"""Runtime setup helpers for the vendored LIBERO benchmark.

LIBERO has used two checkout layouts in this project:

* ``third_party/libero`` (the normal git-submodule layout)
* ``third_party/libero/LIBERO`` (older local checkouts)

The upstream package also prompts for a config file during import.  These helpers
make evaluator startup deterministic without requiring a separate editable
installation of LIBERO.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pathlib
import sys
from typing import Any

_TORCH_LOAD_STATE = {"patched": False}


@dataclasses.dataclass(frozen=True)
class LiberoRuntime:
    checkout_root: pathlib.Path
    config_dir: pathlib.Path


def find_checkout(project_root: pathlib.Path) -> pathlib.Path:
    """Return the LIBERO checkout root for either supported repository layout."""
    candidates = []
    if override := os.environ.get("OPENPI_LIBERO_ROOT"):
        candidates.append(pathlib.Path(override).expanduser())
    candidates.extend(
        [
            project_root / "third_party/libero",
            project_root / "third_party/libero/LIBERO",
        ]
    )

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if (resolved_candidate / "libero/libero/__init__.py").is_file():
            return resolved_candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "LIBERO checkout not found. Run `git submodule update --init --recursive` "
        "or set OPENPI_LIBERO_ROOT. Searched:\n"
        f"{searched}"
    )


def prepare(project_root: pathlib.Path | None = None) -> LiberoRuntime:
    """Put LIBERO on ``sys.path`` and create a non-interactive config file."""
    if project_root is None:
        project_root = pathlib.Path(__file__).resolve().parents[3]
    project_root = project_root.resolve()
    checkout_root = find_checkout(project_root)

    checkout_str = str(checkout_root)
    if checkout_str not in sys.path:
        sys.path.insert(0, checkout_str)

    configured_dir = os.environ.get("LIBERO_CONFIG_PATH")
    config_dir = (
        pathlib.Path(configured_dir).expanduser().resolve()
        if configured_dir
        else (project_root / ".cache/libero").resolve()
    )
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    config_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = checkout_root / "libero/libero"
    config: dict[str, Any] = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(checkout_root / "libero/datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    config_file = config_dir / "config.yaml"
    # JSON is valid YAML. Rewrite the project-local default so moving the clone
    # does not leave stale absolute paths; preserve an explicitly supplied config.
    if not configured_dir or not config_file.exists():
        config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return LiberoRuntime(checkout_root=checkout_root, config_dir=config_dir)


def enable_trusted_torch_load() -> None:
    """Restore pre-PyTorch-2.6 loading for trusted LIBERO init-state files."""
    import torch

    if _TORCH_LOAD_STATE["patched"]:
        return
    original_load = torch.load

    def load_trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_trusted
    _TORCH_LOAD_STATE["patched"] = True


def import_modules(project_root: pathlib.Path | None = None):
    """Prepare LIBERO and return the benchmark API used by the evaluator."""
    prepare(project_root)
    enable_trusted_torch_load()
    libero_package = importlib.import_module("libero.libero")
    benchmark = importlib.import_module("libero.libero.benchmark")
    envs = importlib.import_module("libero.libero.envs")
    return benchmark, libero_package.get_libero_path, envs.OffScreenRenderEnv, envs.SegmentationRenderEnv
