"""Run the OpenPI LIBERO evaluator against an isolated LIBERO-plus checkout."""

import os
import pathlib
import runpy
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBERO_PLUS_ROOT = ROOT / "third_party/libero-plus/LIBERO-plus"
LIBERO_PLUS_PACKAGE_ROOT = LIBERO_PLUS_ROOT / "libero"
LIBERO_PLUS_BENCHMARK_ROOT = LIBERO_PLUS_PACKAGE_ROOT / "libero"


def _write_config() -> pathlib.Path:
    config_dir = pathlib.Path(tempfile.gettempdir()) / "openpi-libero-plus"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": LIBERO_PLUS_BENCHMARK_ROOT,
        "bddl_files": LIBERO_PLUS_BENCHMARK_ROOT / "bddl_files",
        "init_states": LIBERO_PLUS_BENCHMARK_ROOT / "init_files",
        "datasets": LIBERO_PLUS_PACKAGE_ROOT / "datasets",
        "assets": LIBERO_PLUS_BENCHMARK_ROOT / "assets",
    }
    (config_dir / "config.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in config.items()), encoding="utf-8"
    )
    return config_dir


def main() -> None:
    if not LIBERO_PLUS_ROOT.exists():
        raise FileNotFoundError(f"LIBERO-plus checkout not found: {LIBERO_PLUS_ROOT}")
    assets_dir = LIBERO_PLUS_BENCHMARK_ROOT / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"LIBERO-plus assets not found: {assets_dir}. Download and extract assets.zip before evaluation."
        )

    os.environ["LIBERO_CONFIG_PATH"] = str(_write_config())
    sys.path.insert(0, str(LIBERO_PLUS_ROOT))

    # LIBERO-plus init-state files contain trusted NumPy arrays. PyTorch 2.6 changed
    # torch.load's default to weights_only=True, which cannot deserialize them.
    torch_load = torch.load

    def load_trusted_init_state(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    torch.load = load_trusted_init_state

    defaults = {
        "--args.task-category": "Objects Layout",
        "--args.num-trials-per-task": "1",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if "--args.max-tasks" not in sys.argv and "--args.task-ids" not in sys.argv:
        sys.argv.extend(["--args.max-tasks", "20"])

    runpy.run_path(str(pathlib.Path(__file__).with_name("main.py")), run_name="__main__")


if __name__ == "__main__":
    main()
