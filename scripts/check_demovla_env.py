"""Check whether a clone is ready for DemoVLA training and LIBERO evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ok(message: str) -> None:
    print(f"[ok]   {message}")


def _warn(message: str) -> None:
    print(f"[warn] {message}")


def _fail(message: str, failures: list[str]) -> None:
    print(f"[fail] {message}")
    failures.append(message)


def _check_import(module_name: str, failures: list[str]) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _fail(f"cannot import {module_name}: {exc}", failures)
        return
    version = getattr(module, "__version__", None)
    _ok(f"import {module_name}" + (f" ({version})" if version else ""))


def _check_dataset(data_root: pathlib.Path, failures: list[str], *, required: bool) -> None:
    info_path = data_root / "meta/info.json"
    if not info_path.is_file():
        message = f"LeRobot dataset not found at {data_root}"
        if required:
            _fail(message, failures)
        else:
            _warn(message)
        return

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info["features"]
        image_shape = features["image"]["shape"]
        wrist_shape = features["wrist_image"]["shape"]
        if image_shape != [256, 256, 3] or wrist_shape != [256, 256, 3]:
            raise ValueError(f"expected 256x256 RGB images, got image={image_shape}, wrist_image={wrist_shape}")
        parquet_count = sum(1 for _ in data_root.glob("data/**/*.parquet"))
        expected_episodes = int(info["total_episodes"])
        if parquet_count != expected_episodes:
            raise ValueError(f"expected {expected_episodes} parquet files, found {parquet_count}")
    except Exception as exc:
        _fail(f"invalid LeRobot dataset at {data_root}: {exc}", failures)
        return
    _ok(
        f"dataset {data_root} "
        f"({info['total_episodes']} episodes, {info['total_frames']} frames, 256x256 RGB)"
    )


def _check_norm_stats(config_name: str, failures: list[str], *, required: bool) -> None:
    path = PROJECT_ROOT / "assets" / config_name / "local/libero/norm_stats.json"
    if not path.is_file():
        message = f"norm stats not found at {path}"
        if required:
            _fail(message, failures)
        else:
            _warn(message)
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))["norm_stats"]
        if set(payload) != {"state", "actions"}:
            raise ValueError(f"expected state/actions keys, got {sorted(payload)}")
    except Exception as exc:
        _fail(f"invalid norm stats at {path}: {exc}", failures)
        return
    _ok(f"norm stats {path}")


def _check_checkpoint(failures: list[str], *, required: bool) -> None:
    checkpoint = os.environ.get("OPENPI_PI05_LIBERO_CHECKPOINT")
    if not checkpoint:
        message = (
            "OPENPI_PI05_LIBERO_CHECKPOINT is unset; training will download "
            "gs://openpi-assets/checkpoints/pi05_libero on first use"
        )
        if required:
            _fail(message, failures)
        else:
            _warn(message)
        return
    if checkpoint.startswith(("gs://", "http://", "https://")):
        _ok(f"checkpoint URI configured: {checkpoint}")
        return
    root = pathlib.Path(checkpoint).expanduser()
    metadata = root / "params/_METADATA"
    if not metadata.is_file():
        _fail(f"checkpoint metadata not found at {metadata}", failures)
        return
    _ok(f"local checkpoint {root.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-norm-stats", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--config-name", default="demovla_libero_sparse_deep_diverse")
    args = parser.parse_args()

    failures: list[str] = []
    if sys.version_info[:2] != (3, 11):
        _fail(f"Python 3.11 is required, found {sys.version.split()[0]}", failures)
    else:
        _ok(f"Python {sys.version.split()[0]}")

    for module_name in ("openpi", "jax", "torch", "lerobot", "mujoco"):
        _check_import(module_name, failures)
    if importlib.util.find_spec("robosuite") is None:
        _fail("robosuite is not installed", failures)
    else:
        _ok("robosuite is installed (renderer import deferred)")

    try:
        import jax

        devices = jax.devices()
        platforms = sorted({device.platform for device in devices})
        if args.require_gpu and "gpu" not in platforms:
            _fail(f"JAX GPU device required, found platforms={platforms}", failures)
        else:
            _ok(f"JAX devices: {len(devices)} ({', '.join(platforms)})")
    except Exception as exc:
        _fail(f"cannot enumerate JAX devices: {exc}", failures)

    try:
        from openpi.shared import libero_runtime

        runtime = libero_runtime.prepare(PROJECT_ROOT)
        importlib.import_module("libero.libero.benchmark")
        if args.require_gpu:
            libero_runtime.import_modules(PROJECT_ROOT)
        _ok(f"LIBERO checkout {runtime.checkout_root}")
        _ok(f"LIBERO config {runtime.config_dir / 'config.yaml'}")
    except Exception as exc:
        _fail(f"LIBERO runtime is not ready: {exc}", failures)

    data_root = pathlib.Path(os.environ.get("OPENPI_LIBERO_DATA_ROOT", "data/lerobot/local/libero"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    _check_dataset(data_root, failures, required=args.require_data)
    _check_norm_stats(args.config_name, failures, required=args.require_norm_stats)
    _check_checkpoint(failures, required=args.require_checkpoint)

    if failures:
        print(f"\nEnvironment check failed with {len(failures)} issue(s).")
        return 1
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
