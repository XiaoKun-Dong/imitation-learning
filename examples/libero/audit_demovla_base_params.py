"""Compare the non-DemoVLA parameter tensors of two OpenPI checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib

from flax import traverse_util
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model


def _tensor_digest(value: np.ndarray) -> dict[str, object]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest(),
    }


def _checkpoint_digests(params_dir: pathlib.Path) -> dict[str, dict[str, object]]:
    # Policy loading casts both checkpoints to bfloat16. Compare that effective
    # inference representation instead of their potentially different storage
    # dtypes.
    params = _model.restore_params(params_dir, restore_type=np.ndarray, dtype=jnp.bfloat16)
    flat_params = traverse_util.flatten_dict(params)
    result = {
        "/".join(path): _tensor_digest(value)
        for path, value in flat_params.items()
        if not any(component.startswith("demovla_") for component in path)
    }
    del params, flat_params
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dynamic_params", type=pathlib.Path)
    parser.add_argument("pi05_params", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    print(f"Hashing pi0.5 checkpoint: {args.pi05_params}", flush=True)
    pi05 = _checkpoint_digests(args.pi05_params)
    print(f"Hashing DemoVLA checkpoint: {args.dynamic_params}", flush=True)
    dynamic = _checkpoint_digests(args.dynamic_params)

    pi05_paths = set(pi05)
    dynamic_paths = set(dynamic)
    common_paths = pi05_paths & dynamic_paths
    mismatched = sorted(path for path in common_paths if pi05[path] != dynamic[path])
    result = {
        "pi05_tensor_count": len(pi05),
        "dynamic_base_tensor_count": len(dynamic),
        "common_tensor_count": len(common_paths),
        "missing_from_dynamic": sorted(pi05_paths - dynamic_paths),
        "extra_in_dynamic_base": sorted(dynamic_paths - pi05_paths),
        "mismatched_tensor_count": len(mismatched),
        "mismatched_tensors": mismatched,
        "base_parameters_identical": pi05_paths == dynamic_paths and not mismatched,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
