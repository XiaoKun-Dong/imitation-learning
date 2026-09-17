"""Verify the full 40-task LIBERO LeRobot dataset and its normalization stats."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.shared import normalize

EXPECTED = {
    "codebase_version": "v2.0",
    "total_episodes": 1693,
    "total_frames": 273465,
    "total_tasks": 40,
    "fps": 10,
}
ACTION_HORIZON = 10


def _column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values).reshape(len(array), array.type.list_size)
    return np.asarray(array)


def _load_episode_vectors(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["state", "actions"])
    state = _column_to_numpy(table["state"]).astype(np.float64)
    actions = _column_to_numpy(table["actions"]).astype(np.float64)
    if state.shape != (len(table), 8):
        raise ValueError(f"unexpected state shape in {path}: {state.shape}")
    if actions.shape != (len(table), 7):
        raise ValueError(f"unexpected action shape in {path}: {actions.shape}")
    indices = np.minimum(
        np.arange(len(actions))[:, None] + np.arange(ACTION_HORIZON)[None, :],
        len(actions) - 1,
    )
    return state, actions[indices].reshape(-1, actions.shape[-1])


def _statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=pathlib.Path)
    parser.add_argument("norm_stats", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--write-computed-stats-dir",
        type=pathlib.Path,
        help="Write the exact recomputed statistics as <dir>/norm_stats.json.",
    )
    # The released stats were accumulated in batches with float32 inputs,
    # while this independent audit uses float64 and exact quantiles. Small
    # differences are expected even when the underlying rows are identical.
    parser.add_argument("--mean-std-atol", type=float, default=5e-4)
    parser.add_argument("--quantile-atol", type=float, default=5e-3)
    parser.add_argument(
        "--allow-stats-mismatch",
        action="store_true",
        help=(
            "Require the dataset structure to be complete but record, rather than reject, differences from the "
            "provided stats. Use only when evaluating or adapting a frozen checkpoint with intentionally locked stats."
        ),
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    mismatched_metadata = {
        key: {"expected": expected, "actual": info.get(key)}
        for key, expected in EXPECTED.items()
        if info.get(key) != expected
    }
    if mismatched_metadata:
        raise ValueError(f"LIBERO metadata mismatch: {mismatched_metadata}")

    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    task_records = [line for line in tasks_path.read_text().splitlines() if line]
    episode_records = [line for line in episodes_path.read_text().splitlines() if line]
    if len(task_records) != EXPECTED["total_tasks"]:
        raise ValueError(f"expected 40 task records, found {len(task_records)}")
    if len(episode_records) != EXPECTED["total_episodes"]:
        raise ValueError(f"expected 1693 episode records, found {len(episode_records)}")

    incomplete = sorted(root.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"download has {len(incomplete)} incomplete files; first={incomplete[0]}")
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if len(parquet_paths) != EXPECTED["total_episodes"]:
        raise ValueError(f"expected 1693 parquet files, found {len(parquet_paths)}")

    state_parts = []
    action_parts = []
    row_count = 0
    for path in parquet_paths:
        state, actions = _load_episode_vectors(path)
        row_count += len(state)
        state_parts.append(state)
        action_parts.append(actions)
    if row_count != EXPECTED["total_frames"]:
        raise ValueError(f"expected 273465 parquet rows, found {row_count}")

    computed = {
        "state": _statistics(np.concatenate(state_parts)),
        "actions": _statistics(np.concatenate(action_parts)),
    }
    if args.write_computed_stats_dir is not None:
        normalize.save(
            args.write_computed_stats_dir,
            {key: normalize.NormStats(**fields) for key, fields in computed.items()},
        )
    reference = normalize.load(args.norm_stats.parent)
    differences: dict[str, dict[str, float]] = {}
    failures = []
    for key in ("state", "actions"):
        differences[key] = {}
        for field in ("mean", "std", "q01", "q99"):
            actual = computed[key][field]
            expected = np.asarray(getattr(reference[key], field))
            max_abs = float(np.max(np.abs(actual - expected)))
            differences[key][field] = max_abs
            tolerance = args.mean_std_atol if field in ("mean", "std") else args.quantile_atol
            if max_abs > tolerance:
                failures.append(f"{key}.{field}: max_abs={max_abs:.8g} > atol={tolerance:.8g}")

    result = {
        "dataset_root": str(root),
        "metadata": {key: info[key] for key in EXPECTED},
        "parquet_files": len(parquet_paths),
        "parquet_rows": row_count,
        "task_records": len(task_records),
        "episode_records": len(episode_records),
        "norm_stats_path": str(args.norm_stats.resolve()),
        "norm_stats_sha256": hashlib.sha256(args.norm_stats.read_bytes()).hexdigest(),
        "computed_stats_path": (
            str((args.write_computed_stats_dir / "norm_stats.json").resolve())
            if args.write_computed_stats_dir is not None
            else None
        ),
        "max_abs_difference_from_exact_recompute": differences,
        "mean_std_atol": args.mean_std_atol,
        "quantile_atol": args.quantile_atol,
        "stats_match_required": not args.allow_stats_mismatch,
        "stats_matched": not failures,
        "passed": not failures or args.allow_stats_mismatch,
        "failures": failures,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures and not args.allow_stats_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
