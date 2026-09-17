#!/usr/bin/env python3
"""Audit and summarize the frozen DemoVLA content-intervention expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
from typing import Any

import numpy as np

CONDITIONS = ("correct", "shuffled", "zero", "injection_off", "wrong_prompt")
COMPARATORS = ("shuffled", "zero", "injection_off", "wrong_prompt")
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_SAMPLES = 100_000
ACTION_THRESHOLD = 1e-6
TRAJECTORY_THRESHOLD = 1e-6


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _video_frames(path: pathlib.Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _exact_mcnemar(correct_only: int, comparator_only: int) -> float:
    discordant = correct_only + comparator_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, index) for index in range(min(correct_only, comparator_only) + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def _paired_bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    if len(differences) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
    estimates = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _audit_point(
    point: dict[str, Any], manifest: dict[str, Any], run_root: pathlib.Path
) -> tuple[dict[str, Any], list[str]]:
    point_id = point["point_id"]
    run_dir = run_root / point_id
    errors: list[str] = []
    summary_path = run_dir / "summary.json"
    run_manifest_path = run_dir / "run_manifest.json"
    if not summary_path.is_file() or not run_manifest_path.is_file():
        return {"point_id": point_id, "status": "missing"}, [f"{point_id}: missing summary or run manifest"]

    summary = _read_json(summary_path)
    run_manifest = _read_json(run_manifest_path)
    for key in ("task_id", "episode_id", "branch_replan"):
        if summary.get(key) != point[key]:
            errors.append(f"{point_id}: summary {key} mismatch")
        if run_manifest.get(key) != point[key]:
            errors.append(f"{point_id}: run manifest {key} mismatch")

    donor = manifest["donors"][point["donor_id"]]
    if run_manifest.get("shuffled_memory_file_sha256") != donor["sha256"]:
        errors.append(f"{point_id}: donor SHA256 mismatch")
    if run_manifest.get("shuffled_memory_shape") != donor["shape"]:
        errors.append(f"{point_id}: donor shape mismatch")

    reached = summary.get("intervention_reached") is True
    prefix_valid = (
        summary.get("initialization_max_abs_diff") == 0
        and summary.get("prefix_max_abs_diff") == 0
    )
    branch_valid = summary.get("branch_image_equal") is True and summary.get("branch_state_equal") is True
    if not prefix_valid:
        errors.append(f"{point_id}: initialization/prefix audit failed")
    if reached and not branch_valid:
        errors.append(f"{point_id}: branch audit failed")
    if not reached and summary.get("status") != "intervention_not_reached":
        errors.append(f"{point_id}: non-reached point lacks registered status")

    trajectories: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        condition_dir = run_dir / "conditions" / condition
        video_path = condition_dir / "rollout.mp4"
        trajectory_path = condition_dir / "trajectory.npz"
        replans_path = condition_dir / "replans.jsonl"
        if not all(path.is_file() and path.stat().st_size > 0 for path in (video_path, trajectory_path, replans_path)):
            errors.append(f"{point_id}/{condition}: incomplete artifacts")
            continue
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            actions = np.asarray(trajectory["actions"])
            states = np.asarray(trajectory["simulator_states"])
        trajectories[condition] = states
        condition_summary = summary["conditions"][condition]
        expected_steps = condition_summary["final_env_step"]
        if len(actions) != expected_steps or len(states) != expected_steps:
            errors.append(f"{point_id}/{condition}: trajectory length mismatch")
        replan_lines = sum(1 for line in replans_path.read_text(encoding="utf-8").splitlines() if line)
        if replan_lines != condition_summary["num_replans"]:
            errors.append(f"{point_id}/{condition}: replan count mismatch")
        if _video_frames(video_path) != expected_steps:
            errors.append(f"{point_id}/{condition}: video frame mismatch")

    row: dict[str, Any] = {
        "point_id": point_id,
        "task_id": point["task_id"],
        "episode_id": point["episode_id"],
        "phase_bin": point["phase_bin"],
        "branch_replan": point["branch_replan"],
        "status": summary.get("status"),
        "intervention_reached": reached,
        "audit_valid": prefix_valid and (branch_valid if reached else True),
    }
    for condition in CONDITIONS:
        row[f"{condition}_success"] = bool(summary["conditions"][condition]["success"])
        row[f"{condition}_termination"] = summary["conditions"][condition].get("termination")
        row[f"{condition}_final_env_step"] = summary["conditions"][condition]["final_env_step"]

    if reached:
        branch_step = point["branch_replan"] * manifest["shared_protocol"]["replan_steps"]
        for comparator in COMPARATORS:
            action_l2 = summary["branch_action_distances_from_correct"][comparator]["executed_l2"]
            row[f"{comparator}_action_l2"] = action_l2
            row[f"{comparator}_action_nonzero"] = action_l2 > ACTION_THRESHOLD
            if "correct" in trajectories and comparator in trajectories:
                common_length = min(len(trajectories["correct"]), len(trajectories[comparator]))
                if common_length > branch_step:
                    max_abs = float(
                        np.max(
                            np.abs(
                                trajectories["correct"][branch_step:common_length]
                                - trajectories[comparator][branch_step:common_length]
                            )
                        )
                    )
                    row[f"{comparator}_trajectory_max_abs"] = max_abs
                    row[f"{comparator}_trajectory_diverged"] = max_abs > TRAJECTORY_THRESHOLD
    return row, errors


def _comparison_statistics(rows: list[dict[str, Any]], comparator: str) -> dict[str, Any]:
    valid = [row for row in rows if row["intervention_reached"] and row["audit_valid"]]
    correct = np.asarray([row["correct_success"] for row in valid], dtype=np.int8)
    compared = np.asarray([row[f"{comparator}_success"] for row in valid], dtype=np.int8)
    differences = correct - compared
    correct_only = int(np.sum((correct == 1) & (compared == 0)))
    comparator_only = int(np.sum((correct == 0) & (compared == 1)))
    ci_low, ci_high = _paired_bootstrap_ci(differences)
    action_l2 = np.asarray([row[f"{comparator}_action_l2"] for row in valid], dtype=np.float64)
    trajectory_diverged = [row.get(f"{comparator}_trajectory_diverged", False) for row in valid]
    return {
        "comparator": comparator,
        "n_valid_pairs": len(valid),
        "correct_successes": int(correct.sum()),
        "comparator_successes": int(compared.sum()),
        "correct_only": correct_only,
        "comparator_only": comparator_only,
        "paired_success_difference": float(differences.mean()) if len(differences) else math.nan,
        "bootstrap_95_ci": [ci_low, ci_high],
        "exact_mcnemar_p": _exact_mcnemar(correct_only, comparator_only),
        "action_nonzero_count": int(np.sum(action_l2 > ACTION_THRESHOLD)),
        "action_l2_mean": float(action_l2.mean()) if len(action_l2) else math.nan,
        "action_l2_median": float(np.median(action_l2)) if len(action_l2) else math.nan,
        "trajectory_diverged_count": int(sum(trajectory_diverged)),
    }


def _format_rate(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=pathlib.Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    manifest = _read_json(root / "expansion_manifest.json")
    run_root = root / "expansion_runs"

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for point in manifest["points"]:
        row, point_errors = _audit_point(point, manifest, run_root)
        rows.append(row)
        errors.extend(point_errors)
    if errors:
        raise RuntimeError("Expansion audit failed:\n" + "\n".join(errors))

    statistics = [_comparison_statistics(rows, comparator) for comparator in COMPARATORS]
    result = {
        "protocol_id": manifest["protocol_id"],
        "registered_points": len(rows),
        "completed_points": sum(row["status"] == "completed" for row in rows),
        "intervention_not_reached": sum(not row["intervention_reached"] for row in rows),
        "valid_intervention_pairs": sum(row["intervention_reached"] and row["audit_valid"] for row in rows),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "comparisons": statistics,
        "points": rows,
    }
    (root / "expansion_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row})
    with (root / "expansion_results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# DemoVLA closed-loop content intervention: 30 点扩展结果",
        "",
        f"注册点: {len(rows)}; 有效干预 pair: {result['valid_intervention_pairs']}; "
        f"干预点前完成: {result['intervention_not_reached']}。",
        "",
        "## 配对闭环统计",
        "",
        "| comparator | N | correct/comparator success | correct-only | comparator-only | paired diff | 95% bootstrap CI | exact McNemar p | action nonzero | trajectory diverged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stat in statistics:
        ci = stat["bootstrap_95_ci"]
        lines.append(
            f"| {stat['comparator']} | {stat['n_valid_pairs']} | "
            f"{stat['correct_successes']}/{stat['comparator_successes']} | "
            f"{stat['correct_only']} | {stat['comparator_only']} | "
            f"{_format_rate(stat['paired_success_difference'])} | "
            f"[{_format_rate(ci[0])}, {_format_rate(ci[1])}] | "
            f"{stat['exact_mcnemar_p']:.6g} | {stat['action_nonzero_count']} | "
            f"{stat['trajectory_diverged_count']} |"
        )
    lines.extend(
        [
            "",
            "## 逐点结果",
            "",
            "| point | task/episode | phase | branch | status | correct | shuffled | zero | off | wrong prompt |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {row['point_id']} | {row['task_id']}/{row['episode_id']} | {row['phase_bin']} | "
            f"r{row['branch_replan']} | {row['status']} | {int(row['correct_success'])} | "
            f"{int(row['shuffled_success'])} | {int(row['zero_success'])} | "
            f"{int(row['injection_off_success'])} | {int(row['wrong_prompt_success'])} |"
        )
        for row in rows
    )
    lines.extend(
        [
            "",
            "注: `intervention_not_reached` 行的五路结果来自完全相同的共享前缀, 不进入干预比较分母。",
            "Pilot 四点未并入本表统计。CI 为固定 seed 的配对 percentile bootstrap; p 值为双侧 exact McNemar。",
        ]
    )
    (root / "expansion_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "points"}, indent=2))


if __name__ == "__main__":
    main()
