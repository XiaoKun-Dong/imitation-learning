"""Analyze matched same-state DemoVLA interventions around LIBERO replans."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _load_trace(trace_root: pathlib.Path, replan_steps: int) -> list[dict]:
    records = []
    for path in sorted((trace_root / "replans").glob("replan_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["condition_names"]]
            chunks = {name: np.asarray(data["action_chunks"][index]) for index, name in enumerate(names)}
            correct = chunks["correct"][:replan_steps]
            interventions = {}
            for name, chunk in chunks.items():
                if name == "correct":
                    continue
                delta = chunk[:replan_steps] - correct
                interventions[name] = {
                    "executed_l2": float(np.linalg.norm(delta)),
                    "translation_l2": float(np.linalg.norm(delta[:, :3])),
                    "rotation_l2": float(np.linalg.norm(delta[:, 3:6])),
                    "gripper_l2": float(np.linalg.norm(delta[:, 6])),
                    "max_step_l2": float(np.max(np.linalg.norm(delta, axis=-1))),
                    "gripper_sign_disagreement": bool(
                        np.any((np.sign(chunk[:replan_steps, 6]) != np.sign(correct[:, 6])) & (np.abs(delta[:, 6]) > 0.25))
                    ),
                }
            zero_distance = interventions["zero"]["executed_l2"]
            transplant_distance = interventions.get("transplant", {}).get("executed_l2")
            transplant_rescue = None
            if transplant_distance is not None and zero_distance > 1.0e-8:
                transplant_rescue = float(1.0 - transplant_distance / zero_distance)
            records.append(
                {
                    "trace_file": str(path),
                    "replan_index": int(data["replan_index"]),
                    "env_step": int(data["env_step"]),
                    "interventions": interventions,
                    "transplant_rescue_fraction": transplant_rescue,
                }
            )
    if not records:
        raise ValueError(f"no replan NPZ files found under {trace_root}")
    return records


def _robust_threshold(values: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(values))
    scaled_mad = float(1.4826 * np.median(np.abs(values - median)))
    threshold = median + 3.0 * scaled_mad
    return median, scaled_mad, threshold


def _condition_summary(records: list[dict]) -> dict[str, dict[str, float]]:
    names = sorted({name for row in records for name in row["interventions"]})
    summary = {}
    for name in names:
        values = np.asarray(
            [row["interventions"][name]["executed_l2"] for row in records if name in row["interventions"]]
        )
        summary[name] = {
            "count": int(values.size),
            "mean_executed_l2": float(np.mean(values)),
            "median_executed_l2": float(np.median(values)),
            "max_executed_l2": float(np.max(values)),
        }
    return summary


def _markdown(result: dict) -> str:
    lines = [
        "# DemoVLA 关键 replan 同状态干预分析",
        "",
        f"- 输出根目录: `{result['output_root']}`",
        f"- replan steps: {result['replan_steps']}",
        "- 主候选只使用 zero-executed 失败轨迹上的 `zero vs correct` 动作差选择, 避免用 transplant/wrong 结果反向挑点。",
        "",
    ]
    for trajectory, payload in result["trajectories"].items():
        lines.extend(
            [
                f"## {trajectory}",
                "",
                f"- success: {payload['episode_summary']['success']}",
                f"- replans: {payload['episode_summary']['num_replans']}",
                "",
                "| Condition | N | Mean executed L2 | Median | Max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, stats in payload["condition_summary"].items():
            lines.append(
                f"| {name} | {stats['count']} | {stats['mean_executed_l2']:.6f} | "
                f"{stats['median_executed_l2']:.6f} | {stats['max_executed_l2']:.6f} |"
            )
        lines.append("")

    selection = result["zero_trajectory_selection"]
    lines.extend(
        [
            "## Zero 失败轨迹候选",
            "",
            f"- zero-effect median: {selection['median']:.6f}",
            f"- scaled MAD: {selection['scaled_mad']:.6f}",
            f"- 预注册阈值 `median + 3xscaled_MAD`: {selection['threshold']:.6f}",
            "",
            "| Rank | Replan | Env step | Zero L2 | Wrong L2 | Transplant rescue | KO-4 | KO-9 | KO-14 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(selection["ranked_candidates"], start=1):
        interventions = row["interventions"]
        rescue = row["transplant_rescue_fraction"]
        lines.append(
            f"| {rank} | {row['replan_index']} | {row['env_step']} | "
            f"{interventions['zero']['executed_l2']:.6f} | "
            f"{interventions['wrong_prompt']['executed_l2']:.6f} | "
            f"{rescue:.3f} | " if rescue is not None else
            f"| {rank} | {row['replan_index']} | {row['env_step']} | "
            f"{interventions['zero']['executed_l2']:.6f} | "
            f"{interventions['wrong_prompt']['executed_l2']:.6f} | n/a | "
        )
        lines[-1] += (
            f"{interventions['knockout_layer4']['executed_l2']:.6f} | "
            f"{interventions['knockout_layer9']['executed_l2']:.6f} | "
            f"{interventions['knockout_layer14']['executed_l2']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    trajectories = {}
    trace_records = {}
    for name in ("correct", "zero"):
        trace_root = args.output_root / name
        episode_summary = json.loads((trace_root / "summary.json").read_text(encoding="utf-8"))
        records = _load_trace(trace_root, args.replan_steps)
        trace_records[name] = records
        trajectories[name] = {
            "episode_summary": episode_summary,
            "condition_summary": _condition_summary(records),
        }

    zero_records = trace_records["zero"]
    zero_values = np.asarray([row["interventions"]["zero"]["executed_l2"] for row in zero_records])
    median, scaled_mad, threshold = _robust_threshold(zero_values)
    candidates = [
        row
        for row in zero_records
        if row["interventions"]["zero"]["executed_l2"] >= threshold
        or row["interventions"]["zero"]["gripper_sign_disagreement"]
    ]
    ranked = sorted(
        candidates,
        key=lambda row: row["interventions"]["zero"]["executed_l2"],
        reverse=True,
    )[: args.top_k]
    result = {
        "output_root": str(args.output_root.resolve()),
        "replan_steps": args.replan_steps,
        "trajectories": trajectories,
        "zero_trajectory_selection": {
            "median": median,
            "scaled_mad": scaled_mad,
            "threshold": threshold,
            "candidate_count": len(candidates),
            "earliest_candidate_replan": min((row["replan_index"] for row in candidates), default=None),
            "ranked_candidates": ranked,
        },
    }
    analysis_dir = args.output_root / "analysis"
    analysis_dir.mkdir(exist_ok=False)
    (analysis_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (analysis_dir / "summary.md").write_text(_markdown(result), encoding="utf-8")
    print(_markdown(result))


if __name__ == "__main__":
    main()
