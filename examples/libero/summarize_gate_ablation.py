"""Summarize paired DemoVLA gate-ablation rollouts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib

LABELS = ("dynamic", "layer_mean", "injection_off", "static_deep")


def _load_metrics(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    outcomes: dict[tuple[int, int], bool] = {}
    with path.open(encoding="utf-8") as metrics_file:
        for line in metrics_file:
            row = json.loads(line)
            key = (int(row["benchmark_task_id"]), int(row["episode"]))
            if key in outcomes:
                raise ValueError(f"duplicate rollout key {key} in {path}")
            outcomes[key] = bool(row["success"])
    return outcomes


def _mcnemar_exact_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=pathlib.Path)
    args = parser.parse_args()

    outcomes = {label: _load_metrics(args.output_root / label / "metrics.jsonl") for label in LABELS}
    reference_keys = set(outcomes["dynamic"])
    for label, label_outcomes in outcomes.items():
        if set(label_outcomes) != reference_keys:
            missing = sorted(reference_keys - set(label_outcomes))
            extra = sorted(set(label_outcomes) - reference_keys)
            raise ValueError(f"{label} is not paired with dynamic: missing={missing[:5]} extra={extra[:5]}")

    episode_count = len(reference_keys)
    print("success summary")
    for label in LABELS:
        successes = sum(outcomes[label].values())
        print(f"  {label:13s} {successes:4d}/{episode_count:<4d} {successes / episode_count:7.2%}")

    print("paired comparison against dynamic")
    dynamic = outcomes["dynamic"]
    for label in LABELS[1:]:
        contender = outcomes[label]
        dynamic_only = sum(dynamic[key] and not contender[key] for key in reference_keys)
        contender_only = sum(contender[key] and not dynamic[key] for key in reference_keys)
        both_success = sum(dynamic[key] and contender[key] for key in reference_keys)
        both_failure = episode_count - dynamic_only - contender_only - both_success
        exact_p = _mcnemar_exact_p(dynamic_only, contender_only)
        print(
            f"  {label:13s} dynamic_only={dynamic_only:3d} contender_only={contender_only:3d} "
            f"both_success={both_success:3d} both_failure={both_failure:3d} McNemar_p={exact_p:.4g}"
        )


if __name__ == "__main__":
    main()
