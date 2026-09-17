"""Summarize paired DemoVLA evaluations on a LIBERO-plus task panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import pathlib
import random
import re


def _load_metrics(root: pathlib.Path, condition: str, seeds: list[int]) -> dict[tuple[int, int, int], dict]:
    rows: dict[tuple[int, int, int], dict] = {}
    for seed in seeds:
        path = root / condition / f"seed_{seed}" / "metrics.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (seed, int(row["benchmark_task_id"]), int(row["episode"]))
            if key in rows:
                raise ValueError(f"duplicate rollout key {key} in {path}")
            rows[key] = row
    return rows


def _mcnemar_exact_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def _cluster_bootstrap_ci(
    reference: dict[tuple[int, int, int], dict],
    contender: dict[tuple[int, int, int], dict],
    *,
    samples: int = 10_000,
    seed: int = 20260903,
) -> tuple[float, float]:
    by_task: dict[int, list[float]] = defaultdict(list)
    for key, row in reference.items():
        by_task[key[1]].append(float(bool(row["success"])) - float(bool(contender[key]["success"])))
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [rng.choice(task_ids) for _ in task_ids]
        deltas = [delta for task_id in sampled for delta in by_task[task_id]]
        estimates.append(sum(deltas) / len(deltas))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def _level(task_name: str) -> str:
    match = re.search(r"\blevel([1-5])\b", task_name)
    return f"level{match.group(1)}" if match else "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    args = parser.parse_args()

    if args.reference not in args.conditions:
        raise ValueError(f"reference {args.reference!r} is not in {args.conditions}")
    outcomes = {
        condition: _load_metrics(args.output_root, condition, args.seeds) for condition in args.conditions
    }
    reference = outcomes[args.reference]
    reference_keys = set(reference)
    for condition, rows in outcomes.items():
        if set(rows) != reference_keys:
            missing = sorted(reference_keys - set(rows))
            extra = sorted(set(rows) - reference_keys)
            raise ValueError(f"{condition} is not paired: missing={missing[:5]} extra={extra[:5]}")

    lines = [
        "# LIBERO-Plus paired evaluation",
        "",
        f"Episodes: {len(reference_keys)} ({len({key[1] for key in reference_keys})} tasks x "
        f"{len(args.seeds)} seeds)",
        "",
        "## Success rate",
        "",
        "| Condition | Success | Rate |",
        "|---|---:|---:|",
    ]
    for condition in args.conditions:
        success = sum(bool(row["success"]) for row in outcomes[condition].values())
        lines.append(f"| {condition} | {success}/{len(reference_keys)} | {success / len(reference_keys):.2%} |")

    lines.extend(
        [
            "",
            f"## Paired comparisons against {args.reference}",
            "",
            "| Comparator | Difference | Reference-only | Comparator-only | Both success | Both failure | "
            "Cluster bootstrap 95% CI | McNemar exact p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in args.conditions:
        if condition == args.reference:
            continue
        contender = outcomes[condition]
        reference_only = sum(
            bool(reference[key]["success"]) and not bool(contender[key]["success"]) for key in reference_keys
        )
        contender_only = sum(
            bool(contender[key]["success"]) and not bool(reference[key]["success"]) for key in reference_keys
        )
        both_success = sum(
            bool(reference[key]["success"]) and bool(contender[key]["success"]) for key in reference_keys
        )
        both_failure = len(reference_keys) - reference_only - contender_only - both_success
        difference = (reference_only - contender_only) / len(reference_keys)
        ci_low, ci_high = _cluster_bootstrap_ci(reference, contender)
        lines.append(
            f"| {condition} | {difference:+.2%} | {reference_only} | {contender_only} | {both_success} | "
            f"{both_failure} | [{ci_low:+.2%}, {ci_high:+.2%}] | "
            f"{_mcnemar_exact_p(reference_only, contender_only):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Success by named displacement level",
            "",
            "| Level | " + " | ".join(args.conditions) + " |",
            "|---|" + "---:|" * len(args.conditions),
        ]
    )
    levels = sorted({_level(row["task"]) for row in reference.values()})
    for level in levels:
        keys = [key for key, row in reference.items() if _level(row["task"]) == level]
        cells = []
        for condition in args.conditions:
            success = sum(bool(outcomes[condition][key]["success"]) for key in keys)
            cells.append(f"{success}/{len(keys)} ({success / len(keys):.1%})")
        lines.append(f"| {level} | " + " | ".join(cells) + " |")

    summary = "\n".join(lines) + "\n"
    summary_path = args.output_root / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
