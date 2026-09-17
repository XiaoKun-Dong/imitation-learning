"""Summarize paired DemoVLA model-by-normalization rollout matrices."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import pathlib
import random

PHASE_LABELS = {
    "equivalence": (
        "pi05_official_stats",
        "demovla_off_official_stats",
        "pi05_local_stats",
        "demovla_off_local_stats",
    ),
    "causal": (
        "demovla_off_official_stats",
        "demovla_off_local_stats",
        "dynamic_official_stats",
        "dynamic_local_stats",
    ),
}


def _load_metrics(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    outcomes: dict[tuple[int, int], bool] = {}
    with path.open(encoding="utf-8") as metrics_file:
        for line in metrics_file:
            row = json.loads(line)
            key = (int(row["benchmark_task_id"]), int(row["episode"]))
            if key in outcomes:
                raise ValueError(f"duplicate rollout key {key} in {path}")
            outcomes[key] = bool(row["success"])
    if not outcomes:
        raise ValueError(f"no rollout records in {path}")
    return outcomes


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - radius, center + radius


def _mcnemar_exact_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_contrast(
    outcomes: Mapping[str, Mapping[tuple[int, int], bool]],
    contender_label: str,
    reference_label: str,
) -> dict[str, float | int | str]:
    contender = outcomes[contender_label]
    reference = outcomes[reference_label]
    keys = sorted(reference)
    reference_only = sum(reference[key] and not contender[key] for key in keys)
    contender_only = sum(contender[key] and not reference[key] for key in keys)
    both_success = sum(reference[key] and contender[key] for key in keys)
    both_failure = len(keys) - reference_only - contender_only - both_success
    return {
        "contender": contender_label,
        "reference": reference_label,
        "success_rate_delta": (sum(contender.values()) - sum(reference.values())) / len(keys),
        "reference_only": reference_only,
        "contender_only": contender_only,
        "both_success": both_success,
        "both_failure": both_failure,
        "mcnemar_exact_p": _mcnemar_exact_p(reference_only, contender_only),
    }


def _stratified_did_interval(
    outcomes: Mapping[str, Mapping[tuple[int, int], bool]],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    labels = (
        "demovla_off_official_stats",
        "demovla_off_local_stats",
        "dynamic_official_stats",
        "dynamic_local_stats",
    )
    task_keys: dict[int, list[tuple[int, int]]] = {}
    for key in outcomes[labels[0]]:
        task_keys.setdefault(key[0], []).append(key)

    def did(keys: list[tuple[int, int]]) -> float:
        means = {label: sum(outcomes[label][key] for key in keys) / len(keys) for label in labels}
        return (means["dynamic_local_stats"] - means["demovla_off_local_stats"]) - (
            means["dynamic_official_stats"] - means["demovla_off_official_stats"]
        )

    all_keys = sorted(outcomes[labels[0]])
    point = did(all_keys)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        sampled_keys = []
        for keys in task_keys.values():
            sampled_keys.extend(rng.choice(keys) for _ in keys)
        bootstrap.append(did(sampled_keys))
    bootstrap.sort()
    return point, bootstrap[int(0.025 * samples)], bootstrap[int(0.975 * samples)]


def summarize(output_root: pathlib.Path, phase: str) -> tuple[dict[str, object], str]:
    labels = PHASE_LABELS[phase]
    outcomes = {label: _load_metrics(output_root / label / "metrics.jsonl") for label in labels}
    reference_keys = set(outcomes[labels[0]])
    for label, label_outcomes in outcomes.items():
        if set(label_outcomes) != reference_keys:
            missing = sorted(reference_keys - set(label_outcomes))
            extra = sorted(set(label_outcomes) - reference_keys)
            raise ValueError(f"{label} is not paired: missing={missing[:5]} extra={extra[:5]}")

    summaries: dict[str, dict[str, object]] = {}
    for label in labels:
        successes = sum(outcomes[label].values())
        total = len(outcomes[label])
        low, high = _wilson_interval(successes, total)
        per_task: dict[str, dict[str, int | float]] = {}
        for task in sorted({key[0] for key in reference_keys}):
            task_values = [value for key, value in outcomes[label].items() if key[0] == task]
            per_task[str(task)] = {
                "successes": sum(task_values),
                "total": len(task_values),
                "success_rate": sum(task_values) / len(task_values),
            }
        summaries[label] = {
            "successes": successes,
            "total": total,
            "success_rate": successes / total,
            "wilson_95": [low, high],
            "per_task": per_task,
        }

    if phase == "equivalence":
        contrast_pairs = (
            ("demovla_off_official_stats", "pi05_official_stats"),
            ("demovla_off_local_stats", "pi05_local_stats"),
            ("pi05_local_stats", "pi05_official_stats"),
        )
    else:
        contrast_pairs = (
            ("dynamic_official_stats", "demovla_off_official_stats"),
            ("dynamic_local_stats", "demovla_off_local_stats"),
            ("demovla_off_local_stats", "demovla_off_official_stats"),
            ("dynamic_local_stats", "dynamic_official_stats"),
        )
    contrasts = [_paired_contrast(outcomes, contender, reference) for contender, reference in contrast_pairs]

    result: dict[str, object] = {
        "phase": phase,
        "episode_keys": len(reference_keys),
        "conditions": summaries,
        "paired_contrasts": contrasts,
    }
    if phase == "equivalence":
        result["same_stats_outcome_equivalence"] = {
            "official": outcomes["demovla_off_official_stats"] == outcomes["pi05_official_stats"],
            "local": outcomes["demovla_off_local_stats"] == outcomes["pi05_local_stats"],
        }
    else:
        did, did_low, did_high = _stratified_did_interval(outcomes)
        result["difference_in_differences"] = {
            "estimate": did,
            "stratified_bootstrap_95": [did_low, did_high],
        }

    lines = [f"# DemoVLA norm matrix: {phase}", "", "## Success summary", ""]
    lines.extend(("| Condition | Success | Rate | Wilson 95% CI |", "|---|---:|---:|---:|"))
    for label in labels:
        row = summaries[label]
        low, high = row["wilson_95"]
        lines.append(
            f"| `{label}` | {row['successes']}/{row['total']} | {row['success_rate']:.2%} | {low:.2%}-{high:.2%} |"
        )
    lines.extend(("", "## Paired contrasts", ""))
    lines.extend(
        (
            "| Contender - reference | Delta success | Ref only | Contender only | McNemar p |",
            "|---|---:|---:|---:|---:|",
        )
    )
    lines.extend(
        (
            f"| `{contrast['contender']}` - `{contrast['reference']}` | "
            f"{contrast['success_rate_delta']:+.2%} | {contrast['reference_only']} | "
            f"{contrast['contender_only']} | {contrast['mcnemar_exact_p']:.4g} |"
        )
        for contrast in contrasts
    )
    if phase == "equivalence":
        equivalence = result["same_stats_outcome_equivalence"]
        lines.extend(
            (
                "",
                "## Outcome-equivalence gate",
                "",
                f"- Official stats: `{equivalence['official']}`",
                f"- Local stats: `{equivalence['local']}`",
            )
        )
    else:
        did = result["difference_in_differences"]
        did_low, did_high = did["stratified_bootstrap_95"]
        lines.extend(
            (
                "",
                "## Model x normalization interaction",
                "",
                f"Difference-in-differences: `{did['estimate']:+.2%}` "
                f"(task-stratified bootstrap 95% CI `{did_low:+.2%}`-`{did_high:+.2%}`).",
            )
        )
    return result, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--phase", choices=tuple(PHASE_LABELS), required=True)
    args = parser.parse_args()
    result, markdown = summarize(args.output_root, args.phase)
    (args.output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
