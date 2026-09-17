"""Summarize one full-LIBERO run and optionally compare paired runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import pathlib

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _load(root: pathlib.Path) -> dict[tuple[str, int, int], bool]:
    outcomes = {}
    for suite in SUITES:
        path = root / suite / "metrics.jsonl"
        lines = [line for line in path.read_text().splitlines() if line]
        if not lines:
            raise ValueError(f"no rollout records in {path}")
        for line in lines:
            row = json.loads(line)
            key = (suite, int(row["benchmark_task_id"]), int(row["episode"]))
            if key in outcomes:
                raise ValueError(f"duplicate rollout key {key} in {path}")
            outcomes[key] = bool(row["success"])
    if not outcomes:
        raise ValueError(f"no metrics found under {root}")
    return outcomes


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return center - radius, center + radius


def _summary(outcomes: Mapping[tuple[str, int, int], bool]) -> dict[str, object]:
    def summarize_keys(keys: list[tuple[str, int, int]]) -> dict[str, object]:
        successes = sum(outcomes[key] for key in keys)
        low, high = _wilson(successes, len(keys))
        return {
            "successes": successes,
            "total": len(keys),
            "success_rate": successes / len(keys),
            "wilson_95": [low, high],
        }

    keys = sorted(outcomes)
    result = summarize_keys(keys)
    result["per_suite"] = {suite: summarize_keys([key for key in keys if key[0] == suite]) for suite in SUITES}
    return result


def _mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def _contrast(
    contender: Mapping[tuple[str, int, int], bool],
    reference: Mapping[tuple[str, int, int], bool],
    keys: list[tuple[str, int, int]],
) -> dict[str, object]:
    reference_only = sum(reference[key] and not contender[key] for key in keys)
    contender_only = sum(contender[key] and not reference[key] for key in keys)
    return {
        "success_rate_delta": (sum(contender[key] for key in keys) - sum(reference[key] for key in keys)) / len(keys),
        "reference_only": reference_only,
        "contender_only": contender_only,
        "mcnemar_exact_p": _mcnemar(reference_only, contender_only),
    }


def summarize(root: pathlib.Path, reference_root: pathlib.Path | None = None) -> tuple[dict[str, object], str]:
    outcomes = _load(root)
    result: dict[str, object] = {"run": _summary(outcomes)}
    run = result["run"]
    lines = [
        "# Full LIBERO evaluation",
        "",
        "| Suite | Success | Rate | Wilson 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for suite in SUITES:
        row = run["per_suite"][suite]
        low, high = row["wilson_95"]
        lines.append(
            f"| `{suite}` | {row['successes']}/{row['total']} | {row['success_rate']:.2%} | {low:.2%}-{high:.2%} |"
        )
    low, high = run["wilson_95"]
    lines.append(
        f"| **Overall** | **{run['successes']}/{run['total']}** | **{run['success_rate']:.2%}** | "
        f"**{low:.2%}-{high:.2%}** |"
    )

    if reference_root is not None:
        reference = _load(reference_root)
        if set(reference) != set(outcomes):
            raise ValueError("reference and contender rollout keys are not paired")
        all_keys = sorted(outcomes)
        comparison = {
            "overall": _contrast(outcomes, reference, all_keys),
            "per_suite": {
                suite: _contrast(outcomes, reference, [key for key in all_keys if key[0] == suite]) for suite in SUITES
            },
        }
        result["comparison_to_reference"] = comparison
        lines.extend(
            (
                "",
                "## Paired comparison to reference",
                "",
                "| Scope | Delta | Reference only | Contender only | McNemar p |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for suite in SUITES:
            row = comparison["per_suite"][suite]
            lines.append(
                f"| `{suite}` | {row['success_rate_delta']:+.2%} | {row['reference_only']} | "
                f"{row['contender_only']} | {row['mcnemar_exact_p']:.4g} |"
            )
        row = comparison["overall"]
        lines.append(
            f"| **Overall** | **{row['success_rate_delta']:+.2%}** | **{row['reference_only']}** | "
            f"**{row['contender_only']}** | **{row['mcnemar_exact_p']:.4g}** |"
        )
    return result, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--reference-root", type=pathlib.Path)
    args = parser.parse_args()
    result, markdown = summarize(args.output_root, args.reference_root)
    stem = "comparison" if args.reference_root is not None else "summary"
    (args.output_root / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_root / f"{stem}.md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
