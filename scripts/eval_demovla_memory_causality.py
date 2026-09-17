"""Paired fixed-noise flow-loss evaluation for DemoVLA memory interventions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import json
import math
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

MODES: Mapping[str, str] = {
    "correct": "normal",
    "off": "off",
    "shuffled": "batch_shuffle",
    "zero": "zero_memory",
}


def _exact_sign_p(negative: int, positive: int) -> float:
    discordant = negative + positive
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(negative, positive) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def _paired_summary(correct: np.ndarray, comparator: np.ndarray) -> dict[str, float | int | list[float]]:
    delta = correct - comparator
    mean = float(np.mean(delta))
    std = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0
    radius = 1.959963984540054 * std / math.sqrt(delta.size) if delta.size > 1 else 0.0
    negative = int(np.sum(delta < 0.0))
    positive = int(np.sum(delta > 0.0))
    return {
        "correct_minus_comparator_mean": mean,
        "normal_95": [mean - radius, mean + radius],
        "correct_lower": negative,
        "comparator_lower": positive,
        "ties": int(np.sum(delta == 0.0)),
        "correct_lower_rate": negative / delta.size,
        "sign_test_exact_p": _exact_sign_p(negative, positive),
    }


def _render_markdown(result: dict) -> str:
    lines = [
        "# DemoVLA memory causality: paired flow loss",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- samples: {result['num_samples']} ({result['num_batches']} batches x {result['batch_size']})",
        f"- seed: {result['seed']}",
        "- shuffled definition: deterministic half-batch cyclic shift after memory extraction",
        "- all conditions share observations, actions, flow time, and Gaussian flow noise",
        "",
        "| Condition | Mean flow loss | Std |",
        "|---|---:|---:|",
    ]
    for label in MODES:
        row = result["conditions"][label]
        lines.append(f"| `{label}` | {row['mean']:.8f} | {row['std']:.8f} |")
    lines.extend(
        [
            "",
            "| Contrast | Correct - comparator | 95% normal CI | Correct lower | Comparator lower | Sign p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("off", "shuffled", "zero"):
        row = result["paired"][f"correct_vs_{label}"]
        low, high = row["normal_95"]
        lines.append(
            f"| `correct vs {label}` | {row['correct_minus_comparator_mean']:+.8f} | "
            f"[{low:+.8f}, {high:+.8f}] | {row['correct_lower']} | {row['comparator_lower']} | "
            f"{row['sign_test_exact_p']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def _make_eval_fn(model):
    graphdef, state = nnx.split(model)

    def evaluate(
        model_state: nnx.State,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> jax.Array:
        restored_model = nnx.merge(graphdef, model_state)
        loss, metrics = restored_model.compute_loss_with_aux(rng, observation, actions, train=False)
        return loss - metrics.get("demovla_diversity_regularization", 0.0)

    return jax.jit(evaluate), state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="demovla_libero_full_stage_a_fixed_gate")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-gate", type=float, default=0.03)
    args = parser.parse_args()

    if args.batch_size < 2:
        raise ValueError("batch size must be >= 2 for the shuffled-memory intervention")
    if args.batch_size % 2:
        raise ValueError("batch size must be even so half-batch shuffle is a derangement")
    if not 0.0 < args.fixed_gate < 1.0:
        raise ValueError("fixed gate must be strictly between zero and one")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params missing: {args.checkpoint / 'params'}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    base_config = _config.get_config(args.config_name)
    # Ranking is a training-only objective. Disable it so every intervention
    # below reports the same pure per-sample flow loss.
    eval_model_config = dataclasses.replace(base_config.model, interaction_memory_ranking_weight=0.0)
    config = dataclasses.replace(
        base_config,
        model=eval_model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = _data_loader.create_data_loader(
        config,
        shuffle=True,
        num_batches=args.num_batches,
    )

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    evaluators = {}
    for label, mode in MODES.items():
        model = config.model.load(params)
        model.eval()
        fixed_gates = () if mode == "off" else (args.fixed_gate,) * len(config.model.interaction_injection_layers)
        model.configure_interaction_inference_ablation(mode if mode != "normal" else "layer_mean", fixed_gates)
        evaluators[label] = _make_eval_fn(model)

    losses: dict[str, list[np.ndarray]] = {label: [] for label in MODES}
    records_path = args.output_dir / "paired_losses.jsonl"
    started = time.time()
    for batch_index, (observation, actions) in enumerate(loader):
        rng = jax.random.fold_in(jax.random.key(args.seed), batch_index)
        batch_losses = {}
        for label, (eval_fn, state) in evaluators.items():
            values = np.asarray(jax.device_get(eval_fn(state, rng, observation, actions))).mean(axis=-1)
            losses[label].append(values)
            batch_losses[label] = values
        with records_path.open("a", encoding="utf-8") as stream:
            for sample_index in range(args.batch_size):
                stream.write(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "sample": sample_index,
                            **{label: float(values[sample_index]) for label, values in batch_losses.items()},
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        print(f"completed batch {batch_index + 1}/{args.num_batches}", flush=True)

    arrays = {label: np.concatenate(chunks) for label, chunks in losses.items()}
    result = {
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "num_samples": args.num_batches * args.batch_size,
        "seed": args.seed,
        "fixed_gate": args.fixed_gate,
        "elapsed_seconds": time.time() - started,
        "conditions": {
            label: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for label, values in arrays.items()
        },
        "paired": {
            f"correct_vs_{label}": _paired_summary(arrays["correct"], arrays[label])
            for label in ("off", "shuffled", "zero")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = _render_markdown(result)
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
