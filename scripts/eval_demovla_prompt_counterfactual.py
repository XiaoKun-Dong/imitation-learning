"""Paired flow-loss and memory diagnostics for LIBERO prompt counterfactuals."""

from __future__ import annotations

import argparse
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
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _exact_sign_p(negative: int, positive: int) -> float:
    discordant = negative + positive
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(negative, positive) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def _load_tasks(path: pathlib.Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: int(row["task_index"]))
    indices = [int(row["task_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"task indices must be contiguous from zero, got {indices}")
    return [str(row["task"]) for row in rows]


def _suite_swap_indices(num_tasks: int) -> np.ndarray:
    if num_tasks % 10:
        raise ValueError("LIBERO prompt counterfactual expects task groups of 10")
    indices = np.arange(num_tasks, dtype=np.int32)
    return (indices // 10) * 10 + (indices + 1) % 10


def _token_key(tokens: np.ndarray, mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(token) for token in np.asarray(tokens)[np.asarray(mask, dtype=bool)])


def _replace_prompt(
    observation: _model.Observation,
    tokens: np.ndarray,
    masks: np.ndarray,
) -> _model.Observation:
    return observation.replace(
        tokenized_prompt=jnp.asarray(tokens),
        tokenized_prompt_mask=jnp.asarray(masks),
    )


def _make_eval_fn(model):
    graphdef, state = nnx.split(model)

    def evaluate(
        model_state: nnx.State,
        rng: jax.Array,
        action_observation: _model.Observation,
        memory_observation: _model.Observation,
        actions: _model.Actions,
    ):
        restored_model = nnx.merge(graphdef, model_state)
        return restored_model.compute_flow_loss_with_memory_source(
            rng,
            action_observation,
            memory_observation,
            actions,
        )

    return jax.jit(evaluate), state


def _cosine(first: np.ndarray, second: np.ndarray, axis: int = -1) -> np.ndarray:
    numerator = np.sum(first * second, axis=axis)
    denominator = np.linalg.norm(first, axis=axis) * np.linalg.norm(second, axis=axis)
    return numerator / np.maximum(denominator, 1e-12)


def _paired_summary(correct: np.ndarray, counterfactual: np.ndarray) -> dict:
    delta = correct - counterfactual
    mean = float(np.mean(delta))
    std = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0
    radius = 1.959963984540054 * std / math.sqrt(delta.size) if delta.size > 1 else 0.0
    correct_lower = int(np.sum(delta < 0.0))
    counterfactual_lower = int(np.sum(delta > 0.0))
    return {
        "correct_minus_counterfactual_mean": mean,
        "normal_95": [mean - radius, mean + radius],
        "correct_lower": correct_lower,
        "counterfactual_lower": counterfactual_lower,
        "ties": int(np.sum(delta == 0.0)),
        "sign_test_exact_p": _exact_sign_p(correct_lower, counterfactual_lower),
    }


def _render_markdown(result: dict) -> str:
    lines = [
        "# DemoVLA prompt counterfactual: paired flow loss",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- samples: {result['num_samples']}",
        f"- fixed gate: {result['fixed_gate']}",
        "- the action prefix always uses the correct prompt",
        "- only the interaction-memory extractor sees the counterfactual prompt",
        "- all conditions share images, state, actions, flow time, and Gaussian noise",
        "- suite-swap replaces the memory prompt with the next task prompt inside its 10-task LIBERO group",
        "",
        "| Condition | Flow loss | Memory cosine to correct | Visual-attention cosine to correct | Top-patch changed |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, row in result["conditions"].items():
        lines.append(
            f"| `{condition}` | {row['flow_loss_mean']:.8f} | {row['memory_cosine_to_correct']:.4f} | "
            f"{row['visual_attention_cosine_to_correct']:.4f} | {row['top_patch_changed_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Correct - counterfactual | 95% normal CI | Correct lower | Counterfactual lower | Sign p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, row in result["paired"].items():
        low, high = row["normal_95"]
        lines.append(
            f"| `correct vs {condition}` | {row['correct_minus_counterfactual_mean']:+.8f} | "
            f"[{low:+.8f}, {high:+.8f}] | {row['correct_lower']} | {row['counterfactual_lower']} | "
            f"{row['sign_test_exact_p']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="demovla_libero_full_stage_a_fixed_gate")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-gate", type=float, default=0.03)
    args = parser.parse_args()

    if not 0.0 < args.fixed_gate < 1.0:
        raise ValueError("fixed gate must be strictly between zero and one")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params missing: {args.checkpoint / 'params'}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    base_config = _config.get_config(args.config_name)
    config = dataclasses.replace(base_config, batch_size=args.batch_size, num_workers=args.num_workers)
    data_root = pathlib.Path(config.data.root)
    tasks = _load_tasks(data_root / "meta" / "tasks.jsonl")
    tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
    task_tokens = [tokenizer.tokenize(task) for task in tasks]
    task_by_token_key = {
        _token_key(tokens, mask): task_index
        for task_index, (tokens, mask) in enumerate(task_tokens)
    }
    swap_indices = _suite_swap_indices(len(tasks))
    blank_tokens, blank_mask = tokenizer.tokenize("")

    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=args.num_batches)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    model.configure_interaction_inference_ablation(
        "layer_mean",
        (args.fixed_gate,) * len(config.model.interaction_injection_layers),
    )
    eval_fn, model_state = _make_eval_fn(model)

    conditions = ("correct", "suite_swap", "blank")
    losses = {condition: [] for condition in conditions}
    memories = {condition: [] for condition in conditions}
    attentions = {condition: [] for condition in conditions}
    task_indices_all = []
    records_path = args.output_dir / "paired_prompt_losses.jsonl"
    started = time.time()
    for batch_index, (observation, actions) in enumerate(loader):
        prompt_tokens = np.asarray(observation.tokenized_prompt)
        prompt_masks = np.asarray(observation.tokenized_prompt_mask)
        task_indices = np.asarray(
            [task_by_token_key[_token_key(tokens, mask)] for tokens, mask in zip(prompt_tokens, prompt_masks, strict=True)],
            dtype=np.int32,
        )
        swapped = [task_tokens[int(swap_indices[index])] for index in task_indices]
        swapped_tokens = np.stack([item[0] for item in swapped])
        swapped_masks = np.stack([item[1] for item in swapped])
        blank_batch_tokens = np.broadcast_to(blank_tokens, prompt_tokens.shape)
        blank_batch_masks = np.broadcast_to(blank_mask, prompt_masks.shape)
        observations = {
            "correct": observation,
            "suite_swap": _replace_prompt(observation, swapped_tokens, swapped_masks),
            "blank": _replace_prompt(observation, blank_batch_tokens, blank_batch_masks),
        }
        rng = jax.random.fold_in(jax.random.key(args.seed), batch_index)
        batch_results = {}
        for condition, conditioned_observation in observations.items():
            loss, memory, attention = jax.device_get(
                eval_fn(model_state, rng, observation, conditioned_observation, actions)
            )
            loss = np.asarray(loss).mean(axis=-1)
            memory = np.asarray(memory)
            attention = np.asarray(attention)
            losses[condition].append(loss)
            memories[condition].append(memory)
            attentions[condition].append(attention)
            batch_results[condition] = loss
        task_indices_all.append(task_indices)
        with records_path.open("a", encoding="utf-8") as stream:
            for sample_index in range(args.batch_size):
                stream.write(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "sample": sample_index,
                            "task_index": int(task_indices[sample_index]),
                            "correct_prompt": tasks[int(task_indices[sample_index])],
                            "suite_swap_prompt": tasks[int(swap_indices[task_indices[sample_index]])],
                            **{
                                condition: float(batch_results[condition][sample_index])
                                for condition in conditions
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        print(f"completed batch {batch_index + 1}/{args.num_batches}", flush=True)

    loss_arrays = {condition: np.concatenate(values) for condition, values in losses.items()}
    memory_arrays = {condition: np.concatenate(values) for condition, values in memories.items()}
    attention_arrays = {condition: np.concatenate(values) for condition, values in attentions.items()}
    correct_memory = memory_arrays["correct"]
    correct_attention = attention_arrays["correct"]
    task_indices_array = np.concatenate(task_indices_all)
    result = {
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "num_samples": args.num_batches * args.batch_size,
        "seed": args.seed,
        "fixed_gate": args.fixed_gate,
        "elapsed_seconds": time.time() - started,
        "conditions": {},
        "paired": {},
    }
    for condition in conditions:
        memory_cosine = _cosine(correct_memory, memory_arrays[condition], axis=-1)
        flat_correct_attention = correct_attention.reshape(*correct_attention.shape[:2], -1)
        flat_condition_attention = attention_arrays[condition].reshape(*attention_arrays[condition].shape[:2], -1)
        attention_cosine = _cosine(flat_correct_attention, flat_condition_attention, axis=-1)
        top_patch_changed = np.argmax(flat_correct_attention, axis=-1) != np.argmax(flat_condition_attention, axis=-1)
        result["conditions"][condition] = {
            "flow_loss_mean": float(np.mean(loss_arrays[condition])),
            "flow_loss_std": float(np.std(loss_arrays[condition], ddof=1)),
            "memory_cosine_to_correct": float(np.mean(memory_cosine)),
            "visual_attention_cosine_to_correct": float(np.mean(attention_cosine)),
            "top_patch_changed_rate": float(np.mean(top_patch_changed)),
        }
        if condition != "correct":
            result["paired"][condition] = _paired_summary(loss_arrays["correct"], loss_arrays[condition])

    result["per_suite"] = {}
    for suite_index, suite_name in enumerate(("libero_10", "libero_goal", "libero_object", "libero_spatial")):
        mask = task_indices_array // 10 == suite_index
        result["per_suite"][suite_name] = {
            condition: _paired_summary(loss_arrays["correct"][mask], loss_arrays[condition][mask])
            for condition in ("suite_swap", "blank")
        }

    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = _render_markdown(result)
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
