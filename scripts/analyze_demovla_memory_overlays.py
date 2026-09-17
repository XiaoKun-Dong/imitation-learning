"""Summarize temporal and slot diversity in saved DemoVLA memory overlays."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def _mean_off_diagonal_cosine(distributions: np.ndarray) -> float:
    normalized = distributions / np.maximum(np.linalg.norm(distributions, axis=-1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    upper = np.triu_indices(similarity.shape[0], k=1)
    return float(np.mean(similarity[upper]))


def summarize(root: pathlib.Path) -> None:
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no overlay npz files under {root}")

    entropies = []
    top4_masses = []
    slot_cosines = []
    temporal_cosines = []
    camera_masses = []
    pairwise_cosines = []
    action_reads = []
    action_head_reads = []
    effective_reads = []
    adapter_gates = []
    adapter_injection_ratios = []
    injection_layers = None
    previous = None
    prompt = None
    for path in paths:
        with np.load(path) as payload:
            attention = np.asarray(payload["visual_attention"], dtype=np.float64)
            camera_mask = np.asarray(payload["camera_mask"], dtype=bool)
            current_prompt = str(payload["prompt"].item())
            if "action_to_memory_attention" in payload:
                action_reads.append(np.asarray(payload["action_to_memory_attention"], dtype=np.float64))
                if "action_to_memory_head_attention" in payload:
                    action_head_reads.append(
                        np.asarray(payload["action_to_memory_head_attention"], dtype=np.float64)
                    )
                effective_reads.append(np.asarray(payload["effective_visual_attention"], dtype=np.float64))
                adapter_gates.append(np.asarray(payload["adapter_gate"], dtype=np.float64))
                adapter_injection_ratios.append(
                    np.asarray(payload["adapter_injection_ratio"], dtype=np.float64)
                )
                injection_layers = np.asarray(payload["injection_layers"], dtype=np.int32)
        if prompt is None:
            prompt = current_prompt
        elif current_prompt != prompt:
            raise ValueError(f"multiple prompts found under {root}")

        valid_attention = attention[:, camera_mask, :]
        distributions = valid_attention.reshape(valid_attention.shape[0], -1)
        distributions /= np.maximum(np.sum(distributions, axis=-1, keepdims=True), 1e-12)
        entropy = -np.sum(distributions * np.log(np.maximum(distributions, 1e-12)), axis=-1)
        entropies.extend((entropy / np.log(distributions.shape[-1])).tolist())
        top4_masses.extend(np.sort(distributions, axis=-1)[:, -4:].sum(axis=-1).tolist())
        slot_cosines.append(_mean_off_diagonal_cosine(distributions))
        normalized = distributions / np.maximum(np.linalg.norm(distributions, axis=-1, keepdims=True), 1e-12)
        pairwise_cosines.append(normalized @ normalized.T)
        camera_masses.append(np.sum(attention, axis=-1))
        if previous is not None:
            numerator = np.sum(previous * distributions, axis=-1)
            denominator = np.linalg.norm(previous, axis=-1) * np.linalg.norm(distributions, axis=-1)
            temporal_cosines.extend((numerator / np.maximum(denominator, 1e-12)).tolist())
        previous = distributions

    camera_masses_array = np.stack(camera_masses)
    pairwise_cosines_array = np.mean(np.stack(pairwise_cosines), axis=0)
    print(f"# {root.name}")
    print(f"prompt: {prompt}")
    print(f"replans: {len(paths)}")
    print(f"normalized_attention_entropy_mean: {np.mean(entropies):.4f}")
    print(f"top4_attention_mass_mean: {np.mean(top4_masses):.4f}")
    print(f"inter_slot_attention_cosine_mean: {np.mean(slot_cosines):.4f}")
    print(f"same_slot_consecutive_replan_cosine_mean: {np.mean(temporal_cosines):.4f}")
    print("mean_pairwise_slot_attention_cosine:")
    for row in pairwise_cosines_array:
        print("  " + " ".join(f"{value:.3f}" for value in row))
    for slot_index, masses in enumerate(np.mean(camera_masses_array, axis=0)):
        mass_text = ", ".join(f"camera_{index}={mass:.3f}" for index, mass in enumerate(masses))
        print(f"slot_{slot_index}_camera_mass: {mass_text}")
    if action_reads:
        action_reads_array = np.stack(action_reads)
        action_entropy = -np.sum(
            action_reads_array * np.log(np.maximum(action_reads_array, 1e-12)),
            axis=-1,
        )
        dominant_slots = np.argmax(action_reads_array, axis=-1)
        slot_usage = np.mean(action_reads_array, axis=(0, 1, 2, 3))
        dominant_frequency = np.bincount(
            dominant_slots.reshape(-1),
            minlength=action_reads_array.shape[-1],
        ) / dominant_slots.size
        print(f"action_to_memory_entropy_mean: {np.mean(action_entropy):.4f}")
        print("action_to_memory_slot_usage: " + " ".join(f"q{i}={x:.4f}" for i, x in enumerate(slot_usage)))
        print(
            "action_to_memory_dominant_frequency: "
            + " ".join(f"q{i}={x:.4f}" for i, x in enumerate(dominant_frequency))
        )
        for layer_index, layer in enumerate(injection_layers):
            layer_usage = np.mean(action_reads_array[:, :, layer_index], axis=(0, 1, 2))
            print(
                f"layer_{int(layer)}_slot_usage: "
                + " ".join(f"q{i}={x:.4f}" for i, x in enumerate(layer_usage))
            )
        effective_array = np.stack(effective_reads)
        layer_effective = np.mean(effective_array, axis=(1, 3))
        layer_effective = layer_effective.reshape(layer_effective.shape[0], layer_effective.shape[1], -1)
        layer_effective /= np.maximum(np.linalg.norm(layer_effective, axis=-1, keepdims=True), 1e-12)
        layer_similarity = np.einsum("nld,nkd->nlk", layer_effective, layer_effective)
        upper = np.triu_indices(layer_similarity.shape[-1], k=1)
        print(f"effective_attention_inter_layer_cosine_mean: {np.mean(layer_similarity[:, upper[0], upper[1]]):.4f}")
        print(f"adapter_gate_mean: {np.mean(np.stack(adapter_gates)):.4f}")
        print(f"adapter_injection_ratio_mean: {np.mean(np.stack(adapter_injection_ratios)):.4f}")
        if action_head_reads:
            head_reads_array = np.stack(action_head_reads)
            # [replan, flow, layer, head, action, memory]
            head_usage = np.mean(head_reads_array, axis=(0, 1, 2, 4))
            head_entropy = -np.sum(
                head_reads_array * np.log(np.maximum(head_reads_array, 1e-12)),
                axis=-1,
            )
            print(f"per_head_action_to_memory_entropy_mean: {np.mean(head_entropy):.4f}")
            for head, usage in enumerate(head_usage):
                print(
                    f"head_{head}_slot_usage: "
                    + " ".join(f"q{i}={x:.4f}" for i, x in enumerate(usage))
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    for root in args.roots:
        summarize(root)


if __name__ == "__main__":
    main()
