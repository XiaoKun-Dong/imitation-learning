"""Replay one LIBERO episode and query matched DemoVLA memory interventions.

The environment follows one designated policy condition. At every replan, all
policy servers receive the exact same observation and stateless flow-noise key.
The script records their action chunks, the simulator state, the correct-memory
diagnostics, and an optional phase-aligned donor memory for later branching.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib

import imageio
import main as _libero_main
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    task_suite_name: str
    task_id: int
    episode_id: int
    output_dir: pathlib.Path
    correct_port: int
    zero_port: int
    knockout_layer4_port: int
    knockout_layer9_port: int
    knockout_layer14_port: int
    host: str = "127.0.0.1"
    execute_condition: str = "correct"
    donor_memory_dir: pathlib.Path | None = None
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    policy_noise_seed: int = 0


def _max_steps(suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[suite]


def _prompt_similarity(first: str, second: str) -> float:
    first_tokens = set(first.casefold().split())
    second_tokens = set(second.casefold().split())
    return len(first_tokens & second_tokens) / max(1, len(first_tokens | second_tokens))


def _hard_negative_prompt(task_suite, task_index: int) -> str:
    correct = str(task_suite.get_task(task_index).language)
    candidates = [
        str(task_suite.get_task(index).language)
        for index in range(task_suite.n_tasks)
        if index != task_index and str(task_suite.get_task(index).language) != correct
    ]
    if not candidates:
        raise ValueError("task suite does not contain a different prompt")
    return max(candidates, key=lambda prompt: (_prompt_similarity(correct, prompt), prompt))


def _donor_path(donor_dir: pathlib.Path, replan_index: int, env_step: int) -> pathlib.Path | None:
    for stem in ("interaction_replan", "replan"):
        exact = donor_dir / f"{stem}_{replan_index:03d}_step_{env_step:04d}.npz"
        if exact.is_file():
            return exact
        matches = sorted(donor_dir.glob(f"{stem}_{replan_index:03d}_step_*.npz"))
        if len(matches) == 1:
            return matches[0]
    return None


def _action_distance(reference: np.ndarray, contender: np.ndarray, replan_steps: int) -> dict[str, float]:
    delta = np.asarray(contender[:replan_steps], dtype=np.float32) - np.asarray(
        reference[:replan_steps], dtype=np.float32
    )
    return {
        "executed_l2": float(np.linalg.norm(delta)),
        "translation_l2": float(np.linalg.norm(delta[:, :3])),
        "rotation_l2": float(np.linalg.norm(delta[:, 3:6])),
        "gripper_l2": float(np.linalg.norm(delta[:, 6])),
        "max_step_l2": float(np.max(np.linalg.norm(delta, axis=-1))),
    }


def run(args: Args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.execute_condition not in {"correct", "zero"}:
        raise ValueError("execute_condition must be 'correct' or 'zero'")
    # Match the official LIBERO runner exactly. Some robosuite components still
    # consume NumPy's global RNG even when the environment itself is seeded.
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True)
    trace_dir = args.output_dir / "replans"
    trace_dir.mkdir()

    benchmark_dict = _libero_main.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_index = args.task_id - 1
    if not 0 <= task_index < task_suite.n_tasks:
        raise ValueError(f"task_id must be in [1, {task_suite.n_tasks}]")
    task = task_suite.get_task(task_index)
    correct_prompt = str(task.language)
    wrong_prompt = _hard_negative_prompt(task_suite, task_index)
    initial_states = task_suite.get_task_init_states(task_index)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError(f"episode_id must be in [0, {len(initial_states) - 1}]")

    clients = {
        "correct": _websocket_client_policy.WebsocketClientPolicy(args.host, args.correct_port),
        "zero": _websocket_client_policy.WebsocketClientPolicy(args.host, args.zero_port),
        "knockout_layer4": _websocket_client_policy.WebsocketClientPolicy(
            args.host, args.knockout_layer4_port
        ),
        "knockout_layer9": _websocket_client_policy.WebsocketClientPolicy(
            args.host, args.knockout_layer9_port
        ),
        "knockout_layer14": _websocket_client_policy.WebsocketClientPolicy(
            args.host, args.knockout_layer14_port
        ),
    }

    manifest = dataclasses.asdict(args)
    manifest.update({"correct_prompt": correct_prompt, "wrong_prompt": wrong_prompt})
    manifest = {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in manifest.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    env, _ = _libero_main._get_libero_env(task, _libero_main.LIBERO_ENV_RESOLUTION, args.seed)  # noqa: SLF001
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])
    action_plan = collections.deque()
    action_plan_source = ""
    replay_images: list[np.ndarray] = []
    replan_index = 0
    t = 0
    done = False
    max_steps = _max_steps(args.task_suite_name)
    trace_index_path = args.output_dir / "trace_index.jsonl"

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(_libero_main.LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )
        replay_images.append(img)

        if not action_plan:
            env_step = t - args.num_steps_wait
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _libero_main._quat2axisangle(obs["robot0_eef_quat"]),  # noqa: SLF001
                    obs["robot0_gripper_qpos"],
                )
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": correct_prompt,
                _base_policy.FLOW_NOISE_SEED_KEY: np.asarray(
                    [args.policy_noise_seed, args.task_id, args.episode_id, replan_index], dtype=np.uint32
                ),
            }
            results = {label: client.infer(element) for label, client in clients.items()}

            wrong_element = dict(element)
            wrong_element[_base_policy.INTERACTION_MEMORY_PROMPT_KEY] = wrong_prompt
            results["wrong_prompt"] = clients["correct"].infer(wrong_element)

            donor_memory = None
            donor_file = None
            if args.donor_memory_dir is not None:
                donor_file = _donor_path(args.donor_memory_dir, replan_index, env_step)
                if donor_file is not None:
                    with np.load(donor_file, allow_pickle=False) as donor:
                        memory_key = "interaction_memory" if "interaction_memory" in donor.files else "current_memory"
                        donor_memory = np.asarray(donor[memory_key], dtype=np.float32)
                    transplant_element = dict(element)
                    transplant_element[_base_policy.INTERACTION_MEMORY_OVERRIDE_KEY] = donor_memory
                    results["transplant"] = clients["correct"].infer(transplant_element)

            condition_names = tuple(results)
            action_chunks = np.stack([np.asarray(results[name]["actions"]) for name in condition_names])
            if "interaction_memory" not in results["correct"]:
                raise RuntimeError("correct policy server must be started with --interaction-diagnostics")
            current_memory = np.asarray(results["correct"]["interaction_memory"], dtype=np.float32)
            sim_state = np.asarray(env.get_sim_state(), dtype=np.float64)
            np.savez_compressed(
                trace_dir / f"replan_{replan_index:03d}_step_{env_step:04d}.npz",
                condition_names=np.asarray(condition_names),
                action_chunks=action_chunks,
                agent_image=img,
                wrist_image=wrist_img,
                robot_state=state,
                sim_state=sim_state,
                current_memory=current_memory,
                donor_memory=np.asarray(donor_memory) if donor_memory is not None else np.empty((0,), dtype=np.float32),
                replan_index=np.asarray(replan_index, dtype=np.int32),
                env_step=np.asarray(env_step, dtype=np.int32),
            )

            correct_actions = np.asarray(results["correct"]["actions"])
            distances = {
                name: _action_distance(correct_actions, np.asarray(result["actions"]), args.replan_steps)
                for name, result in results.items()
                if name != "correct"
            }
            row = {
                "replan_index": replan_index,
                "env_step": env_step,
                "condition_names": condition_names,
                "executed_condition": args.execute_condition,
                "donor_file": str(donor_file) if donor_file is not None else None,
                "distances_from_correct": distances,
            }
            with trace_index_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(row) + "\n")

            action_plan_source = args.execute_condition
            action_plan.extend(np.asarray(results[action_plan_source]["actions"])[: args.replan_steps])
            replan_index += 1

        action = np.asarray(action_plan.popleft())
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    env.close()
    suffix = "success" if done else "failure"
    imageio.mimwrite(args.output_dir / f"rollout_{suffix}.mp4", replay_images, fps=10)
    summary = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "task": correct_prompt,
        "wrong_prompt": wrong_prompt,
        "executed_condition": action_plan_source or args.execute_condition,
        "success": bool(done),
        "steps": t,
        "num_replans": replan_index,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tyro.cli(run)
