"""Continue a LIBERO rollout from a saved key replan under one intervention."""

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
    trace_file: pathlib.Path
    donor_memory_dir: pathlib.Path
    output_dir: pathlib.Path
    condition: str
    correct_port: int
    zero_port: int
    knockout_layer4_port: int
    knockout_layer9_port: int
    knockout_layer14_port: int
    host: str = "127.0.0.1"
    resize_size: int = 224
    replan_steps: int = 5
    seed: int = 7
    policy_noise_seed: int = 0
    replay_from_initial: bool = False


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
    return max(candidates, key=lambda prompt: (_prompt_similarity(correct, prompt), prompt))


def _load_donor(donor_dir: pathlib.Path, replan_index: int) -> tuple[np.ndarray, pathlib.Path]:
    matches = sorted(donor_dir.glob(f"replan_{replan_index:03d}_step_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one donor for replan {replan_index}, found {len(matches)}")
    with np.load(matches[0], allow_pickle=False) as data:
        return np.asarray(data["current_memory"], dtype=np.float32), matches[0]


def run(args: Args) -> None:
    valid_conditions = {
        "zero",
        "correct",
        "wrong_prompt",
        "transplant",
        "one_shot_correct",
        "one_shot_transplant",
        "knockout_layer4",
        "knockout_layer9",
        "knockout_layer14",
    }
    if args.condition not in valid_conditions:
        raise ValueError(f"condition must be one of {sorted(valid_conditions)}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    np.random.seed(args.seed)

    with np.load(args.trace_file, allow_pickle=False) as trace:
        sim_state = np.asarray(trace["sim_state"], dtype=np.float64)
        branch_replan = int(trace["replan_index"])
        branch_env_step = int(trace["env_step"])

    benchmark_dict = _libero_main.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_index = args.task_id - 1
    task = task_suite.get_task(task_index)
    correct_prompt = str(task.language)
    wrong_prompt = _hard_negative_prompt(task_suite, task_index)
    initial_states = task_suite.get_task_init_states(task_index)
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

    env, _ = _libero_main._get_libero_env(task, _libero_main.LIBERO_ENV_RESOLUTION, args.seed)  # noqa: SLF001
    env.reset()
    if args.replay_from_initial:
        obs = env.set_init_state(initial_states[args.episode_id])
        for _ in range(10):
            obs, _, _, _ = env.step(_libero_main.LIBERO_DUMMY_ACTION)
        replan_index = 0
        env_step = 0
    else:
        obs = env.set_init_state(sim_state)
        replan_index = branch_replan
        env_step = branch_env_step
    action_plan = collections.deque()
    replay_images: list[np.ndarray] = []
    max_steps = _max_steps(args.task_suite_name)
    done = False
    donor_exhausted = False
    branch_records = []

    while env_step < max_steps:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )
        replay_images.append(img)

        if not action_plan:
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
            active_condition = args.condition if replan_index >= branch_replan else "zero"
            if args.condition in {"one_shot_correct", "one_shot_transplant"} and replan_index > branch_replan:
                active_condition = "zero"

            donor_file = None
            if active_condition in {"transplant", "one_shot_transplant"}:
                try:
                    donor_memory, donor_file = _load_donor(args.donor_memory_dir, replan_index)
                except FileNotFoundError:
                    donor_exhausted = True
                    break
                element[_base_policy.INTERACTION_MEMORY_OVERRIDE_KEY] = donor_memory
                result = clients["correct"].infer(element)
            elif active_condition == "wrong_prompt":
                element[_base_policy.INTERACTION_MEMORY_PROMPT_KEY] = wrong_prompt
                result = clients["correct"].infer(element)
            else:
                client_condition = "correct" if active_condition == "one_shot_correct" else active_condition
                result = clients[client_condition].infer(element)

            action_chunk = np.asarray(result["actions"])
            action_plan.extend(action_chunk[: args.replan_steps])
            branch_records.append(
                {
                    "replan_index": replan_index,
                    "env_step": env_step,
                    "active_condition": active_condition,
                    "donor_file": str(donor_file) if donor_file is not None else None,
                }
            )
            replan_index += 1

        action = np.asarray(action_plan.popleft())
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        env_step += 1

    env.close()
    suffix = "success" if done else "failure"
    imageio.mimwrite(args.output_dir / f"branch_{suffix}.mp4", replay_images, fps=10)
    with (args.output_dir / "replans.jsonl").open("w", encoding="utf-8") as output:
        for row in branch_records:
            output.write(json.dumps(row) + "\n")
    summary = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "condition": args.condition,
        "branch_replan": branch_replan,
        "branch_env_step": branch_env_step,
        "final_env_step": env_step,
        "success": bool(done),
        "donor_exhausted": donor_exhausted,
        "num_branch_replans": len(branch_records),
        "wrong_prompt": wrong_prompt,
        "replay_from_initial": args.replay_from_initial,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tyro.cli(run)
