"""Run matched DemoVLA interventions from one exactly shared zero-policy prefix.

Each condition owns an independent LIBERO environment. Before ``branch_replan``
the runner queries the zero-memory policy once and applies the same actions to
every environment, checking simulator states after every step. At the branch,
conditions receive their own memory/gate intervention. This avoids restoring an
incomplete MuJoCo state and makes any within-run outcome difference causal to
the post-branch policy intervention.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import pathlib

import main as _libero_main
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

CONDITIONS = (
    "zero",
    "correct",
    "wrong_prompt",
    "transplant",
    "one_shot_correct",
    "one_shot_wrong_prompt",
    "one_shot_transplant",
    "one_shot_wrong_phase",
    "knockout_layer4",
    "knockout_layer9",
    "knockout_layer14",
)


@dataclasses.dataclass
class Args:
    task_suite_name: str
    task_id: int
    episode_id: int
    branch_replan: int
    donor_memory_dir: pathlib.Path
    output_dir: pathlib.Path
    correct_port: int
    zero_port: int
    knockout_layer4_port: int
    knockout_layer9_port: int
    knockout_layer14_port: int
    host: str = "127.0.0.1"
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    policy_noise_seed: int = 0
    wrong_phase_replan: int = 74


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


def _load_donor(donor_dir: pathlib.Path, replan_index: int) -> tuple[np.ndarray, pathlib.Path]:
    matches = sorted(donor_dir.glob(f"replan_{replan_index:03d}_step_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one donor for replan {replan_index}, found {len(matches)}")
    with np.load(matches[0], allow_pickle=False) as data:
        return np.asarray(data["current_memory"], dtype=np.float32), matches[0]


def _policy_element(
    obs: dict,
    prompt: str,
    *,
    resize_size: int,
    noise_components: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _libero_main._quat2axisangle(obs["robot0_eef_quat"]),  # noqa: SLF001
            obs["robot0_gripper_qpos"],
        )
    )
    return (
        {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": prompt,
            _base_policy.FLOW_NOISE_SEED_KEY: noise_components,
        },
        image,
        state,
    )


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(args: Args) -> None:
    if args.output_dir.exists():
        existing_entries = {path.name for path in args.output_dir.iterdir()}
        if existing_entries - {"logs"}:
            raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.branch_replan < 0:
        raise ValueError("branch_replan must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conditions").mkdir()

    np.random.seed(args.seed)
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
        "knockout_layer4": _websocket_client_policy.WebsocketClientPolicy(args.host, args.knockout_layer4_port),
        "knockout_layer9": _websocket_client_policy.WebsocketClientPolicy(args.host, args.knockout_layer9_port),
        "knockout_layer14": _websocket_client_policy.WebsocketClientPolicy(args.host, args.knockout_layer14_port),
    }

    envs = {}
    observations = {}
    try:
        for condition in CONDITIONS:
            # Reset the global RNG before each construction so every environment
            # receives the same initialization-side random stream.
            np.random.seed(args.seed)
            env, _ = _libero_main._get_libero_env(  # noqa: SLF001
                task, _libero_main.LIBERO_ENV_RESOLUTION, args.seed
            )
            env.reset()
            obs = env.set_init_state(initial_states[args.episode_id])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(_libero_main.LIBERO_DUMMY_ACTION)
            envs[condition] = env
            observations[condition] = obs

        reference_state = np.asarray(envs["zero"].get_sim_state())
        initialization_max_abs_diff = max(
            float(np.max(np.abs(np.asarray(env.get_sim_state()) - reference_state))) for env in envs.values()
        )
        if initialization_max_abs_diff != 0.0:
            raise RuntimeError(
                "lockstep environments differ immediately after initialization: "
                f"max_abs_diff={initialization_max_abs_diff}"
            )

        action_plans = {condition: collections.deque() for condition in CONDITIONS}
        finished = dict.fromkeys(CONDITIONS, False)
        success = dict.fromkeys(CONDITIONS, False)
        final_env_step = {condition: _max_steps(args.task_suite_name) for condition in CONDITIONS}
        donor_exhausted = dict.fromkeys(CONDITIONS, False)
        records = {condition: [] for condition in CONDITIONS}
        prefix_max_abs_diff = 0.0
        branch_payload: dict[str, np.ndarray] = {}
        prefix_zero_action_chunks: list[np.ndarray] = []
        env_step = 0
        replan_index = 0
        max_steps = _max_steps(args.task_suite_name)

        while env_step < max_steps and not all(finished.values()):
            if replan_index < args.branch_replan and not action_plans["zero"]:
                element, image, state = _policy_element(
                    observations["zero"],
                    correct_prompt,
                    resize_size=args.resize_size,
                    noise_components=np.asarray(
                        [args.policy_noise_seed, args.task_id, args.episode_id, replan_index],
                        dtype=np.uint32,
                    ),
                )
                actions = np.asarray(clients["zero"].infer(element)["actions"])
                prefix_zero_action_chunks.append(actions)
                for condition in CONDITIONS:
                    action_plans[condition].extend(actions[: args.replan_steps])
                    records[condition].append(
                        {
                            "replan_index": replan_index,
                            "env_step": env_step,
                            "active_condition": "zero_shared_prefix",
                        }
                    )
                if replan_index == 0:
                    branch_payload["initial_image"] = image
                    branch_payload["initial_robot_state"] = state
                replan_index += 1

            if replan_index >= args.branch_replan:
                active_replan = replan_index
                # All action plans become empty together at the first branch.
                if any(not finished[c] and not action_plans[c] for c in CONDITIONS):
                    for condition in CONDITIONS:
                        if finished[condition] or action_plans[condition]:
                            continue
                        element, image, state = _policy_element(
                            observations[condition],
                            correct_prompt,
                            resize_size=args.resize_size,
                            noise_components=np.asarray(
                                [args.policy_noise_seed, args.task_id, args.episode_id, active_replan],
                                dtype=np.uint32,
                            ),
                        )
                        active_condition = condition
                        if (
                            condition
                            in {
                                "one_shot_correct",
                                "one_shot_wrong_prompt",
                                "one_shot_transplant",
                                "one_shot_wrong_phase",
                            }
                            and active_replan > args.branch_replan
                        ):
                            active_condition = "zero"

                        donor_file = None
                        if active_condition in {"transplant", "one_shot_transplant"}:
                            try:
                                donor_memory, donor_file = _load_donor(args.donor_memory_dir, active_replan)
                            except FileNotFoundError:
                                donor_exhausted[condition] = True
                                finished[condition] = True
                                continue
                            element[_base_policy.INTERACTION_MEMORY_OVERRIDE_KEY] = donor_memory
                            result = clients["correct"].infer(element)
                        elif active_condition == "one_shot_wrong_phase":
                            try:
                                donor_memory, donor_file = _load_donor(args.donor_memory_dir, args.wrong_phase_replan)
                            except FileNotFoundError:
                                donor_exhausted[condition] = True
                                finished[condition] = True
                                continue
                            element[_base_policy.INTERACTION_MEMORY_OVERRIDE_KEY] = donor_memory
                            result = clients["correct"].infer(element)
                        elif active_condition in {"wrong_prompt", "one_shot_wrong_prompt"}:
                            element[_base_policy.INTERACTION_MEMORY_PROMPT_KEY] = wrong_prompt
                            result = clients["correct"].infer(element)
                        else:
                            client_condition = "correct" if active_condition == "one_shot_correct" else active_condition
                            result = clients[client_condition].infer(element)

                        actions = np.asarray(result["actions"])
                        action_plans[condition].extend(actions[: args.replan_steps])
                        records[condition].append(
                            {
                                "replan_index": active_replan,
                                "env_step": env_step,
                                "active_condition": active_condition,
                                "donor_file": str(donor_file) if donor_file is not None else None,
                            }
                        )
                        if active_replan == args.branch_replan:
                            branch_payload[f"{condition}_image"] = image
                            branch_payload[f"{condition}_robot_state"] = state
                            branch_payload[f"{condition}_action_chunk"] = actions
                    replan_index += 1

            for condition in CONDITIONS:
                if finished[condition]:
                    continue
                if not action_plans[condition]:
                    raise RuntimeError(f"empty action plan for active condition {condition}")
                action = np.asarray(action_plans[condition].popleft())
                obs, _, done, _ = envs[condition].step(action.tolist())
                observations[condition] = obs
                if done:
                    finished[condition] = True
                    success[condition] = True
                    final_env_step[condition] = env_step

            if replan_index <= args.branch_replan:
                active_states = [
                    np.asarray(envs[condition].get_sim_state()) for condition in CONDITIONS if not finished[condition]
                ]
                if active_states:
                    reference = active_states[0]
                    step_max_abs_diff = max(float(np.max(np.abs(state - reference))) for state in active_states)
                    prefix_max_abs_diff = max(prefix_max_abs_diff, step_max_abs_diff)
                    if step_max_abs_diff != 0.0:
                        raise RuntimeError(
                            "shared zero prefix lost exact lockstep before branch: "
                            f"env_step={env_step} max_abs_diff={step_max_abs_diff}"
                        )
            env_step += 1

        for condition in CONDITIONS:
            condition_dir = args.output_dir / "conditions" / condition
            condition_dir.mkdir()
            with (condition_dir / "replans.jsonl").open("w", encoding="utf-8") as output:
                for row in records[condition]:
                    output.write(json.dumps(row) + "\n")

        branch_payload["prefix_zero_action_chunks"] = np.stack(prefix_zero_action_chunks)
        np.savez_compressed(args.output_dir / "branch_audit.npz", **branch_payload)
        branch_images = [branch_payload[f"{condition}_image"] for condition in CONDITIONS]
        branch_states = [branch_payload[f"{condition}_robot_state"] for condition in CONDITIONS]
        branch_image_equal = all(np.array_equal(branch_images[0], image) for image in branch_images[1:])
        branch_state_equal = all(np.array_equal(branch_states[0], state) for state in branch_states[1:])
        summary = {
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "episode_id": args.episode_id,
            "task": correct_prompt,
            "wrong_prompt": wrong_prompt,
            "branch_replan": args.branch_replan,
            "branch_env_step": args.branch_replan * args.replan_steps,
            "wrong_phase_replan": args.wrong_phase_replan,
            "initialization_max_abs_diff": initialization_max_abs_diff,
            "prefix_max_abs_diff": prefix_max_abs_diff,
            "branch_image_equal": branch_image_equal,
            "branch_state_equal": branch_state_equal,
            "branch_image_sha256": _digest(branch_images[0]),
            "conditions": {
                condition: {
                    "success": success[condition],
                    "final_env_step": final_env_step[condition],
                    "num_replans": len(records[condition]),
                    "donor_exhausted": donor_exhausted[condition],
                }
                for condition in CONDITIONS
            },
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        for env in envs.values():
            env.close()


if __name__ == "__main__":
    tyro.cli(run)
