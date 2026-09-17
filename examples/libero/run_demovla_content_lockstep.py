"""Run one-shot DemoVLA memory-content interventions from a shared correct prefix.

Every condition owns an independent LIBERO environment. Before the registered
branch replan, the runner queries the correct-memory policy once and applies
the exact same actions to every environment. At the branch, only the memory
condition changes. All conditions return to the correct-memory policy at the
next replan. This isolates the long-term effect of one memory-content pulse.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import pathlib

import imageio
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

CONDITIONS = ("correct", "shuffled", "zero", "injection_off", "wrong_prompt")


def _import_libero_main():
    # Keep LIBERO / MuJoCo initialization out of module import so pure helper
    # tests do not require an EGL device.
    import main as libero_main

    return libero_main


@dataclasses.dataclass
class Args:
    task_suite_name: str
    task_id: int
    episode_id: int
    branch_replan: int
    shuffled_memory_file: pathlib.Path
    output_dir: pathlib.Path
    correct_port: int
    zero_port: int
    injection_off_port: int
    host: str = "127.0.0.1"
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    policy_noise_seed: int = 0
    video_fps: int = 10


def _max_steps(suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[suite]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_memory(path: pathlib.Path) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"shuffled memory file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("current_memory", "interaction_memory"):
            if key in data.files:
                memory = np.asarray(data[key], dtype=np.float32)
                if memory.size == 0:
                    raise ValueError(f"{key} is empty in {path}")
                return memory, key
    raise KeyError(f"expected current_memory or interaction_memory in {path}")


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


def _policy_element(
    obs: dict,
    prompt: str,
    *,
    resize_size: int,
    noise_components: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    libero_main = _import_libero_main()
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            libero_main._quat2axisangle(obs["robot0_eef_quat"]),  # noqa: SLF001
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


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def _validate_args(args: Args) -> None:
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if args.video_fps <= 0:
        raise ValueError("video_fps must be positive")
    max_replans = (_max_steps(args.task_suite_name) + args.replan_steps - 1) // args.replan_steps
    if not 0 <= args.branch_replan < max_replans:
        raise ValueError(f"branch_replan must be in [0, {max_replans - 1}]")


def run(args: Args) -> None:
    _validate_args(args)
    libero_main = _import_libero_main()
    existing_entries = set()
    if args.output_dir.exists():
        existing_entries = {path.name for path in args.output_dir.iterdir()}
    if existing_entries - {"logs"}:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conditions").mkdir()

    shuffled_memory, shuffled_memory_key = _load_memory(args.shuffled_memory_file)
    np.random.seed(args.seed)
    task_suite = libero_main.benchmark.get_benchmark_dict()[args.task_suite_name]()
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
        "injection_off": _websocket_client_policy.WebsocketClientPolicy(args.host, args.injection_off_port),
    }
    manifest = dataclasses.asdict(args)
    manifest.update(
        {
            "conditions": CONDITIONS,
            "prefix_condition": "correct",
            "post_branch_condition": "correct",
            "intervention_duration_replans": 1,
            "correct_prompt": correct_prompt,
            "wrong_prompt": wrong_prompt,
            "shuffled_memory_key": shuffled_memory_key,
            "shuffled_memory_shape": list(shuffled_memory.shape),
            "shuffled_memory_file_sha256": _sha256(args.shuffled_memory_file),
        }
    )
    manifest = {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in manifest.items()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    envs = {}
    observations = {}
    video_writers = {}
    try:
        for condition in CONDITIONS:
            np.random.seed(args.seed)
            env, _ = libero_main._get_libero_env(  # noqa: SLF001
                task, libero_main.LIBERO_ENV_RESOLUTION, args.seed
            )
            env.reset()
            obs = env.set_init_state(initial_states[args.episode_id])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
            envs[condition] = env
            observations[condition] = obs
            condition_dir = args.output_dir / "conditions" / condition
            condition_dir.mkdir()
            video_writers[condition] = imageio.get_writer(
                condition_dir / "rollout.mp4", fps=args.video_fps, codec="libx264"
            )

        reference_state = np.asarray(envs["correct"].get_sim_state())
        initialization_max_abs_diff = max(
            float(np.max(np.abs(np.asarray(env.get_sim_state()) - reference_state))) for env in envs.values()
        )
        if initialization_max_abs_diff != 0.0:
            raise RuntimeError(
                "content-lockstep environments differ immediately after initialization: "
                f"max_abs_diff={initialization_max_abs_diff}"
            )

        action_plans = {condition: collections.deque() for condition in CONDITIONS}
        finished = dict.fromkeys(CONDITIONS, False)
        success = dict.fromkeys(CONDITIONS, False)
        max_steps = _max_steps(args.task_suite_name)
        final_env_step = dict.fromkeys(CONDITIONS, max_steps)
        records = {condition: [] for condition in CONDITIONS}
        trajectory_actions = {condition: [] for condition in CONDITIONS}
        trajectory_states = {condition: [] for condition in CONDITIONS}
        prefix_max_abs_diff = 0.0
        prefix_action_chunks = []
        branch_payload = {}
        replan_index = 0
        env_step = 0

        while env_step < max_steps and not all(finished.values()):
            active_conditions = [condition for condition in CONDITIONS if not finished[condition]]
            empty_active_plans = [condition for condition in active_conditions if not action_plans[condition]]
            if empty_active_plans and len(empty_active_plans) != len(active_conditions):
                raise RuntimeError("active conditions lost the shared replan schedule")

            if empty_active_plans:
                noise_components = np.asarray(
                    [args.policy_noise_seed, args.task_id, args.episode_id, replan_index], dtype=np.uint32
                )
                if replan_index < args.branch_replan:
                    element, image, state = _policy_element(
                        observations["correct"],
                        correct_prompt,
                        resize_size=args.resize_size,
                        noise_components=noise_components,
                    )
                    actions = np.asarray(clients["correct"].infer(element)["actions"])
                    prefix_action_chunks.append(actions)
                    for condition in active_conditions:
                        action_plans[condition].extend(actions[: args.replan_steps])
                        records[condition].append(
                            {
                                "replan_index": replan_index,
                                "env_step": env_step,
                                "active_condition": "correct_shared_prefix",
                                "image_sha256": _array_sha256(image),
                                "state_sha256": _array_sha256(state),
                            }
                        )
                else:
                    for condition in active_conditions:
                        element, image, state = _policy_element(
                            observations[condition],
                            correct_prompt,
                            resize_size=args.resize_size,
                            noise_components=noise_components,
                        )
                        active_condition = "correct"
                        if replan_index == args.branch_replan:
                            active_condition = condition
                            if condition == "shuffled":
                                element[_base_policy.INTERACTION_MEMORY_OVERRIDE_KEY] = shuffled_memory
                                result = clients["correct"].infer(element)
                            elif condition == "wrong_prompt":
                                element[_base_policy.INTERACTION_MEMORY_PROMPT_KEY] = wrong_prompt
                                result = clients["correct"].infer(element)
                            else:
                                result = clients[condition].infer(element)
                            branch_payload[f"{condition}_image"] = image
                            branch_payload[f"{condition}_robot_state"] = state
                            branch_payload[f"{condition}_action_chunk"] = np.asarray(result["actions"])
                        else:
                            result = clients["correct"].infer(element)
                        actions = np.asarray(result["actions"])
                        action_plans[condition].extend(actions[: args.replan_steps])
                        records[condition].append(
                            {
                                "replan_index": replan_index,
                                "env_step": env_step,
                                "active_condition": active_condition,
                                "image_sha256": _array_sha256(image),
                                "state_sha256": _array_sha256(state),
                            }
                        )
                replan_index += 1

            for condition in active_conditions:
                action = np.asarray(action_plans[condition].popleft())
                trajectory_actions[condition].append(action)
                obs, _, done, _ = envs[condition].step(action.tolist())
                observations[condition] = obs
                trajectory_states[condition].append(np.asarray(envs[condition].get_sim_state()))
                frame = image_tools.convert_to_uint8(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                video_writers[condition].append_data(frame)
                if done:
                    finished[condition] = True
                    success[condition] = True
                    final_env_step[condition] = env_step + 1

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
                            "shared correct prefix lost exact lockstep before branch: "
                            f"env_step={env_step} max_abs_diff={step_max_abs_diff}"
                        )
            env_step += 1

        for condition in CONDITIONS:
            condition_dir = args.output_dir / "conditions" / condition
            with (condition_dir / "replans.jsonl").open("w", encoding="utf-8") as output:
                for row in records[condition]:
                    output.write(json.dumps(row) + "\n")
            np.savez_compressed(
                condition_dir / "trajectory.npz",
                actions=np.asarray(trajectory_actions[condition]),
                simulator_states=np.asarray(trajectory_states[condition]),
            )

        condition_summaries = {
            condition: {
                "success": success[condition],
                "termination": "success" if success[condition] else "timeout",
                "final_env_step": final_env_step[condition],
                "num_replans": len(records[condition]),
            }
            for condition in CONDITIONS
        }
        intervention_reached = all(
            f"{condition}_action_chunk" in branch_payload for condition in CONDITIONS
        )
        if not intervention_reached:
            np.savez_compressed(
                args.output_dir / "prefix_audit.npz",
                prefix_correct_action_chunks=np.asarray(prefix_action_chunks),
            )
            summary = {
                "status": "intervention_not_reached",
                "intervention_reached": False,
                "task_suite_name": args.task_suite_name,
                "task_id": args.task_id,
                "episode_id": args.episode_id,
                "task": correct_prompt,
                "wrong_prompt": wrong_prompt,
                "branch_replan": args.branch_replan,
                "branch_env_step": args.branch_replan * args.replan_steps,
                "prefix_condition": "correct",
                "post_branch_condition": "correct",
                "intervention_duration_replans": 1,
                "initialization_max_abs_diff": initialization_max_abs_diff,
                "prefix_max_abs_diff": prefix_max_abs_diff,
                "branch_image_equal": None,
                "branch_state_equal": None,
                "branch_image_sha256": None,
                "branch_action_distances_from_correct": None,
                "conditions": condition_summaries,
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            return

        branch_images = [branch_payload[f"{condition}_image"] for condition in CONDITIONS]
        branch_states = [branch_payload[f"{condition}_robot_state"] for condition in CONDITIONS]
        branch_image_equal = all(np.array_equal(branch_images[0], image) for image in branch_images[1:])
        branch_state_equal = all(np.array_equal(branch_states[0], state) for state in branch_states[1:])
        branch_action_chunks = {
            condition: np.asarray(branch_payload[f"{condition}_action_chunk"]) for condition in CONDITIONS
        }
        branch_payload["prefix_correct_action_chunks"] = np.asarray(prefix_action_chunks)
        np.savez_compressed(args.output_dir / "branch_audit.npz", **branch_payload)

        action_distances = {
            condition: _action_distance(
                branch_action_chunks["correct"], branch_action_chunks[condition], args.replan_steps
            )
            for condition in CONDITIONS
            if condition != "correct"
        }
        summary = {
            "status": "completed",
            "intervention_reached": True,
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "episode_id": args.episode_id,
            "task": correct_prompt,
            "wrong_prompt": wrong_prompt,
            "branch_replan": args.branch_replan,
            "branch_env_step": args.branch_replan * args.replan_steps,
            "prefix_condition": "correct",
            "post_branch_condition": "correct",
            "intervention_duration_replans": 1,
            "initialization_max_abs_diff": initialization_max_abs_diff,
            "prefix_max_abs_diff": prefix_max_abs_diff,
            "branch_image_equal": branch_image_equal,
            "branch_state_equal": branch_state_equal,
            "branch_image_sha256": _array_sha256(branch_images[0]),
            "branch_action_distances_from_correct": action_distances,
            "conditions": condition_summaries,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        for writer in video_writers.values():
            writer.close()
        for env in envs.values():
            env.close()


if __name__ == "__main__":
    run(tyro.cli(Args))
