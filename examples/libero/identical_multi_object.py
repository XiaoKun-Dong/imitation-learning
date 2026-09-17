"""Continuous LIBERO stress test with multiple identical objects.

The task places several instances of one LIBERO object asset in a cluttered
floor workspace. The policy is asked to put every instance in the basket. An
instance is counted as complete after it is released inside the basket, then
parked outside the workspace. Parking implements an effectively unbounded bin
and prevents the policy from repeatedly selecting an already completed item.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import pathlib
import re

import imageio
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from openpi.shared import libero_runtime as _libero_runtime

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
_OBJECT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    object_type: str = "bbq_sauce"
    num_objects: int = 6
    prompt: str | None = None
    num_trials: int = 10
    max_steps: int = 1800
    num_steps_wait: int = 10

    # Shared clutter region in the LIBERO floor workspace. The standard basket
    # remains at y=0.25, outside this region.
    clutter_x_min: float = -0.225
    clutter_y_min: float = -0.27
    clutter_x_max: float = 0.18
    clutter_y_max: float = 0.10

    # A target switch is recorded only when the new nearest item is clearly
    # closer, which suppresses nearest-neighbor jitter between adjacent items.
    approach_distance: float = 0.16
    approach_switch_margin: float = 0.02

    seed: int = 7
    policy_noise_seed: int | None = None
    video_out_path: str = "data/libero/videos/identical_multi_object"
    video_fps: int = 20
    live_preview: bool = False


def _validate_args(args: Args) -> None:
    if not _OBJECT_TYPE_PATTERN.fullmatch(args.object_type):
        raise ValueError("object_type must be a lowercase LIBERO category such as 'bbq_sauce'")
    if args.num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if args.video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if args.policy_noise_seed is not None and not 0 <= args.policy_noise_seed <= np.iinfo(np.uint32).max:
        raise ValueError("policy_noise_seed must be in [0, 2**32 - 1] or None")
    if not args.clutter_x_min < args.clutter_x_max or not args.clutter_y_min < args.clutter_y_max:
        raise ValueError("clutter region minima must be smaller than maxima")


def _object_names(object_type: str, num_objects: int) -> tuple[str, ...]:
    return tuple(f"{object_type}_{index}" for index in range(1, num_objects + 1))


def build_bddl(args: Args) -> tuple[str, str, tuple[str, ...]]:
    """Return generated BDDL, policy prompt, and identical object names."""
    _validate_args(args)
    object_names = _object_names(args.object_type, args.num_objects)
    object_label = args.object_type.replace("_", " ")
    prompt = args.prompt or (
        f"Pick up every {object_label} item from the floor and place it in the basket, "
        "one item at a time. Continue until all items are in the basket."
    )
    object_declaration = " ".join(object_names)
    object_init = "\n".join(f"    (On {name} floor_clutter_region)" for name in object_names)
    object_goals = "\n".join(f"      (In {name} basket_1_contain_region)" for name in object_names)
    objects_of_interest = "\n".join(f"    {name}" for name in object_names)

    bddl = f"""(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {prompt})
  (:regions
    (bin_region
      (:target floor)
      (:ranges ((-0.01 0.25 0.01 0.27)))
    )
    (clutter_region
      (:target floor)
      (:ranges (({args.clutter_x_min} {args.clutter_y_min} {args.clutter_x_max} {args.clutter_y_max})))
    )
    (contain_region
      (:target basket_1)
    )
  )
  (:fixtures
    floor - floor
  )
  (:objects
    {object_declaration} - {args.object_type}
    basket_1 - basket
  )
  (:obj_of_interest
{objects_of_interest}
    basket_1
  )
  (:init
{object_init}
    (On basket_1 floor_bin_region)
  )
  (:goal
    (And
{object_goals}
    )
  )
)
"""
    return bddl, prompt, object_names


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if denominator < 1e-6:
        return np.zeros(3)
    return quat[:3] * (2.0 * np.arccos(quat[3]) / denominator)


def _policy_observation(obs: dict, prompt: str, resize_size: int) -> tuple[dict, np.ndarray]:
    image = _agentview_frame(obs)
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
    element = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
    }
    return element, image


def _agentview_frame(obs: dict) -> np.ndarray:
    return image_tools.convert_to_uint8(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))


def _show_live_frame(frame: np.ndarray, episode_idx: int) -> bool:
    """Display an RGB frame and return False when the user presses q or Esc."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("live_preview requires opencv-python") from error
    cv2.imshow(f"DemoVLA identical-object rollout {episode_idx}", frame[:, :, ::-1])
    return cv2.waitKey(1) & 0xFF not in {ord("q"), 27}


def _grasped_object(base_env, active_names: set[str]) -> str | None:
    gripper = base_env.robots[0].gripper
    for name in sorted(active_names):
        if base_env._check_grasp(gripper, base_env.objects_dict[name]):  # noqa: SLF001
            return name
    return None


def _objects_released_in_basket(base_env, active_names: set[str], grasped_name: str | None) -> list[str]:
    return [
        name
        for name in sorted(active_names)
        if name != grasped_name
        and base_env._eval_predicate(["in", name, "basket_1_contain_region"])  # noqa: SLF001
    ]


def _park_object(base_env, object_name: str, completed_index: int) -> None:
    """Move a completed free-joint object outside the visible workspace."""
    obj = base_env.objects_dict[object_name]
    joint_name = obj.joints[-1]
    qpos = np.asarray(base_env.sim.data.get_joint_qpos(joint_name)).copy()
    qpos[:3] = np.array([2.0 + 0.15 * completed_index, 2.0, 0.15])
    base_env.sim.data.set_joint_qpos(joint_name, qpos)
    try:
        qvel = np.asarray(base_env.sim.data.get_joint_qvel(joint_name)).copy()
        base_env.sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except (AttributeError, KeyError):
        logging.debug("Could not zero joint velocity for %s", object_name, exc_info=True)
    base_env.sim.forward()


def _refresh_observation(env):
    env._post_process()  # noqa: SLF001
    env._update_observables(force=True)  # noqa: SLF001
    return env.env._get_observations()  # noqa: SLF001


def _object_distances(base_env, obs: dict, active_names: set[str]) -> dict[str, float]:
    eef_pos = np.asarray(obs["robot0_eef_pos"])
    distances = {}
    for name in active_names:
        obj = base_env.objects_dict[name]
        body_id = base_env.sim.model.body_name2id(obj.root_body)
        distances[name] = float(np.linalg.norm(eef_pos - base_env.sim.data.body_xpos[body_id]))
    return distances


def _clutter_restart_target(
    distances: dict[str, float],
    object_xy: dict[str, np.ndarray],
    clutter_bounds: tuple[float, float, float, float],
    distance_threshold: float,
) -> str | None:
    """Return the nearest approached object that is still in the clutter area."""
    x_min, y_min, x_max, y_max = clutter_bounds
    candidates = {
        name: distance
        for name, distance in distances.items()
        if distance <= distance_threshold
        and name in object_xy
        and x_min <= object_xy[name][0] <= x_max
        and y_min <= object_xy[name][1] <= y_max
    }
    return min(candidates, key=candidates.get) if candidates else None


def _object_xy(base_env, active_names: set[str]) -> dict[str, np.ndarray]:
    positions = {}
    for name in active_names:
        obj = base_env.objects_dict[name]
        body_id = base_env.sim.model.body_name2id(obj.root_body)
        positions[name] = np.asarray(base_env.sim.data.body_xpos[body_id, :2]).copy()
    return positions


def _update_approach_target(
    distances: dict[str, float],
    current_target: str | None,
    distance_threshold: float,
    switch_margin: float,
) -> tuple[str | None, bool]:
    if not distances:
        return None, False
    nearest_name = min(distances, key=distances.get)
    nearest_distance = distances[nearest_name]
    if nearest_distance > distance_threshold:
        return current_target, False
    if current_target is None or current_target not in distances:
        return nearest_name, False
    if nearest_name == current_target:
        return current_target, False
    if nearest_distance + switch_margin < distances[current_target]:
        return nearest_name, True
    return current_target, False


def eval_identical_multi_object(args: Args) -> None:
    bddl, prompt, object_names = build_bddl(args)
    _, _, offscreen_render_env = _libero_runtime.import_modules(pathlib.Path(__file__).resolve().parents[2])
    output_dir = pathlib.Path(args.video_out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    bddl_path = output_dir / f"identical_{args.object_type}_{args.num_objects}.bddl"
    bddl_path.write_text(bddl, encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    env = offscreen_render_env(
        bddl_file_name=str(bddl_path),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        horizon=args.max_steps + args.num_steps_wait + 1,
        ignore_done=True,
    )
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    try:
        for episode_idx in tqdm.trange(args.num_trials):
            episode_seed = args.seed + episode_idx
            np.random.seed(episode_seed)
            env.seed(episode_seed)
            obs = env.reset()
            action_plan: collections.deque[np.ndarray] = collections.deque()
            active_names = set(object_names)
            completed_names: list[str] = []
            ever_grasped: set[str] = set()
            grasp_order: list[str] = []
            placement_steps: dict[str, int] = {}
            approach_target: str | None = None
            approach_switch_count = 0
            post_placement_restart_events: list[dict] = []
            pending_restart_from_step: int | None = None
            last_grasped_name: str | None = None
            replan_index = 0
            control_step = 0
            stopped_by_user = False
            video_path = output_dir / f"rollout_episode_{episode_idx:03d}.mp4"
            video_writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264")

            try:
                for env_step in range(args.max_steps + args.num_steps_wait):
                    if env_step < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        frame = _agentview_frame(obs)
                        video_writer.append_data(frame)
                        if args.live_preview and not _show_live_frame(frame, episode_idx):
                            stopped_by_user = True
                            break
                        continue

                    if not action_plan:
                        element, _ = _policy_observation(obs, prompt, args.resize_size)
                        if args.policy_noise_seed is not None:
                            element[_base_policy.FLOW_NOISE_SEED_KEY] = np.asarray(
                                [args.policy_noise_seed, episode_idx, replan_index, 0], dtype=np.uint32
                            )
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"Policy returned {len(action_chunk)} actions, "
                                f"fewer than replan_steps={args.replan_steps}"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])
                        replan_index += 1

                    obs, _, _, _ = env.step(action_plan.popleft().tolist())
                    control_step += 1
                    base_env = env.env
                    grasped_name = _grasped_object(base_env, active_names)
                    if grasped_name is not None:
                        ever_grasped.add(grasped_name)
                        if grasped_name != last_grasped_name:
                            grasp_order.append(grasped_name)
                    last_grasped_name = grasped_name

                    distances = _object_distances(base_env, obs, active_names)
                    approach_target, switched = _update_approach_target(
                        distances,
                        approach_target,
                        args.approach_distance,
                        args.approach_switch_margin,
                    )
                    approach_switch_count += int(switched)

                    if pending_restart_from_step is not None:
                        restart_target = _clutter_restart_target(
                            distances,
                            _object_xy(base_env, active_names),
                            (
                                args.clutter_x_min,
                                args.clutter_y_min,
                                args.clutter_x_max,
                                args.clutter_y_max,
                            ),
                            args.approach_distance,
                        )
                        if restart_target is not None:
                            post_placement_restart_events.append(
                                {
                                    "after_placement_step": pending_restart_from_step,
                                    "restart_step": control_step,
                                    "latency_steps": control_step - pending_restart_from_step,
                                    "target": restart_target,
                                }
                            )
                            logging.info(
                                "Episode %d: returned to grasp range at step %d, %d steps after placement (%s)",
                                episode_idx,
                                control_step,
                                control_step - pending_restart_from_step,
                                restart_target,
                            )
                            pending_restart_from_step = None

                    newly_completed = _objects_released_in_basket(base_env, active_names, grasped_name)
                    if newly_completed:
                        for name in newly_completed:
                            active_names.remove(name)
                            completed_names.append(name)
                            placement_steps[name] = control_step
                            _park_object(base_env, name, len(completed_names))
                        obs = _refresh_observation(env)
                        action_plan.clear()
                        approach_target = None
                        last_grasped_name = None
                        pending_restart_from_step = control_step
                        logging.info(
                            "Episode %d: completed %d/%d at step %d (%s)",
                            episode_idx,
                            len(completed_names),
                            len(object_names),
                            control_step,
                            ", ".join(newly_completed),
                        )

                    frame = _agentview_frame(obs)
                    video_writer.append_data(frame)
                    if args.live_preview and not _show_live_frame(frame, episode_idx):
                        stopped_by_user = True
                        break

                    if not active_names:
                        break
            finally:
                video_writer.close()

            if args.live_preview:
                import cv2

                cv2.destroyAllWindows()

            success = not active_names
            episode_metrics = {
                "episode": episode_idx,
                "episode_seed": episode_seed,
                "object_type": args.object_type,
                "num_objects": len(object_names),
                "success": success,
                "stopped_by_user": stopped_by_user,
                "video_path": str(video_path),
                "completed_count": len(completed_names),
                "completed_order": completed_names,
                "remaining_objects": sorted(active_names),
                "ever_grasped_count": len(ever_grasped),
                "ever_grasped": sorted(ever_grasped),
                "grasp_order": grasp_order,
                "placement_steps": placement_steps,
                "approach_switch_count": approach_switch_count,
                "post_placement_restart_events": post_placement_restart_events,
                "pending_restart_from_step": pending_restart_from_step,
                "control_steps": control_step,
                "replans": replan_index,
                "policy_noise_seed": args.policy_noise_seed,
                "prompt": prompt,
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(episode_metrics) + "\n")
            logging.info(
                "Episode %d result: success=%s, completed=%d/%d, approach_switches=%d, video=%s",
                episode_idx,
                success,
                len(completed_names),
                len(object_names),
                approach_switch_count,
                video_path,
            )
            if stopped_by_user:
                break
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_identical_multi_object)
