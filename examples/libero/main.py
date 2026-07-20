import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Literal

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import SegmentationRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image
from robosuite.utils import camera_utils
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    object_condition: Literal["none", "3d"] = "3d"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    metrics_path = pathlib.Path(args.video_out_path) / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_target_grasps, total_wrong_object_grasps = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_target_grasps, task_wrong_object_grasps = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            target_grasped = False
            wrong_object_grasped = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        if args.object_condition == "3d":
                            target_mask, target_bbox, target_crop, target_point = _get_target_object_condition(
                                env, obs, img, args.resize_size
                            )
                            element.update(
                                {
                                    "target_mask": target_mask,
                                    "target_bbox": target_bbox,
                                    "target_crop": target_crop,
                                    "target_point": target_point,
                                }
                            )

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    step_target_grasped, step_wrong_object_grasped = _get_grasp_state(env)
                    target_grasped |= step_target_grasped
                    wrong_object_grasped |= step_wrong_object_grasped
                    if done:
                        # A successful pick-and-place implies the target was grasped even if a transient
                        # finger contact was missed between evaluation samples.
                        target_grasped = True
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception("Caught exception during rollout")
                    break

            task_episodes += 1
            total_episodes += 1
            task_target_grasps += int(target_grasped)
            total_target_grasps += int(target_grasped)
            task_wrong_object_grasps += int(wrong_object_grasped)
            total_wrong_object_grasps += int(wrong_object_grasped)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx:02d}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            episode_metrics = {
                "task_id": task_id,
                "task": task_description,
                "episode": episode_idx,
                "success": bool(done),
                "target_grasped": target_grasped,
                "wrong_object_grasped": wrong_object_grasped,
                "steps": t,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(episode_metrics) + "\n")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(
                "# target grasp failures: %d (%.1f%%)",
                total_episodes - total_target_grasps,
                (total_episodes - total_target_grasps) / total_episodes * 100,
            )

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current task grasp failure rate: {1.0 - task_target_grasps / task_episodes}")
        logging.info(
            "Current task post-grasp failure rate: %.3f",
            (task_target_grasps - task_successes) / task_target_grasps if task_target_grasps else 0.0,
        )
        logging.info(f"Current task wrong-object grasp rate: {task_wrong_object_grasps / task_episodes}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total grasp failure rate: {1.0 - total_target_grasps / total_episodes}")
    logging.info(
        "Total post-grasp failure rate: %.3f",
        (total_target_grasps - total_successes) / total_target_grasps if total_target_grasps else 0.0,
    )
    logging.info(f"Total wrong-object grasp rate: {total_wrong_object_grasps / total_episodes}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_segmentations": "instance",
        "camera_depths": True,
    }
    env = SegmentationRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _get_target_object_condition(env, obs, image: np.ndarray, resize_size: int):
    """Build target object mask/bbox/crop in the same frame as observation/image.

    LIBERO's BDDL obj_of_interest may include destination objects. For LIBERO Object,
    the first object is the manipulated target, which is the signal we want here.
    """
    seg = _get_agentview_segmentation(obs)
    seg = np.ascontiguousarray(seg[::-1, ::-1])
    raw_target_mask = _target_mask_from_segmentation(env, seg)
    raw_target_bbox = _bbox_from_mask(raw_target_mask)
    depth = np.ascontiguousarray(_get_agentview_depth(obs)[::-1, ::-1])
    target_point = _target_point_from_depth(env, raw_target_mask, raw_target_bbox, depth)

    mask_rgb = np.repeat(raw_target_mask[..., None], 3, axis=-1).astype(np.uint8) * 255
    target_mask = image_tools.resize_with_pad(
        mask_rgb, resize_size, resize_size, method=Image.Resampling.NEAREST
    )
    target_mask = target_mask[..., 0] > 127
    target_bbox = _bbox_from_mask(target_mask)
    target_crop = _crop_from_bbox(image, target_bbox)
    return target_mask, target_bbox, target_crop, target_point


def _get_agentview_segmentation(obs):
    for key in ("agentview_segmentation_instance", "agentview_instance_segmentation", "agentview_segmentation"):
        if key in obs:
            segmentation = np.asarray(obs[key])
            if segmentation.ndim == 3:
                # LIBERO / robosuite may render instance ids as HxWx1 or in the first
                # channel of a multi-channel segmentation image.
                segmentation = segmentation[..., 0]
            return segmentation
    seg_keys = [key for key in obs if "agentview" in key and "seg" in key]
    raise KeyError(f"Could not find agentview segmentation in obs. Available segmentation-like keys: {seg_keys}")


def _get_agentview_depth(obs):
    for key in ("agentview_depth", "agentview_depth_image", "agentview_image_depth"):
        if key in obs:
            depth = np.asarray(obs[key])
            return depth[..., 0] if depth.ndim == 3 else depth
    depth_keys = [key for key in obs if "agentview" in key and "depth" in key]
    raise KeyError(f"Could not find agentview depth in obs. Available depth-like keys: {depth_keys}")


def _target_mask_from_segmentation(env, segmentation_image):
    if not env.obj_of_interest:
        return np.zeros(segmentation_image.shape[:2], dtype=bool)
    target_obj = env.obj_of_interest[0]
    target_id = env.instance_to_id[target_obj]
    return segmentation_image == target_id


def _bbox_from_mask(mask):
    """Return absolute pixel bbox [x1, y1, x2, y2) with x2/y2 exclusive."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _crop_from_bbox(image, bbox):
    x1, y1, x2, y2 = bbox.astype(np.int32)
    padded = np.zeros_like(image)
    if x2 <= x1 or y2 <= y1:
        return padded
    padded[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return padded


def _target_point_from_depth(env, mask, bbox, depth):
    """Return agentview-camera-frame target center [x, y, z] in meters."""
    if not np.any(mask):
        return np.zeros((3,), dtype=np.float32)

    metric_depth = camera_utils.get_real_depth_map(env.sim, depth)
    valid_depth = metric_depth[mask]
    valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0)]
    if valid_depth.size == 0:
        return np.zeros((3,), dtype=np.float32)

    z = np.median(valid_depth).astype(np.float32)
    x1, y1, x2, y2 = bbox.astype(np.float32)
    u = (x1 + x2 - 1.0) * 0.5
    v = (y1 + y2 - 1.0) * 0.5
    height, width = depth.shape[:2]
    fovy = float(env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")])
    fy = 0.5 * height / np.tan(np.deg2rad(fovy) * 0.5)
    fx = fy
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.asarray([x, y, z], dtype=np.float32)


def _get_grasp_state(env):
    """Return whether the robot currently grasps the target or any other movable object."""
    base_env = env.env
    gripper = base_env.robots[0].gripper
    target_name = env.obj_of_interest[0]
    target_grasped = base_env._check_grasp(gripper, base_env.objects_dict[target_name])  # noqa: SLF001
    wrong_object_grasped = any(
        base_env._check_grasp(gripper, obj)  # noqa: SLF001
        for name, obj in base_env.objects_dict.items()
        if name != target_name
    )
    return bool(target_grasped), bool(wrong_object_grasped)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
