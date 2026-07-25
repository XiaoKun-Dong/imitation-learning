"""
Convert local LIBERO Object HDF5 demos to LeRobot format with GT target masks.

This script is tailored for the pi0.5 object-mask experiment. It reads the local
LIBERO Object HDF5 files, restores each recorded MuJoCo state in
SegmentationRenderEnv, extracts the first BDDL obj_of_interest as the manipulated
target, and writes target_mask / target_bbox / target_crop / target_point
alongside the normal RGB, state, action, and task fields.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py

By default this reads:
third_party/libero/LIBERO/libero/datasets/libero_object
"""

from collections.abc import Iterator
import json
import pathlib
import shutil
import xml.etree.ElementTree as ET

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from libero.libero.envs import utils as libero_env_utils
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from robosuite.utils import camera_utils
import tyro

REPO_NAME = "your_hf_username/libero_object"
DEFAULT_DATA_DIR = pathlib.Path("third_party/libero/LIBERO/libero/datasets/libero_object")
DEFAULT_OUTPUT_ROOT = pathlib.Path("data/lerobot")
IMAGE_SIZE = 128
LIBERO_ASSETS_DIR = pathlib.Path("third_party/libero/LIBERO/libero/libero/assets").resolve()


def _iter_hdf5_files(data_dir: pathlib.Path) -> Iterator[pathlib.Path]:
    yield from sorted(path for path in data_dir.glob("*_demo.hdf5") if path.is_file())


def _bddl_path_from_demo_path(demo_path: pathlib.Path) -> pathlib.Path:
    task_name = demo_path.name[: -len("_demo.hdf5")]
    return pathlib.Path(get_libero_path("bddl_files")) / "libero_object" / f"{task_name}.bddl"


def _make_env(bddl_file: pathlib.Path) -> SegmentationRenderEnv:
    env = SegmentationRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
        camera_segmentations="instance",
        camera_depths=True,
    )
    env.reset()
    return env


def _load_demo_xml(env: SegmentationRenderEnv, model_xml: str) -> None:
    # HDF5 demos contain absolute asset paths from the machine that generated them.
    # LIBERO's helper rewrites robosuite/libero asset paths for the local checkout.
    model_xml = libero_env_utils.postprocess_model_xml(model_xml, {}, demo_generation=True)
    model_xml = _rewrite_libero_asset_paths(model_xml)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()


def _rewrite_libero_asset_paths(model_xml: str) -> str:
    """Rewrite legacy LIBERO demo asset paths to this checkout.

    Some released HDF5 demos store paths from the data-generation machine, e.g.
    `/Users/.../chiliocosm/assets/...`. LIBERO's helper handles robosuite paths,
    but not every historical LIBERO asset prefix, so patch file attributes here.
    """
    root = ET.fromstring(model_xml)
    asset = root.find("asset")
    if asset is None:
        return model_xml

    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        file_path = elem.get("file")
        if not file_path:
            continue
        parts = pathlib.PurePosixPath(file_path).parts
        if "assets" not in parts:
            continue
        asset_idx = parts.index("assets")
        local_path = LIBERO_ASSETS_DIR.joinpath(*parts[asset_idx + 1 :])
        if local_path.exists():
            elem.set("file", str(local_path))
    return ET.tostring(root, encoding="unicode")


def _restore_obs(env: SegmentationRenderEnv, state: np.ndarray) -> dict:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def _get_agentview_segmentation(obs: dict) -> np.ndarray:
    for key in ("agentview_segmentation_instance", "agentview_instance_segmentation", "agentview_segmentation"):
        if key in obs:
            return _as_2d_segmentation(obs[key])
    seg_keys = [key for key in obs if "agentview" in key and "seg" in key]
    raise KeyError(f"Could not find agentview segmentation. Available segmentation-like keys: {seg_keys}")


def _get_agentview_depth(obs: dict) -> np.ndarray | None:
    for key in ("agentview_depth", "agentview_depth_image", "agentview_image_depth"):
        if key in obs:
            depth = np.asarray(obs[key])
            if depth.ndim == 3:
                depth = depth[..., 0]
            return depth.astype(np.float32)
    return None


def _as_2d_segmentation(segmentation_image: np.ndarray) -> np.ndarray:
    segmentation_image = np.asarray(segmentation_image)
    if segmentation_image.ndim == 3:
        if segmentation_image.shape[-1] == 1:
            return segmentation_image[..., 0]
        # robosuite segmentation can include multiple channels depending on renderer settings.
        # Instance ids are stored in the first channel for the LIBERO object-mask path.
        return segmentation_image[..., 0]
    return segmentation_image


def _target_mask_from_segmentation(env: SegmentationRenderEnv, segmentation_image: np.ndarray) -> np.ndarray:
    segmentation_image = _as_2d_segmentation(segmentation_image)
    if not env.obj_of_interest:
        return np.zeros(segmentation_image.shape[:2], dtype=bool)
    target_obj = env.obj_of_interest[0]
    target_id = env.instance_to_id[target_obj]
    return segmentation_image == target_id


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return absolute pixel bbox [x1, y1, x2, y2) with x2/y2 exclusive."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _crop_from_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.int32)
    padded = np.zeros_like(image)
    if x2 <= x1 or y2 <= y1:
        return padded
    # Keep the crop in the original image coordinate frame so target_crop,
    # target_mask, and target_bbox all share the same spatial semantics.
    padded[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return padded


def _target_point_from_depth(
    env: SegmentationRenderEnv,
    mask: np.ndarray,
    bbox: np.ndarray,
    depth: np.ndarray | None,
) -> np.ndarray:
    """Return approximate agentview-camera-frame target point [x, y, z] in meters."""
    if depth is None or not np.any(mask):
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


def _save_debug_overlay(
    image: np.ndarray, mask: np.ndarray, bbox: np.ndarray, crop: np.ndarray, output_path: pathlib.Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_pil = Image.fromarray(image).convert("RGB")
    overlay = image_pil.convert("RGBA")
    red_mask = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
    mask_alpha = (mask.astype(np.uint8) * 96)
    red_mask.putalpha(Image.fromarray(mask_alpha, mode="L"))
    overlay = Image.alpha_composite(overlay, red_mask)

    x1, y1, x2, y2 = bbox.astype(np.int32).tolist()
    if x2 > x1 and y2 > y1:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 255, 0, 255), width=2)

    mask_vis = ImageOps.colorize(Image.fromarray(mask.astype(np.uint8) * 255, mode="L"), black="black", white="red")
    crop_pil = Image.fromarray(crop).convert("RGB")
    sheet = Image.new("RGB", (image_pil.width * 4, image_pil.height), "white")
    sheet.paste(image_pil, (0, 0))
    sheet.paste(overlay.convert("RGB"), (image_pil.width, 0))
    sheet.paste(mask_vis, (image_pil.width * 2, 0))
    sheet.paste(crop_pil, (image_pil.width * 3, 0))
    sheet.save(output_path)


def _state_from_demo(demo: h5py.Group, frame_idx: int) -> np.ndarray:
    ee_pos = demo["obs/ee_pos"][frame_idx]
    ee_ori = demo["obs/ee_ori"][frame_idx]
    gripper = demo["obs/gripper_states"][frame_idx]
    return np.concatenate([ee_pos, ee_ori, gripper], axis=0).astype(np.float32)


def _create_dataset(repo_name: str, output_root: pathlib.Path) -> LeRobotDataset:
    output_path = output_root / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "target_mask": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "target_bbox": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["x1_y1_x2_y2_pixel_exclusive"],
            },
            "target_crop": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "target_point": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["x_y_z_agentview_camera"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def main(
    data_dir: pathlib.Path = DEFAULT_DATA_DIR,
    *,
    repo_name: str = REPO_NAME,
    output_root: pathlib.Path = DEFAULT_OUTPUT_ROOT,
    push_to_hub: bool = False,
    max_files: int | None = None,
    max_demos_per_file: int | None = None,
    debug_overlay_dir: pathlib.Path | None = None,
    debug_frames_per_task: int = 0,
):
    hdf5_files = list(_iter_hdf5_files(data_dir))
    if max_files is not None:
        hdf5_files = hdf5_files[:max_files]
    if not hdf5_files:
        raise FileNotFoundError(f"No *_demo.hdf5 files found in {data_dir}")

    dataset = _create_dataset(repo_name, output_root)

    for hdf5_path in hdf5_files:
        bddl_file = _bddl_path_from_demo_path(hdf5_path)
        env = _make_env(bddl_file)
        print(f"Converting {hdf5_path.name} with target={env.obj_of_interest[:1]}")

        with h5py.File(hdf5_path, "r") as f:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            task = problem_info["language_instruction"]
            demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))
            if max_demos_per_file is not None:
                demos = demos[:max_demos_per_file]

            for demo_name in demos:
                demo = f[f"data/{demo_name}"]
                model_xml = demo.attrs["model_file"]
                num_frames = demo["actions"].shape[0]
                _load_demo_xml(env, model_xml)
                debug_saved = 0

                for frame_idx in range(num_frames):
                    obs = _restore_obs(env, demo["states"][frame_idx])
                    seg = _get_agentview_segmentation(obs)
                    target_mask = _target_mask_from_segmentation(env, seg)
                    target_bbox = _bbox_from_mask(target_mask)
                    target_depth = _get_agentview_depth(obs)
                    target_point = _target_point_from_depth(env, target_mask, target_bbox, target_depth)
                    image = demo["obs/agentview_rgb"][frame_idx]
                    target_crop = _crop_from_bbox(image, target_bbox)
                    if debug_overlay_dir is not None and debug_saved < debug_frames_per_task:
                        debug_name = f"{hdf5_path.stem}_{demo_name}_frame_{frame_idx:06d}.png"
                        _save_debug_overlay(image, target_mask, target_bbox, target_crop, debug_overlay_dir / debug_name)
                        debug_saved += 1

                    dataset.add_frame(
                        {
                            "image": image,
                            "wrist_image": demo["obs/eye_in_hand_rgb"][frame_idx],
                            "state": _state_from_demo(demo, frame_idx),
                            "actions": demo["actions"][frame_idx].astype(np.float32),
                            "task": task,
                            "target_mask": np.repeat(target_mask[..., None], 3, axis=-1).astype(np.uint8) * 255,
                            "target_bbox": target_bbox,
                            "target_crop": target_crop,
                            "target_point": target_point,
                        }
                    )
                dataset.save_episode()

        env.close()

    if push_to_hub:
        dataset.push_to_hub(tags=["libero", "panda", "object-mask", "object-3d"], private=False, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
