"""Render DemoVLA attention for selected frames from a Kuavo LeRobot v3 dataset."""

from __future__ import annotations

import dataclasses
from io import BytesIO
import json
import logging
import pathlib
import subprocess

import numpy as np
from PIL import Image
import polars as pl
import tyro

from openpi.models import demovla_visualization
from openpi.policies import policy_config
from openpi.training import config as train_config

_CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_r",
)


@dataclasses.dataclass
class Args:
    dataset_root: str = "/home/dongxiaokun/小件钢圈上料/lerobot"
    checkpoint_dir: str = "/home/dongxiaokun/checkpoints/xiaojianshangliao_dynamic_gate_v3_retry1/29999"
    config_name: str = "demovla_kuavo_right_dynamic_gate"
    episode_index: int = 0
    frame_indices: tuple[int, ...] = (35, 72)
    output_dir: str = "outputs/kuavo_dataset_attention"
    top_k: int = 4


def _decode_video_frame(video_path: pathlib.Path, timestamp: float) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    encoded = subprocess.run(command, check=True, capture_output=True).stdout
    return np.asarray(Image.open(BytesIO(encoded)).convert("RGB"))


def _read_episode_frame(
    dataset_root: pathlib.Path,
    episode_index: int,
    frame_index: int,
    fps: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_rows = pl.read_parquet(episodes_path).filter(pl.col("episode_index") == episode_index)
    if episode_rows.height != 1:
        raise ValueError(f"expected one metadata row for episode {episode_index}, got {episode_rows.height}")
    episode = episode_rows.row(0, named=True)
    if not 0 <= frame_index < int(episode["length"]):
        raise ValueError(f"frame {frame_index} is outside episode {episode_index} length {episode['length']}")

    data_path = dataset_root / (
        f"data/chunk-{int(episode['data/chunk_index']):03d}/file-{int(episode['data/file_index']):03d}.parquet"
    )
    frame_rows = pl.read_parquet(data_path).filter(
        (pl.col("episode_index") == episode_index) & (pl.col("frame_index") == frame_index)
    )
    if frame_rows.height != 1:
        raise ValueError(f"expected one data row for episode {episode_index} frame {frame_index}")
    frame = frame_rows.row(0, named=True)

    images = {}
    for camera_key in _CAMERA_KEYS:
        chunk_index = int(episode[f"videos/{camera_key}/chunk_index"])
        file_index = int(episode[f"videos/{camera_key}/file_index"])
        from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
        video_path = dataset_root / f"videos/{camera_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        images[camera_key] = _decode_video_frame(video_path, from_timestamp + frame_index / fps)

    task = str(episode["tasks"][0])
    return images, np.asarray(frame["observation.state"], dtype=np.float32), task


def main(args: Args) -> None:
    if not args.frame_indices:
        raise ValueError("at least one frame index is required")
    dataset_root = pathlib.Path(args.dataset_root)
    info = json.loads((dataset_root / "meta/info.json").read_text())
    fps = float(info["fps"])

    # Visual attention is computed by a diagnostics-only prefix pass and does
    # not depend on the flow sampler. Two denoising steps keep this offline
    # visualization fast while preserving the exact attention diagnostics.
    policy = policy_config.create_trained_policy(
        train_config.get_config(args.config_name),
        args.checkpoint_dir,
        sample_kwargs={"interaction_diagnostics": True, "num_steps": 2},
    )

    output_dir = pathlib.Path(args.output_dir)
    for replan_index, frame_index in enumerate(args.frame_indices):
        raw_images, state, prompt = _read_episode_frame(
            dataset_root,
            args.episode_index,
            frame_index,
            fps,
        )
        result = policy.infer(
            {
                "cam_h": raw_images["observation.images.head_cam_h"],
                "cam_r": raw_images["observation.images.wrist_cam_r"],
                "state": state,
                "prompt": prompt,
            }
        )
        images = {
            "base_0_rgb": raw_images["observation.images.head_cam_h"],
            "right_wrist_0_rgb": raw_images["observation.images.wrist_cam_r"],
        }
        output_path = demovla_visualization.save_interaction_attention_replan(
            images=images,
            camera_names=result["interaction_camera_names"],
            visual_attention=result["interaction_visual_attention"],
            camera_mask=result["interaction_camera_mask"],
            patch_grid_shape=result["interaction_patch_grid_shape"],
            output_dir=output_dir,
            stem=f"episode_{args.episode_index:04d}_frame_{frame_index:04d}",
            replan_index=replan_index,
            env_step=frame_index,
            top_k=args.top_k,
        )
        logging.info("Rendered dataset attention: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
