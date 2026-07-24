import argparse
import io
import os
import pathlib

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw


def _decode_image(value) -> np.ndarray:
    if isinstance(value, dict) and "bytes" in value:
        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    return np.asarray(value)


def _decode_mask(value) -> np.ndarray:
    image = _decode_image(value)
    if image.ndim == 3:
        image = image[..., 0]
    return image > 127


def _overlay(image: np.ndarray, gt_mask: np.ndarray, yolo_mask: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    yolo_mask = np.asarray(yolo_mask, dtype=bool)
    if gt_mask.shape != image.shape[:2]:
        gt_mask = np.asarray(Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(image.shape[1::-1])) > 127
    if yolo_mask.shape != image.shape[:2]:
        yolo_mask = np.asarray(Image.fromarray(yolo_mask.astype(np.uint8) * 255).resize(image.shape[1::-1])) > 127

    gt_only = gt_mask & ~yolo_mask
    yolo_only = yolo_mask & ~gt_mask
    overlap = gt_mask & yolo_mask
    out = image.copy()
    alpha = 0.45
    colors = {
        "gt": np.asarray([0, 255, 80], dtype=np.float32),
        "yolo": np.asarray([255, 40, 40], dtype=np.float32),
        "overlap": np.asarray([255, 220, 0], dtype=np.float32),
    }
    out[gt_only] = (1 - alpha) * out[gt_only] + alpha * colors["gt"]
    out[yolo_only] = (1 - alpha) * out[yolo_only] + alpha * colors["yolo"]
    out[overlap] = (1 - alpha) * out[overlap] + alpha * colors["overlap"]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_box(draw: ImageDraw.ImageDraw, bbox: np.ndarray, color: tuple[int, int, int], width: int = 2) -> None:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def _add_legend(image: np.ndarray, *, gt_bbox: np.ndarray, yolo_bbox: np.ndarray, confidence: float) -> Image.Image:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    _draw_box(draw, gt_bbox, (0, 255, 80), width=2)
    _draw_box(draw, yolo_bbox, (255, 40, 40), width=2)
    draw.rectangle([4, 4, 126, 54], fill=(0, 0, 0))
    draw.rectangle([10, 12, 22, 24], fill=(0, 255, 80))
    draw.text((28, 9), "GT mask/box", fill=(255, 255, 255))
    draw.rectangle([10, 32, 22, 44], fill=(255, 40, 40))
    draw.text((28, 29), f"YOLO {confidence:.2f}", fill=(255, 255, 255))
    return canvas


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    parser = argparse.ArgumentParser(description="Overlay LIBERO GT masks and YOLO-seg masks on RGB frames.")
    parser.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("data/lerobot/local/libero_object_mask"))
    parser.add_argument("--yolo-model", type=pathlib.Path, default=pathlib.Path("checkpoints/yolo26s-seg.pt"))
    parser.add_argument("--ultralytics-repo", type=pathlib.Path, default=pathlib.Path("ultralytics"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/libero_yolo_overlay"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--class-name", action="append", default=None)
    parser.add_argument("--class-id", action="append", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    episode_path = args.dataset_root / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    df = pd.read_parquet(episode_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from openpi.policies.ultralytics_provider import UltralyticsObjectConditionProvider

    provider = UltralyticsObjectConditionProvider(
        args.yolo_model,
        repo_path=args.ultralytics_repo,
        class_names=args.class_name,
        class_ids=args.class_id,
        conf=args.conf,
        device=args.device,
    )

    indices = list(range(0, len(df), args.stride))[: args.num_frames]
    for frame_index in indices:
        row = df.iloc[frame_index]
        image = _decode_image(row["image"])
        gt_mask = _decode_mask(row["target_mask"])
        gt_bbox = np.asarray(row["target_bbox"], dtype=np.float32)
        prediction = provider(image)
        yolo_mask = prediction["target_mask"]
        yolo_bbox = prediction["target_bbox"]
        confidence = float(np.asarray(prediction["object_condition_confidence"]).reshape(-1)[0])
        overlaid = _overlay(image, gt_mask, yolo_mask)
        canvas = _add_legend(overlaid, gt_bbox=gt_bbox, yolo_bbox=yolo_bbox, confidence=confidence)
        out_path = args.output_dir / f"episode_{args.episode:06d}_frame_{int(row['frame_index']):06d}.png"
        canvas.save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
