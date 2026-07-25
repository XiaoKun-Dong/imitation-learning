"""Utilities for rendering DemoVLA interaction-query patch attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import pathlib

import numpy as np
from PIL import Image
from PIL import ImageDraw


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if image.min() < 0.0:
            image = (image + 1.0) / 2.0
        if image.max() <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _attention_overlay(image: np.ndarray, heatmap: np.ndarray, normalizer: float) -> Image.Image:
    base = Image.fromarray(_as_uint8_image(image)).convert("RGBA")
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.clip(heatmap / normalizer, 0.0, 1.0) if normalizer > 0.0 else np.zeros_like(heatmap)
    alpha = Image.fromarray(np.round(heatmap * 150.0).astype(np.uint8)).resize(
        base.size,
        resample=Image.Resampling.BILINEAR,
    )
    red = Image.new("RGBA", base.size, (255, 32, 32, 0))
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def _render_interaction_attention(
    *,
    images: Mapping[str, np.ndarray],
    camera_names: Sequence[str],
    visual_attention: np.ndarray,
    camera_mask: np.ndarray,
    top_patch_view_indices: np.ndarray,
    top_patch_xy: np.ndarray,
    top_patch_weights: np.ndarray,
    patch_grid_shape: Sequence[int],
    output_path: pathlib.Path,
    label: str,
) -> pathlib.Path:
    """Render one query-by-camera attention grid."""
    visual_attention = np.asarray(visual_attention)
    camera_mask = np.asarray(camera_mask, dtype=bool)
    top_patch_view_indices = np.asarray(top_patch_view_indices)
    top_patch_xy = np.asarray(top_patch_xy)
    top_patch_weights = np.asarray(top_patch_weights)
    grid_height, grid_width = (int(value) for value in patch_grid_shape)

    if visual_attention.ndim != 3:
        raise ValueError(f"visual_attention must have shape [queries, views, patches], got {visual_attention.shape}")
    num_queries, num_views, num_patches = visual_attention.shape
    if num_patches != grid_height * grid_width:
        raise ValueError(f"attention has {num_patches} patches but grid is {grid_height}x{grid_width}")
    if len(camera_names) != num_views:
        raise ValueError(f"received {len(camera_names)} camera names for {num_views} views")
    if camera_mask.shape != (num_views,):
        raise ValueError(f"camera_mask must have shape {(num_views,)}, got {camera_mask.shape}")

    valid_view_indices = [index for index in range(num_views) if camera_mask[index]]
    if not valid_view_indices:
        raise ValueError("at least one camera view must be valid")
    top_k = top_patch_view_indices.shape[-1]
    if top_patch_view_indices.shape != (num_queries, top_k):
        raise ValueError("top_patch_view_indices must have shape [queries, top_k]")
    if top_patch_xy.shape != (num_queries, top_k, 2):
        raise ValueError("top_patch_xy must have shape [queries, top_k, 2]")
    if top_patch_weights.shape != (num_queries, top_k):
        raise ValueError("top_patch_weights must have shape [queries, top_k]")

    panel_width, panel_height = 224, 224
    label_height = 24
    prepared_images = {
        name: np.asarray(
            Image.fromarray(_as_uint8_image(images[name])).resize(
                (panel_width, panel_height),
                resample=Image.Resampling.BILINEAR,
            )
        )
        for name in camera_names
        if name in images
    }
    missing_images = [camera_names[index] for index in valid_view_indices if camera_names[index] not in prepared_images]
    if missing_images:
        raise ValueError(f"missing RGB images for camera views: {missing_images}")

    canvas = Image.new(
        "RGB",
        (
            panel_width * len(valid_view_indices),
            (panel_height + label_height) * num_queries,
        ),
        color=(20, 20, 20),
    )
    for query_index in range(num_queries):
        query_attention = visual_attention[query_index]
        normalizer = float(np.max(query_attention[valid_view_indices]))
        for column, view_index in enumerate(valid_view_indices):
            camera_name = camera_names[view_index]
            heatmap = query_attention[view_index].reshape(grid_height, grid_width)
            panel = _attention_overlay(prepared_images[camera_name], heatmap, normalizer)
            draw = ImageDraw.Draw(panel)
            for rank, selected_view in enumerate(top_patch_view_indices[query_index]):
                if int(selected_view) != view_index:
                    continue
                center_x, center_y = top_patch_xy[query_index, rank]
                x0 = round((float(center_x) - 0.5 / grid_width) * panel_width)
                y0 = round((float(center_y) - 0.5 / grid_height) * panel_height)
                x1 = round((float(center_x) + 0.5 / grid_width) * panel_width)
                y1 = round((float(center_y) + 0.5 / grid_height) * panel_height)
                draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
                draw.text(
                    (max(0, x0), max(0, y0 - 11)),
                    f"{rank + 1}:{top_patch_weights[query_index, rank]:.3f}",
                    fill=(255, 255, 0),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

            left = column * panel_width
            top = query_index * (panel_height + label_height)
            canvas.paste(panel, (left, top + label_height))
            ImageDraw.Draw(canvas).text(
                (left + 4, top + 4),
                f"q{query_index} | {camera_name} | {label}",
                fill=(255, 255, 255),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def save_interaction_attention_replan(
    *,
    images: Mapping[str, np.ndarray],
    camera_names: Sequence[str],
    visual_attention: np.ndarray,
    camera_mask: np.ndarray,
    patch_grid_shape: Sequence[int],
    output_dir: pathlib.Path | str,
    stem: str,
    replan_index: int,
    env_step: int,
    top_k: int = 4,
) -> pathlib.Path:
    """Render the interaction attention extracted for one policy replan."""
    visual_attention = np.asarray(visual_attention)
    camera_mask = np.asarray(camera_mask, dtype=bool)
    if visual_attention.ndim != 3:
        raise ValueError(f"visual_attention must have shape [queries, views, patches], got {visual_attention.shape}")

    num_queries, num_views, num_patches = visual_attention.shape
    if camera_mask.shape != (num_views,):
        raise ValueError(f"camera_mask must have shape {(num_views,)}, got {camera_mask.shape}")
    valid_patch_count = int(np.sum(camera_mask)) * num_patches
    if not 0 < top_k <= valid_patch_count:
        raise ValueError(f"top_k must be in [1, {valid_patch_count}], got {top_k}")

    flat_attention = visual_attention.reshape(num_queries, num_views * num_patches)
    flat_valid_mask = np.repeat(camera_mask, num_patches)
    masked_attention = np.where(flat_valid_mask[None, :], flat_attention, -np.inf)
    top_indices = np.argsort(masked_attention, axis=-1)[:, ::-1][:, :top_k]
    top_weights = np.take_along_axis(flat_attention, top_indices, axis=-1)
    top_view_indices = top_indices // num_patches
    patch_indices = top_indices % num_patches

    grid_height, grid_width = (int(value) for value in patch_grid_shape)
    if num_patches != grid_height * grid_width:
        raise ValueError(f"attention has {num_patches} patches but grid is {grid_height}x{grid_width}")
    patch_y = patch_indices // grid_width
    patch_x = patch_indices % grid_width
    top_patch_xy = np.stack(
        [
            (patch_x.astype(np.float32) + 0.5) / grid_width,
            (patch_y.astype(np.float32) + 0.5) / grid_height,
        ],
        axis=-1,
    )

    output_dir = pathlib.Path(output_dir)
    output_path = output_dir / f"{stem}_replan_{replan_index:03d}_step_{env_step:04d}.png"
    return _render_interaction_attention(
        images=images,
        camera_names=camera_names,
        visual_attention=visual_attention,
        camera_mask=camera_mask,
        top_patch_view_indices=top_view_indices,
        top_patch_xy=top_patch_xy,
        top_patch_weights=top_weights,
        patch_grid_shape=patch_grid_shape,
        output_path=output_path,
        label=f"replan={replan_index:03d} step={env_step:04d}",
    )
