"""Utilities for rendering DemoVLA interaction-query patch attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import pathlib
import textwrap

import numpy as np
from PIL import Image
from PIL import ImageDraw


# Paper-oriented rendering defaults. The scene is intentionally desaturated
# against white so that attention magnitude, rather than image texture or
# top-patch decorations, is the primary visual signal.
_BASE_IMAGE_OPACITY = 0.45
_ATTENTION_COLOR = (178, 0, 34)
_ATTENTION_MAX_ALPHA = 238
_ATTENTION_GAMMA = 0.65


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
    base_rgb = _as_uint8_image(image).astype(np.float32)
    base_rgb = np.rint(255.0 * (1.0 - _BASE_IMAGE_OPACITY) + base_rgb * _BASE_IMAGE_OPACITY).astype(np.uint8)
    base = Image.fromarray(base_rgb).convert("RGBA")
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.clip(heatmap / normalizer, 0.0, 1.0) if normalizer > 0.0 else np.zeros_like(heatmap)
    # Gamma < 1 preserves low-but-meaningful mass while making high-attention
    # tokens visibly darker than the underlying RGB scene.
    heatmap = np.power(heatmap, _ATTENTION_GAMMA)
    alpha = Image.fromarray(np.round(heatmap * _ATTENTION_MAX_ALPHA).astype(np.uint8)).resize(
        base.size,
        resample=Image.Resampling.BILINEAR,
    )
    red = Image.new("RGBA", base.size, (*_ATTENTION_COLOR, 0))
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def _resize_with_pad_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an RGB image using the same letterbox geometry as model preprocessing."""
    image = _as_uint8_image(image)
    source_height, source_width = image.shape[:2]
    ratio = max(source_width / width, source_height / height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    pad_top, remainder_height = divmod(height - resized_height, 2)
    pad_left, remainder_width = divmod(width - resized_width, 2)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(resized, (pad_left, pad_top))

    # Keep the same asymmetric remainder convention as ``resize_with_pad``.
    assert pad_top + resized_height + pad_top + remainder_height == height
    assert pad_left + resized_width + pad_left + remainder_width == width
    return np.asarray(canvas)


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
    title: str | None = None,
    row_labels: Sequence[str] | None = None,
    show_top_patch_annotations: bool = False,
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
    if row_labels is None:
        row_labels = [f"mem q{index}" for index in range(num_queries)]
    if len(row_labels) != num_queries:
        raise ValueError(f"received {len(row_labels)} row labels for {num_queries} attention rows")
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
        name: _resize_with_pad_image(images[name], panel_height, panel_width)
        for name in camera_names
        if name in images
    }
    missing_images = [camera_names[index] for index in valid_view_indices if camera_names[index] not in prepared_images]
    if missing_images:
        raise ValueError(f"missing RGB images for camera views: {missing_images}")

    canvas_width = panel_width * len(valid_view_indices)
    title_lines = textwrap.wrap(title, width=max(24, canvas_width // 7)) if title else []
    title_height = 8 + 18 * len(title_lines) if title_lines else 0
    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            title_height + (panel_height + label_height) * num_queries,
        ),
        color=(20, 20, 20),
    )
    canvas_draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(title_lines):
        canvas_draw.text(
            (6, 4 + 18 * line_index),
            line,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    for query_index in range(num_queries):
        query_attention = visual_attention[query_index]
        normalizer = float(np.max(query_attention[valid_view_indices]))
        for column, view_index in enumerate(valid_view_indices):
            camera_name = camera_names[view_index]
            heatmap = query_attention[view_index].reshape(grid_height, grid_width)
            panel = _attention_overlay(prepared_images[camera_name], heatmap, normalizer)
            draw = ImageDraw.Draw(panel)
            if show_top_patch_annotations:
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
            top = title_height + query_index * (panel_height + label_height)
            canvas.paste(panel, (left, top + label_height))
            camera_mass = float(np.sum(query_attention[view_index]))
            camera_label = {
                "base_0_rgb": "base",
                "left_wrist_0_rgb": "left-wrist",
                "right_wrist_0_rgb": "right-wrist",
            }.get(camera_name, camera_name)
            canvas_draw.text(
                (left + 4, top + 4),
                f"{row_labels[query_index]} | {camera_label} | mass={camera_mass:.3f}",
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
    prompt: str | None = None,
    save_raw: bool = False,
    action_to_memory_attention: np.ndarray | None = None,
    action_to_memory_head_attention: np.ndarray | None = None,
    interaction_memory: np.ndarray | None = None,
    effective_visual_attention: np.ndarray | None = None,
    adapter_gate: np.ndarray | None = None,
    adapter_injection_ratio: np.ndarray | None = None,
    flow_times: np.ndarray | None = None,
    injection_layers: np.ndarray | None = None,
    show_top_patch_annotations: bool = False,
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
    title_parts = []
    if prompt:
        title_parts.append(f"Task: {prompt}")
    title_parts.append(f"replan={replan_index:03d} step={env_step:04d}")
    rendered_path = _render_interaction_attention(
        images=images,
        camera_names=camera_names,
        visual_attention=visual_attention,
        camera_mask=camera_mask,
        top_patch_view_indices=top_view_indices,
        top_patch_xy=top_patch_xy,
        top_patch_weights=top_weights,
        patch_grid_shape=patch_grid_shape,
        output_path=output_path,
        title=" | ".join(title_parts),
        show_top_patch_annotations=show_top_patch_annotations,
    )
    if save_raw:
        raw_payload = {
            "visual_attention": visual_attention,
            "camera_mask": camera_mask,
            "camera_names": np.asarray(camera_names),
            "patch_grid_shape": np.asarray(patch_grid_shape, dtype=np.int32),
            "prompt": np.asarray(prompt or ""),
            "replan_index": np.asarray(replan_index, dtype=np.int32),
            "env_step": np.asarray(env_step, dtype=np.int32),
        }
        optional_payload = {
            "action_to_memory_attention": action_to_memory_attention,
            "action_to_memory_head_attention": action_to_memory_head_attention,
            "interaction_memory": interaction_memory,
            "effective_visual_attention": effective_visual_attention,
            "adapter_gate": adapter_gate,
            "adapter_injection_ratio": adapter_injection_ratio,
            "flow_times": flow_times,
            "injection_layers": injection_layers,
        }
        raw_payload.update({key: np.asarray(value) for key, value in optional_payload.items() if value is not None})
        np.savez_compressed(output_path.with_suffix(".npz"), **raw_payload)
    return rendered_path


def save_effective_interaction_attention_replan(
    *,
    images: Mapping[str, np.ndarray],
    camera_names: Sequence[str],
    effective_visual_attention: np.ndarray,
    action_to_memory_attention: np.ndarray,
    camera_mask: np.ndarray,
    patch_grid_shape: Sequence[int],
    injection_layers: Sequence[int],
    output_dir: pathlib.Path | str,
    stem: str,
    replan_index: int,
    env_step: int,
    prompt: str | None = None,
    top_k: int = 4,
    show_top_patch_annotations: bool = False,
) -> pathlib.Path:
    """Render end-to-end action-conditioned patch attention for one replan.

    Flow steps and action positions are averaged for the compact panel. The
    unaggregated arrays remain available in the adjacent raw ``.npz`` file.
    """
    effective_visual_attention = np.asarray(effective_visual_attention, dtype=np.float32)
    action_to_memory_attention = np.asarray(action_to_memory_attention, dtype=np.float32)
    camera_mask = np.asarray(camera_mask, dtype=bool)
    injection_layers = np.asarray(injection_layers, dtype=np.int32)
    if effective_visual_attention.ndim != 5:
        raise ValueError(
            "effective_visual_attention must have shape [flow, layers, action, views, patches], "
            f"got {effective_visual_attention.shape}"
        )
    if action_to_memory_attention.shape[:3] != effective_visual_attention.shape[:3]:
        raise ValueError("action-to-memory and effective attention axes do not match")
    _, num_layers, _, num_views, num_patches = effective_visual_attention.shape
    if injection_layers.shape != (num_layers,):
        raise ValueError(f"injection_layers must have shape {(num_layers,)}, got {injection_layers.shape}")
    if camera_mask.shape != (num_views,):
        raise ValueError(f"camera_mask must have shape {(num_views,)}, got {camera_mask.shape}")

    layer_attention = np.mean(effective_visual_attention, axis=(0, 2))
    flat_attention = layer_attention.reshape(num_layers, num_views * num_patches)
    flat_valid_mask = np.repeat(camera_mask, num_patches)
    masked_attention = np.where(flat_valid_mask[None, :], flat_attention, -np.inf)
    top_indices = np.argsort(masked_attention, axis=-1)[:, ::-1][:, :top_k]
    top_weights = np.take_along_axis(flat_attention, top_indices, axis=-1)
    top_view_indices = top_indices // num_patches
    patch_indices = top_indices % num_patches
    grid_height, grid_width = (int(value) for value in patch_grid_shape)
    patch_y = patch_indices // grid_width
    patch_x = patch_indices % grid_width
    top_patch_xy = np.stack(
        [
            (patch_x.astype(np.float32) + 0.5) / grid_width,
            (patch_y.astype(np.float32) + 0.5) / grid_height,
        ],
        axis=-1,
    )

    slot_usage = np.mean(action_to_memory_attention, axis=(0, 2))
    title_parts = []
    if prompt:
        title_parts.append(f"Task: {prompt}")
    title_parts.append(f"effective action attention | replan={replan_index:03d} step={env_step:04d}")
    title_parts.extend(
        f"L{int(layer)} slots=" + ",".join(f"q{slot}:{weight:.2f}" for slot, weight in enumerate(usage))
        for layer, usage in zip(injection_layers, slot_usage, strict=True)
    )
    output_dir = pathlib.Path(output_dir)
    output_path = output_dir / f"{stem}_replan_{replan_index:03d}_step_{env_step:04d}.png"
    return _render_interaction_attention(
        images=images,
        camera_names=camera_names,
        visual_attention=layer_attention,
        camera_mask=camera_mask,
        top_patch_view_indices=top_view_indices,
        top_patch_xy=top_patch_xy,
        top_patch_weights=top_weights,
        patch_grid_shape=patch_grid_shape,
        output_path=output_path,
        title=" | ".join(title_parts),
        row_labels=[f"layer {int(layer)}" for layer in injection_layers],
        show_top_patch_annotations=show_top_patch_annotations,
    )
