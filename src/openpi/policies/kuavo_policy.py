"""Input and output transforms for the Kuavo 5W right-arm policy."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_kuavo_right_example() -> dict:
    """Create an example observation matching the Kuavo deployment interface."""
    return {
        "observation.state": np.random.rand(8).astype(np.float32),
        "observation.images.head_cam_h": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation.images.wrist_cam_r": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "prompt": "Pick and Place",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    # The Kuavo ROS deployment returns torch tensors with a leading singleton
    # batch dimension, while the training transforms operate on individual
    # samples. Accept both representations at this boundary.
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.max(initial=0.0) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Kuavo RGB image must have shape HWC or CHW with three channels, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class KuavoRightArmOnly(transforms.DataTransformFn):
    """Select the right arm from a 16-D left+right Kuavo sample.

    Existing 8-D right-arm datasets pass through unchanged. This keeps the
    deployment interface and normalization statistics consistently 8-D.
    """

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        for key in ("observation.state", "action"):
            if key not in result:
                continue
            value = np.asarray(result[key])
            if value.dtype == object:
                value = np.asarray(value.tolist())
            if value.shape[-1] == 16:
                result[key] = value[..., 8:16]
            elif value.shape[-1] != 8:
                raise ValueError(f"Kuavo {key} must be 8-D or 16-D, got {value.shape}")
        return result


@dataclasses.dataclass(frozen=True)
class KuavoRightInputs(transforms.DataTransformFn):
    """Map head/right-wrist RGB and 8-D right-arm state into OpenPI inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["cam_h"])
        right_wrist_image = _parse_image(data["cam_r"])
        state = np.asarray(data["state"])
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.shape != (8,):
            raise ValueError(f"Kuavo right-arm state must be 8-D, got {state.shape}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 8:
                raise ValueError(f"Kuavo right-arm actions must be 8-D, got {actions.shape}")
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class KuavoRightOutputs(transforms.DataTransformFn):
    """Return seven right-arm joints and one Leju-claw command."""

    def __call__(self, data: dict) -> dict:
        outputs = {"actions": np.asarray(data["actions"])[..., :8]}
        outputs.update({key: value for key, value in data.items() if key.startswith("interaction_")})
        return outputs
