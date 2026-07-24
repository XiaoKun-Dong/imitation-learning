import dataclasses
from collections.abc import Mapping
from typing import Protocol

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "target_mask": np.zeros((224, 224), dtype=bool),
        "target_bbox": np.zeros((4,), dtype=np.float32),
        "target_crop": np.zeros((224, 224, 3), dtype=np.uint8),
        "target_point": np.zeros((3,), dtype=np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_mask(mask) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] in (1, 3) and mask.shape[-1] not in (1, 3):
            mask = einops.rearrange(mask, "c h w -> h w c")
        mask = mask[..., 0]
    return mask > 0


def _crop_from_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.int32)
    crop = np.zeros_like(image)
    if x2 <= x1 or y2 <= y1:
        return crop
    height, width = image.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    crop[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return crop


class ObjectConditionProvider(Protocol):
    """Produces online target-object fields from the current base camera image."""

    def __call__(self, image: np.ndarray, *, prompt: str | None = None) -> Mapping[str, np.ndarray | float]: ...


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    # Optional online detector/segmenter hook. For example, this can wrap a YOLO-seg
    # worker and return target_mask, target_bbox, target_crop, and confidence.
    object_condition_provider: ObjectConditionProvider | None = None

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        provider_outputs = {}
        has_spatial_condition = any(key in data for key in ("target_mask", "target_bbox", "target_point"))
        if self.object_condition_provider is not None and not has_spatial_condition:
            provider_outputs = dict(
                self.object_condition_provider(
                    base_image,
                    prompt=data.get("prompt"),
                )
            )

        # Optional object-centric inputs for the pi0.5 mask experiment.
        # These are aligned to observation/image and are consumed only by the action expert.
        object_data = {**provider_outputs, **data}
        if "target_mask" in object_data:
            inputs["target_mask"] = _parse_mask(object_data["target_mask"])
        if "target_bbox" in object_data:
            inputs["target_bbox"] = np.asarray(object_data["target_bbox"], dtype=np.float32)
        if "target_crop" in object_data:
            inputs["target_crop"] = _parse_image(object_data["target_crop"])
        elif "target_bbox" in object_data:
            inputs["target_crop"] = _crop_from_bbox(base_image, inputs["target_bbox"])
        if "target_point" in object_data:
            inputs["target_point"] = np.asarray(object_data["target_point"], dtype=np.float32)
        if "object_condition_confidence" in object_data:
            inputs["object_condition_confidence"] = np.asarray(
                object_data["object_condition_confidence"], dtype=np.float32
            )
        if "object_semantic_tokens" in object_data:
            inputs["object_semantic_tokens"] = np.asarray(object_data["object_semantic_tokens"], dtype=np.float32)

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., :7])}
