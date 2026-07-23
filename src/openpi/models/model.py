import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)

# Target bboxes are always absolute pixel coordinates in the current image frame:
# [x1, y1, x2, y2], with x2/y2 exclusive. They are not normalized to [0, 1].
TARGET_BBOX_FORMAT = "xyxy_pixel_exclusive"


def _resize_target_mask(mask: at.Array, image_resolution: tuple[int, int]) -> at.Bool[at.Array, "*b h w"]:
    mask = jnp.asarray(mask)
    resized = image_tools.resize_with_pad(
        mask[..., None].astype(jnp.float32),
        *image_resolution,
        method=jax.image.ResizeMethod.NEAREST,
    )
    return resized[..., 0] > 0.5


def _resize_target_crop(crop: at.Array, image_resolution: tuple[int, int]) -> at.Array:
    crop = jnp.asarray(crop)
    if crop.shape[-1] != 3:
        raise ValueError(f"target_crop must have 3 channels, got shape {crop.shape}")
    return image_tools.resize_with_pad(crop, *image_resolution)


def _resize_target_bbox(
    bbox: at.Array,
    original_hw: tuple[int, int],
    image_resolution: tuple[int, int],
) -> at.Array:
    """Resize an absolute pixel bbox [x1, y1, x2, y2) through resize-with-pad geometry."""
    bbox = jnp.asarray(bbox, dtype=jnp.float32)
    has_bbox = jnp.any(jnp.abs(bbox) > 0, axis=-1)
    cur_height, cur_width = original_hw
    target_height, target_width = image_resolution
    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    pad_h0 = (target_height - resized_height) // 2
    pad_w0 = (target_width - resized_width) // 2

    scale = jnp.asarray([resized_width / cur_width, resized_height / cur_height] * 2, dtype=jnp.float32)
    offset = jnp.asarray([pad_w0, pad_h0, pad_w0, pad_h0], dtype=jnp.float32)
    resized_bbox = bbox * scale + offset
    return jnp.where(has_bbox[..., None], resized_bbox, 0.0)


def _bbox_from_target_mask(mask: at.Bool[at.Array, "b h w"]) -> at.Float[at.Array, "b 4"]:
    """Return absolute [x1, y1, x2, y2) boxes for a batch of masks."""
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    rows = jnp.any(mask, axis=2)
    cols = jnp.any(mask, axis=1)
    has_mask = jnp.any(mask, axis=(1, 2))
    y1 = jnp.argmax(rows, axis=1)
    x1 = jnp.argmax(cols, axis=1)
    y2 = mask.shape[1] - jnp.argmax(rows[:, ::-1], axis=1)
    x2 = mask.shape[2] - jnp.argmax(cols[:, ::-1], axis=1)
    bbox = jnp.stack([x1, y1, x2, y2], axis=1).astype(jnp.float32)
    return jnp.where(has_mask[:, None], bbox, 0.0)


def _bbox_to_corners(bbox: at.Float[at.Array, "b 4"]) -> at.Float[at.Array, "b 4 2"]:
    """Convert xyxy-exclusive boxes to Augmax (y, x) keypoints."""
    bbox = jnp.asarray(bbox, dtype=jnp.float32)
    x1, y1, x2, y2 = jnp.moveaxis(bbox, -1, 0)
    x2 = jnp.maximum(x1, x2 - 1.0)
    y2 = jnp.maximum(y1, y2 - 1.0)
    return jnp.stack(
        [
            jnp.stack([y1, x1], axis=-1),
            jnp.stack([y1, x2], axis=-1),
            jnp.stack([y2, x1], axis=-1),
            jnp.stack([y2, x2], axis=-1),
        ],
        axis=1,
    )


def _bbox_from_corners(
    corners: at.Float[at.Array, "b 4 2"], has_bbox: at.Bool[at.Array, " b"], image_resolution: tuple[int, int]
) -> at.Float[at.Array, "b 4"]:
    """Convert transformed (y, x) corners back to clipped xyxy-exclusive boxes."""
    height, width = image_resolution
    ys = corners[..., 0]
    xs = corners[..., 1]
    x1 = jnp.clip(jnp.floor(jnp.min(xs, axis=1)), 0, width)
    y1 = jnp.clip(jnp.floor(jnp.min(ys, axis=1)), 0, height)
    x2 = jnp.clip(jnp.ceil(jnp.max(xs, axis=1)) + 1.0, 0, width)
    y2 = jnp.clip(jnp.ceil(jnp.max(ys, axis=1)) + 1.0, 0, height)
    bbox = jnp.stack([x1, y1, x2, y2], axis=1).astype(jnp.float32)
    return jnp.where(has_bbox[:, None], bbox, 0.0)


def _augment_base_image_and_object_condition(
    rng: at.KeyArrayLike,
    image: at.Float[at.Array, "b h w 3"],
    target_mask: at.Bool[at.Array, "b h w"] | None,
    target_crop: at.Float[at.Array, "b h w 3"] | None,
    target_bbox: at.Float[at.Array, "b 4"] | None,
) -> tuple[at.Float[at.Array, "b h w 3"], at.Array | None, at.Array | None, at.Array | None]:
    """Apply one sampled geometric transform to base RGB and all spatial object inputs."""
    height, width = image.shape[1:3]
    inputs = {"image": image / 2.0 + 0.5}
    input_types = {"image": augmax.InputType.IMAGE}
    if target_mask is not None:
        inputs["target_mask"] = target_mask[..., None].astype(jnp.float32)
        input_types["target_mask"] = augmax.InputType.MASK
    if target_crop is not None:
        inputs["target_crop"] = target_crop
        input_types["target_crop"] = augmax.InputType.IMAGE
    has_bbox = None
    if target_bbox is not None:
        has_bbox = jnp.any(jnp.abs(target_bbox) > 0, axis=1)
        inputs["bbox_corners"] = _bbox_to_corners(target_bbox)
        input_types["bbox_corners"] = augmax.InputType.KEYPOINTS

    transform = augmax.Chain(
        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
        augmax.Resize(width, height),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
        input_types=input_types,
    )
    sub_rngs = jax.random.split(rng, image.shape[0])
    augmented = jax.vmap(transform)(sub_rngs, inputs)

    image = augmented["image"] * 2.0 - 1.0
    if target_mask is not None:
        target_mask = augmented["target_mask"][..., 0] > 0.5
    if target_crop is not None:
        target_crop = augmented["target_crop"]
    if target_mask is not None:
        target_bbox = _bbox_from_target_mask(target_mask)
    elif target_bbox is not None:
        target_bbox = _bbox_from_corners(augmented["bbox_corners"], has_bbox, (height, width))
    return image, target_mask, target_crop, target_bbox


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Optional target-object conditioning for object-centric action generation.
    # These are used by the pi0-style action expert as a lightweight conditioning signal
    # without changing the main vision-language backbone.
    target_mask: at.Bool[ArrayT, "b h w"] | None = None
    # Absolute pixel bbox [x1, y1, x2, y2) in the same image frame as target_mask/crop.
    # x2/y2 are exclusive, so area is (x2 - x1) * (y2 - y1).
    target_bbox: at.Float[ArrayT, "b 4"] | None = None
    target_crop: at.Float[ArrayT, "b h w c"] | None = None
    # 3D target point [x, y, z] derived from depth, in the camera/world frame
    # chosen by the dataset conversion pipeline.
    target_point: at.Float[ArrayT, "b 3"] | None = None
    # Optional detector/segmenter confidence in [0, 1].
    object_condition_confidence: at.Float[ArrayT, "b 1"] | None = None
    # Optional precomputed frozen semantic patch tokens (for example DINOv2).
    # Exclude the CLS token; the patch count must form a square grid.
    object_semantic_tokens: at.Float[ArrayT, "b n d"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        target_crop = data.get("target_crop")
        if target_crop is not None:
            # Object crops are adapter inputs, not backbone RGB inputs. Keep them in
            # [0, 1] so black padded/missing regions stay at zero and match dropout.
            if getattr(target_crop, "dtype", None) == np.uint8:
                target_crop = target_crop.astype(np.float32) / 255.0
            elif hasattr(target_crop, "dtype") and target_crop.dtype == torch.uint8:
                target_crop = target_crop.to(torch.float32) / 255.0
            else:
                target_crop = target_crop.astype(np.float32)
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            target_mask=data.get("target_mask"),
            target_bbox=data.get("target_bbox"),
            target_crop=target_crop,
            target_point=data.get("target_point"),
            object_condition_confidence=data.get("object_condition_confidence"),
            object_semantic_tokens=data.get("object_semantic_tokens"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    first_image_shape = observation.images[image_keys[0]].shape[1:3]
    target_mask = observation.target_mask
    if target_mask is not None and target_mask.shape[-2:] != image_resolution:
        target_mask = _resize_target_mask(target_mask, image_resolution)

    target_crop = observation.target_crop
    if target_crop is not None and target_crop.shape[1:3] != image_resolution:
        target_crop = _resize_target_crop(target_crop, image_resolution)

    target_bbox = observation.target_bbox
    if target_bbox is not None and first_image_shape != image_resolution:
        target_bbox = _resize_target_bbox(target_bbox, first_image_shape, image_resolution)

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            if key == image_keys[0] and any(value is not None for value in (target_mask, target_crop, target_bbox)):
                image, target_mask, target_crop, target_bbox = _augment_base_image_and_object_condition(
                    rng, image, target_mask, target_crop, target_bbox
                )
                out_images[key] = image
                continue

            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    with at.disable_typechecking():
        return Observation(
            images=out_images,
            image_masks=out_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            target_mask=target_mask,
            target_bbox=target_bbox,
            target_crop=target_crop,
            target_point=observation.target_point,
            object_condition_confidence=observation.object_condition_confidence,
            object_semantic_tokens=observation.object_semantic_tokens,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        expected_params = state.to_pure_dict()
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(expected_params, params)

        flat_expected = traverse_util.flatten_dict(expected_params)
        flat_params = traverse_util.flatten_dict(params)
        missing_paths = set(flat_expected) - set(flat_params)
        legacy_object_paths = {path for path in missing_paths if path and path[0].startswith("object_condition_")}
        if legacy_object_paths == missing_paths:
            loaded_has_object_params = any(path and path[0].startswith("object_condition_") for path in flat_params)
            if loaded_has_object_params and legacy_object_paths:
                raise ValueError(
                    "Object-condition checkpoint is missing parameters required by this model config. "
                    "Use the checkpoint's original config or an explicit migration."
                )
            for path in legacy_object_paths:
                flat_params[path] = flat_expected[path]
            params = traverse_util.unflatten_dict(flat_params)

        at.check_pytree_equality(expected=expected_params, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
