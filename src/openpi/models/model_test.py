from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as _train_config


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_model_with_object_condition():
    key = jax.random.key(1)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    with _model.at.disable_typechecking():
        obs = obs.replace(
            target_mask=jnp.zeros((batch_size, 16, 16), dtype=jnp.bool_),
            target_bbox=jnp.ones((batch_size, 4), dtype=jnp.float32),
            target_crop=jnp.ones((batch_size, 16, 16, 3), dtype=jnp.float32),
        )

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=5)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_object_condition_dropout():
    key = jax.random.key(2)
    config = pi0_config.Pi0Config(object_condition_dropout_rate=1.0)
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)
    with _model.at.disable_typechecking():
        obs = obs.replace(
            target_mask=jnp.ones((batch_size, 16, 16), dtype=jnp.bool_),
            target_bbox=jnp.ones((batch_size, 4), dtype=jnp.float32),
            target_crop=jnp.ones((batch_size, 16, 16, 3), dtype=jnp.float32),
        )

    dropped = model._drop_object_condition(key, obs, train=True)
    assert not jnp.any(dropped.target_mask)
    assert jnp.allclose(dropped.target_bbox, 0.0)
    assert jnp.allclose(dropped.target_crop, 0.0)


def test_target_bbox_resize_pixel_exclusive_convention():
    bbox = jnp.asarray([[10, 20, 30, 50]], dtype=jnp.float32)

    resized = _model._resize_target_bbox(bbox, original_hw=(100, 200), image_resolution=(224, 224))

    assert jnp.allclose(resized, jnp.asarray([[11.2, 78.4, 33.6, 112.0]], dtype=jnp.float32))
    assert resized.max() > 1.0
    assert _model.TARGET_BBOX_FORMAT == "xyxy_pixel_exclusive"


def test_preprocess_target_object_shapes_and_dtypes():
    batch_size = 2
    obs = _model.Observation(
        images={
            "base_0_rgb": jnp.zeros((batch_size, 100, 200, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.zeros((batch_size, 100, 200, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.zeros((batch_size, 100, 200, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        },
        state=jnp.zeros((batch_size, 8), dtype=jnp.float32),
        target_mask=jnp.zeros((batch_size, 100, 200), dtype=jnp.bool_),
        target_bbox=jnp.asarray([[10, 20, 30, 50], [0, 0, 0, 0]], dtype=jnp.float32),
        target_crop=jnp.zeros((batch_size, 100, 200, 3), dtype=jnp.float32),
    )

    processed = _model.preprocess_observation(None, obs, image_resolution=(224, 224))

    assert processed.target_mask.shape == (batch_size, 224, 224)
    assert processed.target_mask.dtype == jnp.bool_
    assert processed.target_crop.shape == (batch_size, 224, 224, 3)
    assert processed.target_crop.dtype == jnp.float32
    assert processed.target_bbox.shape == (batch_size, 4)
    assert processed.target_bbox.dtype == jnp.float32
    assert jnp.allclose(processed.target_bbox[0], jnp.asarray([11.2, 78.4, 33.6, 112.0], dtype=jnp.float32))
    assert jnp.allclose(processed.target_bbox[1], jnp.asarray([0.0, 56.0, 0.0, 56.0], dtype=jnp.float32))


def test_observation_from_dict_converts_uint8_target_crop_to_float():
    data = {
        "image": {
            "base_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
        },
        "image_mask": {
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        },
        "state": jnp.zeros((1, 8), dtype=jnp.float32),
        "target_mask": jnp.zeros((1, 224, 224), dtype=jnp.bool_),
        "target_bbox": jnp.asarray([[0, 0, 2, 2]], dtype=jnp.float32),
        "target_crop": jnp.full((1, 224, 224, 3), 255, dtype=jnp.uint8),
    }

    obs = _model.Observation.from_dict(data)

    assert obs.target_crop.dtype == jnp.float32
    assert jnp.allclose(obs.target_crop, 1.0)


def test_resize_target_crop_requires_rgb_channels():
    with pytest.raises(ValueError, match="target_crop must have 3 channels"):
        _model._resize_target_crop(jnp.zeros((1, 16, 16, 1), dtype=jnp.float32), (224, 224))


def test_pi05_object_mask_freeze_filter():
    config = _train_config.get_config("pi05_libero_object_mask")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(3))
    trainable_state = nnx.state(abstract_model, config.trainable_filter).flat_state()
    trainable_paths = ["/".join(str(part) for part in path) for path in trainable_state]

    assert trainable_paths
    assert any("object_condition_proj" in path for path in trainable_paths)
    assert any("action_in_proj" in path for path in trainable_paths)
    assert any("action_out_proj" in path for path in trainable_paths)
    assert any("time_mlp" in path for path in trainable_paths)
    assert any("PaliGemma/llm" in path and "_1" in path for path in trainable_paths)
    assert not any("PaliGemma/img" in path for path in trainable_paths)
    assert not any("PaliGemma/llm" in path and "_1" not in path for path in trainable_paths)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
