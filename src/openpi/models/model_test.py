from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.policies import libero_policy
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
            target_point=jnp.ones((batch_size, 3), dtype=jnp.float32),
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
            target_point=jnp.ones((batch_size, 3), dtype=jnp.float32),
        )

    dropped = model._drop_object_condition(key, obs, train=True)
    assert not jnp.any(dropped.target_mask)
    assert jnp.allclose(dropped.target_bbox, 0.0)
    assert jnp.allclose(dropped.target_crop, 0.0)
    assert jnp.allclose(dropped.target_point, 0.0)


def test_pi0_object_cross_attention_starts_as_noop():
    key = jax.random.key(4)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    model = config.create(key)
    obs = config.fake_obs(1)
    actions = config.fake_act(1)
    timestep = jnp.asarray([0.5], dtype=jnp.float32)
    with _model.at.disable_typechecking():
        conditioned_obs = obs.replace(
            target_mask=jnp.ones((1, 16, 16), dtype=jnp.bool_),
            target_bbox=jnp.asarray([[20.0, 30.0, 80.0, 100.0]], dtype=jnp.float32),
            target_crop=jnp.ones((1, 16, 16, 3), dtype=jnp.float32),
            target_point=jnp.asarray([[0.1, -0.2, 1.0]], dtype=jnp.float32),
        )

    plain_tokens = model.embed_suffix(obs, actions, timestep)[0]
    conditioned_tokens = model.embed_suffix(conditioned_obs, actions, timestep)[0]

    assert jnp.array_equal(plain_tokens, conditioned_tokens)


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
        target_point=jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float32),
    )

    processed = _model.preprocess_observation(None, obs, image_resolution=(224, 224))

    assert processed.target_mask.shape == (batch_size, 224, 224)
    assert processed.target_mask.dtype == jnp.bool_
    assert processed.target_crop.shape == (batch_size, 224, 224, 3)
    assert processed.target_crop.dtype == jnp.float32
    assert processed.target_bbox.shape == (batch_size, 4)
    assert processed.target_bbox.dtype == jnp.float32
    assert processed.target_point.shape == (batch_size, 3)
    assert processed.target_point.dtype == jnp.float32
    assert jnp.allclose(processed.target_point, obs.target_point)
    assert jnp.allclose(processed.target_bbox[0], jnp.asarray([11.2, 78.4, 33.6, 112.0], dtype=jnp.float32))
    assert jnp.allclose(processed.target_bbox[1], jnp.zeros((4,), dtype=jnp.float32))


def test_preprocess_synchronizes_base_image_and_object_geometric_augmentation():
    batch_size = 2
    target_mask = jnp.zeros((batch_size, 224, 224), dtype=jnp.bool_).at[:, 60:120, 80:150].set(True)
    target_crop = jnp.repeat(target_mask[..., None], 3, axis=-1).astype(jnp.float32)
    base_image = target_crop * 2.0 - 1.0
    obs = _model.Observation(
        images={
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": jnp.zeros_like(base_image),
            "right_wrist_0_rgb": jnp.zeros_like(base_image),
        },
        image_masks={key: jnp.ones((batch_size,), dtype=jnp.bool_) for key in _model.IMAGE_KEYS},
        state=jnp.zeros((batch_size, 8), dtype=jnp.float32),
        target_mask=target_mask,
        target_bbox=jnp.asarray([[80, 60, 150, 120]] * batch_size, dtype=jnp.float32),
        target_crop=target_crop,
    )

    processed = _model.preprocess_observation(jax.random.key(10), obs, train=True)
    repeated = _model.preprocess_observation(jax.random.key(10), obs, train=True)

    assert jnp.allclose(processed.images["base_0_rgb"], processed.target_crop * 2.0 - 1.0, atol=1e-5)
    assert jnp.array_equal(processed.target_bbox, _model._bbox_from_target_mask(processed.target_mask))
    assert jnp.array_equal(processed.target_mask, repeated.target_mask)
    assert jnp.allclose(processed.target_crop, repeated.target_crop)
    assert not jnp.array_equal(processed.target_mask, target_mask)


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
        "target_point": jnp.asarray([[0.1, 0.2, 0.3]], dtype=jnp.float32),
    }

    obs = _model.Observation.from_dict(data)

    assert obs.target_crop.dtype == jnp.float32
    assert jnp.allclose(obs.target_crop, 1.0)
    assert jnp.allclose(obs.target_point, jnp.asarray([[0.1, 0.2, 0.3]], dtype=jnp.float32))


def test_libero_parse_mask_accepts_chw_images():
    mask = np.zeros((3, 16, 16), dtype=np.uint8)
    mask[:, 4:8, 6:10] = 255

    parsed = libero_policy._parse_mask(mask)

    assert parsed.shape == (16, 16)
    assert parsed.dtype == np.bool_
    assert parsed.sum() == 16


def test_resize_target_crop_requires_rgb_channels():
    with pytest.raises(ValueError, match="target_crop must have 3 channels"):
        _model._resize_target_crop(jnp.zeros((1, 16, 16, 1), dtype=jnp.float32), (224, 224))


def test_pi05_object_mask_freeze_filter():
    config = _train_config.get_config("pi05_libero_object_mask")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(3))
    trainable_state = nnx.state(abstract_model, config.trainable_filter).flat_state()
    trainable_paths = ["/".join(str(part) for part in path) for path in trainable_state]

    assert trainable_paths
    assert any("object_condition_" in path for path in trainable_paths)
    assert any("action_in_proj" in path for path in trainable_paths)
    assert any("action_out_proj" in path for path in trainable_paths)
    assert any("time_mlp" in path for path in trainable_paths)
    assert any("PaliGemma/llm" in path and "_1" in path for path in trainable_paths)
    assert not any("PaliGemma/img" in path for path in trainable_paths)
    assert not any("PaliGemma/llm" in path and "_1" not in path for path in trainable_paths)


def test_pi05_object_cross_attention_only_trains_object_condition():
    config = _train_config.get_config("pi05_libero_object_cross_attention")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(3))
    trainable_state = nnx.state(abstract_model, config.trainable_filter).flat_state()
    trainable_paths = ["/".join(str(part) for part in path) for path in trainable_state]

    assert trainable_paths
    assert all("object_condition_" in path for path in trainable_paths)
    assert any("object_condition_query_proj" in path for path in trainable_paths)
    assert any("object_condition_output_proj" in path for path in trainable_paths)


def test_pi05_object_2d_cross_attention_excludes_target_point():
    config = _train_config.get_config("pi05_libero_object_2d_cross_attention")

    assert config.data.object_condition_keys == ("target_mask", "target_bbox", "target_crop")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(4))
    trainable_state = nnx.state(abstract_model, config.trainable_filter).flat_state()
    trainable_paths = ["/".join(str(part) for part in path) for path in trainable_state]
    assert trainable_paths
    assert all("object_condition_" in path for path in trainable_paths)


def test_pi05_object_2d_step1_enables_encoder_and_gate():
    config = _train_config.get_config("pi05_libero_object_2d_step1")

    assert config.data.object_condition_keys == ("target_mask", "target_bbox", "target_crop")
    assert config.model.object_condition_encoder_layers == 2
    assert config.model.object_condition_use_gate
    assert jnp.isclose(jax.nn.sigmoid(config.model.object_condition_gate_init), 0.01798621)

    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(5))
    traverse_util.flatten_dict(nnx.state(abstract_model).to_pure_dict(), sep="/")
    trainable_state = nnx.state(abstract_model, config.trainable_filter).flat_state()
    trainable_paths = ["/".join(str(part) for part in path) for path in trainable_state]
    assert trainable_paths
    assert all("object_condition_" in path for path in trainable_paths)
    assert any("object_condition_encoder_qkv_projs" in path for path in trainable_paths)
    assert any("object_condition_encoder_mlp_in" in path for path in trainable_paths)
    assert any("object_condition_gate" in path for path in trainable_paths)


def test_pi05_object_2d_legacy_config_keeps_step1_disabled():
    config = _train_config.get_config("pi05_libero_object_2d_cross_attention")

    assert config.model.object_condition_encoder_layers == 0
    assert not config.model.object_condition_use_gate


def test_pi0_step1_object_encoder_masks_tokens_and_starts_as_noop():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        object_condition_encoder_layers=2,
        object_condition_use_gate=True,
    )
    model = config.create(jax.random.key(6))
    obs = config.fake_obs(batch_size=1)
    target_mask = jnp.zeros((1, 224, 224), dtype=jnp.bool_).at[:, 64:128, 80:144].set(True)
    target_crop = jnp.repeat(target_mask[..., None], 3, axis=-1).astype(jnp.float32)
    obs = obs.replace(
        target_mask=target_mask,
        target_bbox=jnp.asarray([[80.0, 64.0, 144.0, 128.0]], dtype=jnp.float32),
        target_crop=target_crop,
        target_point=None,
    )

    object_condition = model._embed_object_condition(obs)
    assert object_condition is not None
    object_tokens, object_token_mask, has_condition = object_condition
    assert object_tokens.shape == (1, 257, 1024)
    assert object_token_mask.shape == (1, 257)
    assert jnp.all(has_condition)
    assert jnp.all(jnp.isfinite(object_tokens))
    assert jnp.allclose(object_tokens * (~object_token_mask[..., None]), 0.0)
    assert jnp.isclose(jax.nn.sigmoid(model.object_condition_gate.value), 0.01798621)

    action_tokens = jax.random.normal(jax.random.key(7), (1, config.action_horizon, 1024))
    conditioned = model._cross_attend_object_condition(action_tokens, object_condition)
    assert jnp.allclose(conditioned, action_tokens)

    empty_obs = obs.replace(
        target_mask=jnp.zeros_like(target_mask),
        target_bbox=jnp.zeros((1, 4), dtype=jnp.float32),
        target_crop=jnp.zeros_like(target_crop),
    )
    empty_condition = model._embed_object_condition(empty_obs)
    assert empty_condition is not None
    empty_tokens, _, empty_has_condition = empty_condition
    assert jnp.all(jnp.isfinite(empty_tokens))
    assert not jnp.any(empty_has_condition)
    assert jnp.allclose(model._cross_attend_object_condition(action_tokens, empty_condition), action_tokens)


def test_pi05_loads_legacy_checkpoint_without_object_encoder():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        object_condition_encoder_layers=2,
        object_condition_use_gate=True,
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    _, state = nnx.split(abstract_model)
    legacy_params = state.to_pure_dict()
    for name in list(legacy_params):
        if name.startswith("object_condition_"):
            legacy_params.pop(name)

    loaded_model = config.load(legacy_params)

    loaded_state = nnx.state(loaded_model).to_pure_dict()
    assert "object_condition_query_proj" in loaded_state
    assert "object_condition_output_proj" in loaded_state
    assert "object_condition_encoder_qkv_projs" in loaded_state
    assert "object_condition_gate" in loaded_state


def test_pi05_rejects_object_checkpoint_with_incompatible_step1_structure():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        object_condition_encoder_layers=2,
        object_condition_use_gate=True,
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    _, state = nnx.split(abstract_model)
    old_object_params = state.to_pure_dict()
    for name in list(old_object_params):
        if name.startswith("object_condition_encoder_") or name == "object_condition_gate":
            old_object_params.pop(name)

    with pytest.raises(ValueError, match="Object-condition checkpoint is missing parameters"):
        config.load(old_object_params)


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
