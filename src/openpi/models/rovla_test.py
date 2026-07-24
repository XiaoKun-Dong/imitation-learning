import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import rovla_config
from openpi.training import config as _training_config
from openpi.training import weight_loaders


def test_rovla_config_is_registered_and_only_adapter_is_trainable():
    train_config = _training_config.get_config("rovla_libero")
    assert isinstance(train_config.model, rovla_config.RoVLAConfig)
    assert train_config.model.model_type is _model.ModelType.PI05

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    trainable_paths = set(nnx.state(abstract_model, nnx.All(nnx.Param, train_config.trainable_filter)).flat_state())
    assert trainable_paths
    assert all(path[0].startswith("rovla_") for path in trainable_paths)


def test_rovla_checkpoint_loader_keeps_initialized_adapter_params():
    merged = weight_loaders._merge_params(  # noqa: SLF001
        {"base": {"weight": np.ones((1,), dtype=np.float32)}},
        {
            "base": {"weight": np.zeros((1,), dtype=np.float32)},
            "rovla_adapter": {"weight": np.full((1,), 2.0, dtype=np.float32)},
        },
        missing_regex=".*(lora|rovla).*",
    )
    assert merged["rovla_adapter"]["weight"].item() == 2.0


def test_rovla_config_rejects_invalid_corruption_size():
    with pytest.raises(ValueError, match="mask corruption"):
        rovla_config.RoVLAConfig(rovla_mask_shift_pixels=-1)


def test_rovla_accepts_external_semantic_tokens_with_matching_width():
    config = rovla_config.RoVLAConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=10,
        rovla_external_semantic_dim=384,
    )
    model = config.create(jax.random.key(1))
    obs = config.fake_obs(batch_size=1).replace(
        target_bbox=jnp.asarray([[32.0, 32.0, 96.0, 96.0]], dtype=jnp.float32),
        object_semantic_tokens=jnp.ones((1, 16, 384), dtype=jnp.float32),
    )

    object_condition = model._embed_object_condition(obs)

    assert object_condition is not None
    object_tokens, object_mask, has_condition, confidence = object_condition
    expected_tokens = min(config.rovla_num_object_tokens, 16) + 1
    assert object_tokens.shape == (1, expected_tokens, 64)
    assert object_mask.shape == (1, expected_tokens)
    assert jnp.all(has_condition)
    assert jnp.allclose(confidence, 1.0)


def test_rovla_rejects_external_semantic_token_width_mismatch():
    config = rovla_config.RoVLAConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=10,
        rovla_external_semantic_dim=384,
    )
    model = config.create(jax.random.key(2))
    obs = config.fake_obs(batch_size=1).replace(
        target_bbox=jnp.asarray([[32.0, 32.0, 96.0, 96.0]], dtype=jnp.float32),
        object_semantic_tokens=jnp.ones((1, 16, 768), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="object_semantic_tokens dim"):
        model._embed_object_condition(obs)


def test_rovla_confidence_alone_does_not_activate_condition():
    config = rovla_config.RoVLAConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=10,
    )
    model = config.create(jax.random.key(3))
    obs = config.fake_obs(batch_size=1).replace(
        target_mask=None,
        target_bbox=None,
        target_crop=None,
        target_point=None,
        object_condition_confidence=jnp.ones((1, 1), dtype=jnp.float32),
    )

    object_condition = model._embed_object_condition(obs)

    assert object_condition is not None
    _, _, has_condition, confidence = object_condition
    assert not jnp.any(has_condition)
    assert jnp.allclose(confidence, 1.0)
