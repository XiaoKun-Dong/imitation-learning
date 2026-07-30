import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import demovla
from openpi.models import demovla_visualization
from openpi.models import model as _model
from openpi.training import config as _train_config


def _path_strings(state: nnx.State) -> list[str]:
    return ["/".join(str(part) for part in path) for path in state.flat_state()]


@pytest.fixture(scope="module")
def demovla_model():
    config = demovla.DemoVLAConfig()
    return config, config.create(jax.random.key(1))


@pytest.fixture(scope="module")
def dynamic_demovla_model():
    config = demovla.DemoVLAConfig(interaction_gate_mode="dynamic")
    return config, config.create(jax.random.key(12))


def test_demovla_config_rejects_invalid_attention_shape():
    with pytest.raises(ValueError, match="divisible"):
        demovla.DemoVLAConfig(interaction_num_heads=7)


def test_demovla_config_rejects_invalid_injection_layers():
    with pytest.raises(ValueError, match="sorted and unique"):
        demovla.DemoVLAConfig(interaction_injection_layers=(9, 4, 14))
    with pytest.raises(ValueError, match=r"must be in \[0, 18\)"):
        demovla.DemoVLAConfig(interaction_injection_layers=(4, 9, 18))


def test_demovla_config_rejects_invalid_diversity_settings():
    with pytest.raises(ValueError, match="must be non-negative"):
        demovla.DemoVLAConfig(interaction_attention_diversity_weight=-1.0)
    with pytest.raises(ValueError, match=r"must be in \[-1, 1\]"):
        demovla.DemoVLAConfig(interaction_memory_diversity_margin=1.1)


def test_demovla_config_rejects_dynamic_gate_for_single_shot():
    with pytest.raises(ValueError, match="requires sparse_deep"):
        demovla.DemoVLAConfig(
            interaction_injection_mode="single_shot",
            interaction_gate_mode="dynamic",
        )


def test_pairwise_diversity_penalizes_collapsed_queries():
    collapsed = jnp.ones((1, 4, 4), dtype=jnp.float32)
    diverse = jnp.eye(4, dtype=jnp.float32)[None, ...]

    collapsed_loss, collapsed_mean, collapsed_max = demovla._pairwise_diversity_loss(collapsed, 0.5)  # noqa: SLF001
    diverse_loss, diverse_mean, diverse_max = demovla._pairwise_diversity_loss(diverse, 0.5)  # noqa: SLF001

    assert jnp.allclose(collapsed_loss, 0.25)
    assert jnp.allclose(collapsed_mean, 1.0)
    assert jnp.allclose(collapsed_max, 1.0)
    assert jnp.allclose(diverse_loss, 0.0)
    assert jnp.allclose(diverse_mean, 0.0)
    assert jnp.allclose(diverse_max, 0.0)


def test_demovla_inputs_do_not_require_oracle_object_condition():
    observation, actions = demovla.DemoVLAConfig().inputs_spec(batch_size=2)

    assert observation.target_mask is None
    assert observation.target_bbox is None
    assert observation.target_crop is None
    assert observation.target_point is None
    assert actions.shape == (2, 10, 32)


@pytest.mark.parametrize(
    ("config_name", "injection_mode"),
    [
        ("demovla_libero_single_shot", "single_shot"),
        ("demovla_libero_sparse_deep", "sparse_deep"),
        ("demovla_libero_sparse_deep_diverse", "sparse_deep"),
        ("demovla_libero_sparse_deep_dynamic_gate", "sparse_deep"),
    ],
)
def test_demovla_training_configs_are_adapter_only(config_name, injection_mode):
    config = _train_config.get_config(config_name)
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable_paths = _path_strings(nnx.state(abstract_model, config.trainable_filter))

    assert isinstance(config.model, demovla.DemoVLAConfig)
    assert config.model.interaction_injection_mode == injection_mode
    assert config.data.object_condition_keys is None
    assert not config.data.include_object_condition
    assert trainable_paths
    assert all(path.startswith("demovla_") for path in trainable_paths)
    assert any("demovla_interaction_queries" in path for path in trainable_paths)
    assert any("demovla_action_output_proj" in path for path in trainable_paths)
    if config_name.endswith("_diverse"):
        assert config.model.interaction_attention_diversity_weight == 1e-3
        assert config.model.interaction_memory_diversity_weight == 1e-4
    if config_name.endswith("_dynamic_gate"):
        assert config.model.interaction_gate_mode == "dynamic"
        assert any("demovla_dynamic_gate_mlp_out" in path for path in trainable_paths)


def test_demovla_shared_adapter_and_layer_gates_start_as_exact_noop(demovla_model):
    config, model = demovla_model
    action_tokens = jax.random.normal(jax.random.key(2), (1, config.action_horizon, 1024))
    memory = jax.random.normal(jax.random.key(3), (1, config.num_interaction_tokens, 1024))

    injected = model.inject_interaction_memory(action_tokens, memory, gate_index=1)

    assert jnp.array_equal(injected, action_tokens)
    assert model.demovla_interaction_gates.value.shape == (len(config.interaction_injection_layers),)
    assert jnp.allclose(jax.nn.sigmoid(model.demovla_interaction_gates.value), 0.01798621)


def test_sparse_deep_adapter_only_updates_configured_layers():
    width = 8
    params = {
        "norm_scale": jnp.ones((width,)),
        "norm_bias": jnp.zeros((width,)),
        "query_kernel": jnp.eye(width),
        "query_bias": jnp.zeros((width,)),
        "key_kernel": jnp.eye(width),
        "key_bias": jnp.zeros((width,)),
        "value_kernel": jnp.eye(width),
        "value_bias": jnp.zeros((width,)),
        "output_kernel": jnp.eye(width),
        "output_bias": jnp.zeros((width,)),
        "gates": jnp.full((2,), 10.0),
    }
    action_hidden = jax.random.normal(jax.random.key(8), (1, 3, width))
    memory = jax.random.normal(jax.random.key(9), (1, 2, width))
    injection_layers = jnp.asarray((1, 3))

    unchanged, _ = demovla._apply_sparse_deep_adapter(  # noqa: SLF001
        2,
        "scalar",
        params,
        memory,
        jnp.zeros((1, width)),
        injection_layers,
        jnp.asarray(0),
        [None, action_hidden],
    )
    updated, _ = demovla._apply_sparse_deep_adapter(  # noqa: SLF001
        2,
        "scalar",
        params,
        memory,
        jnp.zeros((1, width)),
        injection_layers,
        jnp.asarray(1),
        [None, action_hidden],
    )

    assert jnp.array_equal(unchanged[1], action_hidden)
    assert not jnp.allclose(updated[1], action_hidden)


def test_dynamic_gate_parameters_start_as_base_model_noop(dynamic_demovla_model):
    config, model = dynamic_demovla_model
    assert model.demovla_dynamic_gate_slot_embeddings.value.shape == (
        config.action_horizon,
        config.interaction_dynamic_gate_embedding_dim,
    )
    assert model.demovla_dynamic_gate_layer_embeddings.value.shape == (
        len(config.interaction_injection_layers),
        config.interaction_dynamic_gate_embedding_dim,
    )
    assert jnp.all(model.demovla_dynamic_gate_mlp_out.kernel.value == 0.0)
    assert jnp.all(model.demovla_dynamic_gate_mlp_out.bias.value == config.interaction_gate_init)
    assert jnp.all(model.demovla_action_output_proj.kernel.value == 0.0)
    assert jnp.all(model.demovla_action_output_proj.bias.value == 0.0)

    action_hidden = jax.random.normal(
        jax.random.key(13),
        (1, config.action_horizon, model.demovla_action_output_proj.out_features),
    )
    memory = jax.random.normal(
        jax.random.key(14),
        (1, config.num_interaction_tokens, model.demovla_action_output_proj.out_features),
    )
    adapter = model._make_sparse_deep_adapter(  # noqa: SLF001
        memory,
        jnp.ones((1, model.demovla_action_output_proj.out_features)),
    )
    (_, injected), aux = adapter(jnp.asarray(config.interaction_injection_layers[0]), [None, action_hidden])

    assert jnp.array_equal(injected, action_hidden)
    assert jnp.allclose(aux["gate"], jax.nn.sigmoid(config.interaction_gate_init))
    assert jnp.all(aux["injection_ratio"] == 0.0)


def test_dynamic_gate_loss_exports_layer_flow_bin_and_slot_metrics(dynamic_demovla_model):
    config, model = dynamic_demovla_model
    observation = config.fake_obs(batch_size=1)
    actions = config.fake_act(batch_size=1)

    loss, metrics = nnx.eval_shape(
        lambda module: module.compute_loss_with_aux(jax.random.key(15), observation, actions),
        model,
    )

    assert loss.shape == (1, config.action_horizon)
    assert metrics["demovla_dynamic_gate_mean"].shape == ()
    assert metrics["demovla_dynamic_injection_ratio_layer_9_mean"].shape == ()
    assert metrics["demovla_dynamic_layer_4_flow_bin_0_slot_0_gate_mean"].shape == ()
    assert metrics["demovla_dynamic_layer_14_flow_bin_4_slot_9_gate_mean"].shape == ()
    assert metrics["demovla_dynamic_layer_14_flow_bin_4_sample_count"].shape == ()


def test_dynamic_gate_has_layer_slot_and_flow_time_capacity():
    width = 4
    hidden_dim = 3
    embedding_dim = 2
    horizon = 3
    params = {
        "norm_scale": jnp.ones((width,)),
        "norm_bias": jnp.zeros((width,)),
        "query_kernel": jnp.eye(width),
        "query_bias": jnp.zeros((width,)),
        "key_kernel": jnp.eye(width),
        "key_bias": jnp.zeros((width,)),
        "value_kernel": jnp.eye(width),
        "value_bias": jnp.zeros((width,)),
        "output_kernel": jnp.eye(width),
        "output_bias": jnp.zeros((width,)),
        "gate_norm_scale": jnp.ones((width,)),
        "gate_norm_bias": jnp.zeros((width,)),
        "slot_embeddings": jnp.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "layer_embeddings": jnp.asarray([[0.0, 0.0], [0.0, 1.0]]),
        "gate_mlp_in_kernel": jnp.zeros((2 * width + 2 * embedding_dim, hidden_dim))
        .at[width, 0]
        .set(1.0)
        .at[width + embedding_dim, 1]
        .set(1.0)
        .at[2 * width + embedding_dim, 2]
        .set(1.0),
        "gate_mlp_in_bias": jnp.zeros((hidden_dim,)),
        "gate_mlp_out_kernel": jnp.ones((hidden_dim, 1)),
        "gate_mlp_out_bias": jnp.full((1,), -4.0),
    }
    action_hidden = jnp.arange(horizon * width, dtype=jnp.float32).reshape(1, horizon, width) + 1.0
    memory = jnp.ones((1, 2, width), dtype=jnp.float32)
    injection_layers = jnp.asarray((1, 3))

    (_, layer_1_hidden), layer_1_aux = demovla._apply_sparse_deep_adapter(  # noqa: SLF001
        2,
        "dynamic",
        params,
        memory,
        jnp.zeros((1, width)),
        injection_layers,
        jnp.asarray(1),
        [None, action_hidden],
    )
    (_, layer_3_hidden), layer_3_aux = demovla._apply_sparse_deep_adapter(  # noqa: SLF001
        2,
        "dynamic",
        params,
        memory,
        jnp.ones((1, width)),
        injection_layers,
        jnp.asarray(3),
        [None, action_hidden],
    )

    assert layer_1_aux["gate"].shape == (1, horizon)
    assert layer_1_aux["injection_ratio"].shape == (1, horizon)
    assert not jnp.allclose(layer_1_aux["gate"][:, 0], layer_1_aux["gate"][:, -1])
    assert not jnp.allclose(layer_1_aux["gate"], layer_3_aux["gate"])
    assert not jnp.allclose(layer_1_hidden, layer_3_hidden)


def test_demovla_memory_has_expected_shape_and_is_finite(demovla_model):
    config, model = demovla_model
    observation = config.fake_obs(batch_size=1)
    image_tokens = len(observation.images) * 256
    prefix_length = image_tokens + config.max_token_len
    prefix_hidden = jax.random.normal(jax.random.key(5), (1, prefix_length, 2048))

    with _model.at.disable_typechecking():
        memory = model.extract_interaction_memory(observation, prefix_hidden)

    assert memory.shape == (1, config.num_interaction_tokens, 1024)
    assert jnp.all(jnp.isfinite(memory))


def test_demovla_patch_diagnostics_have_positions_and_normalized_attention(demovla_model):
    config, model = demovla_model
    observation = config.fake_obs(batch_size=1)
    image_tokens = len(observation.images) * 256
    prefix_length = image_tokens + config.max_token_len
    prefix_hidden = jax.random.normal(jax.random.key(10), (1, prefix_length, 2048))

    with _model.at.disable_typechecking():
        memory, diagnostics = model.interaction_patch_diagnostics(
            observation,
            prefix_hidden,
            top_k=3,
        )

    assert memory.shape == (1, config.num_interaction_tokens, 1024)
    assert diagnostics["interaction_visual_attention"].shape == (
        1,
        config.num_interaction_tokens,
        len(observation.images),
        256,
    )
    assert diagnostics["interaction_top_patch_view_indices"].shape == (1, config.num_interaction_tokens, 3)
    assert diagnostics["interaction_top_patch_xy"].shape == (1, config.num_interaction_tokens, 3, 2)
    assert jnp.all(diagnostics["interaction_top_patch_xy"] > 0.0)
    assert jnp.all(diagnostics["interaction_top_patch_xy"] < 1.0)
    attention_sum = jnp.sum(diagnostics["interaction_visual_attention"], axis=(-2, -1))
    assert jnp.allclose(attention_sum, 1.0)


def test_demovla_loss_and_sampling_interfaces(demovla_model):
    config, model = demovla_model
    observation = config.fake_obs(batch_size=1)
    actions = config.fake_act(batch_size=1)

    loss = nnx.eval_shape(lambda module: module.compute_loss(jax.random.key(6), observation, actions), model)
    loss_with_aux, loss_metrics = nnx.eval_shape(
        lambda module: module.compute_loss_with_aux(jax.random.key(6), observation, actions),
        model,
    )
    sampled = nnx.eval_shape(
        lambda module: module.sample_actions(jax.random.key(7), observation, num_steps=2),
        model,
    )
    debug_sampled, diagnostics = nnx.eval_shape(
        lambda module: module.sample_actions_with_interaction_diagnostics(
            jax.random.key(11),
            observation,
            num_steps=2,
            top_k=3,
        ),
        model,
    )
    diagnostics_only = nnx.eval_shape(
        lambda module: module.interaction_diagnostics(
            observation,
            top_k=3,
        ),
        model,
    )

    assert loss.shape == (1, config.action_horizon)
    assert loss_with_aux.shape == loss.shape
    assert loss_metrics["demovla_attention_diversity_loss"].shape == ()
    assert loss_metrics["demovla_memory_diversity_loss"].shape == ()
    assert loss_metrics["demovla_diversity_regularization"].shape == ()
    assert sampled.shape == (1, config.action_horizon, config.action_dim)
    assert debug_sampled.shape == sampled.shape
    assert diagnostics["interaction_visual_attention"].shape == (
        1,
        config.num_interaction_tokens,
        len(observation.images),
        256,
    )
    assert diagnostics["interaction_top_patch_xy"].shape == (1, config.num_interaction_tokens, 3, 2)
    assert diagnostics_only["interaction_visual_attention"].shape == diagnostics["interaction_visual_attention"].shape
    assert diagnostics_only["interaction_top_patch_xy"].shape == diagnostics["interaction_top_patch_xy"].shape


def test_demovla_visualization_writes_one_replan_panel(tmp_path):
    visual_attention = np.zeros((2, 3, 4), dtype=np.float32)
    visual_attention[..., 0, 0] = 0.6
    visual_attention[..., 1, 3] = 0.4

    path = demovla_visualization.save_interaction_attention_replan(
        images={
            "base_0_rgb": np.full((32, 32, 3), 64, dtype=np.uint8),
            "left_wrist_0_rgb": np.full((32, 32, 3), 128, dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        },
        camera_names=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        visual_attention=visual_attention,
        camera_mask=np.asarray([True, True, False]),
        patch_grid_shape=(2, 2),
        output_dir=tmp_path,
        stem="attention",
        replan_index=3,
        env_step=15,
        top_k=2,
    )

    assert path.name == "attention_replan_003_step_0015.png"
    assert path.exists()
    assert path.stat().st_size > 0
