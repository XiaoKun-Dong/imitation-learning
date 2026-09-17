import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

from openpi.models import demovla
from openpi.models import demovla_visualization
from openpi.models import model as _model
from openpi.models import pi0_config
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
    with pytest.raises(ValueError, match="must be non-negative"):
        demovla.DemoVLAConfig(interaction_output_init_std=-1e-3)


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


def test_demovla_inputs_match_pi_observation_contract():
    observation, actions = demovla.DemoVLAConfig().inputs_spec(batch_size=2)

    assert set(observation.images) == set(_model.IMAGE_KEYS)
    assert observation.state.shape == (2, 32)
    assert actions.shape == (2, 10, 32)


@pytest.mark.parametrize(
    ("config_name", "injection_mode"),
    [
        ("demovla_libero_single_shot", "single_shot"),
        ("demovla_libero_sparse_deep", "sparse_deep"),
        ("demovla_libero_sparse_deep_diverse", "sparse_deep"),
        ("demovla_libero_sparse_deep_dynamic_gate", "sparse_deep"),
        ("demovla_libero_sparse_deep_dynamic_gate_m3", "sparse_deep"),
    ],
)
def test_demovla_training_configs_are_adapter_only(config_name, injection_mode):
    config = _train_config.get_config(config_name)
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable_paths = _path_strings(nnx.state(abstract_model, config.trainable_filter))

    assert isinstance(config.model, demovla.DemoVLAConfig)
    assert config.model.interaction_injection_mode == injection_mode
    assert trainable_paths
    assert all(path.startswith("demovla_") for path in trainable_paths)
    assert any("demovla_interaction_queries" in path for path in trainable_paths)
    assert any("demovla_action_output_proj" in path for path in trainable_paths)
    if config_name.endswith("_diverse"):
        assert config.model.interaction_attention_diversity_weight == 1e-3
        assert config.model.interaction_memory_diversity_weight == 1e-4
    if "_dynamic_gate" in config_name:
        assert config.model.interaction_gate_mode == "dynamic"
        assert any("demovla_dynamic_gate_mlp_out" in path for path in trainable_paths)


def test_demovla_libero_dynamic_gate_m3_matches_four_gpu_training_plan():
    config = _train_config.get_config("demovla_libero_sparse_deep_dynamic_gate_m3")
    pi05_config = _train_config.get_config("pi05_libero")

    assert config.model.interaction_gate_init == -3.0
    assert jnp.isclose(jax.nn.sigmoid(config.model.interaction_gate_init), 0.04742587)
    assert config.project_name == "demovla"
    assert config.wandb_enabled
    assert config.num_train_steps == pi05_config.num_train_steps == 30_000
    assert config.batch_size == 128
    assert config.num_workers == 16
    assert config.fsdp_devices == 4


def test_full_libero_comparison_configs_are_strictly_matched():
    control = _train_config.get_config("pi05_libero_full_matched")
    contender = _train_config.get_config("demovla_libero_full_matched")

    assert isinstance(control.model, pi0_config.Pi0Config)
    assert isinstance(contender.model, demovla.DemoVLAConfig)
    assert control.model.pi05
    assert contender.model.pi05
    assert control.model.action_horizon == contender.model.action_horizon == 10
    assert control.model.discrete_state_input == contender.model.discrete_state_input is False
    assert control.data == contender.data
    assert str(control.data.assets.assets_dir).endswith("assets/libero_full_matched")
    assert control.batch_size == contender.batch_size == 32
    assert control.num_workers == contender.num_workers == 16
    assert control.lr_schedule == contender.lr_schedule
    assert control.optimizer == contender.optimizer
    assert control.ema_decay == contender.ema_decay == 0.999
    assert control.num_train_steps == contender.num_train_steps == 30_000
    assert control.save_interval == contender.save_interval == 5_000
    assert control.keep_period == contender.keep_period == 10_000
    assert control.seed == contender.seed
    assert control.fsdp_devices == contender.fsdp_devices == 4
    assert control.wandb_enabled == contender.wandb_enabled is False
    assert isinstance(control.freeze_filter, nnx.Nothing)
    assert isinstance(contender.freeze_filter, nnx.Nothing)
    assert control.weight_loader.params_path == contender.weight_loader.params_path


def test_demovla_full_stage_a_freezes_backbone_and_scalar_gates():
    config = _train_config.get_config("demovla_libero_full_stage_a_fixed_gate")
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(24))
    trainable_paths = _path_strings(nnx.state(abstract_model, config.trainable_filter))

    assert isinstance(config.model, demovla.DemoVLAConfig)
    assert config.model.interaction_gate_mode == "scalar"
    assert config.model.interaction_injection_mode == "sparse_deep"
    assert jnp.isclose(jax.nn.sigmoid(config.model.interaction_gate_init), 0.05)
    assert config.model.interaction_output_init_std == 1e-3
    assert trainable_paths
    assert all(path.startswith("demovla_") for path in trainable_paths)
    assert not any("demovla_interaction_gates" in path for path in trainable_paths)
    assert any("demovla_interaction_queries" in path for path in trainable_paths)
    assert any("demovla_action_output_proj" in path for path in trainable_paths)
    assert str(config.data.assets.assets_dir).endswith("checkpoints/pi05_libero/assets")
    assert config.batch_size == 128
    assert config.fsdp_devices == 4
    assert config.num_train_steps == 5_000
    assert config.wandb_enabled


def test_demovla_b1_readout_pair_is_strictly_matched():
    control = _train_config.get_config("demovla_libero_full_b1_readout_control")
    recovery = _train_config.get_config("demovla_libero_full_b1_readout_recovery")

    assert control.data == recovery.data
    assert control.batch_size == recovery.batch_size == 128
    assert control.lr_schedule == recovery.lr_schedule
    assert control.optimizer == recovery.optimizer
    assert control.seed == recovery.seed == 42
    assert control.num_train_steps == recovery.num_train_steps == 1_200
    assert control.fsdp_devices == recovery.fsdp_devices == 4
    assert jnp.isclose(jax.nn.sigmoid(control.model.interaction_gate_init), 0.03)
    assert jnp.isclose(jax.nn.sigmoid(recovery.model.interaction_gate_init), 0.03)
    assert control.weight_loader.params_path == recovery.weight_loader.params_path
    assert control.weight_loader.skip_regex == recovery.weight_loader.skip_regex

    control_model = nnx.eval_shape(control.model.create, jax.random.key(25))
    recovery_model = nnx.eval_shape(recovery.model.create, jax.random.key(26))
    control_paths = _path_strings(nnx.state(control_model, control.trainable_filter))
    recovery_paths = _path_strings(nnx.state(recovery_model, recovery.trainable_filter))
    for paths in (control_paths, recovery_paths):
        assert paths
        assert all(
            any(
                component in path
                for component in (
                    "demovla_action_query_",
                    "demovla_memory_key_proj",
                    "demovla_memory_value_proj",
                    "demovla_action_output_proj",
                    "demovla_readout_",
                )
            )
            for path in paths
        )
        assert not any("demovla_interaction_gates" in path for path in paths)
        assert not any("demovla_interaction_queries" in path for path in paths)
    assert control.model.interaction_readout_mode == "standard"
    assert recovery.model.interaction_readout_mode == "recovery"
    assert recovery.model.interaction_memory_ranking_weight == 1.0
    assert not any("demovla_readout_" in path for path in control_paths)
    assert any("demovla_readout_layer_embeddings" in path for path in recovery_paths)
    assert any("demovla_readout_temperature_logits" in path for path in recovery_paths)


def test_demovla_s1_semantic_pair_changes_only_ranking_objective():
    control = _train_config.get_config("demovla_libero_full_s1_semantic_control")
    semantic = _train_config.get_config("demovla_libero_full_s1_semantic_ranking")

    assert control.data == semantic.data
    assert control.batch_size == semantic.batch_size == 128
    assert control.lr_schedule == semantic.lr_schedule
    assert control.optimizer == semantic.optimizer
    assert control.seed == semantic.seed == 42
    assert control.num_train_steps == semantic.num_train_steps == 1_200
    assert control.freeze_filter == semantic.freeze_filter
    assert control.weight_loader == semantic.weight_loader
    assert control.model.interaction_readout_mode == semantic.model.interaction_readout_mode == "standard"
    assert control.model.interaction_memory_ranking_weight == 0.0
    assert semantic.model.interaction_memory_ranking_weight == 0.25
    assert semantic.model.interaction_memory_ranking_margin == 2e-5
    assert semantic.model.interaction_memory_ranking_mode == "prompt_hard_negative"

    control_model = nnx.eval_shape(control.model.create, jax.random.key(27))
    semantic_model = nnx.eval_shape(semantic.model.create, jax.random.key(28))
    assert _path_strings(nnx.state(control_model, control.trainable_filter)) == _path_strings(
        nnx.state(semantic_model, semantic.trainable_filter)
    )


def test_prompt_hard_negative_excludes_duplicate_tasks(demovla_model):
    _, model = demovla_model
    tokens = jnp.asarray(
        [
            [1, 2, 3, 0],
            [1, 2, 3, 0],
            [1, 2, 4, 0],
            [9, 9, 9, 0],
        ],
        dtype=jnp.int32,
    )
    masks = jnp.asarray([[True, True, True, False]] * 4)
    observation = _model.Observation(
        images={name: jnp.zeros((4, 2, 2, 3), dtype=jnp.float32) for name in _model.IMAGE_KEYS},
        image_masks={name: jnp.ones((4,), dtype=jnp.bool_) for name in _model.IMAGE_KEYS},
        state=jnp.zeros((4, 32), dtype=jnp.float32),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=masks,
    )

    negative, similarity = model._make_prompt_hard_negative(observation)  # noqa: SLF001

    assert jnp.all(jnp.any(negative.tokenized_prompt != tokens, axis=-1))
    assert jnp.array_equal(negative.tokenized_prompt[0], tokens[2])
    assert jnp.array_equal(negative.tokenized_prompt[1], tokens[2])
    assert jnp.isclose(similarity[0], 2 / 3)
    assert jnp.isclose(similarity[1], 2 / 3)


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
    updated, updated_aux = demovla._apply_sparse_deep_adapter(  # noqa: SLF001
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
    assert jnp.all(updated_aux["ungated_delta_ratio"] > 0.0)
    assert jnp.all(updated_aux["injection_ratio"] > 0.0)
    assert jnp.allclose(
        updated_aux["injection_ratio"],
        jax.nn.sigmoid(10.0) * updated_aux["ungated_delta_ratio"],
    )


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


def test_dynamic_gate_zero_injection_has_finite_gradients():
    width = 4
    embedding_dim = 2
    hidden_dim = 3
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
        # Match the exact no-op initialization used by DemoVLA.
        "output_kernel": jnp.zeros((width, width)),
        "output_bias": jnp.zeros((width,)),
        "gate_norm_scale": jnp.ones((width,)),
        "gate_norm_bias": jnp.zeros((width,)),
        "slot_embeddings": jnp.zeros((horizon, embedding_dim)),
        "layer_embeddings": jnp.zeros((1, embedding_dim)),
        "gate_mlp_in_kernel": jax.random.normal(
            jax.random.key(18),
            (2 * width + 2 * embedding_dim, hidden_dim),
        ),
        "gate_mlp_in_bias": jnp.zeros((hidden_dim,)),
        "gate_mlp_out_kernel": jnp.zeros((hidden_dim, 1)),
        "gate_mlp_out_bias": jnp.full((1,), -4.0),
    }
    action_hidden = jax.random.normal(jax.random.key(19), (1, horizon, width))
    memory = jax.random.normal(jax.random.key(20), (1, 2, width))
    flow_time_embedding = jax.random.normal(jax.random.key(21), (1, width))

    def adapter_output_sum(adapter_params):
        (_, injected), _ = demovla._apply_sparse_deep_adapter(  # noqa: SLF001
            1,
            "dynamic",
            adapter_params,
            memory,
            flow_time_embedding,
            jnp.asarray((1,)),
            jnp.asarray(1),
            [None, action_hidden],
        )
        return jnp.sum(injected)

    grads = jax.grad(adapter_output_sum)(params)

    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in jax.tree.leaves(grads))


def test_dynamic_gate_layer_mean_ablation_uses_fixed_per_layer_gates(dynamic_demovla_model):
    config, model = dynamic_demovla_model
    layer_mean_gates = (0.0211, 0.0242, 0.0277)
    model.configure_interaction_inference_ablation("layer_mean", layer_mean_gates)
    action_hidden = jax.random.normal(
        jax.random.key(22),
        (1, config.action_horizon, model.demovla_action_output_proj.out_features),
    )
    memory = jax.random.normal(
        jax.random.key(23),
        (1, config.num_interaction_tokens, model.demovla_action_output_proj.out_features),
    )
    adapter = model._make_sparse_deep_adapter(  # noqa: SLF001
        memory,
        jnp.ones((1, model.demovla_action_output_proj.out_features)),
    )

    for layer, expected_gate in zip(config.interaction_injection_layers, layer_mean_gates, strict=True):
        (_, _), aux = adapter(jnp.asarray(layer), [None, action_hidden])
        assert jnp.allclose(aux["gate"], expected_gate)

    model.configure_interaction_inference_ablation("normal")


def test_dynamic_gate_layer_mean_ablation_validates_gate_count(dynamic_demovla_model):
    _, model = dynamic_demovla_model
    with pytest.raises(ValueError, match="one gate probability per injection layer"):
        model.configure_interaction_inference_ablation("layer_mean", (0.02,))


def test_scalar_gate_layer_mean_ablation_uses_fixed_per_layer_gates(demovla_model):
    config, model = demovla_model
    fixed_gates = (0.01, 0.03, 0.05)
    model.configure_interaction_inference_ablation("layer_mean", fixed_gates)
    action_hidden = jax.random.normal(
        jax.random.key(24),
        (1, config.action_horizon, model.demovla_action_output_proj.out_features),
    )
    memory = jax.random.normal(
        jax.random.key(25),
        (1, config.num_interaction_tokens, model.demovla_action_output_proj.out_features),
    )
    adapter = model._make_sparse_deep_adapter(  # noqa: SLF001
        memory,
        jnp.ones((1, model.demovla_action_output_proj.out_features)),
    )

    for layer, expected_gate in zip(config.interaction_injection_layers, fixed_gates, strict=True):
        (_, _), aux = adapter(jnp.asarray(layer), [None, action_hidden])
        assert jnp.allclose(aux["gate"], expected_gate)

    model.configure_interaction_inference_ablation("normal")


def test_scalar_gate_layer_knockout_accepts_exact_zero(demovla_model):
    config, model = demovla_model
    fixed_gates = (0.0, 0.03, 0.03)
    model.configure_interaction_inference_ablation("layer_mean", fixed_gates)
    memory = jax.random.normal(
        jax.random.key(124),
        (1, config.num_interaction_tokens, model.demovla_action_output_proj.out_features),
    )
    adapter = model._make_sparse_deep_adapter(  # noqa: SLF001
        memory,
        jnp.ones((1, model.demovla_action_output_proj.out_features)),
    )
    action_hidden = jax.random.normal(
        jax.random.key(125),
        (1, config.action_horizon, model.demovla_action_output_proj.out_features),
    )

    for layer, expected_gate in zip(config.interaction_injection_layers, fixed_gates, strict=True):
        (_, _), aux = adapter(jnp.asarray(layer), [None, action_hidden])
        assert jnp.allclose(aux["gate"], expected_gate)

    model.configure_interaction_inference_ablation("normal")


def test_memory_ablation_zero_and_deterministic_batch_shuffle(dynamic_demovla_model):
    _, model = dynamic_demovla_model
    memory = jnp.arange(4 * 2 * 3, dtype=jnp.float32).reshape(4, 2, 3)

    model.configure_interaction_inference_ablation("zero_memory")
    assert jnp.array_equal(model._apply_interaction_memory_ablation(memory), jnp.zeros_like(memory))  # noqa: SLF001

    model.configure_interaction_inference_ablation("batch_shuffle")
    assert jnp.array_equal(  # noqa: SLF001
        model._apply_interaction_memory_ablation(memory),
        jnp.roll(memory, shift=2, axis=0),
    )
    with pytest.raises(ValueError, match="batch size >= 2"):
        model._apply_interaction_memory_ablation(memory[:1])  # noqa: SLF001

    model.configure_interaction_inference_ablation("normal")


def test_zero_memory_composes_with_fixed_gate_override(demovla_model):
    config, model = demovla_model
    fixed_gates = (0.03, 0.03, 0.03)
    model.configure_interaction_inference_ablation("zero_memory", fixed_gates)
    memory = jax.random.normal(
        jax.random.key(26),
        (1, config.num_interaction_tokens, model.demovla_action_output_proj.out_features),
    )
    ablated_memory = model._apply_interaction_memory_ablation(memory)  # noqa: SLF001
    assert jnp.array_equal(ablated_memory, jnp.zeros_like(memory))

    adapter = model._make_sparse_deep_adapter(  # noqa: SLF001
        ablated_memory,
        jnp.ones((1, model.demovla_action_output_proj.out_features)),
    )
    action_hidden = jax.random.normal(
        jax.random.key(27),
        (1, config.action_horizon, model.demovla_action_output_proj.out_features),
    )
    for layer in config.interaction_injection_layers:
        (_, _), aux = adapter(jnp.asarray(layer), [None, action_hidden])
        assert jnp.allclose(aux["gate"], 0.03)

    model.configure_interaction_inference_ablation("normal")


def test_dynamic_gate_model_instances_have_matching_graphdefs():
    config = demovla.DemoVLAConfig(interaction_gate_mode="dynamic")
    # Graph structure does not require materializing two multi-billion-parameter
    # models. Keeping the models abstract also prevents this regression test from
    # exhausting a single GPU after the module-scoped model fixtures are loaded.
    first = nnx.graphdef(nnx.eval_shape(config.create, jax.random.key(16)))
    second = nnx.graphdef(nnx.eval_shape(config.create, jax.random.key(17)))

    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)


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
    memory_source_loss, memory_source, memory_source_attention = nnx.eval_shape(
        lambda module: module.compute_flow_loss_with_memory_source(
            jax.random.key(6),
            observation,
            observation,
            actions,
        ),
        model,
    )
    sampled = nnx.eval_shape(
        lambda module: module.sample_actions(jax.random.key(7), observation, num_steps=2),
        model,
    )
    memory_source_sampled = nnx.eval_shape(
        lambda module: module.sample_actions_with_memory_source(
            jax.random.key(7),
            observation,
            observation,
            num_steps=2,
        ),
        model,
    )
    transplanted_sampled = nnx.eval_shape(
        lambda module: module.sample_actions_with_memory_source(
            jax.random.key(7),
            observation,
            observation,
            num_steps=2,
            interaction_memory_override=jnp.zeros(
                (1, config.num_interaction_tokens, model.demovla_action_output_proj.out_features)
            ),
        ),
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
    assert memory_source_loss.shape == loss.shape
    assert memory_source.shape == (1, config.num_interaction_tokens, 1024)
    assert memory_source_attention.shape == (
        1,
        config.num_interaction_tokens,
        len(observation.images) * 256,
    )
    assert loss_metrics["demovla_attention_diversity_loss"].shape == ()
    assert loss_metrics["demovla_memory_diversity_loss"].shape == ()
    assert loss_metrics["demovla_diversity_regularization"].shape == ()
    assert sampled.shape == (1, config.action_horizon, config.action_dim)
    assert memory_source_sampled.shape == sampled.shape
    assert transplanted_sampled.shape == sampled.shape
    assert debug_sampled.shape == sampled.shape
    assert diagnostics["interaction_visual_attention"].shape == (
        1,
        config.num_interaction_tokens,
        len(observation.images),
        256,
    )
    assert diagnostics["interaction_top_patch_xy"].shape == (1, config.num_interaction_tokens, 3, 2)
    assert diagnostics["interaction_action_to_memory_attention"].shape == (
        1,
        2,
        len(config.interaction_injection_layers),
        config.action_horizon,
        config.num_interaction_tokens,
    )
    assert diagnostics["interaction_action_to_memory_head_attention"].shape == (
        1,
        2,
        len(config.interaction_injection_layers),
        config.interaction_num_heads,
        config.action_horizon,
        config.num_interaction_tokens,
    )
    assert diagnostics["interaction_effective_visual_attention"].shape == (
        1,
        2,
        len(config.interaction_injection_layers),
        config.action_horizon,
        len(observation.images),
        256,
    )
    assert diagnostics["interaction_adapter_gate"].shape == (
        1,
        2,
        len(config.interaction_injection_layers),
        config.action_horizon,
    )
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
        prompt="pick up the red mug and place it on the plate",
        save_raw=True,
    )

    assert path.name == "attention_replan_003_step_0015.png"
    assert path.exists()
    assert path.stat().st_size > 0
    raw = np.load(path.with_suffix(".npz"))
    assert raw["visual_attention"].shape == visual_attention.shape
    assert raw["prompt"].item() == "pick up the red mug and place it on the plate"
    with Image.open(path) as rendered:
        assert rendered.height > 2 * (224 + 24)


def test_demovla_visualization_uses_model_letterbox_geometry():
    image = np.full((480, 848, 3), 255, dtype=np.uint8)

    prepared = demovla_visualization._resize_with_pad_image(image, 224, 224)  # noqa: SLF001

    assert prepared.shape == (224, 224, 3)
    assert np.all(prepared[:49] == 0)
    assert np.all(prepared[49:175] == 255)
    assert np.all(prepared[175:] == 0)


def test_demovla_visualization_deemphasizes_rgb_and_deepens_attention():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    rendered = np.asarray(demovla_visualization._attention_overlay(image, np.ones((1, 1)), 1.0))  # noqa: SLF001

    # The washed-out background remains visible, while maximal attention is a
    # dark, saturated red rather than a translucent pink overlay.
    assert np.all(rendered[..., 0] > rendered[..., 1] + 100)
    assert np.all(rendered[..., 1] < 50)


def test_demovla_visualization_writes_effective_action_attention(tmp_path):
    effective_attention = np.full((2, 3, 4, 3, 4), 1.0 / 12.0, dtype=np.float32)
    action_to_memory = np.full((2, 3, 4, 2), 0.5, dtype=np.float32)

    path = demovla_visualization.save_effective_interaction_attention_replan(
        images={
            "base_0_rgb": np.full((32, 32, 3), 64, dtype=np.uint8),
            "left_wrist_0_rgb": np.full((32, 32, 3), 128, dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        },
        camera_names=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        effective_visual_attention=effective_attention,
        action_to_memory_attention=action_to_memory,
        camera_mask=np.asarray([True, True, False]),
        patch_grid_shape=(2, 2),
        injection_layers=(4, 9, 14),
        output_dir=tmp_path,
        stem="effective",
        replan_index=2,
        env_step=10,
        prompt="put both mugs on the plate",
    )

    assert path.name == "effective_replan_002_step_0010.png"
    assert path.exists()
    with Image.open(path) as rendered:
        assert rendered.height > 3 * (224 + 24)
