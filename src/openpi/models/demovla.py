"""DemoVLA: self-grounded interaction memory for pi0.5.

The visual-language prefix, its KV cache, and one interaction memory are
computed once per policy replan. The action expert reads that memory either
once at its input or at a few configured Transformer depths. No external
object annotation is used.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

_SIGLIP_PATCH_SIZE = 14
_DYNAMIC_GATE_FLOW_BINS = 5


def _linear(
    x: at.Float[at.Array, "... in_features"],
    kernel: at.Float[at.Array, "in_features out_features"],
    bias: at.Float[at.Array, " out_features"],
) -> at.Float[at.Array, "... out_features"]:
    return jnp.matmul(x, kernel) + bias


def _layer_norm(
    x: at.Float[at.Array, "... features"],
    scale: at.Float[at.Array, " features"],
    bias: at.Float[at.Array, " features"],
) -> at.Float[at.Array, "... features"]:
    x_float = x.astype(jnp.float32)
    mean = jnp.mean(x_float, axis=-1, keepdims=True)
    mean_square = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    variance = jnp.maximum(0.0, mean_square - jnp.square(mean))
    normalized = (x_float - mean) * jax.lax.rsqrt(variance + 1.0e-6)
    return normalized * scale + bias


def _pairwise_diversity_loss(
    tokens: at.Float[at.Array, "b q d"],
    margin: float,
) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, ""], at.Float[at.Array, ""]]:
    """Penalize only query pairs whose cosine similarity exceeds a margin."""
    num_queries = tokens.shape[-2]
    if num_queries < 2:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero, zero

    tokens = tokens.astype(jnp.float32)
    tokens = tokens / jnp.maximum(jnp.linalg.norm(tokens, axis=-1, keepdims=True), 1.0e-6)
    similarity = jnp.einsum("bqd,bkd->bqk", tokens, tokens)
    off_diagonal = 1.0 - jnp.eye(num_queries, dtype=jnp.float32)
    pair_count = similarity.shape[0] * num_queries * (num_queries - 1)
    pairwise_mean = jnp.sum(similarity * off_diagonal[None, ...]) / pair_count
    pairwise_max = jnp.max(
        jnp.where(
            off_diagonal[None, ...].astype(jnp.bool_),
            similarity,
            -jnp.inf,
        )
    )
    excess_similarity = jax.nn.relu(similarity - margin)
    loss = jnp.sum(jnp.square(excess_similarity) * off_diagonal[None, ...]) / pair_count
    return loss, pairwise_mean, pairwise_max


def _apply_sparse_deep_adapter(
    num_heads,
    gate_mode,
    params,
    interaction_memory,
    flow_time_embedding,
    injection_layers,
    layer_index,
    hidden_groups,
    readout_mode="standard",
    temperature_min=0.5,
    temperature_max=2.0,
):
    """Pure JAX action-token hook compatible with Flax's scanned blocks."""
    action_hidden = hidden_groups[1]
    if action_hidden is None:
        return hidden_groups

    def inject(hidden):
        matches = layer_index == injection_layers
        gate_index = jnp.argmax(matches)
        query_hidden = hidden
        if readout_mode == "recovery":
            query_hidden = query_hidden + params["readout_layer_embeddings"][gate_index][None, None, :]
        query = _linear(
            _layer_norm(query_hidden, params["norm_scale"], params["norm_bias"]),
            params["query_kernel"],
            params["query_bias"],
        )
        key = _linear(interaction_memory, params["key_kernel"], params["key_bias"])
        value = _linear(interaction_memory, params["value_kernel"], params["value_bias"])
        head_dim = query.shape[-1] // num_heads
        query_heads = einops.rearrange(query, "b q (h d) -> b h q d", h=num_heads)
        key_heads = einops.rearrange(key, "b k (h d) -> b h k d", h=num_heads)
        value_heads = einops.rearrange(value, "b k (h d) -> b h k d", h=num_heads)
        if readout_mode == "recovery":
            query_heads = query_heads * jax.lax.rsqrt(
                jnp.mean(jnp.square(query_heads.astype(jnp.float32)), axis=-1, keepdims=True) + 1.0e-6
            ).astype(query_heads.dtype)
            key_heads = key_heads * jax.lax.rsqrt(
                jnp.mean(jnp.square(key_heads.astype(jnp.float32)), axis=-1, keepdims=True) + 1.0e-6
            ).astype(key_heads.dtype)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query_heads, key_heads) * (head_dim**-0.5)
        if readout_mode == "recovery":
            temperature = temperature_min + (temperature_max - temperature_min) * jax.nn.sigmoid(
                params["readout_temperature_logits"][gate_index]
            )
            logits = logits * temperature.astype(logits.dtype)
        weights = jax.nn.softmax(logits, axis=-1)
        delta = jnp.einsum("bhqk,bhkd->bhqd", weights, value_heads)
        delta = einops.rearrange(delta, "b h q d -> b q (h d)")
        delta = _linear(delta, params["output_kernel"], params["output_bias"])

        if gate_mode == "dynamic":
            if hidden.shape[-2] != params["slot_embeddings"].shape[0]:
                raise ValueError(
                    "dynamic gate action-slot embeddings must match the action hidden sequence length: "
                    f"{params['slot_embeddings'].shape[0]} != {hidden.shape[-2]}"
                )
            normalized_hidden = _layer_norm(
                hidden,
                params["gate_norm_scale"],
                params["gate_norm_bias"],
            )
            slot_embedding = jnp.broadcast_to(
                params["slot_embeddings"][None, :, :],
                (*hidden.shape[:2], params["slot_embeddings"].shape[-1]),
            )
            layer_embedding = jnp.broadcast_to(
                params["layer_embeddings"][gate_index][None, None, :],
                (*hidden.shape[:2], params["layer_embeddings"].shape[-1]),
            )
            expanded_flow_time = jnp.broadcast_to(
                flow_time_embedding[:, None, :],
                (*hidden.shape[:2], flow_time_embedding.shape[-1]),
            )
            gate_input = jnp.concatenate(
                [normalized_hidden, slot_embedding, expanded_flow_time, layer_embedding],
                axis=-1,
            )
            gate_hidden = jax.nn.silu(_linear(gate_input, params["gate_mlp_in_kernel"], params["gate_mlp_in_bias"]))
            raw_gate = _linear(
                gate_hidden,
                params["gate_mlp_out_kernel"],
                params["gate_mlp_out_bias"],
            )[..., 0]
        else:
            raw_gate = jnp.broadcast_to(params["gates"][gate_index], hidden.shape[:2])

        gate = jax.nn.sigmoid(raw_gate).astype(delta.dtype)
        gated_delta = gate[..., None] * delta
        # These values are diagnostics only. In particular, ``gated_delta`` is
        # exactly zero at initialization because the adapter output projection
        # is zero-initialized. Differentiating ``linalg.norm`` at that point is
        # undefined and can leak NaNs through the auxiliary output of the
        # scanned/conditional layer adapter even though the diagnostic has no
        # loss weight. Stop gradients before computing every diagnostic.
        diagnostic_gated_delta = jax.lax.stop_gradient(gated_delta.astype(jnp.float32))
        diagnostic_delta = jax.lax.stop_gradient(delta.astype(jnp.float32))
        diagnostic_hidden = jax.lax.stop_gradient(hidden.astype(jnp.float32))
        diagnostic_weights = jax.lax.stop_gradient(weights.astype(jnp.float32))
        injection_ratio = jnp.linalg.norm(diagnostic_gated_delta, axis=-1) / jnp.maximum(
            jnp.linalg.norm(diagnostic_hidden, axis=-1),
            1.0e-6,
        )
        ungated_delta_ratio = jnp.linalg.norm(diagnostic_delta, axis=-1) / jnp.maximum(
            jnp.linalg.norm(diagnostic_hidden, axis=-1),
            1.0e-6,
        )
        attention_entropy = -jnp.mean(
            jnp.sum(diagnostic_weights * jnp.log(jnp.maximum(diagnostic_weights, 1.0e-8)), axis=-1),
            axis=1,
        )
        aux = {
            "attention_entropy": attention_entropy,
            "gate": jax.lax.stop_gradient(gate.astype(jnp.float32)),
            "injection_ratio": injection_ratio,
            "head_slot_attention": diagnostic_weights,
            "raw_gate": jax.lax.stop_gradient(raw_gate.astype(jnp.float32)),
            "slot_attention": jnp.mean(diagnostic_weights, axis=1),
            "ungated_delta_ratio": ungated_delta_ratio,
        }
        return (hidden + gated_delta).astype(hidden.dtype), aux

    matches = layer_index == injection_layers
    empty_aux = {
        "attention_entropy": jnp.zeros(action_hidden.shape[:2], dtype=jnp.float32),
        "gate": jnp.zeros(action_hidden.shape[:2], dtype=jnp.float32),
        "injection_ratio": jnp.zeros(action_hidden.shape[:2], dtype=jnp.float32),
        "head_slot_attention": jnp.zeros(
            (action_hidden.shape[0], num_heads, action_hidden.shape[1], interaction_memory.shape[-2]),
            dtype=jnp.float32,
        ),
        "raw_gate": jnp.zeros(action_hidden.shape[:2], dtype=jnp.float32),
        "slot_attention": jnp.zeros(
            (*action_hidden.shape[:2], interaction_memory.shape[-2]),
            dtype=jnp.float32,
        ),
        "ungated_delta_ratio": jnp.zeros(action_hidden.shape[:2], dtype=jnp.float32),
    }
    action_hidden, aux = jax.lax.cond(
        jnp.any(matches),
        inject,
        lambda hidden: (hidden, empty_aux),
        action_hidden,
    )
    return [hidden_groups[0], action_hidden], aux


@dataclasses.dataclass(frozen=True)
class DemoVLAConfig(pi0_config.Pi0Config):
    """Configuration for DemoVLA interaction-memory models."""

    pi05: bool = True
    action_horizon: int = 10
    discrete_state_input: bool = False

    use_interaction_memory: bool = True
    num_interaction_tokens: int = 4
    interaction_num_heads: int = 8
    interaction_mlp_ratio: int = 4
    interaction_gate_init: float = -4.0
    interaction_gate_mode: Literal["scalar", "dynamic"] = "scalar"
    interaction_dynamic_gate_hidden_dim: int = 256
    interaction_dynamic_gate_embedding_dim: int = 64
    interaction_max_camera_views: int = 3
    interaction_injection_mode: Literal["single_shot", "sparse_deep"] = "sparse_deep"
    interaction_injection_layers: tuple[int, ...] = (4, 9, 14)
    interaction_adapter_share_weights: bool = True
    interaction_attention_diversity_weight: float = 0.0
    interaction_attention_diversity_margin: float = 0.5
    interaction_memory_diversity_weight: float = 0.0
    interaction_memory_diversity_margin: float = 0.5
    interaction_output_init_std: float = 0.0
    interaction_readout_mode: Literal["standard", "recovery"] = "standard"
    interaction_readout_temperature_min: float = 0.5
    interaction_readout_temperature_max: float = 2.0
    interaction_readout_temperature_init: float = 1.0
    interaction_memory_ranking_weight: float = 0.0
    interaction_memory_ranking_margin: float = 2.0e-5
    interaction_memory_ranking_mode: Literal["batch_shuffle", "prompt_hard_negative"] = "batch_shuffle"

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("DemoVLA currently supports pi0.5 only")
        if self.num_interaction_tokens <= 0:
            raise ValueError("num_interaction_tokens must be positive")
        if self.interaction_num_heads <= 0:
            raise ValueError("interaction_num_heads must be positive")
        if self.interaction_mlp_ratio <= 0:
            raise ValueError("interaction_mlp_ratio must be positive")
        if self.interaction_gate_mode not in ("scalar", "dynamic"):
            raise ValueError(f"unsupported interaction_gate_mode: {self.interaction_gate_mode}")
        if self.interaction_dynamic_gate_hidden_dim <= 0:
            raise ValueError("interaction_dynamic_gate_hidden_dim must be positive")
        if self.interaction_dynamic_gate_embedding_dim <= 0:
            raise ValueError("interaction_dynamic_gate_embedding_dim must be positive")
        if self.interaction_max_camera_views <= 0:
            raise ValueError("interaction_max_camera_views must be positive")
        if self.interaction_injection_mode not in ("single_shot", "sparse_deep"):
            raise ValueError(f"unsupported interaction_injection_mode: {self.interaction_injection_mode}")
        if not self.interaction_adapter_share_weights:
            raise ValueError("DemoVLA currently requires shared interaction adapter weights")
        if self.interaction_gate_mode == "dynamic" and self.interaction_injection_mode != "sparse_deep":
            raise ValueError("dynamic gate requires sparse_deep interaction injection")
        if self.interaction_attention_diversity_weight < 0.0:
            raise ValueError("interaction_attention_diversity_weight must be non-negative")
        if self.interaction_memory_diversity_weight < 0.0:
            raise ValueError("interaction_memory_diversity_weight must be non-negative")
        if self.interaction_output_init_std < 0.0:
            raise ValueError("interaction_output_init_std must be non-negative")
        if self.interaction_readout_mode not in ("standard", "recovery"):
            raise ValueError(f"unsupported interaction_readout_mode: {self.interaction_readout_mode}")
        if not 0.0 < self.interaction_readout_temperature_min < self.interaction_readout_temperature_max:
            raise ValueError("readout temperature bounds must satisfy 0 < min < max")
        if not (
            self.interaction_readout_temperature_min
            < self.interaction_readout_temperature_init
            < self.interaction_readout_temperature_max
        ):
            raise ValueError("readout temperature init must lie strictly inside its bounds")
        if self.interaction_memory_ranking_weight < 0.0:
            raise ValueError("interaction_memory_ranking_weight must be non-negative")
        if self.interaction_memory_ranking_margin < 0.0:
            raise ValueError("interaction_memory_ranking_margin must be non-negative")
        if self.interaction_memory_ranking_mode not in ("batch_shuffle", "prompt_hard_negative"):
            raise ValueError(f"unsupported interaction_memory_ranking_mode: {self.interaction_memory_ranking_mode}")
        if not -1.0 <= self.interaction_attention_diversity_margin <= 1.0:
            raise ValueError("interaction_attention_diversity_margin must be in [-1, 1]")
        if not -1.0 <= self.interaction_memory_diversity_margin <= 1.0:
            raise ValueError("interaction_memory_diversity_margin must be in [-1, 1]")

        action_config = _gemma.get_config(self.action_expert_variant)
        action_width = action_config.width
        if action_width % self.interaction_num_heads != 0:
            raise ValueError("action expert width must be divisible by interaction_num_heads")
        if self.interaction_injection_mode == "sparse_deep":
            if not self.interaction_injection_layers:
                raise ValueError("sparse_deep requires at least one interaction injection layer")
            if tuple(sorted(set(self.interaction_injection_layers))) != self.interaction_injection_layers:
                raise ValueError("interaction_injection_layers must be sorted and unique")
            if self.interaction_injection_layers[0] < 0 or self.interaction_injection_layers[-1] >= action_config.depth:
                raise ValueError(
                    f"interaction_injection_layers must be in [0, {action_config.depth}), "
                    f"got {self.interaction_injection_layers}"
                )

    @override
    def create(self, rng: at.KeyArrayLike) -> DemoVLA:
        return DemoVLA(self, rngs=nnx.Rngs(rng))

class DemoVLA(pi0.Pi0):
    """Pi0.5 with a self-grounded sparse interaction-memory adapter."""

    def __init__(self, config: DemoVLAConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.use_interaction_memory = config.use_interaction_memory
        self.num_interaction_tokens = config.num_interaction_tokens
        self.interaction_num_heads = config.interaction_num_heads
        self.interaction_max_camera_views = config.interaction_max_camera_views
        self.interaction_injection_mode = config.interaction_injection_mode
        self.interaction_injection_layers = config.interaction_injection_layers
        self.interaction_gate_mode = config.interaction_gate_mode
        # Inference-only controls. They are plain graph metadata rather than
        # parameters, so changing them before JIT compilation remains checkpoint
        # compatible and cannot modify the trained weights.
        self.interaction_inference_ablation = "normal"
        self.interaction_inference_layer_mean_gates = ()
        self.interaction_attention_diversity_weight = config.interaction_attention_diversity_weight
        self.interaction_attention_diversity_margin = config.interaction_attention_diversity_margin
        self.interaction_memory_diversity_weight = config.interaction_memory_diversity_weight
        self.interaction_memory_diversity_margin = config.interaction_memory_diversity_margin
        self.interaction_readout_mode = config.interaction_readout_mode
        self.interaction_readout_temperature_min = config.interaction_readout_temperature_min
        self.interaction_readout_temperature_max = config.interaction_readout_temperature_max
        self.interaction_memory_ranking_weight = config.interaction_memory_ranking_weight
        self.interaction_memory_ranking_margin = config.interaction_memory_ranking_margin
        self.interaction_memory_ranking_mode = config.interaction_memory_ranking_mode

        vlm_width = _gemma.get_config(config.paligemma_variant).width
        action_width = _gemma.get_config(config.action_expert_variant).width
        mlp_width = action_width * config.interaction_mlp_ratio

        query_key = rngs.params()
        camera_key = rngs.params()
        self.demovla_interaction_queries = nnx.Param(
            jax.random.normal(query_key, (config.num_interaction_tokens, action_width), dtype=jnp.float32) * 0.02
        )
        self.demovla_camera_embeddings = nnx.Param(
            jax.random.normal(
                camera_key,
                (config.interaction_max_camera_views, action_width),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.demovla_spatial_proj = nnx.Linear(2, action_width, rngs=rngs)

        self.demovla_text_query_norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.demovla_text_context_norm = nnx.LayerNorm(vlm_width, rngs=rngs)
        self.demovla_text_query_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.demovla_text_key_proj = nnx.Linear(vlm_width, action_width, rngs=rngs)
        self.demovla_text_value_proj = nnx.Linear(vlm_width, action_width, rngs=rngs)
        self.demovla_text_output_proj = nnx.Linear(action_width, action_width, rngs=rngs)

        self.demovla_visual_query_norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.demovla_visual_context_norm = nnx.LayerNorm(vlm_width, rngs=rngs)
        self.demovla_visual_query_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.demovla_visual_key_proj = nnx.Linear(vlm_width, action_width, rngs=rngs)
        self.demovla_visual_value_proj = nnx.Linear(vlm_width, action_width, rngs=rngs)
        self.demovla_visual_output_proj = nnx.Linear(action_width, action_width, rngs=rngs)

        self.demovla_memory_ffn_norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.demovla_memory_ffn_in = nnx.Linear(action_width, mlp_width, rngs=rngs)
        self.demovla_memory_ffn_out = nnx.Linear(mlp_width, action_width, rngs=rngs)

        self.demovla_action_query_norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.demovla_action_query_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.demovla_memory_key_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.demovla_memory_value_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        # A zero output projection guarantees exact pi0.5 behavior at initialization.
        self.demovla_action_output_proj = nnx.Linear(
            action_width,
            action_width,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        if config.interaction_readout_mode == "recovery":
            self.demovla_readout_layer_embeddings = nnx.Param(
                jnp.zeros((len(config.interaction_injection_layers), action_width), dtype=jnp.float32)
            )
            initial_fraction = (
                (config.interaction_readout_temperature_init - config.interaction_readout_temperature_min)
                / (config.interaction_readout_temperature_max - config.interaction_readout_temperature_min)
            )
            initial_logit = jnp.log(initial_fraction) - jnp.log1p(-initial_fraction)
            self.demovla_readout_temperature_logits = nnx.Param(
                jnp.full((len(config.interaction_injection_layers),), initial_logit, dtype=jnp.float32)
            )
        if config.interaction_output_init_std > 0.0:
            output_key = rngs.params()
            self.demovla_action_output_proj.kernel.value = (
                jax.random.normal(
                    output_key,
                    self.demovla_action_output_proj.kernel.value.shape,
                    dtype=jnp.float32,
                )
                * config.interaction_output_init_std
            )
        if config.interaction_gate_mode == "dynamic":
            gate_embedding_dim = config.interaction_dynamic_gate_embedding_dim
            slot_key = rngs.params()
            layer_key = rngs.params()
            self.demovla_dynamic_gate_slot_embeddings = nnx.Param(
                jax.random.normal(
                    slot_key,
                    (config.action_horizon, gate_embedding_dim),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            self.demovla_dynamic_gate_layer_embeddings = nnx.Param(
                jax.random.normal(
                    layer_key,
                    (len(config.interaction_injection_layers), gate_embedding_dim),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            self.demovla_dynamic_gate_norm = nnx.LayerNorm(action_width, rngs=rngs)
            gate_input_dim = 2 * action_width + 2 * gate_embedding_dim
            self.demovla_dynamic_gate_mlp_in = nnx.Linear(
                gate_input_dim,
                config.interaction_dynamic_gate_hidden_dim,
                rngs=rngs,
            )
            self.demovla_dynamic_gate_mlp_out = nnx.Linear(
                config.interaction_dynamic_gate_hidden_dim,
                1,
                kernel_init=nnx.initializers.zeros,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            # Assign the requested initial value after construction instead of using
            # `initializers.constant`, which creates a new closure for every model
            # instance and makes otherwise identical NNX GraphDefs compare unequal.
            self.demovla_dynamic_gate_mlp_out.bias.value = jnp.full(
                self.demovla_dynamic_gate_mlp_out.bias.value.shape,
                config.interaction_gate_init,
                dtype=jnp.float32,
            )
        else:
            num_gates = (
                1 if config.interaction_injection_mode == "single_shot" else len(config.interaction_injection_layers)
            )
            self.demovla_interaction_gates = nnx.Param(
                jnp.full((num_gates,), config.interaction_gate_init, dtype=jnp.float32)
            )

    def _attention_with_weights(
        self,
        query: at.Float[at.Array, "b q d"],
        key: at.Float[at.Array, "b k d"],
        value: at.Float[at.Array, "b k d"],
        key_mask: at.Bool[at.Array, "b k"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b q d"],
        at.Float[at.Array, "b h q k"],
    ]:
        num_heads = self.interaction_num_heads
        head_dim = query.shape[-1] // num_heads
        query = einops.rearrange(query, "b q (h d) -> b h q d", h=num_heads)
        key = einops.rearrange(key, "b k (h d) -> b h k d", h=num_heads)
        value = einops.rearrange(value, "b k (h d) -> b h k d", h=num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) * (head_dim**-0.5)

        if key_mask is not None:
            mask = key_mask[:, None, None, :]
            logits = jnp.where(mask, logits, -1.0e30)
            weights = jax.nn.softmax(logits, axis=-1) * mask.astype(logits.dtype)
            weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-6)
        else:
            weights = jax.nn.softmax(logits, axis=-1)

        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        return einops.rearrange(attended, "b h q d -> b q (h d)"), weights

    def _attention(
        self,
        query: at.Float[at.Array, "b q d"],
        key: at.Float[at.Array, "b k d"],
        value: at.Float[at.Array, "b k d"],
        key_mask: at.Bool[at.Array, "b k"] | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        attended, _ = self._attention_with_weights(query, key, value, key_mask)
        return attended

    def _patch_grid_shape(self) -> tuple[int, int]:
        return (
            _model.IMAGE_RESOLUTION[0] // _SIGLIP_PATCH_SIZE,
            _model.IMAGE_RESOLUTION[1] // _SIGLIP_PATCH_SIZE,
        )

    def _prefix_groups(
        self,
        obs: _model.Observation,
        prefix_hidden: at.Float[at.Array, "b prefix_s vlm_d"],
    ) -> tuple[
        at.Float[at.Array, "b visual_s vlm_d"],
        at.Bool[at.Array, "b visual_s"],
        at.Float[at.Array, "b text_s vlm_d"],
        at.Bool[at.Array, "b text_s"],
        at.Float[at.Array, "b visual_s interaction_d"],
    ]:
        grid_height, grid_width = self._patch_grid_shape()
        tokens_per_view = grid_height * grid_width
        num_views = len(obs.images)
        if num_views > self.interaction_max_camera_views:
            raise ValueError(
                f"DemoVLA received {num_views} camera views, but interaction_max_camera_views="
                f"{self.interaction_max_camera_views}"
            )

        visual_length = num_views * tokens_per_view
        visual_tokens = prefix_hidden[:, :visual_length]
        text_tokens = prefix_hidden[:, visual_length:]
        visual_mask = jnp.concatenate(
            [einops.repeat(obs.image_masks[name], "b -> b s", s=tokens_per_view) for name in obs.images],
            axis=1,
        )
        if obs.tokenized_prompt_mask is None:
            text_mask = jnp.zeros(text_tokens.shape[:2], dtype=jnp.bool_)
        else:
            text_mask = obs.tokenized_prompt_mask

        ys = jnp.linspace(0.0, 1.0, grid_height, dtype=jnp.float32)
        xs = jnp.linspace(0.0, 1.0, grid_width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        coordinates = jnp.stack([xx, yy], axis=-1).reshape(tokens_per_view, 2)
        spatial = self.demovla_spatial_proj(coordinates)
        spatial = jnp.tile(spatial[None, :, :], (num_views, 1, 1))
        spatial = spatial + self.demovla_camera_embeddings.value[:num_views, None, :]
        spatial = spatial.reshape(1, visual_length, -1)
        spatial = jnp.broadcast_to(spatial, (prefix_hidden.shape[0], visual_length, spatial.shape[-1]))
        return visual_tokens, visual_mask, text_tokens, text_mask, spatial

    def _extract_interaction_memory_and_attention(
        self,
        obs: _model.Observation,
        prefix_hidden: at.Float[at.Array, "b prefix_s vlm_d"],
    ) -> tuple[
        at.Float[at.Array, "b interaction_s interaction_d"],
        at.Float[at.Array, "b interaction_s visual_s"],
    ]:
        """Extract sparse memory and head-averaged visual patch attention."""
        visual_tokens, visual_mask, text_tokens, text_mask, spatial = self._prefix_groups(obs, prefix_hidden)
        batch_size = prefix_hidden.shape[0]
        queries = jnp.broadcast_to(
            self.demovla_interaction_queries.value[None, :, :],
            (batch_size, self.num_interaction_tokens, self.demovla_interaction_queries.value.shape[-1]),
        )

        text_query = self.demovla_text_query_proj(self.demovla_text_query_norm(queries))
        text_context = self.demovla_text_context_norm(text_tokens)
        text_delta = self._attention(
            text_query,
            self.demovla_text_key_proj(text_context),
            self.demovla_text_value_proj(text_context),
            text_mask,
        )
        queries = queries + self.demovla_text_output_proj(text_delta)

        visual_query = self.demovla_visual_query_proj(self.demovla_visual_query_norm(queries))
        visual_context = self.demovla_visual_context_norm(visual_tokens)
        visual_delta, visual_attention_heads = self._attention_with_weights(
            visual_query,
            self.demovla_visual_key_proj(visual_context) + spatial,
            self.demovla_visual_value_proj(visual_context) + spatial,
            visual_mask,
        )
        memory = queries + self.demovla_visual_output_proj(visual_delta)
        normalized = self.demovla_memory_ffn_norm(memory)
        memory = memory + self.demovla_memory_ffn_out(nnx.swish(self.demovla_memory_ffn_in(normalized)))
        visual_attention = jnp.mean(visual_attention_heads, axis=1)
        return memory, visual_attention

    def extract_interaction_memory(
        self,
        obs: _model.Observation,
        prefix_hidden: at.Float[at.Array, "b prefix_s vlm_d"],
    ) -> at.Float[at.Array, "b interaction_s interaction_d"]:
        """Extract language-conditioned sparse interaction memory."""
        memory, _ = self._extract_interaction_memory_and_attention(obs, prefix_hidden)
        return memory

    def interaction_patch_diagnostics(
        self,
        obs: _model.Observation,
        prefix_hidden: at.Float[at.Array, "b prefix_s vlm_d"],
        *,
        top_k: int = 4,
    ) -> tuple[
        at.Float[at.Array, "b interaction_s interaction_d"],
        dict[str, at.Array],
    ]:
        """Return interaction memory plus selected visual patch positions.

        Attention is averaged across extractor heads. Top-k patches are selected
        jointly across all camera views for each interaction query.
        """
        grid_height, grid_width = self._patch_grid_shape()
        tokens_per_view = grid_height * grid_width
        num_views = len(obs.images)
        visual_length = num_views * tokens_per_view
        if not 0 < top_k <= visual_length:
            raise ValueError(f"top_k must be in [1, {visual_length}], got {top_k}")

        memory, visual_attention = self._extract_interaction_memory_and_attention(obs, prefix_hidden)
        top_weights, top_indices = jax.lax.top_k(visual_attention, top_k)
        view_indices = top_indices // tokens_per_view
        patch_indices = top_indices % tokens_per_view
        patch_y = patch_indices // grid_width
        patch_x = patch_indices % grid_width
        patch_xy = jnp.stack(
            [
                (patch_x.astype(jnp.float32) + 0.5) / grid_width,
                (patch_y.astype(jnp.float32) + 0.5) / grid_height,
            ],
            axis=-1,
        )
        visual_attention = visual_attention.reshape(
            visual_attention.shape[0],
            self.num_interaction_tokens,
            num_views,
            tokens_per_view,
        )
        camera_mask = jnp.stack([obs.image_masks[name] for name in obs.images], axis=-1)
        diagnostics = {
            "interaction_visual_attention": visual_attention,
            "interaction_top_patch_indices": top_indices,
            "interaction_top_patch_weights": top_weights,
            "interaction_top_patch_view_indices": view_indices,
            "interaction_top_patch_xy": patch_xy,
            "interaction_camera_mask": camera_mask,
        }
        return memory, diagnostics

    def inject_interaction_memory(
        self,
        action_tokens: at.Float[at.Array, "b action_s action_d"],
        interaction_memory: at.Float[at.Array, "b interaction_s action_d"],
        gate_index: int | at.Int[at.Array, ""] = 0,
    ) -> at.Float[at.Array, "b action_s action_d"]:
        """Let the current action hidden states retrieve cached interaction memory."""
        if self.interaction_gate_mode != "scalar":
            raise ValueError("inject_interaction_memory is only used by the scalar single-shot gate")
        query = self.demovla_action_query_proj(self.demovla_action_query_norm(action_tokens))
        delta = self._attention(
            query,
            self.demovla_memory_key_proj(interaction_memory),
            self.demovla_memory_value_proj(interaction_memory),
        )
        delta = self.demovla_action_output_proj(delta)
        gate = jax.nn.sigmoid(self.demovla_interaction_gates.value[gate_index]).astype(delta.dtype)
        return action_tokens + gate * delta

    def configure_interaction_inference_ablation(
        self,
        mode: Literal["normal", "layer_mean", "off", "zero_memory", "batch_shuffle"],
        layer_mean_gates: tuple[float, ...] = (),
    ) -> None:
        """Configure parameter-free memory and gate interventions before JIT compilation.

        An explicit ``layer_mean_gates`` tuple may be composed with a memory
        intervention such as ``zero_memory``. This keeps the gate probabilities
        identical between correct- and ablated-memory policies.
        """
        if mode not in ("normal", "layer_mean", "off", "zero_memory", "batch_shuffle"):
            raise ValueError(f"unsupported interaction inference ablation: {mode}")
        if mode == "layer_mean" and not layer_mean_gates:
            raise ValueError("layer_mean ablation requires fixed gate probabilities")
        if layer_mean_gates:
            if mode == "off":
                raise ValueError("fixed gate probabilities cannot be combined with the off ablation")
            if self.interaction_injection_mode != "sparse_deep":
                raise ValueError("fixed gate probabilities require a sparse-deep model")
            if len(layer_mean_gates) != len(self.interaction_injection_layers):
                raise ValueError(
                    "fixed gate override requires one gate probability per injection layer: "
                    f"expected {len(self.interaction_injection_layers)}, got {len(layer_mean_gates)}"
                )
            if any(not 0.0 <= gate < 1.0 for gate in layer_mean_gates):
                raise ValueError("fixed gate probabilities must be in [0, 1)")

        self.interaction_inference_ablation = mode
        self.interaction_inference_layer_mean_gates = tuple(float(gate) for gate in layer_mean_gates)

    def _apply_interaction_memory_ablation(
        self,
        interaction_memory: at.Float[at.Array, "b interaction_s action_d"],
    ) -> at.Float[at.Array, "b interaction_s action_d"]:
        """Apply a memory-only intervention without changing the VLM prefix.

        ``batch_shuffle`` uses a deterministic half-batch cyclic shift. This is
        a derangement for even evaluation batches and avoids consuming another
        RNG stream, so correct and shuffled losses retain identical flow noise.
        It is intentionally rejected for online batch-1 rollout, where a batch
        shuffle would otherwise be the identity intervention.
        """
        if self.interaction_inference_ablation == "zero_memory":
            return jnp.zeros_like(interaction_memory)
        if self.interaction_inference_ablation == "batch_shuffle":
            batch_size = interaction_memory.shape[0]
            if batch_size < 2:
                raise ValueError("batch_shuffle memory ablation requires batch size >= 2")
            return jnp.roll(interaction_memory, shift=max(1, batch_size // 2), axis=0)
        return interaction_memory

    def _make_prompt_hard_negative(
        self,
        observation: _model.Observation,
    ) -> tuple[_model.Observation, at.Float[at.Array, " b"]]:
        """Replace only the prompt with the closest different prompt in the batch.

        Similarity is positional token overlap over valid prompt positions. Exact
        duplicates are excluded, so repeated examples from the same LIBERO task
        cannot become false negatives. Images, state, and image augmentation stay
        bit-identical to the correct observation.
        """
        tokens = observation.tokenized_prompt
        masks = observation.tokenized_prompt_mask
        if tokens is None or masks is None:
            raise ValueError("prompt hard-negative ranking requires tokenized prompts")
        if tokens.ndim != 2 or tokens.shape[0] < 2:
            raise ValueError("prompt hard-negative ranking requires a batch size of at least two")

        token_equal = tokens[:, None, :] == tokens[None, :, :]
        mask_equal = masks[:, None, :] == masks[None, :, :]
        same_prompt = jnp.all(token_equal & mask_equal, axis=-1)
        jointly_valid = masks[:, None, :] & masks[None, :, :]
        overlap = jnp.sum(token_equal & jointly_valid, axis=-1).astype(jnp.float32)
        union_length = jnp.sum(masks[:, None, :] | masks[None, :, :], axis=-1).astype(jnp.float32)
        similarity = overlap / jnp.maximum(union_length, 1.0)
        similarity = jnp.where(same_prompt, -jnp.inf, similarity)
        negative_indices = jnp.argmax(similarity, axis=-1)
        has_negative = jnp.any(~same_prompt, axis=-1)
        fallback_indices = (jnp.arange(tokens.shape[0]) + 1) % tokens.shape[0]
        negative_indices = jnp.where(has_negative, negative_indices, fallback_indices)
        selected_similarity = jnp.take_along_axis(
            similarity,
            negative_indices[:, None],
            axis=1,
        )[:, 0]
        selected_similarity = jnp.where(has_negative, selected_similarity, 0.0)
        return (
            observation.replace(
                tokenized_prompt=tokens[negative_indices],
                tokenized_prompt_mask=masks[negative_indices],
            ),
            selected_similarity,
        )

    def _make_sparse_deep_adapter(
        self,
        interaction_memory: at.Float[at.Array, "b interaction_s action_d"],
        flow_time_embedding: at.Float[at.Array, "b action_d"],
    ):
        """Build the generic Gemma hook used only during the action-expert pass."""
        params = {
            "norm_scale": self.demovla_action_query_norm.scale.value,
            "norm_bias": self.demovla_action_query_norm.bias.value,
            "query_kernel": self.demovla_action_query_proj.kernel.value,
            "query_bias": self.demovla_action_query_proj.bias.value,
            "key_kernel": self.demovla_memory_key_proj.kernel.value,
            "key_bias": self.demovla_memory_key_proj.bias.value,
            "value_kernel": self.demovla_memory_value_proj.kernel.value,
            "value_bias": self.demovla_memory_value_proj.bias.value,
            "output_kernel": self.demovla_action_output_proj.kernel.value,
            "output_bias": self.demovla_action_output_proj.bias.value,
        }
        if self.interaction_readout_mode == "recovery":
            params.update(
                {
                    "readout_layer_embeddings": self.demovla_readout_layer_embeddings.value,
                    "readout_temperature_logits": self.demovla_readout_temperature_logits.value,
                }
            )
        adapter_gate_mode = self.interaction_gate_mode
        if self.interaction_inference_layer_mean_gates:
            gate_probabilities = jnp.asarray(self.interaction_inference_layer_mean_gates, dtype=jnp.float32)
            params["gates"] = jnp.where(
                gate_probabilities == 0.0,
                -jnp.inf,
                jnp.log(gate_probabilities) - jnp.log1p(-gate_probabilities),
            )
            adapter_gate_mode = "scalar"
        elif self.interaction_gate_mode == "dynamic":
            params.update(
                {
                    "gate_norm_scale": self.demovla_dynamic_gate_norm.scale.value,
                    "gate_norm_bias": self.demovla_dynamic_gate_norm.bias.value,
                    "slot_embeddings": self.demovla_dynamic_gate_slot_embeddings.value,
                    "layer_embeddings": self.demovla_dynamic_gate_layer_embeddings.value,
                    "gate_mlp_in_kernel": self.demovla_dynamic_gate_mlp_in.kernel.value,
                    "gate_mlp_in_bias": self.demovla_dynamic_gate_mlp_in.bias.value,
                    "gate_mlp_out_kernel": self.demovla_dynamic_gate_mlp_out.kernel.value,
                    "gate_mlp_out_bias": self.demovla_dynamic_gate_mlp_out.bias.value,
                }
            )
        else:
            params["gates"] = self.demovla_interaction_gates.value
        return jax.tree_util.Partial(
            functools.partial(
                _apply_sparse_deep_adapter,
                self.interaction_num_heads,
                adapter_gate_mode,
                readout_mode=self.interaction_readout_mode,
                temperature_min=self.interaction_readout_temperature_min,
                temperature_max=self.interaction_readout_temperature_max,
            ),
            params,
            interaction_memory,
            flow_time_embedding,
            jnp.asarray(self.interaction_injection_layers, dtype=jnp.int32),
        )

    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        interaction_memory: at.Float[at.Array, "b interaction_s action_d"] | None = None,
    ):
        tokens, input_mask, ar_mask, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)
        if (
            self.use_interaction_memory
            and self.interaction_injection_mode == "single_shot"
            and interaction_memory is not None
            and self.interaction_inference_ablation != "off"
        ):
            action_tokens = self.inject_interaction_memory(tokens[:, -self.action_horizon :], interaction_memory)
            tokens = tokens.at[:, -self.action_horizon :].set(action_tokens)
        return tokens, input_mask, ar_mask, adarms_cond

    def _encode_prefix(self, observation: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_hidden, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        assert prefix_hidden is not None
        return prefix_tokens.shape[1], prefix_mask, kv_cache, prefix_hidden

    def _forward_action_expert(
        self,
        suffix_tokens: at.Float[at.Array, "b suffix_s action_d"],
        suffix_mask: at.Bool[at.Array, "b suffix_s"],
        suffix_ar_mask: at.Bool[at.Array, " suffix_s"],
        adarms_cond: at.Float[at.Array, "b action_d"] | None,
        prefix_length: int,
        prefix_mask: at.Bool[at.Array, "b prefix_s"],
        kv_cache: _gemma.KVCache,
        interaction_memory: at.Float[at.Array, "b interaction_s action_d"],
        *,
        return_adapter_aux: bool = False,
    ):
        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        if full_attn_mask.shape[-1] != prefix_length + suffix_tokens.shape[1]:
            raise ValueError("prefix KV cache and suffix attention mask lengths do not match")
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        layer_adapter = None
        if self.interaction_injection_mode == "sparse_deep" and self.interaction_inference_ablation != "off":
            if adarms_cond is None:
                raise ValueError("sparse-deep dynamic injection requires the pi0.5 flow-time embedding")
            layer_adapter = self._make_sparse_deep_adapter(interaction_memory, adarms_cond)
        if return_adapter_aux:
            (prefix_out, suffix_out), _, adapter_aux = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                layer_adapter=layer_adapter,
                return_layer_adapter_aux=True,
            )
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                layer_adapter=layer_adapter,
            )
            adapter_aux = None
        assert prefix_out is None
        assert suffix_out is not None
        return suffix_out, adapter_aux

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_aux(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Return flow loss plus query-diversity regularizers and diagnostics."""
        if not self.use_interaction_memory:
            return super().compute_loss(rng, observation, actions, train=train), {}

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_length, prefix_mask, kv_cache, prefix_hidden = self._encode_prefix(observation)
        interaction_memory, visual_attention = self._extract_interaction_memory_and_attention(
            observation, prefix_hidden
        )
        interaction_memory = self._apply_interaction_memory_ablation(interaction_memory)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, interaction_memory
        )
        suffix_out, adapter_aux = self._forward_action_expert(
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
            prefix_length,
            prefix_mask,
            kv_cache,
            interaction_memory,
            return_adapter_aux=(
                self.interaction_injection_mode == "sparse_deep"
                and self.interaction_inference_ablation != "off"
            ),
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(velocity - u_t), axis=-1)

        ranking_regularization = jnp.asarray(0.0, dtype=jnp.float32)
        shuffled_flow_loss = jnp.asarray(0.0, dtype=jnp.float32)
        counterfactual_flow_loss = jnp.asarray(0.0, dtype=jnp.float32)
        prompt_negative_similarity = jnp.asarray(0.0, dtype=jnp.float32)
        ranking_active_rate = jnp.asarray(0.0, dtype=jnp.float32)
        if self.interaction_memory_ranking_weight > 0.0:
            if self.interaction_memory_ranking_mode == "batch_shuffle":
                counterfactual_memory = jnp.roll(interaction_memory, shift=1, axis=0)
            else:
                negative_observation, negative_similarity = self._make_prompt_hard_negative(observation)
                _, _, _, negative_prefix_hidden = self._encode_prefix(negative_observation)
                counterfactual_memory, _ = self._extract_interaction_memory_and_attention(
                    negative_observation,
                    negative_prefix_hidden,
                )
                counterfactual_memory = self._apply_interaction_memory_ablation(counterfactual_memory)
                prompt_negative_similarity = jnp.mean(negative_similarity)
            counterfactual_suffix_out, _ = self._forward_action_expert(
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                prefix_length,
                prefix_mask,
                kv_cache,
                counterfactual_memory,
                return_adapter_aux=False,
            )
            counterfactual_velocity = self.action_out_proj(
                counterfactual_suffix_out[:, -self.action_horizon :]
            )
            counterfactual_flow = jnp.mean(jnp.square(counterfactual_velocity - u_t), axis=-1)
            counterfactual_flow_loss = jnp.mean(counterfactual_flow)
            if self.interaction_memory_ranking_mode == "batch_shuffle":
                shuffled_flow_loss = counterfactual_flow_loss
            ranking_hinge = jax.nn.relu(
                self.interaction_memory_ranking_margin + flow_loss - counterfactual_flow
            )
            ranking_active_rate = jnp.mean(ranking_hinge > 0.0)
            ranking_regularization = self.interaction_memory_ranking_weight * jnp.mean(ranking_hinge)

        attention_diversity_loss, attention_pair_cosine_mean, attention_pair_cosine_max = _pairwise_diversity_loss(
            visual_attention,
            self.interaction_attention_diversity_margin,
        )
        memory_diversity_loss, memory_pair_cosine_mean, memory_pair_cosine_max = _pairwise_diversity_loss(
            interaction_memory,
            self.interaction_memory_diversity_margin,
        )
        diversity_regularization = (
            self.interaction_attention_diversity_weight * attention_diversity_loss
            + self.interaction_memory_diversity_weight * memory_diversity_loss
        )
        auxiliary_regularization = diversity_regularization + ranking_regularization
        total_loss = flow_loss + auxiliary_regularization.astype(flow_loss.dtype)
        metrics = {
            "demovla_flow_loss": jnp.mean(flow_loss),
            "demovla_attention_diversity_loss": attention_diversity_loss,
            "demovla_attention_pair_cosine_mean": attention_pair_cosine_mean,
            "demovla_attention_pair_cosine_max": attention_pair_cosine_max,
            "demovla_memory_diversity_loss": memory_diversity_loss,
            "demovla_memory_pair_cosine_mean": memory_pair_cosine_mean,
            "demovla_memory_pair_cosine_max": memory_pair_cosine_max,
            "demovla_diversity_regularization": diversity_regularization,
            "demovla_memory_ranking_regularization": ranking_regularization,
            "demovla_shuffled_memory_flow_loss": shuffled_flow_loss,
            "demovla_counterfactual_memory_flow_loss": counterfactual_flow_loss,
            "demovla_prompt_negative_similarity": prompt_negative_similarity,
            "demovla_memory_ranking_active_rate": ranking_active_rate,
            "demovla_auxiliary_regularization": auxiliary_regularization,
        }
        if adapter_aux is not None:
            injection_layer_indices = jnp.asarray(self.interaction_injection_layers, dtype=jnp.int32)
            dynamic_aux = jax.tree.map(lambda value: value[injection_layer_indices], adapter_aux)
            gates = dynamic_aux["gate"]
            metrics.update(
                {
                    "demovla_dynamic_gate_mean": jnp.mean(gates),
                    "demovla_dynamic_gate_std": jnp.std(gates),
                    "demovla_dynamic_gate_raw_mean": jnp.mean(dynamic_aux["raw_gate"]),
                    "demovla_dynamic_injection_ratio_mean": jnp.mean(dynamic_aux["injection_ratio"]),
                    "demovla_dynamic_ungated_delta_ratio_mean": jnp.mean(dynamic_aux["ungated_delta_ratio"]),
                    "demovla_dynamic_attention_entropy_mean": jnp.mean(dynamic_aux["attention_entropy"]),
                    "demovla_dynamic_flow_time_mean": jnp.mean(time),
                    "demovla_adapter_gate_mean": jnp.mean(gates),
                    "demovla_adapter_injection_ratio_mean": jnp.mean(dynamic_aux["injection_ratio"]),
                    "demovla_adapter_ungated_delta_ratio_mean": jnp.mean(dynamic_aux["ungated_delta_ratio"]),
                }
            )
            for layer_offset, layer in enumerate(self.interaction_injection_layers):
                layer_gates = gates[layer_offset]
                layer_injection_ratio = dynamic_aux["injection_ratio"][layer_offset]
                layer_ungated_delta_ratio = dynamic_aux["ungated_delta_ratio"][layer_offset]
                layer_attention_entropy = dynamic_aux["attention_entropy"][layer_offset]
                metrics[f"demovla_dynamic_gate_layer_{layer}_mean"] = jnp.mean(layer_gates)
                metrics[f"demovla_dynamic_gate_layer_{layer}_std"] = jnp.std(layer_gates)
                metrics[f"demovla_dynamic_injection_ratio_layer_{layer}_mean"] = jnp.mean(layer_injection_ratio)
                metrics[f"demovla_dynamic_ungated_delta_ratio_layer_{layer}_mean"] = jnp.mean(
                    layer_ungated_delta_ratio
                )
                metrics[f"demovla_adapter_injection_ratio_layer_{layer}_mean"] = jnp.mean(layer_injection_ratio)
                metrics[f"demovla_adapter_ungated_delta_ratio_layer_{layer}_mean"] = jnp.mean(
                    layer_ungated_delta_ratio
                )
                metrics[f"demovla_dynamic_attention_entropy_layer_{layer}_mean"] = jnp.mean(layer_attention_entropy)
                for slot in range(self.action_horizon):
                    metrics[f"demovla_dynamic_gate_layer_{layer}_slot_{slot}_mean"] = jnp.mean(layer_gates[:, slot])
                for flow_bin in range(_DYNAMIC_GATE_FLOW_BINS):
                    lower = flow_bin / _DYNAMIC_GATE_FLOW_BINS
                    upper = (flow_bin + 1) / _DYNAMIC_GATE_FLOW_BINS
                    flow_mask = (time >= lower) & (
                        time <= upper if flow_bin == _DYNAMIC_GATE_FLOW_BINS - 1 else time < upper
                    )
                    flow_mask_float = flow_mask.astype(jnp.float32)
                    sample_count = jnp.sum(flow_mask_float)
                    safe_count = jnp.maximum(sample_count, 1.0)
                    slot_gate_mean = jnp.sum(layer_gates * flow_mask_float[:, None], axis=0) / safe_count
                    centered_gate = layer_gates - slot_gate_mean[None, :]
                    slot_gate_std = jnp.sqrt(
                        jnp.sum(jnp.square(centered_gate) * flow_mask_float[:, None], axis=0) / safe_count
                    )
                    metric_prefix = f"demovla_dynamic_layer_{layer}_flow_bin_{flow_bin}"
                    metrics[f"{metric_prefix}_sample_count"] = sample_count
                    metrics[f"{metric_prefix}_gate_mean"] = jnp.mean(slot_gate_mean)
                    metrics[f"{metric_prefix}_gate_std"] = jnp.mean(slot_gate_std)
                    metrics[f"{metric_prefix}_injection_ratio_mean"] = jnp.sum(
                        layer_injection_ratio * flow_mask_float[:, None]
                    ) / (safe_count * self.action_horizon)
                    metrics[f"{metric_prefix}_ungated_delta_ratio_mean"] = jnp.sum(
                        layer_ungated_delta_ratio * flow_mask_float[:, None]
                    ) / (safe_count * self.action_horizon)
                    metrics[f"{metric_prefix}_attention_entropy_mean"] = jnp.sum(
                        layer_attention_entropy * flow_mask_float[:, None]
                    ) / (safe_count * self.action_horizon)
                    for slot in range(self.action_horizon):
                        metrics[f"{metric_prefix}_slot_{slot}_gate_mean"] = slot_gate_mean[slot]
        return total_loss, metrics

    def compute_flow_loss_with_memory_source(
        self,
        rng: at.KeyArrayLike,
        action_observation: _model.Observation,
        memory_observation: _model.Observation,
        actions: _model.Actions,
    ) -> tuple[
        at.Float[at.Array, "*b ah"],
        at.Float[at.Array, "*b interaction_s action_d"],
        at.Float[at.Array, "*b interaction_s views patches"],
    ]:
        """Compute flow loss while sourcing memory from a counterfactual prompt.

        The action prefix/KV cache always comes from ``action_observation``.
        Only the interaction-memory extractor sees ``memory_observation``, which
        isolates task-conditioned memory from the base VLM prompt pathway.
        """
        if not self.use_interaction_memory:
            raise ValueError("memory-source diagnostics require use_interaction_memory=True")
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        action_observation = _model.preprocess_observation(
            preprocess_rng,
            action_observation,
            train=False,
        )
        memory_observation = _model.preprocess_observation(
            preprocess_rng,
            memory_observation,
            train=False,
        )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_length, prefix_mask, kv_cache, _ = self._encode_prefix(action_observation)
        _, _, _, memory_prefix_hidden = self._encode_prefix(memory_observation)
        interaction_memory, visual_attention = self._extract_interaction_memory_and_attention(
            memory_observation,
            memory_prefix_hidden,
        )
        interaction_memory = self._apply_interaction_memory_ablation(interaction_memory)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            action_observation,
            x_t,
            time,
            interaction_memory,
        )
        suffix_out, _ = self._forward_action_expert(
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
            prefix_length,
            prefix_mask,
            kv_cache,
            interaction_memory,
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(velocity - u_t), axis=-1)
        return flow_loss, interaction_memory, visual_attention

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if not self.use_interaction_memory:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_length, prefix_mask, kv_cache, prefix_hidden = self._encode_prefix(observation)
        interaction_memory = self.extract_interaction_memory(observation, prefix_hidden)
        interaction_memory = self._apply_interaction_memory_ablation(interaction_memory)

        def step(carry):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                batch_time,
                interaction_memory,
            )
            suffix_out, _ = self._forward_action_expert(
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                prefix_length,
                prefix_mask,
                kv_cache,
                interaction_memory,
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions

    def sample_actions_with_memory_source(
        self,
        rng: at.KeyArrayLike,
        action_observation: _model.Observation,
        memory_observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        interaction_memory_override: at.Float[at.Array, "b interaction_s action_d"] | None = None,
    ) -> _model.Actions:
        """Sample actions while changing only the interaction-memory source.

        ``action_observation`` always supplies the action VLM prefix, suffix,
        state, and KV cache. ``memory_observation`` is encoded independently and
        is visible only to the DemoVLA memory extractor. Supplying
        ``interaction_memory_override`` bypasses that extraction entirely. This
        interface supports prompt-only counterfactuals and memory transplant
        experiments without changing the base policy prompt pathway.
        """
        if not self.use_interaction_memory:
            raise ValueError("memory-source sampling requires use_interaction_memory=True")

        action_observation = _model.preprocess_observation(None, action_observation, train=False)
        memory_observation = _model.preprocess_observation(None, memory_observation, train=False)
        dt = -1.0 / num_steps
        batch_size = action_observation.state.shape[0]
        if memory_observation.state.shape[0] != batch_size:
            raise ValueError("action and memory observations must have matching batch sizes")
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_length, prefix_mask, kv_cache, _ = self._encode_prefix(action_observation)
        if interaction_memory_override is None:
            _, _, _, memory_prefix_hidden = self._encode_prefix(memory_observation)
            interaction_memory = self.extract_interaction_memory(memory_observation, memory_prefix_hidden)
        else:
            interaction_memory = interaction_memory_override
            expected_prefix = (batch_size, self.num_interaction_tokens)
            if interaction_memory.ndim != 3 or interaction_memory.shape[:2] != expected_prefix:
                raise ValueError(
                    "interaction memory override must have shape "
                    f"[batch, {self.num_interaction_tokens}, width], got {interaction_memory.shape}"
                )
        interaction_memory = self._apply_interaction_memory_ablation(interaction_memory)

        def step(carry):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                action_observation,
                x_t,
                batch_time,
                interaction_memory,
            )
            suffix_out, _ = self._forward_action_expert(
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                prefix_length,
                prefix_mask,
                kv_cache,
                interaction_memory,
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions

    def sample_actions_with_interaction_diagnostics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        top_k: int = 4,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        """Sample actions and return extractor plus action-read diagnostics."""
        actions = self.sample_actions(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        diagnostics = self.interaction_action_diagnostics(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            top_k=top_k,
        )
        return actions, diagnostics

    def interaction_action_diagnostics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        top_k: int = 4,
    ) -> dict[str, at.Array]:
        """Replay denoising and expose how each injection layer reads visual memory.

        This diagnostic runs separately from the standard action sampler. Given
        the same RNG/noise, it follows the same denoising trajectory while
        collecting adapter auxiliaries; returned policy actions still come from
        the unmodified standard sampling graph.
        """
        if not self.use_interaction_memory:
            raise ValueError("interaction diagnostics require use_interaction_memory=True")
        if self.interaction_injection_mode != "sparse_deep":
            raise ValueError("action-to-memory diagnostics require sparse_deep injection")
        if self.interaction_inference_ablation == "off":
            raise ValueError("action-to-memory diagnostics are unavailable when injection is off")

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_length, prefix_mask, kv_cache, prefix_hidden = self._encode_prefix(observation)
        interaction_memory, patch_diagnostics = self.interaction_patch_diagnostics(
            observation,
            prefix_hidden,
            top_k=top_k,
        )
        patch_diagnostics["interaction_memory"] = interaction_memory
        interaction_memory = self._apply_interaction_memory_ablation(interaction_memory)
        injection_layer_indices = jnp.asarray(self.interaction_injection_layers, dtype=jnp.int32)

        def step(carry, _):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                batch_time,
                interaction_memory,
            )
            suffix_out, adapter_aux = self._forward_action_expert(
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                prefix_length,
                prefix_mask,
                kv_cache,
                interaction_memory,
                return_adapter_aux=True,
            )
            assert adapter_aux is not None
            adapter_aux = jax.tree.map(lambda value: value[injection_layer_indices], adapter_aux)
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            next_carry = (x_t + dt * velocity, time + dt)
            step_aux = {
                "gate": adapter_aux["gate"],
                "head_slot_attention": adapter_aux["head_slot_attention"],
                "injection_ratio": adapter_aux["injection_ratio"],
                "slot_attention": adapter_aux["slot_attention"],
            }
            return next_carry, step_aux

        (_, _), denoising_aux = jax.lax.scan(
            step,
            (noise, jnp.asarray(1.0, dtype=jnp.float32)),
            xs=None,
            length=num_steps,
        )
        # scan: [flow, layer, batch, action, ...] -> [batch, flow, layer, action, ...]
        slot_attention = jnp.transpose(denoising_aux["slot_attention"], (2, 0, 1, 3, 4))
        head_slot_attention = jnp.transpose(denoising_aux["head_slot_attention"], (2, 0, 1, 3, 4, 5))
        gates = jnp.transpose(denoising_aux["gate"], (2, 0, 1, 3))
        injection_ratio = jnp.transpose(denoising_aux["injection_ratio"], (2, 0, 1, 3))
        visual_attention = patch_diagnostics["interaction_visual_attention"]
        effective_visual_attention = jnp.einsum(
            "btlaq,bqvp->btlavp",
            slot_attention,
            visual_attention,
        )
        flow_times = 1.0 + dt * jnp.arange(num_steps, dtype=jnp.float32)
        patch_diagnostics.update(
            {
                "interaction_action_to_memory_attention": slot_attention,
                "interaction_action_to_memory_head_attention": head_slot_attention,
                "interaction_effective_visual_attention": effective_visual_attention,
                "interaction_adapter_gate": gates,
                "interaction_adapter_injection_ratio": injection_ratio,
                "interaction_flow_times": jnp.broadcast_to(flow_times, (batch_size, num_steps)),
                "interaction_injection_layers": jnp.broadcast_to(
                    injection_layer_indices,
                    (batch_size, len(self.interaction_injection_layers)),
                ),
            }
        )
        grid_height, grid_width = self._patch_grid_shape()
        patch_diagnostics["interaction_patch_grid_shape"] = jnp.broadcast_to(
            jnp.asarray([grid_height, grid_width], dtype=jnp.int32),
            (batch_size, 2),
        )
        return patch_diagnostics

    def interaction_diagnostics(
        self,
        observation: _model.Observation,
        *,
        top_k: int = 4,
    ) -> dict[str, at.Array]:
        """Return replan diagnostics without changing the action-sampling path."""
        if not self.use_interaction_memory:
            raise ValueError("interaction diagnostics require use_interaction_memory=True")

        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        _, _, _, prefix_hidden = self._encode_prefix(observation)
        memory, diagnostics = self.interaction_patch_diagnostics(
            observation,
            prefix_hidden,
            top_k=top_k,
        )
        diagnostics["interaction_memory"] = memory
        grid_height, grid_width = self._patch_grid_shape()
        diagnostics["interaction_patch_grid_shape"] = jnp.broadcast_to(
            jnp.asarray([grid_height, grid_width], dtype=jnp.int32),
            (batch_size, 2),
        )
        return diagnostics
