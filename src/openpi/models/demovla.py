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
    params,
    interaction_memory,
    injection_layers,
    layer_index,
    hidden_groups,
):
    """Pure JAX action-token hook compatible with Flax's scanned blocks."""
    action_hidden = hidden_groups[1]
    if action_hidden is None:
        return hidden_groups

    def inject(hidden):
        query = _linear(
            _layer_norm(hidden, params["norm_scale"], params["norm_bias"]),
            params["query_kernel"],
            params["query_bias"],
        )
        key = _linear(interaction_memory, params["key_kernel"], params["key_bias"])
        value = _linear(interaction_memory, params["value_kernel"], params["value_bias"])
        head_dim = query.shape[-1] // num_heads
        query_heads = einops.rearrange(query, "b q (h d) -> b h q d", h=num_heads)
        key_heads = einops.rearrange(key, "b k (h d) -> b h k d", h=num_heads)
        value_heads = einops.rearrange(value, "b k (h d) -> b h k d", h=num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query_heads, key_heads) * (head_dim**-0.5)
        weights = jax.nn.softmax(logits, axis=-1)
        delta = jnp.einsum("bhqk,bhkd->bhqd", weights, value_heads)
        delta = einops.rearrange(delta, "b h q d -> b q (h d)")
        delta = _linear(delta, params["output_kernel"], params["output_bias"])

        matches = layer_index == injection_layers
        gate_index = jnp.argmax(matches)
        gate = jax.nn.sigmoid(params["gates"][gate_index])
        return (hidden + gate * delta).astype(hidden.dtype)

    matches = layer_index == injection_layers
    action_hidden = jax.lax.cond(jnp.any(matches), inject, lambda hidden: hidden, action_hidden)
    return [hidden_groups[0], action_hidden]


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
    interaction_max_camera_views: int = 3
    interaction_injection_mode: Literal["single_shot", "sparse_deep"] = "sparse_deep"
    interaction_injection_layers: tuple[int, ...] = (4, 9, 14)
    interaction_adapter_share_weights: bool = True
    interaction_attention_diversity_weight: float = 0.0
    interaction_attention_diversity_margin: float = 0.5
    interaction_memory_diversity_weight: float = 0.0
    interaction_memory_diversity_margin: float = 0.5

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
        if self.interaction_max_camera_views <= 0:
            raise ValueError("interaction_max_camera_views must be positive")
        if self.interaction_injection_mode not in ("single_shot", "sparse_deep"):
            raise ValueError(f"unsupported interaction_injection_mode: {self.interaction_injection_mode}")
        if not self.interaction_adapter_share_weights:
            raise ValueError("DemoVLA currently requires shared interaction adapter weights")
        if self.interaction_attention_diversity_weight < 0.0:
            raise ValueError("interaction_attention_diversity_weight must be non-negative")
        if self.interaction_memory_diversity_weight < 0.0:
            raise ValueError("interaction_memory_diversity_weight must be non-negative")
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

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation, actions = super().inputs_spec(batch_size=batch_size)
        # DemoVLA deliberately has no externally supplied object condition.
        with at.disable_typechecking():
            observation = observation.replace(
                target_mask=None,
                target_bbox=None,
                target_crop=None,
                target_point=None,
            )
        return observation, actions


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
        self.interaction_attention_diversity_weight = config.interaction_attention_diversity_weight
        self.interaction_attention_diversity_margin = config.interaction_attention_diversity_margin
        self.interaction_memory_diversity_weight = config.interaction_memory_diversity_weight
        self.interaction_memory_diversity_margin = config.interaction_memory_diversity_margin

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
        query = self.demovla_action_query_proj(self.demovla_action_query_norm(action_tokens))
        delta = self._attention(
            query,
            self.demovla_memory_key_proj(interaction_memory),
            self.demovla_memory_value_proj(interaction_memory),
        )
        delta = self.demovla_action_output_proj(delta)
        gate = jax.nn.sigmoid(self.demovla_interaction_gates.value[gate_index]).astype(delta.dtype)
        return action_tokens + gate * delta

    def _make_sparse_deep_adapter(
        self,
        interaction_memory: at.Float[at.Array, "b interaction_s action_d"],
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
            "gates": self.demovla_interaction_gates.value,
        }
        return jax.tree_util.Partial(
            functools.partial(_apply_sparse_deep_adapter, self.interaction_num_heads),
            params,
            interaction_memory,
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
    ) -> at.Float[at.Array, "b suffix_s action_d"]:
        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        if full_attn_mask.shape[-1] != prefix_length + suffix_tokens.shape[1]:
            raise ValueError("prefix KV cache and suffix attention mask lengths do not match")
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        layer_adapter = None
        if self.interaction_injection_mode == "sparse_deep":
            layer_adapter = self._make_sparse_deep_adapter(interaction_memory)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            layer_adapter=layer_adapter,
        )
        assert prefix_out is None
        assert suffix_out is not None
        return suffix_out

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
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, interaction_memory
        )
        suffix_out = self._forward_action_expert(
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
        total_loss = flow_loss + diversity_regularization.astype(flow_loss.dtype)
        metrics = {
            "demovla_flow_loss": jnp.mean(flow_loss),
            "demovla_attention_diversity_loss": attention_diversity_loss,
            "demovla_attention_pair_cosine_mean": attention_pair_cosine_mean,
            "demovla_attention_pair_cosine_max": attention_pair_cosine_max,
            "demovla_memory_diversity_loss": memory_diversity_loss,
            "demovla_memory_pair_cosine_mean": memory_pair_cosine_mean,
            "demovla_memory_pair_cosine_max": memory_pair_cosine_max,
            "demovla_diversity_regularization": diversity_regularization,
        }
        return total_loss, metrics

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

        def step(carry):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                batch_time,
                interaction_memory,
            )
            suffix_out = self._forward_action_expert(
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
        """Sample actions and return the interaction patches for this replan."""
        if not self.use_interaction_memory:
            raise ValueError("interaction diagnostics require use_interaction_memory=True")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_length, prefix_mask, kv_cache, prefix_hidden = self._encode_prefix(observation)
        interaction_memory, diagnostics = self.interaction_patch_diagnostics(
            observation,
            prefix_hidden,
            top_k=top_k,
        )

        def step(carry):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                batch_time,
                interaction_memory,
            )
            suffix_out = self._forward_action_expert(
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
        grid_height, grid_width = self._patch_grid_shape()
        diagnostics["interaction_patch_grid_shape"] = jnp.broadcast_to(
            jnp.asarray([grid_height, grid_width], dtype=jnp.int32),
            (batch_size, 2),
        )
        return actions, diagnostics
