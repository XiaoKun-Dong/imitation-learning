import math

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import rovla_config
from openpi.shared import array_typing as at


class RoVLA(pi0.Pi0):
    """Robust object-conditioned action adapter on top of pi0.5."""

    def __init__(self, config: rovla_config.RoVLAConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        action_width = _gemma.get_config(config.action_expert_variant).width
        semantic_width = _gemma.get_config(config.paligemma_variant).width
        if action_width % config.rovla_num_heads:
            raise ValueError("action expert width must be divisible by rovla_num_heads")
        self.rovla_num_object_tokens = config.rovla_num_object_tokens
        self.rovla_num_heads = config.rovla_num_heads
        self.rovla_residual_scale = config.rovla_residual_scale
        self.rovla_gate_bias = config.rovla_gate_bias_init
        self.rovla_condition_dropout_rate = config.rovla_condition_dropout_rate
        self.rovla_mask_shift_pixels = config.rovla_mask_shift_pixels
        self.rovla_mask_morphology_radius = config.rovla_mask_morphology_radius
        self.rovla_bbox_jitter_pixels = config.rovla_bbox_jitter_pixels
        self.rovla_point_noise_std = config.rovla_point_noise_std
        self.rovla_external_semantic_dim = config.rovla_external_semantic_dim

        self.rovla_backbone_semantic_proj = nnx.Linear(semantic_width, action_width, rngs=rngs)
        self.rovla_external_semantic_proj = nnx.Linear(config.rovla_external_semantic_dim, action_width, rngs=rngs)
        self.rovla_position_proj = nnx.Linear(2, action_width, rngs=rngs)
        # bbox(4), mask moments(5), point(3), confidence(1), availability(3), role(2)
        self.rovla_geometry_proj = nnx.Linear(18, action_width, rngs=rngs)
        self.rovla_query_norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.rovla_query_time_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.rovla_query_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.rovla_key_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.rovla_value_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.rovla_output_proj = nnx.Linear(
            action_width,
            action_width,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.rovla_gate_state_proj = nnx.Linear(config.action_dim, action_width, rngs=rngs)
        self.rovla_gate_time_proj = nnx.Linear(action_width, action_width, rngs=rngs)
        self.rovla_gate_confidence_proj = nnx.Linear(1, action_width, rngs=rngs)
        self.rovla_gate_output_proj = nnx.Linear(
            action_width,
            1,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    @staticmethod
    def _normalized_grid(grid_size: int):
        axes = (jnp.arange(grid_size, dtype=jnp.float32) + 0.5) / grid_size
        yy, xx = jnp.meshgrid(axes, axes, indexing="ij")
        return jnp.stack([xx, yy], axis=-1).reshape(-1, 2)

    def _semantic_grid(self, obs: _model.Observation):
        if obs.object_semantic_tokens is not None:
            semantic = jnp.asarray(obs.object_semantic_tokens, dtype=jnp.float32)
            if semantic.shape[-1] != self.rovla_external_semantic_dim:
                raise ValueError(
                    f"object_semantic_tokens dim must be {self.rovla_external_semantic_dim}, got {semantic.shape[-1]}"
                )
            tokens = self.rovla_external_semantic_proj(semantic)
        else:
            semantic, _ = self.PaliGemma.img(obs.images["base_0_rgb"], train=False)
            tokens = self.rovla_backbone_semantic_proj(semantic.astype(jnp.float32))
        grid_size = math.isqrt(tokens.shape[1])
        if grid_size * grid_size != tokens.shape[1]:
            raise ValueError("semantic patch tokens must form a square grid and exclude any CLS token")
        positions = self._normalized_grid(grid_size)[None]
        return tokens, jnp.broadcast_to(positions, (tokens.shape[0], tokens.shape[1], 2)), grid_size

    def _condition_features(self, obs: _model.Observation, grid_size: int):
        batch_size = obs.state.shape[0]
        mask_grid = jnp.zeros((batch_size, grid_size, grid_size), dtype=jnp.float32)
        if obs.target_mask is not None:
            mask_grid = jax.image.resize(
                jnp.asarray(obs.target_mask[..., None], dtype=jnp.float32),
                (batch_size, grid_size, grid_size, 1),
                method=jax.image.ResizeMethod.LINEAR,
            )[..., 0]
        has_mask = jnp.any(mask_grid > 0.05, axis=(1, 2))

        bbox = jnp.zeros((batch_size, 4), dtype=jnp.float32)
        if obs.target_bbox is not None:
            bbox = jnp.clip(jnp.asarray(obs.target_bbox, dtype=jnp.float32) / 224.0, 0.0, 1.0)
        has_bbox = (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])

        point = jnp.zeros((batch_size, 3), dtype=jnp.float32)
        if obs.target_point is not None:
            point = jnp.asarray(obs.target_point, dtype=jnp.float32)
        has_point = jnp.any(jnp.abs(point) > 0, axis=1)

        positions = self._normalized_grid(grid_size)
        x, y = positions[:, 0][None], positions[:, 1][None]
        in_bbox = (x >= bbox[:, 0, None]) & (x <= bbox[:, 2, None]) & (y >= bbox[:, 1, None]) & (y <= bbox[:, 3, None])
        flat_mask = mask_grid.reshape(batch_size, -1)
        selector = jnp.maximum(flat_mask, in_bbox.astype(jnp.float32))
        denom = jnp.sum(flat_mask, axis=1, keepdims=True) + 1e-6
        cx = jnp.sum(flat_mask * x, axis=1, keepdims=True) / denom
        cy = jnp.sum(flat_mask * y, axis=1, keepdims=True) / denom
        moments = jnp.concatenate(
            [
                jnp.mean(flat_mask, axis=1, keepdims=True),
                cx,
                cy,
                jnp.sum(flat_mask * jnp.square(x - cx), axis=1, keepdims=True) / denom,
                jnp.sum(flat_mask * jnp.square(y - cy), axis=1, keepdims=True) / denom,
            ],
            axis=1,
        )
        has_condition = has_mask | has_bbox | has_point
        confidence = has_condition.astype(jnp.float32)[:, None]
        if obs.object_condition_confidence is not None:
            confidence = jnp.clip(
                jnp.asarray(obs.object_condition_confidence, dtype=jnp.float32).reshape(batch_size, 1), 0, 1
            )
        availability = jnp.stack([has_mask, has_bbox, has_point], axis=1).astype(jnp.float32)
        role = jnp.broadcast_to(jnp.asarray([[1.0, 0.0]], dtype=jnp.float32), (batch_size, 2))
        geometry = jnp.concatenate([bbox, moments, point, confidence, availability, role], axis=1)
        return selector, geometry, confidence, has_condition

    def _embed_object_condition(self, obs: _model.Observation):
        if all(
            value is None
            for value in (obs.target_mask, obs.target_bbox, obs.target_point, obs.object_condition_confidence)
        ):
            return None
        semantic_tokens, positions, grid_size = self._semantic_grid(obs)
        selector, geometry, confidence, has_condition = self._condition_features(obs, grid_size)
        selector += jnp.linspace(0.0, 1e-5, selector.shape[1], dtype=selector.dtype)[None]
        k = min(self.rovla_num_object_tokens, selector.shape[1])
        scores, indices = jax.lax.top_k(selector, k)
        token_indices = jnp.broadcast_to(indices[..., None], (*indices.shape, semantic_tokens.shape[-1]))
        position_indices = jnp.broadcast_to(indices[..., None], (*indices.shape, 2))
        selected = jnp.take_along_axis(semantic_tokens, token_indices, axis=1)
        selected_positions = jnp.take_along_axis(positions, position_indices, axis=1)
        selected = nnx.swish(selected) + self.rovla_position_proj(selected_positions)
        geometry_token = nnx.swish(self.rovla_geometry_proj(geometry))[:, None, :]
        object_tokens = jnp.concatenate([selected, geometry_token], axis=1)
        token_mask = jnp.concatenate([scores > 0.05, jnp.ones((scores.shape[0], 1), dtype=jnp.bool_)], axis=1)
        return object_tokens, token_mask, has_condition, confidence

    def _cross_attend_object_condition(
        self, action_tokens, object_condition, *, state=None, time_features=None, return_metrics=False
    ):
        object_tokens, object_mask, has_condition, confidence = object_condition
        if state is None or time_features is None:
            raise ValueError("RoVLA requires state and flow-time features")
        query_features = self.rovla_query_norm(action_tokens)
        query_features = query_features + nnx.swish(self.rovla_query_time_proj(time_features))[:, None, :]
        query = self.rovla_query_proj(query_features)
        key = self.rovla_key_proj(object_tokens)
        value = self.rovla_value_proj(object_tokens)
        head_dim = query.shape[-1] // self.rovla_num_heads
        query = einops.rearrange(query, "b s (h d) -> b h s d", h=self.rovla_num_heads)
        key = einops.rearrange(key, "b s (h d) -> b h s d", h=self.rovla_num_heads)
        value = einops.rearrange(value, "b s (h d) -> b h s d", h=self.rovla_num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) * head_dim**-0.5
        logits = jnp.where(object_mask[:, None, None, :], logits, -jnp.inf)
        attended = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, axis=-1), value)
        delta = self.rovla_output_proj(einops.rearrange(attended, "b h s d -> b s (h d)"))

        gate_features = query_features
        gate_features = gate_features + nnx.swish(self.rovla_gate_state_proj(state))[:, None, :]
        gate_features = gate_features + nnx.swish(self.rovla_gate_time_proj(time_features))[:, None, :]
        gate_features = gate_features + nnx.swish(self.rovla_gate_confidence_proj(confidence))[:, None, :]
        gate = jax.nn.sigmoid(
            self.rovla_gate_output_proj(nnx.swish(gate_features)).astype(jnp.float32) + self.rovla_gate_bias
        )
        effective_scale = (
            self.rovla_residual_scale
            * gate.astype(delta.dtype)
            * confidence[:, None, :].astype(delta.dtype)
            * has_condition[:, None, None].astype(delta.dtype)
        )
        residual = effective_scale * delta
        output = action_tokens + residual
        if not return_metrics:
            return output
        return output, {
            "rovla_action_token_rms": jnp.sqrt(jnp.mean(jnp.square(action_tokens.astype(jnp.float32)))),
            "rovla_delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta.astype(jnp.float32)))),
            "rovla_residual_rms": jnp.sqrt(jnp.mean(jnp.square(residual.astype(jnp.float32)))),
            "rovla_effective_scale": jnp.mean(effective_scale.astype(jnp.float32)),
            "rovla_gate": jnp.mean(gate),
            "rovla_gate_min": jnp.min(gate),
            "rovla_gate_max": jnp.max(gate),
            "rovla_condition_confidence": jnp.mean(confidence),
        }

    def _drop_object_condition(self, rng: at.KeyArrayLike, obs: _model.Observation, *, train: bool):
        if not train:
            return obs
        dropout_rng, mask_rng, morphology_rng, bbox_rng, point_rng = jax.random.split(rng, 5)
        batch_size = obs.state.shape[0]
        keep = jax.random.bernoulli(dropout_rng, 1.0 - self.rovla_condition_dropout_rate, (batch_size,))

        def keep_value(value):
            if value is None:
                return None
            shape = (batch_size,) + (1,) * (value.ndim - 1)
            return value * keep.reshape(shape).astype(value.dtype)

        mask = keep_value(obs.target_mask)
        if mask is not None and self.rovla_mask_shift_pixels > 0:
            shifts = jax.random.randint(
                mask_rng,
                (batch_size, 2),
                -self.rovla_mask_shift_pixels,
                self.rovla_mask_shift_pixels + 1,
            )

            def shift_mask(single_mask, shift):
                height, width = single_mask.shape
                source_y = jnp.arange(height) - shift[0]
                source_x = jnp.arange(width) - shift[1]
                valid_y = (source_y >= 0) & (source_y < height)
                valid_x = (source_x >= 0) & (source_x < width)
                shifted = single_mask[jnp.clip(source_y, 0, height - 1)[:, None], jnp.clip(source_x, 0, width - 1)]
                return shifted & valid_y[:, None] & valid_x[None, :]

            mask = jax.vmap(shift_mask)(mask, shifts)
        if mask is not None and self.rovla_mask_morphology_radius > 0:
            radius = self.rovla_mask_morphology_radius
            window = (1, 2 * radius + 1, 2 * radius + 1)
            float_mask = mask.astype(jnp.float32)
            dilated = jax.lax.reduce_window(float_mask, 0.0, jax.lax.max, window, (1, 1, 1), "SAME") > 0.5
            eroded = jax.lax.reduce_window(float_mask, 1.0, jax.lax.min, window, (1, 1, 1), "SAME") > 0.5
            operation = jax.random.randint(morphology_rng, (batch_size,), 0, 3)
            mask = jnp.where(
                (operation == 1)[:, None, None],
                dilated,
                jnp.where((operation == 2)[:, None, None], eroded, mask),
            )

        bbox = keep_value(obs.target_bbox)
        if bbox is not None and self.rovla_bbox_jitter_pixels > 0:
            bbox += jax.random.normal(bbox_rng, bbox.shape) * self.rovla_bbox_jitter_pixels * keep[:, None]
            bbox = jnp.clip(bbox, 0.0, 224.0)
        point = keep_value(obs.target_point)
        if point is not None and self.rovla_point_noise_std > 0:
            point += jax.random.normal(point_rng, point.shape) * self.rovla_point_noise_std * keep[:, None]
        with at.disable_typechecking():
            return obs.replace(
                target_mask=mask,
                target_bbox=bbox,
                target_crop=keep_value(obs.target_crop),
                target_point=point,
                object_condition_confidence=keep_value(obs.object_condition_confidence),
                object_semantic_tokens=obs.object_semantic_tokens,
            )
