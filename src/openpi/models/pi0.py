import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

OBJECT_CONDITION_GRID_SIZE = 16
OBJECT_CONDITION_PATCH_DIM = 6  # RGB crop, mask, and normalized x/y coordinates.
OBJECT_CONDITION_GEOMETRY_DIM = 12  # bbox, mask moments, and target point.


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.object_condition_dropout_rate = config.object_condition_dropout_rate
        self.object_condition_num_heads = config.object_condition_num_heads
        self.object_condition_residual_scale = config.object_condition_residual_scale
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if action_expert_config.width % self.object_condition_num_heads != 0:
            raise ValueError("action expert width must be divisible by object_condition_num_heads")
        self.object_condition_patch_proj = nnx.Linear(
            OBJECT_CONDITION_PATCH_DIM, action_expert_config.width, rngs=rngs
        )
        self.object_condition_geometry_proj = nnx.Linear(
            OBJECT_CONDITION_GEOMETRY_DIM, action_expert_config.width, rngs=rngs
        )
        self.object_condition_query_proj = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.object_condition_key_proj = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.object_condition_value_proj = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        # Zero initialization makes a legacy policy's initial behavior exactly unchanged.
        self.object_condition_output_proj = nnx.Linear(
            action_expert_config.width,
            action_expert_config.width,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _resize_object_map(
        self, x: at.Float[at.Array, "b h w c"], *, method: jax.image.ResizeMethod
    ) -> at.Float[at.Array, "b 16 16 c"]:
        return jax.image.resize(
            x,
            (x.shape[0], OBJECT_CONDITION_GRID_SIZE, OBJECT_CONDITION_GRID_SIZE, x.shape[-1]),
            method=method,
        )

    def _embed_object_condition(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b object_s emb"],
        at.Bool[at.Array, "b object_s"],
        at.Bool[at.Array, " b"],
    ] | None:
        """Encode spatial object patches and geometry as cross-attention tokens."""
        if (
            obs.target_mask is None
            and obs.target_bbox is None
            and obs.target_crop is None
            and obs.target_point is None
        ):
            return None

        batch_size = obs.state.shape[0]
        crop_grid = jnp.zeros(
            (batch_size, OBJECT_CONDITION_GRID_SIZE, OBJECT_CONDITION_GRID_SIZE, 3),
            dtype=jnp.float32,
        )
        mask_grid = jnp.zeros(
            (batch_size, OBJECT_CONDITION_GRID_SIZE, OBJECT_CONDITION_GRID_SIZE, 1),
            dtype=jnp.float32,
        )
        bbox_features = jnp.zeros((batch_size, 4), dtype=jnp.float32)
        point_features = jnp.zeros((batch_size, 3), dtype=jnp.float32)

        if obs.target_crop is not None:
            crop = jnp.asarray(obs.target_crop, dtype=jnp.float32)
            crop_grid = self._resize_object_map(crop, method=jax.image.ResizeMethod.LINEAR)

        if obs.target_bbox is not None:
            bbox_scale = jnp.asarray(
                [_model.IMAGE_RESOLUTION[1], _model.IMAGE_RESOLUTION[0]] * 2, dtype=jnp.float32
            )
            bbox_features = jnp.asarray(obs.target_bbox, dtype=jnp.float32) / bbox_scale

        if obs.target_point is not None:
            point_features = jnp.asarray(obs.target_point, dtype=jnp.float32)

        if obs.target_mask is not None:
            mask = jnp.asarray(obs.target_mask[..., None], dtype=jnp.float32)
            mask_grid = self._resize_object_map(mask, method=jax.image.ResizeMethod.NEAREST)

        # Spatial moments give the action expert an explicit coarse target center,
        # extent, and visible area even when target_crop is missing.
        flat_mask = mask_grid[..., 0].reshape(batch_size, -1)
        area = jnp.mean(flat_mask, axis=-1, keepdims=True)
        ys = jnp.linspace(0.0, 1.0, OBJECT_CONDITION_GRID_SIZE, dtype=jnp.float32)
        xs = jnp.linspace(0.0, 1.0, OBJECT_CONDITION_GRID_SIZE, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        flat_x = xx.reshape(1, -1)
        flat_y = yy.reshape(1, -1)
        denom = jnp.sum(flat_mask, axis=-1, keepdims=True) + 1e-6
        cx = jnp.sum(flat_mask * flat_x, axis=-1, keepdims=True) / denom
        cy = jnp.sum(flat_mask * flat_y, axis=-1, keepdims=True) / denom
        var_x = jnp.sum(flat_mask * jnp.square(flat_x - cx), axis=-1, keepdims=True) / denom
        var_y = jnp.sum(flat_mask * jnp.square(flat_y - cy), axis=-1, keepdims=True) / denom
        mask_features = jnp.concatenate([area, cx, cy, var_x, var_y], axis=-1)

        coordinates = jnp.stack([xx, yy], axis=-1)
        coordinates = jnp.broadcast_to(coordinates, (*crop_grid.shape[:3], 2))
        patch_features = jnp.concatenate([crop_grid, mask_grid, coordinates], axis=-1)
        patch_features = patch_features.reshape(batch_size, -1, OBJECT_CONDITION_PATCH_DIM)
        patch_tokens = nnx.swish(self.object_condition_patch_proj(patch_features))

        geometry_features = jnp.concatenate([bbox_features, mask_features, point_features], axis=-1)
        geometry_token = nnx.swish(self.object_condition_geometry_proj(geometry_features))[:, None, :]
        object_tokens = jnp.concatenate([patch_tokens, geometry_token], axis=1)

        if obs.target_mask is not None:
            patch_mask = mask_grid[..., 0] > 0.5
        elif obs.target_crop is not None:
            patch_mask = jnp.any(jnp.abs(crop_grid) > 0, axis=-1)
        else:
            patch_mask = jnp.zeros(crop_grid.shape[:3], dtype=jnp.bool_)
        patch_mask = patch_mask.reshape(batch_size, -1)
        object_token_mask = jnp.concatenate(
            [patch_mask, jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1
        )

        # Object dropout and missing segmentations must be true no-ops even though patch
        # coordinates themselves are nonzero.
        has_condition = (
            jnp.any(jnp.abs(crop_grid) > 0, axis=(1, 2, 3))
            | jnp.any(mask_grid > 0, axis=(1, 2, 3))
            | jnp.any(jnp.abs(bbox_features) > 0, axis=1)
            | jnp.any(jnp.abs(point_features) > 0, axis=1)
        )
        return object_tokens, object_token_mask, has_condition

    def _cross_attend_object_condition(
        self,
        action_tokens: at.Float[at.Array, "b action_s emb"],
        object_condition: tuple[
            at.Float[at.Array, "b object_s emb"],
            at.Bool[at.Array, "b object_s"],
            at.Bool[at.Array, " b"],
        ],
    ) -> at.Float[at.Array, "b action_s emb"]:
        """Let each action token independently retrieve target-object information."""
        object_tokens, object_token_mask, has_condition = object_condition
        query = self.object_condition_query_proj(action_tokens)
        key = self.object_condition_key_proj(object_tokens)
        value = self.object_condition_value_proj(object_tokens)

        num_heads = self.object_condition_num_heads
        head_dim = query.shape[-1] // num_heads
        query = einops.rearrange(query, "b s (h d) -> b h s d", h=num_heads)
        key = einops.rearrange(key, "b s (h d) -> b h s d", h=num_heads)
        value = einops.rearrange(value, "b s (h d) -> b h s d", h=num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) * (head_dim**-0.5)
        logits = jnp.where(object_token_mask[:, None, None, :], logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        attended = einops.rearrange(attended, "b h s d -> b s (h d)")
        delta = self.object_condition_output_proj(attended)
        delta = delta * has_condition[:, None, None].astype(delta.dtype)
        return action_tokens + self.object_condition_residual_scale * delta

    def _drop_object_condition(
        self, rng: at.KeyArrayLike, obs: _model.Observation, *, train: bool
    ) -> _model.Observation:
        """Drop object inputs per example during training for segmentation robustness."""
        rate = self.object_condition_dropout_rate
        if not train or rate == 0.0:
            return obs
        if (
            obs.target_mask is None
            and obs.target_bbox is None
            and obs.target_crop is None
            and obs.target_point is None
        ):
            return obs

        batch_size = obs.state.shape[0]
        keep = jax.random.bernoulli(rng, 1.0 - rate, (batch_size,))

        def apply_keep(value):
            if value is None:
                return None
            keep_shape = (batch_size,) + (1,) * (value.ndim - 1)
            keep_mask = keep.reshape(keep_shape)
            if value.dtype == jnp.bool_:
                return jnp.logical_and(value, keep_mask)
            return value * keep_mask.astype(value.dtype)

        with at.disable_typechecking():
            return obs.replace(
                target_mask=apply_keep(obs.target_mask),
                target_bbox=apply_keep(obs.target_bbox),
                target_crop=apply_keep(obs.target_crop),
                target_point=apply_keep(obs.target_point),
            )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        object_cond = self._embed_object_condition(obs)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        # Each action token retrieves the spatial/3D target information it needs.
        if object_cond is not None:
            action_expert_tokens = self._cross_attend_object_condition(action_expert_tokens, object_cond)

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, object_dropout_rng, noise_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        observation = self._drop_object_condition(object_dropout_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
