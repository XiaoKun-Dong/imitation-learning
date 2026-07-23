import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.rovla import RoVLA


@dataclasses.dataclass(frozen=True)
class RoVLAConfig(pi0_config.Pi0Config):
    """Configuration for the robust object-conditioned pi0.5 adapter."""

    pi05: bool = True
    discrete_state_input: bool = False
    rovla_num_object_tokens: int = 32
    rovla_external_semantic_dim: int = 768
    rovla_num_heads: int = 8
    rovla_residual_scale: float = 0.1
    rovla_gate_bias_init: float = -3.4760987
    rovla_condition_dropout_rate: float = 0.1
    rovla_mask_shift_pixels: int = 8
    rovla_mask_morphology_radius: int = 2
    rovla_bbox_jitter_pixels: float = 8.0
    rovla_point_noise_std: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if self.rovla_num_object_tokens <= 0:
            raise ValueError("rovla_num_object_tokens must be positive")
        if self.rovla_external_semantic_dim <= 0:
            raise ValueError("rovla_external_semantic_dim must be positive")
        if self.rovla_num_heads <= 0:
            raise ValueError("rovla_num_heads must be positive")
        if not 0.0 < self.rovla_residual_scale <= 1.0:
            raise ValueError("rovla_residual_scale must be in (0, 1]")
        if not 0.0 <= self.rovla_condition_dropout_rate <= 1.0:
            raise ValueError("rovla_condition_dropout_rate must be in [0, 1]")
        if self.rovla_mask_shift_pixels < 0 or self.rovla_mask_morphology_radius < 0:
            raise ValueError("RoVLA mask corruption sizes must be non-negative")

    @override
    def create(self, rng: at.KeyArrayLike) -> "RoVLA":
        from openpi.models.rovla import RoVLA

        return RoVLA(self, rngs=nnx.Rngs(rng))
