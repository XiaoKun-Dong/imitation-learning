import abc
from typing import Dict


# Reserved observation key used to request stateless, deterministic flow noise.
# Policy implementations must remove this field before applying observation transforms.
FLOW_NOISE_SEED_KEY = "_openpi_flow_noise_seed"


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
