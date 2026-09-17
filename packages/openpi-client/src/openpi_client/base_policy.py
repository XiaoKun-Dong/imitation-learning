import abc
from typing import Dict


# Reserved observation key used to request stateless, deterministic flow noise.
# Policy implementations must remove this field before applying observation transforms.
FLOW_NOISE_SEED_KEY = "_openpi_flow_noise_seed"

# Reserved inputs for DemoVLA causal interventions. Policies must remove these
# fields before applying the normal observation transforms. The prompt override
# changes only the interaction-memory extractor input; the action VLM prefix
# continues to receive the original ``prompt``. The tensor override bypasses
# extraction and injects an explicitly supplied interaction memory.
INTERACTION_MEMORY_PROMPT_KEY = "_openpi_interaction_memory_prompt"
INTERACTION_MEMORY_OVERRIDE_KEY = "_openpi_interaction_memory_override"


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
