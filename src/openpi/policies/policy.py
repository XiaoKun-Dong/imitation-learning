from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _deterministic_noise_from_seed_components(
    seed_components: Sequence[int] | np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Generate stateless Gaussian noise from stable integer seed components."""
    components_array = np.asarray(seed_components)
    if components_array.ndim != 1 or components_array.size == 0:
        raise ValueError("flow noise seed must be a non-empty one-dimensional integer sequence")
    if not np.issubdtype(components_array.dtype, np.integer):
        raise ValueError("flow noise seed components must be integers")

    components = [int(component) for component in components_array]
    if any(component < 0 or component > np.iinfo(np.uint32).max for component in components):
        raise ValueError("flow noise seed components must be in [0, 2**32 - 1]")

    rng = np.random.default_rng(np.random.SeedSequence(components))
    return rng.standard_normal(shape, dtype=np.float32)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = dict(sample_kwargs or {})
        self._interaction_diagnostics = bool(self._sample_kwargs.pop("interaction_diagnostics", False))
        interaction_ablation = self._sample_kwargs.pop("interaction_ablation", "normal")
        interaction_layer_mean_gates = tuple(self._sample_kwargs.pop("interaction_layer_mean_gates", ()))
        if interaction_ablation != "normal" or interaction_layer_mean_gates:
            configure_ablation = getattr(model, "configure_interaction_inference_ablation", None)
            if configure_ablation is None:
                raise ValueError("interaction ablations require a DemoVLA model")
            configure_ablation(interaction_ablation, interaction_layer_mean_gates)
            logging.info(
                "Configured DemoVLA interaction ablation: mode=%s layer_mean_gates=%s",
                interaction_ablation,
                interaction_layer_mean_gates,
            )
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            if self._interaction_diagnostics:
                raise ValueError("interaction_diagnostics is only supported by JAX DemoVLA models")
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_memory_source = None
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            memory_source_sampler = getattr(model, "sample_actions_with_memory_source", None)
            self._sample_actions_with_memory_source = (
                nnx_utils.module_jit(memory_source_sampler) if memory_source_sampler is not None else None
            )
            self._interaction_diagnostics_fn = None
            self._interaction_diagnostics_uses_sampling_context = False
            if self._interaction_diagnostics:
                debug_method = getattr(model, "interaction_action_diagnostics", None)
                if debug_method is not None:
                    self._interaction_diagnostics_uses_sampling_context = True
                else:
                    debug_method = getattr(model, "interaction_diagnostics", None)
                if debug_method is None:
                    raise ValueError("interaction_diagnostics requires a model with interaction diagnostic support")
                self._interaction_diagnostics_fn = nnx_utils.module_jit(debug_method)
            self._rng = rng or jax.random.key(0)

    def _sample_actions_and_maybe_diagnostics(
        self,
        rng_or_device: at.KeyArrayLike | str,
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
        *,
        memory_observation: _model.Observation | None = None,
        interaction_memory_override: at.Array | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Use one action sampler regardless of whether diagnostics are enabled."""
        if memory_observation is not None or interaction_memory_override is not None:
            if self._sample_actions_with_memory_source is None:
                raise ValueError("memory-source interventions require a JAX DemoVLA model")
            actions = self._sample_actions_with_memory_source(
                rng_or_device,
                observation,
                memory_observation if memory_observation is not None else observation,
                interaction_memory_override=interaction_memory_override,
                **sample_kwargs,
            )
            # Existing action-read diagnostics assume one shared observation.
            # Returning no diagnostics here avoids silently labeling action-prefix
            # diagnostics as if they came from the counterfactual memory source.
            return actions, None

        actions = self._sample_actions(rng_or_device, observation, **sample_kwargs)
        diagnostics = None
        if self._interaction_diagnostics:
            assert self._interaction_diagnostics_fn is not None
            if getattr(self, "_interaction_diagnostics_uses_sampling_context", False):
                diagnostics = self._interaction_diagnostics_fn(
                    rng_or_device,
                    observation,
                    **sample_kwargs,
                )
            else:
                diagnostics = self._interaction_diagnostics_fn(observation)
        return actions, diagnostics

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Remove protocol-only inputs before observation transforms see them.
        flow_noise_seed = obs.get(_base_policy.FLOW_NOISE_SEED_KEY)
        memory_prompt = obs.get(_base_policy.INTERACTION_MEMORY_PROMPT_KEY)
        interaction_memory_override = obs.get(_base_policy.INTERACTION_MEMORY_OVERRIDE_KEY)
        protocol_keys = {
            _base_policy.FLOW_NOISE_SEED_KEY,
            _base_policy.INTERACTION_MEMORY_PROMPT_KEY,
            _base_policy.INTERACTION_MEMORY_OVERRIDE_KEY,
        }
        raw_inputs = {key: value for key, value in obs.items() if key not in protocol_keys}
        # Make a copy since transformations may modify the inputs in place.
        raw_inputs = jax.tree.map(lambda x: x, raw_inputs)
        if flow_noise_seed is not None:
            if noise is not None:
                raise ValueError("provide either explicit noise or a flow noise seed, not both")
            noise = _deterministic_noise_from_seed_components(
                flow_noise_seed,
                (self._model.action_horizon, self._model.action_dim),
            )

        inputs = self._input_transform(raw_inputs)
        memory_inputs = None
        if memory_prompt is not None:
            if not isinstance(memory_prompt, str):
                raise ValueError("interaction memory prompt override must be a string")
            memory_raw_inputs = dict(raw_inputs)
            memory_raw_inputs["prompt"] = memory_prompt
            memory_inputs = self._input_transform(memory_raw_inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            if memory_inputs is not None:
                memory_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], memory_inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            if memory_inputs is not None or interaction_memory_override is not None:
                raise ValueError("memory-source interventions are only supported by JAX DemoVLA models")
            # Convert inputs to PyTorch tensors and move to correct device
            import torch

            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            if self._is_pytorch_model:
                import torch

                noise = torch.from_numpy(noise).to(self._pytorch_device)
            else:
                noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        memory_observation = _model.Observation.from_dict(memory_inputs) if memory_inputs is not None else None
        memory_override_array = None
        if interaction_memory_override is not None:
            memory_override_array = jnp.asarray(interaction_memory_override)
            if memory_override_array.ndim == 2:
                memory_override_array = memory_override_array[None, ...]
        start_time = time.monotonic()
        # Diagnostics must not select a different action-sampling graph. Always
        # generate actions through the standard sampler, then compute diagnostics
        # independently when requested.
        actions, diagnostics = self._sample_actions_and_maybe_diagnostics(
            sample_rng_or_pytorch_device,
            observation,
            sample_kwargs,
            memory_observation=memory_observation,
            interaction_memory_override=memory_override_array,
        )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if diagnostics is not None:
            outputs.update(diagnostics)
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if diagnostics is not None:
            outputs["interaction_camera_names"] = list(observation.images)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
