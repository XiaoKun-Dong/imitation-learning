import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_deterministic_flow_noise_depends_only_on_seed_components():
    shape = (10, 32)
    seed = np.asarray([7, 4, 2, 9], dtype=np.uint32)

    first = _policy._deterministic_noise_from_seed_components(seed, shape)  # noqa: SLF001
    repeated = _policy._deterministic_noise_from_seed_components(seed.copy(), shape)  # noqa: SLF001
    next_replan = _policy._deterministic_noise_from_seed_components([7, 4, 2, 10], shape)  # noqa: SLF001

    assert first.shape == shape
    assert first.dtype == np.float32
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, next_replan)


@pytest.mark.parametrize("seed", [[], [[1, 2]], [1.5, 2.5], [-1, 0], [2**32, 0]])
def test_deterministic_flow_noise_rejects_invalid_seed_components(seed):
    with pytest.raises(ValueError, match="flow noise seed"):
        _policy._deterministic_noise_from_seed_components(seed, (2, 3))  # noqa: SLF001


def test_diagnostics_do_not_change_the_action_sampling_path():
    policy = object.__new__(_policy.Policy)
    noise = np.arange(6, dtype=np.float32).reshape(2, 3)
    sample_calls = []

    def sample_actions(rng, observation, **kwargs):
        sample_calls.append((rng, observation))
        return kwargs["noise"] * 2.0

    policy._sample_actions = sample_actions  # noqa: SLF001
    policy._interaction_diagnostics = False  # noqa: SLF001
    policy._interaction_diagnostics_fn = None  # noqa: SLF001
    plain_actions, plain_diagnostics = policy._sample_actions_and_maybe_diagnostics(  # noqa: SLF001
        "device",
        "observation",
        {"noise": noise},
    )

    policy._interaction_diagnostics = True  # noqa: SLF001
    policy._interaction_diagnostics_fn = lambda observation: {"marker": observation}  # noqa: SLF001
    diagnostic_actions, diagnostics = policy._sample_actions_and_maybe_diagnostics(  # noqa: SLF001
        "device",
        "observation",
        {"noise": noise},
    )

    assert np.array_equal(plain_actions, diagnostic_actions)
    assert plain_diagnostics is None
    assert diagnostics == {"marker": "observation"}
    assert sample_calls == [("device", "observation"), ("device", "observation")]


def test_action_diagnostics_receive_the_sampling_rng_and_noise():
    policy = object.__new__(_policy.Policy)
    noise = np.arange(6, dtype=np.float32).reshape(2, 3)
    policy._sample_actions = lambda rng, observation, **kwargs: kwargs["noise"] * 2.0  # noqa: SLF001
    policy._interaction_diagnostics = True  # noqa: SLF001
    policy._interaction_diagnostics_uses_sampling_context = True  # noqa: SLF001
    policy._interaction_diagnostics_fn = (  # noqa: SLF001
        lambda rng, observation, **kwargs: {"rng": rng, "observation": observation, "noise": kwargs["noise"]}
    )

    actions, diagnostics = policy._sample_actions_and_maybe_diagnostics(  # noqa: SLF001
        "rng",
        "observation",
        {"noise": noise},
    )

    assert np.array_equal(actions, noise * 2.0)
    assert diagnostics["rng"] == "rng"
    assert diagnostics["observation"] == "observation"
    assert np.array_equal(diagnostics["noise"], noise)


def test_memory_source_intervention_bypasses_shared_observation_sampler_and_diagnostics():
    policy = object.__new__(_policy.Policy)
    noise = np.arange(6, dtype=np.float32).reshape(2, 3)
    calls = []
    policy._sample_actions = lambda *args, **kwargs: pytest.fail("shared sampler should not run")  # noqa: SLF001
    policy._sample_actions_with_memory_source = (  # noqa: SLF001
        lambda rng, action_observation, memory_observation, **kwargs: calls.append(
            (rng, action_observation, memory_observation, kwargs["interaction_memory_override"])
        )
        or kwargs["noise"] * 3.0
    )
    policy._interaction_diagnostics = True  # noqa: SLF001
    policy._interaction_diagnostics_fn = lambda *_args, **_kwargs: pytest.fail("diagnostics should not run")  # noqa: SLF001
    override = np.ones((1, 4, 8), dtype=np.float32)

    actions, diagnostics = policy._sample_actions_and_maybe_diagnostics(  # noqa: SLF001
        "rng",
        "action_observation",
        {"noise": noise},
        memory_observation="memory_observation",
        interaction_memory_override=override,
    )

    assert np.array_equal(actions, noise * 3.0)
    assert diagnostics is None
    assert len(calls) == 1
    assert calls[0][:3] == ("rng", "action_observation", "memory_observation")
    assert np.array_equal(calls[0][3], override)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
