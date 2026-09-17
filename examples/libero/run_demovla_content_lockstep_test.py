from __future__ import annotations

import pathlib

import numpy as np
import pytest
import run_demovla_content_lockstep


def test_load_memory_accepts_current_memory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "memory.npz"
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.savez(path, current_memory=expected)

    actual, key = run_demovla_content_lockstep._load_memory(path)  # noqa: SLF001

    assert key == "current_memory"
    np.testing.assert_array_equal(actual, expected)


def test_load_memory_rejects_empty_memory(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "memory.npz"
    np.savez(path, interaction_memory=np.empty((0,), dtype=np.float32))

    with pytest.raises(ValueError, match="empty"):
        run_demovla_content_lockstep._load_memory(path)  # noqa: SLF001


def test_action_distance_uses_only_executed_prefix() -> None:
    reference = np.zeros((10, 7), dtype=np.float32)
    contender = np.zeros((10, 7), dtype=np.float32)
    contender[:5, 0] = 1.0
    contender[5:, :] = 100.0

    distance = run_demovla_content_lockstep._action_distance(reference, contender, 5)  # noqa: SLF001

    assert distance["executed_l2"] == pytest.approx(np.sqrt(5.0))
    assert distance["translation_l2"] == pytest.approx(np.sqrt(5.0))
    assert distance["rotation_l2"] == 0.0
    assert distance["gripper_l2"] == 0.0


def test_conditions_include_registered_primary_controls() -> None:
    assert run_demovla_content_lockstep.CONDITIONS[:4] == (
        "correct",
        "shuffled",
        "zero",
        "injection_off",
    )
