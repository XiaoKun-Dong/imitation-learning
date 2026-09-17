import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import kuavo_policy


def test_kuavo_right_arm_only_slices_bimanual_samples():
    result = kuavo_policy.KuavoRightArmOnly()(
        {
            "observation.state": np.arange(16, dtype=np.float32),
            "action": np.arange(3 * 16, dtype=np.float32).reshape(3, 16),
        }
    )

    np.testing.assert_array_equal(result["observation.state"], np.arange(8, 16, dtype=np.float32))
    np.testing.assert_array_equal(result["action"], np.arange(3 * 16, dtype=np.float32).reshape(3, 16)[:, 8:])


def test_kuavo_right_arm_only_keeps_right_only_samples():
    state = np.arange(8, dtype=np.float32)
    actions = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    result = kuavo_policy.KuavoRightArmOnly()({"observation.state": state, "action": actions})

    np.testing.assert_array_equal(result["observation.state"], state)
    np.testing.assert_array_equal(result["action"], actions)


def test_kuavo_right_arm_only_stacks_arrow_object_batches():
    state = np.empty(2, dtype=object)
    state[:] = [np.arange(16, dtype=np.float32), np.arange(16, 32, dtype=np.float32)]
    actions = np.empty((2, 3), dtype=object)
    for batch_index in range(2):
        for step in range(3):
            actions[batch_index, step] = np.arange(16, dtype=np.float32) + batch_index + step

    result = kuavo_policy.KuavoRightArmOnly()({"observation.state": state, "action": actions})

    assert result["observation.state"].shape == (2, 8)
    assert result["action"].shape == (2, 3, 8)


def test_kuavo_right_inputs_map_two_cameras_and_actions():
    transform = kuavo_policy.KuavoRightInputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "cam_h": np.zeros((3, 12, 16), dtype=np.float32),
            "cam_r": np.full((12, 16, 3), 127, dtype=np.uint8),
            "state": np.arange(8, dtype=np.float32),
            "actions": np.zeros((10, 8), dtype=np.float32),
            "prompt": "Pick and Place",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["right_wrist_0_rgb"].shape == (12, 16, 3)
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert result["state"].shape == (8,)
    assert result["actions"].shape == (10, 8)
    assert result["prompt"] == "Pick and Place"


def test_kuavo_right_inputs_accept_ros_batched_tensors():
    transform = kuavo_policy.KuavoRightInputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "cam_h": np.zeros((1, 3, 12, 16), dtype=np.float32),
            "cam_r": np.ones((1, 3, 12, 16), dtype=np.float32),
            "state": np.arange(8, dtype=np.float32)[None, :],
            "prompt": "Pick and Place",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (12, 16, 3)
    assert result["state"].shape == (8,)


def test_kuavo_right_inputs_reject_wrong_state_dimension():
    transform = kuavo_policy.KuavoRightInputs(model_type=_model.ModelType.PI05)
    with pytest.raises(ValueError, match="state must be 8-D"):
        transform(
            {
                "cam_h": np.zeros((12, 16, 3), dtype=np.uint8),
                "cam_r": np.zeros((12, 16, 3), dtype=np.uint8),
                "state": np.zeros(16, dtype=np.float32),
            }
        )


def test_kuavo_right_outputs_keep_eight_action_dimensions_and_diagnostics():
    result = kuavo_policy.KuavoRightOutputs()(
        {
            "actions": np.zeros((10, 32), dtype=np.float32),
            "interaction_gate": np.ones((10, 1), dtype=np.float32),
        }
    )

    assert result["actions"].shape == (10, 8)
    assert result["interaction_gate"].shape == (10, 1)
