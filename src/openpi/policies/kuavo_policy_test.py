import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import kuavo_policy


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
