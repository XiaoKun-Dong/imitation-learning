import numpy as np
import pytest
import torch

from openpi.serving import kuavo_attention_policy_server


class _FakePolicy:
    def __init__(self):
        self.observation = None

    def infer(self, observation):
        self.observation = observation
        attention = np.zeros((2, 3, 4), dtype=np.float32)
        attention[:, 0, 0] = 0.6
        attention[:, 2, 3] = 0.4
        return {
            "actions": np.zeros((10, 8), dtype=np.float32),
            "interaction_camera_names": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            "interaction_visual_attention": attention,
            "interaction_camera_mask": np.asarray([True, False, True]),
            "interaction_patch_grid_shape": np.asarray([2, 2]),
        }


def _ros_observation():
    return {
        "observation.images.head_cam_h": torch.zeros((1, 3, 12, 16)),
        "observation.images.wrist_cam_r": torch.ones((1, 3, 12, 16)),
        "observation.state": torch.arange(8, dtype=torch.float32)[None, :],
        "prompt": "Pick and Place",
    }


def test_prepare_kuavo_observation_maps_ros_keys():
    prepared = kuavo_attention_policy_server.prepare_kuavo_observation(_ros_observation())

    assert set(prepared) == {"cam_h", "cam_r", "state", "prompt"}
    assert prepared["state"].shape == (1, 8)


def test_prepare_kuavo_observation_rejects_missing_camera():
    observation = _ros_observation()
    observation.pop("observation.images.wrist_cam_r")

    with pytest.raises(KeyError, match="wrist_cam_r"):
        kuavo_attention_policy_server.prepare_kuavo_observation(observation)


def test_kuavo_attention_policy_saves_panel_and_returns_actions(tmp_path):
    fake_policy = _FakePolicy()
    policy = kuavo_attention_policy_server.KuavoAttentionPolicy(fake_policy, tmp_path, top_k=2)

    reset_result = policy.reset()
    actions = policy.select_action_chunk(_ros_observation())

    assert reset_result == {"status": "ok", "episode_index": 0}
    assert actions.shape == (10, 8)
    assert fake_policy.observation.keys() == {"cam_h", "cam_r", "state", "prompt"}
    assert (tmp_path / "latest.png").is_file()
    assert (tmp_path / "episode_0000" / "kuavo_attention_replan_000_step_0000.png").is_file()
