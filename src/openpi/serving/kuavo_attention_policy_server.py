"""Kuavo-compatible ZMQ server with server-side DemoVLA attention rendering."""

from __future__ import annotations

from collections.abc import Callable
import io
import logging
import pathlib
import shutil
from typing import Any

import numpy as np
import torch
import zmq

from openpi.models import demovla_visualization

_KUAVO_INPUT_KEYS = {
    "cam_h": "observation.images.head_cam_h",
    "cam_r": "observation.images.wrist_cam_r",
    "state": "observation.state",
}
_RAW_CAMERA_KEYS = {
    "base_0_rgb": ("observation.images.head_cam_h", "cam_h"),
    "left_wrist_0_rgb": ("observation.images.wrist_cam_l", "cam_l"),
    "right_wrist_0_rgb": ("observation.images.wrist_cam_r", "cam_r"),
}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_hwc_image(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected a singleton-batched CHW or HWC RGB image, got {image.shape}")
    return image


def prepare_kuavo_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert the Kuavo ROS deployment keys to the policy inference keys."""
    missing = [source for source in _KUAVO_INPUT_KEYS.values() if source not in observation]
    if missing:
        raise KeyError(f"Kuavo observation is missing required keys: {missing}")
    prepared = {target: observation[source] for target, source in _KUAVO_INPUT_KEYS.items()}
    if "prompt" in observation:
        prepared["prompt"] = observation["prompt"]
    return prepared


def extract_camera_images(observation: dict[str, Any], camera_names: list[str]) -> dict[str, np.ndarray]:
    images = {}
    for camera_name in camera_names:
        for raw_key in _RAW_CAMERA_KEYS.get(camera_name, ()):
            if raw_key in observation:
                images[camera_name] = _to_hwc_image(observation[raw_key])
                break
    return images


class KuavoAttentionPolicy:
    """Adapt an OpenPI policy to Kuavo action-chunk calls and save attention panels."""

    def __init__(
        self,
        policy,
        output_dir: pathlib.Path | str,
        *,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._policy = policy
        self._output_dir = pathlib.Path(output_dir)
        self._top_k = top_k
        self._episode_index = -1
        self._replan_index = 0

    def reset(self) -> dict[str, Any]:
        self._episode_index += 1
        self._replan_index = 0
        return {"status": "ok", "episode_index": self._episode_index}

    def _save_attention(self, observation: dict[str, Any], result: dict[str, Any]) -> pathlib.Path:
        required = {
            "interaction_camera_names",
            "interaction_visual_attention",
            "interaction_camera_mask",
            "interaction_patch_grid_shape",
        }
        missing = required - result.keys()
        if missing:
            raise RuntimeError(
                "DemoVLA attention diagnostics are incomplete; missing "
                f"{sorted(missing)}. Load a JAX DemoVLA policy with interaction_diagnostics enabled."
            )

        if self._episode_index < 0:
            self.reset()
        camera_names = list(result["interaction_camera_names"])
        images = extract_camera_images(observation, camera_names)
        episode_dir = self._output_dir / f"episode_{self._episode_index:04d}"
        output_path = demovla_visualization.save_interaction_attention_replan(
            images=images,
            camera_names=camera_names,
            visual_attention=result["interaction_visual_attention"],
            camera_mask=result["interaction_camera_mask"],
            patch_grid_shape=result["interaction_patch_grid_shape"],
            output_dir=episode_dir,
            stem="kuavo_attention",
            replan_index=self._replan_index,
            env_step=self._replan_index,
            top_k=self._top_k,
            prompt=str(observation.get("prompt", "")) or None,
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, self._output_dir / "latest.png")
        logging.info("Saved Kuavo DemoVLA attention: %s", output_path)
        self._replan_index += 1
        return output_path

    def select_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        result = self._policy.infer(prepare_kuavo_observation(observation))
        self._save_attention(observation, result)
        return np.asarray(result["actions"])

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        actions = self.select_action_chunk(observation)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"policy returned an invalid action chunk shape: {actions.shape}")
        return actions[0]


class KuavoZmqPolicyServer:
    """Serve the endpoint protocol used by ``kuavo_deploy.PolicyClient``."""

    def __init__(
        self,
        policy: KuavoAttentionPolicy,
        *,
        host: str = "0.0.0.0",
        port: int = 5555,
        api_token: str | None = None,
    ) -> None:
        self._policy = policy
        self._api_token = api_token
        self._running = True
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, tuple[Callable, bool]] = {
            "ping": (lambda: {"status": "ok", "message": "Server is running"}, False),
            "reset": (policy.reset, False),
            "select_action": (policy.select_action, True),
            "select_action_chunk": (policy.select_action_chunk, True),
            "kill": (self._kill, False),
        }
        logging.info("Kuavo DemoVLA ZMQ server listening on tcp://%s:%d", host, port)

    @staticmethod
    def _serialize(data: Any) -> bytes:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(data: bytes) -> dict[str, Any]:
        return torch.load(io.BytesIO(data), weights_only=False)

    def _kill(self) -> dict[str, str]:
        self._running = False
        return {"status": "ok"}

    def serve_forever(self) -> None:
        try:
            while self._running:
                request = self._deserialize(self._socket.recv())
                try:
                    if self._api_token is not None and request.get("api_token") != self._api_token:
                        raise PermissionError("Unauthorized: invalid API token")
                    endpoint = request.get("endpoint", "select_action")
                    if endpoint not in self._endpoints:
                        raise ValueError(f"Unknown endpoint: {endpoint}")
                    handler, requires_input = self._endpoints[endpoint]
                    response = handler(request.get("data", {})) if requires_input else handler()
                except Exception as exc:  # Keep the REP socket usable after a bad request.
                    logging.exception("Kuavo policy request failed")
                    response = {"error": str(exc)}
                self._socket.send(self._serialize(response))
        finally:
            self._socket.close()
            self._context.term()
