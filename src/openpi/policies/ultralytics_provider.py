import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np


def _crop_from_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.int32)
    crop = np.zeros_like(image)
    if x2 <= x1 or y2 <= y1:
        return crop
    height, width = image.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    crop[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return crop


class UltralyticsObjectConditionProvider:
    """YOLO-seg provider that returns RoVLA target-object fields from an RGB image."""

    def __init__(
        self,
        model_path: str | pathlib.Path,
        *,
        repo_path: str | pathlib.Path | None = None,
        class_names: Sequence[str] | None = None,
        class_ids: Sequence[int] | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | int | None = None,
        predictor: Any | None = None,
    ):
        if class_names is not None and class_ids is not None:
            raise ValueError("Use either class_names or class_ids, not both")
        self.class_names = tuple(class_names) if class_names is not None else None
        self.class_ids = tuple(class_ids) if class_ids is not None else None
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = predictor if predictor is not None else self._load_model(model_path, repo_path)

    @staticmethod
    def _load_model(model_path: str | pathlib.Path, repo_path: str | pathlib.Path | None):
        if repo_path is not None:
            repo_path = pathlib.Path(repo_path).resolve()
            sys.path.insert(0, str(repo_path))
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import YOLO

        return YOLO(str(model_path))

    def __call__(self, image: np.ndarray, *, prompt: str | None = None) -> dict[str, np.ndarray]:
        del prompt
        image = np.asarray(image)
        results = self.model.predict(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        if not results:
            return self._empty(image)
        return self._result_to_condition(results[0], image)

    def _result_to_condition(self, result: Any, image: np.ndarray) -> dict[str, np.ndarray]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._empty(image)

        xyxy = np.asarray(boxes.xyxy.detach().cpu().numpy(), dtype=np.float32)
        conf = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=np.float32)
        cls = np.asarray(boxes.cls.detach().cpu().numpy(), dtype=np.int32)
        keep = self._class_filter(result, cls)
        if not np.any(keep):
            return self._empty(image)
        candidate_indices = np.where(keep)[0]
        selected = candidate_indices[np.argmax(conf[candidate_indices])]
        bbox = xyxy[selected]

        mask = np.zeros(image.shape[:2], dtype=bool)
        if result.masks is not None and result.masks.data is not None:
            masks = np.asarray(result.masks.data.detach().cpu().numpy())
            if selected < masks.shape[0]:
                mask = masks[selected] > 0.5
                if mask.shape != image.shape[:2]:
                    mask = self._resize_mask(mask, image.shape[:2])

        return {
            "target_mask": mask,
            "target_bbox": bbox.astype(np.float32),
            "target_crop": _crop_from_bbox(image, bbox),
            "object_condition_confidence": np.asarray([conf[selected]], dtype=np.float32),
        }

    def _class_filter(self, result: Any, cls: np.ndarray) -> np.ndarray:
        if self.class_ids is not None:
            return np.isin(cls, np.asarray(self.class_ids, dtype=np.int32))
        if self.class_names is None:
            return np.ones(cls.shape, dtype=bool)
        names = getattr(result, "names", {})
        selected = {name.lower() for name in self.class_names}
        labels = np.asarray([str(names.get(int(class_id), int(class_id))).lower() for class_id in cls])
        return np.isin(labels, list(selected))

    @staticmethod
    def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        from PIL import Image

        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (shape[1], shape[0]),
            resample=Image.Resampling.NEAREST,
        )
        return np.asarray(resized) > 127

    @staticmethod
    def _empty(image: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "target_mask": np.zeros(image.shape[:2], dtype=bool),
            "target_bbox": np.zeros((4,), dtype=np.float32),
            "target_crop": np.zeros_like(image),
            "object_condition_confidence": np.zeros((1,), dtype=np.float32),
        }
