import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def _example() -> dict:
    return {
        "observation/state": np.zeros((8,), dtype=np.float32),
        "observation/image": np.full((224, 224, 3), 7, dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "pick the red block",
    }


def test_libero_inputs_uses_online_object_condition_provider():
    calls = []

    def provider(image, *, prompt=None):
        calls.append((image.copy(), prompt))
        mask = np.zeros((224, 224), dtype=bool)
        mask[20:40, 10:30] = True
        return {
            "target_mask": mask,
            "target_bbox": np.asarray([10, 20, 30, 40], dtype=np.float32),
            "object_condition_confidence": 0.75,
        }

    transform = libero_policy.LiberoInputs(
        model_type=_model.ModelType.PI05,
        object_condition_provider=provider,
    )

    output = transform(_example())

    assert len(calls) == 1
    assert calls[0][1] == "pick the red block"
    assert np.array_equal(calls[0][0], _example()["observation/image"])
    assert output["target_mask"].shape == (224, 224)
    assert output["target_mask"].any()
    assert np.allclose(output["target_bbox"], np.asarray([10, 20, 30, 40], dtype=np.float32))
    assert output["target_crop"].sum() > 0
    assert np.allclose(output["object_condition_confidence"], 0.75)


def test_libero_inputs_keeps_explicit_object_condition_over_provider():
    def provider(image, *, prompt=None):
        del image, prompt
        return {
            "target_bbox": np.asarray([10, 20, 30, 40], dtype=np.float32),
            "object_condition_confidence": 0.75,
        }

    data = _example()
    data["target_bbox"] = np.asarray([1, 2, 3, 4], dtype=np.float32)
    transform = libero_policy.LiberoInputs(
        model_type=_model.ModelType.PI05,
        object_condition_provider=provider,
    )

    output = transform(data)

    assert np.allclose(output["target_bbox"], np.asarray([1, 2, 3, 4], dtype=np.float32))
    assert "object_condition_confidence" not in output
