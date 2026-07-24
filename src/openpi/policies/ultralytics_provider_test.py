import numpy as np

from openpi.policies import ultralytics_provider


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[1, 2, 5, 6], [10, 12, 18, 20]])
        self.conf = _FakeTensor([0.4, 0.9])
        self.cls = _FakeTensor([0, 1])

    def __len__(self):
        return 2


class _FakeMasks:
    def __init__(self):
        mask0 = np.zeros((16, 16), dtype=np.float32)
        mask1 = np.zeros((16, 16), dtype=np.float32)
        mask0[2:6, 1:5] = 1.0
        mask1[12:16, 10:16] = 1.0
        self.data = _FakeTensor([mask0, mask1])


class _FakeResult:
    boxes = _FakeBoxes()
    masks = _FakeMasks()
    names = {0: "blue_block", 1: "red_block"}


class _FakePredictor:
    def predict(self, image, **kwargs):
        del image, kwargs
        return [_FakeResult()]


def test_ultralytics_provider_selects_filtered_high_confidence_mask():
    provider = ultralytics_provider.UltralyticsObjectConditionProvider(
        "unused.pt",
        class_names=("red_block",),
        predictor=_FakePredictor(),
    )
    image = np.full((16, 16, 3), 7, dtype=np.uint8)

    output = provider(image)

    assert np.allclose(output["target_bbox"], np.asarray([10, 12, 18, 20], dtype=np.float32))
    assert output["target_mask"].shape == (16, 16)
    assert output["target_mask"][12:16, 10:16].all()
    assert output["target_crop"].sum() > 0
    assert np.allclose(output["object_condition_confidence"], np.asarray([0.9], dtype=np.float32))


def test_ultralytics_provider_returns_empty_condition_without_matching_class():
    provider = ultralytics_provider.UltralyticsObjectConditionProvider(
        "unused.pt",
        class_ids=(9,),
        predictor=_FakePredictor(),
    )
    image = np.full((16, 16, 3), 7, dtype=np.uint8)

    output = provider(image)

    assert not output["target_mask"].any()
    assert np.allclose(output["target_bbox"], 0.0)
    assert np.allclose(output["object_condition_confidence"], 0.0)
