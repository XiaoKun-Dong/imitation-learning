from types import SimpleNamespace

import numpy as np

from examples.libero import object_condition


def test_empty_object_condition_is_all_zero():
    image = np.full((8, 12, 3), 127, dtype=np.uint8)

    mask, bbox, crop, point = object_condition.empty_condition(image)

    assert mask.shape == (8, 12)
    assert not mask.any()
    assert np.array_equal(bbox, np.zeros(4, dtype=np.float32))
    assert not crop.any()
    assert point is None


def test_wrong_condition_object_excludes_objects_of_interest():
    env = SimpleNamespace(
        obj_of_interest=["target", "basket"],
        instance_to_id={"target": 1, "basket": 2, "z_distractor": 3, "a_distractor": 4},
        env=SimpleNamespace(
            objects_dict={"target": object(), "basket": object(), "z_distractor": object(), "a_distractor": object()}
        ),
    )

    assert object_condition.select_wrong_object(env) == "a_distractor"
    segmentation = np.asarray([[0, 1, 4], [3, 4, 2]])
    assert np.array_equal(
        object_condition.mask_from_segmentation(env, segmentation, "a_distractor"),
        np.asarray([[False, False, True], [False, True, False]]),
    )
