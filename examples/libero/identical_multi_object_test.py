import dataclasses

import numpy as np
import pytest

from examples.libero import identical_multi_object


def test_build_bddl_declares_and_requires_every_identical_object() -> None:
    args = identical_multi_object.Args(object_type="bbq_sauce", num_objects=4)

    bddl, prompt, object_names = identical_multi_object.build_bddl(args)

    assert object_names == ("bbq_sauce_1", "bbq_sauce_2", "bbq_sauce_3", "bbq_sauce_4")
    assert "bbq_sauce_1 bbq_sauce_2 bbq_sauce_3 bbq_sauce_4 - bbq_sauce" in bddl
    for object_name in object_names:
        assert f"(On {object_name} floor_clutter_region)" in bddl
        assert f"(In {object_name} basket_1_contain_region)" in bddl
    assert "one item at a time" in prompt


@pytest.mark.parametrize("object_type", ["BBQ sauce", "../bbq_sauce", "bbq-sauce"])
def test_build_bddl_rejects_unsafe_object_type(object_type: str) -> None:
    with pytest.raises(ValueError, match="object_type"):
        identical_multi_object.build_bddl(identical_multi_object.Args(object_type=object_type))


def test_update_approach_target_uses_hysteresis() -> None:
    target, switched = identical_multi_object._update_approach_target(  # noqa: SLF001
        {"item_1": 0.10, "item_2": 0.11}, None, 0.16, 0.02
    )
    assert (target, switched) == ("item_1", False)

    target, switched = identical_multi_object._update_approach_target(  # noqa: SLF001
        {"item_1": 0.11, "item_2": 0.10}, target, 0.16, 0.02
    )
    assert (target, switched) == ("item_1", False)

    target, switched = identical_multi_object._update_approach_target(  # noqa: SLF001
        {"item_1": 0.14, "item_2": 0.09}, target, 0.16, 0.02
    )
    assert (target, switched) == ("item_2", True)


def test_clutter_restart_target_ignores_object_left_at_basket() -> None:
    target = identical_multi_object._clutter_restart_target(  # noqa: SLF001
        {"item_at_basket": 0.04, "item_in_clutter": 0.12},
        {
            "item_at_basket": np.array([0.0, 0.25]),
            "item_in_clutter": np.array([0.1, 0.0]),
        },
        (-0.225, -0.27, 0.18, 0.10),
        0.16,
    )

    assert target == "item_in_clutter"


def test_validate_args_rejects_invalid_counts_and_bounds() -> None:
    with pytest.raises(ValueError, match="num_objects"):
        identical_multi_object.build_bddl(identical_multi_object.Args(num_objects=1))

    args = dataclasses.replace(
        identical_multi_object.Args(), clutter_x_min=0.2, clutter_x_max=-0.2
    )
    with pytest.raises(ValueError, match="clutter region"):
        identical_multi_object.build_bddl(args)
