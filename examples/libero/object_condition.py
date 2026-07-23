import numpy as np


def empty_condition(image: np.ndarray):
    height, width = image.shape[:2]
    return (
        np.zeros((height, width), dtype=bool),
        np.zeros((4,), dtype=np.float32),
        np.zeros_like(image),
        None,
    )


def select_wrong_object(env) -> str:
    """Select one deterministic movable distractor, excluding target and destination objects."""
    excluded = set(env.obj_of_interest)
    candidates = sorted(
        name for name in env.env.objects_dict if name not in excluded and name in env.instance_to_id
    )
    if not candidates:
        raise ValueError("No non-target movable object is available for the wrong-mask ablation")
    return candidates[0]


def mask_from_segmentation(env, segmentation_image: np.ndarray, object_name: str | None) -> np.ndarray:
    if not env.obj_of_interest:
        return np.zeros(segmentation_image.shape[:2], dtype=bool)
    selected_object = object_name or env.obj_of_interest[0]
    if selected_object not in env.instance_to_id:
        raise KeyError(f"No segmentation id for condition object {selected_object!r}")
    return segmentation_image == env.instance_to_id[selected_object]
