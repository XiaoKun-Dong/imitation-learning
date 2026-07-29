"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run python examples/libero/convert_libero_data_to_lerobot.py --data-dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run python examples/libero/convert_libero_data_to_lerobot.py --data-dir /path/to/your/data --push-to-hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The default output matches the DemoVLA training configs:
``data/lerobot/local/libero`` with repo id ``local/libero``.
Running this conversion script will take approximately 30 minutes.
"""

import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

DEFAULT_REPO_ID = "local/libero"
DEFAULT_OUTPUT_ROOT = "data/lerobot/local/libero"
RAW_DATASET_NAMES = [
    "libero_object_no_noops",
]


def main(
    data_dir: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    repo_id: str = DEFAULT_REPO_ID,
    overwrite: bool = False,
    push_to_hub: bool = False,
):
    output_path = pathlib.Path(output_root).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
