"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from collections.abc import Iterator
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _arrow_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values).reshape(len(array), array.type.list_size)
    return np.asarray(array)


def create_lerobot_v3_parquet_batches(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[Iterator[dict], int] | None:
    """Build norm-stat batches directly from LeRobot v3 parquet files.

    Video observations are irrelevant to state/action normalization. Reading the
    parquet columns directly avoids decoding two videos for every frame.
    """
    if data_config.root is None or data_config.norm_stats_transforms is None:
        return None

    root = pathlib.Path(data_config.root).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        return None

    parquet_paths = sorted(root.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot v3 parquet files found under {root / 'data'}")

    state_key = "observation.state"
    required_keys = {state_key, "episode_index", "index", *data_config.action_sequence_keys}
    feature_keys = set(info.get("features", {}))
    missing_keys = required_keys - feature_keys
    if missing_keys:
        raise KeyError(f"LeRobot v3 dataset is missing norm-stat columns: {sorted(missing_keys)}")

    table = pq.read_table(parquet_paths, columns=sorted(required_keys))
    columns = {key: _arrow_column_to_numpy(table[key]) for key in required_keys}
    order = np.argsort(columns["index"], kind="stable")
    columns = {key: value[order] for key, value in columns.items()}

    episode_indices = columns["episode_index"]
    num_frames = len(episode_indices)
    boundaries = np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries - 1, [num_frames - 1]))
    episode_end = np.empty(num_frames, dtype=np.int64)
    for start, end in zip(starts, ends, strict=True):
        episode_end[start : end + 1] = end

    query_indices = np.minimum(
        np.arange(num_frames, dtype=np.int64)[:, None] + np.arange(action_horizon, dtype=np.int64)[None, :],
        episode_end[:, None],
    )
    if max_frames is not None and max_frames < num_frames:
        selected_indices = np.random.default_rng(0).choice(num_frames, size=max_frames, replace=False)
    else:
        selected_indices = np.arange(num_frames)
    num_batches = len(selected_indices) // batch_size
    selected_indices = selected_indices[: num_batches * batch_size]
    transform = transforms.compose(data_config.norm_stats_transforms.inputs)

    def iterate_batches() -> Iterator[dict]:
        for batch_index in range(num_batches):
            batch_indices = selected_indices[batch_index * batch_size : (batch_index + 1) * batch_size]
            raw_batch = {
                state_key: columns[state_key][batch_indices],
                **{key: columns[key][query_indices[batch_indices]] for key in data_config.action_sequence_keys},
            }
            yield transform(raw_batch)

    return iterate_batches(), num_batches


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    norm_stats_transforms = data_config.norm_stats_transforms
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        load_videos=norm_stats_transforms is None,
    )
    input_transforms = (
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
        if norm_stats_transforms is None
        else list(norm_stats_transforms.inputs)
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *input_transforms,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    parquet_batches = create_lerobot_v3_parquet_batches(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        max_frames,
    )
    if parquet_batches is not None:
        data_loader, num_batches = parquet_batches
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Keep the output location identical to the location used by training and
    # inference. repo_id can contain slashes or differ from an explicit asset_id.
    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
