import dataclasses
import functools
import json
import logging
import pathlib
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_DEMOVLA_PARAMS_FILTER = nnx_utils.PathRegex("demovla_.*")
_DEMOVLA_INJECTION_FILTER = nnx_utils.PathRegex(
    r"demovla_(action_query_.*|memory_(key|value)_proj.*|action_output_proj.*|interaction_gates|dynamic_gate_.*)"
)
_DEMOVLA_EXTRACTOR_FILTER = nnx.All(_DEMOVLA_PARAMS_FILTER, nnx.Not(_DEMOVLA_INJECTION_FILTER))
_DEMOVLA_OUTPUT_PROJ_FILTER = nnx_utils.PathRegex("demovla_action_output_proj/.*")


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _append_metrics_jsonl(path: pathlib.Path, step: int, metrics: dict[str, Any]) -> None:
    record = {"step": step, **{key: np.asarray(value).item() for key, value in metrics.items()}}
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    flat_params_shape = traverse_util.flatten_dict(params_shape)
    flat_loaded_params = traverse_util.flatten_dict(loaded_params)
    extra_loaded_keys = set(flat_loaded_params) - set(flat_params_shape)
    if extra_loaded_keys:
        raise ValueError(f"Loaded checkpoint contains unexpected params: {sorted(extra_loaded_keys)}")
    for key, loaded_value in flat_loaded_params.items():
        expected_value = flat_params_shape[key]
        if expected_value.shape != loaded_value.shape:
            raise ValueError(f"Shape mismatch at {key}: expected {expected_value.shape}, got {loaded_value.shape}")
        if expected_value.dtype != loaded_value.dtype:
            raise ValueError(f"Dtype mismatch at {key}: expected {expected_value.dtype}, got {loaded_value.dtype}")

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in flat_loaded_params.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        model_metrics = {}
        if hasattr(model, "compute_loss_with_aux"):
            chunked_loss, model_metrics = model.compute_loss_with_aux(rng, observation, actions, train=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        loss = jnp.mean(chunked_loss)
        task_chunked_loss = chunked_loss - model_metrics.get("demovla_diversity_regularization", 0.0)
        horizon_metrics = {
            "loss_action_first": jnp.mean(task_chunked_loss[..., 0]),
            "loss_action_middle": jnp.mean(task_chunked_loss[..., task_chunked_loss.shape[-1] // 2]),
            "loss_action_last": jnp.mean(task_chunked_loss[..., -1]),
            **model_metrics,
        }
        return loss, horizon_metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, horizon_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        **horizon_metrics,
        "learning_rate": config.lr_schedule.create()(state.step),
        "grad_norm": optax.global_norm(grads),
        "update_norm": optax.global_norm(updates),
        "param_norm": optax.global_norm(kernel_params),
    }
    if hasattr(model, "demovla_action_output_proj"):
        info.update(
            {
                "demovla_adapter_grad_norm": optax.global_norm(grads.filter(_DEMOVLA_PARAMS_FILTER)),
                "demovla_adapter_param_norm": optax.global_norm(new_params.filter(_DEMOVLA_PARAMS_FILTER)),
                "demovla_extractor_grad_norm": optax.global_norm(grads.filter(_DEMOVLA_EXTRACTOR_FILTER)),
                "demovla_injection_grad_norm": optax.global_norm(grads.filter(_DEMOVLA_INJECTION_FILTER)),
                "demovla_output_proj_grad_norm": optax.global_norm(grads.filter(_DEMOVLA_OUTPUT_PROJ_FILTER)),
                "demovla_output_proj_update_norm": optax.global_norm(updates.filter(_DEMOVLA_OUTPUT_PROJ_FILTER)),
                "demovla_output_proj_param_norm": optax.global_norm(new_params.filter(_DEMOVLA_OUTPUT_PROJ_FILTER)),
            }
        )
        if hasattr(model, "demovla_interaction_gates"):
            raw_gates = model.demovla_interaction_gates.value
            gates = jax.nn.sigmoid(raw_gates)
            info["demovla_gate_mean"] = jnp.mean(gates)
            if model.interaction_injection_mode == "single_shot":
                gate_labels = ("input",)
            else:
                gate_labels = tuple(f"layer_{layer}" for layer in model.interaction_injection_layers)
            for label, raw_gate, gate in zip(gate_labels, raw_gates, gates, strict=True):
                info[f"demovla_gate_{label}_raw"] = raw_gate
                info[f"demovla_gate_{label}"] = gate
    if getattr(model, "object_condition_use_gate", False):
        raw_gate = model.object_condition_gate.value
        gate = jax.nn.sigmoid(raw_gate)
        info["object_condition_gate_raw"] = raw_gate
        info["object_condition_gate"] = gate
        info["object_condition_effective_scale"] = gate
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            latest_info = jax.device_get(infos[-1])
            # Optimization statistics are averaged over the interval. State-like
            # values should show the value at the current checkpoint boundary.
            for key, value in latest_info.items():
                is_scalar_gate_state = key == "demovla_gate_mean" or key.startswith("demovla_gate_layer_")
                if key == "learning_rate" or key.endswith("_param_norm") or is_scalar_gate_state:
                    reduced_info[key] = value
            console_info = {
                key: value
                for key, value in reduced_info.items()
                if "_flow_bin_" not in key and "_slot_" not in key
            }
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in console_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            _append_metrics_jsonl(config.checkpoint_dir / "metrics.jsonl", step, reduced_info)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
