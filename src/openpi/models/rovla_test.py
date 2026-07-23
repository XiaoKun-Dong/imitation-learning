import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import rovla_config
from openpi.training import config as _training_config
from openpi.training import weight_loaders


def test_rovla_config_is_registered_and_only_adapter_is_trainable():
    train_config = _training_config.get_config("rovla_libero")
    assert isinstance(train_config.model, rovla_config.RoVLAConfig)
    assert train_config.model.model_type is _model.ModelType.PI05

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    trainable_paths = set(nnx.state(abstract_model, nnx.All(nnx.Param, train_config.trainable_filter)).flat_state())
    assert trainable_paths
    assert all(path[0].startswith("rovla_") for path in trainable_paths)


def test_rovla_checkpoint_loader_keeps_initialized_adapter_params():
    merged = weight_loaders._merge_params(  # noqa: SLF001
        {"base": {"weight": np.ones((1,), dtype=np.float32)}},
        {
            "base": {"weight": np.zeros((1,), dtype=np.float32)},
            "rovla_adapter": {"weight": np.full((1,), 2.0, dtype=np.float32)},
        },
        missing_regex=".*(lora|rovla).*",
    )
    assert merged["rovla_adapter"]["weight"].item() == 2.0


def test_rovla_config_rejects_invalid_corruption_size():
    with pytest.raises(ValueError, match="mask corruption"):
        rovla_config.RoVLAConfig(rovla_mask_shift_pixels=-1)
