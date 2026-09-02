from pathlib import Path

import pytest
import torch
import yaml

from hypercast4d.layers import HyperDense
from hypercast4d.models import PaperForecaster, parameter_count


NOTEBOOK_TABLES = {
    "quaternion": torch.tensor([[-1, 1, -1], [-1, -1, 1], [1, -1, -1]]),
    "coquaternion": torch.tensor([[-1, 1, -1], [-1, 1, -1], [1, 1, 1]]),
    "cl11": torch.tensor([[1, 1, 1], [-1, -1, 1], [-1, -1, 1]]),
}


def test_committed_config_stays_inside_paper_search_space() -> None:
    path = Path(__file__).parents[1] / "configs" / "evaluation.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    training = config["training"]
    adam = training["optimizer"]
    assert adam == {
        "name": "adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1.0e-7,
        "amsgrad": False,
    }
    assert training["loss"] == "mse"
    assert training["batch_size"] == 32
    assert training["epochs"] == 50
    assert training["early_stopping_patience"] is None
    assert training["restore_best_weights"] is False

    models = config["models"]
    assert models["cnn"]["units"] in {8, 16, 32, 64, 128}
    assert models["lstm"]["units"] in {8, 16, 32, 64, 128}
    assert models["hyper"]["units"] in {1, 2, 4, 8, 16, 32}
    for name in ("cnn", "lstm", "hyper"):
        settings = models[name]
        assert settings["dense_units"] in {8, 16, 32, 64}
        assert settings["activation"] in {"linear", "relu"}
        assert settings["dropout"] == 0.5
        assert settings["pool_size"] == 2
    assert models["cnn"]["kernel_size"] == 3

    paper_windows = {10, 20, 40, 60}
    paper_horizons = {1, 5, 10, 20}
    for cell in config["experiment"]["cells"]:
        assert cell["window"] in paper_windows
        assert cell["horizon"] in paper_horizons


@pytest.mark.parametrize("algebra", NOTEBOOK_TABLES)
def test_hyperdense_matches_supplementary_block_matrix(algebra: str) -> None:
    layer = HyperDense(2, 3, algebra=algebra, bias=False).double()
    inputs = torch.randn(5, 8, dtype=torch.float64)
    table = NOTEBOOK_TABLES[algebra]
    real, imag_i, imag_j, imag_k = layer.weight
    block_real = torch.cat(
        [real, table[0, 0] * imag_i, table[1, 1] * imag_j, table[2, 2] * imag_k]
    )
    block_i = torch.cat([imag_i, real, table[1, 2] * imag_k, table[2, 1] * imag_j])
    block_j = torch.cat([imag_j, table[0, 2] * imag_k, real, table[2, 0] * imag_i])
    block_k = torch.cat([imag_k, table[0, 1] * imag_j, table[1, 0] * imag_i, real])
    notebook_weight = torch.cat([block_real, block_i, block_j, block_k], dim=1)
    assert torch.allclose(layer(inputs), inputs @ notebook_weight)


def test_hyperdense_bias_matches_notebook_zero_initialization() -> None:
    layer = HyperDense(1, 3, algebra="coquaternion")
    assert torch.count_nonzero(layer.bias) == 0


def test_paper_parameter_count_for_hyper_model() -> None:
    # HyperDense: 4*1*8 + 4*8 = 64
    # dense-before: 32*32 + 32 = 1056
    # dense-after: (5*32)*32 + 32 = 5152
    # output: 32*1 + 1 = 33
    model = PaperForecaster(
        "hyper",
        window=10,
        horizon=1,
        first_layer_units=8,
        dense_before_pool=True,
        dense_after_pool=True,
        dense_units=32,
        activation="relu",
        dropout=0.5,
        pool_size=2,
        conv_kernel_size=3,
        algebra="quaternion",
    )
    assert parameter_count(model) == 64 + 1056 + 5152 + 33


def test_lstm_matches_notebook_sequence_then_pool_shape() -> None:
    model = PaperForecaster(
        "lstm",
        window=10,
        horizon=5,
        first_layer_units=16,
        dense_before_pool=False,
        dense_after_pool=False,
        dense_units=32,
        activation="linear",
        dropout=0.5,
        pool_size=2,
        conv_kernel_size=3,
    )
    sequence, _ = model.first(torch.randn(2, 10, 4))
    pooled = model.pool(sequence.transpose(1, 2)).transpose(1, 2)
    assert sequence.shape == (2, 10, 16)
    assert pooled.shape == (2, 5, 16)
    assert model(torch.randn(2, 10, 4)).shape == (2, 5)
    assert parameter_count(model) == 1344 + 405
