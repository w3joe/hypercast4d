import math

import torch
from torch import nn
from torch.utils.data import TensorDataset

from hypercast4d.training import fit_model


def test_fit_model_reports_each_epoch_to_callback() -> None:
    features = torch.linspace(0, 1, 24).reshape(12, 2)
    targets = features.sum(dim=1, keepdim=True)
    dataset = TensorDataset(features, targets)
    observed: list[tuple[int, float, float]] = []
    model = nn.Linear(2, 1)

    result = fit_model(
        model,
        dataset,
        dataset,
        seed=7,
        epochs=3,
        batch_size=4,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-7,
        adam_amsgrad=False,
        loss_name="mse",
        shuffle=True,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        restore_best_weights=False,
        device=torch.device("cpu"),
        epoch_callback=lambda epoch, train, validation: observed.append(
            (epoch, train, validation)
        ),
    )

    assert result.epochs_ran == 3
    assert [epoch for epoch, _, _ in observed] == [1, 2, 3]
    assert all(math.isfinite(value) for row in observed for value in row[1:])
