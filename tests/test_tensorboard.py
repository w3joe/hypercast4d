from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from hypercast4d.experiment import _log_tensorboard_result


def test_completed_run_writes_scalars_text_and_hparams(tmp_path) -> None:
    log_dir = tmp_path / "run"
    writer = SummaryWriter(log_dir=str(log_dir))
    _log_tensorboard_result(
        writer,
        window=10,
        horizon=1,
        seed=7,
        model="hyper_quaternion",
        epochs_ran=3,
        epochs_requested=3,
        parameters=6305,
        train_seconds=0.25,
        metrics={"mae": 0.1, "mse": 0.02},
        model_settings={"units": 8, "activation": "relu"},
        training_config={
            "epochs": 50,
            "batch_size": 32,
            "loss": "mse",
            "shuffle": True,
            "early_stopping_patience": None,
            "restore_best_weights": False,
            "optimizer": {
                "name": "adam",
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-7,
                "amsgrad": False,
            },
        },
    )
    writer.close()

    accumulator = EventAccumulator(str(log_dir)).Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert {
        "metrics/test_mae_original_units",
        "metrics/test_mse_original_units",
        "model/trainable_parameters",
        "timing/train_seconds",
    } <= scalar_tags
    assert "run/effective_configuration/text_summary" in accumulator.Tags()["tensors"]
    assert list((log_dir / "hparams").glob("events.out.tfevents.*"))
