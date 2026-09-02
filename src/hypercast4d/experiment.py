"""Run the bounded, leakage-safe HyperCast4D evaluation matrix."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .data import load_paper_data, prepare_windows
from .models import build_model, parameter_count
from .training import error_metrics, fit_model, predict, seed_everything


SUPPORTED_MODEL_NAMES = (
    "persistence",
    "linear",
    "cnn",
    "lstm",
    "hyper_quaternion",
    "hyper_coquaternion",
    "hyper_cl11",
)

RESULT_COLUMNS = (
    "window",
    "horizon",
    "seed",
    "model",
    "mae",
    "mse",
    "parameters",
    "train_seconds",
    "process_peak_rss_mb",
    "epochs_ran",
    "best_validation_mse_scaled",
    "train_samples",
    "validation_samples",
    "test_samples",
)


def _atomic_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _write_live_results(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_status(
    path: Path,
    *,
    state: str,
    completed: int,
    total: int,
    started_at: str,
) -> None:
    status = {
        "state": state,
        "completed": completed,
        "total": total,
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_text(path, json.dumps(status, indent=2) + "\n")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _tensorboard_log_path(
    root: Path,
    experiment_id: str,
    window: int,
    horizon: int,
    model: str,
    seed: int,
) -> Path:
    return root / experiment_id / f"w{window}_h{horizon}" / model / f"seed_{seed}"


def _log_tensorboard_result(
    writer: SummaryWriter,
    *,
    window: int,
    horizon: int,
    seed: int,
    model: str,
    epochs_ran: int,
    epochs_requested: int,
    parameters: int,
    train_seconds: float,
    metrics: dict[str, float],
    model_settings: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    step = max(epochs_ran, 0)
    writer.add_scalar("metrics/test_mae_original_units", metrics["mae"], step)
    writer.add_scalar("metrics/test_mse_original_units", metrics["mse"], step)
    writer.add_scalar("model/trainable_parameters", parameters, step)
    writer.add_scalar("timing/train_seconds", train_seconds, step)
    effective_config = {
        "model": model,
        "window": window,
        "horizon": horizon,
        "seed": seed,
        "epochs_requested": epochs_requested,
        "epochs_ran": epochs_ran,
        "batch_size": int(training_config["batch_size"]),
        "loss": str(training_config["loss"]),
        "shuffle": bool(training_config["shuffle"]),
        "early_stopping_patience": (
            training_config["early_stopping_patience"]
            if training_config["early_stopping_patience"] is not None
            else "disabled"
        ),
        "restore_best_weights": bool(training_config["restore_best_weights"]),
        **{
            f"optimizer_{key}": value
            for key, value in training_config["optimizer"].items()
        },
        **{f"model_{key}": value for key, value in model_settings.items()},
    }
    writer.add_text(
        "run/effective_configuration",
        f"```json\n{json.dumps(effective_config, indent=2)}\n```",
        0,
    )
    writer.add_hparams(
        effective_config,
        {
            "hparam/test_mae": metrics["mae"],
            "hparam/test_mse": metrics["mse"],
        },
        run_name="hparams",
    )
    writer.flush()


def _plot(summary: pd.DataFrame, destination: Path) -> None:
    labels = [f"w{row.window}/h{row.horizon}" for row in summary.itertuples()]
    models = list(dict.fromkeys(summary["model"]))
    cells = list(dict.fromkeys(labels))
    x = np.arange(len(cells))
    width = 0.8 / len(models)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    for offset, model in enumerate(models):
        rows = summary[summary["model"] == model]
        by_cell = {
            f"w{row.window}/h{row.horizon}": row.mae_mean for row in rows.itertuples()
        }
        axis.bar(
            x + (offset - (len(models) - 1) / 2) * width,
            [by_cell[c] for c in cells],
            width,
            label=model,
        )
    axis.set_xticks(x, cells)
    axis.set_ylabel("Test MAE (original target units; lower is better)")
    axis.set_title("Leakage-safe forecasting evaluation")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "mae_by_cell.png", dpi=180)
    plt.close(fig)

    efficiency = summary[summary["model"] != "persistence"]
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    if efficiency.empty:
        axis.text(
            0.5,
            0.5,
            "No trainable models enabled",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        for model, rows in efficiency.groupby("model", sort=False):
            axis.scatter(rows["parameters_mean"], rows["mae_mean"], label=model, s=55)
        axis.set_xscale("log")
        axis.legend(fontsize=8)
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Mean test MAE")
    axis.set_title("Accuracy–parameter trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "accuracy_vs_parameters.png", dpi=180)
    plt.close(fig)


def run(
    config: dict[str, Any], quick: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_config = config["data"]
    experiment_config = config["experiment"]
    training_config = config["training"]
    optimizer_config = training_config["optimizer"]
    tensorboard_config = config["tensorboard"]
    model_config = config["models"]
    model_names = tuple(model_config["enabled"])
    unknown_models = sorted(set(model_names) - set(SUPPORTED_MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"Unsupported models in configuration: {unknown_models}")
    if not model_names:
        raise ValueError("At least one model must be enabled")
    if len(model_names) != len(set(model_names)):
        raise ValueError("Enabled model names must be unique")
    frame = load_paper_data(data_config["path"], data_config.get("target_column"))
    output = Path(experiment_config["output_dir"])
    if quick:
        output = output.with_name(f"{output.name}_quick")
    output.mkdir(parents=True, exist_ok=True)
    quick_config = experiment_config["quick"]
    if quick:
        quick_cell_count = int(quick_config["cell_count"])
        quick_seed_count = int(quick_config["seed_count"])
        quick_epochs = int(quick_config["epochs"])
        if min(quick_cell_count, quick_seed_count, quick_epochs) < 1:
            raise ValueError(
                "Quick-run cell_count, seed_count, and epochs must be positive"
            )
        cells = experiment_config["cells"][:quick_cell_count]
        seeds = experiment_config["seeds"][:quick_seed_count]
        epochs = min(quick_epochs, int(training_config["epochs"]))
    else:
        cells = experiment_config["cells"]
        seeds = experiment_config["seeds"]
        epochs = int(training_config["epochs"])
    if not cells or not seeds:
        raise ValueError("Experiment cells and seeds must not be empty")
    if (
        min(
            epochs,
            int(training_config["batch_size"]),
            int(training_config["evaluation_batch_size"]),
        )
        < 1
    ):
        raise ValueError("Epoch and batch-size settings must be positive")
    patience_value = training_config["early_stopping_patience"]
    patience = None if patience_value is None else int(patience_value)
    if patience is not None and patience < 1:
        raise ValueError("early_stopping_patience must be null or positive")
    if float(training_config["early_stopping_min_delta"]) < 0:
        raise ValueError("early_stopping_min_delta must not be negative")
    if optimizer_config["name"] != "adam":
        raise ValueError("This paper-aligned runner supports the Adam optimizer")
    beta1 = float(optimizer_config["beta1"])
    beta2 = float(optimizer_config["beta2"])
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if (
        min(
            float(optimizer_config["learning_rate"]),
            float(optimizer_config["epsilon"]),
        )
        <= 0
    ):
        raise ValueError("Adam learning_rate and epsilon must be positive")
    tensorboard_enabled = bool(tensorboard_config["enabled"])
    tensorboard_flush_seconds = int(tensorboard_config["flush_seconds"])
    if tensorboard_flush_seconds < 1:
        raise ValueError("tensorboard.flush_seconds must be positive")
    device = _device(training_config["device"])
    rows: list[dict[str, Any]] = []
    results_path = output / "runs.csv"
    status_path = output / "status.json"
    started = datetime.now(UTC)
    started_at = started.isoformat()
    experiment_id = started.strftime("%Y%m%dT%H%M%S_%fZ")
    if quick:
        experiment_id = f"{experiment_id}_quick"
    tensorboard_run_dir = (
        Path(tensorboard_config["log_dir"]) / experiment_id
        if tensorboard_enabled
        else None
    )
    total_runs = len(cells) * len(seeds) * len(model_names)
    _write_live_results(results_path, rows)
    _write_status(
        status_path,
        state="running",
        completed=0,
        total=total_runs,
        started_at=started_at,
    )

    for cell in cells:
        window, horizon = int(cell["window"]), int(cell["horizon"])
        prepared = prepare_windows(
            frame,
            window,
            horizon,
            float(data_config["train_fraction"]),
            float(data_config["validation_fraction"]),
        )
        train_data = prepared.train.as_dataset()
        validation_data = prepared.validation.as_dataset()
        test_data = prepared.test.as_dataset()
        target_scaled = prepared.test.y
        target = prepared.scaler.inverse_target(target_scaled)

        for seed in seeds:
            for name in model_names:
                seed_everything(int(seed))
                model_settings: dict[str, Any] = {}
                writer = None
                if tensorboard_run_dir is not None:
                    log_path = _tensorboard_log_path(
                        Path(tensorboard_config["log_dir"]),
                        experiment_id,
                        window,
                        horizon,
                        name,
                        int(seed),
                    )
                    writer = SummaryWriter(
                        log_dir=str(log_path), flush_secs=tensorboard_flush_seconds
                    )
                if name == "persistence":
                    prediction_scaled = np.repeat(
                        prepared.test.x[:, -1, 0:1], horizon, axis=1
                    )
                    train_seconds = 0.0
                    peak_memory = float("nan")
                    epochs_ran = 0
                    validation_loss = float("nan")
                    parameters = 0
                else:
                    if name == "linear":
                        model = build_model(name, window, horizon)
                    else:
                        settings_name = "hyper" if name.startswith("hyper_") else name
                        model_settings = dict(model_config[settings_name])
                        model = build_model(
                            name,
                            window,
                            horizon,
                            first_layer_units=int(model_settings["units"]),
                            conv_kernel_size=(
                                int(model_settings["kernel_size"])
                                if name == "cnn"
                                else None
                            ),
                            dense_before_pool=bool(model_settings["dense_before_pool"]),
                            dense_after_pool=bool(model_settings["dense_after_pool"]),
                            dense_units=int(model_settings["dense_units"]),
                            activation=str(model_settings["activation"]),
                            dropout=float(model_settings["dropout"]),
                            pool_size=int(model_settings["pool_size"]),
                        )
                    parameters = parameter_count(model)

                    def log_epoch(
                        epoch: int, train_loss: float, validation_loss: float
                    ) -> None:
                        if writer is not None:
                            writer.add_scalar("loss/train_scaled", train_loss, epoch)
                            writer.add_scalar(
                                "loss/validation_scaled", validation_loss, epoch
                            )

                    fitted = fit_model(
                        model,
                        train_data,
                        validation_data,
                        seed=int(seed),
                        epochs=int(epochs),
                        batch_size=int(training_config["batch_size"]),
                        learning_rate=float(optimizer_config["learning_rate"]),
                        adam_beta1=beta1,
                        adam_beta2=beta2,
                        adam_epsilon=float(optimizer_config["epsilon"]),
                        adam_amsgrad=bool(optimizer_config["amsgrad"]),
                        loss_name=str(training_config["loss"]),
                        shuffle=bool(training_config["shuffle"]),
                        early_stopping_patience=patience,
                        early_stopping_min_delta=float(
                            training_config["early_stopping_min_delta"]
                        ),
                        restore_best_weights=bool(
                            training_config["restore_best_weights"]
                        ),
                        device=device,
                        epoch_callback=log_epoch,
                    )
                    prediction_scaled = predict(
                        model,
                        test_data,
                        device,
                        int(training_config["evaluation_batch_size"]),
                    )
                    train_seconds = fitted.train_seconds
                    peak_memory = fitted.process_peak_rss_mb
                    epochs_ran = fitted.epochs_ran
                    validation_loss = fitted.best_validation_loss
                prediction = prepared.scaler.inverse_target(prediction_scaled)
                metrics = error_metrics(prediction, target)
                if writer is not None:
                    _log_tensorboard_result(
                        writer,
                        window=window,
                        horizon=horizon,
                        seed=int(seed),
                        model=name,
                        epochs_ran=epochs_ran,
                        epochs_requested=epochs,
                        parameters=parameters,
                        train_seconds=train_seconds,
                        metrics=metrics,
                        model_settings=model_settings,
                        training_config=training_config,
                    )
                    writer.close()
                row = {
                    "window": window,
                    "horizon": horizon,
                    "seed": int(seed),
                    "model": name,
                    **metrics,
                    "parameters": parameters,
                    "train_seconds": train_seconds,
                    "process_peak_rss_mb": peak_memory,
                    "epochs_ran": epochs_ran,
                    "best_validation_mse_scaled": validation_loss,
                    "train_samples": len(prepared.train.x),
                    "validation_samples": len(prepared.validation.x),
                    "test_samples": len(prepared.test.x),
                }
                rows.append(row)
                _write_live_results(results_path, rows)
                _write_status(
                    status_path,
                    state="running",
                    completed=len(rows),
                    total=total_runs,
                    started_at=started_at,
                )
                print(
                    f"{name:22s} w={window:2d} h={horizon:2d} seed={seed:2d} "
                    f"MAE={metrics['mae']:.5g}"
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["window", "horizon", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            parameters_mean=("parameters", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            process_peak_rss_mb_mean=("process_peak_rss_mb", "mean"),
        )
        .sort_values(["window", "horizon", "mae_mean"])
    )
    _write_live_results(results_path, rows)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "protocol": "chronological-v1",
        "quick": quick,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "output_dir": str(output),
        "tensorboard_run_dir": (
            str(tensorboard_run_dir) if tensorboard_run_dir is not None else None
        ),
        "columns": list(frame.columns),
        "rows": len(frame),
        "config": config,
    }
    _atomic_text(
        output / "metadata.json",
        json.dumps(metadata, indent=2) + "\n",
    )
    _plot(summary, output)
    _write_status(
        status_path,
        state="complete",
        completed=len(rows),
        total=total_runs,
        started_at=started_at,
    )
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one cell, one seed and at most three epochs",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run(config, args.quick)


if __name__ == "__main__":
    main()
