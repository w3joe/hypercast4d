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

from .data import load_paper_data, prepare_windows
from .models import build_model, parameter_count
from .training import error_metrics, fit_model, predict, seed_everything


MODEL_NAMES = (
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
    for model, rows in efficiency.groupby("model", sort=False):
        axis.scatter(rows["parameters_mean"], rows["mae_mean"], label=model, s=55)
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Mean test MAE")
    axis.set_title("Accuracy–parameter trade-off")
    axis.legend(fontsize=8)
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
    model_config = config["models"]
    frame = load_paper_data(data_config["path"], data_config.get("target_column"))
    output = Path(experiment_config["output_dir"])
    if quick:
        output = output.with_name(f"{output.name}_quick")
    output.mkdir(parents=True, exist_ok=True)
    cells = experiment_config["cells"][:1] if quick else experiment_config["cells"]
    seeds = experiment_config["seeds"][:1] if quick else experiment_config["seeds"]
    epochs = min(3, training_config["epochs"]) if quick else training_config["epochs"]
    device = _device(training_config.get("device", "cpu"))
    rows: list[dict[str, Any]] = []
    results_path = output / "runs.csv"
    status_path = output / "status.json"
    started_at = datetime.now(UTC).isoformat()
    total_runs = len(cells) * len(seeds) * len(MODEL_NAMES)
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
            for name in MODEL_NAMES:
                seed_everything(int(seed))
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
                    model = build_model(
                        name,
                        window,
                        horizon,
                        hyper_hidden=int(model_config["hyper_hidden"]),
                        cnn_channels=int(model_config["cnn_channels"]),
                        lstm_hidden=int(model_config["lstm_hidden"]),
                        dropout=float(model_config["dropout"]),
                    )
                    parameters = parameter_count(model)
                    fitted = fit_model(
                        model,
                        train_data,
                        validation_data,
                        seed=int(seed),
                        epochs=int(epochs),
                        batch_size=int(training_config["batch_size"]),
                        learning_rate=float(training_config["learning_rate"]),
                        patience=int(training_config["patience"]),
                        device=device,
                    )
                    prediction_scaled = predict(model, test_data, device)
                    train_seconds = fitted.train_seconds
                    peak_memory = fitted.process_peak_rss_mb
                    epochs_ran = fitted.epochs_ran
                    validation_loss = fitted.best_validation_loss
                prediction = prepared.scaler.inverse_target(prediction_scaled)
                metrics = error_metrics(prediction, target)
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
