"""Worker process for validation and final-test playground jobs."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .architecture import build_architecture, normalize_architecture_spec
from .data import MinMaxStats, load_paper_data, prepare_windows
from .models import parameter_count
from .training import error_metrics, fit_model, predict, seed_everything


EVALUATION_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "cells": [{"window": 10, "horizon": 1}],
        "seeds": [7],
        "epochs": 3,
        "folds": [{"train_fraction": 0.70, "validation_fraction": 0.15}],
    },
    "standard": {
        "cells": [
            {"window": 10, "horizon": 1},
            {"window": 20, "horizon": 5},
            {"window": 60, "horizon": 20},
        ],
        "seeds": [7, 19, 31, 43, 59],
        "epochs": 50,
        "folds": [{"train_fraction": 0.70, "validation_fraction": 0.15}],
    },
    "robust": {
        "cells": [
            {"window": window, "horizon": horizon}
            for window in (10, 20, 40, 60)
            for horizon in (1, 5, 10, 20)
        ],
        "seeds": [7, 19, 31, 43, 59],
        "epochs": 50,
        "folds": [
            {"train_fraction": 0.55, "validation_fraction": 0.10},
            {"train_fraction": 0.65, "validation_fraction": 0.10},
            {"train_fraction": 0.75, "validation_fraction": 0.10},
        ],
    },
}


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    converted = int(value)
    if converted != value or not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return converted


def _positive_float(value: Any, name: str) -> float:
    converted = float(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def normalize_evaluation(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    preset_name = str(raw.get("preset", "quick"))
    if preset_name not in EVALUATION_PRESETS:
        raise ValueError(f"unknown evaluation preset {preset_name!r}")
    preset = json.loads(json.dumps(EVALUATION_PRESETS[preset_name]))
    cells_raw = raw.get("cells", preset["cells"])
    if not isinstance(cells_raw, list) or not cells_raw:
        raise ValueError("cells must be a non-empty list")
    cells: list[dict[str, int]] = []
    for cell in cells_raw:
        if not isinstance(cell, dict):
            raise ValueError("each cell must be an object")
        cells.append(
            {
                "window": _integer(cell.get("window"), "window", 2, 512),
                "horizon": _integer(cell.get("horizon"), "horizon", 1, 128),
            }
        )
    if len({(cell["window"], cell["horizon"]) for cell in cells}) != len(cells):
        raise ValueError("evaluation cells must be unique")
    seeds_raw = raw.get("seeds", preset["seeds"])
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("seeds must be a non-empty list")
    seeds = [_integer(seed, "seed", 0, 2**31 - 1) for seed in seeds_raw]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    loss = str(raw.get("loss", "mse"))
    if loss not in {"mse", "mae", "huber"}:
        raise ValueError("loss must be mse, mae, or huber")
    device = str(raw.get("device", "cpu"))
    if device not in {"cpu", "auto", "mps", "cuda"}:
        raise ValueError("device must be cpu, auto, mps, or cuda")
    patience_raw = raw.get("early_stopping_patience", 10)
    patience = (
        None
        if patience_raw is None
        else _integer(patience_raw, "early_stopping_patience", 1, 1000)
    )
    target_column = str(raw.get("target_column", "Copper")).strip()
    if not target_column or len(target_column) > 128:
        raise ValueError("target_column is invalid")
    data_path = str(raw.get("data_path", "data/raw/paper_data.xlsx"))
    path = Path(data_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("data_path must be a relative path inside the project")
    if not data_path.startswith("data/"):
        raise ValueError("data_path must be inside data/")
    return {
        "preset": preset_name,
        "data_path": data_path,
        "target_column": target_column,
        "cells": cells,
        "seeds": seeds,
        "epochs": _integer(raw.get("epochs", preset["epochs"]), "epochs", 1, 5000),
        "batch_size": _integer(raw.get("batch_size", 32), "batch_size", 1, 8192),
        "evaluation_batch_size": _integer(
            raw.get("evaluation_batch_size", 256),
            "evaluation_batch_size",
            1,
            8192,
        ),
        "learning_rate": _positive_float(raw.get("learning_rate", 0.001), "learning_rate"),
        "loss": loss,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": float(raw.get("early_stopping_min_delta", 0.0)),
        "device": device,
        "folds": preset["folds"],
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return device


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _status(job_dir: Path, **updates: Any) -> None:
    path = job_dir / "status.json"
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(updates)
    current["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(path, current)


def _metrics_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    persistence: np.ndarray,
    *,
    window: int,
    horizon: int,
    seed: int,
    fold: int,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = error_metrics(prediction, target)
    persistence_metrics = error_metrics(persistence, target)
    aggregate = {
        "window": window,
        "horizon": horizon,
        "seed": seed,
        "fold": fold,
        "split": split,
        **metrics,
        "persistence_mae": persistence_metrics["mae"],
        "persistence_mse": persistence_metrics["mse"],
        "mae_ratio": metrics["mae"] / persistence_metrics["mae"],
        "mse_ratio": metrics["mse"] / persistence_metrics["mse"],
    }
    per_lead: list[dict[str, Any]] = []
    for lead in range(horizon):
        lead_metrics = error_metrics(prediction[:, lead], target[:, lead])
        lead_persistence = error_metrics(persistence[:, lead], target[:, lead])
        per_lead.append(
            {
                "window": window,
                "horizon": horizon,
                "seed": seed,
                "fold": fold,
                "split": split,
                "lead": lead + 1,
                **lead_metrics,
                "persistence_mae": lead_persistence["mae"],
                "persistence_mse": lead_persistence["mse"],
            }
        )
    return aggregate, per_lead


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["window", "horizon"], as_index=False).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        persistence_mae=("persistence_mae", "mean"),
        persistence_mse=("persistence_mse", "mean"),
        parameters=("parameters", "mean"),
        train_seconds=("train_seconds", "mean"),
        epochs_median=("epochs_ran", "median"),
    )
    grouped["mae_ratio"] = grouped["mae_mean"] / grouped["persistence_mae"]
    grouped["mse_ratio"] = grouped["mse_mean"] / grouped["persistence_mse"]
    return json.loads(grouped.to_json(orient="records"))


def _fit_kwargs(evaluation: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "batch_size": evaluation["batch_size"],
        "learning_rate": evaluation["learning_rate"],
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-7,
        "adam_amsgrad": False,
        "loss_name": evaluation["loss"],
        "shuffle": True,
        "early_stopping_min_delta": evaluation["early_stopping_min_delta"],
        "device": device,
    }


def run_validation(job_dir: Path, request: dict[str, Any]) -> None:
    architecture = normalize_architecture_spec(request["architecture"])
    evaluation = normalize_evaluation(request.get("evaluation"))
    frame = load_paper_data(evaluation["data_path"], evaluation["target_column"])
    device = resolve_device(evaluation["device"])
    total = len(evaluation["cells"]) * len(evaluation["seeds"]) * len(evaluation["folds"])
    rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    _status(job_dir, state="running", completed=0, total=total, phase="validation")
    print(
        f"Validation started: {architecture['name']} · {total} fitted runs",
        flush=True,
    )
    for cell in evaluation["cells"]:
        window, horizon = cell["window"], cell["horizon"]
        for fold_index, fold in enumerate(evaluation["folds"], start=1):
            prepared = prepare_windows(
                frame,
                window,
                horizon,
                fold["train_fraction"],
                fold["validation_fraction"],
            )
            for seed in evaluation["seeds"]:
                print(
                    f"Training w{window}/h{horizon} fold {fold_index} seed {seed}",
                    flush=True,
                )
                seed_everything(seed)
                model = build_architecture(architecture, window, horizon)

                def report_epoch(
                    epoch: int, train_loss: float, validation_loss: float
                ) -> None:
                    _status(
                        job_dir,
                        current={
                            "window": window,
                            "horizon": horizon,
                            "fold": fold_index,
                            "seed": seed,
                            "epoch": epoch,
                            "epochs": evaluation["epochs"],
                            "train_loss": train_loss,
                            "validation_loss": validation_loss,
                        },
                    )
                    if epoch == 1 or epoch % 5 == 0:
                        print(
                            f"  epoch {epoch}: train={train_loss:.6g} "
                            f"validation={validation_loss:.6g}",
                            flush=True,
                        )

                fitted = fit_model(
                    model,
                    prepared.train.as_dataset(),
                    prepared.validation.as_dataset(),
                    seed=seed,
                    epochs=evaluation["epochs"],
                    early_stopping_patience=evaluation["early_stopping_patience"],
                    restore_best_weights=True,
                    epoch_callback=report_epoch,
                    **_fit_kwargs(evaluation, device),
                )
                prediction_scaled = predict(
                    model,
                    prepared.validation.as_dataset(),
                    device,
                    evaluation["evaluation_batch_size"],
                )
                prediction = prepared.scaler.inverse_target(prediction_scaled)
                target = prepared.scaler.inverse_target(prepared.validation.y)
                persistence = prepared.scaler.inverse_target(
                    np.repeat(prepared.validation.x[:, -1, 0:1], horizon, axis=1)
                )
                row, leads = _metrics_rows(
                    prediction,
                    target,
                    persistence,
                    window=window,
                    horizon=horizon,
                    seed=seed,
                    fold=fold_index,
                    split="validation",
                )
                row.update(
                    {
                        "parameters": parameter_count(model),
                        "train_seconds": fitted.train_seconds,
                        "epochs_ran": fitted.epochs_ran,
                        "best_validation_loss_scaled": fitted.best_validation_loss,
                        "train_samples": len(prepared.train.x),
                        "validation_samples": len(prepared.validation.x),
                    }
                )
                rows.append(row)
                lead_rows.extend(leads)
                _atomic_csv(job_dir / "runs.csv", rows)
                _atomic_csv(job_dir / "per_lead.csv", lead_rows)
                _status(job_dir, completed=len(rows), total=total)
                print(
                    f"  complete: MAE={row['mae']:.6g} MSE={row['mse']:.6g} "
                    f"MAE ratio={row['mae_ratio']:.4f}",
                    flush=True,
                )
    summary = _summary(rows)
    _atomic_json(job_dir / "summary.json", summary)
    _atomic_csv(job_dir / "summary.csv", summary)
    _status(job_dir, state="complete", completed=total, total=total, current=None)
    print("Validation complete", flush=True)


def _final_datasets(
    frame: pd.DataFrame, window: int, horizon: int
) -> tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray, MinMaxStats]:
    values = frame.to_numpy(dtype=np.float64)
    boundary = int(len(values) * 0.85)
    scaler = MinMaxStats.fit(values[:boundary])
    scaled = scaler.transform(values).astype(np.float32)
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    test_x: list[np.ndarray] = []
    test_y: list[np.ndarray] = []
    for target_start in range(window, len(values) - horizon + 1):
        target_stop = target_start + horizon
        x = scaled[target_start - window : target_start]
        y = scaled[target_start:target_stop, 0]
        if target_stop <= boundary:
            train_x.append(x)
            train_y.append(y)
        elif target_start >= boundary:
            test_x.append(x)
            test_y.append(y)
    if not train_x or not test_x:
        raise ValueError("final split produced no train or test samples")
    train_x_array, train_y_array = np.stack(train_x), np.stack(train_y)
    test_x_array, test_y_array = np.stack(test_x), np.stack(test_y)
    train = TensorDataset(
        torch.as_tensor(train_x_array), torch.as_tensor(train_y_array)
    )
    test = TensorDataset(torch.as_tensor(test_x_array), torch.as_tensor(test_y_array))
    return train, test, test_x_array, test_y_array, scaler


def run_final_test(job_dir: Path, request: dict[str, Any]) -> None:
    architecture = normalize_architecture_spec(request["architecture"])
    evaluation = normalize_evaluation(request.get("evaluation"))
    parent_dir = Path(request["parent_job_dir"])
    parent_runs = pd.read_csv(parent_dir / "runs.csv")
    frame = load_paper_data(evaluation["data_path"], evaluation["target_column"])
    device = resolve_device(evaluation["device"])
    total = len(evaluation["cells"]) * len(evaluation["seeds"])
    rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    _status(job_dir, state="running", completed=0, total=total, phase="final_test")
    print(
        f"Final test started: {architecture['name']} · {total} fitted runs",
        flush=True,
    )
    for cell in evaluation["cells"]:
        window, horizon = cell["window"], cell["horizon"]
        matching = parent_runs[
            (parent_runs["window"] == window) & (parent_runs["horizon"] == horizon)
        ]
        if matching.empty:
            raise ValueError(f"validation job has no epoch selection for w{window}/h{horizon}")
        selected_epochs = max(1, int(round(float(matching["epochs_ran"].median()))))
        train, test, test_x, test_y, scaler = _final_datasets(
            frame, window, horizon
        )
        for seed in evaluation["seeds"]:
            print(
                f"Refitting w{window}/h{horizon} seed {seed} for "
                f"{selected_epochs} epochs",
                flush=True,
            )
            seed_everything(seed)
            model = build_architecture(architecture, window, horizon)

            def report_epoch(
                epoch: int, train_loss: float, validation_loss: float
            ) -> None:
                _status(
                    job_dir,
                    current={
                        "window": window,
                        "horizon": horizon,
                        "fold": 0,
                        "seed": seed,
                        "epoch": epoch,
                        "epochs": selected_epochs,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    },
                )

            fitted = fit_model(
                model,
                train,
                train,
                seed=seed,
                epochs=selected_epochs,
                early_stopping_patience=None,
                restore_best_weights=False,
                epoch_callback=report_epoch,
                **_fit_kwargs(evaluation, device),
            )
            prediction_scaled = predict(
                model, test, device, evaluation["evaluation_batch_size"]
            )
            prediction = scaler.inverse_target(prediction_scaled)
            target = scaler.inverse_target(test_y)
            persistence = scaler.inverse_target(
                np.repeat(test_x[:, -1, 0:1], horizon, axis=1)
            )
            row, leads = _metrics_rows(
                prediction,
                target,
                persistence,
                window=window,
                horizon=horizon,
                seed=seed,
                fold=0,
                split="test",
            )
            row.update(
                {
                    "parameters": parameter_count(model),
                    "train_seconds": fitted.train_seconds,
                    "epochs_ran": selected_epochs,
                    "best_validation_loss_scaled": None,
                    "train_samples": len(train),
                    "validation_samples": 0,
                }
            )
            rows.append(row)
            lead_rows.extend(leads)
            _atomic_csv(job_dir / "runs.csv", rows)
            _atomic_csv(job_dir / "per_lead.csv", lead_rows)
            _status(job_dir, completed=len(rows), total=total)
            print(
                f"  complete: MAE={row['mae']:.6g} MSE={row['mse']:.6g}",
                flush=True,
            )
    summary = _summary(rows)
    _atomic_json(job_dir / "summary.json", summary)
    _atomic_csv(job_dir / "summary.csv", summary)
    _status(job_dir, state="complete", completed=total, total=total, current=None)
    print("Final test complete", flush=True)


def run_job(job_dir: Path) -> None:
    request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    started = datetime.now(UTC).isoformat()
    _status(job_dir, state="starting", started_at=started, error=None)
    try:
        if request.get("phase", "validation") == "final_test":
            run_final_test(job_dir, request)
        else:
            run_validation(job_dir, request)
    except BaseException as error:
        _status(job_dir, state="failed", error=str(error))
        traceback.print_exc()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    run_job(args.job_dir.resolve())


if __name__ == "__main__":
    main()
