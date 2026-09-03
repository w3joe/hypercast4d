"""Reproduce the paper's supplementary GridSearchCV protocol in PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .data import load_paper_data
from .models import build_model, parameter_count
from .training import error_metrics, fit_model, predict, seed_everything


MODEL_TYPES = ("cnn", "lstm", "hyper")
FOLD_COLUMNS = (
    "input_order",
    "window",
    "horizon",
    "model",
    "candidate",
    "fold",
    "mae_scaled",
    "mse_scaled",
    "parameters",
    "train_seconds",
    "declared_algebra",
    "effective_algebra",
    "settings_json",
)
CANDIDATE_COLUMNS = (
    "input_order",
    "window",
    "horizon",
    "model",
    "candidate",
    "mean_test_score",
    "std_test_score",
    "mae_mean_scaled",
    "mae_std_scaled",
    "parameters",
    "train_seconds_mean",
    "declared_algebra",
    "effective_algebra",
    "settings_json",
)


@dataclass(frozen=True)
class PaperWindows:
    features: np.ndarray
    targets: np.ndarray
    columns: tuple[str, ...]
    target_column: str


def _atomic_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _initialize_csv(path: Path, columns: tuple[str, ...]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()
    temporary.replace(path)


def _append_csv(path: Path, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writerow(row)


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def prepare_paper_windows(
    frame: pd.DataFrame,
    *,
    input_columns: list[str],
    target_column: str,
    window: int,
    horizon: int,
    feature_range: tuple[float, float],
) -> PaperWindows:
    """Match the notebook: global min-max scaling, then overlapping windows."""
    if len(input_columns) != 4 or len(set(input_columns)) != 4:
        raise ValueError("Each input order must contain four unique columns")
    if set(input_columns) != set(frame.columns):
        raise ValueError(
            "Each input order must contain every dataset column exactly once"
        )
    if target_column not in input_columns:
        raise ValueError("target_column must be present in the input order")
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive")

    ordered = frame.loc[:, input_columns]
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(ordered)
    target_index = input_columns.index(target_column)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for target_start in range(window, len(scaled) - horizon + 1):
        features.append(scaled[target_start - window : target_start])
        targets.append(scaled[target_start : target_start + horizon, target_index])
    return PaperWindows(
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        columns=tuple(input_columns),
        target_column=target_column,
    )


def candidate_grids(search_spaces: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if set(search_spaces) != set(MODEL_TYPES):
        raise ValueError(f"search_spaces must contain exactly {MODEL_TYPES}")

    # GridSearchCV's ParameterGrid sorts parameter names. Generate candidates
    # under the notebook's original names so candidate indices and traversal
    # order also match its saved cv_results_ workbooks.
    aliases = {
        "cnn": {
            "units": "n_filters",
            "dense_before_pool": "n_dense1",
            "dense_after_pool": "n_dense2",
            "dense_units": "n_units",
            "activation": "activation",
        },
        "lstm": {
            "units": "n_unitsLSTM",
            "dense_before_pool": "n_dense1",
            "dense_after_pool": "n_dense2",
            "dense_units": "n_units",
            "activation": "activation",
        },
        "hyper": {
            "units": "n_hunits",
            "dense_before_pool": "n_dense1",
            "dense_after_pool": "n_dense2",
            "dense_units": "n_units",
            "activation": "activation",
            "declared_algebra": "algebra",
        },
    }
    grids: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_TYPES:
        mapping = aliases[model]
        if set(search_spaces[model]) != set(mapping):
            raise ValueError(
                f"search_spaces.{model} must contain exactly {tuple(mapping)}"
            )
        original_names = {
            mapping[configured_name]: values
            for configured_name, values in search_spaces[model].items()
        }
        reverse = {original: configured for configured, original in mapping.items()}
        grids[model] = [
            {reverse[name]: value for name, value in candidate.items()}
            for candidate in ParameterGrid(original_names)
        ]
    return grids


def _effective_model_name(
    model_type: str,
    candidate: dict[str, Any],
    algebra_behavior: str,
    forced_algebra: str,
) -> tuple[str, str | None, str | None]:
    if model_type != "hyper":
        return model_type, None, None
    declared = str(candidate["declared_algebra"])
    if algebra_behavior == "force_quaternion":
        effective = forced_algebra
    elif algebra_behavior == "declared":
        effective = declared
    else:
        raise ValueError(
            "hyper_algebra_behavior must be 'force_quaternion' or 'declared'"
        )
    return f"hyper_{effective}", declared, effective


def _build_candidate(
    model_type: str,
    candidate: dict[str, Any],
    architecture: dict[str, Any],
    algebra_behavior: str,
    forced_algebra: str,
    window: int,
    horizon: int,
) -> tuple[torch.nn.Module, str | None, str | None]:
    model_name, declared, effective = _effective_model_name(
        model_type, candidate, algebra_behavior, forced_algebra
    )
    model = build_model(
        model_name,
        window,
        horizon,
        first_layer_units=int(candidate["units"]),
        conv_kernel_size=(
            int(architecture["convolution"]["kernel_size"])
            if model_type == "cnn"
            else None
        ),
        dense_before_pool=bool(candidate["dense_before_pool"]),
        dense_after_pool=bool(candidate["dense_after_pool"]),
        dense_units=int(candidate["dense_units"]),
        activation=str(candidate["activation"]),
        dropout=float(architecture["dropout"]),
        pool_size=int(architecture["pooling"]["size"]),
    )
    return model, declared, effective


def _dataset(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))


def _status(
    path: Path,
    *,
    state: str,
    completed: int,
    total: int,
    started_at: str,
) -> None:
    payload = {
        "state": state,
        "completed_fits": completed,
        "total_fits": total,
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_text(path, json.dumps(payload, indent=2) + "\n")


def run_reproduction(
    config: dict[str, Any], quick: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_config = config["data"]
    cv_config = config["cross_validation"]
    training = config["training"]
    optimizer = training["optimizer"]
    execution = config["execution"]
    tensorboard = execution["tensorboard"]
    architecture = config["architecture"]

    if config["protocol"]["metric"] != "mae":
        raise ValueError("The supplementary notebook selects candidates by MAE")
    if config["protocol"]["metric_scale"] != "normalized":
        raise ValueError("The supplementary notebook scores normalized targets")
    scaler_config = data_config["scaler"]
    if (
        scaler_config["name"] != "minmax"
        or scaler_config["fit_scope"] != "complete_dataset"
    ):
        raise ValueError("Paper reproduction requires global MinMax scaling")
    if optimizer["name"] != "adam" or training["loss"] != "mse":
        raise ValueError("The supplementary notebook uses Adam with MSE loss")

    frame = load_paper_data(data_config["path"])
    input_orders = list(data_config["input_orders"])
    cells = [
        (int(window), int(horizon))
        for window in data_config["windows"]
        for horizon in data_config["horizons"]
    ]
    grids = candidate_grids(config["search_spaces"])
    folds = int(cv_config["folds"])
    epochs = int(training["epochs"])
    if quick:
        quick_config = execution["quick"]
        input_orders = input_orders[: int(quick_config["order_count"])]
        cells = cells[: int(quick_config["cell_count"])]
        folds = int(quick_config["folds"])
        epochs = int(quick_config["epochs"])
        candidate_limit = int(quick_config["candidates_per_model"])
        grids = {
            model: candidates[:candidate_limit] for model, candidates in grids.items()
        }
    if not input_orders or not cells or folds < 2 or epochs < 1:
        raise ValueError(
            "Orders/cells must be non-empty, folds >= 2, and epochs positive"
        )
    if any(not candidates for candidates in grids.values()):
        raise ValueError("Every model search space must contain at least one candidate")

    output = Path(execution["output_dir"])
    if quick:
        output = output.with_name(f"{output.name}_quick")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False)
    )
    fold_path = output / "folds.csv"
    candidate_path = output / "candidates.csv"
    status_path = output / "status.json"
    started = datetime.now(UTC)
    started_at = started.isoformat()
    experiment_id = started.strftime("%Y%m%dT%H%M%S_%fZ")
    if quick:
        experiment_id = f"{experiment_id}_quick"
    tensorboard_root = (
        Path(tensorboard["log_dir"]) / experiment_id
        if bool(tensorboard["enabled"])
        else None
    )
    total_fits = (
        len(input_orders)
        * len(cells)
        * folds
        * sum(len(candidates) for candidates in grids.values())
    )
    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    _initialize_csv(fold_path, FOLD_COLUMNS)
    _initialize_csv(candidate_path, CANDIDATE_COLUMNS)
    _status(
        status_path,
        state="running",
        completed=0,
        total=total_fits,
        started_at=started_at,
    )

    device = _device(str(training["device"]))
    random_seed_value = training["random_seed"]
    random_seed = None if random_seed_value is None else int(random_seed_value)
    feature_range = tuple(float(value) for value in scaler_config["feature_range"])
    if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
        raise ValueError(
            "scaler.feature_range must contain increasing lower/upper values"
        )

    for order in input_orders:
        order_name = str(order["name"])
        input_columns = [str(column) for column in order["columns"]]
        target_column = str(order["target_column"])
        for window, horizon in cells:
            prepared = prepare_paper_windows(
                frame,
                input_columns=input_columns,
                target_column=target_column,
                window=window,
                horizon=horizon,
                feature_range=feature_range,
            )
            split = train_test_split(
                prepared.features,
                prepared.targets,
                test_size=float(data_config["holdout"]["fraction"]),
                shuffle=bool(data_config["holdout"]["shuffle"]),
                random_state=data_config["holdout"]["random_state"],
            )
            train_features, holdout_features, train_targets, holdout_targets = split
            holdout_dataset = _dataset(holdout_features, holdout_targets)
            kfold = KFold(
                n_splits=folds,
                shuffle=bool(cv_config["shuffle"]),
                random_state=cv_config["random_state"],
            )
            fold_indices = list(kfold.split(train_features))

            for model_type in MODEL_TYPES:
                for candidate_index, candidate in enumerate(grids[model_type]):
                    settings_json = json.dumps(candidate, sort_keys=True)
                    candidate_fold_rows: list[dict[str, Any]] = []
                    for fold_index, (fit_indices, score_indices) in enumerate(
                        fold_indices, start=1
                    ):
                        seed_everything(random_seed)
                        model, declared, effective = _build_candidate(
                            model_type,
                            candidate,
                            architecture,
                            str(config["hyper_algebra_behavior"]),
                            str(config["forced_hyper_algebra"]),
                            window,
                            horizon,
                        )
                        parameters = parameter_count(model)
                        writer = None
                        if tensorboard_root is not None:
                            log_dir = (
                                tensorboard_root
                                / order_name
                                / f"w{window}_h{horizon}"
                                / model_type
                                / f"candidate_{candidate_index}"
                                / f"fold_{fold_index}"
                            )
                            writer = SummaryWriter(
                                log_dir=str(log_dir),
                                flush_secs=int(tensorboard["flush_seconds"]),
                            )

                        def log_epoch(
                            epoch: int, train_loss: float, validation_loss: float
                        ) -> None:
                            if writer is not None:
                                writer.add_scalar(
                                    "loss/train_scaled", train_loss, epoch
                                )
                                writer.add_scalar(
                                    "loss/validation_holdout_scaled",
                                    validation_loss,
                                    epoch,
                                )

                        fitted = fit_model(
                            model,
                            _dataset(
                                train_features[fit_indices], train_targets[fit_indices]
                            ),
                            holdout_dataset,
                            seed=random_seed,
                            epochs=epochs,
                            batch_size=int(training["batch_size"]),
                            learning_rate=float(optimizer["learning_rate"]),
                            adam_beta1=float(optimizer["beta1"]),
                            adam_beta2=float(optimizer["beta2"]),
                            adam_epsilon=float(optimizer["epsilon"]),
                            adam_amsgrad=bool(optimizer["amsgrad"]),
                            loss_name=str(training["loss"]),
                            shuffle=bool(training["shuffle"]),
                            early_stopping_patience=training["early_stopping_patience"],
                            early_stopping_min_delta=float(
                                training["early_stopping_min_delta"]
                            ),
                            restore_best_weights=bool(training["restore_best_weights"]),
                            device=device,
                            epoch_callback=log_epoch,
                        )
                        score_dataset = _dataset(
                            train_features[score_indices], train_targets[score_indices]
                        )
                        prediction = predict(
                            model, score_dataset, device, int(training["batch_size"])
                        )
                        metrics = error_metrics(
                            prediction, train_targets[score_indices]
                        )
                        if writer is not None:
                            writer.add_scalar(
                                "metrics/fold_mae_scaled", metrics["mae"], epochs
                            )
                            writer.add_scalar(
                                "metrics/fold_mse_scaled", metrics["mse"], epochs
                            )
                            writer.add_text("candidate/settings", settings_json, 0)
                            writer.close()
                        fold_row = {
                            "input_order": order_name,
                            "window": window,
                            "horizon": horizon,
                            "model": model_type,
                            "candidate": candidate_index,
                            "fold": fold_index,
                            "mae_scaled": metrics["mae"],
                            "mse_scaled": metrics["mse"],
                            "parameters": parameters,
                            "train_seconds": fitted.train_seconds,
                            "declared_algebra": declared,
                            "effective_algebra": effective,
                            "settings_json": settings_json,
                        }
                        fold_rows.append(fold_row)
                        candidate_fold_rows.append(fold_row)
                        _append_csv(fold_path, fold_row, FOLD_COLUMNS)
                        _status(
                            status_path,
                            state="running",
                            completed=len(fold_rows),
                            total=total_fits,
                            started_at=started_at,
                        )

                    fold_maes = np.asarray(
                        [row["mae_scaled"] for row in candidate_fold_rows]
                    )
                    candidate_row = {
                        "input_order": order_name,
                        "window": window,
                        "horizon": horizon,
                        "model": model_type,
                        "candidate": candidate_index,
                        "mean_test_score": -float(fold_maes.mean()),
                        "std_test_score": float(fold_maes.std(ddof=0)),
                        "mae_mean_scaled": float(fold_maes.mean()),
                        "mae_std_scaled": float(fold_maes.std(ddof=0)),
                        "parameters": candidate_fold_rows[0]["parameters"],
                        "train_seconds_mean": float(
                            np.mean(
                                [row["train_seconds"] for row in candidate_fold_rows]
                            )
                        ),
                        "declared_algebra": candidate_fold_rows[0]["declared_algebra"],
                        "effective_algebra": candidate_fold_rows[0][
                            "effective_algebra"
                        ],
                        "settings_json": settings_json,
                    }
                    candidate_rows.append(candidate_row)
                    _append_csv(candidate_path, candidate_row, CANDIDATE_COLUMNS)
                    print(
                        f"{order_name:12s} w={window:2d} h={horizon:2d} "
                        f"{model_type:5s} candidate={candidate_index:3d} "
                        f"MAE={candidate_row['mae_mean_scaled']:.6f}"
                    )

    candidates_frame = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    best = (
        candidates_frame.sort_values("mae_mean_scaled")
        .groupby(["input_order", "window", "horizon", "model"], as_index=False)
        .first()
        .sort_values(["input_order", "window", "horizon", "model"])
    )
    best.to_csv(output / "best.csv", index=False)
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    metadata = {
        "protocol": config["protocol"],
        "quick": quick,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sklearn": __import__("sklearn").__version__,
        "device": str(device),
        "elapsed_seconds": time.time() - started.timestamp(),
        "tensorboard_run_dir": (
            str(tensorboard_root) if tensorboard_root is not None else None
        ),
        "config": config,
    }
    _atomic_text(output / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    _status(
        status_path,
        state="complete",
        completed=len(fold_rows),
        total=total_fits,
        started_at=started_at,
    )
    return folds_frame, candidates_frame, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/paper_reproduction.yaml")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the configured bounded reproduction subset",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_reproduction(config, args.quick)


if __name__ == "__main__":
    main()
