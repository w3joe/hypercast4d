"""Point-forecast diagnostics in original units, with training-only references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RETURN_TOLERANCE = 1e-6
LARGE_MOVE_QUANTILE = 0.90
DIAGNOSTIC_COLUMNS = (
    "directional_accuracy",
    "direction_baseline_accuracy",
    "return_correlation",
    "bias",
    "p95_abs_error",
    "large_move_mae",
    "persistence_large_move_mae",
    "large_move_mae_ratio",
    "large_move_count",
    "sample_count",
    "return_sample_count",
)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def returns(values: np.ndarray, origin: np.ndarray) -> np.ndarray:
    result = np.full(np.shape(values), np.nan, dtype=np.float64)
    np.divide(values - origin, origin, out=result, where=origin > 0)
    return result


def direction(values: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values) <= RETURN_TOLERANCE, 0, np.sign(values))


def training_references(
    training_target: np.ndarray, horizon: int
) -> list[dict[str, Any]]:
    """Each h-step reference uses pairs wholly contained in training rows."""
    y = np.asarray(training_target, dtype=np.float64)
    references = []
    for lead in range(1, horizon + 1):
        changes = returns(y[lead:], y[:-lead])
        changes = changes[np.isfinite(changes)]
        if not len(changes):
            references.append(
                {"large_move_threshold": None, "direction_baseline": None}
            )
            continue
        labels, counts = np.unique(direction(changes), return_counts=True)
        references.append(
            {
                "large_move_threshold": float(
                    np.quantile(np.abs(changes), LARGE_MOVE_QUANTILE)
                ),
                "direction_baseline": int(labels[np.argmax(counts)]),
            }
        )
    return references


def forecast_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
    references: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(
        target, dtype=np.float64
    )
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or len(origin) != len(target)
    ):
        raise ValueError(
            "predictions and targets must be [origin, lead], with one origin price per row"
        )
    if len(references) != target.shape[1]:
        raise ValueError("one training reference is required per lead")
    if not all(np.isfinite(x).all() for x in (prediction, target, origin)):
        raise ValueError("forecast diagnostics require finite prices")
    rows = []
    for j, reference in enumerate(references):
        actual_return = returns(target[:, j], origin)
        predicted_return = returns(prediction[:, j], origin)
        valid = np.isfinite(actual_return) & np.isfinite(predicted_return)
        actual, predicted = actual_return[valid], predicted_return[valid]
        residual = prediction[:, j] - target[:, j]
        threshold = reference["large_move_threshold"]
        large = (
            valid & (np.abs(actual_return) > threshold)
            if threshold is not None
            else np.zeros(len(target), dtype=bool)
        )
        large_mae = float(np.mean(np.abs(residual[large]))) if large.any() else None
        baseline_mae = (
            float(np.mean(np.abs(target[large, j] - origin[large])))
            if large.any()
            else None
        )
        correlation = None
        if (
            len(actual) >= 2
            and np.ptp(actual) > RETURN_TOLERANCE
            and np.ptp(predicted) > RETURN_TOLERANCE
        ):
            correlation = float(np.clip(np.corrcoef(actual, predicted)[0, 1], -1, 1))
        rows.append(
            {
                "lead": j + 1,
                **reference,
                "sample_count": len(target),
                "return_sample_count": int(valid.sum()),
                "directional_accuracy": (
                    float(np.mean(direction(actual) == direction(predicted)))
                    if len(actual)
                    else None
                ),
                "direction_baseline_accuracy": (
                    float(np.mean(direction(actual) == reference["direction_baseline"]))
                    if len(actual) and reference["direction_baseline"] is not None
                    else None
                ),
                "return_correlation": correlation,
                "bias": float(np.mean(residual)),
                "p95_abs_error": float(np.quantile(np.abs(residual), 0.95)),
                "large_move_count": int(large.sum()),
                "large_move_mae": large_mae,
                "persistence_large_move_mae": baseline_mae,
                "large_move_mae_ratio": (
                    safe_ratio(large_mae, baseline_mae)
                    if large_mae is not None
                    else None
                ),
            }
        )
    # Preserve lead-specific interpretation: correlations and tail percentiles
    # are NOT recomputed from a mixture of different forecast horizons.
    aggregate = {}
    for key in DIAGNOSTIC_COLUMNS:
        values = [row[key] for row in rows if row[key] is not None]
        aggregate[key] = (
            (int(sum(values)) if key.endswith("count") else float(np.mean(values)))
            if values
            else None
        )
    return aggregate, rows


def initialize_diagnostics(destination: Path) -> None:
    (destination / "predictions.csv").write_text("", encoding="utf-8")
    (destination / "diagnostics.json").write_text(
        json.dumps(
            {
                "version": 1,
                "return_definition": "(forecast or actual - origin_price) / origin_price; origin_price > 0",
                "direction_flat_tolerance": RETURN_TOLERANCE,
                "correlation_minimum_return_spread": RETURN_TOLERANCE,
                "direction_baseline": "most frequent training h-step direction; ties choose smallest label (-1, 0, 1)",
                "large_move_quantile": LARGE_MOVE_QUANTILE,
                "large_move_rule": "absolute actual h-step return strictly exceeds training quantile",
                "aggregation": "run diagnostics average defined per-lead scores; counts sum origin/lead pairs; summaries average runs",
                "undefined": "null in JSON and empty in CSV; never zero-filled",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def append_predictions(
    destination: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    target_start: np.ndarray,
    references: list[dict[str, Any]],
    **identity: Any,
) -> None:
    starts = np.asarray(target_start)
    count, horizon = target.shape
    origins = frame.iloc[starts - 1, 0].to_numpy(dtype=np.float64)
    origin_prices = np.repeat(origins, horizon)
    actual = target.reshape(-1)
    forecast = prediction.reshape(-1)
    thresholds = np.tile([r["large_move_threshold"] for r in references], count)
    data = pd.DataFrame(
        {
            **identity,
            "forecast_origin": np.repeat(frame.index[starts - 1].astype(str), horizon),
            "origin_row": np.repeat(starts - 1, horizon),
            "target_date": frame.index[
                (starts[:, None] + np.arange(horizon)).reshape(-1)
            ].astype(str),
            "lead": np.tile(np.arange(1, horizon + 1), count),
            "origin_price": origin_prices,
            "actual": actual,
            "prediction": forecast,
            "persistence": origin_prices,
            "actual_return": returns(actual, origin_prices),
            "predicted_return": returns(forecast, origin_prices),
            "large_move_threshold": thresholds,
            "direction_baseline": np.tile(
                [r["direction_baseline"] for r in references], count
            ),
        }
    )
    path = destination / "predictions.csv"
    data.to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
    )
