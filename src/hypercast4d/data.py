"""Chronological, leakage-safe data preparation for the paper's spreadsheet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class MinMaxStats:
    minimum: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxStats":
        minimum = np.nanmin(values, axis=0)
        maximum = np.nanmax(values, axis=0)
        span = maximum - minimum
        span = np.where(span == 0, 1.0, span)
        return cls(minimum=minimum, span=span)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.minimum) / self.span

    def inverse_target(self, values: np.ndarray, target_index: int = 0) -> np.ndarray:
        return values * self.span[target_index] + self.minimum[target_index]


@dataclass(frozen=True)
class WindowSplit:
    x: np.ndarray
    y: np.ndarray
    target_start: np.ndarray

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(
            torch.as_tensor(self.x, dtype=torch.float32),
            torch.as_tensor(self.y, dtype=torch.float32),
        )


@dataclass(frozen=True)
class PreparedData:
    train: WindowSplit
    validation: WindowSplit
    test: WindowSplit
    scaler: MinMaxStats
    columns: tuple[str, ...]
    train_end: int
    validation_end: int


def load_paper_data(path: str | Path, target_column: str | None = None) -> pd.DataFrame:
    """Read the supplementary workbook and return four numeric series.

    The first usable sheet is selected. A date-like first column is retained as
    the index when present. Missing observations are rejected rather than
    silently imputed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `hypercast4d-download` first."
        )
    frame = pd.read_excel(path, sheet_name=0)
    if frame.empty:
        raise ValueError(f"No rows found in {path}")

    first = frame.iloc[:, 0]
    parsed = pd.to_datetime(first, errors="coerce")
    if parsed.notna().mean() > 0.9:
        frame = frame.iloc[:, 1:].copy()
        frame.index = parsed

    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] != 4:
        raise ValueError(
            f"Expected exactly four numeric series, found {numeric.shape[1]}: "
            f"{list(numeric.columns)}"
        )
    if numeric.isna().any().any():
        missing = int(numeric.isna().sum().sum())
        raise ValueError(f"Dataset contains {missing} missing numeric values")
    if target_column is not None:
        if target_column not in numeric.columns:
            raise ValueError(
                f"Target {target_column!r} is not in {list(numeric.columns)}"
            )
        ordered = [
            target_column,
            *[column for column in numeric if column != target_column],
        ]
        numeric = numeric.loc[:, ordered]
    numeric.columns = [str(column) for column in numeric.columns]
    return numeric.astype(np.float64)


def _empty(window: int, horizon: int, features: int) -> WindowSplit:
    return WindowSplit(
        x=np.empty((0, window, features), dtype=np.float32),
        y=np.empty((0, horizon), dtype=np.float32),
        target_start=np.empty((0,), dtype=np.int64),
    )


def prepare_windows(
    frame: pd.DataFrame,
    window: int,
    horizon: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PreparedData:
    """Scale on training rows and create non-overlapping target partitions.

    Validation and test inputs may use earlier historical rows, but every target
    in a window belongs wholly to one split. This prevents the shuffled,
    overlapping-window leakage present in the archived notebook.
    """
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below one")

    values = frame.to_numpy(dtype=np.float64)
    row_count, feature_count = values.shape
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end <= window or validation_end + horizon > row_count:
        raise ValueError("Not enough rows for the requested window, horizon and splits")

    scaler = MinMaxStats.fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for target_start in range(window, row_count - horizon + 1):
        target_stop = target_start + horizon
        sample = (
            scaled[target_start - window : target_start],
            scaled[target_start:target_stop, 0],
            target_start,
        )
        if target_stop <= train_end:
            buckets["train"].append(sample)
        elif target_start >= train_end and target_stop <= validation_end:
            buckets["validation"].append(sample)
        elif target_start >= validation_end:
            buckets["test"].append(sample)

    def stack(name: str) -> WindowSplit:
        rows = buckets[name]
        if not rows:
            return _empty(window, horizon, feature_count)
        return WindowSplit(
            x=np.stack([row[0] for row in rows]),
            y=np.stack([row[1] for row in rows]),
            target_start=np.asarray([row[2] for row in rows], dtype=np.int64),
        )

    prepared = PreparedData(
        train=stack("train"),
        validation=stack("validation"),
        test=stack("test"),
        scaler=scaler,
        columns=tuple(frame.columns),
        train_end=train_end,
        validation_end=validation_end,
    )
    if (
        min(len(prepared.train.x), len(prepared.validation.x), len(prepared.test.x))
        == 0
    ):
        raise ValueError("At least one chronological split produced no samples")
    return prepared
