import numpy as np
import pandas as pd

from hypercast4d.data import prepare_windows


def frame(rows: int = 100) -> pd.DataFrame:
    values = np.arange(rows, dtype=np.float64)
    return pd.DataFrame({f"series_{index}": values + index for index in range(4)})


def test_scaler_is_fit_only_on_training_rows() -> None:
    data = frame()
    prepared = prepare_windows(
        data, window=5, horizon=3, train_fraction=0.6, validation_fraction=0.2
    )
    assert prepared.scaler.minimum[0] == 0
    assert prepared.scaler.span[0] == 59
    assert prepared.scaler.transform(data.to_numpy())[-1, 0] > 1.0


def test_target_ranges_do_not_overlap_boundaries() -> None:
    prepared = prepare_windows(
        frame(), window=5, horizon=3, train_fraction=0.6, validation_fraction=0.2
    )
    assert np.all(prepared.train.target_start + 3 <= prepared.train_end)
    assert np.all(prepared.validation.target_start >= prepared.train_end)
    assert np.all(prepared.validation.target_start + 3 <= prepared.validation_end)
    assert np.all(prepared.test.target_start >= prepared.validation_end)


def test_chronological_order_is_preserved() -> None:
    prepared = prepare_windows(
        frame(), window=5, horizon=3, train_fraction=0.6, validation_fraction=0.2
    )
    for split in (prepared.train, prepared.validation, prepared.test):
        assert np.all(np.diff(split.target_start) == 1)
