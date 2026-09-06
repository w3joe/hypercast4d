import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypercast4d.diagnostics import (
    append_predictions,
    forecast_diagnostics,
    initialize_diagnostics,
    training_references,
    safe_ratio,
)
from hypercast4d.playground_runner import _metrics_rows, run_job
from hypercast4d.architecture import presets


def test_known_forecasts_and_large_move_subset():
    actual = np.array([[110.0], [90.0], [120.0], [100.0]])
    prediction = np.array([[105.0], [95.0], [110.0], [101.0]])
    aggregate, (row,) = forecast_diagnostics(
        prediction,
        actual,
        np.full(4, 100.0),
        [{"large_move_threshold": 0.15, "direction_baseline": 1}],
    )
    assert row["directional_accuracy"] == 0.75
    assert row["direction_baseline_accuracy"] == 0.5
    assert row["bias"] == -2.25
    assert row["p95_abs_error"] == pytest.approx(9.25)
    assert row["large_move_count"] == 1
    assert row["large_move_mae"] == 10
    assert row["persistence_large_move_mae"] == 20
    assert row["large_move_mae_ratio"] == 0.5
    assert row["return_correlation"] == pytest.approx(
        np.corrcoef([0.1, -0.1, 0.2, 0], [0.05, -0.05, 0.1, 0.01])[0, 1]
    )
    assert aggregate["sample_count"] == 4


def test_constant_predictions_empty_tail_and_zero_origin_are_undefined():
    refs = [{"large_move_threshold": 0.5, "direction_baseline": 0}]
    summary, (row,) = forecast_diagnostics(
        np.full((3, 1), 100.0),
        np.array([[101.0], [99.0], [100.0]]),
        np.full(3, 100.0),
        refs,
    )
    assert row["return_correlation"] is None
    assert row["large_move_mae"] is None
    assert row["large_move_mae_ratio"] is None
    assert row["large_move_count"] == 0
    assert row["directional_accuracy"] == pytest.approx(1 / 3)
    json.dumps(summary, allow_nan=False)
    _, (row,) = forecast_diagnostics(
        np.ones((3, 1)), np.ones((3, 1)), np.zeros(3), refs
    )
    assert row["return_sample_count"] == 0
    assert row["directional_accuracy"] is None
    assert row["return_correlation"] is None
    assert safe_ratio(0, 0) is None


def test_training_references_are_horizon_specific_and_float_noise_is_flat():
    train = np.array([100.0, 110.0, 99.0, 108.9])
    refs = training_references(train, 2)
    assert refs[0]["large_move_threshold"] == pytest.approx(0.1)
    assert refs[0]["direction_baseline"] == 1
    assert refs[1]["large_move_threshold"] == pytest.approx(0.01)
    assert refs[1]["direction_baseline"] == -1
    _, (row,) = forecast_diagnostics(
        np.array([[100.000001]]), np.array([[100.0]]), np.array([100.0]), refs[:1]
    )
    assert row["directional_accuracy"] == 1
    assert row["return_correlation"] is None


def test_export_aligns_origins_and_targets_and_appends_seeds(tmp_path):
    frame = pd.DataFrame(
        {"Copper": [100.0, 110.0, 120.0, 130.0, 140.0]},
        index=pd.date_range("2020-01-01", periods=5),
    )
    target = np.array([[120.0, 130.0], [130.0, 140.0]])
    refs = training_references(frame.Copper.to_numpy()[:3], 2)
    initialize_diagnostics(tmp_path)
    for seed in [7, 19]:
        append_predictions(
            tmp_path,
            target,
            target,
            frame,
            np.array([2, 3]),
            refs,
            window=2,
            horizon=2,
            seed=seed,
            fold=1,
            split="validation",
        )
    saved = pd.read_csv(tmp_path / "predictions.csv")
    assert len(saved) == 8
    assert saved.origin_row.tolist() == [1, 1, 2, 2] * 2
    assert saved.lead.tolist() == [1, 2, 1, 2] * 2
    assert saved.target_date.iloc[1] == "2020-01-04"
    assert saved.predicted_return.iloc[1] == pytest.approx(130 / 110 - 1)
    assert saved.persistence.iloc[1] == 110


def test_constant_series_metrics_are_json_safe():
    x = np.ones((4, 2))
    row, leads = _metrics_rows(
        x,
        x,
        x,
        window=2,
        horizon=2,
        seed=7,
        fold=1,
        split="validation",
        references=training_references(np.ones(10), 2),
    )
    assert row["mae_ratio"] is None
    assert row["mse_ratio"] is None
    assert row["return_correlation"] is None
    json.dumps([row, leads], allow_nan=False)


def test_validation_and_final_test_save_only_their_partition(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        10 + np.cumsum(rng.normal(0, 0.1, (140, 4)), axis=0),
        columns=["Copper", "FCX", "CLP", "SCCO"],
    )
    frame.insert(0, "Date", pd.date_range("2020-01-01", periods=140))
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    frame.to_excel(data / "paper_data.xlsx", index=False)
    architecture = next(p for p in presets() if p["preset_id"] == "paper-quaternion")
    evaluation = {
        "preset": "standard",
        "epochs": 1,
        "seeds": [7],
        "cells": [{"window": 10, "horizon": 5}],
    }
    parent = tmp_path / "validation"
    parent.mkdir()
    (parent / "request.json").write_text(
        json.dumps(
            {
                "phase": "validation",
                "architecture": architecture,
                "evaluation": evaluation,
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    run_job(parent)
    final = tmp_path / "final"
    final.mkdir()
    (final / "request.json").write_text(
        json.dumps(
            {
                "phase": "final_test",
                "architecture": architecture,
                "evaluation": evaluation,
                "parent_job_dir": str(parent),
            }
        )
    )
    run_job(final)
    for path, boundary, stop, split in [
        (parent, 98, 119, "validation"),
        (final, 119, 140, "test"),
    ]:
        predictions = pd.read_csv(path / "predictions.csv")
        leads = pd.read_csv(path / "per_lead.csv")
        assert set(predictions.split) == {split}
        assert predictions.origin_row.min() == boundary - 1
        assert (predictions.origin_row + predictions.lead).max() == stop - 1
        expected = training_references(frame.Copper.to_numpy()[:boundary], 5)
        assert leads.large_move_threshold.tolist() == pytest.approx(
            [r["large_move_threshold"] for r in expected]
        )
        for row in leads.itertuples():
            sample = predictions[predictions.lead == row.lead]
            assert row.bias == pytest.approx((sample.prediction - sample.actual).mean())
            assert (
                row.large_move_count
                == (sample.actual_return.abs() > row.large_move_threshold).sum()
            )
        assert json.loads((path / "status.json").read_text())["state"] == "complete"


def test_forecast_download_and_missing_artifact(tmp_path):
    from fastapi.testclient import TestClient
    from hypercast4d.playground import create_app

    root = tmp_path / "playground"
    path = root / "jobs" / "example"
    path.mkdir(parents=True)
    (path / "status.json").write_text('{"state":"complete"}')
    (path / "request.json").write_text("{}")
    app = create_app(root, tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/v1/jobs/example/predictions.csv").status_code == 404
        (path / "predictions.csv").write_text("lead,actual,prediction\n1,10,11\n")
        response = client.get("/api/v1/jobs/example/predictions.csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert response.text.endswith("1,10,11\n")
        assert client.get("/api/v1/jobs/missing/predictions.csv").status_code == 404


def test_main_runner_emits_diagnostics_and_forecasts(tmp_path):
    import yaml
    from hypercast4d.experiment import run

    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        10 + np.cumsum(rng.normal(0, 0.1, (140, 4)), axis=0),
        columns=["Copper", "FCX", "CLP", "SCCO"],
    )
    frame.insert(0, "Date", pd.date_range("2020-01-01", periods=140))
    workbook = tmp_path / "data.xlsx"
    frame.to_excel(workbook, index=False)
    config = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    config["data"]["path"] = str(workbook)
    config["experiment"].update(
        output_dir=str(tmp_path / "main"),
        cells=[{"window": 10, "horizon": 5}],
        seeds=[7],
    )
    config["models"]["enabled"] = ["persistence"]
    config["tensorboard"]["enabled"] = False
    runs, summary = run(config)
    assert len(runs) == 1
    assert "directional_accuracy" in summary
    assert runs.iloc[0].return_correlation is None
    saved = pd.read_csv(tmp_path / "main/runs.csv")
    assert "p95_abs_error" in saved
    assert len(pd.read_csv(tmp_path / "main/per_lead.csv")) == 5
    forecasts = pd.read_csv(tmp_path / "main/predictions.csv")
    assert set(forecasts.model) == {"persistence"}
    assert forecasts.origin_row.min() == 118
