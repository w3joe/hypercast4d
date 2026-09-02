import json
from pathlib import Path

from hypercast4d.dashboard import DASHBOARD_HTML, handler_for, load_runs, load_status


def test_load_runs_coerces_numeric_fields(tmp_path: Path) -> None:
    (tmp_path / "runs.csv").write_text(
        "window,horizon,seed,model,mae,mse,parameters,train_seconds\n"
        "10,1,7,hyper_quaternion,0.125,0.02,225,1.5\n",
        encoding="utf-8",
    )
    rows = load_runs(tmp_path)
    assert rows == [
        {
            "window": 10,
            "horizon": 1,
            "seed": 7,
            "model": "hyper_quaternion",
            "mae": 0.125,
            "mse": 0.02,
            "parameters": 225,
            "train_seconds": 1.5,
            "process_peak_rss_mb": None,
            "epochs_ran": None,
            "best_validation_mse_scaled": None,
            "train_samples": None,
            "validation_samples": None,
            "test_samples": None,
        }
    ]


def test_status_falls_back_to_existing_results(tmp_path: Path) -> None:
    assert load_status(tmp_path, 0)["state"] == "waiting"
    assert load_status(tmp_path, 4) == {
        "state": "complete",
        "completed": 4,
        "total": 4,
    }


def test_status_file_takes_precedence(tmp_path: Path) -> None:
    status = {"state": "running", "completed": 2, "total": 7}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert load_status(tmp_path, 99) == status


def test_dashboard_refresh_rate_is_injected(tmp_path: Path) -> None:
    handler = handler_for(tmp_path, refresh_seconds=2.5)
    closure_values = [cell.cell_contents for cell in handler.do_GET.__closure__ or ()]
    page = next(value for value in closure_values if isinstance(value, str))
    assert "setInterval(poll,2500)" in page
    assert "updates every 2.5 seconds" in page
    assert "__REFRESH_" not in page
    assert "__REFRESH_" in DASHBOARD_HTML
