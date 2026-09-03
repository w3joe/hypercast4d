import json
from pathlib import Path

import numpy as np
import pandas as pd

from hypercast4d.architecture import presets
from hypercast4d.modal_runner import _execute_payload


def test_modal_payload_produces_standard_playground_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    values = 10 + np.cumsum(rng.normal(0.01, 0.05, size=(140, 4)), axis=0)
    frame = pd.DataFrame(values, columns=["Copper", "FCX", "CLP", "SCCO"])
    frame.insert(0, "Date", pd.date_range("2020-01-01", periods=len(frame)))
    workbook = tmp_path / "paper_data.xlsx"
    frame.to_excel(workbook, index=False)
    architecture = next(
        item for item in presets() if item["preset_id"] == "paper-quaternion"
    )
    request = {
        "phase": "validation",
        "architecture": architecture,
        "evaluation": {
            "preset": "quick",
            "data_path": "data/raw/paper_data.xlsx",
            "epochs": 1,
        },
    }

    result = _execute_payload(
        json.dumps(request), workbook.read_bytes(), None, require_cuda=False
    )

    assert result["ok"] is True
    assert result["status"]["state"] == "complete"
    assert set(result["files"]) == {
        "runs.csv",
        "per_lead.csv",
        "summary.json",
        "summary.csv",
    }
    assert "Validation complete" in result["log"]
