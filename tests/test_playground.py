import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from hypercast4d.architecture import presets
from hypercast4d.playground import JobManager, create_app
from hypercast4d.playground_runner import normalize_evaluation, run_job
from hypercast4d.upstream_models import UPSTREAM_MODELS


def _preset(preset_id: str) -> dict:
    return next(item for item in presets() if item["preset_id"] == preset_id)


def test_evaluation_presets_normalize_with_safe_defaults() -> None:
    quick = normalize_evaluation({"preset": "quick"})
    assert quick["cells"] == [{"window": 10, "horizon": 1}]
    assert quick["seeds"] == [7]
    assert quick["data_path"] == "data/raw/paper_data.xlsx"
    assert quick["protocol"] == "chronological-v1"
    assert quick["loss"] == "mse"
    assert quick["early_stopping_patience"] is None
    assert quick["restore_best_weights"] is False


def test_playground_fit_defaults_match_main_evaluation_config() -> None:
    config = yaml.safe_load(Path("configs/evaluation.yaml").read_text(encoding="utf-8"))
    evaluation = normalize_evaluation({"preset": "standard"})
    training = config["training"]
    optimizer = training["optimizer"]

    assert evaluation["cells"] == config["experiment"]["cells"]
    assert evaluation["seeds"] == config["experiment"]["seeds"]
    assert evaluation["epochs"] == training["epochs"]
    assert evaluation["batch_size"] == training["batch_size"]
    assert evaluation["evaluation_batch_size"] == training["evaluation_batch_size"]
    assert evaluation["learning_rate"] == optimizer["learning_rate"]
    assert evaluation["adam_beta1"] == optimizer["beta1"]
    assert evaluation["adam_beta2"] == optimizer["beta2"]
    assert evaluation["adam_epsilon"] == optimizer["epsilon"]
    assert evaluation["adam_amsgrad"] == optimizer["amsgrad"]
    assert evaluation["loss"] == training["loss"]
    assert evaluation["shuffle"] == training["shuffle"]
    assert evaluation["early_stopping_patience"] == training["early_stopping_patience"]
    assert evaluation["restore_best_weights"] == training["restore_best_weights"]


def test_evaluation_rejects_paths_outside_data() -> None:
    try:
        normalize_evaluation({"data_path": "../private.xlsx"})
    except ValueError as error:
        assert "relative path" in str(error)
    else:
        raise AssertionError("unsafe path was accepted")


def test_job_manager_persists_and_cancels_a_queued_job(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "results", tmp_path)
    job = manager.submit_validation(_preset("residual-tcn"), {"preset": "quick"})
    assert job["status"]["state"] == "queued"
    assert job["status"]["execution_target"] == "local"
    cancelled = manager.cancel(job["id"])
    assert cancelled["status"]["state"] == "cancelled"
    recovered = JobManager(tmp_path / "results", tmp_path)
    assert recovered.get_job(job["id"])["status"]["state"] == "cancelled"


def test_running_jobs_are_marked_interrupted_on_recovery(tmp_path: Path) -> None:
    job_dir = tmp_path / "results" / "jobs" / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text("{}", encoding="utf-8")
    (job_dir / "status.json").write_text(
        json.dumps({"state": "running"}), encoding="utf-8"
    )
    manager = JobManager(tmp_path / "results", tmp_path)
    assert manager.get_job("job-1")["status"]["state"] == "interrupted"


def test_modal_job_records_gpu_and_forces_cuda(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "hypercast4d.playground.modal_capability",
        lambda: {"sdk_installed": True, "authenticated": True},
    )
    manager = JobManager(tmp_path / "results", tmp_path)
    job = manager.submit_validation(
        _preset("paper-quaternion"),
        {"preset": "quick", "device": "cpu"},
        {"target": "modal", "gpu": "L4"},
    )
    assert job["status"]["execution_target"] == "modal"
    assert job["status"]["gpu"] == "L4"
    assert job["request"]["evaluation"]["device"] == "cuda"


def test_final_test_is_locked_to_one_job_per_candidate(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "results", tmp_path)
    parent = manager.submit_validation(
        _preset("residual-tcn"),
        {
            "preset": "standard",
            "cells": [{"window": 10, "horizon": 1}],
            "seeds": [7],
            "epochs": 1,
        },
    )
    status_path = manager.jobs_root / parent["id"] / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["state"] = "complete"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    final_job = manager.submit_final_test(parent["id"])
    assert final_job["status"]["phase"] == "final_test"
    try:
        manager.submit_final_test(parent["id"])
    except ValueError as error:
        assert "already has" in str(error)
    else:
        raise AssertionError("duplicate final test was accepted")


def test_playground_api_exposes_catalog_validation_and_saved_architectures(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "playground", tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"ok": True}
        catalog = client.get("/api/v1/catalog")
        assert catalog.status_code == 200
        assert len(catalog.json()["method_collection"]["methods"]) == 52
        assert any(item["type"] == "tcn" for group in catalog.json()["categories"] for item in group["layers"])

        response = client.post(
            "/api/v1/architectures/validate",
            json={"architecture": _preset("residual-tcn"), "window": 20, "horizon": 5},
        )
        assert response.status_code == 200
        assert response.json()["output_shape"] == "[B, 5]"

        saved = client.post("/api/v1/architectures", json=_preset("residual-tcn"))
        assert saved.status_code == 200
        records = client.get("/api/v1/architectures").json()
        assert records[0]["id"] == saved.json()["id"]
        assert client.get("/").status_code == 200


def test_compute_api_reports_modal_without_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "hypercast4d.playground.modal_capability",
        lambda: {
            "available": False,
            "sdk_installed": True,
            "authenticated": False,
            "gpus": [{"id": "L4", "label": "L4", "description": "Recommended"}],
            "setup_command": "modal setup",
        },
    )
    app = create_app(tmp_path / "playground", tmp_path)
    with TestClient(app) as client:
        payload = client.get("/api/v1/compute").json()
        assert payload["local"] == {"available": True}
        assert payload["modal"]["available"] is False
        assert "token" not in json.dumps(payload).lower()


def test_playground_reads_main_reproduction_results(tmp_path: Path) -> None:
    reproduction = tmp_path / "paper_reproduction"
    reproduction.mkdir()
    (reproduction / "candidates.csv").write_text(
        "input_order,window,horizon,model,candidate,mae_mean_scaled,"
        "mae_std_scaled,parameters,train_seconds_mean,declared_algebra,"
        "effective_algebra,settings_json\n"
        'published,10,1,cnn,0,0.08,0.01,145,1.2,,,"{}"\n'
        'published,10,1,cnn,1,0.06,0.02,145,1.3,,,"{}"\n',
        encoding="utf-8",
    )
    app = create_app(tmp_path / "playground", tmp_path, reproduction)
    with TestClient(app) as client:
        rows = client.get("/api/runs").json()
        assert len(rows) == 1
        assert rows[0]["candidate"] == 1
        assert rows[0]["mae"] == 0.06


@pytest.mark.parametrize("preset_id", ["residual-tcn", "research-dlinear", "research-patchtst", "research-itransformer"] + [
    f"tslib-{name}" for name in UPSTREAM_MODELS
] + ['tslib-tsmixer-internal'] + [f'tslib-{name}-graph' for name in UPSTREAM_MODELS])
def test_worker_completes_a_real_validation_job(
    tmp_path: Path, monkeypatch, preset_id: str
) -> None:
    data_directory = tmp_path / "data" / "raw"
    data_directory.mkdir(parents=True)
    rng = np.random.default_rng(7)
    changes = rng.normal(0.01, 0.05, size=(140, 4))
    values = 10 + np.cumsum(changes, axis=0)
    frame = pd.DataFrame(values, columns=["Copper", "FCX", "CLP", "SCCO"])
    frame.insert(0, "Date", pd.date_range("2020-01-01", periods=len(frame)))
    frame.to_excel(data_directory / "paper_data.xlsx", index=False)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    architecture = _preset(preset_id.removesuffix('-internal').removesuffix('-graph'))
    if preset_id.endswith('-graph'):
        from hypercast4d.graph_architecture import convert_to_graph
        architecture = convert_to_graph(architecture)
    if preset_id.endswith('-internal'):
        architecture['layers'][0]['internal_overrides'] = {
            'model.0.temporal': {'hidden_units': [16], 'activation': 'gelu', 'dropout': .1, 'bias': True}}
    request = {
        "phase": "validation",
        "architecture": architecture,
        "evaluation": {
            "preset": "quick",
            "data_path": "data/raw/paper_data.xlsx",
            "epochs": 1,
        },
    }
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    run_job(job_dir)

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    runs = pd.read_csv(job_dir / "runs.csv")
    assert list(runs["split"]) == ["validation"]
    assert np.isfinite(runs.loc[0, "mae_ratio"])


def test_graph_api_migration_drafts_and_all_cell_validation(tmp_path):
    from hypercast4d.graph_architecture import convert_to_graph
    manager = JobManager(tmp_path / 'results', tmp_path)
    original = manager.save_architecture(_preset('tslib-tsmixer'))
    graph = convert_to_graph(original['spec'])
    view = {'positions': {'input': {'x': 10, 'y': 20}}, 'collapsed': ['layer-core']}
    saved = manager.save_architecture({**graph, 'view': view})
    assert saved['id'] != original['id']
    assert manager.get_architecture(original['id'])['spec']['schema_version'] == 1
    assert manager.get_architecture(saved['id'])['view'] == view
    # Drafts may be saved, but are not accepted for execution.
    graph['edges'].pop()
    manager.save_architecture(graph)
    with pytest.raises(ValueError, match='ports'):
        manager.submit_validation(graph, {'preset': 'quick'})
    graph = convert_to_graph(original['spec'])
    graph['nodes'].append({'id': 'fixed-head', 'kind': 'dense', 'params': {'units': 1}})
    graph['edges'].append({'source': graph['output'], 'target': 'fixed-head', 'port': 'x'})
    graph['output'] = 'fixed-head'
    with pytest.raises(ValueError, match='Forecast output'):
        manager.submit_validation(graph, {'preset': 'quick', 'cells': [{'window': 10, 'horizon': 1}, {'window': 10, 'horizon': 3}]})


def test_convert_endpoint_returns_executable_nodes(tmp_path):
    app = create_app(tmp_path / 'results', tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/v1/architectures/convert', json={'architecture': _preset('tslib-frets')})
        assert response.status_code == 200
        graph = response.json()
        assert graph['schema_version'] == 2 and len(graph['nodes']) > 20
        response = client.post('/api/v1/architectures/validate', json={'architecture': graph, 'window': 10, 'horizon': 3})
        assert response.status_code == 200
        assert any(n['label'] == 'Linear' for n in response.json()['graph_nodes'].values())
