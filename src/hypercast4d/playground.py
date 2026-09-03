"""Local API and static frontend for the HyperCast4D architecture playground."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import sys
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .architecture import (
    ArchitectureError,
    architecture_hash,
    layer_catalog,
    normalize_architecture_spec,
    presets,
    validate_architecture,
)
from .playground_runner import (
    EVALUATION_DEFAULTS,
    EVALUATION_PRESETS,
    normalize_evaluation,
)


TERMINAL_STATES = {"complete", "failed", "cancelled", "interrupted"}


def _read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    row[key] = None
                    continue
                try:
                    numeric = float(value)
                    row[key] = int(numeric) if numeric.is_integer() else numeric
                except ValueError:
                    row[key] = value
            rows.append(row)
        return rows


class JobManager:
    """Persistent, single-worker FIFO queue for local training processes."""

    def __init__(self, root: Path, project_root: Path) -> None:
        self.root = root.resolve()
        self.project_root = project_root.resolve()
        self.jobs_root = self.root / "jobs"
        self.architectures_root = self.root / "architectures"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.architectures_root.mkdir(parents=True, exist_ok=True)
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()
        self.active_id: str | None = None
        self.active_process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self._recover_jobs()

    def _recover_jobs(self) -> None:
        for directory in sorted(self.jobs_root.iterdir()):
            if not directory.is_dir():
                continue
            status_path = directory / "status.json"
            status = _read_json(status_path, {})
            state = status.get("state")
            if state in {"running", "starting"}:
                status.update(state="interrupted", updated_at=_utc_now())
                _atomic_json(status_path, status)
            elif state == "queued":
                self.pending.put(directory.name)

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.lock:
            process = self.active_process
            active_id = self.active_id
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            if active_id is not None:
                self._update_status(active_id, state="interrupted")
        self.pending.put(None)
        if self.thread is not None:
            self.thread.join(timeout=6)

    def _update_status(self, job_id: str, **updates: Any) -> None:
        path = self.jobs_root / job_id / "status.json"
        status = _read_json(path, {})
        status.update(updates)
        status["updated_at"] = _utc_now()
        _atomic_json(path, status)

    def _worker_loop(self) -> None:
        while True:
            job_id = self.pending.get()
            if job_id is None:
                return
            job_dir = self.jobs_root / job_id
            status = _read_json(job_dir / "status.json", {})
            if status.get("state") != "queued":
                continue
            self._update_status(job_id, state="starting")
            with (job_dir / "training.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "hypercast4d.playground_runner",
                        "--job-dir",
                        str(job_dir),
                    ],
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with self.lock:
                    self.active_id = job_id
                    self.active_process = process
                return_code = process.wait()
                with self.lock:
                    self.active_id = None
                    self.active_process = None
            latest = _read_json(job_dir / "status.json", {})
            if return_code and latest.get("state") not in {"failed", "cancelled"}:
                self._update_status(
                    job_id,
                    state="failed",
                    error=f"worker exited with status {return_code}",
                )

    def save_architecture(self, raw: dict[str, Any]) -> dict[str, Any]:
        spec = normalize_architecture_spec(raw)
        architecture_id = str(raw.get("id") or uuid.uuid4())
        try:
            uuid.UUID(architecture_id)
        except ValueError as error:
            raise ValueError("architecture id must be a UUID") from error
        path = self.architectures_root / f"{architecture_id}.json"
        existing = _read_json(path, {})
        now = _utc_now()
        record = {
            "id": architecture_id,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "hash": architecture_hash(spec),
            "spec": spec,
        }
        _atomic_json(path, record)
        return record

    def list_architectures(self) -> list[dict[str, Any]]:
        records = [
            record
            for path in self.architectures_root.glob("*.json")
            if (record := _read_json(path))
        ]
        return sorted(records, key=lambda record: record["updated_at"], reverse=True)

    def get_architecture(self, architecture_id: str) -> dict[str, Any]:
        record = _read_json(self.architectures_root / f"{architecture_id}.json")
        if record is None:
            raise KeyError(architecture_id)
        return record

    def _create_job(self, request: dict[str, Any]) -> dict[str, Any]:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        job_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / job_id
        job_dir.mkdir()
        request = {"job_id": job_id, **request}
        _atomic_json(job_dir / "request.json", request)
        status = {
            "id": job_id,
            "state": "queued",
            "phase": request["phase"],
            "completed": 0,
            "total": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "candidate_hash": request["candidate_hash"],
            "architecture_name": request["architecture"]["name"],
            "preset": request["evaluation"]["preset"],
            "protocol": request["evaluation"]["protocol"],
            "error": None,
        }
        _atomic_json(job_dir / "status.json", status)
        self.pending.put(job_id)
        return self.get_job(job_id)

    def submit_validation(
        self, architecture: dict[str, Any], evaluation: dict[str, Any] | None
    ) -> dict[str, Any]:
        spec = normalize_architecture_spec(architecture)
        normalized_evaluation = normalize_evaluation(evaluation)
        candidate_hash = architecture_hash(spec, normalized_evaluation)
        return self._create_job(
            {
                "phase": "validation",
                "candidate_hash": candidate_hash,
                "architecture": spec,
                "evaluation": normalized_evaluation,
            }
        )

    def submit_final_test(self, parent_id: str) -> dict[str, Any]:
        parent = self.get_job(parent_id)
        if parent["status"].get("state") != "complete":
            raise ValueError("validation job must be complete")
        request = parent["request"]
        if request.get("phase") != "validation":
            raise ValueError("a final test must reference a validation job")
        if request["evaluation"]["preset"] == "quick":
            raise ValueError("Quick runs cannot unlock the held-out test set")
        candidate_hash = request["candidate_hash"]
        for job in self.list_jobs():
            status = job["status"]
            if (
                status.get("phase") == "final_test"
                and status.get("candidate_hash") == candidate_hash
                and status.get("state") not in {"failed", "cancelled", "interrupted"}
            ):
                raise ValueError("this candidate already has a final-test run")
        return self._create_job(
            {
                "phase": "final_test",
                "candidate_hash": candidate_hash,
                "parent_job_id": parent_id,
                "parent_job_dir": str(self.jobs_root / parent_id),
                "architecture": request["architecture"],
                "evaluation": request["evaluation"],
            }
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        state = job["status"].get("state")
        if state in TERMINAL_STATES:
            return job
        self._update_status(job_id, state="cancelled")
        with self.lock:
            if self.active_id == job_id and self.active_process is not None:
                self.active_process.terminate()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        job_dir = self.jobs_root / job_id
        if not job_dir.is_dir() or job_dir.parent != self.jobs_root:
            raise KeyError(job_id)
        return {
            "id": job_id,
            "request": _read_json(job_dir / "request.json", {}),
            "status": _read_json(job_dir / "status.json", {}),
            "summary": _read_json(job_dir / "summary.json", []),
            "runs": _csv_rows(job_dir / "runs.csv"),
            "per_lead": _csv_rows(job_dir / "per_lead.csv"),
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for directory in sorted(self.jobs_root.iterdir(), reverse=True):
            if directory.is_dir():
                try:
                    jobs.append(self.get_job(directory.name))
                except KeyError:
                    pass
        return jobs

    def log(self, job_id: str) -> str:
        path = self.jobs_root / job_id / "training.log"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-100_000:]


def create_app(
    results_root: Path = Path("results/playground"),
    project_root: Path | None = None,
    legacy_results: Path = Path("results/evaluation"),
) -> FastAPI:
    project_root = (project_root or Path.cwd()).resolve()
    manager = JobManager(results_root, project_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        yield
        manager.stop()

    app = FastAPI(title="HyperCast4D Playground", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(ArchitectureError)
    async def architecture_error_handler(_, error: ArchitectureError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v1/catalog")
    def catalog() -> dict[str, Any]:
        return {
            **layer_catalog(),
            "presets": presets(),
            "evaluation_presets": EVALUATION_PRESETS,
            "evaluation_defaults": EVALUATION_DEFAULTS,
        }

    @app.post("/api/v1/architectures/validate")
    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            window = int(payload.get("window", 10))
            horizon = int(payload.get("horizon", 1))
            return validate_architecture(payload["architecture"], window, horizon)
        except (KeyError, TypeError, ValueError, ArchitectureError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/architectures")
    def list_architectures() -> list[dict[str, Any]]:
        return manager.list_architectures()

    @app.post("/api/v1/architectures")
    def save_architecture(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return manager.save_architecture(payload)
        except (ValueError, ArchitectureError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/architectures/{architecture_id}")
    def get_architecture(architecture_id: str) -> dict[str, Any]:
        try:
            return manager.get_architecture(architecture_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="architecture not found") from error

    @app.get("/api/v1/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return manager.list_jobs()

    @app.post("/api/v1/jobs")
    def submit_job(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return manager.submit_validation(
                payload["architecture"], payload.get("evaluation")
            )
        except (KeyError, ValueError, ArchitectureError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return manager.get_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/api/v1/jobs/{job_id}/log", response_class=PlainTextResponse)
    def get_log(job_id: str) -> str:
        try:
            manager.get_job(job_id)
            return manager.log(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return manager.cancel(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.post("/api/v1/jobs/{job_id}/final-test")
    def final_test(job_id: str) -> dict[str, Any]:
        try:
            return manager.submit_final_test(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/runs")
    def legacy_runs() -> list[dict[str, Any]]:
        # Use the same reader as the main dashboard so both leakage-safe
        # evaluation runs and released-notebook reproduction candidates appear
        # correctly in the unified Runs view.
        from .dashboard import load_runs

        return load_runs(legacy_results)

    @app.get("/api/status")
    def legacy_status() -> dict[str, Any]:
        from .dashboard import load_status

        rows = legacy_runs()
        return load_status(legacy_results, len(rows))

    static_root = Path(__file__).with_name("web_dist")
    assets = static_root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        candidate = static_root / path
        if path and candidate.is_file() and candidate.resolve().is_relative_to(static_root.resolve()):
            return FileResponse(candidate)
        index = static_root / "index.html"
        if index.exists():
            return FileResponse(index)
        return PlainTextResponse(
            "The playground frontend has not been built. Run npm run build in web/.",
            status_code=503,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/playground"))
    parser.add_argument("--results", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=1.5,
        help="Deprecated compatibility option; the workbench refreshes automatically",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the playground is local-only; host must be 127.0.0.1 or localhost")
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        create_app(args.results_root, Path.cwd(), args.results),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
