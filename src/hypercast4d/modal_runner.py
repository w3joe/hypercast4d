"""Submit one playground job to Modal and copy its artifacts back locally."""

from __future__ import annotations

import io
import json
import os
import signal
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from .compute import normalize_execution

try:  # Modal stays optional for users who only run on their own machine.
    import modal
except ImportError:  # pragma: no cover - exercised through capability checks
    modal = None


RESULT_FILES = ("runs.csv", "per_lead.csv", "summary.json", "summary.csv")


def _atomic_bytes(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def _execute_payload(
    request_json: str,
    dataset: bytes,
    parent_runs: bytes | None,
    *,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Run the normal worker inside an isolated filesystem and return artifacts."""
    from .playground_runner import run_job

    request = json.loads(request_json)
    data_path = Path(request["evaluation"]["data_path"])
    if data_path.is_absolute() or ".." in data_path.parts:
        raise ValueError("remote data path must stay inside the job workspace")

    with tempfile.TemporaryDirectory(prefix="hypercast4d-modal-") as temporary:
        root = Path(temporary)
        job_dir = root / "job"
        job_dir.mkdir()
        remote_data = root / data_path
        remote_data.parent.mkdir(parents=True, exist_ok=True)
        remote_data.write_bytes(dataset)
        if request.get("phase") == "final_test":
            if parent_runs is None:
                raise ValueError("final-test Modal jobs require parent validation runs")
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "runs.csv").write_bytes(parent_runs)
            request["parent_job_dir"] = str(parent_dir)
        (job_dir / "request.json").write_text(
            json.dumps(request), encoding="utf-8"
        )

        output = io.StringIO()
        previous_directory = Path.cwd()
        ok = True
        error_message: str | None = None
        actual_gpu: str | None = None
        try:
            os.chdir(root)
            with redirect_stdout(output), redirect_stderr(output):
                if require_cuda:
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("Modal allocated a worker without CUDA")
                    actual_gpu = torch.cuda.get_device_name(0)
                    print(f"Modal GPU: {actual_gpu}", flush=True)
                run_job(job_dir)
        except BaseException as error:  # Return the worker's useful error/logs.
            ok = False
            error_message = str(error)
            traceback.print_exc(file=output)
        finally:
            os.chdir(previous_directory)

        status_path = job_dir / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {"state": "failed", "error": error_message}
        )
        files = {
            name: (job_dir / name).read_bytes()
            for name in RESULT_FILES
            if (job_dir / name).exists()
        }
        return {
            "ok": ok,
            "error": error_message,
            "actual_gpu": actual_gpu,
            "status": status,
            "files": files,
            "log": output.getvalue(),
        }


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "numpy>=1.26",
            "openpyxl>=3.1",
            "pandas>=2.1",
            "torch>=2.2",
        )
        .add_local_python_source("hypercast4d")
    )
    app = modal.App("hypercast4d-playground-jobs")

    @app.function(image=image, cpu=2.0, memory=4096, timeout=24 * 60 * 60)
    def execute_remote(
        request_json: str, dataset: bytes, parent_runs: bytes | None
    ) -> dict[str, Any]:
        result = _execute_payload(request_json, dataset, parent_runs)
        print(result["log"], end="", flush=True)
        return result

else:  # Keep imports working when the optional integration is not installed.
    app = None
    execute_remote = None


def run_modal_job(job_dir: Path) -> None:
    """Run a queued job remotely and materialize the normal local result files."""
    if modal is None or app is None or execute_remote is None:
        raise RuntimeError("Modal support is not installed; install the 'modal' extra")

    from .playground_runner import _status, normalize_evaluation

    job_dir = job_dir.resolve()
    request_path = job_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    execution = normalize_execution(request.get("execution"))
    if execution["target"] != "modal":
        raise ValueError("modal_runner requires a Modal execution target")
    gpu = str(execution["gpu"])
    evaluation = normalize_evaluation(request.get("evaluation"))
    data_path = Path(evaluation["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    dataset = data_path.read_bytes()
    parent_runs: bytes | None = None
    if request.get("phase") == "final_test":
        parent_runs_path = Path(request["parent_job_dir"]) / "runs.csv"
        parent_runs = parent_runs_path.read_bytes()

    total = len(evaluation["cells"]) * len(evaluation["seeds"])
    if request.get("phase", "validation") == "validation":
        total *= len(evaluation["folds"])
    _status(
        job_dir,
        state="running",
        completed=0,
        total=total,
        current=None,
        gpu=gpu,
    )
    print(f"Submitting job to Modal on {gpu}", flush=True)

    function_call: Any = None

    def cancel_remote(_signal: int, _frame: Any) -> None:
        if function_call is not None:
            function_call.cancel(terminate_containers=True)
        raise SystemExit(130)

    previous_sigterm = signal.signal(signal.SIGTERM, cancel_remote)
    previous_sigint = signal.signal(signal.SIGINT, cancel_remote)
    try:
        with modal.enable_output():
            with app.run():
                function_call = execute_remote.with_options(gpu=gpu).spawn(
                    request_path.read_text(encoding="utf-8"),
                    dataset,
                    parent_runs,
                )
                dashboard_url = function_call.get_dashboard_url()
                _status(
                    job_dir,
                    modal_call_id=function_call.object_id,
                    modal_dashboard_url=dashboard_url,
                )
                print(f"Modal call: {function_call.object_id}", flush=True)
                if dashboard_url:
                    print(f"Modal dashboard: {dashboard_url}", flush=True)
                result = function_call.get()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    remote_log = str(result.get("log", ""))
    if remote_log:
        print(remote_log, end="" if remote_log.endswith("\n") else "\n", flush=True)
    for name, contents in result.get("files", {}).items():
        if name not in RESULT_FILES or not isinstance(contents, bytes):
            continue
        _atomic_bytes(job_dir / name, contents)
    remote_status = result.get("status", {})
    if result.get("ok"):
        _status(
            job_dir,
            state="complete",
            completed=int(remote_status.get("completed", total)),
            total=int(remote_status.get("total", total)),
            current=None,
            actual_gpu=result.get("actual_gpu"),
            error=None,
        )
        return
    error_message = str(result.get("error") or "Modal worker failed")
    _status(job_dir, state="failed", error=error_message, current=None)
    raise RuntimeError(error_message)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    run_modal_job(args.job_dir)


if __name__ == "__main__":
    main()
