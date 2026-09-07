"""Ephemeral Compute Engine GPU VM lifecycle, using the user's gcloud identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
import zipfile

from .compute import normalize_execution
from .gcp_compute import gcp_config
from .playground_runner import _status, normalize_evaluation

RESULT_FILES = ("runs.csv", "per_lead.csv", "summary.json", "summary.csv",
                "predictions.csv", "diagnostics.json", "status.json", "training.log")


def cloud(*args: str, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", *args, "--quiet"], capture_output=True, text=True,
                          timeout=timeout, check=check, start_new_session=True)


def package_job(job_dir: Path, destination: Path) -> None:
    request = json.loads((job_dir / "request.json").read_text())
    evaluation = normalize_evaluation(request["evaluation"])
    request["evaluation"] = {**evaluation, "device": "cuda"}
    data = Path(evaluation["data_path"])
    root = Path.cwd().resolve()
    if not data.resolve().is_relative_to(root / "data"):
        raise ValueError("Dataset must resolve inside the project data directory")
    source = Path(__file__).parent
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json"} and "web_dist" not in path.parts:
                archive.write(path, "src/hypercast4d/" + path.relative_to(source).as_posix())
        archive.write(data, data.as_posix())
        if request.get("phase") == "final_test":
            archive.write(Path(request["parent_job_dir"]) / "runs.csv", "parent/runs.csv")
            request["parent_job_dir"] = "/opt/hypercast-job/parent"
        archive.writestr("job/request.json", json.dumps(request))


def startup_script(prefix: str, count: int) -> str:
    # All configuration is server-owned; quote even validated values.
    prefix = shlex.quote(prefix)
    return f'''#!/bin/bash
set -euo pipefail
mkdir -p /opt/hypercast-job/job
cd /opt/hypercast-job
export PYTHONPATH=/opt/hypercast-job/src
finish() {{
  code=$?
  trap - EXIT
  set +e
  echo "$code" > job/exit-code
  python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
with ZipFile('result.zip', 'w', ZIP_DEFLATED) as archive:
    for path in Path('job').iterdir():
        if path.is_file():
            archive.write(path, str(path))
PY
  gcloud storage cp result.zip {prefix}/result.zip
  # Keep the provider runtime timer active until the controller deletes this VM.
}}
trap finish EXIT
exec > >(tee /opt/hypercast-job/job/bootstrap.log) 2>&1
gcloud storage cp {prefix}/payload.zip payload.zip
python3 -m zipfile -e payload.zip .
python3 -m hypercast4d.gcp_worker --job-dir /opt/hypercast-job/job --gpu-count {count}
'''


def create_arguments(config: dict, name: str, machine: str, script: Path) -> list[str]:
    return ["compute", "instances", "create", name,
            f"--project={config['project']}", f"--zone={config['zone']}",
            f"--machine-type={machine}", f"--image={config['image']}",
            f"--image-project={config['image_project']}",
            f"--service-account={config['service_account']}", "--scopes=cloud-platform",
            f"--subnet={config['subnet']}", "--no-address",
            "--boot-disk-size=100GB", "--boot-disk-type=pd-balanced", "--boot-disk-auto-delete",
            "--maintenance-policy=TERMINATE", "--no-restart-on-failure",
            f"--max-run-duration={config['max_hours']}h", "--instance-termination-action=DELETE",
            f"--metadata-from-file=startup-script={script}", "--labels=app=hypercast4d"]


def collect_result(archive_path: Path, job_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for filename in (*RESULT_FILES, "bootstrap.log"):
            name = f"job/{filename}"
            if name in archive.namelist():
                if filename == "status.json":
                    remote_status = json.loads(archive.read(name))
                    _status(job_dir, **{key: remote_status[key] for key in
                            ("state", "completed", "total", "current", "error") if key in remote_status})
                elif filename in {"training.log", "bootstrap.log"}:
                    print(archive.read(name).decode("utf-8", errors="replace"), flush=True)
                else:
                    (job_dir / filename).write_bytes(archive.read(name))
        if archive.read("job/exit-code").strip() != b"0":
            raise RuntimeError("GCP worker failed; see training log")
        if not all(f"job/{name}" in archive.namelist() for name in RESULT_FILES):
            raise RuntimeError("GCP returned incomplete results")


def run_gcp_job(job_dir: Path) -> None:
    config = gcp_config()
    request = json.loads((job_dir / "request.json").read_text())
    execution = normalize_execution(request.get("execution"))
    if execution["target"] != "gcp":
        raise ValueError("Expected a GCP job")
    name = "hypercast-" + uuid.uuid4().hex[:24]
    prefix = f"gs://{config['bucket']}/hypercast4d/{name}"
    scope = [f"--project={config['project']}", f"--zone={config['zone']}"]
    _status(job_dir, state="starting", gcp_instance=name, gcp_project=config["project"],
            gcp_zone=config["zone"], gcp_prefix=prefix, gpu_count=execution["gpu_count"])

    def cancel(signum, frame):
        raise SystemExit(130)

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    attempted = False
    uploaded = False
    try:
        with tempfile.TemporaryDirectory(prefix="hypercast-gcp-") as temporary:
            directory = Path(temporary)
            package_job(job_dir, directory / "payload.zip")
            # Generated runtime artifact, not a source edit.
            (directory / "startup.sh").write_text(startup_script(prefix, execution["gpu_count"]))
            uploaded = True
            cloud("storage", "cp", str(directory / "payload.zip"), prefix + "/payload.zip", timeout=600)
            attempted = True
            cloud(*create_arguments(config, name, execution["machine_type"], directory / "startup.sh"), timeout=300)
            print(f"GCP VM {name}: startup / training; results and logs download when finished", flush=True)
            _status(job_dir, state="running", current=None)
            deadline = time.monotonic() + int(config["max_hours"]) * 3600
            while time.monotonic() < deadline:
                result = cloud("storage", "cp", prefix + "/result.zip", str(directory / "result.zip"),
                               check=False, timeout=120)
                if result.returncode == 0:
                    collect_result(directory / "result.zip", job_dir)
                    return
                vm = cloud("compute", "instances", "describe", name, *scope,
                           "--format=value(status)", check=False)
                if vm.returncode != 0 or vm.stdout.strip() == "TERMINATED":
                    raise RuntimeError("GCP VM stopped or is unavailable before results arrived; check VM serial logs and IAM")
                time.sleep(10)
            raise TimeoutError("GCP job exceeded its configured runtime limit")
    except SystemExit:
        _status(job_dir, state="cancelled", error=None)
        raise
    except BaseException as error:
        _status(job_dir, state="failed", error=str(error))
        raise
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        errors = []
        if attempted:
            try:
                result = cloud("compute", "instances", "delete", name, *scope, check=False)
                if result.returncode and "not found" not in result.stderr.lower():
                    errors.append(f"Check VM {name}: {result.stderr.strip()}")
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(f"Check VM {name}: {error}")
        if uploaded:
            try:
                result = cloud("storage", "rm", "--recursive", prefix + "/", check=False)
                if result.returncode:
                    errors.append(f"Check job storage {prefix}: {result.stderr.strip()}")
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(f"Check job storage {prefix}: {error}")
        _status(job_dir, gcp_cleanup="; ".join(errors) if errors else "complete")
        for error in errors:
            print(f"GCP cleanup warning: {error}", flush=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    run_gcp_job(parser.parse_args().job_dir.resolve())


if __name__ == "__main__":
    main()
