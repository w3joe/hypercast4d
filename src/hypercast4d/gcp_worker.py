"""VM-side execution: independent cell/seed trials, one process per GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from .playground_runner import _atomic_csv, _atomic_json, _status, _summary, normalize_evaluation
from .diagnostics import initialize_diagnostics


def trial_requests(request: dict, shard_index: int = 0, shard_count: int = 1) -> list[dict]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid VM shard")
    evaluation = normalize_evaluation(request["evaluation"])
    return [{**request, "evaluation": {**evaluation, "cells": [cell], "seeds": [seed]}}
            for cell in evaluation["cells"] for seed in evaluation["seeds"]][shard_index::shard_count]


def merge_trials(job_dir: Path, directories: list[Path]) -> None:
    import pandas as pd
    initialize_diagnostics(job_dir)
    for filename in ("runs.csv", "per_lead.csv", "predictions.csv"):
        frames = [pd.read_csv(path / filename) for path in directories]
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(job_dir / filename, index=False)
        if filename == "runs.csv":
            rows = json.loads(combined.to_json(orient="records"))
    summary = _summary(rows)
    _atomic_json(job_dir / "summary.json", summary)
    _atomic_csv(job_dir / "summary.csv", summary)
    _status(job_dir, state="complete", completed=len(rows), total=len(rows), current=None)


def run_gpu_trials(job_dir: Path, gpu_count: int, shard_index: int = 0, shard_count: int = 1) -> None:
    import torch
    if torch.cuda.device_count() < gpu_count:
        raise RuntimeError(f"Requested {gpu_count} GPUs; CUDA sees {torch.cuda.device_count()}")
    request = json.loads((job_dir / "request.json").read_text())
    trials = trial_requests(request, shard_index, shard_count)
    if not trials:
        initialize_diagnostics(job_dir)
        for filename in ("runs.csv", "per_lead.csv", "summary.csv", "training.log"):
            (job_dir / filename).write_text("")
        _atomic_json(job_dir / "summary.json", [])
        _status(job_dir, state="complete", completed=0, total=0, current=None)
        return
    directories = []
    for index, trial in enumerate(trials):
        path = job_dir / f"trial-{index}"
        path.mkdir()
        _atomic_json(path / "request.json", trial)
        directories.append(path)

    def run_lane(gpu: int) -> None:
        for index in range(gpu, len(trials), gpu_count):
            path = directories[index]
            with (path / "training.log").open("w") as log:
                subprocess.run(
                    [sys.executable, "-m", "hypercast4d.playground_runner", "--job-dir", str(path)],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )

    try:
        with ThreadPoolExecutor(max_workers=min(gpu_count, len(trials))) as pool:
            list(pool.map(run_lane, range(min(gpu_count, len(trials)))))
        merge_trials(job_dir, directories)
    finally:
        with (job_dir / "training.log").open("w") as log:
            for path in directories:
                if (path / "training.log").exists():
                    log.write(f"\n{path.name}\n" + (path / "training.log").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    try:
        run_gpu_trials(args.job_dir.resolve(), args.gpu_count, args.shard_index, args.shard_count)
    except BaseException as error:
        _status(args.job_dir, state="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
