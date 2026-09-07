"""Explicit, server-owned GCP configuration. No cloud resources are created here."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any

GCP_MACHINES = {
    "L4": {1: "g2-standard-4", 2: "g2-standard-24", 4: "g2-standard-48", 8: "g2-standard-96"},
    "A100-40GB": {**{n: f"a2-highgpu-{n}g" for n in (1, 2, 4, 8)}, 16: "a2-megagpu-16g"},
}


def normalize_gcp_execution(raw: dict[str, Any]) -> dict[str, Any]:
    gpu = raw.get("gpu", "L4")
    count = raw.get("gpu_count", 1)
    if not isinstance(gpu, str) or gpu not in GCP_MACHINES:
        raise ValueError("GCP GPU must be L4 or A100-40GB")
    if type(count) is not int or count not in GCP_MACHINES[gpu]:
        raise ValueError(f"unsupported GCP GPU count for {gpu}")
    return {"target": "gcp", "gpu": gpu, "gpu_count": count,
            "machine_type": GCP_MACHINES[gpu][count]}


def gcp_config() -> dict[str, str]:
    config = {}
    for key in ("PROJECT", "ZONE", "BUCKET", "SERVICE_ACCOUNT", "IMAGE", "IMAGE_PROJECT", "SUBNET"):
        value = os.environ.get(f"HYPERCAST_GCP_{key}", "").strip()
        if not value or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._@-]*", value):
            raise ValueError(f"Set HYPERCAST_GCP_{key}; see docs/gcp_gpu.md")
        config[key.lower()] = value
    hours = os.environ.get("HYPERCAST_GCP_MAX_HOURS", "6")
    if not hours.isdigit() or not 1 <= int(hours) <= 24:
        raise ValueError("HYPERCAST_GCP_MAX_HOURS must be 1–24")
    config["max_hours"] = hours
    return config


def gcp_capability() -> dict[str, Any]:
    available = False
    message = "Install Google Cloud CLI and run gcloud auth login; see docs/gcp_gpu.md"
    config: dict[str, str] = {}
    try:
        config = gcp_config()
        if shutil.which("gcloud"):
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            available = result.returncode == 0 and bool(result.stdout.strip())
            if available:
                message = f"Configured · {config['zone']} · quota and capacity checked at launch"
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        message = str(error)
    return {"available": available, "message": message,
            "gpus": [{"id": gpu, "label": gpu.replace("-", " "), "counts": list(machines)}
                     for gpu, machines in GCP_MACHINES.items()]}
