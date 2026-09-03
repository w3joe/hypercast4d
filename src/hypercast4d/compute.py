"""Execution-target validation and local Modal capability discovery."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


MODAL_GPUS: tuple[dict[str, str], ...] = (
    {"id": "T4", "label": "T4", "description": "Economy"},
    {"id": "L4", "label": "L4", "description": "Recommended"},
    {"id": "A10", "label": "A10", "description": "Balanced"},
    {"id": "L40S", "label": "L40S", "description": "High throughput"},
    {"id": "A100-40GB", "label": "A100 40 GB", "description": "Training"},
    {"id": "A100-80GB", "label": "A100 80 GB", "description": "Large memory"},
    {"id": "H100", "label": "H100", "description": "Fastest common"},
    {"id": "H200", "label": "H200", "description": "Maximum memory"},
)
MODAL_GPU_IDS = frozenset(item["id"] for item in MODAL_GPUS)


def normalize_execution(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the serializable compute choice attached to a job."""
    raw = raw or {}
    target = str(raw.get("target", "local"))
    if target == "local":
        return {"target": "local", "gpu": None}
    if target != "modal":
        raise ValueError("execution target must be local or modal")
    gpu = str(raw.get("gpu", "L4"))
    if gpu not in MODAL_GPU_IDS:
        raise ValueError(
            f"unsupported Modal GPU {gpu!r}; choose one of {sorted(MODAL_GPU_IDS)}"
        )
    return {"target": "modal", "gpu": gpu}


def _modal_executable() -> str | None:
    beside_python = Path(sys.executable).with_name("modal")
    if beside_python.is_file():
        return str(beside_python)
    return shutil.which("modal")


def modal_capability(timeout: float = 8.0) -> dict[str, Any]:
    """Report whether this Python environment can submit authenticated jobs."""
    sdk_installed = importlib.util.find_spec("modal") is not None
    executable = _modal_executable()
    authenticated = False
    if sdk_installed and executable is not None:
        try:
            completed = subprocess.run(
                [executable, "token", "info"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            authenticated = completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            authenticated = False
    return {
        "available": sdk_installed and authenticated,
        "sdk_installed": sdk_installed,
        "authenticated": authenticated,
        "gpus": list(MODAL_GPUS),
        "setup_command": (
            "modal setup"
            if sdk_installed
            else "uv pip install -e '.[modal]' && modal setup"
        ),
    }
