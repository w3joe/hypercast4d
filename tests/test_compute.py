import subprocess

import pytest

from hypercast4d.compute import MODAL_GPU_IDS, modal_capability, normalize_execution


def test_execution_defaults_to_local() -> None:
    assert normalize_execution(None) == {"target": "local", "gpu": None}


@pytest.mark.parametrize("gpu", sorted(MODAL_GPU_IDS))
def test_supported_modal_gpus_are_normalized(gpu: str) -> None:
    assert normalize_execution({"target": "modal", "gpu": gpu}) == {
        "target": "modal",
        "gpu": gpu,
    }


def test_unknown_modal_gpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Modal GPU"):
        normalize_execution({"target": "modal", "gpu": "ImaginaryGPU"})


def test_modal_capability_does_not_expose_credentials(monkeypatch) -> None:
    monkeypatch.setattr("hypercast4d.compute.importlib.util.find_spec", lambda _: object())
    monkeypatch.setattr("hypercast4d.compute._modal_executable", lambda: "/bin/modal")
    monkeypatch.setattr(
        "hypercast4d.compute.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    capability = modal_capability()
    assert capability["available"] is True
    assert set(capability) == {
        "available",
        "sdk_installed",
        "authenticated",
        "gpus",
        "setup_command",
    }
