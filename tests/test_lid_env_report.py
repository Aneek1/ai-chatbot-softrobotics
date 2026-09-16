import subprocess

import pytest

from training.env_report import (
    CudaUnavailable,
    GpuBusy,
    gpu_from_nvidia_smi,
    llama_server_running,
    package_versions,
    require_cuda,
    wait_for_gpu,
)


def completed(stdout: str, returncode: int = 0):
    return lambda *args, **kwargs: subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")


def test_parses_nvidia_smi():
    run = completed("NVIDIA GeForce RTX 5060 Laptop GPU, 592.15, 8151, 12.0\n")
    assert gpu_from_nvidia_smi(run) == {
        "name": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "driver": "592.15",
        "memory_mib": 8151,
        "compute_capability": "12.0",
    }


def test_missing_nvidia_smi_means_no_gpu():
    def run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert gpu_from_nvidia_smi(run) is None


def test_package_versions_reports_missing_packages_as_none():
    versions = package_versions(("pytest", "no-such-package-for-sure"))
    assert versions["pytest"]
    assert versions["no-such-package-for-sure"] is None


def test_cpu_only_torch_is_rejected():
    with pytest.raises(CudaUnavailable, match="CPU-only build"):
        require_cuda({"version": "2.14.0+cpu", "cuda": None, "cuda_available": False})


def test_missing_torch_is_rejected():
    with pytest.raises(CudaUnavailable, match="uv sync --group train"):
        require_cuda(None)


def test_cuda_build_without_a_visible_gpu_is_rejected():
    with pytest.raises(CudaUnavailable, match="sees no GPU"):
        require_cuda({"version": "2.14.0+cu130", "cuda": "13.0", "cuda_available": False})


def test_cuda_build_with_gpu_passes():
    require_cuda({"version": "2.14.0+cu130", "cuda": "13.0", "cuda_available": True})


def test_llama_server_detected_on_windows():
    run = completed("llama-server.exe            91696 Console   1  1,978,168 K\n")
    assert llama_server_running(run, system=lambda: "Windows") is True
    idle = completed("INFO: No tasks are running which match the specified criteria.\n")
    assert llama_server_running(idle, system=lambda: "Windows") is False


def test_llama_server_detected_with_pgrep_elsewhere():
    assert llama_server_running(completed("1234\n", 0), system=lambda: "Linux") is True
    assert llama_server_running(completed("", 1), system=lambda: "Darwin") is False


def test_wait_returns_immediately_when_idle():
    sleeps = []
    assert wait_for_gpu(is_busy=lambda: False, sleep=sleeps.append, log=lambda m: None) == 0
    assert sleeps == []


def test_wait_polls_every_five_minutes_until_idle():
    states = iter([True, True, False])
    sleeps = []
    waited = wait_for_gpu(is_busy=lambda: next(states), sleep=sleeps.append, log=lambda m: None)
    assert sleeps == [300, 300]
    assert waited == 600


def test_wait_gives_up_after_two_hours():
    sleeps = []
    with pytest.raises(GpuBusy, match="after 120 minutes"):
        wait_for_gpu(is_busy=lambda: True, sleep=sleeps.append, log=lambda m: None)
    assert len(sleeps) == 24
