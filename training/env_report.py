"""Hardware and software facts recorded in every run, plus the GPU checks run before training.

Usage:
    uv run --group train python -m training.env_report                  # print the report
    uv run --group train python -m training.env_report --require-cuda   # also fail on a CPU-only torch
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch",
    "transformers",
    "onnx",
    "onnxruntime",
    "numpy",
    "pyarrow",
    "fasttext-wheel",
    "fasttext-predict",
    "opencc",
)


class CudaUnavailable(RuntimeError):
    pass


class GpuBusy(RuntimeError):
    pass


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_from_nvidia_smi(run: Callable = subprocess.run) -> dict | None:
    query = "--query-gpu=name,driver_version,memory.total,compute_cap"
    try:
        result = run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name, driver, memory, capability = (
        part.strip() for part in result.stdout.strip().splitlines()[0].split(",")
    )
    return {
        "name": name,
        "driver": driver,
        "memory_mib": int(float(memory)),
        "compute_capability": capability,
    }


def torch_info() -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    info = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        major, minor = torch.cuda.get_device_capability(0)
        info["device"] = torch.cuda.get_device_name(0)
        info["capability"] = f"{major}.{minor}"
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def git_commit(run: Callable = subprocess.run) -> str | None:
    try:
        result = run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu_from_nvidia_smi(),
        "torch": torch_info(),
        "commit": git_commit(),
        "packages": package_versions(),
    }


def require_cuda(info: dict | None) -> None:
    if info is None:
        raise CudaUnavailable("PyTorch is not installed. Run: uv sync --group train")
    if info["cuda"] is None:
        raise CudaUnavailable(
            f"PyTorch {info['version']} is a CPU-only build. Reinstall the CUDA build: "
            "uv sync --group train --reinstall-package torch"
        )
    if not info["cuda_available"]:
        raise CudaUnavailable(
            f"PyTorch was built for CUDA {info['cuda']} but sees no GPU; check the NVIDIA driver."
        )


def llama_server_running(run: Callable = subprocess.run, system: Callable[[], str] = platform.system) -> bool:
    if system() == "Windows":
        result = run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "llama-server.exe" in result.stdout.lower()
    result = run(["pgrep", "-f", "llama-server"], capture_output=True, text=True, check=False)
    return result.returncode == 0


def wait_for_gpu(
    is_busy: Callable[[], bool] = llama_server_running,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: int = 300,
    timeout_seconds: int = 7200,
    log: Callable[[str], None] = print,
) -> int:
    """DaybreakOS latency measurements run llama-server on this GPU; never train alongside them."""
    waited = 0
    while is_busy():
        if waited >= timeout_seconds:
            raise GpuBusy(
                f"llama-server is still running after {waited // 60} minutes; GPU training not started."
            )
        log(f"llama-server is running; checking again in {poll_seconds // 60} minutes")
        sleep(poll_seconds)
        waited += poll_seconds
    return waited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = environment()
    print(json.dumps(report, indent=2))
    if args.require_cuda:
        require_cuda(report["torch"])


if __name__ == "__main__":
    main()
