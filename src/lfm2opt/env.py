"""Capture everything needed to reproduce a measurement: versions, GPU, clocks, power limits."""

import importlib.util
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def git_commit() -> str | None:
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    if commit is None and (REPO_ROOT / ".git_commit").exists():  # code shipped without .git (Colab, rsync)
        return (REPO_ROOT / ".git_commit").read_text().strip()
    dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    if commit and dirty:
        commit += "-dirty"
    return commit


def gpu_short_name() -> str:
    """Short, filename-safe GPU name, e.g. 'T4', 'RTXPRO5000'."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name()
    for prefix in ("NVIDIA ", "Tesla ", "GeForce "):
        name = name.replace(prefix, "")
    return "".join(ch for ch in name if ch.isalnum())[:24]


def nvml_info() -> dict:
    """Clocks, power limit, temperature from NVML (absent on Jetson / CPU)."""
    try:
        import pynvml as nv

        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = {
            "driver": nv.nvmlSystemGetDriverVersion(),
            "power_limit_w": nv.nvmlDeviceGetPowerManagementLimit(h) / 1000,
            "power_default_limit_w": nv.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
            "sm_clock_max_mhz": nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM),
            "mem_clock_max_mhz": nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM),
            "sm_clock_now_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "temperature_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "persistence_mode": nv.nvmlDeviceGetPersistenceMode(h),
        }
        try:
            info["applications_clock_sm_mhz"] = nv.nvmlDeviceGetApplicationsClock(h, nv.NVML_CLOCK_SM)
        except nv.NVMLError:
            pass
        return info
    except Exception as e:  # pynvml missing, no NVML (Jetson), permission issue
        return {"nvml_error": repr(e)}


def collect_env() -> dict:
    import transformers

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "transformers": transformers.__version__,
        "git_commit": git_commit(),
        # Which optional fast paths transformers may silently pick up (changes the baseline!)
        "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None,
        "kernels_installed": importlib.util.find_spec("kernels") is not None,
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            gpu_name=torch.cuda.get_device_name(),
            gpu_capability=f"sm_{props.major}{props.minor}",
            gpu_sm_count=props.multi_processor_count,
            gpu_mem_gb=round(props.total_memory / 2**30, 2),
            gpu_l2_mb=round(getattr(props, "L2_cache_size", 0) / 2**20, 2),
            bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
        )
        env.update(nvml_info())
    return env
