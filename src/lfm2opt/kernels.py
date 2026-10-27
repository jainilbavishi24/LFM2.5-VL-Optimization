"""JIT-built CUDA extension with our fused decode kernels (kernels/lfm2_decode.cu).

Needs a Python with headers (Python.h) and nvcc: on the course server use `.venv-ref` (uv's standalone Python);
the system Python there has no headers. The build is cached under ~/.cache/torch_extensions.
"""

import os

import torch

from .env import REPO_ROOT

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _EXT = load(
            name="lfm2_decode",
            sources=[str(REPO_ROOT / "kernels" / "lfm2_decode.cpp"), str(REPO_ROOT / "kernels" / "lfm2_decode.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
        # Programmatic Dependent Launch between our kernels (on by default; LFM2_PDL=0 for A/B measurements)
        _EXT.set_pdl(os.environ.get("LFM2_PDL", "1") != "0")
    return _EXT



_EXTS = {}


def _build(name: str, stem: str):
    """JIT-build kernels/<stem>.cpp + kernels/<stem>.cu once per process (cached on disk by torch)."""
    if name not in _EXTS:
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _EXTS[name] = load(
            name=name,
            sources=[str(REPO_ROOT / "kernels" / f"{stem}.cpp"), str(REPO_ROOT / "kernels" / f"{stem}.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXTS[name]


def prep_ext():
    """GPU image preprocessing kernels (kernels/image_prep.cu), used by lfm2opt.preprocess."""
    return _build("lfm2_image_prep", "image_prep")


def vision_ext():
    """Vision-encoder kernels (kernels/vision_ops.cu), used by lfm2opt.vision."""
    return _build("lfm2_vision_ops", "vision_ops")


def int8_ext():
    """INT8 tensor-core GEMMs for prefill and the vision encoder (kernels/int8_gemm.cu), used by lfm2opt.int8."""
    return _build("lfm2_int8_gemm", "int8_gemm")
