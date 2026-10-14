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


_PREP = None


def prep_ext():
    """GPU image preprocessing kernels (kernels/image_prep.cu), used by lfm2opt.preprocess."""
    global _PREP
    if _PREP is None:
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _PREP = load(
            name="lfm2_image_prep",
            sources=[str(REPO_ROOT / "kernels" / "image_prep.cpp"), str(REPO_ROOT / "kernels" / "image_prep.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _PREP
