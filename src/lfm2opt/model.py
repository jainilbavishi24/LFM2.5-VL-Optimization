"""Load LFM2.5-VL and build model inputs for a workload."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import logging as hf_logging

from .env import REPO_ROOT

hf_logging.disable_progress_bar()

DEFAULT_MODEL = "LiquidAI/LFM2.5-VL-1.6B"
# The revision we read the code against (README "Third-party code").
PINNED_REVISIONS = {"LiquidAI/LFM2.5-VL-1.6B": "919fde3d022e3f90a4716006f993938ee8c2eb97"}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    """'auto' = bf16 where the GPU has native BF16 (sm_80+), else fp16 (e.g. T4)."""
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return DTYPES[name]


def dtype_name(dtype: torch.dtype) -> str:
    return {v: k for k, v in DTYPES.items()}[dtype]


def resolve_model_path(model: str) -> tuple[str, str | None]:
    """Prefer a local copy in models/<name> (from scripts/download_model.sh), else the Hub with pinned revision."""
    local = REPO_ROOT / "models" / model.split("/")[-1]
    if (local / "config.json").exists():
        return str(local), None
    if Path(model).exists():
        return model, None
    return model, PINNED_REVISIONS.get(model)


def load(model: str = DEFAULT_MODEL, dtype: str = "auto", attn_impl: str = "sdpa", device: str = "cuda"):
    path, revision = resolve_model_path(model)
    torch_dtype = resolve_dtype(dtype)
    processor = AutoProcessor.from_pretrained(path, revision=revision)
    net = AutoModelForImageTextToText.from_pretrained(
        path, revision=revision, dtype=torch_dtype, attn_implementation=attn_impl
    ).to(device)
    net.eval()
    return net, processor


def build_inputs(processor, image: Image.Image | None, prompt: str, batch_size: int = 1, device: str = "cuda"):
    """Apply the chat template and run the (CPU) processor. Returns a BatchFeature on `device`."""
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    conversation = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    kwargs = {"text": [text] * batch_size, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = [[image]] * batch_size
    return processor(**kwargs).to(device)


def describe_inputs(inputs, image_token_id: int) -> dict:
    """Token/tile counts that determine the workload size."""
    ids = inputs["input_ids"]
    info = {
        "batch_size": ids.shape[0],
        "prompt_tokens": ids.shape[1],
        "image_tokens": int((ids[0] == image_token_id).sum()),
        "vision_tiles": 0,
        "vision_patches": 0,
    }
    if "pixel_values" in inputs:
        info["vision_tiles"] = inputs["pixel_values"].shape[0] // ids.shape[0]
        info["vision_patches_padded"] = inputs["pixel_values"].shape[0] * inputs["pixel_values"].shape[1]
        info["vision_patches"] = int(inputs["pixel_attention_mask"].sum())
    return info
