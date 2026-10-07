"""Fixed benchmark workloads. Same inputs on every GPU so results are comparable.

Input size is what matters for performance: a single 512x512-class image goes through the
vision tower as one tile, larger images are split into 512x512 tiles + a thumbnail. Larger
inputs are made by resizing one real photo so the content stays the same.
"""

from dataclasses import dataclass

from PIL import Image

from .env import REPO_ROOT

SOURCE_IMAGE = REPO_ROOT / "assets" / "images" / "coco_000000039769.jpg"  # 640x480, COCO val2017

IMAGE_PROMPT = "Describe this image in detail."
TEXT_PROMPT = "Explain in detail how a bicycle works."


@dataclass(frozen=True)
class Workload:
    name: str
    image_size: tuple[int, int] | None  # (width, height); None = text-only
    prompt: str
    description: str

    def image(self) -> Image.Image | None:
        if self.image_size is None:
            return None
        img = Image.open(SOURCE_IMAGE).convert("RGB")
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.Resampling.BICUBIC)
        return img


WORKLOADS = {
    w.name: w
    for w in [
        Workload("text", None, TEXT_PROMPT, "Text only: language model alone"),
        Workload("img_small", (640, 480), IMAGE_PROMPT, "VGA photo: single tile"),
        Workload("img_hd", (1280, 720), IMAGE_PROMPT, "720p: multi-tile + thumbnail"),
        Workload("img_fhd", (1920, 1080), IMAGE_PROMPT, "1080p: multi-tile + thumbnail"),
    ]
}
