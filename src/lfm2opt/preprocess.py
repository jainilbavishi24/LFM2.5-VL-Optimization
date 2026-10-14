"""GPU image preprocessing for LFM2-VL: drop-in replacement for the HF image processor.

HF's processor resizes, tiles, normalizes and patchifies on the CPU (69 ms for a 1080p image on the server's
Core Ultra 9, more than the whole vision tower). Here the uint8 image is copied to the GPU once and two kernels
(kernels/image_prep.cu) produce the padded `pixel_values` directly: an antialiased bilinear horizontal pass into a uint8
buffer, then the vertical pass fused with rescale/normalize, tile split and 16x16 patchify. The int16 fixed-point
weights are computed here with the same formula as ATen's uint8 resize, so the output is bit-identical to the CPU path.

    processor = GpuImageProcessor.wrap(processor)   # same call signature, pixel_values come back on the GPU
"""

import copy
import math
from functools import lru_cache

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.lfm2_vl.image_processing_lfm2_vl import Lfm2VlImageProcessor

from .kernels import prep_ext


def aa_bilinear_weights(in_size: int, out_size: int):
    """ATen's antialiased bilinear weights (uint8 path): per output index (xmin, xsize) and int16 weights.

    Mirrors HelperInterpLinear::_compute_index_ranges_weights<double> + _compute_index_ranges_int16_weights
    (aten/src/ATen/native/cpu/UpSampleKernel.cpp), which follows Pillow's resample.
    """
    scale = in_size / out_size
    support = scale if scale >= 1.0 else 1.0  # interp_size (2) * 0.5 * scale
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    max_interp = int(math.ceil(support)) * 2 + 1
    bounds = np.zeros((out_size, 2), np.int32)
    wf = np.zeros((out_size, max_interp), np.float64)
    wt_max = 0.0
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(max(min(int(center + support + 0.5), in_size) - xmin, 0), max_interp)
        ws = [max(0.0, 1.0 - abs((j + xmin - center + 0.5) * invscale)) for j in range(xsize)]
        total = sum(ws)
        if total != 0.0:
            ws = [w / total for w in ws]
            wt_max = max(wt_max, max(ws))
        bounds[i] = (xmin, xsize)
        wf[i, :xsize] = ws
    prec = 0
    while prec < 22 and int(0.5 + wt_max * (1 << (prec + 1))) < (1 << 15):
        prec += 1
    v = wf * (1 << prec)
    wi = np.where(v < 0, np.trunc(v - 0.5), np.trunc(v + 0.5)).astype(np.int16)
    return bounds, wi, prec


@lru_cache(maxsize=64)
def _weights_on_gpu(in_size: int, out_size: int):
    bounds, w, prec = aa_bilinear_weights(in_size, out_size)
    return torch.from_numpy(bounds).cuda(), torch.from_numpy(w).cuda(), prec


class GpuImageProcessor(Lfm2VlImageProcessor):
    """Lfm2VlImageProcessor whose resize/normalize/patchify runs as fused CUDA kernels (PIL / uint8 HWC input)."""

    @classmethod
    def wrap(cls, processor):
        """Shallow copy of an Lfm2VlProcessor with this image processor in place of the HF one."""
        gp = cls.__new__(cls)
        gp.__dict__.update(processor.image_processor.__dict__)
        gp._staging = None
        gp._copied = None
        out = copy.copy(processor)
        out.image_processor = gp
        return out

    def _upload(self, img) -> torch.Tensor:
        """PIL / numpy HWC uint8 -> CUDA [H, W, 3] through a reused pinned buffer."""
        if hasattr(img, "convert") and img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img)
        n = arr.size
        if self._copied is not None:
            self._copied.synchronize()  # the previous upload still reads the staging buffer
        if self._staging is None or self._staging.numel() < n:
            self._staging = torch.empty(max(n, 1920 * 1080 * 3), dtype=torch.uint8, pin_memory=True)
        stage = self._staging[:n].view(arr.shape)
        stage.numpy()[...] = arr
        dev = torch.empty(arr.shape, dtype=torch.uint8, device="cuda")
        dev.copy_(stage, non_blocking=True)
        self._copied = torch.cuda.Event()
        self._copied.record()
        return dev

    def _layout(self, height: int, width: int):
        """Same decisions as Lfm2VlImageProcessor.resize_and_split: list of resize jobs + rows/cols/size."""
        new_w, new_h = self.smart_resize(height, width, self.downsample_factor, self.min_image_tokens,
                                         self.max_image_tokens, self.encoder_patch_size)
        large = self._is_image_too_large(height, width, self.max_image_tokens, self.encoder_patch_size,
                                         self.downsample_factor, self.max_pixels_tolerance)
        split = self.do_image_splitting and not self.min_tiles == self.max_tiles == 1
        if large and split:
            gw, gh, tw, th, _ = self._get_grid_layout(height, width, self.min_tiles, self.max_tiles, self.tile_size)
            jobs = [(th, tw, self.tile_size, self.tile_size, gw, gw * gh)]
            if self.use_thumbnail and gw * gh != 1:
                jobs.append((new_h, new_w, new_h, new_w, 1, 1))
            return jobs, gh, gw, [new_h, new_w]
        return [(new_h, new_w, new_h, new_w, 1, 1)], 1, 1, [new_h, new_w]

    def __call__(self, images, **kwargs):
        assert self.resample in (2, None) and self.do_rescale and self.do_normalize and self.do_pad
        flat = [im for sub in images for im in sub] if isinstance(images, (list, tuple)) and images and \
            isinstance(images[0], (list, tuple)) else list(images if isinstance(images, (list, tuple)) else [images])
        ps = self.encoder_patch_size
        max_patches = max(self.max_image_tokens * self.downsample_factor**2, (self.tile_size // ps) ** 2)
        mean = [m / self.rescale_factor for m in self.image_mean]
        std = [s / self.rescale_factor for s in self.image_std]
        ext = prep_ext()

        plans, n_tiles = [], 0
        for img in flat:
            w, h = img.size if hasattr(img, "size") and not isinstance(img, np.ndarray) else (img.shape[1], img.shape[0])
            jobs, rows, cols, size = self._layout(h, w)
            plans.append((img, h, w, jobs, rows, cols, size))
            n_tiles += sum(j[5] for j in jobs)

        out = torch.empty((n_tiles, max_patches, 3 * ps * ps), dtype=torch.float32, device="cuda")
        mask = torch.zeros((n_tiles, max_patches), dtype=torch.int32, device="cuda")
        shapes, rows_l, cols_l, sizes_l, t0 = [], [], [], [], 0
        for img, h, w, jobs, rows, cols, size in plans:
            dev = self._upload(img)
            for oh, ow, tile_h, tile_w, tiles_x, ntiles in jobs:
                hb, hw, hp = _weights_on_gpu(w, ow)
                vb, vw, vp = _weights_on_gpu(h, oh)
                tmp = torch.empty((h, ow, 3), dtype=torch.uint8, device="cuda")
                ext.resize_h(dev, tmp, hb, hw, hp)
                ext.resize_v_patchify(tmp, out, vb, vw, vp, oh, tile_h, tile_w, tiles_x, t0, mean, std)
                n = (tile_h // ps) * (tile_w // ps)
                if n < max_patches:
                    out[t0:t0 + ntiles, n:].zero_()
                mask[t0:t0 + ntiles, :n] = 1
                shapes += [[tile_h // ps, tile_w // ps]] * ntiles
                t0 += ntiles
            rows_l.append(rows)
            cols_l.append(cols)
            sizes_l.append(size)
        data = {"pixel_values": out, "pixel_attention_mask": mask,
                "spatial_shapes": torch.tensor(shapes, dtype=torch.int64, device="cuda"),
                "image_rows": rows_l, "image_cols": cols_l, "image_sizes": sizes_l}
        return BatchFeature(data=data)
