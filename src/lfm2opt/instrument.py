"""Phase-level instrumentation of LFM2-VL via module hooks.

The same hook points serve three tools, so all of them agree on phase boundaries:
  * "events":          CUDA events (GPU-stream time per phase, no host sync) + host timestamps
  * "nvtx":            NVTX ranges for Nsight Systems / Nsight Compute (--nvtx-include "lm_decode/")
  * "record_function": torch.profiler annotations (used to attribute kernels to phases)

Phases:
  vision_tower  - SigLIP2 encoder, all tiles in one call (prefill only)
  projector     - pixel-unshuffle + MLP, called once per tile (prefill only)
  lm_prefill    - first language-model forward of a request
  lm_decode     - every later language-model forward (one per generated token)
  lm_head       - logits projection of a decode step
  lm_head_prefill - logits projection of the prefill step
"""

import time
from contextlib import contextmanager

import torch


class PhaseInstrumentor:
    def __init__(self, net, modes=("events",)):
        self.modes = set(modes)
        self.cuda = torch.cuda.is_available()
        self.modules = {
            "vision_tower": net.model.vision_tower,
            "projector": net.model.multi_modal_projector,
            "lm": net.model.language_model,
            "lm_head": net.lm_head,
        }
        self._handles = []
        self._rf_stack = []
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        """Call before every request: the first LM call after a reset is the prefill."""
        self.records = []  # (phase, start_event, end_event, host_t_start)
        self._open = {}
        self._lm_calls = 0

    def attach(self):
        for key, mod in self.modules.items():
            self._handles.append(mod.register_forward_pre_hook(self._pre(key)))
            self._handles.append(mod.register_forward_hook(self._post(key)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextmanager
    def attached(self):
        self.attach()
        try:
            yield self
        finally:
            self.detach()

    # ---------------------------------------------------------------- hooks
    def _phase(self, key):
        if key == "lm":
            return "lm_prefill" if self._lm_calls == 0 else "lm_decode"
        if key == "lm_head" and self._lm_calls <= 1:  # LM forward of the prefill step has just finished
            return "lm_head_prefill"
        return key

    def _pre(self, key):
        def hook(module, args):
            phase = self._phase(key)
            start = None
            if "events" in self.modes and self.cuda:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
            self._open[key] = (phase, start, time.perf_counter())
            if "nvtx" in self.modes and self.cuda:
                torch.cuda.nvtx.range_push(phase)
            if "record_function" in self.modes:
                rf = torch.profiler.record_function(phase)
                rf.__enter__()
                self._rf_stack.append(rf)

        return hook

    def _post(self, key):
        def hook(module, args, output):
            phase, start, host_t = self._open.pop(key)
            end = None
            if "events" in self.modes and self.cuda:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            self.records.append((phase, start, end, host_t))
            if "nvtx" in self.modes and self.cuda:
                torch.cuda.nvtx.range_pop()
            if "record_function" in self.modes:
                self._rf_stack.pop().__exit__(None, None, None)
            if key == "lm":
                self._lm_calls += 1

        return hook

    # ---------------------------------------------------------------- results
    def summary(self) -> dict:
        """GPU-stream milliseconds per phase. Synchronizes once."""
        if self.cuda:
            torch.cuda.synchronize()
        out = {"vision_ms": 0.0, "projector_ms": 0.0, "projector_calls": 0, "lm_prefill_ms": None,
               "lm_head_prefill_ms": None, "lm_decode_ms": [], "lm_head_ms": [], "decode_step_host_ms": []}
        lm_host_starts = []
        for phase, start, end, host_t in self.records:
            ms = start.elapsed_time(end) if (start is not None and end is not None) else None
            if phase == "vision_tower":
                out["vision_ms"] += ms or 0.0
            elif phase == "projector":
                out["projector_ms"] += ms or 0.0
                out["projector_calls"] += 1
            elif phase == "lm_prefill":
                out["lm_prefill_ms"] = ms
                lm_host_starts.append(host_t)
            elif phase == "lm_decode":
                out["lm_decode_ms"].append(ms)
                lm_host_starts.append(host_t)
            elif phase == "lm_head":
                out["lm_head_ms"].append(ms)
            elif phase == "lm_head_prefill":
                out["lm_head_prefill_ms"] = ms
        # Host-side interval between consecutive LM forwards = wall time of one generation step
        # (generate() syncs once per step on CUDA, so this tracks real step latency).
        out["decode_step_host_ms"] = [(b - a) * 1e3 for a, b in zip(lm_host_starts[:-1], lm_host_starts[1:])]
        return out
