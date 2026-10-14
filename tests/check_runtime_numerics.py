"""Teacher-forced numerics check of the O1 runtime (src/lfm2opt/runtime.py) against Hugging Face generate().

HF generates N greedy tokens (min_new_tokens=N, raw logits kept). Our runtime then decodes the *same* token
sequence (each step's input token is forced to HF's), so both see identical contexts and their per-step logits
can be compared directly: max |logit difference|, argmax agreement, and HF's top-1/top-2 margin wherever the
argmax differs (a near-tie flip is a numerics effect, a large-margin flip would be a bug).

    python tests/check_runtime_numerics.py --workloads text,img_small,img_fhd --gen 128
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.runtime import StaticDecoder  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="text,img_small,img_fhd")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fused", action="store_true", help="check the fused-kernel runtime (O2-O4)")
    ap.add_argument("--gemv", default="cublas", choices=["cublas", "bf16", "int8", "int4"], help="fused path GEMV (O6: int8)")
    ap.add_argument("--w4-group", type=int, default=128)
    ap.add_argument("--w4-clip", action="store_true", help="int4: per-group clipping search")
    ap.add_argument("--w4-int8", default="", help="int4: weight kinds kept in INT8 (qkv,o,in,out,w13,w2,head)")
    ap.add_argument("--fused-prefill", action="store_true", help="prompt through our prefill kernels (O3b)")
    ap.add_argument("--vision-fused", action="store_true", help="fused SigLIP2 encoder (O7d)")
    ap.add_argument("--awq", action="store_true", help="int4: AWQ scales calibrated on variants.CALIB_PROMPTS")
    args = ap.parse_args()

    net, processor = load()
    calib = None
    if args.awq:
        from lfm2opt.runtime import collect_calibration
        from lfm2opt.variants import CALIB_PROMPTS

        img = WORKLOADS["img_small"].image()
        calib = collect_calibration(net, [(build_inputs(processor, img if i % 2 == 0 else None, p), 64)
                                          for i, p in enumerate(CALIB_PROMPTS)])
    dec = StaticDecoder(net, keep_logits=True, fused=args.fused or args.gemv != "cublas", gemv=args.gemv,
                        w4_group=args.w4_group, w4_clip=args.w4_clip,
                        w4_keep_int8=tuple(k for k in args.w4_int8.split(",") if k), calib=calib,
                        fused_prefill=args.fused_prefill)
    if args.vision_fused:  # HF reference keeps HF's vision tower: compare to an unpatched copy of the forward
        from lfm2opt.vision import FusedSiglip2Encoder
        venc = FusedSiglip2Encoder(net.model.vision_tower)
    report = {}
    for name in args.workloads.split(","):
        wl = WORKLOADS[name]
        inputs = build_inputs(processor, wl.image(), wl.prompt)
        T = inputs["input_ids"].shape[1]
        hf = net.generate(**inputs, max_new_tokens=args.gen, min_new_tokens=args.gen, do_sample=False,
                          pad_token_id=processor.tokenizer.pad_token_id, output_logits=True,
                          return_dict_in_generate=True)
        seq = hf.sequences[:, T:]                                   # [1, gen]
        hf_logits = torch.stack(hf.logits, dim=1)[0].float()        # [gen, V] raw logits per step

        model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        if args.vision_fused:
            vt = net.model.vision_tower
            orig, vt.forward = vt.forward, venc
        first, s = dec.prefill(inputs["input_ids"], args.gen, attention_mask=inputs.get("attention_mask"),
                               **model_inputs)
        if args.vision_fused:
            vt.forward = orig
        first_diff = (dec.last_prefill_logits[0].float() - hf_logits[0]).abs().max().item() \
            if getattr(dec, "last_prefill_logits", None) is not None else float("nan")
        diffs, agree, flips = [], 0, []
        for t in range(1, args.gen):
            s["tok"].copy_(seq[:, t - 1 : t])                       # force HF's token as this step's input
            s["graph"].replay()
            ours = s["logits"][0]
            ref = hf_logits[t]
            diffs.append((ours - ref).abs().max().item())
            if ours.argmax() == ref.argmax():
                agree += 1
            else:
                top2 = ref.topk(2).values
                flips.append(dict(step=t, hf_margin=(top2[0] - top2[1]).item(), max_abs_diff=diffs[-1]))
        d = torch.tensor(diffs)
        report[name] = dict(prompt_tokens=T, steps=args.gen - 1, argmax_agree=agree,
                            max_abs_logit_diff=d.max().item(), median_abs_logit_diff=d.median().item(),
                            first_token_match=bool(first[0] == seq[0, 0]), first_max_abs_logit_diff=first_diff,
                            flips=flips)
        print(f"{name:9s} T={T:5d}  argmax agree {agree}/{args.gen - 1}  max|dlogit| {d.max():.3f}  "
              f"median {d.median():.3f}  first token {'ok' if first[0] == seq[0, 0] else 'DIFF'} "
              f"(max|dlogit| {first_diff:.3f})  flips at near-ties: {[round(f['hf_margin'], 3) for f in flips]}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
