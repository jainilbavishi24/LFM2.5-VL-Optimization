# L40 sessions: how we use our 6 hours per week

_Our measurement GPU is an **NVIDIA L40 (48 GB)**, reached over SSH from our own machine, in one **6-hour slot per team per week**. This replaced the RTX PRO 5000 we planned with earlier. Everything is developed and rehearsed on Colab T4 first; the L40 slot is only for measuring. Created 2026-10-11._

## 1. The L40 and what it changes for us

| | Colab T4 (dev) | **L40 (measurement)** | Jetson Orin (edge, later) |
| --- | --- | --- | --- |
| Architecture | Turing, sm_75 | **Ada Lovelace, sm_89** | Ampere, sm_87 |
| Memory / bandwidth | 15 GB / 320 GB/s | **48 GB GDDR6 / 864 GB/s** | 8–64 GB shared / 68–205 GB/s |
| SMs / L2 | 40 / 4 MB | **142 / 96 MB** | 8–16 / 4 MB |
| Tensor-core precisions | FP16, INT8 | **FP16, BF16, TF32, FP8 (E4M3/E5M2), INT8** | FP16, BF16, INT8 |
| Power limit | 70 W | **300 W** (passively cooled server card) | 7–60 W |
| FlashAttention-2 / SDPA flash backend | no | **yes** | yes |
| Decode floor (2.34 GB of weights per token) | 7.3 ms | **2.7 ms (≈370 tok/s)** | 11–34 ms |

Consequences for the plan ([optimization_plan.md](optimization_plan.md)):

- **Precision:** baseline in **BF16** (native on sm_89); FP16 only on the T4.
- **Decode will be even more overhead-bound than on the T4.** GPU work per token shrinks ~3× (2.7 ms floor vs 7.3), while the ~790 kernel launches per token cost about the same CPU time. So O1 (CUDA graphs) and O2–O4 (fusion) matter more on the L40.
- **Attention:** the SDPA flash backend exists on sm_89, so the FP32 math fallback we saw on the T4 may disappear or change. `profiling/sdpa_probe.py` checks this in session 1 (Q5.1, Q5.2).
- **Low precision (O6):** **FP8 weights are possible on the L40** (Ada FP8 tensor cores; `torch._scaled_mm` supports sm_89). NVFP4 is Blackwell-only and is dropped from the plan. INT8 / INT4 weight-only GEMV stays (also needed for the Jetson).
- **96 MB L2** is large: prefill activations and some per-layer weights fit. Watch for this in ncu (L2 hit rate) when comparing with the T4.
- Builds: compile our kernels for `sm_75` (T4), `sm_89` (L40), `sm_87` (Jetson).

## 2. Rules for a slot

1. **Nothing gets debugged on the L40.** Every command we run in a slot has been run, end to end, on Colab with the same scripts (`scripts/run_suite.sh`, same flags) in the days before. New code without a Colab rehearsal goes in the P3 "flex" block only.
2. **The GPU is never idle.** Jobs run in tmux via `scripts/remote/remote.sh bg`, queued back to back; we prepare the next command while one runs.
3. **Timing runs alone.** Decode is CPU/launch-bound, so any CPU load (pip installs, builds, downloads) skews TPOT and energy. Rule:
   - timing blocks (`SUITE=timing`, `SUITE=variant`, reference benchmarks): **nothing else runs on the machine**;
   - Nsight Compute blocks (`SUITE=ncu`) replay kernels in isolation, so builds/installs/downloads are scheduled **during** them.
4. **Never leave results only on the server.** `remote.sh sync` after every block (it is incremental), and a final sync + local check before the slot ends.
5. **Priorities decide what gets dropped.** Each block is tagged P1 (must have) / P2 (should) / P3 (if time). If something fails, skip to the next block and note it; don't debug for more than 10 minutes.
6. **One person drives, one person watches the clock and writes the session log** (times, what ran, return codes, anomalies) → experiment log afterwards.
7. **Check who else is on the GPU** (`remote.sh check`) at the start and before every timing block. If another process is using the GPU, timing/energy numbers are invalid: do ncu/builds first and timing later, and record it.

## 3. One-time setup (before the first slot)

- [ ] Get from the course staff: server address, username, port, how the slot is booked, whether home-dir files persist between slots, sudo or not (see §8).
- [ ] Add the `l40` entry to `~/.ssh/config` (template at the top of `scripts/remote/remote.sh`), copy our public key, test `bash scripts/remote/remote.sh check`.
- [ ] If we may log in outside our slot: run `remote.sh push` and `remote.sh setup` early (installs the venv and downloads the weights without using GPU time). If not, setup is the first 20 minutes of slot 1.
- [ ] Write and rehearse on Colab what session 1 needs that doesn't exist yet: the `hf_compile` variant (O0b), a vLLM benchmark script (O0c), a llama.cpp benchmark script (O0d), and `profiling/sdpa_probe.py` (written, not yet run).
- [ ] Rehearse the whole session-1 command list on Colab (§5 prep checklist).

## 4. Session 1: baseline on the L40 (before Evaluation 1)

Goal: L40 baseline + bottleneck evidence (same suite as the T4), the attention-backend question, and the reference baselines (O0), so Eval 1 has a cross-GPU story.

Time estimates are from the T4 runs; the L40 should be faster for timing steps but similar for ncu (dominated by replay overhead). They are re-calibrated after this session.

| Time | Block | Prio | Command (from our machine) | Notes |
| --- | --- | --- | --- | --- |
| 0:00–0:20 | Connect, push, setup | P1 | `remote.sh check` → `remote.sh push` → `remote.sh setup` | preflight report saved; ~3 min if the venv/weights persisted |
| 0:20–0:25 | Smoke test | P1 | `remote.sh run "python benchmarks/bench.py --workloads img_small --gen-lens 1,16 --repeats 2 --idle-seconds 2 --tag smoke"` | output text must be sensible |
| 0:25–1:00 | **Timing block** (alone on the machine) | P1 | `remote.sh bg timing "SUITE=timing bash scripts/run_suite.sh"` | grid, batch sweep, BF16-vs-FP32, hook overhead (~19 min on T4) |
| 1:00–1:15 | Profile block | P1 | `remote.sh bg profile "SUITE=profile bash scripts/run_suite.sh"` → `sync` | torch profiler ×3, nsys ×2 |
| 1:15–1:25 | Attention backend probe | P1 | `remote.sh bg sdpa "python profiling/sdpa_probe.py"` | answers Q5.1 / Q5.2 / O5a |
| 1:25–3:25 | **Nsight Compute block** | P1 decode+vision, P2 rest | `remote.sh bg ncu "SUITE=ncu bash scripts/run_suite.sh"` | ~2 h on T4. **In parallel (CPU only):** build the `causal-conv1d` wheel (`pip wheel … -w wheels/`, **don't install yet**: it would change the baseline mid-run), build llama.cpp with CUDA, create the vLLM venv, download the GGUF. `sync` every 30 min |
| 3:25–4:25 | Reference baselines O0 (alone) | P2 | install the causal-conv1d wheel → `SUITE=variant TAG=ref_causalconv1d`; `VARIANT=hf_compile`; vLLM and llama.cpp benchmark scripts | gives the "existing best" numbers |
| 4:25–5:15 | Flex | P3 | first optimization variants that passed Colab rehearsal (O1/O2) with `SUITE=variant`; or 3B baseline (G4); or reruns of failed blocks | |
| 5:15–5:45 | Final sync and wrap-up | P1 | `remote.sh sync`, compare file counts, kill leftover tmux jobs, `nvidia-smi` clean | then commit results and write the log entry |
| 5:45–6:00 | Buffer | | | |

If the GPU counters are not accessible to us (no sudo), `run_suite.sh` skips the ncu steps automatically. Then the ncu block becomes: reference baselines (needs exclusivity for timing) + flex, and the CPU-side builds move into the setup block.

## 5. Later sessions (one per week; to be adjusted after each)

| Slot | Course week | Goal | Main blocks |
| --- | --- | --- | --- |
| 2 | 6 | First optimizations | per variant (O1 CUDA graphs, O2 fused RMSNorm, O3 fused short-conv): `SUITE=variant` (~30 min each) + targeted ncu of the new kernels (`profile_ncu.sh` with small `LAUNCHES`); block-size sweeps of our kernels for sm_89 |
| 3 | 7 (Eval 2) | Working prototype numbers | cumulative stack O1+O2+O3(+O4) full `SUITE=all`; output-agreement checks; before/after nsys timelines for the slides; O5 attention |
| 4 | 8 | Precision + vision + generalization | O6 INT8/FP8 weight GEMV, O7 vision attention, LFM2.5-VL-3B baseline vs final stack, quality eval set |
| 5 | 9 (Final) | Final numbers + demo | final stack full suite (all workloads, batch sweep, energy), anything missing, record the demo |

Rough capacity: with ~1 h of setup/sync/buffer, a slot fits **about 8 variant runs** (`SUITE=variant`), or one full baseline-style suite plus ~4 variants.

**Prep checklist before every slot (on Colab, the days before):**

- [ ] Every command of the slot plan runs end to end on the T4 (`colab.sh bg`), with the exact flags.
- [ ] New variants pass their tests in `tests/` and match baseline outputs (`compare_outputs.py`).
- [ ] The slot's command list is written down in order, with priorities, in the experiment log draft.
- [ ] Code committed and pushed; `remote.sh push` sends exactly that commit (recorded in every result).

## 6. Failure playbook

| Problem | What to do |
| --- | --- |
| Another process on the GPU | ncu + CPU-side work first, timing later; if it persists, run timing anyway and flag those results as "shared GPU" |
| No internet on the server | push the venv requirements as wheels and the model weights from our machine (`rsync`); ask staff for a proxy/mirror |
| Driver too old for torch cu130 | `setup_env.sh` picks cu128/cu126 automatically; note the torch version (results record it) |
| `ERR_NVGPUCTRPERM` (no ncu) | automatic skip; ask staff whether ncu can be enabled (`NVreg_RestrictProfilingToAdminUsers=0`) or run with sudo |
| SSH drops | jobs are in tmux; reconnect, `remote.sh jobs` / `log` |
| Disk quota | delete `profiling/reports/*/*.ncu-rep` on the server after `sync` (they are large) |
| A step crashes | `remote.sh log <job> 100`, note it, continue with the next block |

## 7. Commands cheat-sheet

```bash
R=scripts/remote/remote.sh
bash $R check                      # connectivity, GPU, other users
bash $R push && bash $R setup      # code + env + weights + profilers (+ preflight report)
bash $R bg <job> "<command>"       # start in tmux on the server
bash $R log <job> 30               # follow progress
bash $R jobs                       # what is running
bash $R sync                       # copy results/raw + profiling/reports back (repeat freely)
bash $R shell                      # interactive shell on the server
```

## 8. Open questions for the course staff / team

1. Server address, username, port; how do we get our public key onto it?
2. Is the 6-hour slot one contiguous block? Fixed day/time each week? Which weeks (how many slots before Eval 1, Eval 2, Final)?
3. Can we log in (without using the GPU) outside our slot to install the environment and download weights? Do files in our home directory persist between slots? Disk quota?
4. Is the GPU exclusive to us during the slot, or shared with other teams/users?
5. sudo / root? Needed for Nsight Compute counters (unless the admin has enabled profiling for all users) and for locking clocks / setting the power limit.
6. Internet access from the server (pip, Hugging Face, GitHub)?
7. Is `nvcc` (CUDA toolkit) installed, and which version? Docker / conda available?
8. Is `tmux` available (our scripts use it to survive SSH drops)?
