# GPU sessions: how we use our 2-hour slots

_Measurement GPU: the course **"RTX 5000" server** (team account `team21`, SSH from our own machine), **three 2-hour slots per week**. Development and rehearsal happen on Colab T4 (unlimited). Replaces the earlier L40 plan (6 h/week) and the original RTX PRO 5000 plan (8 h/week). Updated 2026-10-13._

**Never put the team password in a file, a commit, a chat log or a script.** It is typed once by a person (`ssh-copy-id`); afterwards everything uses our SSH key.

---

## 1. Which GPU is it?

"RTX 5000" can mean three quite different cards. `scripts/preflight.sh` tells us on the first login; the plan works for all three, only the numbers change.

| | Quadro RTX 5000 | RTX 5000 Ada | **RTX PRO 5000 Blackwell** |
| --- | --- | --- | --- |
| Architecture / CC | Turing, sm_75 | Ada, sm_89 | Blackwell, sm_120 |
| Memory / bandwidth | 16 GB / 448 GB/s | 32 GB / 576 GB/s | 48 GB / 1344 GB/s |
| BF16 / FP8 / FP4 tensor cores | no / no / no | yes / yes / no | yes / yes / yes |
| Decode floor (2.34 GB/token) | 5.2 ms | 4.1 ms | 1.7 ms |
| Baseline dtype | FP16 | BF16 | BF16 |
| PyTorch wheel | any | any | needs CUDA ≥ 12.8 build (handled by `setup_env.sh`) |

Same conclusion in all three cases: decode is launch-overhead bound on the T4 already, and a faster GPU shrinks the GPU work but not the ~790 CPU-side launches per token, so O1 (CUDA graphs) and fusion matter even more.

## 2. Rules (why this is hard to get wrong)

1. **Nothing is debugged on the server.** Every slot preset is first run end to end on Colab with the same command and the same deadline ("dress rehearsal"). New, unrehearsed code is never in a slot.
2. **One command per slot.** A slot is a preset of `scripts/run_suite.sh` (`SUITE=slot1`, …) started with `remote.sh slot`. It runs in **tmux** on the server, so a dropped SSH connection or a closed laptop changes nothing.
3. **Hard deadline built in.** The job gets `SLOT_MINUTES`; a step starts only if its estimate fits, every step has a timeout (2× estimate), and whatever didn't fit is listed as `SKIPPED` at the end of the log → first in the next slot. A slot can't overrun, and a hung step can't eat it.
4. **Results leave the server every 10 minutes** (`remote.sh watch` syncs while it waits) and once more at the end. Nothing important ever exists only on the server.
5. **Timing runs alone.** Decode is CPU/launch-bound: any other CPU load (pip, builds) skews TPOT/energy. Builds and downloads are only allowed during Nsight Compute steps (they replay kernels in isolation).
6. **Check for other users** (`remote.sh check`) before starting; the suite also logs every other process on the GPU. Timing done while someone else uses the GPU is flagged as invalid.
7. **Fixed roles per slot:** the *driver* runs the commands; the *scribe* keeps the session log (times, what ran, anomalies) which becomes the experiment-log entry.
8. **Setup never uses slot time if we can avoid it:** if login outside slots is allowed, `remote.sh setup` (env + 3.2 GB weights) runs the day before.

## 3. One-time setup (before slot 1)

- [ ] Get the server address (and port / VPN requirement) from the course staff.
- [ ] `~/.ssh/config` entry `gpu` (template at the top of `scripts/remote/remote.sh`). If the server refuses keys, add `ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 4h`: then one person logs in once per slot with the password and all scripts reuse that connection.
- [ ] `ssh-copy-id gpu` (a person types the password once), then `bash scripts/remote/remote.sh check`.
- [ ] If login outside slots is allowed: `remote.sh push && remote.sh setup` now (installs everything, saves a preflight report). If not: setup is the first ~15 min of slot 1 (already budgeted).
- [ ] Backup for a server without internet: download the weights to our machine (`bash scripts/download_model.sh`, ~12 min here) so they can be `rsync`ed over.
- [ ] Dress rehearsal of `SUITE=slot1` on Colab with the same deadline (running 2026-10-13).

## 4. Slot 1: baseline on the server (what we do in the first 2 hours)

Assumes setup could **not** be done beforehand (worst case). If it was, the job gets 100 minutes instead of 85 and more of the list fits.

| Clock | What | Command (from our machine) | Done when |
| --- | --- | --- | --- |
| 0:00–0:05 | Connect; GPU model, other users | `bash scripts/remote/remote.sh check` | GPU listed, no foreign processes |
| 0:05–0:07 | Send code | `remote.sh push` | rsync summary |
| 0:07–0:20 | Preflight + venv + weights + profilers (idempotent) | `remote.sh setup` | two preflight reports, `matmul ok`, model downloaded |
| 0:20–0:23 | Smoke test | `remote.sh run "python benchmarks/bench.py --workloads img_small --gen-lens 1,16 --repeats 2 --idle-seconds 2 --tag smoke"` | sensible caption printed |
| 0:23 | **Start the slot job** (deadline 85 min) | `remote.sh slot slot1 85` then `remote.sh watch slot1_<HHMM>` | — |
| 0:23–1:48 | Runs by itself, in priority order: benchmark grid → batch sweep → BF16-vs-FP32 outputs → hook overhead → summaries → PyTorch-profiler breakdowns (3 inputs) → Nsight Systems (2) → attention-path probe → Nsight Compute decode step → vision → prefill (whatever fits) | (watch syncs every 10 min) | `DONE / FAILED / SKIPPED` lines in the log |
| 1:48–1:55 | Final sync + check | `remote.sh sync`; open `results/raw/*_<GPU>_baseline_full/summary.md` | summaries present locally |
| 1:55–2:00 | Leave clean | `remote.sh jobs` (nothing left), `remote.sh check` (GPU idle) | — |
| after | Commit results, experiment-log entry, set `EST_SCALE` from measured step times | — | — |

Expected (T4-based estimates; the server should be faster): the timing + profiling part takes ~32 min, the attention probe ~4, Nsight Compute decode ~28 and vision ~16 → about 80 min, so prefill ncu probably moves to slot 2.

## 5. Following slots

| Slot | Content | Preset / command | Needs beforehand (rehearsed on Colab) |
| --- | --- | --- | --- |
| 2 | Rest of the profiling: anything SKIPPED in slot 1, then Nsight Compute prefill (both inputs), 1080p vision, deep decode, long-context decode | `remote.sh slot slot2 100` (+ `STEPS=` for carry-over) | — |
| 3 | Reference baselines O0 (timing, alone): `causal-conv1d` installed, HF `torch.compile` + static cache, vLLM, llama.cpp | new preset `slot3` | the O0 scripts (not written yet) |
| 4+ | Optimizations: per variant `SUITE=variant` (~20 min on T4 → ~3–4 variants per slot) + targeted Nsight Compute of the new kernels; then the cumulative stack, 3B check, final numbers | `VARIANT=<name> SUITE=variant` | each variant passes `tests/` and matches baseline outputs on Colab |

Rhythm with 3 slots a week: **slot A** measures what was finished; **between slots** we develop on Colab; **slot B/C** measure the next batch. A slot is never used to find out whether something works.

## 6. Failure playbook

| Problem | Action |
| --- | --- |
| Can't connect | check VPN/campus network; `ssh -v gpu`; if the staff confirm the slot is ours, report it immediately (slot time is lost otherwise) |
| Another process on the GPU | still run ncu/profiling steps (ratios stay valid); timing steps are flagged; tell the staff |
| No internet on the server | `rsync` weights from our machine (§3); pip wheels the same way; ask staff for a proxy |
| Old driver / CUDA | `setup_env.sh` picks a matching wheel (cu126/cu128/cu130) and logs the torch version |
| `ERR_NVGPUCTRPERM` (no ncu) | suite skips ncu steps automatically; ask staff to enable profiling for users |
| SSH drops | job is in tmux; reconnect, `remote.sh log <job> 40` |
| A step fails | it is listed as FAILED; the rest continues; we look at the log after the slot, not during |
| Slot ending | nothing to do: the deadline stops the job; final `sync` |

## 7. Command cheat-sheet

```bash
R=scripts/remote/remote.sh
bash $R check                     # who/what is on the GPU
bash $R push && bash $R setup     # code + env + weights + profilers (+ preflight report)
bash $R slot slot1 85             # start a slot preset with a hard deadline (minutes)
bash $R watch <job>               # sync every 10 min, show progress, stop when the job ends
bash $R log <job> 40              # last lines of a job
bash $R stop <job>                # stop a job cleanly
bash $R sync                      # results back to our machine (repeat freely)
```

## 8. Open questions

1. Server address / port; VPN or campus network needed?
2. Which card exactly (Quadro RTX 5000, RTX 5000 Ada, RTX PRO 5000 Blackwell)?
3. Slot days and times; is the GPU exclusive to us during the slot? Can we log in outside slots (setup only)?
4. sudo? (Nsight Compute counters, clock locking.) Internet? Home directory persistent between slots, quota?
5. Date of the first slot and of Evaluation 1.
