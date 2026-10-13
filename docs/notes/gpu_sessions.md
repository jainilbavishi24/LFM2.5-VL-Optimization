# GPU sessions: how we use our 2-hour slots

_Measurement GPU: the course **"RTX 5000" server**, `team21@10.4.25.63`, team folder `/data/cs3406/team21`, **three booked 2-hour slots per week**. Development and dress rehearsals happen on Colab T4 (unlimited). Replaces the earlier L40 plan (6 h/week) and the original RTX PRO 5000 plan (8 h/week). Updated 2026-10-13._

**Never put the team password in a file, a commit, a chat log or a script.** A person types it once per slot to open the master SSH connection.

**Server rules (course staff), and what they force on us:**

| Rule | Consequence for our setup |
| --- | --- |
| Login works **only during our slot** ("Your account has expired" otherwise) | No setup in advance: slot 1 starts with setup (budgeted). Everything else must be ready beforehand. |
| At slot end we are logged out and **all our programs stop** (10-min warning first) | Hard deadline: our job stops scheduling work **12 min before the slot end**; results are synced continuously. |
| **Logging out early also stops everything**; nohup / screen / tmux don't survive | One SSH **master connection** is opened at the start and kept open for the whole slot; all our commands reuse it. A dropped connection kills the running step, so results are synced every 5 min and every step writes its own files. |
| Keep all files in `/data/cs3406/team21` (team-only) | Code, venv, model weights, caches (pip/uv/HF) and results all live there (`scripts/remote/remote.sh` sets this up), so slots 2+ reuse the setup in ~1 min. |

---

## 0. Deadline: Evaluation 1 starts Thursday 22 Oct 2026

Course staff (13 Oct): use the booked slots for **baseline experiments**; baseline results are expected at Eval 1; an unused booked slot may not be replaced. Eval 1 asks for: problem understanding, working baseline, reliable measurements, initial profiling, identified bottleneck, optimization strategy. T4 side is done; what is missing is the **server baseline**.

| Date | What | Must have? |
| --- | --- | --- |
| Tue 13 Oct | **Book slots**; Colab dress rehearsal of slot 1; all commands ready | yes |
| Wed 14 – Thu 15 Oct | **Slot 1**: setup + baseline timing + profiling + attention probe + ncu decode/vision (`slot1`) | **yes** |
| Fri 16 – Sat 17 Oct | **Slot 2**: whatever slot 1 skipped + remaining Nsight Compute (`slot2`) | yes |
| Mon 19 – Tue 20 Oct | **Slot 3**: buffer (re-run anything failed) or reference baselines (O0) if their scripts are rehearsed | buffer |
| Tue 20 – Wed 21 Oct | Results → plots, Eval 1 slides (T4 vs server, bottleneck evidence, plan); no GPU work planned | yes |
| Thu 22 Oct | Evaluation 1 | — |

Book the three slots as early in these windows as possible, so a failed slot can still be repeated before the 21st.

## 1. Which GPU is it?

"RTX 5000" can mean three quite different cards. `scripts/preflight.sh` tells us on the first login; the plan works for all three, only the numbers change.

| | Quadro RTX 5000 | RTX 5000 Ada | **RTX PRO 5000 Blackwell** |
| --- | --- | --- | --- |
| Architecture / CC | Turing, sm_75 | Ada, sm_89 | Blackwell, sm_120 |
| Memory / bandwidth | 16 GB / 448 GB/s | 32 GB / 576 GB/s | 48 GB / 1344 GB/s |
| BF16 / FP8 / FP4 tensor cores | no / no / no | yes / yes / no | yes / yes / yes |
| Decode floor (2.34 GB/token) | 5.2 ms | 4.1 ms | 1.7 ms |
| Baseline dtype (`--dtype auto`) | FP16 | BF16 | BF16 |
| PyTorch wheel | any | any | needs CUDA ≥ 12.8 build (handled by `setup_env.sh`) |

Same conclusion in all three cases: decode is launch-overhead bound on the T4 already, and a faster GPU shrinks the GPU work but not the ~790 CPU-side launches per token, so O1 (CUDA graphs) and fusion matter even more.

## 2. Rules (why this is hard to get wrong)

1. **Nothing is debugged on the server.** Every slot preset is first run end to end on Colab with the same command and deadline (dress rehearsal). Unrehearsed code never goes into a slot.
2. **One command per slot.** A slot is a preset of `scripts/run_suite.sh` (`slot1`, `slot2`, …) started with `remote.sh slot <preset> <slot end HH:MM>`.
3. **Hard deadline built in.** The job stops scheduling 12 minutes before the slot end; a step starts only if its estimate fits; every step has a timeout (2× estimate, capped by time left); whatever didn't fit is listed as `SKIPPED` → first in the next slot. A slot can't overrun, and a hung step can't eat it.
4. **Results leave the server every 5 minutes** (`remote.sh watch`) and once more at the end. If the connection dies, at most the current step is lost.
5. **The master connection is sacred.** The laptop stays awake, plugged in and on the network for the full 2 hours (disable sleep; don't close VS Code/WSL; don't run `remote.sh disconnect` until the end).
6. **Timing runs alone.** Decode is CPU/launch-bound: any other CPU load (pip, builds) skews TPOT/energy. Builds/downloads only during Nsight Compute steps.
7. **Check for other users** at the start (`remote.sh check`); the suite logs every other process on the GPU, and timing done while someone else uses it is flagged.
8. **Fixed roles per slot:** the *driver* runs the commands; the *scribe* keeps the session log (times, what ran, anomalies) → experiment log.

## 3. Before slot 1 (all done without logging in)

- [x] `~/.ssh/config` entry `gpu` → `team21@10.4.25.63` with ControlMaster/ControlPersist and keep-alives (on Jainil's machine).
- [x] `scripts/remote/remote.sh` adapted to the rules above (team folder, caches, master connection checks, `slot <preset> <end>`).
- [ ] Dress rehearsal of `slot1` on Colab (running 2026-10-13).
- [ ] Confirm we can reach `10.4.25.63` from this machine (campus network / VPN? WSL2 + VPN sometimes needs extra setup). A TCP check (`nc -zv 10.4.25.63 22`) does not log in.
- [ ] Optional fallback against "no internet on the server": download the model weights to this machine (`bash scripts/download_model.sh`, 3.2 GB, ~12 min) so `remote.sh pushmodel` can copy them.
- [ ] Book slots; write the slot end times here.

## 4. Slot 1: what we do in the first 2 hours

Times relative to the slot start S; slot end E = S + 2:00. Setup must happen inside the slot (no login outside slots).

| Clock | Who | What | Command | Done when |
| --- | --- | --- | --- | --- |
| S+0:00 | person | **Open the master connection** (team password, once) | `ssh -fN gpu` in a terminal | prompt returns without error |
| S+0:01 | Claude | Master alive? GPU model, other users, disk | `bash scripts/remote/remote.sh status` → `check` | GPU listed, no foreign processes |
| S+0:03 | Claude | Send code | `remote.sh push` | rsync summary |
| S+0:04 | Claude | Preflight + venv + torch + model weights + profilers | `remote.sh setup` | 2 preflight reports, `matmul ok`, weights present |
| S+0:22 | Claude | Smoke test | `remote.sh run "python benchmarks/bench.py --workloads img_small --gen-lens 1,16 --repeats 2 --idle-seconds 2 --tag smoke"` | sensible caption |
| S+0:25 | Claude | **Start the slot job** | `remote.sh slot slot1 <E as HH:MM>` | `started job slot1_HHMM` |
| S+0:25 → E−0:12 | (automatic) | Priority order: benchmark grid → batch sweep → BF16-vs-FP32 → hook overhead → summaries → PyTorch-profiler breakdowns (3) → Nsight Systems (2) → attention probe → Nsight Compute decode → vision → prefill | `remote.sh watch slot1_HHMM` (syncs every 5 min) | log ends with `DONE / FAILED / SKIPPED` |
| E−0:12 → E−0:05 | Claude | Final sync + check summaries locally | `remote.sh sync` | `results/raw/*_baseline_full/summary.md` here |
| E−0:05 | Claude | Leave clean; then the slot ends | `remote.sh jobs` (none), `remote.sh disconnect` | — |
| after | team | Commit results, experiment-log entry, calibrate `EST_SCALE` from step times | — | — |

Budget: setup ≈ 20 min (torch ≈ 2.5 GB + weights 3.2 GB; depends on the server's internet) → about 83 min for the job. T4-based estimates: timing + summaries + profiling ≈ 32 min, attention probe 4, ncu decode 28, ncu vision 16 → ~80 min; ncu prefill most likely moves to slot 2. The server GPU should be faster than the T4, so more may fit.

If setup is slow (no internet, old Python): the deadline logic still protects the slot; whatever doesn't fit moves to slot 2, where setup takes ~1 min.

## 5. Following slots

| Slot | Content | Command | Needs beforehand (rehearsed on Colab) |
| --- | --- | --- | --- |
| 2 | Anything SKIPPED/FAILED in slot 1, then Nsight Compute prefill (both inputs), 1080p vision, deep decode, long-context decode | `ssh -fN gpu`; `remote.sh status`; `remote.sh push`; `remote.sh setup` (~1 min, verifies); `remote.sh slot slot2 <E>` (carry-over: `STEPS="..." remote.sh slot all <E>`) | — |
| 3 | Buffer, or reference baselines O0 (timing, alone): `causal-conv1d`, HF `torch.compile` + static cache, vLLM, llama.cpp | preset `slot3` (to be written) | the O0 scripts |
| 4+ | Optimizations: per variant `SUITE=variant` (~20 min at T4 speed → 3–4 per slot) + targeted Nsight Compute; then the cumulative stack, 3B check, final numbers | `VARIANT=<name> SUITE=variant` via `remote.sh bg` | each variant passes `tests/` and matches baseline outputs on Colab |

## 6. Failure playbook

| Problem | Action |
| --- | --- |
| "Your account has expired" | not our slot (wrong time / time zone); check the booking |
| Can't reach 10.4.25.63 | campus network / VPN; `ssh -v gpu`; report to staff immediately (slot time is lost otherwise) |
| Master connection drops mid-slot | person re-runs `ssh -fN gpu`; Claude: `remote.sh jobs` (it will be gone), `remote.sh sync`, then restart the remaining steps: `STEPS="<skipped/failed>" remote.sh slot all <E>` |
| Another process on the GPU | run profiling/ncu steps first (ratios stay valid); timing steps are flagged; tell the staff |
| No internet on the server | `remote.sh pushmodel` (weights from this machine); pip packages: ask staff for a proxy/mirror |
| Python < 3.10 on the server | `setup_env.sh` installs `uv` and a Python 3.12 into the team folder (needs internet) |
| Old driver / CUDA | `setup_env.sh` picks a matching wheel (cu126/cu128/cu130) and logs the torch version |
| `ERR_NVGPUCTRPERM` (no ncu) | suite skips ncu steps automatically; ask staff to enable profiling for users |
| A step fails | listed as FAILED; the rest continues; we read the log after the slot |
| 10-minute warning appears | expected: our job already stopped at E−12; just let the final sync finish |

## 7. Command cheat-sheet

```bash
ssh -fN gpu                       # (person) open the master connection, once per slot
R=scripts/remote/remote.sh
bash $R status                    # master alive?
bash $R check                     # GPU model, other users, disk
bash $R push && bash $R setup     # code + env + weights + profilers (+ preflight report)
bash $R slot slot1 16:00          # start a preset; stops at 15:48 if the slot ends at 16:00
bash $R watch slot1_1425          # sync every 5 min, show progress, stop when the job ends
bash $R log slot1_1425 40         # last lines of a job
bash $R stop slot1_1425           # stop the job
bash $R sync                      # results back to this machine (repeat freely)
bash $R disconnect                # at the very end only
```

## 8. Open questions

1. Which card exactly (Quadro RTX 5000, RTX 5000 Ada, RTX PRO 5000 Blackwell)? (preflight will tell)
2. Do we need the campus network / VPN to reach 10.4.25.63 from our laptops?
3. Is the GPU exclusive to our team during our slot?
4. sudo? (Nsight Compute counters, clock locking.) Internet access from the server? Quota of `/data/cs3406/team21`?
5. Booked slot dates and times.
