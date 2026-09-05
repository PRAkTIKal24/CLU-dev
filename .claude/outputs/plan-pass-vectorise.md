# plan-pass-vectorise (RE-POINTED: the lane-parallel controller) — experiment-engineer report

Task + acceptance criterion: **put the plan pass's `batch` lane-calls in worker processes (layers stay
sequential), prove the `WritePlan` is unchanged, cut the Python term ≥ 4× at batch 8, and land §7.27's
in-flight store watch.** Status: **done** (both halves; acceptance met at **4.93×**).

⭐ **DOWNSTREAM RECONCILIATION LIST (§5 corollary — needs an owner at the review that accepts this):**
1. **`pilot-placement-probe` §8.1's "GPU-idle < 30 % on any of the above brackets" is now MEASURED-FALSE
   on the 1 s bracket** (34.5–42.9 %; it holds on the 4 s bracket at 11.6–15.8 %). It assumed the full
   8×; the measured cut is 4.93×. Sites: probe §8.1, the `[C2W5]` review §10 entry, charter §A18.4 if
   quoted there.
2. **§7.27 now has an instrument** (`store_health_probe`, on by default) but still **no owner for the
   cause** — the one gradient probe distinguishing φ-drift from amp-suppression is still unrun.
3. **§7 needs three new entries** (see *Proposed handover updates*): `plan_workers`/`plan_mp_start`/
   `store_watch` defaults, the JAX-fork prohibition, and the watch's GPU-backend caveat.

## ⭐ DIAL DECLARATION (echoed before the first result)
⚠ The task file predates/omits the §7 block, so I declare it:
- **Dial:** **none — instrument/engineering.** No claim about admission, lifetimes, isolation or
  compute-adaptive reads is made or moved here. The only claims are (a) wall-clock and (b) *nothing
  else changed*.
- **Laundering control:** the trivial substitute for "make the plan pass faster" is **making a
  different plan** (drop the real controller / round-robin it). The control against it is the
  **bit-identity test** on all 7 plan fields + the monitor-summary and trip-state tests; `plan_workers`
  never touches `real_controller`.
- **Falsifies the claim:** any plan field differing serial-vs-pooled, any monitor count/trip-state
  differing, or a speedup < 4× at batch 8.
- **Does NOT falsify the claim:** losing to an ideal 8× (heterogeneous cores + IPC are known costs);
  the end-to-end plan-pass wall clock barely moving **on this laptop**, where the CPU-side JAX forward
  is 25× the Python term — that is the metric-native-ceiling situation the whole exercise is about,
  and the A100 numbers are explicitly brackets.

## 0. FLAG PROVENANCE (every number in this report)
| item | value |
|---|---|
| commits | `b81f487` (lane-parallel), `8c1d14d` (§7.27 watch); branch `lane-parallel-controller` off `main` = `eaecc91` |
| harness | `.claude/scratch/plan-pass-vectorise/bench_lane_parallel.py`, `bench_watch.py` (scratch, untracked) |
| machine | **Apple M1 Pro, 8 cores (6 P + 2 E)**, JAX **0.9.0** (main venv, reused per §4 — no worktree `uv sync`), CPU backend |
| seeds | 0, 1, 2 (identity); 0 (timing) |
| scale flags (timing) | `d_model=64 n_layers=2 seq_len=1024 batch=8 vocab=256` + **pilot store geometry** `addr_dim=8 payload_dim=4 capacity=32 atoms_per_item=256 budget=24`, `memory: chunk=64 write_inner_steps=40 atom_place_radius=0.3`, `store: write_margin=0.6` (= the probe's recommended CSF3 block) |
| lane-chunk counts | **8 lanes × 16 chunks × 12 layer-iterations = 1536 lane-chunks/step — the PILOT count** (`n_layers` iterated 12×; `d_model` is irrelevant to the controller term) |
| new flags | `plan_workers=0` (default, serial), `plan_mp_start="spawn"`, `store_watch=True` |
| unchanged | `real_controller=True`, all 7 stage flags True, `soft_certificate=True`, `leak=0.02`, `retry_max_rounds=1` |
| pilot smoke | `--scale toy --stage s3 --quick --seed 0 --set plan_workers=2 monitor_every=2` (toy geometry: `capacity=8 atoms=128 addr=2 chunk=32 write_inner_steps=4`) |

---

## 1. ⭐⭐ THE HEADLINE — 4.93× on the Python term, decision-identical

`_plan_lanes` at pilot lane-chunk counts (median of 3, 12 sequential layer-batches = one step):

| path | per-step Python | speedup | pool setup (one-off) | plan identical to serial? |
|---|---|---|---|---|
| **serial** (shipped) | **1.597 s** | 1.00× | — | — (reference) |
| pool, 2 workers | 1.018 s | **1.57×** | 5.1 s | ✅ all 7 fields |
| pool, 4 workers | 0.584 s | **2.73×** | 5.8 s | ✅ all 7 fields |
| **pool, 8 workers** | **0.324 s** | **⭐ 4.93×** | 6.5 s | ✅ all 7 fields |

- ✅ **Acceptance met: ≥ 4× at batch 8.**
- **Identity** (`slot, admitted, group_scale, reset, sites, live, retry`, exact `array_equal`, not
  `allclose`): **batch 8 × seeds 0/1/2 in the harness**, plus **batch 3 × seeds 0/1/2 as a blocking
  test**, plus identical monitor summaries and identical trip states for all 14 monitors.
- Fitting `T(W) = s·T_ser/W + c` to the three rows: **s = 1.16** (straggler/contention inflation)
  and **c = 7.7 ms per layer** (the 8-lane round trip). `c` is 28 % of the pooled per-layer time at
  this scale, and it is what stops the cut at 4.93× rather than 8/1.16 = 6.9×.

**⚠ Honest scope on the absolute serial number.** Steady-state I measure **1.04 ms per lane-chunk**,
*below* the probe's 1.7–2.4 ms band (the probe's own band is reproduced by my **first** call, 1.86 ms,
which carries JAX warm-up). The reason is visible in the profile: at this synthetic rig the admission
gate **refuses 120 of 128 offers**, leaving **1 live item per lane**, and the per-chunk cost scales
with the number of live records (`slot_of`, the live-state fill). A store carrying its budget of 24
items is **more** expensive per lane-chunk, and since `c` is fixed, **the speedup then gets better,
not worse** — 4.93× is a conservative floor for the pilot.

## 2. GPU-idle brackets, re-computed (task §2's fourth bullet)

Both columns keep the probe's two A100 bracketing assumptions for the JAX-side step (**1 s** and
**4 s**); nothing on a GPU was measured from here (declared NOT-RUN).

| Python term/step | serial | pooled (÷4.93) | idle @1 s JAX | idle @4 s JAX |
|---|---|---|---|---|
| **probe's band, 2.6–3.7 s** (the published pilot number) | 2.6–3.7 s | **0.53–0.75 s** | **72.2–78.7 % → 34.5–42.9 %** | **39.4–48.1 % → 11.6–15.8 %** |
| my direct measurement, 1.60 s (n_live = 1 floor) | 1.60 s | 0.32 s | 61.5 % → **24.5 %** | 28.6 % → **7.5 %** |

⛔ **The probe's "< 30 % on any of the above brackets" is FALSE at the 1 s bracket** under its own
Python-term band — it assumed 8×. It is true at the 4 s bracket, and true at both brackets only if the
Python term is at the low (my-measurement) end. **This was pre-registered as the likely outcome and it
happened** (PREREG §"registered in advance").

## 3. End-to-end `plan_pass` on this laptop — and why it is not the number that matters

| | serial | pool 8 |
|---|---|---|
| `plan_pass`, 2 real layers, batch 8, warm | 8.94 s | 8.25 s (**1.08×**) |

At this rig the *CPU* JAX forward (`write_inner_steps = 40`, 16 chunks) is ~25× the Python term, so
the cut is invisible end-to-end **here**. That is exactly the probe's argument in reverse: the ratio
`Python : JAX` is what an A100 changes by 1–2 orders, and the Python term is the one no GPU touches.
Reported, not hidden — and it is why the honest acceptance criterion was on the Python term.

---

## 4. ⭐ HOW THE TWO PICKLABILITY BLOCKERS WERE CLOSED (probe §8.1)

1. **The live `CluControllerV0` → `LaneControllerSummary`** (guard counts, verb log, live records,
   stop state, projected policy). ⭐ **The serial path returns it too**, so serial and pooled are the
   same object graph and the equivalence test compares like with like. It duck-types
   `guard_fire_counts()`. `MonitorContext.controller` receives it; **no monitor in `monitors.py` reads
   `ctx.controller`** (grepped) — M14 reads `extras["canary_guard_counts"]`, which is merged from the
   per-lane summaries and is asserted equal.
2. **The shared monitor `registry` → `_ClassITrips`**, a snapshot of `class_i_tripped()` taken **once
   per pass**. The controller only *reads* that list, and nothing inside a plan pass calls
   `registry.observe`, so the snapshot is exact. ⭐ Asserted with a registry **carrying a real class-I
   trip** (`blank`), not an empty one — otherwise the claim is vacuous.

**Safety properties built in:** `spawn`, never `fork` (JAX is not fork-safe with live backend
threads); workers get `JAX_PLATFORMS=cpu` + `XLA_PYTHON_CLIENT_PREALLOCATE=false` + single-threaded
BLAS **before** they import, so N workers cannot each grab the job's GPU; one **persistent** pool per
`(workers, start)` created lazily and `atexit`-torn-down; a broken pool falls back to the serial path
with a loud one-line message (a wall-clock failure, never a correctness one).

---

## 5. ⚠ THE §7.27 RIDER — and it fires immediately

`store_health_probe` logs the probe's §10-row-10 **pair** at `monitor_every` cadence on **fixed**
tokens, with an untrained baseline taken **before the first update**, into
`rec["arms"][*]["train"]["store_health"]` — i.e. **the CSF3 artifact carries it by default**
(`store_watch=True`) — and prints each reading in flight.

Toy pilot, `--quick` (6 steps), seed 0, `monitor_every=2`, printed live:

| step | tag | depth_median | ×untrained | q\* payload spread | ×untrained | n_live |
|---|---|---|---|---|---|---|
| 0 | untrained | 0.06864 | 1.00 | 0.14233 | 1.00 | 3 |
| 2 | trained | 0.02953 | **0.43** | 0.09326 | **0.655** | 4 |
| 4 | trained | 0.02287 | **0.333** | 0.06561 | **0.461** | 4 |
| 5 | trained | 0.02531 | **0.369** | 0.04542 | **0.319** | 4 |

⛔ **§7.27's mechanism is visible in SIX training steps** at toy scale: depth to 0.37× and the
between-item `q*` spread to 0.32× of untrained, **monotonically**. This is one seed at toy scale and
is a *demonstration that the instrument works*, not a result — but it says the CSF3 run will have this
signal in its log from step 25 onward instead of at hour 12.

**Both numbers, never one (§7.26):** depth alone is not health — a bigger `write_margin` deepens every
well *at a shared payload location* (0.114 → 0.054 spread). The watch would call that a pass; the pair
calls it what it is.

**Cost, measured and then fixed.** The reading replays a sequence's writes; **eager it cost 49.2 s per
reading** at pilot store geometry (`write_inner_steps=40`). Two in-scope fixes: **(i)** only **one
lane** is planned (the diagnostic replays lane 0; lanes are independent), **(ii)** the per-chunk write
goes through a `filter_jit`ed `_cell_write`. **49.2 s → 2.44 s (20.2×)**, depth identical to
**9 significant figures** (3.8902763365e-01 → 3.8902763330e-01, float32 ULP from XLA fusion). At toy
scale the steady-state reading is **~1.0 s**. At `monitor_every=25` over 4000 steps = 160 readings,
this is ≈ 1 extra monitor-pass-equivalent per observation.

⚠ Two honest caveats: (a) when a lane has **1 live item the spread is `nan`** (undefined) — it happened
at my synthetic pilot-geometry rig (94 % of offers refused) and did not at the toy pilot rig (3–4 live);
(b) `monitor_pass`'s own eager replay is deliberately **not** jitted (its numbers are published and it
is out of my ownership), so the watch's depth and monitor #9's depth can differ in the last bits.

---

## 6. PREREG SCORECARD (registered in `.claude/outputs/plan-pass-vectorise/PREREG.md` before any timing)

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | speedup, batch 8, W = 8: **5.0×**, band [4.0, 7.0] | **4.93×** | ✅ **CONFIRMED** (band + acceptance) |
| **P2** | W = 4: 3.2×, [2.5, 4.0] | 2.73× | ✅ in band |
| **P3** | W = 2: 1.8×, [1.5, 2.0] | 1.57× | ✅ in band |
| **P4** | per-layer overhead `c` = 3 ms, [1, 10] | **7.7 ms** (fitted with s = 1.16); naive excess-over-ideal 10.4–18.3 ms | ◐ **in band, point off 2.6×** — and the naive estimator I would have quoted (10.4 ms) is *outside* the band, so the decomposition mattered |
| **P5** | exact identity, batch 8, 3 seeds, 7 fields | exact, 3/3 seeds (+ 3 more at batch 3 in the suite) | ✅✅ **CONFIRMED** |
| **P6** | pilot Python term after: 0.52–0.74 s | 0.53–0.75 s | ✅ (arithmetic from P1 — weak) |
| **P7** | idle @1 s: 34–43 % | 34.5–42.9 % | ✅ (arithmetic — weak) |
| **P8** | idle @4 s: 12–16 % | 11.6–15.8 % | ✅ (arithmetic — weak) |
| **P9** | pool setup 20 s, [10, 60] | **5.1–6.5 s** | ⛔ **REFUTED — 3–4× CHEAPER**; spawned children import off a warm page cache, so `spawn`'s cost is not a reason to reach for `fork` |
| **risk** | *"I predict the probe's < 30 % claim FAILS on the 1 s bracket"* | 34.5–42.9 % | ⛔ **the probe's claim is refuted, as registered** |

Registered but *not* measured, declared: anything on an A100/CSF3; `fork`; a real-latent (enwik8)
pilot-scale end-to-end.

---

## 7. ⭐ THE CSF3 DELTA (probe §10 base + mine)

The delta is **two flags and one sbatch option — no file is edited** (a module edited on the cluster
is a provenance hole; `plan_workers` lands in the artifact's `flags.pilot` block verbatim).

```bash
# probe §10 base + `plan-pass-vectorise` delta   (delta = `plan_workers=8` and `-c 12`)
sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,\
MEM="atom_place_radius=0.3 write_inner_steps=40",\
STORE="write_margin=0.6",\
SET="monitor_every=25 plan_workers=8" \
       -c 12 --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
```

- `plan_workers=8` = one worker per lane (`batch = 8`). `-1` would also resolve to 8 there.
- **`-c 12`** overrides the script's `#SBATCH -c 8` **without editing the file**; gpuA allows ≤ 12 host
  cores per GPU (documented in the script's own header) and 8 workers + the parent need 9. Optional —
  at `-c 8` the parent contends with one worker and the cut lands nearer the 4.4–4.9× measured here.
- **§7.27's watch needs no flag** (`store_watch=True` by default); it will appear as
  `arms.*.train.store_health` and as `[watch/clu_store] …` lines in `logs/%x-%j.out`.
- ⚠ If a worker cannot init CPU-only JAX on the node, the run **does not fail** — it prints
  `[plan_pass] ⚠ lane pool unavailable (…); falling back to the serial controller` and proceeds
  exactly as the pre-probe submission. Grep the log for that line before quoting the speedup.

---

## 8. How I verified (commands + observed output)
- `pytest tests/test_lane_parallel_controller.py tests/test_cluformer_pilot.py tests/test_placement_probe.py -q`
  → **49 passed in 166 s** (and again at the final state, **27 passed** for the two pilot files after
  the commit split; the third file is unchanged by the split).
- Blast radius (everything importing `train_cluformer`/`monitors`): `pytest tests/test_blocks.py
  tests/test_clu_system.py tests/test_clu_controller.py tests/test_memory_gym.py tests/test_monitors.py
  tests/test_soft_certificate.py tests/test_traj_write_objective.py -q` → **169 passed in 240 s**.
- `ruff check chlu/training/train_cluformer.py tests/test_lane_parallel_controller.py` → **All checks
  passed**. (⚠ `ruff format --check` reports this file as unformatted **on `main` too** — pre-existing,
  not touched, out of scope.)
- `bench_lane_parallel.py` → §1's table; `bench_watch.py` → §5's 49.2 s → 2.44 s.
- Real runner end-to-end, twice: `python -m chlu.experiments.exp_cluformer_pilot --scale toy --stage s3
  --quick --seed 0 --arms clu_store none --set plan_workers=2 monitor_every=2` → artifact contains
  `flags.pilot.plan_workers = 2`, `monitors_final.plan.lane_mode = "pool[2/spawn]"`, 4 `store_health`
  entries on `clu_store` and **0 on `none`** (inapplicable, correctly), `static bpc 8.0117`.
- `_resolve_workers`: `0→0`, `-1→8`, `2→2`, `99→8` (clamped to `batch`); defaults are
  `plan_workers=0, plan_mp_start='spawn', store_watch=True`.

## 9. New tests (11, in a NEW file — no shared test file was edited)
`tests/test_lane_parallel_controller.py`:
`test_lane_parallel_plan_is_bit_identical_to_serial[0/1/2]` (⭐ the blocking equivalence test; also
asserts the pool was *actually used*, so a silent fallback cannot pass it) ·
`test_lane_parallel_monitor_summaries_are_unchanged` · `test_monitor_trip_states_are_identical_under_the_pool` ·
`test_the_lane_summary_is_picklable_and_duck_types_the_controller` ·
`test_class_i_snapshot_reproduces_the_live_registry` · `test_store_health_probe_reports_depth_and_qstar_spread` ·
`test_store_health_probe_is_inapplicable_off_the_store` ·
`test_the_watch_fires_during_training_and_lands_in_the_artifact` · `test_the_watch_can_be_switched_off`.
Module-scoped autouse float32 fixture, per §7.23/N211's documented pattern.

## 10. Git footprint
- **Branch `lane-parallel-controller`** (the task file's name; the Head's spawn template said
  `agent/experiment-engineer/<slug>` — I followed the **task file**, flagged here so the Hub looks in
  the right place), off `main` = `eaecc91`. Rebase onto local `main` = no-op (`main` has not moved).
  **Not pushed.**
- `b81f487` — lane-parallel plan pass (`chlu/training/train_cluformer.py`,
  `tests/test_lane_parallel_controller.py`).
- `8c1d14d` — the §7.27 in-flight watch (same two files).
- Touched **exactly two files**, both inside the declared ownership (`plan_pass` /
  `_controller_plan_for_lane` call path / new tests). ⛔ `monitors.py`, the factored-store files and
  `chlu/core/controller*` were **not** modified.
- Worked in a **scoped worktree** (`../CHLU-lane-parallel-controller`, main venv reused per §4);
  verified `git -C <main-repo> log main..lane-parallel-controller` shows **both** hashes **before**
  `git worktree remove` (§3.2). ⚠ Two other agents' worktrees are live (`CHLU-mamba2`,
  `CHLU-null-arms`) — the worktree was the right call. `orgdiv-null-arms` has committed only
  `chlu/core/null_arms.py` (no overlap); `bprime-mamba2-arm` has **no commits yet but is very likely to
  touch `train_cluformer.py`'s arm plumbing** (`solve_arms`/`build_arm`/`arms`) — **no overlap with my
  hunks** (`PilotConfig` runtime block, `plan_pass`, `train_arm`'s loop, two new functions), so a
  textual conflict is unlikely but the Hub should merge mine first or expect a trivial one.
- ⚠ One process note: I first committed both changes together, then **split them** (`git reset --soft`
  + reconstruct) and verified the reconstructed tree was **byte-identical** to the pre-split commit
  (`git diff 2b4a18d` empty). `2b4a18d` is now unreferenced (reflog only).

## 11. Open questions / follow-ups / risks
1. ⚠ **Backend asymmetry on a GPU host (the one real correctness risk).** Workers are pinned to CPU;
   on CSF3 the parent's JAX is on the A100. The controller's *decisions* are pure-numpy
   (`min_separation`, `d_safe`, the record loops) — the only JAX-touched quantity in a decision is
   `allocator.store.amps` in `_pick_victim`, whose entries differ by `exp(-leak·Δage)` and are
   tie-broken by `item_id`, so a backend rounding difference flipping a victim is possible but needs a
   near-exact amp tie. **My identity tests run CPU-parent vs CPU-worker and cannot see this.**
   Cheapest mitigation, ~10 min on the cluster: run `STG=s1` once with `plan_workers=0` and once with
   `8` and diff `monitors_init.plan`. I recommend the Head/Hub make that the first CSF3 action.
2. **`c = 7.7 ms/layer` is the remaining ceiling**, not the lane work. If more is wanted, the next cut
   is one `ex.map` over **(layer-independent) lanes only once per step** — impossible (layers are
   sequential) — or shrinking the payload (`sites` is 24 KB/lane and is the bulk of it; it could be
   sent as int8 slot indices + a small centre table). Not built, priced at ~2 h.
3. **The watch's cost is now dominated by `plan_pass` itself**, so it inherits this task's speedup;
   but at `monitor_every=25` it is still ≈ one monitor-pass per observation. If the 12 h budget gets
   tight, `store_watch=False` is a flag — **but that is exactly the number §7.27 says not to lose.**
4. `n_live = 1` ⇒ `qstar_payload_spread = nan`. Real enwik8 latents gave 3–4 live items in every rig I
   ran, so this is unlikely at scale; if the pilot log shows `nan` spreads, the finding is *"the
   admission gate is refusing almost everything"*, which is itself the story (and matches the probe's
   120/128 refusals here).
5. **Not run, declared:** `fork`; a real pilot-scale (26–47 M) end-to-end; any GPU; a sweep of
   `plan_mp_start`; threads (the controller is Python-bound, so a thread pool cannot help).

---

## Proposed handover updates (for the Hub)
- **§3 (config) — new `PilotConfig` flags**, all preserving current behaviour by default:
  `plan_workers = 0` (0 serial · N workers · −1 = `min(batch, cpu_count)`), `plan_mp_start = "spawn"`,
  `store_watch = True`. All three reachable as `--set` / `SET=` on the CSF3 job and recorded in
  `flags.pilot` (⚠ `as_flag_table` omits defaults, so `store_watch=True` shows only via the
  `store_health` block it produces).
- **§7.27 — update from "OPEN, no owner"** to *"OPEN (cause); **instrumented** by
  `store_health_probe`, on by default, `arms.*.train.store_health` in the artifact + `[watch/…]` lines
  in the job log. Depth ×0.37 and `q*` spread ×0.32 of untrained within **6** toy steps (`8c1d14d`).
  Still unrun: the one gradient probe separating φ-drift from amp-suppression."*
- **§7 NEW — 7.30 [ENV/standing] JAX is not fork-safe.** Any multiprocessing in this repo must use
  `spawn` (measured cost: 5–6.5 s for a 2–8-worker pool, *not* the 20 s I expected) and must set
  `JAX_PLATFORMS=cpu` in the child **before** import or every worker opens a handle on the job's GPU.
  Pattern: `train_cluformer._lane_pool`.
- **§7 NEW — 7.31 [correctness watch] CPU-pinned workers vs a GPU parent.** The lane controller's
  decisions are numpy, except `_pick_victim`'s `amps` comparison; a backend difference could in
  principle flip a victim on an exact tie. Verified identical CPU-vs-CPU only. First CSF3 action
  should be the `STG=s1` `plan_workers` 0-vs-8 diff (§11.1).
- **§10 running log / `[C2W5]` §10 entry — correct the probe's GPU-idle claim:** the lane-parallel cut
  is **4.93×**, not 8×; GPU idle after it is **34.5–42.9 % (1 s A100 bracket)** and **11.6–15.8 %
  (4 s bracket)** against the probe's own 2.6–3.7 s Python band. *"< 30 % on any bracket"* is
  measured-false at the 1 s bracket. (Pre-registered; see PREREG.)
- **The gate condition in the task file is met**, so on the Head's ruling ("lane-parallel first, then
  submit") the CSF3 submission is unblocked; the recommended line is §7 above.
