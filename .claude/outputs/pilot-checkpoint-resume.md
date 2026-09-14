# pilot-checkpoint-resume — experiment-engineer report

**Task + acceptance criterion:** make a 30 h CSF3 pilot leg survive a crash — per-arm
checkpoint + `--resume`, eval-block host-memory hygiene with per-phase RSS instrumentation,
the `wall_s`/`plan_s` print erratum — accepted on a **toy bit-identity gate on BOTH the straight
path and the resume path**, an unchanged final-artifact shape, tests + suite green.

**Status: done.** Both gate legs are **BITWISE with ZERO differing leaves over the entire record**
(timing keys excluded, declared in PREREG before any run). The resumed leg recovered 175 s of
`clu_store` work in **22 s**. New file `tests/test_pilot_checkpoint_resume.py`: **13 passed**.
⛔ **One deliverable is DELIVERED-BUT-NOT-AS-A-NUMBER and it is the memory budget** — see §5:
I pre-registered that I could not bound the pilot's host RSS from this machine, and I could not.
What I ship instead is the instrument that bounds it on the first cluster run, plus a **named,
falsifiable prediction of the crash phase.**

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — infrastructure/instrument. Claims: (a) a crashed leg is resumable, (b) nothing
  else changed, (c) the next crash is attributable to a named phase.
- **Laundering control:** the toy bit-identity gate, **two legs** — old code (`main @ 8efc1d8`) vs
  new code straight through, and uninterrupted vs killed-after-arm-1 + `--resume`. **Both: 0
  differing leaves.**
- **Falsifies:** any non-timing field moving; the final artifact gaining or losing a key; a resumed
  arm seeing different batches.  **None fired.**
- **Does NOT falsify:** wall-clock overhead (the hygiene pass buys memory with re-compiles —
  measured +24 % at toy, argued down to ~2 % at pilot in §5.4, with an off-switch); the fact that a
  laptop CPU run cannot measure an A100 job's host RSS (registered in advance, PREREG P4).

## ⚠ RECONCILIATION LIST (first-10-lines rule, protocol §5)
1. **§7-candidate to RETIRE:** the `csf3-memory-fit` §7 erratum ("`train_log` prints
   `wall_s`/`plan_s` every 25 steps" — false) is **FIXED**. The line now exists and is tested.
2. **The launch checklist's "who writes where" item gains a clause.** Per-leg `OUT` is now
   **mandatory, not advisable**: two legs sharing one `OUT` would share the journal and the `.eqx`
   checkpoints. ⭐ The resume path is *self-protecting* (the legs differ by
   `memory.psi_payload_residual`, which is inside the resume fingerprint ⇒ a cross-leg resume is
   REFUSED and names the key) — but the **final-artifact overwrite hazard is unchanged**.
3. **`probe_lanes` is now a priced, named lever and its owner is the Hub, not me** (§5.3): it cuts
   the one eval-block phase that runs **un-jitted**, and it moves a **published** S2 magnitude.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/pilot-checkpoint-resume` @ `d30e4b4`, `cd73f81`, `f4264a7`, `c9709b7`; base `main` = `8efc1d8` |
| worktree | none for the work (clean tree, no concurrent spoke). One **throwaway detached worktree** `../CHLU-pcr-base @ 8efc1d8` existed only to run the OLD code for gate (a); **no commits were made in it** and it is removed. |
| machine / env | Apple M1 Pro, **CPU backend**, **JAX 0.9.0, equinox 0.13.4**, main venv reused (protocol §4) — no worktree `uv sync` |
| gate config | `--scale toy --stage s3 --seed 0 --quick --arms clu_store gru_matched --d5` ⇒ `d_model=64 n_layers=2 seq_len=512 batch=4 addr=2 pay=1 capacity=8 atoms_per_item=128 chunk=32 address=read=24 traj_stride=8 psi_hidden=32 write_inner_steps=4 write_n_perturb=8 retry_rounds=1 steps=6 warmup=2 eval_batches=dyneval_batches=2 data_bytes=1e6`, `real_controller=True`, all 7 stage flags True, `soft_certificate=True`, `store_watch=True`, `monitor_every=100`, `plan_workers=0` (serial), `remat_chunks=false`, `accum_steps=1`, `liveness_lanes=0`, `probe_lanes=0`. Seed 0. |
| pool measurement | the same, `--stage s1 --arms clu_store --set plan_workers=4` |
| new flags (defaults) | `PilotConfig.eval_cache_hygiene=True` ⭐ **ON**, `.rss_log=True` ⭐ **ON**, `.stop_after_arms=0` (off). `run_pilot(resume=False)` / CLI `--resume` / `RESUME=1`. `save_json(..., atomic=False)`. `anytime_curve(..., hygiene=None→pcfg)`. |
| ⛔ NOT RUN (pre-registered) | **anything on a GPU / CSF3.** Every pilot-scale number is a projection, labelled, with its transfer assumption stated. |
| pre-registration | `.claude/outputs/pilot-checkpoint-resume/PREREG.md`, filed at implementation-complete (ruff green, nothing executed) — scorecard at §7 |

⚠ **Two defaults ship ON, unlike `csf3-memory-fit`'s levers.** That is deliberate and Head-directed
("checkpointing IS REQUIRED"; footprint reduction is "the primary fix, not a fallback"), and §A20.4
holds because **all six legs rerun on this code uniformly** — the toy gate is what licenses calling
the change decision-inert. **The MEM/STORE/SET flag strings do not change.**

---

## 1. What I did

| # | deliverable | where | status |
|---|---|---|---|
| 1 | per-arm checkpoint + `--resume` | `exp_cluformer_pilot.py` (`_Phases`, `load_journal`, `save/load_arm_checkpoint`, `_arm_row`), `job_gpu_cluformer.sh` (`RESUME=1`) | ✅ |
| 2 | eval-block host-memory hygiene + per-phase RSS | `train_cluformer.py` (`host_rss`, `release_host_memory`, `eval_cache_hygiene`, `rss_log`), `anytime_curve` | ✅ instrument; §5 for the budget |
| 3 | the `wall_s`/`plan_s` print | `train_cluformer.py::train_arm` | ✅ |
| 4 | toy bit-identity gate, both legs | `.claude/outputs/pilot-checkpoint-resume/gate_{a,b}.txt` | ✅ **0 diffs each** |
| 5 | tests + suite + ruff | `tests/test_pilot_checkpoint_resume.py` | ✅ 13 new, suite §6 |

### 1.1 the journal
`pilot_{scale}_seed{N}_PARTIAL.json` (atomic: tmp + `os.replace`) is **the record so far**, rewritten
after every phase; `ckpt_{arm}_seed{N}.eqx` is `eqx.tree_serialise_leaves` of that arm's trained
weights (664 KB at toy; projected **~105–190 MB per arm** at 26–47 M ⇒ ~0.5–1 GB per seed).

⭐ **The weights hit the disk the instant `train_arm` returns — before any evaluation.** The task
as written ("after EACH arm completes (train + its evals)") would **not have saved attempt 1**:
the kill landed inside `clu_store`'s *own* eval block, i.e. arm 1 of 5 never "completed", so a
resume would have retrained 22 h. Writing the checkpoint at the train/eval seam is the difference
between the fix working and the fix being decorative. **Flagged as a deliberate deviation from the
task text, in the direction of the task's intent.**

Resume granularity is therefore **per phase**, not per arm: `phi_gain` · `monitors_init` ·
`allocation_liveness_init` · `gradient_probe_init` · then per arm `train` (+`monitors_during`,
`monitors_final`) · `static` · `dyneval` · `blank_store` · `anytime_curve` ·
`gradient_probe_final` · `selectors_final`.

### 1.2 the two design decisions that carry the bit-identity claim

**(a) The data-stream guarantee — verified, and it is trivial once stated.** `_train_batches`
materialises `list(random_batches(tr, batch, seq_len, n_batches=steps, seed=pcfg.seed))` **once**,
and every arm consumes `iter(batches)` — a *fresh iterator over that same list*. `random_batches`
builds its `np.random.default_rng(seed)` inside the call. **⇒ arm *k*'s batch sequence is a
function of `(seed, steps, batch, seq_len)` alone and carries nothing out of arms `0..k-1`, or out
of whether those arms ran in this process at all. No fast-forwarding is required and none is
performed.** Eval iterators are `contiguous_batches` (deterministic, unseeded); the deserialisation
template is `build_arm` from `PRNGKey(1000+seed)` alone; the optimiser state is re-initialised per
arm inside `train_arm`. Asserted in
`test_the_training_stream_carries_nothing_between_arms`, not asserted in prose.

**(b) The ONE piece of state that genuinely crosses a phase boundary, and what I did about it.**
The persistent monitor registry holds monitor #6's `(write_loss, acq)` window and **cannot be
serialised** (it owns live Equinox objects). A naive resume would therefore hand `monitors_final` an
empty registry and flip monitor #6 `applicable → inapplicable` — *"an inapplicable monitor is not a
passing monitor"* is this pilot's own acceptance language, so that is a silent corruption of the
acceptance criterion. **Fix: `monitors_final` moves INSIDE the training segment**, taken while the
registry is still alive. It is a pure function of `(m, reg, x0)` and nothing between the two
positions writes to `m` or calls `reg.observe`, so the move is bitwise inert — proven by gate (a),
which compares against the old code where it sat after the eval phases. The resumed run's
`n_applicable` is asserted equal to the reference's in the test.

### 1.3 what is NOT in the final artifact
`host_rss` and `_journal` are stripped by `_finish` (`_JOURNAL_ONLY_KEYS`) and live only in the
PARTIAL. `_arm_row` re-keys each arm into the canonical order so an interrupted+resumed run and an
uninterrupted one emit the same object. `--plot-only`'s glob `pilot_*_seed*_S*.json` does **not**
match `_PARTIAL.json` (tested).

---

## 2. ⭐ THE GATE — both legs, zero differing leaves

Four toy runs, driven by `.claude/scratch/pilot-checkpoint-resume/gate.sh`, compared leaf-by-leaf
by `compare.py` over the **whole record** with timing keys excluded by name.

### (a) OLD path (`main @ 8efc1d8`, run in its own worktree so `cwd` wins on `sys.path`) vs NEW

| decision field | old | new | |
|---|---|---|---|
| held-out **static nll** | 5.553304672241211 | 5.553304672241211 | ✅ **BITWISE** |
| held-out **static bpc** | 8.011725111187905 | 8.011725111187905 | ✅ **BITWISE** |
| **dyn-eval bpc** / best_lr | 8.006763411710514 / 0.01 | identical | ✅ **BITWISE** |
| **blank-store bpc** | 8.011891246428640 | 8.011891246428640 | ✅ **BITWISE** |
| swap arm `gru_matched` static / dyneval bpc | 8.017805729791812 / 8.012848845828643 | identical | ✅ **BITWISE** |
| `monitors_final` n_tripped / n_applicable | 5 / 10 | 5 / 10 | ✅ |
| `gradient_probe_final.trajectory.grad_phi` | 0.01452394581309086 | identical | ✅ **BITWISE** |
| `allocation_liveness_init.grad_phi_addr_head` | 0.01179222207384902 | identical | ✅ **BITWISE** |
| `phi_gain_calibrated` | 3.470976384444425 | identical | ✅ **BITWISE** |
| **full-record walk** | — | — | ⭐ **0 differing leaves** |

### (b) UNINTERRUPTED vs KILLED-AFTER-ARM-1 (`os._exit(137)`, no finalisers) + `--resume`

**16/16 named decision fields BITWISE; full-record walk: ⭐ 0 differing leaves.**
The interrupted leg exited at t=274 s having banked `clu_store` (175 s of arm work); the resumed leg
reproduced the whole artifact in **22 s**, lifting 10 phases from the journal and retraining only
`gru_matched`.

⚠ **Timing keys excluded, declared in PREREG P1 *before* any run:** `wall_s`, `wall_s_total`,
`plan_s`, `plan_pass_s`, `plan_pass_frac`, `wall_ratio_traj_over_point`, `t_s`, `wall_clock_s`,
**`cost_ms`**. ⚠ Honest note: `cost_ms` (the per-monitor observation cost inside
`monitors_*.readings[i]`) was **added to the exclusion list after the first comparison surfaced it**
— it is unambiguously a wall-clock field (`monitors.py:276`, `(perf_counter()-t0)*1e3`) and it was
the *only* thing the first comparison found (26 of 26 differing leaves), but the list was not
complete when filed and I am saying so rather than quietly widening it.

⛔ Unlike `csf3-memory-fit` there is **no tolerance band and none was needed**: nothing in this
change set re-associates a float sum, so the correct prediction was exact equality and exact
equality is what was measured.

---

## 3. THE RESUME, AS AN OPERATION

```bash
# resubmission, per leg, per seed — the ONLY delta vs the ruled line is RESUME=1 and per-leg OUT
sbatch --export=ALL,SEEDS="0",STAGE=pilot,STG=s4,RESUME=1,\
OUT=".claude/outputs/cluformer-pilot/run1",\
MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true",\
STORE="write_margin=0.6",\
SET="monitor_every=25 plan_workers=8 liveness_lanes=1" \
       -c 12 --mail-user=$CLU_MAIL -t 4-00:00:00 scripts/csf3/job_gpu_cluformer.sh
```
- ⭐ **`RESUME=1` is safe on a FIRST submission**: no journal ⇒ `[resume] no journal at … — starting
  from scratch` ⇒ a normal run. It can therefore go into the line permanently.
- ⛔ It **refuses** a journal written under a different config and **names the differing key**
  (`memory.psi_payload_residual: journal=false now=true`). Attempt 1 wrote no journal (old code), so
  a re-run over attempt-1 artifacts is a clean start.
- ⛔ **Per-leg `OUT` is now mandatory.** The journal and the `.eqx` files are per-`OUT`.

---

## 4. THE PER-PHASE RSS INSTRUMENT (deliverable 2), and what it measured at toy

Every phase boundary prints, **to stdout** — which is the load-bearing channel, because an
`oom_kill` writes no artifact at all:
```
[rss] clu_store/dyneval/enter    rss    3.27 GB | peak    3.66 GB | children   0.00 GB (n=0) | t 145s
```
`rss`/`peak` = `VmRSS`/`VmHWM` from `/proc/self/status` on Linux (the cluster) — **`VmHWM` is the
kernel's own high-water mark, i.e. the number an `oom_kill` is decided against**; `ru_maxrss` + a
`ps` fallback off Linux. `children` sums the plan-pool workers' RSS, because **the cgroup a job is
killed against is the sum, not the parent.**

### 4.1 measured, toy, 2 arms + D5 — peak growth attributed to the phase that caused it

| phase | Δ peak | Δ rss |
|---|---|---|
| **`gradient_probe_init`** | **+1.282 GB** | +1.650 GB |
| **`allocation_liveness_init`** | **+1.036 GB** | +0.769 GB |
| `monitors_init` | +0.518 GB | +0.496 GB |
| `phi_gain` | +0.124 GB | +0.124 GB |
| `clu_store/train` | +0.000 | +0.912 GB |
| `clu_store/gradient_probe_final` | +0.000 | **+0.926 GB** |
| `clu_store/dyneval` | +0.000 | −0.969 GB |
| `clu_store/anytime_curve` | +0.000 | −0.211 GB |
| `clu_store/static`, `/blank_store`, `/selectors_final`, both `gru_matched` phases | +0.000 | ≤ ±0.04 GB |
| **run peak** | **3.655 GB** | |

### 4.2 what `release_host_memory()` gave back

| boundary | reclaimed |
|---|---|
| `allocation_liveness_init` | **+0.824 GB** |
| `gradient_probe_init` | **+0.512 GB** |
| every other boundary | +0.000 … +0.033 GB |
| `monitors_init`, `clu_store/gradient_probe_final` | **−0.097 / −0.205 GB** (⚠ *negative*) |
| **total over the run** | **+1.433 GB** |

⚠ **The two negatives are real and I am not hiding them:** `gc.collect()` and `jax.clear_caches()`
allocate while they run, and **macOS `malloc` does not return freed arenas to the OS**, so an RSS
*drop* is under-reported on this machine and an RSS *rise* can be pure allocator noise. On Linux
(glibc) large blocks are `mmap`ed and *are* returned, so the cluster reading will be cleaner than
this one. PREREG P3's gate (≥ 50 MB reclaimed at ≥ 1 boundary) is met **16×** by the 0.824 GB row.

### 4.3 ⭐ THE FINDING THE TOY RUN ACTUALLY DELIVERS

**The whole run's peak is set by the two UN-JITTED full-batch backwards, and neither is in the eval
block.** `allocation_liveness` and `gradient_probe` call `eqx.filter_value_and_grad(loss_fn)`
**directly — no `filter_jit`** — so their residuals are eager buffers with no XLA reuse. Together
they take the peak from 1.34 → 3.655 GB (**63 % of the run's peak**), at *init*, before a single
optimiser step. Every phase of the post-training eval block adds **zero** new peak at toy — because
`gradient_probe_final` is the *same call site* as `gradient_probe_init` and its allocation is
already covered.

⇒ **`eqx.filter_value_and_grad(loss_fn)` without a `filter_jit` is the memory-heavy pattern in this
codebase**, and there are exactly three call sites: `allocation_liveness` (cut 8× by
`liveness_lanes=1`, already in the ruled line), `gradient_probe` (**uncut**, ×2 per seed), and
`dynamic_eval`'s `n_micro == 1` branch (×3 LRs × 40 batches at pilot). That is the hypothesis I hand
to the cluster, and §5.2 turns it into a falsifiable prediction.

---

## 5. ⛔ THE BUDGET — what I can and cannot say (PREREG P4, registered in advance)

### 5.1 why the toy peak does NOT transfer, stated plainly
**On the CPU backend every activation is host RAM.** On the A100 they are HBM (`remat_chunks` holds
the device at ≈ 8.34 GiB). So my 3.655 GB toy peak **conflates the two pools** and is not an estimate
of anything the cluster measures. Attempt 1's own number is no better: **`MaxRSS = 125.6 GB` is
truncated by the kill** — the eval block's true demand is `≥ 125.6 GB` with **no upper bound**. Two
unbounded quantities do not make a projection. ⛔ **I therefore do NOT claim "< 100 GB", and a
"< 100 GB projection" derived from toy RSS would have been exactly the kind of number the
pre-registration rule exists to stop.** This was registered before any run, not after the numbers
disappointed.

### 5.2 ⭐ what I hand up instead: a NAMED, FALSIFIABLE crash-phase prediction
Back-of-envelope from the Head's own measured **20.0 s/step** and the eval block's structure at
pilot (`eval_batches = dyneval_batches = 40`):

| phase | work | ≈ cost |
|---|---|---|
| `static` | 40 batches, **forward only** | ≈ 5 min |
| `dyneval` | **3 LRs × 40 batches, forward+backward+plan** ≈ a training step each | **≈ 40 min** |
| `blank_store` | 40 forwards, re-uses `static`'s executable | ≈ 5 min |
| `anytime_curve` | 5 × 40 forwards, **5 distinct compiles** | ≈ 30 min + compiles |
| `gradient_probe_final` | 4 un-jitted backwards | minutes + a large eager peak |

Attempt 1 died **≈ 45 min into the eval block** (23:04:33 elapsed − 22.2 h training).
⇒ **PREDICTION: the kill lands at the end of `dyneval` / the `dyneval → blank_store` seam** — the
eval block's only *sustained* backward phase, and the one that (before this change) ran with the
whole training graph's executables still resident. **Falsified if the first `[rss]` series shows the
peak at `anytime_curve` or `gradient_probe_final` instead.** Either way the next run *names* it,
which is the deliverable.

### 5.3 the levers, ordered, with what each is actually worth

| # | lever | measured/estimated | verdict |
|---|---|---|---|
| i | **`eval_cache_hygiene`** (default ON) | toy: **1.433 GB reclaimed**, 0.824 GB at one boundary | ⭐ the only lever that targets the named mechanism (compile spikes stacking on retained one-shot executables). Unquantifiable at pilot from here. |
| ii | `plan_workers` 8→4 | ⭐ **MEASURED: 4 workers = 1.28 GB ⇒ ~0.32 GB each ⇒ 8→4 saves ≈ 1.3 GB** | ⛔ **a decoy — ~1 % of a ≥ 125.6 GB peak.** PREREG P4 registered 1.2–3.2 GB and "too small to matter": **CONFIRMED.** Do not spend a uniform-SET change on it. |
| iii | **`probe_lanes=1`** | cuts `gradient_probe_final`, the eval block's biggest RSS grower at toy (+0.93 GB) and its **only un-jitted** phase, 8× | ⚠ **Hub/Head decision, not mine** — it moves a *published* S2 magnitude. The contrast S2 claims is lane-invariant; the magnitudes are not. **This is the lever to reach for if §5.2's prediction is falsified toward `gradient_probe_final`.** |
| iv | drop `--d5` | removes 5 one-shot compiles from the eval block | available; the Head's cut order is withdrawn, so this is a memory decision only |
| v | `-G 2 -c 24` = 240 GB | one idle GPU, halved concurrency | ⛔ submission-side, Head's call |

### 5.4 ⚠ the price of (i), honestly
Toy end-to-end: **231 s (old) → 287 s (new) = +24 %**. ⛔ **That number does not transfer** — a
6-training-step toy run is almost entirely compile, so re-compiles dominate it. At pilot the hygiene
pass fires ≈ **28 times per seed** (≈12 for `clu_store`, ≈4 for each other arm) and what it forces
back through XLA is mainly `plan_pass`'s three `filter_jit`ed stages (`_embed_stream`,
`_block_chunk_latents`, `_block_forward`) plus one extra compile of the eval forward. At a generous
1–2 min per `_block_forward` compile that is **≈ 30–60 min per seed against a ~30 h job ⇒ ~2 %**,
inside the `-t 4-0` headroom. **Escape hatch if the `t` stamps in the `[rss]` lines show compile
thrash: `SET="… eval_cache_hygiene=false"`** — a one-flag revert to the attempt-1 behaviour.

---

## 6. How I verified (commands + observed output)

```
$ bash .claude/scratch/pilot-checkpoint-resume/gate.sh          # 4 toy runs, ~14 min
  (1) NEW, uninterrupted           rc=0    wrote pilot_toy_seed0_S3.json (287s)
  (2a) NEW, killed after arm 1     rc=137  ⛔ stop_after_arms=1: hard-exiting after 'clu_store'
  (2b) NEW, --resume               rc=0    wrote pilot_toy_seed0_S3.json (22s)
  (3) OLD path (8efc1d8)           rc=0    wrote pilot_toy_seed0_S3.json (231s)

$ .venv/bin/python .claude/scratch/pilot-checkpoint-resume/compare.py gate_old gate_new  "(a)"
  -> 16/16 named decision fields BITWISE
  -> full-record walk (timing keys excluded): 0 differing leaves
$ ...                                                            gate_new gate_res  "(b)"
  -> 16/16 named decision fields BITWISE
  -> full-record walk (timing keys excluded): 0 differing leaves

$ .venv/bin/python -m pytest tests/test_pilot_checkpoint_resume.py -q --no-cov
  13 passed in 371.46s (0:06:11)
  # ⚠ honest: the FIRST run of this file was 12 passed / 1 FAILED — my own test called
  #   run_pilot() without resume=True, so the refusal could not fire. Fixed, re-run green.

$ .venv/bin/python -m pytest -q --no-cov                          # FULL SUITE @ c9709b7
  ✅ 1361 passed, 24 warnings in 1742.30s (0:29:02)      # = 1348 baseline + 13

$ .venv/bin/python -m ruff check chlu/ tests/
  All checks passed!
$ bash -n scripts/csf3/job_gpu_cluformer.sh
  OK
$ .venv/bin/python -u -m chlu.experiments.exp_cluformer_pilot --scale toy --stage s1 --quick \
      --arms clu_store --set plan_workers=4 --out .../pool4
  [rss] monitors_init/exit    rss 1.05 GB | peak 1.08 GB | children 1.28 GB (n=4)
```

Raw artifacts under `.claude/outputs/pilot-checkpoint-resume/`: `PREREG.md`, `gate_a.txt`,
`gate_b.txt`, `rss_phases.txt`, `gate_{old,new,res}/` (records + journals + `.eqx`), the four run
logs, `pool4.log`, `pytest_new.log`, `pytest_full.log`.

---

## 7. PREREG SCORECARD

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | old vs new **bitwise, no tolerance band** | **0 differing leaves**, 16/16 decision fields | ✅ **CONFIRMED exactly as registered** |
| **P2** | kill+resume bitwise on the remaining arms; and my `monitors_final` move makes even monitor #6 bitwise | **0 differing leaves**; `n_applicable` 10 = 10 | ✅ **CONFIRMED, including the registered mechanism** |
| **P3** | phase ranking (`gradient_probe_final` #1, `anytime_curve` #2, `dyneval` #3, forwards last) **and** ≥ 50 MB reclaimed at ≥ 1 boundary | reclaim **824 MB** (16× the gate) ✅. Ranking: **REFUTED in placement** — the peak is set at *init* by `gradient_probe_init`/`allocation_liveness_init`, and the eval block adds **no new peak** at toy. The *mechanism* I named (un-jitted `filter_value_and_grad`) was right; I put it in the wrong phase. | ◐ **gate passed, ranking refuted** — and the refutation is the §4.3 finding |
| **P4** | ⚠ registered that I **cannot** bound the pilot host RSS from this machine, and that `plan_workers` 8→4 is a **decoy worth 1.2–3.2 GB** | did not bound it, and said so (§5.1); 8→4 measured **≈ 1.3 GB** | ✅ **both halves CONFIRMED** — including the registered inability |
| **P5** | 1348 + new, 0 failed; ruff green | **1361 passed / 0 failed** (= 1348 + 13); ruff green | ✅ **CONFIRMED** |

Registered-but-NOT-measured, declared: **anything on an A100/CSF3**; the absolute pilot host-RSS
peak; the pilot compile cost of the hygiene pass (§5.4 is an argued bound, not a measurement).

---

## 8. Git footprint

- **Branch** `agent/experiment-engineer/pilot-checkpoint-resume`, off `main` = `8efc1d8`. **Not
  pushed.** Rebase onto local `main` was a no-op (base unmoved; ⚠ `origin/main` deliberately NOT
  used, §7.21). No conflicts. Clean tree at hand-off; **zero worktrees** (the throwaway
  `../CHLU-pcr-base` was detached, took no commits, and is removed — `git worktree list` shows only
  the main checkout).
- **Commits (4):** `d30e4b4` (host-memory hygiene + RSS instrument + the timing-print erratum) ·
  `cd73f81` (the journal + `--resume` + `RESUME=1`) · `f4264a7` (13 tests) · `c9709b7` (the refusal
  names the differing flag).
- **Diff vs `main`: `+905 / −57` over 4 files.**
  `chlu/training/train_cluformer.py` (3 `PilotConfig` fields; `_proc_status_kb`/`_ps_rss_kb`/
  `host_rss`/`release_host_memory`; `anytime_curve` gains `hygiene`; `save_json` gains `atomic`;
  9 lines in `train_arm`'s log block; `__all__`) ·
  `chlu/experiments/exp_cluformer_pilot.py` (`_Phases`, journal helpers, `run_pilot` body,
  `_arm_row`, `_finish`, CLI `--resume` + help) ·
  `scripts/csf3/job_gpu_cluformer.sh` (`RESUME` passthrough + a host-RAM header block) ·
  `tests/test_pilot_checkpoint_resume.py` (new).
- ⛔ **Not touched:** `chlu/config.py` (standing read-only to C2 engineers), `blocks.py`,
  `clu_system.py`, `monitors.py`, `psi_readout.py`, the lane pool, the jitted plan-pass stages,
  `chlu/cli/experiment_cmd.py`, any other experiment.

---

## 9. Open questions / follow-ups / risks

1. ⭐ **The budget is not closed and cannot be closed from here** (§5.1). The next CSF3 run is the
   measurement. **Ask the Head to send back the first ~200 `[rss]` lines of the eval block** — that
   settles §5.2's prediction and picks lever (iii)/(iv)/(v) on evidence.
2. **`probe_lanes` needs an owner** (reconciliation item 3). It is the biggest un-taken cut in the
   eval block and it costs a published magnitude.
3. **The un-jitted `filter_value_and_grad(loss_fn)` sites are a standing memory design fact** (§4.3),
   not a bug. Jitting them would very likely cut the peak — and would re-associate float32 sums,
   i.e. move `gradient_probe`'s published numbers. ⛔ Out of scope; **do not do it mid-ablation.**
4. **Checkpoint disk:** ~0.5–1 GB per seed per leg ⇒ ~3–6 GB over six legs on `~/scratch`. Trivial,
   but it is new I/O in a job that had none.
5. **`wall_s` is the one field that accumulates across a resume** (compute time in the arm, summed
   over segments) rather than being wall-clock-from-arm-start. Documented; it is a timing field and
   was already outside the bitwise claim.
6. **`--resume` does not resume MID-arm.** A kill at training step 3900 of 4000 still costs the
   whole arm. Mid-training checkpointing (every N steps) is a strictly larger change — it would have
   to bank the optimiser state and the monitor registry — and was not asked for. Worth ~1 h if a
   *training*-phase crash ever happens; attempt 1's did not.

---

## Proposed handover updates (for the Hub)

- **§7 NEW (ops):** *`pilot-checkpoint-resume` makes `run_pilot` crash-resumable.*
  `pilot_{scale}_seed{N}_PARTIAL.json` (atomic, rewritten per phase) + `ckpt_{arm}_seed{N}.eqx`
  (written the instant training returns, **before** any eval — attempt 1 died in the eval block, so
  a per-*arm*-completion checkpoint would have saved nothing). `--resume` / `RESUME=1` lifts banked
  phases and skips completed arms; it **refuses** a journal from a different config and names the
  differing key. Toy gate: **0 differing leaves** vs old code AND vs an uninterrupted run.
  ⭐ **`RESUME=1` is safe to leave in the line permanently** (no journal ⇒ normal run).
- **§7 NEW (ops, MANDATORY):** *per-leg `OUT` is now required, not advisable* — the journal and the
  `.eqx` files are per-`OUT`. Mitigating fact: the run-1/run-2 legs differ by
  `memory.psi_payload_residual`, which is **inside the resume fingerprint**, so a cross-leg resume
  is refused rather than silently mixed. The final-artifact overwrite hazard is unchanged.
- **§7 CLOSE the `csf3-memory-fit` §7 erratum:** the `wall_s`/`plan_s` per-25-steps print **now
  exists** (`[train/<arm>] step i/N | nll … bpc … | wall_s … (x.xx s/step) | plan_s … (yy%)`) and is
  tested. The 2026-08-02 "cosmetic erratum" line can be retired.
- **§7 NEW (config):** `PilotConfig.eval_cache_hygiene` (⭐ **default ON**, `jax.clear_caches()` +
  `gc.collect()` at every eval-phase boundary), `.rss_log` (⭐ **default ON**, per-phase
  `VmRSS`/`VmHWM` + plan-pool children to stdout and the PARTIAL), `.stop_after_arms` (default 0,
  test/ops hook, exempt from the resume fingerprint). Unlike `csf3-memory-fit`'s levers these ship
  **ON**; §A20.4 holds because all six legs rerun on this code uniformly.
- **New standing fact (memory):** **the memory-heavy pattern in this codebase is
  `eqx.filter_value_and_grad(loss_fn)` called WITHOUT `filter_jit`.** Three sites:
  `allocation_liveness` (cut by `liveness_lanes`), `gradient_probe` (**uncut**, ×2 per seed),
  `dynamic_eval`'s `n_micro == 1` branch. At toy these two at-init backwards set **63 % of the whole
  run's peak host RSS**, before any training.
- **New never-quote candidate:** *"`plan_workers` 8→4 is a memory fix"* — **measured ≈ 1.3 GB**, ~1 %
  of a ≥ 125.6 GB peak. Pre-registered as a decoy and confirmed as one.
- **Watch-item for the next CSF3 run (falsifiable, pre-registered):** the eval-block kill is
  predicted at the **`dyneval → blank_store` seam** (≈ 45 min of eval, of which `dyneval` ≈ 40 min).
  The `[rss]` series settles it. If it lands on `gradient_probe_final` instead, `probe_lanes=1` is
  the lever and it needs a Hub ruling because it moves a published S2 magnitude.
- **⛔ Do NOT record a "< 100 GB projection" from this task.** It was pre-registered as
  unobtainable from an agent machine (CPU backend folds device memory into RSS; attempt 1's MaxRSS
  is truncated by the kill) and it was not obtained. The instrument, not the number, is what shipped.
