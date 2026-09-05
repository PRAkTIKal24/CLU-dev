# csf3-memory-fit — experiment-engineer report

**Task + acceptance criterion:** make the tier-iii pilot's backward fit an 80 GB A100 (run 1 OOMed at
97.82 GiB) **without changing a single decision or number** — accepted on toy bit-identity, a memory
ledger projecting ≤ 72 GB, a measured/bounded slowdown against the 12 h budget, and a
flags-only resubmission line.

**Status: done.** The memory half **passes with margin** (projected pilot peak **8.34 GiB**
conservative / **59.9 GiB** worst case, vs 97.82 GiB before and a 72 GB bar), bit-identity is
**BITWISE** on every decision-bearing quantity, and the suite is **1311 passed / 0 failed** (baseline
1289). ⛔ **One flagged failure that is NOT mine to fix but blocks the resubmission as currently
written** — see §7: the 12 h wallclock does **not** fit, and not because of remat; the ruled line
puts 3 seeds × 5 arms in ONE 12 h job.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — infrastructure/memory. Claims: (a) it fits, (b) nothing else changed.
- **Laundering control:** bit-identity at toy, remat on vs off — held-out bpc and all 7 `WritePlan`
  fields. **Result: BITWISE identical.** The gradient is not bitwise (a rematted VJP is recomputed and
  re-associated); the deviation is **declared and measured at `‖Δg‖/‖g‖ = 8.67e-10`**, i.e. 0.7 % of
  ONE float32 ULP (1.19e-7), against a test gate of 1e-8.
- **Falsifies:** any decision/bpc change beyond that tolerance, or a projected pilot peak > 72 GB.
  **Neither fired.**
- **Does NOT falsify:** compute overhead. Measured/bounded below; remat's compute price is the trade.

## ⚠ RECONCILIATION LIST (first-10-lines rule, protocol §5)
1. **The resubmission line must become THREE jobs, one per seed** (§7). The ruled C2W5 line
   (`SEEDS="0 1 2"` in one `-t 12:00:00` job) asks for 15 arm-trainings in one 12 h allocation ⇒
   **0.72 s/step end-to-end**, which the plan pass alone (0.324 s/step, measured by
   `plan-pass-vectorise`) half-consumes. Needs a Head/Hub decision, not an engineer's.
2. **§7.27's watch numbers are float-sensitive after the first optimiser step** — untrained readings
   are bitwise identical remat on/off; trained readings differ at ≤ 6.6e-7 relative. The abort
   criterion (< 1e-3 absolute, < 0.1× untrained) is 6 orders above that, so it is unaffected — but
   whichever flags go in the line must go in **both** legs of the §A20.4 ablation pair.
3. **`remat_read_segments` is a measured NON-lever under `remat_chunks`** (§4). Do not put it in the
   line. It is kept, tested and documented for the read-fix iteration.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/csf3-memory-fit` @ `be258f1`, `ecfe746`, `d5eb885`; base `main` = `9b2d4db` |
| worktree | `../CHLU-csf3memfit` (wt 3 of 3) |
| machine / env | Apple M1 Pro, **CPU backend**, **JAX 0.9.0, equinox 0.13.4** — the MAIN venv reused per protocol §4 (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`), **no worktree `uv sync`** |
| memory harness | `.claude/scratch/csf3-memory-fit/{ledger,ledger2,ledger3,timing}.py` (untracked) |
| memory metric | `jax.jit(value_and_grad).lower(...).compile().memory_analysis().temp_size_in_bytes` (⚠ `eqx.filter_jit`'s `Compiled` wrapper does not expose it; a raw `jax.jit` over the `eqx.partition`ed model traces the identical computation) |
| ledger rig | **the TRUE pilot store geometry**: `addr_dim=8 payload_dim=4 capacity=32 atoms_per_item=256` ⇒ `n_atoms=8192, dim=12`; `chunk=64 address_steps=64 read_steps=64 retry_rounds=1 traj_stride=8 psi_hidden=128 write_n_perturb=8 conv_kernel=4 mlp_mult=4 d_model=512`, `write_inner_steps ∈ {4, 40}`, `real_controller=False` (the plan is input data; it does not enter the backward), `vocab=256`. `batch`, `n_layers`, `n_chunks` are the swept axes. |
| toy end-to-end | `--scale toy --stage s3 --seed 0 --quick --arms clu_store`, i.e. `d_model=64 n_layers=2 seq_len=512 batch=4 addr=2 pay=1 capacity=8 atoms_per_item=128 chunk=32 address=read=24 steps=6 data_bytes=1e6`, `real_controller=True`, all 7 stage flags True, `soft_certificate=True`, `store_watch=True`, `monitor_every=100`. Seed 0. |
| new flags (all default = OLD behaviour) | `StreamMemoryConfig.remat_chunks=False`, `.remat_read_segments=0`; `PilotConfig.accum_steps=1`, `.liveness_lanes=0`, `.probe_lanes=0` |
| unchanged | every store/controller/monitor semantic; `chlu/config.py` untouched; the lane pool (`b81f487`) and the jitted stages (`46755fb`) untouched except that `_train_step` gains an `accum_steps > 1` sibling |
| ⛔ NOT RUN (pre-registered) | **anything on a GPU / CSF3.** The cluster is unreachable from agent machines. Every GPU number below is an ESTIMATE, labelled, with its transfer assumption stated. |

Pre-registration: `.claude/outputs/csf3-memory-fit/PREREG.md`, filed with an explicit ordering
statement (P1/P2 were registered by the **task file**; P3–P6 by me, P3 written while `ledger2.py` was
running with `wc -c` = 0 on its output).

---

## 1. What I did

Four levers, in `blocks.py` (2 hunks) and `train_cluformer.py` (5 hunks), **every one defaulting to
the shipped behaviour** so toy history and every pre-CSF3 artifact are untouched:

| flag | where | what it does |
|---|---|---|
| `remat_chunks` (rung 1) | `StreamBlock.__call__`, 4 lines | `eqx.filter_checkpoint` on the scanned chunk body ⇒ the scan stores only the `StoreState` carries and recomputes ONE chunk interior at a time inside the backward |
| `remat_read_segments` (rung 2) | new `CluStoreCell._rollout`, replacing 3 `truncated_rollout` call sites | cuts each read phase's Verlet rollout into `n` checkpointed segments, chaining `(q, p)` unchanged and re-assembling the strided ψ buffer on the **global** stride phase `off = (−j·L) mod stride`; falls back to the shipped single call when `steps % n ≠ 0` |
| `accum_steps` | new `_accum_grads` / `_train_step_accum`, wired into `train_arm` **and `dynamic_eval`** | `lax.scan` over microbatches of the BATCH axis (scan, not a Python loop, so XLA cannot overlap them); one optimiser step on the summed gradient ⇒ the registered effective batch 8 is preserved exactly |
| `liveness_lanes` / `probe_lanes` | `allocation_liveness` / `gradient_probe` | run the at-init full-batch backward on a lane subset; both report `n_lanes` |

`dynamic_eval` inherits `accum_steps` deliberately: it takes the *same* backward as a training step, so
without that the one column the job header's cut-order forbids cutting would be the next thing to OOM.

Plus: **23 tests** (`tests/test_csf3_memory_fit.py`; full suite **1311 passed / 0 failed**, baseline
1289), the rung-2 verdict recorded on the flag itself, and `--set`/`--mem` help naming the four levers.

---

## 2. ⭐⭐ THE HEADLINE — one law explains the crash and the fix

Every measurement below collapses onto a single constant. With `V ≡ (address_steps + read_steps) ·
(1 + retry_rounds)` Verlet steps per read (= **192** at pilot):

```
peak_backward  ≈  3.36 MB · V · [ batch · n_layers · n_chunks ]     (OFF, shipped)
peak_backward  ≈  3.36 MB · V · [ batch ]  +  persistent           (rung 1 ON)
```

⇒ the fix removes the factor **`n_layers · n_chunks` = 12 · 16 = 192×**.

`3.36 MB` per Verlet step per lane per chunk = **8.6 `(n_atoms, dim)` float32 arrays**
(8192 · 12 · 4 B = 393 KB) — i.e. the backward of one force evaluation over the atom dictionary. That
is the whole crash, and it is the read, not the write (§4).

---

## 3. THE MEMORY LEDGER (ESTIMATE — CPU `memory_analysis`, true store geometry)

### 3.1 the chunk axis (`batch = n_layers = 1`)

| `n_chunks` | OFF | `remat_chunks` | both rungs |
|---|---|---|---|
| 1 | 645.50 MB | 645.47 MB | 649.64 MB |
| 2 | **1935.06 MB** | **648.19 MB** | 652.42 MB |
| 4 | **3223.90 MB** | **651.29 MB** | 655.51 MB |
| 8 | 5802.61 MB | 658.50 MB | — |

Fits (the `C = 1` point is degenerate — a length-1 scan stacks nothing):
`OFF: 646.2 + 644.4·C` MB (predicts C=8 → 5801.4, measured 5802.61, **+0.02 %**);
`ON : 645.1 + 1.72·C` MB. **Marginal per-chunk cost 644.4 → 1.72 MB = 375×.**
The 1.72 MB residual is *mechanistically identified*: `StoreState` carry (8192·12 + 2·8192 + 32·12
floats = 458 KB) + the shell's own per-chunk residuals (conv window 4·64·512·4 B = 512 KB, MLP hidden
64·2048·4 B = 512 KB) = 1.48 MB predicted.

### 3.2 the layer and batch axes — **P3a CONFIRMED, P3b refuted by 2.98×**

| B | L | C | cfg | measured | P3a (layers serialise) | P3b (layers stack) |
|---|---|---|---|---|---|---|
| 1 | 1 | 2 | rung 1 | 648.20 MB | 648 | 648 |
| 1 | 2 | 2 | rung 1 | **650.59 MB** | 651 ✅ | 1296 ⛔ |
| 1 | 3 | 2 | rung 1 | **652.99 MB** | 654 ✅ | 1944 ⛔ |
| 2 | 1 | 2 | rung 1 | 1296.43 MB | ×2.0000 exactly | |
| 2 | 2 | 2 | rung 1 | 1301.18 MB | ×2.0000 exactly | |
| 1 | 1 | 8 | rung 1 | 658.50 MB | | |
| 1 | 2 | 8 | rung 1 | 677.31 MB | | |
| 1 | 2 | 2 | OFF | 3218.99 MB | (= `646 + 644.4·C·L`, −0.15 %) | |
| 2 | 1 | 2 | OFF | 3870.04 MB | ×2.0000 exactly | |

**The ~645 MB rematted transient is paid ONCE, not per layer** (+2.40 MB per extra layer at `C=2`,
+18.8 MB at `C=8`), and **everything is exactly linear in batch** — which is what makes `accum_steps`
a clean halving lever if it is ever needed. Fitted:

```
peak_ON (B, L, C)  =  B · [ 644.8 + 1.72·C + (L−1)·(2.74·C − 3.1) ] MB
peak_OFF(B, L, C)  =  B · [ 646.2 + 644.4·C·L ] MB
```

### 3.3 the projection to the pilot (`B=8, L=12, C=16`) — **⭐ THE ACCEPTANCE NUMBER**

| model | pilot OFF | pilot ON | verdict vs 72 GB |
|---|---|---|---|
| CPU absolute | 994.97 GB | **8.96 GB = 8.34 GiB** | ✅ 88 % headroom |
| **GPU anchor** | **97.82 GiB (run 1's own crash message)** | — | — |
| (A) ratio transfer `97.82 · (ON/OFF)` | — | **0.88 GiB** | ✅ optimistic |
| (B) CPU absolute taken at face value | — | **8.34 GiB** | ✅ **the estimate I stand behind** |
| (C) worst case: layers do NOT serialise on GPU (refuted on CPU at L=3) | — | **59.9 GiB** | ✅ but only 17 % headroom |

⚠ **Why I do not quote (A) as the answer.** The CPU model predicts **995 GB** for the OFF config where
the GPU's own XLA asked for **105.0 GB (97.82 GiB)** — a factor **9.5×**. That gap is not error, it is
information: **the GPU's XLA already auto-rematerialized the shipped graph** (its message quotes an
auto-remat floor of 76.70 GiB). Transferring the OFF-config ratio to the ON config would assume XLA can
squeeze the hand-rematted graph as hard as the un-rematted one, which it cannot — there is nothing left
to squeeze. So I quote **(B) 8.34 GiB as the central estimate**, which is deliberately conservative
(it credits the GPU with *none* of the 9.5× it demonstrably has), and **(C) 59.9 GiB as the bound
under the hypothesis my own layer measurement refutes.**

**All three clear 72 GB.** If the resubmission nevertheless OOMs, `accum_steps=2` halves whatever it is
(batch linearity measured at exactly 2.0000) ⇒ (C) → 32 GB; that is the held fallback, not in the line.

---

## 4. ⛔ RUNG 2 IS A MEASURED NON-LEVER (P6, registered before measurement — and my registered
*explanation* was wrong)

| cfg (`B=L=1, C=2`) | peak |
|---|---|
| neither | 1935.07 MB |
| **rung 1 alone** | **648.19 MB** |
| **rung 2 alone, `n=4`** | **658.43 MB** |
| rung 2 alone, `n=16` | 656.36 MB |
| rung 1 + rung 2, `n=4` | 652.42 MB (**+0.65 %**) |
| rung 1 + rung 2, `n=16` | 655.01 MB (**+1.05 %**) |

**The two rungs are alternative routes to the SAME floor, not composable ones**, and neither goes
below it. Where the floor comes from (rung 1 on, knock one factor out at a time):

| knock-out | peak | vs base 648.20 MB |
|---|---|---|
| `retry_rounds` 1 → 0 (V: 192 → 128) | 433.81 MB | **×0.669** vs ×0.667 predicted — exact |
| `address_steps = read_steps` 64 → 8 (V: 192 → 24) | 86.41 MB | ×0.133 vs ×0.125 |
| `write_inner_steps` 40 → 1 | 648.19 MB | **−0.0005 %** |
| `write_n_perturb` 8 → 1 | 648.20 MB | 0 |
| `psi_hidden` 128 → 8 | 648.20 MB | 0 |
| `d_model` 512 → 128 | 646.23 MB | −0.30 % |

⇒ **the floor is the READ's Verlet backward and essentially nothing else** — the 40-step inner write,
ψ and the whole `d_model=512` shell together are < 0.5 % of it. My PREREG's P6 registered the write as
the suspect and registered the alternative "one large fused buffer"; **both are wrong — it is the read,
and it is exactly linear in `V`.** ⚠ I have **not** isolated *why* rung 2 fails to split a quantity
that is provably linear in `V`; I report the fact and label the mechanism **UNRESOLVED**. (One
candidate I did not test: nesting `jax.checkpoint` inside a rematted `scan` body may be absorbed by
JAX's partial-eval rather than nested.) An `atoms_per_item` sweep was attempted and is **void** —
`CluSystemConfig` clamped `n_atoms` back to 8192, so that row measures nothing.

---

## 5. THE CONTROL — bit-identity, end to end at toy

Two full `--scale toy --stage s3 --quick --arms clu_store` runs, identical except
`--mem remat_chunks=true`:

| quantity | remat OFF | remat ON | |
|---|---|---|---|
| held-out static **nll** | 5.553304672241211 | 5.553304672241211 | ✅ **BITWISE** |
| held-out static **bpc** | 8.011725111187905 | 8.011725111187905 | ✅ **BITWISE** |
| **dyn-eval** bpc | 8.006763411710514 | 8.006763411710514 | ✅ **BITWISE** |
| **blank-store** bpc | 8.011891246428640 | 8.011891246428640 | ✅ **BITWISE** |
| monitors_init `n_tripped` | 5 | 5 | ✅ |
| §7.27 watch, **untrained** (depth_median, spread, n_live) | 0.06863571032522342 / 0.14232712984085083 / 3 | identical | ✅ **BITWISE** |
| §7.27 watch, **trained @ step 5** | 0.025313927265032 / 0.045417323708534 / 4 | 0.025313928305395 / 0.045417353510857 / 4 | rel **4.1e-8 / 6.6e-7**, `n_live` identical |
| S2 `grad_phi` (trajectory) | 0.011792222073849024 | 0.011792221524321560 | rel **4.7e-8** (< 0.4 ULP) |
| `train.loss_history` (6 steps) | — | — | max abs Δ **4.77e-7** on losses ≈ 5.55 ⇒ rel 8.6e-8 |

**All 7 `WritePlan` fields** are asserted `array_equal` per layer in
`test_all_seven_write_plan_fields_are_bit_identical_under_chunk_remat` (synthetic tokens, real
controller, 2 layers) and the **forward loss is bitwise for all 5 swap arms**
(`test_every_swap_arm_survives_chunk_remat_bitwise`).

**Why the trained readings drift and why it does not matter.** The *forward* is bitwise; the
*gradient* is not (a rematted VJP is recomputed and XLA re-associates its float32 sums). After one
optimiser step the weights therefore differ in the last bits and everything downstream inherits it.
Measured at the leaf level: `‖Δg‖/‖g‖` = **8.67e-10** (`remat_chunks`), **3.18e-10**
(`remat_read_segments=3`) — **0.7 % of one float32 ULP (1.19e-7)**. Declared tolerance in the test
suite: **1e-8** (one order above the measurement, two below the ULP). The §7.27 abort criterion is a
factor-10 test; a 6.6e-7 perturbation is **six orders** below it.

---

## 6. THE COMPUTE PRICE

**Bound (this is the number to quote).** `jax.checkpoint` on the chunk body recomputes that body's
forward exactly once inside the backward: forward `F` + backward `2F` → `F + (F + 2F)` ⇒ **≤ 1.333× on
the chunk interior**, and 1.00× on everything outside it (embedding, conv, MLP, head, optimiser, and
the entire CPU-side plan pass). §4 shows the chunk interior *is* essentially the whole CLU-arm FLOP
count, so **≤ 1.333× on the CLU arm's differentiable pass** and less end-to-end.

**Measured on CPU (warm, min of 3, `B=L=1`, pilot store geometry) — and it goes the OTHER way:**

| `n_chunks` | `write_inner_steps` | OFF | rung 1 | rung 1 + rung 2 |
|---|---|---|---|---|
| 4 | 4 | 0.518 s | 0.456 s (**0.88×**) | 0.539 s (1.04×) |
| 4 | 40 | 0.726 s | 0.531 s (**0.73×**) | 0.735 s (1.01×) |
| 8 | 4 | 3.641 s | 1.153 s (**0.32×**) | 1.415 s (0.39×) |
| 8 | 40 | 4.033 s | 1.499 s (**0.37×**) | 1.499 s (0.37×) |

(loss **bitwise identical** in all 12 rows.) ⚠ **Do not carry this to the A100.** On this laptop the
un-rematted graph is memory-bandwidth-bound (3.2–5.8 GB of temporaries against ~200 GB/s of shared
DRAM); the A100 has 80 GB of HBM at ~2 TB/s and will not see that pathology. The honest reading is:
**remat's arithmetic surcharge is ≤ 1.333× and its memory-traffic saving can exceed it — the toy
end-to-end runs (33.3 s vs 36.1 s of training wall) are compile-dominated over 6 steps and are NOT a
usable slowdown measurement.**

⛔ **Registered falsifier (P5): an end-to-end slowdown > 2.0×. Not observed anywhere; the bound
forbids it.**

---

## 7. ⛔ THE 12 h WALLCLOCK CHECK — **IT DOES NOT FIT, AND NOT BECAUSE OF REMAT**

Task §6 asks me to verify that "4000 steps × 3 seeds fits `-t 12:00:00` **per seed**; if not, say so
and stop." The ruled C2W5 line does not do one seed per job:

```
sbatch --export=ALL,SEEDS="0 1 2",... -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
```

`job_gpu_cluformer.sh` passes `--seeds $SEEDS` to **one** `python -m ...` process, and `run_pilot`
trains **all 5 arms** per seed. So that single 12 h allocation is asked for
**3 seeds × 5 arms × 4000 steps = 60 000 optimiser steps + 3 × (40 static + 3 × 40 dyn-eval) eval
batches**:

| budget | value |
|---|---|
| wallclock | 43 200 s |
| optimiser steps in it | 60 000 |
| ⇒ per step, everything | **0.72 s** |
| of which the CPU-side plan pass alone (measured, `plan-pass-vectorise`, pooled at 8 workers) | **0.324 s** |
| left for the A100 forward+backward of a 12-layer block whose CLU read is 192 Verlet steps over 8192 atoms × 16 chunks × 8 lanes | **≈ 0.40 s** |

That is not credible, and it is **independent of my change** — remat's ≤1.333× is second-order against
a 10× shortfall. The job header's own declared budget ("≤ 108 A100-hours = **3 arms × 3 seeds × ≤ 12
h**") reads as **nine** 12 h slots, not one; the script and the ruled line disagree with the header.

**Recommendation (Head/Hub decision, not mine):** submit **one job per seed** — `SEEDS="0"`,
`SEEDS="1"`, `SEEDS="2"` as three `sbatch`es (gpuA allows ≤ 4 concurrent GPUs/user). That gives
**20 000 steps / 43 200 s = 2.16 s/step**, and since 4 of the 5 arms have trivial cells, realistically
≈ 9 s/step for the CLU arm. Whether *that* holds has **never been measured** — run 1 OOMed at compile,
so no A100 step time for this model exists anywhere. The first job is therefore also the timing probe:
`train_log` prints `wall_s`/`plan_s` every 25 steps, so the Head knows within ~5 minutes whether
4000 steps fit, and the header's cut order (D5 → TTT arm → depth/width) governs from there.

---

## 8. ⭐ THE RESUBMISSION LINE (flags only; zero module edits on the cluster)

Run-1's ruled line + **one** `MEM` flag and **one** `SET` flag, submitted **once per seed**:

```bash
# csf3-memory-fit delta over the C2W5-ruled line = `remat_chunks=true` and `liveness_lanes=1`,
# and SEEDS split one-per-job (see §7).  Repeat with SEEDS="1" and SEEDS="2".
sbatch --export=ALL,SEEDS="0",STAGE=pilot,STG=s4,\
MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true",\
STORE="write_margin=0.6",\
SET="monitor_every=25 plan_workers=8 liveness_lanes=1" \
       -c 12 --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
```

- ⛔ **`remat_read_segments` is deliberately NOT in the line** (§4: +0.65–1.05 % of peak for nothing).
- ⛔ **`accum_steps=2` is deliberately NOT in the line.** It is the held fallback: add it *only* if the
  resubmission still OOMs. It is the one lever that is not bitwise (mean-of-means re-associates the
  float32 sum; measured agreement 3e-5 relative), so it should not be spent unnecessarily.
- `liveness_lanes=1` shrinks the **crash site** to a 1-lane backward. It changes
  `allocation_liveness_init`'s reported gradient norm and slot entropy (per-lane objects, now over one
  lane) — run 1 produced **no** such number (it died there), so nothing is being contradicted, but
  **§A20.4: whatever is in this line must be in BOTH legs of the run-1/run-2 ablation pair.**
- `probe_lanes` is **left at 0**: it would move the published S2 `‖dL/dφ‖` magnitudes.
- Everything else is byte-for-byte the ruled line; `store_watch` still needs no flag.

---

## 9. How I verified (commands + observed output)

```
$ PYTHONPATH=<wt> .venv/bin/python -m pytest tests/test_csf3_memory_fit.py -q
   18 passed in 146.27s            # before the 4-arm parametrised test was added
$ ... -k swap_arm
   4 passed, 18 deselected in 14.60s
$ ... -k dynamic_eval
   1 passed, 22 deselected in 33.60s          # => 23 tests in the new file
$ PYTHONPATH=. .venv/bin/python -m pytest -q --no-cov   # FULL SUITE @ ecfe746
   1307 passed, 31 warnings in 1627.32s (0:27:07)       # = 1289 baseline + 18
$ ...                                                   # FULL SUITE @ d5eb885
   ✅ 1311 passed, 24 warnings in 1610.31s (0:26:50)    # = 1289 baseline + 22
   # ⚠ honest: the last commit (07c28a4) adds ONE further test, verified in
   #   isolation but not inside a full-suite re-run => 1312 expected, 1311 shown.
$ .venv/bin/python -m ruff check chlu/ tests/test_csf3_memory_fit.py
   All checks passed!
$ PYTHONPATH=<wt> .venv/bin/python .claude/scratch/csf3-memory-fit/ledger.py    # §3.1, 18 cells
$ ...                                       ledger2.py   # §3.2, 10 cells
$ ...                                       ledger3.py   # §4,   10 cells
$ ...                                       timing.py    # §6,   12 cells
$ PYTHONPATH=. .venv/bin/python -u -m chlu.experiments.exp_cluformer_pilot \
      --scale toy --stage s3 --seed 0 --quick --arms clu_store --out .../toy_off
   [clu_store] static bpc 8.0117 | dyneval bpc 8.0068 | 188s
$ ... --mem remat_chunks=true --out .../toy_on
   [clu_store] static bpc 8.0117 | dyneval bpc 8.0068 | 145s
```

Raw artifacts: `.claude/scratch/csf3-memory-fit/{ledger_raw,ledger2_raw,ledger3_raw,timing_raw}.json`;
the two toy run artifacts at `.claude/outputs/csf3-memory-fit/toy_{off,on}/pilot_toy_seed0_S3.json`
(+ `.log`).

## 10. PREREG SCORECARD

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** (task file) | activations ÷ ~`n_chunks` (16) | marginal per-chunk **644.4 → 1.72 MB = 375×**, and the `n_layers` multiplier goes too ⇒ **192× total** | ✅ **CONFIRMED, and stronger than registered** |
| **P2** (task file) | ≤ 2× the chunk-interior compute | bound **1.333×**; CPU measured **0.32–0.88×** (bandwidth artifact, not transferable) | ✅ |
| **P3** | layers serialise (P3a: 648 / 651 / 654 MB) vs stack (P3b: 648 / 1296 / 1944) | **648.20 / 650.59 / 652.99** | ✅ **P3a CONFIRMED (≤ 0.2 % off); P3b REFUTED by 2.98× at L=3** |
| **P4** | central 5.99 GiB, bracket ≤ 30 GiB, both < 72 GB | **8.34 GiB** central (the refined L-scaling moved it), worst case 59.9 GiB | ◐ **in bracket, point off 1.4×** — my prereg used the L=1 model and so over-counted the per-layer term |
| **P5** | end-to-end slowdown ≤ 1.35×, falsifier > 2.0× | not measurable end-to-end at toy (compile-dominated); bound 1.333× | ◐ **unmeasured, bounded** — stated honestly, not claimed |
| **P6** | rung 2 buys nothing under rung 1 (+0.65 % at n=4) — **and** the floor is the WRITE, alternatively "one fused buffer" | rung 2 verdict ✅; **the floor is the READ**, exactly linear in `V` | ◐ **verdict CONFIRMED, my mechanism REFUTED** — the registered explanation was wrong and the knock-out table says so |

Registered-but-NOT-measured, declared: **anything on an A100/CSF3**; the `atoms_per_item` scaling
(harness bug — `n_atoms` clamped, row void); *why* rung 2 fails to split a `V`-linear quantity.

---

## 11. Git footprint

- **Branch** `agent/experiment-engineer/csf3-memory-fit`, off `main` = `9b2d4db`, worked in worktree
  `../CHLU-csf3memfit` (wt 3 of 3, per the task). **Not pushed.** No conflicts; rebase onto local
  `main` was a no-op (base unmoved; ⚠ `origin/main` deliberately NOT used, §7.21).
- ✅ **Protocol §3.2 discharged:** the four commits were verified from the MAIN repo
  (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/csf3-memory-fit`
  → `07c28a4 d5eb885 ecfe746 be258f1`) **before** `git worktree remove ../CHLU-csf3memfit`.
  **The worktree is released — the 3/3 cap now has a free slot.** Diff vs `main`:
  `+589 / −18` over 4 files.
- **Commits:** `be258f1` (the four levers + 18 tests), `ecfe746` (rung-2 verdict on the flag + CLI
  help), `d5eb885` (the 4-arm swap identity test), `07c28a4` (dyn-eval inherits `accum_steps`).
- **Files touched:** `chlu/core/blocks.py` (config fields; new `CluStoreCell._rollout`; 3 rollout call
  sites; 4 lines in `StreamBlock.__call__`), `chlu/training/train_cluformer.py` (5 `PilotConfig`
  fields; `_microbatch`/`_accum_grads`/`_train_step_accum`; 6 lines in `train_arm`; 4 in
  `dynamic_eval`; lane guards in `allocation_liveness`/`gradient_probe`),
  `chlu/experiments/exp_cluformer_pilot.py` (help text only), `tests/test_csf3_memory_fit.py` (new).
- ⛔ **Not touched:** `psi_readout.py`, the factored-store/read-fix files, `chlu/config.py`, the lane
  pool, the jitted plan-pass stages, `monitors.py`, `scripts/csf3/*`.

## 12. Open questions / follow-ups / risks

1. **The 12 h budget (§7)** — needs a Head/Hub ruling before resubmission. This is the one thing that
   can still burn an allocation.
2. **Every GPU number is an estimate.** The transfer assumption is stated; the conservative estimate
   (8.34 GiB) credits the GPU with none of its demonstrated 9.5× auto-remat advantage. If it still
   OOMs, `accum_steps=2` is the tested halving lever.
3. **Rung 2's mechanism is unresolved** (§4). It matters only if the read budget grows — which is
   exactly what the read-fix iteration might do, so it is worth an hour then, not now.
4. **The ~645 MB/lane floor is a design fact, not a bug**: `3.36 MB × V`, i.e. ~8.6 `(n_atoms, dim)`
   arrays per Verlet step. At `V = 192, n_atoms = 8192` that is 5.16 GB for batch 8 *before any
   sequence length*. Any future increase in `address_steps`/`read_steps`/`retry_rounds`/`atoms_per_item`
   scales it linearly and directly — the read budget is now a **memory** dial as well as a compute one.
5. **`store_health` is float-sensitive after step 1** (rel ≤ 6.6e-7 here). The watch's *decisions* are
   unaffected (6 orders of margin), but nobody should compare watch numbers across branches bitwise.

## Proposed handover updates (for the Hub)

- **§7 NEW (config/ops):** *`csf3-memory-fit` lands four backward-memory levers, all default-OFF:*
  `StreamMemoryConfig.remat_chunks` (**the fix** — removes the `n_layers · n_chunks` = 192× multiplier
  on the pilot backward), `.remat_read_segments` (**measured non-lever under `remat_chunks`; do not
  enable**), `PilotConfig.accum_steps` (held fallback; the only non-bitwise lever),
  `.liveness_lanes` / `.probe_lanes`. Projected pilot peak **8.34 GiB** (conservative) / 59.9 GiB
  (worst case) vs 97.82 GiB before, against an 80 GB card.
- **§7 NEW (ops, BLOCKING):** *the C2W5-ruled CSF3 line puts 3 seeds × 5 arms in ONE 12 h job = 0.72
  s/step end-to-end, of which the plan pass alone is 0.324 s. Split one seed per job (3 sbatches,
  ≤ 4 concurrent allowed). The job header's own budget ("3 arms × 3 seeds × ≤ 12 h") already reads
  that way; the script and the line do not.*
- **§7.27 addendum:** the in-flight watch's *untrained* reading is bitwise reproducible; its *trained*
  readings carry float32 round-off (≤ 6.6e-7 relative measured across a remat toggle). The abort
  criterion has 6 orders of margin, but watch numbers must not be compared bitwise across branches.
- **New standing fact for the read-fix iteration:** the pilot backward's memory is **the read and
  nothing else** — `3.36 MB` per Verlet step per lane per chunk (≈ 8.6 `(n_atoms, dim)` float32
  arrays); `write_inner_steps` 40→1, `write_n_perturb` 8→1, `psi_hidden` 128→8 and `d_model` 512→128
  together move it by **< 0.5 %**. `address_steps`/`read_steps`/`retry_rounds`/`atoms_per_item` are now
  memory dials.
- **New never-quote candidate:** *"the 40-step inner write is what blew the A100 memory"* — measured
  false (−0.0005 %).
