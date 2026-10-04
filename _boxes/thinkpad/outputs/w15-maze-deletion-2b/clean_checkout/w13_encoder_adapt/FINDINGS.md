# w13 hub/encoder-adapt — FINDINGS

**Report to the Hub.** Box FORGIS_14, branch `w13/encoder-adapt`, worktree
`C:/chlu/worktrees/CHLU-w13-encoder-adapt`, base `base/iter8 = aa91785` + the folded
`w12/single-design-crossover` harness. **CPU only, 0 GPU** (none visible: `jax.devices()`
is `[CpuDevice(id=0)]` and there is no torch on this box). 4 threads
(`OMP/MKL/OpenBLAS/NumExpr = 1` per lane, at most 4 lanes).
⭐ `chlu.__file__ = C:\chlu\worktrees\CHLU-w13-encoder-adapt\chlu\__init__.py` — asserted
in code (`exp_w13_encoder_adapt.assert_chlu_in_worktree`) before any number and written
into every artefact.
⛔ **Nothing under `$CHLU_CLAUDE_ROOT` was read.** The harness's own resolvers load the
FROZEN INSTRUMENT CHECKPOINTS from there (bridge, external classifier), exactly as w8 and
w12 did; no report, findings file or number under that root was opened.
⭐ **SCORED RUNS SPENT: 1 of 3** on target `encoder-adapt` (`/c/chlu/SCORED-LEDGER.md`,
row appended 2026-09-21T21:27:53Z, **before** the first test-condition process at
21:27:58Z). Runs 2 and 3 remain.

---

## 0. THE ANSWER, in four sentences

1. ⭐ **The pre-registered bar is MET, and only just.** `anchor`'s margin grows from
   `frozen` on **3 of 3** seeds and `adapter`'s on **2 of 3**; the bar was "≥ 2 of 3",
   declared before any number.
2. ⛔⛔ **But the growth is not a store result, and this hub says so.** In **all six**
   treatment cells the single design's level and its best baseline's level move in the
   **same direction and by similar amounts**; the margin is the small residual. **That is
   a finding about the ENCODER, which every baseline also gets — not about the store.**
3. ⛔ **Nothing clears the programme's own margin.** No leg of the nine reaches
   +1.0/corner with a bootstrap CI95 low above +1.0. Seven of nine CI95s straddle zero.
4. ⛔ **The seed dominates the arm by ~5×.** The frozen single design reads
   23.312 / 11.202 / 20.294 across seeds 0/1/2 — a spread of **12.11** — while the largest
   arm effect *within* a seed is **3.51**.

---

## 1. WHAT WAS MEASURED — 3 seeds × 3 configurations = 9 arm-runs

⛔ **That is 3 SEEDS, never "9 seeds".** Seed count and configuration count are separate
numbers throughout.

| arm | encoder | role | enc params in Optax |
|---|---|---|---|
| `frozen` | frozen | ⭐ **REFERENCE** | **0** |
| `adapter` | rank-32 residual on the penultimate feature; **encoder weights frozen** | treatment | **131,072** |
| `anchor` | full unfreeze + penalty on distance from frozen weights AND features | treatment | **428,576** |

`run/seed_{0,1,2}/{arm}/arm.json → compute`. **Equal steps AND equal images on every one
of the nine runs: 400 steps × 25,600 images.** Seconds 2297/2368/2523 (seed 0),
2291/2308/2309 (seed 1), 2211/2047/1640 (seed 2) — ⭐ **equal steps is within 10 % of
equal seconds on seeds 0 and 1, and on seed 2 the treatments were FASTER than the
reference**, so no seconds-matched control changes any verdict and none is needed.
All three arms carry the identical objective — `recon + 1.0·codebook + 0.25·commitment
(= the STORE READ ERROR on fitting rows) + β_swap·swap` — on the identical batches from
the identical PRNG stream; they differ only in which encoder leaves Optax is given, plus
`anchor`'s penalty.

**One bit-identical warm start per seed, and it is visible in the number:** all three arms
start at loss **0.19270** on seed 0 (`arm.json → compute.history_first`). The adapter's
up-projection is zero-initialised, so the adapted encoder is **bit-identical to the frozen
one at step 0** — asserted on real images in `stage_train`, which refuses the run if it is
not (`arm.json → adapter.bit_identical_to_frozen_at_step_0: true`).

---

## 2. ⭐⭐ THE BAR — the MARGIN, per seed, never pooled

`score/test_seed{S}_{arm}_K32/score.json → MARGIN`. 109 corners (60 pair sub-corners +
49 n-way), TEST condition, metric `distinct_valid_novel_tuples` per corner, paired by
corner through `fw.paired_estimand`. Best learned baseline is chosen **per arm** as the
strongest of that arm's own refits; it was `mlp_180k` (178,952 params) on **all nine legs**.

| seed | arm | single design | best learned | **MARGIN** | Δ vs frozen | bootstrap CI95 | won/tied/lost | sign p |
|---|---|---|---|---|---|---|---|---|
| 0 | frozen | 23.3119 | 22.7431 | **+0.5688** | — | [−0.743, +1.917] | 50/9/50 | 1.0000 |
| 0 | adapter | 22.0642 | 20.4404 | **+1.6239** | **+1.0550** | **[+0.661, +2.606]** | 63/9/37 | **0.0120** |
| 0 | anchor | 25.5688 | 24.2385 | **+1.3303** | **+0.7615** | [−0.009, +2.725] | 55/13/41 | 0.1843 |
| 1 | frozen | 11.2018 | 11.2294 | **−0.0275** | — | [−0.807, +0.752] | 45/19/45 | 1.0000 |
| 1 | adapter | 13.3486 | 12.9817 | **+0.3670** | **+0.3945** | [−0.587, +1.394] | 47/15/47 | 1.0000 |
| 1 | anchor | 11.1560 | 11.1651 | **−0.0092** | **+0.0183** | [−0.752, +0.780] | 41/30/38 | 0.8221 |
| 2 | frozen | 20.2936 | 19.1927 | **+1.1009** | — | [−0.156, +2.303] | 68/5/36 | 0.0022 |
| 2 | adapter | 18.2844 | 17.9817 | **+0.3028** | **−0.7982** | [−0.853, +1.395] | 55/12/42 | 0.2229 |
| 2 | anchor | 20.8165 | 19.5138 | **+1.3028** | **+0.2018** | **[+0.165, +2.422]** | 56/10/43 | 0.2276 |

> ⭐ **VERDICT ON THE BAR:** `anchor` grows on **3 of 3** seeds (+0.7615 / +0.0183 /
> +0.2018). `adapter` grows on **2 of 3** (+1.0550 / +0.3945 / **−0.7982**).
> Declared bar: grow on ≥ 2 of 3. **Both treatments meet it.**

⚠ **And here is everything that qualifies it.**
- ⛔ **Nothing clears the programme's own margin of +1.0/corner with CI95 low > +1.0.**
  Two legs have a CI95 that excludes zero at all (`s0 adapter` [+0.661, +2.606];
  `s2 anchor` [+0.165, +2.422]) and neither has a low above +1.0.
- ⚠ `adapter`'s three deltas are **+1.06 / +0.39 / −0.80** — a spread of 1.85 around a mean
  of **+0.217**. `anchor`'s are **+0.76 / +0.02 / +0.20**, mean **+0.327**. On this spread
  a 3-seed mean is not a measurement of a small effect, and it is not reported as one.
- ⭐ **The margin does NOT start negative here, and the reason is declared and is against
  this hub.** Wave 12 measured −0.054…−0.077 *of ceiling* against a learner byte-matched
  to the store's **own 24.4× total**; these baselines are matched to **118,784 / 179,968
  parameters**, not to this object's own 2,123,898–2,160,818 B four-part total. A
  1×-matched rival would be **STRONGER**, so this comparison **flatters the single
  design** and the **1× column is a DECLARED NOT-RUN**. ⛔ No row here is a 1× result, and
  the sign difference from wave 12 is a difference of comparator, not of object.

---

## 3. ⛔⛔ THE TRAP FIRED: THIS IS AN ENCODER FINDING, NOT A STORE FINDING

`Δmargin = Δ(single design) − Δ(best baseline)`, both measured against that seed's
`frozen` arm, from the same nine `score.json` files:

| seed | arm | Δ single design | Δ best baseline | Δ margin | same direction? |
|---|---|---|---|---|---|
| 0 | adapter | **−1.248** | **−2.303** | +1.055 | ⭐ yes |
| 0 | anchor | **+2.257** | **+1.495** | +0.761 | ⭐ yes |
| 1 | adapter | **+2.147** | **+1.752** | +0.394 | ⭐ yes |
| 1 | anchor | −0.046 | −0.064 | +0.018 | ⭐ yes |
| 2 | adapter | **−2.009** | **−1.211** | −0.798 | ⭐ yes |
| 2 | anchor | +0.523 | +0.321 | +0.202 | ⭐ yes |

⛔⛔ **In 6 of 6 treatment cells the store and its strongest baseline move together**, and
the *level* movement (0.05–2.26 tuples/corner) is comparable to or larger than the margin
residual (0.02–1.06). **What light adaptation moved is the FEATURE SPACE, which every
learned baseline is refit on and therefore also gets.** That is the exact reading the
task's second falsifier demands, and it is the primary paragraph of this report, not a
caveat. ⭐ On seed 0 `adapter` the store got **worse** (23.31 → 22.06) and the margin grew
only because the baseline got worse faster.

⭐ **The seed dominates.** Frozen single design across seeds: **23.312 / 11.202 / 20.294**
(spread **12.11**). Largest arm effect within a seed: **3.51 / 2.19 / 2.53**. The
between-seed variation is ~5× the between-arm variation, and the judge accuracy tracks it
(seed 0 0.41–0.48, seed 1 0.21–0.25, seed 2 0.34–0.40).

---

## 4. ⭐ THE CONTROL GATE — passed on every one of the nine legs

`score.json → MARGIN.controls_must_be_zero`, and independently `per_arm`:

**`landscape_removed`, `landscape_shuffled`, `landscape_shuffled_across_wells` and
`random_atoms` read EXACTLY 0.0 on all 9 legs**, against a single design reading
11.16–25.57. `primary_degenerate: false` on all 9 — on TEST the novel metric separates and
is the one used. 60 pair + 49 n-way corners on all 9; `mlp_180k` the best baseline on all 9.

**The object, on every leg:** K\* = 32, A = 63, wells 64…64, rank/width max 0.9688–0.9844,
four-part ledger **4.051×–4.121×** the 524,288 B REFERENCE column (summed, never netted).
Address book fitted in 879–2,185 s per leg. Dead channels (union of the partition file's
and the arm's own) 64–164 of 2,048, reported per leg, never absorbed.

---

## 5. ⭐ THE FACTOR PROBE — a MARGIN over random-init, PER SEED, NEVER POOLED

`run/seed_{S}/{arm}/probe.json → MARGIN_OVER_RANDOM_INIT`:

| seed | frozen | adapter | anchor | random-init floor (absolute) |
|---|---|---|---|---|
| 0 | +0.0001 | +0.0007 | −0.0016 | 0.9735 |
| 1 | **−0.0160** | **−0.0064** | **−0.0059** | 0.9655 / 0.9655 / 0.9661 |
| 2 | +0.0095 | +0.0077 | +0.0088 | 0.9609 |

⛔ **The sign of the probe margin is not stable across seeds** — every arm is negative on
seed 1 and positive on seed 2. Absolutes are 0.9495–0.9742 against random-init floors of
0.9609–0.9661, reproducing w8's measurement that an **untrained** encoder probes Shapes3D
at 0.962–0.972 and w9's independent finding of the same instability.
⛔ **No absolute probe number here is a claim, and no arm buys anything over an untrained
encoder of the same architecture.** ⭐ Reported per seed; never pooled.

---

## 6. ⛔ THE SWAP LOSS — two NULLS, both measured

**(a) It is ILL-POSED at K\* = 32, and this stopped the wave's first configuration.**
With the in-loop store at the task's K\* = 32, `gen_swap_loss.derive_fixed_ownership`
refused by name (`dev/logs/smoke_warm_s0.log`, `chlu/core/gen_swap_loss.py:148`):

> `AssertionError: well(s) [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
> 22, 23, 24, 26, 27, 28, 29, 30, 31] own no factor -- the swap loss has no target for
> them and is ill-posed; reported, not patched`

**26 of 32 wells own no factor.** The swap loss assigns each well exactly ONE owned factor
by a Hungarian match on a measured transfer matrix, and Shapes3D has **6** factors, so at
most 6 wells can own one at any K. ⛔ **Not patched** — restricting the draw to the owning
wells would override a refusal the frozen harness makes deliberately. Resolved by
separating the two objects (AMENDMENT 1, written before any spend): the **scored** single
design stays at K\* = 32; the **in-loop VQ store** is w8's K = 5 / A = 64 object whose
atom block is 64 × 2048 × 4 = **524,288 B**, w8's pin exactly. At K = 5 ownership is
well-posed on every seed (`own = [4,1,0,3,2] / [4,1,0,3,2] / [1,3,0,5,2]`), with one
factor unowned "by construction" exactly as the frozen docstring says.

**(b) At the compute this hub could afford, the swap term does nothing.**
`swap_exact` — the fraction of swaps the judge reads back as exactly the intended tuple —
is **0.0000 at the first step on every arm and every seed**, and at the last step reads
0.0000 / 0.0000 / 0.0000 (seed 0), 0.0000 / 0.0156 / 0.0156 (seed 1), 0.0000 / 0.0312 /
0.0156 (seed 2) — i.e. at most 2 of 64. ⛔ **A NULL on the swap loss's own headline
statistic, on 3 of 3 seeds.** It is consistent with the identity ceiling measured on the
warm starts (0.1641 / 0.1523 / 0.1406, `run/seed_{S}/warm_start.json`). w8's positive swap
result (+0.0100 / +0.0129 / +0.0075) was measured on GPU at **10× steps and 4× data**; ⛔
these two are not comparable and are not put in one table.

⭐ **The swap term used here IS the frozen one**, asserted numerically on a real batch
before the first update of every run: `abs_delta_loss = 0.0` and `abs_delta_exact = 0.0`,
**exactly zero**, on all nine arm-runs (`arm.json → compute.swap_identity_vs_frozen_swap_loss`).
A non-zero delta refuses the run. (It is composed from `gen_swap_loss`'s own primitives in
that module's own order with the well's channel row traced instead of static, so K = 32
compiles a handful of XLA graphs rather than 32.)

---

## 7. ⭐ `anchor` DOES NOT DIVERGE — and `adapter` is not "light" where it counts

`arm.json → compute.displacement_final`, relative L2 **squared** (so the declared
reference displacement ρ = 0.10 corresponds to 0.01):

| seed | frozen | adapter (no penalty) | anchor (penalised) |
|---|---|---|---|
| 0 | **0.000000** | 0.157581 | **0.002146** |
| 1 | **0.000000** | 0.111499 | **0.002393** |
| 2 | **0.000000** | 0.137014 | **0.001711** |

⭐ **`anchor`'s feature displacement is 47×–80× smaller than the unpenalised `adapter`'s
and sits 4.2×–5.8× BELOW the declared ρ = 0.10 reference, on 3 of 3 seeds.** ⛔ The
falsifier *"anchor diverges from the frozen features"* **does not fire**, and λ was set
once from the shared warm start by the declared rule `λ = recon₀/ρ²`
(2.0279 / 1.8943 / 1.8861) and ⛔ **never retuned**. Weight displacements: 0.000325 /
0.000247 / 0.000324 against `adapter`'s 0.126956 / 0.104833 / 0.115534.

⚠ **Conversely, `adapter` — the arm whose label is "light" — moves the features by
rel-L2 ≈ 0.33–0.40, i.e. about 40 %.** "Light" describes its **131,072 trainable
parameters (524,288 B, the programme's own reference pin)**, not its effect on the feature
space, and this hub says so rather than letting the word do work the number does not.

---

## 8. ⭐ kill -9 RESUMABILITY — PROVED BEFORE ANY SPEND

`dev/killproof/`, run at 17:18–17:38Z, **before** the ledger row (21:27:53Z).

| part | result | evidence |
|---|---|---|
| checkpoint on a wall-clock timer, then **SIGKILL of the run's OWN recorded PID** | ⭐ PASS | `killproof.log`: victim PID 132774, `ckpt_000000000001.npz`, `SIGKILL sent to 132774`, `victim reaped; exit status 137` |
| relaunch must RESUME | ⭐ PASS | `run_resumed.log`: `[adapter] RESUMED from checkpoint step 1; 59 steps remain`, exit 0, 23 saves |
| resumed == uninterrupted | ⭐ PASS on every TRAINED object | `killproof.json`: `encoder`, `base_encoder`, `store`, `joint_decoder` hashes IDENTICAL; fingerprints equal (`5b616fc355eda4b7`) |
| a DIFFERENT fingerprint must be refused **by name** | ⭐ PASS | `FingerprintMismatch: checkpoint ... was written by config fingerprint 552cea0330a2b8d9, this job is f7659c4708fd127b. Refusing to resume.` |

⛔ **Only PID 132774 was signalled; no box-wide kill at any point in this wave.** Every
PID this hub launched is in `MY_PIDS.txt`; the three K = 32 warm runs retired after
FINDING 1 were killed by their own recorded PIDs (132149/132152/132155).
⚠ **One object did NOT match and it is reported, not hidden:** `render_decoder`
(`0fb21401…` resumed vs `45e12010…` clean). It is refit **after** the training loop, so
the checkpoint does not cover it; everything the checkpoint does cover matched
bit-for-bit, and the displacement figures agree to 16 digits across the two runs
(`0.015152376145124435` in both). An isolating control (two uninterrupted runs under
different box load) is a **DECLARED NOT-RUN** — the box was saturated with the nine
scored lanes and the check was not worth displacing a lane.

---

## 9. ⛔ A REPAIR TO A FROZEN MODULE, DECLARED

`chlu/eval/rivals/w4_learners.py:train` did `Xnp = np.asarray(X0, np.float32)`
unconditionally. The two `tokens` families feed those ids to `eqx.nn.Embedding`, which
jax 0.9.0 refuses to index with a float:

> `TypeError: Indexer must have integer or boolean type, got indexer with type float32`

so **`transformer` and `lstm` could not be FITTED at all**. `predict()` never made the
cast, so inference always preserved the dtype and only training was broken.
⚠ **This is not new to w13:** `w8-encoder-loop/run/seed_*/{alone,loop_swap}/rivals.json`
records `families_done = ["mlp_180k"]` on **all six** of its arm-seeds.
⭐ **Repaired rather than declared not-run — a rival that cannot be fitted is a rival that
cannot lose.** The repair preserves an integer input's dtype and nothing else: ⛔ no
change to the protocol, the grids, the architecture, the schedule or the parameter budget.
Verified after the repair (4096 rows × 2 epochs): transformer / lstm / mlp / cont_mlp all
fit and all return a finite val MSE. All five families then fitted on **all nine**
arm-runs, at identical parameter counts across arms: `mlp` 117,878, `mlp_180k` 178,952,
`cont_mlp` 119,525, `transformer` 119,858, `lstm` 119,125.

⚠ **A fact that runs against this hub and is printed anyway:** the baselines are
**uniformly best on `anchor`'s features**, on **5 of 5 families on 3 of 3 seeds**
(e.g. seed 0 `mlp_180k` val MSE 0.014637 / 0.015666 / 0.015934 for anchor / frozen /
adapter). `anchor`'s margin is therefore measured against a **harder** bar than
`adapter`'s, which makes `anchor`'s 3-of-3 growth the stronger of the two results.

---

## 10. ⛔ DECLARED NOT-RUNS — never a zero, never a blank

| not run | reason, measured where possible |
|---|---|
| **LACE** (and `poe_ebm`) | `w4_learners.builders()` — the protocol every other baseline here is refit under — has no `lace` entry; LACE is an energy model with its own fit routine and **no entry point that refits it on a NEW feature cache**. Writing one would be this hub authoring a fitting protocol for a competitor. w12's scored leg records the same call. |
| **the 1×-matched baseline column** | The available comparators are matched to 118,784 / 179,968 params, not to this object's own ~2.15 MB four-part total. A 1×-matched rival would be **STRONGER**, so the error is AGAINST this hub and every row says so. |
| **seeds 3 and 4** | w12's own measurements: `CACHE-READY.json` gives seed 4 `status: NOT_RUN` / `array_shape: null`; seed 3 passes the cache rule but `fw.classifier_dir(3)` does not exist, so **no corner of it can be judged on this instrument**. |
| **GPU** | None visible on this box (`jax.devices() == [CpuDevice(id=0)]`; no torch installed). Every number here is CPU. |
| **the render-decoder determinism control** | See §8. The box was saturated by the nine lanes; not worth displacing one. |
| **a deletion / refusal leg** | Out of this hub's question. Not attempted, not claimed. |
| **scored runs 2 and 3 on `encoder-adapt`** | **UNSPENT and returned.** One run covered the whole pre-registered comparison. |

---

## 11. WHAT WOULD PROVE THIS HUB WRONG — scored against its own list

| falsifier, declared in advance | fired? |
|---|---|
| "the margin does not grow on ≥ 2 of 3 seeds ⇒ adaptation does not buy a store result" | ⭐ **Did not fire.** `anchor` 3/3, `adapter` 2/3. |
| "every arm improves together ⇒ it is an ENCODER gain, not a store gain" | ⛔⛔ **FIRED, in 6 of 6 treatment cells.** §3. This is the headline. |
| "`anchor` diverges from the frozen features ⇒ the penalty is mis-scaled" | ⭐ **Did not fire.** 47×–80× below the unpenalised arm, 4.2×–5.8× below the declared ρ. §7. |

---

## 12. WHAT THE HUB SHOULD TAKE FORWARD

1. ⭐ **Light adaptation buys a small, seed-unstable margin, and what it actually moves is
   the encoder.** The one-line answer to the target: *the frozen harness is not beaten by
   adapting it; the margin's growth is a residual of a feature-space shift that the
   baselines share.*
2. ⭐ **`anchor` is the better of the two treatments and it is the cheaper claim**: 3 of 3
   seeds, against a measurably harder baseline, at a feature displacement 47×–80× smaller
   than the unpenalised adapter. If one adaptation is carried into the paper, it is this
   one, stated as an encoder effect.
3. ⛔ **The swap loss should not be carried at K\* = 32 without a decision.** It is
   structurally ill-posed there (26 of 32 wells unowned) and empirically null at this
   compute (`swap_exact` ≈ 0 on 3 of 3 seeds). Either the ownership rule changes — a Head
   decision, not a hub one — or the term is retired for fine partitions.
4. ⚠ **The comparator pin is the open question, not the arms.** Wave 12's negative margin
   and this wave's positive one differ by **which learner the store is matched against**,
   not by the object. A 1×-matched baseline is the measurement that would settle it, and
   it is a declared NOT-RUN here.
5. ⭐ **Two runs remain on `encoder-adapt`.** The obvious spend is the 1×-matched
   comparator column on these same nine caches — every artefact needed is on disk.
