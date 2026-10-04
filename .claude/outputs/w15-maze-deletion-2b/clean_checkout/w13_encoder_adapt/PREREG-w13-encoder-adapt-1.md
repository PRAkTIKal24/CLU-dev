# PREREG — w13 hub/encoder-adapt, run 1/3 on target `encoder-adapt`

**Written BEFORE any scored number exists.** Timestamp of this file: see
`PREREG-w13-encoder-adapt-1.timestamp.txt` beside it.
Branch `w13/encoder-adapt`. Output `/c/chlu/outputs/w13-encoder-adapt/`.
`chlu.__file__` is asserted inside `C:\chlu\worktrees\CHLU-w13-encoder-adapt` in code
(`exp_w13_encoder_adapt.assert_chlu_in_worktree`) before any number, and is written into
every artefact.

⛔ **Budget.** Target `encoder-adapt`, **0/3 spent** at the time of writing
(`/c/chlu/SCORED-LEDGER.md`, Head ruling row 2026-09-21). ⛔ It does **NOT** draw on
`encoder-loop`. The plan's heading "target encoder-loop, 2 left" is STALE; the ledger is
authoritative.

---

## 1. THE ONE QUESTION

**The frozen harness is the primary object. Light adaptation is optional — what does it
buy?**

## 2. THE ARMS — equal steps, equal images, one bit-identical warm start per seed

| arm | encoder | role |
|---|---|---|
| `frozen` | frozen | ⭐ **REFERENCE** |
| `adapter` | rank-32 low-rank residual adapter on the penultimate feature; **encoder weights frozen** | treatment |
| `anchor` | full unfreeze **+ penalty on distance from the frozen weights AND features** | treatment |

**The objective is IDENTICAL in all three arms** and is w8's, plus the anchor term in
`anchor` only:

```
L = recon + 1.0 * codebook + 0.25 * commitment + beta_swap * swap   (+ lam * anchor)
```

* `commitment = || sg(gather) − z ||²` **is** the store read error on the fitting rows,
  from the encoder's side (`chlu/core/gen_bridge.py:joint_loss_terms`).
* `recon` is kept exactly as in w8.
* The **swap loss** is in because w8 measured it repairing what unfreezing breaks —
  positive on all 3 GPU seeds (+0.0100 / +0.0129 / +0.0075) at 10× steps and 4× data.
* The arms differ **only** in which encoder leaves Optax is given, and (anchor only) the
  extra penalty. `frozen` carries the swap term too, so the step-cost profile, the batch
  sequence and the PRNG stream are identical across arms.

**`beta_swap`** is `"auto"` — `recon_0 / swap_0`, both measured **once** on the SHARED
warm start before the first update, so the two terms enter at equal magnitude. Verbatim
w8's rule.

**`lam` (anchor)** is `recon_0 / rho²` with a **REFERENCE DISPLACEMENT `rho = 0.10`**:
a 10 % relative displacement of the features (or of the weights) costs exactly the warm
start's reconstruction loss. Measured once on the shared warm start, deterministic,
written into the manifest. ⛔ **It is never selected against any number, and it is never
retuned to make the distance look better.**

**The adapter's size is PINNED, not tuned:** rank 32 over D = 2048 →
`2 × 2048 × 32 = 131,072` float32 = **524,288 B**, the programme's own reference pin.
Its up-projection is initialised to **exactly zero**, so the adapted encoder is
**bit-identical to the frozen one at step 0** — asserted on real images in
`stage_train`, which refuses the run if it is not.

**Equal steps AND equal images are ASSERTED and PRINTED** on every arm manifest
(`compute.gradient_steps`, `compute.images_seen`). If `anchor` needs more wall clock, the
seconds are printed beside the steps and the report says which decides.

## 3. THE STORE — FIXED AT K\* = 32, selected by wave 12, never re-selected here

`C:/chlu/evidence/outputs/w12-single-design-crossover/CROSSOVER-val.json` →
`K_star = 32`, `A_requested = 63`, well width 64, `rank_over_width` 0.2656 … **0.9844**,
four-part ledger 2,143,138 B = 4.0877× the 524,288 B reference. The selection rule was
frozen and timestamped at 2026-09-21T11:10:24Z before any val composition number existed.

Two objects carry that K and they are different objects:
* the **in-loop VQ store** the encoder trains through — K = 32 wells, A = 63 modes;
* the **scored single design** — `CovContentStore` at K\* = 32, rebuilt from **each
  arm's own** generation-2 cache by wave 12's own unmodified
  `scripts/w12_crossover_lib.py`.

## 4. ⭐⭐ THE BAR — it is the MARGIN, and it starts NEGATIVE

Every learned baseline is **refit on each arm's own features** by
`chlu/eval/rivals/w4_learners.fit_family`, protocol verbatim, on each arm's own
generation-2 cache.

```
margin(arm) = mean over corners of [ distinct_valid_novel_tuples(single design)
                                   − distinct_valid_novel_tuples(best learned baseline) ]
```

paired by corner; the best learned baseline is chosen **per arm** as the strongest of
that arm's own refits, so the bar is always the strongest available competitor.

> **BAR: `margin` must GROW from `frozen` to an adapted arm, on ≥ 2 of 3 seeds.**

⚠ **Wave 12 measured this margin NEGATIVE.** Derived from
`w12-composition-ladder/FINDINGS.md` §0 (store 30.5389 / 59.7500 / 62.7083 / 64.7194 /
64.5556 at k = 2…6, `learned_1x` 33.3222 / 55.5083 / 67.9917 / 68.6194 / 68.6167,
ceiling-normalised store 0.8483 / 0.8299 / 0.8709 / 0.8989 / 0.8966): the margin as a
fraction of ceiling is **−0.0773 (k=2) / −0.0734 (k=4) / −0.0542 (k=5) / −0.0564 (k=6)**
— the task file's "−0.054 to −0.077 of ceiling at n = 2, 4, 5, 6", reproduced
arithmetically here before any w13 number existed.

⭐ So the question is **"does adaptation close a deficit?"**, ⛔ not "does it extend a
lead". **A margin that grows and stays negative is a real and reportable result and will
be reported in exactly those words.**

⛔⛔ **A feature gain shared by ALL arms is a finding about the ENCODER, not about the
store, and will be reported as such.** The arm-level composition LEVEL is printed beside
the margin for exactly this reason.

## 5. THE FACTOR PROBE — a MARGIN over random-init, per seed, never pooled

`accuracy(arm) − accuracy(random_init)`, where `random_init` is an untrained encoder of
the identical architecture carried through the identical pipeline
(`exp_w8_encoder_loop.stage_probe`, unmodified). ⛔ **No absolute probe number is a
claim**: w8 measured a random-init encoder probing Shapes3D at 0.962–0.972 aggregate and
`alone`'s margin went NEGATIVE on 2 of 3 seeds; w9's harness independently found the same
instability. ⭐ **Per-seed, never pooled.**

## 6. CONTROLS — run on every row, no row claimed unless all pass

Wave 12's own four charter-C controls in wave 12's own definitions:
`landscape_removed` (zero), `landscape_shuffled` (value permute),
`landscape_shuffled_across_wells` (well permute), `random_atoms` (random span).

## 7. SEEDS

**3 seeds — 0, 1, 2 — ALL OR NONE.** ⛔ The best is never reported alone.
⛔ Never "9/9 seeds": the seed count and the configuration count are reported separately
(3 seeds × 3 arms = 9 arm-runs; that is **3 seeds**, not 9).
Seeds 3 and 4 are DECLARED NOT-RUN with the reason wave 12 measured and recorded:
`CACHE-READY.json` gives seed 4 `status: NOT_RUN` / `array_shape: null`, and seed 3 has no
`fw.classifier_dir(3)` — no corner of it can be judged on this instrument.

## 8. WHAT WOULD PROVE THIS HUB WRONG

* **The margin does not grow on ≥ 2 of 3 seeds** ⇒ adaptation does not buy a store
  result. Reported; that is the secondary paragraph either way.
* **Every arm improves together** ⇒ it is an ENCODER gain, not a store gain — and the
  report says which.
* **`anchor` diverges from the frozen features** ⇒ the penalty is mis-scaled. The
  distance is reported (`compute.displacement_final`, both feature and weight, relative
  L2 squared) and ⛔ **is not tuned until it looks better.**

## 9. DECLARED NOT-RUNS (never a zero, never a blank)

* **LACE** — `w4_learners.builders()`, the protocol every other baseline here is refit
  under, has no `lace` entry; LACE is an energy model with its own fit routine and no
  entry point that refits it on a NEW feature cache. Refitting it per arm × seed would be
  a new fitting protocol written by this hub for a competitor. w12's scored leg records
  the same call: `lace`/`poe_ebm` DECLARED NOT-RUN by spoke A.
* **GPU** — none is visible on this box (`jax.devices() == [CpuDevice(id=0)]`, no torch
  installed). Every number is CPU.
* Any further not-run is recorded in `LOG.md` with its reason at the moment it is taken.

## 10. RESUMABILITY — PROVED BEFORE ANY SPEND

`scripts/w13_killproof.sh`: a run is launched with a wall-clock checkpoint timer,
**SIGKILLed by its own recorded PID** (⛔ never a box-wide kill), relaunched, must print
`RESUMED from checkpoint step N`, and its trained parameter hashes are compared against an
uninterrupted run of the identical configuration. A resume offered a **different
fingerprint** must be refused by name. Artefacts in `killproof/`.
