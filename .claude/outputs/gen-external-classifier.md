# gen-external-classifier — experiment-engineer report

**Task + acceptance criterion:** build a classifier that reads factor values off a **rendered image**, trained on
**real train-split images only**; gate it at in-corner `shape` balanced accuracy ≥ 0.95 on REAL held-out images;
then report what it says about the frozen decoder's renders versus the linear probe.

**Status: done** — all 8 acceptance items met, 3/3 seeds, except item 6's cluster line which is **NOT RUN by
design** (CSF3 unreachable) and the wide-condition stretch item, **declared NOT RUN with its cause named** (§8).

> **⭐ Downstream reconciliation list: YES, 3 items — §9.** Two need a Hub owner at this review: the `dec_recipe`
> default, and the fact that **`gen-cell-harness`'s gate file is now green and its content check can be run
> mechanically**.

---

# ⭐⭐ THE TWO RESULTS

**1. THE GATE PASSES, 3/3 seeds, at the ceiling.**

| | seed 0 | seed 1 | seed 2 | floor | verdict |
|---|---|---|---|---|---|
| external, in-corner `shape` **balanced accuracy on REAL held-out images** | **1.0000** | **1.0000** | **1.0000** | 0.95 | ⭐ **PASS 3/3** |
| probe companion, same rows, same statistic | 1.0000 | 1.0000 | 1.0000 | — | — |

(TPR 1.0000, TNR 1.0000, AUROC 1.0000 on every seed; n_pos = 72, n_neg = 198/199/199.)
⭐ `"gate": "PASS"` is written **last** to `.claude/outputs/gen-external-classifier/classifier_gate.json`.

**2. ⭐⭐ THE CEILINGS STAND — and the instrument is NOT a second copy of the probe.**

| on the **SAME** renders, in-corner `shape` balanced accuracy | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| **external classifier** | 0.9085 | 0.9472 | 0.9514 | **0.9357 ± 0.0193** |
| **linear probe** (recomputed in-run) | 0.9223 | 0.9365 | 0.9653 | **0.9414 ± 0.0179** |
| **gap Δ = external − probe** | −0.0139 | **+0.0108** | −0.0139 | **−0.0057 ± 0.0116** |

⭐ **`H_AGREE`, pre-registered:** |mean Δ| = 0.0057 is below **both** halves of the materiality rule (0.05 absolute
*and* 2× the seed spread), **and the sign flips across seeds**. ⛔ The phase's ceiling stack is **not** an artefact
of measuring the decoder with its own family. Every ceiling number stands as published.

⭐⭐ **And it is not a second copy.** The task's own falsifier is *"agrees with the probe everywhere, including
where the probe is saturated."* Per **item**, not per level:

| set | agreement on **REAL** images | agreement on **RENDERS** | disagreements go both ways? |
|---|---|---|---|
| A (n=1024×3) | **1.0000** (0 / 3072) | 0.9759 (74 disagree) | yes — ext-only-right 10/23/10, probe-only-right 12/14/5 |
| B (n=72×3) | **1.0000** (0 / 216) | 0.9398 (13 disagree) | yes — 5/3/0 vs 3/0/2 |
| MS (n≈199×3) | **1.0000** (0 / 596) | 0.9547 (27 disagree) | yes — 1/5/0 vs 12/9/0 |

⛔ **0 of 3,884 real-image items disagree; 114 of 3,884 rendered items do (2.94 %).** The two instruments are
*identical* exactly where R15 said the probe is saturated (a known confirmation, ⛔ not news) and *genuinely
different* exactly where independence is testable — on the decoder's renders, in both directions. The falsifier
did **not** fire.

⚠ **The one thing the matching level hides, and it is worth carrying.** On seed 0 the two instruments reach nearly
the same balanced accuracy by **opposite errors**: external TPR 0.9583 / TNR 0.8586, probe TPR 0.9306 / TNR 0.9141.
The external reader over-calls cylinder on rendered matched-seen negatives; the probe under-detects rendered
held-out cylinders. ⛔ A future arm comparison read on TPR alone would move with the choice of instrument even
though the balanced number does not.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *the instrument.* ⛔ No arm, no CLU number, no store number, no kill-rule verdict appears below.
- **Laundering control:** independence of the decoder, **demonstrated** in leg 3 (renders, where the two can
  disagree) and quantified per item, ⛔ never argued from architecture.
- **Falsifies:** the external classifier scoring the decoder's renders materially below the probe ⇒ every phase-1
  ceiling is optimistic. **It did not fire** (Δ = −0.0057 ± 0.0116, sign-inconsistent).
- **Does NOT falsify:** agreement on REAL images. Both read 1.0000 there; that is the **metric-native saturation**
  R15 already established, stated in advance as a known confirmation.

---

## 1. Flag provenance (mandatory, M4)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-external-classifier` — `6baae71`, `0156d7e` (base `main @ 5fee3f4`, unmoved) |
| worktree | `/Users/user/Desktop/CHLU-gen-extclf` (fresh, this spoke only) |
| env | main venv reused, ⛔ **no `uv sync`** — jax **0.9.0**, equinox 0.13.4, optax, numpy 2.4.1, **h5py 3.16.0** (imports; trap 3 discharged), Python 3.11.13, `JAX_PLATFORMS=cpu`, float32 (`jax_enable_x64` off) |
| seeds | **0, 1, 2** (all three complete; ⛔ none NOT RUN) |
| dataset / condition / pretext | `shapes3d` / `recomb2element` (Montero) / `recon` |
| **decoder** | ⭐ `dec_recipe = v1_capacity_schedule` selected **explicitly**, ⛔ never the `"baseline"` default. Param hashes asserted per seed: `01ef91f75eeaae7b…` / `001011bc603a7cb9…` / `2d7eea75add77044…` — **plus the recipe NAME**, both via `assert_dream_decoder` |
| frozen encoder / probe | `98d326028316ae2c…`/`c03bfb8916d00932…`, `fc05e2c5f39694b2…`/`fb16109ded84c908…`, `154d498610648fac…`/`5cdaa90a6b67f1b3…` — all asserted against `bridge_manifest.json` |
| feature cache (trap 2) | seed 0 `scope="full"` **480,000** rows · seeds 1–2 `scope="used"` **9,288** rows; checksum verified; coverage extended through the **frozen encoder** for the 196–199 matched-seen rows no `used` cache holds, with `_assert_cache_reencodes` proving 64 cached rows re-encode **bit-identically** first |
| **classifier subsample (required)** | **n = 8,192** real train-split rows per seed, RNG `numpy.default_rng(seed·104_729 + 20_260_913)` = **20260913 / 20365642 / 20470371**; pool 469,730 / 469,717 / 469,726 |
| classifier | `ExternalFactorCNN`, channels `[32,64,128,128]`, kernel **5**, stride 2, GroupNorm 8, **SiLU**, hidden **256**, 6 linear heads; **978,873** params; Adam lr 1e-3, batch 128, **1,500 steps**; loss 2.185 → **8.3e-05** / 7.7e-05 / 7.7e-05; hashes `c8741ac417d88ebf…` / `aa69b436989b2de5…` / `ad16c4e532731958…` |
| non-default flags vs shipped config | ⛔ **none** — every number above is at `GenClassifierConfig`'s defaults (the block is new, so its defaults *are* the production configuration; the whole block is in §7) |
| wallclock | 669 s / 452 s / 495 s per seed, 27 min total, CPU, with two other spokes live |

---

## 2. Leg 1 — build: real pixels, train split only

**What "external" means here, operationally — and all four separations are asserted, not asserted-in-prose:**

1. **Rows.** The training draw is disjoint from the bridge's own 8,192 fit rows (the rows the decoder **and** the
   probe were fitted on, from the same key split), from `eval_a`, from `eval_b`, and from **the entire Montero
   photometric corner**. Measured overlaps: `0 / 0 / 0 / 0` on every seed.
2. **Renders.** ⛔ Never sees one — and it is *verified*: 64 rows per seed are re-streamed straight out of
   `3dshapes.h5` and required **bit-identical** to the training tensor.
3. **Architecture.** 4 stages / kernel 5 / SiLU / hidden projection, against the probe's 3 stages / kernel 4 /
   ReLU / direct linear heads.
4. **Key chain.** `PRNGKey(seed·104_729 + 20_260_913)`, ⛔ never the bridge's `seed·10_007 + 20_260_910`.

⚠ **Honest limit:** 1 is the only one of the four that is *evidence*; 3 and 4 are hygiene. Independence is only
testable in leg 3, and that is where it was tested.

**Excluded counts, per seed (acceptance item 2):**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **held-out-combination images in training** | **0** (72/72 excluded, 8,192 rows re-checked through the grid bijection) | **0** (72/72) | **0** (72/72) |
| **decoder renders in training** | **0** (64 rows bit-identical to the h5) | **0** (64) | **0** (64) |
| rows excluded from the pool | 10,270 (incl. **288** corner + **936** wide held-out) | 10,283 | 10,274 |
| matched-seen negatives realised | 198 | 199 | 199 |

⭐ **All three assertions are shown ABLE TO FIRE** on deliberately seeded violations, in the test file
(`gen-wide-split` precedent): a bridge row smuggled into the draw raises `LineageLeak`; a held-out row raises
`HoldoutViolation`; **one perturbed pixel of one real image** raises `RenderLeak` from the bit-identity check.

---

## 3. ⭐ Leg 2 — the gate

**Statistic (R20/R23, verbatim):** positives = the **72 REAL** held-out-combination images (all `shape = cylinder`);
negatives = the **REAL** matched-seen non-cylinder images in the same corner. Argmax decision, chance 0.5.

```
⭐ GATE seed 0: external in-corner shape balacc on REAL = 1.0000 (floor 0.95) -> PASS | probe 1.0000 | n_pos=72 n_neg=198
⭐ GATE seed 1: external in-corner shape balacc on REAL = 1.0000 (floor 0.95) -> PASS | probe 1.0000 | n_pos=72 n_neg=199
⭐ GATE seed 2: external in-corner shape balacc on REAL = 1.0000 (floor 0.95) -> PASS | probe 1.0000 | n_pos=72 n_neg=199
⭐ gate = PASS (3/3 seeds, min 1.0000, floor 0.95)
```

⛔ The rule is **every seed**, not the mean — a 2/3 pass whose mean clears 0.95 is a FAIL, and that is a test, not a
convention (`test_pass_requires_every_seed_not_the_mean`). Aggregate per-factor on REAL images is **1.0000** for
both instruments on A, B and MS: ⭐ the external reader is **saturated on reals too**, so like the probe it can only
ever *under*-report decoder damage, never over-report it — the safe direction for a ceiling.

**Prediction P1 (pre-registered: PASS, per-seed in [0.97, 1.00], point 0.995) — ✅ held**, at the top of the range.

---

## 4. ⭐⭐ Leg 3 — the comparison that could have killed the ceilings

Both instruments read **one** rendered array per set, produced once by the frozen `v1_capacity_schedule` decoder
from the frozen cached features. ⛔ Nothing was re-rendered between the two reads.

### 4.1 The primary row (in-corner `shape` balanced accuracy on renders)

| | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| external | 0.9085 | 0.9472 | 0.9514 | 0.9357 ± 0.0193 |
| probe (recomputed here) | 0.9223 | 0.9365 | 0.9653 | 0.9414 ± 0.0179 |
| **R22's declared reference** | 0.9223 | 0.9365 | 0.9653 | **0.9414 ± 0.0179** |
| **Δ** | −0.0139 | +0.0108 | −0.0139 | **−0.0057 ± 0.0116** |
| external AUROC / probe AUROC | 0.9611 / 0.9720 | — | — | (companion, ⛔ never the gate) |

⭐⭐ **The probe's in-run recomputation reproduces `gen-r0-rescore`'s number BIT-IDENTICALLY on all three seeds —
`max |Δ| = 0.0` against `rescore_seed_{0,1,2}.json`.** So the gap is a **within-run** difference between two
readers of the same pixels, ⛔ not a cross-report comparison, and the render pipeline here is provably the same
pipeline R22 measured.

**Verdict under the pre-registered rule:** `|mean Δ| = 0.0057 < 0.05` **and** `< 2 × 0.0116` ⇒ **not material** ⇒
**`H_AGREE`**. ⭐ The ceilings stand and the instrument has earned its place.

### 4.2 Companions on the A-set renders (n = 1024 per seed)

| | external | probe | Δ |
|---|---|---|---|
| aggregate (6 factors) | 0.9912 ± 0.0029 | 0.9902 ± 0.0030 | **+0.0010** |
| `shape` top-1 | 0.9772 ± 0.0093 | 0.9733 ± 0.0090 | **+0.0039** |

⚠ **Prediction P3 was FALSIFIED, in the useful direction.** I pre-registered external ≈ 0.96 ± 0.02 aggregate on A
renders, i.e. a *small negative* gap, derived from the probe's fit images being the decoder's reconstruction
targets. The external classifier is in fact **marginally better** than the probe on A renders (+0.0010 aggregate,
+0.0039 `shape`) while being marginally worse in the corner. ⭐ **The lineage-coupling mechanism I predicted is not
detectable at this precision** — which is itself the cleanest available evidence that the probe is not flattering
the decoder.

**Prediction P2 (mean Δ = −0.04, H_AGREE "but only just") — half right.** The *sign call* on the pooled mean was
right (−0.0057) and the *hypothesis* was right (H_AGREE), but the magnitude was **7× too large** and I did not
predict that the sign would flip across seeds. ⛔ Do not quote "the external reader is slightly worse": on 1 of 3
seeds it is better, and the pooled difference is a quarter of the seed spread.

**Prediction P4 (|Δ| ≥ 0.01 on renders while both ≥ 0.98 on reals) — ✅ held** on every seed (|Δ| = 0.0139 /
0.0108 / 0.0139; both instruments 1.0000 on reals), and the per-item analysis (§ THE TWO RESULTS) makes it much
stronger than the |Δ| test alone.

---

## 5. How I verified — commands and observed output

**Environment (protocol §4):** ⛔ no `uv sync` in the worktree; main venv reused —
`PYTHONPATH=/Users/user/Desktop/CHLU-gen-extclf CHLU_GEN_DATA_ROOT=…/.claude/data/gen
CHLU_GEN_BRIDGE_ROOT=…/.claude/outputs/gen-feature-bridge/runs CHLU_CLAUDE_ROOT=…/.claude JAX_PLATFORMS=cpu
/Users/user/Desktop/CHLU/.venv/bin/python …` ⇒ **jax stays at 0.9.0** (w6 lesson).

**h5py (trap 3):** verified it imports — `h5py 3.16.0` — so the 8 real-data tests are live, not silently skipped.

```
# smoke (⚠ plumbing only, ⛔ never a number): 512 rows / 20 steps
python -u -m chlu.experiments.exp_gen_classifier --out …/scratch/…/smoke --stage all --quick
  ⭐ GATE seed 0: external … = 0.7014 (floor 0.95) -> ⛔ FAIL   # the FAIL path exercised on purpose
  ⛔ seed 0: gate FAILED — leg 3 NOT RUN, returning to the Hub

# production, 3 seeds
python -u -m chlu.experiments.exp_gen_classifier --out …/outputs/gen-external-classifier --stage all
  rows: clf_train=8192 (pool 469730, 10270 excluded incl. 288 corner + 936 wide held-out) · eval_a=1024 eval_b=72 matched_seen=198
  ⭐ cache re-encode check: 64 rows bit-identical through the frozen encoder
  ⭐ decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b… (hash + recipe asserted)
  ⭐ no-leak: 8192 training rows checked, 72/72 held-out-combination images excluded, 0 leaked;
     0 decoder renders in training (64 rows bit-identical to the h5);
     lineage overlap with bridge_train/eval_a/corner = 0/0/0
  ⭐ GATE seed 0 … = 1.0000 -> PASS | probe 1.0000
  ⭐⭐ LEG 3 seed 0: on the SAME renders — external 0.9085 vs probe 0.9223 (gap -0.0139); R22 reference 0.9223
  ⭐ gate = PASS (3/3 seeds, min 1.0000, floor 0.95)
```

**Bit-identity of the probe against R22 (the provenance check that makes leg 3 within-run):**
```
R22 on disk    : [0.9223484848484849, 0.9364530988274706, 0.9652777777777778]
recomputed here: [0.9223484848484849, 0.9364530988274706, 0.9652777777777778]
max |delta|    : 0.0
```

**Targeted tests** (⛔ no full suite — the Hub runs one at wave close):
```
pytest tests/test_gen_classifier.py -q --no-cov            -> 26 passed in 16.03s
pytest tests/test_gen_bridge_wide.py tests/test_gen_compose.py tests/test_config.py -q --no-cov
                                                           -> 63 passed in 23.71s   (config-block regression)
ruff check chlu/eval/gen_classifier.py chlu/experiments/exp_gen_classifier.py chlu/config.py tests/… -> clean
```
⚠ One genuine failure found and fixed on the way: my render-leak test passed the **gen data root** where
`shapes3d_root()` expects the **dataset** dir (`…/gen` vs `…/gen/shapes3d`), so it raised `FileNotFoundError`
instead of exercising the assertion. Fixed in the test by resolving the path the way the experiment does.
⛔ `ruff format` would reformat my two new files — and also `chlu/core/gen_bridge.py` and
`chlu/experiments/exp_gen_compose.py`, i.e. **the repo is not `ruff format`-clean**, so I left formatting alone
rather than introduce a diff the rest of the tree does not carry.

---

## 6. Artefacts

`.claude/outputs/gen-external-classifier/`
- ⭐ **`classifier_gate.json`** — `"gate": "PASS"`, **written last**; carries the per-seed values, the rule, the
  decoder recipe + per-seed hashes, and the no-leak counts. **This is the file `gen-cell-harness` gates on.**
- `PREREG.md` — mtime **17:44:06**; `chlu/eval/gen_classifier.py` **17:44:43**,
  `chlu/experiments/exp_gen_classifier.py` **17:45:51**, first smoke run **17:46**, first production run
  **17:47:30**. ⛔ Verify by mtime, not by claim. ⚠ The margin is **37 seconds**, not hours — the prereg was
  written before the harness but in the same session, so read it as *"committed before the first measurement"*,
  ⛔ not as an overnight gap.
- `summary.json` · `classifier_gate.png` (2 panels: the gate on reals, the two instruments on renders) ·
  `instrument_agreement.json` (per-item agreement) · `run.log`
- `seed_{0,1,2}/classifier_manifest.json` + `seed_{0,1,2}/external_classifier.pkl` (3.9 MB each)

Scratch: `.claude/scratch/gen-external-classifier/{smoke/,agreement.py}`.

⭐ **Incrementality (acceptance item 7):** every seed writes its own manifest and its own checkpoint; the training
loop checkpoints every 250 steps; a seed whose manifest exists is **skipped** on re-invocation; the aggregate
reports any absent seed as `seeds_not_run`, ⛔ never as a zero. Nothing was held to the end.

---

## 7. The config block (acceptance: one block, appended at EOF)

`GenClassifierConfig` appended at the **end** of `chlu/config.py` behind `gen-read-compose`, plus the three house
wiring sites (`CHLUConfig` field, `load_config`, `save_config`) with `BEGIN/END gen-external-classifier` markers.
⭐ **`git diff --numstat` for `chlu/config.py`: `132 insertions, 0 deletions`** — ⛔ nothing existing was modified,
and the peer-conflict rule (take the peer's whole file, re-apply, assert 0 deletions) has nothing to undo.
No conflict with `gen-arms-a7-wiring` (it edits `GenBaselinesConfig`, a different block) or with
`gen-commitment-collapse` (same tail anchor; both are pure appends).

⛔ **I did not add a CLI hook.** `chlu/cli/experiment_cmd.py` is not in my task's file-ownership list and is a
high-collision shared file this wave; the module ships its own argparse `main()` like `exp_gen_bridge`/
`exp_gen_compose`, and is runnable as `python -m chlu.experiments.exp_gen_classifier`. **The one-line CLI hook is
a Hub decision, noted in §9.**

---

## 8. ⛔ Declared NOT RUN

**(a) The CSF3 re-fit — NOT RUN.** The cluster is unreachable (2FA is Head-only). No compute was committed.
⚠ **The launcher cannot run this stage as it stands**, and ⛔ `scripts/csf3/` is outside my ownership, so I did not
edit it. Both forms of the submit line, honestly labelled:

```bash
# (i) launcher-native — REQUIRES a 3-line edit to scripts/csf3/job_gpu_gen.sh that I did NOT make:
#     add `classifier` to the STAGE case list (line ~110), set UPSTREAM="decoder" (line ~388),
#     and map MODULE=chlu.experiments.exp_gen_classifier. The module also ignores --arm.
export GEN_GIT_COMMIT=$(git rev-parse HEAD)          # 0156d7e on this branch
cd ~/scratch/CHLU && mkdir -p logs
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=classifier,ARMS=clu,SEEDS=0 1 2,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" \
       -a 0-2%2 -t 02:00:00 scripts/csf3/job_gpu_gen.sh

# (ii) no launcher edit needed — works today:
sbatch -p gpuA -G 1 -n 1 -c 8 -t 02:00:00 -J clu-extclf -o logs/%x-%j.out -e logs/%x-%j.err --wrap \
 "source scripts/csf3/env.sh && python -u -m chlu.experiments.exp_gen_classifier \
    --out \$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/classifier --stage all"
```
**Status: NOT RUN.** ⛔ A cluster re-fit would change nothing about the verdict — the gate is at 1.0000 and the
laptop budget (`n_clf_train = 8192`, 1,500 steps) already saturates it. Its value is a **larger** subsample
(`--n-train 65536`), which is the only lever that could make the external reader strong enough to separate from
the probe on renders. That is the follow-up worth the GPU, not a reproduction.

**(b) The wide-condition (11-cell, `0.9894 ± 0.0021`) comparison — NOT RUN, and ⛔ not merely for budget.**
I pre-registered it as a stretch item. It must **not** be run on these three classifiers, and here is the exact
reason, which is more useful than the number would have been: the training draw excludes the wide condition's
**held-out cells** (936 rows, confirmed in every manifest) but **not** each wide cell's **matched-seen negatives**
— the seen, same-corner, other-shape rows that a wide in-corner read uses as its negative class. Those are only
excluded for *Montero's* corner (288 rows). Scoring the wide cells with this classifier would therefore read a
**leak-contaminated TNR**, and it would look entirely normal. **The fix is one extra exclusion set and a refit**
(~30 min/seed on this machine): union the 13 cells' negative pools into `classifier_rows`'s `extra_exclude`.
⛔ Do not run the wide comparison against `seed_*/external_classifier.pkl` as shipped.

---

## 9. ⭐ Downstream reconciliation list (needs a Hub owner)

1. ⭐ **`gen-cell-harness`'s gate is now mechanically satisfiable.** `classifier_gate.json` exists with
   `"gate": "PASS"` at the pinned path. R30's sequencing also requires `refusal_scores.json` = `complete` and the
   R27.3 whitening content check — ⛔ neither is mine and neither is checked here. **Owner: Hub.**
2. ⛔ **`GenBridgeConfig.dec_recipe` still defaults to `"baseline"`** — the decoder that FAILED the gate. This is
   now the **third** module to defend against it in its own config (`gen_dream`, `gen_compose`, `gen_classifier`).
   The one-line flip is written out in `gen-decoder-lift` §9.1 and is still unowned. **Owner: Hub/Head.**
3. **A CLI hook** (`chlu exp-gen-classifier`) is the only piece of the house style I deliberately skipped, because
   `chlu/cli/experiment_cmd.py` is shared with two live spokes and is not in my ownership list. ~35 lines,
   copy-shaped from `cmd_exp_gen_compose`. **Owner: Hub, at the merge.**

---

## 10. Git footprint

- **Branch:** `agent/experiment-engineer/gen-external-classifier`, off **`main @ 5fee3f4`** (base unmoved; `git
  rebase main` was a no-op — ⛔ I did **not** rebase onto the stale `origin/main`, §7.21).
- **Commits:** `6baae71` (the instrument + experiment + tests + config block) · `0156d7e` (pin R22's reference at
  full precision).
- **Verified from the MAIN repo** (protocol §3.2): `git log --oneline main..agent/experiment-engineer/gen-external-classifier`
  lists both hashes; `git diff --stat` = **4 files, 1,637 insertions, 0 deletions**.
  ⛔ The worktree `/Users/user/Desktop/CHLU-gen-extclf` is **left in place** for Hub review.
- **Files:** `chlu/eval/gen_classifier.py` (new, 386) · `chlu/experiments/exp_gen_classifier.py` (new, 744) ·
  `tests/test_gen_classifier.py` (new, 375) · `chlu/config.py` (**+132 / −0**, one block at EOF + 3 wiring sites).
- ⛔ Not pushed, no PR, nothing merged. ⛔ No `.pkl`, no figure, no dataset entered git.
- **No conflicts.** ⛔ I never touched the shared main checkout (it holds another agent's branch,
  `pilot-ttt-nan-and-d5-wiring` @ `7fcef50`, which does not even contain the `gen_*` modules).

---

## 11. Open questions / follow-ups / risks

1. ⚠ **Both instruments are saturated on real images, so neither can resolve anything above its own ceiling.**
   Every render comparison in this phase is therefore measured with two readers that are perfect on reals and
   lossy only through the decoder. Good for a gate; it means a *future* ceiling claim cannot be sharpened by a
   better reader — only by a better decoder.
2. ⚠ **The external reader is not obviously "stronger" than the probe.** It is +0.001 on A renders and −0.006
   in the corner. ⛔ Do not present it as an upgrade; present it as an **independent second reader** whose value is
   that it was fitted on rows the decoder never saw. If the Hub wants a *stronger* external reader (the honest way
   to make the ceiling claim more adversarial) the lever is `--n-train 65536` on a GPU — §8(a).
3. ⭐ **The TPR/TNR asymmetry (§ THE TWO RESULTS) is a live risk for the headline table.** Any arm metric read on
   detection rather than balanced accuracy will move with the choice of instrument. ⛔ Keep balanced accuracy as
   the gate statistic, exactly as R23.1 rules.
4. ⚠ `seed_*/classifier_manifest.json` carries `probe_reproduces_reference: true` for seed 0 and `false` for
   seeds 1–2 — that flag was computed against the **4-dp report figures** I first transcribed, not the disk
   values. The disk comparison is **exact on all three seeds** (§5). Commit `0156d7e` fixes the constants; ⛔ I did
   not re-run three seeds to refresh a derived boolean. **Read §5's check, not that flag.**
5. **dSprites is not wired here** — the matched-seen corner is Shapes3D-specific by construction
   (`matched_seen_rows` refuses any other dataset), so the instrument is Shapes3D-only until someone defines the
   corner elsewhere. ⛔ Do not invent one.

---

## Proposed handover updates (for the Hub)

- **§7 new item — `dec_recipe` default.** Third module in a row to carry a private guard against
  `GenBridgeConfig.dec_recipe = "baseline"`. It is now a standing trap with three defences and no fix; either flip
  the default (one line + one test, `gen-decoder-lift` §9.1) or record it in §7 as permanent.
- **§7 new item — the repo is not `ruff format`-clean.** `chlu/core/gen_bridge.py` and
  `chlu/experiments/exp_gen_compose.py` (among others) would be reformatted by `ruff format`. The working bar is
  `ruff check`. Worth stating so no future spoke "tidies" a shared file and creates a 500-line diff.
- **§7 new item — worktree path resolution.** `shapes3d_root(root)` treats an explicit `root` as the **dataset**
  directory while `$CHLU_GEN_DATA_ROOT` is the **parent**. That asymmetry cost me one red test and will cost the
  next spoke one too.
- **Result to record:** Shapes3D `recon`, Montero condition, seeds {0,1,2}, `v1_capacity_schedule` decoder —
  external classifier gate **PASS 3/3 at 1.0000** (floor 0.95); on the same renders **external 0.9357 ± 0.0193 vs
  probe 0.9414 ± 0.0179, Δ = −0.0057 ± 0.0116, NOT material, `H_AGREE`** ⇒ ⭐ **the phase's ceiling stack stands**;
  per-item agreement **1.0000 on reals / 0.9706 on renders** ⇒ ⭐ the instrument is independent where independence
  is testable and ⛔ is not a second copy of the probe.
- **Carry as a NOT RUN with a cause, not as a gap:** the wide 11-cell comparison needs its own row-selection pass
  (§8b) — running it on the shipped classifiers would read a leak-contaminated TNR.
