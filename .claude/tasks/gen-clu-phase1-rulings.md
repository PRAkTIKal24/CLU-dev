# gen-clu-phase1 — BINDING RULINGS (Head + Advisor). Every phase-1 task file references this.

⛔ This is a ledger of decisions already taken, not a brief. Spokes read the rows their task file points them at.
Amendments are appended with a date; nothing here is edited in place.

---

## R1 — Encoder pretext (Head, 2026-09-09)
**Reconstruction is the primary pretext, mandatory.** The positioning sentence against LACE is *"their energy is derived
from label supervision; our store is learned from data."* A factor-classification pretext collapses that sentence — the
factor structure would enter through labels one stage upstream, and the semantic-organisation claim reads as inherited
from supervision. It also matches the bar: the Montero negative result was established on reconstruction-trained models.
- **`cls` pretext is kept as a cheap diagnostic arm, never a headline row.** Its only job is fault isolation: if `recon`
  fails a kill rule and `cls` passes, the failure is in the features, not the store/addressing. If both fail, it is ours.
- ⭐ **CLARIFICATION — do not over-apply this.** The **evaluation probe classifier still trains on factor labels.**
  That is measurement, not the generative path, and it stays. "No label supervision" governs the generative path only.

## R2 — The C set is defined in factor space (Head, 2026-09-09)
Structurally invalid requests, generated from the factor grid alone, **fixed before grouping runs, identical for every
arm**: (a) two different values of the same factor requested simultaneously; (b) a factor value outside its range.
⛔ It must not reference wells, modes or clusters — two of the four arms never have them, and an arm that defines its
own C set is scoring its own exam. Built by `gen-data-splits` as `.claude/data/gen/<dataset>/c_invalid.json`.
- **Addendum (Head, 2026-09-09):** keep a **well-space version as a reported mapping, not a query set.** After grouping
  runs, show how the factor-space C queries land on wells. Diagnostic for grouping quality; costs nothing.
  **Owner: the grouping spoke.** ⛔ It never becomes the query set.

## R3 — The decoder ceiling is a GATE, not just a control (Head, 2026-09-09)
One frozen decoder is shared by every arm forever. It may simply be unable to render a combination it never saw, in
which case B is low for every arm, store-removed matches CLU, and kill rule (i) fires on the decoder rather than the
store. So the ceiling is measured first, by feeding **real cached held-out features** through the frozen decoder and
probing the result.
1. **Pre-registered floor.** If `ceiling_B < 0.8 x ceiling_A` (same probe, same protocol, A-split equivalent), the
   **phase stops before any arm runs** and returns to Head + Advisor. ⛔ Do not run arms against a wall.
2. ⭐ **Kill rule (i) is read RELATIVE to the ceiling.** store-removed vs CLU are compared on B accuracy **as a fraction
   of `ceiling_B`**, not in absolute terms. This amends the original kill rule as filed.
3. **No leak.** The decoder never trains on held-out-combination images; they are used at evaluation only.
   **Assert it and report the count.** Later "just fix the decoder" pressure will point at exactly this leak.

## R4 — LACE (Advisor read, 2026-09-09)
LACE = attribute classifiers on frozen-StyleGAN latents; energy = classifier logits + Gaussian prior; ODE sampler;
AND/OR/NOT compose classifier energies; needs full attribute labels. **Confirmed absent: refusal, invalid-combination
detection, deletion, and any persistent or enumerable concept store.** Both our claims stand.
- ⛔ **Composition alone is table stakes, not a contribution.** Zero-shot generation of unseen attribute combinations on
  a frozen generator was done well in 2021. The paper must never present capability 1 as novel on its own.
- LACE is the tier-1 refusal baseline: thresholding its energy is the canonical post-hoc comparison, replacing the
  vague "EBM with threshold" row with a named, published method.
- ⛔ **0.94–0.98 is NOT a bar we inherit** — StyleGAN at 1024², full attribute labels, their splits. Only our own
  re-implementation on our features and our splits is comparable.
- **Supervision asymmetry is declared, not hidden:** LACE gets full factor labels over the train split; our store does
  not. This makes LACE the stronger baseline, which is the direction we want.
- Method-section difference to state: their energy is derived from label supervision; ours is a learned store written
  from data — which is why deletion, provenance and a dynamics-derived barrier exist for us and cannot exist for LACE.

## R5 — Compute and layout (Head, 2026-09-09)
Smoke locally on CPU; launch on CSF3 in line with existing scripts, but **one** new launch script
(`scripts/csf3/job_gpu_gen.sh`) and **fresh folders**:
`$SCRATCH/CHLU-gen/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/`. ⛔ Nothing writes into a C2/C3 path.
Worktree cap **4 active spokes**; the cold worktrees `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are
**parked, not free** — new spokes create their own.

## R6 — Cell scope (Hub, 2026-09-09)
dSprites gets a loader in phase 1; **the A/B/C cell is Shapes3D only.** Base for every phase-1 spoke: `main @ b0257bc`.

## OPEN — not yet ruled
- **D4:** does a kill rule bind our certificate against LACE-threshold on C? Needed before the prereg is filed.
  Bears on R4: since composition is table stakes, all three filed kill rules currently police B (the gate) and none
  police C (the headline).

---

## R7 — The kill-rule set, consolidated as of 2026-09-09 (Head)
⭐ **Governing principle: phase-1 kill rules test whether the mechanism EXISTS, not whether it is WINNING.** Losing a
first-cell margin to a baseline is iteration signal. Kill rules are reserved for *unattributable or meaningless* —
store unused, retrieval in disguise, certificate at chance. ⛔ Binding a competitive margin into phase 1 would kill a
live mechanism for being untuned.

| # | Fails if | Note |
|---|---|---|
| i | store-removed matches CLU on B within seed spread | ⚠ **AMENDED (R3.2):** compared as a fraction of `ceiling_B`, not in absolute terms. Composition isn't the store's. |
| ii | CLU's B accuracy is not separable from the kNN arm | mechanism is retrieval |
| iii | certificate AUROC on C ≤ 0.5 + spread, thresholds from valid data only | certificate is at chance |
| iv | **NEW** — certificate AUROC on C-vs-(A∪B) ≤ the **kNN-threshold** arm's, beyond seed spread | see below |

**Why (iv) is kNN and not LACE.** kNN's distance threshold is the **same supervision class as us**: no labels, no
dynamics, no invalid examples. If a derived barrier criterion cannot beat plain distance-to-store, the derivation adds
nothing over naive geometry — a mechanism failure, which is what kill rules are for.
⭐ **The paired metric does the work:** a naive distance threshold should score *poorly* because it refuses B as well as
C. If it still beats us on C-vs-(A∪B), we deserve the kill.

**LACE-threshold is NOT a kill rule.** Its refusal energy comes from a factor-labelled classifier; our certificate uses
zero labels and zero invalid examples. **Losing to a label-supervised detector in cell one is survivable and expected.**
⛔ But it is **not** ignored: the tier-1 paper claim already commits us to beating post-hoc thresholds by submission, so
**the LACE margin is pre-registered and reported from phase 1 onward** — reported, never phase-killing.

⛔ A failed kill rule returns to Head + Advisor. It does **not** return to more tuning.

## R8 — Request embedding: declared once, used by every arm (Head, 2026-09-09)
Refusal baselines need a way to embed a **request**, because impossible combinations have no images. Declare the
mapping once and apply it identically across arms; ⛔ never let an arm choose its own.
- **kNN arm:** composed **per-factor prototype features**, computed from the cached train features.
- **LACE arm:** composed **classifier energies**.
- **CLU arm:** its own addressing head — that is the mechanism under test, not a baseline convenience.
**Ownership:** the baselines spoke builds and owns the kNN and LACE request embeddings and the per-factor prototypes;
the grouping spoke owns CLU addressing. ⛔ Neither reimplements the other's.
**The request SETS (A, B, C) are single-sourced from `gen-data-splits`** and are byte-identical across arms.

---

## R9 — LACE arm build rulings (Hub, from `gen-lace-spec`, 2026-09-09)
Official code pinned: `NVlabs/LACE@dcfd4a57bee40fcb17ac8304413f1e35a64323cd` (5 commits total; HEAD *is* the paper
release). NVIDIA **non-commercial** licence — note before any code is vendored.
1. **Whitening is mandatory, not a nicety.** LACE's `½‖z‖²` is the *exact* log-density of StyleGAN's z-space, and that
   exactness is what licenses its ODE drift to omit the prior score. Fit a shrinkage Gaussian over cached train
   features and sample in whitened coordinates. ⛔ **A bare `½‖x‖²` on raw post-ReLU features is a silent crippling and
   is refused.** ⛔ A learned flow prior is also refused — it breaks the invariance identity and stops being LACE.
2. **Scoring is image-space through the ONE shared frozen decoder** (resolves the scout's R1). Every arm decodes
   through the same decoder; no arm substitutes its own. LACE is not generator-free in our setup, and that is fair
   because it is not generator-free for anyone.
3. **Two concessions granted to LACE, declared in the prereg** (scout's R2): dense factor labels, and a validation
   split for tuning the OR/NOT constants (`cof`, and the `0.1` in α). ⭐ Freezing those at the published 20/0.1 is the
   easiest way to hobble OR/NOT — the authors' own comments are literally *"these values work well"*. They are tuned
   constants, not laws.
4. **The Langevin arm runs at ≥2 temperatures** (scout's R3): as-published, and at `std = √(2·lr)`. The published
   constants are ~200× colder than their own target (`T_eff = std²/(2·lr) = 0.005`), so as-published LACE-LD is
   near-deterministic gradient ascent and will not cover modes. **Report LACE's best.** This is the scout's arithmetic
   from primary inputs, flagged as a derivation, not an authors' claim.
5. **No Metropolis correction exists in LACE, and its absence is defensible for the ODE arm** — LACE never diffuses the
   classifiers, only the N(0,I) prior, which is VP-SDE-invariant, so composition applies to undiffused log-densities.
   Du et al. 2023's critique bites only if we run the mistuned LD sampler or break the invariance. Our own sampler
   ruling still stands on its own merits and must be declared in the prereg regardless.
6. ⛔ **Traps when copying constants:** shipped `n_epochs=50` (not the appendix's 100, not argparse's 200); batch 128
   FFHQ / 512 CIFAR (argparse default 64 was **not** what was run); the staircase decay schedule **never fires**;
   `dis_temp` is not a clean temperature — do not sweep it. OR is **non-commutative** and there is a hard **10-operand
   limit** (operands are parsed as single characters).
7. ⛔ **Do not write "LACE needs 50k human labels."** Every dataset uses *machine* labels from a teacher; a labelling-cost
   argument is weak and would be an overstatement. The load-bearing asymmetry is **structural**: closed vocabulary, new
   concept ⇒ new labelling pass + new head, and no deletion, no refusal, no invalid-combination detection, no
   enumerable store, no admission decision. Every query is answered.
8. **`DO_NOT_USE_AS_A_BAR`:** CIFAR-10 FID 6.63 / ACC 0.972; FFHQ-glasses FID 20.93 / ACC 0.998; unseen-combination
   0.94–0.98. StyleGAN up to 1024², their splits, their labels. Only our re-implementation is comparable.

## R10 — Montero splits pinned FROM CODE (Hub, from `gen-montero-splits-spec`, 2026-09-09)
Pinned at `mmrl/disent-and-gen@bff35a7570d08fdef0d10c21b093b9a24d98dde6`, `scripts/configs/datasplits.py`.
⛔ **The paper's prose contradicts the code for both datasets and contradicts itself on dSprites rotation. The code
wins; the prose is not a spec.** Deterministic boolean masks, no seed, no sampling.
- **Shapes3D held-out = 72 of 480,000** (Hub-recomputed independently). **dSprites = 5,808 of 737,280**, ±float
  boundary (5,808 / 6,292 / 6,776) — realised count to be reported by the data spoke, never assumed.
- Traps folded into `gen-data-splits.md` before spawn: colour-column slice (silently empty test set), swapped angle
  constants (an engineer "fixing" the apparent bug inverts the held-out band), float boundaries, and the
  `latent_classes` vs `latent_values` literal collision.
- ⛔ Never repeat "the dSprites hold-out is 1 of 243 bins" as fact — the code's cell is ~1.9× that.

## OPEN — raised 2026-09-09, needs a Head ruling
- **B is 72 combinations, and that breaks the C-vs-(A∪B) metric.** B is **0.0150%** of A∪B, so an arm can refuse all of
  B at almost no AUROC cost — which removes exactly the penalty kill rule (iv) was designed to impose on the naive
  kNN distance threshold. See the Hub's options to the Head, 2026-09-09.

---

## R11 — Realised split counts, MEASURED (Hub, from `gen-data-splits`, 2026-09-09)
⛔ **RETIRE 5,808 wherever it appears.** It was a good-faith nominal estimate; the realised count is **6,292**.

| Dataset | n_total | **n_test (realised)** | n_train |
|---|---|---|---|
| Shapes3D | 480,000 | **72** | 479,928 |
| dSprites | 737,280 | **6,292** | 730,988 |

Both **Hub-verified independently** — my own filter code against the raw files, not the spoke's. dSprites: the float
boundary admits **13** rotation values, not the nominal 12 (`grid_rot[13] > a240` is True; `grid_rot[26] < a120` is
False; `0.6 > 0.6` is False ⇒ 4 scale values) ⇒ `1·4·13·11·11 = 6,292`. Corrections back-filled into
`montero_splits.json` under `_hub_corrections`.
**Merged to `main @ 9e87f72`.** Merged tree byte-identical to the branch tip, so the spoke's suite run
(**1919 → 1943, +24 = exactly its new tests, 0 failed**) is the post-merge state.

## R12 — Known issue: h5py is undeclared (Hub, 2026-09-09) — OWNER: Hub, scheduled
`chlu/data/shapes3d.py` requires h5py; it is **not** in `pyproject.toml`, so any `uv sync` prunes it and 8 real-data
tests **silently turn into skips** (`importorskip` ⇒ nothing goes red, nothing goes green either). Also absent from
`scripts/csf3/requirements-csf3.txt`.
⛔ Not fixed mid-wave: adding an extra without re-locking breaks every concurrent agent's `uv sync --frozen`.
**Scheduled: add a `gen = ["h5py>=3.0"]` extra, re-lock, regenerate `requirements-csf3.txt` — at the next window with
no engineer spoke in flight** (i.e. once `gen-csf3-launcher` lands). Until then downstream spokes install it manually
and say so.

## R13 — Task-template defect found and fixed (Hub, 2026-09-09)
My engineer task files carried ⛔ *"read no campaign ledger, charter or handover"*. That is right for information
hygiene but **too broad**: it also excluded `.claude/handover_context.md` **§7 Known Issues**, so a spoke could
re-introduce a fixed bug unknowingly. `gen-data-splits` flagged this itself. ⭐ **Spot-checked: no regression** — §7.10
(`--quick` unwired), §7.11 (unseeded subsample) and §7.12 (CLI env bug) were all handled correctly anyway.
**Fix applied to every unspawned engineer task file: the ⛔ stands, EXCEPT §7 Known Issues, which is required reading.**

---

## R14 — Rulings imported from `iclr_pivot_head_advisor.md`, 2026-09-09 (Head + Advisor)
These CLOSE items previously marked OPEN in this ledger. ⛔ Do not re-litigate them.

1. ⭐ **Refusal metric: PRIMARY = AUROC(B unseen-valid vs C impossible).** Kill rules **(iii) and (iv) bind on it.**
   C-vs-A and C-vs-(A∪B) are reported alongside. **This closes the Hub's open D4.**
2. ⭐ **C is sized comparably to B**, pre-registered, sampled from the factor-space definition.
   **DONE — `main @ 215fc4d`:** `c_invalid_n_per_kind` now defaults to `None` and derives as `ceil(n_test/2)`.
   Regenerated: **Shapes3D |C| = 72** (sha256 `b886da239e97b19e9001a88ce04fded329dda4884d301f78c3097ea6cc66ba49`),
   **dSprites |C| = 6,292** (sha256 `10a39e713864c9f0c28f740caad49394cbda6f34f3d1d29bbc61dab9fc288666`).
   ⛔ These supersede the 32-record files and the checksums in `gen-data-splits.md` §6.
3. **Encoder/LACE trade-off RULED — our choices never weaken a competitor.** Either pick a shared feature width LACE
   can whiten **natively**, or give LACE a **declared PCA projection and report it both ways.** ⛔ This supersedes the
   "come back to the Head if `D > 2e4`" wording in `gen-feature-bridge.md`: it is now a build instruction, not an escalation.
4. **Backup push: the internal (code + state) fork ONLY.** ⛔ Never the code-only base repo, ⛔ never `origin`.
   The runbook script pushes both, so it is **not** run as written.
5. **Confirmed, unchanged:** the decoder gate (≥0.8× floor, stop-and-return) · the store-deletion check before arms
   (kill rule (i), moved early) · Head + Advisor sign the point predictions and kill thresholds before the cell switch.
6. ⛔ **Never-quote, binding on every draft:** capability 1 (compose unseen-valid) is **never presented as novel on its
   own** — LACE owned it in 2021. The claims are the accept/refuse **asymmetry** and the **structural store**.
7. **Binding design rule inherited from the CAMELS work, applies to our headline:** every (B) unseen-but-valid split
   must be protected against *the donor arm in disguise*. On factor data that means held-out combinations verified
   **genuinely absent** (done: the assertion, proven able to fire) **and the probe able to distinguish**.
   ⭐ **B-vs-A separation is a measured property of the split, never an assumption** — the bridge's ceiling measures it.

## STILL OPEN — needs the Head
- **Widened unseen set as a labelled SECONDARY split.** Advisor recommends YES (Montero's 72 stays the headline and
  the protocol is untouched; our extra held-out combinations reported separately for statistical power, verified
  absent by the same assertion). **Head to confirm.** If yes it is one more spoke, sequenced after the cell.

---

## R15 — GATE 1 PASSED (`gen-feature-bridge`, merged `main @ fcdca43`, 2026-09-09/10)
**Shapes3D · `recon` · 3 seeds · Hub-recomputed from `ceiling.json`, not read off the report.**

| | value |
|---|---|
| `ceiling_A` | **0.9507 ± 0.0129** (n=1024) |
| `ceiling_B` | **0.8472 ± 0.0218** (n=72) |
| ratio | **0.8912** (per-seed 0.9161 / 0.8724 / 0.8850) |
| floor (R3.1) | 0.8 |
| **verdict** | ⭐ **PROCEED**, 3/3 seeds; worst seed still clears |

- **R3.3 no-leak holds:** 72/72 held-out images excluded, **0 leaked**, per seed.
- **The probe reads REAL images at 1.0000** on both A and B ⇒ the instrument is sound and **saturated**. ⛔ Nobody may
  later read "probe accuracy" as a difficulty axis; all loss in this phase is through the decoder.
- ⭐ **`D = 2048 ≤ 20000` ⇒ LACE whitens NATIVELY** (ruling option 1, R14.3). ⛔ No PCA projection, no channel-only
  fallback, no declared deviation needed. `N_train_features = 8192`; shrinkage not mandatory.
- **`PREREG.md` timestamps ~10 h before the first run** — verified by mtime, not by claim. One prediction was
  **falsified** (the hard factor was predicted to be `orientation`; it is `scale` then `shape`) and that falsification
  is what produced the finding below.
- Suite **1976 → 2010**, delta exactly its 34 new tests, 0 failed. ⭐ The §7 reading exemption (R13) paid for itself:
  §7.23's x64-at-module-import hazard broke 9 of its tests in the full suite and passed in isolation; it root-caused it
  from §7 rather than chasing it.

### ⭐ THE FINDING — the B deficit is mostly PHOTOMETRIC, not compositional
Montero's cell requires **all three hues ≥ 0.8**, which puts floor, wall and object in the same band and collapses
object/background contrast. Against a **photometrically matched SEEN control** (same hue corner, seen shape, n=209):

| | aggregate | scale | shape |
|---|---|---|---|
| `A_random` (the reported `ceiling_A`) | 0.9476 | 0.859 | 0.904 |
| **`A_matched`** (same hues, seen shape) | 0.9203 | **0.703** | 0.876 |
| `B` (held out) | 0.8681 | **0.708** | **0.500** |

⭐ **`scale` on B is statistically indistinguishable from `scale` on matched SEEN data** ⇒ a low-contrast rendering
artefact, not a composition failure. ⛔ **The only genuinely compositional term is `shape`: 0.876 → 0.500.**
Against the stricter normaliser the gate passes *more* comfortably (B / `A_matched` = **0.943**).

## OPEN — needs the Head, before the arms spoke spawns
- ⭐ **How kill rule (i) reads the ceiling.** R3.2 compares store-removed vs CLU as a fraction of `ceiling_B`, but two
  thirds of `ceiling_B`'s deficit is photometric and four of six factors sit at ≈1.0 and carry no compositional
  information. On the flat aggregate an arm can pass or fail on rendering contrast rather than on composition.
  ⚠ **This is a scoring refinement made BEFORE any arm exists and BEFORE the cell runs**, motivated by a property of
  the split and the decoder — not by any arm's result. It must be **pre-registered in `PREREG.md`** and ⛔ **may not be
  revisited once the cell has run.**

---

## R16 — ⛔ GATE 1 REVERSED ON THE COMPOSITIONAL TERM: **STOP** (Head, 2026-09-10)
**Scoring RULED (option 1, irrevocable):** kill rule (i) reads on the **compositional term** — `shape` primary,
`scale` normalised against the **matched-seen** control — with the aggregate reported alongside. ⛔ Option 3
(renormalise everything) rejected: it moves more than the one broken thing, and a comfortable 0.943 is exactly the kind
of number that should make us suspicious of a scoring change.
⭐ **The matched-seen control becomes a permanent companion row in every table in this phase.**

### ⭐ THE CONSEQUENCE THE HUB MISSED (Head drew it): the gate must read on the same term as the kill rule
Otherwise the gate and the kill rule test different things. Hub-recomputed from `ceiling.json` and
`matched_control.json`:

| term | normaliser | A | B | ratio | vs floor 0.8 |
|---|---|---|---|---|---|
| **`shape`** | matched-seen (seed 0) | 0.8756 | 0.5000 | **0.571** | ⛔ **FIRES** |
| **`shape`** | random-A (3 seeds) | 0.9076 | 0.5926 | **0.653** | ⛔ **FIRES** |
| `scale` | **matched-seen** (seed 0) | 0.7033 | 0.7083 | **1.007** | ✅ passes — ⭐ the photometric attribution holds exactly |
| aggregate | random-A (3 seeds) | 0.9507 | 0.8472 | 0.8912 | ✅ passes (but is 2/3 photometric) |

⛔ **The decoder cannot render held-out shapes from REAL features.** Comparing arms in a 0.25–0.50 band on 72 items
measures the decoder's ceiling, not the store. **This is the capped-B failure the gate exists to catch, and it caught it.**

### ⚠ Hub diagnostic the Head has not yet seen — it bounds how much option 1 can buy
The **`cls`** diagnostic pretext (R1's fault-isolation arm) reaches `shape` **0.6944 / 0.8988 = 0.773** — **better than
`recon`'s 0.653, and still below the 0.8 floor.** ⚠ `cls` differs in *both* features and the decoder fitted to them, so
this is not a clean features-only bound. But it is the one empirical datapoint we have, and it says **the whole
pipeline under a more favourable pretext still does not clear 0.8 on `shape`.**
⇒ ⭐ **Pre-register the expectation that a decoder-only fix will NOT clear 0.8**, so that falling back to option 2 is a
pre-registered outcome rather than a retreat. If it does clear, that is a genuine, pre-registered win.

### The two ways out, in the Head's order
1. **Improve the decoder, bounded** — train-split only, frozen afterwards, **two variants maximum**, a day of effort,
   re-probe. `gen-decoder-lift`.
2. **If the ceiling does not move** — accept the band, pre-register the floor **at the measured ceiling** with the
   reason stated, and the widened split becomes **mandatory**.

## R17 — Widened split is MANDATORY (Head, 2026-09-10)
Not optional, not a nice-to-have. With `shape` as the compositional term, **72 items × 3 seeds inside a half-width band
cannot separate two arms at any confidence a reviewer will accept.** The extra held-out combinations — verified absent
by assertion, reported separately, Montero's 72 untouched as the headline — are what turn this from an anecdote into a
measurement.

## R18 — What is NOT invalidated (Hub, 2026-09-10)
⭐ `gen-store-grouping` **continues**. The store trains on cached features from the **frozen encoder**; the decoder is
downstream of it. A decoder-only fix therefore leaves the feature cache and the store untouched.
⛔ **This is why `gen-decoder-lift` is constrained to the decoder alone.** If the fix needed a different encoder, the
cache changes and every downstream spoke's work is voided — that would be a much larger decision and it returns to the
Head, not to the spoke.

---

## R19 — Wave 3 merged and verified on the MERGED state (Hub, 2026-09-11, `main @ e4f3d71`)
Three branches merged `--no-ff`, **zero conflicts** — the `config.py` block split (R16 ownership note) held.
**Full suite on the merged tree: `2078 passed, 0 failed` (45:03), HEAD `e4f3d71` before AND after the run.**
⭐ Test accounting is exact: `2010 (bridge) + 18 (wide-split) + 19 (decoder-lift) + 31 (store-grouping) = 2078`.
Nothing was lost or double-counted in the three-way merge.

### The three results
1. ⭐ **`gen-decoder-lift`: `shape` CLEARS.** 0.5926 → **0.9398** on B (3 seeds); ratio **0.9964** vs matched-seen,
   **0.9656** vs random-A; 3/3 seeds. ⛔ **R16's STOP is lifted; option 2 does NOT activate.** The winner was the
   variant filed as the **null control** (capacity + longer schedule, uniform pixel MSE untouched); the
   object-weighted loss the Hub nominated as "the single most useful clue" **lost badly** (0.537). ⭐ The Hub's
   pre-registered expectation P5 ("a decoder-only fix will NOT clear 0.8") is **FALSIFIED** — which is exactly what
   filing it was for.
2. ⭐ **`gen-store-grouping`: both falsifiers held.** `delete_zero_delta_aggregate = +0.4752` against a near-zero line
   of 0.01 — **475× clear of the programme's standing failure mode.** ⭐ Stronger than the number: `read_from_codes`
   **does not take the features as an argument**, and the only encoder→decoder channel is **30 bits**. A bypass is not
   merely unmeasured, it is **structurally inexpressible.** Correlation beats random by **+0.1504** aggregate /
   **+0.2015** on `shape`, margin **81×** the largest seed spread.
3. **`gen-wide-split`:** positive control fires **13/13** cells; Montero's condition re-emits **bit-identically**.

### ⛔ CARRIED FORWARD — open, not resolved by this merge
- ⭐ **The `shape` ratio compares a DETECTION rate to a DISCRIMINATION rate.** Montero's cell *is* `shape == cylinder`,
  so `shape` on B has **exactly 1 realised class, majority rate 1.000**, while matched-seen has 3 (majority 0.354).
  **Survives:** the 0.59 → 0.94 lift (detection vs detection, same probe, same set). **Does not survive:** the
  ratio-to-ceiling, which is what kill rule (i) reads. **Hub recommendation, NOT yet ruled:** interim normaliser =
  matched-seen restricted to a single non-cylinder shape (1-class vs 1-class), plus the widened split structurally.
  ⛔ Reject "drop `shape` from the pooled metric" — it discards the only compositional factor.
- ⭐ **The ladder x-axis is degenerate as literally specified** (near-zero denominators; per-factor leakages to −12.9).
  Must be fixed **before R1…R5**, and R0 re-scored under it.
- ⭐ **R0's read numbers were measured against the SUPERSEDED decoder.** They need re-scoring against the
  decoder-lift winner regardless. ⭐ **These three fold into ONE cheap re-score spoke** — the store is untouched
  (it trains on features) and leakage probes never touch the decoder, so nothing retrains.
- **`gen-baselines` byte-match target: `524,288 B` exactly** (`runs/shapes3d_recon/store_bytes.json`).
- ⚠ **Feature caches are non-uniform:** seed 0 is `scope="full"` (480,000 rows); seeds 1–2 are `scope="used"`
  (9,288 rows) and contain **none** of the matched-seen control rows. Every future gen spoke hits this.
- **Launcher needs a `CONDITION` axis** for the wide split (scoping item; no launcher edit was made).

---

## R20 — Metric corrections RULED, and one result promoted to a design constant (Head, 2026-09-11)

### ⭐ The store result is the programme's biggest in three campaigns — and it is the STRUCTURE, not the number
⛔ The headline is **not** `+0.4752`. It is that **a bypass is inexpressible**: the read never sees the features, and
the only encoder→decoder channel is **30 bits**. The routing-around failure that consumed two campaigns is **closed by
construction, not by measurement.**
- ⭐ **That sentence goes into the paper's method section verbatim.** (Head ruling; the Head writes the prose.)
- ⭐ **The 30-bit channel width is now a DECLARED DESIGN CONSTANT** — `K·log₂M = 5·6`. It is reported as a design
  choice, never as an incidental implementation detail, and any later change to it is a declared change to the claim.

### ⭐ Kill rule (i) reads DISCRIMINATION WITHIN THE CORNER — not detection
Confirmed: one-class matched-seen normaliser as the interim, widened split structurally. **Plus the Head's tightening,
which supersedes the Hub's weaker proposal:** with a one-class held-out cell, a probe that answers "cylinder" for
*anything* in that colour corner scores 1.0. So the like-for-like number is:
> **Positives:** the held-out cylinders. **Negatives:** matched-seen **non-cylinders in the same colours**.
> **Score:** binary **AUROC or balanced accuracy** — cylinder vs not, *within the corner*.
⛔ This is what kill rule (i) reads **until the widened split's multi-shape cells make it moot.**

### ⭐ Ladder x-axis: transfer of a BALANCED metric, never a ratio of raw accuracies
> `leakage = 1 − [balanced accuracy of the per-well shape probe on (held-out ∪ matched-seen-other-shapes), same corner]
>            ÷ [the same metric on seen corners]`
⛔ **Balanced accuracy cannot go degenerate; ratios of raw accuracy against a one-class cell will always produce
−12.9-style nonsense.** This supersedes the leakage definition in the `gen-store-grouping` amendment.

### ⭐ The decoder gate must be re-run on MULTI-SHAPE held-out cells
`0.94` is a **detection** rate. ⛔ **We have not yet seen the decoder discriminate shapes in a held-out corner.**
⚠ **Hub scoping note the Head should have:** this is **not** cheap and does **not** belong in the re-score spoke. The
current encoder *and* decoder trained on Montero's train split, which **contains** the widened cells' rows — they have
seen them. A clean held-out ceiling on multi-shape cells therefore requires **a full bridge re-run at the wide
configuration** (encoder + decoder fitted with the wide cells excluded). ⭐ That is the second configuration R17
already requires, so it is not new cost — but it is a run, not a re-score. **Split into two spokes accordingly.**

### What survives the corrections unchanged
- ⭐ **`+0.4752` is a read-space number and survives the decoder change** (Head). The store's load-bearing status is
  not affected by any of the above.
- **Correlation beats random by +0.20 on `shape` at 81× seed spread** — the first evidence that the wells carry factor
  structure, and **the R0 point on the ladder**.

### For the negatives registry (Head ruling — both go in)
1. **"A decoder-only fix will not clear 0.8" — FALSIFIED.** It reached 0.94.
2. **The nominated "best clue" LOST to the null control.** The object-weighted loss scored 0.537; plain **capacity +
   longer schedule** with uniform pixel MSE untouched won.
⭐ **Lesson to keep:** the shipped decoder was **under-resourced**, and capacity + a longer schedule beat every clever
objective. ⛔ **Default to that first next time**, before reaching for a designed loss.

## R21 — Standing gate (Head, 2026-09-11)
⛔ **The launcher's live CSF3 smoke runs BEFORE any arm.** It is a Head action (2FA) and it gates the arms, not the
re-score.

---

## R22 — `gen-r0-rescore` accepted (Hub, 2026-09-11). No branch, no merge — read-only analyst.

### ⭐ SURVIVES: kill rule (i)'s corrected metric changes the LEVEL, not the VERDICT
Positives = 72 held-out cylinders; negatives = matched-seen **non**-cylinders in the same corner (n≈198). Chance 0.5.
⛔ **NOT comparable in level to the old `shape` accuracies (0.5926 / 0.9398).**

| decoder | balanced accuracy | AUROC | TPR | TNR |
|---|---|---|---|---|
| baseline | 0.7191 ± 0.0305 | 0.8806 | 0.5926 | 0.8457 |
| ⭐ **V1** | **0.9414 ± 0.0179** | **0.9886** | 0.9398 | 0.9429 |

**V1 clears 0.8 on 3/3 seeds (worst 0.9223); the baseline on 0/3 (best 0.7587).**
⭐ **The elegant confirmation:** the old `shape`-on-B accuracy **is exactly this metric's TPR** — `0.5926` and
`0.9398` reappear verbatim. The correction's entire content is the **TNR column the old ratio could not see**: the
baseline called **15.4 %** of *seen* non-cylinders "cylinder"; V1 calls **5.7 %**. ⇒ **the lift was real
discrimination, not a detection artefact.**

### ⭐ SURVIVES: R0's grouping evidence is decoder-independent
Read margin **+0.1504 → +0.1504** (identical to 4 dp), `shape` **+0.2015 → +0.2008** (−0.3 %). The better decoder
lifts **both** arms by the same amount (+0.0110 corr, +0.0111 rand). ⭐ The store-read vectors were never what the
decoder was fixing. No-store ceiling rises to 0.9902 aggregate / 0.9733 `shape` under V1.

### ⭐ SURVIVES: the corrected leakage is bounded
**Hub-verified from `leakage_cells.json`: 180 cells, ZERO non-finite, range [−0.5534, 1.0000].**
⛔ The old definition: **55 of 180 cells NaN (30.6 %)** and a worst cell of **−198.0** (a detection rate 1.000 ÷ a
3-class rate 0.005025). ⚠ **R19's carry-forward text quoted "−12.9" — that under-stated it by 15×. Corrected here.**

### ⛔ DOES NOT SURVIVE: the `shape`-specific LEAKAGE advantage
Old margin **+1.0381** → corrected **+0.0147 ± 0.0771** — **0.19× its own seed spread, sign-inconsistent per seed
(−0.0318 / −0.0475 / +0.1233).** Hub-verified from `summary.json`.
⛔ **Any claim that the wells carry `shape` structure ON THE LEAKAGE AXIS must be re-worded.** ⭐ The same claim on
the **read** axis is untouched (**+0.2008, 12.2× spread, 3/3 seeds**) — that is where R0's claim rests.
**Pooled leakage also weakens:** +0.2701 (5.2× spread) → **+0.1337 ± 0.0592 (2.26× spread)**. Still 3/3 positive and
directionally right, ⛔ but it can no longer carry a headline. R0's claim rests on the **read margin** (16× spread)
and the **code-space companions** (MIG 9.9×, DCI-C 8.8×).

### ⚠ In-corner store read — honest level
Correlation **0.6866 ± 0.0681** balanced accuracy vs random **0.5491 ± 0.0146** (barely above chance); margin
+0.1375 (1.89× spread) / AUROC +0.1788 (5.3× spread), 3/3 positive both. ⭐ **A 30-bit channel carries little
in-corner shape identity for either grouping** — correlation's 0.6866 sits far below the real-feature 0.9414.
That is a **store-capacity** statement, not a grouping one. ⛔ Not what kill rule (i) reads.

## OPEN — needs the Head
1. **Which statistic kill rule (i) reads.** Analyst recommends **balanced accuracy**, AUROC as a mandatory
   threshold-free companion: the 0.8 floor was written against an *accuracy*, and AUROC is the more generous here
   (0.9886 vs 0.9414) so reading it would **weaken** the gate. ⭐ Hub concurs.
2. ⭐ **The ladder's x-axis may lack resolution on the factor that matters.** Leakage cannot separate correlation
   from random on `shape` at R0. Options: **(a)** run as designed, read capability claims off pooled leakage +
   code-space companions, declaring `shape`-leakage underpowered; **(b)** ⭐ **sequence the ladder AFTER
   `gen-bridge-wide`** — with all four shapes held out, `shape` leakage becomes 4-class on both sides and may regain
   resolution, at no extra cost since that configuration is already mandated and in flight; **(c)** change the x-axis
   to a code-space measure (MIG / DCI-C), which separates cleanly at R0. **Hub recommends (b), fallback (a).**

---

## R23 — ADVISOR VERDICT, 2026-09-11. All binding. ⛔ Do not re-litigate.
1. **Kill rule (i) reads BALANCED ACCURACY** on the compositional term. AUROC is a **mandatory companion, never the
   gate statistic.** Floor stays **0.8×**.
2. **The ladder is sequenced AFTER `gen-bridge-wide`.** x-axis `= 1 − balanced-accuracy transfer` (held-out ∪
   matched-seen-other-shapes, same corner) ÷ seen-corner balanced accuracy. ⛔ **Never a ratio of raw accuracies.**
   **MIG and DCI-C are pre-registered companions, not replacement axes.**
   ⭐ **Fallback, PRE-REGISTERED NOW:** if `shape` leakage on the widened split separates the groupings by **less than
   ~3× seed spread**, P1 is read off pooled leakage + companions and `shape`-leakage is declared **underpowered, in
   those words, in the paper.**
3. **Shared-corner cells (`montero`, `wide-000`) are quoted as a TWO-CLASS corner, never pooled.** The pooled headline
   runs on the **other 11 cells**; the corner pair is reported alongside.
4. ⭐ **Claim wording, effective immediately:** *"the wells carry factor structure"* is established on the **READ axis
   only** (+0.2008, 12× spread). ⛔ It is **NOT** established on the leakage axis at R0 (+0.0147 ± 0.0771,
   sign-inconsistent). **No semantic-organisation sentence rests on leakage until the ladder produces one.**
5. **The 30-bit channel is not touched in phase 1.** The in-corner store read (**0.6866** vs the **0.9414**
   real-feature ceiling) is **declared in the cell as the STORE ceiling, companion to the decoder ceiling.**
   Capacity-vs-bypass-risk is a separate, later, pre-registered axis.
6. ⛔ **Cluster smoke runs before any pre-registered cell. No exceptions.**

### ⭐ Ruling 3 applied — Hub-recomputed from `summary.json`
| | balacc held-out | balacc seen | ratio | AUROC |
|---|---|---|---|---|
| all 13 cells (as first reported) | 0.9798 ± 0.0088 | 0.9774 | 1.0025 | 0.9994 |
| ⭐ **11 cells (the headline)** | **0.9894 ± 0.0021** | 0.9861 | **1.0034** | 0.9993 |
| `montero` (2-class corner) | 1.0000 ± 0.0000 | — | — | 1.0000 |
| `wide-000` (2-class corner) | 0.8542 ± 0.0967 | — | — | 1.0000 (TPR 0.7083) |

⭐ **The ruling does more than tidy the metric: the seed spread tightens 4× (0.0088 → 0.0021)**, because
`wide-000`'s ±0.0967 instability was the dominant contributor. Worst of the 11 is `wide-001` at **0.9470**.

## ⚠ TWO GAPS IN THE ORDER OF WORK — Hub flags, not objections
**(b) and (c) are already DONE.** `gen-r0-rescore` is accepted (R22) and `gen-bridge-wide` is merged
(`main @ 9a4457d`). The "decoder ceiling gate on the widened split's multi-shape cells" that (b) attaches was in fact
delivered by (c). ⇒ **remaining: a, d, e.**

1. ⭐ **There is no step between (d) and (e) and there needs to be.** The A/B/C cell has **no arms to run**. Three
   task files are written and have **never been spawned**: `gen-dream-sampler` (conditional dreaming + the
   store-removed ablation), `gen-certificate` (the refusal criterion), `gen-baselines` (byte-matched kNN + LACE).
   Plus `gen-prereg`. ⛔ (e) cannot start until those land.
2. ⭐ **R0 is on the WRONG CONFIGURATION for the ladder.** The ladder requires "same store config, same bytes, K
   fixed" across rungs, and ruling 2 puts R1–R5 on the **widened** configuration — but the shipped R0 store was
   fitted on **Montero's** feature cache. As it stands R0 and R1–R5 would not be comparable.
   **Hub recommendation:** re-fit R0 on the **wide** feature cache (which `gen-bridge-wide` has now produced at
   `scope="full"` for all 3 seeds) so all six rungs share one configuration, and **keep the Montero R0 as the
   published link to the literature**. The store fit is ~1200 Adam steps — cheap. ⛔ Needs a ruling before R1 runs.

---

## R24 — R0 is re-fitted on the wide cache; the ladder runs entirely on the wide configuration (Head, 2026-09-11)
⭐ *"A monotonicity claim on a mixed baseline is no claim at all"* — and the fix costs minutes. **Three riders, all binding:**
1. ⭐ **Re-derive the GROUPING on the wide train cache, not just the store weights.** The correlation clustering was
   computed on Montero's training features and the wide train set differs by the held-out cells. Expect near-identical
   channel correlations — **but measure it: report adjusted-Rand agreement between the Montero and wide groupings.**
   ⭐ High agreement documents a non-issue; **low agreement is a finding about grouping stability that we want BEFORE
   R1, not after.** ⛔ `K` stays fixed — it is a ladder constant (`store_plan.json`, `assert_plan_matched`).
2. ⛔ **Montero R0 stays as the literature link, OFF the ladder.** It appears in the compositional-results table
   against the published split; **the ladder figure carries wide-configuration rungs only. Never both on one axis.**
3. ⭐ **THE GATE — wide-R0 and Montero-R0 side by side, once**, as a consistency check on the **caches**.
   ⛔ **If they diverge beyond seed spread on the read margin, STOP before R1 — that is a PLUMBING FAULT, not a
   result.** Montero's shipped comparison values: aggregate margin **+0.1504 ± 0.0094**, `shape` **+0.2008 ± 0.0165**.
**⛔ R1 does not start until the re-fit lands AND the agreement number is in this ledger.**
**Task file:** `.claude/tasks/gen-r0-wide-refit.md`.

## Hub housekeeping, 2026-09-11
- **Worktree `CHLU-gen-store` removed** after verifying from the main repo that `gen-store-grouping` is an ancestor of
  `main` (0 unmerged commits, 0 uncommitted files). 5 worktrees remain; 3 are parked from earlier campaigns.
- ⛔ **Standing trap, now in every live task file:** `chlu/config.py` `dec_recipe` defaults to **`"baseline"`** — the
  decoder that **FAILED** the gate (0.7191). The winner is **`v1_capacity_schedule`** (0.9414). The default was left
  alone deliberately so Montero's history reproduces. ⭐ **Every spoke must select the recipe explicitly and assert the
  decoder parameter hash. A run through the baseline looks entirely normal in the logs and is void.**

---

## R25 — Wave 4 results (Hub, 2026-09-12). `main @ a0034d4` + certificate merge; suite **2166 passed, 0 failed**, HEAD stable.

### `gen-dream-sampler` — ⛔ **THE PRE-REGISTERED FALSIFIER FIRED. The dial's answer is NO.**
Contrast `full − zero_step = +0.0095 ± 0.0082` vs a pre-registered `+0.03`. ⭐ **The informative failure, not the
vacuous one:** the chain genuinely samples (95.3 % of rollouts change a code; 499/512 configurations vs 278 for the
clamp alone; acceptance 0.952; energy moves 1.0× its own scale) and it moves **exactly one factor tightly** —
`shape`, `Δ = +0.0215 ± 0.0033`, 3/3 seeds, 6.5× spread.
⛔ **But the separability control takes all of it:** with coupling off, where the clamp **provably cannot reach** a
relaxed well, `Δshape = +0.0200 ± 0.0181` ⇒ **steering-attributable = +0.0015 ± 0.0204 = ZERO.** The gain is the chain
settling onto a mode, not the clamp steering. ⭐ The spoke built the control that killed its own favourable reading.

### `gen-r0-wide-refit` — ⭐ **GATE: `CONSISTENT`** (0.70× pooled seed spread, both terms). No plumbing fault.
Wide read margins **+0.1410** aggregate (14.5× spread) / **+0.1813** `shape` (8.0×). `assert_plan_matched` passes;
Montero R0 untouched and bit-identical.

### `gen-certificate` — ⭐ the derivation is NOT decoration over naive geometry, and it took a control to show it
- Zero-invalid-supervision audit: **0 C-record ids in 15 fits / 78,848 ids**, every arm, every seed.
- ⚠ On the **contaminated** dup metric: naive `distance_reach` 0.8372 vs barrier 0.9138 ⇒ **+0.077, BELOW** its
  pre-registered +0.10 — an equivocal result. ⭐ **Remove the giveaway (A7) and the control collapses to 0.5630 while
  the barrier holds at 0.9146** ⇒ margin **+0.3516 ± 0.0372, 9.5× spread, 3/3.**
- ⭐ **Attribution fires for the right reason**, and no term count can fake it: names the culprit factor at
  **0.797** (dup) / **1.000** (oor) vs chance 0.167 = **4.8× chance**; correlation beats random **+0.130 ± 0.026, 3/3**.
- ⛔ `distance` is **at chance vs A (0.5264)** and **below chance on out-of-range (0.4836)** — its B-vs-C number is a
  property of the B set, not a refusal ability. ⛔ Never quote the naive baseline without that column.
- ⛔ **`distance_reach` is the in-house control, NOT the byte-matched kNN arm kill rule (iv) names.** That is
  `gen-baselines`'. The collapse is a **prediction** there, and `gen-cell-run` must measure it.

### ⭐⭐ THE C-SET TERM-COUNT LEAK — a defect in the Hub's own spec, Hub-verified independently
All **36** `duplicate_factor` records carry **7 terms**; A and B carry **6**. A function that only counts terms scores
**AUROC 1.0000** on C-dup-vs-B, 0.5000 on C-oor, **0.7500 pooled** — with no model in it. ⛔ Every arm inherits it and
it is invisible in an aggregate C number. **Cause: the Hub's C spec never required C records to be structurally
indistinguishable from A/B.** Fix shipped by the spoke (**A7**: valid requests carrying a *redundant* term — same
image, 7 terms, built from valid data alone, so the supervision claim is untouched).
⛔ **`gen-baselines` ran all 3 seeds (15:20 / 16:27 / 17:10) with the contaminated set — its C numbers need re-scoring.**

---

## R26 — ADVISOR RULINGS, 2026-09-13. All binding.

### 3 (first, because it changes the cell) — ⛔ CONDITIONAL DREAMING IS DROPPED AS THE COMPOSITION MECHANISM
The falsifier fired cleanly and the coupling-off control makes it unambiguous: **the chain samples a mode, the clamp
does not steer.** Item 4 says the same from the other direction — the request already specifies **4.42 of 5 wells**, so
there is nothing to relax.
⭐ **Composition is the k-particle launch READ over the requested modes, then decode.** That is the validated mechanism
(0.86 exact-set; +0.18 `shape` read margin) and the one the Head originally named. **Clamp-and-relax was the Advisor's
own design addition and is retired.**
- **B path:** `request → read → code → frozen decoder → probe`. **C path unchanged** (certificate on the request).
- Dreaming survives **only** as within-mode sampling for diversity, **never as steering**, and is **off the phase-1
  critical path**.
- **Partial-request completion is a DIFFERENT capability — banked as secondary, not scored in this cell.**

### 1 — A7 approved; **re-score** the baselines' C numbers, do not re-run
⭐ **NEW STANDING RULE:** every C set carries a mandatory **no-model detector control** — a function seeing only
request *structure* (term count, ordering, id ranges) must score **≤ 0.5 + spread, pre-registered, before any arm is
scored.** The certificate spoke's control found this one; the rule means it never has to again.

### 2 — Ladder x-axis: **promote DCI-C**, chance-corrected transfer as companion
⚠ The Advisor rejected this last time as metric-shopping. **What changed:** the leakage axis failed **structurally**
(class-count mismatch, sign inversion across configurations — diagnosed, not unlucky) and the pre-registered fallback
triggered. ⭐ **The principled criterion is CONFIGURATION-INVARIANCE — the axis must give the same value for the same
features under different partitions — and it was chosen before seeing any rung result.** DCI-C meets it;
probe-transfer leakage does not. **Stated in the paper in those words.**
Companion: **chance-corrected balanced accuracy `(BA − 1/c)/(1 − 1/c)`**, which removes the class-count dependence.
⭐ **STANDING RULE, third strike:** **every comparison metric on the ladder or the cell gets a pre-registered
invariance check — same features, different partition, same value — before it is used.**
⛔ **R1 unblocks only once that check passes for DCI-C.**

### 5 — Re-derive the grouping per seed
Only `K`, `n_modes`, `D`, `store_bytes` are portable. ⛔ Labels indexing a seed's own encoder channels never travel.
**Price of the earlier contract, for the ledger: store ceiling 0.5579 vs 0.6208 on seed 1.**

### 6 — the ARI rider is retired
Cross-encoder ARI is bounded by encoder reproducibility (~0.06), so 0.115 cannot mean what it was meant to.
⭐ **The consistency check that DOES hold is the read-margin agreement at 0.70× spread — that is the plumbing test,
and it passed.** The ARI number stays with its caveat.

### 7 — launcher + data_root
Add the **`certificate` stage** to `scripts/csf3/job_gpu_gen.sh`. ⛔ The `gen_data.data_root` ambiguity that silently
skips the C set is *"declared not-run reported as null"* in code form — **it must fail loudly, no default.**
⛔ **Cluster smoke still gates the cell; nothing pre-registered runs before it.**

### ⭐ The certificate is tier 1 of the refusal claim — the pivot's first headline-grade number
Barrier **0.9146** vs naive control **0.5630** after the giveaway is removed (9.5× spread), attribution naming the
culprit factor at up to **1.000** against chance **0.167**, **zero invalid ids across every fit** — on an honest C set.

## ⚠ R26 CONSEQUENCE THE HUB FOUND — the byte budget reverts, and the kNN arm is now OVER-resourced
**Hub-verified:** `fit_code_coupling` / `coupling_bytes` are referenced **only** inside `chlu/core/gen_dream.py`;
`gen_certificate.py` does not import them (its `pair` term is built from per-well mode histograms over valid rows, a
different structure). ⇒ ⭐ **With dreaming off the critical path, the 163,840 B coupling field leaves the deployed
store and the byte-match target reverts 688,128 → 524,288 B.**
⚠ `gen-baselines` ran all 3 seeds at **688,128** (the Hub relayed it mid-run as amendment A5). ⇒ **the kNN arm carries
a ~31 % byte ADVANTAGE it should not have.** ⭐ This is the **conservative** direction — an over-resourced competitor
that still loses is a stronger result, not a weaker one — but it **must be declared, not left silent.**
⚠ **Separately, and not yet in any ledger:** the certificate deploys its own state (per-(factor, grid-value) mode
histograms + the mode prior). **That is a real byte cost of the refusal capability** and the competitors' refusal
thresholds are ~1 float. ⛔ It belongs in the cell's byte ledger. **Hub estimate ≈ 74 KB — to be MEASURED, not assumed.**

---

## R27 — ADVISOR/HEAD RULINGS, 2026-09-13. All binding.

### 1 — ⛔ Kill rule (ii) reads the BYTE-MATCHED arm (`knn_proto`, unbounded channel), **as pre-registered**
⭐ *"Reading the channel-matched number would be an amendment made after seeing a baseline result, in the kinder
direction. Every scoring refinement we have accepted so far was neutral or stricter; this would be the first that
is not, and it would be the one a reviewer finds."* **The rule stands as written.**

**Decided NOW, before the CLU arm runs:**
- ⭐ **If (ii) fires it kills the COMPOSITION claim, not the phase.** Each kill rule maps to a claim: (i) store
  attribution · (ii) composition · (iii)/(iv) refusal. Composition was already table stakes after the LACE read.
  A prototype-mean table beating our ceiling says **recombination-to-element on Shapes3D is solved by additive
  prototypes once features are decent** — a finding about the task, and reportable. ⛔ **Refusal and structural
  deletion were never conditioned on (ii) and do not inherit its verdict.**
- **Both rows go in the paper, neither hidden.** Byte-matched: prototype means **0.8884** beat our store ceiling by
  **+0.20**. Channel-matched: at 30 bits our store reads **0.6866** where a lookup of the same width sits at
  **chance (0.5206)**. ⛔ The second is a **bits-efficiency statement, not a rescue**, and is presented as exactly that.
- ⭐ **NEW ARM, pre-registered now: `CLU-wide`.** Decode from the store's **composed per-well mode vector** at the
  same channel width as `knn_proto`. ⛔ **store → decoder, NEVER encoder → decoder** — the bypass guarantee stays
  intact. It is the honest like-for-like and **it is not kinder**: if composed modes are just prototype means with
  machinery attached, it will tie `knn_proto` and we say so. ⛔ It does **not** replace what (ii) reads; it sits beside it.

### 2 — Kill rule (iv) reads the **ORACLE** bound (**0.8373 ± 0.0109**), not the deployable (0.7998 ± 0.0443)
No valid-only statistic can select a refusal configuration for detection power, and a surrogate invalid set would
break the supervision-class match (iv) rests on. ⭐ **The rival gets its best configuration chosen with oracle access;
we do not. Harder for us, as it should be.**

### 3 — LACE whitening exactness **gates its configuration**
⭐ *"A rival weakened by our own selection procedure is our fault."* If the pre-registered selection yields a
non-exact whitening on a seed, that seed **falls back to the nearest configuration satisfying exactness, or runs the
general form with the prior score included** — both reported, **the stronger quoted**.
⛔ **Never-weaken-a-competitor applies to our selection code as much as to library defaults.**

### 4 — ⭐ C results are ALWAYS reported BY KIND; pooled is a companion only
Neither rival covers both kinds (kNN: oor **0.8045** / dup **0.3907**; LACE: oor **0.3971** / dup **0.9991**); ours
does. **That differentiator is invisible pooled.** ⛔ **Not quotable until the rivals are re-scored under A7.**

## ⚠ HUB TRAP FOUND WHILE SCOPING `CLU-wide` — the easiest way to void the structural claim
`chlu/core/gen_store.py::soft_read(store, X, tau)` computes `sum_m pi_m · mode_m` — **exactly the composed per-well
mode vector R27.1 asks for** — but it takes **`X`, the FEATURES**, and derives `pi` from them. ⛔ **Using it as-is is
an encoder → decoder path and voids the no-bypass guarantee** (`read_from_codes`'s whole point is that it takes
*no* features). ⭐ **`CLU-wide` must derive its mixture weights from the REQUEST** — `address_requests`' per-well mode
log-scores, softmaxed — never from features. **Assert the arm touches no feature tensor.**

---

## R28 — Wave 5 merged (Hub, 2026-09-13). `main @ 5fee3f4`. ⚠ Full suite deferred to wave close (new policy).
**New verification policy:** spokes already run the full suite on their own branch (their acceptance item). The Hub's
unique value is the **merged** state, so: **targeted tests per merge** (117 passed in 42 s across the touched
modules) and **ONE full suite per wave, on the final merged tip, when the machine is quiet.** ⛔ A full suite per
merge was competing with spokes for memory on a 16 GB machine and is very likely what tipped the DCI OOM.

### ⭐ `gen-metric-invariance` — **DCI-C PASSES. R1 UNBLOCKS.**
| axis | role | verdict |
|---|---|---|
| `leak_raw` (raw BA transfer leakage) | ⭐ **negative control, scored FIRST** | **FAIL** ✅ the check has teeth |
| `leak_cc` (chance-corrected, the Advisor's companion) | companion | **FAIL** |
| **`dci_c`** | **the candidate (R26.2)** | ⭐ **PASS** |

DCI-C: relabelling invariance **1.28e-16**, class-count stress **0.367 / 0.019**, **6/6** configuration signs agree,
magnitude ratio **1.34** inside the pre-registered [1/3, 3] band. ⭐ **The broken axis was rejected by the identical
code path before the candidate was scored** — the check was shown capable of failing something first.

**Three carried findings:**
- ⚠ **Class-count matching does NOT repair the leakage axis** — the sign inversion **survives** it (3/6).
  ⛔ `gen-r0-wide-refit` §4.2's "6-vs-15-class" account is *a* cause and demonstrably **not the whole cause.**
  ⭐ We therefore qualify DCI-C on the **invariance criterion**, not on having understood the failure.
- ⛔ **The Advisor's chance-corrected companion is WORSE, not better** (P7 falsified in direction) — leg 2 0.657/1.219,
  3/6 signs, magnitude ratio 11.37. **The offered route back to a transfer axis is closed.**
- ⚠ **Sign-convention clash:** this report's tables are `corr − rand`, the **opposite** of R22's. Bookkeeping, but it
  will produce a wrong reading if quoted across reports. **Owner: Hub, at the ladder prereg.**

### `gen-read-compose` — composition IS the store's, and the capacity limit is now named
- ⭐ **Falsifier did NOT fire:** `read − store_removed = +0.1759 ± 0.0107`, 3/3 seeds, **16.4×** spread.
- ⛔ **`CLU-wide` does NOT tie `knn_proto`**: **−0.1400 ± 0.0307, 0/3 seeds.** R27.1's pre-registered reading is
  **rejected**; the like-for-like channel does **not** close the gap.
- ⭐⭐ **THE DIAGNOSIS: the 72 held-out requests collapse to 9.0 ± 0.8 distinct code configurations** (seen: 495/512).
  `clu_wide` and `proto` render 72/72 distinct. ⭐ **The store gives nine answers to seventy-two questions** — the
  capacity statement in its sharpest form, and **invisible in every accuracy number.**
- ⚠ **"0.86 exact-set" (R26) needs a unit before it is quoted anywhere:** request-vs-feature code agreement is
  **0.871 per WELL** but **0.510 per full K-tuple** (`0.871^5 ≈ 0.50`). ⛔ If the original meant per-well, the stated
  justification for the read path is materially weaker. **Needs Head/Advisor.**

### `gen-c-set-a7` / `gen-launcher-fixes`
A7 + the **disk-discovered** no-model detector registry shipped (24 tests). Certificate stage added with its own
workspace path, every pre-existing guard still firing. ⛔ **Every cluster item still NOT RUN** — the smoke has never run.

---

## R29 — ADVISOR REPLY + ⭐ HARD PIVOT (Head ruled, 2026-09-13). Binding. `main @ 5fee3f4`.

### Q1 — "0.86" is PER FULL SET
Whole subset recovered, precondition *"handed correct addresses"*. ⭐ **Our comparable is 0.510 per K-tuple, NOT
0.871 per well. The gap is ADDRESSING.** ⛔ Verify the unit against the artifact row before quoting anywhere, and
⛔ never mix units in one comparison.

### Q2 — ⛔ COMPOSITION IS RELEASED AS A CLAIM
**Kill rule (ii) fired as pre-registered** (`CLU-wide − knn_proto = −0.1400`, 0/3 seeds).
⭐ **Report 9-of-72 as a measured COMMITMENT collapse, NOT a capacity limit** — 30 bits ≫ 72 codes, so capacity was
never the binding constraint. **Disk-only rider: do the 9 codes coincide with SEEN configurations? One line in the
write-up either way.**

### Q3 — ladder confirmed
Wide config · R0 re-fitted there · grouping re-derived per seed · **DCI-C axis, MIG companion** ·
⭐ **pin the sign convention in the prereg: `corr − rand`, positive = good** · qualify DCI-C **on the invariance
criterion alone**.

### Q4 — the escalation has ALREADY FIRED IN PART
Pre-registered now: **a surviving rival's margin sets the refusal comparison, not the naive 0.5630.** If the barrier
fails to clear it, **refusal tier 1 becomes a companion.**
⭐ **kNN 0.8045 ± 0.1227 TIES the certificate's 0.797 on `out_of_range`. Report that half as a TIE — never a win,
never a loss.**
⭐ **The refusal claim is: ONE dynamics-derived criterion covers BOTH kinds with ZERO invalid supervision, where each
rival covers one.** ⛔ Nothing quotable until the `duplicate_factor` spoke lands.
- **LACE:** run the pre-registered fallback on the 2 non-exact seeds **before any LACE number enters a table**;
  0.4694 is a **provisional lower bound**. Both forms reported, stronger quoted.
- **Composition re-run trigger: discharge BY PROOF in the table row** (prototype store size = factors × values ×
  width, already under budget ⇒ re-run returns the same number). **Moot once composition is released.**

## ⭐⭐ THE PIVOT — phase 1 is a MECHANISM STUDY; the wave as scoped produces no compositional-generation result
**New order. ⛔ The ladder must not consume the wave.**
1. ⭐ **PRIMARY — the commitment collapse.** Pre-register: **distinct codes per 72 requests vs stopping rule /
   multiplicity / launch temperature.** **PASS ≈ ≥60/72 distinct AND the store→decoder arm within spread of
   `knn_proto`.** ⛔ **Fail ⇒ new approaches (particle traversal etc.), NOT abandonment.**
2. ⭐ **IN PARALLEL — the generation harness end to end:** request → launch read → codes → decoder → image →
   **external classifier** on held-out combinations, vs **LACE · energy-based composition · `knn_proto`**.
   ⭐ **This is the headline table.**
3. **Encoder unfreeze with the store in the loop enters the plan NOW** (bypass guarantee kept). **Cluster; needs the
   Head's 2FA step.**
4. **Navigation** (well-to-well traversal, classifier along the path) queued directly after 1–2.
5. **Ladder R1–R5 = mechanism appendix, background.**

### Paper honesty lines (binding on every draft)
- ⛔ **Store load-bearing is BY CONSTRUCTION ⇒ method section, NOT headline.**
- ⛔ **Deletion differentiates only through the DYNAMICS** — `knn_proto` can drop a mean too.

**⛔ Process:** bring the Head a **worklist for 1–3 (spoke · cost · falsifier · artifact path) BEFORE launching.
The Head launches.**

---

## R30 — Wave 6 scoped under the R29 pivot (Hub, 2026-09-13). `main @ 5fee3f4`. Head approved all items.

**Head directions taken this session:** (1) launch `gen-arms-a7-wiring`; (2) hand over the CSF3 smoke block;
(3) ⭐ **full suite waived for this wave in the interest of time** — targeted tests per merge only, and the standing
suite debt at `5fee3f4` is carried, not discharged; (4) proceed on pivot items 1–3; (5) run the 9-codes check.

### ⭐ THE 9-CODES RIDER (R29 Q2) — ANSWERED IN PART, AND IT REFRAMES THE QUESTION
⛔ **The exact question cannot be answered from disk: `gen-read-compose` persisted the code-configuration COUNT, never
the code TUPLES.** The store `.pkl` holds weights only; recomputing the tuples is a run, not a read. Reported as
**NOT RUN**, with the cause named. ⭐ **Fix shipped into scope:** `gen-commitment-collapse` must persist the per-request
tuples for A, B and MS, so this is a spreadsheet forever after.

⭐ **What disk DOES answer, Hub-derived from `seed_*/compose_manifest.json` → `request_addressing.n_distinct_code_configs`:**

| set | what it is | seed 0 / 1 / 2 | distinct fraction |
|---|---|---|---|
| **A** | 512 seen requests, spread over the factor space | 494 / 493 / 498 of 512 | **≈ 0.97** |
| **MS** | ~198 matched-seen requests — **SEEN shapes, SAME colour corner** | 35 / 24 / 24 of ~198 | **≈ 0.14** |
| **B** | 72 held-out requests, same corner | 9 / 8 / 10 of 72 | **≈ 0.125** |

⭐⭐ **THE COLLAPSE IS A PROPERTY OF THE COLOUR CORNER, NOT OF HELD-OUT-NESS.** Seen data in the same corner collapses
at the same rate as held-out data (≈0.14 vs ≈0.125); data spread over the factor space does not collapse at all
(≈0.97). ⛔ **"Seen vs unseen" is therefore the wrong axis for the 9 codes**, and any write-up framing it that way is
wrong. **Binding on every draft.** ⭐ It also makes MS a **mandatory companion on every commitment number** — a knob
that lifts B and leaves MS flat has found a B-specific artefact, not a fix.
⚠ **Not claimed:** which codes they are. The tuple-overlap number is owed by `gen-commitment-collapse`.

### The worklist (Head-approved before launch, per R29's process rule)
| # | spoke | pivot item | cost | falsifier | artifact |
|---|---|---|---|---|---|
| 1 | `gen-arms-a7-wiring` | unblocks refusal | ~110 min | a rival that got **weaker** | `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` |
| 2 | ⭐ `gen-commitment-collapse` | **1 — PRIMARY** | ~4 h | distinct count rises, accuracy falls (= noise); or B moves and MS does not | `.claude/outputs/gen-commitment-collapse/` |
| 3 | `gen-external-classifier` | 2 (the instrument) | ~5 h | it agrees with the probe everywhere, incl. where the probe is saturated ⇒ adds nothing | `.claude/outputs/gen-external-classifier/classifier_gate.json` |
| 4 | `gen-cell-harness` | 2 (headline table) | ~6 h | an arm moves and you cannot name whether decoder, classifier or arm moved it | `.claude/outputs/gen-cell-harness/` |
| 5 | `gen-encoder-unfreeze` | 3 | ~5 h local | the bypass test passes on a deliberately broken variant | `.claude/outputs/gen-encoder-unfreeze/` |

**Sequencing.** 1–3 launch together (3 code worktrees, at the charter cap). 4 is gated on **both** `classifier_gate.json`
= `PASS` **and** `refusal_scores.json` = `complete`, **plus** a content check that no seed's quoted LACE configuration is
still the non-exact whitening (R27.3). 5 is gated on `.claude/outputs/gen-cluster-smoke/smoke_result.json` = `ok`,
which only the Head can produce. ⛔ Ladder R1–R5 stays appendix background and does not consume this wave (R29).

### Hub housekeeping
- ⭐ **Three fully-merged worktrees removed** — `CHLU-gen-a7`, `CHLU-gen-inv`, `CHLU-gen-launchfix` — each verified from
  the main repo first: **0 commits unmerged into `main`, 0 uncommitted files.** 6 worktrees remain, 3 parked from
  earlier campaigns. This is what freed the capacity for items 2–3.
- **`gen-arms-a7-wiring` acceptance item 1 amended before spawn:** its per-request refusal scores now have a **pinned
  path** with a `"status": "complete"` key written last. ⛔ It previously said *"path in the report"*, which is a prose
  gate — `gen-cell-harness` could not have been gated on it mechanically.
- **`.claude/tasks/gen-cell-run.md` marked SUPERSEDED** (pre-pivot: four-arm cell, conditional dreaming, kill-rule
  verdicts). Replaced by `gen-cell-harness.md`. ⛔ Not to be spawned.
- ⚠ **`chlu/config.py` is shared by three concurrent spokes this wave.** `gen-arms-a7-wiring` edits the **existing**
  `GenBaselinesConfig` block; the other two **append one new block at EOF**. ⛔ Merge order is a7 → collapse →
  classifier, and same-anchor conflicts are resolved by taking the peer's whole file and re-applying, asserting
  **0 deletions** — hunk-by-hunk "keep both" has broken this file's syntax twice.
- ⚠ **Suite debt carried:** `5fee3f4` has had targeted tests only, and the Head waived the wave-6 full suite. Last full
  green remains **2166 passed @ `a0034d4`**. ⛔ Do not describe any later tip as fully verified.

---

## R31 — Wave 6 merged and reviewed (Hub, 2026-09-13). `main @ be27713`. Targeted tests only (Head waived the suite).

Three branches merged `--no-ff`. ⚠ **One conflict, the predicted one:** `chlu/config.py`, both tail-append blocks at
the same anchor. Resolved by the whole-file rule — both blocks kept in merge order, **0 deletions against the prior
merged tip**, `ast.parse` clean. ⚠ The 11 lines that differ against the classifier branch are `gen-arms-a7-wiring`'s
`include_coupling_bytes` fix (`True` → `False`, R26.3), which that branch predates. **Verified `False` on the tip.**
**Targeted tests on the MERGED tip: 196 passed, 1 skipped, 0 failed (2:32).** ⛔ **Full-suite debt still stands at
`a0034d4`/2166** — `5fee3f4` and `be27713` have had targeted tests only.

### ⛔⛔ HUB ERRATUM — R30's stated MECHANISM was WRONG. The CONCLUSION survives.
R30 (and the `gen-commitment-collapse` task file) asserted that Montero's corner makes the three hue factors
**constant** across the 72 held-out requests. **They are not constant** — each hue takes **2** values in the corner
(`2·2·2 × scale 3 × shape 1 × orientation 3 = 72`). ⛔ **Never repeat the constant-hue account.**
⭐ **What survives untouched, and is now stronger:** *"the corner collapses, not the hold-out."* It is no longer an
inference from two rates — it is a **set identity**, Hub-re-derived from the persisted tuples:
**|B ∩ A| = 0 of 9 / 0 of 8 / 0 of 10** and **|B ∩ MS| = 9 of 9 / 7 of 8 / 10 of 10**, and **|MS ∩ A| = 0** on all
three seeds. ⇒ **every code a held-out request receives is also given to a SEEN request in the same corner, and none
of them is used anywhere else in the factor space.** R29 Q2's rider is **DISCHARGED**.

### ⛔⛔ THE SPOKE'S OWN MECHANISM LINE ALSO NEEDS CORRECTING — `shape` flip = 0.00 on B is VACUOUS
The report's reconciliation item 1 quotes B's `code_flip_rate` row (*"scale 0.20 · shape 0.00 · orientation 0.13"*) to
argue the code is *"≈ the hue triple and almost nothing else."* ⛔ **The `shape` cell is an artefact.**
**Hub-verified from `commit_manifest_s*.json` → `factor_leverage`:** on B, `shape` has
**`n_values_realised = 1`** (Montero's cell *is* `shape == cylinder`), and `factor_leverage` returns
`float(np.mean(flips)) if flips else 0.0` — so an empty flip list reports **0.00**, meaning *"no other value exists
to flip to"*, **not** *"editing shape does not move the code."*
⚠ Worse, the function's own note reads *"flip ≈ 0 with a large evidence share ⇒ OUTVOTED"*, and `shape` carries a
**0.124** evidence share on B — so the vacuous cell reads exactly like a real finding.
⭐ **The control refutes it in the same table:** in the SAME corner with `shape` varying (MS, 3 values), the flip rate
is **0.85 / 0.86 / 0.82**. **The code tracks `shape` perfectly well.**

### ⭐ THE CORRECTED MECHANISM — Hub-derived, and it is arithmetic, not narrative
**The committed code is approximately `(hue triple × shape)`. `scale` and `orientation` are dropped inside the corner.**

| | flip rate on B | flip rate on MS | distinct codes | predicted by `(hue,shape)` |
|---|---|---|---|---|
| floor / wall / object hue | 0.98 / 1.00 / 0.56 | 0.99 / 1.00 / 0.63 | — | — |
| **`shape`** | ⛔ **vacuous (1 value)** | **0.84** | — | — |
| `scale` | **0.20** | **0.28** | — | — |
| `orientation` | **0.13** | **0.17** | — | — |
| **B** (shape constant) | | | **9 / 8 / 10** | **8** = the hue triples |
| **MS** (shape varies ×3) | | | **35 / 24 / 24** | **24** = 8 × 3 — ⭐ **exact on 2/3 seeds** |

⭐ **84 % of requests take the modal code of their `(hue, shape)` cell** (B 0.958/0.833/0.792; MS 0.798/0.849/0.789).
⭐ **This is consistent with R22's read margin on `shape` (+0.2008, 12× spread) instead of contradicting it** — which
the uncorrected reading would have done. ⛔ **Binding on every draft: the dropped factors are `scale` and
`orientation`, never `shape`.**
⭐ **Standing rule, fourth strike:** a per-factor statistic is **not reportable without its `n_values_realised`**. A
factor with one realised value produces a confident-looking zero. **Owner: Hub, at the next table review.**

### The three verdicts
1. ⛔ **`gen-commitment-collapse`: FAIL, pre-registered, 3/3 seeds, all four configs.** ⭐ **The right kind of fail.**
   Distinctness and correctness were shown **separable**: at **160.2 bits** (k = 8, 5.34× the design constant) the
   same store renders **70.3 ± 1.2** distinct B codes and still scores `scale` **0.000**. ⇒ ⛔ **The collapse is NOT a
   channel shortage** — 30 bits was never the binding constraint, which confirms R29 Q2's reading.
   ⭐ **`argmax` is provably temperature-blind** (softmax is monotone; measured bit-identical over τ ∈ [0.25, 8]).
   ⛔ **Any future task or draft listing "launch temperature" as a knob on the shipped read is listing a no-op.**
   ⭐ **No B-only knob exists:** over 66 configs, *r*(B distinct, MS distinct) = **0.9907**, 0 configs exceed a +0.10
   gap. Every knob moves the **corner**. Per R29 this routes to new approaches, ⛔ not to abandonment, ⛔ not to re-tuning.
2. ⭐ **`gen-external-classifier`: GATE PASSES 3/3 at 1.0000** on real held-out images (floor 0.95), and ⭐⭐ **the
   phase's ceiling stack SURVIVES its own adversarial check**: external **0.9357 ± 0.0193** vs probe
   **0.9414 ± 0.0179** on the *same* renders, mean Δ **−0.0057 ± 0.0116**, **sign flips across seeds**. ⛔ The
   ceilings are **not** an artefact of measuring the decoder with its own family. **`gen-cell-harness`'s gate is green.**
3. ⭐ **`gen-arms-a7-wiring`: falsifier did NOT fire — no rival got weaker.** `out_of_range` bit-identical, **worst
   |Δ| = 0.000e+00 over 30 numbers**. ⭐ **R27.4 is now QUOTABLE on a controlled metric:** LACE's `duplicate_factor`
   holds at **0.9936 ± 0.0057** under A7 (⇒ ⛔ it was **not** the term-count giveaway, it is mechanism) while kNN's
   falls to **0.3338 ± 0.0692**; kNN holds `out_of_range` at **0.8045** where LACE sits at 0.3971–0.5183.
   ⇒ **"neither rival covers both kinds, ours does" is evidenced, not asserted.**
   ⭐ **The Head's mandatory re-run trigger is a MEASURED no-op:** `knn_proto` in-corner balanced accuracy is
   **bit-identical at both budgets** (Δ = 0.000000, 3/3). ⭐ At the corrected budget the kNN arm gets **better**
   (`out_of_range` AUROC 0.8045 → **0.8281**) — the 31 % advantage was, if anything, hobbling it.
   ⭐ LACE byte-invariance **measured**, not argued (one re-run seed reproduces every number), so LACE never re-runs.

### ⛔ Carried forward — open, not resolved by this merge
- ⚠ **The A7-controlled number moves TWO things at once and must never be quoted bare.** A7 is built from **train**
  rows, so B → A7 changes the term count (6→7) **and** the seen-ness of the negative pool. The dedup set `A7D`
  separates them: for kNN the legs are **+0.2653** (seen/unseen) and **−0.3221** (term count) and they nearly cancel.
  ⛔ **Every quoted A7 number carries this decomposition.**
- ⚠ **`gen-baselines` §2a's `±` is SAMPLE sd (ddof=1); its flag-provenance table declares POPULATION sd (ddof=0).**
  Means bit-identical, spreads differ by √(3/2) (kNN oor **0.1002** vs the published 0.1227). **Not a re-run — a
  convention clash to fix once. Owner: Hub.**
- ⛔ **`dec_recipe` still defaults to `"baseline"`**, the recipe that FAILED the gate. Both new spokes asserted the
  hash explicitly and were correct. **Standing trap, unchanged.**
- **Cluster smoke still NOT RUN** — it gates `gen-encoder-unfreeze` and every pre-registered cell (R21, R23.6).

### ⚠ R31 addendum — a SILENT-SKIP trap the Hub hit on its own merge check (2026-09-13)
The merged-tip run reported **196 passed, 1 skipped**. The skip is
`tests/test_gen_classifier.py:251 — "Shapes3D not on disk"`, and ⛔ **it is an artefact of running pytest from the
`CHLU-merge` WORKTREE, not a defect and not a missing dataset.**
**Cause, read from code:** `chlu/data/gen_splits.py::default_gen_data_root` resolves to `<repo>/.claude/data/gen`;
`.claude/` is gitignored, so it exists **only in the primary checkout** — a worktree has none. The docstring says this
in as many words. The dataset is present (`/Users/user/Desktop/CHLU/.claude/data/gen/shapes3d/3dshapes.h5`, 268 MB)
and `h5py` imports fine (3.16.0), so neither R12's nor §7's usual suspects apply.
⭐ **The fix is one environment variable, and it is now mandatory for every worktree test run:**
`CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`
⛔ **Why this matters more than one skip:** the skipped test is the **render-leak bit-identity assertion** — the check
that makes *"the classifier never saw a decoder render"* a measurement rather than a claim. ⭐ **Every spoke this wave
ran its suite from a worktree.** Any real-data test in this family has been skipping for them too, and a skip is
invisible in a green line. ⛔ **A spoke reporting "N passed" from a worktree has NOT exercised the real-data path.**
⭐ **STANDING RULE, fifth strike:** ⛔ **a green suite is not evidence until its SKIP COUNT is read and every skip is
named.** Report `passed / skipped / failed`, never `passed` alone. **Owner: Hub, folded into the task template.**

**⭐ R31 addendum, CLOSED — the honest merged-tip number.** Re-run with `CHLU_GEN_DATA_ROOT` set so the real-data path
actually executes: **221 passed, 0 skipped, 0 failed (2:27)** on `main @ be27713`, across
`test_config · test_gen_baselines · test_gen_compose · test_gen_classifier · test_gen_certificate · test_gen_store ·
test_gen_data_splits`. ⛔ **Supersedes R31's "196 passed, 1 skipped"** — that run was from the worktree with the
real-data tests silently skipping, and the render-leak bit-identity assertion **did not execute in it**. It executes
now and passes. ⚠ **Full-suite debt is unaffected and still stands at `a0034d4` / 2166.**

---

## R32 — ⛔ THE CLUSTER SMOKE FAILED TO SUBMIT. Zero jobs, zero job ids. (Head ran it, 2026-09-13)

**What happened:** `PLAN=1` printed correctly. **Both `sbatch` calls were REJECTED by the scheduler:**
`sbatch: error: You can only request 1 core in this partition. / Batch job submission failed: CPU count
specification invalid`. `squeue --me` empty; no `logs/clu-gen-*.out` exist.
⛔ **Nothing ran. This is NOT a smoke result — the gate is still shut.**

### The cause — a defect in the launcher's OWN documented submit block, Hub-verified from code
`scripts/csf3/job_gpu_gen.sh` hardcodes, at lines 73–76: **`-p gpuA · -G 1 · -n 1 · -c 8`**.
`gen-launcher-fixes.md` §3.2 then prescribes submitting that script with **`-p serial`**. ⛔ **CSF3's `serial`
partition is 1 core and takes no `-n`/`-c`/`-G` at all** — and **the repo already says so**, in
`scripts/csf3/job_cpu_smoke.sh` line 11: *"1-core Intel CPU partition (no -n/-c/-G: serial=1 core)"*.
⇒ The prescribed block could **never** have worked. ⭐ **It was never caught because the smoke had never been run** —
the exact failure mode the standing "a check that cannot fail proves nothing" rule exists to prevent, arriving from
the other direction: **a check that was never executed proved nothing either.**
⚠ `-c` can be overridden on the command line, but `-G 1` cannot be un-requested on a GPU-less partition. ⛔ **Do not
patch this with `-c 1`; it will fail again on the GPU request.**

### RULED (Hub): run the smoke on `gpuA`, the partition the real runs use
⭐ Dropping `-p serial` is not a workaround, it is the **more faithful** smoke: a plumbing test on a partition we will
never use does not validate the submit path we will actually use. The payload is trivial (`SMOKE=1` writes one JSON),
so the A100 cost is seconds of wallclock plus queue time.
⭐ **And it buys a second check for free: the JAX GPU preflight.** Submitted as a third job on its own seed (so it gets
its own run directory and overwrites nothing), it answers *"does JAX initialise the A100 in this environment?"* —
⛔ which is worth far more before a 4-day run than after one. **A preflight failure is a real finding, not a bug.**

### Owed
- **Launcher defect** — `gen-launcher-fixes.md` §3.2's submit block is **wrong as written** and must be corrected at
  source, not just in the Head's terminal. ⛔ Not a mid-gate code change; **scoped to the next engineer spoke.**
- **The gate is unchanged and still shut:** R21 / R23.6 — no pre-registered cell runs before a green smoke, and
  `gen-encoder-unfreeze` stays blocked on `.claude/outputs/gen-cluster-smoke/smoke_result.json`.

### ⚠ R32 addendum — a SHELL trap that manufactures a false "these differ" (Hub, 2026-09-13)
Checking whether wave 6 changed the launcher, the Hub ran a per-commit hash **inside a `for` loop**:
`for c in A B; do git show "$c:path" | shasum -a 256; done` → returned **two different hashes**, deterministically,
on **both** runs. ⛔ **Both were wrong.** The identical command run **outside** a loop returns one stable hash
(`816387ed…`) for both commits, corroborated by `git diff` (empty), `git log -- scripts/csf3/` (empty), equal byte
sizes (21,823) and a clean `diff` of the two extracted files.
⇒ **In this zsh, a `git show | shasum` pipeline inside a `for` loop corrupts what reaches the hasher.**
⛔ **Never verify file identity with a hash computed inside a loop here.** Extract to files and `diff`, or hash
outside a loop. ⭐ This came within one sentence of telling the Head the cluster launcher had changed when it had not —
the false positive, not the false negative, was the dangerous direction.
**Established: `scripts/csf3/job_gpu_gen.sh` is BYTE-IDENTICAL at `5fee3f4` and `be27713`** — wave 6 touched no file
under `scripts/csf3/` (all three spokes were forbidden it).

### ⭐ R32 update — THE SUBMIT PATH WORKS. Jobs queued 2026-09-13 20:18. (Head)
Dropping `-p serial` fixed it. **Three jobs accepted, 5 array tasks, partition `gpuA`, 8 CPUs, 20 min limit:**

| job id | name | tasks | writes to |
|---|---|---|---|
| **20325640** | `clu-gen-smoke` | `[0]` | `encoder/clu_s0` — plumbing, preflight skipped |
| **20325641** | `clu-gen-cert-smoke` | `[0-2%2]` | `certificate/clu_s{0,1,2}` — the new stage's own path |
| **20325642** | `clu-gen-gpuprobe` | `[0]` | `encoder/clu_s1` — ⭐ JAX **preflight ON** |

All **PD (Priority)**; Slurm's start estimates were 22:22 / 22:45, i.e. ~2 h of queue. ⛔ **Queued is not green.
This is still NOT a smoke result and the gate is still shut** — R21 / R23.6 unchanged.
⭐ **The two verdicts are SEPARATE and must not be conflated:** `20325640` + `20325641` decide the **plumbing gate**;
`20325642` decides whether **JAX initialises the A100**. ⛔ A gpuprobe failure does **not** fail the plumbing gate —
it is a finding that blocks *real* GPU runs and nothing else, which is exactly why it was submitted on its own seed.

---

## R33 — Wave 7 scoped while the cluster queue runs (Hub, 2026-09-13). `main @ be27713`.

**Launchable now: 2 spokes.** 3 code-worktree slots are free (all gen worktrees removed at the wave-6 close).

| spoke | pivot item | cost | falsifier | state |
|---|---|---|---|---|
| `gen-cell-harness` | **2 — the headline table** | ~6 h | an arm moves and you cannot name whether decoder, classifier or arm moved it | ⭐ **gates now GREEN** |
| `gen-navigation` | **4, and the pre-registered response to item 1's FAIL** | ~5 h | linear interpolation between endpoint reads traverses as well ⇒ the store adds nothing | ⭐ **ready** |

⛔ **`gen-encoder-unfreeze` stays BLOCKED** on `.claude/outputs/gen-cluster-smoke/smoke_result.json` — jobs 20325640 /
20325641 are still queued. ⛔ **Ladder R1–R5 stays appendix background** and does not consume this wave (R29).

### ⛔ HUB ERRATUM — `gen-cell-harness`'s LACE precondition was WRONG as written, corrected before spawn
It said: *"if any seed's quoted LACE configuration is still the non-exact whitening, stop."* ⛔ **That would have
blocked on a compliant — and pro-rival — case.** R27.3's rule is *both legs run, **the stronger quoted***, and on
**composition**, the axis the harness measures, the **non-exact** leg is the stronger on both gated seeds. ⭐ Quoting
it is therefore **correct and required**, and LACE's composition stays as published (`ode` 0.6797 ± 0.0207).
**Corrected to:** verify both legs ran on seeds 0 and 2 and that the stronger is quoted per axis.
⭐ **The general lesson:** never-weaken-a-competitor is a rule about the **number**, ⛔ never about the configuration
label. A precondition phrased over the label inverts it.
⚠ **Residue, carried unresolved:** R27.3's second route — the general form **with** the prior score — is **NOT RUN**
(it needs a new drift term in `sample_ode`, a sampler change). ⛔ Not to be attempted inside another spoke.

### ⭐ Why `gen-navigation` is the right second spoke, and why it is unblocked *now*
The collapse fail leaves one question open: the commitment lands on **9 points** — ⭐ **is the space BETWEEN them
structured?** Smooth, semantically ordered traversal would be a mechanism result that **survives** the commitment
failure; its absence bounds the store and we say so. R29 names both routes and they converge here:
item 1's fail route (*"particle traversal etc."*) and item 4 (*"well-to-well traversal, classifier along the path"*).
⭐ **It became launchable exactly when `gen-external-classifier` landed** — "classifier along the path" needs an
instrument that is not the decoder's own family, and until this wave we did not have one.
⭐ **Its pre-registered endpoint rule must include `scale` and `orientation` pairs** — R31 established those are the
factors the code **drops** inside the corner, so they are where traversal is most likely to fail. ⛔ Testing only the
hue axes would be scoring our own exam.

### ⛔ HELD, deliberately: the launcher fix
`gen-launcher-fixes.md` §3.2's submit block is wrong at source (R32) and wants fixing. ⛔ **Not now.** Three cluster
jobs are in flight against the pushed tree, and if the smoke surfaces further launcher defects we want **one** spoke
fixing all of them, not two passes. **Scheduled: the first engineer slot after the smoke reports.**

### Hub-owned, no spoke needed
- The `ddof` convention clash (R31 carry-forward) — Hub, at the next table review.
- `n_values_realised` beside every per-factor statistic, and `CHLU_GEN_DATA_ROOT` + *"report passed/skipped/failed"*
  — ⭐ **both already folded into `gen-navigation.md`**; to be folded into the standing task template.

### R33 addendum — the third spoke: `gen-bridge-dsprites` (Hub, 2026-09-13)
**Head asked for a third local spoke. Scoped: the dSprites decoder-ceiling gate.** Fills the worktree cap at 3.

⭐ **Why this one.** Every number this programme has comes from **one dataset**, and *"does it work on anything else?"*
is the first question a referee asks. ⭐ **dSprites is already fully wired and has never been run** — Hub-verified on
disk: the loader (`chlu/data/dsprites.py`), the `dsprites` axis in `gen_bridge.py` and **every** `exp_gen_*` module,
and the staged data (`dsprites_…64x64.hdf5` + `.npz`, `c_invalid.json`, `a7_redundant.json`, `query_sets.json`,
`factor_index.npz`). ⇒ the marginal cost is a **run**, not a build. It is a **gate**, so a negative is as useful as a
positive — and it is the cheapest honest answer to the generalisation question.

### ⚠ A Hub inference CAUGHT BEFORE IT WAS WRITTEN DOWN — dSprites does NOT escape the one-class problem
From R11's arithmetic (`1·4·13·11·11 = 6,292`) the Hub inferred the `1` was the **colour** factor, i.e. that `shape`
varies on dSprites' held-out cell and the dataset would therefore give a **cleaner metric** than Shapes3D. ⛔ **Wrong,
and it was about to be offered to the Head as the main selling point.** The pinned filter, read from
`montero_splits.json`, begins **`latent_classes[:, shp] == 1`** — and colour was **already dropped at load** (5
columns, not 6). ⇒ **the `1` is `shape`.** dSprites pins shape to ONE value exactly as Montero's Shapes3D cell pins
`shape == cylinder`.
⭐ **Consequence, and it is the good one:** R20's in-corner discrimination metric carries over **unchanged**
(positives = held-out `shp == 1`; negatives = matched-seen other shapes in the same corner; balanced accuracy the
gate, AUROC the companion). ⛔ **No new metric may be invented for this dataset** — reusing the ruled one is precisely
what makes the two datasets comparable. **The value of dSprites is generalisation, not a friendlier metric.**
⭐ **Standing lesson, sixth strike:** ⛔ **never infer a split's structure from a factor-count product.** Read the
pinned filter. The product `1·4·13·11·11` is consistent with at least two different held-out designs.

### ⚠ Cost, stated plainly
~8 h, the longest in the wave — dSprites train is **730,988** rows vs Shapes3D's 479,928. **Per-seed incremental
writes and skip-if-exists are mandatory**, and fewer-seeds-fully-reported beats a partial number. It may run overnight.

---

## R34 — `gen-navigation` merged and reviewed (Hub, 2026-09-13). `main @ 275fe18`.

### ⛔ THE DIAL'S ANSWER IS NO — `NAV = 0.0000 ± 0.0000`, 3/3 seeds, Hub-verified from `summary.json`
The spoke's own named falsifier **FIRED**. `NAV` = fraction of qualifying pairs where the store path traverses and
**neither control does** — zero, on every seed, both pools. ⭐ **A statistic about EXISTENCE, not level**, which is why
the channel caveat below does not undercut it. Per R29 a clean negative routes to the Head, ⛔ not to re-tuning.
**Instrument provenance asserted:** decoder `v1_capacity_schedule` hash `01ef91f75eea…`; external-classifier gate
`PASS`, param hash checked against the manifest. **3 seeds run, `seeds_not_run = []`.**

### ⭐⭐ THE REAL FINDING — the binding clause is ENDPOINT FIDELITY, not geometry
`endpoints_read_ok` is **exactly 0.000 on all three seeds in the corner** and **0.517 on the grid**. ⛔ **You cannot
measure navigation between two points the read cannot represent.** ⇒ **this is not "the landscape is not navigable";
it is "the landscape was largely untestable through a 30-bit read that loses the requested factors."**
⛔ **Binding on every draft: do not write the navigation negative as a geometry result.**

### ⭐⭐ R31's MECHANISM REPLICATES ON AN INDEPENDENT INSTRUMENT — and it is GLOBAL, not corner-local
The **external classifier** (⛔ not the linear probe, ⛔ a different harness) reads endpoint fidelity in the corner as
`scale` **0.021** / `orientation` **0.158**, reproducing `gen-commitment-collapse`'s probe numbers (0.019 / 0.194) to
within noise. ⭐ **NEW:** on the *spread* grid it is `scale` **0.229** (chance 0.125) and `orientation` **0.236**
(chance 0.067) — barely above chance. ⇒ ⭐ **the read drops `scale` and `orientation` EVERYWHERE, not only inside
Montero's corner.** R31 was about **distinctness** and was corner-local; this is about **fidelity** and is global.
⭐⭐ **One mechanism, now confirmed three ways — two instruments, two harnesses, two request pools:
the 30-bit read carries HUE and SHAPE and discards SCALE and ORIENTATION.** ⭐ **That is a measured loss profile of a
lossy channel, not a malfunction, and it is the sharpest mechanism statement the phase has produced.**

### ⚠ THE COMPARISON IS NOT CHANNEL-MATCHED — Hub-found, missing from the spoke's reconciliation list
From `summary.json → channel_by_arm`: the `linear` control carries **65,536 bits per interior point** (a float32
D=2048 vector, `exceeds_design_constant = True`); `store_snap` carries **30**. ⇒ ⭐ **the winning control has a
2,185× wider per-point channel.** ⛔ **This must travel with the sentence "interpolation does it at least as well"** —
without it the comparison reads far more damningly than it is.
⭐ **Precedent governs the reading, and we follow it: R27.1.** The pre-registered comparison is **primary**; the
channel-matched observation is a **companion, never a rescue**. Reading the kinder number after seeing the control's
result is exactly the amendment R27.1 refused. ⛔ **`NAV = 0` stands as the headline.**

### ⭐ The spoke argued against itself, correctly, and that is why the result is trustworthy
On the **most generous reading available to the store** (GRID, stability clause dropped, conditioned on endpoints
reading correctly) the control **still wins**: `visits_intermediate` **0.550 ± 0.147** vs **0.712 ± 0.134**,
Δ **−0.162 ± 0.027**, sign-consistent 3/3. The one axis where the store leads is `off_target_constancy` **+0.028**,
an order of magnitude below its own pre-registration. ⭐ It labelled the conditioning **post-hoc**, flagged that it
selects on an arm's own outcome, and noted both arms share the same endpoint rate so the comparison is fair.

### ⛔ CARRIED — controls that must travel with any future "smooth latent interpolation" claim
1. ⭐ **`visits_intermediate` is NOT evidence on its own.** The **`scramble`-deleted** store — content destroyed —
   visits an intermediate factor value on **0.483 ± 0.012** of GRID pairs and is weakly monotone on 0.350.
   ⛔ Any smoothness claim without this control is unsupported.
2. ⚠ **`deleted_mean`'s `off_target_constancy = 1.000` is ARITHMETIC** (a constant image is perfectly constant).
   ⛔ Never read as the deletion control "doing well".
3. ⚠ **CORNER cells in the conditional table are UNDEFINED, not zero** — 0 pairs have readable endpoints.
4. ⚠ **The store path is a STAIRCASE:** 5.92 of 8 steps exactly zero-length, `jump_ratio` **5.63 ± 0.17** vs the
   control's **1.55 ± 0.03**; only 3.08 of 9 points carry a distinct code. ⭐ Yet it is **more direct** in image space
   (`path_efficiency` **0.841** vs 0.747). ⛔ Quoting either fact alone is misleading.
5. ⚠ **`GRID/orientation`'s single traversing pair** (1 in 24, one seed) ⛔ is not quotable — the control traversed
   the same pair.

### ⭐ The R31 rule earned its keep on first use
`n_values_realised` is carried on **every** per-factor row, and it is what let the spoke mark **`CORNER/shape` as
`NOT APPLICABLE / NOT RUN`** (1 realised value) instead of emitting the confident-looking zero that produced R31's
erratum. ⭐ **The rule caught the exact defect it was written for, one spoke later.**

### ⚠ Process note — the pre-registration mtime trail was OVERWRITTEN by its own addendum
The report claims the filing order is *"mtime-verifiable"* at 21:30:27 / 21:33 / 21:34. ⛔ **It is not:** `PREREG.md`'s
mtime on disk is **21:37:15**, because Addendum 1 was written **into the same file**, destroying the original
timestamp. ⭐ **The ordering claim nevertheless HOLDS on the conservative reading** — Hub-verified: every production
artefact postdates 21:37:15 (`seed_0` 21:38:26 → `seed_2` 21:39:46 → tables 21:44), so the prereg precedes every
number it governs. ⚠ And prereg integrity is **least** load-bearing here: this is a **negative** that fires the
spoke's own falsifier, the direction nobody fakes.
⭐ **STANDING RULE, seventh strike: an addendum is written as a NEW FILE (`PREREG-addendum-N.md`), ⛔ never appended
in place** — an in-place addendum destroys the very evidence the pre-registration rule exists to create.
**Owner: Hub, into the task template.**

**R34 test line.** Targeted tests on the merged tip `275fe18`, run **with `CHLU_GEN_DATA_ROOT` set** per the new rule:
**102 passed, 0 skipped, 0 failed (36 s)** across `test_gen_navigate · test_gen_compose · test_config ·
test_gen_classifier`. ⭐ **Zero skips — the data-root rule works and the real-data path executed.**
Worktree `CHLU-gen-nav` removed after verifying 0 unmerged commits and 0 uncommitted files (8 worktrees remain:
`CHLU-gen-harness` and `CHLU-gen-dsprites` live, the rest parked).
⚠ Full-suite debt unchanged: still `a0034d4` / 2166.

⚠ **Hub process note, same session:** the first attempt to write this entry ran with cwd inside `CHLU-merge` and the
heredoc **failed silently** (`no such file or directory`) while the trailing `echo` still printed "ledger-updated".
⛔ **`.claude/` exists ONLY in the primary checkout** — the same fact behind R31's silent-skip trap, biting a second
time in a different way. ⭐ **Rule: every `.claude/` write is issued from `/Users/user/Desktop/CHLU`, never from a
worktree, and a success echo must not be chained to a command that can fail independently.**

---

## R35 — SMOKE ROUND 1: 2 of 3 green, and ⛔ THE CLUSTER IS RUNNING A STALE TREE (Hub, 2026-09-13)

| job | result | verdict |
|---|---|---|
| **20325640** `clu-gen-smoke` (encoder) | `COMPLETED 0:0`, manifest **ok**, artifact present | ✅ path plumbing works |
| **20325641** `clu-gen-cert-smoke` ×3 | `FAILED 2:0`, **NO MANIFEST**, no artifact | ⛔ **the stage does not exist on the cluster** |
| **20325642** `clu-gen-gpuprobe` | `COMPLETED 0:0`, preflight **ON** | ⭐ **JAX initialises the A100** |

### ⛔ THE CAUSE — Hub-verified locally from git, no cluster access needed
Every manifest records **`git_commit = 3faef27`**. At `3faef27`:
- `job_gpu_gen.sh` contains **zero** occurrences of `certificate`; its guard reads
  `STAGE=encoder|decoder|store|cell` (lines 83, 96). ⇒ `STAGE=certificate` **dies at §1 validation with `exit 2`,
  before the `on_exit` trap is installed** — which is exactly why there is **no manifest**, not even a `failed` one.
- The certificate stage arrives at **`8940f1c`**, merged by **`54219a0`** — ⭐ **both AFTER `3faef27`.**
⇒ ⛔ **This is not a launcher defect and not an `sbatch --export` mangling. The pushed tree simply predates the
feature.** (The competing hypothesis — `SEEDS="0 1 2"` losing its spaces through `--export`, which would trip the
`SLURM_ARRAY_TASK_COUNT` guard at the same `exit 2` with no manifest — is **ruled out** by the absent stage.)

### ⚠ CONSEQUENCE: the GREEN result is also stale and does not discharge the gate
`20325640` validated the **`3faef27`** launcher, ⛔ **not the one we merged.** Everything `gen-launcher-fixes`
shipped — the `certificate` stage and its separate workspace path, the loud-failing `gen_data.data_root`, the new
guards — is **still entirely untested on the cluster.** ⛔ **R21 / R23.6 remain unsatisfied. The gate stays SHUT and
`.claude/outputs/gen-cluster-smoke/smoke_result.json` must NOT be written.**
⭐ **Note the manifest recorded the *exported env var*, not the rsync'd tree** — they happened to agree here, and the
certificate failure is what proved it. ⛔ `GEN_GIT_COMMIT` is a **claim** about provenance, not a measurement of it.

### ⭐ WHAT IS GENUINELY BANKED — the GPU preflight, and it transfers across tree versions
The preflight **is present at `3faef27`** (`assert jax.default_backend() == "gpu"`), the probe ran with it **enabled**,
and it exited **0**. ⇒ ⭐ **JAX initialises the A100 in this environment — first confirmation the programme has ever
had.** That is a statement about the **environment**, so it survives the re-push and does not need re-running.
⚠ **Hub error in the check command, recorded:** the Hub grepped `logs/stage.log` for the preflight output. §8 writes to
the **slurm `.out`**, because `stage.log` only captures the tee'd payload (§9). The grep printed nothing and that was
the Hub's bug, ⛔ not a missing preflight. **Rule: `stage.log` holds the PAYLOAD only; everything the launcher does
around it lands in the slurm streams.**

### THE FIX — re-push, re-export, re-run all three
⛔ **Re-run the encoder smoke too**, not just the certificate: the passing one tested the wrong launcher.
⭐ **Lesson for the ledger:** *"the smoke has never run"* was true for six days; when it finally ran it tested a tree
six days old and **reported green**. ⛔ **A green result whose provenance is unverified is not evidence.** The
certificate failure is the only reason we found out — ⭐ **the stage that failed is what audited the stage that passed.**

### ⭐ R35 addendum — ROOT CAUSE: the cluster `git pull`s from `clu-dev`, which is 39 commits STALE
The Head clarifies round 1 was updated by **`git pull` on CSF3**, not by `push_repo.sh`. Hub-verified locally:

| ref | tip | state |
|---|---|---|
| local `main` | **275fe18** | current |
| **`clu-dev/main`** | ⛔ **3faef27** | ⭐ **exactly the commit in every stale manifest — main is 39 commits ahead** |
| `clu-dev/internal-state` | **97f259e**, snapshot **2026-09-11 @ e4f3d71** | ⛔ **stale by the whole pivot** |
| `clu/main` | b0257bc | code-only release base, deliberately frozen |
| `origin/main` | 40c2f31 | ⛔ FROZEN PUBLIC — never pushed |

⇒ ⛔ **The stale smoke was not an accident of this session; it is structural.** `git pull` on CSF3 serves `clu-dev`,
and `clu-dev` has not been pushed since wave 5. **It will keep serving stale code until we push it.**

### ⛔ TWO UPDATE PATHS THAT DISAGREE — pick one
`push_repo.sh` rsyncs the working tree **excluding `.git/`** (which is why `GEN_GIT_COMMIT` must be exported at all).
⇒ after an rsync onto a git clone, **the working tree and `.git` disagree**, and a later `git pull` there will
clobber or conflict unpredictably. ⭐ **Ruling: `push_repo.sh` is the ONE canonical path to CSF3. ⛔ Do not `git pull`
on the cluster.** The rsync idiom is what every existing script documents.

### ⚠ BACKUP EXPOSURE — flagged to the Head, Hub recommends acting at wave close
**39 commits of code and the entire internal state from R24 → R35 exist ONLY on this laptop.** The last internal-state
snapshot predates the pivot, `gen-baselines`, `gen-certificate`, both merged waves and every ruling since.
⛔ **The runbook must NOT be run as written** — line 41 is `git push -u clu main`, and `clu` is the code-only release
base that is deliberately frozen. **Correct subset: line 42 (`clu-dev main`) + the internal-state plumbing (§4).**
⭐ The §4 plumbing is **worktree-safe by construction** (temp-index + `commit-tree`, never a checkout) — which matters
because a `git checkout internal-state` in the main worktree has wiped force-added ignored files on this programme
before, and the runbook's own line 64 says NEVER do it.
⭐ **Safe to run with spokes in flight:** it pushes refs and builds a snapshot commit via plumbing; it touches no
worktree. ⛔ Still a network-visible push to the wrong-remote-adjacent target, so it goes to the Head as a decision.

---

## R36 — `push_repo.sh` had TWO defects; the canonical CSF3 path had never worked (Hub, 2026-09-13). `main @ 22f9dc5`

### Defect 1 — bash 3.2 + `set -u` + an empty array (`main @ f66fdda`)
macOS ships **bash 3.2.57**, where `"${ARR[@]}"` on an **empty** array under `set -u` raises *"unbound variable"*.
`DELETE_FLAG` is empty unless `PUSH_DELETE=1`. ⇒ ⛔ **the DEFAULT path of `push_repo.sh` has never run on this
machine** — it died at line 39 before rsync was invoked. Fix: `${DELETE_FLAG[@]+"${DELETE_FLAG[@]}"}`.
⭐⭐ **This is the root of the entire stale-cluster episode.** With the canonical rsync path broken, the cluster was
updated by `git pull` from `clu-dev` — 39 commits behind — so the first CSF3 smoke ran six-day-old code **and
reported green**. ⛔ One broken line in a helper script silently invalidated a gate.

### Defect 2 — an rsync exclude pattern that matches directories only (`main @ 22f9dc5`)
`--exclude '.git/'`: **a trailing slash makes an rsync pattern match DIRECTORIES ONLY.** In a git **worktree** `.git`
is a regular **file** (`gitdir: …`). Hub-verified empirically on a scratch tree: with `'.git/'` the file transfers;
with `'.git'` it does not, and `.gitignore` is unaffected. ⇒ rsync tried to write a 59-byte file over the remote's
`.git` **directory** → **partial transfer, exit 23**.
⚠ **Latent because the script assumed the primary checkout** — and the handover instructs us to push from a worktree
instead. Two correct-sounding rules combined into a defect.

### ⛔⛔ A HUB NEAR-MISS WORTH MORE THAN BOTH FIXES
The first attempt at defect 2 put the explanatory comment **inside the rsync command's backslash continuation**.
Backslash-newline is removed **before** tokenisation, so the `#` began a comment that swallowed **every remaining
argument** — all other excludes, the source and the destination. ⭐ **`bash -n` reported OK**, because it is
syntactically valid. The rsync stub showed the truth: `RSYNC ARGS: -avz --progress --exclude .git` — **no source, no
destination.**
⭐ **STANDING RULE, eighth strike: ⛔ never place a comment inside a backslash-continued command**, and ⛔ **`bash -n`
is a syntax check, not a behaviour check — verify a shell command by observing the arguments it actually passes**
(a stub on `PATH` costs seconds). ⭐ This is the same lesson as *"a check that cannot fail proves nothing"*, arriving
as *"a check that passes may be checking the wrong thing."*

### ⭐ STRUCTURAL FIX ORDERED — delete `.git` on the cluster
`~/scratch/CHLU/.git` exists only because someone cloned there. ⛔ It is **not needed**: the launcher takes provenance
from `GEN_GIT_COMMIT`, and `push_repo.sh` is the one canonical path. ⭐ **Removing it makes `git pull` on the cluster
IMPOSSIBLE, so the stale-tree failure class cannot recur.** ⛔ Do not use `PUSH_DELETE=1` as a substitute — `logs/` is
not excluded, so `--delete` would destroy the slurm smoke logs.

### ⭐ The readiness test, before any resubmission
⛔ Do not trust the manifest after the fact. **Check the launcher ON THE CLUSTER first:**
`grep -c certificate ~/scratch/CHLU/scripts/csf3/job_gpu_gen.sh` — **0 means the tree is still stale and submitting is
pointless.** ⭐ A pre-flight check on the artefact beats a post-hoc check on its provenance claim.

---

## R37 — Wave 7 merged: the headline table and a SECOND DATASET (Hub, 2026-09-14). `main @ 994ca9b`.
Targeted tests on the merged tip with `CHLU_GEN_DATA_ROOT` set: **146 passed, 0 skipped, 0 failed.**
⚠ Full-suite debt unchanged: still `a0034d4` / 2166. Both worktrees removed after verifying 0 unmerged, 0 dirty.

### ⭐⭐ `gen-bridge-dsprites` — THE GATE PASSES, 3/3. The stack exists on a second dataset.
Hub-recomputed from the three `seed_*/discrimination.json`, not read off the report: held-out in-corner `shape`
balanced accuracy **0.9616**, per seed **0.9518 / 0.9751 / 0.9579**; AUROC **0.9991**; ratio to the matched-seen
normaliser **1.007**; floor 0.8×. ⭐ **Above Shapes3D's Montero-configuration 0.9414 on the identical metric — the
second dataset is not the harder one**, and *"does it work on anything else?"* now has a measurement behind it.
⚠ **Artifact-naming trap:** the top-level `discrimination_gate.json` holds **seed 2 only** (0.9579), not the pooled
mean. The report quotes the 3-seed mean correctly, but ⛔ **a later reader taking the top-level file as the headline
would under-quote it.** **Owner: Hub, at the table review.**

### ⭐⭐ THE BUG IT CAUGHT IS WORTH MORE THAN THE GATE
The dSprites bridge path **had never executed**. Two defects, and the second is the dangerous kind:
1. `ValueError: too many values to unpack` — loud, immediate.
2. ⭐⭐ **`/255` applied to a `{0,1}` uint8 mask.** dSprites images are binary, not 0–255. ⇒ **an entirely black
   dataset that trains, evaluates and reports without a single error.** ⛔ Every dSprites number we would ever have
   taken would have been silently void. ⭐ **This is the strongest argument yet for running a second dataset early:
   the port is what exposes assumptions the first dataset let us keep.**

### ⛔ DO NOT CARRY R15's PROBE CLAIM ACROSS DATASETS
On dSprites the probe is **NOT saturated** — it reads `orientation` on **real** images at **0.15** (the shapes'
own 90°/180° symmetry caps it, pre-registered as P8). ⇒ ⛔ **R15's *"the probe reads real images at 1.0000"* is a
Shapes3D fact, not a property of the instrument**, and the dSprites *aggregate* ceiling is structurally
uninterpretable. **The compositional-term gate is unaffected** — it reads `shape`, which is clean.

### ⭐⭐ `gen-cell-harness` — THE HEADLINE TABLE, on images, 7 rows × 3 seeds, 0 NOT RUN
In-corner `shape` balanced accuracy via the **external** classifier (Hub-verified against `cell_results.json`):

| arm | external | probe | Δ |
|---|---|---|---|
| **`knn_proto`** | **0.9119** | 0.8884 | +0.0235 |
| `lace/ld_corrected` | 0.6605 | 0.6420 | +0.0185 |
| **`clu`** | **0.6568** | 0.6759 | −0.0191 |
| `lace` (`ode`) | 0.6347 | 0.6389 | −0.0041 |
| `lace/ld_published` | 0.5580 | 0.5924 | −0.0344 |
| `ebm_compose` | 0.5408 | 0.5318 | +0.0090 |
| ⭐ **`store_removed`** | **0.5000** | 0.5000 | 0.0000 |
| *decoder ceiling* | *0.9357* | *0.9414* | *−0.0057* |

⭐ **Cross-check that the plumbing did not drift:** `clu`'s probe number **0.6759** reproduces `gen-read-compose`
bit-identically, and the ceiling's Δ (−0.0057) reproduces `gen-external-classifier`'s leg-3 number to 4 dp.

### ⛔ NEVER-QUOTE, new and binding: "CLU beats LACE's best Langevin leg"
Top and bottom of the ranking are **instrument-invariant**, ⛔ but **one adjacent pair FLIPS**: `clu` and
`lace/ld_corrected` are separated by **0.0037** on the external instrument and **0.0339** on the probe, **with
opposite signs**. Both gaps sit at or below the sampler-noise floor. ⇒ ⭐ **report it as a TIE inside the noise, on
both instruments.** ⭐ **The table's value is precisely that it can say this** — a single-instrument table would have
asserted a win.

### ⛔ CARRIED — three, each needing a Hub owner
1. ⭐ **LACE's per-sampler number carries a CLASSIFIER-REFIT spread never previously reported.** `ode` **0.6797 ±
   0.0207** (R27-era) is a **seed** spread over **one** classifier fit per seed. Four independent refits at fixed
   (seed 0, configuration, decoder, instrument) give **0.6476 ± 0.0227**, and the published **0.7077** sits **2.65
   sample-sd above** it at *identical* classifier quality (own-val 0.9867 vs 0.9870). ⛔ **No single LACE per-sampler
   number is quotable without this second spread.**
2. ⚠ **Re-fitting a rival on a smaller row budget silently weakens it.** At 1,024 fit rows `knn_proto` fell
   0.8763 → **0.8308** and LACE's whitening went rank-deficient (|diag−1| 0.2382 → **0.6329**). ⭐ **Standing rule:
   a spoke that re-fits a rival inherits `GenBaselinesConfig`'s 32,256/512, ⛔ never its own budget.**
3. ⚠ **Third strike on transcribing constants from RENDERED 4-dp tables instead of from disk** (same family as
   `gen-external-classifier` §11.4). With corrected constants the CLU and `knn_proto` arms reproduce their published
   probe numbers **bit-identically, 3/3 seeds, Δ = 0.0 exactly**. ⭐ **Rule: a reference constant is read from the
   artifact, never from a report's table.**
4. ⚠ **The `ddof` clash is STILL live and has now appeared in a third report** — dSprites' `± 0.0099` is population
   sd; the sample sd is **0.0121**. ⛔ Pin one convention programme-wide. **Owner: Hub, unchanged from R31.**

---

## R38 — ⛔ HUB HYPOTHESIS REFUTED BY THE `.err`; the real state is two DIFFERENT failures (Hub, 2026-09-14)

**The Head's `.err` (a glob over `clu-gen-cert-smoke-*.err`, `head -5`) returns FIVE lines from TWO rounds:**
```
FATAL: STAGE='certificate' is not one of: encoder decoder store cell     x3   <- ROUND 1, the stale tree
FATAL: cannot resolve the git commit (push_repo.sh excludes .git) ...    x2   <- A LATER round
```
⛔ **The Hub's `sbatch --export` space-mangling hypothesis (R35/R37 working theory) is REFUTED.** `SEEDS="0 1 2"` was
never implicated; no array-count error appears anywhere. ⭐ **The launcher's own NOTE — that a value may contain
spaces provided the whole `--export` argument is one shell word — is neither confirmed nor contradicted and remains
UNTESTED.** ⛔ Do not record it as verified.

### ⛔ THE TRAP THE HUB WALKED INTO: a log GLOB pools every round
`logs/clu-gen-*.err` accumulates across submissions, so `head -5` returned **round 1's** errors first and the newest
round's last. ⭐ **A diagnosis drawn from a glob over an append-only log directory is a diagnosis of the OLDEST
failure, not the current one.**
⭐ **STANDING RULE, ninth strike: inspect logs BY JOB ID (`logs/<name>-<jobid>_*.err`), ⛔ never by a wildcard over a
directory that survives previous rounds — and move old logs aside before a re-run so the next round is unambiguous.**
⚠ This is the third time this programme has been misled by stale artefacts surviving into a new round (R31's silent
skips, R35's stale tree, now this).

### ⭐ WHAT THE SECOND ERROR ACTUALLY PROVES — and it is the launcher behaving CORRECTLY
*"cannot resolve the git commit"* fires only when **all three** provenance sources fail: `GEN_GIT_COMMIT` unset,
**no `.git`** at `$CLU_REPO`, and no `GIT_COMMIT` file. ⇒
1. ⭐ **`rm -rf ~/scratch/CHLU/.git` was carried out** — the Hub's own R36 instruction, which deliberately removed the
   git fallback so a stale `git pull` can never recur.
2. ⛔ **`GEN_GIT_COMMIT` was EMPTY in the login-node shell.** ⭐ **`$GEN_GIT_COMMIT` in an `sbatch --export` expands
   where `sbatch` RUNS — the login node — NOT on the laptop where it was exported.** Exporting it locally does
   nothing for the submission.
⭐ **The launcher refused to burn wallclock on an unprovenanced run, exactly as designed.** ⛔ That is a PASS for the
guard, not a defect: *"a run without provenance is not worth its wallclock."*
⚠ **Consequence of R36's `.git` deletion, now explicit and accepted:** with no fallback, `GEN_GIT_COMMIT` is
**mandatory on the login node for every submission, forever.** That is the intended trade.

### The corrected state of the gate
| item | state |
|---|---|
| encoder plumbing | ⚠ passed **against the stale tree**, provenance stamped `3faef27` — **re-run for a valid record** |
| certificate stage | ⛔ **still never executed** — round 1 died on the stale tree, the retry died on provenance |
| GPU preflight (A100) | ⭐ **banked, environment fact, does not need re-running** |
| **the gate** | ⛔ **SHUT.** `smoke_result.json` must not be written. |

**R38 corroboration (2026-09-14).** The archived log names confirm the two-round structure exactly:
**round 1 = `20325640` (encoder) · `20325641` (cert ×3) · `20325642` (gpuprobe)**;
**round 2 = `20326970` (encoder) · `20326971` (cert ×3)**, submitted after the rsync and after `.git` was removed.
⇒ ⭐ **every round-2 job died on `cannot resolve the git commit` and wrote NO manifest**, which is why the
`encoder/clu_s0` manifest the Head read was **round 1's leftover**, still stamped `3faef27`. ⛔ **A run directory
retains the previous round's manifest when the new run dies before the trap — so a manifest is evidence about SOME
run, not necessarily the last one.** Read it together with the job id in `manifest.slurm.job_id`.
**Round 3 preconditions verified by the Head before submitting:** `GEN_GIT_COMMIT=[22f9dc5]` non-empty on the
**login node**, and `grep -c certificate` on the cluster's launcher = **14**. Old logs archived to `logs/old/`.

---

## R39 — ⭐⭐ THE CLUSTER SMOKE IS GREEN. THE GATE IS OPEN. (Hub, 2026-09-14)
Round 3, Head-run, Hub-verified from `run_manifest.json` (`status` + `exit_code` + `git_commit` + `slurm.job_id`):

| run dir | status | commit | job |
|---|---|---|---|
| `encoder/clu_s0` | **ok** rc=0 | **22f9dc5** | 20344190 |
| `certificate/clu_s0` | **ok** rc=0 | **22f9dc5** | 20344191 |
| `certificate/clu_s1` | **ok** rc=0 | **22f9dc5** | 20344192 |
| `certificate/clu_s2` | **ok** rc=0 | **22f9dc5** | 20344193 |

⭐ **All four carry the CURRENT commit and FRESH job ids** — the two checks that distinguish a real result from a
leftover manifest (R38). ⭐⭐ **The `certificate` stage has now executed for the first time anywhere**, and it landed
under **`certificate/`**, not `cell/` — the exact defect `gen-launcher-fixes` was written to fix, finally confirmed
on the cluster rather than in a local dry run.
**Gate file written:** `.claude/outputs/gen-cluster-smoke/smoke_result.json`, `"status": "ok"`.
⇒ ⭐ **R21 / R23.6 are SATISFIED. `gen-encoder-unfreeze` is UNBLOCKED.**

### ⛔ What the green does NOT cover — stated so it is never over-read
1. **`SMOKE=1` payload only.** No stage module ran; **no science happened.** This is path plumbing, the manifest and
   provenance — nothing more.
2. ⛔ **A real `certificate` run is still NOT ready.** `exp_gen_certificate` reads its inputs from `.claude/outputs/…`
   and `push_repo.sh` **excludes `.claude/`**, so the bridge and store artefacts do not exist on the cluster. It will
   fail **loudly** (`FileNotFoundError`), so no wrong number can escape — but the artefacts must be staged first.
3. ⚠ **`sbatch --export` with spaces in a value remains UNTESTED.** Round 3 used **single-seed** submissions, so the
   launcher's own NOTE about spaces is still unverified. ⛔ Do not record it as confirmed; use one seed per job.

### ⭐ The cost of this gate, for the record — 3 rounds, ~4 wasted queue cycles, 5 distinct causes
`-p serial` incompatible with the script's own `-G 1 -c 8` · a stale tree from `git pull` against a 39-commit-behind
fork · `push_repo.sh` dying on bash 3.2's empty-array expansion · `push_repo.sh` shipping a worktree's `.git` **file**
into a `.git` **directory** · `GEN_GIT_COMMIT` exported on the laptop instead of the login node.
⭐⭐ **Every one was invisible until the smoke actually ran, and the first round reported GREEN while running
six-day-old code.** ⛔ **The standing lesson: "it has never been run" is a defect report, not a scheduling note.**
