# REFEREE-v0 — adversarial review of `w10-draft-v0/DRAFT-v0.md`

**w10 `spoke/referee` · agent `paper-referee` · 2026-09-21.**
**Scored runs spent: ⛔ ZERO.** This spoke produced no number. Every figure below is quoted with the
path that holds it. ⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
**Target reviewed:** `/c/chlu/outputs/w10-draft-v0/DRAFT-v0.md` (74,813 B, 753 lines, mtime 10:53
local — the gate was discharged before this spoke started).

---

## 0. WHAT THIS REVIEW ACTUALLY DID

⛔ Not a read-through. Every number in the draft that carries a pointer was resolved to the file at
the end of that pointer, and where the file disagrees with the draft, the file wins.

| I opened | and did what |
|---|---|
| `w10-seeds-and-tables/TABLES.md` + `TABLES.json` | read whole; **re-ran `verify_tables.py` myself** |
| `w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` | dumped `/display/delete_green_exactness`, `/ledger`, `/env_columns`, `/pipeline_ceiling_rows`, `/span_ceiling_rows`, `/baseline_params` |
| `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json` | dumped **all 21 arms × 3 seeds** (class, params, store_B, address_B, r3, r4) and recomputed every paired delta |
| `w8-downdate/DOWNDATE-READY-test.json` | dumped **every leaf field**, per seed |
| `w8-downdate/LOG.md`, `w8-single-design/LOG.md`, `w5-compose-field/LOG.md`, `w9-maze/LOG.md` | read the cited line ranges and each FINAL REPORT |
| `w10-physics-section-evidence/EVIDENCE.md`, `BENCHMARK-STATEMENT.md` | read whole |
| `w10-registry-fold/FINDINGS.md` + the `N-I8-*` fold in `negative_results.md` | read whole |
| `w8-novelty/NOVELTY.md` (and its `LOG.md` correction) | read whole |
| `evidence/outputs/w4-,w5-,w6-,w8-advisor-report.md`, `advisor-wave-plan-w8-w10.md` | grepped every cited figure |

### What reproduced cleanly — do not re-litigate these

- ⭐ **`verify_tables.py` PASSES on a fresh run by this spoke:** 27 sources registered, 27 source
  hashes re-verified, 240 cells, **720 per-seed values** re-resolved (3 via a declared reduction),
  5 figures, 5 figure hashes, 5 NOT-RUNs, 4 corrections. The draft's App. M claim is sound.
  *Path:* `/c/chlu/outputs/w10-seeds-and-tables/verify_tables.py`.
- ⭐ **Every w9/maze deletion cell is exact.** `7.5217e-16 / 8.8265e-16 / 7.8176e-16`, projector
  `0.0 / 0.0 / 0.0`, rows `216/992 · 226/975 · 193/1084`, `refit False`, `passes_over_retained_data 0`,
  `items_kept False`, `basis_change_bytes_deployed 0`. *Path:*
  `w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` `/display/delete_green_exactness`.
- ⭐ **Every w5 composition delta reproduces from the raw arm table.** Row 4
  `63.6122−53.4694 = +10.143`, `64.3265−55.6531 = +8.673`, `67.6531−59.8571 = +7.796`; row 3
  `+4.033 / +2.483 / +2.417`; the `oracle_content_items` gap `+31.224 / +25.204 / +25.878`;
  `oracle_content_pool − mlp_180k` `+10.980 / +8.224 / +8.551`. *Path:*
  `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`.
- ⭐ **K15 is exact and I recomputed it independently:** max |span ceiling − pipeline ceiling| =
  **0.013761**, **exactly equal on 7 of 12** row×seed cells.
- ⭐ TABLE 1's BAR-1 deltas (−0.0560 / −0.0209 / −0.0434), TABLE 2's `A′`/`A″`/`A` and CI95 lows, the
  novelty counts (3,038 / 759 / 254 / ≈4,050; 10 of 3,038; 0 of 3,038; 4 of 6), the D32b break-out
  (sums to 1,339,083 exactly; the 16.5 % / 83.5 % split is right), `7.48×` (3,922,450/524,288 =
  7.4814), `0.991×` (519,794/524,288 = 0.99143), and every advisor-report figure quoted in §11 — all
  check out.

⇒ The draft's arithmetic on the numbers it *chose to print* is good. **Everything below is about
numbers it did not print, numbers it printed from a superseded source, and one claim-STOP breach.**

---

## 1. ⛔ VERDICT — simulated

> ## **REJECT as it stands. Accept-with-major-revision is reachable, and the revision costs 0 scored runs.**

Simulated ICLR panel, hostile but fair:

| reviewer | score | the one thing that decides it |
|---|---|---|
| **R1 — machine unlearning** | **3 (reject)** → 6 after MF1–MF7 | *"Exact" is never defined, the frozen encoder/decoder is never refit, and the sufficient statistic that buys the exactness is **23.5× the byte budget the paper pins everything else to** and appears in no table.* |
| **R2 — compositional generation** | **5 (marginal reject)** → 6 | *Two main-text tables report the same arm's composition in two metrics on two splits; the parameter match is quoted at −0.01 % from a figure the source spoke corrected to −0.30 %/+0.50 %.* |
| **R3 — benchmarks / empirical rigour** | **6 (marginal accept)** | The negatives discipline and `verify_tables.py` are genuinely above the field. Docks it for a main-text composition row (row 4) with no caveat column, and for an environment mislabel. |
| **AC** | **reject, invite resubmission** | Two artefact-level contradictions inside the file the paper cites for its lead result: `a1_subspace.passed_every_seed = false`, and `note: "No scored run and no claim"`. |

⭐ **None of the sixteen must-fixes below requires a scored run, a fitted model or a new
measurement.** Every one is a number already on disk and absent from the draft, or a number in the
draft that a later artefact superseded. The verdict is about *reporting*, not about the science —
which is why it is fixable inside the prose pass.

⛔ **And the falsifier the draft checked itself against does not fire in the direction it thinks.**
§10 concludes "a contribution paragraph survives". It does — but the *pointer the draft hands the
Head for the contribution sentence* points at a STOP-violating sentence (MF1). The failure mode is
not "no paragraph survives"; it is "the draft hands the Head the wrong one".

---

## 2. ⛔ MUST-FIX — sixteen items

Per the acceptance criterion, each names **the claim · the evidence it fails against · the smallest
change that fixes it.**

---

### MF1 ⛔⛔ The draft points the Head at a sentence that violates its own novelty STOP #1

- **The claim.** §1.2: *"⭐ **The contribution sentence.** A compositional generative store whose
  whole content is a sufficient statistic, so deletion **is** the refit. | `NOVELTY.md` §3 final ¶ —
  **the survivable form is verbatim there**."*
- **The evidence it fails against.** `w8-novelty/NOVELTY.md` §3, final paragraph, verbatim:
  > *"the whole generative object is a sufficient statistic, so a generative model with **a
  > request-indexed tensor head** admits deletion that is exactly the refit"*

  `request-indexed tensor head` is **STOP #1 word for word** — the draft's own §1 NEVER-CLAIM row
  (*"multilinear/bilinear factored generation with a request-indexed tensor head"* — Tenenbaum &
  Freeman 2000, Vasilescu & Terzopoulos 2002), and NOVELTY.md §4's own stop list repeats it. A Head
  told "lift this verbatim" lifts TensorFaces into the abstract.
- **Smallest fix.** Delete the words *"the survivable form is verbatim there"*; repoint §1.2 at the
  draft's **own** §10 paragraph, which already avoids the phrase by construction; and add one line:
  *⛔ NOVELTY.md §3's sentence contains STOP #1 and must be paraphrased, never lifted.*

---

### MF2 ⛔⛔ "whose whole content is a sufficient statistic" is false of **both** implementations

- **The claim.** §1.2, §1.3 and the §10 contribution paragraph: the store's *content* **is** a
  sufficient statistic, therefore deletion **is** the refit.
- **The evidence it fails against.** In neither implementation is the content the statistic. The
  content is the *leading eigenvectors of* a statistic kept **separately and never netted into any
  byte ledger**:

  | leg | deployed content | sufficient statistic | statistic / content |
  |---|---|---|---|
  | w9/maze | 1,642,752 B deployed (`/ledger/deployed_bytes`) | **35,904 B** float64 (`/ledger/statistic_bytes`) | 2.19 % |
  | w8/downdate | **524,288 B** store (`store_bytes`, the pin, tol 0) | **12,310,672 / 11,941,360 / 11,850,912 B** (`cov_bytes_per_seed`) | **2,348 % / 2,278 % / 2,261 %** |

  *Paths:* `w9-maze/runs/scored1_seed0_test.json` `/ledger`;
  `w8-downdate/DOWNDATE-READY-test.json` `/cov_bytes_per_seed`. The w9 ledger's own `note` says it
  outright: *"the float64 sufficient statistic is what buys EXACT deletion and is **declared beside
  them, never netted in**"*.
- **Smallest fix.** Rewrite the contribution noun phrase to what was measured: *"a generative store
  whose content is **derived from a retained sufficient statistic**, so downdating the statistic
  **induces** the refit of the content"*, and state the statistic's byte cost in the same sentence
  (MF3). ⛔ The current phrasing is also what makes MF3 invisible.

---

### MF3 ⛔⛔ The paper's Table 1 (the lead) prints no byte row — the object is **24.5× the pin**

- **The claim.** §3's rendering of TABLE 4 has eight property rows for the w9 leg and five for the
  w8 leg. **No byte row on either.** §5 meanwhile spends a section on the compose-and-forget object
  being *"7.48× the byte budget"* and rules that it *"is quoted with its overrun attached or not at
  all"*.
- **The evidence it fails against.** The lead result's own gate file records the overrun as a named
  field: `times_over_budget` = **24.5466 / 23.8422 / 23.6697**; `honest_total_bytes` =
  **12,869,482 / 12,500,170 / 12,409,722 B**. The downdate spoke's own FINAL-REPORT table prints it
  in the same row as each seed's A1 verdict: *"24.55x / 23.84x / 23.67x"*. *Paths:*
  `w8-downdate/DOWNDATE-READY-test.json` `/times_over_budget`, `/honest_total_bytes`,
  `/per_seed/{0,1,2}/times_over_budget`; `w8-downdate/LOG.md:632-635`.
- **Why it is fatal rather than cosmetic.** The paper penalises the §5 object for 7.48× and exempts
  its own lead result at **24.5×** — three times worse — one section later. A referee who opens one
  JSON finds `times_over_budget: 24.5466` in the file cited for the headline. ⛔ And §5's chain
  (*"the covariance costs Σ_k d_k²; fitting it inside 524,288 B forces K ≥ 36"*) is **precisely the
  statement that the §3 object does not fit**.
- **Smallest fix.** One row on TABLE 4's w8 leg: *"honest total bytes (store + statistic) ·
  12,869,482 / 12,500,170 / 12,409,722 = **24.55× / 23.84× / 23.67×** the 524,288 B pin"*; one row on
  the w9 leg: *"statistic declared beside the read · 35,904 B = 2.2 % of deployed"*. Then one
  sentence in §5: *the object that forgets exactly is itself 24.5× over the pin, which is why §5
  exists.* ⭐ That sentence converts the paper's worst omission into its cleanest framing.

---

### MF4 ⛔ "Exact" is never defined, and the only definition the measurements support is store-scoped

- **The claim.** §3: *"deleting a set of rows from the store is **exactly** the store refitted
  without those rows"*; §10: *"deletion that is **exactly** the refit"*; NOVELTY.md §4 calls it
  *"refit-equal erasure of a **generator**"*.
- **The evidence it fails against.** The renderer is **frozen and is never refit**. Draft §2.4 says
  so; `w9-maze/LOG.md:376` says every arm *"is rendered by the same **frozen decoder** through the
  same latent standardisation"*. That decoder's training set includes eval-bank items:
  `ae_train_items = {'eval_bank_train': 543, 'pool': 4860, 'pool_sites': 270}`
  (`w9-maze/runs/scored1_seed0_test.json`). The 216/226/193 deleted rows leave the statistic; the
  encoder/decoder weights fitted with them present are untouched. **Retrain-equivalence of the
  generative model is not what was measured.**
- **Smallest fix.** One definition sentence in §3, before the table: *"**Exact** here means the
  post-deletion **store content** equals, to float roundoff, the content fitted from the retained
  rows alone, **holding the frozen encoder and decoder fixed**. It is not a claim that the renderer
  forgets."* Then change "of a generative model" / "of a generator" to "of a generative model's
  **store**". ⛔ Without this, R1 rejects on sight — it is the standard unlearning objection and the
  paper currently prints no answer.

---

### MF5 ⛔ "Exact deletion" is printed with no cost, against a ruling the programme wrote itself

- **The claim.** §3 carries exactness with no cost statement anywhere in the section.
- **The evidence it fails against.** `evidence/outputs/w5-advisor-report.md` §6.3, the programme's
  own ruling request: *"**What may the deletion claim be?** 'The hole is exact and free' is
  supported; **'deletion is exact and free' is NOT** — eviction costs composition."* And §2:
  eviction of `red` *"refuses perfectly but **FAILS the composition bar** (**−2.09 / −0.91 / −6.74**
  against a declared ±1.0)"*, with the post-hoc control that *"an item store loses composition when
  it loses atoms, **whichever atoms they are**"*. The pattern reproduces on the maze: surviving
  targets **0.1024 → 0.0000** on TEST, three seeds (`TABLES.md` T1 §1d) — which the draft banks in
  §11.4, four sections from the claim it qualifies.
- **Smallest fix.** One sentence at the end of §3: *"Exact does not mean free. On TEST the same
  deletion takes surviving-target composition from 0.1024 to 0.0000 (§11.4); recovering it costs the
  address-book carry of §5; and w5 measured atom eviction on the item store at −2.09 / −0.91 / −6.74.
  The claim is exactness, never cost-free erasure."*

---

### MF6 ⛔ The gate file cited for the lead carries two caveats the draft never quotes

Both are leaf fields of `w8-downdate/DOWNDATE-READY-test.json`, the file K2 names.

- **(a) `cut_is_marginal_any_seed = True`**, with `per_seed/{0,1,2}/cut_is_marginal = True` on **all
  three seeds**. The hub commissioned to score the object wrote, on clearing the gate:
  > *"⚠ One field I flag rather than pass over: `cut_is_marginal_any_seed: true`, carried as a bare
  > boolean with no per-seed detail in the gate. A marginal cut means a direction sits near the
  > threshold, so a small perturbation could change WHICH directions are kept — the same instability
  > class as the original failure. ⛔ It does not block the gate and I am not treating it as a defect,
  > **but it belongs in the caption of anything I score on that object.**"*

  (`w8-single-design/LOG.md`, 2026-09-20T22:45Z.) **The draft has no caption.**
- **(b) `a1_subspace.passed_every_seed = **False**`, `seeds_failing = ["2"]`**,
  `worst_in_float32_ulp = 60.36282`, against its own
  `worst_floor_subspace_in_float32_ulp = 66.1107`. The top-level `all_three_assertions_passed = true`
  **excludes** `a1_subspace`. A referee opening the cited JSON reads `passed: false` inside the
  paper's lead-result artefact, and the paper says nothing.
- **Smallest fix.** Two clauses in TABLE 4's caption: *"the store is built under a variance-share
  admission cut that is **marginal on every seed** (`cut_is_marginal_any_seed = true`); the
  byte-level assertion `downdate_equals_refit` passes on all three seeds, while the separate
  subspace-level assertion `a1_subspace` on the **pre-cut** store fails on seed 2 at 60.36 ulp —
  **below its own 66.11 ulp floor**, so that failure is a property of the content and not of the
  downdate, which is what the artefact's `reproducibility_floor.what` field states."*

---

### MF7 ⛔ K2 is labelled **RESULT · 3 / TEST**; the artefact says "No scored run and no claim"

- **The claim.** §B register K2: *"| **RESULT** | 3 / TEST |"*; §3's TABLE 4 labels the w8 leg
  `class **RESULT**`.
- **The evidence it fails against.** `w8-downdate/DOWNDATE-READY-test.json` `/note`:
  > *"⛔ **No scored run and no claim**: these are properties of the object (deviations, byte counts,
  > row counts), not outcomes. **hub/single-design scores the object.**"*

  And the `w8-downdate/LOG.md` FINAL REPORT header: *"⛔ **No scored run. No claim.**"* The hub that
  was to score it — `hub/single-design` — scored **0 of 3** and is the draft's own **NR3**
  (`TABLES.md` §NR3: *"NOT RUN — 0 of 3 scored, full budget intact"*). ⇒ **Nobody scored this leg.**
  (The w9/maze leg *is* inside scored runs S1/S2/S3, so K1's RESULT label is correct.)
- **Smallest fix.** Reclassify K2 and TABLE 4's w8 leg as **`ASSERTION BATTERY (object property) ·
  3 seeds · TEST condition · 0 scored runs`**, and add to §3: *"The second implementation is an
  assertion battery on the object, not a scored row; the hub commissioned to score it returned its
  full budget (NR3)."* ⭐ This costs the paper nothing — two independent *implementations* is still
  the claim — and it removes the one label a referee can call inflation.

---

### MF8 ⛔ TABLE 4's per-seed absolutes are not the numbers the source spoke reports

- **The claim.** §3 TABLE 4, w8 leg: *"max **absolute** deviation, worst well | **2.365e-14 /
  1.126e-09 / 7.105e-15**"*, printed under the banner *"⛔⛔ THE NUMBER THE PAPER MUST QUOTE IS THE
  ULP FIGURE OR THE TRUE ABSOLUTE — NEVER `1e-14`."*
- **The evidence it fails against.** The same JSON carries **two** per-seed sets, and the draft
  prints the smaller:

  | source | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | `/downdate_equals_refit/max_abs_dev_per_well_by_seed` (max of the list) — what `TABLES.md` T4 and the draft print | 2.365e-14 | 1.126e-09 | 7.105e-15 |
  | `/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` — what the spoke's own FINAL REPORT §2 prints ("pinned 11,024 rows + random 64") | **4.547e-13** | 1.126e-09 | **2.842e-14** |

  *Paths:* `w8-downdate/DOWNDATE-READY-test.json`; `w8-downdate/LOG.md:900-906` (the A1
  before/after table) and `:1010-1024` (FINAL REPORT §2). The discrepancy is **19×** on seed 0 and
  **4×** on seed 2. A correction (C2) whose whole point is to stop an understated figure being
  printed, and which then prints a *different* understated figure, is the worst available outcome
  for that correction.
- **Smallest fix.** Print the spoke's own row — **4.547e-13 / 1.126e-09 / 2.842e-14** — and add the
  column header that distinguishes the two deletions: *"worst over the pinned 11,024-row deletion
  **and** the random-64 deletion"*. Keep the ulp figure beside it.

---

### MF9 ⛔ Only one of the two unresolvable legs is declared

- **The claim.** §3: *"⚠ **One honest limit to print beside the numbers:** `spoke/downdate-build`'s
  worst leg (TEST **seed 1**, 1.126e-09) sits below its own measured reproducibility floor of
  4.282e-09."*
- **The evidence it fails against.** `w8-downdate/LOG.md:900-906`, the A1-after-cut table, gives
  deviation against floor on all four legs:

  | leg | A1 worst AFTER | floor AFTER | resolvable? |
  |---|---|---|---|
  | val 0 | 1.164e-10 | 3.887e-12 | yes |
  | test 0 | 4.547e-13 | 2.127e-13 | yes |
  | **test 1** | 1.126e-09 | 4.282e-09 | **NO** |
  | **test 2** | 2.842e-14 | 5.684e-14 | **NO** |

  ⇒ **two of the three TEST seeds**, not one.
- **Smallest fix.** Replace with *"on **two of three** TEST seeds the deviation sits below that
  leg's own reproducibility floor (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs 5.684e-14),
  so on those legs the residual is not resolvable and the claim rests on the floor rather than on
  the deviation."* ⭐ Stated that way it is *stronger*: an unresolvable residual is the best possible
  outcome for an exactness claim, and the draft currently hides two thirds of it.

---

### MF10 ⛔ The scored maze rows are **CMAZE-2**; the draft calls the whole section CMAZE-1

- **The claim.** §6 and every table slot: *"**TABLE 1** … CMAZE-1, w9 `hub/maze`"*. Never-quote #3:
  *"Never merge T1 with T3 (**same environment**, different budget)"*.
- **The evidence it fails against.** `w9-maze/LOG.md` FINAL REPORT §1:
  > *"**CMAZE-2** — a REQUEST layer over CMAZE-1. The engine, the solver and the exact integer judge
  > are `w8_envscout.maze_colour`, imported unmodified … The layer adds named agents and named
  > targets inside one maze …"*

  The registry says the same thing as its own self-check: `w10-registry-fold/FINDINGS.md` §6 —
  *"`N-I8-17` — w9's BLIND-SEQUENTIAL of 0.9037 is recorded as a **REQUEST-LAYER** weakness against
  **CMAZE-2**, **not** as a defect in the adopted CMAZE-1, because that is what the measurement
  shows."* TABLE 3 (the harness tier) **is** CMAZE-1 proper.
- **Why this is a must-fix and not a nit.** The draft's Correction **C4** is the claim that 0.4980 is
  *"a property of the REQUEST LAYER, not of CMAZE-1"*. If both tables are labelled CMAZE-1, C4
  contradicts itself on the page — the 0.9037 column *is then* a CMAZE-1 number. The triple
  0.4980 (published CMAZE-1) / 0.5317 (CMAZE-1 at the harness tier) / 0.9037 (CMAZE-2) is unreadable
  under one label.
- **Smallest fix.** Relabel T1 and all of §6/§6.1/§6.2 **CMAZE-2**; define it in one line at §6's
  head (*"a request layer over CMAZE-1; engine, solver and exact judge imported unmodified"*); keep
  T3 as CMAZE-1; rewrite never-quote #3's parenthesis to *"different environment **and** different
  budget"*.

---

### MF11 ⛔ "FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere" is false on the hub's own split

- **The claim.** §6.2: *"FRAGMENT-LOOKUP r=1 and r=2 **0.0000 on every seed** (TABLE 3)"*; K16:
  *"with FRAGMENT-LOOKUP r=1/r=2 at **0.0000 everywhere**"*.
- **The evidence it fails against.** `w9-maze/LOG.md` §3, the hub's own caveat table:

  | column | published CMAZE-1 | THIS hub's split (mean of 3 scored seeds) |
  |---|---|---|
  | FRAGMENT-LOOKUP r=1 / r=2, **cross-maze** | 0.0000 / 0.0000 | 0.0006 / 0.0000 |
  | FRAGMENT-LOOKUP r=1 / r=2, **same-maze** | (not measured there) | **0.0741 / 0.5105** |

  **0.5105** against row 2's pipeline ceiling of **0.6319** is the largest arm number anywhere in
  that table — a fragment table reaches 81 % of the pipeline ceiling where the CLU reaches 20 % and
  the best learned baseline 29 %. The hub names it rather than hiding it: *"It is also not a
  like-for-like arm: FRAG reads the true maze at every step, which charter B.3 forbids the CLU."*
- **Smallest fix.** In §6.2 and K16: *"FRAGMENT-LOOKUP **cross-maze** r=1/r=2 = 0.0006 / 0.0000 —
  the split does not leak; the **same-maze** table reads 0.0741 / **0.5105** and is not a
  like-for-like arm, because it reads the true maze at every step, which charter B.3 forbids the
  CLU."* ⭐ And put 0.5105 into §6.1(i): it is the strongest single piece of evidence for *"every
  piece of every answer is already written"*, and the draft leaves its best exhibit on the floor.

---

### MF12 ⛔ The parameter match is quoted from a figure the source spoke corrected away

- **The claim.** §2.3: *"Parameter budget SECONDARY: **178,928** vs `mlp_180k`'s 178,952 =
  **−0.01 %**"*, pointer *"`w5-compose-field/LOG.md` **02:45Z** entry"*. §4 repeats
  *"parameter-matched to **−0.01 %**"*.
- **The evidence it fails against.** `w5-compose-field/LOG.md`, the 01:58 UTC entry, headed
  *"⛔ **CORRECTION to a number I put in the PREREG.**"*:
  > *"§1 states the arm's address book as 191,424 B / 47,856 parameters, giving a total of 178,928
  > against `mlp_180k`'s 178,952 (−0.013 %). **That figure came from the `val` leg.**"*

  | scored leg | store | address | TOTAL params | vs `mlp_180k` (178,952) |
  |---|---|---|---|---|
  | test seed 0 | 524,288 B / 131,072 | 189,344 B / 47,336 | 178,408 | **−0.304 %** |
  | test seed 1 | 524,288 B / 131,072 | 195,064 B / 48,766 | 179,838 | **+0.495 %** |
  | test seed 2 | 524,288 B / 131,072 | 195,064 B / 48,766 | 179,838 | **+0.495 %** |

  and the spoke's own reading: *"the secondary parameter match is **within ±0.5 %** rather than the
  ±0.013 % I wrote, and **on two of three seeds the arm carries slightly MORE parameters than the
  bar**. ⛔ I state it that way round rather than quoting the seed that flatters me."* I re-derived
  it from the arm table: `param_count 131072` + `address_bytes 189344 / 4 = 47336` = **178,408**
  (`w5-compose-field/scored/test_seed0/READ-TABLE.json`). **The draft quotes the val figure — 50×
  tighter than the truth — in the very place the spoke wrote a correction bullet to prevent it.**
  The pointer is also wrong: the 02:45Z entry is the `val` arm-clears-the-bar bullet (+16.875), not
  a byte or parameter entry.
- **Smallest fix.** Replace the figure with the three-seed table above and the spoke's own sentence;
  repoint to `w5-compose-field/LOG.md` **01:58 UTC**.

---

### MF13 ⛔ "at 524,288 B" invites a byte comparison the arm does not win

- **The claim.** §4: *"`pca_tensor` beats `mlp_180k` … at **524,288 B tolerance 0**"*; §5's Table-2
  pin summary repeats it.
- **The evidence it fails against.** The rival has no store and no address book. Total deployed:

  | | store | address book | total |
  |---|---|---|---|
  | `pca_tensor` seed 0 | 524,288 B | 189,344 B | **713,632 B** |
  | `pca_tensor` seeds 1, 2 | 524,288 B | 195,064 B | **719,352 B** |
  | `rival_mlp_180k` | 0 | 0 | 178,952 × 4 = **715,808 B** |

  *Path:* `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`, fields `store_bytes`,
  `address_bytes`, `param_count`. ⇒ the arm sits at **99.70 % / 100.50 % / 100.50 %** of the rival's
  bytes — a genuinely matched comparison — but *"at 524,288 B"* reads as *27 % cheaper*, and that is
  what a referee will check and find false.
- **Smallest fix.** State the form the numbers support: *"the **store** is pinned at 524,288 B at
  tolerance 0; the address book (189,344–195,064 B) is declared beside it; **total deployed bytes
  are 713,632–719,352 against the rival's 715,808 — matched to within ±0.5 %.**"* ⭐ This is a
  stronger claim than the draft's and it is the one the artefacts prove.

---

### MF14 ⛔ Row 4 is a main-text composition row with **no BLIND-SEQUENTIAL column in existence**

- **The claim.** §6's main table prints row 4 (budget interpolation) as a composition row and reports
  BAR 1 failing on it at −0.0434. The charter rule printed at the head of `TABLES.md` is *"Every
  CMAZE-1 composition row carries BLIND-SEQUENTIAL measured on its own split."*
- **The evidence it fails against.** `TABLES.md` §1c carries caveat columns for **row 2 and row 3
  only**. I went to source: `w9-maze/runs/scored1_seed0_test.json` `/env_columns` has exactly four
  keys — `unseen_pair`, `nway`, `fragment`, `fragment_cross_maze`. **There is no blind block for row
  4 on any seed.** The rule is unsatisfiable for that row, and neither `TABLES.md` nor the draft says
  so. Separately, the draft's slotting sends **all** of §1c to App. A (*"Table 3 (main) | TABLE 1
  §1a — RESULT rows + the C.2 controls"*; *"App. A | … caveat columns 1c"*), so **no** composition
  row carries its caveat column in main text at all.
- **Smallest fix.** Two changes. (i) Move BLIND-SEQUENTIAL (and BLIND-FACTOR) for rows 2 and 3 into
  the main-text table, per the charter. (ii) Add a declared gap under row 4: *"⛔ **row 4 has no
  BLIND column on any seed** — `/env_columns` carries blind blocks only for `unseen_pair` and
  `nway`. DECLARED NOT-MEASURED, not a blank."*

---

### MF15 ⛔ §11's negatives list omits a tier-A negative with a 4× pre-registration miss

- **The claim.** §11 opens *"this section is where they are collected so a referee can count them"*,
  and lists sixteen.
- **The evidence it fails against.** `w10-registry-fold/negative_results.md` **`N-I8-19`** — tier A,
  CURATOR-PROPOSED, w9 — appears nowhere in the draft:
  > *"⛔ **The CLU's refusal on CMAZE-2 is NOVELTY DETECTION, not wall detection** — and its
  > false-refusal rate on grounded-possible requests is **0.8182 against a pre-registered 0.20**."*

  with the control that decides it: the object refuses the **reachable-but-unwritten** combination at
  **0.9781**, as often as the **walled** one at **0.9801**; and *"the supervised refuser, which HAS
  seen impossible examples, **separates better**. No refusal win is claimed."* The draft's App. A
  would print exactly those DISPLAY rows (T1 §1d: 0.9801, 0.9781, 0.8182) **without the finding that
  reads them**, and the draft's never-quote #14 restates the *"no impossible example was ever seen"*
  line that `N-I8-19` says is not a win.
- **Smallest fix.** Add as §11 item 17, in main text: *"**Refusal is novelty detection, not wall
  detection** — the store refuses reachable-but-unwritten pairs (0.9781) as often as walled ones
  (0.9801), at a false-refusal rate of 0.8182 on grounded-possible requests against a pre-registered
  0.20. No refusal claim is made. (`N-I8-19`.)"*

---

### MF16 ⛔ The paper's "own open problem" rests on one configuration of the five that fit the pin

- **The claim.** §5: *"at K ≥ 36 composition **collapses**"*; §11.3 / K8: *"No object composes and
  forgets inside 524,288 B."* Stated unconditionally, as the paper's central open problem.
- **The evidence it fails against.** `w8-single-design/LOG.md`, the 18:32Z `ward_balanced` sweep,
  lists **five** configurations that fit the 524,288 B pin at float64:

  | K | wells | A | span | total B | × budget | composition measured? |
  |---|---|---|---|---|---|---|
  | 36 | 56..57 | 4 | 144 | 523,680 | 0.999 | ⛔ no |
  | 38 | 53..54 | 7 | 266 | 523,744 | 0.999 | ⛔ no |
  | 40 | 51..52 | **9** | **360** | 518,080 | 0.988 | ⛔ no |
  | 42 | 48..49 | 12 | 504 | 522,704 | 0.997 | ⛔ no |
  | 48 | 42..43 | 18 | **864** | 521,984 | 0.996 | ⛔ no |

  The **only** configuration whose composition was measured is **K = 40 with A = 7, span 280**
  (23:22Z: *"Span ranks 7 in every one of the 40 wells, sum 280"*; 519,794 B = 0.991×) — **below the
  sweep's own K = 40 row**, and below the composing object's realised span of 308. Three of the five
  that fit the pin (spans 360, 504, 864) carry **more** deployed span than the object that composes
  at 0.8918. ⛔ The K ≥ 36 threshold is also **dtype-conditional**: the same hub's earlier sweep
  (`LOG.md:99`) prints a **float32** matched-bytes point — **K = 21, A = 94, 517,952 B = 0.988× the
  pin, span 336** — excluded by argument (*"the peer measured downdate residuals of 1e-10..5e-10 …
  and reports float32 would not hold them; **I accept that measurement** and treat f64 as
  required"*) and **never measured**.
- **Smallest fix.** Two qualifiers, no runs. (i) §5's row 3 becomes *"at K = 40, A = 7 (span 280,
  0.991× the pin — **the one configuration at K ≥ 36 whose composition was measured**) composition
  collapses to 0.0648"*, footnoting the four unmeasured pin-fitting configurations and their spans.
  (ii) K8 and §11.3 become *"**No object with a float64 sufficient statistic** composes and forgets
  inside 524,288 B, on the one such configuration measured."* ⭐ This is still the wave's answer; it
  is just the answer the evidence supports. Left unconditional it is the single easiest thing for a
  referee to break, because the counter-table is in the paper's own source LOG.

---

## 3. ⚠ SHOULD-FIX — twenty items

| # | claim in the draft | what the artefact says | smallest fix |
|---|---|---|---|
| **SF1** | §4: *"the inherited field store's own **`oracle_span`** sits below the bar (row 4 −12.531 / −5.224 / −5.653)"* | those are **`oracle_span_field`**'s numbers (40.9388 / 50.4286 / 54.2041 − `mlp_180k`). `oracle_span` reads 44.1837 / 50.8980 / 53.0000 ⇒ **−9.286 / −4.755 / −6.857**, the pair that replicates wave 4's −6.966. Two different ceilings. *(`READ-TABLE.json`; `w5-compose-field/LOG.md` §7 items 2–3)* | use `oracle_span_field` in §4, as K6 already correctly does |
| **SF2** | App. E's ceiling ladder lists five rungs with no byte column | `oracle_span` and `span_head` carry **`address_bytes = 1,766,400`** — **9.3×** the arm's own book and **3.4×** the store pin; every other rung carries 0 | print `address_bytes` per rung. A byte-ledger paper cannot ship a ladder whose rungs differ 9× in book size silently |
| **SF3** | §3: *"a variance-share admission cut with **no free parameter**, `tau = 4.941056e-16 / 1.1920929e-07`"* | `w8-downdate/LOG.md:840-855`: `delta_rel_worst = 4.9411e-16` **is test seed 2's leg worst** — the failing leg. The spoke is transparent and argues that rounding up would be anti-conservative, but the constant is calibrated on the held-out TEST legs | keep "no free parameter"; add *"its numerator is the worst accumulate-order perturbation measured across the four legs, including the failing one; the spoke declined to round it up because that would have made seed 2 more likely to pass"* |
| **SF4** | §3 / App. D describe the cut; its invoice is not printed | `w8-downdate/LOG.md:910-922`: directions dropped **3 / 1 / 4 / 5** (val0 / test0 / test1 / test2), worst variance share lost **8.94e-09**, and *"`rank/d` in well 3 moves from 0.31 to **0.21** on test seed 2, so the deployed span in that well **shrinks by a fifth**"* | print the invoice row in App. D. It is the only place the exactness claim costs content |
| **SF5** | §5 and never-quote #6: the carry *"rewrites **127,488 B** of the address book's 158,664 B"* | **superseded.** `w8-downdate/LOG.md:972`: *"The carry's byte cost falls slightly, 127,488 B → **125,952 B**, because the deployed span is one direction narrower"*; its FINAL REPORT (`:1067`) prints **125,952**. `w8-single-design/LOG.md:631` still carries the stale 127,488, which is the route by which the draft inherited it | print **125,952 B of 158,664 B**, noting 127,488 was the pre-cut figure |
| **SF6** | §7: *"the swap loss REPAIRS that damage (`A″` positive on 4 of 6, **two at p ≤ 0.0003**)"* | among the four positive `A″` rows the two-sided sign-test p's are 0.0001751, 0.000198, 0.000346, 0.0001564 ⇒ **three** at p ≤ 0.0003. Inherited verbatim from `TABLES.md` | write "three at p ≤ 0.0003" (or "four at p ≤ 0.00035") |
| **SF7** | §7: *"the learner is at 0.000 either way"* | `rival_mlp_180k @ alone` on **row 3** is **0.006** (0.000 / 0.017 / 0.000) | "at or within 0.02 of 0.000" |
| **SF8** | §8.2: *"49 corners × 3 seeds, **192,975 rows**"* | `EVIDENCE.md` §2.3: 192,975 is the **total**; *"of which **41,493 / 41,405 / 41,358** rows sit in the 49 **scored** corners"* (= 124,256), the remainder in 3 aux + 4 control corners | *"192,975 rows across 56 corners, of which 124,256 in the 49 scored corners"* |
| **SF9** | §8.3: *"the instrument floor … median **0.0349** ⇒ B1's 0.10 bar is ≈ 3× the instrument's own floor"* | `EVIDENCE.md` §2.2 gives **median 0.0349 / p90 0.0749 / max 0.1601** over 17 cells — **the max exceeds B1's own 0.10 bar**, at `h/h* = 1.42`, the exceptional point | print all three, and state that the bar is safe on the **median** reading, which is what B1 scores |
| **SF10** | §8.2: *"beats a one-parameter rival fitted with sight of the held-out measurements … (B2 mean paired Δ −0.5968 / −0.5796 / −0.5982)"* | the rival's own level is in the source: FIT1 median error **0.640132 / 0.623155 / 0.641281** against the CRR's **0.0433** — 15× worse, i.e. it is the naive `C/(εμ)²` scaling. §C.1 does not apply, so FIT1 is the only floor there is | print the rival's own error beside the delta, so the reader can judge whether it is a strawman rather than being asked to assume it is not |
| **SF11** | §8.4: *"with **all** the error in the single bucket 0.5–2"* | the buckets are 0.0008 / 0.0264 / **0.1482** / 0.0454 / **0.0262** — the `> 10` bucket is 33× the `< 0.1` bucket. Inherited verbatim from `EVIDENCE.md` §2.5 | *"the error is dominated by the single bucket straddling the exceptional point (0.1482 against 0.0008–0.0454 elsewhere)"* |
| **SF12** | §6's main table prints means only | row 3 conjunction CLU = 0.0284 (**0.0383 / 0.0469 / 0.0000**) — one of three seeds is **exactly zero**; row 2 CLU = 0.1279 (0.1590 / 0.0597 / 0.1651), a 2.8× spread. The charter rule is *"all three seeds or none — no row reports a best-of"* | print the per-seed triples in the main table, as `TABLES.md` does. A main-text mean concealing a 0.0000 inverts the rule's purpose |
| **SF13** | Main Table 2 (w5 composition) and main Table 4 (compose-XOR-forget) both report `pca_tensor` composition | Table 2: **63.612 / 64.327 / 67.653 distinct valid novel tuples per corner**, test, 49 + 60 paired corners. Table 4: **0.8941 valid mean**, val, 24 corners, one arm. `63.612 / 72 = 0.8835` — close enough to 0.8941 that a reader will try to reconcile them | one line under Table 4: *"a different metric (valid mean) on a different split (val, 24 corners); not comparable with Table 2's tuple counts"* |
| **SF14** | §5: *"the covariance costs **`Σ_k d_k²`**"* | it is the symmetric triangle at float64: `Σ_k d_k(d_k+1)/2 × 8 B + D×8 + K×8` (`w8-single-design/LOG.md:55,89,121`). The draft's form overstates the count by 2× and understates the dtype by 2×, and **K ≥ 36 is entirely a float64 fact** (*"Below K=36 **no atoms fit at all** — the **float64** statistic alone exceeds 524,288 B"*, `:148`) | print the cost function with its dtype. Without it the reader cannot check K ≥ 36 and derives K ≥ 32 |
| **SF15** | §G: *"⚠ **Two** open items the registry fold raised for the Hub"* | `w10-registry-fold/FINDINGS.md` §5 raises **six**. The draft carries §5.1 and §5.2 and drops §5.3 (two wave-8 Advisor questions carry no ruling), §5.4 (live budget: generation-table run 3 and single-design 3/3 unspent), **§5.5 (spot-check `N-I8-5`'s byte ledger against `DOWNDATE-READY*.json`)** and §5.6 (tiers are CURATOR-PROPOSED and never self-confirmed) | carry all six. **§5.5 is the spot-check that produces MF3, MF6 and MF7** — the registry asked for it and the draft dropped it |
| **SF16** | §F lists eight gaps, F1–F8 | `NOVELTY.md` §E names the one outstanding pre-submission action and the draft does not: *"It is no longer the cheapest outstanding insurance — that is now **a careful read of the two 2024 works in §C.1 and §C.4 in full**"* | add F9 with that sentence and its path |
| **SF17** | §3 stands alone; the deletion's cost is four sections away in §11.4 | see MF5 | add the cross-reference even if MF5's sentence is judged too strong |
| **SF18** | §6.1(i) *"Every piece of every answer is already written"* | the source's own clinching number is omitted: *"it explains why a same-maze fragment table reaches **0.5105** while every parametric arm sits at 0.13-0.18"* (`w9-maze/LOG.md:297`) | add it, with the charter-B.3 caveat from MF11 |
| **SF19** | never-quote #3: *"Never merge T1 with T3 (**same environment**, different budget)"* | different **environment** as well (MF10) | fix the parenthesis |
| **SF20** | K1 cites the registry's `N-I8-1` band | the registry's §2 row prints CMAZE seed 2 as **8.2e-16**; the artefact says **7.818e-16**. The draft uses the artefact's value and is **right** — but the Head should know the registry row is off, because it is the row the claims matrix inherits | flag to the Hub for the registry; no change to the draft |

---

## 4. ○ NICE-TO-HAVE — eight

1. **§6 / K15:** on seed 2 `unseen_pair` the **store-span ceiling (0.7018) exceeds the pipeline
   ceiling (0.6881)** — that cell *is* the 0.0138 maximum. A referee will ask how a projection
   ceiling sits above the pipeline it is a sub-case of. One sentence pre-empts it. *(recomputed from
   `w9-maze/runs/scored3_seed2_test.json` `/pipeline_ceiling_rows`, `/span_ceiling_rows`)*
2. **§9 requirement 4** cites *"§5 of this paper"* as the forcing measurement; `EVIDENCE.md` §5.3
   item 8 cites **R2** (the w5 ≈18.6-tuples/corner price and the coupling hole). Cite both.
3. **§9 requirement 1** omits the source's own qualifier: *"Under an optimality-gap metric O3 only
   falls to **0.9367**, still above the 0.90 pass line."*
4. **§9 requirement 3** omits GYM-1's matched-bytes stitching **0.8520** and GYM-2's **T3 0.0003
   against GYM-1's 0.7027**, both of which sharpen *"exact-key lookup is not a tripwire"*.
5. **K25** says *"six orthogonal sweeps return zero"*; `NOVELTY.md` §A says the orthogonal sweep is
   **21 title/abstract queries** over all of OpenAlex, of which §B lists **six** returning zero. Say
   "six of the sweep's 21 queries".
6. **§11.5** drops the name the source gives the failing precondition: *"canonical
   **Blelloch–Golovin** placement"*.
7. **§8.5** already carries *"the PREDICTED curve's own argmin over `γ*_exact` is 1.0000 on every
   seed"*; adding V5's own closing line, *"the instrument gap is a level, not a rate"*
   (`EVIDENCE.md` §2.6), finishes the argument in the source's words.
8. **§11 item 8** could carry `w5-advisor-report.md` §3's own premise correction: *"'96 unseen
   conjunctions' is a **val** figure — **the scored split has 3**"*, a sample-size caveat on both
   refusal rows.

---

## 5. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — checked against the result

> **The task's falsifier:** *"You return no must-fix items. ⭐ That is not a pass — re-read §1 and §2
> and check the arithmetic in the byte ledgers, because a skeleton assembled this fast has them."*

**It does not fire: sixteen must-fixes, and the byte ledgers hold nine of them.** Against the task's
six named push-points:

| push-point | result |
|---|---|
| **1** every number traced to `/c/chlu/evidence/outputs/**` and `/c/chlu/outputs/**` | done exhaustively. **Two numbers trace to nothing current** — the `−0.01 %` parameter match (MF12, corrected at 01:58Z) and the `127,488 B` carry (SF5, superseded at `LOG.md:972`). One traces to a **different field of the same file** than the source spoke's own headline (MF8). |
| **2** tables merging compute budgets / decoder recipes | ⭐ **the draft is disciplined here.** T1/T2/T3/T4 are kept apart, the `w8_v1_short` instrument is walled off by charter, and the two deletion legs are not merged. ⛔ But it merges two **environments** under one label (MF10) and two **metrics** across adjacent main tables (SF13). |
| **3** the novelty STOP | ⛔ **breached at §1.2** (MF1). The draft's own §10 paragraph survives the STOP; the pointer it hands the Head does not. |
| **4** negatives in the MAIN TEXT | ⭐ **the draft's strongest suit.** §5, §6, §7 and §11 are all main text, and the route-task failure and the compose/forget incompatibility lead their own sections. ⛔ One tier-A negative is missing entirely (MF15), and one bar's printed max is dropped while its sibling's miss is carried (below). |
| **5** every CMAZE composition claim carries its own-split BLIND-SEQUENTIAL | ⛔ **fails twice** — the caveat columns are slotted to App. A while the rows sit in main text, and **row 4 has no BLIND column in any artefact** (MF14). |
| **6** attack the deletion claim | ⛔ **it does not survive as written.** "Exact" is undefined (MF4); the downdate-vs-refit comparison holds the frozen renderer fixed, so it is not retrain-equivalence (MF4); the byte accounting does **not** net the address book in — correctly — but it never nets or even prints the **sufficient statistic at 24.5× the pin** (MF2, MF3), which is the number that decides whether the claim is about a 524,288 B object at all. |

### One item promoted out of should-fix, because it is the exact twin of a miss the draft *does* carry

⛔ **B3's printed max over γ is absent from the draft.** `EVIDENCE.md` §2.3 prints it on the same bar
table as B1/B2/B4: *"B3 … **max over γ** (printed, the stricter reading) | **59.1158** | 0.1612 |
**1883.2934** | ⚠ printed, bar not moved"* — against a bar of ≤ 0.10. The draft gives **B4**'s
three-seed miss a headline slot (§8.5, *"carried, not footnoted"*) and drops B3's, which misses by
**four orders of magnitude on two of three seeds**. A section whose stated discipline is "carry the
miss" cannot carry one miss and drop a larger one.
**Smallest fix:** add to §8.5 — *"B3 passes on the median reading (0.0357 / 0.0282 / 0.0343) and its
**max over γ** is printed and not moved: **59.1158 / 0.1612 / 1883.2934**. The median is the scored
statistic; the max is carried because the same file carries B4's."*

---

## 6. ⭐ WHAT THE PAPER SHOULD KEEP EXACTLY AS IT IS

So the Head does not over-correct. These survived a hostile read:

1. **The failure-agreement paragraph (§3).** Two implementations reproducing the *same seed-2
   failure* with different code and different sign conventions (32.90 / 35.46 ulp vs 14.377 ulp) is
   genuinely unusual evidence, and the draft is right that it is stronger than agreement on success.
   ⚠ One clause must be added: **both implementations also found that failure sits below the
   object's own reproducibility floor** — spoke, 4.746e-06 floor against 4.227e-06 observed;
   single-design, 36.9305 ulp floor against 14.4518 ulp observed
   (`w8-single-design/LOG.md`, 21:30Z and 22:20Z; *"Seed 2's subspace is not a function of its
   data"*). ⭐ With that clause the paragraph becomes a statement about **content well-posedness**,
   which is a better finding than a downdate failure.
2. **§6.1, plan-vs-latent.** Every cell verified against `w9-maze/LOG.md:288-346`; the controls
   (agent's-plan-wins 0.5722 / 0.4250 / 0.4252; target's-plan-plus-random-half 0.3557 / 0.3700 /
   0.4065) are exact, and the REFERENCE-not-an-arm caveat is correctly placed. **This is the best
   thing in the paper.**
3. **The `oracle_content_items` floor choice (§4).** Declining the flattering 0.000 lookups for the
   informative 32.388 / 39.122 / 41.776 oracle floor is the right call, and it was pre-declared in
   Amendment 1 before any held-out number existed.
4. **App. M and `verify_tables.py`.** I re-ran it: 27 / 27 / 240 / 720 / 5, PASS. Very few ICLR
   papers can do this and it should be said in the paper, not only in an appendix.
5. **§8.1 opening the physics section on the scope limit**, with `pca_settle` at
   −39.959 / −46.918 / −39.061 refuting the scoreboard reading on the same object and the same seeds.
6. **C6, C7, C8** — the +1.1385 reporting gap, the *"do not write D18 up as a confirmation"* ruling,
   and the tautological `d14_confinement` field — all reproduced faithfully from `EVIDENCE.md`
   §§4.2–4.4, and all exactly the kind of thing a referee is never told.
7. **§D (deleted claims).** Deleting rather than flagging unsourced sentences is correct, and the
   Campaign-2 skeleton exclusion (`iclr-long-skeleton.md` is a C2-era planning document with a
   PENDING spine) is the right call and worth keeping visible.

---

## 7. ⛔ THE CONTRIBUTION PARAGRAPH, REWRITTEN TO SURVIVE THIS REVIEW

Offered as a diff target, not as prose the Head must accept. Every change is forced by a must-fix
above; nothing new is asserted.

> We show that a generative store whose content is **derived from a retained sufficient statistic**
> admits deletion of the **store** that is exactly the content refitted from the retained rows,
> **holding the frozen encoder and decoder fixed** — verified by two independent implementations on
> two environments, three seeds each, on held-out test data, with no refit performed, no pass over
> retained data and items never kept **[K1, K2]**, at the cost of a statistic that is 2.2 % of the
> deployed read in one environment and **24.5× the byte pin in the other** **[MF3]**; that the two
> implementations agree not only on where the mechanism works but on where it breaks, and that where
> it breaks the residual is **below the object's own reproducibility floor** **[K3]**; that the same
> store class composes, beating a parameter-matched trained learner on both composition rows on all
> three seeds at **matched total deployed bytes (±0.5 %)**, with the store pinned at 524,288 B
> **[K4, K5, K6]**; and that at that pin, **for a float64 statistic and on the one configuration
> measured**, the two properties are incompatible, for a reason we measure rather than argue
> **[K7, K8, K9]**. The composition result does not transfer to an exact-judge route task on
> **CMAZE-2**, and we isolate why: the read composes latents where the task composes plans — the
> same two halves compose to 1.0000 in plan space and 0.13 in latent space **[K11–K15]**.

⭐ It is longer, and it is the version that cannot be broken by opening one JSON.

---

## 8. WHAT THIS SPOKE DID NOT DO

- ⛔ **No scored run, no model fitted, no number produced.** The only thing this spoke executed is
  `verify_tables.py` (read-only, ~3 s, no scored budget), plus read-only JSON and log inspection.
- ⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
- ⛔ **No background job was launched**, none was left running, and **no box-wide kill** was issued or
  was possible from anything this spoke ran.
- ⛔ **Nothing was written outside `/c/chlu/outputs/w10-referee/`** and this spoke's own worktree
  (branch `w10/referee`, under `projects/w10-referee/`). **No `.claude/` directory was created.**
- ⛔ **DECLARED NOT-CHECKED, each with its reason — never a silent omission:**
  1. **The five figure PNGs were not opened as images.** Their **hashes** were re-verified by
     `verify_tables.py` (5 of 5) and their source manifests read, but no figure's *content* was
     inspected — so the draft's figure-slotting claims are checked for existence and provenance
     only, not for what the figures show.
  2. **`claims_matrix.md`'s 13 `CM-I8-*` rows were not read cell by cell.** The 21 `N-I8-*` registry
     entries and `FINDINGS.md` were. Where the draft cites `CM-I8-1(b)`, I verified the *content* of
     that claim against the artefacts rather than against the matrix row.
  3. **The V1 / V2 / V5 shorts were not opened.** Every `[shorts]` status in `EVIDENCE.md` §3 is
     taken on that file's provenance labelling, which is what the label exists for.
  4. **`w8-encoder-loop/CROSS-ARM.json` was not dumped directly.** TABLE 2's cells were checked
     against `TABLES.md` and re-verified en bloc by `verify_tables.py`'s 720-value pass.
  5. **The gym LOGs (`w5-gym1`, `w51-gym2`) were not opened.** §9's numbers were checked against
     `BENCHMARK-STATEMENT.md` and `EVIDENCE.md` §5, which are this wave's own restatement of them.
  6. **The three sibling w10 spokes' bundles were not verified.** Out of scope; each spoke verifies
     its own.
