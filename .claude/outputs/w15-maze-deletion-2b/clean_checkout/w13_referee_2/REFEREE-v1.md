# REFEREE-v1 — adversarial re-review of `w13-draft-revise/DRAFT-v1.md`

**w13 `spoke/referee-2` · agent `paper-referee` · 2026-09-22.**
**Scored runs spent: ⛔ ZERO. Compute spent: ⛔ ZERO fitted models, ZERO new measurements, 1 thread.**
This spoke produced no experimental number. Every figure below is quoted with the path that holds it,
and every recomputation below is arithmetic on fields already on disk.
⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
**Target reviewed:** `/c/chlu/outputs/w13-draft-revise/DRAFT-v1.md` (124,769 B, 1,308 lines).
**Gate discharged before start:** that spoke's `DRAFT-v1.md` **and** its `END_UTC.txt` were both on disk.
**Prior verdict re-tested:** `/c/chlu/evidence/outputs/w10-referee/REFEREE-v0.md` — REJECT, 16 must-fix.

---

## 0. WHAT THIS REVIEW ACTUALLY DID

⛔ Not a read-through, and ⛔ not a check of the draft's own disposition table. Each of MF1–MF16 was
tested **against the artefact**, not against the draft's claim about the artefact. The new headline —
the K\* sign flip — was re-derived from its own JSON, and its **attribution** was attacked.

| I opened | and did what |
|---|---|
| `w13-draft-revise/DRAFT-v1.md` | read whole, §0–§J |
| `w13-seeds-and-tables/TABLES.md` + `TABLES.json` | read T1, T4, T5, T6, T7, T8, T9, T11, T12, the completeness gates, the NOT-RUN register; **re-ran `verify_tables.py` myself** |
| ⭐ `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` | dumped **all 45 leg records** (5 rungs × 3 legs × 3 seeds): store, `learned_1x`, margin, `learned_1x_target_params`, `ledger_four_part`, `controls`; plus `/verdict`, `/seed_matched_K5_vs_Kstar`, `/margin_swing_decomposed`, `/axis_label` |
| `w12b .../kstar_ladder/*/LEG.json`, `rows_learned_1x.json`, `rows_single_design_K*.json`, `rows_*__random_atoms.json` | dumped train recipe, fit rows, corner ids, control `form` strings |
| `w12b .../NR6-T7.json`, `FINDINGS.md` §§1–13 | read whole, including §12 LIMITS |
| `w12-composition-ladder/BYTES.json`, `AMENDMENT-1.md`, `LOG.md`, `runs/*/rows_*.json`, `runs/*/LEG.json` | dumped `well_sizes`, `p2_covariance_bytes`, `p2_if_symmetric_triangle_float32`, all 15 `learned_1x` param counts, all per-corner score vectors |
| `w12-single-design-crossover/LEDGER-ACROSS-K.json` | checked the `SMOKE_NOT_A_CLAIM` flag and the ledger rows |
| `w8-downdate/DOWNDATE-READY-test.json` | re-checked the ulp fields, the floors, `cut_is_marginal`, `a1_subspace` |

### ⭐ What reproduced cleanly — do not re-litigate these

- ⭐ **`verify_tables.py` PASSES on a fresh run by this spoke**, and the draft's App. M counts are exact:
  **52 sources registered · 52 source hashes re-verified · 872 cells · 1,786 per-seed values** (3 via a
  declared reduction) **· 8 figures, 8 hashes · 10 NOT-RUNs · 19 corrections · PASS.**
- ⭐ **All 15 K\* per-seed margins re-derive positive**, minimum **+3.117** (n = 2, seed 1), from
  `KSTAR-LADDER.json` `/legs/{n}/kstar_K32/{s}/margin`. The flip is real as a number.
- ⭐ **The seed means the Hub quoted reproduce exactly:** +4.356 / +14.072 / +10.022 / +8.356 / +7.978.
- ⭐ **The w8 four-part ledger closes to the byte on all three seeds** (524,288 + 12,327,096 + 11,488 +
  5,288 + 1,322 = 12,869,482, and the seed-1 and seed-2 sums likewise), and 12,869,482 / 524,288 =
  24.5466 exactly.
- ⭐ **The K\* per-seed ledgers sum correctly** (516,096 + 560,608 + {1,060,864 | 1,054,104 | 1,058,264}
  + 6,610 = {2,144,178 | 2,137,418 | 2,141,578}).
- ⭐ **T7's three-seed row is arithmetically right:** 32.350 ± 0.815 and 67.177 ± 1.320 are the mean and
  sample SD of (31.600 / 32.233 / 33.217) and (66.959 / 65.980 / 68.592).
- ⭐ **`3,131,564 – 3,260,030` is the exact min/max of `param_count` over all 15 ladder `learned_1x` legs.**
- ⭐ **13 of 15 K\* control legs read exactly 0.0000 and the two exceptions are n = 3 seeds 0 and 1 at
  0.008333**, precisely as the draft states.

⇒ **The draft's arithmetic on the numbers it chose to print is good, and its discipline is now
unusual for this field.** Everything below is about **which convention those numbers are priced in**,
**which opponent they are measured against**, and **two of the sixteen that are not actually closed.**

---

## 1. ⛔ VERDICT — simulated

> ## **MAJOR REVISION. ⛔ No longer a reject — and ⛔ not yet an accept.**
> **The revision that gets it there costs 0 scored runs and 0 fitted models. Every new must-fix below
> is arithmetic on a field already on disk, or a column the source artefact already carries and the
> draft dropped.**

| reviewer | v0 | v1 | the one thing that decides it |
|---|---|---|---|
| **R1 — machine unlearning** | 3 (reject) | **7 (accept)** | *"Exact" is now defined store-scoped before the table, the frozen encoder/decoder are declared outside the deletable object, the statistic's 24.55× is printed on the same page as the claim, and §3.6 makes the scope a **measurement** (32.25 → 32.25) rather than a definition. This is the best-defended unlearning claim I have refereed this cycle.* Docks one point for §I calling the four-part total "a statistic" (**N7**). |
| **R2 — compositional generation** | 5 (marginal reject) | **5 (marginal reject)** → 7 after N1–N4 | *The composition section's headline reversed between submissions, and the paper's own attribution of the reversal does not survive its own byte convention. The K = 5 defeat is measured against an opponent priced by `Σ_k d_k²` while the K\* win is priced by the symmetric triangle — and the draft's SF14 closure says, two sections later, that `Σ_k d_k²` is the wrong cost function.* |
| **R3 — benchmarks / empirical rigour** | 6 (marginal accept) | **7 (accept)** | *`verify_tables.py`, the four-part never-netted ledger, the declared NOT-RUNs with launch lines, and negatives in main text are above the field. Docks it for a control gate that changes strength across the comparison that carries the headline (**N4**) and for one ratio printed against three totals (**N5**).* |
| **AC** | reject, invite resubmission | **major revision** | *Fourteen of sixteen prior must-fixes are genuinely closed against the artefacts. Two are not (**MF2**, **MF14**). The new result is load-bearing and its attribution is currently unsound; the fix is bookkeeping, not science.* |

⛔ **The falsifier the task set for THIS spoke does not fire:** eight new must-fixes, and **the byte
ledgers hold five of them**.

---

## 2. ⭐⭐ MF1–MF16 — CLOSED / PARTIAL / OPEN, item by item

⛔ **The draft claims all sixteen CLOSED. Fourteen are. Two are not.**
Each row states what I opened and what it said.

| MF | status | the evidence I checked it against |
|---|---|---|
| **MF1** — STOP-#1 pointer | ⭐ **CLOSED** | §1 slot 1.2 deletes *"the survivable form is verbatim there"*, prints the STOP warning at the pointer, and repoints at §I. I grepped §I: the strings *"request-indexed tensor head"*, *"tensor-product label conditioning"* and *"multilinear"* are **absent**. The Head can no longer be handed TensorFaces. |
| **MF2** — *"whose whole content is a sufficient statistic"* | ⚠ **PARTIAL** | The noun phrase **is** fixed (§I: *"content **derived from** a retained sufficient statistic … downdating the statistic **induces** the refit"*), and §A's disposition quotes the ratios correctly (2.19 % / 2,261–2,348 %). ⛔ **But MF2's second half — "state the statistic's byte cost in the same sentence" — is not done: §I's sentence prints `24.55×`, which is the FOUR-PART HONEST TOTAL, not the statistic.** The statistic alone is 12,327,096 / 524,288 = **23.51×**. See **N7**. |
| **MF3** — no byte row on the lead table | ⭐ **CLOSED** | §3.3 prints four parts on all three legs, never netted, with the 16,384 B declared residual; I re-summed every row and all close to the byte. The framing sentence v0 asked for is there. ⛔ New defects **in** that row are **N1**, **N5**, **N6** — but the omission MF3 named is gone. |
| **MF4** — *"exact"* undefined | ⭐ **CLOSED, and strengthened beyond the fix I asked for** | §3.0 prints the store-scoped definition **before** the table, verbatim from a citable field (`TABLES.json` `/exactness_definition`), with `ae_train_items.eval_bank_train` = 543 / 529 / 552 as the reason; §2.5 declares the frozen encoder and decoder **outside the deletable object**; exact-JUDGE and exact-DELETION are separated. ⭐ And §3.6 converts the scope from a definition into a **measurement**: at K\* the deletion of all 11,044 red rows leaves the 20 red-requesting corners at **32.2500 → 32.2500**. ⭐ That is a stronger answer to R1 than the sentence v0 asked for. |
| **MF5** — exactness with no cost | ⭐ **CLOSED** | §3.5 sits at the end of §3: 0.1024 (0.1569 / 0.0702 / 0.0802) → 0.0000; eviction −2.0899 / −0.9101 / −6.7416 against the hole's exactly 0.0000; the programme's own §6.3 ruling quoted. SF17's cross-reference is in place. |
| **MF6** — caveats inside the cited gate file | ⭐ **CLOSED** | §3.2, in T4's caption: `cut_is_marginal` True on all three seeds; `a1_subspace.passed_every_seed` False, `seeds_failing ["2"]`, 60.36282 ulp against its own 66.11072 ulp floor; `all_three_assertions_passed` **excludes** it. I re-dumped all four fields from `DOWNDATE-READY-test.json`; they read as printed. |
| **MF7** — RESULT label on a "no claim" artefact | ⭐ **CLOSED** | §3.1 leg 2 is **ASSERTION BATTERY (object property) · 3 seeds · TEST · 0 SCORED RUNS**, with NR3 named, and K2 in §B carries the same class. ⭐ The draft's own line — *"the artefact was not relabelled to match the table; the table was relabelled to match the artefact"* — is the right instinct and should survive into the paper. |
| **MF8** — the narrower of two per-seed fields | ⭐ **CLOSED** | §3.1 prints **4.5475e-13 / 1.1255e-09 / 2.8422e-14** under a header naming **both** deletions (the pinned 11,024 rows and the random 64), keeps the narrower field beside it **labelled as such**, and keeps the ulp figure (`worst_in_float32_ulp_at_unit_scale` = 0.009442, which I re-read from the JSON). `fig4` is redrawn from the corrected field and its hash re-verifies. |
| **MF9** — one of two unresolvable legs declared | ⭐ **CLOSED** | §3.1 prints both against their own floors (2.1268e-13 / 4.2817e-09 / 5.6843e-14) with a `resolvable at all?` row reading **True / False / False**, and states it the strong way. |
| **MF10** — CMAZE-2 mislabelled CMAZE-1 | ⭐ **CLOSED** | §6.0 defines CMAZE-2 in one line from the source's own words, relabels §6/§6.1/§6.2/§6.3, keeps T3 as CMAZE-1 proper, forbids sharing a table, and fixes never-quote #3's parenthesis (SF19). The 0.4980 / 0.5317 / 0.9037 triple is now readable. |
| **MF11** — *"FRAGMENT-LOOKUP 0.0000 everywhere"* | ⭐ **CLOSED, improved** | §6.3 prints cross-maze (0.0006 / 0.0000) and same-maze (0.0741 / **0.5114** unweighted seed mean, **0.5105** pooled — correction C11) as two labelled quantities, with the charter-B.3 caveat. ⭐ And 0.5114 is promoted into §6.2(i) as the strongest exhibit, which is what SF18 asked for. |
| **MF12** — parameter match from a superseded `val` figure | ⭐ **CLOSED** | §2.3 prints **178,408 / 179,838 / 179,838 vs 178,952 = −0.304 % / +0.495 % / +0.495 %**, carries the spoke's own sentence that **on two of three seeds the arm carries MORE parameters than the bar**, and repoints to `w5-compose-field/LOG.md` **01:58 UTC**. |
| **MF13** — *"at 524,288 B"* | ⭐ **CLOSED** | §2.3 and §4.1 state the supported form: store pinned at 524,288 B tol 0, book declared beside it, **total deployed 713,632 / 719,352 / 719,352 against 715,808 = 99.70 % / 100.50 % / 100.50 %**. |
| **MF14** — row 4's missing BLIND column | ⚠ **PARTIAL** | Part (ii) is **CLOSED and well done**: row 4 carries *"⛔ no BLIND column on any seed — `/env_columns` carries exactly four keys"*, DECLARED NOT MEASURED, never a blank. ⛔ **Part (i) is NOT done.** MF14 asked that BLIND-SEQUENTIAL **and BLIND-FACTOR** for rows 2 and 3 be **moved into the main-text table**. §6.1's printed table has five columns — CLU, best learned baseline, lookup, `random_atoms`, pipeline ceiling — and **none of them is a BLIND column**. The BLIND-SEQUENTIAL numbers appear in **§6.3**, a separate sub-section, and ⛔ **BLIND-FACTOR's values appear NOWHERE in the draft**: I grepped `DRAFT-v1.md` for `BLIND-FACTOR` and got two hits, both prose instructions, no number. The artefact has them — `TABLES.md` §1c: BLIND-FACTOR **0.4762** (0.4564 / 0.4677 / 0.5046) on row 2 and **0.4720** (0.4447 / 0.4665 / 0.5048) on row 3. **The composition rows are still in one place and their caveat columns in another**, which is the defect MF14 named, one level finer. |
| **MF15** — the missing tier-A negative | ⭐ **CLOSED** | §11 item **17**, main text: 0.9781 (0.9721 / 0.9794 / 0.9829) vs 0.9801 (0.9606 / 0.9846 / 0.9952), false-refusal **0.8182 against a pre-registered 0.20**, the supervised refuser separating better, no refusal win claimed. `N-I8-19` cited. |
| **MF16** — the open problem on one of five configurations | ⭐ **CLOSED, and superseded in substance** | §5.1 prints the K = 40 / A = 7 / span 280 / 0.991× qualifier as *"the one configuration at K ≥ 36 whose composition was measured"*, footnotes the pin-fitting sweep rows and the excluded float32 point (K = 21, A = 94, 0.988×, span 336), and rewrites §11.3 as *"no object **with a float64 sufficient statistic**…"*. ⭐ And the nine-K sweep replaces the argument with a measurement: minimum honest four-part total **3.9942× at K = 24**. ⚠ One count error inside the closure — **SF-b** below. |

> ### ⭐ Score: **14 CLOSED · 2 PARTIAL · 0 OPEN.** That is a genuine revision, not a cosmetic one.
> ⛔ **But "all sixteen closed" is not true, and a disposition table that overstates by two is the
> first thing an AC checks.** §A must read **14 CLOSED, 2 PARTIAL**, with the two named.

---

## 3. ⛔ NEW MUST-FIX — eight

Per the acceptance criterion each names **the claim · the evidence it fails against · the smallest
change that fixes it**. ⛔ None requires a scored run, a fitted model or a new measurement.

---

### N1 ⛔⛔ The K = 5 defeat is priced by a covariance convention the draft itself declares wrong — and the source file carries the alternative in the next field

- **The claim.** §0.2 bullet 2 and §4.2: *"At **K = 5** the store loses to a 1×-matched learner at 4 of
  5 rungs and clears at none."* §4.4: *"the single design costs **12,523,008 – 13,041,864 B =
  23.886× – 24.875×** the 524,288 B reference; `pca_tensor` costs 1.358×–1.372× — **≈18× less**."*
  §I: *"a bridge arm attributes **62–79 %** of the swing to the K step."*
- **The evidence it fails against.** The ladder's opponent is sized as *(the store's four-part total) ÷ 4*
  (`w12-composition-ladder/AMENDMENT-1.md`). That total is dominated by the statistic, and the
  statistic is priced on the **full square**. I verified it closes to the byte on every seed from
  `BYTES.json` `/legs/*/single_design_K5/well_sizes`:

  | seed | well widths | `Σ d_k² × 8` | + means/counts 16,424 | `p2_statistics_bytes` on disk |
  |---|---|---|---|---|
  | 0 | 777 / 936 / 216 / 60 / 59 | 12,268,496 | 12,284,920 | **12,284,920** ✓ |
  | 1 | 814 / 880 / 210 / 72 / 72 | 11,931,712 | 11,948,136 | **11,948,136** ✓ |
  | 2 | 836 / 847 / 223 / 68 / 74 | 11,809,072 | 11,825,496 | **11,825,496** ✓ |

  ⛔ **Every other ledger in the programme is the symmetric triangle**, and those also close to the
  byte: the K\* object's `p2_covariance_bytes` = 549,120 = 32·(64·65/2)·8 + 2048·8 + 32·8 **exactly**;
  the crossover / bridge K = 5 object's = 3,380,064 = the triangle over five ~410-wide wells **exactly**.
- **⛔ And the draft says so itself, in its own SF14 closure, 80 lines later (§5.1):** *"the covariance
  costs the symmetric TRIANGLE at float64, `Σ_k d_k(d_k+1)/2 × 8 B + D×8 + K×8` — ⛔ **not `Σ_k d_k²`.**
  v0's form overstates the count 2×."* ⇒ **the one ledger in the paper still on the form the paper
  declares wrong is the one that sizes the opponent the store loses to.**
- **⛔ The source file even carries the corrected number.** `BYTES.json` `/legs/*/single_design_K5/`
  has a field named **`p2_if_symmetric_triangle_float32`** = **3,071,220 / 2,987,024 / …** — computed,
  stored, and never printed by the draft.
- **What it does to the headline.** Recomputing the ladder's own K = 5 ledger on the paper's own
  convention (pure arithmetic from `well_sizes`, zero runs), seed 0:

  | rung | four-part total, **as printed** | × pin | four-part total, **symmetric triangle f64** | × pin | 1× opponent, as printed | 1× opponent, corrected |
  |---|---|---|---|---|---|---|
  | n = 2 | 12,967,352 | 24.733× | **6,841,296** | **13.049×** | 3,241,838 | **1,710,324** |
  | n = 3 | 12,898,080 | 24.601× | **6,806,400** | **12.982×** | 3,224,520 | **1,701,600** |
  | n = 4 | 13,041,864 | 24.875× | **6,876,992** | **13.117×** | 3,260,466 | **1,719,248** |
  | n = 5 | 12,949,816 | 24.700× | **6,832,008** | **13.031×** | 3,237,454 | **1,708,002** |
  | n = 6 | 12,956,296 | 24.712× | **6,835,248** | **13.037×** | 3,239,074 | **1,708,812** |

  ⇒ §4.4's *"≈18× less"* becomes ≈9.5× less; §0.2's *"23.886×–24.875×"* becomes ≈13×; and the
  opponent the store is said to lose to is **1.9× too large** on the draft's own float64 convention,
  **3.5× too large** on the file's own float32 one.
- **Smallest fix.** Print the ladder's K = 5 ledger on the same convention as every other ledger in
  the paper — the recomputation above, from `well_sizes`, **0 runs** — and add one sentence to §4.2:
  *⛔ the K = 5 margins are measured against an opponent sized from a full-square statistic; on the
  symmetric-triangle convention this paper states in §5.1, the same object's four-part total is
  ≈13.0× the reference and its 1×-matched opponent carries ≈1.71 M parameters, not 3.24 M. The K = 5
  column is therefore not on the paper's own ledger and is labelled as such.*

---

### N2 ⛔⛔ w12b already fitted the same-convention opponent at K = 5, on the identical corners — and against it "loses at 4 of 5 rungs" becomes "within ±1.3 tuples at four rungs"

- **The claim.** §0.2, §4.2, §11 item 19, §I: the K = 5 verdict is *"loses at 4 of 5 rungs"* with
  per-seed margins **−4.567 / −2.217 / −1.567** (n = 2) through **−5.775 / −4.567 / −1.842** (n = 6),
  and *"the bridge STILL LOSES at 4 of 5 rungs — the identical pattern"*.
- **The evidence it fails against.** The bridge leg is not only a different construction; it carries a
  **1,029,776-parameter** `learned_1x` (`w12b .../kstar_ladder/{n}_seed0_K5/rows_learned_1x.json`
  `/summary/param_count`, target 1,029,508) fitted on **the same rung, the same seed, the same
  generation-2 cache, the same 32,768 fit rows, the same 20 epochs / 2,560 steps / lr 1e-3 / batch
  256**, and — I checked this explicitly — **on the identical corner set**: the `per_corner`
  `corner_id` lists of `w12-composition-ladder/runs/{n}_seed0/rows_learned_1x.json` and
  `w12b .../kstar_ladder/{n}_seed0_K5/rows_learned_1x.json` are **equal as sets**, 60 at n = 2 and 120
  at n ≥ 3. So the paired estimand is computable **with zero new runs**, and I computed it per corner:

  | rung (seed 0) | ladder store | opponent **as printed** (3.13–3.26 M params) | margin printed | opponent at **1,029,776 params** | ⭐ **margin, same convention** | corners won by > 1.0 |
  |---|---|---|---|---|---|---|
  | n = 2 | 28.6333 | 33.2000 | **−4.5667** | 29.8667 | **−1.2333** | 19 / 60 |
  | n = 3 | 56.9500 | 52.1333 | +4.8167 | 46.2917 | **+10.6583** | 78 / 120 |
  | n = 4 | 60.9333 | 67.1750 | **−6.2417** | 61.3250 | **−0.3917** | 46 / 120 |
  | n = 5 | 63.2417 | 68.0500 | **−4.8083** | 63.3667 | **−0.1250** | 38 / 120 |
  | n = 6 | 62.6917 | 68.4667 | **−5.7750** | 63.2750 | **−0.5833** | 38 / 120 |

  ⚠ **Stated honestly and against my own interest:** 1,029,776 is not *the* right opponent either —
  the honest bracket for the ladder object's 1× opponent is **938,413 params** (the file's own
  `p2_if_symmetric_triangle_float32`) to **1,710,324 params** (float64 triangle). 1,029,776 sits
  **inside** that bracket; **3,241,076 sits 1.9×–3.5× outside it**. The store still does not *clear*
  the +1.0 bar at four rungs. ⛔ **But "loses by 2–6 novel tuples per corner" and "loses by 0.1–1.2"
  are different papers**, and the draft prints only the first.
- **Smallest fix.** One appendix row, 0 runs: *ladder store K = 5 vs the same-convention 1× opponent
  already fitted by w12b, seed 0, identical corners: **−1.233 / +10.658 / −0.392 / −0.125 / −0.583***,
  labelled **1 seed · replication · not scored**, with the 938,413–1,710,324 bracket stated. Then
  §0.2 bullet 2 becomes *"at K = 5 the store fails the every-seed bar at every rung; the size of that
  failure is a function of how the statistic is priced, and against a same-convention opponent it is
  ≤ 1.3 tuples at four rungs."*

---

### N3 ⛔⛔ The bridge table drops the two columns its own source provides — and those columns say the construction step is the opponent shrinking, not the store improving

- **The claim.** §4.2's bridge table and §0.2 bullet 2: *"**62–79 % of every swing is the K step, not
  the construction**"*; *"Changing the construction alone flips nothing. The flip is the K step."*
  §4.2 also states *"the store's own score **rises at every rung** (+1.86 to +3.38)"* — which is true
  of the **ladder → K\*** end-to-end step, and I verified it (+2.106 / +1.864 / +3.378 / +1.862 /
  +2.047 as three-seed means).
- **The evidence it fails against.** `KSTAR-LADDER.json` `/margin_swing_decomposed/{n}/` carries
  **`store_ladder_K5`, `store_bridge_K5`, `store_kstar_K32`** and **`opponent_params{ladder_K5,
  bridge_K5, kstar_K32}`**. ⛔ **The draft prints neither, and both contradict the reading its table
  invites.** All seed 0, from those fields:

  | rung | store: ladder → bridge → K\* | **store across the CONSTRUCTION step** | opponent params: ladder → bridge → K\* | opponent score: ladder → K\* | ⭐ **share of swing that is the opponent falling** |
  |---|---|---|---|---|---|
  | n = 2 | 28.633 → 29.617 → 31.800 | **+0.98** | 3,241,076 → 1,029,776 → 534,866 | 33.200 → 25.133 | **71.8 %** |
  | n = 3 | 56.950 → 54.717 → 60.150 | ⛔ **−2.23** | 3,224,228 → 1,029,776 → 534,866 | 52.133 → 42.275 | **75.5 %** |
  | n = 4 | 60.933 → 60.308 → 65.242 | ⛔ **−0.63** | 3,260,030 → 1,029,776 → 534,866 | 67.175 → 52.542 | **77.3 %** |
  | n = 5 | 63.242 → 62.850 → 65.642 | ⛔ **−0.39** | 3,236,864 → 1,029,776 → 534,866 | 68.050 → 55.292 | **84.2 %** |
  | n = 6 | 62.692 → 60.725 → 65.908 | ⛔ **−1.97** | 3,238,970 → 1,029,776 → 534,866 | 68.467 → 56.533 | **78.8 %** |

  ⇒ ⛔ **Across the "construction step" the store's own score FALLS at 4 of 5 rungs while its margin
  improves at all five** — the whole construction gain is the byte-matched opponent being cut
  **3.15×**. And the "K step" is itself a further opponent cut of **1.93×**. ⛔ **Neither step is a
  clean manipulation of one variable: both are opponent-budget cuts, because the opponent is defined
  as the object's own total.** End to end, **71.8–84.2 % of the swing is the opponent falling** and
  only **15.8–28.2 %** is the store reading better.
- **⛔ The artefact's own `why` field says the construction step is compound** — *"the partition rule,
  the store class and **the covariance convention** changing together"* — and the draft quotes the
  first two and drops the third from its table caption, which is what makes **N1** invisible.
- **Smallest fix.** Add the two source columns (`store_*` and `opponent_params`) to §4.2's bridge
  table — they are already in the JSON — and one sentence: *both steps cut the byte-matched opponent
  (3.15× then 1.93×); across the construction step the store's own score falls at 4 of 5 rungs; end to
  end **72–84 % of the swing is the opponent falling** and 16–28 % is the store reading better.*
  ⭐ **Stated that way the finding is still real and is a byte-efficiency result — which is the paper's
  actual thesis — instead of a composition-depth result it cannot support.**

---

### N4 ⛔ The control gate is a **weaker control** at K\* than at K = 5, and the cited artefact's own LIMITS says so

- **The claim.** §0.2 bullet 4: *"At K = 5, `landscape_shuffled` and `random_atoms` are beaten by the
  arm's entire score at every rung … At K\*, **13 of 15 legs read exactly 0.0000** … ⇒ the store
  genuinely carries its score."* §4.5: *"the control gate was MEASURED at seeds 1 and 2, not assumed:
  all four controls read exactly 0.0000."* K14.
- **The evidence it fails against.** `w12b .../kstar_ladder/*/rows_single_design_K32__random_atoms.json`
  `/summary/form`, verbatim:
  > *"the crossover hub's control gate. **The ladder's STRONGER `random_atoms` — random atoms with the
  > address book REFIT against them — is a DECLARED NOT-RUN here**: the book costs about an hour of
  > single-thread CPU per leg at K\*."*

  and `w12b .../FINDINGS.md` **§12 LIMITS**, verbatim: *"**the ladder's `random_atoms` is the weaker
  post-fit form at K\***."* The strong form at K = 5 does **not** read 0.0000: it reads **0.25** novel
  tuples, judge accuracy 0.006944, at 671.7 s
  (`w12-composition-ladder/runs/2_seed0/rows_single_design_K5__random_atoms.json`); the weak form at
  K\* takes 24.1 s and reads 0.0. T8 prints the K = 5 control as a **paired delta** of +28.3833
  against an arm of 28.6333 — i.e. **not** "the arm's entire score", short by exactly that 0.25.
- **Why it is a must-fix and not a nit.** The draft's own framing is *"the brief's '0.000 throughout'
  is true at K = 5 and is not true at K\*"*. ⛔ **It is the other way round where it matters:** the
  K = 5 gate is the expensive control and does not read zero; the K\* gate reads zero because the
  expensive control was not run. A referee who reads "the controls collapse harder at K\*" is reading
  a comparison of **two different controls**, and the caveat lives in the LIMITS section of the file
  the draft cites eleven times.
- **Smallest fix.** One caption clause on every K\* control row: *⚠ `random_atoms` at K\* is the
  **post-fit** form (atoms randomised, address book **not** refit). The ladder's refit form is a
  DECLARED NOT-RUN at K\* (≈1 h single-thread CPU per leg) and reads **0.25**, not 0.0000, at K = 5.*
  And delete the phrase *"beaten by the arm's entire score"* in favour of the printed paired delta.

---

### N5 ⛔ The byte row prints **one ratio against three different totals**, and **two different ratios for the same object**

- **The claim.** §3.3's K\* row: address book **1,060,864 / 1,054,104 / 1,058,264**, honest total
  **2,144,178 / 2,137,418 / 2,141,578**, *"× the 524,288 B pin | **4.0897×**"* — a single value.
  §0.2 bullet 1 repeats *"**4.0897×** on the K\* object (`TABLES.md` T7 byte ledger)"*.
- **The evidence it fails against.** `TABLES.md` **T7 and T9 are ONE-SEED tables** (`seeds (n) | 1`,
  seed 0) and their 4.0897 is `2,144,178 / 524,288` for **that seed alone**. The three-seed totals come
  from `w12b .../FINDINGS.md` §4, which prints **three totals and no ratio**. Correct, per seed:
  **4.0897× / 4.0768× / 4.0847×**. ⛔ And §5.2 / `TABLES.md` T12 print the **same object** at
  **4.0877×** (2,143,138 B, address book 1,059,824) — the **val** pin leg, a fourth number.
- **Why it matters here of all places.** This is the row MF3 created and the row the paper's whole
  byte discipline rests on. ⛔ A four-part-never-netted ledger that prints a seed-0 ratio beside a
  three-seed total, and a val ratio two sections later without saying it is val, hands a referee the
  one objection this paper cannot afford.
- **Smallest fix.** §3.3's last column becomes **`4.0897× / 4.0768× / 4.0847×`** and its parts 1, 2, 4
  gain the note *"constant across seeds; only part 3 varies"* (which is the source's own sentence).
  §5.2's and T12's 4.0877× gains the word **val**.

---

### N6 ⛔ Two different objects are both called **"K = 5"**, 80 lines apart, with byte totals **3.1× apart** and no cross-reference

- **The claim.** §4.4: *"At **K = 5**, over 18 fitted legs: the single design costs **12,523,008 –
  13,041,864 B = 23.886× – 24.875×** the 524,288 B reference."* §5.1, same draft:
  *"at **K = 5** composition holds — **0.8918** vs the incumbent's 0.8941 — at an honest four-part
  ledger of **4,118,034 B = 7.8545×** the pin."* §5.2's table repeats the second as its K = 5 column.
- **The evidence it fails against.** They are different objects. The first is the w12 ladder's
  `DowndatableStore` on the **label-MI `factor`** partition (`BYTES.json` `/summary/single_design_K5`,
  well widths 777/936/216/60/59, full-square statistic). The second is the crossover / bridge
  `CovContentStore` on the **label-free `ward_balanced`** cut (`TABLES.md` T12 K = 5 row;
  `KSTAR-LADDER.json` `/legs/*/bridge_K5/0/ledger_four_part`, five ~410-wide wells, triangular
  statistic, 195,584 B book). ⛔ **The draft knows** — its own correction **C20** in §G reads *"the two
  conventions price the same statistic differently — full-square 12,284,920 B vs symmetric-triangle
  3,391,552 B — on different partitions"* — but C20 is filed as a **never-quote note aimed at the
  advisor report**, and §4.4 and §5.1 carry no cross-reference to it or to each other.
  ⚠ C20 is also itself imprecise: it compares two numbers that differ in **both** convention and
  partition, so it reads as a 3.6× convention effect when the convention alone is 2× (see **N1**).
- **Smallest fix.** Give the two objects names and use them everywhere — *K = 5 (MI partition,
  full-square)* and *K = 5 (ward, triangle)* — and add one line to each of §4.4 and §5.1 pointing at
  the other and at C20. ⛔ Without it a referee reads the draft's two K = 5 byte totals as a
  contradiction, which on the page is exactly what they are.

---

### N7 ⛔ The contribution paragraph calls the four-part honest total **"a statistic"**

- **The claim.** §I, the sentence most likely to be lifted verbatim into the abstract: *"…at the cost
  of a statistic that is **2.19 %** of the deployed read in one environment and **24.55× the byte
  pin** in the other **[K4]**."*
- **The evidence it fails against.** `TABLES.md` T4 byte ledger: 24.5466× is
  `honest_total_bytes / 524,288` = **atoms + statistic + request table + coupling + holes**. The
  **statistic alone** is 12,327,096 B = **23.5117×** the pin. ⛔ And the two halves of the clause quote
  **two different ratios** — statistic ÷ deployed read (w9) against four-part total ÷ pin (w8) — as
  though they were one quantity. That is the netting-and-labelling discipline the paper's lead result
  is built on, breached in the paper's lead sentence. It is also the half of **MF2** that is still open.
- **Smallest fix.** *"…at the cost of a statistic that is **2.19 %** of the deployed read in one
  environment and **23.5× the store** in the other, taking the four-part honest total to **24.55× the
  pin** **[K4]**."* Eleven words, and MF2 closes with it.

---

### N8 ⛔ The draft says **three different things** about where the K = 5 loss is printed

- **The claim.** §4.2 prints the K = 5 per-seed losses in a **main-text §4 table**. §C slots
  *"**Table 5 (main)** | the **K = 5 vs K\*** ladder verdict with the bridge column"*. §11 item 19 says
  *"the K = 5 reading the programme published first is **withdrawn from main text** and carried in the
  **appendix**"*. `TABLES.md` T8's own header reads *"**Placement: APPENDIX** — NOT main text: the main
  text carries ONE K, K\*, and never both (Head ruling, 2026-09-21T12:38:07Z)"*.
- **The evidence it fails against.** §11 item 19's sentence is lifted from `w12b .../FINDINGS.md` §13
  item 1, which is a **recommendation to the Hub**, not a ruling already taken — that section's own
  heading is *"WHAT THE HUB IS ASKED TO RULE ON"*. So the draft has converted an open question into a
  statement, and that statement contradicts its own §C.
- **Why it is a must-fix.** The Head writes prose from this file. The ladder loss at K = 5 is the
  paper's **largest negative**, and the only one of the three instructions that survives a careless
  read is *"withdrawn from main text"*. ⛔ Appendixing it is overclaiming by omission, and the draft
  currently authorises exactly that.
- **Smallest fix.** Delete *"withdrawn from main text"* from §11 item 19 and replace with: *the Head's
  one-K ruling governs which K the composition **claim** is stated at, not which K's **numbers** are
  printed; the K = 5 margins appear in main text beside K\*'s (§4.2, §C Table 5), and T8's appendix
  placement is the full 13-arm table, not the verdict.*

---

## 4. ⚠ SHOULD-FIX — twelve

| # | claim in the draft | what the artefact says | smallest fix |
|---|---|---|---|
| **SF-a** | §A: *"⭐ **All sixteen CLOSED.**"* | MF2 and MF14 are PARTIAL (§2 above) | *"Fourteen CLOSED, two PARTIAL — MF2's byte clause and MF14's main-table caveat columns"* |
| **SF-b** | §5.1 / §A MF16: *"the **four** unmeasured pin-fitting configurations (K = 36/38/40/42/48, spans 144/266/360/504/864)"* | that is **five** K values and **five** spans, and all five are unmeasured — the measured point is K = 40 **A = 7 span 280**, which is not the sweep's K = 40 row (A = 9, span 360) | write **five**, and say the measured K = 40 point sits *below* the sweep's own K = 40 row |
| **SF-c** | §0.2 bullet 4 / §4.4: *"`landscape_shuffled` and `random_atoms` are beaten by the **arm's entire score** at every rung"* | T8's `random_atoms` paired delta is **+28.3833** against an arm of 28.6333 — the control scores **0.25**, not 0 | quote the paired delta, not "the entire score" (and see **N4**) |
| **SF-d** | §0.3: *"Held out … **288,000 train / 192,000 test**"*, offered as the replication's scale | `AMENDMENT-1.md`: both the store's content **and** `learned_1x` are fitted from *"that leg's own pool, **first `PINS['n_store_train'] = 32 768` rows** — the same rows the store's content is fitted from"*, and every `rows_learned_1x.json` carries `n_fit_rows: 32768` | print **32,768 fit rows, matched between arm and opponent** — ⭐ a **stronger** statement than the split size, because it forecloses the obvious *"the store saw 9× more data"* objection |
| **SF-e** | §4.4: *"`pca_tensor` … scores **IDENTICALLY — to four decimals at every rung and every seed** — to the single design"* | true, and stronger than stated: the two arms' `random_atoms` controls are also identical to 16 digits (0.25 / sd 0.7041042 / judge 0.0069444 on both), i.e. the reads are the **same object** | say that — *the statistic is not consulted at generation time at all, which is why the two arms are bit-equal* |
| **SF-f** | §4.2's K\* column and §4.4's K = 5 ledger sit in the same section | the ladder's ledger is a **three-part** total (`BYTES.json` carries parts 1–3; part 4 DECLARED ABSENT) while the K\* legs carry **four** (+6,610 B) | one line: the two totals differ in part count as well as convention (6,610 B, 0.05 % — small, but a never-netted ledger cannot let it pass silently) |
| **SF-g** | §C App. M: *"`verify_tables.py` re-resolves 52 sources / 872 cells / 1,786 per-seed values / 8 figure hashes, **PASS**"* | ⭐ I re-ran it and it does, exactly. ⚠ But it verifies **pointer → value**, not **currency**: it PASSES on the T7 and T8 rows the draft's own OPEN-1 declares superseded | add the sentence the verifier cannot say for itself: *it checks that every printed cell is the value on disk at the pointer it records; it does not check that the pointer is the latest artefact* |
| **SF-h** | §3.1: `worst_in_float32_ulp_at_unit_scale = 0.009442` labelled *"artefact's own bar"* | it is a measured **worst**, not a bar; and `per_seed/0` and `per_seed/1` carry the identical value 0.009441560599952936 while `per_seed/2` carries 2.384e-07 | call it *"the artefact's own worst-over-seeds ulp figure"*, and note seeds 0 and 1 share it |
| **SF-i** | §4.5: *"⛔ The 1× comparator column on T7 remains DECLARED NOT-RUN (NR8)"* | ⭐ correct, and the error direction is stated correctly. ⚠ But §4.2's K\* ladder column **is** a 1× result on the same object class, two sub-sections away | one line: *the scored T7 rows have no 1× column; the K\* ladder replication does, and it is the only 1× evidence at K\** |
| **SF-j** | §3.3's w8 row heads part 2 *"statistic"* at 12,327,096 | that is covariance + running means + per-well counts; T4 splits it 2a/2b | keep T4's 2a/2b split in the paper's row, since the paper elsewhere reasons about `Σ_k d_k` separately |
| **SF-k** | §I: *"beating a **parameter-matched** trained learner"* | after MF12 the match is **−0.304 % / +0.495 % / +0.495 %**, with the arm carrying **more** parameters on two of three seeds | *"a learner matched to within ±0.5 % on parameters and ±0.5 % on total deployed bytes"* |
| **SF-l** | §11 item 3 and K18: *"no object with a float64 sufficient statistic composes and forgets inside 524,288 B"* | ⭐ well qualified. ⚠ But `LEDGER-ACROSS-K.json`'s first key is `SMOKE_NOT_A_CLAIM: true`, and the draft says so only in §5.2 | repeat the smoke flag in K18's own row, where the claim register will be read |

---

## 5. ○ NICE-TO-HAVE — five

1. **§4.3 is the best-handled thing in the revision and could say one thing more.** The draft labels
   the n ≥ 3 axis honestly and shows the flip survives at both ends of the only genuine contrast
   (n = 2: −2.783 → +4.356; n = 6: −4.061 → +7.978 — both re-derived and correct). ⭐ Add the
   artefact's own decisive control: *every arm **not** fitted on the rung's pool is **exactly
   constant** across n = 3…6*, which is what makes "training-data axis" a measurement rather than a
   concession.
2. **§4.2's n = 3 paragraph** correctly refuses to call the K = 5 n = 3 result a win. ⭐ Add the number
   that makes the refusal airtight: `sign_flipped` is **false** for n = 3 in
   `KSTAR-LADDER.json` `/seed_matched_K5_vs_Kstar/3` — it is the one rung where nothing flipped,
   because it was already positive.
3. **§4.5's rank-deficiency note** (seed 1 0.5938–0.9844, seed 2 0.4375–0.9844) is good. Add that the
   K\* ladder legs are uniform at 0.9688 — so the spread is a property of the **crossover object's**
   seeds, not of K\* as such.
4. **§5.2's U-curve** is the cleanest new result in the draft and its minimum is at K = 24, not K\*.
   One sentence would let the paper own it: *the object the paper promotes is not its own cheapest
   configuration, and we say so.*
5. **§3.6's `cos_min` contrast** (0.1999 at rank/width 0.156 vs 0.9976 random) is doing a lot of work.
   Print the K\* rank/width beside it (0.9531–0.9688, `TABLES.md` T9) so the reader can see the axis.

---

## 6. ⭐ WHAT THE REVISION GOT RIGHT AND MUST NOT BE UNDONE

So the Head does not over-correct a second time.

1. ⭐⭐ **§3.0 + §2.5 + §3.6 together.** Defining exactness store-scoped, declaring the frozen
   encoder/decoder outside the deletable object **before** the claim, and then **measuring** the
   scope's cost (32.2500 → 32.2500 at K\*) is a better answer than MF4 asked for. ⛔ Do not soften
   §3.6 into a footnote: it is the reason R1 moves from 3 to 7.
2. ⭐⭐ **§3.3's framing sentence** — *the object that forgets exactly is itself 24.55× the pin the
   paper measures everything else against, which is why §5 exists* — converts the worst omission into
   the paper's cleanest structural argument. Keep it verbatim (with **N7**'s eleven words).
3. ⭐ **§3.4's failure agreement with the floor clause.** Two implementations reproducing the same
   seed-2 failure, **both below their own reproducibility floors**, is a statement about content
   well-posedness. Keep it.
4. ⭐ **§6.2 plan-vs-latent**, again verified, and **0.5114 promoted into §6.2(i)** where it belongs.
5. ⭐ **The declared-NOT-RUN discipline**, the ten NR rows with launch lines, and **§0.1** — declaring
   that `w12b` sits outside the evidence fold and flagging it at every use, rather than quietly
   folding it — is exactly the right instinct and should be said in the paper, not only in a log.
6. ⭐ **§4.5's direction-of-error statement** (*"a 1×-matched rival would carry ≈4.09× these parameters
   and be STRONGER, so the printed comparison FLATTERS the arm"*) is the single most credible
   sentence in the composition section. ⛔ **N1–N3 exist because that sentence's discipline was not
   applied to the K = 5 column.**
7. ⭐ **OPEN-1 declared rather than silently fixed**, and no other spoke's artefact edited.

---

## 7. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — checked against the result

> **The task's falsifier:** *"You return no must-fix items ⇒ that is not a pass. Re-check §2 and §3 and
> the arithmetic in the byte ledgers — a draft revised this fast has them."*

**It does not fire: eight new must-fixes, five of them in the byte ledgers, and two of the sixteen
prior items are only partly closed.** Against the task's eight named push-points:

| push-point | result |
|---|---|
| **1** numbers that trace to no file | ⭐ **the draft is clean here.** Every distinctive figure I chased resolved: 3,131,564–3,260,030, 62–79 %, +1.86 to +3.38, 5.92–5.99×, 1.358×–1.372×, 32.350 ± 0.815, 4.5475e-13, 0.009442, 0.5114/0.5105, 0.4762 (in the artefact, not the draft). ⛔ **One number is derived and wrong (4.0897× against three totals, N5) and one is mislabelled (24.55× as "a statistic", N7).** |
| **2** ⛔⛔ tables merging budgets / recipes / environments | ⛔ **fails four times, and this is the section's real damage.** Two covariance conventions across the K = 5 ↔ K\* comparison (**N1**); two objects both called "K = 5" (**N6**); two control recipes across the same comparison (**N4**); a val ratio and a TEST ratio for one object (**N5**). ⭐ The environment hazards v0 named (CMAZE-1/2, `w8_v1_short`, T1/T3, 1-seed vs 3-seed rows) are **all correctly walled off**. |
| **3** the byte row, four parts, never netted | ⭐ **printed on all three legs, never netted, closes to the byte, with a declared residual.** MF3 is genuinely closed. ⛔ The row's own ratio column is wrong on two seeds (**N5**). |
| **4** is "exact" DEFINED store-scoped, encoder outside the object | ⭐⭐ **yes, and better than asked** — a citable field, before the table, plus a measurement of the scope's limit. **MF4 CLOSED.** |
| **5** negatives in the MAIN TEXT, incl. the ladder loss | ⭐ **§5, §6, §7 and §11's twenty items are all main text**, the route-task failure and compose/forget lead their own sections, and MF15 is item 17. ⛔ **But the draft gives three contradictory instructions about where the ladder loss is printed (N8)**, and the only one that survives a careless read appendixes it. |
| **6** ⚠ attack the n = 3 win | ⭐ **the draft does not lean on it.** It reports n = 3 at K = 5 as UNEXPLAINED, names the fact that the **opponent fell** rather than the arm improving (0.7709 against 0.9256–0.9530 elsewhere), refuses to invent a mechanism, and notes it fails the every-seed bar anyway. Two additions at §5 item 2. **No must-fix.** |
| **7** ⚠ the ladder's axis for n ≥ 3 | ⭐⭐ **labelled honestly and at the right altitude.** §4.3 states the identical 120 corners, the held-out row counts (300,000 / 235,200 / 213,600 / 200,640), the r = −0.96…−0.98 tracking, that arms not fitted on the pool are exactly constant, and that **only n = 2 vs n = 6 is a genuine depth contrast** — then shows the flip is present at both ends of that contrast. ⛔ **This is not passed off as composition depth and I could not break it. No must-fix.** |
| **8** ⭐ attack the deletion claim properly | ⛔ **the downdate-vs-refit comparator is the right one and is now correctly bounded.** The reference can delete only by refitting from scratch, reading 120,028–131,072 rows at 9.259 s against 0.8998 s (§3.7), and §3.6 states the limit of what the deletion buys. ⛔ **The byte accounting does NOT net the address book anywhere** — I checked all three legs and the book is a separate printed part in every one. ⚠ The residual exposure is **N7**'s label, not the accounting. |

### ⛔ The one thing a hostile AC will do that this draft is not yet ready for

Open `w12-composition-ladder/BYTES.json`, read the field **two lines below** the one the paper quotes
(`p2_if_symmetric_triangle_float32`), and ask why the store the paper penalises at 24.9× is priced on
`Σ_k d_k²` while the store it promotes at 4.09× is priced on `Σ_k d_k(d_k+1)/2`, when §5.1 of the same
paper says the second is the correct cost function. ⭐ **There is a good answer and the paper can give
it in three sentences (N1), but it has to give it first.**

---

## 8. WHAT THIS SPOKE DID NOT DO

- ⛔ **No scored run, no compute budget, no model fitted, no experimental number produced.** The only
  things executed are `verify_tables.py` (read-only, ~4 s, 1 thread, no scored budget) and read-only
  JSON arithmetic in-process. Every recomputation above is a sum or a mean over fields on disk and is
  reproducible from the paths named.
- ⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
- ⛔ **Other agents are running on this box.** `/c/chlu/outputs/w12b-nr6-and-kstar/` and
  `/c/chlu/outputs/w13-draft-revise/` were read **read-only** and only after each carried its own
  `END_UTC.txt`. ⛔ **No session, worktree, output directory or process belonging to another agent was
  touched. No background job was launched, none was left running, and NO BOX-WIDE KILL** was issued or
  was possible from anything this spoke ran.
- ⛔ **Nothing was written outside `/c/chlu/outputs/w13-referee-2/`** and this spoke's own worktree
  (branch `w13/referee-2`). ⛔ **`/c/chlu/evidence/` was not written. No `.claude/` directory was
  created. No other spoke's table, figure, generator or registry was edited.**
- ⛔ **DECLARED NOT-CHECKED, each with its reason — never a silent omission:**
  1. **The eight figure PNGs were not opened as images.** Their **hashes** were re-verified by
     `verify_tables.py` (8 of 8); no figure's *content* was inspected.
  2. **The `.pkl` and `.eqx` artefacts were not loaded** (address books, stores, rival weights). Every
     byte figure above is taken from the JSON ledgers, which I checked close arithmetically.
  3. **Seeds 1 and 2 of the bridge do not exist**, so **N2's substitution is seed 0 only** and is
     labelled as such. I did not extrapolate it to three seeds.
  4. **The physics block (§8) was re-checked only where v0 raised an item** (SF8–SF11, B3's max,
     NTH7). `EVIDENCE.md` was not re-read end to end; §8's other cells are taken on v0's pass.
  5. **`claims_matrix.md`'s `CM-W12-*` rows were not read cell by cell.** Where the draft cites one I
     checked the underlying artefact instead.
  6. **`w13-registry-fold` was not re-audited.** Out of scope; v0 audited `w10-registry-fold`.
  7. **The three sibling w13 spokes' bundles were not verified.** Each spoke verifies its own.
