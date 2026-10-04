# DRAFT-v0 — the ICLR-long SKELETON for the Naive-CLU paper

**w10 `spoke/draft-v0` · agent `paper-writer` · 2026-09-21.**
**Scored runs spent: ⛔ ZERO.** This spoke produced no number. Every figure below is quoted from an
artefact with the path that holds it.
**Evidence base:** `/c/chlu/evidence/outputs/` (+ `claims_matrix.md`, `negative_results.md`) and the
three w10 sibling spokes. ⛔ `$CHLU_CLAUDE_ROOT` **was not read at any point.**

> ⛔ **THE HEAD WRITES THE PROSE.** Nothing below is finished sentences except where a sentence is
> quoted as *the* sentence the evidence forces. This file is **structure + claim placement +
> evidence pointers**, built so the Head edits placement, not invents it.

---

## 0. GATE — the mechanical precondition, discharged

| required by the task file | found at | status |
|---|---|---|
| `/c/chlu/outputs/w10-seeds-and-tables/TABLES.json` | same path, 263,860 B | ✅ PRESENT |
| `/c/chlu/outputs/w10-physics-section/EVIDENCE.md` | `/c/chlu/outputs/w10-physics-section-evidence/EVIDENCE.md`, 64,732 B | ✅ PRESENT (that spoke's directory carries the `-evidence` suffix; it holds `START_UTC.txt`, `LOG.md`, `BENCHMARK-STATEMENT.md` and its bundle, so it is the named artefact and not a different one) |
| the updated negatives registry | `/c/chlu/outputs/w10-registry-fold/negative_results.md` (1,442,067 B) + `claims_matrix.md` (442,589 B) + `FINDINGS.md` | ✅ PRESENT — 21 entries `N-I8-1…21`, 13 matrix rows `CM-I8-1…13` |

⚠ **One declared NOT-AVAILABLE input:** `/c/chlu/outputs/w10-referee/` was **empty** at 08:50Z. It is
not a precondition for this spoke and nothing below depends on it, but the Head should read the
referee spoke's output against §§E–H of this file before the prose pass.

---

## 1. THE FRAMING — carried from the task file, not relitigated here

| | |
|---|---|
| ⭐ **LEAD CLAIM** | **Exact deletion of a compositional generative model.** Confirmed twice over, on two environments, by two independent implementations, and **unclaimed in ≈4,050 screened works**. |
| ⭐ **THE SENTENCE THE PAPER CARRIES** | **"Downdate the statistic, not the truncated factors."** |
| ⛔ **NEVER CLAIM** | *"multilinear/bilinear factored generation with a request-indexed tensor head"* — Tenenbaum & Freeman (2000), Vasilescu & Terzopoulos (2002). |
| ⛔ **NEVER CLAIM** | *"tensor-product label conditioning for unseen attribute combinations"* — MLC-VAE, Georgopoulos et al., ICML 2020. |
| ⛔ **NEVER CLAIM** | the downdate **primitive** ("covariance downdate + re-eigendecompose = exact refit") — Cao & Yang (2015) owns it in one line from the summation form. |
| ⭐ **BUT DO NOT CONCEDE** | exactness to the decremental-PCA / eigenspace-arithmetic line. **It does not have it.** Hall–Marshall–Martin subtract *deflated* eigenspaces and say so themselves: *"yields a result **approximating**…"*, *"tends to instability"*, *"best used to remove a **small fraction**"*, *"no rigorous error analysis… anecdotal in character"*. Source: `/c/chlu/evidence/outputs/_thinkpad/w8/w8-novelty/NOVELTY.md` §D + `LOG.md` 16:53Z. ⚠ **Reading `NOVELTY.md` without its `LOG.md` gives away this ground.** |

**Novelty status of the composite object:** ⚠ **ADJACENT**, on a *systematic* negative, not on
absence of evidence — ≈4,050 unique works screened in three independent directions (3,038
generation-side forward citations; 759 deletion-side mirror crawl; 254 orthogonal all-of-OpenAlex).
Deletion language in the generation-side set: **10 of 3,038 (0.3 %)**, seven false positives, three
real and all approximate. Frozen-decoder + tensor-head + label-index together: **0 of 3,038**.
Highest atom count reached by any single work: **4 of 6 (CausalX)**; nothing reached 5.
*Source:* `NOVELTY.md` §§A–E.

---

## 2. MAIN TEXT — section structure, target ≈ 9 pp

> **Main-vs-appendix rule for this paper:** main text carries (i) the deletion result, (ii) the one
> composition RESULT, (iii) the byte-pin incompatibility that joins them, (iv) the route-task
> negative with its isolated mechanism, and (v) the physics section's scope limit. Everything else
> is an appendix. ⛔ **Nothing is omitted before the dedicated pruning passes** (appendix-maximalism).

### §1 Introduction

| slot | what goes here | pointer |
|---|---|---|
| 1.1 | The problem sentence: exact unlearning exists for k-means, bilinear matrix completion and ridge heads on frozen backbones; for *generative* models the literature calls retrain-equivalence out of reach and substitutes approximations. | `NOVELTY.md` §3 (arXiv:2206.14439 opens by saying so; arXiv:2603.12977 states it covers *"ridge regression only — no PCA, eigendecomposition, or generative models"*) |
| 1.2 | ⭐ **The contribution sentence.** A compositional generative store whose whole content is a sufficient statistic, so deletion **is** the refit. | `NOVELTY.md` §3 final ¶ — the survivable form is verbatim there |
| 1.3 | ⭐ **The mechanism sentence.** *"Downdate the statistic, not the truncated factors."* | `NOVELTY.md` §D; `/c/chlu/outputs/w10-registry-fold/FINDINGS.md` §4; registry `N-I8-1`(F), matrix `CM-I8-1`(c) |
| 1.4 | Contributions enumerated on page 1. Four, in this order: **(a)** exact deletion, two implementations, 3 seeds each, TEST; **(b)** the composition RESULT at an asserted byte pin; **(c)** the measured incompatibility of (a) and (b) *at that pin*; **(d)** the route-task negative and its isolated mechanism. | §§3–6 below |
| 1.5 | ⭐ **The honest-thickness sentence, in the introduction and not buried.** One environment with three scored seeds and a clean instrument returns a **negative**; one claim is positive, replicated across two implementations and three seeds, and unclaimed. | `/c/chlu/outputs/w10-seeds-and-tables/FINDINGS.md` §6 final ¶ |
| 1.6 | Headline figure named. **Figure 1 = `figures/fig4_exact_deletion.png`** (both deletion implementations, log scale). | `/c/chlu/outputs/w10-seeds-and-tables/figures/fig4_exact_deletion.png` |

⛔ **The introduction must not open on composition.** Composition is the crowded half of the
literature and `NOVELTY.md` §5 states plainly it *"cannot be won on its own"*.

### §2 The object

| slot | content | pointer |
|---|---|---|
| 2.1 | The store: per-well leading **principal subspace of the fit pool** — 64 atoms × 2048 float32, pool mean as the first atom. ⭐ **Unsupervised; no label enters the content at all.** | `/c/chlu/outputs/w5-compose-field/LOG.md` §2 ("What the object is") |
| 2.2 | The read: a tensor-product address book, R=128, **request-only** — `feature_argument_present: false`, `bitwise_stable_under_feature_perturbation: true`, asserted on the deployed object per leg. Address bytes **189,344–195,064 B declared BESIDE the store and never netted into it**. | `w5-compose-field/LOG.md` §2; charter B.3 assertions |
| 2.3 | The byte pin: **524,288 B, tolerance 0**, `matches_budget: true` asserted before every leg. Parameter budget SECONDARY: 178,928 vs `mlp_180k`'s 178,952 = **−0.01 %**. | `w5-compose-field/LOG.md` 02:45Z entry |
| 2.4 | The renderer: a **frozen** nonlinear decoder; labels touch the model only at the address book. This is the interface split. | `NOVELTY.md` §4 claim 2 |
| 2.5 | ⚠ Naming/continuity discipline (**Head decision, not this spoke's**): the paper object is the **Naive-CLU** per the wave plan. | `/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md` §0 |

⚠ **Write §2 so that §2.1 + §2.4 are visibly the *join* of two literatures.** The multilinear line
(A3+A4) builds its bases *out of the labels* and renders linearly; the frozen-generator line (A2+A5)
has no label request and no product table. ⛔ Do not present the join itself as the headline — that
is claim 2 of 3 in `NOVELTY.md` §4, not claim 1.

### §3 ⭐⭐ EXACT DELETION — the lead result

**The claim, in the only form the artefacts support:** deleting a set of rows from the store is
**exactly** the store refitted without those rows — no refit performed, no pass over retained data,
items never kept — verified on **two independent implementations, different code, different authors,
three seeds each, on the held-out TEST condition**, with a negative control firing on every seed.

**TABLE 4** is the main-text table. Both legs, side by side, **not merged** (the deviations are in
different units: a relative residual on the statistic vs an absolute max deviation per well).
*Source:* `/c/chlu/outputs/w10-seeds-and-tables/TABLES.md` § TABLE 4.

| leg | statistic | seed 0 / 1 / 2 |
|---|---|---|
| w9/maze `CluRead.delete_rows_exact`, TEST | statistic max **relative** residual (downdate vs refit) | **7.522e-16 / 8.826e-16 / 7.818e-16** |
| | subspace max **projector** residual | **0.000e+00 / 0.000e+00 / 0.000e+00** |
| | rows removed / kept | 216/992 · 226/975 · 193/1084 |
| | refit performed · passes over retained data · items kept · address-book bytes deployed by the carry | **False · 0 · False · 0** |
| w8/downdate `exp_w8_downdate`, TEST | max **absolute** deviation, worst well | **2.365e-14 / 1.126e-09 / 7.105e-15** |
| | negative control fired | **True / True / True** |
| | no pass over data · items kept | **True / False** |
| | artefact's own bar | `passed_every_seed = True`; `worst_max_abs_dev_over_seeds = 1.1255e-09`; **`worst_in_float32_ulp_at_unit_scale = 0.009442`** |

⛔⛔ **THE NUMBER THE PAPER MUST QUOTE IS THE ULP FIGURE OR THE TRUE ABSOLUTE — NEVER `1e-14`.**
The w8 advisor report's range *"1e-16…1e-14"* understates the worst observed deviation by ~5 orders
of magnitude. The claim survives (0.009442 float32 ulp is under one hundredth of an ulp, and
`passed_every_seed = true`), but the printed range does not. *Correction C2:*
`/c/chlu/outputs/w10-seeds-and-tables/TABLES.md` §CORRECTIONS C2; artefacts
`/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` and `DOWNDATE-READY.json` `/downdate_equals_refit`.

⛔⛔ **AND THE PAPER MUST NOT SAY "9/9 SEEDS".** The phrase is **3 configurations × 3 seeds inside
ONE implementation** (`hub/single-design`'s matched-bytes candidates K=40/44/48), not nine seeds, and
the `1e-16…1e-14` band is **not shared by both implementations** — `spoke/downdate-build`'s own four
legs run **2.842e-14 · 4.547e-13 · 1.164e-10 · 1.126e-09**. *Source:*
`/c/chlu/outputs/w10-registry-fold/FINDINGS.md` §3; `/c/chlu/outputs/w8-single-design/LOG.md:481,503`;
`/c/chlu/outputs/w8-downdate/LOG.md:1010-1024`.

⭐ **The number to lead the section with** (registry `CM-I8-1(b)`, which prints the three
implementations as three separate rows so no draft can quote one implementation's band as both):
**CMAZE, all three scored seeds, statistic 7.5e-16…8.8e-16 and subspace projector EXACTLY 0.000e+00,
replicated from a clean checkout with ZERO numeric differences.**

⭐ **The paragraph the Head should not cut — the failure agreement.** The exactness assertion
**failed first**: on TEST seed 2 at **3.921e-06 = 32.90 float32 ulp**, on the incumbent's coarse
imbalanced partition at A=64 (`/c/chlu/outputs/w8-downdate/LOG.md:618-640`). The fix is a
variance-share admission cut with **no free parameter**, `tau = 4.941056e-16 / 1.1920929e-07 =
4.144858e-09`, **declared before the re-run and not moved**. And `hub/single-design` reproduced the
same seed-2 failure **independently, with different code and its own sign convention**, at 14.377 ulp
against the spoke's 32.90/35.46 (`/c/chlu/outputs/w8-single-design/LOG.md:481`).
⇒ **Two implementations agreeing on where the mechanism breaks is stronger evidence than two
implementations agreeing on where it works.** That is a paragraph, and it belongs in main text.

⚠ **One honest limit to print beside the numbers:** `spoke/downdate-build`'s worst leg (TEST seed 1,
1.126e-09) sits **below its own measured reproducibility floor of 4.282e-09**, so at that leg the
deviation is not resolvable at all. *Source:* `w10-registry-fold/FINDINGS.md` §3(2).

**Figure slot:** `figures/fig4_exact_deletion.png` — both implementations, log scale.

**A second, independent deletion property, carried from the physics block:**
**decay commutes with deletion**, `evict ∘ decay^n == decay^n ∘ evict`, **0 bytes differing in a
2,054,866 B image, in all 12 cells per seed, all three seeds, zero tolerance** — and on a *learned*
store, which V5 does not claim. *Source:* `/c/chlu/outputs/w10-physics-section-evidence/EVIDENCE.md`
§3.5 (D32a), from `evidence/outputs/_thinkpad/w6/w6-retention/LOG.md` §3 §5 §6.
⛔ Its companion **D32b FAILS as pre-registered** on all three seeds and must travel with it — see §11.

### §4 ⭐ COMPOSITION AT AN ASSERTED BYTE PIN — the one RESULT against a trained learner

⚠ **This result is NOT in `TABLES.md`.** The w10 tables spoke read w8/w9 artefacts and the physics
sources; the programme's 3-seed held-out-TEST composition win is **w5 `hub/compose-field`**. The Head
must not conclude from `TABLES.md` alone that no composition RESULT exists. ⚠ It is **also not in
tension** with that spoke's "no Shapes3D composition scoreboard at 3 seeds": that statement is about
the **incumbent generation-cell instrument** (K=64, A=1,216, `eval_b`, a validity *rate*), which is
`NR1` and has **one** scored seed. compose-field's metric is **distinct valid novel tuples per
corner** on Montero corners. ⛔ **The two do not share a table.**

**The claim:** `pca_tensor` beats `mlp_180k` on **both** target rows, **all three seeds**, beyond a
margin declared before the first held-out number, at **524,288 B tolerance 0** and parameter-matched
to **−0.01 %**.

| | row 4 (49 n-way corners) | row 3 (60 pair sub-corners) |
|---|---|---|
| mean paired Δ vs `mlp_180k` | **+10.143 / +8.673 / +7.796** | **+4.033 / +2.483 / +2.417** |
| bootstrap 2.5 % quantile | +6.632 / +5.816 / +5.082 | +2.117 / **+1.033** / +1.200 |
| bar (Δ ≥ +1.0 **and** CI95-low > +1.0, every seed) | ⭐ **PASS 3/3** | ⭐ **PASS 3/3** |

*Source:* `/c/chlu/outputs/w5-compose-field/LOG.md:196,247`; `/c/chlu/evidence/outputs/w5-advisor-report.md` §1.
Condition `test`, cache generation 2, bridge seeds 0/1/2, 49 + 60 paired corners, 20 arms scored per
seed, zero failures (`w5-compose-field/LOG.md` 01:45Z).

⚠ **Print the tight leg as tight.** Row 3 seed 1 clears +1.0 by **0.033**. The hub itself refused to
call it comfortable; the paper does not get to either.

⭐ **The floor that counts, declared before the numbers.** The two matched-bytes lookups score 0.000
and the hub **declined to claim them as a §C.1 pass** (a single principal *direction* is not a
plausible answer, so 0.000 is nearly automatic). The informative floor is `oracle_content_items` — 64
whole written rows at the same 524,288 B, read by an **oracle** — scoring **32.388 / 39.122 / 41.776**
on row 4. The arm beats it by **+31.224 / +25.204 / +25.878 while the oracle is handed the true
feature and the arm is not.** *Source:* `w5-compose-field/LOG.md` §7 (02:20Z ladder; 01:45Z table).

⭐ **The measurement that says the win is the CONTENT, not the budget.** The inherited field store's
own `oracle_span` sits **below** the bar (row 4 **−12.531 / −5.224 / −5.653**), while at the *same*
524,288 B, the same K=5 partition and training rows only, `oracle_content_pool` − `mlp_180k` is
**+10.980 / +8.224 / +8.551**. ⇒ the wave-4 ceiling was a property of 64 *fitted modes*, not of the
byte budget. *Source:* `w5-compose-field/LOG.md` §7 item 3.

⛔ **Figure gap, declared:** there is **no figure for this result in any w10 artefact.**
`figures/fig1…fig5` cover CMAZE-1, the n-way denominators, the encoder-loop decomposition, deletion
and BLIND-SEQUENTIAL. **A composition figure must be built or the section ships table-only.** See §F.

### §5 ⭐⭐ THE JOIN — compose XOR forget, at the pin. This is the paper's own open problem.

This section exists because §3 and §4 are **measured on different objects**, and a reader will ask
whether one object does both. The programme asked, and the answer is measured:

| link in the chain | number | source |
|---|---|---|
| exact per-item deletion works on the covariance store | downdate == refit, 0 rows refitted, 0 passes, items never kept | `/c/chlu/outputs/w8-single-design/LOG.md:553-620` §2(1) |
| the covariance costs `Σ_k d_k²`; fitting it inside 524,288 B **forces K ≥ 36** | below K=36 the statistic alone exceeds the budget with **zero atoms left for content** | same, §2(2) |
| at K ≥ 36 composition **collapses** | K=40 / A=7, honest four-part ledger **519,794 B = 0.991×** the pin ⇒ valid mean **0.0648** | same, §2(3) |
| at K = 5 composition **holds** | **0.8918** against the incumbent's **0.8941** — a tie — but the honest ledger is **3,922,450 B = 7.48×** the pin (label-free `ward_balanced` cut), or 24.5× on the spoke's factor partition | same, §2(4) and §3 |

⭐⭐ **The sentence this forces:** *both properties DO coexist in one object — at 7.5× the byte
budget.* ⛔ **It is quoted with its overrun attached or not at all,** and it is **`val`, seed 0, one
arm, 0 of 3 scored runs spent** — an observation, not a RESULT row. The byte pin is PRIMARY at
tolerance 0, so ⛔ **no RESULT row may be written for that object.**

⭐ **And the cost of keeping items instead is priced:** a principal direction is not a written item,
so charter §B.4's exact per-item deletion does not hold of the composition-winning content; keeping
items costs **≈ 18.6 tuples/corner** on row 4. *Source:* `/c/chlu/evidence/outputs/w5-advisor-report.md` §1
final ¶ (registry `N-I8-5`, matrix `CM-I8-5`).

⚠ **The repair that makes post-deletion composition survivable, and its price.** Deletion changes the
atoms, so an address book fitted before it is stale after it. Carrying the **book** rather than the
atoms — `W_k ← W_k M_k`, `M_k = U_kᵀ U'_k`, formed at delete time, **no data row read, zero deployed
bytes** — recovers judge-space composition **0.0584 → 0.8964 against 0.8941 before deletion = 100.3 %
recovery with no refit of anything**. ⛔ Its cost, printed rather than inferred: it rewrites
**127,488 B of the address book's 158,664 B**, and is **never** quoted beside the coupling hole's
1 byte. *Source:* `/c/chlu/outputs/w8-single-design/LOG.md` §4.1. ⚠ **`val`, not TEST.**

⭐ **The store-level control that says the carry is targeted, not generic:** `M_k`'s principal-angle
spectrum under the red deletion is **cos_min 0.1999, median 0.9999, 93.8 % of directions above 0.99**,
while under a **size-matched random** deletion of the same 11,044 rows **cos_min 0.9976 — nothing
moves**. ⛔ A property of the store, **not** a composition number.
*Source:* `w8-single-design/LOG.md` §4.2.

⛔ **DECLARED NOT-RUN NR3 travels with this section, in the main text:** the single-design object's
RESULT rows 3/4 and post-deletion rows are **0 of 3 scored, full budget intact** — the object was
falsified on `val` before a scored run was justified. That is the wave's answer, not a gap.
*Source:* `/c/chlu/outputs/w10-seeds-and-tables/TABLES.md` §NR3.

### §6 ⛔ IT DOES NOT TRANSFER TO A ROUTE TASK — a main-text result, not an appendix

> ⛔ **A draft that buries this is wrong.** The task file is explicit and the evidence is clean.

**TABLE 1** is the second main-text table: CMAZE-1, w9 `hub/maze`, held-out TEST, 3 of 3 scored
seeds, K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, read = 410,688 floats (1,642,752 B deployed)
with the float64 sufficient statistic (35,904 B) declared beside it and **never netted in**;
baselines matched at mlp 410,920 / lstm 409,846 / transformer 405,776 params.
*Source:* `/c/chlu/outputs/w10-seeds-and-tables/TABLES.md` § TABLE 1.

| row | CLU | best learned baseline | lookup (matched bytes) | `random_atoms` (C.2) | pipeline ceiling |
|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.9865 (transformer) | **1.0000** | 0.9601 | 1.0000 |
| 2 unseen pair | 0.1279 | **0.1840** (transformer) | 0.1565 | 0.1414 | 0.6319 |
| 3 n-way, CONJUNCTION | 0.0284 | **0.0493** (lstm) | 0.0159 | 0.0195 | 0.5255 (per route) |
| 3 n-way, PER ROUTE | 0.0852 | **0.1246** (transformer) | 0.0946 | 0.0693 | 0.5255 |
| 4 budget interpolation | 0.0996 | **0.1430** (transformer) | 0.1122 | 0.0905 | 0.5573 |

- ⛔ **BAR 1 fails on all three composition rows:** −0.0560 (row 2) / −0.0209 (row 3 conjunction) /
  −0.0434 (row 4) against the best learned baseline.
- ⛔ **BAR 3 fails, and that is the mechanism of the table.** `landscape_shuffled` **0.9509** and
  `random_atoms` **0.9601** against the CLU's 1.0000 on row 1; on row 2 `random_atoms` **0.1414** sits
  **inside** the CLU's own seed spread (0.1590 / 0.0597 / 0.1651). Only `landscape_removed` collapses
  (0.0000 everywhere). ⇒ **Under the hub's own pre-registration NO RESULT ROW IS CLAIMED for the CLU
  arm on CMAZE-1.** The paper prints the table and claims nothing from it.
- ⭐ Row 1's 1.0000 is real but **uninformative**: the matched-bytes lookup also reads 1.0000.
- ⭐ Every arm sits far below its ceiling, and at this rank the **store span tracks the pipeline
  ceiling to within 0.0138 on every row and every seed — exactly equal on 7 of the 12 row×seed
  cells** — so the atoms are not the binding constraint. **The address book is.**

⚠ **Row 3 must be printed TWICE.** The scored n-way row is a **CONJUNCTION** (n = 183/192/191); its
ceilings are computed **PER ROUTE** (n = 389/403/414). `0.0284 vs 0.5255` is **not** a fraction of
ceiling. *Correction C3.* **Figure slot:** `figures/fig2_nway_conjunction_vs_route.png`.

**Figure slot (main):** `figures/fig1_cmaze1_main_scoreboard.png`.

#### §6.1 ⭐⭐ THE ISOLATED MECHANISM — plan space **1.0000** vs latent space **0.13**

This is what makes §6 a *result* rather than a failure report. Three analyses of the
already-reported scored legs; **no new arm, no new held-out number, no bar moved.**

1. **Every piece of every answer is already written.** PLAN COVERAGE of a held-out unseen-pair route
   by the union of the TRAINING routes of the *same* maze: **mean 1.000, median 1.000; 194 of 195
   pairs at ≥ 0.95** (seed 0).
2. **The factorisation the CLU assumes is CORRECT.** Coverage by same-agent plans **0.838**, by
   same-target plans **0.885**, **UNION 0.997**, **complete for 99.1 % of pairs** — i.e. precisely the
   pair blocks of the address book carry the whole answer.
3. ⭐⭐ **The composition is destroyed by WHERE it is performed.** Same two halves, composed in two
   different spaces, executed and graded by the same exact judge:

| composition performed in… | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **PLAN space** (overlay the two half-plans, target's plan wins conflicts) | **1.0000** | **1.0000** | **1.0000** |
| **LATENT space** (average the two halves' latents, decode) | **0.1649** | **0.0950** | **0.1449** |
| the same-agent half alone | 0.2835 | 0.1900 | 0.2523 |
| the same-target half alone | 0.1701 | 0.1750 | 0.2523 |

   Controls (it is not a tie-break artefact; both halves are load-bearing): agent's-plan-wins
   **0.5722 / 0.4250 / 0.4252**; target's plan + a RANDOM training plan instead of the agent's half
   **0.3557 / 0.3700 / 0.4065**. In plan space the overlay is not merely valid, it is **OPTIMAL**
   (score 1.0000 = opt/len on every pair).

⭐⭐ **THE FINDING, as the paper should state it:** *the read is linear in the request design, so
composing two addresses composes their **latents**; an exact-judge route task composes in the
**plan**. The information is present, the factorisation is right, and the operator is available and
perfect — what fails is that a point-in-latent-space read cannot express "overlay these two plans,
the second winning".*

*Source:* `/c/chlu/outputs/w9-maze/LOG.md:288-346` (10:55Z entry); registry `N-I8-3`, matrix `CM-I8-3`.

⛔ **Two caveats that travel with it, and neither is optional.**
(a) The plan-space overlay is a **REFERENCE, not an arm** — it keeps whole written items and is not
matched to the CLU's byte count. ⛔ It is reported to locate the failure, **never as a win**.
(b) That the overlay is *perfect* is itself a statement that these routes decompose additively —
the same property that leaves BLIND-SEQUENTIAL at 0.9037 here (§6.2). In an environment whose routes
need two or more keys the overlay would not be perfect.

⭐ A second, subsuming reading the Head may keep or cut: rank decides whether a single plan can be
**rendered at all** (0.937 explained variance needed to render 98 % of training routes; 0.544 renders
2.6 %), and even with rendering perfect — at the scored configuration the store span **equals** the
pipeline ceiling — latent superposition still caps the row at ~0.15. ⭐ The sentence that explains why
the same content transferred to Shapes3D and not here: *on Shapes3D the read is graded by a
classifier over an image, which tolerates blur; here one wrong action is a zero, and a
variance-optimal compressor's discarded residual is exactly what decides legality.*
*Source:* `/c/chlu/outputs/w9-maze/LOG.md` 08:05Z + 10:55Z.

#### §6.2 ⛔ The caveat column, measured on THIS split — and a correction to the Advisor

BLIND-SEQUENTIAL on the hub's own split is **0.9077 / 0.8905 / 0.9128 (mean 0.9037)** on row 2 and
**0.9229 / 0.8933 / 0.8841 (mean 0.9001)** on row 3 — ⛔ **not** the published CMAZE-1 **0.4980**.

⭐ **And the published tripwire reproduces intact at the harness tier**: BLIND-SEQUENTIAL **0.5317
(0.5550 / 0.5300 / 0.5100)**, FRAGMENT-LOOKUP r=1 and r=2 **0.0000 on every seed** (TABLE 3).
⇒ ⛔ **This is a property of the REQUEST LAYER, not of CMAZE-1.** The hub's *unseen pair on a SEEN
maze* layer puts **1,242 of 1,449 optimal routes** on exactly one key, so there is no key ORDER for
rule B to trap. *Correction C4* — the w8 advisor's *"wave 9 inherits the 0.4980 column as a caveat"*
mis-attributes it. **Figure slot:** `figures/fig5_blind_sequential_on_this_split.png`.

⚠ **Band correction C1:** the band *"0.9037–0.9130"* is not a band in the artefact. 0.9037 is the
3-seed **mean**; 0.9128 (not 0.9130) is seed 2; the per-seed **minimum is 0.8905**.

### §7 ⛔ WHAT THE STORE DOES NOT DO — it does not shape the representation

**TABLE 2** (Shapes3D generation cell, w8 `hub/encoder-loop`, `w8_v1_short` decoder,
`joint_steps=400`, `n_train=8192`, 3 seeds). ⛔ **Separate table by charter** — the shortened decoder
is a different *instrument*, not a different setting, and the metric is a **count**
(`distinct_valid_novel_tuples`, ceiling 36/corner on row 3 and 72 on row 4), not a rate.

| bar | row 3 | row 4 |
|---|---|---|
| BAR A passed (mean paired Δ > +1.0 **and** CI95-low > +1.0, all three seeds) | **False** | **False** |
| BAR B falsifier fired (Δ ≤ 0 on two or more seeds) | ⛔ **True** | False |

⭐ **The decomposition is the finding, and the net is a wash because two real effects cancel:**
unfreezing the encoder with the store on the gradient path **HURTS** (`A′` negative on **5 of 6**
seed-rows, two at **p ≤ 0.0013**), and the swap loss **REPAIRS** that damage (`A″` positive on **4 of
6**, two at **p ≤ 0.0003**). Net `A` = **−1.772** (row 3) and **+1.361** (row 4), CI95-low **−3.456**
and **−1.592**. *Source:* `TABLES.md` § TABLE 2; `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json` `/VERDICT`.

⭐ **And the ablation that reads it correctly:** the CLU's margin over the matched-parameter learner
**does not grow** (+12.078 → +10.311 on row 3; +24.687 → +26.048 on row 4 — the learner is at 0.000
either way). **This is a finding about the ENCODER, not about the STORE.**

⛔ **"The store shapes the representation" is NOT SUPPORTED and the paper must not print it.**
Registry `N-I8-4`, matrix `CM-I8-4`. **Figure slot:** `figures/fig3_encoder_loop_decomposition.png`.

⛔ **NR4 travels here:** `hub/maze-encoder-loop` was **contingent on a positive w8 result and the
contingency did not fire**; zero budget drawn. *Source:* `TABLES.md` §NR4.

### §8 THE EXTENDED PHYSICS SECTION — and its scope limit, stated first

> ⛔ **THE SCOPE SENTENCE OPENS THE SECTION, NOT THE RESULT.**
> **The Curvature–Retention Relationship predicts a settle half-life. It supports no scoreboard claim.**
> *Source:* `/c/chlu/outputs/w10-physics-section-evidence/EVIDENCE.md` §1; Advisor ruling
> `/c/chlu/evidence/outputs/w6-advisor-report.md` §6.2.

| slot | content | pointer |
|---|---|---|
| 8.1 | ⛔ The scope limit, and **the same object refuting the scoreboard reading on the same seeds**: `pca_settle` costs row 4 **−39.959 / −46.918 / −39.061** and row 3 **−15.800 / −25.533 / −20.350** against `pca_tensor`, while paying **81,312 B more read-side**; `pca_settle_zerostep` is **bit-identical** to the incumbent (`np.array_equal` asserted) at Δ **0.000** on every seed. ⭐ The one-sentence version is written out verbatim in EVIDENCE.md §1 and should be lifted. | EVIDENCE.md §1 |
| 8.2 | ⭐⭐ **THE RESULT.** The CRR predicts the field store's settle half-life **parameter-free** to a factor **1.105** at the median held-out corner (B1 = **0.0433 / 0.0435 / 0.0431**, bar 0.10, seed spread 0.0004) and beats a one-parameter rival **fitted with sight of the held-out measurements** on **49 of 49 corners on every one of three seeds** (B2 mean paired Δ **−0.5968 / −0.5796 / −0.5982**). 49 corners × 3 seeds, **192,975 rows**, byte pin 524,288 B tolerance 0, field sha recorded. ⭐ B1, B2, the byte pin, the field sha, the integrator check, the instrument floor and B6 all **RECOMPUTE from the raw artefacts**. | EVIDENCE.md §2.3 |
| 8.3 | ⭐ The instrument floor, measured **before** the bar was fixed: median **0.0349** on an exact quadratic ⇒ B1's 0.10 bar is ≈ **3×** the instrument's own floor. | EVIDENCE.md §2.2 |
| 8.4 | ⭐ **Where the law stops, resolved by `h/h*`:** median error **0.0008 (0.2 %)** at `h/h* < 0.1` and **0.0262** at `> 10`, with **all** the error in the single bucket 0.5–2 straddling the exceptional point, where D10 says the 2×2 map is **defective**. **The theory predicts where its own simplest form stops.** | EVIDENCE.md §2.5 |
| 8.5 | ⛔ **B4 is a MISS on all three seeds** (0.3362 / 0.3301 / 0.3520 against a factor-2 bar; 0.3782 / 0.3784 / 0.3834 against the exact inversion) — **carried, not footnoted**, and **pre-registered with its cause in a dated amendment before the run**. The predicted curve's own argmin over `γ*_exact` is **1.0000** on every seed, so the grid resolves the critical point and the shortfall is in the rollout, not in D16. | EVIDENCE.md §2.6 |
| 8.6 | ⭐⭐ The instrument was **named in advance** and was worth an order of magnitude: envelope vs first-crossing differ by **9.24×** deep underdamped; first-crossing's error is **1.0088–1.0180** against the envelope's **0.0284–0.0296**. ⛔ Scoring first-crossing would have reported the law as broken when the instrument was wrong. | EVIDENCE.md §2.7 |
| 8.7 | ⭐ **A second, independent reproduction of the same law** (`w6 hub/retention`, different hub, target and code path): floor **27.026815** vs the papers' 27.03; simulated envelope vs closed form **4e-13 – 8e-11 over ~7 orders of magnitude** of `n_half`; stability limit `1 + tr + det = 0` to **3.3e-16**; latch `ρ = 1.0` exact. | EVIDENCE.md §2.11 |
| 8.8 | ⭐ Bit-exact replication of seed 0 from a clean checkout cloned from the bundle; **runs 2/3 and 3/3 RETURNED UNSPENT**; 1 h 55 m local CPU, no GPU, no cloud, no cluster. | EVIDENCE.md §2.10 |
| 8.9 | The mechanism that connects 8.1 and 8.2 — ⛔ **not scored, not a row.** The settle does **not** leave the principal subspace (off-span ≤ **2e−7**; orthogonal fraction predicts the loss at chance, AUC **0.457–0.559**). What breaks the read is motion **inside** it: the read's answers form a dense lattice with one-factor spacing **ρ₁ = 1.71/1.89/2.00** against a state norm of **31.3/32.3/32.8** (**5.5 %** of the state), the judge's decision cells **are** that lattice's cells (flip probability crosses ½ at `s ≈ 0.5` in all six factors over 15,552 judged states), and the settle places **0.78–0.86** of its step in the 51 answer-changing directions against the isotropic **0.407** — **3.7–4.4×** the isotropic share. Displacement in lattice units predicts the loss at **AUC 0.904**. | EVIDENCE.md §3.10 |
| 8.10 | ⭐ **Deletion in the physics block, and it splits** — D32a **CONFIRMED [CHLU]** (§3 above); D32b **REFUTED on this object, failing step named** (§11.5). ⭐ Two consequences derived **before** measuring and then exact: `n_refuse = 19.9316 × n_half` for every curvature, measured **539 = 19.9432** half-lives on all three seeds; and **`μ²` cancels** — two clocks a decade apart give **16/16 horizons byte-identical, worst lifetime gap 0.000e+00**, so ⛔ **no curvature can trade refusal against composition.** | EVIDENCE.md §3.5 |
| 8.11 | ⛔ **Status discipline is part of the section, not an appendix.** Every physical claim carries CONFIRMED [CHLU] / CONFIRMED [shorts] / REFUTED / UNTESTED / DEFINITION, and **[shorts] means provenance is a paper, not an artefact in this tree** — the programme never re-ran those measurements. | EVIDENCE.md §0 |
| 8.12 | ⭐ The controls, stated because §C.2 is satisfied in the **spectrum**, not in the error: the CRR error is **identical across arms** (0.0428–0.0435 on `field`/`removed`/`shuffled`/`pca_tensor`) *because the law is about the dynamics — stated in advance*; the arms separate in the curvature spectrum (`field` **[0.2727, 1.0000]** vs `removed` **[1.0000, 1.0000]**, no non-trivial curvature at all). ⛔ **§C.1 DOES NOT APPLY and was not given a manufactured pass:** there is no lookup comparator for *"does a closed form predict a rollout"*, and **that hub claims no row against any rival.** | EVIDENCE.md §2.8 |

⛔ **The section must quote the exact-root form of the half-life, never the asymptotic one** — the
asymptotic `2γ ln2/[(2−γ)(εμ)²]` is **2.4 % off already at `h = 0.30 h*`** and was **DECLINED as the
operator** by the hub that used it. *Source:* EVIDENCE.md §2.11 final ¶.

⛔ **Three reporting defects the Head must not walk into** — see §G below: the headline constant
**+1.1385**, the D18 "thin margin", and the tautological `d14_confinement` field.

### §9 WHAT A BENCHMARK WOULD HAVE TO CONTAIN — a requirement list, not a plea

⭐ Drafted from the w5/w6 **negatives**; ⛔ **not** an argument that this programme deserves a
benchmark. Ten requirements, each followed by the measurement that forces it. Full standalone text
to lift: `/c/chlu/outputs/w10-physics-section-evidence/BENCHMARK-STATEMENT.md`; summary
`EVIDENCE.md` §5.3.

The five load-bearing ones for main text (the other five go to App. J):

1. **At least one rule must be able to render a candidate ILLEGAL, not merely more expensive.**
   *Forced by:* GYM-2's composition-blind oracle **O3 = 1.0000** (0 faults) while obstacle-blind
   scores 0.0387 and speed-blind 0.4213 — a speed limit is an upper bound, so ⛔ *"no metric and no
   number of dimensions makes it composable."*
2. **Score with an optimality gap, not feasibility-or-goal-reaching.** *Forced by:* the same
   addendum — the gap drops TW-B from **0.9825 to 0.3900**, below the 0.60 line (stitched plans are
   valid but sloppy: 1.25 extra steps, only 39 % exactly optimal). ⚠ It is strictly better **and** it
   does not rescue requirement 1.
3. **Run a matched-bytes STITCHING tripwire with a `min_frag` sweep before adoption.** *Forced by:*
   ⛔ GYM-1's **T1 exact-key lookup alone scored 0.0000 and would have certified a dead gym as
   sound**; in GYM-2 whole-trajectory retrieval died (T1 0.0000, T3 0.0003) while coarse
   recombination still fired at **0.9825**. **Exact-key lookup is not a tripwire.**
4. **It must contain a deletion request, scored on the SAME object and the SAME metric as
   composition.** *Forced by:* §5 of this paper. A benchmark that scores only composition cannot see
   the programme's central open problem, which is **measured, not argued**.
5. ⛔ **It must not be able to score a dynamics prediction as a taxonomy row.** *Forced by:* §8.1.

⛔ **What this list is not,** and the sentence must appear: it is **not** a claim that the field lacks
such a benchmark. **No scout in this evidence base surveyed the field for one at this wave level**;
the Campaign-3 benchmark scouts are declared **NOT USED**. The Head's own ruling stands: *"benchmark
for the physics CLU declared future work."* *Source:* `EVIDENCE.md` §5.3 final ¶;
`/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md` §0.

### §10 RELATED WORK — positioned so a referee cannot spring it

⛔ **The two claim-STOPs are enforced in this section by construction**, by citing the owners first.

| block | what to say | pointer |
|---|---|---|
| 10.1 the multilinear half | Tenenbaum & Freeman (2000); **TensorFaces** (Vasilescu & Terzopoulos, ECCV 2002) — which **itself states it subsumes PCA as the 1-factor case and bilinear style–content as the 2-factor case**; Vlasic et al. (2005); **CausalX / M-mode Block SVD** (the one genuine block-multilinear ancestor, and the highest atom count in the whole crawl at **4 of 6**); **MLC-VAE** (ICML 2020) — the closest modern statement of the composition claim. ⇒ they own **A3+A4**. ⛔ What they cannot shed: a **complete labelled Cartesian-product tensor**, bases **defined by the label modes**, and a **linear** renderer. | `NOVELTY.md` §§1–2 |
| 10.2 the frozen-generator half | GANSpace (NeurIPS 2020), TCA (BMVC 2021), **PandA** (ICLR 2023) and its **TPAMI 2024** journal version — still *"entirely unsupervised"*, **no labels and no address book**, deletion-atom grep returns **0 across the entire paper**. ⇒ they own **A2+A5**. | `NOVELTY.md` §2, §C.4, `LOG.md` 17:02Z |
| 10.3 exact unlearning | Cao & Yang (S&P 2015) — the summation form, **which owns the primitive**; Ginart et al. (NeurIPS 2019) k-means; Xu et al. (2023) bilinear matrix completion; the ridge-head-on-frozen-backbone result (arXiv:2603.12977) that **explicitly excludes "PCA, eigendecomposition, or generative models"**. | `NOVELTY.md` §3 |
| 10.4 ⭐ the one sentence that is ours | **Hall–Marshall–Martin and the decremental-PCA line are explicitly approximate — they subtract *deflated* eigenspaces, so truncation error compounds under subtraction.** Our route is exact for the opposite reason: it downdates the **full, untruncated second-moment statistic** and re-eigendecomposes afterwards. **"Downdate the statistic, not the truncated factors."** | `NOVELTY.md` §D; `LOG.md` 16:53Z |
| 10.5 pre-empt, do not be shown | **Feature Unlearning for Pre-trained GANs and VAEs** (AAAI 2024) — the nearest thing in existence, and it **unlearns a feature not rows**, is approximate, and **fine-tunes the generator** (so the decoder is not frozen); **Data Redaction from Pre-trained GANs** (SaTML 2023), useful to us because it states outright *"redaction is a fundamentally different task from data deletion"*; GANs Unlearning (TDSC 2025), approximate by construction; **Incremental N-Mode SVD** (TIP 2014) — the only work in ≈4,050 bridging eigenspace-update numerics and multilinear generative models, and it is **incremental only, no removal operator**. | `NOVELTY.md` §C |
| 10.6 the theory ancestor to cite for §6.1 | *Compositional Generalization via Forced Rendering of Disentangled Latents* (arXiv:2501.18797) + Compositional Risk Minimization — the modern statement that **latent factorisation can fail to survive the decoder**. This is direct prior motivation for §6.1's plan-vs-latent finding and for keeping the read linear up to a frozen renderer. | `NOVELTY.md` §2 honourable mentions |

⚠ **Residual novelty risk, stated in the paper's own limitations and not hidden:** OpenAlex coverage
of sub-3-month preprints and non-indexed workshop papers is imperfect, and abstract-level screening
cannot catch a match buried in a method section. The scout's own judgement: **low, not zero.**

### §11 LIMITATIONS, NEGATIVES AND THE DECLARED NOT-RUNS

⛔ **This section is a list of measured facts, not a hedge paragraph.** Every entry is already placed
in a main-text section above; this section is where they are collected so a referee can count them.

1. **No RESULT row for the CLU on CMAZE-1** — BAR 3 fails; the hub's own pre-registration forbids it (§6).
2. **"The store shapes the representation" is falsified** — BAR B fired on row 3 (§7).
3. **No object composes and forgets inside 524,288 B** — falsified on `val`, 0/3 scored (§5).
4. **Post-deletion composition on TEST fails as pre-registered:** surviving targets **0.1024 → 0.0000**
   (per seed 0.1569 / 0.0702 / 0.0802 → 0.0000). The cause is diagnosed — the w8 carry `W_k ← W_k R_k`
   is stated for the orthonormal basis `U_k` while this read is expressed in the deployed modes `G_k`
   — and the repaired mechanism gives **−0.0120 on VAL seed 0**, inside BAR 6's ±0.05. ⛔ **One VAL
   measurement is not three TEST measurements: this is NR5, not a row.** *Source:* `TABLES.md` §1d, §NR5.
5. **D32b REFUTED** — the post-delete store is **not** byte-identical to the never-inserted store:
   **1,339,083 / 1,338,974 / 1,343,641 B differ = 65.17 / 65.16 / 65.39 %** of the image. ⭐ The
   break-out is worth more than the bar: `coupling_v` **798,926**, `term_mu` **267,684**, `atoms`
   **221,590**, `coupling_u` 33,595, fitted widths/gaps/counts 16,769, live-set book-keeping 519 ⇒
   **placement is only 16.5 % of the obstruction; 83.5 % is that the address book and coupling table
   are FITTED ON THE POOL.** ⚠ This falsifies the **paper's precondition on this store**, not the
   paper's law — doing the canonical-placement thing would close **at most one sixth** of the gap.
   *Source:* `EVIDENCE.md` §3.5.
6. **B4 misses on all three seeds** (§8.5).
7. **A lifetime is strictly worse than instant eviction** — composition cost differs from instant
   eviction's by **+0.0562 / +0.0337 / +0.0112** against a declared margin of 1.0, and is paid
   strictly *before* any refusal arrives (`t_pay(50 %)` = **7.992 / 5.994 / 7.992** half-lives vs
   `t_refuse` = **19.943**). ⭐ **96.30–99.83 %** of the composition cost is paid by a **continuous**
   weight while the **discrete** liveness bit has not yet flipped. *Source:*
   `/c/chlu/evidence/outputs/w6-advisor-report.md` §4; `EVIDENCE.md` §5.2 R3; registry `N-I8-7`.
8. **Symbolic refusal: no claim on rows 5 and 6.** An **8-byte rule** ties the field store exactly at
   **(1.0000, 1.0000)** on both rows, and `table_absence` **is** `bitmap8` at a byte multiple of
   **59,392×**. ⛔ Pre-registered as the expected outcome and reported first. ⭐ The surviving claim is
   refuse-after-conjunction-deletion, and it is **expressed by a flag (`coupling_hole`), not by
   geometry** — with a separate 8 + 166 B pair-deletion table matching it exactly. *Source:*
   `/c/chlu/evidence/outputs/w5-advisor-report.md` §3; registry `N-I8-6`.
9. **Row 2 (unseen-valid) belongs to a generic learner** — a parameter-matched continuous-input MLP
   beats the arm by **−0.0192 ± 0.0028** (hub-built, 573,440 params = 100.1 % of the arm) and
   **−0.0170 ± 0.0024** (built independently by `spoke-a`), **3/3 seeds with CI95 < 0** on both.
   *Source:* `/c/chlu/evidence/outputs/w4-advisor-report.md` §2; registry `N-I8-9`.
10. **Placement is exhausted** — row 4 vs the label-MI partition **+1.388 / −12.755 / +2.163**, mean
    **−3.068**, seed spread **14.918**; bar not met, runs 2 and 3 returned unspent. ⚠ And the later w8
    measurement narrows the Advisor's "closed" ruling: at the incumbent's granularity the **label-free**
    cut reads **0.8918 vs 0.8941**, so ⭐ **what is closed is the DIAL, not the criterion.** *Source:*
    `/c/chlu/evidence/outputs/w4-advisor-report.md` §3 + `/c/chlu/outputs/w8-single-design/LOG.md` §4.3;
    registry `N-I8-8`.
11. **Two purpose-built gyms failed their own adoption gates** on two different grounds (§9).
12. ⛔⛔ **A DECLARED ABSENCE:** *"Mexican-hat width / separation sets the interpolation floor and the
    refusal gap"* **does not exist in the programme's own papers.** "interpolation" occurs **0** times;
    "Mexican hat" occurs **once**, as a float64 round-off testbed with no width and no separation
    parameter; "centre-surround", "DoG", "RBF", "lateral inhibition" **0** times. ⭐ Established
    independently twice. ⭐⭐ And the nearest genuinely derived object **inverts** it — the retention
    curve's floor is curvature-**IN**dependent. *Source:* `EVIDENCE.md` §3.9; registry `N-I8-14`.
13. **A1–A8 are asserted without derivation and are UNTESTED as laws.** ⛔ A section must not present
    any of them as a result of the theory — including the **Lifetime Trilemma** (A1), whose
    *consequence* is measured but whose statement has **no proof and no impossibility argument in
    either paper**. ⭐ V2 itself says of A6: *"This is a direction on which this paper offers no
    evidence, but is part of our future work."* *Source:* `EVIDENCE.md` §3.8.
14. **The V1 certificate family (D34–D45) is UNTESTED on the CHLU field store** — [shorts] throughout,
    including V1's own measured limit that `det J = 1` plus zero energy change is **not sufficient**
    for BIBO stability (at exit `b = 5.0`: ΔH = 0.0 exactly, the sub-level test admits, and the
    trajectory escapes — max excursion 0.09 certified vs **104.83** blind). *Source:* `EVIDENCE.md` §3.6.
15. **Five DECLARED NOT-RUNs, each with a launch line and a cost** — NR1…NR5, reproduced verbatim in
    App. K. ⛔ Never a blank, never a zero.
16. **Residual novelty risk** (§10 final ¶).

---

## 3. APPENDICES — appendix-maximalism, nothing pruned at this stage

| app | content | source |
|---|---|---|
| **A** | Full TABLE 1 (all 11 columns + ceilings 1b + caveat columns 1c + DISPLAY rows 1d + instrument audits 1e), per-seed values throughout | `TABLES.md` § TABLE 1 |
| **B** | Full TABLE 2 (encoder-loop, both rows, `A′`/`A″`/`A`, sign-test p, bootstrap CI, both matched-parameter ablations) | `TABLES.md` § TABLE 2 |
| **C** | Full TABLE 3 (CMAZE-1 at the ~60k harness tier: all learned baselines, both table floors, ORACLE, all five BLIND columns, both FRAGMENT-LOOKUP radii, the DISPLAY refuser) | `TABLES.md` § TABLE 3 |
| **D** | Full TABLE 4 + the failure-agreement narrative + the admission cut `tau = 4.144858e-09` and why it has no free parameter | `TABLES.md` § TABLE 4; `/c/chlu/outputs/w8-downdate/LOG.md:618-640` |
| **E** | The w5 composition RESULT in full: the ceiling ladder (`oracle_span_field`, `oracle_content_pool`, `oracle_content_items`, `oracle_content_fine`, `oracle_true_feature`), all controls, and the declared NOT-RUN rivals (`lace` at **95.1×** and `poe_ebm` at **27.1×** the store budget, both refused in their own words because the output is a sampler draw, not a function of the request — *a dash, never a win for us*) | `/c/chlu/outputs/w5-compose-field/LOG.md` §§1–7 |
| **F** | ⭐ **The byte-ledger discipline appendix.** Why the read floats and the sufficient statistic are declared **beside** the store and never netted in; the four-part honest ledger; the 7.48× / 0.991× pair; `rank/d_k` on every row (the incumbent is **already saturated in 2 of its 5 wells**, `d_k = 58 < A = 64`, realised deployed span **308, not 320**) | `w8-single-design/LOG.md` §§2–4.4; `TABLES.md` T1 pins |
| **G** | Instrument audits: exact judge in all four parts; **no tolerance used anywhere in scoring**; **0 / 750** stepper disagreements; **0** impossible examples seen by a non-refuser arm; read is request-only (8 checks per seed, bitwise stable under feature perturbation); **0** plan-map collisions on every row and seed; independent replication of scored run 1 from a clean checkout with the encoder retrained from scratch and **zero numeric differences** | `TABLES.md` §1e |
| **H** | The physics claim ledger D1–D50 + A1–A8 with status labels, and the **34/34, 0 FAIL** numerical reproduction battery (pure stdlib, no `chlu` import, reruns in ~4 s) | `EVIDENCE.md` §§3–4 |
| **I** | The V1 certificate family (D34–D45), [shorts] throughout, with V1's own BIBO counter-measurement | `EVIDENCE.md` §3.6 |
| **J** | The remaining five benchmark requirements (state-space size vs `4·n_params/frag_bytes`; rule-blind oracle as a mandatory gate; held-out path/target diversity; exact zero-tolerance validity; continuous-vs-discrete channel separation) | `BENCHMARK-STATEMENT.md`; `EVIDENCE.md` §5.3 |
| **K** | ⛔ **The DECLARED NOT-RUN register, verbatim, with launch lines and costs** — NR1 (Shapes3D incumbent cell at 3 seeds; seed 0 only, arm 0.8495 / 51 tuples at 454,336 B vs `knn_proto` 0.8588 / 57 at 475,136 B, Δ −0.0093 = TIE under R1; ~40 min CPU but spends one of **1** remaining run); NR2 (the GPU pin; **$26.46** for 3 seeds, A100-40 SPOT at $1.47/h, concurrency ONE so sequential, `kill -9` resumability proved); NR3; NR4; NR5 | `TABLES.md` §DECLARED NOT-RUN |
| **L** | ⭐ **The corrections appendix** — C1–C4 plus the four this spoke adds (§G below). A paper that prints a corrected number without the correction invites the referee to find the uncorrected one | `TABLES.md` §CORRECTIONS; `w10-registry-fold/FINDINGS.md` §3; `EVIDENCE.md` §§4.2–4.4 |
| **M** | Provenance: the 27-entry source register with bytes and sha256 for every file the tables were built from, and the statement that `verify_tables.py` re-resolves **27 sources / 240 cells / 720 per-seed values / 5 figure hashes** from `TABLES.json` alone, **PASS**, with byte-identical figures across rebuilds **and from the bundle** | `TABLES.md` §SOURCE REGISTER; `w10-seeds-and-tables/FINDINGS.md` §0 |
| **N** | The physics-block negatives that do not fit §11: the `knn_density` trap cleared with numbers (`well_sigma / nn_median` = 0.805 / 0.895 / 0.900 / 1.759 / 1.207 — not 1.0 on any well and on **both sides** of 1, so the w5 refusal-rival failure at 0.0203 is **not inherited**); §C.4 measured and not firing (settled minimum 0.326–1.224 σ from its nearest atom, read drawing on 30.9–59.5 of 64 atoms); the mechanism note that **no stored value owns a basin** (all five wells below the Δ/σ = 2 bifurcation at 1.339 / 0.812 / 0.768 / 0.736 / 0.404) ⚠ **with well 0's two minima reported MISSING for the two-atom rule** | `EVIDENCE.md` §2.8, §2.9 |

---

## 4. §B — THE CLAIM REGISTER

⭐ **Acceptance criterion of this spoke:** every claim carries a pointer to the table row or artefact
that supports it. ⛔ **A claim with no pointer was DELETED by this spoke** — see §D.

| # | claim as the paper may state it | class | n seeds / condition | pointer |
|---|---|---|---|---|
| **K1** | Deleting rows from the store is exactly the store refit without them: statistic max relative residual **7.5e-16…8.8e-16**, subspace projector residual **exactly 0.000e+00**, 0 refits, 0 passes, items never kept | **RESULT** | 3 / TEST | `TABLES.md` T4 (w9/maze leg); `w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` |
| **K2** | Independently reproduced by different code and a different author: worst max absolute deviation **1.1255e-09 = 0.009442 float32 ulp at unit scale**, `passed_every_seed = true`, negative control firing every seed | **RESULT** | 3 / TEST | `TABLES.md` T4 (w8/downdate leg); `w8-downdate/DOWNDATE-READY-test.json` `/downdate_equals_refit` |
| **K3** | The two implementations also agree on **where the mechanism breaks** — the same seed-2 failure, at 32.90/35.46 ulp and 14.377 ulp respectively, closed by a no-free-parameter admission cut declared before the re-run | **RESULT (supporting)** | 3 / TEST | `w8-downdate/LOG.md:618-640`; `w8-single-design/LOG.md:481` |
| **K4** | `pca_tensor` beats `mlp_180k` on rows 3 and 4, all three seeds, CI95-low above the declared +1.0 margin on every seed, at 524,288 B tolerance 0 | **RESULT** | 3 / TEST | `w5-compose-field/LOG.md:196,247`; `w5-advisor-report.md` §1 |
| **K5** | It also beats every other rival that can enter those rows (`lstm`, `mlp`, `transformer`, `cont_mlp`, `table`), both rows, all three seeds | **RESULT** | 3 / TEST | `w5-compose-field/LOG.md` §1 |
| **K6** | The win is a property of the **content**, not of the byte budget — `oracle_span_field` − bar is −12.531/−5.224/−5.653 while `oracle_content_pool` − bar is +10.980/+8.224/+8.551 at the same 524,288 B | **RESULT (supporting)** | 3 / TEST | `w5-compose-field/LOG.md` §7(3) |
| **K7** | The composition-winning content forfeits exact per-item deletion by construction; keeping items costs **≈18.6 tuples/corner** on row 4 | **measured cost** | 3 / TEST | `w5-advisor-report.md` §1 final ¶; registry `N-I8-5` |
| **K8** | At the byte pin the two properties are incompatible: the covariance forces **K ≥ 36**, and at K=40/A=7 (**0.991×** the pin) composition is **0.0648** | **val falsification** | val, 0/3 scored | `w8-single-design/LOG.md` §2(2)(3) |
| **K9** | Both properties coexist in one object at **7.48×** the pin (**0.8918** vs incumbent **0.8941**) | ⛔ **unscored val observation** | val, seed 0, 1 arm | `w8-single-design/LOG.md` §3 |
| **K10** | The basis-change carry recovers post-deletion composition **0.0584 → 0.8964 (100.3 %)** with no refit, no data row read, zero deployed bytes — and rewrites **127,488 B of 158,664 B** of the address book | **val measurement** | val | `w8-single-design/LOG.md` §4.1 |
| **K11** | On CMAZE-1 a matched-parameter transformer beats the CLU on every composition row (BAR 1: −0.0560 / −0.0209 / −0.0434) | **RESULT (negative)** | 3 / TEST | `TABLES.md` T1 1a; `w9-maze/LOG.md` §5 |
| **K12** | The C.2 controls do not collapse (`landscape_shuffled` 0.9509, `random_atoms` 0.9601 on row 1; `random_atoms` 0.1414 inside the CLU's seed spread on row 2) ⇒ **no RESULT row is claimed for the CLU on CMAZE-1** | **RESULT (negative)** | 3 / TEST | `TABLES.md` T1 1a |
| **K13** | The mechanism is isolated: the same two halves compose to **1.0000 / 1.0000 / 1.0000** in plan space and **0.1649 / 0.0950 / 0.1449** in latent space | **analysis of scored legs** | 3 / TEST | `w9-maze/LOG.md:288-346`; registry `N-I8-3` |
| **K14** | The information is present (plan coverage mean 1.000) and the factorisation is right (union coverage 0.997, complete for 99.1 %) | **analysis** | 3 / TEST | `w9-maze/LOG.md:288-346` |
| **K15** | The store span tracks the pipeline ceiling to within **0.0138** on every row and seed, exactly equal on **7 of 12** row×seed cells ⇒ the atoms are not the binding constraint | **RESULT (supporting)** | 3 / TEST | `TABLES.md` T1 1b; `/span_ceiling_rows` |
| **K16** | BLIND-SEQUENTIAL is a **request-layer** property: 0.9037 on the hub's split, **0.5317** at the harness tier against the published 0.4980, with FRAGMENT-LOOKUP r=1/r=2 at **0.0000** everywhere | **RESULT (instrument)** | 3 / both tiers | `TABLES.md` T1 1c + T3; C4 |
| **K17** | Unfreezing the encoder with the store on the gradient path hurts (`A′` negative 5/6, p ≤ 0.0013) and the swap loss repairs it (`A″` positive 4/6, p ≤ 0.0003); the net is a wash and **BAR B fired on row 3** | **RESULT (negative)** | 3 / test | `TABLES.md` T2; `w8-encoder-loop/CROSS-ARM.json` `/VERDICT` |
| **K18** | The CLU's margin over the matched-parameter learner **does not grow** under the encoder loop ⇒ a finding about the encoder, not the store | **ablation** | 3 / test | `TABLES.md` T2 ablation rows |
| **K19** | The CRR predicts the settle half-life parameter-free to a factor **1.105** at the median held-out corner and beats a sighted fitted rival **49/49 on every seed** | **RESULT** | 3 / held-out | `EVIDENCE.md` §2.3; `w6-potentials/RESULTS-LEDGER.md` |
| **K20** | The CRR's error sits **entirely** in the `h/h*` bucket 0.5–2 straddling the exceptional point, where the map is provably defective ⇒ **the theory predicts where its own simplest form stops** | **RESULT (supporting)** | seed 0 held-out | `EVIDENCE.md` §2.5 |
| **K21** | ⛔ The same dynamics **cost** composition at every strength at which they move the state (row 4 −39.959 / −46.918 / −39.061), so the CRR supports **no scoreboard claim** | **RESULT (negative)** | 3 | `EVIDENCE.md` §1 |
| **K22** | Decay commutes with deletion exactly: **0 bytes differing in a 2,054,866 B image**, 12/12 cells per seed, 3 seeds, zero tolerance — on a **learned** store | **RESULT** | 3 | `EVIDENCE.md` §3.5 D32a |
| **K23** | ⛔ The post-delete store is **not** byte-identical to the never-inserted store (**65.17–65.39 %** of the image differs), and the break-out shows **83.5 %** of the obstruction is the fitted address book and coupling table, not placement | **RESULT (negative)** | 3 | `EVIDENCE.md` §3.5 D32b |
| **K24** | `n_refuse = 19.9316 × n_half` for every curvature, measured **539 = 19.9432** half-lives on all three seeds; and `μ²` cancels, giving **16/16 horizons byte-identical** ⇒ **no curvature trades refusal against composition** | **RESULT** | 3 | `EVIDENCE.md` §3.5 |
| **K25** | The novelty position: **0 of 3,038** citing works combine frozen-decoder + tensor-head + label-index; highest atom count **4 of 6**; ≈4,050 works screened across three directions; six orthogonal sweeps return zero | **scout result** | — | `NOVELTY.md` §§A–B; registry `N-I8-18` |
| **K26** | The eigenspace-arithmetic line is **explicitly approximate by its authors' own statement**, so exactness is not conceded to it | **scout result** | — | `NOVELTY.md` §D; `NOVELTY/LOG.md` 16:53Z |

---

## 5. §C — TABLE AND FIGURE SLOTTING

### Tables

| paper slot | table | pin summary | source |
|---|---|---|---|
| **Table 1 (main)** | TABLE 4 — exact deletion, two implementations | 3 seeds each, TEST, deviations in their own units, never merged | `TABLES.md` § TABLE 4 |
| **Table 2 (main)** | the w5 composition RESULT | test, 3 seeds, 524,288 B tolerance 0, 49 + 60 paired corners | `w5-compose-field/LOG.md` §1 ⚠ **must be typeset from the LOG — there is no `TABLES.md` row for it** |
| **Table 3 (main)** | TABLE 1 §1a — CMAZE-1 RESULT rows + the C.2 controls | 3/3 scored seeds, TEST, matched params, row 3 printed twice | `TABLES.md` § TABLE 1 |
| **Table 4 (main)** | the compose-XOR-forget ledger (§5) | val, 0/3 scored; 0.991× and 7.48× rows | `w8-single-design/LOG.md` §2 |
| **Table 5 (main)** | the CRR bar table B1–B6 | 3 seeds, held-out, 192,975 rows | `EVIDENCE.md` §2.3 |
| App. A | TABLE 1 §§1b–1e in full | | `TABLES.md` |
| App. B | TABLE 2 in full | 3 seeds, `w8_v1_short` instrument | `TABLES.md` |
| App. C | TABLE 3 in full | 3 seeds, ~60k tier | `TABLES.md` |
| App. K | the NR register | | `TABLES.md` |

### Figures — what exists, and what does not

| paper slot | file | status |
|---|---|---|
| **Figure 1 (headline)** | `/c/chlu/outputs/w10-seeds-and-tables/figures/fig4_exact_deletion.png` | ✅ EXISTS, hash-verified, byte-identical across rebuilds and from the bundle |
| **Figure 2** | `figures/fig1_cmaze1_main_scoreboard.png` | ✅ EXISTS |
| **Figure 3** | `figures/fig2_nway_conjunction_vs_route.png` | ✅ EXISTS |
| **Figure 4** | `figures/fig3_encoder_loop_decomposition.png` | ✅ EXISTS |
| **Figure 5** | `figures/fig5_blind_sequential_on_this_split.png` | ✅ EXISTS |
| **Figure 6 — the composition RESULT** | — | ⛔ **DOES NOT EXIST.** Declared gap. Would plot `pca_tensor` vs `mlp_180k` and the `oracle_content_items` floor on rows 3/4, 3 seeds as dots, from `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json` |
| **Figure 7 — compose XOR forget** | — | ⛔ **DOES NOT EXIST.** Would plot composition against the byte ledger with the pin as a vertical line (0.991× → 0.0648; 7.48× → 0.8918; incumbent 0.8941) |
| **Figure 8 — the CRR** | — | ⛔ **DOES NOT EXIST in any w10 artefact.** The `h/h*` bucket table (§2.5) and the B1/B2 per-corner data are in `w6-potentials/test/s{0,1,2}/results.json`; a figure would have to be built there |
| **Figure 9 — plan vs latent** | — | ⛔ **DOES NOT EXIST.** Would be a 3-bar-group plot of the §6.1 table |

⚠ **Four of the nine figure slots have no artefact.** That is a declared production gap for the Head,
not a silent omission. ⛔ None of them may be drawn from a number this spoke invented; every one has
its source JSON named above.

---

## 6. §D — CLAIMS THIS SPOKE DELETED (no pointer ⇒ deleted, not flagged)

Per the acceptance criterion, these were candidate paper sentences drawn from reports, task files or
narrative summaries that **no artefact on disk supports as stated**. They are recorded here so the
Head can see what was removed and why, rather than rediscovering them.

| deleted sentence | why it was deleted |
|---|---|
| *"exact deletion holds to 1e-16…1e-14, 9/9 seeds, two independent implementations"* | ⛔ **Three defects at once.** "9/9 seeds" is 3 configurations × 3 seeds inside ONE implementation; the band is not shared by both implementations (the other runs 2.842e-14…1.126e-09); and it is not unconditional — it FAILED first at 32.90 ulp. Replaced by K1/K2/K3. *Source:* `w10-registry-fold/FINDINGS.md` §3 |
| *"the store shapes the representation"* | ⛔ BAR B falsifier **fired** on row 3. `w8-encoder-loop/CROSS-ARM.json` `/VERDICT/r3_comp_2way/BAR_B_FALSIFIED = true` |
| *"the CLU composes on a route task"* / any RESULT row for the CLU on CMAZE-1 | ⛔ BAR 1 and BAR 3 both fail; the hub's own pre-registration forbids the row |
| *"an object that composes and forgets inside 524,288 B"* | ⛔ Falsified on val (K8); 0/3 scored |
| *"post-deletion composition holds"* | ⛔ The only held-out number is the failing as-pre-registered one (0.1024 → 0.0000). The repaired number is VAL, seed 0 (NR5) |
| *"multilinear/bilinear factored generation with a request-indexed tensor head"* as a contribution | ⛔ Novelty STOP #1 — TensorFaces 2002, Tenenbaum & Freeman 2000 |
| *"tensor-product label conditioning for unseen attribute combinations"* | ⛔ Novelty STOP — MLC-VAE, ICML 2020 |
| *"covariance downdate then re-eigendecompose equals an exact refit"* as a contribution | ⛔ Novelty STOP #2 (narrowed) — Cao & Yang 2015 owns it from the summation form |
| *"Mexican-hat width / separation sets the interpolation floor and the refusal gap"* | ⛔ **The law does not exist in the corpus** — 0 occurrences of "interpolation", 1 of "Mexican hat" (a round-off testbed). And the nearest derived object INVERTS it |
| *"wave 9 inherits the 0.4980 blind-sequential column as a caveat"* | ⛔ Correction C4 — a request-layer property, not an environment property; the published value reproduces at 0.5317 at the harness tier |
| *"the CRR result supports a scoreboard claim"* | ⛔ Refuted on the same object and the same seeds (K21); Advisor-ruled |
| *"the Lifetime Trilemma"* as a result of the theory | ⛔ Stated as fact in V5 §2.3 and V1 §5 with **no proof and no impossibility argument in either**. Its *consequence* is measured; the trilemma is UNTESTED |
| *"throat latch impact μ² ∝ A/l²"* (A2) and *"the minimum-separation certificate"* (A5) as derived laws | ⛔ Both **refused** by `hub/potentials` as substitutes for the absent law; `A` and `l` are never defined anywhere in the paper, and 1.540000 is a measured constant with no closed form |
| *"the programme's claims need a benchmark that does not exist"* | ⛔ **No scout in this evidence base surveyed the field for one at this wave level.** The requirement list is a specification, not a survey claim |
| *"a benchmark is needed / deserved"* framing generally | ⛔ Same reason; `EVIDENCE.md` §5.3 final ¶ forbids it in terms |
| the Campaign-2 ICLR-long thesis (voraus-AD accept-maker, interference firewall, priced channel, reversible O(1) training) | ⛔ **Different object, different campaign.** `/c/chlu/evidence/outputs/iclr-long-skeleton.md` is a C2-era planning document whose spine (`g7b-torus-voraus`) is PENDING and whose smoke trends negative. ⚠ It is **not** the paper the wave plan commissions — the wave plan's paper object is the **composition-led store (Naive-CLU)**. Recorded so the Head does not merge two skeletons |

---

## 7. §E — THE NEVER-QUOTE LIST (one line each, for the Head's desk)

1. ⛔ Never `1e-14` for the deletion band. Quote **0.009442 float32 ulp** or the true absolute.
2. ⛔ Never **"9/9 seeds"**.
3. ⛔ Never merge T1 with T3 (same environment, different budget), T2 with the incumbent's rows
   (different instrument), or the two deletion legs (different units).
4. ⛔ Never quote the n-way conjunction against the per-route ceiling as a fraction.
5. ⛔ Never print **0.4980** where **0.9037** belongs, or the reverse.
6. ⛔ Never quote the **127,488 B** carry rewrite beside the coupling hole's **1 byte**.
7. ⛔ Never quote the **7.48×** object without its overrun attached.
8. ⛔ Never present the plan-space overlay as a win — it is a reference, unmatched in bytes.
9. ⛔ Never quote `d14_confinement.max_abs_residual` (it is **tautological, exactly 0.0 on 60 cells**);
   quote `hessian_fd_check` (**3.519e-11 / 1.202e-10 / 2.699e-11**).
10. ⛔ Never quote the asymptotic half-life form; quote the exact root.
11. ⛔ Never write D18 up as a confirmation (see §G3).
12. ⛔ Never claim the three novelty-STOP sentences (§1).
13. ⛔ Never state that the field lacks a composition-and-deletion benchmark.
14. ⛔ Never let a RESULT row and a DISPLAY row share a comparator: the only DISPLAY comparator is the
    supervised refuser, and our line is *"no impossible example was ever seen"*.

---

## 8. §F — WHAT THIS DRAFT NEEDS THAT DOES NOT EXIST YET

⛔ Each is a **declared gap with a named source**, never a blank.

| # | gap | what would close it | cost |
|---|---|---|---|
| F1 | **Figure 6** — the composition RESULT has no figure | plot rows 3/4 from `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json` | analyst-minutes, **0 scored runs** |
| F2 | **Figure 7** — compose XOR forget has no figure | plot composition vs byte ledger from `w8-single-design/LOG.md` §2 | analyst-minutes, 0 scored runs |
| F3 | **Figure 8** — the CRR has no figure in any w10 artefact | build from `w6-potentials/test/s{0,1,2}/results.json` (`estimand.corners`, the `h/h*` buckets) | analyst-minutes, 0 scored runs |
| F4 | **Figure 9** — plan vs latent has no figure | the §6.1 table is already tabulated in `w9-maze/LOG.md:288-346` | analyst-minutes, 0 scored runs |
| F5 | **Table 2 of the paper has no `TABLES.json` provenance row** — the composition RESULT was outside the tables spoke's source register | extend `build_tables.py`'s register with `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json` and re-run `verify_tables.py` | analyst-minutes, 0 scored runs |
| F6 | **The per-cell D17/D18 `(γ_window, slope_measured, slope_derived)` rows are in no machine-readable artefact** | whoever owns the w6 branch emits them to JSON | ⛔ **CANNOT be run from the artefacts**; declared reporting gap (`EVIDENCE.md` §4.2, §6) |
| F7 | **The referee spoke's output** was not available at this spoke's window | read `/c/chlu/outputs/w10-referee/` against §§E–H before the prose pass | — |
| F8 | **No abstract, no title.** ⛔ Deliberately not written — the Head writes prose. The abstract's first sentence is fixed by §1.2 and its mechanism sentence by §1.3 | | — |

---

## 9. §G — CORRECTIONS THE HEAD MUST ABSORB BEFORE WRITING

⭐ Eight. Four are the tables spoke's C1–C4; four more this spoke carries forward from the registry
fold and the physics evidence. **None changes a verdict. All change a number the paper would
otherwise print.**

| # | as narrated | as the artefact says | consequence |
|---|---|---|---|
| **C1** | BLIND-SEQUENTIAL band "0.9037–0.9130" | 0.9077 / 0.8905 / 0.9128, mean 0.9037 | the band mixes a mean with a per-seed value; the true minimum is **0.8905** |
| **C2** | deletion "1e-16…1e-14" | worst 1.1255e-09 abs = **0.009442 float32 ulp** | understates by ~5 orders; quote the ulp figure |
| **C3** | n-way row vs its ceiling | row n = 183/192/191 (conjunction); ceiling n = 389/403/414 (per route) | not a fraction of ceiling; print row 3 twice |
| **C4** | "wave 9 inherits the 0.4980 column" | 0.5317 at the harness tier, 0.9037 on the hub's split | a **request-layer** property, not an environment property |
| **C5** | "9/9 seeds, two independent implementations" | 3 configurations × 3 seeds inside ONE implementation; the band is not shared | ⛔ the phrase is wrong as written, in three ways |
| **C6** | the D17 headline pair "+1.1385 derived vs +1.1368 measured" | the **law** reproduces exactly and `> +1` is verified over the whole band, but **+1.1385 is not reproducible from any recorded machine-readable artefact** — `1 + γ/(2−γ) = 1.1385` at γ = 0.243303, which is **not a grid point**, and the per-cell fit windows were never written to JSON | ⚠ the paper's **sharpest** claim currently has its **weakest** artefact trail. A *reporting* gap, not evidence the number is wrong — nothing contradicts it. Fix = F6 |
| **C7** | D18 "underdamped slope confirmed" | derived −1.0050, measured −1.1090, margin **thin on every seed** (0.1368 vs 0.1417 etc.), and the shorts' own values are **−1.006** and **−1.0020 ± 0.0003** — so w6's −1.1090 is the outlier, not the derivation | ⛔ **Do not write D18 up as a confirmation.** Write: *confirmed on the shorts' own designed and emergent testbeds, inconclusive on this store, by a margin the hub itself declared thin* |
| **C8** | `d14_confinement.max_abs_residual = 0.0` in `results.json` | it compares `hess_eig(α)` to `hess_eig(0) + 2α` — **the same algebra on both sides**, so it can only ever pass. The hub caught this itself and replaced it | ⛔ **A writer reading `results.json` cold would quote the wrong field.** Quote `hessian_fd_check` |

⚠ **Two open items the registry fold raised for the Hub and this spoke does not decide:** the
`N89`/`N90` duplicate allocation in the registry, and the fact that the `⟲ 2026-09-11` gen-pivot
negatives still have **no claims-matrix row at all**. *Source:*
`/c/chlu/outputs/w10-registry-fold/FINDINGS.md` §5.

---

## 10. §H — WHAT WOULD PROVE THIS SPOKE WRONG, CHECKED AGAINST THE RESULT

> **The task file's falsifier:** *"The evidence base does not support a contribution paragraph that
> survives the novelty STOP ⇒ report that plainly; it is a finding about the programme, not a
> writing problem to style around."*

**Verdict: the falsifier does NOT fire. A contribution paragraph survives.** Here it is, with every
clause's pointer, so the Head can check the survival claim rather than take it:

> We show that a compositional generative store whose content is a sufficient statistic admits
> deletion that is **exactly** the refit — verified by two independent implementations on two
> environments, three seeds each, on held-out test data, with no refit performed, no pass over
> retained data and items never kept **[K1, K2]**; that the two implementations agree not only on
> where the mechanism works but on where it breaks **[K3]**; that the same store class composes,
> beating a parameter-matched trained learner on both composition rows on all three seeds at an
> asserted byte pin **[K4, K5, K6]**; and that at that pin the two properties are **incompatible**,
> for a reason we measure rather than argue **[K7, K8, K9]**. The composition result does **not**
> transfer to an exact-judge route task, and we isolate why: the read composes **latents** where the
> task composes **plans** — the same two halves compose to 1.0000 in plan space and 0.13 in latent
> space **[K11–K15]**.

**Why it survives the novelty STOP, clause by clause:**

- ⭐ It does **not** claim multilinear factored generation, a request-indexed tensor head, or
  tensor-product label conditioning. Those words are absent from the paragraph by construction.
- ⭐ It does **not** claim the downdate primitive. It claims the *object*: exact deletion of a
  **generative** model, which the crawl finds in **0 of ≈4,050** screened works **[K25]**.
- ⭐ Its mechanism sentence — *downdate the statistic, not the truncated factors* — is a distinction
  the eigenspace-arithmetic line **does not have**, by its own authors' words **[K26]**.
- ⭐ The composition clause is deliberately stated as *"beats a parameter-matched trained learner at
  an asserted byte pin"*, which is a **measured comparison**, not a novelty claim in the crowded
  compositional-generation literature.

⚠ **Three honest weaknesses in that paragraph, stated rather than styled around:**

1. The deletion result and the composition result are on **different objects**, and the object that
   does both is a **val, seed-0, 7.48×-over-budget observation**. The paper's strength is that it
   *measures* the incompatibility; its weakness is that it has no scored object exhibiting both.
2. The programme has **one** environment with three scored seeds and a clean instrument, and that
   environment returns a **negative**. Two of the four tables in the tables spoke's file exist only
   because a second compute pin was never run.
3. The flagship generative cell at the **incumbent** instrument has **one seed** (NR1), and one of
   the remaining generation-table scored runs would have to be spent to get a second.

⛔ **None of the three is a reason to restyle the paragraph.** All three are already in §11 and
App. K, and the Head should keep them there.

---

## 11. WHAT THIS SPOKE DID NOT DO

- ⛔ **No scored run, no model fitted, no number produced.** Every figure above is quoted with the
  path that holds it.
- ⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
- ⛔ **No prose.** The Head writes it. Where a sentence is given verbatim above it is because the
  evidence fixes its content, and it is marked as such.
- ⛔ **Nothing written outside `/c/chlu/outputs/w10-draft-v0/`** and this spoke's own worktree
  (branch `w10/draft-v0`). No `.claude/` directory was created.
- ⛔ **No background job launched, none left running, no box-wide kill at any point.** This spoke
  launched no long-running process at all.
- ⛔ **No pruning.** Appendix-maximalism is in force: nothing was omitted ahead of the dedicated
  pruning passes. §D lists only sentences that had **no artefact**, which the acceptance criterion
  requires be deleted rather than flagged.
