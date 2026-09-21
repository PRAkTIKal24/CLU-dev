# w10 spoke/draft-v0 — FINDINGS (report to the Hub)

**Agent:** `paper-writer`. **Branch:** `w10/draft-v0`, bundled as `w10-draft-v0.bundle` (that branch
only, verified). **Window:** `START_UTC.txt` 2026-09-21T08:49:19Z → `END_UTC.txt`.
**Scored runs: ⛔ ZERO.** No model fitted, no number produced. ⛔ **`$CHLU_CLAUDE_ROOT` never opened.**
**Deliverable:** `/c/chlu/outputs/w10-draft-v0/DRAFT-v0.md` (the skeleton) · `LOG.md` · this file.

---

## 0. GATE — passed, with one naming discrepancy recorded rather than assumed away

| required | found | status |
|---|---|---|
| `/c/chlu/outputs/w10-seeds-and-tables/TABLES.json` | same path, 263,860 B | ✅ |
| `/c/chlu/outputs/w10-physics-section/EVIDENCE.md` | `/c/chlu/outputs/w10-physics-section-evidence/EVIDENCE.md`, 64,732 B | ✅ — that spoke's directory carries a `-evidence` suffix and holds `START_UTC.txt`, `LOG.md`, `BENCHMARK-STATEMENT.md`, `checks/` and its bundle, so it is the named artefact |
| updated negatives registry | `/c/chlu/outputs/w10-registry-fold/{negative_results.md, claims_matrix.md, FINDINGS.md}` — 21 entries `N-I8-1…21`, 13 rows `CM-I8-1…13` | ✅ |

⚠ **Declared NOT-AVAILABLE input:** `/c/chlu/outputs/w10-referee/` was **empty** at 08:50Z. Not a
precondition; recorded so the Hub knows the skeleton was written without it. §§E–H of `DRAFT-v0.md`
are the sections the referee output should be read against.

---

## 1. ⭐⭐ THE ANSWER TO THE FALSIFIER: A CONTRIBUTION PARAGRAPH SURVIVES THE NOVELTY STOP

The task's falsifier was *"the evidence base does not support a contribution paragraph that survives
the novelty STOP."* **It does not fire.** The paragraph is written out in `DRAFT-v0.md` §H with a
pointer on every clause. Why it survives, clause by clause:

- It never uses the three stopped phrases. *Multilinear factored generation*, *request-indexed tensor
  head*, *tensor-product label conditioning* and *the downdate primitive* are absent **by
  construction**, not by hedging.
- It claims the **object**, not the primitive: exact deletion of a *generative* model — found in
  **0 of ≈4,050** screened works, with **0 of 3,038** citing works combining frozen-decoder +
  tensor-head + label-index and a **maximum atom count of 4 of 6** (CausalX)
  (`evidence/outputs/_thinkpad/w8/w8-novelty/NOVELTY.md` §§A–B).
- Its mechanism sentence — ⭐ ***"downdate the statistic, not the truncated factors"*** — is ground
  the `NOVELTY.md` **LOG** gives back and the file alone concedes. Hall–Marshall–Martin subtract
  *deflated* eigenspaces and call their own result *"approximating"*, *"tends to instability"*,
  *"anecdotal in character"* (`NOVELTY.md` §D; `w8-novelty/LOG.md` 16:53Z).
  ⛔ **A draft written from `NOVELTY.md` alone would have given this away.** The task file was right
  to mandate both.

---

## 2. ⛔ THE ONE PLACE THE SIBLING SPOKES' PICTURE IS INCOMPLETE — and the Hub should act on it

⭐⭐ **`TABLES.md` contains no composition RESULT, and the programme has one.**

The tables spoke's source register (27 files) covers **w8, w9 and the env scout only**. The
programme's 3-seed, held-out-TEST, pre-registered composition win is **w5 `hub/compose-field`**:

| | row 4 (49 n-way corners) | row 3 (60 pair sub-corners) |
|---|---|---|
| mean paired Δ vs `mlp_180k` | **+10.143 / +8.673 / +7.796** | **+4.033 / +2.483 / +2.417** |
| bootstrap 2.5 % quantile | +6.632 / +5.816 / +5.082 | +2.117 / **+1.033** / +1.200 |
| bar (Δ ≥ +1.0 **and** CI95-low > +1.0, every seed) | ⭐ **PASS 3/3** | ⭐ **PASS 3/3** |

`test`, cache generation 2, seeds 0/1/2, 20 arms scored per seed, zero failures, store **524,288 B
tolerance 0** asserted per leg, parameter-matched to **−0.01 %**.
*Paths:* `/c/chlu/outputs/w5-compose-field/LOG.md:196,247`;
`/c/chlu/evidence/outputs/w5-advisor-report.md` §1; raw
`/c/chlu/outputs/w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json` (all three verified present).

⚠ **This is NOT a contradiction of the tables spoke.** Its *"no Shapes3D composition scoreboard at
3 seeds"* is about the **incumbent generation-cell instrument** (K=64, A=1,216, `eval_b`, a validity
*rate*) — that is `NR1` and has one scored seed. compose-field's metric is **distinct valid novel
tuples per corner** on Montero corners. ⛔ **Different instrument; the two do not share a table.**

⇒ **Hub action, cheap and zero scored runs:** extend `build_tables.py`'s source register with the
three `READ-TABLE.json` files and re-run `verify_tables.py`, so the paper's Table 2 has the same
machine-checkable provenance as Tables 1/3/4. Listed as gap **F5** in `DRAFT-v0.md` §F.

---

## 3. THE SKELETON, IN ONE TABLE

Lead = deletion. Negatives in main text. Physics scoped. ⛔ The Head writes the prose.

| § | what it carries | the number it turns on |
|---|---|---|
| **1** Introduction | the problem (generative deletion is called out of reach), the contribution, the mechanism sentence, the honest-thickness sentence | — |
| **2** The object | unsupervised principal-subspace content · request-only tensor address book · frozen decoder · 524,288 B tolerance 0 | address bytes 189,344–195,064 B declared **beside**, never netted in |
| **3** ⭐⭐ **Exact deletion — THE LEAD** | two implementations, 3 seeds each, TEST, negative control firing every seed | statistic **7.522e-16 / 8.826e-16 / 7.818e-16**, projector **0.000e+00**; worst absolute **1.1255e-09 = 0.009442 float32 ulp** |
| **4** Composition at a byte pin | the one RESULT against a trained learner | **+10.143 / +8.673 / +7.796** (row 4); floor `oracle_content_items` beaten by **+31.224 / +25.204 / +25.878** |
| **5** ⭐⭐ **compose XOR forget** — the paper's own open problem | covariance forces **K ≥ 36**; at 0.991× the pin composition is **0.0648**; at **7.48×** it is 0.8918 vs incumbent 0.8941 | ⛔ val, seed 0, **0/3 scored** — an observation, quoted with its overrun or not at all |
| **6** ⛔ **No transfer to a route task** — main text | BAR 1 **−0.0560 / −0.0209 / −0.0434**; BAR 3 fails (`random_atoms` **0.9601**) ⇒ no RESULT row claimed | **PLAN 1.0000 / 1.0000 / 1.0000 vs LATENT 0.1649 / 0.0950 / 0.1449** |
| **7** The store does not shape the representation | BAR B fired on row 3 | A′ negative 5/6 (p ≤ .0013); A″ positive 4/6 (p ≤ .0003); margin over the learner does not grow |
| **8** Physics, scope limit first | ⛔ *"predicts a settle half-life; supports no scoreboard claim"* | B1 **0.0433 / 0.0435 / 0.0431** (factor 1.105); B2 **49/49 on every seed** vs a *sighted, fitted* rival; ⛔ B4 misses ×3 |
| **9** Benchmark requirement list | ⛔ a specification, **not** a survey claim | O3 composition-blind **1.0000**; optimality gap drops TW-B **0.9825 → 0.3900**; T1 lookup **0.0000** |
| **10** Related work | the two literatures the object joins, cited as owners first | 0 / 3,038 · 4 of 6 max · ≈4,050 screened |
| **11** Limitations | 16 numbered measured facts, each already placed above | five DECLARED NOT-RUNs with launch lines |
| **App. A–N** | appendix-maximalism, nothing pruned | |

---

## 4. ⭐ THE ACCEPTANCE CRITERION, DISCHARGED — and what it deleted

> *"Every claim carries a pointer to the table row or artefact that supports it. A claim with no
> pointer is DELETED by this spoke, not flagged for later."*

- **26 claims** are registered in `DRAFT-v0.md` §B (`K1`…`K26`), each with class, seed count,
  condition and the path/row that holds it.
- **16 candidate sentences were DELETED** and are listed with their reasons in §D. The ones the Hub
  should know about:
  1. ⛔ *"1e-16…1e-14, 9/9 seeds, two independent implementations"* — **wrong in three ways at once**
     (see §5 below).
  2. ⛔ *"the store shapes the representation"* — BAR B **fired**.
  3. ⛔ *"an object that composes and forgets inside 524,288 B"* — falsified on val, 0/3 scored.
  4. ⛔ *"Mexican-hat width / separation sets the interpolation floor"* — **the law does not exist in
     the corpus**, and the nearest derived object inverts it.
  5. ⛔ *"the programme's claims need a benchmark that does not exist"* — **no scout in this evidence
     base surveyed the field for one** at this wave level.
  6. ⛔ **The entire Campaign-2 ICLR-long thesis** in `/c/chlu/evidence/outputs/iclr-long-skeleton.md`
     (voraus-AD accept-maker, interference firewall, priced channel, reversible O(1) training). ⚠ It
     is a **different object from a different campaign**, its spine `g7b-torus-voraus` is PENDING and
     its smoke trends negative, and the wave plan's paper object is the **composition-led store
     (Naive-CLU)**. ⭐ **Recorded explicitly so no one merges two skeletons.**

---

## 5. ⛔ EIGHT CORRECTIONS THE PROSE PASS MUST ABSORB (four inherited, four carried forward)

None changes a verdict. All change a number the paper would otherwise print. Full table in
`DRAFT-v0.md` §G.

| # | the defect in one line |
|---|---|
| **C1** | the BLIND-SEQUENTIAL "band" mixes a 3-seed mean with a per-seed value; the true minimum is **0.8905** |
| **C2** | the deletion range **`1e-14` understates the worst deviation by ~5 orders**; quote **0.009442 float32 ulp** |
| **C3** | the n-way row is a CONJUNCTION and its ceiling is PER ROUTE — `0.0284 vs 0.5255` is **not** a fraction |
| **C4** | **0.4980 is a request-layer property, not an inherited environment caveat** — it reproduces at **0.5317** at the harness tier |
| **C5** | ⛔ **"9/9 seeds" is 3 configurations × 3 seeds inside ONE implementation**, the band is not shared by both implementations (the other runs 2.842e-14…1.126e-09), and the assertion **failed first** at 32.90 ulp |
| **C6** | ⚠ **the paper's sharpest number has its weakest artefact trail:** `+1.1385` is `1 + γ/(2−γ)` at γ = 0.243303, which is **not a grid point**, and the per-cell fit windows are in **no machine-readable artefact**. A reporting gap, not evidence the number is wrong |
| **C7** | ⛔ **do not write D18 up as a confirmation** — the shorts' own values are −1.006 and −1.0020 ± 0.0003, so w6's −1.1090 is the outlier, and the hub itself called the margin thin |
| **C8** | ⛔ `d14_confinement.max_abs_residual` is **tautological** (the same algebra on both sides, exactly 0.0 on 60 cells). Quote `hessian_fd_check` (**3.519e-11 / 1.202e-10 / 2.699e-11**) |

⭐ **C5's strongest form is the one to keep, not the one to drop:** the two implementations agree on
**where the mechanism breaks** — the same seed-2 failure, independently, with different code and its
own sign convention (14.377 ulp against 32.90/35.46). That is a better paragraph than nine clean
seeds would have been, and `DRAFT-v0.md` §3 places it in main text.

---

## 6. ⛔ WHAT THE DRAFT NEEDS THAT DOES NOT EXIST — declared, never blank

| # | gap | cost |
|---|---|---|
| **F1–F4** | **four of nine figure slots have no artefact**: the composition RESULT, compose-XOR-forget, the CRR, and plan-vs-latent. Every one has its source JSON named in `DRAFT-v0.md` §F | analyst-minutes, **0 scored runs each** |
| **F5** | the composition RESULT has no `TABLES.json` provenance row (§2 above) | analyst-minutes, 0 scored runs |
| **F6** | ⛔ **CANNOT be run from the artefacts** — the per-cell D17/D18 `(γ_window, slope_measured, slope_derived)` rows were never written to JSON. Needs whoever owns the w6 branch | one emit |
| **F7** | the referee spoke's output was unavailable in this window | — |
| **F8** | ⛔ no abstract, no title — **deliberately**; the Head writes prose | — |

⛔ **Five DECLARED NOT-RUNs are carried into the paper itself** (App. K, verbatim with launch lines
and costs): NR1 Shapes3D incumbent cell at 3 seeds (**1 generation-table run left**); NR2 the GPU pin
(**$26.46**, A100-40 SPOT, concurrency ONE); NR3 the single-design object (**0/3, budget intact**);
NR4 maze-encoder-loop (**contingency did not fire**); NR5 post-deletion composition with the repaired
deletion on TEST (**VAL seed 0 only, −0.0120**; the maze budget is 3/3 spent).

---

## 7. THE THREE HONEST WEAKNESSES OF THE PAPER AS THE EVIDENCE STANDS

⛔ Stated here because they are the Hub's to price, not the writer's to style around. All three are
already in `DRAFT-v0.md` §11 and §H and the Head should keep them there.

1. **The lead result and the composition result are on different objects.** The object that does both
   is a **val, seed-0, 7.48×-over-budget observation**. The paper's strength is that it *measures*
   the incompatibility at the pin; its weakness is that no **scored** object exhibits both.
2. **One environment with three scored seeds and a clean instrument, and it returns a negative.**
   Two of the tables spoke's four tables exist only because a second compute pin was never run.
3. **The flagship generative cell at the incumbent instrument has one seed** (NR1, Δ −0.0093 = TIE
   under R1). Closing it spends one of the **1** remaining generation-table scored runs.

---

## 8. COST AND DISCIPLINE

- ⛔ **Zero scored runs, zero compute, zero GPU, no background job launched, no PID captured, no
  box-wide kill possible or issued.** Reading and writing documents only.
- ⛔ **`$CHLU_CLAUDE_ROOT` not read at any point.**
- ⛔ **Nothing written outside `/c/chlu/outputs/w10-draft-v0/`** and this spoke's own worktree
  (`projects/w10-draft-v0/`, additive). **No `.claude/` directory created.**
- Files read, in full: the three w10 sibling deliverables; `NOVELTY.md` **and** its `LOG.md`;
  `w5-compose-field/LOG.md`; `w5-advisor-report.md`; `w4-advisor-report.md` §§2–3;
  `w6-advisor-report.md` §4; `w8-single-design/LOG.md` (revised final report);
  `w9-maze/LOG.md:280-350`; `advisor-wave-plan-w8-w10.md`; `iclr-long-skeleton.md` (read **to rule it
  out**, and the ruling is recorded in §4.6 above).
- Branch `w10/draft-v0` committed and bundled as `w10-draft-v0.bundle` — **that branch only, never
  `--all`** — and the bundle verified.
