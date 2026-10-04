# w13 spoke/draft-revise — FINDINGS (report to the Hub)

**Agent:** `paper-writer`. **Branch:** `w13/draft-revise`. **Worktree:**
`C:\chlu\worktrees\CHLU-w13-draft-revise`. **Box** FORGIS_14, **1 thread**.
**Window:** `START_UTC.txt` **2026-09-22T06:09:50Z** → `END_UTC.txt`.

> ⛔ **SCORED RUNS: ZERO. COMPUTE: ZERO. GPU: ZERO. Background jobs / PIDs launched: ZERO.
> Box-wide kill: NEVER issued, never possible.** ⛔ `$CHLU_CLAUDE_ROOT` was **not read**.
> ⛔ **`/c/chlu/evidence/` was not written.** Nothing was written outside
> `/c/chlu/outputs/w13-draft-revise/` and this worktree. No `.claude/` directory was created.
> ⛔ **Another hub was running on this box** (`w13/encoder-adapt`, and `w12b/nr6-and-kstar` until
> 06:07:28Z). No session, worktree, output directory or process of any other agent was touched,
> listed or signalled.

**Deliverables:** `DRAFT-v1.md` (1,308 lines, 124,769 B), `FINDINGS.md`, `LOG.md`, `START_UTC.txt`,
`END_UTC.txt`, `w13-draft-revise.bundle`.

---

## 0. ⭐⭐ THE THREE THINGS TO READ FIRST

### 0.1 ⭐ ALL SIXTEEN MUST-FIXES ARE CLOSED, AT ZERO SCORED RUNS

`REFEREE-v0.md` rejected the draft with MF1–MF16. **Every one is closed with the change named and
the artefact that licenses it — `DRAFT-v1.md` §A, one row per item.** None required a scored run, a
fitted model or a new measurement, which is what the referee predicted.
Nine of the sixteen were already closed at the **artefact** level by `w12r/artefact-mustfix` and
`w13/seeds-and-tables`; this spoke closed them at the **draft** level and closed MF1, MF2, MF5 and
the prose halves of MF3, MF4, MF12, MF13, MF15 and MF16 itself.
**SF1–SF20, the eight nice-to-haves and the referee's one promoted item (B3's max over γ) are also
placed**, each at its own number in the section that carries it (`DRAFT-v1.md` §A tail).

### 0.2 ⛔⛔ THE HEADLINE MOVED AGAIN, **AFTER** THE TASK FILE WAS WRITTEN — AND IT REVERSES

The brief instructed: *"the composition claim does not survive an honest byte match … report n = 3 as
unexplained, not as a win"* and *"T7, the headline scored row, is currently 1 SEED … `w12b` is running
the remaining seeds; if its numbers land before you finish, use them and say so."*

⭐ **They landed.** `/c/chlu/outputs/w12b-nr6-and-kstar/END_UTC.txt` = **2026-09-22T06:07:28Z**, two
minutes before this spoke's START. **This spoke re-derived both load-bearing results from the JSON
itself rather than quoting that hub's prose.**

| | brief's state (w12, K = 5) | measured now (w12b, K\* = 32) | artefact |
|---|---|---|---|
| does the store clear a **1×-matched** learner? | ⛔ **no rung** under the every-seed bar | ⭐⭐ **all five rungs, every seed**; all 15 margins positive, **minimum +3.117** | `KSTAR-LADDER.json` `/verdict/kstar_K32`, `/legs/{n}/kstar_K32/{s}/margin` |
| T7's seed count | ⚠ **1** | ⭐ **3** — row 3 **32.350 ± 0.815**, row 4 **67.177 ± 1.320**; the seed-0 headline sits **inside** the spread | `NR6-T7.json` `/arms/cov_tensor` |
| controls | "0.000 throughout" | ⚠ at K\*, **13 of 15 exactly 0.0000**; **n = 3 seeds 0 and 1 read 0.0083** | `w12b .../FINDINGS.md` §1 |
| the ladder's 1× opponent | "1,029,508 params" | ⛔ measured **3,131,564 – 3,260,030** | `w12-composition-ladder/runs/*/rows_learned_1x.json` `/summary/param_count` |

⭐⭐ **The `bridge` arm makes the flip attributable and it is the most important single artefact of
the wave.** Holding the K\* construction fixed and setting K back to 5, the store **still loses at 4
of 5 rungs** — the identical pattern to the ladder's own K = 5 verdict. ⇒ **changing the construction
alone flips nothing; the flip is the K step, worth 62–79 % of every swing.**
(`KSTAR-LADDER.json` `/margin_swing_decomposed`; ⚠ **one seed**.)

⇒ **`DRAFT-v1.md` leads with exact deletion, as instructed, and makes the composition claim
explicitly K-CONDITIONAL** (§4.2). The K = 5 verdict is **withdrawn from main text** and carried in
the appendix *with its sign stated as a property of K = 5, not of the store.* ⛔ Neither reading is
suppressed and neither is presented as the whole.

### 0.3 ⛔ THE DISTINCTION THE BRIEF'S HEADLINE ELIDES, AND IT MATTERS TO THE PAPER

*"The composition claim does not survive an honest byte match"* is true of **the single design on the
ladder** and is **not** true of the w5 arm.

- `learned_1x` is matched to **the single design's own total** — the leg files say so in words:
  *"1x the single design's own MEASURED total (12,967,352 B = 24.73× the 524,288 B reference pin)"*,
  3,241,076 params on rung 2 seed 0 (`LADDER.json` `/rungs/2/arms/learned_1x/per_seed/0`).
  ⇒ ≈18× `pca_tensor`'s deployed bytes. **`pca_tensor` has no 1×-matched opponent on the ladder at
  all** — three of the four 1× families are **NR8**, and none was fitted at *its* size.
- The one honest byte match `pca_tensor` **does** have is w5's, and it **wins** it: total deployed
  **713,632 / 719,352 / 719,352 B against the rival's 715,808 B** (99.70 % / 100.50 % / 100.50 %),
  parameters **178,408 / 179,838 / 179,838 vs 178,952** (−0.304 % / +0.495 % / +0.495 %), beating it
  on **both** rows on **all three seeds** with CI95-low above the pre-declared +1.0
  (`TABLES.md` T6, T11; `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`).

⇒ **The paper's honest architecture, and what `DRAFT-v1.md` is built on:** composition is carried by
an object matched to its own bytes on a scored TEST leg at 3 seeds (w5); the **extra** bytes of the
four-part ledger are what buys **exact deletion** — at K = 5, where `pca_tensor` at **1.358×–1.372×**
scores identically to four decimals, at every rung and every seed, to the single design at
**23.886×–24.875×** (`TABLES.md` T8 byte ledger; `BYTES.json` `/summary`; registry `N328`).
⚠ **That equality is a K = 5 statement. `pca_tensor` was never run at K\*** — **OPEN-2 / F11.**

---

## 1. WHAT WAS READ, AND WHAT WAS RE-DERIVED RATHER THAN QUOTED

| read whole | why |
|---|---|
| `/c/chlu/tasks/w13-spoke-draft-revise.md` | the job, including the standing-rules line |
| `evidence/outputs/w10-draft-v0/DRAFT-v0.md` (753 lines) | the target |
| `evidence/outputs/w10-referee/REFEREE-v0.md` (631 lines) | 16 MF, 20 SF, 8 NTH, 1 promoted, the keep-list, the rewritten paragraph |
| `evidence/outputs/w12-advisor-report.md` (158 lines) | what changed in wave 12 |
| `evidence/outputs/w13-seeds-and-tables/{FINDINGS.md, TABLES.md}` | the corrected tables — definitions, T1 §§1c–1e, T4, T5, T6, T7–T12, C1–C19, figures |
| `evidence/outputs/w12r-artefact-mustfix/README.md` | which MF the generator already closed |
| `evidence/outputs/w13-registry-fold/FINDINGS.md` | N326–N336, CM-W12-1…10, `N-I8-21`(b), the six open flags |
| `evidence/outputs/w12-positioning-scout/POSITIONING.md` §§4, 5.0–5.5 | the rung-3 ruling for §8 |
| `evidence/outputs/w12-deletion-baselines/FINDINGS.md` §§0–2 | the naive-vs-naive comparator rows |
| ⚠ `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` | **outside the evidence fold**, read on the task file's own instruction |

**Re-derived by this spoke from raw fields, not taken from prose** (⛔ the standing rule that the Hub
was wrong three times where it paraphrased):

1. **All 15 K\* per-seed margins**, from `KSTAR-LADDER.json` `/legs/{n}/kstar_K32/{seed}/margin`:
   n = 2 **+6.667 / +3.117 / +3.283** · n = 3 **+17.875 / +12.808 / +11.533** · n = 4 **+12.700 /
   +8.742 / +8.625** · n = 5 **+10.350 / +6.517 / +8.200** · n = 6 **+9.375 / +6.108 / +8.450**.
   ⭐ All positive; minimum **+3.117**. The hub's §1 table quotes the worst seed per rung and agrees.
2. **T7's three-seed rows**, from `NR6-T7.json` `/arms/cov_tensor`: row 3 31.600 / 32.233 / 33.217;
   row 4 66.959 / 65.980 / 68.592; judge accuracy 0.9012 / 0.9048 / 0.9362; K = 5 seed 0 28.817 /
   62.122 / 0.8285.
3. **`learned_1x`'s byte matching**, from `LADDER.json` `/rungs/2/arms/learned_1x/per_seed/*` — the
   basis of §0.3 and correction **C20**.
4. **The K\* legs' arm inventory**, from `w12b .../kstar_ladder/*/` — `single_design_K32`,
   `learned_1x` and three controls. ⛔ **No `pca_tensor` arm exists at K\*** (OPEN-2).
5. **The K\* legs' status**, from `KSTAR-LADDER.json` `/REPLICATION_NOT_SCORED` = `true` and
   `kstar_ladder/*/LEG.json` `/corner_check/all_rows_held_out` = `true` (288,000 train / 192,000
   test) — held out, 3 seeds, **0 scored runs drawn** (OPEN-4).

---

## 2. THE SIXTEEN MUST-FIXES — ONE LINE EACH. **Full disposition with the how: `DRAFT-v1.md` §A.**

| MF | status | the change, in one line |
|---|---|---|
| MF1 | ⭐ **CLOSED** | the STOP-violating pointer is **deleted**; §1.2 repoints at `DRAFT-v1.md` §I, which avoids the phrase by construction |
| MF2 | ⭐ **CLOSED** | *"content **derived from** a retained sufficient statistic; downdating the statistic **induces** the refit"*, with the statistic's byte cost in the same sentence |
| MF3 | ⭐ **CLOSED** | four-part ledger on every store table, never netted: **24.5466× / 23.8422× / 23.6697×**, 2.19 %, 4.0897×, + the 16,384 B declared residual |
| MF4 | ⭐ **CLOSED** | the **STORE-SCOPED** definition printed before the table, verbatim from `TABLES.json` `/exactness_definition`; frozen encoder and decoder **declared outside the deletable object** |
| MF5 | ⭐ **CLOSED** | *"exact is not free"* at the end of §3: 0.1024 → 0.0000; eviction −2.0899 / −0.9101 / −6.7416 vs the hole's 0.0000 |
| MF6 | ⭐ **CLOSED** | `cut_is_marginal` 3/3 and `a1_subspace` failing seed 2 at 60.36 vs its own 66.11 floor, in T4's caption |
| MF7 | ⭐ **CLOSED** | the w8 leg is **ASSERTION BATTERY · 0 SCORED RUNS**; the artefact was not relabelled to match the table |
| MF8 | ⭐ **CLOSED** | the spoke's own field **4.5475e-13 / 1.1255e-09 / 2.8422e-14**, header naming both deletions |
| MF9 | ⭐ **CLOSED** | **two** unresolvable legs, each against its own floor, stated the strong way |
| MF10 | ⭐ **CLOSED** | §6 relabelled **CMAZE-2**, defined in one line; T3 stays CMAZE-1 proper; never-quote #3 fixed |
| MF11 | ⭐ **CLOSED** | cross-maze **0.0006 / 0.0000** vs same-maze **0.0741 / 0.5114**, both labelled; 0.5114 promoted into §6.2(i) |
| MF12 | ⭐ **CLOSED** | **−0.304 % / +0.495 % / +0.495 %**, repointed to `w5-compose-field/LOG.md` **01:58 UTC** |
| MF13 | ⭐ **CLOSED** | total deployed **713,632–719,352 vs 715,808 — ±0.5 %**, a stronger claim than v0's |
| MF14 | ⭐ **CLOSED** | BLIND columns for rows 2 and 3 into main text; **row 4 DECLARED NOT MEASURED** |
| MF15 | ⭐ **CLOSED** | §11 item 17: **0.9781 ≈ 0.9801**, false-refusal **0.8182 vs a pre-registered 0.20**; no refusal win claimed |
| MF16 | ⭐ **CLOSED + superseded** | the K ≥ 36 qualifier printed with its four unmeasured siblings and the excluded float32 point; ⭐ **and the nine-K sweep replaces the argument with a measurement — minimum honest total 3.9942× at K = 24** |

---

## 3. ⛔ SIX CORRECTIONS THIS SPOKE CARRIES OR ISSUES

**C20 ⭐ NEW — issued by this spoke.** `w12-advisor-report.md:146` says the ladder's verdict stands
against a **1,029,508-param** opponent. Measured, `learned_1x` carried **3,131,564 – 3,260,030
params** on all 15 legs. 1,029,508 is the **crossover construction's** K = 5 figure, read from a file
whose first key is `SMOKE_NOT_A_CLAIM: true`. The two conventions price the same statistic
differently — full-square **12,284,920 B** vs symmetric-triangle **3,391,552 B** — on different
partitions. *"Roughly halved"* describes the **bridge → K\*** step (1.92×), **not** the ladder's
opponent, against which K\* is **5.92–5.99×** smaller. *Source:* `w12b .../FINDINGS.md` §3, verified
by this spoke against `rows_learned_1x.json`.

**C21 ⭐ NEW — carried.** *"Seed 4 has no external classifier"* understates it: **seed 3 has none
either.** Both pass the generation-2 cache rule and **neither can be judged** —
`decoder_v1_s{3,4}.pkl` and `gen-external-classifier/seed_{3,4}` are both **ABSENT**;
`winner_hashes.json`'s keys are exactly `['seed_0','seed_1','seed_2']`. ⇒ ⛔ **no row in this
programme can exceed 3 seeds, and that is a measurement, not an omission.**

**C15 amended.** The brief's *"4 seeds"* for the scored crossover row was corrected to **1** by
`w13/seeds-and-tables`; it is now **3**, via w12b runs 2/3 and 3/3.

**C16, C18, C11, C12** are carried verbatim into `DRAFT-v1.md` §G. **C6-phys / C7-phys / C8-phys**
(the +1.1385 trail, D18, `d14_confinement`) are carried unchanged: ⛔ **D18 is not written up as a
confirmation**, per the brief.

⚠ **And a reading this spoke sharpened rather than repeated.** The advisor report reports the store
*winning* at n = 3 on the K = 5 ladder. **On the pre-declared every-seed bar it does not:** the
per-seed margins are **+4.8167 / −0.0333 / +7.9417**, so it clears on **2 of 3 seeds** and the rung
fails. `TABLES.md` T8's own verdict says so — *"under the pre-declared every-seed rule it clears at
NO rung"* — and the registry files the same reading (`N326`). ⭐ **Report n = 3 as unexplained, as
the brief says, and as a bar failure, which is stronger.**

---

## 4. ⛔ DECLARED OPEN ITEMS — ten, each with its reason. **Nothing was silently dropped.**

| # | open | reason |
|---|---|---|
| **OPEN-1** | ⛔⛔ **`TABLES.md` T7 and T8 are superseded by w12b and the manifest has not been rebuilt.** T7 prints `seeds (n) = 1` where the artefact has 3; T8's *"clears at NO rung"* is a K = 5 statement | rebuilding `TABLES.json`/`TABLES.md`/figures and re-running `verify_tables.py` is the **tables spoke's** job and needs the generator. ⛔ A spoke does not edit another spoke's artefact. **The paper cannot freeze before this is done.** |
| **OPEN-2** | ⚠ **`pca_tensor` was never run at K\***, so *"the ledger buys deletion, not composition"* is a **K = 5** statement | one replication leg per rung at K\*; harness exists (`w12b_ladder_kstar.py`). **Not run, not costed here** (F11) |
| **OPEN-3** | ⚠ **T9 / T10-ours (deletion, refusal on the K\* object) are ONE seed** | `single-design` is now **3/3 spent**. A further seed needs a new scored-ledger row — **a Hub decision** (F12) |
| **OPEN-4** | ⚠ **the K\* 1× win is a REPLICATION, not a scored row**; the bridge attribution is **one seed** | `KSTAR-LADDER.json` `/REPLICATION_NOT_SCORED` = true. Stated at every mention |
| **OPEN-5** | ⚠ **four figures do not exist** (F1, F3, F4, F10); `fig6` exists only at K = 5 | analyst-minutes, 0 scored runs — but not this spoke's to draw, and ⛔ none may be drawn from an invented number |
| **OPEN-6** | ⚠ **the extra abstain output for MLP / cont-MLP / LSTM is NOT BUILT** | `w12/deletion-baselines` ran before the Head's naive-vs-naive amendment. DECLARED NOT-RUN, carried |
| **OPEN-7** | ⚠ **+1.1385 has no machine-readable trail**; F6 cannot be run from the artefacts | a reporting gap in the w6 branch. ⛔ D18 is not written up as a confirmation |
| **OPEN-8** | ⚠ **`N-I8-21`(b), the six registry-fold flags, and the `pca_tensor`-vs-single-design column choice** | ⛔ Hub rulings. A spoke does not promote a standing rule or confirm a claims-matrix version unilaterally |
| **OPEN-9** | ⚠ **`w12b` is not in `/c/chlu/evidence/outputs/`** | folding the evidence base is the Hub's. Read on the task file's instruction; **every citation flagged** |
| **OPEN-10** | ⚠ **`w12-advisor-report.md` carries two numbers its own successors correct** (C20, C21) | editing another wave's report is not this spoke's. Both printed as corrections |

**Carried DECLARED NOT-RUNs, restated so they are not lost:** NR1 · NR2 (the GPU pin, **$26.46**) ·
NR3 (**partially discharged** — T7 is now its 3-seed composition instance; T9/T10 are 1 seed) ·
NR4 · NR5 · NR6 (K = 5 on TEST at seeds 1 and 2; the bridge at seeds 1 and 2) · NR7 (any 5-seed
row — **impossible**, C21) · NR8 (the 1× comparator column on T7) · NR9 (the refusal head-to-head) ·
NR10 (`generation-table` run 3, ⭐ **unspent**). Both unlearners remain **measured on the fit pool and
unscored**, HELD post-submission by Head ruling. SISA and the voting CLU ensemble are **UNBUILT by
ruling.**

---

## 5. ⛔ WHAT WOULD HAVE PROVED THIS SPOKE WRONG — checked against the result

> *"The evidence cannot support a contribution paragraph that survives the novelty STOP ⇒ say so
> plainly. That is a finding about the programme, not a writing problem to style around."*

⭐ **It does not fire. A paragraph survives, and it is stronger than v0's** — `DRAFT-v1.md` §I, with
every clause pointed at a K-number and every K-number at a file. It avoids all three STOP phrases by
construction, claims the **object** and not the downdate primitive, and states the composition clause
as a measured comparison **at stated bytes and a stated K**.

⭐ **The acceptance criterion is met:** every claim in `DRAFT-v1.md` carries a pointer to a table row
or artefact, and **every MF1–MF16 is CLOSED with how** (§A). Claims with no pointer were **deleted,
not flagged** — §D lists 28, of which **four are new WITHDRAWALS** forced by w12b (the K = 5 verdict
as *the* verdict; *"T7 is a one-seed row"*; *"the controls read 0.000 throughout"*; *"the ladder's
opponent carried 1,029,508 params"*).

⚠ **Five honest weaknesses in the paragraph, stated in the draft and not styled around:** the
deletion and w5 composition results are on **different objects**; at K\* exact deletion **does not
change what the object answers** (32.25 → 32.25); the K\* 1× win is a **replication** and the bridge
is **one seed**; **no row can exceed 3 seeds**; and the one environment with three scored seeds and a
clean instrument returns a **negative**.

---

## 6. ⭐ WHAT THE HUB IS ASKED TO RULE ON

1. ⛔⛔ **OPEN-1 first, before anything else.** The corrected tables now disagree with their own
   sources in two rows. Commission the rebuild, or the paper cites a table its evidence has
   overturned.
2. ⛔ **Which composition reading leads the paper.** This spoke recommends: **w5 at matched total
   bytes (3 seeds, TEST, scored) leads; the K = 5 → K\* flip is reported as a K-conditional finding
   with the bridge attributing it; the K = 5 verdict goes to the appendix with its sign labelled a
   property of K = 5.** `DRAFT-v1.md` §4 is built that way and either alternative is a small edit.
3. ⚠ **Does the bridge column go in the paper?** w12b recommends one appendix row; this spoke agrees
   — ⛔ **without it, "K\* flips it" is unattributable** — and notes it is **one seed**.
4. ⚠ **`pca_tensor` at K\* (F11/OPEN-2).** The programme's single cleanest sentence — *the ledger
   buys deletion, not composition* — is currently a K = 5 statement. One replication would settle it
   at **0 scored runs**.
5. ⚠ **A further scored seed for T9/T10 (F12/OPEN-3)?** `single-design` is exhausted; the deletion
   legs of the lead object are still one seed.
6. ⚠ **`N-I8-21`(b)** — six instances, two of them wave 12. Standing or wave-scoped; where filed;
   whether `N-I8-21` is split. ⛔ Not this spoke's.
7. ⚠ **`w12b` into the evidence fold** (OPEN-9), and the two corrections its artefacts force on
   `w12-advisor-report.md` (C20, C21).

---

## 7. COST

⛔ **Zero scored runs. Zero compute. Zero GPU. No cluster leg. No background job. No PID launched.
No box-wide kill possible or issued. One thread.** One session, reading and writing documents.
Deliverables in `/c/chlu/outputs/w13-draft-revise/`, committed on branch `w13/draft-revise` under
`w13_draft_revise/`, and bundled to `w13-draft-revise.bundle` — **that branch only**, verified,
`list-heads` showing exactly one ref.
