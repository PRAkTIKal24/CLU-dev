
## §I8.18 ◈◈◈◈◈ THE PASS-5 ROWS: w20 `shapes-data-matched` (the Shapes margin is +7, and M1's cause was schedule), M3 measured, the four w21 units, the w19 relabel, and the M1 / M3 carry-forward

Filed 2026-09-24 by `w21 spoke/registry-fold`, `doc-curator`, fold symbol **◈**, pass 5. ⚠ CURATOR-PROPOSED. Registry entries: `N372`–`N390`.

> ⚠ **Paths:** unprefixed paths are relative to `~/clu/evidence/outputs/`. **CA** = `~/clu/outputs/w21-registry-fold/curator/curator_arithmetic.json`
> (script beside it; reads evidence only).
> ⛔⛔ **The Head's 2026-09-24 rule binds this section: no pre-registration, no pre-declared bar or label.** A PREREG bar appears below only as the
> record of what a spoke declared. Where it and a measured number disagree, the number is reported.
> ⛔ `tasks/w17/AMENDMENT-1.md` still binds drafts; no row below authorises a number in one.

### **CM-W20-1** ⭐⭐ THE DATA-MATCHED SHAPES MARGIN: **+7.03, NOT +39…+49. It shrinks about 6× and does not collapse.**

*(Registry: `N372`, `N380`, `N381` item 4. Source: `w20-shapes-data-matched/SHAPES-MATCHED.json` → `headline`, `rows34.*`; `FINDINGS.md` §0–§4, §9.)*

- ✅ **Approved content:** rows 3–4, K\* = 32 on 131,072 rows, against the strongest data-matched rival fitted on the **same** 131,072 rows at the
  read-path pin: **+7.03** (seeds +7.49 / +8.60 / +5.02; JSON mean 7.033639143730887), **+6.54** against the rival at its best over every fit incl.
  30×; every per-seed CI above 0 (lowest lower bound +3.73). **With the P1 convergence table beside it** (w20 FINDINGS l.255–256).
- ⛔⛔ **FORBIDDEN:**
  - *"+39 to +49"* (or any number in that range) as a margin over rivals without *"against comparators trained for 1,280 steps"* and the data-matched
    +7.03 beside it;
  - the mlp seed-0 rival called converged (30× is still +1.47 over 10×; *"+6.02 (best-over-all-fits) is the honest seed-0 number"*);
  - the rival search called exhaustive (every family selected the grid edge, lr = 3e-3 and 40 epochs; grids not widened);
  - `headline.rule`'s *"P1-carried"* quoted as the definition of +7.03: it is max(DECLARED, TENX) at seed 2 (the literal P1-carried mean is
    7.342507645259938, CA).
- ⚠ **`learned_1x` is absent from w20's tables on the read-path matching rule** (`N380`). Its existing w12 value at the shipped leg, 48.8991 /
  49.5413 / 49.4404, sits above w20's K\* = 32 store (47.50 / 47.40 / 49.12) on every seed (unpaired, memory-ledger-matched, not a margin). ⛔ **FORBIDDEN:**
  any sentence that the store beats *every* learned rival on Shapes rows 3–4 without the `learned_1x` column or its stated exclusion rule.
- **Supersedes** the size in `CM-W17-1`'s framing (the ≈ +39…+49 margin); `CM-W17-1` carries a `⟲ PASS 5` line.

### **CM-W20-2** ⭐⭐ M1's MECHANISM, MEASURED: **SCHEDULE, NOT DATA.**

*(Registry: `N373`. Source: `SHAPES-MATCHED.json` → `rows34.arms.{mlp__declared@test16k_stepmatched, …, w12comp_mlp__as_trained}`; `FINDINGS.md` l.24–31.)*

- ✅ **Approved content:** *"The draft's rivals scored 5.5 (MLP) because they were trained for 1,280 steps, untuned. An MLP given the rows-3–4 selected
  schedule (20,480 steps) scores 38.6 on the same 16,384 rows … At matched steps, 8× more data buys the MLP ~0.5 novel tuple per corner. The store is just
  as insensitive to data (K\* = 32: 47.28 at 16,384 rows → 48.01 at 131,072)."* (w20 FINDINGS l.24–29.)
- ⛔ **FORBIDDEN:** *"the rivals lost because they saw 8× less data"*; *"data matching closes the gap"* (the Hub: *"Fixing the data match alone would not have
  fixed the paper"*); extending this from the MLP family to all rivals (A1.1 / A2.3 are MLP-only diagnostics).
- ⭐ **M1 reading:** "matched" must mean data **and** optimiser steps **and** input **and** a width the rival can train at (`CM-W17-8`).

### **CM-W20-3** ⛔⛔ DELETION UNDER THE HEAD'S DEFINITION: **the mechanism is exact; the outcome is not. M3, measured and replicated.**

*(Registry: `N374`, `N375`, `N376`, `N381` items 1, 2, 5, 6. Source: `SHAPES-MATCHED.json` → `deletion_matched`; `FINDINGS.md` §5; Head `AMENDMENT-2.md`.)*

- ✅ **Approved content, two statements and never one:**
  1. **Exact:** the store downdate equals refit-without, and the book refit from the downdated statistic equals the refit-without book, within the declared
     bars on 9/9 legs ((a) max 5.383044481277466e-07 against 1e-5; (b1) 0; (b2) max 7.812795388417726e-08 against 1e-3), and composition is identical
     to the refit-without pipeline on every corner.
  2. **Not harmless:** deleting `red ∧ sphere` costs the surviving corners −6.13 / −6.43 / −6.30 (3/3 seeds); a random-1 % deletion moves −0.02 / −0.11 /
     −0.01. *"Under the Head's definition, deleting a conjunction removes the store's ability to compose one of its values with other values."*
- ⛔⛔ **FORBIDDEN:**
  - *"deletion leaves the other values' composition unchanged"* (the w15 row held only because the book was **carried**);
  - *"c still composes after forgetting c"* in any table or figure (RETIRED by AMENDMENT-2; evidence stays in artefacts);
  - *"exact to 1e-8"* for (a) or (b2);
  - *"data-free deletion"* or *"reads 0 rows"* as a size or privacy claim for the address book: its exact statistic is 967304800 bytes, the fit data
    itself, and the refit costs 2,367–5,913 s;
  - *"the rivals lose < 1 where the store loses 6"* per seed (true for the family means of mlp / lstm / transformer only; cont_mlp s2 −9.568807339449542);
  - the −32.1 breakdown quoted beyond seed 0.
- ⭐ **The store's real advantage, stated as cost:** the downdate reads **0** pool rows in seconds; rival retrain-without reads 120,028–129,761 rows in
  548–2,065 s. ⚠ `red ∧ sphere` is not significant against mlp s1 / s2 or lstm s2. w15's 16k-rival deletion columns: **appendix only**.

### **CM-W20-4** ⚠ THE LADDER, LACE / PoE AT 131k, AND WHAT IS NOT CLAIMABLE

*(Registry: `N377`, `N378`, `N379`, `N380`, `N381` item 3. Source: `SHAPES-MATCHED.json` → `ladder.<n>`, `LACE_PoE_at_131072`; `FINDINGS.md` §6–§8.)*

- ✅ **Approved:** ladder margins against rivals fitted on each rung's own 131,072 rows: **+5.13 (n = 2), +11.18, +12.55, +10.69, +10.30 (n = 3–6)**,
  every per-seed CI above 0 (lowest +2.46), rebuilt K\* = 32 stores reproducing w12b on 15/15 legs. Rungs n = 3–6 probe the identical 120 corners.
- ⛔ **NOT CLAIMABLE:** the EXISTING ladder K = 5 arm at **n = 4 and n = 5** (`random_atoms` 1.2416666666666667 and 2.033333333333333 at seed 2, over
  the 1.0 bar; `landscape_removed` and across-wells never run on that arm). ⇒ Drop, re-run or print as not-claimable: the Advisor's (w20 report §7 item 5).
- ✅ **LACE / PoE @131,072, printed beside w15's 32,256 rows, with the scope verbatim:** *"LACE/PoE as instantiated in this feature space at their own
  declared budgets — not a refutation of LACE in general."* LACE 0.018 / 0.046 / 0.073, PoE 0.000. ⛔ FORBIDDEN: comparing them to the VAL store's 64.92
  (`N381` item 3). ⛔ LACE/PoE @131k on the rungs is a DECLARED NOT-RUN, not a null.
- ⭐ **Process:** w20's five self-caught defects, including VOIDED seed-1/2 rung stores (per-seed dead masks), are reported as voids (`N379`).

### **CM-W21-1** ⛔ WORDS-FULL ON TEST: **LOSS** (w21 runs 4–6/9)

*(Registry: `N382`. Source: `w21-words-full-test/test/BARS-WORDS-FULL-test.json` → `HEAD_LABEL`; `FINDINGS.md`.)*

- ✅ **Approved:** *"WORDS-FULL on TEST: LOSS"*, Δ̄ −0.017517974813034298, σ 0.014018294086938362, 0/3 seeds ahead; *"the shared pair block carries the
  composition, not the store"*; store gate holds; VAL instrument reproduced bit-for-bit. ⚠ **Printed beside it (formerly "never the label"):** under the
  CLU's own per-seed sd (0.1373909749670931) or R\*'s (0.15140703330856564) it would read COMPARABLE.
- ⛔ **FORBIDDEN:** *"the store composes where rivals cannot"* on CMAZE-T WORDS-FULL; the LOSS reported without the σ-sensitivity line, or the COMPARABLE
  reading reported without the LOSS.

### **CM-W21-2** ⚠⚠ CLU-REL-A10: **A WIN BY THE HEAD'S RULE, MOSTLY CARRIED BY THE CODING**

*(Registry: `N383`, `N386`. Source: `w21-maze-t-clu-improve/test/LABEL-BOTH-test.json`, `val_full/LABEL-{BOTH,WORDS-FULL}-val.json`; `FINDINGS.md`.)*

- ✅ **Approved, three statements that travel together:**
  1. *"The CLU, with a relational request design, beats every matched rival on the frozen input on every seed"*: BOTH TEST 0.9285039268690677 against
     0.6851358576873765, 3/3 (runs 7–9/9).
  2. The same relational columns given to the rivals lift the best rival to 0.8621086355989208; the CLU keeps **+0.066** and is ahead on **2 of 3** seeds.
  3. On WORDS-FULL VAL a matched transformer with the relational columns reads 0.9414055233621977 against the CLU's 0.9371139657869874: **the store's
     contribution there is zero.** *"Mechanism, named: colour-relational request coding (an input representation) … It is not a store effect."*
- ⛔⛔ **FORBIDDEN:** the 0.929 / 0.685 WIN without statement 2 beside it; *"the store"* credited with the margin; *"CLU-REL-A10 generalises to new
  colours"* (`value` is LOSS, 0.500 against 0.792); the store gate called clean (seed 0 across-wells 0.778 > 0.758; post-hoc controls climb with atoms to
  0.615 / 0.937; ⛔ **seed 4 VAL `random_atoms` = CLU = 0.8994252873563219 and `landscape_removed` 0.2413793103448276**, `N386`); *"69 candidates"* (68,
  57 gate-passers incl. one inadmissible).

### **CM-W21-3** ⛔⛔ THE A1 BOTH ROW AT SIX SEEDS: **heterogeneous, estimand-dependent, and bounded by a measured rival defect**

*(Registry: `N384`, `N385`, `N386`, `N387`, `N388`. Source: `w21-maze-t-seed4/MAZE-T-SEED4.json`, `FINDINGS.md`, `TABLES.md`, `test/BOTH_test_seed3.json`; CA `seed4_estimands`.)*

- ✅ **Approved content:**
  - *"6 seeds tried, 6 kept, 0 discarded"*; easy / hard by **measured ceiling** (cut 0.6407297594090047), with the sensitivity (matched-only ceiling ⇒
    hard = {1, 3, 4}) printed;
  - seed 5 (easiest; ceiling 0.939): CLU 0.6818181818181818 against the matched mlp-80k's 0.8666666666666667, traced to 0.000 on two class-A tuples that
    eight arms (**one of them a control**) solve;
  - rollout validity 0.000 on all six seeds.
- ⛔⛔ **Estimands — print both, never one** (`N385`): per-seed-best **+0.046** at six seeds (+0.065 at three); R\*-by-mean **+0.128** at six seeds, ahead
  6/6 (+0.0988 at three). ⛔ FORBIDDEN: *"the margin fell from +0.099 to +0.046"* (mixed estimands); *"the win replicates at six seeds"* without the seed-5
  miss; *"the win does not replicate"* without the R\*-by-mean line.
- ⛔ **FORBIDDEN:** seed 3's +0.262 without *"against a non-converged matched MLP"*; a hard seed described as *"inherently hard data"* (CLU-REL-A10 reads
  0.914 / 0.899 there; `N388`); a matched-float rival on the 954-dim input presented as a trained baseline without its loss (`N387`).

### **CM-W19-5** ⭐⭐ THE w19 RELABEL UNDER THE HEAD'S RULE: **BOTH TEST → WIN, OBS → COMPARABLE**

*(Registry: `N389`. Source: `w19-advisor-report.md` §0a; `w19-maze-t-symmetric` BARS JSONs; CA `w19_relabel_check`.)*

- ✅ **Approved:** the table in `N389`, with the Hub's caveat verbatim: *"Against the per-seed best arm, seed 1 flips — the 8k lstm reads 0.5674 against the
  CLU's 0.5660 … Print both readings; do not let 'ahead 3/3' stand unqualified."*
- ⛔ **FORBIDDEN:** the WIN printed as a store-over-rivals margin on CMAZE-T without WORDS-FULL's LOSS beside it (`CM-W21-1`): the BOTH rivals read
  observation ⊕ design at 4 hidden units (`N387`); OBS's COMPARABLE quoted as parity (the failing link is READ, and its spread comes from seed 0).
- **Supersedes** the verdict column of `CM-W19-3` for the paper; that row carries a `⟲ PASS 5` line. ⛔ `CM-W19-3`'s forbidden-wordings list stands.

### **CM-W17-8** ⚠ OPEN-ITEMS LEDGER, PASS 5 — **M1 and M3 are now MEASURED; their wording is still the Head's and the Advisor's**

*(Registry: `N339` and `N340` with their `⟲ PASS 5` lines; `N372`–`N377`, `N387`, `N390`. Supersedes nothing: `CM-W17-5`…`CM-W17-7` stay the earlier records.)*

| item | status at pass 5 | what pass 5 adds |
|---|---|---|
| **M1** | ⭐ **MEASURED** on Shapes; ⛔ wording **HELD** for the Head + Advisor | Data-matched margin +7.03 (`N372`). ⭐⭐ The operative asymmetry was **optimiser steps**, not data (`N373`). A fourth axis is measured on CMAZE-T: the float match forces a 4-unit rival that cannot train (`N387`). ⇒ "matched" = data, steps, input **and** a trainable width |
| **M3** | ⭐ **MEASURED and REPLICATED**; ⛔ wording **HELD** | Under the Head's refit definition the mechanism is exact and the outcome is not: `red ∧ sphere` −6.13 / −6.43 / −6.30 on survivors; random-1 % control ≈ 0 (`N374`). The book's exact statistic is the fit data (`N375`) |
| **process** | ⭐ the PREREG ban applied | Everything withheld for want of pre-registration is folded with its status (`N380`, `N390`) |

- ⛔ The curator does not rule. ⛔ `CM-W17-1` and `CM-W17-2` stay in force with their `⟲ PASS 5` lines until the ruling.

## §I8.19 ⚠ OPEN FLAGS FOR THE HUB: the ◈ pass 5

These are raised and deliberately not resolved by the curator.

1. ⛔⛔ **The abstract's Shapes headline.** +39…+49 is not a data-matched number; the w20 record says ≈ +7 with the P1 table beside it (`CM-W20-1`). The live
   draft was **not** read or edited at this pass (no lock taken). ⇒ Owner: the Head / the draft spoke.
2. ⛔⛔ **M1's paper sentence.** Say plainly that the published comparator was trained for 1,280 steps (`N373`), or report only the corrected number?
   The w20 Hub report asks the Advisor exactly this (§7 item 2).
3. ⛔⛔ **M3's paper sentence.** Exactness (a, b) holds; outcome (d) fails for a conjunction (`N374`); the book's statistic is the fit data (`N375`).
4. ⚠ **`learned_1x`.** Absent from w20's tables; its existing value is above the K\* store on every seed (`N380`). Appendix with the rule stated, or
   absent? The ~3.2M-param number is not on batmac (`NEEDS-ARTEFACTS.json`).
5. ⚠ **The existing ladder K = 5 rows at n = 4 / 5** fail P2 at seed 2 (`N377`); this touches the w12 / w13 ladder rows. Drop, re-run, or print as
   not-claimable?
6. ⚠ **Which estimand for the six-seed BOTH row** (`N385`), and whether the per-seed-best caveat of `CM-W19-5` becomes the paper's rule.
7. ⚠ **The matched-float rule on wide inputs** (`N387`): a rule change (e.g. a width floor, or matching on something other than read-path floats) is a
   programme decision.
8. ⚠ **CLU-REL-A10's store gate** on seed 4 VAL (`N386`): the atoms-6 variant had a cleaner gate on VAL (`N383`). Re-run or print as is?
9. ⚠ **Prose-vs-artefact fixes owed:** w20 FINDINGS l.156 and l.183–184; the w20 Hub report §5 item 1 (64.92); `headline.rule`'s label; the
   w18 Hub report's MF-2 reason; w19 FINDINGS l.36–38 (still uncorrected); the clu-improve PREREG / ledger "69" (correctly left, corrected in FINDINGS).
10. ⚠ **Sync:** the w20 per-corner request map (for the −32.1 breakdown at seeds 1–2) and every persisted w20 / w21 object are on the ThinkPad only.
11. ⚠ **v2.26 is PROPOSED.** ⛔ This pass does not confirm v2.25 or below. The tiers of `N372`–`N390` are curator-proposed.
