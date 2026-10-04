
## §I8.20 ◈◈◈◈◈◈ THE PASS-6 ROWS: CLU-REL-A10 on six seeds (an input representation, not a store), MF-2 answered, single-fit RNG variance, and the CMAZE-T figure defect

Filed 2026-09-25 by `w22 spoke/registry-fold` pass 6, `doc-curator`, fold symbol **◈**, pass 6. ⚠ CURATOR-PROPOSED. Registry entries: `N391`–`N397`.

> ⚠ **Paths:** unprefixed paths are relative to `~/clu/evidence/outputs/`. **CA6** = `~/clu/outputs/w22-registry-fold6/curator/curator_arithmetic_p6.json`
> (script beside it; reads evidence only). figures3 JSONs: branch `w22/figures3` @ `cb5e0ac`, `w22_figures3/`.
> ⛔⛔ **The Head's 2026-09-24 rule binds this section: no pre-registration, no pre-declared bar or label.** A pre-declared bar appears below only as the
> record of what a spoke declared.
> ⛔ `tasks/w17/AMENDMENT-1.md` still binds drafts; no row below authorises a number in one.

### **CM-W21-4** ⭐⭐ / ⛔⛔ CLU-REL-A10 ON SIX SEEDS: **0.925 against A1's 0.677, at the ceiling on every seed. The margin is the INPUT CODING, not the store.**

*(Registry: `N391`, `N392`, `N396` items 3–4. Source: `w21-rel-newseeds/REL-NEWSEEDS.json` → `nway_summary`; `TABLES.md` §0, §2–§4; `replay/*_PERTUPLE.json`; CA6.)*

- ✅ **Approved content, three statements that travel together:**
  1. *"With colour-relational request coding (an input representation carried through the closed-form head), the CLU reads 0.925 on CMAZE-T BOTH TEST
     over six seeds (A1: 0.677), reaching the trained-arm ceiling on every seed. Its margin over the best matched frozen-input rival is +0.202 on average,
     and on seeds 3–5 that rival is converged (no WORSE@80k; the transformer is unflagged)."*
  2. *"Rivals handed the same relational columns come within +0.031 of the CLU on average (−0.007 … +0.165)."* The residual sits mostly on seed 2 (+0.165).
  3. *"On seed 4 a random-atom store matches the CLU exactly (0.9004, all 16 held-out tuples); on seed 5 a matched rival ties it (0.939)."*
- ⛔⛔ **FORBIDDEN:**
  - the 0.925 or the +0.202 margin credited to **the store** or to memory;
  - *"the store composes where rivals cannot"* on CMAZE-T;
  - statement 1 without statement 2 beside it;
  - *"fixes A1's weak seeds"* naming seeds 1 and 5 without seed 4 (A1's lowest seed, where REL gains most **and** ties `random_atoms`);
  - *"+rel erases the margin on every seed"* (seed 2 keeps +0.165);
  - *"CLU-REL-A10 generalises to new colours"* (value row: the CLU is below the best matched rival on **all six** seeds);
  - rollout claims (validity 0.267 / 0.156 / 0.000 on seeds 3–5);
  - the store gate called clean on seed 4 (TEST and VAL).
- **Supersedes** the three-seed scope of `CM-W21-2` for the six-seed question. `CM-W21-2`'s forbidden list stands and carries a `⟲ PASS 6` line.

### **CM-W21-5** ⭐⭐ MF-2, MEASURED: **real on paper, immaterial measured. Pool barely matters at matched steps; steps do.**

*(Registry: `N393`. Source: `w21-ladder-rivals-32k/LADDER-RIVALS-32K.json` → `rungs.<n>.margins_K32_store_minus_rival.*`; `TABLES.md` L3 / F / P1; `checks/pool_identity.json`; CA6 `ladder_margins`.)*

- ✅ **Approved:** ladder margins with **all three rival sets and both budgets named** (pool AND steps in every column header; `CAPTION.md` is the drop-in).
  Mean over rungs: **A +17.53** (32,256 rows, 5,040 steps), **B +9.97** (131,072 rows), **C +10.28** (the store's own 32,768 content rows), **C-sm
  +10.75**. Every CI95 lower bound in B / C / C-sm is above 0 (lowest +1.98). Pool identity is proven on 15/15 legs. *"At matched steps the pool size
  barely moves these rivals; the step budget is what moves them."*
- ⛔ **FORBIDDEN:**
  - *"MF-2 inflated the ladder margin"* (C lands within −1.21 … +1.45 of B per rung, mixed sign);
  - *"MF-2 CLOSED"* citing w20's PREREG A2.1 (`N381` item 7): the measurement is `N393`;
  - the ladder margin quoted without the pool **and** the step budget;
  - set A's +17.53 as a data-matched number (set A holds the value-holdout's held-out values);
  - cont_mlp called converged (up to +19.33 still gaining at 204,800);
  - the rival search called exhaustive (60/60 selections at the grid edge).
- ⚠ **Carries `CM-W21-6`:** the C − B differences (±1.5) and family differences under ≈ 4 are inside single-fit RNG variance.
- **Supersedes nothing.** `CM-W20-4` stands and carries a `⟲ PASS 6` line.

### **CM-W21-6** ⚠⚠⚠ SINGLE-FIT RNG-DRAW VARIANCE: **up to 4.13 novel/corner on the PRNG draw alone. It bears on every single-fit margin in the paper.**

*(Registry: `N394`, `N396` items 1–2. Source: `LADDER-RIVALS-32K.json` → `rungs.<n>.crosscheck.{X1_same_rows_same_steps, X1_diagnostic_ref_recipe_replay}`; `TABLES.md` X1; `LOG.md`; CA6 `ladder_X1`.)*

- ✅ **Approved content:** *"Two mlp fits identical in rows (32,768), steps (2,560), width, parameters, trainer and schedule, differing only in PRNG draw,
  differ per seed by −2.15 to +4.13 novel tuples per corner; the per-corner CI95 excludes 0 on 12 of 15 legs. A replay with the reference's own keys
  reproduces it identically per corner on 15/15 legs."* Always with its scope: mlp, 2,560 steps (an under-trained regime), Shapes ladder.
- ⛔⛔ **FORBIDDEN:**
  - *"10/15"* or *"±4.1"* (the artefact: **12/15**, **−2.15 … +4.13**);
  - a **per-seed** margin, or a between-family / between-set / between-pool difference, **under ≈ 4** presented as real on the strength of its per-corner
    CI alone. This includes Shapes seed 2's +5.02, which clears 4.13 by only 0.8850152905198776 (CA6);
  - any sentence that a per-corner bootstrap CI covers refit variance;
  - X1's *"REPRODUCES"* quoted as a verdict (a pre-declared bar, LOG 13:58Z; the ban).
- ⭐ **What stays approvable:** the **sign** of the 3-seed mean margins (Shapes +7.03; ladder +5.13 … +14.54 in B / C / C-sm), all per-seed margins
  positive. The recommended companion wording is *"single-fit margins; seed-level refit variance up to ≈ 4 novel/corner (w21 X1)"*.
- ⚠ **Not measured:** refit variance at the carried budgets (20,480 / 204,800 steps) and on CMAZE-T. Indirect evidence: mlp step-matched − TENX
  −4.76 … +2.02. ⇒ `§I8.21` flag 1.

### **CM-W22-1** ⚠ THE CMAZE-T FIGURE DEFECT: **the panels drew the state, the judge scores the fact vector. Measured, fixed to 0, no number affected.**

*(Registry: `N395`, `N396` items 5–8. Source: `w22/figures3` → `w22_figures3/AMBIGUITY.json`, `GLYPH-IDENTITY.json`, `PALETTE-MAZE_RGB*.txt`; `w22-figures3-FINDINGS.md`; CA6 `figures3`.)*

- ✅ **Approved (captions):** *"The tick is the judge's verdict on all four facts (legality, move, key event, pocket-next); the picture shows only the
  resulting state; the strip below prints the facts."* Before the strip, 33,412 of 92,628 opposite-mark pairs (A1, seeds 0–2) were pictorially identical.
  After it, 0 are, in every scope, CLU-REL-A10 included (154,671 of 373,583 → 0).
- ⛔ **FORBIDDEN:**
  - *"only legality was invisible"* or *"3,740 ambiguous panels"* as the full count (3,740 is the legal-bit-only class; 16,283 same-picture pairs have
    pocket-next as the reason);
  - the Hub note's *"88,888 visibly different … figure is fine"*;
  - *"a number changed"*: none did (132/132 REL arm scores float-`==` the run JSONs);
  - the pre-fix palette described as validated (blue↔purple dE 13.8 < 15; CVD 7.2).
- ⭐ **Consistency:** task-figure glyph = panel glyph (7/7 discs pixel-identical); c1 red everywhere.

### **CM-W17-9** ⚠ OPEN-ITEMS LEDGER, PASS 6 — **MF-2 is MEASURED; single-fit variance is a NEW standing caveat; CMAZE-T's positive is the coding**

*(Registry: `N391`–`N397`. Supersedes nothing: `CM-W17-5`…`CM-W17-8` stay the earlier records.)*

| item | status at pass 6 | what pass 6 adds |
|---|---|---|
| **MF-2** | ⭐ **MEASURED: immaterial** | Rivals on the store's own 32,768 rows give the same ladder margins as the 131k set (`N393`) |
| **M1** | ⭐ MEASURED (pass 5); ⛔ wording **HELD** | Independently confirmed on the ladder: steps, not pool, move the rivals (`N393`). ⚠ New companion: single-fit RNG variance up to 4.13 (`N394`) |
| **M3** | unchanged (pass 5) | — |
| **CMAZE-T positive** | ⭐ six-seed **input-representation** result; ⛔ **not** a store result | `N391`, `N392` |
| **process** | ⭐ the PREREG ban applied | One pre-declared bar found (X1) and quoted as record only (`N394`, `N397`) |

- ⛔ The curator does not rule. `CM-W17-1`, `CM-W17-2` and `CM-W17-8` stay in force until the ruling.

## §I8.21 ⚠ OPEN FLAGS FOR THE HUB: the ◈ pass 6

These are raised and deliberately not resolved by the curator.

1. ⛔⛔ **Refit variance at the carried budgets** (`N394`). X1 exists only at 2,560 steps. k PRNG replicates of the carrying rival per seed, at the headline
   budget (Shapes rows 3–4) and on CMAZE-T, would put a refit interval beside every per-corner CI. Run it, or print the X1 caveat beside every single-fit
   margin? ⇒ Owner: the Head.
2. ⛔⛔ **CMAZE-T's paper sentence** (`N392`). The positive is colour-relational request coding. Is it an input-representation result in the main text, with
   the `+rel` table beside it, or is CMAZE-T moved out of the store claims? Seed 2's +0.165 residual is the only place the CLU keeps a margin beyond the
   coding.
3. ⚠ **The seed-4 store gate** fails on TEST as on VAL (`N392`; `§I8.19` flag 8). Print as is, or re-run with a different atom count?
4. ⚠ **The 32,768-row ladder figure slot** is still `not_landed` (`N395`). `ladder-rivals-32k`'s `CAPTION.md` and `caption.json` are the drop-in.
   ⇒ Owner: the figures spoke.
5. ⚠ **Prose-vs-artefact fixes owed** (`N396`): the brief's "10/15" and "±4.1" (carried into any Hub or advisor report?); figures3 FINDINGS' palette
   digits (13.7 / 7.1) and its "defect file does not exist"; the Hub defect note's "88,888 … figure is fine"; the ladder and figures3 LOG timestamps.
6. ⚠ **Draft state reported by figures3, not verified by the curator:** the live draft no longer places `fig_headline_steps` or the three
   `rendered/rc_w19_t7_*` figures, and `\PLACEHOLDER` reads 4 (pass 2 reported 15). No draft was read at this pass.
7. ⚠ **v2.27 is PROPOSED.** ⛔ This pass does not confirm v2.26 or below. The tiers of `N391`–`N397` are curator-proposed.
