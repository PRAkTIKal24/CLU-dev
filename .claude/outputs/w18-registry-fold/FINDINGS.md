# w18 spoke/registry-fold — FINDINGS (FOURTH PASS; ⛔ FIFTH PASS REQUIRED for w20 `shapes-data-matched` only)

**Agent:** `doc-curator`. **Branch:** `w18/registry-fold`. **Machine:** batmac.
**Commissioned by:** the w18 registry-fold fourth-pass launch (2026-09-23; pre-authorised by the Head).
**Scored runs: NONE. Compute: NONE. Background jobs: NONE. Sub-agents: NONE.** tmux was not touched and no process was killed.
⛔ `~/clu/evidence/` was not written, and no draft was edited. The base md5s are unchanged (registry `5568a6f2…`, matrix `6b8042dd…`).

## 0. WHAT SHIPPED IN PASS 4

| file | pass 3 → pass 4 | what pass 4 adds |
|---|---|---|
| `negative_results.md` | 7,167 → **7,466** lines | **`N366`–`N371`**; `◈.12`–`◈.14`; a pass-4 not-runs table; a header note; and inserted `⟲ PASS 4` lines at `N339` (M1), `N348`, `N349`, `N363` and `◈.11` |
| `claims_matrix.md` | 2,019 → **2,131** lines, **v2.25 (PROPOSED)** | **`§I8.16`**: `CM-W19-3`, `CM-W19-4`, `CM-W17-7`. **`§I8.17`**: 9 open flags. Also `⟲ PASS 4` lines at `CM-W17-1`, `CM-W16-2`, `CM-W17-5`, the `§I8.14` note, `CM-W19-1`, and `§I8.15` flags 1 and 4 |

**Additivity was checked with difflib against the pass-3 files:**
- Registry: 299 lines inserted, **0 replaced, 0 deleted**. There are now **363** `### N` headings with **zero duplicate numbers**, and each of N366–N371 appears exactly once.
- Matrix: 112 inserted. Line 1 (the title) is the only replaced line, and the prior title is kept verbatim as its tail (`endswith` True). The 53 `CM-` row headings have no duplicates.

**Number audit.** Every decimal and every token of four or more digits in the new text was checked mechanically against the sources read at this pass: 123 tokens in the registry block and 41 in the matrix block. All were found verbatim, except draft line numbers observed by grep and sha256 prefixes.

## 1. THE RESULT: w19 `maze-t-symmetric` CONTRADICTS A WAVE-16 HEADLINE

**⛔⛔ All four conditions are LOSS.** Budget: **3 of 9** scored runs spent (BOTH, TEST); **6 DECLARED NOT-RUN**.

| condition | CLU | R\* | margin | verdict | entry |
|---|---|---|---|---|---|
| WORDS-MAIN (VAL) | 0.3343737955355819 | 0.7473459065764733 (transformer, 80k) | −0.4129721110408914 | LOSS (VB1, VB2) | `N366`, `N367` |
| **WORDS-FULL (VAL)** | 0.7247150720644506 | 0.7419822433104506 (transformer, **8k**) | −0.017267171245999968 | LOSS (VB1). ⭐⭐ *"the pair block, not the store, carries the w16 margin"* | **`N367`** |
| OBS (VAL) | 0.3513038895074918 | 0.45663721986855893 (transformer, raw obs, 80k) | −0.10533333036106712 | LOSS (VB1, VB4); link **READ** | `N368` |
| BOTH (VAL → TEST) | 0.7247150720644506 → **0.724626907621842** | 0.6168149530251911 → 0.6257902235030398 (lstm, 80k) | +0.10790011903925956 → **+0.0988366841188022** | VAL PASS (knife-edge on seed 1) → **TEST LOSS** (SB1; behind the 8k LSTM on seed 1) | `N369` |
| *OBS-X10 (diagnostic)* | 0.44708633392890107 | 0.46080381873040394 | −0.013717484801502866 | *would read LOSS*; never a bar | `N368` |

**⛔⛔ The consequence, recorded as the launch asked:**
- **What was asymmetric.** w16's CMAZE-T Config-A1 "STORE row" (CLU 0.7246 against 0.4035) gave the CLU the five words through a 318-wide `[bias|main|pair]` design. Every rival read the observation instead.
- **Only the rivals changed.** The CLU is the same object: BOTH TEST reproduces w16's `BARS-A1-test.json` CLU mean, **0.724626907621842, to every digit**, and its controls too.
- **The margin does not survive symmetric inputs.** It had survived 80k-step rivals, but that check tested steps, not input.
- **Where it is annotated.** ⟲ lines at `N348` and `N349` (the w16 entries that relied on the A1 row), and at `CM-W16-2` and `CM-W17-5` in the matrix. ⛔ The original entries are **not edited**.
- **C-512**, w16's replication row, was not re-run. It is **exposed** to the same asymmetry and may not be promoted in A1's place.

**What survives. Two statements, never one:**
1. The **store gate**: `landscape_removed` is 0.000 on every condition, and the controls collapse on WORDS-FULL, OBS and BOTH. ⚠ **They do NOT collapse on WORDS-MAIN** (VB2 fails).
2. **Store-scoped deletion exactness** on BOTH TEST.

Neither is a margin over rivals.

**⛔⛔ M1, a THIRD time** (`⟲ PASS 4` at `N339`; `CM-W17-7`). Each time, the rival was given less than the store:
- the **data pool** in Table 1 (Shapes);
- the **training steps** on w16 CMAZE-O;
- the **input** on w16 CMAZE-T.

⚠ The launch's "now on the maze rather than on Shapes" is imprecise, because instance 2 was already on a maze. What is new is the **axis**, and that it reverses the **headline**. **M1 and M3 remain OPEN and HELD. The curator does not rule.**

**⭐ Positive** (`N370`, `CM-W19-4`): w19 persisted **384 objects**, each hashed and manifested with 0 mismatches. This closes `N363`'s defect class for w19, and the A1 CLU side is now recoverable. ⚠ Three limits:
- **Only the 18 manifests are on batmac.** The objects are on the ThinkPad.
- **w16's rivals are not recoverable**, because the BOTH rivals read a different input.
- **The `persist_dir` pointers are stale** (pre-move paths).

## 2. ⚠ WHERE THE LAUNCH (OR w19's PROSE) DISAGREED WITH THE PRIMARY RECORD — `N371`

1. *"BOTH: val PASS knife-edge (margin +0.099 < the 0.10 bar)"*: the **VAL** margin is +0.10790011903925956 (a PASS). +0.0988… is **TEST**. The VAL knife-edge is the seed-1 lead (0.5688888888888889 against 0.5674074074074074).
2. w19 FINDINGS l.36–38, *"On every condition the four charter-C controls collapse"*, is contradicted by w19's own `BARS-WORDS-MAIN-val.json` (`B2_store_gate: false`: 0.18744092621014888 against a bar of 0.1843737955355819).
3. *"The extractor recovers the whole word"*: FINDINGS l.23 says **only 6–22 %** of held-out n-way tuples. The launch's wording drops that qualifier and inverts the finding.
4. *"786 objects"*: it is **384 objects**. 786 is a file count (384 objects + 384 `.sha256` + 18 manifests).
5. *"634 / 29 dyn"*: the measured values are **636 / 31** (w19 FINDINGS l.72–73).
6. The `persist_dir` fields name pre-move paths.

## 3. ⛔ WHAT DID NOT RUN — FIFTH PASS REQUIRED

- **w20 `shapes-data-matched` is still running** (ledger 1/3, per the launch). batmac has only its task file: no output directory and no ledger. It is **NOT FOLDED, and nothing is guessed** (registry `◈.14`). ⭐ It is the data-matched Shapes test that M1 asks for.
- **Not done, by design:**
  - Verifying or rebuilding any persisted object: they are not on batmac.
  - Reading `logs/`: it is empty on batmac.
  - Auditing `TABLES.md` or `MAZE-T-SYMMETRIC.json` cell by cell: they were hashed only, and every number here comes from the BARS JSONs or from FINDINGS.
  - Any draft or evidence edit. ⚠ The live `DRAFT-v2.tex` (`e49a6910…`) still carries the w16 A1 win at L121–122, L1082–1083, L2163 and L2618 (`§I8.17` item 1).
  - Any ruling on M1, M3, the `N363` re-run, the C-512 re-run, or the w5 count restart.

## 4. OPEN FLAGS — `claims_matrix.md` §I8.17 (9 items). Tiers and v2.25 are CURATOR-PROPOSED.

---

# PASS 3 (kept verbatim below, for the record)

# w18 spoke/registry-fold — FINDINGS (THIRD PASS; ⛔ FOURTH PASS REQUIRED for w19 `maze-t-symmetric` only)

**Agent:** `doc-curator`. **Branch:** `w18/registry-fold`. **Machine:** batmac.
**Commissioned by:** the w18 registry-fold third-pass launch (2026-09-23).
**Scored runs: NONE. Compute: NONE. Background jobs: NONE. Sub-agents: NONE.** tmux was not touched and no process was killed.
There was one scratchpad `bibtex` invocation, to check a claim (`N365`); it wrote nothing under `~/clu/`.
⛔ `~/clu/evidence/` was not written. The base md5s are unchanged (registry `5568a6f2…`, matrix `6b8042dd…`).

## 0. WHAT SHIPPED IN PASS 3

| file | pass 2 → pass 3 | what pass 3 adds |
|---|---|---|
| `negative_results.md` | 6,819 → **7,167** lines | **`N358`–`N365`**, `◈.8`–`◈.11`, a header note, and inserted `⟲ PASS 3` lines at `N-I8-21`, `N321`, `N327`, `N339`, `N340` and `◈.7` |
| `claims_matrix.md` | 1,899 → **2,019** lines, **v2.24 (PROPOSED)** | **`§I8.14`**: `CM-W15-2`, `CM-W15-3`, `CM-W15-4`, `CM-W19-1`, `CM-W19-2`, `CM-W17-6`. **`§I8.15`**: 10 open flags. Also `⟲` lines at the `§I8.12` note and at `§I8.13` flag 1 |

**Additivity was checked with difflib against the pass-2 files:**
- Registry: 348 lines inserted, **0 replaced, 0 deleted**.
- Matrix: 120 inserted. Line 1 (the title) is the only replaced line, and the prior title is kept verbatim as its tail (`endswith` True).
- The registry now has **357** `### N` headings with **zero duplicate numbers**, and each of N358–N365 appears exactly once. The 50 `CM-` row headings have no duplicates.

**Number audit.** Every decimal and every token of four or more digits in the new text was checked mechanically against the sources read at this pass. That was 127 tokens in the registry block and 40 in the matrix block. All were found verbatim, except the log line numbers 2398/2399 (observed directly) and the section id `§I8.14`.

## 1. THE LAUNCH ITEMS

| # | item | entry | result (the primary record's numbers) |
|---|---|---|---|
| 1 | **bar-d-score** | `N358`, `CM-W15-2` | BAR D-GPU **MIXED on row 4 and on row 3** ⇒ **no claim either way** (`BAR-D.json` VERDICT). **0 VOID** seeds. Charter C collapses on every scored leg (`charter_C_collapse_all_seeds: true`), but the margin is ~0, so there is **no store claim in either direction**. `loop_swap` saturates: descriptive, not a bar. GPU and CPU are two blocks, never merged. ⭐ **Ordering NOT REPRODUCED** on GPU (A′ < 0 in 2/6) |
| — | *(curator-added)* | `N359` | ⛔⛔ The w8 scorer **never loaded the refit baseline**; it loaded spoke A's frozen-feature rival. ⇒ w8's CPU `rival_mlp_180k` = 0.000 rows are not the refit baseline. **CPU BAR D is UNMEASURED.** `⟲` at `N-I8-21` |
| 2 | **lace-poe-rows** | `N360`, `CM-W15-3` | Store 47.495 / 47.404 / 49.119 against LACE 0.165 / 0.073 / 0.229 and PoE 0.009 / 0.000 / 0.000. 109/109 corners against each, every seed, every rung. ⛔⛔ **The spoke's scoping is recorded verbatim as a standing caveat:** LACE/PoE *"as re-implemented at these budgets"*, *"not as 'the CLU out-composes LACE in general'"* |
| 2b | generation-table run 3/3 | `N361`, `CM-W15-3` | **The store claim STANDS** (S1 PASS). **"Beats the shipped read" does NOT stand**: S2 FAILS on seed 0 (−0.0185), and S3 loses to `knn_proto`. `generation-table` is exhausted |
| 3 | **coverage-control** | `N362`, `CM-W15-4` | For n ≥ 3 the rise is coverage **by construction**, with coverage share 1.00 on every seed. The test cell (61.614) is **DERIVED, not measured**. The val verdict is **"NO RISE TO DECOMPOSE ON VAL"**. `composition` is EXHAUSTED: 0 runs spent and a NOTE row written. `⟲` at `N327` |
| 4a | ⭐⭐ CMAZE-T panels not buildable | `N363`, `CM-W19-1` | `w16_cmaze_t.py::full` trains everything in-process and writes nothing, so only aggregates survive. **A pipeline that cannot re-render its own figures did not persist enough.** Unblocking needs a TEST-leg persistence re-run, which is a Hub / ledger decision |
| 4b | ⭐ rival succeeds / CLU fails | `N364`, `CM-W19-1` | **30 / 24 / 30 of 5,688** at seeds 0 / 1 / 2. This is printed in Fig. 4's caption |
| 4c | ⭐⭐ bibtex | `N365`, `CM-W19-2` | ⚠⚠ **DISPUTED — see §2.** The process rule is recorded; the claim as briefed is **not** |
| 5 | M1, M3 | `⟲` at `N339`/`N340`; `CM-W17-6` | **OPEN, HELD for the Head and the Advisor** |

## 2. ⚠⚠ ONE LAUNCH ITEM DID NOT SURVIVE THE PRIMARY RECORD

The launch (and `w19-draft-figure-slots/FIGURE-SLOTS.md` l.46) says *"bibtex had NEVER succeeded: every w17 compile carried 226 undefined citations while being reported as a clean compile."*
The w17 log on disk, `~/clu/outputs/w17-paper/compile-v2.log`, says otherwise:
- **bibtex ran and exited 0** (l.969–974), and its `.blg` reads *"You've used 62 entries"*.
- Its final pass reports *"(no undefined citations/references, no errors)"* (l.2399).
- The **226** is `grep -c undefined` over all four passes: **164** hits before bibtex, **61** in pass 2, **0** in passes 3–4, and **1** that is the summary line itself. The style log has **228**.
- A scratchpad reproduction with no `BSTINPUTS` found the `.bst` and resolved **63** entries.

**What is supported:** the w19 **baseline** compile exited 0 with **61** undefined citations in its log, and w19's final compiles have 0. So the rule *"an exit code is not a resolved bibliography"* stands, and it is recorded as binding (`CM-W19-2`), alongside a second rule: *a grep count over a multi-pass log is not a count of unresolved citations*.
The specific claim is recorded as **DISPUTED** and is forbidden in reports. This is `N351`'s rule applied to the brief itself.
Open question for the Hub (`§I8.15` item 3): which compile produced the w17 PDF that was reported as clean.

## 3. ⛔ WHAT DID NOT RUN — FOURTH PASS REQUIRED

- **w19 hub `maze-t-symmetric` is still running.** batmac has no output directory for it, so it is **NOT FOLDED and nothing is guessed** (registry `◈.11`). It feeds Table 27 `tab:cmaze-sym`, placeholders 4–10: WORDS-MAIN, WORDS-FULL, OBS, BOTH, the rival budget + 10× diagnostic, deletion, and rollout.
- **Not done by design:**
  - any draft, placeholder, figure or evidence edit (the four bar-d / lace-poe `\PLACEHOLDER`s are now fillable, and they belong to `placeholder-fill`);
  - any ruling on M1, M3, the w5 `composition` count restart, or the CMAZE-T persistence re-run;
  - editing `N-I8-21`'s body (it is annotated by `⟲` only);
  - re-deriving or rounding any number.
- **Path note:** the w19 spokes are at `~/clu/outputs/w19-*`, **not** `evidence/outputs/` (`§I8.15` item 2).

## 4. OPEN FLAGS — `claims_matrix.md` §I8.15 (10 items). Tiers and v2.24 are CURATOR-PROPOSED.

---

# PASS 2 (kept verbatim below, for the record)

# w18 spoke/registry-fold — FINDINGS (SECOND PASS; ⛔ THIRD PASS REQUIRED for three w15 spokes)

**Agent:** `doc-curator`. **Branch:** `w18/registry-fold`. **Machine:** batmac.
**Commissioned by:** `tasks/w18/launch-registry-fold-pass2.txt`.
**Scored runs: NONE. Compute: NONE. Background jobs: NONE. Sub-agents: NONE.** tmux was not touched and no process was killed.
⛔ `~/clu/evidence/` was not written. The base md5s are unchanged from pass 1 (registry `5568a6f2…`, matrix `6b8042dd…`), and the `w16-advisor-report.md` md5 is unchanged since arrival (`aace0197…`).

## 0. WHAT SHIPPED IN PASS 2

| file | pass 1 → pass 2 | what pass 2 adds |
|---|---|---|
| `negative_results.md` | 6,250 → **6,819** lines | **`N345`–`N357`** and `◈.4`–`◈.7`, a header note, and inserted `⟲ PASS 2` lines at `N339`, `N340` and `◈.3` |
| `claims_matrix.md` | 1,731 → **1,899** lines, **v2.23 (PROPOSED)** | **`§I8.12`**: `CM-W16-1`…`CM-W16-4`, `CM-W15-1` and `CM-W17-5`. **`§I8.13`**: 11 open flags. Also `⟲` lines at `CM-W17-1` and `CM-W17-4` |

**Additivity was checked with difflib against the pass-1 files:**
- Registry: 6,250 lines equal, 569 inserted, **0 replaced, 0 deleted**.
- Matrix: 1,731 equal, 168 inserted. Line 1 (the title) is the only replaced line, and the prior title is kept verbatim as its tail (`endswith` True).
- Across the registry's 349 `### N` headings there are **zero duplicate numbers**, and each of N345–N357 appears exactly once.

**Number audit.** Every decimal and thousands token in the new text (235 in the registry block, 57 in the matrix block) was checked mechanically and found **verbatim in a source read at this pass**. The sources were `w16-maze-wm/**`, the six w15 dirs, `w18-seeds-and-tables`, `w18-referee-fixes`, `DRAFT-v2.tex`, w12 crossover and pass-1. The audit caught one curator rounding (the N346 control values), which was replaced with the JSON's exact values.

## 1. THE FIVE LAUNCH ITEMS

| # | item | entry | the one-line result (numbers are the primary record's) |
|---|---|---|---|
| 1 | w16 configs that **failed VAL** | `N345` | **CMAZE-T Config B, VB3**: learned blind-seq 0.8781795… against a bar of 0.8757985… (CLU 0.925798…). The interface lifts the rival **+0.40** and the CLU **+0.20**. Link **interface** |
| | | `N346` | **CMAZE-O Config A, VB0**: best 0.020833… (0.0125 / 0.015 / 0.035) against a bar of ≥ 0.05. Link **content/read** |
| | | `N347` | **CMAZE-O Config C2, VB1 against 30k-step rivals**: CLU 0.13667 (0.0875 / 0.15 / 0.1725) against MLP 0.39667 (0.315 / 0.4125 / 0.4625), margin −0.26. ⚠ The **same CLU number PASSED** against the recipe rivals (MLP 0.0342). Link **read** |
| 2 | the w16 scored config that missed | `N348`, `CM-W16-1` | **C-field, runs 4–6.** The **rival margin is WITHDRAWN BY THE HUB**: the MLP at 30k steps reads 0.9875 / 0.99 / 0.99 against the CLU's VAL 0.64 / 0.73 / 0.73. ⭐ The **STORE GATE HOLDS**: all four controls are 0.000 on every seed (`MAZE-WM.json` l.369). **Two statements, never one**. Deletion **ran** |
| 3 | failing links | `N349` | **CMAZE-O has three links**: content/read, interface and read. **CMAZE-T rollout: READ** (validity 0.000 for the CLU and every matched rival, per-step 0.925 against 0.818) |
| 4 | six landed w15 spokes | `N352`–`N357` | Each is folded from its own FINDINGS, with its declared not-runs. The headline negatives: Shapes deletion is **not capability erasure** (`N355`); CMAZE-2b deletion **does not induce refusal** (`N356`); the CMAZE-O harness is **interface-only** (`N353`); **K\* is not resolvable at n = 3** and **K = 20 fails the gate** (`N354`) |
| 5 | ⭐⭐ the Hub report disagreed in **eight** places | **`N351`**, `CM-W16-4` | The artefact won all eight, and the curator checked each primary pointer. **A paraphrase is not a source.** ⚠ New finding: the on-disk report is the **corrected** version (sha256 `a533a67c…`), and §4b's report line pointers resolve only against the **superseded** version (sha256 `3c2115bd…`), which is no longer on batmac. That wording survives only in `TABLES-w18.json`. A residual remains in the corrected report at l.72 |
| — | *(curator-added)* | `N350`, `CM-W16-3` | The shipped across-wells control was **partly the identity** (`rng.permutation(K)`). The w16 fix is a derangement. ⚠ The w15 CMAZE-T smoke (Config-A K 4 / 4 / 8) is exposed |

**M1 and M3 are carried forward OPEN.** Both are HELD per `w18-referee-fixes/FINDINGS.md` l.4–5, and the ⟲ lines at `N339` and `N340` plus `CM-W17-5` record this.
⭐⭐ **M1 recurred LIVE in w16.** C-field's recorded win was withdrawn purely because the rivals were under-trained, and C2 flips WIN → FAIL on the rivals' budget alone. This is recorded against the M1 entry as evidence that the ruling is **load-bearing, not cosmetic**.
The pass-1 instruction to close items against referee-fixes is **done**: M2, M4, M5, M6 and M7(e) are APPLIED, hash-verified against the live draft (`8e45fa98…`).

## 2. ⛔ WHAT DID NOT RUN — THIRD PASS REQUIRED

**Still awaiting (running on the other box). NOT FOLDED, and nothing guessed** (registry `◈.7`):
- **`bar-d-score`**: the BAR D full-decoder GPU row, which goes to `tab:bard` (never `tab:encloop`); P11, P31.
- **`coverage-control`**: the n = 6 row on n = 3's train index, in `tab:ladder`; P4.
- **`lace-poe-rows`**: separate LACE and PoE rows on `tab:t7`, every ladder rung, and generation-table run 3; P3, P7.

**Declared not-done:**
- **Any draft edit.** The greps show `DRAFT-v2.tex` still carries 23.5117 (L1317, L2262) and "+0.012731 at K = 10" (L1247). MF14(i) is not carried. These belong to the Head or `placeholder-fill`, and AMENDMENT-1 applies.
- **Any evidence edit**: the report's l.72 residual, `w14-read-verb/FINAL-REPORT.md`, `w9-maze/LOG.md` l.220 and `w15-maze-deletion-2b/TABLE.md`.
- **Any ruling on M1 or M3.**
- **Adjudicating w12's 0.0093 against 0.0127.**
- **Re-deriving or rounding any number.**

**A path note:** `w18-seeds-and-tables/` is **not** under `evidence/outputs/`, although the launch file says so. It was read from `~/clu/outputs/w18-seeds-and-tables/` (§I8.13 flag 2).

## 3. OPEN FLAGS — `claims_matrix.md` §I8.13 (11 items). Tiers and v2.23 are CURATOR-PROPOSED.

---

# PASS 1 (kept verbatim below, for the record)

**Agent:** `doc-curator`. **Branch:** `w18/registry-fold`. **Base:** `base/iter8 = 148f3b7`. **Machine:** batmac.
**Window:** `START_UTC.txt` → `END_UTC.txt` in `~/clu/outputs/w18-registry-fold/`.
**Scored runs: NONE. Compute: NONE. Background jobs / PIDs: NONE.** ⛔ No machine-wide kill; tmux untouched.
⛔ `~/clu/evidence/` not written (base md5s re-checked at close, unchanged from arrival).

## 0. WHAT SHIPPED

| file | what | lines |
|---|---|---|
| `negative_results.md` | registry + one header note + the **◈ block**: `N338`–`N344`, fold-in ledger `◈.2`, **awaiting list `◈.3`**, declared not-runs | 5,973 → **6,250** |
| `claims_matrix.md` | matrix, title → **v2.22 (PROPOSED)** (prior title kept verbatim as its tail) + `§I8.10` (`CM-W14-3`, `CM-W17-1`…`CM-W17-4`) + `§I8.11` open flags | 1,624 → **1,731** |
| `FINDINGS.md`, `LOG.md`, `START_UTC.txt`, `END_UTC.txt`, `PIDS.txt` | record | — |

**Base:** `evidence/outputs/w14-refusal-displays-closeout/{negative_results,claims_matrix}.md`. This is the latest pair on batmac, and it is byte-identical to the worktree copy under `w14_refusal_displays_closeout/` (md5: registry `5568a6f2…`, matrix `6b8042dd…`). The repo has no `.claude/` directory, so by the w13 precedent the pair lives as a per-spoke copy in `w18_registry_fold/`.

**Additivity was checked with difflib, not just claimed:**
- Registry: 5,974 lines equal, 277 inserted, **0 replaced, 0 deleted**.
- Matrix: 1,624 lines equal, 107 inserted. The title line is the only replaced line, and the prior title is preserved verbatim as its tail (`endswith` True).
- Each of N338–N344 and each of the five CM rows appears as exactly one heading. Across the registry's 336 `### N<number>` headings there are **zero duplicate numbers**. Nothing was renumbered, so no redirect is owed.

## 1. WHAT WAS ADDED vs WHAT WAS ALREADY PRESENT

| task item | already present? | this pass |
|---|---|---|
| **w14 verb-vs-store (+0.83 vs +0.47)** as a standing caveat | ⛔ **Absent.** No `read-verb`, `0.8299` or `0.4749` in either file | **Added:** `N338` (tier A) plus `CM-W14-3`, a standing caveat with forbidden wordings and a re-opening condition. Numbers come from `w14-read-verb/FINAL-REPORT.md` §5.1 (lookup **+0.8299**, CLU **+0.4749**), §5.2 (controls do not collapse) and §5.3 (lookup **0.9648 / 0.9646 / 0.9808**, which is "keep everything") |
| wave-17 referee must-fix items still open | ⛔ Absent (w17 was never folded) | **Added:** all of M1–M7 are **OPEN at this fold**. `N339` M1 (fit pools **16,384 vs 131,072**; data-matched margin **+3.117…+17.875**, primary `w12b-nr6-and-kstar/KSTAR-LADDER.md` l.66/l.77). `N340` M3 (address book carried, not refit; **1,015,808 of 1,060,864** bytes, `w13-seeds-and-tables/TABLES.md` l.822/l.659; **83.5 %**, `w10-physics-section-evidence/EVIDENCE.md` l.382). `N341` M4 (SISA in the bib but uncited; CDM absent from the bib). `N342` M2. `N344` M5/M6/M7. Matrix rows `CM-W17-1`…`4` |
| claims whose artefact referee-3 could not locate | — | referee-3 found **no unlocated number** apart from the four honest `\TODO`s (`N344` item 4). **Added** one upstream mismatch it surfaced: `N343`, where `w14-read-verb/FINAL-REPORT.md` l.297 gives "8.2e-16" but `TABLES.md` l.360/l.462 and the run JSONs give **7.818e-16** |
| *(curator-added)* other wave-17 declared negatives | — | `N344` items 5–8: rendered-compositions set 2 DECLARED; plots not-drawn; lecun2022path uncited; placeholder count disagrees 26 / 29 / 31 |
| every **w16** config that failed val | — | ⛔ **AWAITING w16 — second pass required** |
| every **w16** scored config that missed its bar | — | ⛔ **AWAITING w16 — second pass required** |
| **w15** declared not-runs | — | ⛔ **AWAITING w15 — second pass required** |

## 2. ⛔⛔ AWAITING w15/w16 — SECOND PASS REQUIRED (exact list in registry `◈.3`)

**Wave 15.** No artefact of this programme's wave 15 exists on batmac:
- `evidence/outputs/` has no `w15-*` directory. `transfer-docs-sync-w15.md` belongs to an earlier campaign.
- `~/clu/outputs/w15-table-fixes/` is **empty**.
- Branch `w15/table-fixes` has **no commits** past `148f3b7`.

The launch says 2 of 5 spokes have landed; they have not landed on this machine. **Missing, per spoke:**
- `crossover-seeds`
- `single-design-seeds` (seeds 1–2 at K\*, plus deletion rows)
- `coverage-control`
- `lace-poe-rows`
- `bar-d-score`
- `maze-deletion-2b`
- `table-fixes`
- `cmaze-t-harness` and `cmaze-o-harness` (audits and val smokes)
- every w15 DECLARED NOT-RUN

**Wave 16 (not started).** Missing:
- every config that failed val, with its numbers and bar
- every scored config that missed its bar, with the failing link
- all CMAZE-T / CMAZE-O rows and declared negatives

⚠ `w17-advisor-report.md` §7 item 3 already quotes a w15 `crossover-seeds` figure. **Its primary artefact is not on batmac, so it was not folded.** Under AMENDMENT-1 it must not reach a draft.

## 3. DECLARED NOT-DONE

- **No w15/w16 fold:** there are no artefacts. Nothing was guessed.
- **No ruling on M1, M3 or M4:** those belong to the Advisor (`w17-advisor-report.md` §7). The curator's wordings are labelled as candidates.
- **No judgement on whether `w18/referee-fixes` closed an item:** that spoke is in flight, and its edits are not evidence yet. The second pass should close items against its FINDINGS.
- **M2's mechanism is not folded:** it exists only in that spoke's in-flight LOG.
- **`w14-read-verb/FINAL-REPORT.md` is not edited:** it is read-only evidence (`N343`).
- **No entries of their own for w13 `encoder-adapt`, w13 `seeds-and-tables`, w12b `nr6-and-kstar` or w14 `request-layer-fix`:** these are outside the named items. The registry cites none of the first three (flag `§I8.11` item 3).
- **No statistic re-derived.** The only operations were quoting and mechanical `grep` counts, which are marked.

## 4. OPEN FLAGS — see `claims_matrix.md` §I8.11 (six items). Tiers and v2.22 are CURATOR-PROPOSED.
