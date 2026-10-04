# FINDINGS — w15 `spoke/table-fixes`

**The zero-compute must-fix pass.** Branch `w15/table-fixes`, base `base/iter8 = 148f3b7`,
worktree `/c/chlu/worktrees/CHLU-w15-table-fixes`, output `/c/chlu/outputs/w15-table-fixes/`.

**Scored runs spent: ZERO.** **Compute: ZERO.** No model fitted, no arm run, no seed scored, no
figure regenerated, no background job launched, no PID owned (`PIDS.txt`), no kill of any kind
issued. No ledger row was needed in `/c/chlu/SCORED-LEDGER.md` because nothing scored was run.
⛔ `verify_tables.py` **NOT touched.** ⛔ `w10/draft-v0` and `w10/referee` **NOT merged** — the base
is `148f3b7` and nothing else.

**Acceptance artefact:** `/c/chlu/outputs/w15-table-fixes/TABLE-FIXES.md` — each of MF2, MF14(i),
N5, N6, N7, N8 with before → after, the file and line changed, and MF2's lead sentence quoted with
its convention.

---

## 1. WHAT I DID

Landed **22 guarded exact-string edits** on one file, `w13_draft_revise/DRAFT-v1.md`
(137 insertions, 40 deletions), closing the six bookkeeping items `w13/referee-2` returned:
**MF2** and **MF14 part (i)** (REFEREE-v1 §2, both ⚠ PARTIAL) and **N5, N6, N7, N8** (REFEREE-v1 §3,
four of eight new must-fixes).

The edit is a script, `w15_table_fixes/apply_table_fixes.py`, not a hand pass: every replacement is
guarded by a unique mark, the run is idempotent (a second run reports `already applied` ×22 and
writes nothing), and `git checkout 148f3b7 -- w13_draft_revise/DRAFT-v1.md` reverts it exactly.

⭐ **Every number this pass wrote into the draft was re-derived from its primary artefact before the
edit — none was copied from the referee.** That caught one referee arithmetic slip (§4 below).

**Artefacts:**

| file | what it is |
|---|---|
| `/c/chlu/outputs/w15-table-fixes/TABLE-FIXES.md` | the acceptance artefact — before → after, item by item, with line numbers |
| `/c/chlu/outputs/w15-table-fixes/DRAFT-v1.BEFORE.md` | the file as it stood at `148f3b7` |
| `/c/chlu/outputs/w15-table-fixes/DRAFT-v1.AFTER.md` | the amended file |
| `/c/chlu/outputs/w15-table-fixes/DRAFT-v1.diff` | `git diff 148f3b7 -- w13_draft_revise/DRAFT-v1.md` |
| `/c/chlu/outputs/w15-table-fixes/apply_table_fixes.py` | the patch script (also in-tree at `w15_table_fixes/apply_table_fixes.py`) |
| `/c/chlu/outputs/w15-table-fixes/LOG.md` · `PIDS.txt` · `START_UTC.txt` · `END_UTC.txt` | liveness and discipline record |
| `/c/chlu/outputs/w15-table-fixes/w15-table-fixes.bundle` | the branch, whole, `git bundle verify`-ed |

---

## 2. THE NUMBERS, EACH WITH THE ARTEFACT PATH IT CAME FROM

⛔ **No number in this file lives only in prose.** Every figure below was read or computed by this
spoke from the path in its right-hand column.

### 2a. The BLIND columns that moved into the maze main-text table (MF14 i)

Read directly from the three scored JSONs — **not** from `TABLES.md` — at
`/env_columns/<row>/blind/<column>/validity`:

| row | column | seed 0 / 1 / 2 | mean | artefact |
|---|---|---|---|---|
| `unseen_pair` | `BLIND-SEQUENTIAL(key order from A)` | 0.9077 / 0.8905 / 0.9128 | **0.9037** | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`, `.../scored2_seed1_test.json`, `.../scored3_seed2_test.json` |
| `unseen_pair` | `BLIND-FACTOR    (A x B local only)` | 0.4564 / 0.4677 / 0.5046 | **0.4762** | same three files |
| `nway` | `BLIND-SEQUENTIAL(key order from A)` | 0.9229 / 0.8933 / 0.8841 | **0.9001** | same three files |
| `nway` | `BLIND-FACTOR    (A x B local only)` | 0.4447 / 0.4665 / 0.5048 | **0.4720** | same three files |

These reproduce `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c cell for cell.
`/env_columns` carries exactly **four** keys — `unseen_pair`, `nway`, `fragment`,
`fragment_cross_maze` — on every scored seed (same three files), which is why rows 1 and 4 carry
**n/a** and **DECLARED NOT MEASURED** rather than a blank or a zero.

CMAZE-1 harness-tier contrast, printed in §6.3 so the rise is attributed to the request layer and
not to one column's rule: BLIND-SEQUENTIAL **0.5317 (0.5550 / 0.5300 / 0.5100)**, BLIND-FACTOR
**0.2350 (0.1850 / 0.2500 / 0.2700)** — `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T3.
⛔ CMAZE-1 and CMAZE-2 remain in **separate tables**; nothing in this pass merged them.

### 2b. The byte figures, each with its convention named (MF2 / N7 / N5)

| object | bytes | ÷ 524,288 B pin | convention | artefact |
|---|---|---|---|---|
| w8 statistic alone (cov + running means + counts), 3 TEST seeds | 12,327,096 / 11,957,784 / 11,867,336 | **23.5121× / 22.8077× / 22.6351×** | statistic ÷ pin | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T4 byte ledger, row `2. covariance / count ledger, TOTAL`; division by this spoke |
| w8 four-part honest total, 3 TEST seeds | 12,869,482 / 12,500,170 / 12,409,722 | **24.5466× / 23.8422× / 23.6697×** | four-part total ÷ pin | same table, rows `honest total` and `x THE 524,288 B PIN` (stored field; reproduced exactly) |
| w9 statistic | 35,904 of 1,642,752 deployed | **2.19 %** | statistic ÷ deployed read | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c-quater |
| K\* = 32, **TEST**, 3 seeds | 2,144,178 / 2,137,418 / 2,141,578 | **4.0897× / 4.0768× / 4.0847×** | four-part total ÷ pin | totals `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4; seed-0 ratio also stored at `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T7; the other two divisions by this spoke |
| K\* = 32, **VAL pin leg** | 2,143,138 | **4.0877×** | four-part total ÷ pin, **val** | `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 row `(val pin)`; `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 row `K = 32` |
| K = 5 (MI partition, full-square), 18 fitted legs | 12,523,008 – 13,041,864 | **23.8857× – 24.8754×** | four-part total ÷ reference | `/c/chlu/outputs/w12-composition-ladder/BYTES.json` `/summary/single_design_K5` (read directly) |
| K = 5 (ward, triangle) | 4,118,034 | **7.8545×** | four-part total ÷ pin | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 K = 5 row |

⛔ **Seed count travels with every one of these rows**: the w8 and K\* TEST rows are **3 seeds**;
T7's own ledger is **1 seed (id 0)** — its header says *"Every row of T7 is a 1-seed row"*; the K\*
val pin is **not a seed** at all. That distinction is what N5 was about and it is now printed in all
six places the draft quotes these ratios (`TABLE-FIXES.md` §N5 table).

### 2c. The comparison the maze table now shows on its face (MF14 i)

Same held-out TEST rows, same exact integer judge, same three seeds
(`/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json`;
`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1a and §1c):

| row | CLU | best learned baseline | BLIND-SEQUENTIAL | BLIND-FACTOR | SEQ ÷ CLU | FACTOR ÷ CLU |
|---|---|---|---|---|---|---|
| 2 unseen pair | 0.1279 | 0.1840 | **0.9037** | **0.4762** | **7.07×** | **3.72×** |
| 3 n-way (conjunction) | 0.0284 | 0.0493 | **0.9001** | **0.4720** | **31.7×** | **16.6×** |

---

## 3. ⛔ THE THING THE TASK FILE ASKED ME TO LOOK FOR, AND IT HAPPENED

> *"If moving the BLIND columns into the main table makes the maze claim look weaker, that is
> information the paper needs — do not soften it, do not move them back. Report it."*

**It does. Reported, not softened, not moved back.**

With the columns in place (§6.1, `DRAFT-v1.md:665-671`), two **request-ignoring reference policies**
outscore both the CLU and the best learned baseline on both composition rows — BLIND-SEQUENTIAL by
**7.07×** and **31.7×** the CLU's own score, BLIND-FACTOR by **3.72×** and **16.6×** (§2c).

⭐ **What it costs the paper, precisely.** It costs **no result**, because §6.1 claims none: BAR 3
already fails and the draft already said *"under the hub's own pre-registration NO RESULT ROW IS
CLAIMED for the CLU arm here."* What it changes is the **reading**, and the change is toward the
stronger statement:

- The weak form the draft could previously be read as making — *"the CLU fails on an exact-judge
  route task"* — is **not supported by this split**, because a policy that never reads the request
  scores **0.90** on it.
- The supported form is narrower and much harder to attack: **the environment's own tripwire fires
  before the arm is reached.** CMAZE-2's request layer puts **1,242 of 1,449** optimal routes on
  exactly one key, so rule B has no key ORDER to trap (`/c/chlu/outputs/w9-maze/LOG.md` §3, quoted
  in §6.3), and the same tripwire reproduces at **0.5317** on CMAZE-1 proper. The split is the
  finding.
- ⛔ Neither blind column may be quoted as a baseline the CLU lost a fair comparison to: both are
  class **`reference`** in the artefact (`scripts/w10_build_tables.py:436-437`), not arms, and they
  are not byte-matched. That caveat is printed in the caption, not left to the reader.

⭐ **This is a net gain for the paper even though the table reads worse**, because §6.2's mechanism
result — plan space **1.0000** vs latent space **0.13** — is the thing §6 actually contributes, and
it does not depend on the CLU's absolute score on this split at all.

---

## 4. TWO THINGS I FOUND THAT NOBODY ASKED FOR

**(a) The referee's own MF2/N7 ratio is a rounding slip.** REFEREE-v1 §2 and §3/N7 give the
statistic's cost as **23.5117×**. Recomputed: `12,327,096 / 524,288 = 23.512069702148438` →
**23.5121×**. The draft now carries **23.5121×**, and `TABLE-FIXES.md` records the correction. ⭐ At
the two decimals the task file names (**23.51×**) the conclusion is unchanged — which is exactly why
re-deriving rather than copying is cheap insurance.

**(b) "TABLES.md" is ambiguous in this draft and four of my six items depended on resolving it.**
The in-tree `w12r_artefact_mustfix/TABLES.md` (built `2026-09-21T12:50:40Z`) carries **only T1–T6**.
T7, T8, T9, T9b, T10, T11 and T12 — cited by N5, N6, N8 and §A — live only in
`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md`. A reader resolving *"`TABLES.md` T7 byte ledger"*
against the in-tree copy finds nothing. ⭐ **Choice recorded per COMMON.md:** every T7+ citation this
pass touched now carries the **full path** to `w13-seeds-and-tables/TABLES.md`; T1–T6 citations keep
the short form, which resolves either way. Closest to an ICLR main-track result: a citation that
resolves. ⚠ The draft still has short-form `TABLES.md` T7+ citations in lines this pass did not
touch — see §6, OPEN-W15-1.

**(c) An environment bug other w15 spokes will hit.** `/c/chlu/env_w3.sh` sets
`PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"` with a POSIX `:` separator, but the pinned
interpreter is **Windows CPython 3.11**, which splits `PYTHONPATH` on `;`. The whole value is read
as one bogus path and the wave-2 site-packages never reaches `sys.path`
(`ModuleNotFoundError: No module named 'equinox'`). `import chlu` still succeeds — from CWD — so the
COMMON.md assert can pass on a box where **no** pinned dependency is importable. Workaround used
here (nothing outside my own tree edited):
`export PYTHONPATH="C:/chlu/worktrees/CHLU-w15-table-fixes;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"`,
after which the assert passes with `jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4`. ⛔ I did not edit
`env_w3.sh` — it is not mine.

---

## 5. WHAT DID **NOT** RUN

All **DECLARED NOT-RUN**, with reasons — none of these is a zero and none is a failure.

| not run | why |
|---|---|
| **Any scored run** | The task is *"Scored: ZERO. Zero compute. Every item here costs 0 runs."* Nothing in MF2, MF14(i) or N5–N8 needs a measurement; all six are bookkeeping on numbers that already exist. |
| **Any val run, any fit, any figure regeneration** | Same reason. The figures referenced by §6.1 (`fig1_cmaze2_main_scoreboard.png`, `fig2_nway_conjunction_vs_route.png`, `fig5_blind_sequential_on_this_split.png`) were **not** redrawn — see OPEN-W15-1. |
| **`verify_tables.py`** | ⛔ Explicitly not mine (`table-fixes.md`: *"Do not touch the verifier"*). Not run, not edited. Its 52 sources / 872 cells / 1,786 per-seed values / 8 figure hashes are unaffected by this pass, which changed **no artefact**, only the draft that cites them. |
| **`TABLES.md`, `TABLES.json`, `build_tables.py`** | Not mine. The draft was corrected to match them, ⛔ never the other way round. |
| **The other four of REFEREE-v1's eight new must-fixes — N1, N2, N3, N4** | ⛔ **DECLARED NOT-DONE, out of scope.** My task file names N5–N8 only. N1 and N2 concern the K = 5 covariance convention and w12b's same-convention opponent; N3 the bridge table's dropped columns; N4 the K\* control gate. All four are **substantive, not bookkeeping**, and N1/N2 in particular may change a printed verdict. They are untouched and still open. |
| **REFEREE-v1's twelve should-fixes and five nice-to-haves** | Out of scope, untouched, still open — except **SF-a**, which is discharged as a side-effect of closing MF2 and MF14 (`DRAFT-v1.md:1026`). |
| **`w10/draft-v0`, `w10/referee`** | ⛔ Deliberately not merged, per the task file. |
| **Sub-agents** | ⛔ None spawned. |
| **Background jobs** | ⛔ None launched; `PIDS.txt` is empty by design. No kill issued. |

---

## 6. OPEN ITEMS THIS PASS LEAVES BEHIND

| id | what |
|---|---|
| **OPEN-W15-1** | ⚠ **`fig1_cmaze2_main_scoreboard.png` no longer matches its table.** §6.1's table gained two columns; the figure was drawn from the five-column version. Redrawing it means touching `build_tables.py` and re-running `verify_tables.py`'s figure hashes — ⛔ both out of my scope. Declared, not silently left. |
| **OPEN-W15-2** | ⚠ Short-form `TABLES.md` citations to **T7–T12** remain in lines this pass did not touch; they resolve only against `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md`, not against the in-tree `w12r` copy. A one-line "which TABLES.md" note at the head of the draft would close it. |
| **OPEN-W15-3** | ⚠ **N1–N4 remain open** (§5). ⛔ The draft's §A now says all sixteen MFs are closed; it does **not** say the eight N-items are, and four of them are not. |
| **OPEN-W15-4** | ⚠ Correction **C20** (`DRAFT-v1.md:1280`) is still filed as a never-quote note aimed at the advisor report. §4.4 and §5.1 now point at it and at each other, and flag that it moves convention and partition together, but C20's own text was **not rewritten** — it is another wave's correction register. |

---

## 7. WHAT WOULD HAVE PROVED THIS WRONG

The pass is bookkeeping, so its falsifiers are arithmetic and provenance, and each was checked:

1. **If `12,327,096 / 524,288` had not been ≈23.51×**, MF2's whole clause would be wrong. Computed:
   **23.512069702148438**. ✅ (And the referee's 23.5117 is the slip, not mine — §4a.)
2. **If the four-part honest total had equalled the statistic**, N7 would be a non-issue and MF2's
   second half already closed. They differ by **542,386 B on seed 0** (24.5466× vs 23.5121×). ✅
   The defect was real.
3. **If BLIND-FACTOR had already appeared as a number in the draft**, MF14(i) would have been
   partly done. `grep -c "0.4762" DRAFT-v1.BEFORE.md` = **0**; `grep -c "BLIND-FACTOR"` = **2**,
   both prose instructions. ✅ The defect was real.
4. **If `/env_columns` had carried a blind block for row 4**, the DECLARED NOT MEASURED cell would be
   a fabrication. It carries exactly four keys on all three seeds. ✅
5. **If T7 had been a 3-seed table**, N5's charge would collapse. Its own seed-count block reads
   *"n = 1 (seed 0). Every row of T7 is a 1-seed row."* ✅
6. **If `w12b .../FINDINGS.md` §13 had been a ruling rather than a request**, N8's fix would be the
   error. Its heading is *"WHAT THE HUB IS ASKED TO RULE ON"*. ✅
7. **If the two "K = 5" objects had been the same object**, N6 would be pedantry. Their four-part
   totals differ by **3.1×** and their partitions (`factor` label-MI, widths 777/936/216/60/59 vs
   `ward_balanced` label-free, five ≈410-wide wells) and conventions (full-square vs
   symmetric-triangle) both differ. ✅
8. **If moving the BLIND columns in had left the maze table reading the same**, the task file's
   falsifier would not have fired. It fired — §3. ✅ Reported, not softened.

⛔ **What would still prove the pass wrong and was not checkable here:** if the Head reads §6.1's
new columns as arms rather than references, the paper acquires a comparison it cannot defend. The
caption forbids it in three sentences; a reader determined to ignore the caption is not something a
table edit can fix.
