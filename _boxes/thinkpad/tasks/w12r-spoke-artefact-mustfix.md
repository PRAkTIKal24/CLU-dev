> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · registry + matrix at `/c/chlu/evidence/`.
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — frozen, older, every number stale.
>   ⛔ `/c/chlu/evidence/` is **READ-ONLY**; write copies into your own output dir.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w12r-artefact-mustfix/`** and your own worktree.
> - ⭐ **START_UTC.txt first, LOG.md as you go, END_UTC.txt last.**
> - ⭐⭐ **ZERO compute, ZERO scored runs.** ⛔ If you think you need a run, you have misread the job —
>   name the run you would need and leave the item open.
> - ⛔⛔ **Never a box-wide kill.** ⚠ **Wave 12 is RUNNING on this box right now** (3 agents,
>   `w12-*` tmux sessions). ⛔ Do not touch their sessions, worktrees, output dirs or processes.
> - ⭐ **Commit and bundle your branch WHOLE:** `git bundle create <f> <your branch>` — ⛔ never
>   `--all`. Verify, and check `list-heads` shows exactly one ref.
> - ⭐ **Anything not closed is a DECLARED OPEN ITEM with the reason** — ⛔ never silently dropped.
> - ⭐ **Every number names the file it came from.**

# w12r spoke/artefact-mustfix — close the referee's ARTEFACT-level must-fixes

- **Agent:** `experiment-engineer`. ⭐ **ZERO compute. ZERO scored runs.** Artefacts and tables only.
- **Output:** `/c/chlu/outputs/w12r-artefact-mustfix/` (LOG.md, FINDINGS.md, corrected artefacts).
- **Owns:** its output dir + branch `w12r/artefact-mustfix`. ⛔ Not the draft prose — **the Head
  writes prose.** You fix the ARTEFACTS the prose has to cite.
- **Read first:** `/c/chlu/evidence/outputs/w10-referee/REFEREE-v0.md` (16 must-fix, MF1–MF16) and
  `/c/chlu/evidence/outputs/w10-seeds-and-tables/TABLES.json`.

## THE ONE JOB
Close the **artefact-level** must-fixes — the ones that are wrong *in a file*, not wrong in a
sentence. ⭐ The Head fixes sentences; you fix what they point at.

## THE FIVE, IN THE HEAD'S ORDER
1. ⛔⛔ **MF3 — THE BYTE ROW.** Table 1 (the lead) prints **no byte row**, and the object is
   **24.5× the byte pin** everything else is held to. ⭐ **Add the four-part ledger as a printed row
   on every table that carries the store** — atoms · covariance/count ledger · address book ·
   coupling/hole table. ⛔ **Never netted into one number.** Without this the paper rejects on sight.
2. ⛔ **MF4 — DEFINE "EXACT."** The measurements support **STORE-SCOPED** exactness only: ⭐ the
   frozen encoder and decoder are **not refit**. Add that definition to `TABLES.json` as a field the
   prose can cite, and ⛔ flag every cell where "exact" appears without it.
3. ⛔ **MF7 — THE MISLABELLED ROW.** A row is labelled **RESULT · 3 / TEST** whose own artefact says
   *"No scored run and no claim"*. ⭐ **Relabel it to what the artefact says.** ⛔ Do not relabel the
   artefact to match the table.
4. ⛔⛔ **MF10 — THE ENVIRONMENT RELABEL.** Wave 9 scored **CMAZE-2** (a request layer over CMAZE-1;
   engine, solver and exact judge imported unmodified), ⛔ **not CMAZE-1.** Relabel every w9 maze row
   and define CMAZE-2 in one line. ⭐ **The harness tier IS CMAZE-1 proper.** ⛔ **The two must never
   share a table** — different environment, not merely different budget.
   ⭐ The three numbers only read coherently once separated: **0.4980** published CMAZE-1 ·
   **0.5317** CMAZE-1 at the harness tier · **0.9037** CMAZE-2.
5. ⛔ **MF11 —** *"FRAGMENT-LOOKUP 0.0000 everywhere"* is **false on w9's own split**: same-maze reads
   **0.0741 / 0.5105**. ⭐ The 0.0000 figures are the **cross-maze** quantity. Label both.

## ⛔ AND THE PHRASING THAT THREE SPOKES INDEPENDENTLY FLAGGED
⛔ **Never "9/9 seeds."** It is **3 CONFIGURATIONS × 3 seeds inside ONE implementation.**
⭐ Report seed count and configuration count **as separate fields** in `TABLES.json`.
⭐ The peer implementation's band is **2.842e-14 … 1.126e-09** — five orders coarser, worst leg below
its own **4.282e-09** reproducibility floor. ⭐ Print the **three implementations as three rows.**
⭐⭐ **The strongest form is the FAILURE AGREEMENT** — the second implementation reproduced the same
seed-2 failure with different code (**14.377 ulp** vs 32.90). Make that a citable row.

## ACCEPTANCE CRITERION
⭐ **`verify_tables.py` still PASSES** after your edits (it does not import the builder; it re-reads
every source from the path recorded in the manifest). ⛔ **A fix that breaks the verifier is not a
fix.** Report the verifier's counts before and after.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- A must-fix that cannot be closed without a new measurement ⇒ ⭐ **say so and leave it open**, naming
  the run it needs. ⛔ Do not invent the number.
- The byte ledger for some arm does not exist in any artefact ⇒ that is a **gap in the evidence base**,
  report it as one.
