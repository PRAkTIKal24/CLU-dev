> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · matrix + registry at `/c/chlu/evidence/`.
>   ⭐ Current state = **w9-, w10- and w12-** reports there. ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** —
>   frozen, older, every number stale. ⛔ `/c/chlu/evidence/` is READ-ONLY.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w13-w13-spoke-seeds-and-tables/`** and your own worktree (branch `w13/w13-spoke-seeds-and-tables`).
> - ⭐ **START_UTC.txt first · LOG.md as you go · END_UTC.txt last.**
> - ⭐ **Assert `chlu.__file__` resolves inside YOUR worktree** before any number — the shared
>   site-packages has an editable `.pth` to the Head's checkout; if it wins, every number is void.
> - ⭐ **Poll background jobs to completion INSIDE a turn.** ⛔ Never end a turn with detached work
>   running — nothing will re-invoke you. ⚠ A w12 hub stalled exactly this way at a rung boundary.
> - ⛔⛔ **Never a box-wide kill.** Capture your PIDs at launch; kill only those. ⚠ **Other agents are
>   running on this box** — never touch their sessions, worktrees, output dirs or processes.
> - ⭐ **Commit and bundle your branch WHOLE:** `git bundle create <f> w13/w13-spoke-seeds-and-tables` — ⛔ that branch
>   only, **never `--all`**. Verify, and check `list-heads` shows exactly ONE ref.
> - ⭐ **A published file is not a finished file.** ⚠ Three times this programme, a READY json appeared
>   while its author was still writing. If you gate on someone's artefact, check it is COMPLETE
>   (non-null fields, expected counts), not merely present.
> - ⭐ **Anything not run is a DECLARED NOT-RUN with a reason** — ⛔ never a zero, a blank or silence.
> - ⭐ **Every number names the file it came from.** ⛔ No sub-agents for long work.

# w13 spoke/seeds-and-tables — fold wave 12's rows into the main table, at 5 seeds where possible

- **Agent:** `results-analyst`. ⭐ **ZERO scored runs.** Re-reads artefacts; produces no new model number.
- **Output:** `/c/chlu/outputs/w13-seeds-and-tables/` — TABLES.md · TABLES.json · figures/ · verify_tables.py.
- **Owns:** its output dir + branch `w13/seeds-and-tables`.
- **Start from w10's build**, at `/c/chlu/evidence/outputs/w10-seeds-and-tables/`, and the corrected
  artefacts at `/c/chlu/evidence/outputs/w12r-artefact-mustfix/`. ⛔ Do not start from scratch.

## THE ONE JOB
**The main table = rows 3–4 + the n-way ladder + deletion (value / conjunction / item) + refusal
displays + ablations (composition-only store, item store).** Every row labelled
**RESULT / DISPLAY / ablation**, every cell traceable to a path in `TABLES.json`.

## ⭐⭐ THE TWO NOTATIONS THAT MUST NEVER COLLIDE — the Hub got this wrong once
- **n** = number of jointly composed factors (the ladder rungs, n = 2…6).
- **K** = number of WELLS (the partition; the sweep ran K = 5…32 and chose **K\* = 32**).
⛔ **Never write "k" for both.** Every ladder row states its **K**; every wells row states its **n**.

## SEED COUNTS — now heterogeneous, and that is the hazard
⭐ Generation-2 caches now exist for **seeds 3 and 4** (`CACHE-READY.json`, 59 blocks each, re-encode
gate passed, shape matching seeds 0–2).
⚠ **But the rows already measured are NOT all at the same n:** the crossover's scored row 1/3 was
spent at **4 seeds, declared** (seed 4 landed an hour after it closed); the ladder ran **3 seeds**.
⛔⛔ **NEVER mix 3-, 4- and 5-seed rows in one plot without a printed `n` per row.** ⭐ Print the seed
count **as a column**, not a footnote.

## ⛔ BYTE MATCHING — print BOTH, and caption the flip
Two matchings exist and they **tell opposite stories about the k-trend**:
- **1×** = the single design's own measured total (K=5: 4,118,034 B = **1,029,508** float32 params;
  K\*=32: 2,143,138 B = **535,784** params).
- **524,288 B reference** = **131,072** params (rivals matched to within ±0.63 %).
⭐ Against the 524,288 B rivals the store's margin **GROWS** with n (+7.1…+9.7 at n=2 → +15.0…+22.4 at
n=6). ⭐ Against the 1×-matched MLP it stays **NEGATIVE** (−0.054 … −0.077) at n = 2, 4, 5, 6.
⛔ **Print both columns on every row and caption the flip explicitly** — a reader who sees one column
draws the opposite conclusion.

## ⭐ THE FOUR-PART LEDGER IS A PRINTED ROW, NEVER NETTED
atoms · covariance/count ledger · address book · coupling/hole table. ⭐ The referee rejected the
draft partly because Table 1 printed **no byte row** while the object is **24.5× the pin**.
⭐ And print the K-curve: the honest total **FALLS** 7.48× → 2.07× from K=5 to K=32, because the
covariance shrinks faster than the address book grows.

## ACCEPTANCE CRITERION
⭐ **`verify_tables.py` PASSES** — it does not import the builder; it re-reads every source from the
path in the manifest. ⛔ **A fix that breaks the verifier is not a fix.** Report counts before/after.
⭐ **Edit the GENERATOR, never the outputs by hand** (w12r's method — it is what makes the check real).

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- A cell you cannot trace ⇒ the evidence base is thinner than the claims; say so.
- Rows at different pins, seeds or decoder recipes sharing one table ⇒ the table is void; split it.
