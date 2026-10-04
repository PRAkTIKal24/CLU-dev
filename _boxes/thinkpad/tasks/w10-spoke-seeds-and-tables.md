> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **The evidence base is `/c/chlu/evidence/outputs/`** (live, shipped 2026-09-21).
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it points at a FROZEN, OLDER snapshot of the same
>   tree, and every number in it is out of date. Reading it silently produces a stale paper.
> - ⭐ **You write ONLY to `/c/chlu/outputs/w10-seeds-and-tables/`.** ⛔ No other spoke's tree. ⛔ Never
>   into the Head's profile. ⛔ Never create a `.claude/` directory in your worktree.
> - ⭐ **Poll any background job to completion INSIDE a turn** (loop on its output file with sleep).
>   ⛔ Do not end a turn while detached work is running — nothing will re-invoke you.
> - ⛔⛔ **Never run a box-wide kill** (`taskkill /F /IM python.exe` or equivalent). Kill only PIDs
>   you captured at launch. Another agent's training run has been destroyed this way before.
> - ⭐ **Commit and bundle your branch WHOLE before you finish** — `git bundle create <f> <your
>   branch>`, ⛔ not `--all`, which mixes other agents' branches into your file.
> - ⭐ Write a `LOG.md` in your output dir as you go. It is the record, and the Hub reads it.
> - ⭐ If something is not run, it is a **DECLARED NOT-RUN with a reason**, ⛔ never a zero, a blank
>   or a silent omission.

# w10 spoke/seeds-and-tables — every table row at 3 seeds, from artefacts only

- **Agent:** `results-analyst`.
- **Output:** `/c/chlu/outputs/w10-seeds-and-tables/TABLES.md` + `figures/` + `TABLES.json`.
- **Owns:** `/c/chlu/outputs/w10-seeds-and-tables/**` only. ⛔ Touches no other spoke's tree,
  ⛔ no `chlu/` code, ⛔ no other wave's outputs.
- **Scored runs:** ⛔ **NONE.** This spoke re-reads existing artefacts. If a row needs a run that
  does not exist, it is a **DECLARED NOT-RUN with the launch line emitted**, never a blank and
  never a zero.

## THE ONE JOB
Build the paper's **main scoreboard table** from artefacts on disk: every row at **3 seeds on
generation 2**, each row labelled **RESULT / DISPLAY / ablation**, with figures regenerated from
the artefacts they cite.

## ACCEPTANCE CRITERION
`TABLES.md` contains one table where **every cell traces to a file path recorded in `TABLES.json`**,
and a reader can regenerate every figure from that manifest alone. ⭐ A number with no path is a
defect, not a row.

## ⛔ NON-NEGOTIABLE
- ⭐ **All three seeds or none.** The programme reports three measurements, never a best.
- ⛔ **Runs at different compute budgets DO NOT SHARE A TABLE.** w8's CPU leg ran
  `joint_steps=400, n_train=8192`; its GPU leg ran `4000 / 32768`. They are **two tables**, and
  each says its pins in its caption.
- ⛔ Absolute row-3/row-4 values from `w8/encoder-loop` **do not share a table with the incumbent's**
  — a shortened decoder recipe (`w8_v1_short`) is a different instrument, not a different setting.
- ⚠ **CMAZE-1's n-way row is scored as a CONJUNCTION while its ceilings are per ROUTE.** Report both
  or the row is not comparable with its own ceiling. (w9 caught this against itself.)
- ⭐ Every CMAZE-1 composition row carries its **BLIND-SEQUENTIAL** column **measured on that split**
  (0.9037–0.9130), ⛔ not the published 0.4980.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- A cell you cannot trace to a file ⇒ the artefact base is thinner than the claims and you say so.
- Two rows at different pins sitting in one table ⇒ the table is void and must be split.
