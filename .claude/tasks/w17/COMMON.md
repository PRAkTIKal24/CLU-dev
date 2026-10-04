# WAVE 17 — THE PAPER-DRAFT HUB. COMMON RULES (every spoke reads this first)
**Machine: `batmac` (Apple M4, 10 cores, 16 GB). ⛔ This machine NEVER uses GCloud.**
**ZERO COMPUTE, ZERO SCORED RUNS.** Runs in parallel with wave 16 on the other box.

## Paths on this machine
- Repo (base `base/iter8 = 148f3b7`): `~/clu/workspace` — venv at `~/clu/workspace/.venv` (jax 0.9.0 / numpy 2.4.1 / eqx 0.13.4, pinned identical to the ThinkPad).
- **Evidence base (READ-ONLY, your only source of numbers): `~/clu/evidence/outputs/`**
- Your worktree: `~/clu/worktrees/CHLU-w17-<slug>`, branch `w17/<slug>`.
- Your output dir: `~/clu/outputs/w17-<slug>/` — the ONLY place you write outside your worktree.
- Style reference PDF: `~/clu/ship/PJ_Writing_Style_Context.pdf`
- Shared draft directory (see ownership below): `~/clu/outputs/w17-paper/`

## ⛔⛔ THE ONE RULE ABOVE ALL OTHERS
**Every number in the draft comes from an artefact in `~/clu/evidence/outputs/`, and carries a LaTeX comment naming the file it came from.**
⛔ **You do not invent, round, re-derive or "recall" a number.** If you cannot find a number's artefact, write `\TODO{no artefact for X}` and list it in your report. ⛔ A number that lives only in prose does not exist.

## Claims discipline — these are hard constraints, not style preferences
- **A verb claim is not a store claim.** w14: the same merge lifts the matched-bytes lookup floor **+0.83** vs the CLU's **+0.47**. The composition operator is an **INTERFACE** contribution.
- A **store** claim requires the charter-C controls to **collapse**. Otherwise it is an interface row or nothing.
- The **lookup table is a printed floor**, never a rival. On the maze it reaches **0.97** — print that honestly as the **within-maze** floor **that cannot transfer across mazes**.
- **Naive vs naive:** rivals matched on read-path params (~130k floats on Shapes). ⛔ The inflated-rival row does not exist in the paper.
- ⛔ **CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T never appear in one table.**
- ⛔ **For n ≥ 3 the ladder is a TRAINING-DATA axis** (rungs 3–6 share the identical 120 corners). Only **n = 2 vs n = 6** is a depth contrast. The Head's one-line inference note must appear with the ladder.
- **Deletion:** store-scoped exactness quoted from the deletion short's own scoping; frozen encoder/decoder **declared outside the deletable object**; post-deletion adaptation **light and data-free** — ⛔ never "no changes". Rival deletes only by retrain-without, cost printed.
- ⛔ **Every byte figure travels with its convention named.** Seed count is a column on every table.
- ⛔ **Never "9/9 seeds". Never "no changes after deletion". Never the TensorFaces phrasing.**
- **Notation:** n = factors jointly composed; K = wells.
- ⚠ The **ladder and the crossover priced the 1× opponent differently.** Use ONE convention throughout, or print the **938,413–1,710,324** bracket. State in a LaTeX comment which you did.

## Appendix-maximalism
Main results in the main text; **everything else — corollaries, negatives, extra plots — goes to the appendix. NOTHING is omitted.** ⛔ Do not prune. The venue's page limit is reported in `VENUE.md` but **the Hub does NOT trim to it** — trimming is the Head's later editing pass, outside these waves. Obey the **≤ 12 pp main-text** working bound.

## Anonymity
Double-blind. **Self-citations in the third person.** ⛔ No author names, no institution, no repo URL, no acknowledgements in the submitted `.tex`.

## Operating rules
- ⭐ Everything here is **PRE-AUTHORISED**. ⛔ Do NOT wait for the Hub, the Head or the Advisor. Where a choice is open, take the option closest to an **ICLR main-track result**, record it in `LOG.md`, and proceed.
- ⛔ **No sub-agents.**
- ⛔ **Poll background jobs to completion INSIDE a turn.** Never end a turn with detached work running.
- ⛔ **Never run a machine-wide kill.** Capture your own PIDs into `PIDS.txt`.
- ⭐ Write `LOG.md` continuously — it is your liveness signal. `START_UTC.txt` at the start; `END_UTC.txt` only when genuinely finished.
- ⭐ Commit your branch whole and bundle it before finishing: `git bundle create ~/clu/outputs/w17-<slug>/w17-<slug>.bundle w17/<slug>` then `git bundle verify`.
- ⭐ **Say what did NOT run.** "DECLARED NOT-DONE" with a reason is a complete answer.

## File ownership — ⛔ TWO SPOKES MUST NEVER WRITE THE SAME FILE
| file | sole owner |
|---|---|
| `VENUE.md` | venue-scout |
| `DRAFT-v2.tex` | draft-v2 (cite-tagger inserts `\cite{}` ONLY, after draft-v2 writes `END_UTC.txt`) |
| `DRAFT-v2-style.tex` | style-paraphrase (cite-tagger inserts `\cite{}` ONLY, after) |
| `CITATIONS.md` | citation-scout |
| `references.bib`, `CITATIONS-FOR-ZOTERO.md` | cite-tagger |
| `figures/` | plots |
| `figures/rendered/` | rendered-compositions |
| `REFEREE-v2.md` | referee-3 |

## ⭐⭐ AMENDMENT 1 (Head, 2026-09-22) — binds every spoke
Read `AMENDMENT-1.md`. In short: **wave 17 does NOT gate on any wave**, and ⛔ **no w15 or w16 number enters any draft or figure** — they are `\PLACEHOLDER{}` / marked-placeholder figures until the Head and the Advisor have discussed them. The draft is built on **w14-and-earlier evidence only**.
