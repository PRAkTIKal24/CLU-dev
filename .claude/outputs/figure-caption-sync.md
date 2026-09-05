# figure-caption-sync — paper-writer report

Task + acceptance criterion: rewrite every caption on the consolidated caption-edit list so it describes the figure that actually ships; rebuild; page splits unchanged.
Status: **BLOCKED — mechanical precondition not met.** Zero files edited.

**DIAL DECLARATION (echoed): none — caption-only editorial pass; zero number changes, zero prose changes outside caption bodies. Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.**

**⚠ THIS REPORT CONTAINS A DOWNSTREAM RECONCILIATION LIST AND IT NEEDS AN OWNER (protocol §5 corollary).** Three items, all for the Hub, all in §6: (a) the `v2-neurreps-descoped` variant is **missing from the render pass's owed-edit list** and carries the same defects; (b) `pdflatex` is not on `PATH` on this machine (it is at `/Library/TeX/texbin/pdflatex`) — any task whose "Then" step is a rebuild must say so; (c) the task file's paths are wrong by one directory level (`papers/…` → `.claude/papers/…`).

---

## 1. The gate is closed. `.claude/outputs/figure-layout-fix.md` does not exist.

My task file opens with `⛔ Mechanical precondition: .claude/outputs/figure-layout-fix.md exists (its consolidated list is your worklist)`. The Head's spawn line likewise said *"← after the layout fix lands"*. It has not landed. Four independent checks agree:

| check | command | result |
|---|---|---|
| output file | `ls .claude/outputs/figure-layout-fix.md` | **No such file** (render-pass output is present, 10:36) |
| task file | `ls .claude/tasks/figure-layout-fix.md` | present, 08:01 — task was *written*, not *executed* |
| branch | `git branch -a \| grep -iE 'layout\|figure\|caption'` | **no matching branch**; `git worktree list` shows 5 worktrees, none for this slug |
| the artifacts themselves | see below | still at render-pass state |

The artifacts are the decisive check. The layout pass is mandated to re-lay-out three figures and change `\includegraphics` widths. Neither has happened:

- `.claude/papers/palm-variant/v5/figs/*.png` mtimes are **07:45** (render pass), V2's **07:45**, descoped's **07:20**. No figure regenerated since.
- V5 widths are still `0.60\linewidth` (Fig 1, l.79), `0.74` (Fig 2, l.202), `0.80` (Fig C.2, l.261) — i.e. exactly the footprints the render pass flagged as **UNREACHABLE** for legibility, and exactly what the layout pass exists to change.
- The captions are untouched: l.203 still names `I-J`/`I-R3`/`I-R1`; l.262 still opens *"panels are labelled by the pre-registered prediction each tests"*.

**Liveness note (per standing lesson: verify liveness before recovery).** `ps` shows several live agent processes, so a layout-fix agent may be running right now on another thread. I did **not** treat this as a dead-spoke recovery and did not attempt to do the layout work myself — out of scope, and the task explicitly reserves it.

## 2. Why I did not proceed anyway — proceeding would produce *wrong* captions, not merely premature ones

This is the substantive finding, and it is why "close the layout-independent subset now" is also the wrong call.

Seven of the nine owed edits are on **V5 Fig 2 and V5 Fig C.2** — precisely the two figures the layout task orders re-laid-out taller *"with legends restored and every in-figure numeric label back"* (its rule 1, appendix figures, "this is free"). But the render pass wrote those edits **because those legends and labels had been deleted**. Restore them and the edits invert.

| # | figure | owed edit (render pass §7) | why it was owed | fate once the layout pass runs |
|---|---|---|---|---|
| 1 | V5 F2 | replace `I-J`/`I-R3`/`I-R1` with plain-English instrument names | internal IDs stripped | **survives** (label hygiene is carried forward) — but figure *number* may shift, see §3 |
| 2 | V5 F2(a) | add units to `n_{1/2}` | — | **survives**; render pass itself says no action needed (axis already reads "(steps)") |
| 3 | V5 FC.2 | delete *"panels are labelled by the pre-registered prediction each tests"* | Q-labels stripped | **survives** (prereg item names stay banned) |
| 4 | V5 FC.2(a) | add "circles $T{=}4{\times}10^{-3}$, squares $T{=}8{\times}10^{-3}$" | **legend deleted** | ⚠ **VOIDS** — would duplicate a restored legend |
| 5 | V5 FC.2(c) | add red/grey/blue colour key | **legend deleted** | ⚠ **VOIDS** — same |
| 6 | V5 FC.2(c) | **drop** "and hop fraction" | **per-bar labels no longer fit** | ⚠ **INVERTS** — labels return, so the caption should *keep* "and hop fraction" |
| 7 | V5 FC.2(d) | add red/blue key **and** state $T{=}4{\times}10^{-3}$ | legend deleted / — | ⚠ **half voids** (key), half survives (the $T$ statement) |
| 8 | V2 F3(a) | add open-circle/filled-square marker key | **legend deleted** | ⚠ **conditional** — layout task orders a re-check of V2's three figures at the de-scoped widths; if F3 is re-laid-out, this voids |
| 9 | V2 F3(c,d) | none — numbers live in prose | — | no action either way |

**Score: of nine owed edits, three-and-a-half are layout-independent (1, 2, 3, half of 7); four-and-a-half either void or invert (4, 5, 6, half of 7, and conditionally 8).** Writing them today would ship a V5 Appendix B whose Fig C.2 caption spells out three colour keys that duplicate three restored legends, and which has *deleted* the phrase "and hop fraction" from a figure that by then displays hop fractions on its bars. That is a worse defect than the one I was sent to fix, and it is the kind a reviewer meets immediately.

## 3. The layout pass also *adds* caption work that cannot be guessed

Its rule 2 splits **V5 Fig 1**: the main text keeps a single money panel (the collapse), and *"the complete multi-panel version moves to the appendix at full size under a new figure number."* That implies, none of which exists yet:

- a **rewritten main-text Fig 1 caption** — currently a multi-panel caption for what will become one panel;
- a **brand-new caption** for the split-out appendix figure;
- **renumbering** of every V5 figure after the insertion point, hence new `\ref` targets — the one category my task explicitly permits me to touch outside caption bodies ("except where a caption's change requires a matching in-text figure reference"), and which I cannot resolve without knowing where the new figure lands.

My worklist is by construction the layout pass's *consolidated* list ("yours plus the nine already listed"). The "yours" half is empty and unknowable today.

## 4. Page-split baselines recorded (so the eventual pass has a control)

Measured now with `pdfinfo`, from the currently-built PDFs:

| variant | PDF mtime | **total pages** |
|---|---|---|
| `.claude/papers/palm-variant/v5/submission.pdf` | 07:45 | **9** |
| `.claude/papers/neurreps-variants/v2/submission.pdf` | 07:48 | **13** |
| `.claude/papers/v2-neurreps-descoped/submission.pdf` | 07:52 | **14** |

V5's 9 pp total and V2's 13 pp total match the render pass's reported splits (V5 4.00 pp main / 9 total; V2 5.67 pp main / 13 total), confirming these PDFs are the render-pass builds and nothing has moved since. **Note the constraint interaction:** my task says V5 must stay 4.00/9 and orders me to shorten a caption rather than the paper if it slips — but the *layout* task is separately allowed to change that split (it is told to "report and stop" if the reduced main figure moves 4.00/9). Measuring a caption-induced page delta against a baseline that the layout pass may itself have moved is meaningless. The caption pass must be measured against the **post-layout** build, which is another reason it must run second.

## 5. What I did *not* do

No file was created or modified anywhere outside this report. Specifically: no `.tex` edited, no figure regenerated, no build run, no git branch created, no commit. **Git footprint: none.** (The target files live under `.claude/papers/**`, which is gitignored and untracked — see §6c — so this pass has no git dimension at all, and protocol §3 does not bind it. Worth the Hub knowing: the shared checkout is currently on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` at `7fcef50`, working tree clean.)

## 6. Three defects found in the task chain itself (the reconciliation list)

**(a) The `v2-neurreps-descoped` variant is missing from the owed-edit list.** My task names three files; the render pass's §7 list covers only two (`palm-variant/v5`, `neurreps-variants/v2`). The reason is chronological: the render pass finished at 10:36 but measured figures installed at 07:45, whereas `v2-neurreps-descope` produced the descoped variant separately (output 07:53, figures 07:20). The descoped `submission.tex` carries **its own copy of the V2 Fig 3 caption** — at l.349 `\includegraphics[width=0.60\textwidth]{figs/fig3_gmor_condensate.png}`, with the caption body materially identical to `neurreps-variants/v2` l.340 (verified: same panel (a)–(d) structure, same "slope $-1.05$", same closing "therefore verification" rider). **So owed edit #8 applies twice, and only one instance is listed.** Whoever writes the consolidated list must key it by *all three* files. I confirmed the descoped file contains no V5-specific stale strings (`I-J`/`I-R1`/`I-R3`/"pre-registered prediction each tests" → no matches), as expected for a V2 paper.

**(b) `pdflatex` is not on `PATH`.** `which pdflatex` → not found. It exists at `/Library/TeX/texbin/pdflatex` (and `/usr/local/texlive/2026/bin/universal-darwin/pdflatex`), consistent with the render pass's "TeX Live 2026 / pdfTeX 1.40.29". Any task whose acceptance criterion includes "rebuild ×3 and report the page split" needs that path stated, or it reads as a blocked toolchain. `mutool` and `pdfinfo` *are* on `PATH`.

**(c) The task file's paths are off by one directory.** It names `papers/palm-variant/v5/submission.tex`; the repo has no `papers/` at root. The real location is **`.claude/papers/palm-variant/v5/submission.tex`** (likewise `.claude/papers/neurreps-variants/v2/`, `.claude/papers/v2-neurreps-descoped/`). This cost a round of searching and could cause a less careful agent to report the files as deleted. Same correction applies to the layout task's `papers/v2-short/**` / `papers/v5-short/**` untouchables.

## 7. Reusable instrument already banked

`.claude/outputs/figure-render-pass/` contains `pagesplit.py`, `tap.py` (the data-tap comparator), `size.py`, and the six `new_*.py` regenerators. The layout pass will edit the `new_*.py` files; the caption pass should reuse `pagesplit.py` unchanged for acceptance criterion 3. No scratch directory was needed for this report, so none was created.

## 8. Recommendation

**Re-spawn this task unchanged once `.claude/outputs/figure-layout-fix.md` exists**, with three amendments: fix the paths to `.claude/papers/…`; add the `/Library/TeX/texbin` note; and require the consolidated list to cover **all three** variants including the descoped one. The gate did its job here — it named a file, the file was absent, and the absence was detectable in seconds. That is the mechanism working as designed.

Open questions / follow-ups / risks:
- **Role mismatch, flagged not resolved.** I am configured as **physics-theorist**; this is a `paper-writer` task (caption editorial, zero derivation). I executed it as briefed since the task is model-agnostic and the outcome is a gate check, but the Head should spawn the re-run on `paper-writer` — the surviving work is prose judgement inside `\caption{}` bodies, including verbatim-preserved approved wordings and mandatory riders, which is that spoke's competence, not mine.
- **Risk if a layout agent is live right now:** `.claude/papers/**` is untracked, so two agents writing those `.tex` files have **no git-level protection** — last writer wins silently. This pass touched nothing, so no collision occurred, but the Hub should serialise all `.claude/papers/**` writers explicitly rather than relying on branch discipline that does not apply there.
- I could **not** verify acceptance criterion 1's check ("no caption names a label absent from its figure") against the *shipped* figures, because the shipped figures are about to be replaced. The check itself should be run post-layout; a PNG-side text sweep (render pass §9 did an identifying-string sweep) is the natural instrument.

## Proposed handover updates (for the Hub)

- **§7 (discrepancies):** record that the render pass's nine owed caption edits are **not all still valid** — items 4, 5, 6 and half of 7 are artefacts of legend/label deletions that `figure-layout-fix` is mandated to reverse, and item 6 *inverts* (the caption should keep "and hop fraction", not drop it). The consolidated list must be re-derived post-layout, not concatenated.
- **§7:** add that `v2-neurreps-descoped/submission.tex` carries a second copy of the V2 Fig 3 caption and is absent from the render pass's owed-edit list — owed edit #8 applies twice.
- **§8 (open questions):** V5 Fig 1's main/appendix split creates an unnumbered new appendix figure and renumbers everything after it; the caption pass cannot resolve `\ref` targets until that number is fixed. Flag as a dependency, not a defect.
- **Environment facts (protocol §4):** `pdflatex` is **not on `PATH`** — use `/Library/TeX/texbin/pdflatex`. Worth adding alongside the existing "no `timeout` binary on this macOS" note.
- **Paths:** `.claude/papers/**` (not `papers/**`) is where all six paper variants live, and it is **gitignored/untracked** — so parallel writers there get no branch isolation and must be serialised by the Hub.
