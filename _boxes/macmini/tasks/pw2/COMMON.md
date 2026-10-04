# PW-2 — RULES EVERY SPOKE OBEYS (paper lead Hub, batmac, 2026-09-25)

Your brief is `~/clu/tasks/pw2/<slug>.md`. The Head's plan is `~/clu/tasks/pw2/PLAN.md`. Read **§2 in full**; the item text there is binding and verbatim. Where this file and the plan differ, THE PLAN WINS.

## Paths
- **Live drafts for this wave** (the Hub mirrored them from the Head's two Desktop folders, and installs them back after each spoke):
  - `~/clu/pw2/live/v2/` — `DRAFT-v2.tex`, `refs_clu_iclr.bib`, `figures/`, `iclr2027/`
  - `~/clu/pw2/live/style/` — `DRAFT-v2-style.tex`, the same bib, `figures/`, `iclr2027/`
  - Each folder is self-contained. Build FROM the folder: pdflatex, bibtex, then pdflatex until there is no "Rerun" warning.
- ⛔ **The bib is the Head's.** Never write, edit, reformat or regenerate `refs_clu_iclr.bib` (sha256 `b2cf715f25e020acf02038d5df4a2fd0ce816fb59e0aa3cfb101335684a71bd3`, 56 entries). Check its hash at the start and the end of your run.
- ⛔ **Never write** to `~/clu/outputs/w17-paper/`, `~/clu/outputs/pw1-*`, `~/clu/outputs/paper-visuals/`, any `outputs/w*/`, TABLES.md, the registry or the claims matrix. Evidence (read-only): `~/clu/evidence/outputs/`, `~/clu/stage/`, `~/clu/outputs/w*/`, and the PW-1 records `~/clu/outputs/pw1-*/`.
- Your output dir: `~/clu/outputs/pw2-<slug>/`. Worktree `~/clu/worktrees/CHLU-pw2-<slug>`, branch `pw2/<slug>`. Env: `cd` to your worktree, `source ~/clu/env_bm.sh`. The PW-1 draft tooling is in your tree: `pw1_draft_*/` (style diff-check, provenance checker, residual scan, page markers, index, word diff, compile). Reuse it, pointing it at the live folders.

## Non-negotiable on this machine
1. NEVER run `tmux kill-server`. That server holds this machine's keychain access.
2. Avoid `rm <glob>`. The dangerous-rm guard stalls you indefinitely. Use fresh timestamped subdirs; move instead of delete.
3. Never run a box-wide kill. Kill only your own PIDs. Other sessions come and go; they are none of your concern.
4. No sub-agents. Poll background jobs to completion inside a turn.
5. ⛔ No pre-registration, no bars, no labels, no predictions. Print everything. The Head decides what is quotable.
6. Zero runs, zero fits, zero GPU. If you need a number that is not in an artefact, print the gap and continue.
7. ⛔⛔ **THE NINE-PAGE CUT IS THE HEAD'S OWN PASS, AFTER PW-2.** No spoke trims, shortens for length, or moves material to the appendix for length. Moving numbers from prose into tables and captions (item 5) is not a length cut.

## The `.tex` lock
Before EVERY write to a live draft, run this and abort that write if it prints ABORT:
```bash
for g in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^(pw|w)[0-9]+-'); do
  [ "$g" = "$MY_SESSION" ] && continue
  case $g in *draft*|*figures*|*seeds-and-tables*|*placeholder*|*citation*|*referee*) echo "ABORT: $g live";; esac
done
```

## Draft-edit method (every edit)
1. Snapshot both live `.tex` files (sha256) into your out dir.
2. Stage the edit on SCRATCH copies of both whole folders (a fresh timestamped dir), and build there.
3. Check BOTH drafts:
   - exit 0; 0 undefined citations; 0 undefined references; 0 errors; overfull boxes listed;
   - the style diff-check between the two drafts EMPTY on placeholders, artefact paths and section order;
   - numbers: the MULTISET of numbers equal between the two drafts (numbers may move between text and tables in PW-2; compare the drafts with each other, not before with after);
   - provenance findings identical or improved;
   - `\PLACEHOLDER` / `\TODO` unchanged unless your item removes them (name each one).
4. Before installing, confirm the live files are still cmp-equal to your snapshot. Then install byte-identically into BOTH live folders (the `.tex` and the built PDF, named `DRAFT-v2.pdf` / `DRAFT-v2-style.pdf` inside each folder) and re-hash.
5. Every number that LEAVES the text keeps its artefact pointer in a `%` comment, or moves with its `% SRC` into the table or caption that now carries it.
6. **Lock-step:** both drafts receive the same structural edits at the same places. The style twin keeps its own paraphrased wording where the two differ.

## Head-held text — byte-identical after every pass
- The abstract's opening, and the first paragraph of each main section: Introduction, Methods (which the Head's item-3 text opens), Experiments, Results, Discussion.
- The abstract's maze `\TODO`.
- After item 2, `pw2-draft-sections` writes `~/clu/outputs/pw2-draft-sections/HEAD-HELD.md`, listing these paragraphs with their sha256 per draft. Every later spoke re-checks them before installing. item-3 and item-1 borrowings are the only permitted changes to them, and those spokes update the list.

## Vocabulary (PW-1 rules still bind)
n-CLU / n-CLU-REL; baseline; held-out combination; capsule; memory footprint; the engine's judge; `$10\times$ training steps`; no deletion-sense "exact"; none of the banned pipeline words; cross-references as `Sec.~\ref`, `Fig.~\ref`, `Tab.~\ref`, `App.~\ref`; citations as `text~\citep{}` / `text~\citet{}`.

## Finish
`START_UTC.txt` now; `LOG.md` continuously (the watcher reads its mtime); `FINDINGS.md` with each item done / partly / not done and why, before → after tables, and "What did NOT run". Commit your tooling (never outputs), bundle `pw2/<slug>` into your out dir, and verify the bundle. Write `END_UTC.txt` LAST. Pre-authorised in full: never wait for the Head, the Advisor or the Hub. Where a choice is open, take the option closest to an ICLR main-track paper, log it, and proceed.
