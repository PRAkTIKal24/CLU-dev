# PW-1 — RULES EVERY SPOKE OBEYS (paper lead Hub, batmac, 2026-09-25)

Your brief is `~/clu/tasks/pw1/<slug>.md`. The Head's plan is `~/clu/tasks/pw1/PLAN.md`. Read §1 and your items IN FULL; the item text there is binding and verbatim. Where this file and the plan differ, THE PLAN WINS.

## Paths
- Drafts: `~/clu/outputs/w17-paper/DRAFT-v2.tex` and `DRAFT-v2-style.tex` (untracked, single-writer). Figures: `~/clu/outputs/w17-paper/figures/` and `~/clu/outputs/paper-visuals/`.
- Figure pipeline source of truth: `w21_figures/` in your worktree (branch lineage `w22/figures3`, cb5e0ac). The in-folder copies in `paper-visuals/` are INSTALL TARGETS; edit the repo copy, never the installed copy.
- Evidence (read-only): `~/clu/evidence/outputs/`, `~/clu/stage/`, earlier `~/clu/outputs/w*/` directories. ⛔ Never write to any `outputs/w*/` directory, `TABLES.md`, the negatives registry or the claims matrix.
- Your output dir: `~/clu/outputs/pw1-<slug>/` (the ONLY place you write reports). Your worktree: `~/clu/worktrees/CHLU-pw1-<slug>`, branch `pw1/<slug>`.
- Env: `cd` to your worktree, `source ~/clu/env_bm.sh`. TeX is on PATH via that file.

## Non-negotiable on this machine
1. NEVER run `tmux kill-server`. That server holds this machine's keychain access. Killing it logs out every agent here.
2. Avoid `rm <glob>`. The dangerous-rm guard is NOT covered by bypassPermissions and will stall you indefinitely. Write into fresh timestamped subdirs instead.
3. Never run a box-wide kill (`pkill python`, `killall`). Capture your own PIDs and kill only those. Other agents' sessions (`w*`, other `pw*`) will appear while you work: that is normal and none of your concern. Do not inspect, touch or report them.
4. Memory binds (16 GB). Before any heavy step (model load, rendering), check free memory (`vm_stat`). If it is short, wait in a loop inside your turn rather than fail.
5. No sub-agents. Poll background jobs to completion INSIDE a turn. Do not end a turn while detached work is running.
6. ⛔ PRE-REGISTRATIONS ARE BANNED. No PREREG file, no bars, no WIN/COMPARABLE/LOSS labels, and no predictions before a result. Earlier waves' `PREREG-*.md` files are an anti-pattern: do not copy them. Produce and print everything, including numbers that look bad. The Head decides what is quotable.
7. Zero runs, zero fits, zero GPU. You read artefacts. If you need a number that is not in an artefact, print the gap in FINDINGS and continue. Never commission a measurement.

## The `.tex` lock (only for spokes that edit a draft)
Before EVERY write to either draft, run this, and abort that write if it prints ABORT:
```bash
for g in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^(pw|w)[0-9]+-'); do
  [ "$g" = "$MY_SESSION" ] && continue
  case $g in *draft*|*figures*|*seeds-and-tables*|*placeholder*|*citation*|*referee*) echo "ABORT: $g live";; esac
done
```
(`MY_SESSION` is your own session name, given in your launch prompt.)

## Draft-edit method (every draft edit)
1. Snapshot both live drafts (sha256) into your output dir.
2. Stage the edit on SCRATCH copies of the whole `w17-paper/` tree (a fresh timestamped dir). Build FROM that tree: `\bibliographystyle` hard-codes `iclr2027/`, and a build elsewhere silently produces an EMPTY bibliography. Build = pdflatex, bibtex, pdflatex, pdflatex.
3. Check, for BOTH drafts: exit 0; 0 undefined citations; 0 undefined references; the style diff-check between DRAFT-v2 and DRAFT-v2-style is EMPTY on numbers, placeholders, artefact paths and section order (the earlier tool is `STYLE-DIFF-CHECK*.md` in `w17-paper/` and in the `w18-*`/`w22-*` spokes' branches); the provenance findings are identical or improved; the count and order of `\PLACEHOLDER` are unchanged unless your item removes them (then list each one removed).
4. Before installing, confirm the live drafts are still cmp-equal to your snapshot. Then install byte-identically into BOTH drafts (and the PDFs), and re-hash at the destination.
5. Every number that LEAVES the paper keeps its artefact pointer in a LaTeX `%` comment.

## Figure method (every figure you build)
Build twice into scratch dirs. The two builds must be byte-identical. Write PROVENANCE (sources and sha256 per figure) and SHA256 manifests. Load the `dataviz` skill before touching any plot.

## Vocabulary: every word you write must already obey it
Plan items 2, 3 and 13 and the "pendings resolved" block: n-CLU / n-CLU-REL; baseline (never rival); `$10\times$ training steps`; held-out combination (never corner); capsule (never oblong); memory footprint (never ledger); the engine's judge; no "exact" in the deletion sense; no economic metaphors; none of the banned pipeline words (store, gate, artifact/artefact, pre-registered, A1, A10, C-512, improved, config, spoke, hub, wave, charter, ledger row, scored run, budget). This applies to captions, legends, axis labels, table headers and prose. Your own FINDINGS are exempt.

## Held by the Head: do not touch
The abstract's maze sentence stays `\TODO`. Referee M3's deletion wording as a Head decision beyond item 13, MF-11 (length/trim) and MF-12 (venue items, AI-use statement) are also held. Apply only the plan's items.

## Finish
- `START_UTC.txt` now. Write `LOG.md` continuously, because the Hub's watcher reads its mtime.
- `FINDINGS.md`: each assigned item marked done / partly / not done and why; before→after tables with counts; hashes; and a section **"What did NOT run"**.
- Commit the branch WHOLE (code and tools, never outputs) and `git bundle create ~/clu/outputs/pw1-<slug>/pw1-<slug>.bundle pw1/<slug>`, then `git bundle verify` it. Verify outputs by directory listing, not `git status`.
- Write `END_UTC.txt` LAST, only when genuinely finished. Then stop.
- Pre-authorised in full: never wait for the Head, the Advisor or the Hub. Where a choice is open, take the option closest to an ICLR main-track paper, record it in LOG.md, and proceed.
