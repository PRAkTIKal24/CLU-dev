# bprime-dividend-family-2 — experiment-engineer

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 14; Head greenlight under the quality-first posture, 2026-08-18).** Read `.claude/AGENT_PROTOCOL.md`, then this file.

**DIAL DECLARATION: no new dial.** This task extends the B′ audit with a second task family; every performance number it produces is governed by the audit protocol's own control battery (matched-bytes table launder · +0 B substitute audit · same-keys null · blank-store control · rescue gate). Laundering control = the protocol itself. Falsifies: a dividend claimed without its launder column. Does-not-falsify: the family failing protocol validation (that is a reportable outcome, see rule 3).

## Why this exists
The r6 referee ranked this the program's highest-leverage missing experiment: `draft-r6.md` §6 L1 calls a second dividend family *"the cheapest thing that would strengthen this work"*, and post-R.2 (both real streaming venues retired as dividend venues; a third attempt is a registered stop) it is **the only remaining route to audit width**. It directly blunts two of the three hostile-reviewer quotes (the "expected to win" construction note; the "thin cross-family audit" self-description).

## Read before designing anything
1. `papers/bprime/draft-r6.md` **Appendix N (all of it, N.5's four family-construction rules especially)** + Appendix I (the registration-discipline table) + §2 (the protocol) + Appendix P (the audited cell's spec).
2. `PREREG-Bprime.md` + the F3 tuning-grid protocol (`outputs/bprime-rivals-f3.md`) — the tuning symmetry is mandatory: every rival arm gets the full grid, ours too.
3. The banked audit harness (locate via `outputs/bprime-rivals*` provenance; reuse, do not rebuild).

## Deliverables, in order
1. **PREREG FIRST — written, committed, and timestamped BEFORE any arm trains.** In `.claude/outputs/bprime-dividend-family-2.md` §1: the family definition with an explicit check against each of N.5's four rules; the protocol-validation gate it must pass before scoring; numeric falsifiers (what dividend > 0 looks like, at what SE bar); kill-conditions; seed plan (**n = 9 every arm, ours included**); the byte ledger convention per arm (dtype declared per row).
2. **The family implementation** on a scoped branch (`agent/experiment-engineer/bprime-dividend-family-2`), with tests, suite green before any run.
3. **The full audit run:** all six rival arms (ttt_linear · ttt_mlp · deltanet · gdn · gdn2 · mamba2) + the CLU + the complete control battery, full F3 grid, n = 9. CPU-scale per the audit's class; no CSF3.
4. **The report** (§2 onward of the same output file): per-arm tables in the registry-rider format (modal-value rule on byte figures · selection-stability labels · "reads no worse than" discipline for sub-2-SE margins), raw artifacts paths, and a one-page summary the Advisor can verify against the registries.

## Rules
1. Prereg precedes results in the file and in git history — two commits minimum, prereg first; a falsifier written after a result is void.
2. If the family FAILS protocol validation, that is the deliverable — report it with the failure mechanism. ⛔ No silent family swap: a second design attempt requires its own prereg section, dated, with the first attempt's failure kept in the report.
3. ⛔ No toy-compositional substrate (Add.16 §A44.1); the family is a fresh synthetic per N.5, not a revival of a retired gym.
4. One engineer worktree, declared; commits to the scoped branch only; ⛔ never push `origin`.
5. Every number in the report traces to an artifact path. No draft edits — the paper-writer folds on a later pass.

## Acceptance criteria
1. Prereg committed before any training artifact exists (git timestamps show it).
2. 9 seeds on every arm incl. ours; byte ledger per arm with dtype; F3 grid symmetric.
3. The launder column sits beside every dividend number; the rescue gate is scored for every arm including the CLU.
4. Suite green on the branch; standard `## Proposed handover updates` and `## Flags` sections.
