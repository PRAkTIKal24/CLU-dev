# bprime-s-remeasure — results-analyst

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 14; Head greenlight, 2026-08-18).** Read `.claude/AGENT_PROTOCOL.md`, then this file. You write one report: `.claude/outputs/bprime-s-remeasure.md`. No model-code changes.

**DIAL DECLARATION: none — instrument correction on banked artifacts.** One measured quantity (`s`) is re-estimated under the registered convention; no arm is retrained, no dividend is re-scored. Falsifies: the pre-registered direction failing (see below — if it fails, that is the headline).

## Why this exists
Matrix §0.9 (HUB-CONFIRMED at v2.11; N224): the effective-`s` estimator must subtract `α‖q‖²`; without it it reads ~1.44× high (0.438 vs 0.304 on the reference cell). `bprime-c6`'s **`s = 0.40` is FLAGGED FOR A CHECK, NOT REFUTED**, and the re-measurement has been a declared NOT-RUN since C2W5. The r6 referee made the missing rider MUST-FIX (MF-3; the rider itself lands in r7) — this task closes the *experimental* half. **Direction pre-registered by the registry: the correction makes `s` SMALLER and `d/s` LARGER**, i.e. it strengthens the draft's `exp(−½(d/s)²)` suppression claim.

## Do
1. Locate `bprime-c6`'s banked store/artifacts (`outputs/bprime-c6*`) and the estimator that produced `s = 0.40`; identify its subtraction status from code/artifacts, not from memory.
2. Re-estimate `s` with the `α‖q‖²` subtraction applied, on the same banked cell(s) — same seeds, same store, nothing retrained. Report BOTH conventions side by side, per seed, with the estimator spelled out.
3. Recompute `d/s` and the suppression-fit statement (`exp(−½(d/s)²)`, the R² = 0.995 claim's inputs) under the corrected `s`; state plainly whether every downstream claim moves in the strengthening direction, and by how much.
4. If the measured direction CONTRADICTS the pre-registered direction (s larger, or d/s smaller), that is the report's first line — not a footnote — and no draft language is proposed.
5. Close with: (a) proposed N224-discharge wording for the Hub (the registry disposition is the Hub's to make, you propose only); (b) the exact numbers r7's MF-3 rider should carry if the Head wants them updated beyond the flag.

## Rules
- Read-only on the repo except your report; per-file greps under `.claude/` (directory-level grep false-negatives).
- Every number traces to an artifact path or a command reproduced in the report.
- Standard `## Proposed handover updates` and `## Flags` sections.
