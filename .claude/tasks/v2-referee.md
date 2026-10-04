# Task: v2-referee — adversarial review of the V2 short draft + the F5 note (w7)

- **Agent:** `paper-referee` · **Output:** `.claude/outputs/v2-referee.md`
- **Read first (your agent def lists the standing docs):** protocol · Positioning Charter C-1…C-10 · `.claude/claims_matrix.md` (post-w6 version: CM-1 filled, CM-7/8 rewritten, CM-9…12 new) · `.claude/critique_register.md` · then the drafts: `.claude/papers/v2-short/draft.md` (canonical; .tex/.pdf built) and `.claude/papers/f5-note/f5-note.tex`.
- **Context you must weigh:** the draft was written BEFORE w6 results landed. Two known folds are missing by construction: (a) the **fit-gap-anatomy loan-curve numbers** (crossing ≈700 steps; 92% γ-recovery) — §3.4 currently declines to claim the crossing; it now CAN and should (CM-1 updated); (b) the **anchor-robustness envelope** (λ tradeoff, 3000-ep horizon, tilt-immunity-as-theory, anchor⊥volume-conservation) — §3.5/App B cite the older 2-seed cure. Treat these as MUST-FIX wiring notes (evidence exists, not new experiments).
- **Simulated venue:** NeurReps/ML4PS composite, 4–5 pp main text.

## Specific attack surface (beyond your standard passes)
1. **C-1 POLICY REVERSED (Head, 2026-07-07 — supersedes the charter text the draft was written under):** the §1 audit-disclosure paragraph must be **REMOVED** at revision (no defensive audit confession anywhere). Your job: flag the paragraph for removal, flag any OTHER self-deprecating audit framing elsewhere in the draft, and verify the draft still stands without it — i.e., every mechanism it uses is described in its current fixed form with exact flags, and no claim leans on the legacy paper's specific mechanism-numbers. Also check the F5 note reads as neutral class-level theory (never "our previous paper was wrong").
2. §3.2 Mo head-to-head — a hostile reviewer who has READ Mo: does our "overdamped face" claim survive his framing? (mo-deep-read §4 has the positioning; check the draft didn't dilute it.)
3. §3.3 CM-4 — the step-unit confound (map-applications vs wall-time) and the input-driven-RMSE gap: honestly stated or buried?
4. Erosion App B — every novelty sentence must still be hedged pending the Jul-11 scout; hunt unhedged ones.
5. F5 note — the [TODO-HEAD] items must not read as gaps a reviewer can exploit; check the Cor-3 footnote de-anonymization risk wording.
6. Figures are NOT embedded yet (writer flagged it) — list which figures are load-bearing enough that text-only fails review.

**Report:** verdict + itemized MUST-FIX/SHOULD-FIX/NICE + the three sentences a hostile reviewer would quote + missing-experiment list (should be ≈empty — flag anything that isn't in `.claude/outputs/`).
