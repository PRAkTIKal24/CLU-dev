# Task: v2-revision-3 — close the two number-hygiene MUST-FIX from the clean pass (w11, small)

- **Agent:** `paper-writer` · **Output:** report to `.claude/outputs/v2-revision-3.md`; edit `.claude/papers/v2-short/` in place (CHANGELOG v0.4).
- **Read first:** protocol · `.claude/outputs/v2-referee-2.md` (the two MUST-FIX + SF items — this task closes them) · claims matrix v1.5 (CM-4 now carries the misprediction canonical: ≈3.2× trained / ≈5× exact-map-only).
- **Context:** the clean pass returned weak-accept, submission-ready modulo two wording/label fixes (both M4-class, NO experiments) + two trivial SF items.

## Items
1. **MF2-A (the "5×"):** the trained-model misprediction bottoms at **≈3.2×** (meas/pred 0.31 at δ=4, deepest breaking; the full 14-δ sweep goes no deeper on the underdamped side). Change "up to 5×" in the abstract AND §3.2 to **"≈3.2× at the deepest trained-model breaking; the exact analytic map continues to ≈5× further underdamped (theory note, check k)"** — the referee's recommended option (b), coherent with the F5 note where the genuine 5× lives (γ=0.1). Per CM-4 this is not a canonical constant; change freely.
2. **MF2-B (the 15×/92% splice):** three label/caveat fixes, no runs — (i) label the §3.4 table CLU "**CLU (γ=0 conservative)**"; (ii) add one sentence: the App-G recovery ladder is a **separate wake-only experiment** (`fit-gap-anatomy`, seeds 0–2, `9a13455`); its absolute MSE scale and twin (0.0047) are **not cross-comparable** with the §3.4 `minus-the-physics` table (twin 0.0128, seeds 42–44) — only within-table gaps are meaningful; (iii) reword the abstract to "recovers 92% of the **absolute** contraction-forbidden fit gap (the unit still trails the twin ~5× by ratio but is bounded by construction)."
3. **SF2-1** (F5 note): reconcile "all 14 checks" → "13" (rows a–m), or delete the number (it's in the strip-on-arXiv block, but close it).
4. **SF2-2:** abstract "≈57–69" → "≈56–69" (LEM 56.4, don't round up).
5. Rebuild both PDFs. (Pruning SF2-3 is deferred to the dedicated pruning pass, not this task.)

**Acceptance:** both MUST-FIX closed with the exact referee-recommended wordings; PDFs build; diff-summary. This should make V2 clean-weak-accept, submission-ready modulo pruning + the F5 arXiv id.
