# Task: v2-referee-2 — clean-pass re-review of the revised V2 short + F5 note (w10)

- **Agent:** `paper-referee` · **Output:** `.claude/outputs/v2-referee-2.md`
- **Read first:** protocol · Charter · `.claude/claims_matrix.md` v1.4 (CM-1, CM-4 amended, CM-6 amended) · your prior report `.claude/outputs/v2-referee.md` (the MF-1…5/SF-1…6 punch-list — verify each is CLOSED) · then the revised drafts: `.claude/papers/v2-short/draft.md` (v0.3, 5 figs) + `.claude/papers/f5-note/f5-note.tex` (v0.2, arXiv-clean).
- **Context:** v2-revision + v2-revision-2 applied all 5 MUST-FIX and folded SF-1/2/3. This is the clean-pass: confirm the fixes hold, hunt any NEW inconsistency the edits introduced, and give the submission-readiness verdict.

## Specific checks (beyond re-verifying MF/SF closure)
1. **MF-3 residual (the contradiction you caught):** confirm §3.4↔App D emergent-lifetime numbers now reconcile — the writer resolved by attribution (distinct probes, +12–15% envelope / per-seed +5 to +29%). Verify no cross-section reader can still construct the contradiction.
2. **NEW — the "5×" tension (writer-flagged, unresolved):** the abstract/§3.2 say retention misprediction "up to 5× underdamped" but the SF-1 tabulated ratios bottom at 0.30/0.31 (≈3.3× at δ=4, deepest of 10 rows). Either 5× is the extreme-δ value from the full 14-point sweep (censored rows) or it is stale. **Check the draft's own §3.2/App against v2-referee-experiments SF-1 (14 δ rows) — is "5×" traceable?** If not, this is a MUST-FIX number-hygiene item (a reviewer sees 0.30↔3.3× next to "5×").
3. **SF-2 4×-retirement:** confirm the qualitative triad now leads §3.3 and the compute-inversion is stated honestly (not buried); no residual sentence still claims "4× longer" as a compute win.
4. **F5 note arXiv-readiness:** confirm zero `\todohead`/`[TODO-HEAD]` render in the PDF; Cor-3 footnote reduced to the pure class fact (de-anon clause CUT per Head DEC); count reconciled ("all results" not "twelve").
5. Standard passes (charter/matrix/attack) on the net draft.

**Report:** verdict (is it submission-ready modulo pruning + the F5 arXiv id?) + any residual MUST/SHOULD + the "5×" determination + the three hostile-reviewer sentences. If clean, say so plainly — this draft should be near-final.
