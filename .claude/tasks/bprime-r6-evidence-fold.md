# bprime-r6-evidence-fold — paper-writer

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 8, Head ruling Q6 "update first, condense later", 2026-08-18).** Read `.claude/AGENT_PROTOCOL.md`, then this file. You write exactly two files: `.claude/papers/bprime/draft-r6.md` and a new entry in `.claude/papers/bprime/CHANGELOG.md`. Nothing else.

## Why this exists
B′ (`draft-r5.md`, the citation-clean fold of 2026-08-02) predates C2W8 and C2W10, which banked **three pieces of evidence that are exactly B′'s thesis** — before V6 (the 4–5 pp condensation) can be scoped, they must be folded in. r5 has had no referee pass (verified on disk); the referee pass will run on r6, after this fold.

## The three pieces (fold these; approved wordings are BINDING, from `claims_matrix.md` v2.17)
1. **The no-daylight row (N276, C2W8)** — tier-i thesis at a THIRD substrate. Approved form (matrix §0.12): *"branch (b), 0/9 cells, 3/3 arms, pre-registered at Q6 = 0.70 with nothing tuned; |A3b| ≤ 0.047 on 7/9 ⇒ the store and an 832-byte table are statistically indistinguishable on held-out stream reads — and the null is ATTRIBUTABLE (store not inert 9/9, geometry GO and gate validation both verified in advance)."*
2. **The two criterion-4 tripwires (N294 INSECTS · N295 Metro, C2W10)** — routed to B′ as FB4-class protocol evidence. Approved form (matrix §0.13): *"at laptop byte budgets an exemplar store at matched bytes sits at or above the strong-baseline frontier on both of the best-documented real streaming venues — and on one of them, destroying temporal order does not hurt it."* Framing rider, mandatory: *"a FALLBACK BEING RETIRED, ⛔ NOT a venue crisis."*
3. **The 24-h label embargo (N296, C2W10)** — independently useful benchmarking methodology: plain prequential at a 24 h horizon leaks up to 23 h of future traffic to any continuously-updated learner (**+10.9 %** to a 250-exemplar k-NN, **−0.3 %** to GBDT — asymmetric IN THE DIRECTION OF FIRING); seasonal-naive(`t−24 h`) is DEGENERATE at a 24 h horizon (the non-degenerate naive is `t−168 h`).

## Never-quotes that travel with piece 2 (matrix §0.13, all mandatory)
- ⛔ INSECTS band `b = 4` — terminal band persistence-trivial and at ceiling; `R(4)`/`A(4)` uninterpretable as registered.
- ⛔ "ARF is the reference" without its byte caveat — ARF's state is 9,542,925 B = 14.35× SAM-kNN's; byte-matched to nothing.
- ⛔ "river ships SAM-kNN" — it does not; every cost estimate citing a "one-line baseline" is VOID.
- ⛔ any `out-of-control` result — the stream has no data source; that leg is a DECLARED NOT-RUN.
- Metro tripwire fires against **NINE** references, never "8"; `MAE 300.09` belongs to the unregistered k = 10 anti-hobbling arm — the registered k = 5 arm is **306.76** (both beat gbdt_tuned 335.20).

## Rules
1. **r-convention**: the new file is `draft-r6.md` — never "v2"/"v6" naming for a revision.
2. Every new number carries its registry citation (N-number) and its riders verbatim; no number enters that does not trace to the registry or an `outputs/*` file.
3. Matrix v2.17 (and v2.16) are **PROPOSED, pending Hub confirmation** — treat their content as binding for restriction; if a fold decision would rely on a v2.16/v2.17 clause for *permission*, flag it in the changelog entry rather than assuming.
4. Placement is yours per the appendix-maximalism policy (main results main-text; protocol evidence and methodology to the appropriate FB4/protocol sections or appendices); the existing thesis sentence and headline findings (i)–(iii) are not weakened or restructured — this fold strengthens, it does not re-architect.
5. No toy-compositional claims are added anywhere (Add.16 §A44.1 demotion; a separate sweep is auditing the estate for these).
6. Anonymization discipline unchanged: third-person self-citation, placeholders intact.

## Acceptance criteria
1. All three pieces appear, each with its approved wording and full rider set.
2. A diff summary in the CHANGELOG entry: every touched section listed, with what changed and the registry source per change.
3. Sweep printed in the changelog entry: the new content checked against matrix §0.9–§0.14 never-quotes (per-file sweep, never a directory-level grep over `.claude/` — it false-negatives on this machine).
4. Zero edits outside `draft-r6.md` + `CHANGELOG.md`; `draft-r5.md` is untouched (it remains the frozen citation-clean reference).
