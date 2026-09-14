# v2-revision-7 — paper-writer (V2 → v0.8: the referee closures — page budget, bibliography, and the five blockers)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 21, 2026-08-18).** ⛔ **Mechanical precondition: `.claude/outputs/v2-cite-check.md` EXISTS on disk — do not start without it** (it supplies your bibliography and the novelty-wording verdicts). Read `.claude/AGENT_PROTOCOL.md`, then this file. You edit only `papers/v2-short/` (draft.md + draft.tex + CHANGELOG.md).

**DIAL DECLARATION: none — revision pass; no new measurement.**

## Inputs
1. `.claude/outputs/v2-referee-v07.md` — the findings (numbering below is its).
2. `.claude/outputs/v2-cite-check.md` — verified records + BibTeX + the Part-2 novelty verdicts.
3. `.claude/outputs/venue-follow-up.md` §3 — the Jul-20 erosion-novelty verdicts the draft must wire in.

## Close these
- **MF-1 (page budget: 5 pp → ≤ 4 pp)** per the referee's demotion menu, in its priority order: §4 ¶2 → 3 sentences + App L pointer · abstract trim (~30 %, parentheticals of (i)/(ii)/(iii) — numbers survive in §3) · §1's duplicated pointer sentence. ⛔ Do NOT thin §2's fine print (a)–(c) below one sentence each. State the resulting page estimate in the CHANGELOG; note that final pruning waits for the venue template.
- **MF-2 (bibliography)**: typeset the cite-check's verified BibTeX (references excluded from the 4-pp count); apply the Rusch & Mishra 2021a/b disambiguation at both §4 sites; fix any usage-check finding the cite-check reports.
- **MF-3 (μ²-orders collision)**: one reconciling clause in App K.4 naming the instrument (Jacobian-derived μ² floor of the B.8 γ-grid probe) with a cross-pointer to §3.3's Hessian number.
- **MF-4 (hermeticity leaks)**: H.3's "the paid-access companions" → "future work beyond this paper"; M§3.4's "the lattice-scale companions" → "deferred to future work at lattice scale"; the two "this program" wordings → "this paper"/"the theory note".
- **MF-5 (stale scout markers — now a WIRING fix)**: the scout RAN (`venue-follow-up.md` §3, Jul 20). Replace all three "pending scout confirmation" markers with the verdicts as ruled: claim (a) cited as novel-framing-on-known-substrate; (b)/(c) per the cite-check Part-2 confirm result — if CONFIRMED-NOVEL, state novelty with the coverage-scoped wording; if prior art was found, cite it and soften to the honest form. ⛔ Never claim unscoped novelty (CM-17's cite-don't-claim discipline extends here).
- **SF-1 … SF-8** as specified (abstract splice split · "≈1 %" → "≤3 % (median-consistent with its published 1.013)" · §2(c) reword to the tilt-as-dial fence · the 𝓗/H preamble line in M · Appendix-A cross-ref convention · agent branch names → bare commit hashes · the N51 line in App G · the one-clause symmetry-generalization future-work item (T²/SO(3)→SO(2), directions only, zero evidence claims)).
- **NICE N-1, N-2, N-4** at your judgment (N-1's "(= main-text Fig. 1)" inline notes recommended; N-2's "typically"; N-4's acknowledgement-placement note resolved to the double-blind-safe form).

## Guards
- The A.5 S1 sentence and its inline sign-off flag are NOT touched (colleague sign-off pending, tracked).
- CM-16a/b split · C-8 hermeticity now includes a SEMANTIC pass (companion/program-language, not just short names — the standing rule this revision's own history created) · never-quote sweep (per-file, positive controls, zero-hit list in the CHANGELOG entry) over the touched regions + the two files end-to-end once.
- Zero edits outside `papers/v2-short/`.

## Acceptance criteria
1. Every MF/SF item closed (with location) or explicitly deferred with reason; no silent drops.
2. Main text ≤ 4 pp estimated (stated); references+appendices excluded per venue rule.
3. The bibliography builds; both Rusch & Mishra sites disambiguated; no "pending scout" string remains.
4. Sweep printed; the semantic-hermeticity pass result stated ("companions/program/sibling/our other" class: 0 hits or each hit adjudicated).
