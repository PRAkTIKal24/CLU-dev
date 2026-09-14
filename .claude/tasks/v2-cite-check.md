# v2-cite-check — web-scout (V2's bibliography + the erosion-novelty confirm pass)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 21; closes v2-referee-v07 MF-2 and the MF-5 residual, 2026-08-18).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Read-only; one report: `.claude/outputs/v2-cite-check.md`.

**DIAL DECLARATION: none — citation verification + literature confirmation; no performance claim; no laundering control applies.**

## Part 1 — the bibliography (MF-2: the draft cites ~25 works inline with NO reference list)
For every inline author-year citation in `papers/v2-short/draft.md` (sweep the full file; the referee's known set: Mo 2026 · Kong et al. (DOI owed) · HiPPO/Gu et al. · Jelassi et al. · EDEN · Titans · Csordás & Schmidhuber · Iqbal et al. 2026 · Rusch & Mishra 2021 — ⛔ used for BOTH coRNN (ICLR 2021) and UnICORNN (ICML 2021); disambiguate as 2021a/b · Golubitsky, Stewart & Schaeffer 1988 · Krupa 1990 · Rumberger 2001 · Guo — if present · the remainder you find):
1. Canonical record (venue, year, author order, DOI/URL) verified against publisher/arXiv/proceedings primary — never an aggregator alone; retrieval dates on every record.
2. A BibTeX entry per work in the Appendix-Q house pattern (caveats in the `note` field; never-copy traps recorded — author-order, venue-label, year/DOI mismatches).
3. Usage-check where the draft leans on a source's *content* (quote the sentence, quote the source's own words, verdict). Priority: the Mo claims (§3.2's law/estimator sentences), the CM-21 retirement citations (L.1–L.4), the Rusch & Mishra pair.
4. ⚠ Known trap from the program's own record (`venue-follow-up.md` escalation 3): the Wang/Mo-citer arXiv ID is **2606.24946**, not 2606.24945 — if the draft cites it, verify which.

## Part 2 — the erosion-novelty CONFIRM pass (the MF-5 residual)
`venue-follow-up.md` §3 (Jul 20) ruled the `sleep-erosion-study` claims: (a) PARTIAL — novel framing on a known substrate; (b) degeneracy-specificity demarcation NOVEL; (c) NOVEL — both **novel-by-absence, single-sourced**, with its own stated residual: *"confirm against continual-EBM + equilibrium-propagation before"* relying on them. Do that confirm pass now:
1. Search the continual-learning-EBM and equilibrium-propagation literatures (and anything since Jul 20) specifically for prior statements of claims (b) and (c) as §3 words them.
2. Verdict per claim: CONFIRMED-NOVEL (still absent, searched where) / PRIOR ART FOUND (cite it, quote it) / AMBIGUOUS (nearest neighbor + why it differs).
3. State your search coverage honestly — venues/terms searched, so absence is scoped, not asserted.

## Rules
- Per-file greps under `.claude/`; SSO-blocked records flagged with best-available public record labelled.
- ⛔ No draft edits — the revision spoke (`v2-revision-7`) consumes this report.
- A draft sentence whose content-lean fails its source is a HEADLINE finding.

## Acceptance criteria
1. Every inline citation in the draft has a section: verified record + BibTeX + usage-check or "identity-only, nothing to check".
2. Part 2's verdicts carry coverage statements; any prior art found includes the quote.
3. Standard `## Proposed handover updates` and `## Flags`.
