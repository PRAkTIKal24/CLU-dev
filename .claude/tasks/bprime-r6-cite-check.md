# bprime-r6-cite-check — web-scout

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 10), 2026-08-18.** Read `.claude/AGENT_PROTOCOL.md`, then this file. Read-only on the repo; you write exactly one report: `.claude/outputs/bprime-r6-cite-check.md`.

**DIAL DECLARATION: none — citation verification; no performance claim; no laundering control applies.**

## Why this exists
The r6 fold (`papers/bprime/draft-r6.md`, changelog item 10) added six citations declared **UNVERIFIED**: **Souza · Losing · Gomes (ARF) · river · UCI Metro Interstate Traffic Volume · Webb 2016**. None is in Appendix Q (the verified-BibTeX appendix). B′'s standing citation discipline (the r5 cite-fold precedent, `outputs/bprime-draft-r5-cite-fold.md`): every citation verified against its canonical source before a freeze; never-copy traps recorded.

## Deliverable — one report, per-citation sections
For each of the six: (a) the canonical bibliographic record (venue, year, authors in canonical order, DOI/URL), verified against the publisher/ACL-Anthology/arXiv/UCI page — never an aggregator alone; (b) a verified BibTeX entry in the Appendix-Q house pattern with its caveat note; (c) a check of what `draft-r6.md` *uses* the citation for (grep the draft for each cite site, quote the sentence) — does the cited source actually say it? Quote the source's own words where the draft leans on content; (d) flags: single-sourced records, author-order traps, venue-label traps (the \*SEM ≠ SemEval precedent), preprint-vs-published text divergences.

## Rules
1. ⛔ You never edit the draft or Appendix Q — the paper-writer folds your verified entries at the next revision; you report only.
2. Retrieval dates on every record; SSO-blocked or paywalled sources flagged as such with the best-available public record labelled accordingly.
3. Per-file greps only when searching `.claude/` paths (directory-level grep false-negatives on this machine).

## Acceptance criteria
1. Six sections, each with verified record + BibTeX + usage-check + flags (or an explicit "could not verify — here is what blocks it").
2. Any draft sentence whose content-lean on a source is NOT supported by the source's own text is a headline finding, not a footnote.
3. Standard `## Proposed handover updates` and `## Flags` sections.
