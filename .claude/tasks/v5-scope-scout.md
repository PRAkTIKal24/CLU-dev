# v5-scope-scout — web-scout (PALM's scope + the venue-native literature + V5's citation records)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 26; closes v5-referee-v02 SF-7/ME-7 + the V5 bibliography inputs, 2026-08-19).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Read-only; one report: `.claude/outputs/v5-scope-scout.md`.

**DIAL DECLARATION: none — venue/literature scouting + citation verification; no performance claim; no laundering control applies.**

## Part 1 — PALM's topic scope (the standing "known gap", Add.6/A5.7 Q7)
From PALM's own site (verbatim quotes + retrieval dates): the full CFP topic list · what "long-term memory for AI systems" spans in their own words · invited-speaker/organizer research areas (they predict the reviewer pool) · any stated preference for systems vs theory contributions · submission mechanics not yet banked (format/template, supplementary rules beyond the known code-anonymization clause, dual-submission language). ⛔ Facts only; fit judgments are the Advisor's/Head's.

## Part 2 — the venue-native literature brief (feeds V5 §4, which currently has ZERO venue-native citations)
A cited brief on the agent/LLM long-term-memory literature a PALM reviewer expects contact with: external-memory agents (MemGPT-class) · retrieval-augmented long-term stores · memory consolidation/forgetting policies in deployed systems · TTL/expiry and deletion practices in production memory systems · anything speaking to *controllable decay* or *verified deletion* in that stack (the two hooks V5 actually offers that audience). Per item: verified record + one sentence on what it claims + how V5 relates (contrast, not competition, where true). 8–15 works, quality over coverage.

## Part 3 — V5's citation records (the V6-style verification; V5 has ~30 inline cites and no bibliography)
1. **Already verified in `outputs/v2-cite-check.md` — do NOT re-verify, just carry forward:** Mo 2026 · Rusch & Mishra 2021a/b · HLW · Fischer & Igel 2010/2011 · Nijkamp 2020 · Golubitsky et al. · Krupa · Decelle 2021 · Agoritsas 2023 · Toledo-Marin 2025 · Kong 2024.
2. **Verify fresh (the deletion/unlearning + physics set):** Blelloch & Golovin FOCS'07 · Blelloch, Golovin & Vassilevska SWAT 2008 · Guo et al. 2020 · Ginart et al. 2019 · Sekhari et al. 2021 · Bourtoule et al. (SISA) 2021 · Micciancio STOC'97 · Naor & Teague STOC'01 · Hartline et al. 2005 · Buchbinder & Petrank · Minami & Hidaka 2018 + 2020 · Di Bernardo et al. 2025 · Iqbal et al. 2026 · Hinton et al. 1995 · Tieleman 2008 · Du & Mordatch 2019 · Hochreiter & Schmidhuber 1997 · Rusch et al. 2022 · SILO 2024 · PALL 2025 · Ticketed L-U COLT'23 · MUSE · CURE4Rec · anything else you find inline. Per work: canonical record + BibTeX (App-Q house pattern, caveats in notes, never-copy traps) + usage-check where V5 leans on content (priority: Guo §3 Eq.(1) · Ginart Def. A.5 · Sekhari Def. 3 · BG'07's actual claims · Minami & Hidaka's diffusive-NG result).
3. **SF-10's competing work:** the "recent preprint" in E.7/K.2 (the near-deletion result, gap 0.56 ± 0.21) — recover its arXiv id and the N168-compliant citation form ("an ICLR-2026 workshop paper (oral)", workshop name quarantined).

## Rules
Primary sources only (aggregators as corroboration); retrieval dates everywhere; SSO-blocks flagged with labelled fallbacks; a draft content-lean its source does not support is a HEADLINE finding; per-file greps under `.claude/`. Standard `## Proposed handover updates` + `## Flags`.
