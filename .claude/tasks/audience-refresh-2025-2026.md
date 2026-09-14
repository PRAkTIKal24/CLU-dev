# audience-refresh-2025-2026 — web-scout (NeurReps AND PALM audience profiles built on 2025–2026 data)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 46; Head directive 2026-08-21 — "we want both audiences for neurreps and PALM according to 2026 and 2025 data").** Read `.claude/AGENT_PROTOCOL.md`, then this file. Read-only; one report: `.claude/outputs/audience-refresh-2025-2026.md`.

**DIAL DECLARATION: none — audience/literature scouting + citation verification; no performance claim; no laundering control applies.**

## Why this exists
Both audience profiles currently rest on the wrong years. **NeurReps:** `outputs/neurreps-audience-scout.md` verified the 2026 CFP and two PMLR volumes — **v197 (2022)** and **v228 (2023, published 2024)** — but the **2025 edition's accepted-paper list was NOT retrievable** (`neurreps.org/accepted-submissions` returned 404) and 2024 is unindexed. So the audience's *recent* work is a gap. **PALM:** its 2026 CFP, topics, organizers and invited speakers are verified (`outputs/v5-scope-scout.md`), but we have no picture of what this community's papers actually looked like in 2025. The Head wants both profiles on **2025 + 2026** data.

## Part 1 — NeurReps 2025 (and 2024), the gap the last scout declared
Recover the accepted-paper lists. The prior pass tried the workshop site alone; try every surface before declaring failure: **OpenReview venue/group pages for the 2024 and 2025 editions** · the **NeurIPS 2025 workshop schedule page** · `web.archive.org` snapshots of `neurreps.org/accepted-submissions` and the site's 2024/2025 pages · DBLP and Semantic Scholar workshop indices · PMLR (in case a v3/v4 volume exists under a title the previous search missed) · the organizers' own listings.
Deliver: per edition, the accepted-paper list (or an honest "unavailable, surfaces tried"), and from whatever is recovered, **the topic distribution** — how many papers on equivariant architectures, on representational geometry in neural data, on dynamics, on topology, and so on — with the classification stated as your inference.

## Part 2 — PALM's lineage and its 2025 proxy
1. **Establish whether NeurIPS-2026 PALM is a first edition.** If a 2025 predecessor exists (same organizers, or the same name at another venue), get its accepted papers.
2. If it is a first edition, build the 2025 proxy from what actually predicts the reviewer pool: **the adjacent 2025 workshops on agent/LLM memory, continual learning and machine unlearning** (name them, with their accepted-paper themes), plus **the recent (2025–2026) publications of PALM's own organizers and invited speakers** — the scout report already names them; their current work is the sharpest available signal.

## Part 3 — what each profile must deliver (the writers use this, not the raw lists)
For **each** venue, from 2025–2026 material only:
1. **Topic distribution** — where the mass actually is, with counts.
2. **Method and vocabulary census** — the terms and framings that recur in recent titles/abstracts, so our prose sounds native rather than four years out of date. ⚠ Explicitly flag any term whose meaning has *shifted* since the 2022/older material the current drafts were written against.
3. **An updated vocabulary map** — our term → their current standard term, marked **exact / approximate / no-equivalent** (the previous map was built off 2022 works; supersede it and say what changed).
4. **The nearest-neighbour set, refreshed** — the 5–8 most recent works each paper should be positioned against, with a verified record and one sentence on what each claims. ⚠ Flag any that narrow one of our novelty claims, and say which claim.
5. **What this audience rewards and what it rejects** — evidenced from the recent accepted set (e.g. does it accept toy-scale results, negative results, theory without experiments), not from impression.

## Rules
- Primary sources only (OpenReview, PMLR, the venues' own sites, publisher records); aggregators as corroboration; retrieval dates on everything; an unavailable list is declared, never inferred.
- ⚠ **Every classification into topics is your inference and must be labelled as such** — the previous pass did this correctly and it matters.
- ⛔ No strategy recommendations — venue choice and framing are the Advisor's and the Head's. Facts, records and the maps.

## Acceptance criteria
1. NeurReps 2025 (and 2024 if reachable) resolved: list recovered, or failure declared with the surfaces tried.
2. PALM's lineage settled, with a 2025 proxy built and its construction stated.
3. All five Part-3 deliverables present for both venues, on 2025–2026 material, with the vocabulary map marked and the superseded 2022-based map explicitly noted.
4. Standard `## Proposed handover updates` and `## Flags`.
