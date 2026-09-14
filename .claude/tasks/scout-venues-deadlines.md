# Task: scout-venues-deadlines — verify our entire publication calendar

- **Agent:** `web-scout` · **Output:** `.claude/outputs/scout-venues-deadlines.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` (§8 pointer), `.claude/research_roadmap.md` (publication strategy).
- **Why:** the whole program schedule (3–4 workshop shorts Aug/Sept 2026 → ICLR 2027 main track) currently rests on *assumed* deadlines. Verify everything with primary sources.

## Acceptance criterion
A verified calendar table with primary-source URLs for every date, plus dual-submission/archival policy quotes. Anything unverifiable is explicitly marked UNVERIFIED with your best evidence.

## Sub-tasks
1. **NeurIPS 2026 workshops:** find the accepted-workshop list (or proposal-results status if not yet posted). Identify workshops fitting our profile — ML4PS (Machine Learning and the Physical Sciences) and neighbors (e.g., symmetry/geometry in ML, AI4Science, DLDE/differential-equations, time-series/sequential, self-supervised/energy-based). For each candidate: paper deadline, page limit, review format, **archival status (we require non-archival)**, and whether workshop papers can later go to a conference.
2. **ICLR 2027:** exact abstract + full-paper deadlines, page limits, **dual-submission policy** (especially: is prior non-archival workshop publication of components allowed? quote the policy text), reviewing timeline.
3. **Backup venues:** if ICLR 2027 timing turns out wrong for us, note the adjacent options (AISTATS 2027, ICML 2027 dates if known) — one line each, no deep dive.
4. **NeurIPS 2026 main-conference dates** (for context: when workshops actually happen, camera-ready timing).

## Method & rules
Primary sources only (official venue sites, OpenReview, official workshop pages); date-stamp every claim; if 2026/2027 pages aren't live yet, say so and give the last-2-cycles pattern as the estimate, clearly labeled ESTIMATE. Never present a pattern-based guess as a confirmed date.

## Output format
Calendar table (venue | event | date | source URL | confirmed/estimate) → policy quotes → implications for our Aug-28 freeze / Sept-1 assembly plan → `## Proposed handover updates`.
