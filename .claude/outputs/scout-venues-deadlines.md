# scout-venues-deadlines — web-scout report

Task + acceptance criterion: Verify the entire publication calendar (NeurIPS 2026 workshops → ICLR 2027, backups) with primary sources; verified table + policy quotes; anything unverifiable marked UNVERIFIED/ESTIMATE.
Status: **done** (with structural unknowns: NeurIPS 2026 workshop list posts Jul 11; ICLR 2027 CFP not live — both explicitly marked)

What I did:
- Fetched primary pages on **2026-07-04**: neurips.cc 2026 (front page, Dates, CallForWorkshops, WorkshopsGuidance), iclr.cc (front, FutureMeetings, 2026 CFP + AuthorGuide, 2025 CFP; all 2027 pages → 404), ml4physicalsciences.github.io (2025 main + guidelines; 2026 → 404), virtual.aistats.org 2026 Dates (aistats.org/aistats2027 → 404), icml.cc FutureMeetings.
- Cross-checked via web search; debunked one circulating false datapoint (see ICLR row).

How I verified: every CONFIRMED cell below was quoted from an official venue page fetched today; aggregator-only claims are marked UNVERIFIED; pattern-based projections are marked ESTIMATE with the underlying cycles shown.

## Answer first
The NeurIPS 2026 accepted-workshop list **does not exist yet — acceptance notifications land Jul 11, 2026 AoE**, so target-workshop selection must be a follow-up task in ~1 week. The venue-level suggested workshop-paper deadline is **Aug 29, 2026 AoE**, one day after our Aug-28 freeze — the freeze must move earlier (~Aug 21–24) or drafting must run fully in parallel. **ICLR 2027 is not announced** beyond "West Coast North America" (no dates); the circulating "Sept 19/24, Brazil" numbers are a **conflation with ICLR 2026** — treat mid/late-Sept 2026 deadlines as ESTIMATE only. Non-archival status is guaranteed venue-wide: NeurIPS itself states all workshop papers are non-archival, and ICLR's dual-submission policy (2026 text) explicitly exempts workshop-presented work.

## Verified calendar table

| Venue | Event | Date | Source | Status |
|---|---|---|---|---|
| NeurIPS 2026 | Main conference (multi-site) | **Sydney Dec 6–12**; satellites **Atlanta & Paris Dec 9–13** | https://neurips.cc/ (front page) | CONFIRMED ⚠ note (a) |
| NeurIPS 2026 | Workshop acceptance notification (accepted-workshop list drops) | **Jul 11, 2026 AoE** | https://neurips.cc/Conferences/2026/CallForWorkshops | CONFIRMED |
| NeurIPS 2026 | **Suggested** workshop-paper submission deadline | **Aug 29, 2026 AoE** (per-workshop discretion; may be earlier) | CallForWorkshops + https://neurips.cc/Conferences/2026/WorkshopsGuidance | CONFIRMED |
| NeurIPS 2026 | **Mandatory** workshop accept/reject notification | **Sep 29, 2026 AoE** | same | CONFIRMED |
| NeurIPS 2026 | Workshop days | Sydney: **Dec 11–12**; Paris & Atlanta: **Dec 12–13** | CallForWorkshops | CONFIRMED |
| NeurIPS 2026 | Main-conf paper cycle (context only) | abstract May 4 / paper May 6 / notify **Sep 24, 2026** | https://neurips.cc/Conferences/2026/Dates | CONFIRMED (passed; not for us) |
| ICLR 2027 | Location | "**ICLR 2027: West Coast North America**" — no city, no dates | https://iclr.cc/Conferences/FutureMeetings | CONFIRMED (that's all that exists) |
| ICLR 2027 | Abstract / paper deadlines | **NOT ANNOUNCED** (iclr.cc/Conferences/2027 and /2027/CallForPapers → 404 on 2026-07-04). Pattern: ICLR 2026 = abstract Sep 19 / paper Sep 24, 2025; ICLR 2025 = abstract Sep 27 / paper Oct 1, 2024 → expect **abstract ~Sep 19–27, 2026; paper ~Sep 24–Oct 1, 2026** | https://iclr.cc/Conferences/2026/CallForPapers , https://iclr.cc/Conferences/2025/CallForPapers | **ESTIMATE** |
| ML4PS @ NeurIPS | 2026 edition | Page not live (ml4physicalsciences.github.io/2026 → 404). 2025 pattern: deadline **Fri Aug 29, 2025, 23:59 AoE** (= that year's venue suggested date), **4 pages excl. refs**, NeurIPS template, **double-blind**, **no rebuttal**, non-archival, workshop day Dec 6, 2025 | https://ml4physicalsciences.github.io/2025/ + /2025/guidelines.html | 2026 UNVERIFIED; 2025 pattern CONFIRMED |
| AISTATS 2027 | Backup venue | Official site: nothing (aistats.org/aistats2027 → 404). Aggregator (PaperPilot) claims Montreal, Feb 16–23, 2027 — **UNVERIFIED**. Pattern (AISTATS 2026): abstract Sep 25 / paper **Oct 2, 2025** AoE, notify Jan 22, 2026, conf May 2–5, 2026, Tangier, Morocco → expect deadline **~late Sep–early Oct 2026** | https://virtual.aistats.org/Conferences/2026/Dates | ESTIMATE |
| ICML 2027 | Backup venue | Official: "**2027: Announcement coming in August**" (2028 = Eastern United States). A search snippet said "South America" — single-sourced, conflicts with the official page; ignore. Pattern deadline ~late Jan 2027 | https://icml.cc/Conferences/FutureMeetings | UNVERIFIED/ESTIMATE |

**(a) Location discrepancy:** neurips.cc/Conferences/2026/Dates still displays "San Diego, CA" (apparently a stale 2025 template artifact), while the front page and CallForWorkshops agree on Sydney + Atlanta + Paris with mutually consistent workshop-day dates. Treat front page + CallForWorkshops as authoritative.

## Policy quotes (verbatim, load-bearing)

1. **NeurIPS workshop archival status** — NeurIPS 2026 Workshops Guidance:
   > "All NeurIPS workshop papers are non-archival and therefore do not appear in proceedings."

   → Our non-archival requirement is satisfied **venue-wide**, regardless of which workshops we pick. Organizer discretion only covers whether papers are *posted* ("We leave it to the organizers whether they wish to provide access to accepted papers on OpenReview or other platforms.").

2. **NeurIPS deadline flexibility** — same doc:
   > "the submission date for workshop contributions is suggested, and there is a trade-off between how much time workshops give authors to submit papers versus how much time reviewers have to provide reviews."

   → Individual workshops may set deadlines **earlier** than Aug 29.

3. **ICLR dual submission** — ICLR 2026 Author Guide (2027 text UNVERIFIED until its CFP posts; wording stable across recent cycles):
   > "Submissions that are identical (or substantially similar) to versions that have been previously published, or accepted for publication, or that have been submitted in parallel to this or other conferences or journals, are not allowed and violate our dual submission policy. However, papers that cite previous related work by the authors and papers that have appeared on non-peer reviewed websites (like arXiv) or that have been presented at workshops (i.e., venues that do not have publication proceedings) do not violate the policy."

   → **The shorts→ICLR pipeline is explicitly legal** under current policy. ICLR page limit: "At the time of submission, the main text should be 9 pages or fewer," 10 pages at camera-ready; refs/ethics/reproducibility excluded.

4. **ML4PS 2025 guidelines** (pattern for the flagship target):
   > "This workshop is not archival, so we will consider papers containing content that is published in an archival venue other than the main NeurIPS conference (e.g. a physics journal)."

   > "we strictly prohibit submitting to multiple workshops simultaneously."

   → One paper = one workshop. Our 3-distinct-shorts plan complies, but **no hedging a single short across two workshops**.

## Candidate workshops (status today)
The 2026 list is unknowable until **Jul 11**. Verified-recurring neighbors from the 2025 cycle:
- **ML4PS** — ran at NeurIPS 2025 (Dec 6, 2025); GitHub org continuous since ~2017; 4pp, double-blind, no rebuttal, non-archival. Best fit for V2 (Goldstone/mass-spectrum) and arguably V3.
- **AI4Science** — 6th edition at NeurIPS 2025 (OpenReview group: NeurIPS.cc/2025/Workshop/AI4Science).
- **NeurReps** (symmetry/geometry in neural representations) — present at NeurIPS 2025; thematically ideal for V2's SSB/symmetry story.
- DLDE, time-series-specific, and EBM-specific workshops: **not verified this pass** — fold into the Jul-11 follow-up rather than guess.

## Implications for the Aug-28 freeze / Sept-1 assembly plan
1. **Freeze collides with the deadline.** Suggested workshop deadline = Aug 29 AoE and some workshops will pick earlier dates. Recommend: **move results-freeze to ~Aug 21–24**, shorts submission-ready by **Aug 27**; confirm each target workshop's actual deadline in the week of Jul 11.
2. **Sequencing works for us:** ICLR 2027 deadline (est. late Sept 2026) lands **before** workshop notifications (≤ Sep 29) and the workshops (Dec) — legal under the quoted ICLR policy (workshops = no proceedings). The ICLR paper can cite the arXiv/submitted versions of the shorts, not acceptances.
3. **ICLR 2027 = West Coast North America** (not Brazil — that's ICLR 2026/Rio). If ICLR 2027 timing slips or misfits, **AISTATS 2027** (est. deadline ~Oct 2026) is the near-adjacent fallback; **ICML 2027** (deadline pattern ~late Jan 2027) is the slow fallback; official ICML 2027 announcement due **August 2026**.
4. **Multi-site logistics:** workshops run Sydney (Dec 11–12) and Paris/Atlanta (Dec 12–13). Paris presumably the practical site from Manchester — but which site hosts which workshop is unknown until the list posts.

## Confidence & gaps
- CONFIRMED (primary, fetched 2026-07-04): NeurIPS 2026 workshop timeline + non-archival policy + multi-site dates; ICLR 2026/2025 deadline pattern + dual-submission text + page limits; ML4PS 2025 pattern; AISTATS 2026 pattern.
- ESTIMATE/UNVERIFIED: all ICLR 2027 dates; ML4PS/AI4Science/NeurReps **2026** editions; AISTATS 2027 (aggregator-only); ICML 2027 (official says announcement in August).
- **Next scout pass: Jul 12–15, 2026** — pull the accepted-workshop list, build per-workshop table (actual deadline / page limit / review format / site), and watch iclr.cc for the 2027 CFP (~Aug per past cycles).

Git footprint: none (no code changes; this file only).

Open questions / follow-ups / risks:
- R: any target workshop setting a deadline **before** Aug 29 compresses us further — unknowable until Jul 11.
- R: ICLR 2027 could deviate from pattern (e.g., earlier abstract deadline); re-verify the moment the CFP posts.
- Q: which satellite site (Sydney/Paris/Atlanta) do we prefer if a target workshop runs at multiple sites or only one?
- Note: earlier in-thread I mis-stated that this agent lacked a Write tool; corrected — this file is the canonical deliverable.

## Proposed handover updates (for the Hub)
- **Roadmap "Submit late Aug–early Sept"** → replace with: "venue-wide *suggested* deadline **Aug 29, 2026 AoE**; per-workshop deadlines may be earlier; **move freeze to Aug 21–24**, submission-ready Aug 27."
- **Roadmap "ICLR historically abstract ~Sept 19 / paper ~Sept 24 — verify"** → ICLR 2027 is **unannounced** except "West Coast North America" (iclr.cc/FutureMeetings); keep abstract ~Sep 19–27 / paper ~Sep 24–Oct 1, 2026 as ESTIMATE. The "Brazil / Sept 19, 2026" numbers on aggregators are ICLR-**2026** data — do not propagate.
- **Add calendar checkpoint:** **Jul 11, 2026 = NeurIPS accepted-workshop list drops** → spawn immediate follow-up scout (per-workshop deadlines + site assignments).
- **Record constraints:** (i) one short per workshop — simultaneous multi-workshop submission prohibited (ML4PS precedent); (ii) NeurIPS workshop papers **non-archival venue-wide** (verbatim quote above); (iii) ICLR dual-submission policy explicitly exempts workshop-presented work (verbatim quote above) — pipeline policy-clean end-to-end.
- **Backups:** AISTATS 2027 est. deadline ~late Sep–early Oct 2026 (unannounced; aggregator says Montreal Feb 2027, UNVERIFIED); ICML 2027 official announcement due Aug 2026.
