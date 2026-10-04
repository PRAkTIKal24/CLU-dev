# venue-policy-recheck — web-scout report

Task + acceptance criterion: re-verify every load-bearing venue-policy fact for the shorts→ICLR→NMI pipeline against the CURRENT cycle's primary sources, with verbatim quotes + URL + retrieval date; plus a maximally-inclusive sweep of current-cycle NeurIPS workshops with published deadlines.
Status: **done** (with two structural absences that are themselves findings: the central NeurIPS-2026 accepted-workshop list is **not yet publicly posted**, and **ML4PS has no 2026 page**).

**DIAL DECLARATION (echo):** the task file carried no dial block. Interpretation: **none — instrument/recon** (venue-policy reconnaissance). No laundering control, no falsification condition applies; this report contains **facts and quotes only, no strategy** (task acceptance criterion 4).

**Retrieval date for every quote below: 2026-08-05** unless stated otherwise.

**⚠ RECONCILIATION LIST OWNER NEEDED (first-10-lines rule, protocol §5):** three program-held facts are now **superseded** and need an owner — (i) ICLR-cycle policy is no longer "ICLR 2026 text, 2027 unannounced" — **the ICLR 2027 CFP + Author Guidelines are LIVE with hard dates**; (ii) the blanket claim "NeurIPS workshops are non-archival venue-wide, so any workshop is safe" is **incomplete** — **NeurReps 2026 runs an ARCHIVAL PMLR Proceedings track** alongside its non-archival track; (iii) the "one-short-one-workshop, ML4PS precedent" rule is **not this cycle's ML4PS rule** (no 2026 ML4PS page exists); the live analogue is Sim2Science's softer "we discourage".

---

## 1. ICLR (current cycle = **ICLR 2027**) — dual submission / prior publication / preprint / anonymity

**The ICLR 2027 CFP and Author Guidelines are LIVE** (both 404'd at the program's last check on 2026-07-19). This is the single biggest delta vs. the file.

### 1.0 Dates (quoted exactly as published; not interpreted)
Source: https://iclr.cc/Conferences/2027/CallForPapers (retrieved 2026-08-05)
> "Abstract deadline Sep 18, 2026 AOE
> Paper deadline Sep 25, 2026 AOE
> Reviews released Nov 05, 2026
> Author-reviewer discussion Nov 05, 2026–Nov 18, 2026
> Reviewer-AC discussion Nov 19, 2026–Dec 16, 2026
> Final decisions: Dec 16, 2026"

(The page states all times are AoE / UTC-12h.) Venue: "ICLR 2027: West Coast North America" per https://iclr.cc/Conferences/FutureMeetings (previously verified 2026-07-19; not re-fetched this pass — **flagged**).

### 1.1 (a) Do non-archival workshop papers count as prior publication / dual submission? — **NO, explicitly exempt.**
Source: https://iclr.cc/Conferences/2027/AuthorGuidelines
> "Submissions that are identical (or substantially similar) to versions that have been previously published, or accepted for publication, or that have been submitted in parallel to this or other conferences or journals, are not allowed and violate our dual submission policy."

> "papers that cite previous related work by the authors and papers that have appeared on non-peer reviewed websites (like arXiv) or that have been presented at workshops (i.e., venues that do not have publication proceedings) do not violate the policy."

**⚠ The exemption is scoped by the parenthetical "venues that do not have publication proceedings."** A workshop track with proceedings (see §2, NeurReps Proceedings/PMLR) is **not** covered by the quoted exemption text. This is a juxtaposition of two verbatim policies, not an interpretation of intent.

### 1.2 (b) arXiv / preprint policy for submissions under review — **permitted.**
Source: https://iclr.cc/Conferences/2027/CallForPapers
> "Having papers on arxiv is allowed per the dual submission policy outlined in the author guidelines."

On reviewers searching: I found **no** general instruction to reviewers not to search for the paper. The only "don't search" clause I located is **challenge-leaderboard-specific** — reproducing its full paragraph for context so it is not over-read:
Source: https://iclr.cc/Conferences/2027/AuthorGuidelines
> "It is ok to report the results on the leaderboard of a challenge. The authors can include the ranking and the name of the challenge. The reviewers will be advised to not intentionally search the authors by examining the leaderboard."

**⚠ Do not quote this as a general "reviewers are told not to search arXiv" policy — it is not.** Absence of a general clause is a *negative* finding from one page; the Reviewer Guide for 2027 was not fetched this pass (**gap, §Flags**).

### 1.3 (c) Anonymity clauses touching authors' own public preprints / workshop papers during the review window
Source: https://iclr.cc/Conferences/2027/AuthorGuidelines
> "ICLR 2027 is double blind, which means that all submitted papers should be anonymous. Any paper where author identity is revealed in either the main text or the supplementary material will be desk rejected."

> "Note that related arxiv papers by the same authors do not break anonymity; if cited, these should be cited in third person."

Visibility / de-anonymization mechanics (new detail, not previously on file):
> "After the paper submission deadline, if an author chooses to withdraw a submission, it will remain hosted by OpenReview in a publicly visible 'withdrawn papers' section."

> "All submitted papers (accepted, rejected or withdrawn) will be deanonymized after the notification."

> "OpenReview will allow for public discussion any time during the discussion phase. Anybody who is logged in can post comments that are publicly visible, or restrict visibility to reviewers and up, ACs and up, or just PCs. All comments apart from those of the authors, reviewers, ACs or the organizing committee will be required to be non-anonymous."

Per the same page, submissions are **not** publicly visible during the active review phase; reviews become "publicly visible in OpenReview" after **Nov 5, 2026**, and de-anonymization occurs at notification (**Dec 16, 2026**). ⚠ This last pair is WebFetch's rendering of the page's schedule prose rather than a single quoted sentence — **flagged as single-tool-sourced**.

### 1.4 (d) Did the policy text change vs. the prior cycle's wording on file? — **NO change in the load-bearing sentences.**
- Prior cycle on file (ICLR 2026 Author Guide, quoted in `.claude/outputs/scout-venues-deadlines.md`, retrieved 2026-07-04): *"Submissions that are identical (or substantially similar) … do not violate the policy."*
- Current cycle (ICLR 2027 Author Guidelines, 2026-08-05): **character-for-character the same two sentences** as quoted in §1.1.
- Page limit unchanged: > "the main text should be 9 pages or fewer" (submission); > "the page limit will be increased to 10 pages" (rebuttal); > "The list of references does not count towards the page limit, and unlimited additional pages are allowed for the bibliography/references."
- **New/changed in 2027** (not in the program's file): co-author quota > "No author may appear as a co-author on more than 20 papers."; reciprocal reviewing > "All authors who are on 3 or more papers must serve as a reviewer for at least 6 papers."; > "If none of the authors are registered as a reviewer, it will result in desk rejection"; > "Incorrect information on your profile will be grounds for desk rejection"; > "Placeholder or duplicate abstracts will be removed"; and a separate **AI Policy for Authors** at `/Conferences/2027/AIPolicyForAuthors` (referenced only: > "Please see the AI Policy for Authors." — **not fetched this pass, gap**).

---

## 2. NeurIPS workshop camera-ready publication practices (current cycle)

### 2.0 Venue-level rules (primary, current cycle)
Source: https://neurips.cc/Conferences/2026/WorkshopsGuidance
> "All NeurIPS workshop papers are non-archival and therefore do not appear in proceedings."

> "We leave it to the organizers whether they wish to provide access to accepted papers on OpenReview or other platforms (be sure to inform authors on your website how/if you will provide access to the accepted papers)."

> "Workshops that accept contributions must use OpenReview to manage submissions."

Source: https://neurips.cc/Conferences/2026/CallForWorkshops
> "Workshop Acceptance Notification: July 11, 2026, AoE"
> "Suggested Submission Date for Workshop Contributions: August 29, 2026, AoE"
> "Mandatory Accept/Reject Notification Date: September 29, 2026, AoE"
> "Fri Dec 11 and Sat Dec 12, 2026 (Sydney)" / "Sat Dec 12 and Sun Dec 13, 2026 (Paris, Atlanta)"

⚠ **The venue-wide non-archival statement above is contradicted in practice by at least one accepted 2026 workshop** (NeurReps, §2.1). The guidance sentence is a statement about NeurIPS proceedings; it does not stop a workshop from running its own PMLR volume.

### 2.1 Per-workshop, current cycle (each quote from the workshop's own site, 2026-08-05)

**NeurReps 2026 — Symmetry and Geometry in Neural Representations** (https://neurreps.org/)
- Three tracks. **Proceedings track is ARCHIVAL:** > "Self-contained, highly-developed research papers. Archivally published in a dedicated PMLR volume. Double-blind review via OpenReview." (9 pages excl. refs + appendices)
- **Extended Abstract track:** > "Early-stage results, negative findings, opinion pieces, or novel datasets. Non-archival — may be posted to arXiv. Double-blind review via OpenReview." (4 pages excl. refs + appendices)
- **Findings track:** > "High-impact collaborative work between experimentalists and theorists…Single-blind, editorially reviewed by an advisory panel of experts." (no page limit)
- Dual-submission rule (verbatim): > "Papers in the Proceedings Track will be archivally published. Thus, submissions containing content that has been published or is under review elsewhere must include at least 30% new, unpublished/unsubmitted material." and > "to publish a NeurReps paper in another venue down the line, authors must add at least 30% new material." and > "There are no restrictions on Extended Abstract submissions."
- Deadline > "Submission Deadline August 22, 2026 · AoE"; notification Sep 29, 2026; Sydney, Dec 11–12.

**PALM — Personalized, aligned, long-term memory for AI systems** (https://palm-neurips-2026.github.io/)
- > "Accepted papers will be made public, but rejected submissions and reviews will not." (OpenReview)
- > "Submissions must be fully anonymized. This policy applies to any supplementary or linked material as well, including code."
- Non-archival: > "a non-archival policy, welcoming ongoing and unpublished work"; > "Workshop submissions can be subsequently or concurrently submitted to other venues."
- Deadline > "August 24, 2026" with > "All deadlines are 11:59pm AoE (Anywhere on Earth)."; 9pp full / 4pp short excl. refs; Paris, Dec 12 or 13.

**TTCL — Towards Test-Time Continual Learning Agents** (https://ttcl-agents.github.io/)
- > "Accepted papers will be made publicly available on OpenReview, and all accepted papers will be presented in a poster session."
- > "a double-blind review process" (cite prior work "in the third person"); explicitly "non-archival"; welcomes papers "under review at, or have been accepted by, other venues."
- Deadline > "August 29, 2026", > "All deadlines are 11:59 PM, Anywhere on Earth (AoE)."; 4–9pp excl. refs/appendices; Atlanta, Dec 12 or 13.

**AXIOM 2026 — Foundations of Efficient Deep Learning** (https://axiom-neurips2026.github.io/)
- > "All accepted papers will be presented during the poster session and published on the workshop website."
- > "This workshop is non-archival." · double-blind.
- ⚠ **Deadline is NOT AoE:** > "August 29, 2026 (11:59 PM UTC-0)" — ~12 h earlier than an AoE deadline of the same date. 4 pages excl. refs+appendix; Paris, Dec 12.
- Dual submission (restrictive relative to peers): submissions > "must not duplicate work previously published" at ML conferences, and > "Work presented at the main NeurIPS conference must not also appear in the workshop."

**Sim2Science — ML with Imperfect Scientific Models** (https://www.sim2science.com/ , /cfp)
- > "The workshop will not have formal proceedings, but accepted submissions will be linked on the workshop webpage"; > "Sim2Science is non-archival — there are no formal proceedings, so accepted work remains eligible for submission to archival venues afterward."
- > "No — submissions are kept confidential unless accepted and the authors confirm inclusion in the workshop." (i.e., an **opt-in** to public posting — the only opt-out-like mechanism found this pass)
- > "Double-blind review required — please anonymize your submission (no author names/affiliations, avoid self-identifying references)"
- > "Submission deadline: August 29, 2026 (23:59 AoE)"; > "5 pages (excluding references) for workshop papers, or 2 pages (excluding references) for Tiny Papers"; Paris, Dec 12 or 13.

**AI for Science: "Verification in the Age of AI Scientists"** (https://ai4sciencecommunity.github.io/neurips26.html)
- > "Our workshop is **nonarchival**"; > "accepted papers will be posted on the workshop website".
- > "August 29, 2026 AoE"; > "Submissions should be 4-8 pages, with unlimited references and appendices"; Sydney, Dec 11 or 12. Blinding rule **not stated on the page** (gap).

**Efficient and On-Device AI Agents** (https://efficient-ondevice-ai-agents.github.io/)
- > "Workshop papers are non-archival. Accepted papers may be submitted to venues with archival proceedings."
- > "Double-blind review: Submissions must be anonymized. Author names and affiliations should not appear in the paper."
- Deadline August 29, 2026, > "11:59 PM AoE"; 4pp short / 9pp long + unlimited refs; Sydney, Dec 11–12. Public posting **not stated** (gap).

**Child Safety in AI** (https://childsafety-ai.github.io/)
- > "Accepted papers will be non-archival and may be submitted to other venues"; > "Submissions must be anonymous"; > "August 29, 2026, AoE"; "up to four pages, excluding references"; Atlanta, Dec 12 or 13. Public posting **not stated**.

**Prior-cycle norm (labelled PRIOR-CYCLE PROXY — NOT this year's rule):** ML4PS 2025 (https://ml4physicalsciences.github.io/2025/guidelines.html, quoted in the program's 2026-07-04 scout) — 4pp, double-blind, no rebuttal, non-archival, > "This workshop is not archival, so we will consider papers containing content that is published in an archival venue other than the main NeurIPS conference (e.g. a physics journal)." **There is no ML4PS 2026 page as of 2026-08-05** (see §6).

### 2.2 Summary of (a)–(d) as asked
- **(a) Posted publicly, with names, and when:** varies by workshop and is organizer discretion per the venue guidance. Confirmed public posting: PALM (OpenReview), TTCL (OpenReview), AXIOM (workshop site), AI4Science (workshop site), Sim2Science (site, opt-in). Timing of posting is not stated by any of them beyond "accepted"; notification is ≤ **Sep 29, 2026 AoE** venue-wide. **No workshop states an explicit de-anonymized-camera-ready date.**
- **(b) Opt-out / delayed posting:** only **Sim2Science** offers anything resembling it — > "submissions are kept confidential unless accepted and the authors confirm inclusion in the workshop." No workshop found this pass offers delayed posting.
- **(c) Blinding:** double-blind at NeurReps (Proceedings + Extended Abstract), PALM, TTCL, AXIOM, Sim2Science, Efficient On-Device Agents, Child Safety ("anonymous"). Single-blind: NeurReps **Findings** track only. Not stated: AI4Science, BeNTo, TAE, Interpretability×2.
- **(d) Non-archival statements:** quoted verbatim above per workshop. **Exception: NeurReps Proceedings track is archival (PMLR).**

---

## 3. Workshop-side concurrent-submission rules (can the same/overlapping content sit under review at ICLR while submitted to a NeurIPS workshop?)

| Workshop | Verbatim policy | Effect |
|---|---|---|
| TTCL | "non-archival and welcomes submissions that are under review at, or have been accepted by, other venues" | **Explicitly allowed** |
| PALM | "Papers under review or recently accepted at other venues" permitted subject to those venues' policies; "Workshop submissions can be subsequently or concurrently submitted to other venues." | **Explicitly allowed** |
| Efficient & On-Device AI Agents | "Papers under review at other venues are welcome, subject to those venues' policies." | **Explicitly allowed** |
| Child Safety in AI | "Accepted papers will be non-archival and may be submitted to other venues" | Allowed (forward direction stated; concurrent implied, not stated) |
| Sim2Science | "accepted work remains eligible for submission to archival venues afterward"; **but** "We discourage parallel submission of the same paper to multiple NeurIPS 2026 workshops. Please use your best judgment to select the venue that best fits your work." | Allowed vs. archival venues; **discouraged across NeurIPS workshops** (soft, not prohibited) |
| AXIOM 2026 | submissions "must not duplicate work previously published" at ML conferences; "Work presented at the main NeurIPS conference must not also appear in the workshop." | Silent on *under-review* elsewhere; bars *published*/NeurIPS-main duplicates |
| NeurReps — Extended Abstract | "There are no restrictions on Extended Abstract submissions." | **Unrestricted** |
| NeurReps — **Proceedings** | "submissions containing content that has been published or is under review elsewhere must include at least 30% new, unpublished/unsubmitted material"; and to later publish elsewhere, "authors must add at least 30% new material" | **Restricted both directions — 30% novel-material floor** |
| AI4Science, BeNTo, TAE, Interpretability×2, AI4Meta-Science | not stated on the pages fetched | **UNKNOWN — gap** |

Prior-cycle contrast on file: ML4PS 2025's hard > "we strictly prohibit submitting to multiple workshops simultaneously." **No 2026 workshop found this pass repeats that prohibition**; the nearest live analogue is Sim2Science's "we discourage."

---

## 4. NMI (Nature Machine Intelligence) — prior publication

⚠ **SOURCING CAVEAT (single-sourced / not directly fetched).** `www.nature.com/natmachintell/...` redirects (HTTP 303) to `idp.nature.com` SSO for this tool, so I could not fetch the page body. The NMI-specific policy page **exists** at https://www.nature.com/natmachintell/editorial-policies/preprints-conference-proceedings (confirmed as an indexed page title, 2026-08-05); the wording below is from the **search-engine index of the identical Nature-portfolio-wide page** (same text served for Nature, Nature Communications, Nature Computational Science, Nature Methods, …). Treat as **HIGH-confidence-on-substance, NOT-directly-fetched-verbatim** and re-verify from a browser before it becomes load-bearing.

- **Preprints:** > "Posting of preprints is not considered prior publication and will not jeopardize consideration at Nature Portfolio journals." and > manuscripts posted on preprint servers "will not be taken into account when determining the advance provided by a study under consideration at a Nature Portfolio journal."
- **Conference proceedings:** Nature journals "are happy to consider submissions containing material that has been published in a conference proceedings paper", provided > "The submission should provide a substantial extension of results, methodology, analysis, conclusions and/or implications over the conference proceedings paper"; the final decision on what constitutes a substantial extension rests with each journal's editors; and > "Authors must provide details of the conference proceedings paper with their submission including relevant citation in the submitted manuscript."
- **Non-archival workshop papers:** **No Nature-portfolio text specific to non-archival workshop papers was found.** The policy is written in terms of *preprints* and *conference proceedings papers*. A non-archival workshop paper that is only posted on OpenReview/a workshop site with no proceedings is **not addressed explicitly** — this is a genuine gap in the published policy, not something I could resolve from primary text.
- **Media:** authors posting preprints "are asked to respect the policy on communications with the media"; media coverage of a preprint or conference presentation "will not hinder editorial handling of the submission."

---

## 5. Desk-reject / anonymization checklist deltas (what changes vs. the program's existing checklist)

Additions implied by the **current** primary text quoted above:

1. **ICLR 2027, supplementary material is in scope for desk rejection:** > "Any paper where author identity is revealed in either the main text or the supplementary material will be desk rejected."
2. **ICLR 2027 code/repo anonymization, two sanctioned forms:** > "Anonymize your code, put it in a .zip file and submit it as supplementary materials" **or** > "Make an anonymous repository and put the link in your paper." (A named GitHub link is therefore a desk-reject vector.)
3. **Self-citation form:** > "related arxiv papers by the same authors do not break anonymity; if cited, these should be cited in third person." (Workshop-short self-cites must be third-person too — the shorts and the long share results.)
4. **ICLR 2027 NEW, non-content desk-reject vectors** (not previously on file): reviewer registration — > "If none of the authors are registered as a reviewer, it will result in desk rejection"; reciprocal-review load — > "All authors who are on 3 or more papers must serve as a reviewer for at least 6 papers."; profile hygiene — > "Incorrect information on your profile will be grounds for desk rejection"; abstract hygiene — > "Placeholder or duplicate abstracts will be removed"; co-author cap — > "No author may appear as a co-author on more than 20 papers."
5. **Page-limit desk reject:** > "Papers with main text beyond the page limit will be desk-rejected." (9pp at submission, 10pp at rebuttal; refs unlimited; "Acknowledgements etc. do not count for page limit".)
6. **AI-use policy exists separately** (`/Conferences/2027/AIPolicyForAuthors`) and was **not fetched** — an unchecked checklist item.
7. **Workshop-side anonymization is stricter than assumed at PALM:** > "This policy applies to any supplementary or linked material as well, including code." — an anonymized-artifact requirement on the *shorts*, not just the long.
8. **Archival-track trap:** the ICLR exemption text is scoped to "venues that do not have publication proceedings"; **NeurReps' Proceedings track has a PMLR volume** and additionally imposes a 30%-new-material floor in both directions. Any checklist that treats "NeurIPS workshop ⇒ non-archival ⇒ safe" needs a per-track check.
9. **Timezone trap:** AXIOM's deadline is **UTC-0**, not AoE, while every other workshop found is AoE.

---

## 6. The full venue sweep — current-cycle NeurIPS workshops relevant to the shorts

### 6.0 Structural finding (state of the list, 2026-08-05)
- **The central accepted-workshop list is NOT publicly posted.** https://neurips.cc/Conferences/2026/Workshops → **HTTP 404**. https://neurips.cc/Conferences/2026/Schedule?type=Workshop → renders the schedule shell with **no workshop entries**. The 2026 conference front page shows no accepted-workshop announcement.
- **Announcement mechanism + expected timing, verbatim** (NeurIPS Blog, July 2026 newsletter, https://blog.neurips.cc/2026/07/ , retrieved 2026-08-05): > "The decisions for the workshops will be posted in early August followed by a blog post – stay tuned!" (Organizer notification itself was **Jul 11, 2026 AoE** per CallForWorkshops.)
- Consequently the table below is built **workshop-by-workshop from self-published sites**, and is **necessarily incomplete**: workshops that have not yet put up a page are invisible to this pass.
- **ML4PS (Machine Learning and the Physical Sciences) has NO 2026 page**: https://ml4physicalsciences.github.io/2026/ → **HTTP 404**; the root https://ml4physicalsciences.github.io/ returned no fetchable body this pass and is indexed as "Machine Learning and the Physical Sciences, **NeurIPS 2025**". **Its 2026 status (accepted? deadline?) is UNKNOWN.** Nothing in this report should be read as ML4PS running in 2026.
- **No dedicated energy-based-model / associative-memory / unlearning workshop was found in the NeurIPS 2026 cycle this pass.** Adjacent prior-cycle datum (PRIOR-CYCLE / OTHER-VENUE, not NeurIPS 2026): the associative-memory workshop line ran as **NFAM @ ICLR 2026** (https://nfam2026.amemory.net/), third iteration after ICLR 2025 and NeurIPS 2023 — i.e. that community's current home is ICLR, not NeurIPS. Search-snippet-sourced; **flagged**.

### 6.1 Sweep table (every row's facts from the workshop's own site unless marked)

| # | Workshop | URL | Deadline (as published) | Pages | Blinding | Archival | Site/Date | Fit | Tag |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **NeurReps 2026** — Symmetry & Geometry in Neural Representations | https://neurreps.org/ | **Aug 22, 2026 · AoE** | Proc. 9pp · Ext.Abs. 4pp (excl. refs+appx) · Findings none | Proc./Ext.Abs. double-blind; Findings single-blind | **Proc. = ARCHIVAL (PMLR)**; Ext.Abs. non-archival | Sydney, Dec 11–12 | **V2** (Goldstone/symmetry-retention laws are literally the CFP's subject); V3 partial (geometry of a composed store) | **RELEVANT** |
| 2 | **PALM** — Personalized, Aligned, Long-term Memory for AI systems | https://palm-neurips-2026.github.io/ | **Aug 24, 2026 · 11:59pm AoE** | 9pp full / 4pp short | Double-blind, incl. code & linked material | Non-archival; accepted public on OpenReview | Paris, Dec 12 or 13 | **V5** (lifetimes/decay/deletion in a long-term memory), **V6** (memory-evaluation audit), **V1** partial | **RELEVANT** |
| 3 | **TTCL** — Towards Test-Time Continual Learning Agents | https://ttcl-agents.github.io/ | **Aug 29, 2026 · 11:59 PM AoE** | 4–9pp excl. refs/appx | Double-blind | Non-archival; accepted public on OpenReview | Atlanta, Dec 12 or 13 | **V1** (test-time compute/adaptive inference), **V5** (test-time × continual = forgetting/lifetimes) | **RELEVANT** |
| 4 | **AXIOM 2026** — Foundations of Efficient Deep Learning | https://axiom-neurips2026.github.io/ | **Aug 29, 2026 · 11:59 PM UTC-0** ⚠ not AoE | 4pp excl. refs+appx (+1pp Grand Challenge) | Double-blind | Non-archival; accepted posted on workshop site | Paris, Dec 12 | **V1** (efficient/adaptive inference with theory), **V3** (reversible O(1)-memory training = compression/efficiency), **V2** partial (theory-of-DL) | **RELEVANT** |
| 5 | **TAE (Trust-AI-Eval)** — Can We Trust AI Evaluation? | https://tai-eval.github.io/ | **Aug 29, 2026 (AoE)** | not stated on site | not stated | not stated | Sydney, Dec 11 or 12 | **V6** (matched-byte uniform audit protocol incl. our own store losing = "evaluation as an object of study") | **RELEVANT** |
| 6 | **Sim2Science** — ML with Imperfect Scientific Models | https://www.sim2science.com/ (CFP: /cfp) | **Aug 29, 2026 · 23:59 AoE** | 5pp (2pp Tiny), excl. refs | Double-blind | Non-archival, no proceedings; accepted **linked** on site (opt-in) | Paris, Dec 12 or 13 | **V2** partial (exactly-solvable budgets = physics models known to be imperfect), V6 partial | **PARTIAL** |
| 7 | **AI for Science: Verification in the Age of AI Scientists** (8th AI4Science) | https://ai4sciencecommunity.github.io/neurips26.html | **Aug 29, 2026 AoE** | 4–8pp, unlimited refs/appx | not stated | "nonarchival"; accepted posted on workshop site | Sydney, Dec 11 or 12 | **V2/V6** partial (verification/certification framing fits the audit + solvable-budget shorts); V1 partial | **PARTIAL** |
| 8 | **BeNTo** — Beyond Next-Token Prediction: Diffusion & Flow Models | https://bento-neurips.github.io/ | **Aug 29, 2026 · 23:59 AoE** | 4 or 8pp excl. refs/appx | not stated | not stated | Sydney, Dec 11 or 12 | **V5/V1** partial (the closest live generative-modeling/sampling venue found — Track 1 is "generative modeling, probabilistic inference, optimal transport"; **it is diffusion/flow-framed, not EBM-framed**) | **PARTIAL** |
| 9 | **Efficient and On-Device AI Agents** | https://efficient-ondevice-ai-agents.github.io/ | **Aug 29, 2026 · 11:59 PM AoE** | 4pp short / 9pp long + unlimited refs | Double-blind | Non-archival; "may be submitted to venues with archival proceedings" | Sydney, Dec 11–12 | **V1** partial (compute-adaptive reads under resource constraints), **V3** partial (O(1)-memory training) | **PARTIAL** |
| 10 | **Interpretability for Discovery** | https://interpretability4discovery.github.io/ | **Aug 29, 2026 · 11:59 PM AoE** | not stated ("Current submission requirements are tentative") | not stated | not stated | Atlanta, Dec 12 or 13 | **V2** partial ("what a trained model obeys" = turning what models encode into testable knowledge), V6 partial | **PARTIAL** |
| 11 | **Interpretability as a Science** | https://interpscience.github.io/ | **NOT PUBLISHED on landing page** (CFP page linked at /cfp, not fetched) | not stated | not stated | not stated | Sydney | **V2/V6** partial (rigorous-empirical-science framing; V6's uniform-protocol position case) | **PARTIAL** |
| 12 | **AI for Meta-Science** | https://ai4metascience.org/ | **Aug 29, 2026 (AoE)** | 4pp technical / 8pp position (excl. refs) | not stated | not stated | Paris, Dec 2026 | **V6** partial — one of only two live venues found with an explicit **position-paper track** | **PARTIAL** |
| 13 | **Child Safety in AI** | https://childsafety-ai.github.io/ | **Aug 29, 2026, AoE** | up to 4pp excl. refs | "Submissions must be anonymous" | Non-archival; "may be submitted to other venues" | Atlanta, Dec 12 or 13 | **V5** marginal — the only live NeurIPS-2026 CFP found listing "Machine unlearning and concept erasure with strong guarantees" as a topic | **PARTIAL (marginal)** |
| 14 | **TS4H — Time Series for Health** | https://timeseries4health.github.io/ | **Aug 19, 2026** ⚠ **NOT RE-VERIFIED this pass** (from the program's 2026-07-19 scout) | 4pp | double-blind | non-archival | Sydney | Not a topical fit; retained only as the **earliest known 2026 deadline** datum | **PARTIAL (calendar only)** |

**Not-a-workshop but current-cycle facts encountered (context only, both already past):** NeurIPS 2026 main track has a renamed **Evaluations & Datasets (ED) Track** (https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets; abstract May 4 / paper May 6, 2026 AoE — **passed**) and a **Call for Position Papers** (https://neurips.cc/Conferences/2026/CallForPositionPapers — page seen in search results, **not fetched; flagged**).

**Deadline distribution as published (no interpretation):** Aug 19 (TS4H, unre-verified) · **Aug 22 AoE (NeurReps)** · **Aug 24 AoE (PALM)** · Aug 29 (nine workshops; AXIOM's is UTC-0, the rest AoE) · one unpublished (Interpretability as a Science). Venue-suggested date Aug 29 AoE; mandatory notification Sep 29 AoE.

---

## Confidence & gaps

**Verified this pass (primary source, own site/page, 2026-08-05):** ICLR 2027 dates + dual-submission + anonymity + page limits + new desk-reject vectors; NeurIPS 2026 WorkshopsGuidance + CallForWorkshops; NeurReps three-track structure incl. the PMLR archival track and 30% rule; PALM, TTCL, AXIOM, Sim2Science, AI4Science, Efficient On-Device Agents, Child Safety CFP facts; absence of the central workshop list; absence of an ML4PS 2026 page; the "early August + blog post" newsletter sentence.

**Single-sourced / flagged:**
- **NMI (§4)** — could not fetch (SSO redirect); text is from the search index of the portfolio-wide policy. **Re-verify in a browser before it is load-bearing.**
- Every verbatim string in this report was extracted by **one tool (WebFetch) on one retrieval** — per the CM-17 precedent, treat each as single-tool-sourced. The ICLR dual-submission sentence is the exception: it is **cross-verified** against the program's own 2026-07-04 capture of the ICLR-2026 text and is character-identical.
- ICLR 2027 "submissions not publicly visible during active review" is WebFetch's rendering of schedule prose, not a quoted sentence.
- NFAM/associative-memory-at-ICLR datum is search-snippet-sourced.
- TS4H's Aug 19 deadline is from 2026-07-19, **not re-verified**.

**Known gaps / what to search next:**
1. **ICLR 2027 Reviewer Guide** (is there any general "do not search for the paper" instruction?) and **`/Conferences/2027/AIPolicyForAuthors`** — both unfetched.
2. **ML4PS 2026** — existence and CFP; the single most consequential unknown for V2/V3 given the prior cycle's centrality.
3. **The central accepted-workshop list** — due "early August" per the newsletter; a re-sweep the moment the blog post lands will surface workshops that have not self-published (an EBM/generative or physics-of-ML workshop may exist and be invisible today).
4. Missing per-workshop fields: TAE (pages/blinding/archival/dual), BeNTo (blinding/archival/dual), AI4Science (blinding), Interpretability×2 (everything), AI4Meta-Science (blinding/archival), Efficient On-Device (public posting), Child Safety (public posting).
5. NeurReps **Findings** track's archival status is implied non-archival by the page layout but **not stated verbatim** — confirm before relying on it.
6. Whether any workshop publishes a **camera-ready posting date** (none found) — determines when de-anonymized shorts become public relative to ICLR's Dec 16, 2026 de-anonymization.

Git footprint: none (read-only; this file only).

## Proposed handover updates (for the Hub)

1. **Replace the ICLR entry entirely.** The file's "ICLR 2027 unannounced; ICLR-2026 text used as proxy" is **superseded**: the ICLR 2027 CFP + Author Guidelines are live (fetched 2026-08-05). Record: **abstract Sep 18, 2026 AOE · paper Sep 25, 2026 AOE · reviews released Nov 5 · decisions Dec 16, 2026** and note the earlier ESTIMATE ("abstract ~Sep 19–27 / paper ~Sep 24–Oct 1") is now retired.
2. **Keep the dual-submission fact, but tighten its scope.** The ICLR-2027 text is **verbatim identical** to the 2026 text already on file — *and* the exemption is scoped to "venues that do not have publication proceedings." Add the standing caveat: **a workshop track with proceedings (PMLR) is not covered.**
3. **Amend the "non-archival venue-wide" claim.** The NeurIPS guidance sentence still holds verbatim, but **NeurReps 2026 runs an archival PMLR Proceedings track with a 30%-new-material rule in both directions.** "NeurIPS workshop ⇒ non-archival" must become a **per-track** check in the checklist.
4. **Retire the "one short per workshop (ML4PS precedent)" rule as a current-cycle fact.** No 2026 workshop found repeats ML4PS's strict prohibition; the live analogue is Sim2Science's "we discourage parallel submission … to multiple NeurIPS 2026 workshops." Re-instate only if/when an ML4PS 2026 CFP appears.
5. **Record the deadline set as published** (no schedule interpretation): NeurReps **Aug 22 AoE** and PALM **Aug 24 AoE** are the earliest *target-relevant* published deadlines found; AXIOM is Aug 29 **UTC-0** (not AoE); nine workshops sit on Aug 29; mandatory notification Sep 29 AoE venue-wide.
6. **Add three new desk-reject-checklist items** the program did not have: reviewer-registration requirement (desk reject if no author registered as reviewer), OpenReview profile correctness, and the reciprocal-review load for authors on ≥3 papers; plus the AI Policy for Authors as an unchecked item.
7. **Add the de-anonymization fact:** ICLR 2027 de-anonymizes **all** submissions — accepted, rejected **and withdrawn** — after notification (Dec 16, 2026), and withdrawn papers remain publicly hosted.
8. **Calendar checkpoint:** re-spawn this scout when the NeurIPS blog posts the accepted-workshop list (newsletter says "early August"), specifically to resolve **ML4PS 2026** and any EBM/physics-of-ML workshop that has not self-published.
9. **NMI:** preprints are explicitly not prior publication; conference-proceedings material requires a "substantial extension" plus citation of the proceedings paper; **non-archival workshop papers are not addressed by the published policy** — mark as an open question, and re-verify the NMI page from a browser (tool-blocked by SSO).

## Flags

- **F1 — NMI section is not directly fetched** (SSO redirect); search-index-sourced. Re-verify before use.
- **F2 — Single-tool sourcing:** all verbatim strings are one WebFetch retrieval each (CM-17 precedent), except the ICLR dual-submission sentence (cross-verified against the program's own 2026-07-04 capture).
- **F3 — Coverage is structurally incomplete:** the central accepted-workshop list is unpublished, so §6 can only contain workshops that self-published a page by 2026-08-05.
- **F4 — ML4PS 2026 status UNKNOWN** (`/2026/` → 404). Do not assume it runs; do not assume it does not.
- **F5 — TS4H Aug 19 deadline is 2026-07-19 data, not re-verified this pass.**
- **F6 — AXIOM's deadline is UTC-0, not AoE** — an easily-missed ~12 h difference.
- **F7 — The ICLR "reviewers advised not to search" quote is challenge-leaderboard-specific.** It is **not** a general anti-search policy; quoting it as one would be a misrepresentation.
- **F8 — No strategy content in this report** by task instruction; all deadlines are reproduced as published, never converted into a schedule.

Sources:
- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [NeurIPS 2026 Workshops Guidance](https://neurips.cc/Conferences/2026/WorkshopsGuidance)
- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops)
- [NeurIPS 2026 conference page](https://neurips.cc/Conferences/2026/)
- [NeurIPS Blog — July 2026 newsletter](https://blog.neurips.cc/2026/07/)
- [NeurReps 2026](https://neurreps.org/)
- [PALM @ NeurIPS 2026](https://palm-neurips-2026.github.io/)
- [TTCL @ NeurIPS 2026](https://ttcl-agents.github.io/)
- [AXIOM 2026](https://axiom-neurips2026.github.io/)
- [TAE (Trust-AI-Eval) @ NeurIPS 2026](https://tai-eval.github.io/)
- [Sim2Science @ NeurIPS 2026](https://www.sim2science.com/) · [CFP](https://www.sim2science.com/cfp)
- [AI for Science @ NeurIPS 2026](https://ai4sciencecommunity.github.io/neurips26.html)
- [Beyond Next-Token Prediction (BeNTo)](https://bento-neurips.github.io/)
- [Efficient and On-Device AI Agents](https://efficient-ondevice-ai-agents.github.io/)
- [Interpretability for Discovery](https://interpretability4discovery.github.io/)
- [Interpretability as a Science](https://interpscience.github.io/)
- [AI for Meta-Science](https://ai4metascience.org/)
- [Child Safety in AI](https://childsafety-ai.github.io/)
- [ML4PS (2025 site; no 2026 page)](https://ml4physicalsciences.github.io/)
- [NeurIPS 2026 Evaluations & Datasets Track CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- [Nature Machine Intelligence — Preprints & Conference Proceedings (not directly fetchable)](https://www.nature.com/natmachintell/editorial-policies/preprints-conference-proceedings)
- [NFAM @ ICLR 2026 (associative memory; prior/other cycle)](https://nfam2026.amemory.net/)
