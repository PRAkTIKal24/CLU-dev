# v1-v5-venue-scout — web-scout report

Task + acceptance criterion: Establish the identity and the 8 CFP facts (archival · concurrent-submission · deadline · blinding · page limit · cap · public posting · fit) for the two destination venues **STODY** (→V1) and **RPS** (→V5), every cell carrying a quote + URL, with an explicit ELIGIBLE/BARRED/UNRESOLVED verdict per venue.
Status: **done** (with two named unverifiable cells, both reported as "not stated", which is itself the finding).

**⚠ Reconciliation list — first 10 lines, per §5 corollary.** Three items need an owner at the review that accepts this report:
1. **STODY is BARRED under Add.4 Ruling 1** — its site states *no* archival status, *no* blinding, *no* page limit, *no* dual-submission policy. Only an organizer email (`stody.workshop@gmail.com`) can unblock it; the live deadline (Sep 5, 2026 AoE) leaves ~6 days to ask.
2. **RPS's submission window closed today** — deadline 2026-08-29 23:59 AoE = **2026-08-30 11:59 UTC**, confirmed independently by the OpenReview invitation `duedate` epoch. As of this report's date (Aug 30) it is past or within hours of passing. No extension is recorded (contrast: STODY's *is*).
3. **Possible shorthand mismatch on "RPS"** — the only 2026 ML venue with that acronym is **"Representations for the Physical Sciences"**, a physical-sciences *representation-learning* workshop, not a memory/retention venue. If the Head meant a different RPS, this scout scouted the wrong venue; see §4.

## DIAL DECLARATION (echoed)
**Dials touched: NONE.** Venue recon; no experiment, no claim, no artifact edited. Laundering control: n/a. Falsifies: n/a. Does-not-falsify: n/a.

---

## 0. Answer first

**STODY = "AI for Stochastic Dynamics: From Theoretical Foundations to Scientific Applications", NeurIPS 2026 workshop, Sydney** — unique candidate. Its public materials state **only dates**: archival status, dual-submission policy, blinding, page limit and posting policy are **all "not stated"**, which under charter Add.4 Ruling 1 **BARS the track**. Its deadline has been **extended to 2026-09-06 11:59 UTC (= Sep 5 AoE)**, so the bar is removable by an organizer email within the window.

**RPS = "Representations for the Physical Sciences", NeurIPS 2026 workshop, Paris** — unique candidate; it **clears both bars** (explicitly *"Non-archival"*; *"Concurrent submissions are allowed"*), is **double-blind including linked code**, and is **4pp excl. refs + unlimited appendix**. But its deadline was **2026-08-29 AoE (= Aug 30 11:59 UTC)** — i.e. **today/past** — and its scope is representation learning *for physical systems*, which is a **poor topical match** for V5's retention/deletion-in-a-memory content.

---

## 1. The decision table (items 1–7)

| # | Item | **STODY** (V1 destination) | **RPS** (V5 destination) |
|---|---|---|---|
| — | **Full name (verified)** | > "NeurIPS 2026 Workshop on AI for Stochastic Dynamics: From Theoretical Foundations to Scientific Applications" — OpenReview group header, [api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/STODY](https://api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/STODY); subtitle > "NeurIPS 2026 Workshop STODY" | > "Representations for the Physical Sciences Workshop @ NeurIPS 2026" — OpenReview group header, [api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/RPS](https://api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/RPS); site header > "Representations for the Physical Sciences · NeurIPS 2026" |
| — | **Official URL / contact / city** | https://eethanshi.github.io/stochastic-dynamics-2026/ · `stody.workshop@gmail.com` · **Sydney** (group header "Sydney, Australia"; corroborated by the NeurIPS blog workshop announcement, https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/, which lists it under Sydney). Site: > "International Convention Centre Sydney, Australia", > "December 11 or 12, 2026" | https://representations-physical-sciences.github.io/workshop-2026/ · **Paris** (NeurIPS blog announcement lists it under Paris; corroborated by AI Workshop Tracker, "Paris, France") |
| **1** | **Archival status** ⛔ | **NOT STATED.** Literal-string sweep of the site's one-page HTML for `"archival"` → *"WORD DOES NOT APPEAR"* (https://raw.githubusercontent.com/eethanshi/stochastic-dynamics-2026/main/index_one_page.html). The OpenReview group content carries no policy text. ⇒ **BARS the track under Add.4 Ruling 1.** | **Non-archival.** Verbatim from the CFP: > **"Non-archival"** (https://representations-physical-sciences.github.io/workshop-2026/). ⇒ bar cleared. |
| **2** | **Concurrent / dual submission** ⛔ | **NOT STATED.** Sweep for `"concurrent"`, `"dual"`, `"under review"` → all *"DOES NOT APPEAR"*. | **Concurrent explicitly licensed** (not merely subsequent): > **"Concurrent submissions are allowed, but authors are responsible for checking the other venue's dual-submission policy."** ⇒ the *asymmetry trap does not bite here* — the word is "concurrent", and the burden is pushed back onto the source venue's policy (TTCL/PALM both already permit it, per estate). |
| **3** | **Deadline (TZ) / passed?** | **2026-09-06 11:59 UTC = 2026-09-05 23:59 AoE — NOT passed.** OpenReview `Submission` invitation `duedate` = **1788695940000** ms (`expdate` 1788697740000 = +30 min), which I converted by hand to 2026-09-06T11:59:00Z. Corroborated: AI Workshop Tracker, > "September 6, 2026, 11:59 UTC (extended 7 days from original August 30 deadline)" (https://aiworkshoptracker.com/workshop/neurips-2026-stody/). ⚠ **The workshop's own site is STALE**: > "Paper submission deadline Aug 29, 2026 · AoE". Also on site: > "Decision notification Sep 29, 2026 · AoE", > "Camera-ready deadline Oct 9, 2026 · AoE". | **2026-08-29 23:59 AoE = 2026-08-30 11:59 UTC — today; passed or passing within hours.** Site: > "29 Aug 2026 · AoE" (call opened > "July 29, 2026 (AoE)"); notification > "Author notification **By 29 Sep 2026 · AoE**". OpenReview `duedate` = **1788091140000** ms = 2026-08-30T11:59:00Z (`expdate` 1788092940000 = 12:29 UTC) — **no extension recorded**. |
| **4** | **Blinding** | **NOT STATED.** Sweep for `"blind"` and `"anonym"` → both *"DOES NOT APPEAR"*. | **Double-blind, and it reaches code.** > "Short Papers are submitted through OpenReview and reviewed double-blind, following the NeurIPS main-track approach." Artifacts: > "Your submission must consist of a single PDF file including the main text, references, and, optionally, an appendix. **Additional files are not allowed.** You may link to **properly anonymized code and/or data repositories**." (⇒ no separate supplementary upload exists at all.) |
| **5** | **Page limit (refs/appx?)** | **NOT STATED.** Sweep for `"page"` → *"DOES NOT APPEAR"* (see §3 caveat on this particular negative). | **4pp main text; refs AND appendices excluded** — verified, not assumed: > "Short Papers are limited to four pages of main content; **references do not count toward the limit. Appendices are unlimited, but reviewers are not obliged to read them.**" Second track: > "Research Notes" — no page limit, requirements > details forthcoming. Also: > "Unlike the NeurIPS main track, this workshop does not require the NeurIPS paper checklist." |
| **6** | **Per-author / per-group cap** | **NOT STATED.** | **NOT STATED.** Sweep for `"per author"`, `"number of submissions"` → both *"DOES NOT APPEAR"*. |
| **7** | **Public posting of accepted papers (with names, when?)** | **NOT STATED** — no posting/proceedings language anywhere on the site. Only mechanical fact available: the OpenReview `Submission` invitation `readers` = the venue group + the submitting authors (`${2/note/content/authors/value/*/username}`) ⇒ **submissions are not world-readable at submission time**; nothing states what happens on acceptance. | **NOT STATED.** Sweep for `"public"`, `"posted"`, `"proceedings"`, `"camera-ready"` → **all four DOES NOT APPEAR.** Same mechanical fact: `Submission` invitation `readers` = venue group + authors only. |

---

## 2. Item 8 — topical fit (one honest paragraph each; a poor fit is a finding)

**STODY × V1 "Certified Access: Test-Time Compute on a Conservative Physics-Structured AI Memory" — MARGINAL.** The workshop's own topic list (site, one-page version) is: *stochastic analysis and control, SDEs/SPDEs, neural operators, generative models (diffusion, flow-based, score-based), probabilistic forecasting, uncertainty quantification, mathematical finance, climate modelling, and applications in physics, biology, chemistry, materials science, molecular science, robotics and autonomous systems*; the CFP framing invites work connecting > "stochastic analysis, dynamical systems, scientific machine learning, generative modelling, and AI for science." V1's centre of mass is the opposite of stochastic: it is a **conservative, symplectic, deterministic** latent memory whose selling point is *certified access under a test-time compute budget*. The two genuine contact points are (a) CHLU's **Langevin read/generation is an SDE** and sits squarely in "stochastic analysis / sampling", and (b) **calibrated gating is uncertainty quantification**, an explicitly listed topic. But the workshop's evident centre of gravity is **AI for scientific applications of stochastic dynamics**, whereas V1 uses physics as an architectural primitive for a *general* memory with no scientific application. Fit is defensible only if the abstract leads with the Langevin sampler + calibration/UQ; on the workshop's own axis (does this advance stochastic dynamics for science?) it is weak.

**RPS × V5 "Retention You Can Predict, Scope and Delete" — POOR.** Verbatim scope: > "This workshop provides a tightly scoped forum for **representation learning in physical systems**", inviting > "discussions and contributions on topics around **Self-Supervision, Transfer Learning, Sampling, and Tokenization**" and > "interdisciplinary contributions from core machine learning and every area of AI for science", with emphasis on self-supervision for unlabeled *scientific* data, transfer/OOD generalization in the *physical sciences*, simulator/experiment-in-the-loop data generation, and tokenization of *continuous physical data*. V5 is about **retention laws, scoped forgetting and deletion in a long-term memory** whose dynamics are Hamiltonian. That is physics-*inspired representation*, not **representation *for* physical science data** — the workshop wants the physical system to be the *subject*, and in V5 it is the *mechanism*. "Sampling" is the one keyword that overlaps. This is the mismatch flagged in the reconciliation list: it is plausible the Head's "RPS" shorthand denotes a different venue entirely.

---

## 3. Retrieval discipline — what I actually fetched, and what returned nothing

| URL fetched | Result |
|---|---|
| `https://api2.openreview.net/groups?parent=NeurIPS.cc/2026/Workshop` (and `&limit=200`) | 200, JSON — **paginated/truncated**: returned only 10 group ids, neither STODY nor RPS. **Do not read this as "the venues don't exist"** — the direct-id queries below resolved both. |
| `https://api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/STODY` | 200 — full name, city, homepage, contact. |
| `https://api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/RPS` | 200 — full name, homepage. |
| `https://api2.openreview.net/invitations?id=.../STODY/-/Submission` | 200 — `duedate` 1788695940000, `expdate` 1788697740000, `readers`. |
| `https://api2.openreview.net/invitations?id=.../RPS/-/Submission` | 200 — `duedate` 1788091140000, `expdate` 1788092940000, `readers`. |
| `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/STODY` (and `.../Workshop`) | 200 but **content-empty** — JS-rendered shell, nav + footer only. **A silence here is a rendering artefact, not a policy absence.** Hence the API route above. |
| `https://eethanshi.github.io/stochastic-dynamics-2026/` and `/index.html` | 200 — nav + hero only; sections not rendered into text. |
| `https://eethanshi.github.io/stochastic-dynamics-2026/call-for-papers/` | **HTTP 404 Not Found** — guessed path, does not exist. |
| `https://eethanshi.github.io/stochastic-dynamics-2026/index_one_page.html` and its `raw.githubusercontent.com` twin | 200, **real content** — dates, topics, venue. This is the source of the STODY "not stated" findings. |
| `https://github.com/eethanshi/stochastic-dynamics-2026` | 200 — repo has only `index.html`, `index_one_page.html`, `pdf_v0.pdf`, `assets/`; **no separate CFP file exists**, which independently supports "not stated". |
| `https://representations-physical-sciences.github.io/workshop-2026/` (+ `index.html`, `?q=policy`, `?q=topics`) | 200, **real content** — all RPS quotes above. |
| `https://aiworkshoptracker.com/conference/neurips/`, `/workshop/neurips-2026-stody/`, `/workshop/neurips-2026-rps/` | 200 — **secondary** corroboration of names/deadlines/cities and the STODY extension. |
| `https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/` | 200 — both workshops listed; cities confirmed (STODY Sydney, RPS Paris). |

**Two caveats I will not paper over.**
1. **The literal-string sweeps are model-mediated.** The STODY sweep reported `"page"` → *"DOES NOT APPEAR"*, which is implausible for a raw HTML file (CSS/class names). Treat each individual STODY negative as ~90%, but treat the **conjunction** as solid: three independent retrievals (rendered site, raw one-page HTML, repo file listing) all found no policy text, and the repo contains no CFP artifact for one to live in. **STODY's silence is real; the per-word precision is not guaranteed.**
2. **Epoch→date conversions in the table are mine, computed by hand** from the ms epochs. The fetch model mis-converted both (`1788695940000` → it said "May 6, 2026"; correct is 2026-09-06T11:59:00Z). Check: 2026-01-01T00:00:00Z = 1767225600; (1788695940 − 1767225600)/86400 = 248.499 d → day-index 248 = Sep 6, remainder 43,140 s = 11:59. Same method gives RPS 2026-08-30T11:59:00Z, which **exactly matches the RPS site's independently-stated "29 Aug 2026 · AoE"** — that agreement is the cross-check that validates the method, and by transitivity the STODY figure.

---

## 4. Verdicts

**STODY → 🔴 BARRED.** Reason named: **archival status not stated** (item 1), which bars under charter Add.4 Ruling 1 exactly as it closed the NeurReps Findings Track. Dual-submission status is *also* not stated (item 2 — a second, independent bar). Note this is a **removable** bar, not a substantive rejection: the venue publishes only its dates, has no CFP document in its repo, and lists a contact address (`stody.workshop@gmail.com`); its live deadline is **Sep 5, 2026 AoE**, ~6 days out, so an organizer query could resolve items 1, 2, 4, 5, 7 inside the window. Fit is independently **marginal** (§2). *(Whether to ask is the Head's call — out of scope here.)*

**RPS → 🟡 ELIGIBLE ON POLICY, WINDOW CLOSED.** Both bars are cleared with explicit verbatim policy: *"Non-archival"* and *"Concurrent submissions are allowed…"* — this is a venue that would have accepted a paper under review at PALM. **But the submission deadline is 2026-08-30 11:59 UTC (2026-08-29 AoE), i.e. today or already past, with no extension recorded on either the site or the OpenReview invitation** — so the route is closed on timing, not on policy. Two further findings ride along: the **topical fit is poor** (§2), and the acronym **may not be what the Head meant** (§0/§2). I record this as ELIGIBLE-but-closed rather than BARRED because the *policy* bars — the thing this scout was commissioned to test — are genuinely clear.

**Unresolved cells (both venues):** per-author cap (item 6) and **public-posting timing (item 7)** are "not stated" at RPS; at STODY every item except the dates is "not stated".

---

## 5. The anonymity interaction (item 7 — flagged explicitly, as instructed)

The estate already records that **TTCL and PALM both post accepted papers publicly on OpenReview and neither states *when***, with notification venue-wide ≤ **Sep 29, 2026 AoE**. Against that:

- **V1 (TTCL → STODY).** STODY's review window runs **Sep 5 → Sep 29, 2026**, which **fully overlaps** the interval in which TTCL may post V1's twin publicly under author names. **STODY's blinding is not stated**, so the severity is unknown — if it is double-blind (the NeurIPS-workshop default in this cycle: every workshop the estate checked except NeurReps-Findings is double-blind), the exposure is real and unfixable by the authors. *This is a second, independent reason the STODY organizer email is load-bearing: ask about blinding at the same time as archival status.*
- **V5 (PALM → RPS).** Here the exposure is **concrete, not hypothetical**: RPS is confirmed **double-blind**, its review window is **Aug 30 → Sep 29, 2026**, and PALM's public posting of accepted work can land anywhere inside it. RPS additionally requires **anonymized linked code** with **no supplementary upload channel** — so a V5 submission would have to point at an anonymized artifact repo while a de-anonymized PALM twin might be public on the same platform (OpenReview) during review.
- **Neither destination states a posting or camera-ready policy of its own**, so the interaction cannot be resolved from published sources on either side. It is an organizer question, not a research question.

---

## 6. Bibtex-ready venue refs

```bibtex
@misc{stody2026,
  title  = {NeurIPS 2026 Workshop on AI for Stochastic Dynamics: From Theoretical Foundations to Scientific Applications (STODY)},
  note   = {NeurIPS 2026 Workshop, Sydney, Australia, Dec 11--12, 2026. Submission deadline 2026-09-06 11:59 UTC (extended). Archival status not stated as of 2026-08-30},
  howpublished = {\url{https://eethanshi.github.io/stochastic-dynamics-2026/}},
  year   = {2026}
}

@misc{rps2026,
  title  = {Representations for the Physical Sciences Workshop (RPS) @ NeurIPS 2026},
  note   = {NeurIPS 2026 Workshop, Paris, France. Non-archival; double-blind; 4pp excl. references and appendices; submission deadline 2026-08-29 AoE},
  howpublished = {\url{https://representations-physical-sciences.github.io/workshop-2026/}},
  year   = {2026}
}
```

---

## Proposed handover updates (for the Hub)

1. **Record STODY and RPS as scouted (2026-08-30) and close the "zero facts on either venue" gap.** STODY = *AI for Stochastic Dynamics…* (Sydney); RPS = *Representations for the Physical Sciences* (Paris). Both are NeurIPS 2026 workshops, i.e. the same cycle and the same Sep 29 AoE notification wall as TTCL/PALM/NeurReps.
2. **Mark the V1→STODY track BARRED (Add.4 Ruling 1, archival not stated)** with the removal condition written into the ledger: *an organizer reply from `stody.workshop@gmail.com` stating archival status AND dual-submission policy, before 2026-09-05 AoE.* Two of the four unknowns (blinding, page limit) should ride on the same email.
3. **Mark the V5→RPS track CLOSED-ON-TIMING, not barred:** the policy bars are clean (verbatim *"Non-archival"* + *"Concurrent submissions are allowed"*), the deadline (2026-08-30 11:59 UTC) is the blocker. **Ask the Head to confirm "RPS" was the intended shorthand** — the topical fit for a retention/deletion memory paper is poor, and a mis-expanded acronym is the cheapest explanation.
4. **Add a reusable estate fact:** *the OpenReview v2 API (`api2.openreview.net/groups?id=…`, `…/invitations?id=…/-/Submission`) is the authoritative source for a workshop's live deadline* — it caught STODY's 7-day extension that the workshop's own website still contradicts. Also record the failure mode: the `?parent=` listing is truncated and returned neither venue; **only direct-id queries resolve them**, and `openreview.net/group?id=…` renders empty to fetchers.
5. **Escalate the anonymity-interaction gap from "flagged" to "open organizer question"** — it now has a *confirmed* double-blind destination (RPS) whose review window overlaps PALM's unstated public-posting date, and a destination with *unstated* blinding (STODY) overlapping TTCL's. No published source on any of the four venues resolves the timing.
6. **New checklist item for any RPS-style venue:** "no supplementary files; single PDF; linked code must be independently anonymized" — an artifact constraint that hits the shorts, not just the long paper (same class as the PALM finding already in the estate).
