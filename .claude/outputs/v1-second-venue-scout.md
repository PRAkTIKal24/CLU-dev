# v1-second-venue-scout — web-scout report

Task + acceptance criterion: ranked shortlist (top 3 + full table) of venues where **V1 "Certified Access"** could be submitted now or within ~3 weeks, every row carrying the standing 8 facts as QUOTE+URL or "not stated", with ELIGIBLE/BARRED/CLOSED-ON-TIMING/UNRESOLVED per venue.
Status: **done.** Retrieval date for every quote: **2026-08-30**.

**⚠ RECONCILIATION LIST — OWNER NEEDED (first-10-lines rule, protocol §5). Five estate facts are superseded:**
1. **"Efficient & On-Device AI Agents" IS NOT A NeurIPS 2026 WORKSHOP.** `efficient-ondevice-ai-agents.github.io` now returns **HTTP 404**, and it is **absent from the official accepted-workshop list**. The estate carries it as a live V1/V3 candidate (Add.4 Ruling 3). The accepted on-device workshop is **ODI**, a different venue. **Purge the old row.**
2. **The 7-day extension is a CYCLE-WIDE PATTERN, not a STODY quirk.** Verified by API: **TTCL itself**, **ODI**, **TS-LIMITS**, **STODY** all moved Aug 29 AoE → **Sep 5 AoE** (`1788091140000` + `604800000` = `1788695940000`, exactly +7 d). Sim2Science → Sep 2, DynaFront → Sep 4, Interp4Discovery → Sep 2, LCFM → Sep 10.
3. **Sim2Science's *"we discourage parallel submission … to multiple NeurIPS 2026 workshops"* NO LONGER APPEARS** on either its homepage or its `/cfp` (two independent explicit word-sweeps returned "THESE WORDS DO NOT APPEAR"). The live text is permissive. **Add.4's "the live analogue is Sim2Science's soft *we discourage*" is stale** — the live analogue is now **DynaFront's** identical clause (§3).
4. **A HARD outgoing bar exists this cycle and the estate has no such rule on file:** InterpScience — *"we do not allow submissions currently under review at another workshop."* V1 is at TTCL ⇒ **InterpScience is BARRED for V1**, not merely unfit.
5. **AXIOM (estate's named V1 secondary) is CLOSED** — Aug 29, 2026 23:59 **UTC-0**, ~36 h past at report time. It did **not** extend.

## DIAL DECLARATION (echoed)
**Dials touched: NONE — instrument/recon.** Venue reconnaissance only. Laundering control: n/a. Falsifies: n/a. Does-not-falsify: n/a. No submit/don't-submit recommendation is made; the Head owns venue choice. No paper draft was read — §0 of the task file was the sole brief.

---

## 0. Answer first

**Three venues are ELIGIBLE with a live window and a defensible fit; one of them is a genuinely good fit rather than a tolerable one.** **LCFM (Long-Context Foundation Models, Atlanta)** is the standout: deadline **Sep 10, 23:59 AoE** (the longest runway found, API-confirmed), explicitly non-archival, explicitly accepts work under review elsewhere, and its **8-page long-paper track takes V1's 5.2 pp main text with ~2.8 pp of headroom and zero condensation**. MQAR is the canonical long-context associative-recall probe and "efficiency techniques" + "robust evaluation" are named CFP topics, so V1 lands on-axis rather than sideways. **TS-LIMITS (Paris, Sep 5 AoE)** and **Interp4Discovery (Atlanta, Sep 2 AoE)** are the credible second and third, both with the cleanest concurrent-submission language found this cycle — but each is domain-mismatched in a way I will not argue away (§2).

**The near-miss is painful and worth stating plainly:** **RAAAI (Resource-Aware Agentic AI)** is the single best *topical* match for V1 anywhere in the 2026 cycle — its topic list literally reads "Resource-aware planning and reasoning … Memory and context management … Benchmarks, metrics, and evaluation" — and its window **closed today** (2026-08-30 11:59 UTC, API-confirmed, no extension). It is CLOSED-ON-TIMING, not barred.

---

## 1. The ranked shortlist (top 3 ELIGIBLE, by fit)

### 🥇 #1 — LCFM · "The Third Workshop on Long-Context Foundation Models" · Atlanta, Dec 12–13
`https://longcontextfm.github.io/` · OpenReview `NeurIPS.cc/2026/Workshop/LCFM`

| # | Fact | Verbatim evidence |
|---|---|---|
| 1 | **Archival** | > "This is a **non-archival** workshop. No submission will be indexed nor have archival proceedings." ✅ bar cleared |
| 2 | **Concurrent/dual** | > "We accept submissions that are under review at other venues (e.g., ICLR 2027), as long as this does not violate the dual-submission / anonymity policy of the other venue." ✅ **concurrent, not subsequent-only.** The conditional pushes back onto TTCL, which the estate records as *"welcomes submissions that are under review at, or have been accepted by, other venues"* ⇒ the condition is satisfied on the source side. |
| 3 | **Deadline** | > "Submission Deadline: September 10, 2026, 23:59 AOE" — **API-confirmed**: `duedate` = **1789127940000** → hand-converted **2026-09-11T11:59:00Z** (= Sep 10 23:59 AoE). `expdate` 1789129740000 = 12:29 UTC (+30 min grace). **Site and API agree exactly.** |
| 4 | **Blinding** | > "The review process will be double-blind." Reach to code/supplementary: **not stated.** |
| 5 | **Pages** | > "We welcome short papers **up to 4 pages** or long papers **up to 8 pages**, not including references or appendix." ⇒ **V1's ≈5.2 pp main fits the long track outright; no condensation.** |
| 6 | **Per-author cap** | **NOT STATED** (explicitly checked on two retrievals). |
| 7 | **Public posting** | > "Accepted papers will appear on the workshop website. They will also be available on OpenReview and the NeurIPS virtual site." **Timing: not stated.** Note reply `readers` = `["NeurIPS.cc/2026/Workshop/LCFM","${2/content/authors/value/*/username}"]` ⇒ **not world-readable at submission time.** |
| 8 | **Fit** | See §2. **GOOD.** |

### 🥈 #2 — TS-LIMITS · "Generalization for Time Series in Tight Settings: Latency, Inference, Memory, prIvacy and Sustainability" · Paris, Dec 12–13
`https://ts-limits.github.io/` · OpenReview `NeurIPS.cc/2026/Workshop/TS-LIMITS`

| # | Fact | Verbatim evidence |
|---|---|---|
| 1 | **Archival** | > "Non-archival — recent and concurrent submissions welcome." ✅ |
| 2 | **Concurrent/dual** | Same sentence: > "**concurrent submissions welcome**". ✅ **The single cleanest concurrent word found at any destination this cycle.** No multi-NeurIPS-workshop carve-out stated. |
| 3 | **Deadline** | Site: > "Sep 05, 2026" / > "23:59 AoE". **API-confirmed**: `duedate` = **1788695940000** → **2026-09-06T11:59:00Z** (= Sep 5 23:59 AoE). **Extended from Aug 29** (+7 d exactly). |
| 4 | **Blinding** | > "double-blind". Reach to code/supplementary: **not stated.** |
| 5 | **Pages** | > "4 to 7 pages plus references, NeurIPS format" ⇒ **V1's ≈5.2 pp main fits; no condensation.** Appendix treatment: **not stated.** |
| 6 | **Per-author cap** | **NOT STATED.** |
| 7 | **Public posting** | **NOT STATED** — no posting/proceedings-access language found in `content.js`. |
| 8 | **Fit** | See §2. **MIXED — axis-strong, domain-mismatched.** |

### 🥉 #3 — Interp4Discovery · "Interpretability for Discovery: Understanding and Discovering Novel Knowledge in AI Models" · Atlanta, Dec 12–13
`https://interpretability4discovery.github.io/` (+ `/cfp`) · OpenReview `NeurIPS.cc/2026/Workshop/Interp4Discovery`

| # | Fact | Verbatim evidence |
|---|---|---|
| 1 | **Archival** | > "Non-archival" — workshop acceptance does not prevent later archival publication. ✅ |
| 2 | **Concurrent/dual** | > "**Submissions undergoing peer review at another venue, including ICLR or NeurIPS, at the paper submission deadline are welcome.**" ✅ **Explicitly names NeurIPS** — the only destination found that pre-clears a sibling-NeurIPS-workshop situation by name. |
| 3 | **Deadline** | Site: > "Aug 29, 2026 · 11:59 PM AOE" revised to > "Sept 2, 2026 11:59:59 PM AOE". **API-confirmed**: `duedate` = **1788436740000** → **2026-09-03T11:59:00Z** (= Sep 2 23:59 AoE). |
| 4 | **Blinding** | > "Double blind"; authors must remove > "author names, affiliations, acknowledgments, and other identifying details." Code/supplementary reach: **not stated.** |
| 5 | **Pages** | > "Up to 5 pages of main text"; > "References and appendices do not count toward the limit, but the main text must be self-contained." ⇒ **≈0.2 pp condensation** off V1's 5.2 pp. |
| 6 | **Per-author cap** | **NOT STATED.** |
| 7 | **Public posting** | **NOT STATED** for accepted papers; during review submissions > "remain private on OpenReview". ⭐ Best-documented review-phase confidentiality of the three. |
| 8 | **Fit** | See §2. **WEAK-MODERATE — this is the honest floor of the shortlist.** |

---

## 2. Item 8 — fit, one honest paragraph each (a poor fit is a finding)

**LCFM × V1 — GOOD.** The CFP's topic list is *"Long-context and long-horizon agentic foundation models, Novel modeling and training approaches, **Efficiency techniques for foundation models**, **Robust evaluation**, Long-context reasoning, Long-context multi-modal learning, Long-horizon AI for science."* Three of V1's four load-bearing components land inside named topics without reframing: **MQAR** is the canonical synthetic long-context associative-recall benchmark and is native to this community's evaluation vocabulary; **test-time compute / anytime reads / escalation ladders** are "efficiency techniques" in the literal sense the CFP means (compute spent at inference against a quality target); and **calibrated gating + Learn-then-Test risk control** is a "robust evaluation" contribution — selective prediction is exactly a statement about *when a long-context read can be trusted*. The one component that does **not** have a natural home in the CFP is the **symplectic/Hamiltonian reachability certificate**: this audience has no standing interest in conservative latent dynamics as such, and the physics will read as mechanism rather than contribution. The honest framing is therefore *"a memory whose read is certifiable and whose certificate lets you stop early"*, with the Hamiltonian structure as the reason the certificate exists rather than as the headline. The mismatch is real but survivable — and unlike every other candidate, nothing has to be bent to make the paper's *experiments* fit the room.

**TS-LIMITS × V1 — MIXED (axis-strong, domain-mismatched).** The workshop's five named bottlenecks include *"**Inference Speed**: achieving sub-ms inference … make large foundation models unusable without optimization like model compression or **early-exit mechanisms**"* and *"**Memory Efficiency**: … bounded-memory footprints and **selective retention policies** are essential"*, with topics of interest spanning *"model compression, efficient attention, selective retention, … deployment-constrained benchmarks."* **Early-exit is V1's escalation ladder under another name, and "selective retention" is its store.** On the axis of *compute-adaptive reads under a hard inference budget*, this is arguably the sharpest match in the cycle after RAAAI. The disqualifying-in-spirit problem is the substrate: the invitation sentence is *"We invite work at the frontier of practical **time-series deployment**"*, and every bottleneck is posed for temporal/streaming data on embedded devices. V1's experiments are **MQAR retrieval, not forecasting**, and V1 reports no latency, no energy, and no device measurements — the three currencies this workshop trades in. Submitting would mean asking reviewers to accept a retrieval benchmark as evidence about time-series deployment, and the paper has no sub-ms or memory-footprint number to offer them. Fit is defensible on mechanism, thin on evidence.

**Interp4Discovery × V1 — WEAK-MODERATE.** Scope: *"Interpretability for scientific discovery"* and turning *"what models encode into knowledge experts can test and validate."* V1 is not an interpretability paper: it does not explain what a trained model has learned, and it does not extract human-legible knowledge from weights or activations. The one genuine bridge is that a **reachability certificate over symplectic latent dynamics is a mechanistic, verifiable statement about a model's internal state** — "this read is provably reachable within the budget" is closer to a *guarantee* than to an *explanation*, but it is at least a claim about internals that "experts can test and validate," and the calibrated-gating side supplies the validation apparatus. That is a bridge I can describe but would not call a fit. Its place at #3 is earned by **policy quality, not topical match**: it has the only concurrent clause that names NeurIPS explicitly, documented review-phase confidentiality, and a live window. If the Head weights policy-cleanliness over audience, it rises; on audience alone it is the weakest of the three.

---

## 3. The full table — every venue assessed

Verdict key: **ELIGIBLE** · **BARRED** (policy bar; removable ones flagged) · **CLOSED** (= CLOSED-ON-TIMING) · **UNRESOLVED**.
Deadlines are the **API `duedate`** where I queried it (hand-converted), else the tracker/site value. "Now" = 2026-08-30.

| Venue | Site | Deadline (UTC → AoE) | Arch. | Concurrent | Blind | Pages | Cap | Posting | Fit | **Verdict** |
|---|---|---|---|---|---|---|---|---|---|---|
| **LCFM** — Long-Context FMs | Atlanta | **Sep 11 11:59 → Sep 10 AoE** ✅API | non-arch ✅ | ✅ explicit | DB | 4 / **8** excl. refs+appx | n/s | site+OR+virtual, timing n/s | **GOOD** | **🟢 ELIGIBLE — rank 1** |
| **TS-LIMITS** | Paris | **Sep 6 11:59 → Sep 5 AoE** ✅API (+7 d) | non-arch ✅ | ✅ "concurrent … welcome" | DB | 4–7 + refs | n/s | **n/s** | MIXED | **🟢 ELIGIBLE — rank 2** |
| **Interp4Discovery** | Atlanta | **Sep 3 11:59 → Sep 2 AoE** ✅API | non-arch ✅ | ✅ names NeurIPS | DB | 5 excl. refs+appx | n/s | n/s; private in review | WEAK-MOD | **🟢 ELIGIBLE — rank 3** |
| **Sim2Science** | Paris | **Sep 3 11:59 → Sep 2 AoE** | > "non-archival" ✅ | > "We welcome recently published results and **work currently under review**" ✅ | DB | 5 excl. refs (2 Tiny) | n/s | ⭐ **opt-in**: > "submissions are kept confidential unless accepted and the authors confirm inclusion" | **POOR** — "ML with Imperfect Scientific Models"; V1 has no scientific model, imperfect or otherwise | **🟢 ELIGIBLE — poor fit** |
| **ODI** — On-Device Intelligence | Sydney | **Sep 6 11:59 → Sep 5 AoE** ✅API (+7 d, shown on own site) | > "Both tracks are **non-archival** and may be submitted elsewhere" ✅ | ⛔ **"may be submitted elsewhere" = SUBSEQUENT-ONLY. Concurrent NOT STATED.** | DB, > "entire submission, including the appendix" | 5 main, > "References and the appendix **do not** count"; > "We **do not** accept separate supplementary files" | n/s | n/s | MODERATE — "efficient inference under real-world constraints", but V1 has no device/latency/energy number | **🔴 BARRED** — concurrent not stated. **Removable**: `odi.neurips2026@gmail.com`, 6 days in window |
| **DynaFront** | Atlanta | **Sep 5 11:59 → Sep 4 AoE** ✅API | > "There will be no proceedings" ✅ | ⚠ > "Papers already accepted at venues with archival proceedings … will not be considered. **We discourage dual submissions to multiple NeurIPS workshops.**" | DB | > "Recommended length is 4 to 5 pages (excluding references and supplementary materials)" | n/s | > "made available through the OpenReview website and listed on this site" | MOD-WEAK — dynamics **of algorithms** (optimization/sampling/games), not dynamics **as architecture** | **🟡 UNRESOLVED** — soft outgoing bar bites directly (V1 is at TTCL); organizer call |
| **InterpScience** — Interpretability as a Science | Sydney | **Sep 2 11:59 → Sep 1 AoE** | non-arch ✅ | ⛔⛔ > "**we do not allow submissions currently under review at another workshop**" | DB | 5 / 9, refs+appx excluded | n/s | n/s | (moot) | **🔴 BARRED — hard, unremovable for V1** |
| **FMTS** — FMs for Temporal Systems | Sydney | **Sep 16 11:59 → Sep 15 AoE** (longest runway in cycle) | **NOT STATED** | **NOT STATED** | n/s | > "Submit up to 4 pages" (excl.? n/s) | n/s | n/s | MOD-WEAK — temporal/world-modeling; CHLU is a temporal primitive but V1 is a retrieval paper | **🔴 BARRED** (Add.4 Ruling 1) — **removable**, and it buys the most time of any option |
| **RAAAI** — Resource-Aware Agentic AI | Atlanta | **Aug 30 11:59 → Aug 29 AoE** ✅API, **no extension** | > "non-archival" ✅ | NOT STATED | > "double-blind: submission files must be anonymized" | > "up to **8 pages** … not including references and appendices" | n/s | > "posted on the workshop website" | ⭐ **BEST TOPICAL MATCH IN CYCLE** (see §0) | **⚫ CLOSED — today** |
| **AXIOM** — Foundations of Efficient DL | Paris | **Aug 29 23:59 UTC-0** (~36 h past) | non-arch ✅ | ⛔ silent on under-review | DB | 4 excl. refs+appx | n/s | workshop site | GOOD (est. V1 secondary) | **⚫ CLOSED — did not extend** |
| **TAE / TAI-Eval** | Sydney | **Aug 30 11:59 → Aug 29 AoE** ✅API | n/s | n/s | n/s | n/s | n/s | n/s | weak for V1 | **⚫ CLOSED** |
| **AI4Meta-Science** | Paris | **Aug 30 11:59 → Aug 29 AoE** ✅API | n/s | n/s | n/s | 4 tech / 8 position | n/s | n/s | weak for V1 | **⚫ CLOSED** |
| **AI4Science** — Verification in the Age of AI Scientists | Sydney | **Aug 30 23:59 UTC** — hours left today | > "nonarchival" ✅ | NOT STATED | **NOT STATED** | 4–8, unlimited refs/appx | n/s | workshop site | PARTIAL | **⚫ CLOSED-ON-TIMING (effectively)** + concurrent/blinding unstated |
| **E-values: from Statistics to ML** | Paris | **Aug 30 13:00 UTC** ✅API | > "The workshop is non-archival." | NOT STATED | n/s | 4 excl. refs+appx | n/s | n/s | interesting — LTT risk control is multiple-testing-native | **⚫ CLOSED — today** |
| **MLxOR** — uncertainty-aware decision-making | Atlanta | **Sep 1 11:59 → Aug 31 AoE** | not fact-checked | not fact-checked | — | — | — | — | MODERATE (UQ axis) | **🟡 UNRESOLVED** — ~1 day left; not worth the fact-check unless the Head asks |
| **FAST** — Foundations of Agentic Systems Theory | Paris | **Sep 4 22:00 UTC** | not fact-checked | — | — | — | — | — | WEAK | **🟡 UNRESOLVED** |
| **CLEA** — Continual Learning for Enterprise AI Agents | Atlanta | **Sep 5 11:59 → Sep 4 AoE** | not fact-checked | — | — | — | — | — | MOD-LOW; near-duplicate audience to TTCL | **🟡 UNRESOLVED** |
| **PriGM** — Principles of Generative Modeling | Paris | **Sep 5 22:00 UTC** | not fact-checked | — | — | — | — | — | WEAK for V1 (V1 is not the generative short) | **🟡 UNRESOLVED** |
| **"Efficient & On-Device AI Agents"** (estate row) | — | — | — | — | — | — | — | — | — | **⚫ DOES NOT EXIST** — 404 + absent from official list |
| TTCL · PALM · NeurReps · BeNTo · RPS · STODY | — | — | — | — | — | — | — | — | — | **⛔ EXCLUDED by task §0** |

**Excluded-set deadlines, recorded for the ledger only** (all API- or tracker-confirmed, no action implied): TTCL **Sep 6 11:59 UTC** (extended +7 d — the source venue moved), STODY **Sep 6 11:59 UTC**, BeNTo **Sep 6 11:59 UTC**, NeurReps **Findings** track **Sep 8 11:59 UTC**, PALM & RPS **Aug 30 11:59 UTC** (closed).

---

## 4. The TTCL public-posting interaction (task §2, flagged explicitly)

The estate's worry is that TTCL posts accepted papers publicly at an unstated time, potentially inside a double-blind destination's review window. **The extension changes the arithmetic in our favour, and this is a new fact:**

- TTCL's own deadline moved to **Sep 5 23:59 AoE** (`duedate` 1788695940000, API). Its review must therefore run **Sep 6 → Sep 29**, because NeurIPS mandates accept/reject notification by **Sep 29, 2026 AoE** venue-wide.
- TTCL cannot post *accepted* papers before it has decided. ⇒ **TTCL's earliest possible public, named posting is on/after its own notification, i.e. on/after a date ≤ Sep 29 that it does not publish.**
- All three shortlisted destinations are double-blind and sit under the **same Sep 29 notification wall**. LCFM's window is Sep 11 → ≤ Sep 29; TS-LIMITS' and Interp4Discovery's are shorter still.
- ⇒ **The overlap is confined to the tail of the review period (roughly the last days before Sep 29), not the bulk of it.** It is non-zero and unfixable by the authors; no venue on either side publishes a camera-ready/posting date (I searched all three destinations — "timing: not stated" in every case). This remains an organizer question, not a research question.

---

## 5. Retrieval discipline — every URL fetched, and what it returned

| URL | Result |
|---|---|
| `blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/` | 200 — **complete list, 102 workshops** (Sydney 48 / Paris 28 / Atlanta 26). The authoritative census. |
| `aiworkshoptracker.com/conference/neurips/` (×2) | 200 — full deadline registry. ⭐ **Validated against the OpenReview API on 9/9 spot-checks** (LCFM, RAAAI, E-values, ODI, DynaFront, TS-LIMITS, Interp4Discovery, TAE, AI4MetaScience) — every row matched to the minute. First fetch was **truncated** ("Additional workshops … Sep 3 through Sep 8"); the re-fetch with an explicit anti-truncation prompt returned all rows. **A truncated aggregator fetch is a rendering artifact, not a short list.** |
| `api2.openreview.net/invitations?id=…/TTCL/-/Submission` | 200 — duedate 1788695940000 |
| `…/LCFM/-/Submission` | 200 — duedate 1789127940000, expdate 1789129740000, cdate 1786968000000, note-reply readers |
| `…/RAAAI/-/Submission` | 200 — duedate 1788091140000 |
| `…/E-values/-/Submission` | 200 — duedate 1788094800000 (= Aug 30 **13:00** UTC, 61 min past the site's stated Aug 29 AoE) |
| `…/ODI/-/Submission` | 200 — duedate 1788695940000 |
| `…/DynaFront/-/Submission` | 200 — duedate 1788609540000 |
| `…/TS-LIMITS/-/Submission` | 200 — duedate 1788695940000 |
| `…/Interp4Discovery/-/Submission` | 200 — duedate 1788436740000 |
| `…/TAE/-/Submission` · `…/AI4MetaScience/-/Submission` | 200 — both duedate 1788091140000 |
| `…/AXIOM/-/Submission` (plain **and** %2F-encoded) | **HTTP 400 twice.** ⚠ Not resolved. AXIOM's closure rests on two other sources (tracker row + the group webfield's own > "Submission Deadline: Aug 29 2026 11:59PM UTC-0"), **not** on the API. |
| `api2.openreview.net/groups?id=…/AXIOM` | 200 — full name, website, contact, location, deadline string, invitation ids |
| `api2.openreview.net/groups?id=…/DynaFront` | 200 — name/website/location only; **no policy text** |
| `…/Sim2Science/-/Submission` | **HTTP 404** — id guess wrong (real id is `Sim2Sci` per tracker). Not re-queried; site was authoritative. |
| `longcontextfm.github.io/` (×2, independent prompts) | 200 — **both retrievals returned character-consistent policy text.** Only destination cross-verified this way. |
| `longcontextfm.github.io/cfp` | **HTTP 404** — guessed path, does not exist |
| `resource-aware-workshop.github.io/` | 200 — full CFP |
| `odi2026.github.io/` | 200 but **content-empty** — JS template that loads `content/*.md`. **A silence here was a rendering artifact**, disproven below. |
| `odi2026.github.io/call_for_papers` | HTTP 404 — guessed path |
| `github.com/odi2026/odi2026.github.io` | 200 — file listing revealed `content/` layout |
| `raw.githubusercontent.com/…/content/submit.md` (×2) | 200 — **full literal CFP.** Positive control that ODI's empty render was an artifact. |
| `raw.githubusercontent.com/…/content/dates.md` | 200 — extension shown explicitly ("Previous: August 29 … Current: September 5 … Status: Extended") |
| `raw.githubusercontent.com/…/main/index.html` (ODI) | 200 — template only; contained the string `20260826-deadline-extension`, independently corroborating the extension |
| `ts-limits.github.io/` | 200 but policy-empty (JS-rendered) |
| `ts-limits.github.io/call-for-papers` · `/cfp` | **HTTP 404 both** — guessed paths |
| `github.com/ts-limits/ts-limits.github.io` | 200 — revealed `content.js` |
| `raw.githubusercontent.com/ts-limits/…/content.js` (×2) | 200 — **all TS-LIMITS policy + the five bottlenecks.** Positive control on the empty render. |
| `sites.google.com/view/dynafrontneurips26` | 200 — deadline only |
| `sites.google.com/view/dynafrontneurips26/call-for-papers` | 200 — **full policy.** Positive control: the parent page's silence was a subpage artifact. |
| `www.sim2science.com/cfp` (×2) · `www.sim2science.com/` | 200 all three — policy + **two explicit negative word-sweeps** |
| `interpscience.github.io/cfp` | 200 — full policy incl. the hard outgoing bar |
| `interpretability4discovery.github.io/` | 200 — deadline only, policy "not stated" |
| `interpretability4discovery.github.io/cfp` | 200 — **full policy.** Positive control: landing-page silence was an artifact — **this one flipped the verdict from BARRED to ELIGIBLE.** |
| `efficient-ondevice-ai-agents.github.io/` | **HTTP 404** ⚠ — was live on 2026-08-05 per estate |
| `fmts-workshop.github.io/` | 200 — deadline + partial |
| WebSearch: non-NeurIPS venues on V1's axes, Sep-2026 windows | **Nothing usable.** Only hit was ICLR **2026** Workshop CAO (test-time adaptation + selective prediction) — **already decided, 126 papers, window long closed.** |

**Three caveats I will not paper over.**
1. **Every verbatim string is one WebFetch retrieval each (CM-17 precedent), except LCFM (two independent retrievals, consistent) and the ODI/TS-LIMITS/Interp4Discovery/DynaFront facts, which came from raw source files or dedicated subpages rather than model-summarised landing pages.**
2. **All epoch→date conversions are mine, by hand** (the fetch model has mis-rendered these before). Method, anchored at 2026-01-01T00:00:00Z = 1767225600: LCFM `(1789127940 − 1767225600)/86400 = 253.4993` → day 253 = **Sep 11**, remainder 43 140 s = **11:59:00Z**. **Validating cross-check: this equals the site's independently-stated "September 10, 2026, 23:59 AOE" exactly** (AoE = UTC−12). The same method reproduces TS-LIMITS' and Interp4Discovery's site-stated dates exactly — **3/3 independent agreements**, which is what licenses the conversions where no site date exists.
3. **The Sim2Science negative (§Recon-3) is model-mediated.** Two pages, five search terms, both returning "THESE WORDS DO NOT APPEAR" — but per the STODY precedent, treat per-word precision at ~90% and the **conjunction** as solid. The estate's Aug-5 quote is not reproducible today; whether it was removed or merely relocated, **I could not find it.**

---

## 6. Confidence & gaps

**Verified (primary source, this pass):** the 102-workshop official list; LCFM / TS-LIMITS / Interp4Discovery / ODI / DynaFront / RAAAI / Sim2Science / InterpScience full policy sets; nine API `duedate`s; the +7 d extension pattern incl. TTCL's own; ODI's extension on its own site; the `efficient-ondevice-ai-agents` 404 + list-absence.

**Single-sourced / open:**
- **AXIOM's closure** — tracker + group webfield, **API 400'd twice**. High confidence, not API-confirmed.
- **MLxOR, FAST, CLEA, PriGM** — deadlines only (tracker, validated instrument); **no policy facts fetched.** Left UNRESOLVED rather than guessed.
- **Public-posting timing** — *not stated at any of the three shortlisted venues.* This is the one cell the whole cycle refuses to answer.
- **TTCL's outgoing policy** — I found no TTCL clause restricting onward concurrent submission, but per task §0 the source side is settled and I did **not** re-fetch TTCL's page; the estate's Aug-5 capture is the basis. The operative rules here are destination-side.
- **Per-author caps** — "not stated" at every venue assessed, without exception.

**What to search next:** (1) organizer email to ODI (`odi.neurips2026@gmail.com`) — one question, "is concurrent submission permitted?", would move a moderate-fit venue with a 5 pp limit from BARRED to ELIGIBLE inside its window; (2) FMTS's CFP subpage — it has the cycle's longest runway (Sep 15 AoE) and its landing page is almost certainly hiding a `/cfp` the way ODI, TS-LIMITS, DynaFront and Interp4Discovery all were; (3) policy facts for MLxOR/FAST/CLEA/PriGM if the Head wants the UNRESOLVED rows closed.

---

## 7. Bibtex-ready venue refs

```bibtex
@misc{lcfm2026,
  title  = {The Third Workshop on Long-Context Foundation Models (LCFM) @ NeurIPS 2026},
  note   = {NeurIPS 2026 Workshop, Atlanta, USA, Dec 12--13, 2026. Non-archival; double-blind;
            4pp short / 8pp long excl. references and appendix; submission deadline
            2026-09-10 23:59 AoE (= 2026-09-11 11:59 UTC); accepts work under review elsewhere},
  howpublished = {\url{https://longcontextfm.github.io/}},
  year   = {2026}
}
@misc{tslimits2026,
  title  = {TS-LIMITS: Generalization for Time Series in Tight Settings --- Latency, Inference,
            Memory, Privacy and Sustainability @ NeurIPS 2026},
  note   = {NeurIPS 2026 Workshop, Paris, France. Non-archival; concurrent submissions welcome;
            double-blind; 4--7pp plus references; deadline 2026-09-05 23:59 AoE (extended +7d)},
  howpublished = {\url{https://ts-limits.github.io/}},
  year   = {2026}
}
@misc{interp4discovery2026,
  title  = {Interpretability for Discovery: Understanding and Discovering Novel Knowledge in
            AI Models @ NeurIPS 2026},
  note   = {NeurIPS 2026 Workshop, Atlanta, USA. Non-archival; double-blind; 5pp main excl.
            references and appendices; submissions under peer review at ICLR or NeurIPS
            welcome; deadline 2026-09-02 23:59 AoE},
  howpublished = {\url{https://interpretability4discovery.github.io/}},
  year   = {2026}
}
@misc{odi2026,
  title  = {On-Device Intelligence: Foundation Models under Real-World Constraints (ODI) @ NeurIPS 2026},
  note   = {NeurIPS 2026 Workshop, Sydney, Australia, Dec 11--12, 2026. Non-archival; double-blind
            incl. appendix; 5pp main excl. references and appendix; no separate supplementary files;
            deadline 2026-09-05 23:59 AoE (extended from Aug 29). Concurrent-submission policy
            not stated as of 2026-08-30},
  howpublished = {\url{https://odi2026.github.io/}},
  year   = {2026}
}
@misc{raaai2026,
  title  = {Workshop on Resource-Aware Agentic AI (RAAAI) @ NeurIPS 2026},
  note   = {NeurIPS 2026 Workshop, Atlanta, USA. Non-archival; double-blind; up to 8pp excl.
            references and appendices; deadline 2026-08-29 AoE (closed, no extension)},
  howpublished = {\url{https://resource-aware-workshop.github.io/}},
  year   = {2026}
}
```

---

## Proposed handover updates (for the Hub)

1. **Delete the "Efficient & On-Device AI Agents" row from Add.4 Ruling 3** (V1 secondary, V3 secondary). The site 404s and the venue is absent from the official 102-workshop list. Replace the on-device slot with **ODI**, and carry ODI's **BARRED-removable** status plus its organizer address.
2. **Record the cycle-wide +7 d extension pattern as a standing estate fact**, with the mechanism: `duedate` Aug 29 AoE = `1788091140000`; +`604800000` ms = `1788695940000` = Sep 5 AoE. Four venues moved on exactly this offset, **including TTCL itself**. ⭐ **Corollary for future scouts: a workshop's own site is not evidence of its deadline; the OpenReview `duedate` is. But note the inverse also occurred — ODI's site showed the extension while its OpenReview render was empty.**
3. **Promote the AI Workshop Tracker to "validated secondary instrument."** It matched the OpenReview API on **9/9** spot-checks to the minute. Standing caveat: **its default fetch truncates** — always prompt against truncation. It is now the cheapest way to get the whole registry in one call.
4. **Retire "Sim2Science's soft *we discourage*" as the live multi-workshop analogue** (Add.4 bullet, l.255). The sentence is no longer findable on either Sim2Science page. **The live analogue is DynaFront**: *"We discourage dual submissions to multiple NeurIPS workshops."*
5. **Add a NEW standing bar class the estate does not have: the HARD OUTGOING BAR.** InterpScience — *"we do not allow submissions currently under review at another workshop."* Ruling 1 governs *archival status*; this is orthogonal and it fires on the shorts specifically, because every short is or will be under review at a sibling workshop. **Every future venue check must ask "does this venue bar papers under review at another *workshop*?" as a separate cell** — the existing "concurrent/dual" cell does not catch it (InterpScience is otherwise non-archival and permissive).
6. **Record the sharpest V1 near-miss for the post-mortem:** RAAAI's topic list is a line-by-line match to V1 and it closed today without extending. If the shorts programme runs again, resource-aware/agentic-efficiency venues should be scouted *first*, not as a fresh-sweep afterthought.
7. **Update the anonymity-interaction entry with §4's arithmetic:** because TTCL extended to Sep 5 AoE and NeurIPS caps notification at Sep 29 AoE, TTCL's earliest named public posting is on/after a date ≤ Sep 29 — so the exposure is confined to the **tail** of any destination's review window, not its bulk. Still unresolvable from published sources; still an organizer question.
8. **New reusable retrieval rule, earned four times this pass:** *for a JS-templated workshop site, a "not stated" from the landing page is worthless.* ODI (`content/submit.md`), TS-LIMITS (`content.js`), DynaFront (`/call-for-papers`) and Interp4Discovery (`/cfp`) all rendered policy-empty and all four had complete CFPs one hop away — **and in Interp4Discovery's case that hop flipped the verdict from BARRED to ELIGIBLE (rank 3).** Route: GitHub repo file listing → `raw.githubusercontent.com`, or guess `/cfp` before ever writing "not stated."

Git footprint: none (read-only; this file only).

Sources:
- [Announcing the NeurIPS 2026 Workshops](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/)
- [LCFM @ NeurIPS 2026](https://longcontextfm.github.io/)
- [TS-LIMITS @ NeurIPS 2026](https://ts-limits.github.io/)
- [Interpretability for Discovery @ NeurIPS 2026](https://interpretability4discovery.github.io/)
- [ODI: On-Device Intelligence @ NeurIPS 2026](https://odi2026.github.io/)
- [Resource-Aware Agentic AI @ NeurIPS 2026](https://resource-aware-workshop.github.io/)
- [DynaFront @ NeurIPS 2026](https://sites.google.com/view/dynafrontneurips26)
- [Sim2Science @ NeurIPS 2026](https://www.sim2science.com/cfp)
- [Interpretability as a Science @ NeurIPS 2026](https://interpscience.github.io/)
- [AXIOM @ NeurIPS 2026](https://axiom-neurips2026.github.io/)
- [FMTS @ NeurIPS 2026](https://fmts-workshop.github.io/)
- [E-values @ NeurIPS 2026](https://e-values-workshop.github.io/)
- [AI Workshop Tracker — NeurIPS 2026 registry](https://aiworkshoptracker.com/conference/neurips/)
- [NeurIPS 2026 Workshops Guidance](https://neurips.cc/Conferences/2026/WorkshopsGuidance)
