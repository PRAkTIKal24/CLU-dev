# V1 claims-currency audit — the first pass on the TTCL short

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-26** (charter `.claude/advisor-head-shorts-charter.md`, Add.94 §94.3/§94.5 item 1; driver thread `.claude/tasks/v1-shorts-advisor.md` §4).

**Agent:** `paper-referee` (chosen because this pass needs **Bash** — see §8. ⛔ `doc-curator` and `paper-writer` have no shell; assigning an execution step to them is a repeat of the Add.54 error.)
**Output:** `.claude/outputs/v1-claims-currency-audit.md`
**Writes:** that one file, and nothing else. ⛔ **This is a REPORT-ONLY pass. You never edit a paper file.**

---

## 1. Why this pass exists (read this before anything)

V2 and V5 reached the Head's hand-edit as **registry-current clean bases** — revised, cite-checked and refereed against the live registries first. **V1 has not been.** `papers/v1-short/draft.md` is **v0.4, 2026-07-19** and predates the whole of Campaign 2 — six waves, ~90 registry entries, a campaign boundary.

⇒ **V1 carries two failure modes at once, and they run in opposite directions:**
- **its own retired claims** — statements the registries have since demoted or reversed (V2/V5 never faced this);
- **riders lost in condensation** — the classic V2/V5 mode, and V1 has *already partly incurred it* (see §2).

⭐ **The program's standing finding is that a hand-rewrite keeps every number and sheds the riders.** Shedding riders from a document whose riders are *already missing* is how a claims violation ships. **That is what this pass exists to prevent, and it must run before any hand-edit or typeset.**

⚠ **Two Head rulings shape what a useful finding looks like here:**

1. **V1 will be restructured to LEAD WITH WINS**, with negatives given brief main-text mentions and fuller appendix treatment. ⇒ when you assess a passage, say explicitly **whether its qualification would survive being moved to an appendix while its claim stays in main text.** That is the exact failure the Add.76 audit measured on V2 — *"a materially less-qualified paper without a single number having been edited."* ⭐ A rider that must stay adjacent to a leading claim is a finding; a rider that travels safely to an appendix is not.
2. **A set of post-v0.4 R3 results will be folded in, and §3.5 names them.** ⛔ They are **not** speculative and they are **not** C3 — all are C1/C2 registry-banked and quotable today. **Hunting for their presence or absence is in scope and is §3.5's job.** ⛔ What remains out of scope: drafting the sentences that would carry them.

---

## 2. The two objects, and why you must read both

⛔ **They are not the same document, and the difference is a finding this pass inherits rather than discovers.**

| object | what it is | size |
|---|---|---|
| `.claude/NIPSsubmission/v1-ttcl/submission.tex` | **PRIMARY.** The build base; the Head hand-edits this into `pj_sub.tex`. | 8,886 words |
| `.claude/papers/v1-short/draft.md` | **SECOND AXIS.** The canonical markdown; the fuller document. | 14,177 words |

`submission.tex` is a verbatim copy of `papers/v1-short/draft.tex` — but **`draft.tex` is itself a ~37 % condensation of the markdown**, with the same section skeleton and thinner prose. Advisor-verified on disk, 2026-08-26:

- ⛔ **Appendices A and C in the `.tex` are STUBS that defer to the markdown** — `submission.tex:193` reads *"see `draft.md` Appendix~A for the full four tables"*, and `:247` *"See `draft.md` Appendix~C for…"*. **A submission cannot cite an internal file a reviewer cannot open.**
- ⛔ **The v0.4 MF-B fix did not fully land in the `.tex`.** `submission.tex:80` — §3.2's heading still reads *"The discriminating experiment: **reach steps then collapses**; the wormhole is flat"*. The CHANGELOG records that framing as removed from the "§3.2 heading" in **both** files. The abstract, the table footnote (`:98`) and the body (`:102`, *"the squeeze prices reach; it does not fail at the box"*) all carry the corrected **pricing law** — so **the headline section's heading contradicts its own body**.
- ⛔ **MF-A survives at one site:** `submission.tex:40`, contribution 3 — *"**The router's** map $(q,p)\mapsto(b,p)$ has $\det J=0$"*. MF-A renamed that object to **state-replacing map** everywhere precisely to kill the decision/transport conflation; every other site in the `.tex` complies.
- Occurrence counts diverge: *theory note* **9 (md) / 4 (tex)** · *MQAR* **7 / 5** · *wormhole* **68 / 50**.

**Your reporting rule, for every finding:** mark it **BOTH / BASE-ONLY / CANONICAL-ONLY**. A finding present only in `draft.md` is informational (the `.tex` already dropped that prose). A finding in `submission.tex` is **live**. ⭐ And a rider present in `draft.md` but **absent from `submission.tex`** is the highest-value class this pass can produce — it is a qualification already lost, silently, before anyone edited a claim.

---

## 3. What to audit (the worklist)

### 3.1 ⛔ The named stale objects — check every one against the drafts
These are the registry movements V1 v0.4 predates. For each: does the draft state anything inconsistent with it, and where?

- ⛔⛔ **N103 — the R3-native anytime read is a TIE, never a win.** `CM-23(r)` binds the wording to *"ties"*. This is V1's sharpest live mine: the paper is *about* test-time compute.
- **N90 / N95** — the matched-compute feedforward-NN floor dominates CLU-gated retry in every cell (N90: negative in all 8); N95: a decision-grade NO **with headroom present**. ⚠ Read N95's own ⟲ HEAD RULING annotation — the *verdict wording* was superseded 2026-07-25 while the measured numbers stand.
- **N117** — the retry threshold τ is inert when the low-confidence pool exceeds the compute budget, and it is **not** a φ-space property.
- **CM-23(b)** — the retry claim is **SHAPE-only** (the surviving half of a split whose dominance half was falsified). ⚠ The v0.4 draft has (b) but **not** (l)/(r)/(aa).
- **CM-23(l)** — the headroom NO. **CM-23(aa)** — the τ-regime rule. **CM-22(bb)** — *quote the curve, never the endpoint*.
- **CM-2 / CM-7 / CM-8 / CM-12 / CM-14** — V1's own approved-wording rows; confirm each claim still matches its approved form **verbatim**.
- **The genuine-win bar** (charter §4.1): win-by-construction is **supplementary only**; a primary claim must survive competitive baselines.
- **C1-CLOSE** (2026-07-30) and **all six C2 waves** — the draft predates them entirely.

### 3.2 ⛔ The two mandatory sentences, both ABSENT (Advisor-verified; re-verify, do not assume)
- **The measured score sentence** — *external benchmarks won on their own headline metric = ZERO* (CM-23 head). Advisor measurement: `external benchmark` = 0, `headline metric` = 0 in **both** files.
- **The §A20.5 substrate-scope sentence** — matrix §3.1. Advisor measurement: 0 hits.

For each: confirm absence with a positive control, then state **which specific claims currently stand unqualified because it is missing**, ranked by consequence. ⛔ Do not draft replacement text — quote the approved form and name its site.

### 3.3 The theory-note self-containment assessment (charter Add.94 §94.5 item 2 — Head-approved)
9 mentions in `draft.md`, **4 in `submission.tex`**. For **each mention in `submission.tex`**, classify:
- **DECORATIVE** — a courtesy pointer; removable with no loss.
- **LOAD-BEARING** — a claim, constant, derivation or *"it can be shown"* that the reader cannot reach without the note.
For every LOAD-BEARING one: name the exact object (proposition number, constant, closed form) and say whether it is derivable **from material already in the paper**. ⭐ This is the input that decides whether V1 needs a derivation appendix (the V5 route, Add.86/87) or whether Option B (Add.28, anonymized supplementary) suffices. **Report the classification; ⛔ do not recommend which.**

### 3.4 Add.10 F2 — MQAR vs criterion 4 (Head ruling: fold in here, decide later)
MQAR appears **5× in `submission.tex`**, 7× in the markdown, carrying §4.1's gate pillar, §4.2's routing pillar and §4.3's regime map. Criterion 4 (the matched-bytes exemplar-store tripwire, intervention §6) has **six confirmations**; CAMELS-US is its first measured survivor (C3 charter Add.1).

Answer **two** questions and no more:
1. **Does any MQAR-based sentence read as a CLU capability claim** (as opposed to a boundary, a negative, or a mechanism statement)? Quote each.
2. **§4.1's escalatability claim (CM-2) is the one genuinely positive MQAR pillar.** Does it survive a matched-bytes reading — i.e. would an exemplar store at matched bytes have the same *"graded compute to ration"* property the claim rests on? State what would decide it.

⚠ Advisor's prior, stated so you can **refute** it rather than inherit it: the exposure looks narrower than the mention-count suggests, because §4.2 concedes the loss to a 449-param router (N24), §4.3 concedes Hopfield is cheaper *and* more noise-robust (N32/N37), and §4.1 explicitly disclaims energy-as-signal (N3). ⛔ **If the drafts do not support that reading, say so — a premise handed to you is a claim, not a finding.**

### 3.5 ⭐ THE POST-v0.4 POSITIVES — what the draft is MISSING, not just what is stale

**Added by pre-launch amendment, 2026-08-26, on the Head's direction.** ⭐ **This section is as important as §3.1.** V1's draft predates a set of R3 results that are **registry-banked and quotable today**, and the Head has ruled the paper will **lead with them**. All seven are C1/C2 estate — ⛔ **none is C3, so the A5.6 pending-rule does not fire on any of them.**

For **each** item below: report **PRESENT / PARTIAL / ABSENT** in `submission.tex` and in `draft.md`, quote the site if present, and — if absent — name the section where it would attach. ⛔ **Report only. Do not draft the sentence.**

| # | result | authority | Advisor's prior (⛔ refute if wrong) |
|---|---|---|---|
| 1 | **The retry mechanism attribution** — the lift is the *directed* symplectic re-launch; equal-energy random kicks and k-restart ensembles are **dead flat in all 8 cells**; the gate is load-bearing (ungated retry-all collapses at ~9× compute) | **CM-23(g)**, **N90** | partially present as mechanism prose; the *controls that die* are the load-bearing half and look under-stated |
| 2 | **The R3-native TIE** — pixel-space corruption of a φ-addressed store, where **no mask oracle can be constructed**; gated anytime read **ties** the matched-compute feedforward-in-φ floor (**+0.8 ± 1.6 pp, 6 seeds**), auto-stops at **1.40 ± 0.20×** | **CM-23(r)**, **N103** | ⛔ **ABSENT — the draft predates it entirely** |
| 3 | **The SHAPE claim in its approved wording** (quoted verbatim in §3.1) | **CM-23(b)** | draft has (b)'s substance; check the wording is the approved string |
| 4 | **The τ-regime rule** — τ is load-bearing **iff** `#{cos₀ < τ} < k·step_n` | **CM-23(aa)**, **N117** | ⛔ **ABSENT — post-dates v0.4** |
| 5 | **The trilemma's third corner** — dropping amplitude-independent latency **is** the compute-adaptive-read dial | **CM-23(y)** | ⛔ **ABSENT** |
| 6 | **Gate memory-agnostic; escalatability is the CLU asset** | **CM-2** | present (§4.1) — verify wording |
| 7 | **The regime map, settled** | **CM-8** | present (§4.3) — verify scope clauses |

**⛔ Four riders that bind these, and each is a live trap:**
1. ⛔ **CM-23(y) ends with a hard never-quote:** *"Both proposed fixes are **REFUTED** and neither may be described as available (**N119**)."* Item 5 may say what dropping latency *means*; it may **never** present the gated-stiffness channel as available.
2. ⛔ **The flat-curve disjunction (N308, C2W11) now binds EVERY anytime-curve sentence.** *"A flat anytime curve ⇒ the store carries nothing"* is **refuted as an inference** and replaced by *"carries nothing **OR** cannot be addressed"* — the same store/physics/budgets went **0.0223 → 0.8219 → 0.8711** once addressing was handed to it. ⚠ **That is a three-point curve: quote it as a curve, never as "0.02 → 0.87"** (CM-22(bb)). **N199's aphorism may no longer stand alone.** Report every draft sentence that needs this rider.
3. ⚠ **CM-23's scope line reads *"(r) 6 seeds headline, τ-sub-claim 1 seed."*** Item 2 is 6 seeds; item 4 is what upgraded the τ half to 3 seeds × 2 store snapshots. ⛔ Never quote both under one seed count.
4. ⚠ **N117's Δ has two forms** — end-of-stream **−0.5460 ± 0.0226**, and **−0.4846 ± 0.0640 over all six cells** (range [−0.578, −0.405]). Whichever is used, its scope travels in-sentence.

**⛔ Forbidden forms, in descending likelihood of slipping in** (check the draft for each; **N95** is the falsifier that must be stated in the same section as the retry claims — *a decision-grade NO **with headroom present**, so the retraction cannot be blamed on saturation*):
- *"beats feedforward via test-time compute"* — **absolute dominance is RETRACTED**; the NN floor beats CLU-gated in **all 8 cells** (−3.5 … −42.2 pp).
- the anytime curve as a **uniqueness** claim — it is a **shape** claim and the venue is occupied (DEQs / EBTs / Titans).
- *"the anytime read wins"* — it **ties**.
- *"9–10× savings vs Hopfield"* — it is **intra-CLU** rationing.
- any **energy-as-superior-confidence/routing** claim — three independent refutations (N3, N21, N24).

### 3.6 ✅ Already checked — ⛔ do NOT spend a pass re-deriving these
- The draft's **`wins`/`beats` uses (3 + 3 in the markdown)** are the paper **conceding losses** to a learned router. Advisor-read in context; **not** CLU-performance overclaims.
- **Add.10's toy-retirement sweep found ZERO §A44.1 findings in V1.** The toy-compositional demotion is not a V1 issue.
- Both are settled. Confirm in passing if free; never re-litigate.

---

## 4. Deliverables

1. **The itemized worklist** — the pass's core product. One row per finding: *site (file:line) · what it says · what the registry says · BOTH/BASE-ONLY/CANONICAL-ONLY · severity · the smallest edit that would close it.* ⛔ **Name the edit; never make it.** This list goes to the Head for **line-item approval** before any revision spoke launches (§8 rule 1 of the driver thread — an over-scoped restoration once cost this program a full pass).
2. **The two-mandatory-sentence report** (§3.2) with the unqualified claims ranked.
3. **The theory-note classification table** (§3.3), per mention, in `submission.tex`.
4. **The F2 answer** (§3.4) — two questions, quoted evidence.
5. **The completeness gap** — a list of what `submission.tex` is missing relative to `draft.md`, since the Head has ruled that **all results must be present in the base**. Appendices A and C are known stubs; report anything else. ⛔ Report the gap; a writer restores it in a later pass.
6. **Your sweep log** — every grep, its pattern, its count, and its positive control (§8).

---

## 5. Acceptance criteria

- `.claude/outputs/v1-claims-currency-audit.md` exists and carries all six deliverables.
- ⛔ `NIPSsubmission/v1-ttcl/**` and `papers/v1-short/**` are **byte-untouched** — md5 every file you read, before and after, and print the manifest. *(Advisor note: no other spoke is in flight against these paths, so this criterion is satisfiable — cf. Add.43, where a byte-untouched criterion named a directory another spoke was commissioned to write to.)*
- Every registry citation is checked **on disk at the moment of use**, never quoted from this task file. ⛔ **The premises in §2 and §3 are the Advisor's measurements and are claims: re-verify them. If one is wrong, that is a finding and it outranks the rest of your report.** Three passes in this program were saved only because a spoke refused a task-file premise.
- Every negative is positive-controlled (§8).

---

## 6. ⛔ Prohibitions

1. **You never edit a paper file.** Not a typo, not a heading, not a stale claim. The Head's text is the Head's; a defect you notice goes in the list.
2. **No new numbers, no new claims, no invented registry rows.** Every number traces to `outputs/*` or a registry row, or it does not appear.
3. ⛔ **Never find-replace against registry vocabulary** (Add.10 F3, the lexical-false-friend guard). V1's *"collapse"* is used **five times in `submission.tex` and only one is the retired MF-B framing** — the others (*"collapses a capture ball"*, *"sharpened from a collapse into a pricing law"*, *"Hopfield collapses on correlated keys"*, *"the Hopfield-collapse artifact"*) are correct and must survive. Read every hit in context.
4. ⛔ **C-8 hermetic: V1 never cites, references or reasons across the sibling shorts.** Do not read V2's or V5's drafts. If you need a precedent, this task file states it.
5. ⛔ **Treat every C3-era number as PENDING** (charter A5.6) — it is quotable only if it appears in `claims_matrix.md` or a filed charter addendum. Not from a Hub log, not from a spoke report.
6. **Do not propose page cuts.** The page target is the Head's and is not settled.

---

## 7. ⚠ Two grep hazards on this machine — both will silently lie to you

1. ⛔ **`grep` here is a shell function resolving to `ugrep 7.5.0`**, not BSD grep. On bounded-context patterns (`.\{0,70\}word.\{0,70\}`) over the long lines in `.tex`/`.md` files it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative that looks like success — or **hangs indefinitely**. Both were reproduced in this estate on 2026-08-26. ⇒ **Use `/usr/bin/grep` explicitly for any context-window pattern**, and prefer plain `grep -n -i 'word'` + `grep -o 'word' | wc -l` for counts. ⚠ `grep -c` counts **lines, not occurrences** — on a markdown paragraph that is one line, five hits count as 1.
2. ⛔ **Directory-level grep over `.claude/` returns nothing** — it is gitignored. **Sweep per-file, always.**

⭐ **Positive-control every negative before you report it.** Seed a string you know is present and confirm the instrument finds it. This program has repeatedly had to defeat a false negative from a broken instrument, and on this pass a false "no stale claims found" is the worst possible output.

---

## DIAL DECLARATION
**Dials touched: NONE.** This pass reads, greps and writes exactly one report file. It runs no experiment, changes no configuration, and edits no paper file.
