# V1 revision pass 1 — the Head-approved worklist, and nothing else

**Scoped by the V1 Shorts Advisor, 2026-08-26, on the Head's line-item approval** across three exchanges. Basis: `.claude/outputs/v1-claims-currency-audit.md` (REVIEWED-ACCEPTED, Advisor-verified on disk).

**Agent:** `paper-writer` · **Output file edited: `.claude/NIPSsubmission/v1-ttcl/submission.tex` — that one file only.**
**Report:** `.claude/outputs/v1-revision-1.md` · **Deliverable #1: `BUILD-NOTE-R1.md`** in the `v1-ttcl/` folder.

⚠ **Toolset note:** the live agent registry lists `paper-writer` with **Bash**, which differs from charter §8 rule 9 (recorded at Add.54). **If you can run `pdflatex`, build and report the page split. If you cannot, say so plainly and stop at the edits** — the Advisor builds, as at Add.54. ⛔ Never fake a build.

---

## 0. ⛔⛔ THE BOUNDING RULE — read this before anything else

**Every edit in this pass is ENUMERATED below. Anything not enumerated is FORBIDDEN, however beneficial it looks.**

This is not boilerplate. A previous pass in this program consolidated an entire audit inventory into its worklist, executed it faithfully and additively, and **was reverted in full** because the volume read as a rewrite (Add.60a). The Head has since approved **this** list item by item. ⇒ **Your job is exactly these items. If you notice a defect that is not on the list — and you will, the audit found 27 and only these are approved — it goes in your report's findings section, never into the file.**

⛔ Specifically forbidden however tempting: typo fixes, grammar, capitalisation, rewording, re-ordering, re-wrapping, terminology harmonisation, and any touch to a number, caption, label or heading not named below.

---

## 1. Sources, and which file is which

| file | role |
|---|---|
| `.claude/NIPSsubmission/v1-ttcl/submission.tex` | ⭐ **THE TARGET.** The only file you edit. |
| `.claude/papers/v1-short/draft.md` | **ANCESTOR SOURCE, READ-ONLY.** Most restorations below are verbatim lifts from here. ⛔ **Never edit it** — it is the canonical archive. |
| `.claude/claims_matrix.md` · `.claude/negative_results.md` | **Registry, READ-ONLY.** The fold's approved wordings live here and bind **verbatim**. |

⛔ **`draft.md` is deliberately NOT synced in this pass.** The audit found the two files are *each stale where the other is clean* (V1-17), and a one-directional sync re-introduces a defect. The Head's instruction is that **the results must be in the `.tex`** — that is this pass's scope.

⭐ **Method that worked and is expected here** (Add.60's exemplar): **scripted, assertion-guarded, single-occurrence replacements, each asserting `count == 1` before writing.** A regex that matches 0 or 2 sites must fail loudly rather than write. Every added token carries an **ancestor** (a `draft.md` line, a registry clause, or an `outputs/*` row) recorded in the build note. ⛔ **The orphan list must be EMPTY.**

---

## PART A — the five contradiction fixes (one line each)

| # | site | edit |
|---|---|---|
| **A1 / V1-01** | `submission.tex:80` | Replace the §3.2 heading with `draft.md:78`'s **verbatim**: *"The discriminating experiment: squeeze reach is priced, the wormhole is flat-priced — and only the wormhole hands back a receipt."* ⛔ The current heading asserts the MF-B-falsified *"reach steps then collapses"* framing that the abstract, the table footnote (`:98`) and the body (`:102`) all replaced with the pricing law. |
| **A2 / V1-02** | `submission.tex:40` | `The router's map` → **`An untrained state-replacing map`** (`draft.md:31`'s wording). ⛔ MF-A renamed this object everywhere to kill the decision/transport conflation; every other tex site already complies. |
| **A3 / V1-06** | `submission.tex:250` (App D, the N2 entry) | Replace the clause *"Hopfield dominant at 500 ep (**provisional — under-trained map**)"* with `draft.md:448`'s settled form, condensed: the 500-ep map was an **under-training artifact**; at convergence the gate reverses Hopfield **only on clean/correlated cues at kv ≤ 64**; the barrier is an **epoch-budget wall, not a capacity wall**; tally **6/15**. ⛔ §4.3 of the same document files this result as **settled**; the appendix currently calls it provisional. |
| **A4 / V1-11** | `submission.tex:76` | Restore `draft.md:72`'s parenthetical so **$X$ is defined**: *"(the theory note's coset content, **$X$ the broken generator**)"*. ⛔ $X$ currently appears in the **abstract** (`:26`) with no referent anywhere in the base. |
| **A5 / V1-12** | `submission.tex:56` and `:38` | At `:56` restore `draft.md:49`'s attribution — *"The theory note proves (**Prop-A2**) that…"*. At `:38` restore `draft.md:27`'s provenance tag — `[proven.]` → **`[proven; theory note + Anonymous 2026.]`**. ⛔ The causal-box inclusion $Q_T \subseteq C_T$ is the paper's central theorem and is currently a bare assertion. |

---

## PART B — the two number fixes

**B1 / V1-05 — the savings range.** The paper prints **`9–10×`** at four sites under the word *settled*. **CM-8's approved wording is `"6–10× savings" = intra-CLU rationing`** (Advisor-verified verbatim at `claims_matrix.md:556`), and the settled 198-job grid measures **9.9× (kv32) / 9.5× (kv64) / 6.2× (kv96)** — it **falls with load**.

⇒ **Head ruling: quote the curve.** Replace with **`9.9/9.5/6.2× across kv32/64/96`** (or the nearest phrasing the sentence admits) at each of the four sites: `:43`, `:151`, `:153`, `:157`.
⛔ **NOT a find-replace.** `:157` is a figure caption and `:247` is an appendix stub — read each site and fit the phrasing to it. This is **CM-22(bb)**: *quote the curve, never the endpoint.*

**B2 / V1-04 — the two kv32 numbers.** In one paragraph (`submission.tex:140`) the paper states *"the robust payoff is rationing (**9.9×** intra-CLU)"* and, ~40 words later, *"kv32 gives **1.14 ± 0.06×**"*. **Both are correct numbers from different epoch budgets** — `9.9×` is §4.3's 2000-ep grid, `1.14×` is §4.1's own 400-ep measurement — and nothing reconciles them.

⇒ **Attach the flag in-sentence:** `9.9×` → **`9.9× at kv32/2000 ep (§4.3's grid)`**, so the two figures are visibly differently-flagged.
⛔ **The fix is a flag, never a deletion. Do not remove either number.**

---

## PART C — the claim restatement, and the two mandatory sentences

**C1 / V1-10 — restate §5's advantage claim non-comparatively (Head ruling).** `submission.tex:168` currently reads *"…delimit where this buys **a real ML advantage** (escalatable rationing) and where a cheaper black box wins (routing)."*

⛔ The comparison set behind that advantage is **the paper's own full-budget CLU and a one-shot Hopfield** — no matched-compute floor, no matched-bytes exemplar store, anywhere.

⇒ **Head ruling: restate it as an empirical proof of concept with the comparison deferred.** The claim must become **non-comparative**, and the deferral is stated in the same breath — extra compute buys accuracy **on this memory**, offered as a proof of concept, with the matched-compute / matched-bytes comparison named as **future work**.
⛔⛔ **This is a RESTATEMENT, not an addition.** Adding a deferral clause beside an unchanged *"advantage"* claim is self-contradictory — a referee reads an asserted advantage and then reads that we know we have not tested it. **The word doing the damage is "advantage."** Remove it and the deferral becomes coherent.

**C2 / V1-08 — the measured score sentence (Head: IN).** Absent from both files (`external benchmark` 0, `headline metric` 0, positive-controlled).
⇒ Insert **once**, at the head of §4 (`:133–134`, the *"Reporting grade: evidence"* paragraph — the paper's own perimeter statement), in the approved form: **external benchmarks won on their own headline metric = ZERO.** ⛔ **Approved wording binds verbatim** — you may position it, never paraphrase it (Add.30's boundary).

**C3 / V1-09 — the §A20.5 substrate-scope sentence (Head: IN).** Absent from both files (0 hits, positive-controlled).
⇒ Insert **once**, at §5's *"Scope (stated, not buried)"* paragraph (`:170`), which already carries the oracle-placement and laptop-CPU scopes. Verbatim: **"these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, φ-bytes ledgered."**

---

## PART D — the completeness restoration (Head ruling: all results in the base)

⛔ **Appendices A and C are currently STUBS that defer to `draft.md`** — a file no reviewer can open — while `submission.tex:83` tells the reader *"Table 2 (Appendix C) gives the full grid."*

| # | restore | from | into |
|---|---|---|---|
| **D1 / G-1** | **All five flag-provenance tables** — A.1 §3 certificate stack · A.2 §4.1 gate + Hopfield transfer · A.3 §4.2 routing · A.4 §4.3 regime map · A.5 §3.2.1 payoff + B.2 BIBO | `draft.md:193–282` | replacing the stub paragraph at `submission.tex:193` |
| **D2 / G-2** | **All four grids** — C.1 reach battery + per-arm certificates, C.1.b latch cloud · C.2 routing grid · C.3 Hopfield iteration parity · C.4 regime map full grid (a/b/c) | `draft.md:328–442` | replacing the stub at `submission.tex:247` |
| **D3 / G-3 / V1-07** | **The N2b entry — THE NOISE WALL** (Head: restore, descope later if needed) | `draft.md:449` | the App-D list at `submission.tex:250` |

⭐ **Restore VERBATIM, converting markdown tables to LaTeX faithfully.** ⛔ Zero new numbers. Zero rewording. Every restored value must match its `draft.md` ancestor **to the digit** — value, precision, ±, seed count, units.

⚠ **On D3:** CM-8 states verbatim that the noise wall *"**travels with every reversal claim**"*. It is not a decoration — it is the scope of §4.3's only external positive.

⚠ **Expect the page count to grow.** ⛔ **Report it; do not optimise it.** The Head condenses personally into `pj_sub.tex`, and this pass's job is completeness, not compression.

---

## PART E — the fold (Head-approved: items 2, 5, 1 — in that order of value)

All three are **C1/C2 registry-banked and quotable today**. ⛔ **None is C3**, so charter A5.6's pending-rule fires on none of them.

### E1 — item 2: the R3-native TIE (**CM-23(r)**, **N103**) ⭐ the load-bearing one

**Approved form, binds VERBATIM** (`claims_matrix.md:574`, clause (r)): *"in the R3-native regime — pixel-space corruption of a φ-addressed store, where **no mask oracle can be constructed** — the confidence-gated anytime read **ties** the matched-compute feedforward-in-φ floor (**+0.8 ± 1.6 pp, 6 seeds**)…"*, auto-stopping at **1.40 ± 0.20×**, beating every mechanism control (**+4.6 ± 2.2 pp over an equal-energy random kick, 6/6 seeds**).

- ⛔⛔ **The word is "ties". It may NEVER read "wins."** CM-23(r) binds this explicitly.
- ⭐ **Why it leads:** the genuine-win bar (charter §4.1) names *"the R3 anytime protocol"* by name, and this is it — a **matched-compute** result. It is the one V1 claim that clears that bar.
- **Placement:** §4, as a fourth pillar, and it supplies the measured replacement for the sentence C1 restates.
- ⚠ **6 seeds** — and V1 quotes **5 seeds** throughout §3/§4. See the seed rule below.

### E2 — item 5: the trilemma's third corner (**CM-23(y)**)

**Approved form, binds VERBATIM** — the matrix's own designated *"corner to lead with"*: *"dropping amplitude-independent latency **is** the compute-adaptive-read dial — a faded memory costs more integration steps to read, which is a physical, measurable statement a timestamped row cannot make."*

- **Placement:** §5, *"The position, restated"* (`:167`) — it is a *position* sentence, which is V1's genre.
- ⛔⛔ **The hard never-quote at the end of CM-23(y) binds in full: *"Both proposed fixes are REFUTED and neither may be described as available (N119)."*** ⇒ you may state what dropping latency **means**; you may **never** present the **gated-stiffness channel** as available.
- ⚠ `latency` and `trilemma` are both **0** in the base, so there is no existing sentence to contaminate — the trap fires only on your new prose.

### E3 — item 1: the retry mechanism attribution (**CM-23(g)**, **N90**)

**Approved form, binds VERBATIM** (`claims_matrix.md:574`, clause (g)): *"the lift is the **directed** symplectic re-launch: equal-energy random kicks and ensembles of k independent restarts are **dead flat in all 8 cells**, while confidence-gated CLU retry rises **+6.6…+76.2 pp**"*, auto-stopping at **×1.2–1.8**; the gate is load-bearing (**ungated retry-all collapses 0.96 → 0.004 at 9× compute**).

- ⛔ **This is the structural one, and you must understand what it does before writing it.** V1 currently contains **no retry-experiment content at all** — Advisor-verified: `random kick`, `ensemble`, `restart`, `ungated`, `directed`, `re-launch` are **0 in both files**. Its only retry material is the **N1 null** and **toy-EBM theory** in §5/App F, which states *"no runs on trained CLU checkpoints are claimed."*
- ⇒ **Folding this converts App F.6 from a *specified future experiment* into a *reported* one, and changes §5's reporting grade.** ⛔ **App F.6's "not run" language must be updated in the same edit**, or the paper will simultaneously specify and report the same experiment.
- **Placement:** §5 design-rule 2 (`:180`, *"Mix; do not rely on the squeeze alone"*) — the controls-that-die **are** that rule's empirical content.

### ⛔ Riders that bind every sentence in Part E

1. ⛔⛔ **N95's placement obligation FIRES the moment E1 or E3 lands.** **CM-23(l)/N95 must be stated in the same section as the retry claims** — *a decision-grade NO **with headroom present**, so the retraction cannot be blamed on saturation.* ⚠ **Read N95's own ⟲ HEAD RULING (2026-07-25) at `negative_results.md:899–906` before drafting it:** the *verdict wording* was superseded — ⛔ **do not quote *"R3 failed"*; quote the corrected status.** The measured numbers stand unchanged.
2. ⛔ **The flat-curve disjunction (N308, C2W11) binds any anytime-curve sentence.** *"A flat anytime curve ⇒ the store carries nothing"* is **refuted as an inference**; the standing replacement is *"carries nothing **OR** cannot be addressed."* The separating control took the same store, same physics, same budgets from **0.0223 → 0.8219 → 0.8711** under oracle addressing. ⚠ **That is a three-point curve — quote it as a curve, never as "0.02 → 0.87"** (CM-22(bb)). **N199's aphorism may no longer stand alone.**
3. ⛔ **Seed counts must not merge.** CM-23's own scope line reads *"(r) 6 seeds headline, τ-sub-claim 1 seed."* E1 is **6 seeds**; V1's existing §3/§4 material is **5 seeds**. ⇒ the paper will carry **two different n's**, correctly. **Every folded claim names its own seed count in-sentence.** ⛔ Never quote a folded result under V1's existing "5 seeds."
4. ⛔ **Forbidden forms, checked explicitly before you submit:** *"beats feedforward via test-time compute"* (absolute dominance is **RETRACTED** — the NN floor beats CLU-gated in **all 8 cells**, −3.5…−42.2 pp) · the anytime curve as a **uniqueness** claim (it is a **shape** claim and the venue is occupied — DEQs / EBTs / Titans) · *"the anytime read wins"* (it **ties**) · *"9–10× savings vs Hopfield"* (intra-CLU; and see B1) · any **energy-as-superior-confidence/routing** claim (three refutations: N3, N21, N24).

⛔ **NOT in this pass** (Head: *"we'll see what to do with the rest later"*): §3.5 item 4, the τ-regime rule (**CM-23(aa)/N117**). ⚠ It carries a live hazard worth recording so nobody folds it casually later: **V1's τ is an LTT-selected exit threshold on $p_{\rm wrong}$; N117's τ is a cos₀ cutoff on a retry ladder.** They are different objects and conflating them manufactures a false identity.

---

## Deliverables

1. **`BUILD-NOTE-R1.md`** (deliverable #1, written before you report done): every edit **enumerated by item ID**, printed **before → after**, each with its **ancestor** (`draft.md` line / registry clause / `outputs/*` row). ⛔ **Orphan list must be EMPTY.**
2. **A two-way numeric check**: every number added traces to an ancestor; every number that left is accounted for. Print it.
3. **Positive-controlled sweeps** for the forbidden forms above, with the zero-hit list printed so the check is auditable.
4. **Page split before and after** if you can build; otherwise say you cannot and stop at the edits.
5. **`.claude/outputs/v1-revision-1.md`** — the report, including a **findings section** for every defect you noticed and did **not** touch.

## Acceptance criteria

- Exactly the enumerated items are changed. ⛔ **Zero unenumerated diffs** — this is checked at review by an independent `diff`.
- `papers/v1-short/**` **byte-untouched** (md5 manifest, before and after, printed).
- Every approved wording appears **verbatim**; no paraphrase of a registry string.
- Every folded claim carries its seed count and its riders **in-sentence**.

## ⚠ Grep hazards on this machine

⛔ `grep` here is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long `.tex`/`.md` lines it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative — or **hangs**. ⇒ use **`/usr/bin/grep`** for context patterns; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — **sweep per-file.** ⚠ zsh does not word-split: quote any `--include=*.tex` glob or omit it. **Positive-control every negative.**

## DIAL DECLARATION
**Dials touched: NONE.** This pass edits one `.tex` file. It runs no experiment, changes no configuration, and produces no new measurement.
