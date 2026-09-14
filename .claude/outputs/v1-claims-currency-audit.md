# v1-claims-currency-audit — paper-referee report

**Task + acceptance criterion:** audit `NIPSsubmission/v1-ttcl/submission.tex` (PRIMARY) and `papers/v1-short/draft.md` (SECOND AXIS) for retired claims, lost riders and missing post-v0.4 positives against the live registries; six deliverables; paper files byte-untouched.
**Status: done.**
**Reconciliation-list owner note (protocol §5 corollary): this report IS a reconciliation list. It has 27 line items and NO owner yet. It must be converted into a named revision task at the review that accepts it, or it will rot exactly as the "2.6" retraction did.**

## DIAL DECLARATION (echoed per protocol §7)
**Dial: NONE — instrument/recon/claims-audit.** No experiment run, no configuration changed, no paper file edited. Laundering control: n/a. Falsifies: n/a.

## Byte-untouched manifest (acceptance criterion)
`md5` over `.claude/NIPSsubmission/v1-ttcl/**` (13 files) and `.claude/papers/v1-short/**` (10 files), before and after the pass: **all 23 digests identical.** Load-bearing pair:

| file | md5 before | md5 after |
|---|---|---|
| `NIPSsubmission/v1-ttcl/submission.tex` | `05586b2db9652ea3c83964cb61284466` | `05586b2db9652ea3c83964cb61284466` |
| `papers/v1-short/draft.md` | `00d703d58a15c0cb77051a9c55674684` | `00d703d58a15c0cb77051a9c55674684` |

Full 23-line manifest reproduced in §6 (sweep log). Nothing under either tree was opened for writing.

---

# VERDICT (simulated venue review)

## **REJECT** as it stands — recoverable to weak-accept by editing, not by running.

**Meta-review.** This is a careful, unusually honest position paper whose §3 certificate stack is real, exactly verified, and correctly labelled *verification grade*; three of its six contributions are boundaries or negatives and the paper says so in lead position. That craft is why the failure mode here is dangerous rather than obvious: the document is *internally* disciplined and *externally* fourteen months stale. Three defects individually block submission. (1) **The base contradicts itself in two places a reviewer reaches in the first ten minutes** — §3.2's heading still asserts the falsified "reach steps then collapses" framing that its own abstract, table footnote and body replaced with the pricing law, and Appendix D still files the regime map as *"provisional — under-trained map"* while §4.3 files it as *"settled — full grid, 198 jobs."* A referee who notices either will read the whole paper as un-proofread. (2) **The base's Appendices A and C are stubs pointing at `draft.md`, a file the reviewer cannot open** — which means the flag-provenance tables that C-7 makes *part of the paper* and the full grids §3/§4 cite do not exist in the artifact under review. Under C-10 that is not a page-budget question; it is a missing appendix. (3) **The paper's own scoreboard is missing and its test-time-compute claim has never met an external floor.** The mandatory score sentence (*external benchmarks won on their own headline metric = ZERO*) and the §A20.5 substrate-scope sentence are both absent from both files, positive-controlled; and §5 asserts §4 shows *"where this buys a real ML advantage (escalatable rationing)"* on a comparison set containing only a full-budget CLU and a modern Hopfield — never the matched-compute feedforward floor that N90/N95 have since measured dominating CLU-gated retry in 8/8 cells, nor the R3-native tie of N103. That is the genuine-win bar, and V1 is the one short whose thesis *is* test-time compute.

Underneath those, one number is wrong in the reviewer-checkable direction: the paper quotes the intra-CLU rationing figure as **"9–10×"** in four printed places under the word *settled*, while the settled 198-job grid it cites measures **9.9× / 9.5× / 6.2×** across kv32/64/96 and CM-8 canonises **"6–10×"**. A load-dependent number quoted as a narrower constant is CM-22(bb) exactly. And in the same §4.1 paragraph that quotes "9.9× intra-CLU" the paper also states "kv32 gives 1.14×" — two intra-CLU savings figures for kv32, nine-fold apart, one sentence apart, with no reconciliation. That is the N36 defect class, live, in the base.

**The Advisor's premises were re-verified on disk and three of them are wrong or too weak** — see §A. The most consequential: CM-23(g) is not "partially present"; the entire retry-mechanism estate is **wholly absent** from both files, and CM-23(b)'s approved string is absent too, contrary to the charter's stale-draft ledger row *"CM-23(b) split is IN."*

---

# §A — Premise re-verification (task §2/§3; acceptance criterion "if one is wrong, that is a finding and it outranks the rest")

| # | premise (task file / charter / README) | verdict on disk | evidence |
|---|---|---|---|
| A1 | `submission.tex` is a verbatim copy of `draft.tex` | ✅ **CONFIRMED, and tightened**: `diff` = 4 hunks, all `\includegraphics` path repointing (`fig*.png` → `figs/fig*.png`). Zero prose difference. | `diff submission.tex draft.tex` → lines 63/156/162/241 only |
| A2 | word counts 8,886 (tex) / 14,177 (md) | ✅ CONFIRMED exactly | `wc -w` |
| A3 | App A/C are stubs deferring to `draft.md` at `:193`/`:247` | ✅ CONFIRMED at exactly those lines | tex:193, tex:247 |
| A4 | §3.2 heading still carries the MF-B framing | ✅ CONFIRMED, **and worse than stated**: tex:80 is the *pre-fix* heading verbatim, while md:78 carries the corrected one. The CHANGELOG's v0.4 entry claims *"Both draft.md (canonical) and draft.tex updated"* — for this heading that claim is **false**. | tex:80 vs md:78 vs `CHANGELOG.md` line 3 |
| A5 | MF-A survives at tex:40 ("The router's map") | ✅ CONFIRMED; md:31 is compliant ("An untrained **state-replacing** map"). **New:** MF-A *also* survives in the md at the Figure-1 caption (md:94, "wormhole, **router** and Newtonian control"), where the tex caption (tex:64) is compliant. The two files are each stale at a site the other has fixed. | tex:40, md:31, md:94, tex:64 |
| A6 | counts: theory note 9/4 · MQAR 7/5 · wormhole 68/50 | ✅ ALL THREE CONFIRMED exactly | `grep -o \| wc -l` per file |
| A7 | "collapse" ×5 in tex, exactly one is the retired framing | ✅ CONFIRMED by reading all five in context (tex:40 capture-ball, **tex:80 the retired heading**, tex:102 "sharpened from a collapse into a pricing law", tex:153 Hopfield-collapse, tex:247 Hopfield-collapse artifact). Prohibition 3 holds. | see §6 sweep 11 |
| A8 | score sentence absent; §A20.5 sentence absent | ✅ CONFIRMED, positive-controlled | §2 below |
| A9 | §3.5 item 1 — CM-23(g) "partially present as mechanism prose" | ⛔ **REFUTED. It is wholly ABSENT.** `random kick` = 0, `ensemble` = 0, `restart` = 0, `ungated` = 0, `directed` = 0, `re-launch` = 0 — in **both** files. V1 contains no retry-experiment content whatsoever; its only retry material is a **null** (N1, §3.3 + App D) and **theory** (§5 + App F, explicitly "no runs on trained CLU checkpoints are claimed"). | §6 sweep 8 |
| A10 | §3.5 item 3 — "draft has (b)'s substance; check the wording" | ⚠ **REFUTED as stated.** CM-23(b)'s approved string is absent (`cannot draw` 0, `auto-stop` 0, `accuracy-vs-compute` 0, `saturated feedforward` 0) and so is its *experiment*. What exists is a **different instantiation of the same claim shape** at tex:140 / md:136 — *"A one-shot memory earns no such payoff at any level"* — measured on the §4.1 MQAR rationing study, not on the retry ladder. ⇒ **(b)'s registry sentence cannot be pasted in; it would attribute the wrong experiment.** The charter's stale-draft ledger row *"CM-23(b) split is IN"* is not true of the drafts. | tex:140, md:136; charter:223 |
| A11 | §3.4 Advisor prior — "the MQAR exposure is narrower than the count suggests" | ⚠ **PARTLY REFUTED.** The three concessions the prior names are all real and verified (§4.2 concedes to the 449-param router at tex:148; §4.3 concedes Hopfield cheaper + more noise-robust at tex:151/153; §4.1 disclaims energy-as-signal at tex:139/143). **But two MQAR-carried sentences do read as CLU capability claims** (tex:140, tex:146) and a third is a positive accuracy claim (tex:153, Δ+0.02). See §4. | §4 |
| A12 | §3.6 — `wins`/`beats` are the paper conceding | ✅ CONFIRMED in context (tex:128/146/148 `beats`, tex:134/168 `wins`). ⚠ One exception the premise misses: tex:146 *"a direct edge beats hop-by-hop diffusion, in FLOPs and accuracy"* is not a concession — it is an **intra-CLU win-by-construction** presented as contribution 5. | §6 sweep 7 |
| A13 | §3.6 — zero §A44.1 (toy-compositional) findings in V1 | ✅ CONFIRMED free: `compositional` 0, `occupancy` 0, `d_addr` 0, `organizer` 0 in both files | §6 sweep 12 |
| A14 | README: "the venue-class header is stale" | ⚠ **TRUE OF THE SOURCE, NOT OF THE ARTIFACT.** The only `ML4PS` hit in the tex is **line 2, a `%` comment** — it does not print. The md's line 5 front-matter *does* carry it. ⇒ **CANONICAL-ONLY**, not a printed defect. | tex:2, md:5 |
| A15 | README: "9 theory-note mentions, MQAR 7×" applied to `submission.tex` | ⚠ Those are the **md** counts. The tex has **4** and **5**. The task file's table has it right; the README does not. Cosmetic but it changes the size of the §3.3 job. | §6 sweep 4 |
| A16 | no other spoke in flight against these paths | ✅ CONFIRMED — all 23 digests unchanged across the pass | manifest |

---

# 1 — DELIVERABLE 1: THE ITEMIZED WORKLIST

**Reporting key.** **BOTH** = live in both files · **BASE-ONLY** = live in `submission.tex`, already fixed/absent in `draft.md` (⭐ the highest-value class: a qualification lost silently in condensation, or a fix that never landed) · **CANONICAL-ONLY** = informational, the `.tex` already dropped it.
⛔ **Every "smallest edit" below is NAMED, NOT MADE.** Head line-item approval required before any revision spoke launches.

## 1a. MUST-FIX — blocks submission

| # | site | what it says | what the registry / the paper's own body says | class | sev | smallest edit that closes it |
|---|---|---|---|---|---|---|
| **V1-01** | `submission.tex:80` | §3.2 heading: *"The discriminating experiment: **reach steps then collapses**; the wormhole is flat…"* | The MF-B fix (CHANGELOG v0.4) removed the collapse framing from *"§3.2 heading/table/caption"*; the abstract (`:26`), the table footnote (`:98`) and the body (`:102`, *"the squeeze prices reach; it does not fail at the box"*) all carry the **pricing law**. **The headline section's heading contradicts its own body.** | **BASE-ONLY** | **MUST** | Replace the heading with `draft.md:78`'s verbatim: *"The discriminating experiment: squeeze reach is priced, the wormhole is flat-priced — and only the wormhole hands back a receipt."* One line. |
| **V1-02** | `submission.tex:40` | Contribution 3: *"**The router's** map $(q,p)\mapsto(b,p)$ has $\det J=0$"* | MF-A renamed this object to **state-replacing map** *everywhere* to kill the decision/transport conflation; every other tex site complies (`:26`, `:64`, `:102`, `:118`, `:128`, `:247`). `draft.md:31` is compliant. | **BASE-ONLY** | **MUST** | `The router's map` → `An untrained state-replacing map` (md:31's wording). Two words. |
| **V1-03** | `submission.tex:193` (App A) and `:247` (App C) | *"see `draft.md` Appendix~A for the full four tables"* / *"See `draft.md` Appendix~C for…"* | **A submission cannot cite an internal file a reviewer cannot open.** C-7 makes flag-provenance tables *part of the paper*; C-10 forbids omitting appendix material at drafting time. §3.2 (`:83`) and md:84 both cite "Table 2 (Appendix C)" — which does not exist in the base. | **BASE-ONLY** | **MUST** | Inline `draft.md` §§A.1–A.5 (5 tables) and §§C.1–C.4 (grids) into the base. This is a restoration, not a rewrite — the content exists and is registry-current. **Report the gap; a writer restores it** (task §4 item 5). |
| **V1-04** | `submission.tex:140` = `draft.md:136` (same paragraph) | *"the robust payoff is **rationing ($9.9\times$ intra-CLU)**"* … 40 words later … *"kv$32$ gives $1.14\pm0.06\times$"* | Both are "intra-CLU savings at kv32" and they differ **9-fold**. `9.9×` is the **kv32 cell of §4.3's 2000-ep grid** (`regime-remap-2000ep` Item 1 table); `1.14×` is §4.1's own **400-ep** measurement. The A.5 cross-section note flags the epochs axis in general but **never reconciles these two numbers**. This is the C-7 failure verbatim ("apparent contradictions between differently-flagged runs must be impossible to construct") and the exact class MF-C was written to close — it did not close this instance. | **BOTH** | **MUST** | Attach the flag in-sentence: `9.9×` → `9.9× at kv32/2000 ep (§4.3's grid)`, so the two kv32 numbers are visibly differently-flagged. |
| **V1-05** | `submission.tex:43, 151, 153, 157` (+ `draft.md:37, 152, 154, 448`) | *"the CLU gate's **$9$–$10\times$** figure is intra-CLU rationing"*, under the label **settled** | The settled 198-job grid measures intra-CLU savings **9.9× (kv32) / 9.5× (kv64) / 6.2× (kv96)** — it **falls with load** — and **CM-8's approved wording is "6–10×"**. `9–10×` is the superseded **3-cell, 2-seed provisional pilot** figure (`anchor-robustness`). Quoting a load-dependent number as a narrower constant is **CM-22(bb)** ("quote the curve, never the endpoint"), whose two named instances are of exactly this shape. | **BOTH** | **MUST** | `9–10×` → CM-8's `6–10×` at all four tex sites, or quote the curve `9.9/9.5/6.2× across kv32/64/96`. ⛔ Do not find-replace: `:157` is a figure caption and `:247` an appendix stub. |
| **V1-06** | `submission.tex:250` (App D) | *"**N2** abstention-vs-Hopfield unwinnable; Hopfield dominant at $500$ ep (**provisional — under-trained map**)"* | §4.3 of the same document (`:153`) files the map as *"Accuracy (**settled** — full grid, 198 jobs)"*, and CM-8 is **SETTLED**. `draft.md:448` carries the corrected entry (*"the reversal is regime-specific (SETTLED, full grid, 198 jobs)"*). **A referee who reads the appendix finds the paper calling its own headline §4.3 result provisional.** | **BASE-ONLY** | **MUST** | Replace tex:250's N2 clause with `draft.md:448` in condensed form. |
| **V1-07** | `submission.tex:250` (App D) | The tex's App D lists N1·N2·N3·N24·N30·N31·N23. **`draft.md:449`'s N2b — THE NOISE WALL — has no entry.** | CM-8: *"THE NOISE WALL (dominant negative, **travels with every reversal claim**)"*. C-9: negatives are documented, never dropped; the most prominent go in the appendix. The noise wall survives in §4.3 body and the Fig-2 caption but its **registry entry was dropped in condensation**. md:449 even contains the sentence *"a referee finding it absent would be right to fault the paper"* — a sentence the condensation deleted along with the entry. | **BASE-ONLY** ⭐ | **MUST** | Restore `draft.md:449` (N2b) into the tex's App D list. |
| **V1-08** | absent, **both files** | The mandatory score sentence. `external benchmark` = 0, `headline metric` = 0 in both. Positive-controlled. | **CM-23 head**, charter §4.1 rule 1, and the C3 division's carry-across: *"external benchmarks won on their own headline metric = ZERO"* — **mandatory in any performance section.** V1 has three (§4.1, §4.2, §4.3). | **BOTH** | **MUST** | Insert the approved sentence once, at the head of §4 (`:133–134`, the "Reporting grade: evidence" paragraph — the paper's own perimeter statement). See §2 for the ranked exposure. |
| **V1-09** | absent, **both files** | The §A20.5 substrate-scope sentence. 0 hits (`govern the store`, `measured separately`, `encoder`, `bytes ledgered`, `substrate` all 0). Positive-controlled. | Charter **§A20.5** / matrix **§3.1**, verbatim: *"these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, φ-bytes ledgered."* ⛔ **Every short states it, in its own voice, once.** | **BOTH** | **MUST** | Insert once. Best attachment: §5 "Scope (stated, not buried)" (`:170`), beside the existing scope clause. Second-best: §4 head. See §2. |
| **V1-10** | `submission.tex:168` (§5) | *"The honest pillars of §4 delimit where this buys **a real ML advantage (escalatable rationing)** and where a cheaper black box wins (routing)"* | The comparison set behind "escalatable rationing" is **a full-budget CLU** and **a modern Hopfield**. It has never contained a matched-compute feedforward floor. The program has since measured that floor: **N90** (dominates CLU-gated retry in **all 8 cells**, −3.5…−42.2 pp; 5-seed range mask −3.9…−20.7, noise −9.7…−48.2, negative in 40/40) and **N95** (decision-grade NO **with headroom present**, so it *cannot* be blamed on saturation), with **N103** the best case anywhere in the program — a **tie** (+0.8 ± 1.6 pp, 6 seeds). Charter §4.1: *win-by-construction results are SUPPLEMENTARY only; a short's primary claim must survive competitive baselines.* | **BOTH** | **MUST** | Either (a) scope the sentence to its measured comparison set — *"…a real advantage **over a full-budget CLU and a one-shot Hopfield**"* — or (b) fold N90/N95/N103 in per §3.5 and re-pitch. ⛔ (a) is one clause; (b) is a revision pass. **The Head chooses; this pass does not.** |
| **V1-11** | `submission.tex:26` (abstract), `:64` (Fig-1 caption), `:76`, `:252` | The symbol **$X$** is used four times — including in the **abstract** — and is **never defined in the base**. | `draft.md:72` defines it: *"(the theory note's coset content, **$X$ the broken generator**)"*. The tex condensation dropped the parenthetical. A reviewer meets `p^\top X\Delta` in the abstract with no referent. | **BASE-ONLY** ⭐ | **MUST** | Restore md:72's parenthetical at tex:76. Six words. |
| **V1-12** | `submission.tex:55–58` (§2) | *"In relativistic mode one drift advances $q$ by at most $\varepsilon c/\sqrt{M_i}$ per coordinate, so $Q_T \subseteq C_T$"* — asserted, with **no proof and no pointer**. | `draft.md:49` attributes it: *"The theory note proves (**Prop-A2**) that…"*. The tex dropped the attribution. **This is the paper's central theorem** (the causal box is what makes reach a kinematic failure mode and the whole dichotomy falsifiable) and in the base it is a bare assertion. Contribution 1 is tagged *[proven.]* where md:27 tags it *[proven; theory note + Anonymous 2026.]* — the proof's locus dropped too. | **BASE-ONLY** ⭐ | **MUST** | Restore md:49's `(Prop-A2)` attribution at tex:56 and md:27's provenance tag at tex:38. See §3 — this is also the decisive theory-note self-containment finding. |

## 1b. SHOULD-FIX

| # | site | what it says | what the registry says | class | sev | smallest edit |
|---|---|---|---|---|---|---|
| **V1-13** | `submission.tex:26` (abstract) | *"…**every mechanism** that buys capability at inference **can be made to** carry an explicit physical receipt"* | **C-5**: every generalizing claim carries its scale qualifier in-sentence. The paper verified **two** mechanisms (squeeze, wormhole) plus two controls, at dim 2/4, 5 seeds, oracle placement, laptop-CPU. "Every mechanism" is the only scope-free universal in the base (swept: `CLUs` 0 real hits — "clusters" false friend; `at scale` 2, both scoped; `in general` 1, scoped). | BOTH | SHOULD | Add the qualifier in-sentence: *"every mechanism **we examine**"* or *"the two mechanisms we develop"*. |
| **V1-14** | `submission.tex:140` / `draft.md:136` | *"**A one-shot memory earns no such payoff at any level.**"* | This is the **hedged form of the retired uniqueness claim**. CM-23(b)'s split leaves a **SHAPE** claim only, and its approved sentence carries a **mandatory second half** — *"while remaining below that saturated floor on masked-pixel retrieval"* — which has no counterpart here. Additionally the *venue is occupied*: the paper **cites** EBTs (Gladstone 2025), PonderNet, CALM and ACT in the intro and §5 but never treats any as a counterexample to "no one-shot memory can escalate." | BOTH | SHOULD | Narrow to what was measured: *"a one-shot **associative** memory (Hopfield) earns no such payoff at any of the three levels measured"* — and, if the claim is kept in lead position, add the occupied-venue sentence. |
| **V1-15** | `submission.tex:146` | *"a direct edge **beats** hop-by-hop diffusion, in FLOPs and accuracy"* — contribution 5's positive half | Both arms are CLU wormhole configurations chosen by the authors; the only external entrant in §4.2 **wins** (N24/CM-7). Charter §4.1: win-by-construction = **supplementary only**. Presenting it as a numbered contribution is a promissory frame (C-4 adjacent). | BOTH | SHOULD | Keep the number, demote the frame: label it *[mechanism; intra-CLU cost structure]* alongside the existing *[evidence.]* tag, as §3's arms are graded. |
| **V1-16** | `submission.tex:250` | *"Scope caveat: **N24/N27** rest on a linearly-separable-cue laptop testbed"* | **N27 is cited and never defined** — no App-D entry in either file. This is CHANGELOG-deferred **MF-E** and it is still live in the base. (The other half is closed: `N21` = 0 in the tex, 1 in the md.) | BOTH (N27) / CANONICAL-ONLY (N21) | SHOULD | Either add a one-line N27 entry or drop the reference. |
| **V1-17** | `draft.md:94` (Fig-1 caption) | *"wormhole, **router** and Newtonian control all land at exactly $1.0$"* | MF-A residue in the canonical file; `submission.tex:64` is compliant (*"state-replacing map"*). The two files are each stale where the other is clean — **the condensation was not a strict subset in either direction**, which is itself a process finding for the Head. | **CANONICAL-ONLY** | SHOULD | Sync md:94 to tex:64's wording. |
| **V1-18** | `submission.tex:82` (§3.3 heading) | *"Why a prior null does not bear on this claim"* — a full subsection defending against the program's own N1 | Reviewer-optics: a titled subsection is a signpost to the paper's weakest point, and the defence rests on the authors' own re-scoping ("the two share only the operator $S^{(M)}$"). **C-1 (post-reversal) forbids defensive audit paragraphs.** This is not a legacy-mechanism confession so C-1 does not fire literally, but it is the same instinct and the same optics. Under the Head's lead-with-wins restructure it is a §5/appendix object. | BOTH | SHOULD | Demote to a footnote on the §3.2 table, or to App D under N1 (where the same content already sits, md:447). |
| **V1-19** | `submission.tex:162` caption | Fig 3 (frontier) tagged *"(Candidate for the appendix at the pruning pass.)"* — an internal editorial note **printed in the artifact** | C-10 defers pruning; but the note itself is drafting scaffolding visible to a reviewer, like the `[WORKING TITLE]` and `[AUTHORS PLACEHOLDER]` blocks (`:17`, `:18`). | BASE-ONLY (printed) | SHOULD | Delete the parenthetical at typeset. |
| **V1-20** | registry-side, **not the paper** | **CM-12 is now stale relative to the paper.** Its row still reads *"the squeeze cures escape and **provably cannot beat $C_T$** (landing rate steps up then **drops to 0 past the box**…)"* — the MF-B-falsified framing — while the *same row* also carries the corrected bracket *"crossover a bracket $[L, L+p_0\sinh\zeta/M_0]$ not a knife-edge"*. **CM-12 contradicts itself, and the paper is ahead of it.** | registry | SHOULD | ⛔ Curator/Advisor fold, **not** a paper edit. Flagged, never edited (matrix §3.1 C-3/C-4 discipline). Naming it here so V1's pricing-law wording is not "corrected" backwards by a later pass reading CM-12 literally. |
| **V1-21** | registry-side, **not the paper** | **CM-2 is stale in the same direction.** It reads *"graded relaxation where extra compute buys accuracy (4.8×@kv16 **with gains**)"*; the paper's MF-C fix correctly retracts the accuracy half (*"at convergence… the gate's accuracy **matches** full-budget… the payoff is rationing, not accuracy"*). The paper is **more conservative than its own approved wording**. | registry | SHOULD | Curator fold. ⛔ Do not "restore" CM-2's gains clause into V1 — that would re-open MF-C. |

## 1c. NICE

| # | site | item |
|---|---|---|
| V1-22 | `submission.tex:17, 18` | `[WORKING TITLE: …]` and `[AUTHORS PLACEHOLDER]` still in the base (C-10 allows placeholders until the end; noted for the typeset pass). |
| V1-23 | `submission.tex:2` | Stale venue comment *"% V1 workshop short (ML4PS / NeurReps class)… Canonical content: draft.md."* — **a `%` comment; it does not print.** Refutes the README's implication that this is a printed defect. `draft.md:5` *does* print it (CANONICAL-ONLY). |
| V1-24 | `figs/` | `fig_regime_map.png` and `paid_access_reach.png` are banked but unused by the base (4 of 6 used). No defect; noted so a later pass does not "restore" a superseded headline figure — `paid_access_reach.png` was the **v0.1** headline, replaced at v0.3 by `fig1_certificate.png`. |
| V1-25 | whole base | 18 pp at 8,886 words. ⛔ **No page-cut proposal offered (task §6 prohibition 6).** Recorded only because V1-03's appendix restoration will grow it. |

---

# 2 — DELIVERABLE 2: THE TWO MANDATORY SENTENCES

## 2.1 Absence confirmed, with positive controls

| target string | `submission.tex` | `draft.md` | positive control (same instrument, same files) |
|---|---|---|---|
| `external benchmark` | **0** | **0** | `wormhole` → 50 / 68 ✅ |
| `headline metric` | **0** | **0** | `squeeze` → 41 / 55 ✅ |
| `govern the store` | **0** | **0** | `certificate` → 36 / 45 ✅ |
| `measured separately` | **0** | **0** | ″ |
| `encoder` | **0** | **0** | ″ |
| `bytes ledgered` | **0** | **0** | ″ |
| `substrate` | **0** | **0** | ″ |

Both sentences are **ABSENT from both files.** Advisor measurement reproduced independently.

## 2.2 The score sentence — claims currently standing unqualified, ranked by consequence

> Approved form (CM-23 head, verbatim, unchanged across the C2→C3 boundary): **"external benchmarks won on their own headline metric = ZERO."**
> ⚠ Its w23 frame sentence, which the matrix says *should precede any of these*: *"across three independent tasks a trivial or strong baseline dominates the absolute headline number while CLU owns a mechanism/capability axis those baselines structurally lack."* — that frame is a near-exact description of V1's own §4, which makes its absence the more conspicuous.

| rank | claim standing unqualified | site | why it ranks here |
|---|---|---|---|
| **1** | *"where this buys **a real ML advantage** (escalatable rationing)"* | `tex:168` (§5, the position restated — the last thing a reviewer reads) | The paper's only unhedged performance-advantage assertion, in its closing position, with no external floor anywhere in the comparison set. Directly contradicted by N90/N95 and only tied by N103. **This is the sentence the score sentence exists to prevent.** |
| **2** | *"A one-shot memory earns no such payoff at any level."* | `tex:140`, `md:136` | A structural-uniqueness claim about escalatable compute, on a venue occupied by DEQs / EBTs / Titans — three of which the paper itself cites. Without the score sentence a reviewer reads it as a field-level claim. |
| **3** | *"the gate reverses Hopfield … ($\Delta+0.02$)"* | `tex:153`, `md:156` | The paper's only positive accuracy number against an external method. It is scoped four ways in-sentence (clean/correlated, kv≤64, n=8, 2000 ep) — genuinely well-qualified — but a Δ+0.02 reversal at 6/15 cells with 0/6 under noise is exactly the kind of number the score sentence keeps from reading as a benchmark win. |
| **4** | *"a direct edge beats hop-by-hop diffusion, in FLOPs and accuracy"* | `tex:146` | Win-by-construction, intra-CLU, presented as a contribution (V1-15). |
| 5 | *"$4.81\pm0.44\times$ fewer relaxation steps"* | `tex:41`, `:140` | Already heavily scoped inline (kv16, 400 ep, decay named). Lowest residual exposure — the qualifiers here are exemplary and should be the template for the other three. |

**Placement, named not drafted:** the natural site is §4's opening paragraph (`tex:133–134`), which already reads *"Reporting grade: evidence… Where the receipt says a cheaper black box wins, we say so."* — the score sentence belongs in that breath, once, and then governs §4.1–§4.3.

## 2.3 The §A20.5 substrate-scope sentence — claims currently standing unqualified

> Approved form (charter §A20.5 / matrix §3.1, verbatim): **"these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, φ-bytes ledgered."**
> Rationale on the record: *"the program's laws are store-level, and every end-to-end null it has measured has had an encoder in the path… A short that omits the clause invites a reviewer to read a store-level law as an end-to-end claim, and a store-level null as an end-to-end refutation."*

| rank | claim standing unqualified | site | why it ranks here |
|---|---|---|---|
| **1** | **THE NOISE WALL** — *"gate $0.36$ vs Hopfield $0.71$ at $\sigma{=}0.6$/kv$32$ **despite fidelity $\approx1.0$**"* | `tex:153`, `:157` (Fig-2c caption), `:247` | ⛔ **This is the exact object §A20.5 exists for, and V1 states it without the clause.** The paper *itself* observes the store/end-to-end split — *"the patterns are stored, but the governed relaxation over-commits to the corrupted cue"* — and then presents it as a CLU-level negative. It is a **read-path** result on a store whose fidelity is 1.000. Without the clause the paper hands a reviewer a general refutation of the primitive where the measurement supports a narrower one. |
| **2** | *"CLU fidelity rises to $\approx1.0$ and the accuracy reversal appears"* | `tex:153` | Fidelity is store-level; accuracy is end-to-end. The sentence puts them in a causal chain with nothing separating them. |
| **3** | The whole of §4's *"Reporting grade: evidence… learned memories"* frame | `tex:133–134` | §4's three pillars are all end-to-end (MQAR key–value read-in and read-out are in every path) while §3's certificates are store-level theorems. **The paper has no sentence anywhere marking that boundary** — which is precisely the substrate-scope sentence's job. |
| 4 | *"a self-calibrating gate rations relaxation budget on a trained memory"* | `tex:41`, `:137` | The calibration head is itself an encoder-side object (Platt scaling on a residual). Lower rank only because §4.1's scope line is already dense. |

**Placement, named not drafted:** §5's *"Scope (stated, not buried)"* paragraph (`tex:170`) is the paper's own designated home for exactly this class of clause and already carries the oracle-placement and laptop-CPU scopes. One sentence, in the paper's own voice, appended there.

---

# 3 — DELIVERABLE 3: THEORY-NOTE SELF-CONTAINMENT (per mention in `submission.tex`)

**4 mentions in the base** (9 in `draft.md`). Verified count, per-file, positive-controlled.

| # | site | text | classification | the exact object | derivable from material already in the paper? |
|---|---|---|---|---|---|
| **T1** | `tex:34` | *"the exactly-solvable theory whose certificates we verify is developed in a companion note (Anonymous, 2026; hereafter **the theory note**)"* | **DECORATIVE** (introduces the referent) — **but see the standing MF-1 flag**: this is an anonymous citation to a document that is not public. Removing the *pointer* costs nothing; removing the *note* costs T3. | — | n/a |
| **T2** | `tex:48` | *"Full derivations and machine-precision checks are in the theory note."* | **DECORATIVE — with one caveat.** Everything it covers is *stated* in the paper: the conformal-symplecticity identity $J^\top\Omega J=(1-\gamma)\Omega$, $\det J=(1-\gamma)^d$, the gated $\det J=(1-\gamma\varphi(q'))^d$, the two kinetic modes, the governor $\gamma_n = s\tanh(\max(0,H-E^\star))$. | the damped-Verlet Jacobian | **YES.** Each is a one-line computation from the map given at `:47`. The *"machine-precision checks"* half is separately covered by **App E** (`checks.py`, six rows, in the base at `:257–259`) — so the pointer is redundant with the paper's own appendix. |
| **T3** | `tex:72` | *"Its injected energy is **bounded**, $H(S_\zeta z)\le e^{2\vert\zeta\vert}H(z)$ **(theory note Prop-12)**"* | ⛔ **LOAD-BEARING — the only one, and it is load-bearing for the abstract, contribution 2, §3.1, §5 and Table 1.** | **Prop-12 (C2), the bounded-injection certificate.** Traced on disk to `paid-access-theory` §"squeeze" and §Table-1: `H(S_ζ z) ≤ e^{2|ζ|}H(z)` (Prop-12 C2), companion `∂q'ᵢ/∂ζ|₀ = pᵢ/M_eff,i` (Cor-13/§5.4). | ⛔ **NO.** The base gives the squeeze's action ($\delta q'=\delta q\cosh\zeta + p_0\sinh\zeta$) and its symplecticity ($\det S=1$) but **never the energy inequality's derivation**. The reader can *check* it numerically against the four quoted $(\zeta,\text{ratio},\text{bound})$ pairs and App E row (F), and can see it fail on the quartic (the paper says so) — but cannot derive it. ⭐ **This single proposition is what decides whether V1 needs a derivation appendix (V5 route) or Option B suffices.** ⛔ Per task §3.3 I classify and do **not** recommend which. |
| **T4** | `tex:295` | Reference entry: *"Anonymous (2026). [The theory note — …; third person per hermetic-citation policy]"* | **DECORATIVE** (bibliography). | — | n/a |

## 3.1 ⭐ The finding this classification produced that the mention-count could not

**Two load-bearing theory-note pointers exist in `draft.md` and were dropped, not replaced, in the condensation — so the base asserts them with no attribution at all.** This is the highest-value class the pass can produce (task §2) and it is *invisible* to a per-mention count of the base, because the base's count went *down* by losing exactly the pointers that mattered.

| dropped pointer | `draft.md` | what the base does instead | consequence |
|---|---|---|---|
| ⛔ **Prop-A2 — the causal-box theorem** | `md:49`: *"The theory note proves (**Prop-A2**) that in relativistic mode one Verlet drift advances $q$ by at most $\varepsilon c/\sqrt{M_i}$ per coordinate, so $Q_T \subseteq C_T$"* | `tex:55–58` states the inclusion **as a bare assertion**, no proposition, no pointer, no derivation. | **The paper's central theorem — the object that makes reach kinematic and the whole dichotomy falsifiable — is unsupported in the base.** = **V1-12, MUST-FIX.** |
| ⛔ **the ledger's hard-gate regime** | `md:70`: *"…that must be **ledgered** (**theory note §7.4 hard-gate regime**)"* | `tex:74` keeps *"must be **ledgered**"* and drops the regime pointer, while the *fine print two sentences later* depends on it (*"a gate varying during the jump gives $\det J=1+\nabla g\cdot\Delta\neq1$… the receipt is clean only for a frozen gate"*). | The frozen-gate design guard's **regime of validity** loses its locus. SHOULD-FIX; fold with V1-12. |
| ⚠ **the proof-grade provenance tag** | `md:27`: contribution 1 tagged ***[proven; theory note + Anonymous 2026.]*** | `tex:38` tags it ***[proven.]*** | An unattributed *[proven]* in a contributions list is the weakest possible form of a proof claim. Fold with V1-12. |
| ⚠ **the definition of $X$** | `md:72`: *"(the theory note's coset content, **$X$ the broken generator**)"* | `tex:76` drops it — and $X$ appears in the **abstract** (`:26`). | = **V1-11, MUST-FIX.** |

**Net:** on the base, the theory note is **1 load-bearing citation (T3/Prop-12) + 3 decorative**, but the *true* dependency is **four objects deep** once the dropped pointers are counted — Prop-12, Prop-A2, §7.4, and the definition of $X$.

**C-6 check, since it names Prop-12 directly.** ✅ **SATISFIED, and I decline to manufacture a defect here.** C-6 requires *"Prop-12's compact-set scope"* next to the claim. `paid-access-theory` contains **zero** hits for `compact`; what the paper carries instead at `tex:72` is a **narrower, measured** scope — *"the $e^{2\vert\zeta\vert}$ bound is a matched-quadratic-$H$ certificate; on the quartic well the raw ratio can exceed it (expected), so we quote it in that scope."* The other two C-6 items are also satisfied and adjacent: LTT exchangeability + **ECE $\approx0.100\pm0.021$** at `tex:141`, and *"certified within the stated probe-to-deployment scope"* rather than an absolute. **C-6 is the best-executed charter rule in this paper.**

---

# 4 — DELIVERABLE 4: THE F2 ANSWER (Add.10 — MQAR vs criterion 4)

**Two questions, no more.** MQAR: **5 mentions in `submission.tex`**, 7 in `draft.md` (verified).

## Q1 — Does any MQAR-based sentence read as a CLU **capability** claim (as opposed to a boundary, a negative, or a mechanism statement)?

**YES — two, plus one positive accuracy claim. Quoted verbatim:**

1. ⛔ **`tex:140` / `md:136`** — *"**A one-shot memory earns no such payoff at any level.** Scoped to MQAR vocab-$256$, kv$\in\{16,24,32\}$, $5$ seeds, laptop."*
   → A **comparative capability claim**: escalatability is a property CLU has and a whole class of memories lacks. The scope clause bounds the *measurement*, not the *claim* — "at any level" ranges over the three measured levels but the sentence reads as ranging over one-shot memories in general.
2. ⛔ **`tex:146` / `md:144`** — *"a direct edge **beats** hop-by-hop diffusion, in FLOPs and accuracy ($5$ seeds, MQAR-style, $N\le8$, laptop)."*
   → A **capability claim** in comparative form. Intra-CLU (both arms are the authors' own configurations), so it is win-by-construction — but a reviewer skimming contribution 5 reads a win.
3. ⚠ **`tex:153` / `md:156`** — *"the gate reverses Hopfield ($0.99$–$1.00$ vs $0.97$–$0.98$, $\Delta+0.02$, $n=8$)"*
   → An **external positive**, on MQAR, four-ways scoped in-sentence. Nearest thing in V1 to a benchmark claim.

**Mechanism / boundary / negative (not capability), for completeness:** `tex:41` (contribution 4, scoped rationing), `tex:137` (setup), `tex:170` (§5 scope statement), and the whole §4.2 boundary (`tex:148`, the 449-param router **winning**).

⇒ **The Advisor's prior is refuted as stated.** The three concessions it names are all genuinely present and correctly worded — but the exposure is **not** narrow: it is two capability claims and one positive, all MQAR-carried, and one of them (`tex:140`) is the hedged form of a retired uniqueness claim (**V1-14**).

## Q2 — Does §4.1's escalatability claim (CM-2) survive a **matched-bytes** reading?

**No — it is not yet decided, and on the six standing confirmations the prior is that it does not.** Statement of what is and is not established:

- **What the claim rests on.** CM-2: *"The CLU-specific asset is being an **escalatable** memory — graded relaxation where extra compute buys accuracy…; one-shot memories get no allocation payoff."* The property doing the work is **"graded compute to ration."**
- **What criterion 4 asks.** The matched-bytes exemplar-store tripwire (intervention §6, the C2W10 instrument) puts an exemplar store at the **same byte budget** beside the CLU store. It has **six confirmations**; **CAMELS-US (C3 Add.1) is its first measured survivor** (E = 0.2559 vs S = 0.758, `margin_rel` −0.6625 against a −0.02 threshold, no fire at 536× budget).
- ⛔ **Why the claim is structurally exposed, not merely untested.** On MQAR vocab-256 at kv ≤ 32, an exemplar store holding the same key–value pairs at matched bytes answers in **one pass at or near 1.000**. It therefore has **no graded-compute dial at all** — so the CLU's escalatability is *trivially* unique and *simultaneously* worthless: **there is nothing to ration when the baseline is already at ceiling in one step.** This is not a hypothetical; it is the same geometry that produced N90 (*"NN flat at 0.99–1.00 — it cannot rise **because it is already saturated**"*) and N95 (*"identity retrieval under **known** erasure is information-theoretically trivial… There is no headroom versus the correct ML floor"*).
- ⚠ **And the paper has already conceded the premise.** §4.1 point 2 states that the accuracy headroom *"is a property of an imperfect (**under-trained**) memory, not of the mechanism."* ⇒ **the very imperfection that makes escalatability worth anything is one the paper has retracted as an artifact.** After MF-C, the surviving claim is the *saving* — and a saving measured only against **the authors' own full-budget arm**.
- ✅ **What is genuinely CLU-side and would survive:** the LTT coverage certificate (30/30 valid, risk 0.030 at ε=0.05) is a property of the *apparatus*, not of the comparison, and is untouched by criterion 4.

### ⇒ **What would decide it (name the measurement, do not run it):**

> Run the **C2W10 matched-bytes exemplar-store tripwire** on §4.1's own MQAR cells (vocab-256, kv ∈ {16,24,32}, 5 seeds, laptop), with **three arms at matched bytes**: (i) the calibrated CLU gate, (ii) a 1-shot exemplar store / kNN over the same keys, (iii) the modern-Hopfield arm already in the paper — scoring **accuracy at matched *compute*** and **accuracy at matched *bytes*** on the same axes, and reporting the tripwire's `margin_rel` against the −0.02 threshold.
> - **If the tripwire fires** (E ≥ S at matched bytes, the six-confirmation prior): CM-2's escalatability claim is **supplementary only** under charter §4.1, and §4.1's contribution must be re-pitched as the **certified rationing apparatus** (LTT + calibration), which is what `tex:143` already calls it.
> - **If it does not fire** (CAMELS-US precedent): §4.1 becomes V1's first genuine win, and the score sentence's ZERO gets its first contested challenge.
>
> ⛔ Until then, **`tex:140`'s "A one-shot memory earns no such payoff at any level" must not stand in lead position without the matched-bytes qualifier**, because the sentence's whole force depends on the baseline being unable to reach ceiling — which at matched bytes on this task it can.

---

# 5 — DELIVERABLE 5: THE COMPLETENESS GAP (`submission.tex` vs `draft.md`)

The Head has ruled **all results must be present in the base**. Measured gap: the base is a ~37 % condensation and the loss is **not** uniform prose thinning — five bodies of content are absent outright.

| # | missing from the base | where it lives in `draft.md` | what cites it *from inside the base* | sev |
|---|---|---|---|---|
| **G-1** | ⛔ **Appendix A entire — all five flag-provenance tables** (A.1 §3 certificate stack · A.2 §4.1 gate + Hopfield transfer · A.3 §4.2 routing · A.4 §4.3 regime map · A.5 §3.2.1 payoff + B.2 BIBO) | `md:193–282` | `tex:193` — a single prose paragraph that *narrates* the tables and then defers | **MUST** (C-7: "flag-provenance tables are **part of the paper**") |
| **G-2** | ⛔ **Appendix C entire — four grids** (C.1 reach battery + per-arm certificates, C.1.b latch cloud dim{2,4}×seed{0,7} · C.2 routing grid · C.3 Hopfield iteration parity · C.4 regime map full grid a/b/c) | `md:328–442` | `tex:247` (defers) — **and `tex:83`, which tells the reader "Table 2 (App. C) gives the full grid"** | **MUST** (C-10) |
| **G-3** | ⛔ **App D entry N2b — THE NOISE WALL** | `md:449` | §4.3 and Fig-2c present the noise wall as *the dominant negative*; App D does not carry it | **MUST** = V1-07 (C-9) |
| **G-4** | ⚠ **App E's structure** — the six-row verification table with its per-row verdict column collapses to one prose line | `md:463–477` table vs `tex:257–259` prose | §3's *"verified to machine precision"* rests on it | SHOULD |
| **G-5** | ⚠ **App D's four condensed entries lose their mechanism halves** — N2 (loses the settled body, keeps the retired "provisional"), N24 (loses the linear-separability mechanism), N30 (loses "pins a structureless init"), N23 (loses "calibrated head over-routes local at N=8") | `md:443–462` | App D is the C-9 vehicle | SHOULD |

⚠ **Two riders the condensation dropped that are *not* appendix material** and belong in this ledger because they are the §2 high-value class: **Prop-A2** (`md:49`) and the **definition of $X$** (`md:72`). Filed as V1-12 / V1-11.

⛔ **Report the gap; a writer restores it in a later pass** (task §4 item 5). No restoration attempted here.

---

# 6 — DELIVERABLE 6: SWEEP LOG

**Instrument discipline (task §7).** All context-window patterns run through **`/usr/bin/grep`** explicitly, never the shell's `ugrep` alias. All counts by `grep -o … | wc -l` (occurrences), **never `grep -c`** (lines). All sweeps **per-file**, never directory-level over `.claude/`. **Every negative below is positive-controlled.** T = `submission.tex`, M = `draft.md`.

| # | pattern | T | M | positive control (same file, same instrument) | note |
|---|---|---|---|---|---|
| 1 | `wormhole` / `squeeze` / `certificate` | 50/41/36 | 68/55/45 | *(these ARE the controls)* | matches the task's 68/50 premise exactly |
| 2 | `external benchmark` · `headline metric` | **0** · **0** | **0** · **0** | ctrl #1 ✅ | §2 |
| 3 | `govern the store` · `measured separately` · `encoder` · `bytes ledgered` · `substrate` | **0**×5 | **0**×5 | ctrl #1 ✅ | §2 |
| 4 | `theory note` | 4 | 9 | ctrl #1 ✅ | §3; premise confirmed |
| 5 | `MQAR` | 5 | 7 | ctrl #1 ✅ | §4; premise confirmed |
| 6 | `anytime` · `matched-bytes` · `matched bytes` · `matched-compute` · `trilemma` · `latency` · `exemplar` · `kNN` | **0**×8 | **0**×8 | ctrl #1 ✅ | §7 items 2,4,5 ABSENT |
| 7 | `beats` · `wins` · `dominance` | 3 · 2 · 2 | 3 · 3 · 2 | ctrl #1 ✅ | **all read in context** — see A12 |
| 8 | `random kick` · `ensemble` · `restart` · `ungated` · `directed` · `re-launch` · `relaunch` | **0**×7 | **0**×7 | ctrl #1 ✅ | ⛔ **refutes the §3.5 item-1 prior** (A9) |
| 9 | `cannot draw` · `auto-stop` · `accuracy-vs-compute` · `shape claim` | **0**×4 | **0**×4 | `saturated`→1/1, `rising`→1/5 ✅ | ⛔ **refutes the §3.5 item-3 prior** (A10) |
| 10 | `flat curve` · `flat anytime` · `carries nothing` · `cannot be addressed` | **0**×4 | **0**×4 | ctrl #1 ✅ | N308's rider does not yet fire — see §7 rider 2 |
| 11 | `collapse[sd]?` | 5 (+3 `collapses`) | 5 (+3) | — | **all read in context; 1 of 5 is the retired framing** (tex:80). Prohibition 3 honoured. |
| 12 | `compositional` · `occupancy` · `d_addr` · `organizer` · `CAMELS` · `enwik8` | **0**×6 | **0**×6 (`lattice` 0/2) | ctrl #1 ✅ | §3.6 A44.1 confirmed free |
| 13 | `CLUs` · `every mechanism` · `in general` · `at scale` · `always` | 1†·1·1·2·5 | — | — | †`CLUs` = **false friend** (`clusters`, tex:148). C-5 sweep → one real hit (V1-13) |
| 14 | `N21` · `N23` · `N27` · `N30` | 0·1·**2**·2 | 1·1·**2**·2 | ctrl #1 ✅ | N27 dangling (V1-16) |
| 15 | `9$--$10` (tex) / `9–10` (md) | **4** | **4** | — | V1-05 |
| 16 | `\tau` | 2 | 6 | ctrl #1 ✅ | only `tex:274` is substantive (App F.6) |
| 17 | `ML4PS` · `NeurReps` · `workshop` · `PLACEHOLDER` · `WORKING TITLE` | 1·1·1·1·1 | 1·1·1·1·1 | — | tex hits are **line 2, a `%` comment** (A14) |
| 18 | `includegraphics` | 4 | (4 md image refs + Fig 3) | — | 4 of 6 banked PNGs used |
| 19 | `draft.md` (self-reference from inside the base) | **3** (`:2` comment, **`:193`**, **`:247`**) | — | — | V1-03 |
| 20 | number-provenance spot checks | — | — | — | §4.1 {4.81, 0.894, 629, 0.847, 0.869, 0.431, 30/30, 0.647, 0.100, 1.57, 1.14, 0.286, 2636, 1919, 0.547} **all 15 trace to `outputs/v1-pivot.md`** ✅ · §4.2 {0.948, 8.81, 0.887, 0.715, 1.18, 1.76, 2.94, 449, 0.41, 0.28} **all 10 trace to `outputs/v1-router-baseline.md`** ✅ · **9.9× traces to `outputs/regime-remap-2000ep.md` Item-1 table, kv32 row, 2000 ep** ⚠ (V1-04/V1-05) |
| 21 | `md5` manifest ×23 files, before/after | — | — | — | **identical**; acceptance criterion met |

**Registry files consulted on disk at the moment of use** (never quoted from the task file): `claims_matrix.md` L550 (CM-2), L551 (CM-3), L555 (CM-7), L556 (CM-8), L560 (CM-12), L565/566 (CM-14), L573 (CM-22 incl. **(bb)**), L574 (CM-23 incl. **(b)(g)(l)(r)(y)(aa)**), L466 (**N308** flat-curve disjunction), L598–609 (§3.1 incl. **§A20.5** verbatim) · `negative_results.md` L118 (**N90**), L123 + L899–906 (**N95** incl. the ⟲ HEAD RULING), L131 (**N103**), L145 (**N117**) · `advisor-head-shorts-charter.md` L81 / L174 (genuine-win bar), L176 (§A20.5), L223 (stale-draft ledger) · `advisor-head-c3-charter.md` L33 / L60 / L67 (criterion 4, CAMELS-US) · `critique_register.md` (G1/G2/G3/G5/G6, P16/P19/P20) · `philosophy-synthesis.md` L581–600 (Positioning Charter C-1…C-10) · `outputs/{v1-pivot, v1-router-baseline, regime-remap-2000ep, anchor-robustness, paid-access-theory}.md` · `papers/v1-short/CHANGELOG.md`.

---

# 7 — §3.5: THE POST-v0.4 POSITIVES — PRESENT / PARTIAL / ABSENT

All seven are C1/C2 estate; **A5.6's pending-rule fires on none of them.** Verified per-file.

| # | result | authority | `submission.tex` | `draft.md` | where it would attach | ⛔ notes / refutations of the prior |
|---|---|---|---|---|---|---|
| **1** | Retry mechanism attribution — the lift is the **directed** symplectic re-launch; equal-energy random kicks and k-restart ensembles **dead flat in all 8 cells**; the gate load-bearing (ungated retry-all **0.96 → 0.004 at 9× compute**) | **CM-23(g)**, **N90** | ⛔ **ABSENT** | ⛔ **ABSENT** | §5 design-rule 2 (`tex:180`, "Mix; do not rely on the squeeze alone") — the controls-that-die *are* the empirical content of that rule; and App F.6, which currently **specifies as future work** the very ensemble/kick comparison that has since been run | ⛔ **Advisor prior REFUTED (A9).** Not "partially present as mechanism prose" — **zero** of the six control terms appears in either file. V1 has **no retry-experiment content at all**: only the N1 null (§3.3, App D) and toy-EBM theory (§5, App F, *"no runs on trained CLU checkpoints are claimed"*). ⚠ Folding this in **converts App F.6 from a specified experiment into a reported one** — a structural change to §5's grade, not a sentence. |
| **2** | ⭐ **The R3-native TIE** — pixel-space corruption of a φ-addressed store, **no mask oracle constructible**; gated anytime read **ties** the matched-compute feedforward-in-φ floor (**+0.8 ± 1.6 pp, 6 seeds, 3/6 positive**), beats 1-shot kNN-in-φ by +1.1 ± 1.4, beats every mechanism control (**+4.6 ± 2.2 pp over equal-energy random kick, 6/6**; ensemble +4.4; ungated **0.000**), auto-stops at **1.40 ± 0.20×** | **CM-23(r)**, **N103** | ⛔ **ABSENT** | ⛔ **ABSENT** | §4 as a **fourth pillar**, or §5 replacing the *"real ML advantage"* sentence (`tex:168`) with a measured one | ✅ **Advisor prior CONFIRMED.** ⛔⛔ **The wording is bound to "ties" and may never read "wins"** (CM-23(r) head). ⚠ **The seed split is a live trap**: **6 seeds** for the headline, **1 seed** for the τ sub-claim (matrix scope line, w25). ⛔ Never quote items 2 and 4 under one seed count. ⭐ Free companion, quotable and inverts an R3 framing assumption: **there is no forgetting-by-age** — spread across task ages 6.2 pp, **newest items hardest (0.796) not oldest (0.846)**. |
| **3** | The SHAPE claim in its approved wording — *"CLU draws a rising, auto-stopping accuracy-vs-compute curve that a saturated feedforward memory structurally cannot draw, **while remaining below that saturated floor on masked-pixel retrieval**"* | **CM-23(b)** | ⛔ **ABSENT** (string) / ⚠ **a different instantiation present** at `:140` | same | §4.1 point 2, beside *"A one-shot memory earns no such payoff at any level"* | ⛔ **Advisor prior REFUTED as stated (A10).** The approved string is absent **and so is its experiment**. `tex:140` is the same claim *shape* on a **different measurement** (MQAR rationing vs the retry ladder). ⛔ **Pasting (b) in verbatim would attribute the wrong experiment.** ⚠ And the charter's stale-draft ledger row *"CM-23(b) split is IN"* (charter:223) is **not true of either file** — flagged for the Advisor. |
| **4** | The τ-regime rule — τ load-bearing **iff `#{cos₀ < τ} < k·step_n`**; at p=0.8 all four thresholds bit-identical (`max\|Δ\|=0`, 3 seeds × 2 snapshots, pool ≥160/200); at p=0.5 in the **same store** τ=1.0 costs **−0.485 ± 0.064** (**−0.546 ± 0.023** end-of-stream) at 1.80× vs 1.12–1.30× | **CM-23(aa)**, **N117** | ⛔ **ABSENT** | ⛔ **ABSENT** | App F.6, the only τ site (`tex:274`, *"the learned-$\tau$ escalation gate of §4.1"*) | ✅ prior CONFIRMED (absent) — ⛔ **but with a hazard the prior does not name.** **V1's τ is not N117's τ.** V1's is an **LTT-selected exit threshold on $p_{\rm wrong}$** (§4.1, `tex:137`); N117's is a **cos₀ cutoff on a retry ladder**. Folding CM-23(aa) in without saying so **manufactures a false identity between two different gate objects** — a cross-section defect of exactly the C-7 class. ⛔ **Drop the "φ-space vs pixel-space" framing entirely** (pre-registered and falsified) and **every threshold statement names the corruption level** (CM-22(bb)). |
| **5** | The trilemma's third corner — dropping amplitude-independent latency **is** the compute-adaptive-read dial (*"a faded memory costs more integration steps to read, which is a physical, measurable statement a timestamped row cannot make"*) | **CM-23(y)** | ⛔ **ABSENT** | ⛔ **ABSENT** | §5 "The position, restated" (`tex:167`) — it is a *position* sentence, which is V1's genre | ✅ prior CONFIRMED. ⛔⛔ **The hard never-quote at the end of CM-23(y) applies in full: *"Both proposed fixes are REFUTED and neither may be described as available (N119)."*** Item 5 may say what dropping latency **means**; it may **never** present the **gated-stiffness channel** as available. ⚠ V1's `latency` count is **0** in both files, so there is no existing sentence to contaminate — the trap fires only on new prose. |
| **6** | Gate memory-agnostic; escalatability is the CLU asset | **CM-2** | ✅ **PRESENT** — `tex:41` (contribution 4), `tex:136` (§4.1 heading), `tex:139` (finding 1), `tex:143` | ✅ PRESENT (`md:33, 130, 133, 136`) | — | ✅ prior CONFIRMED. **Numbers match CM-2 exactly**: Hopfield raw **0.18 → 0.88** vs CLU **0.43 → 0.87**; **4.81 ± 0.44×** at kv16 (CM-2's "4.8×@kv16"). ⚠ **The paper is MORE conservative than its own approved wording** — CM-2 still says *"extra compute buys accuracy… with gains"*, which MF-C correctly retracted. ⛔ **Do not "restore" CM-2's gains clause; fold the registry instead** (V1-21). |
| **7** | The regime map, settled | **CM-8** | ✅ **PRESENT, and the scope clauses are complete** — `tex:43`, `:151`, `:153`, `:157`, `:247` | ✅ PRESENT | — | ✅ prior CONFIRMED. **All eight CM-8 scope clauses verified present**: under-training artifact · clean/correlated kv≤64 Δ+0.02 · ρ=0.9 = Hopfield fragility not CLU strength · epoch-budget wall not capacity · kv96@4000 +0.03, kv128 ties · kv32 over-trains 1.00→0.89 · **noise wall 0/6, gate 0.36 vs Hopfield 0.71** · tally 6/15. ⛔ **One number diverges: the paper says "9–10×", CM-8 says "6–10×"** = **V1-05, MUST-FIX.** ⚠ And App D still files it **provisional** = **V1-06**. |

## The four binding riders — status against **these** drafts

1. ⛔ **CM-23(y)'s never-quote (N119).** **Not yet violated** — `latency`, `trilemma`, `gated-stiffness` all 0/0. It becomes live the instant item 5 is drafted. **No draft sentence currently needs it; every new one will.**
2. ⛔ **The flat-curve disjunction (N308, C2W11).** *"A flat anytime curve ⇒ the store carries nothing"* is **REFUTED as an inference**; the standing replacement is *"carries nothing **OR** cannot be addressed"*, separable by the control that took the same store / same physics / same frozen budget grid from **0.0223 → 0.8219 → 0.8711** under oracle addressing against a shipped read **flat at 0.0004**. ⚠ **That is a three-point curve — quote it as a curve, never as "0.02 → 0.87"** (CM-22(bb)). **N199's aphorism may no longer stand alone.**
   → **Status on these drafts: NO sentence currently requires the rider** (`anytime` 0/0, `flat curve` 0/0, `carries nothing` 0/0). ⛔ **But the rider fires on item 2 the moment it lands**, and it fires on one *existing* sentence by analogy that the Head should see: **`tex:169`'s BIBO framing and `tex:102`'s squeeze `→0` entries** already deploy the paper's own version of the correct discipline (*"'priced out of the swept grid,' not 'cannot reach'"*) — V1 has independently discovered N308's logic for its own flat-vs-blocked distinction, which is the strongest possible argument that it must not violate it when the anytime curve arrives.
3. ⚠ **Seed-count separation.** CM-23's scope line: *"(r) 6 seeds headline, τ-sub-claim 1 seed."* Item 2 is **6 seeds**; item 4 (N117) is what upgraded the τ half to **3 seeds × 2 store snapshots**. ⛔ **Never quote both under one seed count.** V1 currently quotes 5 seeds throughout §3/§4 — a **third** seed count — so the moment items 2 and 4 land the paper carries three different n's in one document. **A per-claim seed column is the mechanical fix; C-7 will demand it.**
4. ⚠ **N117's Δ has two forms** — end-of-stream **−0.5460 ± 0.0226**, and **−0.4846 ± 0.0640 over all six cells** (range [−0.578, −0.405]). Verified on disk at `negative_results.md:145` (CM-23(aa) rounds these to −0.546 ± 0.023 / −0.485 ± 0.064). **Whichever is used, its scope travels in-sentence.**

## Forbidden forms — checked, including hedged and implied forms

| forbidden form | literal string | **hedged / implied form present?** |
|---|---|---|
| *"beats feedforward via test-time compute"* — absolute dominance **RETRACTED** (N90: NN floor beats CLU-gated in **all 8 cells**, −3.5…−42.2 pp; 5-seed mask −3.9…−20.7, noise −9.7…−48.2, **40/40 negative**) | ✅ **ABSENT** (`feedforward` = 1, intro only) | ⛔ **YES — `tex:168`, *"where this buys a real ML advantage (escalatable rationing)"***. Same proposition, no floor, closing position. = **V1-10, MUST-FIX.** |
| the anytime curve as a **uniqueness** claim (it is a **shape** claim; venue occupied — DEQs / EBTs / Titans) | ✅ ABSENT (`anytime` 0) | ⛔ **YES — `tex:140`, *"A one-shot memory earns no such payoff at any level"***, on a venue whose incumbents the paper cites but never engages. = **V1-14, SHOULD-FIX.** |
| *"the anytime read wins"* — it **ties** | ✅ ABSENT | ✅ no implied form (nothing anytime exists yet) |
| *"9–10× savings vs Hopfield"* — it is **intra-CLU** rationing | ✅ **ABSENT, and exemplarily so** — the paper says *intra-CLU* at all four sites and adds *"We state this next to every accuracy-improvement claim"* (`tex:151`) | ✅ none. ⚠ **But the range itself is wrong** (6–10×, not 9–10×) = V1-05 — a *different* defect from the forbidden form. |
| any **energy-as-superior-confidence/routing** claim (N3, N21, N24 — three independent refutations) | ✅ **ABSENT** | ✅ **none, and actively disclaimed four times**: `tex:41` (*"not a superior energy signal"*), `:139` (*"not a claim that CLU energy is a better confidence signal"*), `:143` (N3, *"a claim we do not make anywhere"*), `:148` (*"energy is not the routing signal"*). **CM-3 compliance is the strongest thing in this paper.** |

⚠ **N95's placement obligation.** CM-23(l)/N95 must be stated **in the same section as the retry claims** — *a decision-grade NO **with headroom present**, so the retraction cannot be blamed on saturation.* **V1 currently has no retry performance claims, so the obligation does not yet fire.** It fires the moment items 1 or 2 land. ⛔ **And read N95's ⟲ HEAD RULING (2026-07-25) before drafting it:** the *verdict wording* was superseded — *"the **STATIC-RETRIEVAL INSTANTIATION** of R3 is CLOSED; the **DIAL** claim survives all mechanism controls for the third consecutive wave and is NOT falsified by a mask-oracle… **Do not quote 'R3 failed'; quote the corrected status.**"* The measured numbers stand unchanged.

---

# 8 — REVIEWER-HAT ATTACK PASS (the register's composites against *this* draft, plus fresh ones)

| attack | register | how it lands on V1 | the paper's defence | residual |
|---|---|---|---|---|
| **"A unit test on a testbed built to satisfy the theory."** *"You designed an architecturally-invariant analytic double well with an SO(2) sector, placed the channel by oracle, and then verified that your closed-form certificates hold on it. What did I learn that the algebra didn't already tell me?"* | **G1** | ⛔ **Lands hard on §3 — which is 2 of the 6 contributions and the entire headline.** Every §3 number is designed-testbed, oracle-placed, dim 2/4, no training. | **Genuinely strong.** The paper labels §3 *"Reporting grade: verification"* (`tex:70`), tags each contribution *[verification of the theory's exactness; oracle placement.]*, and — decisively — §3.2.1 converts the certificate from a **label** into a **measured downstream consequence** (std(Q_out) 0.0803 vs 0.0000, replicating in 4/4 cells). C-2 compliance is real. | ⚠ **The residual is the Head's restructure, not the current text.** See §9. |
| **"Which component buys what?"** | **G2** | Partially lands. The paper *has* its minus-the-physics control in §3 (the state-replacing map, the random shift, the Newtonian control) and its answer is sharp: **volume alone is not the latch receipt — the matched channel is.** | Strong; the random-shift arm is exactly the ablation a reviewer would demand and it is *in the headline figure*. | ⚠ **App B.2 concedes the harder half honestly:** *"`wormhole_blind` and `no_physics_router` **coincide exactly**… what buys BIBO is **the receipt, not the jump**."* A hostile reviewer quotes that as *"so the physics buys nothing here."* The paper pre-empts it by demoting Payoff B below Payoff A. **Adequate.** |
| **"Toy scale."** *"dim 2 and 4. Five seeds. kv ≤ 32. N ≤ 8. Vocab 256. A laptop."* | **G3** | ⛔ **Lands, and cannot be answered by editing.** | The paper does not hide: *"No claim here is at scale, and none uses learned placement"* (`tex:170`). C-5 compliance is otherwise near-perfect (one exception, V1-13). | ⚠ For a **position/theory** paper this is survivable. For anything pitched as an ML result it is not — which is another reason V1-10 matters. |
| **"Certificate fine print, inverted."** *"det J = 1 certifies volume, not boundedness — and your own appendix shows a **free** ledger escaping to infinity."* | **G5** | The single most invertible thing in the paper. | ⭐ **The paper inverts it first, in three places**, and says so: *"we flag it so the fine print cannot be inverted into the review"* (`tex:252`). **This is the best-executed defence in the document and should be the model for §4.** | ✅ **Closed.** |
| **"Foundational-paper falsifications."** | **G6** | Does not land: C-1 was reversed; no audit paragraph exists; the J&P 2026 citation is introduction-only (`tex:34`). | — | ⚠ **But §3.3 is G6's ghost** — a titled subsection defending against the program's own null (V1-18). |
| **"Salami / de-anonymization."** *"An anonymous companion note, a program-internal `draft.md`, an N-numbered negatives registry, and a 'forthcoming work' on entrance-steering."* | **M2/M3** | ⛔ **LANDS, and this pass makes it worse-looking than it is.** The base cites `draft.md` **twice in the printed body** (`:193`, `:247`), cites an unpublished anonymous note **four times**, and its App D uses **program-internal N-numbers** (N1, N2, N3, N24, N30, N31, N23, N27) as if they were public identifiers. | The N-numbers are at least locally defined; `draft.md` is not. | ⛔ **`draft.md` is the acute one** (V1-03). The N-number convention is a **fresh finding**: to a reviewer, *"App. D N31"* reads as a citation to a document they do not have. **Recommend numbering them locally (D.1…D.7) at typeset.** |
| ⭐ **FRESH — "Your paper contradicts itself twice, and I found both in ten minutes."** | new | §3.2's heading vs its own body; App D's *"provisional"* vs §4.3's *"settled"*. | none | ⛔ **V1-01, V1-06. This is the review-killer** — not because either is substantive, but because both signal the artifact was not read end-to-end before submission. |
| ⭐ **FRESH — "Your two kv32 savings numbers differ by 9×, in one paragraph."** | new (C-7 / N36 class) | `tex:140`: *"rationing (9.9× intra-CLU)"* and *"kv32 gives 1.14×"*, 40 words apart. | The A.5 cross-section note explains the epochs axis in general. It does **not** reconcile these two. | ⛔ **V1-04.** This is the exact defect the referee stage caught pre-submission in V2 (**N36**). It is live here. |
| ⭐ **FRESH — "You call it settled and quote the pilot."** | new (CM-22(bb)) | *"9–10×"* under the word *settled*, when the settled grid measures 9.9 / 9.5 / **6.2**× and falls with load. | none | ⛔ **V1-05.** A reviewer who opens `regime-remap-2000ep`'s Item-1 table finds 6.2× at kv96 immediately. |
| ⭐ **FRESH — "What is $X$?"** | new (craft) | `p^\top X\Delta` in the **abstract**, `X` never defined in the base. | none | ⛔ **V1-11.** An undefined symbol in an abstract is a desk-level defect. |
| ⭐ **FRESH — "Your central theorem is an assertion."** | new (craft / G1-adjacent) | `tex:55–58` states $Q_T \subseteq C_T$ with no proof and no pointer; `draft.md:49` attributes it to **Prop-A2**. | none | ⛔ **V1-12.** The one thing that makes the reach/escape dichotomy falsifiable is unsupported in the artifact. |
| ⭐ **FRESH — "Your escalatability claim never met a floor."** | new (charter §4.1 genuine-win bar) | §4.1's comparison set = {full-budget CLU, modern Hopfield}. No matched-compute feedforward line, no matched-bytes exemplar store. | The paper honestly reports Hopfield winning on cost and noise. | ⛔ **V1-10 + §4-Q2.** The one attack that requires an **experiment**, not an edit. |
| ⭐ **FRESH — "You cite the occupied venue and never engage it."** | new (related-work) | Gladstone 2025 (EBTs), Graves 2016 (ACT), Banino 2021 (PonderNet), Schuster 2022 (CALM), Raposo 2024 (MoD) all appear at `tex:30` and `tex:169` **as genre**, never as baselines — while `tex:140` claims no one-shot memory can escalate. **Titans and DEQs are absent entirely.** | §5's prior-art-honesty paragraph is otherwise excellent (it disclaims novelty of nonlocality, stochastic escape, MCMC and the causal bound). | ⚠ SHOULD-FIX: the disclaimer covers the *physics* lineage thoroughly and the *test-time-compute* lineage not at all. |

---

# 9 — THE APPENDIX-TRAVEL ASSESSMENT (Head ruling 1: "lead with wins; negatives brief in main text, fuller in appendix")

⭐ **Per task §1: a rider that must stay adjacent to a leading claim is a finding; a rider that travels safely is not.** This is the Add.76 measurement — *"a materially less-qualified paper without a single number having been edited."*

## 9a. ⛔ Riders that CANNOT travel — moving them manufactures a claims violation

| rider | currently at | the claim it qualifies | why it cannot move |
|---|---|---|---|
| ⛔ **"Reporting grade: verification… oracle channel placement, dim 2/4, 5 seeds, laptop-CPU, $\gamma=0$"** | `tex:70` (§3 head) | **every number in §3** — i.e. the two headline contributions | Under C-2 this is the *grade label*, not a caveat. Detached, §3 reads as an empirical result. **G1 lands unopposed.** |
| ⛔ **"at a $400$-epoch budget" + "at convergence… accuracy **matches** full-budget… the payoff is rationing, not accuracy"** | `tex:41`, `tex:140` | the **4.81×** headline and its accuracy gain | This *is* MF-C. Move it and the paper re-asserts an accuracy gain it retracted. **The single highest-risk relocation in the document.** |
| ⛔ **THE NOISE WALL** (0/6 close; gate 0.36 vs Hopfield 0.71 despite fidelity ≈1.0) | `tex:153`, Fig-2c | the **Δ+0.02 reversal**, V1's only external positive | **CM-8 makes it explicit: *"travels with every reversal claim."*** ⛔ **It is not a "negative" that can take a brief mention — it is the scope of the win.** ⚠ It also has no App-D entry to travel *to* (**V1-07/G-3**). |
| ⛔ **"intra-CLU… never a cost win over Hopfield"** | `tex:43`, `:151`, `:153`, `:157` | every "9–10×" | The paper's own instruction: *"We state this next to every accuracy-improvement claim"* (`tex:151`). Detached, the number reads as a Hopfield saving — a **named forbidden form**. |
| ⛔ **"energy-**gating** it **loses** to a $449$-param physics-free router"** | `tex:148` | the one-hop-edge contribution (5) | **CM-7 forbids energy-as-routing-signal.** Keep the edge in main text and demote the loss and the reader infers the gated edge is the win. ⚠ §4.2 is *titled* with its own concession (*"and where energy-gating it loses"*) — **that title is load-bearing and must survive the restructure.** |
| ⛔ **C-6's three fine prints** — Prop-12's matched-quadratic scope · *"volume alone is not the latch receipt"* · *"a free ledger does not buy BIBO"* | `tex:72`, `:78`, `:79`, `:118` | the certificate stack | **C-6 requires main-text adjacency by name.** These are also the paper's inversion-proofing (G5) — the one attack it currently closes. |
| ⛔ **LTT exchangeability + ECE $\approx0.100\pm0.021$** | `tex:141` | the 30/30 coverage certificate | **C-6, named explicitly in the register (P20).** |

## 9b. ✅ Riders that travel safely

| rider | why it survives relocation |
|---|---|
| ✅ §3.3's prior-null disambiguation (`tex:82`) | Defensive, not qualifying. **Better in App D under N1**, where md:447 already carries it (V1-18). |
| ✅ App B.2's BIBO battery entire | Already appendix; the paper correctly demotes it below Payoff A and states why (*"Payoff A carries no such caveat, which is why it, not this, is the headline panel"*). |
| ✅ App F's four design rules' derivations | Already appendix; §5 keeps the four rule *statements* with their receipts. **Model execution of the split.** |
| ✅ N23, N30, N31 | Genuine future-work anchors; no main-text claim depends on them. |
| ✅ The dim-4 replication and the seed{0,7} cells | Robustness; main text says "dim 4 identical". |

## 9c. ⛔ The structural finding the Head needs before the restructure

**V1's "wins" and V1's "verification grade" are the same object.** Contributions 1–3 (the certificate stack, the two failure modes, the receipt cashed out) are *all* designed-testbed, oracle-placed, verification-grade. Contributions 4–6 are the learned-memory pillars — and **all three are boundaries or scoped negatives.**

⇒ **"Lead with wins, negatives to the appendix" applied mechanically produces a main text that is 100 % designed-testbed self-verification with the entire learned-memory perimeter in the back matter.** That is precisely the Add.76 failure — *a materially less-qualified paper without a single number having been edited* — and it converts **G1 from an attack the paper currently survives into one it cannot.**

⛔ **The paper's own sentence is the counter-argument, and it is currently in main text at `tex:45`:** *"three of our six contributions are boundaries or negatives. §4 is the paper's honest perimeter, not an incidental ablation."* **If §4 moves, that sentence becomes false.** ⚠ **Flagged, not decided — the restructure is the Head's.**

---

# 10 — MISSING-EXPERIMENT LIST FOR THE HUB

⛔ **Distinguishing "genuinely missing" from "exists in outputs but not cited" — the latter are wiring notes (must-fix), the former are task candidates.**

## 10a. EXISTS IN OUTPUTS BUT NOT CITED — **wiring notes, no compute**

| # | what exists | where | why the base needs it |
|---|---|---|---|
| W-1 | **All five flag-provenance tables** | `draft.md:193–282` | C-7. = **G-1 / V1-03** |
| W-2 | **All four appendix grids** (C.1–C.4) | `draft.md:328–442` | C-10, and `tex:83` cites Table 2. = **G-2 / V1-03** |
| W-3 | **N2b, the noise wall registry entry** | `draft.md:449` | C-9, CM-8. = **G-3 / V1-07** |
| W-4 | **The settled intra-CLU savings curve 9.9 / 9.5 / 6.2× across kv32/64/96** | `outputs/regime-remap-2000ep.md` Item-1 table | Replaces the pilot's "9–10×". = **V1-05** |
| W-5 | **Prop-A2, theory-note §7.4, the definition of $X$** | `draft.md:49, 70, 72` | = **V1-11 / V1-12** |
| W-6 | **N90 / N95 / N103 / N117 in full** | `negative_results.md:118, 123+899, 131, 145`; `claims_matrix.md:574` | The §3.5 fold. **All C1/C2, all quotable today, zero compute.** |

## 10b. GENUINELY MISSING — **task candidates for the Hub**

| # | experiment | what it decides | why a venue reviewer demands it | est. |
|---|---|---|---|---|
| **X-1** ⭐⭐ | **Matched-bytes exemplar-store tripwire on §4.1's own MQAR cells** (arms: calibrated CLU gate · 1-shot exemplar/kNN over the same keys at matched bytes · modern Hopfield; report accuracy at matched **compute** *and* matched **bytes**, plus `margin_rel` vs −0.02) | Whether CM-2's escalatability claim clears the **genuine-win bar** or is supplementary-only | **The §4-Q2 answer.** Criterion 4 has six confirmations and one survivor; V1's primary claim has never faced it. **This is the highest-value experiment on the list.** | laptop |
| **X-2** | **A matched-compute feedforward floor in §4.1's own setting** (the N90/N95 baseline transplanted onto MQAR kv{16,24,32}) | Whether `tex:168`'s *"real ML advantage"* survives, or must be scoped to its internal comparison set | Closes **V1-10** by measurement rather than by hedging. **A reviewer will ask "compared to what?" and the paper currently has no answer outside its own family.** | laptop |
| **X-3** | **The §4.2 hard band** — a routing testbed with **non-linearly-separable** cues | Whether the 449-param router's dominance is a task-ceiling artifact | The paper *names this gap itself* — *"linearly-separable cues drive the router's dominance; **a harder band is the untested fair stress test**"* (`tex:148`) — twice, and never runs it. **A reviewer will quote the paper's own admission back at it.** | laptop |
| **X-4** | **$\gamma>0$ re-absorption sweep** | Closes the one certificate row verified only to leading order at $\gamma=0$ (`tex:174`) | The certificate stack is the contribution; **one row of it is incomplete and the paper says so.** Cheap. | laptop |
| **X-5** | **Multi-seed on N103's τ sub-claim** (currently **1 seed**, "multi-seed owed" on the record) | Whether items 2 and 4 can be quoted under a single seed count | ⛔ Without it, folding both §3.5 items in forces **three different n's** (5, 6, 1/3) into one document — a C-7 cross-section defect built at drafting time. | small |
| **X-6** | **Certifying exits on a *learned* $V_\theta$** (sub-level-set estimator, or restore the $\alpha\|q\|^2$ confinement) | Removes the last oracle from App B.2 | The paper names it as future work (`tex:173`); **both its oracles (channel placement, coercive screen) are on the same critical path**, and G1's residual is exactly "everything is oracle-placed." | medium |

---

# 11 — THE THREE SENTENCES A HOSTILE REVIEWER WOULD QUOTE

> **1.** *"The discriminating experiment: **reach steps then collapses**; the wormhole is flat"* — §3.2 heading — **against** the same paper's *"The **squeeze prices reach; it does not fail at the box**… the falsifiable content, **sharpened from a collapse into a pricing law**"* (§3.2, forty lines below). *"The authors' headline section is titled with the framing their own body says they abandoned. I cannot tell which claim I am being asked to evaluate."*

> **2.** *"**N2** abstention-vs-Hopfield unwinnable; Hopfield dominant at 500 ep (**provisional — under-trained map**)"* — Appendix D — **against** *"Accuracy (**settled** — full grid, 198 jobs)"* — §4.3. *"The paper files its own principal empirical result as settled in the main text and provisional in the appendix, and asks me to trust the tables — which, for Appendices A and C, it does not include, instructing me instead to consult `draft.md`."*

> **3.** *"The honest pillars of §4 delimit where this buys **a real ML advantage (escalatable rationing)**"* — §5 — *"The comparison set for that advantage is the authors' own full-budget model and a one-shot Hopfield. There is no matched-compute learned baseline and no matched-bytes exemplar store anywhere in this paper. 'Extra compute buys accuracy' compared to spending less of your own compute is not an advantage; it is a definition."*

---

# 12 — OPEN QUESTIONS / RISKS

1. ⛔ **The base and the canonical file are each stale where the other is clean** (V1-01/V1-02 base-only; V1-17 canonical-only). ⚠ **A one-directional sync in either direction re-introduces a defect.** Whoever executes the worklist must sync **per item**, never per file. The CHANGELOG's *"Both draft.md and draft.tex updated"* is not reliable — verified false for the §3.2 heading.
2. ⚠ **V1-04's fix must not be a find-replace.** `9.9×` and `1.14×` are both correct numbers from different runs; the fix is a **flag**, not a deletion.
3. ⚠ **Folding CM-23(aa) risks manufacturing a false identity** between V1's LTT exit threshold and N117's cos₀ cutoff (§7 item 4). This is a drafting hazard the Head should see **before** the sentence is written, not at referee.
4. ⚠ **Items 1 and 2 of §3.5 would convert App F.6 from a specified future experiment into a reported one**, changing §5's grade. That is a structural revision, not a paragraph.
5. ⚠ **X-1 and X-2 gate the restructure, not just the text.** If the Head leads with the escalatability win and the tripwire later fires, the lead claim becomes supplementary-only after the fact.
6. ⛔ **The theory note (MF-1) is still un-arXiv'd and is still cited four times, once for a numbered proposition the paper cannot derive** (T3/Prop-12). The §3 classification is the input to that decision and is delivered here; **the decision is not mine.**

---

## Proposed handover updates (for the Hub)

- **V1's stale-draft ledger row (charter:223) needs two corrections.** *"CM-23(b) split is IN"* → **not present in either file, in any form**; and the row should gain *"(g) absent · the score sentence absent · §A20.5 absent · App A/C are stubs."*
- **`v1-ttcl/README.md` carries two inaccuracies** worth an erratum banner (never a body edit, per matrix §3.1 C-3): the *"9 theory-note mentions / MQAR 7×"* counts are `draft.md`'s, not the base's (base = 4 and 5); and *"the venue-class header is stale"* describes a `%` comment that does not print.
- **Two registry rows are now stale relative to the paper and need a curator fold** (flagged, never edited): **CM-12** still carries the MF-B-falsified *"provably cannot beat $C_T$"* alongside its own corrected bracket — it contradicts itself and the paper is ahead of it; **CM-2** still carries the MF-C-retracted *"extra compute buys accuracy… with gains."* ⛔ **A later pass reading either literally would revert a correct paper.**
- **CM-8 vs the drafts: `6–10×` (approved) vs `9–10×` (four printed sites).** Needs a Head/curator ruling on which is canonical before the revision spoke launches; the underlying grid says **9.9 / 9.5 / 6.2×**.
- **Six wiring notes (W-1…W-6) need one owner**; four MUST-FIX contradictions (V1-01, V1-04, V1-05, V1-06) need one revision spoke; two experiments (**X-1**, **X-2**) need Hub scoping and gate the "lead with wins" restructure.
- **§9c is a decision item for the Head, not a finding to file:** V1's wins and V1's verification grade are the same object, so a mechanical lead-with-wins restructure moves the entire learned-memory perimeter to the appendix and falsifies the paper's own §1 sentence.
