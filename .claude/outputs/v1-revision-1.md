# v1-revision-1 — paper-writer report

**Task + acceptance criterion:** execute the Head-approved worklist in `.claude/tasks/v1-revision-1.md` (Parts A–E, 5 contradiction fixes · 2 number fixes · 1 restatement + 2 mandatory sentences · 3 completeness restorations · 3 registry folds) against `.claude/NIPSsubmission/v1-ttcl/submission.tex` **and nothing else**; `papers/v1-short/**` byte-untouched; every wording verbatim; every folded claim carries seed count + riders in-sentence.
**Status: done.** All 13 enumerated items landed. Build succeeded (real `pdflatex`, not pseudo-verified): **18 pp → 25 pp**.
**Deliverable #1 `BUILD-NOTE-R1.md` is written** at `.claude/NIPSsubmission/v1-ttcl/BUILD-NOTE-R1.md` (before→after per item ID, ancestors, orphan list EMPTY, numeric check, sweeps, page split, md5 manifest).

**⚠ RECONCILIATION LIST, NEEDS AN OWNER (protocol §5 corollary, stated in the first 10 lines):** this pass produced **one** downstream list — **the three folded results (E1/E2/E3) are in the paper with no flag-provenance table** (App. A covers only §3/§4.1/§4.2/§4.3). Building A.6/A.7 was not enumerated and I did not improvise it. **Owner needed at the review that accepts this report** (finding F-3; source reports and their flag tables are named there).

## DIAL DECLARATION (echoed, protocol §7)
**Dials touched: NONE.** Editorial pass on one `.tex` file. No experiment run, no configuration changed, no new measurement produced. Laundering control: n/a. Falsifies: n/a.

---

## What I did

- Read, in order: `AGENT_PROTOCOL.md` → the **Positioning Charter** as it reads today (`philosophy-synthesis.md:581–600`, incl. **C-1 REVERSED: no audit-confession paragraph** — none was added; the pass adds no defensive audit prose anywhere) → `claims_matrix.md` (CM-8:556, CM-22(bb), CM-23:574 clauses (b)/(g)/(l)/(r)/(y), §A20.5 substrate sentence:607) → `negative_results.md` (N90:854, **N95:899 incl. its ⟲ Head ruling 2026-07-25**, N103:1002, N119:1166, N308:3297) → the task file.
- Applied **22 scripted, assertion-guarded replacements** (`assert count == 1` before every write) + 3 declared post-patches (2 typesetting inside restored tables, 1 phrasing hardening inside E3). Script + machine-readable log under `.claude/scratch/v1-revision-1/`.
- Restored Appendices A and C **from `draft.md` verbatim**, converting markdown tables to LaTeX; restored the **N2b noise wall** into App. D.
- Folded the three Head-approved registry items with their riders in-sentence, in the Head's order of value: **E1 (CM-23(r), the R3-native TIE) as §4.4**, **E2 (CM-23(y) third corner) in §5's position paragraph**, **E3 (CM-23(g) retry-mechanism attribution) in §5 design-rule 2** — and updated **App. F.6's "not run" language in the same edit**, as the task requires.
- Built twice with `pdflatex`, checked the page split, then re-swept.

## How I verified (commands + observed output)

| check | command | observed |
|---|---|---|
| zero unenumerated diffs | reconstruct BEFORE + logged edits, compare md5 to live file | `caef2272f9dc96d349b46486563d24ee` **== live** → IDENTICAL |
| papers untouched | `md5 .claude/papers/v1-short/*` before/after | **11/11 digests identical** (draft.md `00d703d5…4684` unchanged) |
| build | `/Library/TeX/texbin/pdflatex -interaction=nonstopmode submission.tex` ×2 | `Output written on submission.pdf (25 pages, 1144377 bytes)`; `^!` errors **0**; LaTeX warnings **0**; undefined refs **0** |
| baseline build | same, on the pre-edit copy | `18 pages`, same 3 overfull boxes as now |
| numeric two-way | `numcheck.py` (token multisets vs `draft.md:193–282` / `328–442` / `449`) | D3 **exact**; D1/D2 differences all accounted (BUILD-NOTE §2, six classes, none a changed measurement) |
| forbidden forms | `sweep.py`, pure-Python `re` counts (not `grep`/`ugrep`, per the task's hazard note) | **40 patterns at 0**; 10 positive controls > 0; 3 non-zero hits adjudicated (2 pre-existing CM-3 *disclaimers*, present in baseline with identical counts; `2.6` = a measured `2.6×` savings row from `draft.md`) |
| required wordings | same script, case-sensitive exact | **9/9 present exactly once, verbatim** |

**Page split (deliverable #4).** before **18 pp** — main 1–12, App. A p13. after **25 pp** — main 1–14 (§4 p8, **§4.4 p11**, §5 p12), App. A p15, B p17, C p19, D p22, E p22, F p23, refs p24. **Main +2 pp, appendices +5 pp.** ⛔ Not optimised (task §D: report, do not compress; the Head condenses in `pj_sub.tex`).

## Findings / results

### Items landed (all 13)
**A1** §3.2 heading → `draft.md:78` verbatim pricing-law form (the MF-B-falsified "steps then collapses" framing is gone; Figure 1's caption was already pricing-law consistent, so no contradiction remains). **A2** `The router's map` → `An untrained state-replacing map` (`draft.md:31`). **A3** App. D N2 → settled form (under-training artifact · clean/correlated kv≤64 · epoch-budget wall · 6/15). **A4** `$X$` now defined at first use (`draft.md:72`), which retires the undefined-$X$-in-the-abstract defect. **A5** Prop-A2 attribution restored at the $Q_T\subseteq C_T$ derivation and the contributions tag now reads `[proven; theory note + Anonymous 2026.]`.
**B1** all four `9–10×` sites now quote the curve **9.9/9.5/6.2× across kv32/64/96**, each fitted to its sentence (caption included); **B2** the kv32 collision is flagged, not deleted: `9.9× intra-CLU at kv32/2000 ep — §4.3's grid, a different epoch budget from this subsection's`, sitting 40 words from the unchanged `1.14 ± 0.06×`.
**C1** §5's advantage claim is **restated non-comparatively** — the word *advantage* is gone; the sentence now says a rationed relaxation budget buys accuracy **on this memory**, offered as an **empirical proof of concept**, with *"no matched-compute floor and no matched-bytes exemplar store is run against §4.1–4.3 anywhere in this paper"* and the comparison named as future work, pointing at §4.4's tie as the one matched-compute number the paper has. **C2** the scoreboard sentence sits verbatim at the head of §4. **C3** the substrate-scope sentence sits verbatim in §5's Scope paragraph.
**D1/D2** Appendices A and C are no longer stubs pointing at a file no reviewer can open: five flag-provenance tables and four grids are in the base. **D3** N2b (the noise wall) is in App. D, as CM-8 requires it to travel.
**E1** §4.4 = the tie, 6 seeds, its own scope, with the N95 corrected status and the N308 disjunction attached. **E2** the trilemma + its third corner in §5, with the "both fixes refuted / not available" never-quote discharged immediately. **E3** the directed-re-launch mechanism claim in §5 rule 2 with all three mandatory companions, the 5-seed superseded gap range (not the retired seed-0 range), and the F.6 specification downgraded from "not run" to "partially answered on a different substrate; the certified kernel on a trained checkpoint remains unrun".

### ⛔ Defects I noticed and did **not** touch (bounding rule)

- **F-1 — §4.4 is invisible on page 1.** The abstract (`:26`) and the six-item contributions list (`:38–43`) predate the fold: the paper's **only matched-compute result** (a tie) is not among the enumerated contributions, and the abstract still bills §4 as *"three honest pillars"*. I titled §4.4 *"A matched-compute anytime read (beyond the three pillars): it ties"* precisely so that no existing "three" string is contradicted, but a reviewer reading only the first page will not learn that the comparison exists. **Recommend a Head-approved contributions bullet + one abstract clause in pass 2** (C-3: contributions on page 1).
- **F-2 — §4's preamble sentence is now half-true.** `:134` still reads *"§4 leaves the designed testbed for **trained** memories"*; §4.4's store is **designed** with a **learned, frozen** encoder. I neutralised this locally with §4.4's own grade line (*"a designed store addressed through a learned, frozen encoder φ, not a trained CLU checkpoint"*), but the preamble sentence itself is unedited (not enumerated).
- **F-3 — the folded claims have no flag-provenance table (C-7 / protocol §5).** App. A's A.1–A.5 cover §3, §4.1, §4.2, §4.3 only. E1/E2/E3 numbers therefore sit in the paper without commit/seed/flag rows. Adding A.6/A.7 was **not** on the worklist and would have required numbers I was told not to improvise. Source reports that already carry the tables the Hub needs: **`cl-entry-build.md` §"Flag-provenance table" (E1; commits list begins `b6aa1f5`)**, **`retry-compute-study.md`:17 + `multi-seed-w23.md`:19 (E3)**, **`headroom-retry-benchmark.md`:25 (the N95 rider)**, and N308's control (`negative_results.md:3297`, budget grid `[50,100,200,400,800,1200]`). **This is the reconciliation item that needs an owner.**
- **F-4 — the line-2 source comment is stale** (`% V1 workshop short (ML4PS / NeurReps class) … Canonical content: draft.md.`). Not typeset, so it does not reach a reviewer, but it is the last `draft.md` pointer in the file (3 → 1 this pass) and it names a venue class the README says is superseded (**TTCL**; ML4PS has no 2026 edition).
- **F-5 — venue/title/author placeholders untouched** (`[WORKING TITLE: …]`, `[AUTHORS PLACEHOLDER]`) — correct per charter §6, recorded so nobody reads it as an omission.
- **F-6 — App. C's C.4.b heading now carries a filename** (`fig_frontier_clean.png`) because the ancestor line does. Harmless in an appendix, but a pruning pass may want it gone.
- **F-7 — the occupied-venue sentence in §4.4 carries no citations.** I named the venue in prose (deep-equilibrium models, energy-based transformers, recurrent-depth architectures) and added **no** `\cite`, because adding bibliography entries is not enumerated and hermetic-citation discipline (M1) forbids improvising them. **A cite pass should attach the published anchors** (the reference list already carries the HMC/MALA/attention lineage).
- **F-8 — CM-23(r)'s τ sub-claim is deliberately absent from §4.4.** (r)'s τ caveat ("τ is not the binding gate element in φ-space", 1 seed) is **narrowed by CM-22(bb)/CM-23(aa)/N117** as *too general*, and folding (aa) is explicitly out of scope this pass. Rather than state a superseded form, I omitted the sub-claim. ⚠ Carrying the hazard the task flagged, so it is not lost: **V1's τ is an LTT-selected exit threshold on $p_{\rm wrong}$; N117's τ is a cos₀ cutoff on a retry ladder — different objects; conflating them manufactures a false identity.**
- **F-9 — the audit's other line items remain live** (the task approved 5 of 27). Ones I saw while working and left alone: the abstract's "three pillars" inventory (F-1), `:134` (F-2), the 9 "theory note" mentions never assessed for self-containment, and the 7 MQAR mentions flagged in the folder README as an open venue-admissibility question.

### Registry-compliance notes for the reviewer
1. **C-1 (as REVERSED)**: no audit-confession paragraph was added; the folds describe current, fixed mechanisms and never assert legacy mechanism-numbers. J&P 2026 is cited only for the primitive's introduction (`:34`, unchanged, with the mandatory continuity sentence).
2. **C-2 grading is explicit at every folded site**: §4.4 labels itself *evidence* with the substrate stated; §5 rule 2 labels its measurement *designed store*; §3 remains *verification*.
3. **CM-3 never appears**, even hedged; the two pre-existing energy-signal strings are the paper **refusing** the claim and were untouched (baseline counts identical).
4. **Seed counts never merged**: §4.4 = 6 seeds *of its own protocol*; §5 rule 2 = seed-0 ladder re-measured on 5 seeds *of that protocol*; both say so in-sentence, both explicitly disclaim §3/§4's 5 seeds.
5. **CM-22(bb) "quote the curve"** discharged twice: B1's savings curve and N308's three-point oracle-addressing curve (never "0.02 → 0.87").

## Open questions / risks for the Hub or Head

1. **F-1 is the one that changes how the paper reads.** Does the Head want a 7th contributions bullet + abstract clause for §4.4's tie, or does the tie stay a §4-interior result?
2. **F-3: who builds A.6/A.7?** Until then the paper carries three folded results whose configuration a reviewer cannot reconstruct — the exact M4/C-7 failure mode the flag-provenance rule exists to prevent.
3. **Page budget.** 25 pp with main text at 14. If TTCL's limit binds the *main text*, §4.4 (1.5 pp) and §5's E2/E3 additions (≈0.8 pp) are the newest, most compressible material — but per the task I did **not** compress anything.
4. **`draft.md` and `submission.tex` remain deliberately unsynced** (task §1). After this pass the `.tex` is **ahead** of `draft.md` on B1/C1/C2/C3/E1/E2/E3 and equal on A/D. Whoever syncs later must sync *forward*, never re-import the stale `.tex`-side wordings — and must not undo A1–A5, which came *from* `draft.md`.

## Proposed handover updates (for the Hub)

- `v1-ttcl/submission.tex` is now **registry-current on the 13 approved items**: pricing-law heading, the savings curve (9.9/9.5/6.2×), the scoreboard and substrate-scope sentences, full Appendices A and C, the noise wall, and the three C1/C2-banked folds (CM-23(r)/(y)/(g)) with their riders. Build green, **25 pp** (main 14). `BUILD-NOTE-R1.md` is the audit trail.
- **Open, owner-less:** flag-provenance tables for the folded claims (F-3) — convert to a named task at the review that accepts this report.
- **Recommend for pass 2 (Head decision):** the page-1 exposure of §4.4 (F-1) and the `:134` preamble sentence (F-2); both are one-sentence edits and both are *contradiction* class, not polish.
- The remaining audit items (22 of 27) are still unowned; this pass touched none of them.
