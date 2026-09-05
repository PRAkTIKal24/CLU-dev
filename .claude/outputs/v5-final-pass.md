# v5-final-pass — paper-writer report

**Task + acceptance criterion:** V5 v0.4 (fold the `v5-vcurve-validation` results into the canonical, verbatim) **+** the derived `submission/` build (structure · appendix triage · figure promotion · brevity/ABT · de-bold · main ≤ 4 pp hard · total 8–9 pp target · full anonymization).
**Status: done, with one criterion missed and measured** — main text = **4 pp (hard limit met)**; **total = 10 pp against the 8–9 pp target** (costed menu in BUILD-NOTE §7); the second-visual promotion landed in Appendix B rather than the main text (§ "Open questions" 3).

**DIAL DECLARATION (echoed).** Dial: **none — fold + editorial.** Laundering control: n/a (no new measurement). Falsifies: n/a. The only new numbers entering are from the Advisor-verified `outputs/v5-vcurve-validation.md`; **zero numbers invented, changed, rounded up or smoothed**, verified mechanically by a two-way numeric-token check (below).

> ### ⚠ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — needs an owner)
> 1. **The canonical does not tabulate the source report's 9-row instrument-ratio grid** (`v5-vcurve-validation` §3.2), only its summary. I built the submission's Table 3 from the full grid, then **cut it back** to canonical-covered numbers to keep the submission a strict derivative. **Recommend a canonical top-up (that table + its per-δ rows) at the next stage-1-type pass**; it is the strongest single piece of MF-3 evidence.
> 2. **Two citation-id defects I introduced and fixed inside this pass, worth a registry note:** the canonical's Minami & Hidaka (2018) record carries a **DOI and no arXiv id** (I had attached one; restored), and the canonical has **two** 2026 Wang entries whose arXiv ids are adjacent (`2604.20006`, `2606.29788`) — an easy mis-pick. Any future hand-typed bibliography must be diffed against the canonical entry, not retyped.
> 3. **Erosion (canonical App H) is absent from the V5 submission.** That discharges nothing owed — the canonical keeps it with the MF-10/MF-11 corrections — but it **removes V5's only content overlap with the other draft the referee flagged for duplicate-appendix exposure.** Whoever owns that exposure should know the V5 side is now clear.
> 4. **The 4 pp measured at v0.2/v0.3 was a *generic-article* 4 pp.** In NeurIPS-family geometry the same text is ~4.7 pp. Any future page target for a NeurIPS-class venue must be set against that geometry, not the generic class.

---

## What I did

### Stage 1 — canonical → v0.4 (the validation fold). Canonical frozen after this stage; stage 2 never touched it.

| item | where it landed | source |
|---|---|---|
| **1. MF-3 retired and replaced** | §2.1 new "Two instruments, one law" + App **C.3.1** (new, with the four-instrument table) | report §0, §2, §3.1, §3.2 |
| **2. Instrument-floor → model-side bound** | §2.1 + App C.3 (the v0.3 instrument note kept beside the span sentence) | report §3.4 |
| **3. Falsifier upgraded in both directions** | §2.1 falsifier sentence + C.3.1 | report §3.3, §3.6 |
| **4. ME-3 compact + in full** | §2.2 new paragraph + App **D.7** (new, 6 sub-blocks, 3 tables) | report §5–§9 |
| **5. Apparatus** | App **A.10** (new provenance block) · **Figures C.2 & D.2** (banked artifacts) · App-C figures renumbered to document order · **6 new negatives (J.15–J.20)** · Limitations (iv) rewritten · future-work list updated | report §1, §1.2, §10–§12 |

**Numbers folded, verbatim** (each traced in the CHANGELOG to its report section): argmin `0.9001 ± 0.0052` vs `0.9032 ± 0.0027` · slopes `−0.983 ± 0.013` / `+1.117 ± 0.010` vs `−1.0020 ± 0.0003` / `+1.1182 ± 0.0107` · rates `0.995 ± 0.003` · `n_jac/n_R1 = 1.3307/1.7981/1.8204`, CV `2.2–3.7 %`, `d log ratio/d log γ ∈ [−0.0034, +0.0001]` · `max|θ−δ| = 9.400e-13` rad ⇒ `Γ ≤ 7.309e-17` ⇒ `n₁/₂ ≥ 9.483e15` · finite amplitude argmin `0.897–0.917 γ_crit`, ≤`1.4 %` / ≤`1.9 %` · the integrator identity (`692.5` at `γ=0.002`, two seeds `2.7×` apart in `μ²`) · refrigerator `0.9998 ± 0.0019`, `7.955×` vs `7.942`, coupled bath `0.2235` · `D̂/D_absorb = 1.016–1.103` · **law-referenced vault `106.1 ± 5.0×`** · contrast `23.39 ± 10.06` vs `8.03 ± 0.80` (falsifier fired; control off its own law by `1.18–5.23`) · hop fraction `5.5/43.0/2.4 %` → `0.0000`, scalar control `0.73/10.2/0.26 %` · same-site FPT `>1379×` / `35.5×` / `>1290×` at `86.7/0/93.4 %` censoring · `|∇V|` at `θ=π` = `0.042/0.105/0.041` vs `0.0000` · estimator cross-check `112.58 ± 1.09×`, `14.16 ± 1.38×`, `8.03 ± 0.80`.

**Riders enforced**, each at every site: 3 seeds · `T ∈ {4e-3, 8e-3}` both `> T*` · ⛔ **no emergent `σ_θ` ratio anywhere** (stated instead as a void comparison whose control returns `0.4586 ± 0.1181` where it must return `1.000`) · the `θ=π`-is-not-a-vacuum confound on every emergent FPT statement · the contrast number labelled designed-only with its named failure mode · MF-13(a) softened (abstract + contribution (1) now say the vault's *laws* hold on both arms, the *contrast* is designed-only).

### Stage 2 — the derived `submission/` build

Structure (related work → §2; results renumbered §3.1–§3.3) · appendix triage to plots/results-tables only with every drop mapped · brevity/ABT/de-bold pass · anonymization · measured page fit. Full detail in `papers/v5-short/submission/BUILD-NOTE.md`.

## How I verified

**Builds.** Canonical: `tectonic` → `draft.pdf`, **31 pp**, 0 errors. Submission: `pdflatex ×3` (TeX Live 2026, `/Library/TeX/texbin`) → **10 pp, 0 errors, 0 undefined references**, 2 overfull + 7 underfull hboxes (all in `\tiny` narrow table cells; reported, not silenced).

**Page split, measured from the PDF text block** (not estimated):

| block | pages | detail |
|---|---|---|
| main text | **4.00** | pp. 1–4, ends at the foot of p. 4; **2,279 words + 1 figure** |
| references | 0.8 | 25 entries (from the canonical's 58; 33 orphaned by the appendix cut) |
| appendices | 5.2 | 2,967 words + 2 figures + 7 tables |
| **total** | **10** | ⚠ 1 pp over the 8–9 target |

**Canonical main text for comparison: 5 pp / 2,682 words** in the generic `article` class (v0.3 was 4 pp / 2,241). The canonical is deliberately not cut to the venue limit; the limit is met in the derived build.

**De-bold:** `\textbf` in `submission.tex` = **0** (main text *and* appendix). One `\paragraph{Contributions.}` remains — structural.

**Numeric two-way check** (`\begin{tabular}` specs, `\includegraphics` widths and preamble lengths excluded):
- (i) numeric tokens in `submission.tex` **not** in the canonical: **0**.
- (ii) canonical-main-text tokens absent from the submission main text: **23**, of which **21 travelled into the submission appendix**; the 2 truly absent are `2.3` (a canonical section number, now a `\ref`) and `9.40` (present at the source report's own precision, `9.400`).
- ⚠ This check **caught three real defects**, all fixed before the final build: an arXiv id I had attached to Minami & Hidaka (2018) that the canonical record does not carry; a wrong arXiv id on an entry that had become orphaned (dropped); and a 9-row table in the submission whose full grid the canonical only summarises (cut back to canonical numbers — reconciliation item 1).

**Sweeps** (`scratch/v5-final-pass/{sweep,subsweep}.py`, per-file, positive-controlled):
- **Canonical v0.4:** instrument LIVE (17 positive controls incl. `0.9001`, `106.1`, `9.5e15`, `confines`); **forbidden hits = 3, all the benign v0.3 set** (two literature descriptions of Guo's *certified removal*; one "independent of chain length" immediately followed by its k-regime clause). **The six new v0.4 never-quote patterns are all clean** — the withdrawn "19–43 %" wording, the superseded `233.6/653.3/249.0` cell, the stale "unrun" marker, the emergent `σ_θ` ratio, the emergent contrast number as a result, and the measured/measured ratio as a vault.
- **Submission:** instrument LIVE (14 positive controls). **Zero-list hits = 2**, both the same false positive: `n_{\rm jac}/n_{\rm R1}` and `\Gamma_{\rm jac}/\Gamma_{\rm R3}` in a table header matching an `R1`/`R3` internal-label pattern — these are **this paper's own instrument names**, defined three paragraphs above. Context-checked hits, all compliant: `certified` ×3 · `unlearning` ×3 · "deletion is exact" ×2 (both "store-level") · `CHLU` ×2 (continuity sentence + reference) · `0.99985` ×1 (carries "at full load") · `297.8` ×1 (appendix, "never the vault number") · `23.39` ×3 (all designed-only/falsifier-fired).
- **Semantic hermeticity (C-8):** `companion` / `sibling` / `our other short` / `the program` / `forthcoming` / `in preparation` = **0**. Registry tokens (`N\d\d`), `CM-n`, `MF-n`, `SF-n`, source-report names, checkpoint identifiers, commit hashes, project paths = **0**.

**Anonymization:** `\author{}` blank · no `[WORKING TITLE` / `[AUTHORS PLACEHOLDER]` in the submission · no acknowledgment, funding, URL or repository string · PDF **Title/Author/Subject/Keywords/Creator/Producer all empty** · **0** occurrences of any absolute path, username or project string inside the compressed PDF (`strings` sweep) · third-person self-citation intact and the *only* two occurrences of the author names · **no supplementary PDF attached** (V5 is self-contained since v0.3; the theory note is cited nowhere load-bearing) · figures renamed to neutral filenames.

**Do-not-cut list (`v5-referee-v02` §D) — all present in the submission main text:** N108's sentence · the CM-25(f) verbatim quote · the Blelloch–Golovin attribution · the score sentence · the substrate-scope sentence · the designed-symmetry precondition · the `fdt`+Newtonian fine print. Verified by string search.

## Findings / results

1. **The fold is complete and the canonical is the superset.** v0.4 folds all five stage-1 items; the CHANGELOG maps every folded number to its report section. Canonical: `draft.md` 137,870 B, `draft.tex` 145,118 B, `draft.pdf` 31 pp, all written at 02:54 and **byte-unchanged through stage 2** (submission dir written 03:37–03:41).
2. **The hard 4 pp is met, and the cost is quantified.** −269 words (−10.6 %) out of the v0.4 main text, on top of absorbing the fold. §3.3 (deletion) gave up only 6 words because it is almost entirely protected wording; §3.1 gave up 158.
3. **The 8–9 pp total is missed by 1 pp, and every remaining cut contradicts an instruction.** Costed menu in BUILD-NOTE §7. Measured, not assumed: dropping the emergent-vault figure alone does **not** save a page.
4. **The venue-geometry finding.** The "4 pp" of v0.2/v0.3 was a generic-`article` 4 pp; the NeurIPS block is ~15 % smaller, so that same text is ~4.7 pp there. This is the single largest driver of the compression this pass had to do, and it will recur for any NeurIPS-class target.

## Git footprint

**None.** No tracked file created, modified or deleted; `git status --short` empty at start and end. All artifacts under `.claude/` (gitignored): `papers/v5-short/{draft.md,draft.tex,draft.pdf,CHANGELOG.md,figs/}` and `papers/v5-short/submission/{submission.tex,submission.pdf,main_body.tex,appendix.tex,BUILD-NOTE.md,figs/,neurips_2025_ml4ps.sty}`; scripts and backups in `scratch/v5-final-pass/`.

## Open questions / follow-ups / risks for the Hub and the Head

1. **Title, for the Head to workshop.** The submission carries *"Forgetting You Can Budget, Delete and Schedule: a $(\mu,\gamma,T)$ Retention Law and Exact Store-Level Deletion for a Physics-Structured Recurrent Memory"* (the canonical keeps `[WORKING TITLE: …]`). Alternatives, shortest first: (a) *"A Budget for Forgetting: the $(\mu,\gamma,T)$ Law of a Physics-Structured Memory"*; (b) *"Forgetting You Can Budget and Delete"*; (c) *"The Damping Optimum: Retention, Erasure and Exact Deletion in a Physics-Structured Memory"*. ⚠ (b) and (c) drop the "schedule" third contribution from the title, which is honest given it is a mechanics demonstration.
2. **The 1-pp overshoot is a Head decision.** BUILD-NOTE §7 prices six ways to close it; all six cost content the pass was instructed to keep.
3. **Figure promotion landed in the appendix, not the main text.** The rollout overlay is Appendix B's lead figure. Promoting it to main costs ≈0.35 pp, which the hard 4 pp does not have; the trade is *"Figure 2 in main, and one of the protected §3.3 blocks moves to Appendix D."* I did not take that trade unilaterally.
4. **Missing-experiment note (unchanged from the analyst's list, restated so it does not go stale):** the designed-arm same-site FPT control (the `θ=π` baseline was never checked on designed, where symmetry should protect it) is **owed** and is cited in the paper's own negatives table as owed.
5. **Panel labels retained in banked figures.** Figure 3's panel titles carry the pre-registration item labels (Q1, Q2, Q3, Q5) and Figures 2–3 carry seed short-tags. These are not de-anonymizing, and the captions define them, but a re-render would be cleaner if a figure pass is ever commissioned.

## Proposed handover updates (for the Hub)

### Paper state
> **V5 = v0.4 (canonical) + a built submission artifact.** Canonical `papers/v5-short/draft.{md,tex,pdf}` at v0.4, 31 pp, frozen 2026-08-19 02:54. Derived `papers/v5-short/submission/` — **10 pp total, main text 4 pp (PALM short-track hard limit MET)**, 25 references, 5 appendices, 3 figures, 7 tables, fully anonymized, metadata scrubbed, `\textbf` = 0, numeric two-way check clean in both directions. Template used = `neurips_2025_ml4ps.sty` (venue-neutral, notice box suppressed); ⚠ **re-measure in the real PALM template before freeze.**

### For the next curator pass
> Fold the four `v5-vcurve-validation` reconciliation items **and** this pass's two new ones: (a) the canonical lacks the source report's 9-row instrument-ratio grid (summary only) — top up before the table is wanted in a submission again; (b) the canonical bibliography carries two adjacent 2026 Wang arXiv ids (`2604.20006`, `2606.29788`) and a Minami & Hidaka 2018 record with a DOI and **no** arXiv id — both are hand-retyping hazards and belong in the citation fence.

### Standing measurement fact (do not re-derive)
> **Generic `article` 4 pp ≈ 4.7 pp in NeurIPS geometry** (text block 6.5 × 9 in vs 5.5 × 9 in). Every page target for a NeurIPS-class venue must be set in that class. V5's main text is **2,279 words + 1 figure = exactly 4 pp** there; that is the working ceiling.
