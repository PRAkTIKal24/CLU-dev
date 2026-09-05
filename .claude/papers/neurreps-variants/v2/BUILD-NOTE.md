# V2 — NeurReps-audience reframe (variant build) — build note

**A SEPARATE VARIANT.** Source = `papers/v2-short/submission/submission.tex` (the stylized r9 build).
Same results, same numbers, same claims, re-expressed in the vocabulary and framing of the NeurReps
audience and written more directly. ⛔ **`papers/v2-short/**` is byte-untouched** — verified by a
per-file `shasum` manifest taken before the task and re-taken after (§8). Nothing under `papers/v2-short/`
is a product of this build.

## 1. Files

| file | what it is |
|---|---|
| `submission.tex` | the variant source |
| `submission.pdf` | **13 pp total**: main **5.69 pp** (one figure) · references **1.63 pp** (45 entries) · appendices **5.68 pp** |
| `supplementary-theory-note.tex` / `.pdf` | copied unchanged from the source build (anonymized theory note, 12 pp) |
| `figs/` | the three figures, copied unchanged from the source build |
| `neurips_2025_ml4ps.sty` | the template actually used (see §2) |

Build: `pdflatex` ×3 (TeX Live 2026, `/Library/TeX/texbin`). **0 errors, 0 undefined references, 0 overfull boxes.**

## 2. Template

Identical to the source build and for the same reason: **neither the NeurReps template nor the NeurIPS 2026
style file is on this machine**, so the closest genuine NeurIPS-family style file present
(`neurips_2025_ml4ps.sty`, textwidth 5.5 in × textheight 9 in, 10 pt, submission mode with line numbers and
the `Anonymous Author(s)` block) is used, with the workshop notice box suppressed so the artifact is
venue-neutral. ⚠ **The page count must be re-measured in the real venue template before submission.**

## 3. Page split — measured, and the 4-pp target MISSED

Measured from PDF word bounding boxes against the text block (top 72 pt, bottom 720 pt, page 792 pt) —
the same instrument the source build used, so the two numbers are directly comparable.

| block | source (r9) | **this variant** | note |
|---|---|---|---|
| title + abstract | 0.71 | **0.70** | |
| §1 Introduction | 0.49 | **0.63** | +0.14: the mandatory units conversion (flow-Jacobian vs Hessian eigenvalues) |
| §2 Related work | 0.88 | **0.95** | +0.07 *net* — the whole bridge literature added, paid for by moving the retirement elaborations to Appendix D |
| §3 Setup | 0.79 | **1.24** | Figure 1 floats into this span here; §3 prose is *shorter* than r9's |
| §4 Results | 1.92 | **1.63** | figure floated out of this span; prose tightened |
| §5 Discussion | 0.94 | **0.55** | r9's number included Figure 2, now in the appendix |
| **MAIN TOTAL** | **5.72** | **5.69 pp** | ⚠ **against a ≤4 pp target — MISSED** |
| references | 1.01 (28) | **1.63 pp (45)** | 15 new scout-verified audience citations |
| appendices | 4.26 | **5.68 pp** | + Figure 2 demoted here, + new Appendix D |
| **TOTAL** | **11 pp** | **13 pp** | ⚠ against an 8–9 pp target band — MISSED |

Main text is **3,872 words** plus one figure.

### 3.1 Why 4 pp is not reachable, stated as arithmetic rather than as an excuse

The Add.35 proportion package was applied **by construction, not retro-fitted**: Figure 2 was never promoted
to the main text (−0.39 pp against r9 by direct measurement), §5 was written to EA proportion from the start,
and the retirements were compressed in §2 to one sentence each with their elaborations relocated to an
appendix. Those moves bought back the entire cost of the reframe's new §2 — main text came out **0.03 pp below
r9** while carrying a full bridge-literature section that r9 does not have.

What blocks the remaining 1.69 pp is **protected content**, not prose:

- fine print (a)–(c) + the FDT/kinetic-mode mandatory flag box (§3): **0.24 pp**, C-6, must sit next to the claims they qualify;
- the §5 scope box: **0.28 pp**, C-6;
- the chain-length scope clause (§4.3): verbatim, unmovable from the erosion claim;
- §4's three results subsections are number-dense: every quantity is a claim's evidence, and under the
  zero-dropped-findings boundary none may be deleted.

### 3.2 Costed menu for the Head/Advisor — **every item measured by building it, nothing taken unilaterally**

Each row is a real build with that block removed, measured on the same instrument.

| # | move | main | Δ | why it was not taken |
|---|---|---|---|---|
| — | *this build* | **5.69** | — | |
| M1 | Figure 1 → appendix | 5.30 | **−0.39** | the C-3 headline figure; a 4-pp EA with no figure is a worse artifact |
| M4 | §5 scope box → appendix | 5.41 | **−0.28** | ⛔ C-6: fine print must sit next to the claim |
| M5 | §2 bridge paragraph → cut | 5.41 | **−0.28** | ⛔ this paragraph *is* the reframe |
| M6 | §4.2 "price of the prior" → appendix | 5.44 | **−0.25** | ⛔ C-2/C-4: a lead measured result, contribution (3) |
| M2 | fine print (a)–(c) → appendix | 5.57 | **−0.12** | ⛔ C-6 |
| M3 | FDT flag box → appendix | 5.57 | **−0.12** | ⛔ mandatory rider, relocated *into* §3 at r9 for exactly this reason |
| M7 | §4.3 "Honest gap" → appendix | 5.57 | **−0.12** | ⛔ the scope box states the honest gap is part of the claim |
| M8 | §2 retirements → appendix entire | 5.57 | **−0.12** | CM-21 requires them *stated*; already compressed to one sentence each |

**Only M1 is takeable without violating a binding rule.** M1 alone lands main at 5.30 pp. Reaching 4.0 pp
requires firing M1 + at least three rule-violating items. That decision is the Head's, not the writer's.

## 4. What "reframing" changed, concretely

1. **Lead is geometric.** Title, abstract and §1 open on an energy landscape with a group orbit, the curvature
   spectrum transverse to that orbit, and friction/temperature as motion along it. "Mode-mass budget" becomes
   "transverse-curvature budget"; the spectral mass is led with as the transverse curvature and kept as the
   parenthetical name.
2. **§2 rebuilt for this audience.** The continuous-attractor line (S\'agodi's definition quoted), the
   integrator and measured ring/torus codes, diffusion along the manifold, the fine-tuning problem, the two
   prior-art results that *bound our claims* (Renart et al. 2003 on corrective mechanisms; Vafidis et al. 2022
   on learning producing tuning), then the equivariance/level-set/quotient-coordinate census works, then the
   drift disclaimer, then the comparators.
3. **The one conversion is printed once** (§1): flow-Jacobian eigenvalues are inverse time, our $\mu^2$ are
   eigenvalues of a mass-whitened Hessian of a *potential*, inverse time squared; lifetimes are half-lives
   $n_{1/2}$, not $\tau$ and not $D$.
4. **The latch differentiator is stated, not smoothed** (§3): a classical integrator stores by *not* damping;
   at $\gamma=0$ our flat mode never freezes a write and any $\gamma>0$ gives an exact latch.
5. **The three symmetry cells are glossed in audience terms** once each (single fixed point / exactly-flat
   direction / slow manifold), the taxonomy itself unchanged.
6. **ABT openings** on the abstract, §1, and each of §4.1/§4.2/§4.3. **`\textbf` in main text = 0.**

## 5. Boundaries held

- **Approved wordings, mandatory riders, scope qualifiers, fine print: VERBATIM.** The narrow-claim wording,
  fine print (a)–(c), the FDT flag box, the chain-length scope clause, the §5 scope box, the non-comparability
  caveat, the GMOR precision fine print, the negatives-table reading rules, the width-matching confound and
  the continuity sentence all travel unedited.
- **Where an audience term would widen a claim, our term stays.** "Continuous attractor" is used **once**,
  for the literature's object, never for our unit (our object is a designed flat direction of a trained
  potential at dim 4, and our own negatives record that the emergent arm has no coset register). "Solves the
  fine-tuning problem": 0 occurrences. Any phrasing implying learning *cannot* build a flat direction is
  explicitly refused in §4.2 with the Vafidis citation.
- **The no-biological-claim sentence appears exactly once**, in §2, immediately after the ring/torus sentence.
- **`\emph{coset register}` / "register"** are our terms, defined in one clause ("the coset coordinate, the
  position along the flat direction"); the audience has no equivalent as a storage device.
- **"Drift"** is never bare. All 6 occurrences: "cross-session representational drift" scoped in the same
  sentence as a different phenomenon we do not address (×2, one a reference title); "kick--drift--kick" (×2,
  the integrator step); the verbatim prohibition *"never a 'drift rate'"*; and the `latch drift` column header
  of the recovery-ladder table (source-verbatim, our own dynamics).
- **`\emph{budget cube}`** is not used; the object is written out as "the map from $(\mu,\gamma,T)$ to lifetime".
- ⚠ **Known residual (scout F8):** Xu et al.'s title contains "Conformal isometry", unrelated to our
  conformally symplectic map. The body never uses "conformal" for their work; the collision survives only in
  the reference-list title, which cannot be altered.

## 6. Citations

45 entries. All 30 source entries retained. **15 new, every one from the scout's verified records:**
S\'agodi et al. 2024 · Burak & Fiete 2012 · Khona & Fiete 2022 · Seung 1996 · Kim et al. 2017 · Gardner et al.
2022 · Renart, Song & Wang 2003 · Vafidis et al. 2022 · Xu et al. 2023 · Aslan, Platt & Sheard 2023 ·
Akhtiamov & Thomson 2023 · Wang & Ponce 2023 · van der Ouderaa & van der Wilk 2023 · Driscoll et al. 2017 ·
Aitken et al. 2022 · Dinc et al. 2025.
Scout traps honoured: Seung is **sole author**; Jude is first author (never "Perich et al."); Burak & Fiete's
**Eq. 2 is not reproduced** (a verified 2017 erratum whose text the scout could not read); Dinc et al. is cited
as a **preprint** with no PRX volume; PMLR v197 page ranges are reproduced **as PMLR gives them**, including the
Akhtiamov/Aslan p. 181 overlap; PMLR items are cited with the **2023** publication year throughout (one
convention, per scout F1).

## 7. Anonymization — identical to the source build

- `\author{}` blank; the style file's `Anonymous Author(s) / Affiliation / Address / email` block.
- PDF metadata: **Author, Title, Subject, Keywords all empty**; Creator `LaTeX with hyperref`; Producer `pdfTeX-1.40.29`.
- **Decompressed-stream sweep** (81 streams, 17.4 MB inflated): `Forgis` 0 · `x10719pj` 0 · `Users/user` 0 ·
  `Desktop` 0 · `CERN` 0 · `Manchester` 0 · `.claude` 0 · `neurreps-variants` 0 · `/tmp/` 0.
  **Positive control fired:** `Goldstone` 8.
- Third-person self-citation intact: *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"*.
  `CHLU` ×2 in the PDF text (that sentence + its reference entry).
- The theory note is cited as **"Anonymous (2026), provided in the supplementary material"**; the anonymized
  supplementary PDF is copied across unmodified.

## 8. Verification — every sweep per-file, every negative positive-controlled

**Two-way numeric-token check against `papers/v2-short/submission/submission.tex`** (330 vs 390 distinct tokens):
- **In source, absent from variant: 1 token — `0.9`**, i.e. `width=0.9\linewidth` on the two source figures
  (this build uses `0.68` and `0.86`). Typographic, not content. **No content number was lost.**
- **In variant, absent from source: 61 tokens**, every one a citation-metadata fragment of the 15 new
  references (years, volumes, issues, page ranges, DOIs, arXiv ids) or a regex fragment of an existing number.
  **No content number was added or changed.**

**Never-quote + internal-apparatus sweep — 71 patterns, ZERO HITS:** `commit` · `agent/` · `chlu/` · `.claude` ·
`tectonic` · `draft.md` · `draft.tex` · `Registry`/`registry` · `provenance` · `Appendix M` · `N<digits>` ·
`CM-<n>` · `SF-<n>` · `MF-<n>` · `[WORKING TITLE` · `AUTHORS PLACEHOLDER` · `<!--` · CLU-former · certified ·
unlearning · exact deletion · "the item is gone" · "exact discrete FDT" · "samples Gibbs" · 0.384 · 16.28 ·
CAFE · C-MAPSS · HEPA · CAMELS · bpc · S_eff · z_hole · 0.99985 · 54.56 · 306.76 · 300.09 · deltanet · ttt_mlp ·
MUNKEY · 0.4545 · 13.9 · memory vault · 107.77 · compositional · unaskable · Guo · Ginart · Sekhari · Track A ·
waitlist · paid-access · companion · sibling · "our other" · "this program" · "the program" ·
experiment-engineer · "per the Head" · wormhole · scout · Advisor · charter · handover · PREREG · campaign ·
NeurReps · "NeurIPS 2026" · "solves the fine-tuning problem" · "our unit is a continuous attractor".

**HITS, context-checked, compliant:** `workshop` ×2 — a LaTeX source comment about suppressing the notice box
(not in the PDF) and the venue string of the Jawahar & Pierini reference, both as in the source build.

**Semantic hermeticity — ZERO HITS:** "our companion" · "our other paper" · "in our V" · "elsewhere we show" ·
"a companion note" · "our forthcoming" · "we report elsewhere" · "in a separate paper" · "our other work".
**Positive control:** "separate note" ×1 — the sanctioned Anonymous theory-note pointer.

**POSITIVE CONTROLS FIRED:** GMOR 7 · "introduced as CHLU" 1 · Rusch 5 · verification 6 · evidence 9 ·
Anonymous 3 (PDF text) · "transverse curvature" 4 · "continuous attractor" 1 · "fine-tuning problem" 2 ·
45 reference entries · `Goldstone` 8 in decompressed streams.

**Style:** `\textbf` in main text = **0**. No-biological-claim sentence count = **1**.

**Source-build integrity:** a `shasum` manifest of all 20 files under `papers/v2-short/` was taken before any
work and re-taken at the end. `diff` is empty — **byte-identical**.
⚠ *Process note, recorded because it is the kind of thing that must never be silent:* midway through this build
a `pdflatex` invocation ran in the source directory by mistake and regenerated `submission.pdf` (identical
content, different `/ID` and timestamp) plus `.aux/.log/.out`. It was detected immediately by the manifest,
the strays were deleted, and the PDF was restored **byte-for-byte** from an independent copy whose SHA-1
matches the pre-task manifest exactly (`8a38a532ea8d8967195d87880a3c3391b35aa00d`). The final `diff` above is
the proof. `submission.tex` was never written to.
