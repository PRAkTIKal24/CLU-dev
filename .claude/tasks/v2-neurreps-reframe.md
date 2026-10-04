# v2-neurreps-reframe — paper-writer (V2, reframed in the NeurReps audience's own terms; a SEPARATE variant)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 37; Head directive 2026-08-20).** ⛔ **Mechanical precondition: `.claude/outputs/neurreps-audience-scout.md` exists on disk.** Read `.claude/AGENT_PROTOCOL.md`, then this file.

**You write ONLY into `papers/neurreps-variants/v2/`** (create it): `submission.tex` · `submission.pdf` · `BUILD-NOTE.md` · `figs/`. ⛔ **Nothing under `papers/v2-short/` is touched — not the canonical, not the current submission build.** That artifact stays exactly as the Head left it; this is a parallel variant, kept apart to avoid confusion.

**DIAL DECLARATION: none — reframing/editorial pass; zero content, number or claim changes.**

## Source and the one job
Source = `papers/v2-short/submission/submission.tex` (the stylized r9 build). **Same results, same numbers, same claims — re-expressed in the vocabulary and framing of the NeurReps audience**, and written more directly than r9.

The audience (Head-supplied census, verified records in the scout report): geometric deep learning / equivariance (Lie-group convolutions, Lie-algebraic representations, SO(3)/SE(3) equivariant nets, fundamental-domain projections, group-invariant learning) · neuroscience (population variability across sessions, information geometry of population codes, level sets and invariance manifolds of tuning landscapes, inter-areal interactions, topological ensemble detection) · topology and learning theory.

## What reframing means here, concretely
1. **Lead with the geometry, not with our machinery.** The object is an energy landscape with a **group orbit** along which the curvature vanishes; retention is set by the **curvature spectrum transverse to that orbit**; friction and temperature move state **along** it. Use the scout's vocabulary map: orbit / coset space `G/H` · **invariance manifold** · **level set** · zero-curvature (flat) direction · Hessian curvature spectrum · equivariance error · **continuous attractor** where the scout confirms the mapping is exact. ⛔ Where the map says *no equivalent*, keep our term and define it in one clause — never borrow a term that means something else in their field.
2. **Name the phenomenon classes this audience already studies, and state plainly what we do NOT claim.** Drift along a continuous attractor and cross-session representational drift are their objects; our finite-temperature diffusion along the coset is the same geometric object with a computable rate. ⛔ **We make no claim about biological systems and do not model neural data** — the scout supplies the honest contrast sentence; print one, once, in related work.
3. **Related work (§2) is rebuilt for this audience**: the equivariant-network line (including the head-to-head anchor, which is already native here), invariance-manifold / level-set work, the drift literature, and the four retirements compressed to one sentence each. Every new citation comes from the scout's verified records — ⛔ nothing cited that the scout did not verify.
4. **The two axes stay the paper's spine** (the realization taxonomy is already geometric); express them as: which symmetry the trained potential realizes, and what the curvature spectrum then permits.

## Style (strict — the Head's directive, tightened)
`.claude/PJ_Writing_Style_Context.md` applied strictly and more directly than r9: ABT openings (abstract, §1, each results subsection) · macro-to-micro (geometry → mechanism → number) · **short declarative sentences; plain technical terms; one idea per sentence** · zero weasel words · signposting · "we" for our actions, passive for established facts · **no bold in main text outside structural headers** · numbers carry their scope in the sentence.

## Boundaries (absolute)
1. ⛔ **Matrix-approved wordings, mandatory riders, scope qualifiers and fine print: VERBATIM.** Reframing may change the prose *around* them; it may never paraphrase them, and it may never re-describe a result in audience terms that widen it. If an audience term would widen a claim, keep our term.
2. ⛔ Zero number changes, zero new claims, zero dropped findings. Run and print the two-way numeric-token check against `papers/v2-short/submission/submission.tex`.
3. ⛔ Never-quote sweep + internal-apparatus sweep + semantic hermeticity pass, all per-file with positive controls, printed.
4. Anonymization identical to the source build (blank author block, scrubbed metadata, third-person self-citation, the theory note as Anonymous-supplementary — copy the supplementary PDF across).

## Shape
Main text **≤ 4 pp** (NeurReps EA hard limit) · references and appendices excluded from that count · total in the **8–9 pp** band. The Add.35 proportion package applies here by default since this is a fresh build: Figure 2 lives in the appendix unless the main text fits with it; related work and discussion are written to EA proportion from the start (~300 and ~250 words), not compressed after the fact.

## Acceptance criteria
1. Main ≤ 4 pp; total 8–9 pp; page split printed.
2. Every audience term used is either on the scout's map as exact/approximate (with the approximation stated once) or defined in one clause; every new citation is scout-verified.
3. The no-biological-claim sentence is present exactly once.
4. Numeric check + all sweeps printed; `papers/v2-short/**` byte-untouched (state the check).
