# v2-bib-doi-list — web-scout — assemble the identifier block for V2's `refs.bib` (Zenodo-pasteable) + resolve three open citation sites

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 64).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/v2-bib-doi-list.md`.

**DIAL DECLARATION: none — read-and-report. ⛔ Zero edits to any paper file.** You touch nothing in `.claude/NIPSsubmission/`; your only write is your report.

**Context:** the Head is converting V2 from a hand-built author-year `\section*{References}` list (52 entries) to a BibTeX `refs.bib`. They will paste your identifier block into Zenodo's identifier tool to generate the `.bib`, then a later spoke wires `\cite` keys into the prose. **Your block is the input to that whole chain — its accuracy and completeness are the deliverable.**

## Sources — reuse before you re-verify
1. `.claude/NIPSsubmission/v2-neurreps/pj_sub.tex` — the live paper. Its `\section*{References}` (from l.118) is the authoritative list of what is cited. ⚠ The Head edits this file; read it once at start and state the md5 you read.
2. `.claude/NIPSsubmission/v2-neurreps/submission.tex` — the clean base; **its bibliography still carries DOIs that `pj_sub` stripped**. Harvest these first.
3. `.claude/outputs/v2-cite-check.md` — **26 works already primary-source-verified** (records, DOIs, and the never-copy traps). ⭐ **Carry these forward; do NOT re-verify them.** Mark each identifier in your report as `CARRIED` (from 1–3) or `NEW` (you verified it this pass).
4. `.claude/outputs/v2-referee-experiments.md`, `audience-refresh-2025-2026.md`, `neurreps-audience-scout.md` — banked records for the 2024–25 venue works.

## Deliverable 1 — ⭐ THE BLOCK (the Head pastes this verbatim)
A single fenced code block, **one identifier per line, nothing else** — no numbering, no commentary, no blank lines, no trailing punctuation. Ordering: same order as the paper's reference list.
- **DOI preferred**, bare form (`10.1234/xyz`), no `https://doi.org/` prefix.
- **arXiv ID** (`arXiv:2501.00663`) only where no DOI exists. For a work with both, give the DOI.
- ⛔ **Never invent or pattern-guess an identifier.** An unverified identifier is worse than an absent one — it silently produces a wrong `.bib` entry the Head will not catch. Anything you cannot resolve goes to Deliverable 3, not into the block.

## Deliverable 2 — the coverage table
One row per reference entry: author-year short form · identifier in the block (or `—`) · `CARRIED`/`NEW` · the source that established it. This is how the Head checks the generated `.bib` against the paper.

## Deliverable 3 — the residual list (entries with no resolvable identifier)
Per entry: what it is, what you tried, and **the fields the Head needs to hand-enter** (author, title, venue, year, pages). ⚠ Expect these among older works (Snyder 1977, Micciancio 1997, Naor & Teague 2001, Golubitsky/Stewart/Schaeffer 1988, Krupa 1990, Seung 1996, Rumberger 2001) and workshop/PMLR items. Declare honestly rather than approximating.

## Deliverable 4 — ⭐ the orphan / cited-nowhere list
Cross-check every reference entry against the body text. Report: (a) entries **cited nowhere** — a referee has already flagged decorative entries as a credibility cost, and the Head must rule cite-or-cut **before** the `.bib` is built; (b) any in-text citation with **no matching entry** (dangling). ⚠ Use per-file greps with positive controls — directory-level grep over `.claude/` silently returns nothing.

## Deliverable 5 — three open citation sites needing candidate works
The Head's `\TODO` tags mark three sites where **no work is chosen yet**. Propose **2–3 candidates each, ranked, with one line on what each actually claims and why it fits** — ⛔ you recommend, the Head selects; do not present one option as decided.
1. **l.33 — `\cite{}` + `\TODO{cite welling and another SSB paper}`**, at: *"Symmetries supply these directions natively through spontaneous symmetry breaking."* Needs the Welling-lineage work the Head has in mind (equivariance/group-structured representation learning) **plus** a genuine SSB-in-neural-networks reference. ⚠ Check the 2025–26 NeurReps room first (`audience-refresh-2025-2026.md`: *symmetry breaking* is now titular vocabulary there) — a venue-native citation is worth more than a generic one.
2. **l.33 — `\TODO{cite numbu-goldstone paper/review paper}`** (sic), at: *"This neutral coordinate acts as the Nambu-Goldstone mode of the trained potential."* Needs a standard NG-mode reference or review. ⚠ Prefer one a geometry/ML audience can actually use; note if the program already cites a suitable one (Minami & Hidaka 2018 is in the list — say whether it serves or whether a primary/review cite is additionally needed).
3. **l.33 — `\cite{}` + `\TODO{relevant citations}`**, at: *"while the theoretical existence of such a flat direction is established."* ⚠ **This site is claims-sensitive.** The binding boundary (charter Add.49): the zero-Lyapunov-exponent theorem and the qualitative breaking⇒lifetime prediction belong to **arXiv:2605.03338**, and continuous-attractor flat directions and their fragility are prior art (**Renart, Song & Wang 2003**; **Ságodi et al. 2024**; **Burak & Fiete 2012**), all already in the reference list. Recommend from what is already cited wherever possible, and flag explicitly if a proposed new work would narrow one of our novelty claims.

## Already-known identifiers (do not re-research; verify format only)
- **`\TODO{cite CHLU}` (l.35), the self-citation:** Jawahar & Pierini 2026 — **arXiv:2603.01768** (banked, charter Add.23). ⚠ Third-person self-citation is the sanctioned double-blind form; the entry keeps its authors.
- **l.48 `\TODO{cite}`, l.86 `\cite{}`, l.96 `\cite{}` — all three are the same work:** the single-exponential lifetime estimator, **arXiv:2605.03338** (a preprint, never peer-reviewed; check whether a DOI now exists and report either way). ⚠ **In your report you may name its author; in the paper it appears only as a citation** (charter Add.51).

## Constraints
- **Verify against primary sources** (publisher/PMLR/arXiv landing pages); record what you could not reach rather than inferring around it (OpenReview is bot-walled; `web.archive.org` has been tool-blocked before).
- Preserve the never-copy traps banked in `v2-cite-check.md` (Souza not de-Souza · Pfahringer · the Webb `-015-` DOI on a 2016 paper · the JMLR-not-PyPI river record · Rusch & Mishra 2021a/b · Fischer–Igel **2010** for the landscape-distortion result · conformal symplecticity = **McLachlan & Perlmutter 2001**, not HLW).
- ⚠ If any reference entry's *content* looks wrong (wrong year, wrong venue, misattributed result), flag it — do not silently correct it into the block.

## Acceptance criteria
- Deliverable 1 is a single clean block, one identifier per line, every line resolvable, **zero guessed identifiers**.
- Deliverables 2–5 complete; every identifier marked CARRIED or NEW with its source.
- Zero writes outside `.claude/outputs/v2-bib-doi-list.md`; `pj_sub.tex` untouched (state its md5 at read time).
