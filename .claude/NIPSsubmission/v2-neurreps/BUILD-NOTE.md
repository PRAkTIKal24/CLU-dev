# `NIPSsubmission/v2-neurreps` — BUILD NOTE (V2's clean iteration base)

**This is deliverable #1.** Written against the finished artifacts; every number below was measured on the files in this folder.

**Task:** `.claude/tasks/nips-v2-clean.md` (Shorts Advisor, charter Addendum 48; Head directive 2026-08-21).
**DIAL DECLARATION (echoed): none — framing/editorial pass; zero number changes.**
**Source (read-only, verified byte-identical after this pass):** `.claude/papers/plain/v2/submission.tex` + `figs/` + `neurips_2025_ml4ps.sty`.
**Built with:** `/Library/TeX/texbin/pdflatex -interaction=nonstopmode submission.tex` ×3 (pdfTeX 1.40.29, TeX Live 2026), run **only inside this folder**.
**Folder contents:** `submission.tex` · `submission.pdf` · `figs/` (5 PNGs) · `neurips_2025_ml4ps.sty` (required to build) · `submission.log` (kept as box/page evidence; `.aux`/`.out` removed) · this note.

---

## 1 — Page split ⛔ NOT OPTIMISED, and not to be read as a page result

| block | this build | plain source (for reference only) |
|---|---|---|
| **main text** | **7.80 pp** | 7.00 pp |
| references | 2.45 pp | — |
| appendices | 7.79 pp | — |
| **total** | **18 pp** | 17 pp |

Instrument: `pdftotext -bbox`, text block 72–720 pt, the same one used by the earlier build notes (fractional page = page index + (y−72)/648 at the *References* heading and at the Appendix-A heading).

⛔ **The page count was neither targeted nor reduced.** The task states the page budget is explicitly not a target in this pass; nothing was compressed, no typography was applied, nothing was cut for space. The **+0.80 pp on main text** is entirely the new framing content: two new Related-work paragraphs (the 2025–26 neighbour positioning and the N1 scoping paragraph), the enumerated contributions block, and the longer abstract. **Four new reference entries** account for most of the reference growth. If something reads long here, it is long on purpose and is left long.

**Boxes, reported and not fought:** **1 overfull `\hbox` (3.57 pt)** — the loan-curve ladder table, unchanged from the source, same box, same size — **0 overfull `\vbox`**, 36 underfull boxes (ragged `p`-column negatives tables and float pages). **0 LaTeX errors, 0 undefined references, 0 undefined citations.**

---

## 2 — The audience scoping (2025–2026), change by change

Source of every neighbour used: `.claude/outputs/audience-refresh-2025-2026.md` §1.5 / §1.7 / §3.1 (retrieval date 2026-08-21). The draft's framing had been written against a 2022 census; the changes below re-aim it at the room as it is now. **No number, finding, scope qualifier or piece of fine print was changed by any of them.**

| # | change | where | why (refresh item) |
|---|---|---|---|
| 1 | **Spontaneous symmetry breaking is stated directly, with no apology and no translation** — named in the abstract (*"minimisers on a $G$-orbit — spontaneous symmetry breaking"*), in §1 (*"Symmetry supplies such directions for free, and it supplies them by breaking … spontaneous symmetry breaking, stated in the ordinary sense"*), and in §3 (*"the spontaneously broken case"*). The Nambu–Goldstone mode is named in §1 as the register the paper prices. | abstract, §1, §3 | *symmetry breaking* is now titular in this room (2025 spotlight); the 2022 "no-equivalent / needs translation" reading is wrong for 2025 |
| 2 | **"canonicalization" replaces the 2022 term.** §3 now reads *"the coset coordinate … a canonicalization used as a storage device rather than as an input encoding"*, and the Related-work sentence on quotient coordinates reads *"A canonicalization, a coordinate on the quotient, has been used as an input encoding (Aslan, Platt & Sheard 2023), where ours is a storage device."* ⚠ *fundamental domain projection* appears **nowhere** except inside the verbatim title of the cited reference, where it must stay. | §2, §3 | *canonicalization* is the 2024/2025 standard; the older term is four years stale |
| 3 | ⚠ **"flow" is disambiguated on first use**, inside the new Related-work paragraph: *"One disambiguation, because this literature uses the word three ways: flow here means a one-parameter Lie subgroup acting through time, never Ricci flow and never a normalizing flow."* Consequently the two pre-existing uses of *"a first-order flow cannot exhibit"* (§2, §4.2) now read **"a first-order dynamical system cannot exhibit"** — same statement, no collision with the reserved sense. | §2, §4.2 | the 2025 poster list carries Ricci flow and flow equivariance side by side |
| 4 | ⭐ **N5/N5b is the positioning anchor.** New lead sentence of the neighbour paragraph: flow equivariance (Keller 2025) extended to world models over long horizons of partially observed motion (Lillemark et al. 2026); *"That line and this paper share an object — a continuous symmetry carried in a recurrent state over a long horizon — and split the question cleanly: **they generalize, we price**."* followed by the explicit no-generalization-claim clause. Keller 2025 was previously buried in a three-citation list; it is now positioned. | §2 | best anchor in the room; same object, orthogonal contribution |
| 5 | **N2 is named as the emergent-arm competitor**: *"Symmetry-regularized learning of continuous attractor dynamics (NeurReps 2025 workshop poster) obtains the flat direction by symmetry regularization of a learned model, where our designed arm obtains it by construction and our emergent arm carries no symmetry regularizer at all, which is exactly the scope of the negative in §4.3."* ⚠ See §6 flag F2 — the gloss is deliberately confined to a restatement of the poster's own title. | §2 | the soft route to the same flat direction |
| 6 | **N4 is named before a referee can ask for it**, twice: in §2 (*"the standard instrument for the kind of gap we report … is solution degeneracy across task-trained recurrent networks (Huang, Singh, Martinelli & Rajan 2025); we do not run it here"*) and in §5's directions list (*"applied to §4.3's designed-versus-emergent gap, which would separate what is recipe from what is architecture"*). ⛔ Named as a direction, never as cargo (Charter C-4). | §2, §5 | the instrument this audience will demand for a designed-vs-emergent gap |
| 7 | **N6 is cited as the organizers' own live line** and the arrow is stated in both directions: *"geometric restructuring precedes abrupt learning …, and a temporal-consistency regularizer facilitates attractor formation (Haputhanthri et al. 2025, preprint). §4.4 runs that arrow backwards — our objective destroys a designed flat direction and an anchor holds it up."* | §2 | strongest "cite your reviewers' current work" candidate |
| 8 | **N7 is used as licence, not as a citation.** Nothing in this paper introduces Noether, symplectic integration or the conformal-symplectic property apologetically; no hedging clause was added, and the two 2024 posters are **not cited** because the refresh could recover neither author list (flag F5). Their existence licensed the register; it did not license a citation. | whole paper | precedent that the formal register is accepted here |
| 9 | **"memory lifetime" is bridged to our metric once, in §1:** *"What this literature calls a memory lifetime is our half-life $n_{1/2}$, reported in map applications, not $\tau$ and not diffusion coefficients $D$"* — the metric-naming rule that follows it is unchanged. | §1 | *memory lifetime* is now attested verbatim in the adjacent literature |
| 10 | **This reads as a dynamics paper.** §2 opens on the audience's two live questions and the object is introduced dynamically; §1 leads with the state, the map and the neutral coordinate. No claim was added or widened to achieve this. | §1, §2 | dynamics/attractors/RNNs is a standing CFP topic and ~1 in 7 of the 2025 accepted set |

**Prose (PJ style, applied to the framing only).** ABT openings retained/sharpened for the abstract, §1 and every results subsection; macro-to-micro order in §1 (state → map → neutral coordinate → price); contributions **enumerated** on page 1 (three items, replacing a narrative paragraph); short declarative sentences with one idea each; no weasel words introduced; `\emph{}` used for signposting, `\texttt{}` retained for flags and files. ⛔ **Simplify the prose, never the claim:** every approved wording, rider, scope qualifier, fine-print block, negatives table and appendix paragraph is **byte-unchanged from the source** except where a row of §2 above names it.

---

## 3 — ⛔ The N1 scoping statement (binding, conservative)

**No sentence of this paper claims the zero-mode ⇒ pseudo-gap ⇒ finite-lifetime chain.** Concretely:

1. **The chain is attributed, in the abstract, before any of our results.** *"It also knows that exact equivariance protects a neutral direction: a recent preprint (arXiv:2605.03338) proves that an exactly equivariant field carries zero Lyapunov exponents tangent to the group orbit, and reports that in its controlled breaking experiments the formerly protected direction predicts a finite memory lifetime."*
2. **A dedicated Related-work paragraph states what is theirs and what is ours**: *"That existence result, and that qualitative lifetime prediction, are established there and are not claimed here. What this paper adds is the exchange rate on a trained potential, measured at latent dimension 4 on an $S^1$ testbed on ≤5 seeds: the closed-form price list of §4.1 — the $\mu^{-2}$ law with its measured slope, the curvature-independent floor, the exceptional-point crossover, and the survival of all three under the corrective anchor."*
3. **§4.2 repeats the fence at the point of contact**: *"…that prediction is its result, not ours."*
4. ⭐ **The term *pseudo-gap* now appears ZERO times in the paper** (source: once, in attributed quotation marks). The strictest available reading of the directive was taken. *(Reversible in one line if the Hub prefers the attributed-quotation form for vocabulary alignment with the room — see the report's open questions.)*
5. **Citation-only form preserved:** the author token appears **once in the whole file, in the bibliography entry**, and nowhere in prose, captions, labels or filenames (sweep §5A).
6. ⚠ **Units, load-bearing, stated at both sites.** §1: *"One conversion precedes any cross-field comparison, and it is not a rescaling: flow-Jacobian eigenvalues — Lyapunov exponents — are inverse time, whereas $\mu^2$ are eigenvalues of a mass-whitened Hessian of a potential, inverse time squared. We therefore never convert a rate into a curvature on paper; §4.2 does the conversion the only way we trust, by running a published rate-based estimator unchanged on our trained models."* §2: *"its gap is a Lyapunov exponent, an inverse time, while our $\mu^2$ is a curvature of a trained potential, an inverse time squared."* §4.2: *"…so the two are compared here by running its instrument rather than by converting its units."*

**⚠ One wording change beyond framing, disclosed and reversible — "law" → "estimator".** The in-flight full-text read landed before this pass began (`.claude/outputs/n1-fulltext-and-track-check.md`, 2026-08-21 19:01) and its reconciliation list states, in its first ten lines, that **N1 contains no closed-form gap→lifetime relation** — the relation is a prediction validated by correlation, recoverable only from its released code. The source draft called it a *"lifetime law"* at **six sites** — abstract, contributions paragraph, the §4.2 subsection title, the §4.2 body (×2, counting *"That law is the overdamped face…"*) and the Figure 2 caption (`grep -c "lifetime law"` on the source = **5**, plus the one *"That law is"* anaphor). All six now read **"lifetime estimator"** / *"single-exponential estimator of its own"* / *"That estimator is the overdamped face…"*. ⛔ **`grep -c "lifetime law"` = 0.** This is a citation-accuracy correction owed by that report, not a change to any finding of ours, and it **narrows nothing and widens nothing**: the numbers, the containment framing and the "not ours" fence are unchanged. The Hub may revert it in one pass if it prefers to wait for its own reading of that report.

**⚠ Consequence of the same report, NOT acted on here:** the report's finding that N1's gap→lifetime relation is a prediction rather than a law does **not** trigger the task's "one further narrowing pass" — that pass was conditioned on the opposite outcome (finding a *law*). No further narrowing is owed on this evidence.

---

## 4 — Numeric two-way check (instrument identical to the source's)

Multiset of all numeric tokens, with `\includegraphics` options and `\setlength` arguments excluded as typography.

- distinct tokens: **source 392 · this build 395**; total tokens: source 1295 · this build 1316.
- **IN SOURCE, NOT IN THIS BUILD: `{}` — empty. Zero content numbers lost, dropped or altered.**
- IN THIS BUILD, NOT IN SOURCE (18 occurrences, every one accounted for):
  - `2025`×9, `2026`×2 — years of the four new reference entries and their in-text citations;
  - `2502.07256`, `2410.03972`, `2601.01075` — the three new arXiv identifiers;
  - `2605.03338`×1 — **one additional occurrence** of an identifier already in the source: the N1 attribution moved into the abstract;
  - `1`×2, `2`×2 — repeats of existing formula symbols in reworded framing sentences (`$C^1$`, `$n_{1/2}$`, `$\mu^2$`, `$\mu^{-2}$`, `$S^1$`);
  - `4`×1, `5`×1 — the scale qualifier added in-sentence to the new "what this paper adds" claim (`latent dimension $4$`, `$\le5$ seeds`), both already the paper's own canonical scale numbers (Charter C-5).
- **No new quantitative result, and no rounding, smoothing or adjustment of any existing one.**

---

## 5 — Sweeps (per file, positive-controlled, printed)

**A. Author token.** `grep -c "\bMo\b"` → **1**, and it is the bibliography entry (`Mo, H. H. (2026). … arXiv:2605.03338`). Prose/captions/labels/filenames → **0** (same grep with the bibliography line excluded). **Positive control:** the same regex on the pre-anonymization descoped source fires **9**. False positives confirmed surviving: **Morse ×1** (Akhtiamov & Thomson) and **Moser ×1** (Gardner et al.). Pronoun sweep (`\bhis\b|\bHis\b|\bhe\b|\bhim\b`) → **0**.

**B. Internal apparatus / paths / program vocabulary.** Pattern `SF-[0-9]|CM-[0-9]|Cor-[0-9]|.claude|/Users|scratch/|handover|Advisor|Hub|spoke|never-quote|PREREG|N[0-9]{3}|CSF3|CAMELS|CMAPSS|K5|organizer swap|13.9|bprime|CLU-former|claims matrix|wave-[0-9]|charter` → **0 hits**. **Positive control** on `tasks/nips-v2-clean.md`: **6 hits**.

**C. Semantic hermeticity.** Pattern `companion (paper|short)|our other short|V1|V3|V5|sister paper|our unpublished|under review elsewhere` → **0 hits**. The only unpublished-work references are the source's own, unchanged: the theory note as *(Anonymous, 2026)* (×2) and the naming-continuity sentence *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* (present, verbatim, ×1).

**D. Anonymization posture — unchanged.** `\author{}` empty; the style file's `\@notice` suppression retained (no venue string for *this* submission); PDF metadata Title/Subject/Keywords/Author all empty (`pdfinfo`); decompressed-stream sweep over **68 streams / 31.5 MB inflated** returns **0** for `Forgis`, `x10719pj`, `Users/user`, `Desktop`, `CERN`, `Manchester`, `.claude`, `NIPSsubmission`, `/tmp/`, with **positive control `Goldstone` = 8**.
⚠ **Disclosed exception, deliberate:** the string *"NeurReps 2025 workshop poster"* now appears twice (in-text and reference entry) as the **venue of a cited poster**, not as our venue. This is ordinary citation metadata; the build remains venue-neutral for its own notice.

**E. Typography carry-over (verified, not redone).** `footnotesize|scriptsize|\small|\tiny|raggedbottom|\@startsection` → **0 hits**. All **5** `\includegraphics` at `width=\linewidth`; non-`\linewidth` widths → **0**. `\textbf` before `\appendix` (i.e. in the main text) → **0**; appendix `\textbf` is the source's table/row emphasis, inherited unchanged.

---

## 6 — Figure inventory (5, all byte-identical to the plain source)

| # | file | placement | caption carries |
|---|---|---|---|
| 1 | `figs/fig1_gmor.png` | main text, §4.1 | the price list on trained checkpoints; 5 seeds, dim 4, γ=0.05, laptop CPU; labelled **verification** |
| 2 | `figs/fig_lifetime_headtohead.png` | main text, §4.2 | the head-to-head; 14 regimes, 5 seeds; labelled **evidence**; the author token is absent from the canvas (re-rendered in the source pass) |
| 3 | `figs/fig2_anchor_cure_laws.png` | Appendix A | anchored 3000 epochs, λ=100, 3 seeds; labelled verification-under-cure |
| 4 | `figs/fig3_retention_overlay.png` | Appendix C | ⚠ **single-seed status stated in the caption**: baselines median over 5 seeds, emergent median over 3, **designed curve a single representative checkpoint**; labelled **evidence** |
| 5 | `figs/fig3_gmor_condensate.png` | Appendix F | GMOR proper, 8 trained checkpoints, analytic spurion; labelled verification |

`shasum` of all five equals the plain source's — no figure was re-rendered, re-cropped or re-scaled in this pass. ⛔ `sf1_mo_estimator_overlay.png` remains **excluded** for the reason recorded in the source note (author token printed on its canvas, no surviving generator); its result stays fully in the text of Appendix F, and the owed re-render is still actionable from the banked extract.

---

## 7 — Carried-over verifications (each checked on this file, nothing redone)

| carry-over item | check | result |
|---|---|---|
| no page-fitting typography | sweep §5E | **0 devices**; all figures at natural `\linewidth` |
| no bold outside structural headers | `\textbf` before `\appendix` | **0** in main text |
| author token absent from prose/captions/labels/filenames | sweep §5A + filename listing | **0**; the one hit is the bibliography entry; `Morse`/`Moser` survive |
| banked figures restored with provenance-bearing captions | §6 | 5 figures, provenance in every caption |
| single-seed figures labelled as such | Figure 4 caption | present, verbatim from source |
| anonymization posture | sweep §5D | unchanged |
| source folders read-only | §8 | 168-file manifest byte-identical |

**Nothing had slipped**; no repair was needed on any carry-over item.

---

## 8 — Source folders byte-untouched (the check, stated)

Full-file `shasum` manifests taken **before** the pass and re-taken **after**:
- `.claude/papers/**` — **168 files, `diff` empty → byte-identical** (this covers `plain/`, `v2-short/`, `v5-short/`, `v2-neurreps-descoped/`, `neurreps-variants/`, `palm-variant/`, `bprime/`, `iclr-long/`, `f5-note/`, `v1/v3/v5/v6` shorts).
- `.claude/outputs/v2-full-runs/**` — **16 files, `diff` empty → byte-identical** (the banked figure generators and PNGs).
- `pdflatex` was executed **only** with `cwd = .claude/NIPSsubmission/v2-neurreps`; no other paper folder was written to, read-modified, or compiled in.
