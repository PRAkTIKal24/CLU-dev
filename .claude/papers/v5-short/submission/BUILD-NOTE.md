# V5 submission build — build note

Derived artifact, produced by `tasks/v5-final-pass.md` stage 2. The internal canonical
(`../draft.md`, `../draft.tex`, **v0.4**) is **byte-untouched by this stage**: it was written at stage 1
(the validation fold) and frozen before this directory existed. Every block removed below is mapped to
its canonical home, so nothing is lost and everything is findable.

## 1. Files

| file | what it is |
|---|---|
| `submission.tex` | the submission source, assembled from `main_body.tex` + `appendix.tex` (both kept here for auditability) |
| `submission.pdf` | **10 pp total**: **main text 4 pp (hard limit met)** · references 0.8 pp · appendices 5.2 pp |
| `figs/` | three figures, renamed to neutral filenames (§5) |
| `neurips_2025_ml4ps.sty` | the template actually used (§2) |

Build: `pdflatex` ×3 (TeX Live 2026, `/Library/TeX/texbin`). **0 errors, 0 undefined references.**
2 overfull hboxes and 7 underfull hboxes, all inside the `\tiny` narrow `p{}` cells of the negatives
table and the wide instrument tables (loose word spacing; cosmetic). Reported rather than silenced.

## 2. Template — what was used, and why

**Neither the PALM template nor the NeurIPS 2026 style file is obtainable on this machine.** The closest
genuine NeurIPS-family style present locally is **`neurips_2025_ml4ps.sty`**, which carries the standard
NeurIPS page geometry — **textwidth 5.5 in × textheight 9 in, 10 pt, submission mode with line numbers
and the anonymous author block**. That is what this build uses, with the style file's workshop notice box
suppressed (`\renewcommand{\@notice}{}`) so no venue string appears anywhere in the artifact.

⚠ **The page counts below must be re-measured in the real venue template before submission.** The
geometry is the standard NeurIPS block, so the count should be close, but it is not certified.

⚠ **A measured finding the Head should see.** The canonical's "4 pp" at v0.2/v0.3 was measured in a
*generic* `article` class with 1-in margins (text block 6.5 × 9 in). The NeurIPS block is ~15 % smaller
in area, so the same text runs ~4.7 pp there. Meeting the hard 4 pp in NeurIPS geometry therefore
required cutting **269 words (−10.6 %)** out of the v0.4 main text *on top of* absorbing the v0.4
validation fold — §6 gives the per-section arithmetic.

## 3. Page split (measured from the PDF)

| block | pages | note |
|---|---|---|
| main text | **4.00** | pp. 1–4; ends with the Limitations block at the foot of p. 4. **Hard PALM short-track limit met.** |
| references | **0.8** | 25 entries, `\tiny`, begins on p. 5 |
| appendices | **5.2** | pp. 5–10 |
| **TOTAL** | **10** | ⚠ **1 pp above the 8–9 pp target band** — §7 gives the costed menu |

Main text = **2,279 words + 1 figure**. Appendix body = **2,967 words + 2 figures + 7 tables**.

## 4. Structure (instruction 1) — related work is now §2

New order: **§1 Introduction → §2 Related work → §3 Results (§3.1 V-curve, §3.2 vault, §3.3 deletion)
→ §4 Limitations and future work → References → Appendices A–E.** All cross-references converted to
`\ref`; `pdflatex` reports **0 undefined references** and the label/ref diff is empty.

The prior-art material compressed **into** §2: the history-independence lineage (Snyder → Naor & Teague →
Blelloch & Golovin → Blelloch, Golovin & Vassilevska) sits there in one sentence, with the *full* lineage
— including Micciancio's obliviousness/canonicity distinction and Andersson & Ottmann — relocated to
Appendix D's opening paragraph, which is where the claim it qualifies now lives.

## 5. Appendix triage — every removal mapped to its canonical home

**Surviving appendices** (plots and results-tables only, each cut to figure/table + result sentence + fine print):

| submission | was (canonical) | substance kept |
|---|---|---|
| **A** The $(\mu,\gamma,T)$ budget | B | the knob table (Table 1) + the diffusion-law, sign-flip, latch and designed-V-curve result sentences; the `fdt`+Newtonian scope preamble **verbatim**; the "never an $n_{1/2}$ without $\Delta$ and $\ell_\theta/\Delta$" rule |
| **B** The emergent arm + the second instrument | C | the four-instrument table (Table 2) and the reduced instrument-gap table (Table 3) + **Figure 2** (the rollout overlay) + finite-amplitude, integrator-identity, designed-control, collapse and $T^\star$ result sentences + the 2.7×-in-$\mu^2$ scope rider |
| **C** The friction-hole vault, both arms | D | the refrigerator table (Table 4), the confinement table (Table 5) + **Figure 3** (emergent vault) + mechanism, designed-arm, estimator-cross-check, contrast-is-designed-only, first-passage and scope paragraphs, with the $\theta=\pi$ confound and the void-$\sigma_\theta$ riders **verbatim** |
| **D** Exact deletion: prior art + tables | E | the TTL laundering-control table (Table 6) + the full prior-art lineage + exactness/packing/overflow numbers + the leakage, retrieval-geometry and trilemma result sentences + the no-cost-claim denial |
| **E** Prominent negatives | J | the 20-row compact table (Q4 exception), split into two `tabular` blocks so it breaks across pages |

**Dropped from the submission** (canonical keeps every one in full):

| dropped | canonical home | why |
|---|---|---|
| Flag-provenance appendix A.0–A.10 (commits, seeds, every non-default flag) | `../draft.md` **Appendix A** | prose-only under the tightened plots/tables rule; the source reports it reproduces remain its record |
| Appendix F — the three-state lifecycle in full (leg table, L4-unexercised discussion, owned reconciliations) | `../draft.md` **Appendix F** | prose + a leg table that backs no measured claim; the contribution sentence and the "unexercised" label survive in §1, and the leg's negative survives in Appendix E |
| Appendix G — Coleman / Mermin–Wagner and the `fdt`+Newtonian scope | `../draft.md` **Appendix G** | prose-only; **the mandatory flag relocated verbatim** into §3.2 and into Appendix A's preamble (§6) |
| Appendix H — erosion as symmetry restoration (claims (a)–(d), the k-regime clause, the coverage statement) | `../draft.md` **Appendix H** | prose-only; nothing in the submission's main text depends on it. ⚠ **Side effect, flagged as a positive:** dropping it also removes this short's only content overlap with the other draft the referee flagged for duplicate-appendix exposure |
| Appendix I — the $T_\phi$ horizon | `../draft.md` **Appendix I** | future work; **relocated** to §4's "Named next" as one clause, with the do-not-co-locate design rule kept |
| Appendix K — banked main text (K.1 nomenclature, K.2 related work in full, K.3 learned-store refutations) | `../draft.md` **Appendix K** | K.1 is in §1 compressed; K.2 became §2; K.3's two refutations are rows of Appendix E |
| Figures: diffusion law, sign flip, $T^\star$, un-collapsed emergent V-curve, designed vault | `../figs/` (all five kept) | each backs numbers that survive as prose or table rows; dropped in preference order until the page budget closed |
| 33 reference entries orphaned by the appendix cut | `../draft.md` References (58 entries) | each was cited only from a dropped appendix; verified by a per-name sweep of the remaining body. 58 → 25 |

**Reference entries dropped as orphaned** include the whole erosion cluster (Fischer & Igel, Nijkamp,
Toledo-Marín, Decelle, Agoritsas, Du & Mordatch, Hinton, Tieleman), the extended agent-memory cluster
(MemGPT, MemoryBank, Expire-Span, Infini-attention, Wang & Zhang, Wang et al.), the extended unlearning
cluster (SISA follow-ons, MUSE, CURE4Rec, Ghazi, Min, Özdenizci, Laguna), the equivariance cluster
(Golubitsky, Krupa, Di Bernardo, Iqbal, Hidaka & Minami 2020), Coleman, Mermin & Wagner, GMOR,
Buchbinder & Petrank's journal version, Hartline, Sundar & Tarjan, Hairer/Lubich/Wanner, McLachlan &
Perlmutter, coRNN, Chen, Chhikara-adjacent preprints. **Every number cited from the submission's main
text still resolves inside the submission** — verified in §8.

## 6. Rider-relocation list (riders never drop with their appendix)

| rider | was | now | edit |
|---|---|---|---|
| `fdt` + Newtonian mandatory scope | canonical App G + §2.2 fine print | **§3.2 fine print** and **Appendix A preamble** | verbatim, plus the "under the reference default these laws fail / in the relativistic mode no noise scale targets Gibbs" clause |
| ⛔ no emergent $\sigma_\theta$ ratio (G3 non-stationarity) | canonical D.7(d) | **§3.2 fine print** and **Table 5 caption** | verbatim in substance; the control's $0.4586\pm0.1181$ printed at both sites |
| the $\theta=\pi$-is-not-a-vacuum confound | canonical D.7(e) | **Appendix C** first-passage paragraph and **Appendix E** row | verbatim |
| the contrast number is designed-only | canonical §2.2 + D.7(c) | **§3.2 fine print**, **Appendix C**, **Appendix E** row | verbatim at all three sites |
| the designed-symmetry precondition | canonical §2.2 | **§3.2**, in the paper's voice | verbatim; do-not-cut list honoured |
| $\tau_{\max}=\Gamma/2\alpha$ ceiling + tilt-refuted-in-sign | canonical §2.2 tail + K.3 | **Appendix E** (two rows) | the main-text tilt sentence was removed, so the rider travels with the claim into the negatives table |
| the $T_\phi$/$\gamma_\phi$ co-location design rule | canonical App I | **§4 Named next** | compressed to its one clause |
| the $|c|$-distribution clause on every retention number | canonical E.5(v) | **Appendix D** | verbatim |
| quote-the-curve on $0.99985$ | canonical §2.3 | **§3.3** | verbatim ("at full load") |
| N108's sentence · CM-25(f) verbatim quote · the BG attribution · the score sentence · the substrate-scope sentence · the trilemma corner · the $R_{50}$ differentiator · the TTL laundering control · the CLU continuity sentence · scale-as-scope-choice | — | in place in §1/§3.3/§4 | **verbatim, unmoved** — the entire `v5-referee-v02` §D do-not-cut list is present |

## 7. The residual page gap — costed menu for the Head

Main text is **at** the hard limit. The **total is 10 pp against the 8–9 pp target**; closing it needs one
of these, and each contradicts an instruction of this pass or a Charter rule, which is why the writer
stopped here:

| item | saving | what it costs |
|---|---|---|
| Drop Appendix E (negatives) | ≈0.9 pp | contradicts the explicit instruction that the negatives appendix survives as its compact table (Q4), and C-9 |
| Drop Appendix D (deletion tables + prior art) | ≈1.3 pp | the TTL laundering control (MF-9) and the relocated lineage lose their home; contribution 2 becomes unsupported |
| Drop Figure 2 (rollout overlay) | ≈0.35 pp | contradicts instruction 3 — it *is* the v0.4 validation story |
| Drop Figure 3 (emergent vault) | ≈0.35 pp | measured: does **not** save a page on its own (tried; still 10 pp) |
| Drop Appendix A (budget) | ≈0.7 pp | removes the verification-grade evidence for §3.1's designed arm |
| Cut the reference list further | ≈0.3 pp | the remaining 25 are all cited from the surviving body |

## 8. Verification re-run on this artifact (all printed in the spoke report)

- **De-bold (instruction 5): `\textbf` in the submission = 0**, main text *and* appendix. Nine run-in
  emphasis headers use `\emph`; one `\paragraph{Contributions.}` remains, which is structural.
- **Numeric two-way check.** (i) numeric tokens in `submission.tex` **not** present in the canonical:
  **0**. (ii) numeric tokens of the canonical **main text** absent from the submission **main text**: 23,
  of which **21 travelled into the submission appendix** and 2 are non-content — `2.3` (a canonical
  section number, now rendered by `\ref`) and `9.40` (present at the source report's own precision,
  `9.400`, in Appendix B). **No content number left the submission.**
  ⚠ Two defects this check caught and this build fixed: an arXiv id attached to Minami & Hidaka (2018)
  that the canonical record does not carry (restored to the canonical's DOI), and a wrong arXiv id on an
  entry that had become orphaned anyway (dropped). A third: the reduced instrument-gap table (Table 3)
  originally carried the source report's full 9-row grid, which the canonical summarises rather than
  tabulates — the table was cut back to the canonical's own numbers so the submission stays a strict
  derivative. **Recommend a canonical top-up of that table at the next pass.**
- **Final sweep** (per-file, positive-controlled, `scratch/v5-final-pass/subsweep.py`): **zero-list hits =
  2**, both the same false positive — `n_{\rm jac}/n_{\rm R1}` and `\Gamma_{\rm jac}/\Gamma_{\rm R3}` in
  Table 3's header matching an `R1`/`R3` internal-label pattern; these are **this paper's own instrument
  names**, defined three paragraphs above in Appendix B. Context-checked hits, all compliant: `certified`
  ×3 (two literature descriptions of Guo + the explicit denial) · `unlearning` ×3 (the denial + two
  reference entries) · "deletion is exact" ×2 (both qualified "store-level") · `CHLU` ×2 (the sanctioned
  continuity sentence + its reference entry) · `0.99985` ×1 (carries "at full load") · `297.8` ×1
  (appendix, with "never the vault number") · `23.39` ×3 (all labelled designed-only / falsifier-fired).
  Positive controls fired: 107.77 ×8 · 106.1 ×3 · 0.9001 ×2 · Blelloch ×10 · the N108 sentence ×2 ·
  "confines" ×2 · 8.11 ×6 · Anonymous ×2 · "introduced as CHLU" ×1 · verification ×8 · evidence ×11 ·
  9.5e15 ×1 · 0.4586 ×2 · ZERO ×2.
- **Semantic hermeticity (C-8):** `companion` / `sibling` / `our other short` / `the program` /
  `forthcoming` / `in preparation` = **0**.
- **Anonymization:** `\author{}` blank · no `[WORKING TITLE` · no `[AUTHORS PLACEHOLDER]` · no
  acknowledgment, funding, URL or repository string · PDF **Title, Author, Subject, Keywords, Creator and
  Producer all empty** (`\hypersetup` scrub) · **0** occurrences of any absolute path, username or project
  string anywhere in the compressed PDF (`strings` sweep). Third-person self-citation intact: *"the CLU
  (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* — the only two occurrences of
  those names are that sentence and its reference entry. ⛔ **No supplementary PDF is attached:** V5 has
  been self-contained since v0.3 and the theory note is cited nowhere load-bearing.
- **Figures renamed to neutral filenames:** `fig1_collapse.png` → `fig1_damping_optimum.png`,
  `figC_rollout.png` → `fig2_two_instruments.png`, `figD_emergent_vault.png` → `figC2_vault_emergent.png`.
  ⚠ **Retained deliberately:** the panel legends of Figures 2 and 3 carry seed short-tags (`s42`/`s43`/`s44`)
  and Figure 3's panel titles carry the pre-registration item labels (Q1, Q2, Q3, Q5). These are seed
  identifiers and pre-registration item names, not repository, venue or institutional identifiers; the
  captions define them. Re-rendering was out of scope for this pass (banked artifacts only).

## 9. Brevity, ABT and style (instruction 4)

ABT openings, spot-checkable: **abstract** — *and* forgetting is a budget in three numbers, *but* it is
only useful if you can say how fast and what is gone, *therefore* a damping optimum, a vault and an exact
deletion. **§1** — *and* deployed agent memories forget, *but* forgetting is a side effect nobody can
read off a trained model, *therefore* we study a memory where all three questions are answerable.
**§3.1** — *and* retention is non-monotone in friction, *but* every point is a linear-response Jacobian,
*therefore* we measured the curve a second way. **§3.2** — *and* a designed coset latches forever,
*but* at $T>0$ it diffuses, *therefore* friction preserves and a hole is a vault. **§3.3** — *and* a flat
table deletes exactly by construction, *but* our items are superposed into one energy function,
*therefore* canonical placement makes removal a byte-identity.

Macro-to-micro is enforced section by section (law → mechanism → number → fine print). Weasel words: the
only magnitude adjectives left are data-supported ("marginal" nowhere; "decisive" nowhere in main text).
Signposting is explicit at every section head. "We" is used for actions taken; passive for established
facts and standard algorithms.

Per-section word deltas (canonical v0.4 main → submission main; tokenizer strips math and markup):

| block | canonical v0.4 | submission | delta |
|---|---|---|---|
| Abstract | 201 | 188 | −13 |
| §1 Introduction | 473 | 435 | −38 |
| §2 Related work | 215 | 280 | **+65** (absorbs the prior-art compression, instruction 1) |
| §3 Results setup | 42 | 2 | −40 (redundant with §3.1's own scale clause and Limitations (i)) |
| §3.1 V-curve | 533 | 375 | **−158** (the two fine-print blocks folded to `\footnotesize`; the four-instrument detail to Appendix B) |
| §3.2 vault | 441 | 374 | −67 |
| §3.3 deletion | 448 | 442 | −6 (almost entirely protected wording) |
| §4 Limitations | 183 | 171 | −12 |
| **MAIN TOTAL** | **2,536** | **2,267** | **−269 (−10.6 %)** |

Cosmetic, content-neutral settings: `\textfloatsep`/`\intextsep` tightened to 10 pt; float fractions
relaxed (`\topfraction` 0.92, `\textfraction` 0.06); the appendix block set `\footnotesize`, references
`\tiny`, tables `\scriptsize`, the two widest instrument tables and the negatives table `\tiny`; the
negatives table split into two `tabular` blocks so it can break across pages.
