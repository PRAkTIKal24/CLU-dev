# BUILD-NOTE-R4 — `pj_sub.tex` (V2 / NeurReps) — the HEAD-APPROVED MINIMAL PASS

**Pass:** `pj-minimal-v2` (paper-writer). **Commission:** Shorts-charter **Addendum 61** (2026-08-22) — the Head-approved minimal list, three amendments.
**Predecessor:** R3 (`BUILD-NOTE-R3.md`) was **rejected as over-scoped and reverted**; the file this pass edits is the pre-R3 Head rewrite.

| | |
|---|---|
| input `pj_sub.tex` md5 | `d15de78712d90eb94d2495d4bd9ad948` (matches the md5 named in the task) |
| **output `pj_sub.tex` md5** | **`a5758ad3eafcaf8971c73e7685d21450`** |
| lines | 395 → 431 |
| build | `/Library/TeX/texbin/pdflatex`, in-folder, **run ×2**, both exit 0 — **0 errors, 0 undefined references, 0 multiply-defined labels** |
| pages | 14 → **15** |
| figures | **5/5** PNGs present and used, 0 added, 0 unused (`fig1_gmor`, `fig_lifetime_headtohead`, `fig2_anchor_cure_laws`, `fig3_retention_overlay`, `fig3_gmor_condensate`) |
| other folder files | **byte-identical** (`submission.tex` `ffda703b…`, `neurips_2025_ml4ps.sty` `393afa47…`, all 5 PNGs) — only `pj_sub.{tex,pdf,aux,log,out}` changed |
| sources used | `submission.tex` (base, primary ancestor) + `.claude/scratch/pj-restore-R3-preserved/pj_sub_v2_R3.tex` (the R3 bank, pre-fitted blocks) — per-edit attribution in §2/§3 |

**Method.** One scripted pass (`.claude/scratch/pj-minimal-v2/edit.py`), every replacement **assertion-guarded to match exactly once**; input md5 asserted before the first edit; the section-symbol conversion is a counted regex applied last. Pre-pass backup: `.claude/scratch/pj-minimal-v2/pj_sub.tex.BEFORE`.

---

## 1. Head style rule — the `§` symbol is retired (conversion table)

⛔ `\S\ref{...}` is never used. **11 pre-existing occurrences** were found and converted (the task said 9; the measured count is **11 occurrences on 9 lines** — lines 46 and 58 carry two each, which is exactly the 9-vs-11 discrepancy — reported, not silently reconciled). Two further occurrences arrive inside restored blocks (item 11's caption rider; Part 2(b)) and were converted by the same counted regex, for **13 conversions in total**. Post-pass the file contains **0** `\S\ref` and **13** `Sec.~\ref` (on 11 lines).

| # | line (pre-pass) | reference | before → after | `\label` needed? |
|---|---|---|---|---|
| 1 | 42 | `sec:headtohead` | `\S\ref{}` → `Sec.~\ref{}` | already present |
| 2 | 46 | `sec:pricelist` | ditto | already present |
| 3 | 46 | `sec:cure` | ditto | already present |
| 4 | 47 | `sec:headtohead` | ditto | already present |
| 5 | 48 | `sec:boundary` | ditto | already present |
| 6 | 58 | `sec:boundary` | ditto | already present |
| 7 | 58 | `sec:cure` | ditto | already present |
| 8 | 60 | `sec:headtohead` | ditto | already present |
| 9 | 76 | `sec:cure` | ditto | already present |
| 10 | 78 | `sec:cure` | ditto | already present |
| 11 | 80 | `sec:results` | ditto | already present |
| 12–13 | 304, 408 (post-pass) | `sec:boundary`, `sec:headtohead` | arrive with the restored blocks (item 11 rider, Part 2(b)); converted by the same regex | already present |

**Zero `\label` commands were added** — all five referenced sections (`sec:results`, `sec:pricelist`, `sec:headtohead`, `sec:boundary`, `sec:cure`) already carried labels. Appendix references were already in `Appendix~\ref{...}` form (6 occurrences, untouched). The remaining 8 `\S` matches in the file are `\Sigma`, not section symbols.

---

## 2. Part 1 — the 12 approved main-text items (+ item 13)

| # | site (post-pass line) | before → after | source |
|---|---|---|---|
| 1 | 28 (abstract) | "exact equivariance **is required to protect** a neutral direction" → "exact equivariance **protects** a neutral direction" | base L28 |
| 2 | 60 | "guarantees $\dim(G/\mathcal H)$" → "guarantees **at least** $\dim(G/\mathcal H)$" | base L77 |
| 3 | 44 | "**This chapter** provides" → "**This paper** provides" | base register (L41) |
| 4 | 28 (abstract) | "manifold, infinitesimal perturbations" → "manifold, **most** infinitesimal perturbations" | base L28 |
| 5a | 47 | "a recently **published** single-exponential lifetime estimator" → "recently **posted**" | base L44 |
| 5b | 104 | "we apply a recently **published** lifetime estimator" → "recently **posted**" | base L44 (phrase) |
| 6 | 108 | `…=0.9987$)` → `…=0.9987$, **overdamped-only**)` | R3 L108 — ⚠ the R3 form's second clause ("$0.973$ pooled…") is **excluded**: not enumerated, and its ancestor is an output file, not the base |
| 7 | 268 | "$150$ epochs, $3$ seeds, wake-only objective." → "…wake-only objective **except the $\gamma_\phi$ rung**." | base L225 / R3 L269 (base spells the subscript `\phi`; the task wrote `\gamma_\varphi` — the base/table spelling was used for consistency with the table's `$+\gamma_\phi$` row) |
| 8 | 414 (G.1) | the fused sentence "Above the established crossover $h-h^*=2.6\times10^{-6}$, trajectory and Jacobian frequencies align tightly to $0.06$--$0.3\%$." → base's **two separate facts**: the $\sqrt{h-h^*}$ onset **down to** $h-h^*=2.6\times10^{-6}$ on the fine multiplicative grid $[-4.2\times10^{-3},2.6\times10^{-2}]$ **·** the $f=2,4$ quality-factor condition on the $0.06$–$0.3\%$ agreement **·** the dynamically-silent clause | base L382, as fitted in R3 L417. ⚠ base's fourth clause ("the slight slope excess over $1/2$ is far-field bending") is **not** among the three facts the task enumerates and was **not** restored |
| 9 | 139 (Discussion) | **new bullet** — "**Substrate Scope:** Stated once, in our own voice: these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, with its parameter and state-byte budget ledgered." | base L145(iii), as fitted in R3 L139 (bullet-label register only: base's "The substrate scope, once, in our own voice:" → "Stated once, in our own voice:") |
| 10 | 137 (Discussion) | Head's sentence **kept**, the measured score sentence added beside it: "…superiority. *No external benchmark is won on its own headline metric anywhere in this paper.* Evaluative comparisons…" | base L145(ii) **verbatim**. ⚠ R3's trailing clauses (honest-gap; task-RMSE axis) are **excluded** — not enumerated |
| 11 | 304 (Fig-3 caption, `fig:retention`) | "…median across $5$ and $3$ seeds respectively**, while the designed curve is a single representative checkpoint---the $5/5$-seed latch statement is Sec.~\ref{sec:boundary}'s, not this figure's.**" | base L267 / R3 L307 (verbatim; the trailing period becomes a comma so the rider attaches — punctuation only) |
| 12a | 130 | "the slope holds at $-0.956$**(the per-point fit over all overdamped rows)**," | R3 L130 (the cross-reference half of R3's clause is excluded — the task asks for a label, not a reconciliation) |
| 12b | 256 (`fig:sf3` caption) | "(fitted $-0.961$ against predicted $-1$**; the seed-mean OLS over the $7$ overdamped $\delta$**)" | R3 L257 (same exclusion) |
| 13a | 58 | "Similarly, recent efforts utilize symmetry regularization to induce flat directions (NeurReps 2025 workshop poster)." → "Similarly, **concurrent workshop work has explored soft symmetry regularization for continuous attractors**." | task item 13 / R3 L58 |
| 13b | References | the entry "Symmetry-regularized learning of continuous attractor dynamics (2025). NeurReps 2025 workshop poster." **deleted** | task item 13 |

**References count 51 → 50.** No other bibliography line was touched; no new record was created; 0 dangling `\ref`.

---

## 3. Part 2 — banked information restored to APPENDICES ONLY

Every block below sits in an appendix; **nothing from Part 2 touches main text.**

| item | appendix | what was restored | source |
|---|---|---|---|
| (c1) | **D** (`app:compute`) | scan-amortized timing protocol: "Wall time is the median of $7$ scan-amortized repetitions over $2\times10^5$ steps; the naïve single-call timing is dispatch-bound at a $\approx5\,\mu$s floor and is *not* reported." | base L273 / R3 L313 (verbatim) |
| (c2) | **D** | the width-match confound paragraph ("**Confound, flagged:** …not width-matched, CLU at hidden $64$ … the $263/69/56$ retention numbers come from these specific configurations. … single-core CPU at batch $1$; GPU or batched throughput could shift the wall ratios, though not the FLOPs.") | base L299 / R3 L339 (verbatim; base's trailing sentence about what §4.3 leads with is **not** in this file's register and was not carried) |
| (a) | **E** (`app:neg`) | **the 6 within-row numbers** — row 1 "; tied controls exactly $1.0000$" · row 2 "inversion at epoch $116$/$442$/$959$ by sleep frequency" · row 3 "$\sim\!12$ orders; written $\delta$ retained $\le2.1\times10^{-3}$; capacity $\approx1$–$1.6$ bits" · row 4 "and $+0.3291\to-1.1980$" and "tilt-vacuum residual $0.140$–$0.343$ against a random-direction baseline $1/\mathrm{dim}=0.167$" | base L312/315/318/324, as fitted in R3 L352/355/358/361 |
| (a) | **E** | **the 6 missing rows** — the task-RMSE-axis row appended to table 1; a **second table** with the $n_{1/2}$-exponent row (carrying its own reading rule "No $n_{1/2}$ may be quoted without its $\Delta$ and $\ell_\theta/\Delta$"), the friction-field × curvature-governor row, the friction-cannot-stabilize-a-saddle row, the mean-spectrum-chaos row, and the `sleep_temperature` silent-knob row | base L325–327 + L332–355, as fitted in R3 L365–394 (verbatim) |
| (e) | **E** | the **CM-17 sampler fence** at its FAQ-row site: the Gaussian-smoothed-marginal argument, "**The failure is in the *sampler*, not in the *thermodynamics***", "We therefore never assert that a relativistic unit ''has no equilibrium''", and the `newtonian_learned` no-touch scope | base L361 / R3 L398 (verbatim) |
| (b) | **F** (`app:pos`) | "**The head-to-head reproduced on the preprint's own instrument.** … With the preprint's own finite-horizon estimator $\hat\lambda(T{=}128)$ … $\mathrm{corr}=0.9995$ overdamped, meas/pred $0.86$–$1.03$, and $0.30$ at $\delta=4$ against the exact-gap $0.31$." | base L376 / R3 L411 (verbatim, `\S\ref`→`Sec.~\ref`). ⛔ **appendix only — the main-text §4.2 sentence stays OUT**, as instructed |
| (d) | **G** (`app:gmor`) | the **demotion label** inside the config block: "**GMOR proper---G.3 onward---is demoted from the main text and retained here in full; it is a supporting result, not one of this paper's claims.**" | base L380 / R3 L415 (R3's register adaptation: base says "the main text *of this abstract*") |
| (d) | **G** | the **precision fine print** paragraph, including "**Do not quote ''$2.2\times10^{-16}$ relative'' for this experiment.**" | base App F.5/F.6 / R3 L431 (verbatim) |
| (d) | **G** | **G.5** — the definition of the expansion variable $x$ (de-orphaning Figure 5(d)) **and** the $\delta=0.3$ no-NLO clause | base App F.5 / R3 L433 (verbatim). Per the R3 bank, the base's resonance-saturation *claim* is not imported — only the fence + the definition |
| (d) | **G** | **G.6 Honest scope** — tree-level/classical, probe-only, one architecture family, the emergent-arm $\delta_{\rm eff}$ open falsifiable | base App F.6 / R3 L435 (verbatim) |

---

## 4. ⭐ THE HEAD'S REPORT ITEM — appendices ADDED or CHANGED relative to the Head's rewrite

**No appendix was newly created. The file has the same seven appendices, A–G, in the same order, with the same titles and labels, before and after this pass.**

| appendix | label | status | what changed |
|---|---|---|---|
| **A** The laws survive the training-time correction | `app:anchor` | **CHANGED** | item 12b — the $-0.961$ fit-spec clause in the `fig:sf3` caption |
| **B** The explicit price of the physical prior | `app:loan` | **CHANGED** | item 7 — "except the $\gamma_\phi$ rung" in the config line |
| **C** The autonomous-retention head-to-head | `app:retention` | **CHANGED** | item 11 — the single-representative-checkpoint clause in the Fig-3 caption |
| **D** Per-step compute requirements | `app:compute` | **CHANGED** | Part 2(c) — timing protocol sentence + width-match confound paragraph |
| **E** Documented negative results | `app:neg` | **CHANGED** | Part 2(a) — 6 rows (one appended + a new 5-row table) and 6 within-row numbers; Part 2(e) — the CM-17 sampler-scope paragraph |
| **F** Contextualizing prior work boundaries | `app:pos` | **CHANGED** | Part 2(b) — the finite-horizon head-to-head paragraph |
| **G** GMOR verification on trained checkpoints | `app:gmor` | **CHANGED** | item 8 (un-fused G.1) + Part 2(d) — demotion label, precision fine print, G.5, G.6 |
| — | — | **ADDED: none** | no `\section` was created anywhere in the file |

⚠ **The task's expectation was "E, F, D, G changed".** Measured: **all seven changed.** A, B and C change **only** because three of the Head's own *Part-1* line items (12b, 7, 11) live at sites that are physically inside appendices. No Part-2 bank content entered A, B or C.

---

## 5. Classified edit list — the zero-unenumerated-diffs proof

`difflib` opcode diff of pre-pass vs post-pass: **30 changed line-blocks, every one attributable to an enumerated item.**

| # | line (pre) | classification |
|---|---|---|
| 1 | 28 | items **1 + 4** |
| 2 | 42 | **§-rule** (1 conversion) |
| 3 | 44 | item **3** |
| 4 | 46–48 | **§-rule** (4 conversions) + item **5a** |
| 5 | 58 | item **13a** + **§-rule** (2 conversions) |
| 6 | 60 | item **2** + **§-rule** (1) |
| 7 | 76 | **§-rule** (1) |
| 8 | 78 | **§-rule** (1) |
| 9 | 80 | **§-rule** (1) |
| 10 | 104 | item **5b** |
| 11 | 108 | item **6** |
| 12 | 130 | item **12a** |
| 13 | 137 | item **10** |
| 14 | 139 | item **9** (insert) |
| 15 | 235 | item **13b** (delete) |
| 16 | 256 | item **12b** |
| 17 | 268 | item **7** |
| 18 | 304 | item **11** |
| 19 | 310 | Part 2 **(c1)** |
| 20 | 336 | Part 2 **(c2)** (insert) |
| 21 | 347 | Part 2 **(a)** within-row #1 |
| 22 | 350 | Part 2 **(a)** within-row #2 |
| 23 | 353 | Part 2 **(a)** within-row #3 |
| 24 | 356 | Part 2 **(a)** within-row #4 |
| 25 | 360 | Part 2 **(a)** the 6 rows (insert) |
| 26 | 365 | Part 2 **(e)** (insert) |
| 27 | 377 | Part 2 **(b)** (insert) |
| 28 | 379 | Part 2 **(d)** demotion label |
| 29 | 381 | item **8** |
| 30 | 395 | Part 2 **(d)** fine print + G.5 + G.6 (insert) |

**Unenumerated diffs: 0.** No sentence of the Head's rewrite was reworded, reordered, retitled or deleted outside these 30 blocks. Full diff: `.claude/scratch/pj-minimal-v2/pass.diff`; machine-readable edit log: `.claude/scratch/pj-minimal-v2/editlog.json`.

---

## 6. Two-way numeric check

Method (unchanged from R3, so the two notes are comparable): numeric-token bag `\d+(\.\d+)?` over pre-pass vs post-pass; every token whose count **increased** is traced to a `submission.tex` line, then to the R3 bank.

- **Tokens whose count increased: 59 distinct values. 59/59 have a `submission.tex` (base) ancestor line. ORPHAN LIST EMPTY — zero tokens required an output-file ancestor**, and none was improvised. (R3 needed two output-file ancestors, `0.973` and `0.82`; **neither is used here**, because the clauses that carried them are not on the approved list.)
- Newly-appearing values and their base lines: `0.3291`,`1.1980`,`0.140`,`0.343`,`0.167` → base L324 · `116`,`442`,`959` → base L315 · `2.1`,`1.1`,`1.6` → base L318/L312 · `0.53`,`0.60`,`1.04`,`0.78`,`0.63`,`0.55`,`2.0` → base L321 · `0.861`,`2/6`,`2/3` → base L340 · `1.4` → base L343 · `0.9995`,`0.86`,`1.03`,`0.30`,`0.31`,`128` → base L376 · `2.7`,`2.28`,`4.22`,`2.2` → base L413 · `0.670`,`0.68`,`210`,`0.25` → base L415/417 · `4.2`,`2.6` → base L382 · `1.52`,`2.06` → base L306/333 · `263`,`69`,`56` → base L292/299 · `1.0000` → base L312.
- **Tokens whose count decreased: exactly one — `2025` (−3)**, all three occurrences in the deleted authorless-poster entry and its in-text citation (item 13). Nothing else was removed.
- **No digit, precision, unit, exponent or ± anywhere in the file was altered.** Every added token is on the edit map in §2/§3.
- Reverse direction inherited: `pj-fidelity-v2-r2` §A.1 established that the Head's rewrite carried no number without an ancestor; this pass adds only base-ancestored numbers, so the property holds file-wide.

---

## 7. Sweeps (per-file, positive-controlled — the standing gitignored-dir Grep hazard)

| sweep | `pj_sub.tex` | positive control | verdict |
|---|---|---|---|
| never-quote / internal-leak, multi-alternate (`SF-\d`, `CM-\d`, `.claude`, `/Users`, `handover`, `Advisor`, `Hub`, `spoke`, `PREREG`, `N\d{2,3}`, `wave-\d`, `charter`, `certified`, `unlearning`, `2\.6\b`, `pseudo-gap`, `never-quote`, `claims matrix`) | **1 hit — line 414**, the *same inherited false positive* both fidelity rounds and R3 recorded: `$h-h^*=2.6{\times}10^{-6}$` matching `2\.6\b`; ancestor `submission.tex:382` | `advisor-head-shorts-charter.md` = **350 hits** | ✅ **CLEAN** |
| author-token rule (`\bMo\b|\bMorse\b|\bMoser\b|\bhis\b|\bHis\b|\bhim\b`) | **2 hits** — line **216** the *permitted* bibliography entry; line **150** "…via the lens of **Morse** theory" (the survival trap, correctly survived). **0 in body text, captions, labels or filenames**; all 5 `\includegraphics` paths are `figs/fig*_*.png` | `submission.tex` = **3 hits** | ✅ **COMPLIANT** |
| `pseudo-gap` | **0** | `advisor-head-shorts-charter.md` = 5 | ✅ **0** |
| semantic hermeticity (`companion (paper|short|note)`, `our (other|sibling|companion)`, `the program`, `sibling (paper|short)`, `in a companion`, `our V\d`, `the three shorts`) | **0** | `philosophy-synthesis.md` = **112 hits** | ✅ **0** |
| section symbol (`\S\ref`) | **0** | pre-pass file = 11 | ✅ **0** |
| placeholder leftovers (`WORKING TITLE`, `AUTHORS PLACEHOLDER`, `TODO`, `XXX`) | **0** | — | ✅ **0** (the anonymized `\author{}` is intentionally blank) |

---

## 8. Build, boxes, pages

- **`pdflatex` ×2, both exit 0. 0 errors. 0 undefined references. 0 multiply-defined labels.**
- **Bad boxes: 1 Overfull `\hbox` (11.27979 pt, lines 288–298 — the compute-appendix table).** It is **pre-existing and identical**: rebuilding the pre-pass file in `.claude/scratch/pj-minimal-v2/beforebuild/` gives the same single overfull box at the same width. **No new overfull box.**
- Underfull `\hbox` 7 → 22, Underfull `\vbox` 2 → 4. Every new underfull box is inside **Appendix E's narrow `p{1.52in}` table columns** (log lines 348–387), i.e. ragged short lines in the restored negatives rows. Cosmetic, no content loss; the restored rows render in full (verified in the PDF text layer: task-RMSE row p. 12, chaos/`sleep_temperature` rows p. 13).
- **Page split (same instrument as the prior notes: PDF word-bbox positions of the "References" and "Appendix A" headings):**

| | pre-pass | post-pass |
|---|---|---|
| main text | 6.83 pp | **6.91 pp** |
| references | 1.96 pp | **1.90 pp** |
| appendices A–G | 5.21 pp | **6.19 pp** |
| **total** | **14 pp** | **15 pp** |

Main text grew by **0.08 pp** (the 12 word/clause items); the pass's page cost is **almost entirely appendix** (+0.98 pp), which is the Head's amendment working as designed. **Reported, not optimized** — no page-fitting was attempted.

---

## 9. NOT RESTORED — the honest list

**(i) Banked items whose base home is MAIN TEXT — excluded by the task, listed as required:**
N46 rider at the negative's own site · the anchor non-novelty clause (Renart et al.) at §4.4 · the `legacy`-default noise warning and both flag names · the sleep flags (frequency 5 / 500 steps / CD sampler) at the erosion sentence · the orphan-citation sentences (UnICORNN, EDEN, Ramsauer, Huang et al., Jelassi, Hairer) · the 4.5-vs-4.6-decade distinguishing clauses · the seed attributions (5-seed vs 3-seed re-measurement) · `≈35×` wake MSE · the floor ripple (±8 steps at δ=4) · the coRNN honest-weak footnote · the "generally" widening at line 35 (SF-7, **unruled — still the Head's**) · all intensifier/garble edits.

**(ii) Excluded because not enumerated, though they sit in an appendix** — flagged for the Hub, not restored:
1. **The two Appendix-E reading rules that live in prose rather than in a row**: base L357's "…we report amplitudes and boundedness, never a ''drift rate''" and base L359's "(a) the breaking coefficient … is **not** this paper's integrator step $\varepsilon$". The third reading rule **is** in (it is inside the restored $n_{1/2}$-exponent row).
2. **Base L323's non-numeric clause** in the explicit-breaking row, "A designed degeneracy does not survive superposition" — the task authorises "the 6 missing rows + their within-row *numbers*", and this is neither.
3. **Base L324's final clause** "tangential curvature predicted $0.100$, measured $0.0994$" — absent from the R3 bank's fitted row and not separately enumerated.
4. **Base L382's far-field-bending clause** — item 8 enumerates three facts; this is a fourth.
5. **R3's cross-referencing halves of the two slope labels** (each site naming the other's statistic) and **R3's `0.973` pooled-correlation clause** at the 0.9987 site.
6. **App C's base lead-in one-liner** (base L263) and **App A's pointer prose** — neither is on this list; App A remains figure-only, as in the Head's rewrite.
7. Bibliography DOIs and stripped status annotations (one-word Head option, unchanged from R3).

**(iii) Figures:** ⛔ no figure work, per Part 3. **5/5 available PNGs already used** — confirmed.

---

## 10. Open items this pass deliberately did not resolve

1. **SF-7 — "Our results hold *generally* for the class of damped symplectic recurrences" (line 35)** is still there. It is not on the approved list. **Head's ruling still owed.**
2. **The `−0.956` / `−0.961` reconciliation has no owner.** The paper is now self-consistent — each number carries its fit spec at its own site — but no canonical anchored-overdamped slope is pinned in the claims matrix, so any future artifact quoting "the anchored slope" reopens it.
3. **Page discipline: 15 pp** against the EA track's 4. Deferred by Head ruling; the appendix bank is now larger, i.e. the pruning pass has more to cut from.
4. **`\S`-rule count discrepancy, explained**: the task said 9; the file had **11 occurrences on 9 lines** (lines 46 and 58 carry two each). A line-count grep gives 9, an occurrence-count grep gives 11. All 11 are converted; flagged so the count is not re-derived from the task text.

*Filed by `pj-minimal-v2` (paper-writer), 2026-08-22.*
