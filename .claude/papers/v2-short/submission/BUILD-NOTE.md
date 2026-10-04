# V2 submission build — build note

Derived artifact. The internal canonical (`../draft.md`, `../draft.tex`, v0.8) is **byte-untouched**;
this directory is a *derived* build under the submission-build doctrine (canonical ≠ submission artifact).
Every block removed below is mapped to its canonical home, so nothing is lost and everything is findable.

## 1. Files

| file | what it is |
|---|---|
| `submission.tex` | the submission source (derived from `../draft.tex`) |
| `submission.pdf` | **11 pp total** (r9 style pass, 2026-08-19): **main 5.72 pp** (two figures) · **references 1.02 pp** · **appendices 4.23 pp**. Measured from the PDF text-block bounding boxes, see §9. *(r8 was 20 pp / main 5.4 pp.)* |
| `supplementary-theory-note.tex` / `.pdf` | the anonymized theory note, 12 pp, derived from `../f5-note/f5-note.tex` (**source byte-untouched**) |
| `figs/` | three figures, renamed to neutral filenames (see §5) |
| `neurips_2025_ml4ps.sty` | the template actually used (see §2) |

Build: `pdflatex` ×3 (TeX Live 2026). **0 errors, 0 undefined references, 0 overfull boxes** (r9; 8 cosmetic underfull hboxes in the `\scriptsize` negatives table — see §9).

Two cosmetic build settings, content-neutral: `\raggedbottom` is set for the appendix block (the style file's
`\flushbottom` opened large vertical holes around the big tables), and two table column specs were narrowed by
fractions of an inch to clear overfull boxes.

## 2. Template — what was used, and why

**Neither the NeurReps EA template nor the NeurIPS 2026 style file is obtainable on this machine.** A
filesystem search found no NeurReps or NeurIPS 2026 style file, and the network fetch of the official
NeurIPS style URL returned HTTP 403. The closest genuine NeurIPS-family style file present locally is
**`neurips_2025_ml4ps.sty`** (NeurIPS 2025 ML4PS workshop), which carries the standard NeurIPS
page geometry — **textwidth 5.5 in × textheight 9 in, 10 pt Times, submission mode with line numbers
and the `Anonymous Author(s)` block**. That is what this build uses.

Two deliberate deviations, both flagged:
- **The style file's workshop notice box is suppressed** (`\renewcommand{\@notice}{}`) so that no wrong
  venue string appears anywhere in the PDF. The artifact is venue-neutral.
- **The page count below must be re-measured in the real venue template before submission.** The
  geometry is the standard NeurIPS block, so the count should be close, but it is not certified.

## 3. Page count against the 4-pp limit  ⚠ **HISTORICAL (r8). Superseded by §9** — the r9 style pass re-measured everything and fired the whole costed menu; read §9 first.

### 3-old (r8 record) — the honest number, and what the Add.25 lean bought

- **Main text = 5 pp + 22 lines of p. 6 ≈ 5.42 pp** (a full body page is 53 lines in this template).
  References begin on p. 6 and are excluded from the budget; appendices are excluded.
- **Main text = 3,281 words + a 64-word figure caption + Figure 1 (≈0.38 pp).**
- **The Add.25 lean was applied**: GMOR-proper in §3.1 is now one sentence plus a pointer to the GMOR
  appendix. **Measured, by building both versions: it bought exactly one typeset line (≈0.02 pp).** The
  paragraph reflows, so the word saving does not convert into page saving at this position.
- **Residual overflow: ≈1.42 pp ≈ 940 words ≈ 29 % of the main text.** ⛔ **Nothing further was moved.**
  Closing it requires demoting a lead result or the headline figure, which is a Head-level decision.

**Costed menu (measured word counts in this build; a page is ≈664 words of running text):**

| block | words | note |
|---|---|---|
| Abstract | 241 | |
| §1 Introduction | 390 | the contributions list is the bulk |
| §2 Setup + two axes + fine print | 510 | ⛔ fine print (a)–(c) is C-6 claim-scope, protected |
| §3 opener | 36 | |
| §3.1 budget (verification-grade) | 261 | **cheapest demotion** — verification under C-2; the GMOR appendix already carries it |
| §3.2 head-to-head | 277 | the headline (C-3) — do not touch |
| Figure 1 + caption | ≈0.38 pp + 64 w | moving it frees ≈0.45 pp but costs the C-3 headline figure |
| §3.3 designed-vs-emergent + price | 276 | |
| §3.4 baselines + recipe | 372 | |
| §4 related work | 452 | ¶1 could compress; ¶2 carries the approved narrow-claim wording |
| §5 discussion | 466 | the limitations ¶ could become a scope box |

## 4. Appendix triage — every removal mapped to its canonical home  ⚠ **HISTORICAL (r8). The r9 drop map in §9 supersedes the "Kept" table** (six of the eleven kept appendices were dropped from the submission at r9); the "Removed outright" table below still stands.

### 4-old (r8 record)

**Removed outright**

| removed | canonical home |
|---|---|
| Flag-provenance appendix (B.1–B.9: commits, seeds, every non-default flag) | `../draft.md` / `../draft.tex` **Appendix B**, and the source agent reports it reproduces |
| **Appendix M** — the retained long-form main text, in full (M§1–M§5, plus Figures M1, 2b, 3) | `../draft.md` / `../draft.tex` **Appendix M** |
| The cross-reference-convention note (long-form §N.M numbering) | `../draft.tex` line after the References block — moot once Appendix M is gone |
| Venue-class header, draft-status block, reading-order note | `../draft.md` head matter |
| Appendix L.6 (scope-of-this-appendix paragraph) | `../draft.md` **Appendix L.6** |
| The "placement is a pending editorial decision" notes (Appendices C and K) | `../draft.md` Appendices C, K preambles |
| Attribution sentence in the primer ("adapted from a co-author's tutorial note … acknowledgement placed in the camera-ready") | `../draft.md` **Appendix A** preamble. Removed because an anonymized build carries no acknowledgment; **the camera-ready acknowledgment obligation stands** |
| Figures 2b (Mo-estimator overlay) and 3 (retention overlay), and Figure M1 (GMOR law) | they lived only inside Appendix M; source files remain in `../figs/` |

**Kept, per the Head's visual/measured-results criterion** — each trimmed to result + figure/table + mandatory fine print:

| submission | was | substance kept |
|---|---|---|
| A — $SO(2)$ primer | A | **Exception 1 (Add.11).** Boxed scope note compacted to one paragraph, all four clauses intact; corrected A.5 sentence kept **verbatim**, its inline HTML sign-off flag stripped (the comment, not the sentence) |
| B — erosion study | C | phase diagram, mechanism, cure, envelope, demarcation, cross-link, the $3000$-epoch survival result + **Figure 2**; the coverage statement and the chain-length scope clause kept verbatim |
| C — isotropization / charge oscillation | D | all measured splits, amplitudes and decay-law errors; both reporting cautions kept |
| D — kick-amplitude probe | E | the 3-row table + the non-cross-subtraction caution |
| E — exceptional point + damping corollary | F | E.1 signatures, E.2 the V-curve corollary with both consequences |
| F — negatives | G | **Exception 2 (Q4/C-9).** Now an 11-row compact table: claim tested · result · number. Two prose riders retained beneath it (the learned-store reading rules; the sampler kinetic-mode scope) because they are mandatory claim-scope, not commentary |
| G — loan curve + recovery ladder | H | both tables, the non-comparability caveat verbatim, the reach/certificate rungs |
| H — per-step compute | I | all three tables + the width-matching confound |
| I — GMOR proper | J | the 10-row identity table, the precision fine print verbatim, **Figure 3**, the three-instrument check, the LEC result and its breakdown cap |
| J — $T>0$ coset diffusion | K | the mandatory FDT/kinetic-mode flag box verbatim, the generalizes/does-not split, J.1–J.6 including the instrument caveat |
| K — positioning | L | **Exception 3.** Compressed to ≈half a page: the four retirements (K.1–K.4) + one paragraph of bounding prior art (K.5). L.6 dropped |

Appendix letters were renumbered (C→B, D→C, … L→K) and **all cross-references converted to `\ref`**, so no letter is skipped and nothing points at a removed appendix.

## 5. Other apparatus stripped

- All registry tokens (N4, N5, N46, N51, N149/N150, N6, N12–N15, N19, N22), claims-matrix rows (CM-4) and
  internal task IDs (SF-1…SF-3). **The findings themselves are all retained** — the negatives table carries
  every one of them, keyed by claim rather than by registry number.
- All source-report and experiment names, all commit hashes, all checkpoint identifiers.
- Figures renamed: `fig2_mo.png`→`fig1_mo_headtohead.png`, `sf3_anchored3000_laws.png`→`fig2_anchor_cure_laws.png`,
  `fig_gmor_condensate.png`→`fig3_gmor_condensate.png` (the old names embedded an internal task ID).
- Retained deliberately: implementation flag names (`langevin_noise="fdt"`, `newtonian_learned`,
  `sleep_steps`, `sleep_temperature`, `sleep_friction`). These are **mandatory C-6 riders about the
  reference implementation**, not internal process, and the appendices they qualify are undefined without them.

## 6. Parentheses purge — the log

Counted over the region of the canonical that this build keeps (main text + retained appendices), on
*prose-bearing* parentheticals, i.e. those containing three or more letters outside math:

| | count |
|---|---|
| prose-bearing parentheticals in the canonical kept-region | 392 |
| prose-bearing parentheticals in `submission.tex` | 210 |
| **net reduction** | **182** |
| of the survivors, carried verbatim | 136 |
| of the survivors, re-phrased in place | 74 |

Of the **257** canonical parentheticals that were removed or converted:

| class | n | disposition |
|---|---|---|
| internal provenance / registry / source-report pointer | 33 | **stripped** |
| cross-reference / pointer chain | 32 | **stripped**, or replaced by a single `\ref` |
| process / editorial aside ("(headline)", "(placement pending)") | 4 | **stripped** |
| C-5 scale qualifier | 26 | **converted to prose — retained in the sentence** |
| statistical / comparison qualifier (±, predicted, median, seed counts) | 28 | **converted to prose — retained in the sentence** |
| C-2 verification/evidence label | 3 | **converted to prose — retained in the sentence** |
| technical gloss and other phrasing folded into the sentence | 131 | **converted to prose** |

**Nothing in the mandatory classes disappeared.** Verified two ways: (i) every numeric token in
`submission.tex` occurs in the canonical (0 exceptions, 446 distinct tokens); (ii) in reverse, every
numeric token of the canonical **main text** occurs in the submission main text — the only absences are
the section numbers `3.3`/`3.4`, now rendered by `\ref`, and the registry token `149`.

## 7. Anonymization

- `Anonymous Author(s) / Affiliation / Address / email` (the style file's submission-mode block). No acknowledgments, no funding.
- PDF metadata scrubbed: **Author, Title, Subject, Keywords all empty**; Creator `LaTeX with hyperref`; Producer `pdfTeX-1.40.29`. No absolute path, username or project string anywhere in the file, compressed or otherwise.
- Third-person self-citation intact: *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"*. The two `CHLU` occurrences in the artifact are this sentence and the corresponding reference entry.
- The theory note is cited as **"Anonymous (2026), provided in the supplementary material"** (Option B) and the anonymized supplementary PDF is attached here.
- Supplementary: `\author{Anonymous}`, date cleared, arXiv preprint-class line removed, the internal provenance appendix removed, internal preamble comments removed. Its own metadata is likewise empty. `../f5-note/f5-note.tex` is byte-untouched.

## 8. Final sweep on `submission.tex` (per-file, positive-controlled)

**ZERO HITS:** `commit` · `branch` (as a git token) · `agent/` · `chlu/` · `.claude` · `tectonic` · `draft.md` · `draft.tex` ·
`Registry`/`registry` · `provenance` · `Appendix M` · `N<digits>` · `CM-<n>` · `SF-<n>` · `MF-<n>` · `[WORKING TITLE` ·
`[AUTHORS PLACEHOLDER]` · `<!--` · every source-report name · every checkpoint name ·
CLU-former · certified · unlearning · exact deletion · "the item is gone" · "exact discrete FDT" · "samples Gibbs" ·
0.384 · 16.28 · CAFE · C-MAPSS · N-CMAPSS · HEPA · CAMELS · bpc · S_eff · z_hole · 0.99985 · 54.56 · 306.76 · 300.09 ·
deltanet · ttt_mlp · MUNKEY · 0.4545 · 13.9 · memory vault · 107.77 · compositional · unaskable · Guo · Ginart ·
Sekhari · Track A · waitlist · paid-access · companion · sibling · "our other" · "this program" · "the program" ·
experiment-engineer · "per the Head" · wormhole.

**HITS, context-checked, compliant:** `CHLU` ×2 (the sanctioned continuity sentence + its reference entry) ·
"energy units" ×1 (the refusal statement in the FDT flag box) · `2.6` ×4 (all grid/probe values: $2.6\times10^{-2}$,
$2.6\times10^{-6}$, $2.69\times10^{-14}$ — never the retracted number) · "critical damping" ×1 and "V-curve"/"V-shape" ×4
(**all after `\appendix`; zero in the abstract, §1 or §3**) · `branch` ×3 and `head` (overdamped branch; head-to-head,
headline, single head) · `Schmidhuber` (the false-positive source of a `hub` match).

**POSITIVE CONTROLS FIRED:** GMOR 15 · "introduced as CHLU" 1 · Rusch 7 · verification 11 · evidence 20 · Anonymous 3 · 33 reference entries.

---

# 9. r9 — the Head's six-instruction style pass (2026-08-19)

Executed against `.claude/tasks/v2-style-pass.md` (Advisor Add. 32 + the Head's 2026-08-19 directives).
⛔ Canonical (`../draft.md`, `../draft.tex`) byte-untouched. ⛔ `supplementary-theory-note.*` byte-untouched.
Build: `pdflatex` ×3, TeX Live 2026 — **0 errors, 0 undefined references, 0 overfull boxes.**
⚠ **8 underfull hboxes**, all inside the `\scriptsize` narrow `p{}` cells of the negatives table (loose word
spacing, cosmetic). Ragged-right column specs clear all 8 — **measured: at the cost of +1 page (11 → 12 pp)** —
so justification was kept and the 8 are reported rather than silenced.

## 9.1 The measured page split (the headline number)

Measured from the PDF word bounding boxes against the text block (top 72 pt, bottom 720 pt, page 792 pt):

| block | r8 | r9 | note |
|---|---|---|---|
| main text | 5.42 pp (1 figure) | **5.72 pp** (2 figures) | see the arithmetic in §9.6 |
| references | ≈1.06 pp, 33 entries, `\small` | **1.02 pp**, 28 entries, `\footnotesize` | 5 entries orphaned by the appendix cut, dropped (§9.4) |
| appendices | ≈12 pp | **4.23 pp** | six appendices dissolved or dropped |
| **TOTAL** | **20 pp** | **11 pp** | ✅ under the 12-pp hard ceiling; ⚠ **above the 8–9 pp target band** |

⚠ **Two acceptance criteria are MISSED and the reason is measured, not stylistic:** main text is 5.72 pp
against a ≤4 pp target, and the total is 11 pp against an 8–9 pp target. §9.6 gives the arithmetic and
§9.7 the costed menu of what would close the remaining gap — every item on that menu contradicts one of
the pass's own binding instructions, so the decision is the Head's, not the writer's.

## 9.2 Structure (instruction 1) — related work moved to §2, retirements appendix dissolved

New order: **§1 Introduction → §2 Related work → §3 Setup and the two axes (fine print intact) → §4 Results
→ §5 Discussion.** All cross-references converted; `pdflatex` reports zero undefined references.

- The retirements appendix (r8 Appendix K, "Positioning") **no longer exists**. K.1–K.4 are folded into §2 as
  a `\footnotesize` run-in paragraph, *"Four claims we retire, so that a reviewer's first objections are
  answered in writing"*; K.5's bounding prior art is folded into §2 ¶2 (the Ramsauer/modern-Hopfield
  sentence joins the existing HiPPO / Kong / NTM-DNC / EDEN / UnICORNN / Titans list).
- The approved narrow-claim wording carries **verbatim** in §2 ¶2 (`positive control: "controllability, which
  we evidence, not capability"` present).
- §4 Results now opens on a trained/learned system and the verification block sits in Appendix C (menu item 1).

## 9.3 Figure promotion (instruction 3)

`fig2_anchor_cure_laws.png` (the anchored-laws / retention plot) is now **Figure 2 of the main text**, placed
with the result it illustrates — §4.3's anchor-cure sentence (`every headline law of the budget still holds
under it at ≈20× the erosion horizon`). Its caption travels verbatim from the r8 appendix, with the single
cross-reference edit `\S\ref{sec:budget}` → "the budget" (that section is now Appendix C).
Main-text figures: **Figure 1** (head-to-head, the C-3 headline, untouched in content) at `0.9\linewidth`,
**Figure 2** at `0.9\linewidth`. Figure 3 (GMOR) stays in Appendix C at `0.60\textwidth`
(reduced from `\textwidth` purely to close a near-empty float page; content-neutral).

## 9.4 Appendix drop map (instruction 2 + the instruction-6 overflow rule)

Surviving appendices, merged under three headings:
**A Supplementary results** (A.1 loan curve + recovery ladder, A.2 per-step compute) · **B Prominent
negatives** · **C The mode-mass budget verified, and GMOR proper**.

| r8 appendix | r9 disposition | canonical home |
|---|---|---|
| A — $SO(2)$ primer | **DROPPED** — prose-only. The Add.11 exception is overridden by the Head's tightened plots/tables rule. ⚠ *the camera-ready acknowledgment obligation for the colleague's tutorial note still stands* | `../draft.md` / `../draft.tex` **Appendix A** |
| B — erosion study | **DROPPED** — its only visual (Figure 2) was promoted to the main text under instruction 3, leaving prose. Its mandatory chain-length scope clause was **relocated verbatim** into §4.3 (§9.5) | `../draft.md` **Appendix C** |
| C — isotropization / charge oscillation | **DROPPED** — prose-only. Its reporting caution was **relocated** to Appendix B beneath the isotropization row (§9.5) | `../draft.md` **Appendix D** |
| D — kick-amplitude probe | **DROPPED under the instruction-6 overflow rule** — the least load-bearing surviving table (it backs a limitation clause, not a claim). Its result is now stated inline in the §5 scope box (§9.5); its non-cross-subtraction caution drops with the numbers it qualifies | `../draft.md` **Appendix E** |
| E — exceptional point + damping corollary | **DROPPED** — no table. E.1's signatures and E.2's damping corollary with **both** consequences are folded into Appendix C.1/C.2 | `../draft.md` **Appendix F** |
| F — negatives | **KEPT → Appendix B.** 11-row table complete, now split into two `tabular` blocks so it can break across pages (formatting only; zero row changes). All riders kept | — |
| G — loan curve + recovery ladder | **KEPT → A.1.** Both tables + the non-comparability caveat verbatim. **G.3 (reach/certificate rungs) dropped** — prose-only under the instruction-6 "figure/table + result sentence + fine print" rule | `../draft.md` **Appendix H.3** |
| H — per-step compute | **KEPT → A.2.** Both tables + the width-matching confound verbatim | — |
| I — GMOR proper | **KEPT → Appendix C** (now also the home of the demoted §budget verification) | — |
| J — $T>0$ coset diffusion | **DROPPED** — prose-only, no table. Its **mandatory FDT/kinetic-mode flag box relocated verbatim to §3**, next to fine print (a), the claim it qualifies (§9.5) | `../draft.md` **Appendix K** |
| K — positioning | **DISSOLVED into §2** (instruction 1) | `../draft.md` **Appendix L** |

**References dropped as orphaned by the appendix cut** (each was cited only from a dropped appendix; verified
by a per-name sweep of the remaining body): Agoritsas et al. 2023 · Bhatt, Floyd & Moore 2016 · Decelle,
Furtlehner & Seoane 2021 · Fischer & Igel **2011** (the 2010 entry is still cited in §2) · Toledo-Marín et al.
2025. 33 → 28 entries.

**Submission-absent / canonical-present numbers** (permitted; the canonical is the archive). From the dropped
appendices: the SO(2) primer's emergent-arm figures; the erosion phase diagram/mechanism/cure/envelope/
demarcation numbers (B.1–B.6) *except* those already carried by §4.3 and the Figure 2 caption; the
isotropization splits, amplitudes and decay-law errors *except* those in the Appendix B negatives row; the
kick-probe table (`1.125 / 1.288 / 1.052 / −0.006 / +0.000 / +1.28 / 5.45×10⁻² / 1.07×10⁻¹ / 2.03×10⁻² /
277,257,303 / 247,227,263 / 1.07→1.55`) — its `+5 % … +29 %` headline survives inline in §5; the loan
appendix's reach/certificate rungs (`+77 % at c=0.5`, `v_max 0.62`, `ζ=1`, `Δq 2.18`, `det S 1.0000`,
`3×10⁻⁷`, `2.28 ≤ 7.39`, `κ=0`); the whole $T>0$ appendix (J.1–J.6) *except* the `3.77±0.23×`, `T*≈3×10⁻³`,
`1–1.6 bits`, `2–3 minima` and `13–14 orders` figures, all of which are carried in §3/§4/§5.
⛔ **Every number cited from the main text still resolves inside the submission** — verified in §9.8.

## 9.5 Rider-relocation list (riders never drop with their appendix)

| rider | was | now | edit |
|---|---|---|---|
| **FDT / kinetic-mode mandatory flag box** | r8 Appendix J preamble | **§3**, immediately after fine print (a), which is the claim it qualifies | verbatim except three repositioning words: label "in this appendix" → "in this paper"; "every number below is in scope" → "every finite-temperature number here is in scope"; "Nothing in §Results depends on this appendix" → "…depends on it" |
| **Chain-length scope clause** (`sleep_steps ∈ {50,500}`, frequency-decisive) | r8 Appendix B preamble | **§4.3**, inline after the erosion claim, set `\footnotesize` | verbatim, zero word changes |
| **Charge-oscillation reporting caution** (on-vacuum-orbit only; no proportionality constant may be quoted as a law; mass-tying is about the *current*, not the register) | r8 Appendix C | **Appendix B**, as a rider beneath the negatives table's isotropization row | compressed to its two load-bearing clauses, wording preserved |
| **Damping-corollary consequences (i) and (ii)** | r8 Appendix E.2 | **Appendix C.2** | verbatim |
| **Kick-probe anharmonicity result** | r8 Appendix D | **§5 scope box**, stated inline (`+5 % to +29 %` zero-kick bias, kick-independent 2/3 seeds, amplitude-dependent on the softest-$\mu^2$ seed) | restated so the scope box's pointer does not dangle |
| fine print (a)–(c) · negatives-table reading rules · sampler kinetic-mode scope · non-comparability caveat · GMOR precision fine print · width-matching confound · narrow-claim wording · continuity sentence | — | in place | **verbatim, unmoved** |

Pointer repairs inside the kept negatives table (letters only, zero content change):
`Appendix~\ref{app:iso}` → removed · `Appendix~\ref{app:erosion}` → `\S\ref{sec:baselines}` ·
`Appendix~\ref{app:tcube}` → `\S\ref{sec:price}`.

## 9.6 Brevity + de-bold (instructions 4, 5) and the menu (instruction 4)

**De-bold: `\textbf` in the main text = 0** (grep printed in the spoke report). Nine `\paragraph{}` run-in
headers remain — those are the document class's structural formatting, which the instruction exempts.
Definitions now carry italics or plain prose. Appendix table headers keep structural bold (49 `\textbf` in the
appendix block, as permitted).

**Per-section word counts, r9** (tokenizer: math stripped to one token, LaTeX commands stripped; the r8
column is quoted from §3-old, which used a *different* tokenizer, so treat the delta as indicative ±5 %):

| block | r8 | r9 | note |
|---|---|---|---|
| Abstract | 241 | 234 | ABT rewritten (setup → gap → resolution); all numbers retained |
| §1 Introduction | 390 | 324 | ABT ¶1; contributions enumerated on p. 1 |
| §2 Related work (incl. the four retirements) | 452 | **690** | **+238: the dissolved retirements appendix lands here** (instruction 1). ¶1 compressed — menu item 2 fired |
| §3 Setup + two axes + fine print + **FDT box** | 510 | **595** | **+85: the relocated FDT flag box.** Fine print (a)–(c) verbatim |
| §4 opener | 36 | 83 | now carries the demoted verification headline numbers as a pointer sentence |
| §4.1 head-to-head (+Fig. 1 caption) | 277 + 64 | 348 | ⛔ content untouched; connective wording only |
| §4.2 designed-vs-emergent + price | 276 | 280 | |
| §4.3 baselines + recipe (+**Fig. 2 caption**, + scope clause) | 372 | **529** | **+157: the promoted figure's caption (110) + the relocated chain-length clause (60)** |
| §5 Discussion | 466 | 466 | limitations → scope box — menu item 3 fired; horizon compressed |
| §3.1 budget (verification) | 261 | **0 in main** | **menu item 1 fired**: demoted to Appendix C.1/C.2 |
| **MAIN TOTAL** | 3,345 | **3,549** | |

**The arithmetic of the 5.72 pp, stated plainly.** The main text is 0.30 pp *longer* than r8 despite a full
brevity pass, because the pass was required to *add* to it: `+0.27 pp` for the promoted Figure 2 and its
caption, `+0.36 pp` for the retirements paragraph, `+0.13 pp` for the FDT flag box, `+0.09 pp` for the
chain-length clause, `+0.04 pp` for the inlined kick-probe result and the Ramsauer sentence; against
`−0.39 pp` from the menu-item-1 demotion, `−0.16 pp` from setting the four protected boxes (fine print,
FDT box, scope clause, scope box) and the retirements paragraph in `\footnotesize`, and `−0.04 pp` of prose
compression. **The prose compression is small because the main text is not fat:** of its 3,549 words,
≈1,300 are protected-verbatim blocks that may be repositioned but not paraphrased, ≈520 are riders this pass
was ordered to relocate *into* it, ≈175 are figure captions, and most of the remainder is number- or
citation-bearing results prose. r8 had already condensed by relocation twice.

Cosmetic, content-neutral settings added at r9 (in addition to r8's `\raggedbottom` and column narrowing):
the appendix block is set `\footnotesize`, the references `\footnotesize`, the negatives table `\scriptsize`
and split into two `tabular` blocks so it can break across pages, and the three main-text protected boxes
plus the retirements paragraph are `\footnotesize`.

## 9.7 The residual gap — costed menu for the Head (nothing on it was taken unilaterally)

To reach main ≤4 pp / total 8–9 pp, one or more of these must go. **Each contradicts a binding instruction of
this pass or a Charter rule, which is why the writer stopped here:**

| item | saving | what it costs |
|---|---|---|
| Drop Figure 2 from the main text (return it to an appendix) | ≈0.27 pp | contradicts instruction 3 |
| Move the four retirements back to an appendix | ≈0.36 pp | contradicts instruction 1 (and the CM-21 obligation is currently discharged in main text) |
| Move the §5 scope box to an appendix | ≈0.35 pp | contradicts Charter C-6 (fine print next to the claim) |
| Move fine print (a)–(c) + the FDT box out of §3 | ≈0.48 pp | contradicts C-6 and §9.5's rider rule |
| Drop the per-step-compute appendix (A.2) | ≈0.60 pp | contradicts instruction 2's explicit KEEP; the honest-gap receipts become canonical-only (the ratios themselves survive in §4.3) |
| Drop the recovery-ladder table (A.1, second table) | ≈0.20 pp | contradicts instruction 2's KEEP |
| Demote §4.2's price paragraph to an appendix | ≈0.30 pp | removes an evidence-grade result from the main text (C-2/C-3) |

## 9.8 Verification re-run at r9 (all printed in the spoke report)

- **Numeric two-way check.** Direction (i), every numeric token in `submission.tex` occurs in the canonical:
  **2 exceptions, both typographic** — `1.52` and `2.06` are `p{}` column widths in the negatives table, not
  content. Direction (ii), numeric tokens of the canonical **main text** absent from the submission **main
  text**: **8** — `3.3`, `3.4`, `3.5` (canonical section numbers, now rendered by `\ref`), `22`, `46`, `149`
  (registry tokens, stripped per §5), and `6`, `2.7`, both of which are present elsewhere in the submission
  (they travelled with the demoted verification block into Appendix C). **No content number left the
  submission via the main text.**
- **Final sweep** (the §8 pattern set, per-file, positive-controlled): **ALL CLEAR on every zero-hit pattern**,
  `N<digits>` registry tokens **0**. Context-checked hits: `CHLU` ×2 (the sanctioned continuity sentence in §1
  + the reference entry) · "energy units" ×1 (the FDT flag box's refusal statement, now in §3) · `2.6` ×4 (all
  grid/probe values, **all in the appendix**) · `critical-damping` ×1 and `V-shape` ×1 (**both after
  `\appendix`; zero in the abstract, §1, §2 or §3**) · `V-curve` **0**.
  Positive controls fired: GMOR 13 · "introduced as CHLU" 1 · Rusch 6 · verification 6 · evidence 10 ·
  Anonymous 2 · 28 reference entries.
- **Anonymization** unchanged and re-verified: PDF Title/Subject/Keywords/Author all empty, Creator
  `LaTeX with hyperref`, Producer `pdfTeX-1.40.29`; `\author{}`; no `[WORKING TITLE`, no
  `[AUTHORS PLACEHOLDER]`.
