# V5 — PALM-audience reframe (variant build) — build note

Produced by `tasks/v5-palm-reframe.md` (Shorts-Advisor charter Addendum 38; Head ruling 2026-08-20:
V5 reframes for PALM, V2 for NeurReps). **This is a parallel variant, not a replacement.**

⛔ **`papers/v5-short/**` is byte-untouched by this build** — verified by a 25-file md5 manifest taken
before and after (`scratch/v5-palm-reframe/v5short_manifest_{before,after}.txt`, `diff` empty).

**DIAL DECLARATION: none — reframing/editorial pass; zero content, number or claim changes.**

## 1. What this variant is

Same results, same numbers, same claims as `papers/v5-short/submission/submission.tex`, **re-expressed in
the vocabulary and priorities of the PALM audience** as recorded in `outputs/v5-scope-scout.md` Part 1:
the ordering is **policy question → mechanism → number**, not mechanism first. The physics is demoted to
the derivation apparatus; the three results are stated as answers to *what is the retention policy*,
*can retention be scoped*, and *is deletion real and what leaks*.

| file | what it is |
|---|---|
| `submission.tex` | the submission source, assembled from `preamble.tex` + `main_body.tex` + `refs.tex` + `appendix.tex` (all four kept for auditability) |
| `submission.pdf` | **9 pp total**: main text 4.00 pp · references 0.75 pp · appendices 4.25 pp |
| `figs/` | three figures, byte-identical to the source build's (neutral filenames already applied there) |
| `neurips_2025_ml4ps.sty` | the template actually used, identical to the source build's |

Build: `pdflatex` ×3, TeX Live 2026, `/Library/TeX/texbin`. **0 errors, 0 undefined references.**
2 overfull and 1 underfull hbox — **the two overfull boxes are numerically identical to the source
build's** (`91.6832pt` and `406.18022pt`, both inside the `\tiny` wide instrument tables of Appendix B),
i.e. inherited, not introduced. Reported rather than silenced.

## 2. Page split (measured from the PDF)

| block | pages | note |
|---|---|---|
| main text | **4.00** | pp. 1–4; ends with Limitations at the foot of p. 4. **PALM short-track hard limit met.** |
| references | **0.75** | 30 entries, `\tiny`, two columns, top of p. 5 |
| appendices A–E | **4.25** | pp. 5–9 |
| **TOTAL** | **9** | **inside the 8–9 pp target band** (the source build is 10 pp) |

Main text = **3,051 words + 1 figure**; the source build's main text is **2,572** on the same tokenizer
(its own build note reports 2,267 under a different one). The reframe is *longer* in
words and *shorter* in pages; the difference is bought by typography and layout (§6), never by cutting
content.

⚠ **Template caveat, inherited verbatim from the source build.** Neither the PALM template nor the
NeurIPS 2026 style file is obtainable on this machine. This build uses the same locally-present
NeurIPS-family style as the source build (standard NeurIPS geometry: textwidth 5.5 in × textheight 9 in,
10 pt, submission mode with line numbers and the anonymous author block), with the workshop notice box
suppressed so no venue string appears in the artifact. **Re-measure in the real venue template before
submission.**

## 3. What changed relative to the source build

**Rewritten (vocabulary and ordering only):** abstract · §1 Introduction (incl. contributions) ·
§2 Related work (rebuilt from the scout's 13-work brief) · §3.1/§3.2/§3.3 opening frames and section
titles · §3.3's leakage paragraph re-ordered so the TTL comparison leads.

**Byte-identical to the source build:** Appendix A–E in full (`diff` against source lines 115–312 is
empty) · the nomenclature block · every fine-print block · every protected wording on the
`v5-referee-v02` §D do-not-cut list · the Limitations block · `figs/` · the `.sty`.

**Three substantive deltas, each deliberate and each flagged:**
1. **The TTL row is promoted from Appendix D into the main text** (`0.559` vs `0.996` at
   $\sigma_{\rm obs}=0.1$), because the task makes the TTL comparison the centrepiece of the leakage
   result. Both numbers are present verbatim in the source build's Appendix D table; no number changed.
   ⚠ The task brief quotes this pair as "0.559 vs 1.000"; **the measured table says `0.996` for the TTL
   arm**, and the table's number is what is printed.
2. **Three reference entries are restored to their canonical records.** The source build's bibliography
   carries placeholder titles for Yang (2026), Uddin et al. (2026) and Mo (2026); this build uses the
   verified records from the canonical `draft.md` / `outputs/v5-scope-scout.md`.
3. **Six references are added**, all cited in the rebuilt §2 and all scout-verified: Packer et al. 2023
   (MemGPT), Munkhdalai et al. 2024 (Infini-attention), Zhong et al. 2024 (MemoryBank), Sukhbaatar et al.
   2021 (Expire-Span), Wang & Zhang 2026 (MemLeak), Wang et al. 2026 (agentic unlearning). 24 → 30 entries.

One source typo is silently corrected: §2's `"and and exact methods"`.

## 4. Verification re-run on this artifact

- **Numeric two-way check** (`scratch/v5-palm-reframe/check_numeric.py`, comment-safe tokenizer):
  (i) numeric tokens in this build absent from the source build: **25, of which 24 are bibliographic
  identifiers of the six added / three restored references (all present in the canonical `draft.md`) and
  1 is a LaTeX length**. (ii) numeric tokens of the source build absent here: **1** — `0.84`, the source's
  headline-figure width. (iii) source main-text tokens absent from this main text: **1**, the same `0.84`.
  ⇒ **no content number left the main text and no content number was added.**
- **Compliance sweep** (`scratch/v5-palm-reframe/subsweep.py`, the source build's own instrument,
  per-file, positive-controlled): **zero-list hits = 2**, both the same known false positive as the
  source build (`n_{\rm jac}/n_{\rm R1}` and `\Gamma_{\rm jac}/\Gamma_{\rm R3}` in Appendix B's Table 3
  header — this paper's own instrument names, defined three paragraphs above). Positive controls all
  fired (107.77 ×8 · 106.1 ×3 · 0.9001 ×2 · Blelloch ×10 · the N108 sentence ×2 · confines ×3 · 8.11 ×6 ·
  Anonymous ×2 · "introduced as CHLU" ×1 · verification ×8 · evidence ×12 · 9.5e15 ×1 · 0.4586 ×2 ·
  ZERO ×2). Context-checked hits all compliant: `certified` ×3 (two literature descriptions of Guo +
  the explicit denial) · `unlearning` ×6 (the denial + one literature sentence + four reference entries) ·
  "deletion is exact" ×2 (both qualified store-level) · `CHLU` ×2 (continuity sentence + reference entry) ·
  `0.99985` ×1 (carries "at full load") · `297.8` ×1 (appendix, "never the vault number") · `23.39` ×3
  (all labelled designed-only / falsifier-fired).
- **Semantic hermeticity (C-8):** `companion` / `sibling` / `our other short` / `the program` /
  `forthcoming` / `in preparation` = **0**.
- **De-bold:** `\textbf` = **0**, main text and appendix.
- **Anonymization (identical posture to the source build):** `\author{}` blank · no `[WORKING TITLE`,
  no `[AUTHORS PLACEHOLDER]` · no acknowledgment, funding, URL or repository string · PDF Title, Author,
  Subject, Keywords, Creator and Producer all empty · **0** occurrences of any absolute path, username,
  project string, worktree name or venue string anywhere in the decompressed PDF. The third-person
  self-citation is intact and is the only occurrence of those names besides its reference entry:
  *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"*. PALM's
  code-inclusive anonymization requirement is carried by the closing anonymization note in Appendix E.
- **Honest scope sentence:** present **exactly once** (§1, *"What class of claim this is."*).

## 5. Operational vocabulary — what was adopted and what was refused

Adopted (all scout-backed, §1.2 of `outputs/v5-scope-scout.md`): *retention policy* · *retention dial* ·
*TTL / expiry* · *consolidation* · *scoping / scoped retention* · *deletion guarantee* vs *best-effort
deletion* · *membership adversary / privacy leakage* · *stale entry*.

⛔ Refused, on the Add.37 FLAG-2 boundary (*where an audience term would widen a claim, our term stays*):
**"right-to-be-forgotten"** and **"memory provenance"**. Both name compliance properties of a deployed
system; we have a store-level bit-exactness result with the encoder excluded, and using either term would
convert a mechanism statement into a system guarantee. Neither appears in the document.

Terms with no operational equivalent are kept and defined in one clause on first use: *spectral mass*,
*coset*, *atom*, *friction hole / vault*, *canonical placement*.

## 6. How 4.00 pp was met without cutting content

Word-for-word the reframe is 479 words longer than the source main text (3,051 vs 2,572, same tokenizer). The pages were bought by
layout, in this order:
1. §2 Related work and the nomenclature block set `\scriptsize` (were `\small` / `\footnotesize`).
2. All main-text fine-print blocks and the Limitations block set `\scriptsize` (were `\footnotesize`).
3. Heading skips tightened (`\section` / `\subsection` `\@startsection` redefinition); `\textfloatsep`,
   `\intextsep`, `\abovecaptionskip`, `\belowcaptionskip` tightened.
4. References set in two `multicol` columns at `\tiny` (30 entries fit in 0.75 pp).
5. Appendix body set `\scriptsize` (was `\footnotesize`); every table keeps its own explicit size, so no
   table changed.
6. **Headline figure width 0.84 → 0.60 `\linewidth`.**

⚠ **Item 6 is the one quality cost and it is measured, not guessed.** 0.60 is the *maximum* width at which
the main text still fits 4.00 pp: 0.62, 0.64, 0.66, 0.70, 0.76 and 0.84 were each built and all spill onto
p. 5 (and take the total to 10 pp). At 0.60 the figure is 3.3 in wide, against 4.6 in in the source build.
**Recommendation for the Hub: re-render `fig1_damping_optimum.png` with larger tick/axis fonts for this
variant** — that is a figure-generation job, out of scope for a text-only pass, and nothing else in this
build depends on it. Appendix figure widths are unchanged from the source (0.74 / 0.80).

## 7. Title

`[WORKING TITLE]` is not used because this is an anonymized submission-style build, matching the source
build's convention (real title, blank author block). The title in the artifact is the writer's
recommendation only; alternatives are listed in `outputs/v5-palm-reframe.md` and the choice is the Head's.
