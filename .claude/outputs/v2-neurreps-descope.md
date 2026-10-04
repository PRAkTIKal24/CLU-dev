# v2-neurreps-descope — paper-writer report

**DIAL DECLARATION (echoed): none — scoping/editorial pass; zero number changes, zero retractions.**
Laundering control: n/a (no performance number produced). Falsifies: n/a. Does not falsify: n/a.

Task + acceptance criterion: de-scope V2's extended abstract to ONE contribution (the price list), re-aim §2 at
the current audience, apply the novelty retraction, foreground the negatives, finish the paper, deliver the
condensation aid — in a new folder, with both existing V2 artifacts byte-untouched.
Status: **done, with one acceptance criterion FAILED and fully documented (item 1 below).**

## ⚠ DOWNSTREAM RECONCILIATION LIST (first-10-lines rule, AGENT_PROTOCOL §5) — needs an owner

1. ⛔ **Acceptance criterion 4 is HALF-FAILED and the Advisor must rule.** `papers/v2-short/**` is byte-identical
   (21/21 files, verified). **`papers/neurreps-variants/v2/**` is NOT**: its three figure PNGs were rewritten by
   the **concurrent `figure-render-pass` spoke** (07:45:48 — it re-rendered V5's variant figures in the same
   second), and its `submission.pdf` hash changed via a stray 07:28:46 `pdflatex` in that folder which this pass
   first mis-attributed to itself and then "repaired" twice. Current state is a converged 3-pass build of the
   **byte-identical** `submission.tex` against the folder's **current** figures, 13 pp, main re-measured 5.71 pp
   vs its BUILD-NOTE's 5.69. Byte-exact restoration was attempted and is **not achievable** (reason and evidence
   in `papers/v2-neurreps-descoped/BUILD-NOTE.md` §9). ⭐ **Standing lesson for the Hub: a "byte-untouched"
   criterion must never name a directory another in-flight spoke is commissioned to write to.** The reframe's own
   build note records the identical incident one pass earlier. **Owner: the Shorts Advisor.**
2. ⛔ **Add.40/41's "the de-scope reaches 4 pp" projection is REFUTED by measurement, and the reason is
   structural.** It assumed the price list stays in an appendix; it cannot, because it is now the paper's single
   contribution and Head policy is *main text = main results only*. **Main text is 6.19 pp.** Measured floor
   without any C-6 trade: **5.16 pp**; with the headline figure also demoted: **4.81 pp**. **Owner: the Head**
   (the condensation aid is built for exactly this decision).
3. **Dönmez (2024) is printed without a page range** — the scout retrieved author/title/year/volume but not
   pages. Either drop the entry or retrieve the range. **Owner: the Advisor.**
4. **The figure follow-up is now mechanical and hash-pinned** (BUILD-NOTE §10): this build uses the pre-re-render
   PNGs, with both hash sets tabulated. ⛔ Do not copy the new PNGs in without the analyst's caption edits — the
   file sizes moved a lot (fig3 331 kB → 189 kB, fig1 78 kB → 203 kB). **Owner: the Advisor.**

---

## What I did

**Output: `papers/v2-neurreps-descoped/`** (new) — `submission.tex` · `submission.pdf` · `BUILD-NOTE.md` ·
`figs/` (3 PNGs) · `neurips_2025_ml4ps.sty` · `supplementary-theory-note.{tex,pdf}` (copied across).

- **De-scoped to one contribution.** Main text carries the transverse-curvature price list in three parts (what a
  curvature buys · where the law stops · whether it survives the correction), with the Mo head-to-head as its
  evidence (§4.2) and the designed-vs-emergent boundary as its honest negative (§4.3). Abstract and the
  contributions paragraph claim nothing else.
- **Promoted the price list into main text (§4.1).** In the source reframe its numbers lived only in Appendix C
  while the contributions list pointed at the appendix. A single-contribution paper cannot have its contribution's
  evidence only in an appendix. Appendix F was de-duplicated in the same move and now carries only what §4.1 does
  not (the fine grid range, the prefactor reconciliation, the quality-factor sentence, the real-pair statement).
- **Demoted (never retracted):** GMOR proper → App F.3–F.6 + Fig 3 + the 10-row δ table (has both a plot and a
  table); the price of the prior → App B + two tables (has both); the realization taxonomy → **canonical-only**
  (prose-only, no plot, no table), with a 183-word definitional gloss retained in §3 — see "judgement calls".
  Each appendix home opens with a line saying it is demoted, not a claim of this paper.
- **§2 rewritten for the current audience.** Opens on the two topic areas the paper sits in (representation
  *dynamics*; symmetry + dynamical systems + learning) **without naming a venue** (the venue-neutrality sweep
  forbids it). Structure: the object and its established properties → the five facts that are not ours (carrying
  the binding retraction) → the audience's own nearest work → comparators + drift disclaimer → the retirements.
- **Novelty retraction printed in the scout's scoped form in three places** (§2, §4.3's N46 rider verbatim,
  §4.4), and ⛔ nothing anywhere claims a first report of the destroy-and-restore pattern or its cure.
- **Xu et al. (2023) given the position the scout asked for**: a full sentence naming it as the closest published
  instance of this paper's object, then what we add ("a price list for time").
- **Two v228 citations added**, both scout-verified: Vastola (2024) optimal packing of attractor states;
  Dönmez (2024) memory modification through symmetry and geometry.
- **Negatives foregrounded** in the abstract's THEREFORE clause and in the contributions paragraph; §4.3 is
  titled *"The honest negative: where the price list does not extend."*
- **Renamed the object** from "the transverse-curvature budget" to **"the price list"** throughout, per the
  ruling's own wording (no protected block contains the old term; verified).
- **Wrote the condensation aid** (BUILD-NOTE §6): 21-row block table with words · measured pp · PROTECTED/FREE ·
  purpose; nine cuts each **measured by actually building them**; three combined builds; the three largest free
  blocks ranked; and a ten-row "removal changes this claim" tripwire table.

## How I verified (commands run, real numbers)

- **Build:** `pdflatex ×3`, TeX Live 2026 — **0 errors, 0 undefined references, 0 overfull boxes**, 14 pp.
- **Page split** (PDF word bounding boxes vs the text block, the same instrument the other two builds used):
  abstract 0.85 · §1 0.63 · §2 1.20 · §3 0.95 · §4 2.00 (preamble 0.05 / 4.1 0.70 / 4.2 0.37 / 4.3 0.54 /
  4.4 0.34) · §5 0.55 → **MAIN 6.19 pp**; references **1.66** (47 entries); appendices **6.16**; **TOTAL 14 pp**.
  Main text = **4,314 words** + one figure. Against r9 (5.72 main / 11 total) and the reframe (5.69 / 13).
- **Why main grew (+0.50 vs the reframe), itemized:** taxonomy out −0.29, price-of-the-prior out −0.25,
  **price list promoted in +0.75**, §2 audience rebuild +0.25, abstract negatives +0.15.
- **Measured cut menu** (each a real 3-pass build): fine print+FDT box −0.37 ⛔C-6 · Figure 1 −0.34 ⛔C-3 ·
  scope box −0.30 ⛔C-6 · baselines+honest gap −0.27 ✅free-as-a-pair · §2 nearest-work −0.27 ✅free ·
  curvature gloss −0.27 ⚠ · retirements −0.27 ⛔CM-21 · §5 horizon −0.16 ✅free · "Two masses" −0.16 ⛔.
  **Combined: all-free 5.44 · all-free+gloss 5.16 · +Fig 1 out 4.81.**
- **Two-way numeric-token check** vs `neurreps-variants/v2/submission.tex` (390 vs 392 distinct tokens):
  **in source, absent here: 0 tokens.** In this build, absent from source: **2** — `228` and `425`, the PMLR
  volume and first page of the two new citations. ⭐ No content number added, changed, rounded or moved.
- **Protected-wording verbatim check:** 20 blocks compared character-for-character after whitespace
  normalization → **20/20 VERBATIM**, modulo four cross-reference repairs with zero wording change
  (`sec:baselines`→`sec:boundary` ×3; and App B's `\S\ref{sec:price}` → "above"/"the param-matched table",
  since that table now sits two paragraphs above in the same appendix — a C-6 adjacency improvement).
- **Sweeps — 104 patterns, per-file, on both `.tex` and extracted PDF text: ONE hit.** `workshop` (tex ×2: a
  source comment about suppressing the notice box, not in the PDF, + the J&P reference venue string; pdf ×1),
  identical to both other builds. Patterns included `NeurReps`, `Wigner`, `Weyl`, `Axis 1`, `Axis 2`,
  `price of the physics`, all `N<digits>`/`CM-n`/`SF-n`/`MF-n`, the seed short-tags, prereg item names,
  the full never-quote number list, the deletion/certification bans and the semantic-hermeticity set.
- **Positive controls fired (tex|pdf):** GMOR 9|9 · "introduced as CHLU" 1|1 · verification 6|6 · evidence 13|13 ·
  "price list" 24|24 · Goldstone 8|8 · "continuous attractor" 4|4 · biological 1|1 · Vastola 2|2 · Dönmez 2|2 ·
  Renart 3|3 · Vafidis 3|3 · "separate note" 1|1.
- **F7 pressure points checked by hand:** "continuous attractor" is **never applied to our unit** (its 3 body
  uses: the literature's definition, other people's task-trained networks, Xu et al.'s object); "solves the
  fine-tuning problem" 0; no phrasing implies learning cannot build a flat direction (the N46 rider is verbatim
  and Vafidis is cited at the claim); "drift" is never bare (all 6 occurrences enumerated in BUILD-NOTE §7).
- **Style:** `\textbf` in main text = **0**. No-biological-claim sentence = **exactly 1**.
- **Citations:** all **47** entries cited from the surviving body (mechanical surname match + manual confirmation
  for `Anonymous`, `Di Bernardo`, `Rusch & Mishra 2021b`, `Wang & Ponce`, `Ságodi`, `Dönmez`). **No entry was
  orphaned by the de-scope**, so none was dropped.
- **Anonymization:** `\author{}` blank; PDF Author/Title/Subject/Keywords all empty; decompressed-stream sweep
  (82 streams, 16.6 MB inflated): `Forgis` 0 · `x10719pj` 0 · `Users/user` 0 · `Desktop` 0 · `CERN` 0 ·
  `Manchester` 0 · `.claude` 0 · `neurreps-variants` 0 · `v2-neurreps-descoped` 0 · `/tmp/` 0 ·
  `WORKING TITLE` 0. **Positive control fired:** `Goldstone` 8.
- **Protected folders:** 30-file `shasum` manifest before and after. `v2-short` 21/21 byte-identical.
  `neurreps-variants` — 4 files changed; see reconciliation item 1 and BUILD-NOTE §9.

## Judgement calls the Hub should review

1. **The taxonomy gloss is retained (183 words, 0.27 pp) on a C-6 argument, not a content argument.** Three
   verbatim-protected passages point at it: fine print (c)'s *"an observed instance of **the cell**"*, App D's
   *"a degenerate **pseudo-Goldstone** multiplet"*, and §4.4's prediction of a degenerate `μ²` pair, which only
   Schur's lemma explains. Deleting the gloss alone leaves three protected sentences without a referent.
   ⛔ It appears in no contributions list and claims nothing. Overrule freely — but re-anchor those three in the
   same edit. Everything else about the taxonomy (Axis 1/Axis 2, the three cell headings as a labelled partition,
   contributions item (3)) is **absent from the submission and canonical-only**, listed in BUILD-NOTE §3.4.
2. **The title is unchanged** from the reframe. It now describes the single retained contribution exactly, and
   the `[WORKING TITLE: …]` placeholder is on the never-print sweep for built artifacts (both sibling builds
   print the real title). Flagged rather than silently decided.
3. **The learned-baselines paragraph + honest gap stayed in main text** although the ruling does not name them.
   They are negatives, which the track's stated purpose wants foregrounded. They are the **largest free block**
   in the aid (0.27 pp) with a ⛔ tripwire: keeping the `263`-map-step number without the honest-gap paragraph
   re-asserts a retired compute claim (C-6/CM-4). Cut as a pair or not at all.
4. **The anchor/erosion section stayed in main text as part (c) of the contribution**, on the scout's own
   approved sentence (*"…and whether it survives the correction. We measure all three."*). It is therefore not
   a fourth contribution; §4.4's opening says so explicitly.

## Missing-experiment / missing-fact notes (not improvised)

- **Dönmez (2024) page range** — not in the scout report; printed as `PMLR 228` with no pages.
- Nothing else was needed. Every quantity in the paper is inherited unchanged from the reframe, which inherits
  from the source reports; the two-way check proves zero drift.

## Git footprint

None — this pass wrote only under `.claude/` (gitignored). No tracked file touched, no branch created.

## Proposed handover updates (for the Hub)

- **A third V2 artifact exists and is finished:** `papers/v2-neurreps-descoped/` — 14 pp total, **main 6.19 pp**,
  refs 1.66 (47 entries), appendices 6.16. One contribution, current-audience §2, novelty retraction applied,
  negatives foregrounded, zero number changes, zero retractions, 20/20 protected blocks verbatim.
- **Add.40/41's 4-pp projection is refuted by measurement** and the reason is structural, not prose: the
  retained contribution's own evidence has to be in main text. Free-only floor **5.16 pp**; **4.81 pp** if the
  headline figure is demoted; 4.0 pp requires a C-6 trade, which the writer did not take.
- **Byte-untouched criteria must not name directories held by concurrent spokes.** `figure-render-pass` was
  writing into `papers/neurreps-variants/v2/figs/` throughout this pass by its own commission; the reframe's
  build note records the same class of incident one pass earlier. Recommend: off-tree PDF backups before any
  pass works near a variant, and a mechanical pre-flight check for in-flight spokes sharing a path.
- **V2's gate list is otherwise unchanged** by this pass: the Head's read · the colleague's S1 sign-off
  (camera-ready) · the true-venue-template re-measure · the hash-pinned figure follow-up (BUILD-NOTE §10).
