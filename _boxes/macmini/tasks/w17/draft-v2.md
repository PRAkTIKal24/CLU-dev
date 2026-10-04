# w17 spoke/draft-v2 — WRITE THE PAPER
**Read `COMMON.md` first.** Slug `draft-v2`. Agent type: paper-writer.
⭐ **Gate: start when `~/clu/outputs/w17-paper/VENUE.md` exists** (venue-scout's template is what you compile against). If it is not there after a reasonable wait, start on the generic ICLR template and say so in `LOG.md`.

## The one job
ONE `.tex` file — `~/clu/outputs/w17-paper/DRAFT-v2.tex` — on the venue template, **anonymised**.

## ⭐⭐ FIRST read `AMENDMENT-1.md` — it OVERRIDES this file on what may be written.

## The spine — Head-ruled 2026-09-22. Follow this order.
1. **Positioning.** Compositional generation is less crowded; **compositional + exact-deletion generation is unoccupied.** Use the counted figure from the positioning scout. ⭐ Safe phrasing: *"the intersection with exact deletion is empty."*
2. **Why it matters.** World models need **rule-valid** (= valid under an exact judge), **compositional**, **exactly deletable** generation.
3. **The Naive CLU = the 1× single design on Shapes.** All rows: rows 3–4 at **K\***; the ladder **with the one-line inference note**; deletion (value / conjunction / item); post-deletion refusal via the hole; the **composition-only store at ~1/7×** as the "one property" ablation; the **item store** as the other ablation; **encoder-adapt anchor** as a minor paragraph; the **BAR D decomposition paragraph** (unfreezing hurts / swap loss repairs).
4. **The same properties on the maze.** ⭐ **PLACEHOLDER tables and figures for wave 16's rows** (composition, rollout validity, deletion) — **write them as if they land**, clearly marked `\PLACEHOLDER{...}` so wave 18 can fill them. PLUS the already-banked **deletion transfer on CMAZE-2b** and the **verb finding** — the latter as an **INTERFACE claim**, with the lookup **0.97 printed honestly** as the within-maze floor that cannot transfer across mazes.
5. **Limitations of the naive unit → the leaving note.** The physics-CLU; the clue (zero-dynamics limit measured bit-identical); the laws + validations from the shorts as **this paper's own theory** (the shorts are non-archival, so their content enters here). Use the class name **"Latent Dynamics Model"**. Quote the **store-level exactness scoping**.
6. **Follow-up** = full physics-CLU evaluation.

## Hard requirements
- **Every number carries a LaTeX comment with its artefact path.** ⛔ No exceptions.
- Define a `\PLACEHOLDER{}` macro so wave 18 can find every one mechanically. **List them all in your `FINDINGS.md`.**
- **Appendix-maximalism:** everything not in main text goes to the appendix. ⛔ Nothing omitted, nothing pruned.
- ⛔ No `\cite{}` commands — **cite-tagger inserts those after you finish.** Write the prose so a citation can be slotted in.
- Compile it. If LaTeX is not yet installed on this machine, write the `.tex` anyway and record in `LOG.md` that compilation is pending.

## Acceptance criterion
`DRAFT-v2.tex` exists, is anonymised, follows the six-part spine in order, every number has its artefact comment, every wave-16 gap is a `\PLACEHOLDER{}`, and `FINDINGS.md` lists every placeholder and every `\TODO`.

## Files you own
`DRAFT-v2.tex` only. ⛔ Not the style version, not the bib, not the figures.
