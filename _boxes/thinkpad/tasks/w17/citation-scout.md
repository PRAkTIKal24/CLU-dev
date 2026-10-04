# w17 spoke/citation-scout — FIND EVERY PAPER THE DRAFT MUST CITE
**Read `COMMON.md` first.** Slug `citation-scout`. Agent type: web-scout.
⭐ Runs in parallel with draft-v2. cite-tagger consumes your output.

## The one job
For **every claim, method, baseline, dataset, metric and positioning sentence** the draft makes, find the papers to cite. Write `~/clu/outputs/w17-paper/CITATIONS.md`.

## Seed list (the novelty scout's — start here, do not stop here)
TensorFaces · Tenenbaum–Freeman · MLC-VAE · PandA · Cao–Yang · Hall–Marshall–Martin · exact-unlearning ridge-on-frozen · SISA · Guo et al. certified removal · the memory/retrieval-augmented precedent for parameter matching · world-model literature · compositional-generalisation literature · the LDM-class positioning papers from the positioning scout (`~/clu/evidence/outputs/w12-positioning-scout/`) · **3D Shapes** · **MiniGrid** · **Montero et al.**

## Per-entry format — all of it, every entry
`cite key` · title · first author · year · **arXiv ID** · **DOI** (or the literal string `no DOI`) · one line on **what in our draft it supports**.

## ⛔ Verification
⛔ **Do not invent an arXiv ID or a DOI.** Every ID must be one you actually resolved. If you cannot verify one, write `no DOI` / `no arXiv` — ⛔ never a plausible-looking guess. A fabricated identifier is worse than a missing one; it will be caught at camera-ready and it corrupts the bib.

## Acceptance criterion
`CITATIONS.md` with every entry complete, each ID verified, grouped by the draft section it serves, and a final list of **claims for which you could NOT find a citation** — that list is as important as the citations.

## Files you own
`CITATIONS.md`. ⛔ Nothing else. ⛔ You do NOT write `references.bib` — cite-tagger does.
