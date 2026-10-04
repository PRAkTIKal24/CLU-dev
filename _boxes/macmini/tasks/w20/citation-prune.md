# w20 spoke/citation-prune — 96 REFERENCES DOWN TO ≤ 55
**Agent type: web-scout + paper-writer.** Zero compute, zero scored runs. ⭐ PRE-AUTHORISED.
⛔⛔ **EXCLUSIVE `.tex` LOCK, and you go AFTER `referee3-fixes`** — it rewrites the positioning and removes "the intersection is empty", which changes what must be cited. ⛔ If any `w20-referee3-fixes`, `w18-placeholder-fill` or `w19-draft-figure-slots` session is live, **STOP and say so**.

## The one job
Cut `references.bib` / `CITATIONS.md` from **96** entries to **≤ 55**.

### KEEP — the prior art a referee will name
TensorFaces · Tenenbaum–Freeman · MLC-VAE · PandA · Cao–Yang · **the decremental-PCA line** · **exact unlearning on frozen foundation models** · **SISA** · **certified removal** · **compositional diffusion**.
Plus: **the baselines and their sources** · **the datasets and the hold-out protocol** · **the parameter-matching precedent** (retrieval / memory-augmented models) · **the world-model and compositional-generalisation anchors** · **the LDM-class positioning papers** · **the three shorts' key physics references**.

### DROP
⭐ **Everything cited once for colour.**

## Then
- **Re-tag BOTH `.tex` files.**
- **Regenerate `references.bib`.**
- **Regenerate BOTH Zotero blocks** in `CITATIONS-FOR-ZOTERO.md` — one arXiv ID per line, one DOI per line, each a single fenced block, with the omitted counts stated beneath.
- ⭐ **Prose byte-identical after stripping cites** — prove it, as the w17 cite-tagger did (`PROSE-CHECK.txt` is the template).
- ⭐ **Report the dropped list** — every one of the ~41 removed entries, with one line on why.

## Invariants
⭐ Both drafts compile with **0 undefined citations**. ⚠ bibtex silently never succeeded before wave 19 — **check the count, do not assume**. Every surviving `\cite{}` must resolve.
⭐ Four style diffs EMPTY.
⛔ **Do not fabricate an arXiv ID or DOI.** If an entry you keep lacks one, mark it `no DOI` / `no arXiv` — ⛔ never a plausible-looking guess.

## Acceptance
`CITATION-PRUNE.md`: the kept list (≤ 55) and the dropped list with reasons; regenerated `references.bib` and both Zotero blocks; the prose byte-identity proof; both compile logs showing **0 undefined citations**; four diffs EMPTY.

## What would prove this wrong
⭐ If the keep-list cannot get under 55 without dropping something a referee would name, **say so and stop at the smallest defensible number** — ⛔ do not cut a load-bearing citation to hit a target.
