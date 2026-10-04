# pw1-draft-wording — plan items 7, 8, 13 in BOTH drafts
Session `pw1-draft-wording` · worktree `~/clu/worktrees/CHLU-pw1-draft-wording` · branch `pw1/draft-wording` (from `pw1/draft-figures` a01bea7; the draft checkers are in `pw1_draft_sweeps/`, `pw1_draft_cuts/`) · out `~/clu/outputs/pw1-draft-wording/`
**You hold the `.tex` lock.** Read `COMMON.md`, then PLAN.md §1 items 7, 8, 13 in full, and the previous draft spokes' FINDINGS (`~/clu/outputs/pw1-{draft-sweeps,draft-cuts,draft-figures}/FINDINGS.md`).

## The one job
- **Item 7:** the seed-5 limitation sentence, the relational-coding-limit sentence, and scoping of every "every number verified / traced" sentence. Fix the two `% SRC` comments (`0.6407`; "16 of 16"): find the right file, and verify the value against it verbatim. Keep the PoE multiple at 27.135$\times$. Every number you write is found verbatim in the artefact you cite: print value → file → JSON path.
- **Item 8:** no seed is called hard or easy anywhere (text, tables, captions, appendix splits). Seeds are printed together. The CMAZE-T paragraph keeps only the mechanism, with no ceiling number.
- **Item 13:** retire "exact / exactly / exactness" in the deletion sense everywhere, titles included. Use the plain phrasing and the scoping sentence. Write no general post-deletion-composition claim. Print the conjunction loss and the address-book refit cost as limitations. Add the future-work sentence with the precise empirical comparison (above every retrained baseline on value and item scopes; level with MLP and LSTM on red sphere).
- The abstract's maze `\TODO` stays byte-identical. Other abstract sentences change only as items 7, 8 and 13 require.
- Keep the vocabulary of items 2–3.

## Acceptance (printed in FINDINGS)
- A before→after table for every sentence changed, with counts per item.
- Residual scans of rendered text: `exact` (every surviving use listed with its non-deletion sense), `hard seed`, `easy seed`, `hard`, `easy` (each residual listed), and `ceiling`.
- Both drafts: exit 0, 0 undefined citations/references, style diff-check EMPTY, provenance identical or improved, `\PLACEHOLDER` / `\TODO` lists. Page counts.
- Installed byte-identically.
**What would prove this wrong:** a number in a new sentence that is not found verbatim in its source, or a surviving deletion-sense "exact".

## Hub notes (binding)
- Known "exact" sites (from draft-figures): Sec. 3.3's title, "The probe that bounds the claim" ("Exactness is n-CLU-scoped"), the CMAZE-T "Exact deletion holds …" sentence, Tab. cmazet's "n-CLU-scoped exactness", Tab. impl's "Exact (n-CLU-scoped)", and the paper TITLE "naive-CLU: Exactly Deletable Compositional Generation on a Frozen Encoder". Item 13 says "title included". Rewrite the title without "Exactly", print before → after, and flag it for the Head in FINDINGS.
- Deletion numbers: the measured bounds are content ≤ 5.38e-07 and book rendered-feature ≤ 7.81e-08, with the statistic 0. Never write "1e-8".
- A parallel spoke (`pw1-visuals-tooling`) edits only `paper-visuals/` and `figures/` scripts, never the drafts. Do not touch figure files.
