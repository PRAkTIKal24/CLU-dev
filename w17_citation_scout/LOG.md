# w17 citation-scout LOG
2026-09-22T21:48:28Z start; read AMENDMENT-1, COMMON, citation-scout.md
2026-09-22T21:49:24Z read POSITIONING.md, NOVELTY.md, DRAFT-v1 §1-2,§7-10, gen-lace-spec; testing arXiv/Crossref API reachability
2026-09-22T21:51:30Z arXiv API: 70/70 IDs resolved w/ expected titles (arxiv_results.jsonl). Crossref: 21/22 DOIs resolved; TensorFaces guess 10.1007/3-540-47969-4_29 is WRONG (resolves to Speyer 'Pencil of Lines') -> excluded; searching
2026-09-22T21:53:39Z verification complete: 76 arXiv IDs via export.arxiv.org API, ~32 DOIs via api.crossref.org; writing CITATIONS.md
2026-09-22T21:58:04Z CITATIONS.md written to w17-paper/ (97 keys, 76 arXiv, 48 DOI all re-verified in final pass, 12 uncited claims). Removed w14 figures from §4 note. Snapshot copied here.
## DECLARED NOT-DONE
- Dedicated search for "deletion/unlearning as a world-model requirement" (U-2): not run; deletion leg left as our motivation.
- Survey of benchmarks scoring composition+deletion on one object (U-5): not run (inherits POSITIONING NR-7).
- Harness inspection for optimiser / CI procedure / p-value test (U-7): not done; Adam + Efron marked conditional.
- CMAZE-on-MiniGrid provenance (U-6): not checked.
- LeCun 2022 + Montero 2021 OpenReview records: openreview.net 403 ChallengeRequiredError; LeCun unverified (do-not-insert), Montero forum id from search hit only.
- Pre-2022 NeurIPS / ICLR / ICML DOIs: not swept (generally none exist); marked `no DOI`.
- references.bib: not written (cite-tagger owns it).
2026-09-22T21:58:17Z END. committing + bundling
