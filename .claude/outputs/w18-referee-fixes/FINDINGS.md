# FINDINGS — w18 referee-fixes (full report: REFEREE-FIXES.md)

- APPLIED to both drafts: M2, M4, M5, M6, M7(e). M7(c) was partly applied: SISA is cited, and the CDM bib entry is handed to cite-tagger.
- **HELD, not applied: M1** (the Table 1 rivals' fit-pool asymmetry, and whether "naive vs naive" survives). It changes the headline claim, so the Head and the Advisor decide it.
- **HELD, not applied: M3** (whether exactness is rescoped off the address book). It changes what "exact deletion" claims, so the Head and the Advisor decide it.
- DECLARED NOT-DONE:
  - M7(a) AI statement: the authors' facts.
  - M7(b) 9-page limit: the Head's trim. The main text now ends at p.13 (v2) / p.14 (style).
  - M7(d) deadline: the Head's decision.
  - M7(f) zero PLACEHOLDER/TODO: placeholder-fill and the Head.
  - CDM bib entry: cite-tagger.
  - Split request-layer figure: the figures owner. The combined figure is dropped.
  - Causal explanation of the M2 gap: no artefact isolates it, so it stays a `\TODO`.
  - Visual PDF inspection: no rasteriser on this machine.
- Invariants: the four style diffs are ALL EMPTY on the installed files (artefact comments 287 = 287 in identical order, up from 265 by design). Both drafts compile: exit 0, 35 / 36 pages, 0 undefined references.
- Finding for the Head: against the strongest learned rival available, the store's lead is far smaller than Table 1's +39 to +49. Example: ladder n=2, seed 2 is 33.633 against 24.2500. That stronger rival set was fitted on a pool that contains the held-out values.
- Nothing depended on w15/w16. No w15/w16 number entered either draft. No placeholder argument was altered; the arguments were only reordered by the cmazet/cmazeo split.
