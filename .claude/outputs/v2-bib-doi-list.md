# v2-bib-doi-list — web-scout report

Task + acceptance criterion: assemble the Zenodo-pasteable identifier block for V2's `refs.bib`, plus coverage table, residual list, orphan list, and ranked candidates for the three open `\TODO` citation sites. **Zero guessed identifiers.**
Status: **done**
**DIAL DECLARATION (echoed): none — instrument/recon. No performance claim, no laundering control, no falsification axis applies.**

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — first-10-lines rule, AGENT_PROTOCOL §5)
> 1. ⛔⛔ **THE `.bib` MUST BE TYPESET WITH A NUMERIC CITATION STYLE, NOT AUTHOR–YEAR.** Charter Add.51 says arXiv:2605.03338 "appears only as a citation" in the paper — its author must not be named. If the Head generates `refs.bib` and uses a natbib author–year style, the three `\cite{}` sites (l.48, l.86, l.96) render as **"Mo (2026)"** and the constraint breaks *mechanically*, silently, at build time. The Head's own figure TODO at l.95 ("change 'published' to **the citation number**") already presupposes a numeric style — but nothing in the file enforces it. **Owner: whoever wires the `.bib`.** Same argument applies to the `Anonymous (2026)` theory-note entry.
> 2. ⚠ **Dinc et al. is no longer a 2025 preprint.** arXiv:2501.02378 now carries a journal-ref: **Phys. Rev. X 16(2), 021058 (2026), DOI 10.1103/mjcl-lb4x.** The block gives the DOI (per the DOI-preferred rule), so Zenodo will emit a **2026 PRX** entry against an in-text "Dinc et al. 2025" (l.57). **Head must rule: update the in-text year to 2026, or drop back to `arXiv:2501.02378`.** ⛔ I did not silently correct it.
> 3. ⚠ **9 reference entries are cited nowhere in the body** (Deliverable 4). A referee has already flagged decorative entries as a credibility cost. **Cite-or-cut must be ruled BEFORE the `.bib` is built.**
> 4. ℹ **The task file's premise "Minami & Hidaka 2018 is in the list" is FALSE.** It is absent from `pj_sub.tex`, from `submission.tex`, and from `v2-cite-check.md`'s 26 verified works (which explicitly logs "Minami–Hidaka … not present in the draft"). **The program currently cites no Nambu–Goldstone reference at all** — site 2 needs a genuinely new entry, not a re-use.
> 5. ℹ **Entry count is 50, not 52.** Enumerated from `pj_sub.tex` ll.122–221.

---

## 0. File identity (⚠ md5 not computable)

⛔ **I have no shell in this session** — my toolset is Read / Grep / Glob / WebSearch / WebFetch / Write only. **I cannot compute an md5.** Declaring that rather than inventing a hash. Fingerprint of the file as I read it:

| field | value |
|---|---|
| path | `.claude/NIPSsubmission/v2-neurreps/pj_sub.tex` |
| read at | **2026-08-24**, single read, full file |
| total lines | **433** |
| l.1 / l.433 | `\documentclass{article}` / `\end{document}` |
| reference block | `\section*{References}` at **l.118**, `\endgroup` at **l.222** |
| entries | **50** (ll.122–221, blank-line separated; a stray blank paragraph sits at l.209–210 between Seung and Tieleman) |
| `\TODO` cite-sites | all three on **l.33**; self-cite TODO l.35; estimator TODOs l.48, l.86, l.96 |
| ⛔ writes by me | **none** to any paper file. Only `.claude/outputs/v2-bib-doi-list.md`. |

Retrieval date for every NEW identifier below: **2026-08-24**.

---

## Deliverable 1 — ⭐ THE BLOCK

46 identifiers, one per line, in the reference list's own order. The 4 entries with no resolvable identifier are **omitted** (Deliverable 3), so this block is **not** positionally aligned with the list — use Deliverable 2 to map back.

```
10.1371/journal.pcbi.1010716
arXiv:2202.02164
arXiv:2501.00663
10.1073/pnas.1117386109
arXiv:1904.10278
arXiv:2511.04802
10.1103/mjcl-lb4x
10.1016/j.cell.2017.07.021
10.1007/978-3-642-15825-4_26
10.1038/s41586-021-04268-7
10.1007/978-1-4612-4574-2
arXiv:1410.5401
10.1038/nature20101
arXiv:2008.07669
10.1007/3-540-30666-8
arXiv:2502.07256
10.1126/science.7761831
10.1162/neco.1997.9.8.1735
arXiv:2410.03972
arXiv:2605.14685
arXiv:2603.01768
arXiv:2402.01032
arXiv:2205.09829
arXiv:2510.24965
arXiv:2507.14793
10.1038/s41583-022-00642-0
10.1126/science.aal4835
10.1038/s41467-024-49190-4
10.1137/0521081
arXiv:2601.01075
10.1016/S0393-0440(01)00020-1
arXiv:2605.03338
10.1609/aaai.v34i04.5973
arXiv:2008.02217
10.1016/S0896-6273(03)00255-1
10.3934/dcds.2001.7.91
arXiv:2010.00951
arXiv:2103.05487
arXiv:2110.04744
arXiv:2408.00109
10.1073/pnas.93.23.13339
10.1145/1390156.1390290
10.7554/eLife.69841
arXiv:2504.12429
arXiv:2212.13285
arXiv:2210.02684
```

**Format notes for the Head:** DOIs are case-insensitive; `10.1016/S0393-0440(01)00020-1` and `10.1016/S0896-6273(03)00255-1` are given in publisher (upper-case `S`) form, which is what `submission.tex` carries and what resolves. `10.1007/978-3-642-15825-4_26` contains a literal underscore — keep it, do not escape it in the identifier field.

---

## Deliverable 2 — coverage table

`S2` = `submission.tex` bibliography (DOIs `pj_sub` stripped) · `CC` = `v2-cite-check.md` · `NEW` = verified by me this pass against a primary/registry record.

| # | line | author-year short form | identifier in block | C/N | source that established it |
|---|---|---|---|---|---|
| 1 | 122 | Aitken et al. 2022 | `10.1371/journal.pcbi.1010716` | **CARRIED** (S2) + re-verified | Crossref record: PLOS Comput Biol 18(11):e1010716 ✔ |
| 2 | 124 | Akhtiamov & Thomson 2023 | — | — | **RESIDUAL** (D3-1) |
| 3 | 126 | Anonymous 2026 (theory note) | — | — | **RESIDUAL** (D3-2) |
| 4 | 128 | Aslan, Platt & Sheard 2023 | `arXiv:2202.02164` | **NEW** | arXiv API exact-title hit; PMLR v197:181–218 confirmed on proceedings.mlr.press/v197 |
| 5 | 130 | Behrouz, Zhong & Mirrokni 2025 | `arXiv:2501.00663` | **CARRIED** (CC #6) + arXiv API re-resolve | NeurIPS 2025; no DOI |
| 6 | 132 | Burak & Fiete 2012 | `10.1073/pnas.1117386109` | **CARRIED** (S2) + re-verified | Crossref: PNAS 109(43):17645–17650 ✔ |
| 7 | 134 | Csordás & Schmidhuber 2019 | `arXiv:1904.10278` | **CARRIED** (CC #7) | ICLR 2019; no DOI |
| 8 | 136 | Di Bernardo et al. 2025 | `arXiv:2511.04802` | **CARRIED** (CC #16) + arXiv API re-resolve | v2 46pp; **no journal-ref/DOI** as of today |
| 9 | 138 | Dinc et al. 2025 | `10.1103/mjcl-lb4x` | **NEW ⚠** | arXiv:2501.02378 journal-ref → Crossref: **Phys. Rev. X 16(2):021058, 2026**. See reconciliation item 2 |
| 10 | 140 | Dönmez 2024 | — | — | **RESIDUAL** (D3-3) |
| 11 | 142 | Driscoll et al. 2017 | `10.1016/j.cell.2017.07.021` | **NEW** | Crossref: Cell 170(5):986–999.e16 ✔ (S2 carried no DOI for this one) |
| 12 | 144 | Fischer & Igel 2010 | `10.1007/978-3-642-15825-4_26` | **CARRIED** (S2 + CC #21) | ⛔ trap preserved: **2010** ICANN is the landscape/divergence result, *not* F&I 2011 |
| 13 | 146 | Gardner et al. 2022 | `10.1038/s41586-021-04268-7` | **CARRIED** (S2) + re-verified | Crossref: Nature 602(7895):123–128 ✔ · ⚠ **ORPHAN** |
| 14 | 148 | Golubitsky, Stewart & Schaeffer 1988 | `10.1007/978-1-4612-4574-2` | **CARRIED** (S2 + CC #13) | ⛔ trap: **Vol. II** (3 authors, 1988); Vol. I (1985) is 2 authors |
| 15 | 150 | Graves, Wayne & Danihelka 2014 | `arXiv:1410.5401` | **CARRIED** (CC #23) | never formally published — arXiv is the only record |
| 16 | 152 | Graves et al. 2016 | `10.1038/nature20101` | **CARRIED** (S2 + CC #24) | Nature 538(7626):471–476 |
| 17 | 154 | Gu et al. 2020 (HiPPO) | `arXiv:2008.07669` | **CARRIED** (CC #3) | NeurIPS 2020; no DOI |
| 18 | 156 | Hairer, Lubich & Wanner 2006 | `10.1007/3-540-30666-8` | **NEW** | Crossref book record: *Geometric Numerical Integration*, Springer 2006, SSCM 31, ISBN 3540306633 ✔ · ⚠ **ORPHAN** · ⛔ trap preserved: cite for leapfrog/`h<2`/Ch. XII only — **conformal symplecticity is #34** |
| 19 | 158 | Haputhanthri et al. 2025 | `arXiv:2502.07256` | **NEW** | arXiv API: 10 authors, 2025-02-11, **no journal-ref/DOI** |
| 20 | 160 | Hinton et al. 1995 | `10.1126/science.7761831` | **CARRIED** (S2 + CC #19) | ⛔ trap: pages **1158–1161** (author's page says 1158–1160) |
| 21 | 162 | Hochreiter & Schmidhuber 1997 | `10.1162/neco.1997.9.8.1735` | **CARRIED** (S2 + CC #12) | Neural Computation 9(8):1735–1780 |
| 22 | 164 | Huang et al. 2025 | `arXiv:2410.03972` | **NEW** | arXiv API v3, journal-ref "Advances in NeurIPS (2025)"; no DOI · ⚠ **ORPHAN** |
| 23 | 166 | Iqbal et al. 2026 | `arXiv:2605.14685` | **CARRIED** (CC #8) + arXiv API re-resolve | ⛔ trap: arXiv order **Iqbal, Keller, Song, Miyato, Welling** — never the tweet order. No DOI |
| 24 | 168 | Jawahar & Pierini 2026 | `arXiv:2603.01768` | **CARRIED** (charter Add.23 / CC #26) + arXiv API re-resolve | v2; comments verbatim *"Accepted as a short paper at ICLR 2026 (AI & PDE)"*; **no DOI on the arXiv record** |
| 25 | 170 | Jelassi et al. 2024 | `arXiv:2402.01032` | **CARRIED** (CC #4) | PMLR 235:21502–21521; PMLR issues no DOIs · ⚠ **ORPHAN** |
| 26 | 172 | Jude et al. 2023 | `arXiv:2205.09829` | **NEW** | arXiv API exact-title hit; PMLR v197:234–257 confirmed |
| 27 | 174 | Karuvally et al. 2025 (EDEN) | `arXiv:2510.24965` | **CARRIED** (CC #5) + arXiv API re-resolve | ⛔ trap: distinct from the 2019 "EDEN" DRAM paper (1910.05340) · ⚠ **ORPHAN** |
| 28 | 176 | Keller 2025 | `arXiv:2507.14793` | **CARRIED** (CC #17) + arXiv API re-resolve | comments "NeurIPS '25, Spotlight"; **single author** |
| 29 | 178 | Khona & Fiete 2022 | `10.1038/s41583-022-00642-0` | **CARRIED** (S2) + re-verified | Crossref: Nat Rev Neurosci 23(12):744–766 ✔ · ⚠ **ORPHAN** |
| 30 | 180 | Kim et al. 2017 | `10.1126/science.aal4835` | **CARRIED** (S2) + re-verified | Crossref: Science 356(6340):849–853 ✔ · ⚠ **ORPHAN** |
| 31 | 182 | Kong, Brewer & Lai 2024 | `10.1038/s41467-024-49190-4` | **CARRIED** (S2 + CC #2) | Nat Commun 15:4840 |
| 32 | 184 | Krupa 1990 | `10.1137/0521081` | **CARRIED** (S2 + CC #14) | SIAM J Math Anal 21(6):1453–1486 |
| 33 | 186 | Lillemark et al. 2026 | `arXiv:2601.01075` | **NEW** | arXiv API v2, comments **"Accepted at ICML 2026"** (entry currently says preprint-only — see D5 flags); no DOI |
| 34 | 188 | McLachlan & Perlmutter 2001 | `10.1016/S0393-0440(01)00020-1` | **NEW** | Crossref: J Geom Phys 39(4):276–300 ✔; corroborated by ScienceDirect pii `S0393044001000201`. ⛔ trap preserved: **this** is conformal symplecticity, not #18 |
| 35 | 190 | Mo 2026 | `arXiv:2605.03338` | **CARRIED** (CC #1) + arXiv API re-resolve | ⭐ **task question answered: NO DOI and NO journal-ref exist as of 2026-08-24** — still an unrefereed preprint, single author |
| 36 | 192 | Nijkamp et al. 2020 | `10.1609/aaai.v34i04.5973` | **NEW** | AAAI OJS publisher page: AAAI 34(4):5272–5280 ✔ (two-instrument: OJS page + search record) |
| 37 | 194 | Ramsauer et al. 2021 | `arXiv:2008.02217` | **CARRIED** (CC #25) | ICLR 2021; no DOI · ⚠ **ORPHAN** |
| 38 | 196 | Renart, Song & Wang 2003 | `10.1016/S0896-6273(03)00255-1` | **CARRIED** (S2) + re-verified | Crossref: Neuron 38(3):473–485 ✔ |
| 39 | 198 | Rumberger 2001 | `10.3934/dcds.2001.7.91` | **CARRIED** (S2 + CC #15) | DCDS 7(1):91–113 |
| 40 | 200 | Rusch & Mishra 2021a (coRNN) | `arXiv:2010.00951` | **CARRIED** (CC #9) | ⛔ trap: **a/b split is mandatory** |
| 41 | 202 | Rusch & Mishra 2021b (UnICORNN) | `arXiv:2103.05487` | **CARRIED** (CC #10) | PMLR 139:9168–9178, no DOI · ⚠ **ORPHAN** |
| 42 | 204 | Rusch et al. 2022 (LEM) | `arXiv:2110.04744` | **CARRIED** (CC #11) | ICLR 2022; no DOI |
| 43 | 206 | Ságodi et al. 2024 | `arXiv:2408.00109` | **NEW** | arXiv API v3, journal-ref "Proceedings of NeurIPS 2024"; no DOI |
| 44 | 208 | Seung 1996 | `10.1073/pnas.93.23.13339` | **CARRIED** (S2) + re-verified | Crossref: PNAS 93(23):13339–13344 ✔ |
| 45 | 211 | Tieleman 2008 | `10.1145/1390156.1390290` | **CARRIED** (S2 + CC #20) | ⛔ trap: title contains no "persistent contrastive divergence"; ≠ Tieleman & Hinton 2009 |
| 46 | 213 | Vafidis et al. 2022 | `10.7554/eLife.69841` | **CARRIED** (S2) + re-verified | Crossref ✔ · ⚠ **title truncated in `pj_sub`** — see D3 flags |
| 47 | 215 | van der Ouderaa & van der Wilk 2023 | — | — | **RESIDUAL** (D3-4) |
| 48 | 217 | Vastola 2024 | `arXiv:2504.12429` | **NEW** | arXiv API; comments *"Accepted to the NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations"*, journal-ref names PMLR v228 ⇒ **same work**, despite the 2025 posting date. Pages 425–442 confirmed on proceedings.mlr.press/v228 |
| 49 | 219 | Wang & Ponce 2023 | `arXiv:2212.13285` | **NEW** | arXiv API exact-title hit; PMLR v197:278–300 confirmed |
| 50 | 221 | Xu et al. 2023 | `arXiv:2210.02684` | **NEW** | arXiv API exact-title hit; PMLR v197:370–387 confirmed |

**Tally:** 46 in block (28 CARRIED, 18 NEW) · 4 residual · 0 guessed.

---

## Deliverable 3 — the residual list (hand-entry required)

**None of these has a DOI or an arXiv ID.** PMLR does not mint DOIs, and the three PMLR items below have no arXiv preprint (checked by exact-title arXiv API query **and** by author-name arXiv API query **and** by Crossref bibliographic query — all three surfaces negative, per-item).

**D3-1 — Akhtiamov & Thomson 2023.** Workshop paper, no preprint.
```
author  = Danil Akhtiamov and Matt Thomson
title   = Connectedness of loss landscapes via the lens of Morse theory
booktitle = Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations (NeurReps)
series  = PMLR ; volume = 197 ; pages = 171--181 ; year = 2023
editors = Sanborn, Shewmake, Azeglio, Di Bernardo, Miolane  (vol. published 2023-02-07; workshop 2022-12-03)
url     = https://proceedings.mlr.press/v197/akhtiamov23a.html   ⚠ URL pattern inferred from PMLR convention, NOT clicked — verify before use
```
*Tried:* arXiv title search (0 results), arXiv `au:Akhtiamov` (17 papers enumerated, **none on loss landscapes or Morse theory**), Crossref bibliographic (no match).

**D3-2 — Anonymous 2026, the theory note.** ⛔ **No identifier exists or should exist** (double-blind supplementary companion). Per `v2-cite-check` §27: the entry must carry `note={Anonymous companion note, provided in the supplementary material}` and **must not resolve to a named preprint**.
```
author = Anonymous ; year = 2026
title  = The theory note: an exactly-solvable theory of retention, latching and forgetting for damped symplectic recurrences
note   = Provided in the supplementary material.
```

**D3-3 — Dönmez 2024.** ⭐ `pj_sub` gives **no page range**; the volume does.
```
author  = Arif Dönmez
title   = Discovering latent causes and memory modification: A computational approach using symmetry and geometry
booktitle = Proceedings of the 2nd NeurIPS Workshop on Symmetry and Geometry in Neural Representations (NeurReps)
series  = PMLR ; volume = 228 ; pages = 443--458 ; year = 2024
(vol. published 2024-08-02; workshop 2023-12-16)
```
*Tried:* arXiv title search (0), Crossref bibliographic (no match).

**D3-4 — van der Ouderaa & van der Wilk 2023.**
```
author  = Tycho F. A. van der Ouderaa and Mark van der Wilk
title   = Sparse Convolutions on Lie Groups
booktitle = Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations (NeurReps)
series  = PMLR ; volume = 197 ; pages = 48--62 ; year = 2023
```
*Tried:* arXiv title search (0), arXiv `au:"van der Ouderaa"` (**13 papers enumerated, this title absent**), Crossref bibliographic (no match).

### ⚠ Content problems found in the reference list (flagged, NOT silently corrected)

| entry | problem |
|---|---|
| **Dinc et al. (l.138)** | Listed as a 2025 preprint. It is now **Phys. Rev. X 16(2):021058 (2026)**. Year *and* venue are stale. Crossref also renders the title without the leading "A": *"Ghost Mechanism: An Analytical Model…"*. |
| **Vafidis et al. (l.213)** | Title **truncated**: `pj_sub` says *"Learning accurate path integration in ring attractor models."* The published title is *"…in ring attractor models **of the head direction system**"* (verified Crossref; `submission.tex` has it correct). ⇒ **a regression `pj_sub` introduced against its own base file.** |
| **Karuvally et al. (l.174)** | `pj_sub` renders the title as *"Exponential dynamic energy network for high-capacity sequence memory (EDEN)"*. The arXiv title is *"Exponential Dynamic Energy Network for High Capacity Sequence Memory"* — no parenthetical "(EDEN)", no hyphen in "High Capacity". Cosmetic, but Zenodo will emit the real title and the mismatch will be visible. |
| **Lillemark et al. (l.186)** | Listed as bare arXiv. The arXiv record's comments field now reads **"Accepted at ICML 2026"**. (`audience-refresh` F4 logged three title variants; the current v2 title matches `pj_sub`'s.) |
| **Khona & Fiete (l.178)** | `pj_sub` gives *"23:744–766"* with no issue. Crossref has **23(12)**. Harmless; Zenodo will fill it. |
| **GMOR (l.413, App. G title, l.417)** | ⛔ **The Gell-Mann–Oakes–Renner relation is named and used as a load-bearing structural analogy — and there is NO reference entry for it.** Not a bib-format issue; a missing citation. Head should rule whether App. G needs Gell-Mann, Oakes & Renner (1968), *Phys. Rev.* **175**:2195, `10.1103/PhysRev.175.2195` ⚠ (this DOI is from my general knowledge and was **NOT verified this pass** — do not use until checked). |
| **Task-file premise** | The task warns to expect residuals among *Snyder 1977, Micciancio 1997, Naor & Teague 2001, Seung 1996, Rumberger 2001*. **Snyder, Micciancio and Naor & Teague appear nowhere in this paper** (positive-controlled grep: `Rumberger` and `Seung` return hits; `Snyder`/`Micciancio`/`Naor` return none). That part of the task brief is carried over from a different paper. Seung and Rumberger both resolved to DOIs — not residuals. |

---

## Deliverable 4 — ⭐ orphan / dangling list

**Method:** per-file `Grep` on `pj_sub.tex` with `-n -o`, surname alternation over all 50 entries, then a second targeted pass on system names (`Hopfield`, `EDEN`, `UnICORNN`, `HiPPO`, `copying`, `grid cell`, `ring attractor`, `degeneracy`, `2605.03338`, `Lubich`, `Wanner`). **Positive controls fired** (e.g. `Golubitsky`→l.55 body *and* l.148 entry; `Ouderaa`→l.215 entry *and* l.232 body), so the negatives below are real, not a directory-grep artefact.

### (a) Cited NOWHERE in the body — 9 entries. **Head must rule cite-or-cut before the `.bib` is built.**

| entry | line | nearest place it *could* attach |
|---|---|---|
| **Gardner et al. 2022** (toroidal grid-cell topology) | 146 | §2 l.55–57 continuous-attractor substrate; App. A l.232 (Xu et al. grid-cell sentence) |
| **Hairer, Lubich & Wanner 2006** | 156 | §3 l.60 (leapfrog stability, the `εμ<2` band) — this is exactly what HLW is *for*, and it is the trap-safe use |
| **Huang et al. 2025** (solution degeneracy) | 164 | ⭐ **§5 Discussion l.116 literally says "analysing solution degeneracy between the designed and emergent arms" with no cite.** `submission.tex` l.148 *did* cite it there. **This is a citation `pj_sub` dropped.** |
| **Jelassi et al. 2024** (copying/SSM limits) | 170 | App. C l.234 item (1) or App. E item (1), alongside HiPPO |
| **Karuvally et al. 2025 (EDEN)** | 174 | App. C l.234 (the associative-memory neighbours paragraph, next to Titans) |
| **Khona & Fiete 2022** (attractor/integrator review) | 178 | §2 l.55 first sentence — the natural review anchor |
| **Kim et al. 2017** (Drosophila ring attractor) | 180 | §2 l.55–57 |
| **Ramsauer et al. 2021** (modern Hopfield) | 194 | App. C l.234 item (4) / App. E item (2) |
| **Rusch & Mishra 2021b (UnICORNN)** | 202 | ⚠ **App. A l.232 names the baselines as LSTM / LEM (Rusch et al. 2022) / coRNN (2021a) — 2021b is never used.** Either cite it as the Hamiltonian-symplectic-RNN precedent in §3 (where it is genuinely the closest architectural neighbour) or cut it. Cutting also removes the a/b split hazard. |

⇒ **9/50 = 18 % of the bibliography is decorative as the file stands.** Six of the nine (Gardner, Khona, Kim, Jelassi, Karuvally, Ramsauer) are trivially attachable to sentences that already exist; two (Hairer, Huang) are *restorations* of uses that `submission.tex` had; one (UnICORNN) is the only genuine cut candidate.

### (b) In-text citations with no matching entry — **none**, but two structural hazards

- ⛔ **`arXiv:2605.03338` is written as a bare literal string in the body at l.90, l.230, l.298** (and in the fig. caption at l.298), *not* as a citation. The entry at l.190 exists but nothing links to it. Once `refs.bib` lands, those three literals should become `\cite{}` — **and that is exactly where reconciliation item 1 bites**: under an author–year style they become "Mo (2026)".
- **`Anonymous (2026)`** is referenced in prose at l.35 (*"developed in the supplementary Anonymous, 2026, hereafter the theory note"*) and has an entry at l.126. Not dangling, but it must not be auto-resolved by Zenodo.
- **`\TODO{cite CHLU}` (l.35)**, **`\cite{}\TODO{cite}` (l.48, 86, 96)**, and the three l.33 sites are the only unfilled hooks. All accounted for below.

---

## Deliverable 5 — candidates for the three open sites

⛔ **I recommend; the Head selects. Nothing below is presented as decided.**

### Site 1 — l.33 `\cite{}\TODO{cite welling and another SSB paper}`
> *"Symmetries supply these directions natively through spontaneous symmetry breaking."*

Two things are wanted: a Welling-lineage work, **and** a genuine SSB-in-neural-networks reference. ⭐ Note the venue context: `audience-refresh-2025-2026.md` §1.7 establishes that *"symmetry breaking"* is now **titular** NeurReps vocabulary (2025 spotlight #65), so a venue-native cite outranks a generic one.

| rank | candidate | what it actually claims | fit / cost |
|---|---|---|---|
| **1** | **Iqbal, Keller, Song, Miyato & Welling (2026)**, *"Spontaneous symmetry breaking and Goldstone modes for deep information propagation"*, arXiv:**2605.14685** | SSB of a continuous symmetry in deep/recurrent nets produces **gapless Goldstone degrees of freedom** that *"enable coherent signal propagation across depth and recurrent iterations … without relying on architectural stabilizers such as residual connections or normalization"*; explicitly notes the recurrent/long-term-memory value | ⭐ **Satisfies BOTH halves of the TODO in one cite, is Welling-authored, and is ALREADY entry #23 — zero new bib entries.** ⚠ It is also the program's nearest physics rival (CC #8 flags the referee risk); citing it *for the mechanism* at l.33 is the honest move and pre-empts the "you ignored your neighbour" hit. Also covers half of site 2 (Goldstone). |
| **2** | **Kaba & Ravanbakhsh (2023/24)**, *"Symmetry Breaking and Equivariant Neural Networks"*, arXiv:**2312.09016** (v2 2024-03-22) | Equivariant networks **cannot** break symmetry — their output is at least as symmetric as the input; proposes relaxing equivariance to permit SSB | ⭐ **Venue-native: the arXiv comments field literally reads "Symmetry and Geometry in Neural Representations" (= NeurReps).** This is the reference that establishes SSB as a *problem the equivariant-nets community owns*, which is precisely l.33's premise. New entry, no DOI. |
| **3** | **Goel, Lim, Lawrence, Jegelka & Huang (2026)**, *"Any-Subgroup Equivariant Networks via Symmetry Breaking"*, arXiv:**2603.19486**, ICLR 2026 (+ NeurReps 2025 spotlight #65) | Obtains subgroup equivariance from a permutation-equivariant base by feeding a **symmetry-breaking input whose automorphism group is the target subgroup**; relaxes exact→approximate breaking via 2-closure | Most venue-*current* and the source of the titular vocabulary. ⚠ **Weaker semantic fit:** it is symmetry breaking as an *engineering device for subgroup selection*, not SSB producing a flat/neutral direction. Use only as a third, "the room says this word" cite. |
| *alt* | **Cohen & Welling (2016)**, *"Group Equivariant Convolutional Networks"*, arXiv:**1602.07576**, ICML 2016 | The founding group-equivariant CNN | ⚠ **Only if the Head's "welling" meant the equivariance-lineage founder rather than the SSB paper.** It contains **no** SSB content, so it cannot be the SSB half. |

**My reading of the TODO:** #1 alone likely *is* what the Head means by "welling", in which case the missing piece is the second SSB paper ⇒ **#1 + #2** is the minimal complete answer, adding exactly one new entry.

### Site 2 — l.33 `\TODO{cite numbu-goldstone paper/review paper}` *(sic)*
> *"This neutral coordinate acts as the Nambu-Goldstone mode of the trained potential."*

⚠ **Correction to the task premise first: Minami & Hidaka 2018 is NOT in this paper's reference list**, nor in `submission.tex`, nor among `v2-cite-check`'s 26 (which explicitly records it as absent). **The program currently cites no NG reference of any kind** — a new entry is required, and the only partial substitute already present is #23 (Iqbal et al., which supplies "Goldstone modes" but for *propagation*, not for a *stored coordinate*).

| rank | candidate | what it claims | fit |
|---|---|---|---|
| **1** | **Watanabe (2020)**, *"Counting Rules of Nambu–Goldstone Modes"*, **Annu. Rev. Condens. Matter Phys. 11:169–187**, DOI **10.1146/annurev-conmatphys-031119-050644** | A genuine **review**; gives the modern counting rules and stresses that *"the number of resulting NGMs can be lower than that of broken symmetry generators"* in systems **lacking Lorentz invariance** | ⭐ **Best for a geometry/ML audience**: it is a review, it is short, and its central caveat (non-relativistic counting) is exactly CLU's regime — a damped, non-Lorentz-invariant latent map where `dim(G/H)` counting must be asserted, not assumed. Also insures §3 l.68's `dim(G/\mathcal H)` claim. |
| **2** | **Minami & Hidaka (2018)**, *"Spontaneous symmetry breaking and Nambu-Goldstone modes in dissipative systems"*, **Phys. Rev. E 97:012130**, DOI **10.1103/PhysRevE.97.012130** | NG modes in **dissipative** systems | ⭐ **Closest to CLU's actual setting** — the paper's whole point is a *damped* (`γ>0`) map, and this is the only candidate that addresses NG modes under dissipation. If only one cite is affordable and the Head wants the dissipative caveat covered, take this over #1. |
| **3** | The primaries: **Goldstone (1961)**, *"Field theories with 'Superconductor' solutions"*, Nuovo Cimento **19**(1):154–164, DOI **10.1007/BF02812722** · and/or **Nambu & Jona-Lasinio (1961)**, *Phys. Rev.* **122**:345–358, DOI **10.1103/PhysRev.122.345** | The original theorem | Use **only** at §3 l.68, *"We rely strictly on the classical, tree-level Goldstone theorem evaluated over a Hessian"* — that sentence is the one that genuinely wants a primary. ⚠ At l.33 a 1961 particle-physics primary buys a NeurReps reader less than a review. |

**Suggested shape:** one of {#1, #2} at l.33; optionally Goldstone 1961 additionally at l.68. All four DOIs above are **NEW, Crossref-verified this pass**.

### Site 3 — l.33 `\cite{}\TODO{relevant citations}` ⚠ claims-sensitive
> *"while the theoretical existence of such a flat direction is established"*

⛔ **Recommendation: add NO new work here.** This is the one site where a fresh "existence of flat directions" citation could narrow the CRR novelty claim. Everything needed is already in the list.

| rank | recommendation | why |
|---|---|---|
| **1** | **Golubitsky, Stewart & Schaeffer 1988 + Krupa 1990 + Rumberger 2001** (entries 14, 32, 39) | The classical orbit-neutrality trio. §2 l.55 already uses these three for *exactly* this proposition (*"Neutrality along group-orbit directions is a foundational principle"*). Citing the same three at l.33 makes §1 and §2 consistent at zero novelty cost. Rumberger in particular states that *"drifts along the group orbits disappear"* on the reduced orbit space (CC #15). |
| **2** | **+ arXiv:2605.03338** (entry 35) | ⭐ Per charter Add.49 **the zero-Lyapunov-exponent theorem belongs to this preprint**, and l.33's *"the theoretical existence of such a flat direction is established"* **is that theorem** in a recurrent-network setting. Citing it here is the *sanctioned attribution*, not a concession — App. A l.230 already attributes it. ⚠ **The larger risk is the reverse:** asserting "established" at l.33 while attributing the theorem only in an appendix reads as under-attribution. |
| **3** | **+ Ságodi et al. 2024** (entry 43), optionally **Seung 1996** (entry 44) | Continuous-attractor existence *and* structural fragility — already cited at l.55/57 for both halves. Adds the "and it is fragile" half that l.33's next clause ("*its operational utility … depends on a checklist*") depends on. |
| ⛔ | **Do not reuse site-1/site-2 candidates here** | Kaba & Ravanbakhsh and Watanabe/Minami–Hidaka are about *why symmetries produce neutral modes*, i.e. site 1/2's job. Importing them at site 3 would make "existence is established" rest on a physics theorem rather than on the dynamical-systems prior art the charter names, and would invite a referee to ask why the *quantitative* claim isn't also prior. |

### Already-known identifiers — format check only (as instructed)
- **l.35 `\TODO{cite CHLU}`** → **`arXiv:2603.01768`** ✔ format valid; arXiv API re-resolved to v2, Jawahar & Pierini, comments *"Accepted as a short paper at ICLR 2026 (AI & PDE)"*. ⚠ CC §26's anonymity flag stands: `pj_sub` l.35 currently reads *"similar to~\TODO{cite CHLU}"* with no possessive — **that construction is safe**; do not let the revision spoke reintroduce "our reference unit … introduced in <names>".
- **l.48 / l.86 / l.96** → **`arXiv:2605.03338`** ✔ all three are the same work. ⭐ **Task question answered: no DOI exists.** The arXiv API record shows **no `<arxiv:doi>` and no `<arxiv:journal_ref>`** as of 2026-08-24 — v1 only, single author, unrefereed. The arXiv ID is the only identifier available, and "recently posted preprint" remains the correct label (CC trap (ii)).

---

## Confidence & gaps

**Verified this pass against primary/registry records (2026-08-24):** all 18 NEW identifiers. Crossref DOI-resolution for Aitken, Burak, Driscoll, Gardner, Hairer, Khona, Kim, McLachlan, Minami–Hidaka, Nambu–Jona-Lasinio, Goldstone, Renart, Seung, Vafidis, Watanabe, Dinc(PRX). AAAI OJS publisher page for Nijkamp. arXiv API `id_list`/title/author queries for all arXiv IDs. `proceedings.mlr.press/v197` and `/v228` volume indexes for all PMLR page ranges.

**Two-instrument:** McLachlan (Crossref + ScienceDirect pii), Nijkamp (OJS page + search record), the three PMLR residuals (arXiv title search + arXiv author search + Crossref bibliographic, all negative).

**Single-sourced / not verified:**
- Goel et al. author list — arXiv API gives the five authors; the ICLR-2026 acceptance is from the arXiv comments field plus a search summary of an OpenReview PDF. **OpenReview is bot-walled** (`audience-refresh` §1.1), so I could not read the forum page.
- The PMLR landing-page URL in D3-1 is **inferred from PMLR's `<author><year><letter>` convention**, not clicked. Marked as such.
- Gell-Mann, Oakes & Renner DOI in the D3 flags table — **from general knowledge, NOT verified.** Do not use as-is.
- I did not attempt DOI lookups for NeurIPS/ICLR/ICML proceedings entries: those venues mint no DOIs (verified by the absence of `<arxiv:doi>` on all four 2025 NeurIPS records I re-resolved).

**Not attempted:** `web.archive.org` (tool-blocked, per prior scouts). OpenReview (bot wall). Semantic Scholar (not needed — every item resolved on a primary or a registry).

**What to check next, if the Head wants belt-and-braces:** (i) click the four PMLR landing pages to confirm the `akhtiamov23a`-style slugs before hand-entering D3-1/3/4; (ii) confirm the Gell-Mann–Oakes–Renner DOI if App. G is to cite it; (iii) once `refs.bib` is generated, diff the 46 Zenodo-emitted titles against the 46 `pj_sub` strings — my D3 flags table found **three** title/venue divergences by inspection, so a systematic diff will likely find more.

---

## Bibtex-ready refs for the NEW site-1/site-2 candidates

```bibtex
@article{watanabe2020counting,
  title={Counting Rules of {N}ambu--{G}oldstone Modes},
  author={Watanabe, Haruki},
  journal={Annual Review of Condensed Matter Physics}, volume={11}, pages={169--187}, year={2020},
  doi={10.1146/annurev-conmatphys-031119-050644},
  note={REVIEW. Non-relativistic counting: number of NGMs can be FEWER than broken generators. Retrieved 2026-08-24.}}

@article{minami2018dissipative,
  title={Spontaneous symmetry breaking and {N}ambu--{G}oldstone modes in dissipative systems},
  author={Minami, Yuki and Hidaka, Yoshimasa},
  journal={Physical Review E}, volume={97}, number={1}, pages={012130}, year={2018},
  doi={10.1103/PhysRevE.97.012130},
  note={NOT currently in V2's reference list -- the task brief's premise was wrong. Closest to CLU's damped setting. Retrieved 2026-08-24.}}

@article{goldstone1961field,
  title={Field theories with ``Superconductor'' solutions},
  author={Goldstone, Jeffrey},
  journal={Il Nuovo Cimento}, volume={19}, number={1}, pages={154--164}, year={1961},
  doi={10.1007/BF02812722}, note={Retrieved 2026-08-24.}}

@article{nambu1961dynamical,
  title={Dynamical Model of Elementary Particles Based on an Analogy with Superconductivity. {I}},
  author={Nambu, Yoichiro and Jona-Lasinio, Giovanni},
  journal={Physical Review}, volume={122}, number={1}, pages={345--358}, year={1961},
  doi={10.1103/PhysRev.122.345}, note={Retrieved 2026-08-24.}}

@article{kaba2024symmetry,
  title={Symmetry Breaking and Equivariant Neural Networks},
  author={Kaba, S{\'e}kou-Oumar and Ravanbakhsh, Siamak},
  journal={arXiv preprint arXiv:2312.09016}, year={2023},
  note={v2 2024-03-22. arXiv comments field: ``Symmetry and Geometry in Neural Representations'' (= NeurReps) -- VENUE-NATIVE. No DOI. Retrieved 2026-08-24.}}

@inproceedings{goel2026anysubgroup,
  title={Any-Subgroup Equivariant Networks via Symmetry Breaking},
  author={Goel, Abhinav and Lim, Derek and Lawrence, Hannah and Jegelka, Stefanie and Huang, Ningyuan},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2026},
  note={arXiv:2603.19486; also NeurReps 2025 spotlight (#65). Author list from the arXiv record; ICLR acceptance from the arXiv comments field (OpenReview bot-walled). Retrieved 2026-08-24.}}

@inproceedings{cohen2016group,
  title={Group Equivariant Convolutional Networks},
  author={Cohen, Taco S. and Welling, Max},
  booktitle={Proceedings of the 33rd International Conference on Machine Learning (ICML)}, year={2016},
  note={arXiv:1602.07576. No SSB content -- equivariance lineage only. Retrieved 2026-08-24.}}
```

New/updated standing bib facts (for the program record):
```bibtex
@article{dinc2026ghost,
  title={Ghost Mechanism: An Analytical Model of Abrupt Learning in Recurrent Networks},
  author={Dinc, Fatih and Cirakman, Ege and Kurtkaya, Bariscan and Yuksekgonul, Mert and Jiang, Yiqi and Schnitzer, Mark J. and Tanaka, Hidenori},
  journal={Physical Review X}, volume={16}, number={2}, pages={021058}, year={2026},
  doi={10.1103/mjcl-lb4x},
  note={SUPERSEDES the ``2025 preprint arXiv:2501.02378'' record. Crossref title drops the leading ``A''. Retrieved 2026-08-24.}}

@inproceedings{nijkamp2020anatomy,
  ... booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34}, number={4}, pages={5272--5280}, year={2020}, doi={10.1609/aaai.v34i04.5973},
  note={DOI + pages NEW this pass (AAAI OJS). Retrieved 2026-08-24.}}

@book{hairer2006geometric,
  ... doi={10.1007/3-540-30666-8},
  note={DOI NEW this pass (Crossref book record). NOT 10.1007/978-... . Retrieved 2026-08-24.}}
```

---

Git footprint: **none** (read-only; no tracked file touched; no paper file touched).

Open questions / follow-ups / risks:
1. **Numeric-vs-author-year style** (reconciliation item 1) is the single highest-consequence decision in this conversion and it is invisible in the `.tex` today. Needs an explicit Head ruling before the `.bib` is wired.
2. **Dinc:** DOI (→ 2026 PRX) or arXiv (→ 2025 preprint)? The block currently carries the DOI per the DOI-preferred rule.
3. **Cite-or-cut on 9 orphans** — six attach to existing sentences in one clause each; Hairer and Huang are *restorations* of citations `submission.tex` had and `pj_sub` dropped; UnICORNN is the only genuine cut candidate.
4. Does the Head want the **GMOR** relation itself cited (App. G leans on it hard with no reference)?
5. `pj_sub` truncated the Vafidis title relative to its own base file — worth a broader title-integrity diff of `pj_sub` against `submission.tex` before the `.bib` lands.

## Proposed handover updates (for the Hub)
- **V2 bibliography inputs are complete:** 50 entries · 46 resolvable identifiers (28 CARRIED, 18 NEW) · 4 residuals requiring hand entry (Akhtiamov & Thomson · Anonymous theory note · Dönmez · van der Ouderaa & van der Wilk) · **0 guessed**. Block at `.claude/outputs/v2-bib-doi-list.md` §Deliverable 1.
- **New standing bib facts:** Dinc et al. → **PRX 16(2):021058 (2026), DOI 10.1103/mjcl-lb4x** (no longer a preprint) · Nijkamp → **10.1609/aaai.v34i04.5973**, AAAI 34(4):5272–5280 · Hairer et al. → **10.1007/3-540-30666-8** · McLachlan & Perlmutter → **10.1016/S0393-0440(01)00020-1** · Driscoll → **10.1016/j.cell.2017.07.021** · Dönmez PMLR 228 pages = **443–458** · Lillemark et al. → **accepted at ICML 2026** (arXiv comments).
- ⭐ **arXiv:2605.03338 still has NO DOI and NO journal-ref** (re-checked 2026-08-24). It remains an unrefereed single-author preprint; the arXiv ID is its only identifier.
- ⛔ **Program-record correction:** the standing note that *"Minami & Hidaka 2018 is in the list"* is **wrong** — it is in no V2 bibliography and never has been. V2 cites **no** Nambu–Goldstone reference today.
- ⛔ **`pj_sub.tex` has regressed against `submission.tex` in at least two places:** the Vafidis title is truncated, and the Huang et al. citation at the solution-degeneracy sentence was dropped (making Huang an orphan). A systematic `pj_sub` ↔ `submission` reference diff is cheap and probably worth one spoke-hour.
