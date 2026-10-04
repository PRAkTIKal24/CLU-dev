# WAVE 17 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-23. Eight spokes, machine `batmac` (Apple M4), base `base/iter8 = 148f3b7`.**
**All eight finished. ZERO scored runs. ZERO compute. Zero GPU.** Record: `~/clu/outputs/w17-*` on batmac; key artefacts mirrored to `.claude/outputs/w17-harvest/`.
⚠ Every number below is quoted from a spoke's own artefact. ⛔ **The Hub re-derived nothing.**

---

## 0. The scoreboard

| spoke | verdict | scored |
|---|---|---|
| `venue-scout` | ICLR **2027** call + template downloaded; `VENUE.md` (15.8 KB) | 0 |
| `citation-scout` | `CITATIONS.md` (40 KB), 96 entries, each ID resolved not guessed | 0 |
| `draft-v2` | `DRAFT-v2.tex` — six-part spine, **30 placeholders** (post-`referee-fixes`), 1,824 number tokens checked | 0 |
| `style-paraphrase` | `DRAFT-v2-style.tex` — **all four content diffs EMPTY** | 0 |
| `plots` | 3 figures PDF+PNG, **byte-identical rebuild proven** | 0 |
| `rendered-compositions` | 3 panel sets + deletion panel; **0 mismatches** re-reproducing scored `n_valid` | 0 |
| ⭐⭐ `referee-3` | **3/10 REJECT as submitted**, 7 must-fix | 0 |
| `cite-tagger` | both drafts **compile**; prose **byte-identical** after stripping cites | 0 |

## 1. ⭐⭐ THE HEAD'S AMENDMENT WAS APPLIED AND HELD
**Ruling (2026-09-22):** *"w15, w16 data should not go into the draft until the Advisor and I discuss the wave results first anyways."*
⛔ **No w15 or w16 number entered either draft.** The draft is built on **w14-and-earlier evidence only**. All w15 and w16 results are `\PLACEHOLDER{}` — **30 of them as of post-`referee-fixes`** (31 before it; its M5 table split removed one), identical in both drafts. ⚠ The count MOVES as the draft is edited — quote it with a timestamp, never bare.
⭐ The Hub added one constraint the plan did not carry: **a placeholder must not presume a win.** Each names the row AND that it may land as a negative. Verbatim example:
> *"the value, red-sphere and 1\% item deletion rows … so that the table prints $n=3$. Any of them may land as a negative: exactness above floating-point roundoff, a survivor composition change beyond the declared margin, or a refused leg; Table~\ref{tab:del} then prints it as such."*

## 2. ⭐⭐ THE REFEREE: 3/10 REJECT — and two items are CLAIMS defects, not prose
⛔ **M1 — "naive vs naive" is not data-matched.** Table 1's rivals are the `w12-deletion-baselines` checkpoints, fit pool **16,384 rows**; the store's pool is **131,072**. The paper prints the parameter match and never the pool sizes ⇒ an **8× data advantage presented as fairness**. The one data-matched comparison (the 1× MLP on the same 32,768 rows) gives **+3.117 to +17.875**, against the **+39 to +49** the abstract leads with. ⭐ No new run is needed to fix it.
⛔ **M3 — "exact deletion" claims more than the store delivers.** The source definition's bullet — *"NOT a byte-level claim about the address book"* — was dropped from the draft's scope box. The address book is fitted on the full pool and only **rotated** at delete time (1,015,808 of 1,060,864 bytes rewritten), so the post-deletion object is **not** the object fitted without the rows. Our own evidence agrees: **83.5%** of the post-delete vs never-inserted difference comes from the address book and coupling table being fitted on the pool (`w10-physics-section-evidence`).
⚠ M2 (two rival sets an order of magnitude apart on identical corners) · M4 (positioning contradicted by uncited work) · M5 (environments sharing tables) · M6 (missing seed columns) · M7 (venue compliance, desk-reject risk) are mechanical and land in wave 18.
⭐ **The placeholder test passed:** §6 asks whether the paper stands if **every** w15/w16 row lands negative — each is an appendix row or has a stated fallback.
⭐ **No unlocated number.** The four `\TODO`s are honest gaps.

## 3. Integrity checks the spokes ran on themselves
- `style-paraphrase`: numbers, placeholders, **265 artefact comments in identical order**, 96 headings, 74 labels, 28 table floats — **all EMPTY DIFF**.
- `cite-tagger`: both drafts **BYTE-IDENTICAL after stripping inserted cites**; word counts 20,786 / 21,103 unchanged; **0 changed words**; 32 `\citep` + 6 `\nocite` each; both compiled (2,399 / 2,340-line logs, PDFs on disk).
- `plots`: two independent builds, SHA256 manifests match.
- `rendered-compositions`: same frozen decoder hash-asserted against `winner_hashes.json`; positive control render+classify == `Judge.read_back`; full re-render reproduced **every** per-corner `n_valid` across seeds 0/1/2 with **0 mismatches**.
- ⭐ `rendered-compositions` **refused to cherry-pick**: no held-out request at any seed is valid for all six arms, so panel set 2 is **DECLARED** rather than faked; set 3 shows **every arm failing, the CLU included**.

## 4. (a) Placeholders awaiting wave 16 — and wave 15
**31 total.** ⚠ An earlier Hub statement said 26; that was a `grep -c` LINE count, which undercounts lines carrying two placeholders. The count by argument is 31, and it is the same 31 in both drafts. w16: CMAZE-T and CMAZE-O seen / 2-way / n-way / rollout validity / value hold-out / charter-C controls rows. w15: `crossover-seeds`, `single-design-seeds` (deletion rows + rival retrain-without cost), `coverage-control`, `lace-poe-rows`, `bar-d-score`, `maze-deletion-2b`, and both harness audits.
⛔ Every one is written so the paper stands if it lands as a negative.

## 5. (b) THE TWO COPYABLE CITATION BLOCKS — verbatim, for the later Zotero pass
⭐ Generated from `references.bib` (**96 entries**). The generated `.bib` is what both drafts compile against **now**; ⛔ nothing in waves 17–18 depends on replacing it.

Generated by `make_zotero.py` from `references.bib` (96 entries). Paste each block into Zotero's "Add Item by Identifier".

## Block 1: arXiv IDs

```
2102.12853
2009.04075
2004.02546
2111.11736
2206.00048
2112.12911
2203.08422
1907.10786
2007.06600
2004.06030
2206.01714
2302.11552
2310.09336
1907.05012
2302.06676
2603.12977
1912.03817
1911.03030
2002.10077
1911.04933
2209.02299
2206.14439
2407.20516
2303.05699
2206.14389
2308.09881
2601.22456
2303.07345
2303.13516
2310.12508
2402.11846
2205.01833
1803.10122
1912.01603
2301.04104
1911.08265
2209.00588
2405.12399
2408.14837
2402.15391
2406.03689
1911.12247
2204.13661
1802.05983
2204.02283
2107.08221
1811.12359
2307.05596
2307.02598
2501.18797
2410.06303
1711.00350
1908.08351
1912.09713
1604.00289
1706.03762
2110.10873
2112.04426
1911.00172
1907.05242
2203.08913
2412.09764
2306.13831
1606.01164
2008.02217
2302.07253
2008.07669
2111.00396
2312.00752
1806.07366
2103.00065
1803.03635
1410.5401
2510.24965
1412.6980
2111.00254
```

## Block 2: DOIs

```
10.1162/089976600300015349
10.1007/3-540-47969-4_30
10.1145/1073204.1073209
10.1109/ICPR48806.2021.9412780
10.1109/TIP.2014.2346012
10.5244/C.35.430
10.1109/TPAMI.2024.3415506
10.1109/CVPR52688.2022.01125
10.1109/CVPR52688.2022.01091
10.1109/CVPR42600.2020.00926
10.1109/CVPR46437.2021.00158
10.1007/978-3-031-19790-1_26
10.1109/SP.2015.35
10.1007/978-3-032-37673-2_7
10.1109/SP40001.2021.00019
10.1109/CVPR42600.2020.00932
10.1145/3749987
10.1609/aaai.v38i19.30138
10.1109/SaTML54575.2023.00048
10.1109/TDSC.2025.3564992
10.1109/ICCV51070.2023.00230
10.1109/ICCV51070.2023.02074
10.52202/079017-3055
10.1016/S0262-8856(02)00114-2
10.5244/C.13.45
10.1155/2014/429451
10.1117/1.JEI.30.4.043012
10.1137/S089547989223924X
10.5281/zenodo.1207631
10.1038/s41586-025-08744-2
10.1038/s41586-020-03051-4
10.52202/079017-1873
10.52202/079017-0846
10.52202/068431-0736
10.52202/075280-0304
10.52202/075280-1092
10.1613/jair.1.11674
10.1016/0010-0277(88)90031-5
10.1017/S0140525X16001837
10.1162/neco.1997.9.8.1735
10.1162/089976602760128018
10.52202/075280-3209
10.1073/pnas.79.8.2554
10.52202/075280-1197
10.1038/s41467-024-49190-4
10.1038/nature20101
10.52202/085713-1312
10.1214/aos/1176344552
```

**Omitted counts.** Block 1 holds 76 arXiv IDs. **20 of 96 entries are omitted from Block 1** because they have no arXiv ID: tenenbaum2000separating, vasilescu2002tensorfaces, vlasic2005face, lee2014incremental, oldfield2024bilinear, cao2015towards, hall2002adding, hall1999adding, jin2014evd, lee2021fast, gu1994stable, burgess2018shapes, montero2021role, fodor1988connectionism, hochreiter1997long, hinton2002training, hopfield1982neural, kong2024reservoir, graves2016hybrid, efron1979bootstrap.
Block 2 holds 48 DOIs. **48 of 96 entries are omitted from Block 2** because they have no DOI: georgopoulos2020multilinear, harkonen2020ganspace, oldfield2023panda, du2020compositional, du2023reduce, okawa2023compositional, ginart2019making, xu2023netflix, guo2020certified, izzo2021approximate, kong2022approximate, liu2024machine, fang2026machine, fan2024salun, priem2022openalex, hafner2019dream, micheli2023transformers, valevski2025diffusion, bruce2024genie, kipf2020contrastive, zhao2022toward, burgess2018shapes, kim2018disentangling, montero2021role, schott2021visual, locatello2019challenging, liang2025compositional, mahajan2025compositional, lake2018generalization, keysers2020measuring, vaswani2017attention, nie2021controllable, borgeaud2021improving, khandelwal2020generalization, lample2019large, wu2022memorizing, berges2024memory, krotov2016dense, ramsauer2021hopfield, gu2020hippo, gu2022efficiently, gu2024mamba, chen2018neural, cohen2021gradient, frankle2019lottery, graves2014neural, kingma2015adam, kidger2021equinox.
2 entries are in neither block (burgess2018shapes, montero2021role); add them to Zotero by hand from `references.bib`.
`lecun2022path` is not in `references.bib`, and so it is in neither block. The citation scout marked it unverified and said not to insert it.

## 5b. ⚠ TWO BOOKKEEPING ERRORS FOUND BY `w18/registry-fold` — both are real
- ⛔ **A published number disagrees with its own run JSONs.** `w14-read-verb/FINAL-REPORT.md` l.297 gives the maze deletion residual as **8.2e-16**; `TABLES.md` l.360/l.462 and the run JSONs give **7.818e-16**. ⚠ The w14 advisor report — and this Hub, quoting it — repeated the 8.2e-16 figure. Registry entry `N343`.
- ⚠ **The placeholder count was stated three ways** (26 / 29 / 31) across artefacts. Resolved: **31**, counted by argument in both drafts. The Hub's own earlier "26" was a line count.

## 6. What did NOT run — declared
- ⛔ **No scored run, no compute, no GPU, no cloud.** The only executions were `pdflatex` compiles and figure rendering.
- ⛔ `lecun2022path` is **not** in `references.bib` — the citation scout marked it unverified and said not to insert it. ⭐ It is not cited.
- ⚠ 2 entries (`burgess2018shapes`, `montero2021role`) are in **neither** Zotero block and must be added by hand from `references.bib`.
- ⚠ `rendered-compositions` panel set 2 is **DECLARED**, not rendered as specified — no request is valid for all six arms.
- ⚠ Wave 17 ran on **batmac, not forgis-14**: forgis-14 has **no LaTeX**, so `cite-tagger`'s compile deliverable was impossible there.

## 7. What the Advisor is asked to rule on
1. ⭐⭐ **M1.** Is "naive vs naive" defensible as a **parameter-only** match, or must the paper lead with the **data-matched** 1× comparison (+3.117 to +17.875 rather than +39 to +49)? ⛔ Not the Hub's call.
2. ⭐⭐ **M3.** Does the abstract's "deletion is store-scoped and exact" become "exact on the store **content**; the address book is **carried, not refit**"?
3. ⚠ **K\* resolvability (from w15 `crossover-seeds`).** The frozen selection rule's tolerance is **0.02**; the measured seed sd exceeds it at K = 10, 16, 20, 24, 28 **and at K\* = 32** (0.0259). The main text carries K\* alone. Is the rule reportable as having selected K\*, or must the paper say it cannot separate those K on three seeds?
4. ⚠ **M4.** The positioning sentence *"the intersection with exact deletion is empty"* is contradicted by work the draft neither cites nor distinguishes. Which is it: cite-and-distinguish, or weaken the sentence?
