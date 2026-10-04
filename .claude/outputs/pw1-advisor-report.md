# PW-1 — REPORT FOR THE HEAD AND THE ADVISOR
**Paper lead Hub, 2026-09-25. Machine: batmac. Eleven spokes, all finished, all reaped. Zero runs, zero fits, zero GPU, zero cloud.**
Every number below comes from a spoke's own FINDINGS: `~/clu/outputs/pw1-<spoke>/FINDINGS.md` on batmac, mirrored at `.claude/outputs/pw1/pw1-<spoke>/`.

## 0. The answer
All 17 items are closed (item 6 was withdrawn), plus the Head's added template task. Both drafts compile with 0 errors, 0 undefined citations, 0 undefined references and 0 overfull boxes. They carry the new vocabulary and the new cross-reference style, the maze section is CMAZE-T only, every figure is rebuilt, and the drafts sit on the official ICLR 2027 template in double-blind mode. **Main text: 17.95 pp (DRAFT-v2) / 18.25 pp (style) against a 9-page limit.** Length is MF-11, held by the Head.

**Reading copy (Mac):** `.claude/outputs/pw1-paper/w17-paper/` (207 files) and `.claude/outputs/pw1-paper/paper-visuals/` (145 files). The sha256 of every file equals batmac's `~/clu/outputs/{w17-paper,paper-visuals}/`. Following the Head's ruling, the old local `.claude/outputs/w17-paper/` and `paper-visuals/` were not touched.

## 1. Decisions only the Head can make (nothing below was changed without the Head)
1. **Title.** Item 13 retires "exact", title included. The title is now **"naive-CLU: Deletable Compositional Generation on a Frozen Encoder"**; it was "…Exactly Deletable…". Confirm or reword.
2. **"to 1e-8" does not hold (item 12.7).** The n-CLU's content differs from the refit-without by up to **5.38e-07** (all nine scope×seed legs are above 1e-8). The address book's rendered features differ by up to **7.81e-08** (two legs above 1e-8). The book's statistic differs by **0** on all nine. The draft prints these measured bounds and never writes "1e-8".
3. **"10× steps" rulings conflict.** Item 10.6 prescribes the ladder legend entry "seed at $10\times$ steps", and the figure follows it. Item 1.8 says never "10× steps".
4. **Seed-5 sentence (item 7).** It now prints "a 0.04 margin" (record leaf 0.0393) and "106 of the 317 fit rows". **106 is arithmetic**: the record prints 211 of 317 right, emitting E on 63 and N on 43, so 106 = 317 − 211 = 63 + 43. A `%` comment says so. "Deployed span" is not defined in the text and may need a gloss.
5. **The body-text quote at 9 pt.** The deletion-scope statement in Sec. 2 is set in `\begin{quote}\small`. The template's body size is 10 pt. The template spoke could not tell whether this is a layout override, so it left it in.
6. **The door rows in the CMAZE-T panels come from the n-CLU-REL run.** The base n-CLU gets no door transition right on seeds 0–2 (0 of 204). Seeds 3–5 of the base n-CLU cannot be searched, because they have no persisted objects and no per-item scores. All 44 n-CLU-REL door successes are seed 2 and key colour 2. Each such row is labelled "n-CLU-REL run, seed 2: all columns".
7. **Shapes3D render (item 9).** The drawn n-CLU failure is a request that every model fails. It came from a uniform draw over the n-CLU's 511 failures (124 tied), not from choosing it. The n-CLU object has no manifest hash anywhere. Its identity rests on re-reading it: 2,616 per-combination comparisons against the w20 scores per seed, 0 mismatches on each of seeds 0, 1 and 2.
8. **Choices made during the sweep.** "μ⁻² budget law" became "μ⁻² scaling law". The physics "field store" became "field-mode n-CLU". Memories that are not the n-CLU became "memory". "Gate" became "control check".
9. **Still in the drafts, by the plan's own rulings:**
   - Fig. 1's image says "corner" three times, and Fig. 2's image says "spent"/"spend", the engine's event names. Items 3 and 6 froze both figures.
   - Tab. cmazet keeps "Ceiling set by the request design", a design plateau and not a seed label.
   - Two cited papers carry "Exact" in their titles.
10. **Unresolved from before PW-1:** the CMAZE-T six-seed mean. The Head's §7 has +0.128; the artefacts give +0.046 for the base design and +0.201 for n-CLU-REL. It sits only inside the Head-held abstract `\TODO`, which is untouched.

## 2. Item by item
| item | status | what happened (spoke) |
|---|---|---|
| 1 CMAZE-T conditions figure | **done** | plot-maze: four rows, each seed drawn with its count, no label column, no overlays, no ceiling, the Head's titles, `$10\times$ training steps` sub-labels, a new caption. Three deviations: (a) the legend lists no LSTM, because no row draws one (the six-seed best matched baseline is the Transformer at 10× steps, 0.5490 vs LSTM 0.5351); "mean of 6 seeds" became "mean of 3 or 6 seeds", because two rows have 3 seeds; the words-full row's comparator is the 8,000-step arm, labelled "declared training steps". Item 1.5: draft-figures removed the ceiling / best-of-all-arms row from all 19 places per draft |
| 2 nomenclature sweep | **done** | draft-sweeps; counts in §3. Figure-image vocabulary was done by plot-maze and plot-shapes |
| 3 vocabulary from Fig. 1 | **done** | "oblong" and "held-out composition" were already 0; corner → held-out (factor) combination 104 per draft; ledger → memory footprint 24 / 25 |
| 4 paper-visuals self-contained | **done** | visuals-tooling: `cd paper-visuals && bash build.sh` rebuilds in place, in a flat copy, and from a fresh `env -i` copy, all byte-identical to the installed files. A 61-file engine snapshot now lives in `paper-visuals/engine/`. `fig_headline_steps` is no longer built. Each mirrored figure has a documented rebuild line; `fig_task_shapes` was tested byte-identical, and the rendered Shapes3D panels were documented but not re-rendered |
| 5 cross-reference style | **done** | 0 `\S`, `Section~`, `Figure~`, `Table~`, `Appendix~` before a `\ref`, except 1 `\S\ref` + 1 `Table~\ref` inside the Head-held abstract `\TODO` |
| 6 | **withdrawn** | Head, on the current build |
| 7 wording from W23 | **done** | seed-5 limitation (see §1.4); relational limit "n-CLU-REL reads 0.554 against the best matched baseline's 0.797 (six-seed means) and is behind on all six seeds"; the "every number verified" sentences are scoped to 1,308 of 2,293; both `% SRC` fixes are made (0.6407 → `MAZE-T-SEED4.json` `classification.cut.cut`; "16 of 16" → `w22-figures-redraw2/FINDINGS.md` l.15); PoE 27.135× unchanged. The seed-5 numbers needed a second pass (draft-wording2), because the W23 record existed only on the Mac. The Hub staged it to batmac: 19 files, sha256 identical |
| 8 no hard/easy seed | **done** | 0 hard/easy seed in the text; the one "hard" left is "Hard read isolation" (physics). The CMAZE-T paragraph keeps only the mechanism, with no seed named and no ceiling number |
| 9 Shapes3D rendered panels | **done** | render-shapes: Fig. 3 per-model grid (10 rows: sphere 4, cylinder 3, capsule 3) and Fig. 4 "baseline succeeds, n-CLU fails" (one row per baseline; not all capsules: seed-0 pool is capsule 170 / cylinder 100 / sphere 40). Draw seed 20260925, the rule printed in both captions, the six read-backs in `REQUESTS-pw1.json`. Gates: 116/116 object hashes, one decoder, positive control valid, two builds identical |
| 10 ladder | **done** | plot-shapes: one panel, no title, legend at the right, no line-end labels, new names and y label. Printed observation: the MLP was carried at 10× steps on 10 of 15 points, not "every rung/seed" as the w20 notes say; the caption does not repeat that claim |
| 11 wells curve | **done** | "validation composition"; "Bytes ($\times 10^6$)"; "(b) Memory footprint vs K"; "n-CLU, mean of 3 seeds" |
| 12 deletion figure | **done** | 6 panels → 2, retitled; "red sphere"; the removed panels' facts moved into Tab. del plus one sentence (see §1.2) |
| 13 deletion framing | **done** | draft-wording: "exact" 88 → 34 per draft, and all 34 are other senses (numeric equality, exact-match requests, maths). The scoping sentence reads "The match is on the n-CLU and its address book; the frozen encoder and decoder are outside the object". No general claim about composition after deletion. Limitations: the red-sphere loss (6.13 / 6.43 / 6.30 per held-out combination) and the address-book refit (0.97 GB, 2,367–5,913 s). New future-work paragraph: above every retrained baseline on value and item; level with MLP and LSTM on red sphere, where the 95% interval crosses 0 for MLP seeds 1–2 and LSTM seed 2. "exact judge" → "the engine's judge" (33 per draft) |
| 14 maze = CMAZE-T only | **done** | draft-cuts: all listed material removed. It also removed App. "Maze refusal displays (retired)" and Tab. impl-maze, because both are CMAZE-2 material. CMAZE-1/2/2b/O: 0 in source and PDF. The CMAZE-suite paragraph closes the section. Removed numbers keep `%` pointers |
| 15 Fig. 9 (rel-tuples) out | **done** | its one `\ref` now points to Tab. cmazet-sym |
| 16 CMAZE-T panel figures | **done** | the bottom paragraph is removed from every image and stated once in Sec. 4, with each caption pointing to it. The door-success search is recorded in `DOOR-SEARCH.json`: base n-CLU class **empty**; rows drawn from n-CLU-REL (see §1.6). A "most baselines wrong" row was drawn from 199 qualifying items. Hash gate 132/132 + 96/96 + 96/96; ambiguity 0 |
| 17 Fig. 14 (cmazet-deletion) out | **done** | its placeholder went with it (\PLACEHOLDER 3 → 2) |
| Head-added: ICLR 2027 template | **done** | draft-template. (a) The template was re-downloaded: the zip and all 7 files are identical to the copy in use, so nothing was installed. (b) **Defect fixed:** the running header "Under review as a conference paper at ICLR 2027" printed on **0 of 42 pages**. The system `fancyhdr` v5.2 was loading instead of the template's bundled copy. Now it prints on every page. (c) List-spacing overrides removed: 6 (DRAFT-v2) / 8 (style). (d) Double-blind: 0 identity leaks (PDF metadata empty; 0 identity tokens in 713/714 streams; no names, institutions or acknowledgements). (e) A rendered-text word diff before → after is EMPTY in both drafts |

## 3. Replacement tables (item 2, 3, 5; draft-sweeps)
| class | DRAFT-v2 | DRAFT-v2-style |
|---|---|---|
| store → n-CLU (generic) | 324 | 327 |
| rival → baseline | 190 | 190 |
| cross-references | 168 | 169 |
| corner → held-out (factor) combination | 104 | 104 |
| special (named objects, content sense, controls, record, title, first mention) | 94 | 94 |
| budget (by sense) | 57 | 57 |
| pre-registration / scored run | 52 | 52 |
| configuration labels (A1, A10, C-512, …) | 48 | 48 |
| economic metaphors (one by one) | 43 | 44 |
| exact judge → the engine's judge | 33 | 33 |
| ledger → memory footprint | 24 | 25 |
| gate / charter | 17 | 17 |

Every difference between the two drafts is a wording difference of the style paraphrase. The full 300+ rule table, and every metaphor with its sentence, is in `pw1-draft-sweeps/REPLACEMENTS.md`.

Cross-references (DRAFT-v2 before → replaced): `\S\ref` 43 → 42 (1 inside the `\TODO`); `Figure~` 7 → 7; `Figures~` 3 → 3; `Table~` 80 → 79 (1 inside the `\TODO`); `Tables~` 7 → 7; `Appendix~` 29 → 29; `Appendices~` 2 → 2.
Item 13 (draft-wording): exact* 88 → 34 per draft, all non-deletion. Item 14 (draft-cuts): CMAZE-1 14 → 0, CMAZE-2 24 → 0, CMAZE-2b 26 → 0, CMAZE-O 18 → 0, "verb" 28 → 0, `\ref` 187 → 152.
The closing residual audit covers 55 rules over source and PDF text, with **0 unclassified hits**. Every non-zero hit has a stated reason (`pw1-draft-final/FINAL-INDEX.md` §5).

## 4. Figure and table index (DRAFT-v2, final build; the style draft's numbers are identical)
| fig | label | caption lead | p. (v2 / style) |
|---|---|---|---|
| 1 | fig:task-shapes | The Shapes3D task. | 4 / 4 |
| 2 | fig:task-cmazet | The CMAZE-T task. | 5 / 5 |
| 3 | fig:shapes-per-model | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K*=32). | 8 / 9 |
| 4 | fig:shapes-baseline-wins | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 9 / 10 |
| 5 | fig:ladder | The ladder at K* against data-matched baselines, 3 seeds. | 11 / 12 |
| 6 | fig:crossover | The K-curve, validation, 3 seeds at every K. | 27 / 27 |
| 7 | fig:deletion | Deletion on the single design at K*, rows 3–4, test, fit pool 131,072 rows, seeds 0, 1 and 2. | 29 / 29 |
| 8 | fig:adapt | Encoder adaptation margins. | 31 / 31 |
| 9 | fig:maze-wm | CMAZE-T: the n-CLU against the best matched baseline under four input choices, and six seeds without and with relational coding. | 36 / 36 |
| 10 | fig:cmazet-families | Held-out CMAZE-T transitions, one per rule family, with every model's next state. | 38 / 38 |
| 11 | fig:cmazet-per-arm | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0). | 39 / 39 |
| 12 | fig:cmazet-rel-per-arm | Each CMAZE-T model succeeding and failing with relational coding. | 40 / 40 |
| 13 | fig:cmazet-door-tint | Door-then-tint transitions. | 41 / 41 |

| tab | label | caption lead | p. (v2 / style) |
|---|---|---|---|
| 1 | tab:t7 | Composition at K*=32 against data-matched baselines, Shapes3D value hold-out, TEST, 3 seeds. | 7 / 7 |
| 2 | tab:ledger-kstar | The footprint breakdown of the K*=32 object, per seed, never netted. | 9 / 8 |
| 3 | tab:ladder | The ladder at K*=32, 3 seeds, per seed. | 10 / 11 |
| 4 | tab:del | Deletion on the single design, the four measurements. | 12 / 13 |
| 5 | tab:hole | Refusal DISPLAY rows. | 14 / 14 |
| 6 | tab:gentable-kstar | The generation table, K* single design. | 25 / 25 |
| 7 | tab:lace-poe | LACE and PoE at their own settings. | 25 / 25 |
| 8 | tab:k5 | Ladder at K=5. | 26 / 26 |
| 9 | tab:kcurve | The K-curve, validation. | 26 / 26 |
| 10 | tab:kcurve3 | The K-curve at 3 seeds on every row (a smoke, not a claim). | 27 / 27 |
| 11 | tab:impl | Deletion on Shapes3D, implementations 1 and 2. | 28 / 28 |
| 12 | tab:ablations | The two ablations, earlier composition cell. | 28 / 29 |
| 13 | tab:rivals-del | Deletion cost for the data-matched baselines: retrain without the rows. | 29 / 30 |
| 14 | tab:rivals-del-seeds | The n-CLU after deletion against each baseline retrained without the same rows. | 30 / 30 |
| 15 | tab:adapt | Encoder adaptation at K*. | 30 / 31 |
| 16 | tab:encloop | Encoder loop, shortened decoder. | 31 / 32 |
| 17 | tab:bard-gpu | The unfreezing decomposition on the longer leg only. | 31 / 32 |
| 18 | tab:crr | Declared criteria of the retention law on the field-mode n-CLU. | 32 / 33 |
| 19 | tab:cmazet (+ alias tab:cmaze-del) | Maze world-model rows: CMAZE-T, TEST, base n-CLU, 3 seeds. | 37 / 37 |
| 20 | tab:cmazet-sym | CMAZE-T with symmetric inputs: every condition, unseen n-way. | 42 / 42 |

⚠ Figs. 5–8 are placed but not cited from the body text. Figs. 6–13 float into the appendices.

## 5. Page counts (final build, markers on the shipped page)
| section | DRAFT-v2 start / length | style start / length |
|---|---|---|
| 1 Introduction | 2 / 2.74 | 2 / 2.87 |
| 2 The object, the judge and the memory footprint | 4 / 1.90 | 4 / 1.96 |
| 3 The n-CLU on Shapes3D | 6 / 8.69 | 6 / 8.74 |
| 4 The same properties on a maze | 15 / 1.67 | 15 / 1.67 |
| 5 Limitations and a leaving note | 17 / 1.72 | 17 / 1.76 |
| 6 Follow-up: the physics CLU | 18 / 0.22 | 19 / 0.23 |
| AI use / ethics / reproducibility statements | 18–19 / 0.46 | 19 / 0.48 |
| References | 19 / 4.60 | 19 / 4.29 |
| Appendices A–S | 24–42 | 24–43 |
| **Main text (title → AI use statement)** | **17.95 pp, 8.95 over 9** | **18.25 pp, 9.25 over 9** |
| **Total** | **42** | **43** |

Along the way: 50 pp before PW-1 → 41 after the maze cuts → 42 after the new figures and wording → 42 / 43 after the template (the style draft gained a page when the list-spacing overrides were removed).

## 6. Every remaining `\TODO` and `\PLACEHOLDER` (by argument; identical in both drafts)
| # | kind | v2 line | where | argument |
|---|---|---|---|---|
| 1 | TODO | 128 | abstract | CMAZE-T sentence reserved for the Head (**Head-held; byte-identical**) |
| 2 | TODO | 921 | Sec. 3.6 | no record prints the byte ratio of the composition-only n-CLU to the single design |
| 3 | TODO | 1223 | AI use statement | the authors to confirm the venue's disclosure categories (**MF-12, Head-held**) |
| 4 | TODO | 1245 | Reproducibility statement | anonymised code archive not yet built |
| 5 | PLACEHOLDER | 2208 | App. N, fig:cmazet-families | per-model $s_{t+1}$ for the observation-only input: not drawable, models not persisted |
| 6 | PLACEHOLDER | 2245 | App. N, fig:cmazet-door-tint | the same on door-then-tint transitions: not drawable |

## 7. What did NOT run
- No measurement, fit, GPU or cloud job. No write to any `outputs/w*/`, TABLES.md, the registry or the claims matrix.
- Items held by the Head, untouched: the abstract's maze `\TODO`, M3 beyond item 13, MF-11 (length), MF-12 (the AI-use statement and other venue items).
- The rendered Shapes3D panels were not re-rendered after install. They were installed from render-shapes' own verified build: two builds identical, hashes equal at both destinations.
- `fig_task_shapes` (Fig. 1) and `fig_task_cmazet` (Fig. 2) were not changed (items 3 and 6).
- The base n-CLU's seeds 3–5 door search: absent, because no persisted objects exist for them. This is not a zero.
- The `.tex` comments still carry machine paths (2 home-directory paths per draft), and the PDF dates carry a time-zone offset. Stripping these is PW-2's anonymised-upload step.
- The palette validator did not run: `node` is not installed on batmac.
- Figs. 5–8 are still not cited from the body text; no item asked for that.

## 8. Record
| spoke | branch @ commit | what it did |
|---|---|---|
| pw1-draft-sweeps | d80809f | items 2, 3, 5, engine's judge |
| pw1-plot-maze | 2b1791e | items 1, 16, 8 (figures) |
| pw1-render-shapes | 9549f26 | item 9 |
| pw1-draft-cuts | 08d0b79 | items 14, 15, 17 |
| pw1-plot-shapes | e7e418e | items 10, 11, 12 (figures) |
| base/pw1 (Hub merge) | f571b2b | the figure pipeline that built every installed figure |
| pw1-draft-figures | a01bea7 | install; captions; 1.5; 12.7; Shapes3D slots |
| pw1-draft-wording | 29be0f8 | items 7, 8, 13 |
| pw1-visuals-tooling | f8ef2bb | item 4 |
| pw1-draft-wording2 | fbdcafb | item 7 seed-5 numbers |
| base/pw1-final (Hub merge) | 8d3a274 | wording2 + tooling |
| pw1-draft-final | 513e5ec | closing audit, title line-break fix |
| pw1-draft-template | a4de290 | ICLR 2027 template, running-header fix |

Each spoke's branch is bundled and verified in its output dir on batmac. Final installed hashes are in `pw1-draft-template/SHA256-final.txt`. Task files: `.claude/tasks/pw1/`. Hub log: `.claude/outputs/pw1-hub-log.md`.
The Hub staged two evidence copies into batmac's read-only areas: the w20 TEST objects (116 files, from forgis-14) and the W23 seed-5 diagnosis (19 files, from the Mac), each verified per file by sha256.
**PW-1 is closed. PW-2 is not started.** Its outline (plan §2) waits for the Advisor.
