# pw1-plot-shapes — FINDINGS (PW-1 items 10, 11, 12 figure parts; items 2–3 vocabulary in the owned figures; the Hub's retirement of the readverb/requestlayer dispatch)

Branch `pw1/plot-shapes`, commit `e7e418e` (parent `2b1791e`, pw1/plot-maze). Bundle `pw1-plot-shapes.bundle`; `git bundle verify` says it is okay (ref e7e418e). **Neither draft was edited and nothing was installed**, as the brief requires. Zero runs and zero fits: every value is read from artefacts under `~/clu/evidence/outputs/`.

## Item status

| item | status | what |
|---|---|---|
| 10.1 one panel | done | only the data-matched 131,072-row set is drawn. In `data/ladder_rival_sets.json` the 32,256-row and 32,768-row MLP sets now have `status: "retired"`, with the ruling recorded. The build asserts that exactly one set (`data_matched_131072`) is drawn |
| 10.2 no title | done | the pool size and step counts are in `CAPTIONS.tex` fig:ladder |
| 10.3 x label | done | "n, factors jointly held out". The 120-combination fact is in the caption |
| 10.4 legend right, vertical | done | `loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1`; nothing is below the plot |
| 10.5 no line-end labels | done | removed |
| 10.6 legend names | done | n-CLU · MLP · LSTM · Transformer · Continuous MLP · Lookup table (floor) · each seed; line = mean · seed at $10{\times}$ steps (renders "10×"; `{\times}` removes mathtext's operator spacing). The build asserts that the family entries equal the drawn families in order, and that 204,800 = 10 × 20,480 |
| 10.7 y label | done | "distinct valid novel tuples / per held-out combination" (on two lines) |
| 10.8 rebuild, provenance, caption | done | builds a/b byte-identical; PROVENANCE-shapes.json; CAPTIONS.tex fig:ladder |
| 11.1 (a) y | done | "validation composition" |
| 11.2 (b) y | done | "Bytes ($\times 10^6$)" |
| 11.3 (b) title | done | "(b) Memory footprint vs K" |
| 11.4 legend | done | "n-CLU, mean of 3 seeds" |
| 11.5 caption | done | CAPTIONS.tex fig:crossover. The controls get their plain names (landscape removed, landscape shuffled, shuffled across wells, random atoms, as in pw1-plot-maze); "honest four-part footprint" → "the memory footprint in its four parts" |
| 12.1 only (e), (f) → (a), (b) | done | former (a)–(d) are not drawn. Their assertions (RESULT flags, b1 = 0, DISPLAY flags, survivor counts) still run, and every value is recorded in PROVENANCE-shapes.json `figures.fig_deletion.deletion_facts` |
| 12.2 (a) title | done | "(a) Composition on the surviving combinations, after − before (3 seeds)" (three lines) |
| 12.3 (b) title | done | "(b) n-CLU after deletion − baseline retrained without the rows" (two lines) |
| 12.4 y labels | done | (a) "after − before, novel tuples per held-out combination (95 % bootstrap CI)"; (b) "n-CLU after − baseline retrained without, same rows (95 % CI)" |
| 12.5 red sphere | done | tick labels are "value", "red sphere", "1 % items (control)"; no ∧ anywhere |
| 12.6 legend | moved right | there was no Head space rule to read, so I took the option that cannot overlap an interval: vertical, right of (b) |
| 12.7 caption; numbers → table + one sentence | done | CAPTIONS.tex fig:deletion; DELETION-FACTS.tex. **See the gap below** |
| 13 (figure/caption half) | done | no "exact" in any rendered string, caption or the sentence (the checker bans it) |
| 2–3 in fig_headline_steps | done | "baseline optimiser steps (log)", "(a) MLP baseline: …", "n-CLU, K* = 32: …", y label "… per held-out combination (rows 3–4, test, 109 combinations)", end labels "Transformer" / "Continuous MLP" (x-limit widened so they fit). The figure has no slot in the paper (w22 Ruling 2), so it has no caption here |
| 2–3 in fig_encoder_adapt | done | y label "n-CLU margin over the 180k-parameter MLP baseline, novel tuples per held-out combination" (it was "margin vs MLP-180k (novel tuples / corner)"). An optional caption is in CAPTIONS.tex |
| Hub: stop building readverb / requestlayer | done | the two calls are removed from `make_figures.main()`; both functions are kept |

## Before → after, rendered strings (the five owned figures)

| figure | before | after |
|---|---|---|
| ladder | 3 panels; titles "(a) Read-path rivals / 32,256 rows · 5,040 steps", "(b) Data-matched MLP / 32,768 rows · 2,560 steps", "(c) Data-matched rivals / 131,072 rows · 20,480/204,800 steps" | 1 panel, no title |
| ladder | x "n, factors jointly held out (n ≥ 3: the same 120 corners)" ×3 | "n, factors jointly held out" |
| ladder | y "distinct valid novel tuples / corner" | "distinct valid novel tuples per held-out combination" |
| ladder | end labels CLU, MLP, LSTM, Cont. MLP, Transf., table (×3 panels) | none |
| ladder | legend below: "CLU, K* = 32", "seed carried at 204,800 steps" | legend right: "n-CLU", "seed at 10× steps" (others unchanged) |
| crossover | "val composition (valid fraction)" | "validation composition" |
| crossover | "bytes (×10⁶), four parts, summed, never netted" | "Bytes (×10⁶)" |
| crossover | "(b) Four-part byte ledger vs K" | "(b) Memory footprint vs K" |
| crossover | "CLU, mean of 3 seeds" | "n-CLU, mean of 3 seeds" |
| deletion | 6 panels (a)–(f) | 2 panels (a), (b) |
| deletion | "(e) (d) as a paired change, 3 seeds" | "(a) Composition on the surviving combinations, after − before (3 seeds)" |
| deletion | "(f) against each rival retrained without the rows" | "(b) n-CLU after deletion − baseline retrained without the rows" |
| deletion | "red ∧ sphere" (×2 tick sets) | "red sphere" |
| deletion | "CLU after − rival retrained without, same rows (95 % CI)" | "n-CLU after − baseline retrained without, same rows (95 % CI)" |
| deletion | "after − before, paired (95 % bootstrap CI)" | "after − before, novel tuples per held-out combination (95 % bootstrap CI)" |
| headline_steps | "rival optimiser steps (log)" ×2 | "baseline optimiser steps (log)" |
| headline_steps | "(a) MLP rival: every pool × step cell" | "(a) MLP baseline: every pool × step cell" |
| headline_steps | "distinct valid novel tuples / corner (rows 3–4, TEST, 109 corners)" | "distinct valid novel tuples per held-out combination (rows 3–4, test, 109 combinations)" |
| headline_steps | "CLU, K* = 32: fit on …" | "n-CLU, K* = 32: fit on …" |
| headline_steps | "Transf.", "Cont. MLP" | "Transformer", "Continuous MLP" |
| encoder_adapt | "margin vs MLP-180k (novel tuples / corner)" | "n-CLU margin over the 180k-parameter MLP baseline, novel tuples per held-out combination" |

Counts: 5 figures changed. Ladder 3 → 1 panel, deletion 6 → 2 panels. 0 banned words remain in any rendered string: `make_figures.check_vocab_pw1` asserts this on every Text artist after a draw, and the strings are listed per figure in `PROVENANCE-shapes.json` `rendered_text`.

## The gap in item 12.7 (printed, not smoothed)
The brief says "n-CLU and address book equal the refit-without **to 1e-8** on every scope and seed". **The artefact does not support that.** From `w20-shapes-data-matched/SHAPES-MATCHED.md` (Matched deletion table) and the matching JSON leaves:
- n-CLU content, max |mode dev|: 5.96e-08, 4.82e-08, 2.01e-08 (value); 5.38e-07, 1.07e-08, 6.25e-08 (red sphere); 2.34e-07, 1.49e-08, 3.16e-08 (item). **All nine are above 1e-8.**
- book rendered-feature relative deviation: 1.85e-09, 1.42e-09, 1.22e-09; 3.29e-08, 7.81e-08, 2.03e-09; 5.83e-10, 9.74e-10, 6.87e-10. **Two are above 1e-8** (red sphere, seeds 0 and 1).
- book statistic (count, first moment): 0 on all nine.

So DELETION-FACTS.tex prints the measured bounds ("at most 5.38×10⁻⁷ … at most 7.81×10⁻⁸"). This is the same bound the live draft already prints at l.119 ("content mode deviation at most \sci{5.38}{-07}"). The Head decides the wording.

## Numbers in CAPTIONS.tex / DELETION-FACTS.tex, each checked against its source
| number | source (verbatim hit) |
|---|---|
| 131,072 | build assert: SHAPES-MATCHED.json `ladder.*.convergence_evidence.*.*.*.n_fit_rows == 131072`; FINDINGS.md l.189 "**131,072**-row pool" |
| 20,480 / 204,800 | build assert: `…declared.optimizer_steps == 20480`, `…tenx.optimizer_steps == 204800` (and 204,800 = 10 × 20,480) |
| 120 | w12-composition-ladder/FINDINGS.md l.51 "the *identical* 120 corners" |
| 524,288 | CROSSOVER-3SEED.json `per_K.32.per_seed.0.ledger_four_part.reference_pin_bytes` (drawn value) |
| 89 / 109 | SHAPES-MATCHED.json `…store.<s>.d.n_survivor_corners` |
| 6.13 / 6.43 / 6.30 | w20 FINDINGS.md l.35 "−6.13 / −6.43 / −6.30" |
| 0.02 / 0.11 / 0.01 | w20 FINDINGS.md l.165 "(−0.02 / −0.11 / −0.01)" (the caption gives the magnitudes, as the live caption does) |
| interval crosses 0: MLP s1, s2; LSTM s2 (red sphere) | SHAPES-MATCHED.json `conjunction_live.rivals.{mlp.1,mlp.2,lstm.2}.ci95` lower bounds −0.936, −0.505, −0.872; the other 33 intervals have lower bound > 0 (all 24 on value and item) |
| the 18 deviations | SHAPES-MATCHED.md l.212–220, verbatim |
| 1.000 / 0.000 | SHAPES-MATCHED.md "1.000 / 0.000" |
| 720 / 4,968 / 5,688 | SHAPES-MATCHED.json `c_DISPLAY.n_deleted_corner_requests` 720, `n_survivor_requests` 4968 / 5688 (thousands separator added) |
| 180k-parameter | w13-encoder-adapt/CROSS-ARM.md "mlp_180k"; FINDINGS.md l.71 "178,952 params" (the live draft already says "180k-parameter") |

A standalone compile of CAPTIONS.tex plus the DELETION-FACTS rows and sentence, with the draft's `\figslot` / `\sci` / `\Kstar` definitions and the built PDFs, gave exit 0, 0 undefined, 0 errors. A banned-word grep on the non-comment text of both files is clean. The `% SRC` comments name JSON paths that contain `store` / `rivals` / `corners`, which cannot be avoided in a pointer.

## Observations printed as observed
- **Ladder, the carried fit.** Filled markers are the 26 of 60 baseline points whose carried fit is the 204,800-step fit: mlp 10 of 15, lstm 1 of 15 (n=2, seed 0), cont_mlp 15 of 15, transformer 0 (`PROVENANCE-shapes.json` `tenx_points`). w20 FINDINGS.md l.189 says "mlp and cont_mlp carried at TENX on every rung/seed", which the JSON `carried_rival` does not show for the mlp (5 of 15 points are carried at 20,480). The figure draws the JSON. The caption says "where the 20,480-step fit had not converged", so it does not repeat the "every rung/seed" claim.
- `rival_at_or_above_store` is empty: on every rung and seed the n-CLU is above every drawn baseline.
- The not-landed 32,768-row set (`ladder_rivals_32k`) is still listed as not landed. Landing it would now fail the one-panel assert on purpose: a second panel needs a Head ruling.
- `make_figures.py` still draws its own w17 versions of fig_crossover, fig_ladder and fig_deletion. `make_figures_w22` overwrites them in the same build, so they are never installed. I left them unchanged (the minimal diff). Their strings are not in the new vocabulary.
- The `SHORT` dict in make_figures_w22.py is now unused (both of its users were changed). I left it in place: the brief asks for minimal diffs.

## Checks
- Owned-figure builds: `build/a` and `build/b` (`tools/build_shapes_only.py`) are byte-identical: `SHA256-build1.txt` == `SHA256-build2.txt` (11 files: 5 stems × pdf/png + PROVENANCE-shapes.json).
- **Full `build.sh`** into scratch destinations, on the committed code: exit 0, "REBUILD: BYTE-IDENTICAL", "INSTALLED BYTE-IDENTICAL" into both scratch destinations (`fullbuild.log`, `SHA256-fullbuild-{1,2}.txt`). **No `fig_maze_readverb`, `fig_maze_requestlayer`, `fig_maze_rel_tuples` or `fig_cmazet_deletion` is produced.** All 10 maze figure files (fig_maze_conditions, fig_cmazet_{families,per_arm,rel_per_arm,door_then_tint} × pdf/png) are cmp-equal to `~/clu/outputs/pw1-plot-maze/build/a/` in both destinations. The 10 owned-figure files are cmp-equal to `build/a`. fig_task_cmazet.pdf is cmp-equal to the installed paper-visuals copy.
- All five images were rendered and inspected for collisions: none after the fixes (the first draft's deletion titles overlapped, and the headline y label ran past the axis; both are fixed).

## Hashes (build/a = build/b)
```
0dfda2510acbce7f4335b8c0a1c5f068e80a93f780a1673a60e7ed577bf24c36  fig_crossover.pdf
22262422cea96dd8f7c71311de03ce98e34c18373e4f90fdd127eb5783b3e865  fig_crossover.png
8faeaf9fa28f05d86a1a201b29b8f8cfadb9cd9d4a1e46cc3cd07092f5c73773  fig_deletion.pdf
b444cfe683d2669a8586e7a2a55e7ad6d8841ae516e55990313e8e5d444f99ff  fig_deletion.png
3c0a924da28bf664d96a9fd49970e3a53257e345cc7463626729d31a71bbacdb  fig_encoder_adapt.pdf
fe565909d28da32aaba044b5e4448e8d7dc64c913cb62cc5acb8a43597c06c08  fig_encoder_adapt.png
9513fc3d9df45e4f2ec22d04854e3b3f43b8af4db06e9cae108ab32d54bfd8d6  fig_headline_steps.pdf
f3a8595113078c61a598eb2b169c6a71419b65833095937d4238f574d5d6d77b  fig_headline_steps.png
949d5b6d1ff2fac3048bf203c22326c36d66d8fed43e0e67a2aae93d09685f2e  fig_ladder.pdf
a5d05a35d53e5272831fb0ae3e79b7c2c6687c30ffae79e33fa579f1b9363165  fig_ladder.png
```

## What did NOT run
- **No install** into `w17-paper/figures/` or `paper-visuals/`, and **no draft edit**: both are excluded by the brief. The live captions of fig:ladder, fig:crossover and fig:deletion still describe the old panels until an installing pass takes `CAPTIONS.tex`.
- No `.tex` lock guard and no style diff-check: no draft write was made.
- No change to `build.sh`, to the maze/CMAZE-T code, or to shared helpers (`ARMS`, `PARTS`, `save`, `FORBIDDEN`). The name overrides are local to the owned functions.
- No new measurement. The "to 1e-8" statement is not supported by the artefact (see the gap). No measurement was commissioned to test it.
- `fig_headline_steps` gets no caption: it has no slot in the paper.
- The README of `paper-visuals/` was not updated: that happens at install time, which is not this branch's job.
