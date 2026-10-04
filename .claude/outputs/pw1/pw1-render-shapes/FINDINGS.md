# pw1-render-shapes — FINDINGS (PW-1 item 9: Shapes3D rendered success/failure panels)

Session `pw1-render-shapes` · branch `pw1/render-shapes` (from `w19/success-failure-panels`, ca57c4f) ·
code `pw1_render_shapes/` · out `~/clu/outputs/pw1-render-shapes/`. **Not installed, no draft edited** (brief).
Zero fits, zero training, zero GPU: frozen objects read through the frozen decoder and the hash-pinned judge.

## Item status
| part | status | note |
|---|---|---|
| (a) per-model grid (n-CLU, MLP, LSTM, Transformer, Continuous MLP; success + failure each; floor column) | **done** | `rc_pw1_shapes_per_model.{pdf,png}`, 10 rows × 6 columns |
| (b) "baseline succeeds, n-CLU fails" rows | **done** (exists for all 4 baselines) | `rc_pw1_shapes_baseline_wins.{pdf,png}`, 4 rows |
| fixed-seed diversity draw, rule + seed printed | **done** | `select.py` docstring = rule; seed 20260925; in REQUESTS-pw1.json and both captions |
| judge's six read-backs per drawn request | **done** | REQUESTS-pw1.json `per_model.rows[*].judge.<column>.readback` (all 6 columns × 14 requests) |
| gates: object hash, same decoder, positive control, reproduction, 2 builds identical, PROVENANCE | **done, all pass** | below |
| CAPTIONS.tex (2 captions + intro paragraph) | **done** | 19/19 numbers checked programmatically against REQUESTS-pw1.json / factor-miss-seed0.txt; banned-word scan 0 hits; standalone compile at 5.5×9 in: 0 warnings, 0 undefined |
| placement in the drafts | **not done — by brief** | a later spoke places them |

## The objects (carried fit read from SHAPES-MATCHED.json `rows34.P1_convergence.<family>.0.carried`)
| column | object (seed 0) | fit |
|---|---|---|
| n-CLU | `stage/outputs/w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl` (sha256 0b38020b…2b8e) | K\*=32, 131,072 rows = w20 `store_K32_n131072` (its score file's `extra.source`; w20 `integrity_T7_reproduced` identical 3/3) |
| MLP | `stage/w20-shapes-data-matched/rivals/test/ckpt/seed_0/mlp__tenx.rival.eqx` | tenx (10× training steps; 204,800 steps) |
| LSTM | `…/lstm__declared.rival.eqx` | declared |
| Transformer | `…/transformer__declared.rival.eqx` | declared |
| Continuous MLP | `…/cont_mlp__tenx.rival.eqx` | tenx |
| Lookup table (floor) | `stage/outputs/spoke-a-rivals/rivals/seed_0/table.rival.pkl` | w20 `table_FLOOR` |
Choice logged: the staged `stores/test` holds only K\*=32 at 16,384 / 32,768 rows (Table M-16k); the data-matched
comparison (baselines on 131,072 rows) is against the 131,072-row n-CLU, which w20 reused from T7. That object is
used. It has **no manifest hash anywhere** on batmac; its identity is established by reproduction (G2).

## Gates
- **G1 object hashes:** `shasum -a 256 -c SHA256-w20-test-objects.txt` → **116/116 OK** before any load (HASHCHECK-w20-objects.txt); re-verified inside each build.
- **G2 reproduction:** scan of all 109 corners × 5,688 held-out requests per seed; per corner per column, `n_requests`, `n_accepted`, `n_valid`, `distinct_valid_novel_tuples` equal the w20 score files (`scores/rows34/seed<N>/{store_K32_n131072,mlp__tenx@test,lstm__declared@test,transformer__declared@test,cont_mlp__tenx@test,table_FLOOR}.json`): **2,616 comparisons, 0 mismatches at each of seeds 0, 1, 2**. Every drawn request re-rendered at build time: verdict bits and all six read-backs equal the scan.
- **G3 same decoder:** 91 panels, 1 decoder object, param_hash 01ef91f75eeaae7b… = evidence `winner_hashes.json` seed_0; classifier c8741ac417d88ebf….
- **G4 positive control:** (i) render+classify == `Judge.read_back` on the 14 drawn n-CLU requests (84 elements identical); (ii) training-set request, row 364857 (drawn `default_rng(20260925)`): capsule, object hue 0.0, floor 0.7, wall 0.6, scale 0.75, 21.4° — the dataset's own cached feature renders **valid**; n-CLU **valid** (MLP, LSTM, Transformer valid; continuous MLP and table invalid). Feature cache sha256 = its manifest.
- **Two builds byte-identical:** `build/SHA256-build1.txt` == `SHA256-build2.txt` (6 files):
```
c2e9df8d6a5dc33be722ed7b1f8a0ab8aba83b661f236c497101d8e3d9673e12  PROVENANCE-render.json
18661bc6d18ee5f5901a24afef8a81b59dc0e4718da28ce918caf8e8b90383b4  rc_pw1_shapes_baseline_wins.pdf
11d82bc2ef1de8ac9ded19a776e39d5f362a0b38ece918df307a8316a97e5821  rc_pw1_shapes_baseline_wins.png
f9dfc17eacf0e6add220447fe7d8467c1e24f0ac5e9a6c273d7b8d81bfb8d0e0  rc_pw1_shapes_per_model.pdf
3d1d7c3707ebec57d35d0d04f82d10399b08912646542b85cb775a740beb7d3d  rc_pw1_shapes_per_model.png
12ec37681e432463f3594a690532c74363de6d5d93aa45389c7b71c8cf86978f  REQUESTS-pw1.json
```
None of the "what would prove this wrong" conditions occurred.

## Counts (seed 0 drawn; seeds 1–2 counted only) — REQUESTS-pw1.json `counts_*`
Held-out requests per seed: 5,688 (capsule 1,872, cylinder 1,944, sphere 1,872; the cube is value-held-out and never requested in rows 3–4).

| valid / 5,688 | n-CLU | MLP | LSTM | Transformer | Cont. MLP | table |
|---|---|---|---|---|---|---|
| seed 0 | 5,177 | 4,361 | 3,703 | 2,786 | 3,110 | 897 |
| seed 1 | 5,167 | 4,230 | 3,527 | 3,029 | 3,590 | 553 |
| seed 2 | 5,354 | 4,706 | 4,436 | 4,213 | 4,643 | 1,063 |

**Baseline valid AND n-CLU invalid** (the (b) qualifying pools):
| seed | MLP | LSTM | Transformer | Cont. MLP | any baseline | by shape (any) | table (floor) |
|---|---|---|---|---|---|---|---|
| 0 | 200 | 197 | 110 | 167 | **310** | capsule 170 / cylinder 100 / sphere 40 | 67 |
| 1 | 181 | 98 | 93 | 226 | **332** | 181 / 86 / 65 | 26 |
| 2 | 96 | 105 | 128 | 150 | **239** | 122 / 82 / 35 | 46 |

**Are they all capsules? No.** Capsules are the plurality (55 % at seed 0) but cylinders and spheres qualify on every seed.
The n-CLU's own failures lean to the capsule (seed 0: 291 of 511 invalid are capsules). The reverse direction, n-CLU valid and
every baseline invalid: 568 / 359 / 124 at seeds 0 / 1 / 2 (seed 0 by shape: capsule 141, cylinder 230, sphere 197).

**How the invalid images are wrong** (factor-miss-seed0.txt, all 5,688 seed-0 requests): n-CLU 497 of 511 invalid miss one
factor only (scale 329, orientation 125, shape 71 factor-misses); MLP 1,276/1,327 one-factor (scale 1,061); LSTM 1,768/1,985
(scale 1,653); Transformer 2,295/2,902 (scale 1,733, orientation 1,687); continuous MLP 2,101/2,578 (orientation 1,405,
scale 1,361, object hue 153); table 4,195/4,791 (scale 4,737).

## The draws (seed 20260925; shape order sphere → cylinder → capsule; no row DECLARED)
(a) pools: n-CLU ✓ 5,177 / ✗ 511; MLP ✓ 4,360 / ✗ 1,326; LSTM ✓ 3,701 / ✗ 1,983; Transformer ✓ 2,782 / ✗ 2,900;
Cont. MLP ✓ 3,108 / ✗ 2,572 (pools exclude already-drawn requests). Shapes across the 10 rows: sphere 4, cylinder 3, capsule 3.
Row families: 6 pair (object hue × shape), 4 all-six-factor. Object hues shown: 0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9.
Worth knowing before the caption is trusted as neutral: the drawn n-CLU failure (nway-015, cylinder) happens to be a request every
column fails (bits 000000); the draw is uniform over the n-CLU's 511 failures, 124 of which were tied at the target shape and
hue score — it was not selected for that.
(b) one row per baseline: MLP (sphere, bits 010000), LSTM (cylinder, 011110), Transformer (capsule, 011110),
Continuous MLP (sphere, 000011). Per-row pool sizes, tie counts and all read-backs: REQUESTS-pw1.json.

## Which draft slots these replace
The w19 slots `fig:t7-per-arm` (`rendered/rc_w19_t7_per_arm_success_failure`), `fig:t7-rival-wins`
(`rendered/rc_w19_t7_rival_succeeds_clu_fails`) and `fig:t7-rival-wins-s1` (`…_seed1`), inserted by w19 draft-figure-slots
after Tab. `tab:t7`, were **removed by w22 draft-rulings Ruling 2**; the live drafts carry only a comment block
(DRAFT-v2.tex:477–480, DRAFT-v2-style.tex:476–479) that says a re-render against the data-matched baselines is pending.
Proposed new labels: `fig:shapes-per-model` (replaces `fig:t7-per-arm`) and `fig:shapes-baseline-wins` (replaces
`fig:t7-rival-wins` and `fig:t7-rival-wins-s1`: one seed-0 figure; seeds 1–2 enter as counts in the caption). Placement:
`sec:shapes-t7`, at that comment block. Install targets: `w17-paper/figures/rendered/` and `paper-visuals/`. Figure (a) at
`0.9\linewidth` (at `\linewidth` it overflows a 9 in text height by 34 pt with its caption).

## What did NOT run
- **Palette validator** (`dataviz/scripts/validate_palette.js`): `node` is not installed on batmac. The six colours are the paper's
  existing one-name-one-colour set (w21_figures/make_figures.py), kept for cross-figure consistency; colour never carries the
  verdict (✓/✗ glyphs) and column/row identity is also written as text.
- **No draft edit, no install, no compile of either draft** (brief: a later spoke places the figures). CAPTIONS.tex was compiled only standalone.
- **Seeds 1–2 were scanned for counts only, not drawn**: the brief pins seed 0 for rendering. Their scans needed no feature cache
  (only seed 0's is staged); the feature cache was used only for the seed-0 positive control. No NEEDS-ARTEFACTS.json was needed.
- **No hash-manifest check for the n-CLU object or the lookup table**: none exists on batmac (not in the w20 manifest, which covers
  the baselines and the 16k/32k stores). Identity is by per-corner reproduction of the w20 score files only (G2).
- The staged w20 `stores/test` objects (K\*=32 and K=5 at 16,384 / 32,768 rows) were hash-checked but **not used**.
- The dataset HDF5 (`~/clu/data/gen/shapes3d/3dshapes.h5`) was not opened; the split files come from `stage/outputs/data/gen`
  (the T7/w20 split; its `factor_index.npz` differs from `~/clu/data/gen/shapes3d/factor_index.npz`, which is a different file).
- Superseded trial builds and the first byte-identical build (before the (b) label fix) were kept in `trial/`, not deleted.
