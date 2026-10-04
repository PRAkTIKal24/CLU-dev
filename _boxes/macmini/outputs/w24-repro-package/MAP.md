# MAP — every table and figure of DRAFT-v2.tex → the code that produced it → the package entry

Source paper (read-only): `tasks/w24/draft/DRAFT-v2.tex` (2,568 lines). Counted mechanically:
**20 tables** (21 `\label{tab:…}`; `tab:cmaze-del` is a second label on `tab:cmazet`) and
**13 figures** (11 `\figslot` + 2 `figure` environments).
Origin paths are in the source clone (`HEAD` = `w24/repro-package` @ `677e4e3`; other refs named).
Package entries are paths inside `nclu-iclr-code/`.

Status: **SHIP** = code shipped and its smoke runs · **SHIP (seeds 1–2 need caches)** = runs, but
only seed 0's feature cache exists on this machine · **SHIP-PARTIAL** = shipped with a named gap.
`\TODO` cells: the abstract's CMAZE-T sentence (l.126), the AI-use statement (l.1307) and the
code-archive note (l.1329) — prose, no numbers, no code. `\PLACEHOLDER` occurs only in its
`\newcommand` (l.62): 0 placeholder cells. Declared not-runs (l.2533–2562) print no number.

## Shared inputs

| input | how the package gets it |
|---|---|
| Shapes3D images | user downloads `3dshapes.h5` (sha256 pinned in README) |
| value hold-out split | `python -m nclu.data.gen_value_holdout` (origin `HEAD:chlu/data/gen_value_holdout.py`; regenerated split: every index array identical to the staged one) |
| fit pools / dead-channel masks | `python -m nclu.experiments.factor_wells --stage partition` (origin `HEAD:chlu/experiments/exp_gen_factor_wells.py`; regenerated test pool, dead mask and all partition labels identical to the staged file) |
| frozen-encoder feature caches | not shipped (1.97 GB/seed); sha256 in README; rebuilt from pixels by `python -m nclu.experiments.baseline_suite --stage cache2` with the shipped encoder |
| frozen decoder, judge classifier, frozen encoder, probe, pretext decoder | shipped under `weights/` (array-only, hash-checked) |

## Tables

| # | label (tex line) | origin (branch:path) | package entry | status |
|---|---|---|---|---|
| T1 | `tab:t7` (546) | n-CLU `w20/shapes-data-matched:scripts/w20_store.py` (= the builder of `HEAD:scripts/w12_scored_leg.py`/`w12b_scored_leg.py`); baselines `w20…:w20_fit_rival.py`; scoring `w20…:w20_score.py`; floor `HEAD:chlu/experiments/exp_w4_rivals.py --stage table`; aggregation `w20…:w20_tables.py` | `scripts/shapes_build_nclu.py`, `shapes_score_arms.py`, `shapes_fit_baseline.py`, `python -m nclu.experiments.baseline_suite --stage table`, `shapes_tables.py` | SHIP (seeds 1–2 need caches). **Reproduced: n-CLU seed 0 47.495, controls 0.000, table floor 8.229** |
| T2 | `tab:ledger-kstar` (639) | footprint in `HEAD:scripts/w12_crossover_lib.py` | `shapes_build_nclu.py` (info.json) | SHIP. **Reproduced: seed 0 four-part 2,144,178 B, address book 1,060,864 B** |
| T3 | `tab:ladder` (694) | n-CLU rungs `HEAD:scripts/w12b_ladder_kstar.py`; data-matched set `w20…`; earlier set `HEAD:chlu/experiments/exp_w4_rivals.py --stage learners` + `HEAD:scripts/w12_ladder_leg.py`; n-CLU-content set `w21/ladder-rivals-32k:scripts/w21_*`; coverage control `w15/coverage-control:scripts/w15_coverage_{leg,report}.py`; tables `HEAD:scripts/w12_ladder_table.py`, `w12b_kstar_table.py` | `shapes_ladder_nclu.py`, `shapes_ladder_earlier.py`, `shapes_fit_baseline*.py`, `shapes_score*.py`, `shapes_coverage_control*.py`, `shapes_ladder_*_table.py`, `shapes_tables*.py` | SHIP (seeds 1–2 need caches; coverage report needs all three seeds). The n=6-on-n=3 TEST cell is *derived* text, no code |
| T4 | `tab:del` (841) | `w20…:scripts/w20_delete_store.py` + `HEAD:scripts/w12_deletion_rows.py` | `shapes_delete_nclu.py` | SHIP. **Reproduced (item, seed 0): 1,311 rows; 47.495 → 47.477 (−0.018); downdate − refit +0.000; statistic dev 0; 967,304,800 B** |
| T5 | `tab:hole` (979) | ours: `shapes_delete_nclu.py` (c); comparator `HEAD:scripts/w12_deletion_baselines.py` + `chlu/eval/rivals/w12_deletion_baselines.py` | `shapes_delete_nclu.py`, `shapes_refusal_comparator.py` | SHIP |
| T6 | `tab:gentable-kstar` (1392) | `w15/lace-poe-rows:chlu/experiments/exp_gen_cell.py` (+`exp_gen_cell_kstar.py`, `scripts/w15_lace_poe_gencell.py`) | `scripts/shapes_generation_table.py`, `nclu/experiments/exp_gen_cell.py` | SHIP (K* row + controls run here). **Reproduced seed 0: K* 0.5718, removed 0.0556, shuffled 0.1481, random atoms 0.1458, 2,145,738 B; across-wells 0.1273 vs printed 0.1065 (mismatch)** |
| T7 | `tab:lace-poe` (1426) | `w20…:scripts/w20_lace_poe.py` (= `w15/lace-poe-rows` driver + `--nrows`); aggregation `w15/lace-poe-rows:scripts/w15_lace_poe_table.py` | `shapes_fit_lace_poe.py`, `shapes_tables_lace_poe.py` | SHIP |
| T8 | `tab:k5` (1493) | `w20…` store/score/tables | `shapes_build_nclu.py --K 5`, `shapes_score_arms.py`, `shapes_tables.py` | SHIP |
| T9 | `tab:kcurve` (1548) | `HEAD:scripts/w12_val_sweep.py` | `shapes_sweep_k.py` | SHIP |
| T10 | `tab:kcurve3` (1607) | `w15/crossover-seeds:scripts/w15_crossover_{seeds,report}.py` | `shapes_sweep_k_seeds.py`, `shapes_sweep_k_report.py` | SHIP (seeds 1–2 need caches) |
| T11 | `tab:impl` (1647) | `HEAD:chlu/experiments/exp_w8_downdate.py`; `HEAD:scripts/w8_single_design_dev.py` | `python -m nclu.experiments.deletion_full_square`, `shapes_deletion_battery_candidates.py` | SHIP |
| T12 | `tab:ablations` (1718) | `HEAD:chlu/experiments/exp_w5_compose_field.py` | `python -m nclu.experiments.composition_cell` | SHIP |
| T13 | `tab:rivals-del` (1784) | `w20…:w20_fit_rival.py --delete-scope` | `shapes_fit_baseline.py --delete-scope` | SHIP |
| T14 | `tab:rivals-del-seeds` (1808) | `w20…` score + tables | `shapes_score_arms.py`, `shapes_tables.py` | SHIP |
| T15 | `tab:adapt` (1847) | `HEAD:chlu/experiments/exp_w13_encoder_adapt.py` | — | LEFT OUT (see report: outside the declared scope list; not asked by the amendment) |
| T16 | `tab:encloop` (1892) | `HEAD:chlu/experiments/exp_w8_encoder_loop.py` at the CPU pin + `scripts/w8_encoder_loop_report.py` | `python -m nclu.experiments.encoder_loop`, `scripts/encoder_loop_report.py` | SHIP (same code as T17; toy CPU smoke only) |
| T17 | `tab:bard-gpu` (1930) | `exp_w8_encoder_loop.py` at the GPU pin; scoring through `exp_w5_compose_field.py`; `w15/bar-d-score:chlu/experiments/exp_gen_read.py` (rival-root repair) and `scripts/w15_bar_d_report.py` | `python -m nclu.experiments.encoder_loop`, `nclu.experiments.composition_cell`, `scripts/encoder_loop_report.py`, `scripts/encoder_loop_criteria.py` | SHIP (GPU leg not run; toy CPU smoke; the shell wrappers are replaced by the README command sequence) |
| T18 | `tab:crr` (2004) | `HEAD:chlu/experiments/exp_w6_retention.py` + `HEAD:scripts/w6_retention_bars.py` | `python -m nclu.experiments.retention_law`, `scripts/retention_law_criteria.py` | SHIP |
| T19 | `tab:cmazet` (2251) | `HEAD:chlu/exp_w15_cmaze_t*.py`, `w8_envscout/`, `w9_harness/`, `scripts/w15_cmaze_t_run.py`, `scripts/w16_*.py` | `scripts/cmazet_asymmetric.py`, `cmazet_select_config.py`, `cmazet_criteria_asymmetric.py`, `cmazet_tables_asymmetric.py` | SHIP |
| T20 | `tab:cmazet-sym` (2351) | runners `HEAD:scripts/w19_symmetric.py`, `HEAD:scripts/w21_symmetric.py`, `w21/words-full-test:scripts/w19_symmetric.py`, `w21/maze-t-clu-improve:scripts/w19_symmetric.py` (merged into one, additive changes only) + `w21_clu.py`; tables `HEAD:scripts/w19_*`, `w21_tables.py`, `w21_relnew_tables.py`; diagnosis `HEAD:scripts/w21_seed_diagnosis.py`, `w23_seed5_*.py` | `scripts/cmazet_symmetric.py`, `cmazet_relational_nclu.py`, `cmazet_criteria_symmetric.py`, `cmazet_tables_*.py`, `cmazet_holdout_support.py`, `cmazet_content_diagnosis.py` | SHIP. **Reproduced seed 0: n-CLU 0.8003 (bit-identical to the original run); baselines differ (cross-machine, see report)** |

## Figures

| # | label (tex line) | origin | package entry | status |
|---|---|---|---|---|
| F1 | `fig:task-shapes` (375) | `bm/pw2/draft-final:w19_task_figures/extract_*.py`, `w17-paper/figures/make_task_figures.py` | `python -m nclu.task_assets.extract_images/extract_assets`, `figures/task_figures.py --only shapes` | SHIP |
| F2 | `fig:task-cmazet` (388) | `make_task_figures.py --only cmazet`, `cmazet_panels.py` | `figures/task_figures.py --only cmazet` | SHIP (smoke draws it) |
| F3, F4 | `fig:shapes-per-model` (618), `fig:shapes-baseline-wins` (630) | `bm/pw2/draft-final:pw1_render_shapes/*`, `bm/w17/rendered-compositions:w17_rendered_compositions/*` | `python -m nclu.render_shapes.build` | SHIP (needs the T1 objects) |
| F5 | `fig:ladder` (806) | `w17-paper/figures/make_figures_w22.py` | `figures/results_figures_b.py` | SHIP |
| F6 | `fig:crossover` (1628) | `w15/crossover-seeds:scripts/w15_crossover_figure.py` (paper build: `make_figures_w21.py`) | `scripts/shapes_sweep_k_figure.py`, `figures/results_figures_a.py` | SHIP |
| F7 | `fig:deletion` (1756) | `make_figures_w22.py` | `figures/results_figures_b.py` | SHIP |
| F8 | `fig:adapt` (1879) | `make_figures_w21.py` from T15's output | — | LEFT OUT with T15 |
| F9 | `fig:maze-wm` (2193) | `make_figures_w22.py` (+`maze_six.py`) | `figures/results_figures_b.py` | SHIP |
| F10–F13 | `fig:cmazet-families`, `-per-arm`, `-rel-per-arm`, `-door-tint` (2286–2325) | `make_figures_w21.py`, `cmazet_panels.py`, `cmazet_extract*.py`, `cmazet_door_search.py` | `figures/cmazet_extract*.py`, `figures/results_figures_b.py` | SHIP (needs the persisted T20 objects) |

`w24-advisor-report.md` carries the final state and supersedes this file where they differ.
