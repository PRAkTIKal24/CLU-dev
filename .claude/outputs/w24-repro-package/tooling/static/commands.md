### Shapes3D (after section 3; `<s>` in 0, 1, 2)

**Tab. t7 (composition at K\*=32 against data-matched baselines) and the K\* footprint table.**
```bash
python scripts/shapes_build_nclu.py --where test --seed <s> --K 32 --npool 131072     # measured: 90 s (seed 0)
python scripts/shapes_score_arms.py --corners rows34 --seed <s> \
    --arms store:runs/shapes3d/stores/test/seed<s>/K32_n131072/arm.pkl:store_K32_n131072   # measured: 90 s (n-CLU + 4 controls)
for f in mlp lstm transformer cont_mlp; do
  python scripts/shapes_fit_baseline.py --where test --seed <s> --family $f               # estimated: 0.5-3 h per family (selection + declared + 10x)
  python scripts/shapes_score_arms.py --corners rows34 --seed <s> --arms rival:test:${f}__declared,rival:test:${f}__tenx
done
python scripts/shapes_fit_baseline.py --where test --seed <s> --family mlp --epochs-mult 3 --suffix 30x   # 30x diagnostic rows
python scripts/shapes_score_arms.py --corners rows34 --seed <s> --arms table          # the lookup-table floor (needs the frozen-encoder run, see NEEDS below)
python scripts/shapes_tables.py                                                       # -> runs/shapes3d/SHAPES-MATCHED.json
```
The four-part footprint of the K\* footprint table is printed by the build and written to its `info.json`.

**Tab. ladder (n-way ladder) and Fig. ladder.** Per rung `<n>` in 2..6:
```bash
python scripts/shapes_ladder_earlier.py --rung <n> --seed <s> --out runs/shapes3d/ladder_earlier   # rung pool + hold-out; earlier set and floor
python scripts/shapes_ladder_nclu.py --rung <n> --seed <s> --arms store,controls                  # the K* n-CLU on the rung
python scripts/shapes_build_nclu.py --where rung:<n> --seed <s> --K 32 --npool 131072
for f in mlp lstm transformer cont_mlp; do python scripts/shapes_fit_baseline.py --where rung:<n> --seed <s> --family $f --hp-from test; done
python scripts/shapes_score_arms.py --corners rung:<n> --seed <s> --arms store:...,rival:rung<n>:mlp__tenx,...
python scripts/shapes_fit_baseline_content_rows.py --where rung:<n> --seed <s> --family <f>        # n-CLU-content set
python scripts/shapes_score_content_rows.py --corners rung:<n> --seed <s> --arms rival:...
python scripts/shapes_ladder_earlier_table.py && python scripts/shapes_ladder_nclu_table.py && python scripts/shapes_tables_content_rows.py && python scripts/shapes_tables.py
python figures/results_figures_b.py --out runs/figures --merge runs/figures --panels fig_ladder
```
Estimated: tens of minutes per rung and seed for the n-CLU; baselines as above.

**Tab. del (deletion), Tab. hole (our row), Tab. rivals-del, Tab. rivals-del-seeds, Fig. deletion.**
```bash
python scripts/shapes_delete_nclu.py --seed <s>                                        # value, red sphere, random 1%; see measured time in the proof log
for f in mlp lstm transformer cont_mlp; do for d in value conjunction_live item; do
  python scripts/shapes_fit_baseline.py --where test --seed <s> --family $f --delete-scope $d
done; done
python scripts/shapes_tables.py && python figures/results_figures_b.py --out runs/figures --merge runs/figures --panels fig_deletion
python scripts/shapes_refusal_comparator.py --seed <s>                                 # Tab. hole comparator rows (refuse-token transformer)
```

**Tab. lace-poe.** `python scripts/shapes_fit_lace_poe.py --rung shipped --seed <s> --nrows 131072` (131,072 rows) and
`--nrows 32768` (32,256 fit rows); per rung `--rung <n>`. Estimated: 1.5-8 h per leg.

**Tab. gentable-kstar (the generation table, K\* row and its controls).**
`python scripts/shapes_generation_table.py --out runs/shapes3d/generation_table --seed <s> --arm kstar --stage all`
(measured: 7 min for seed 0). The other rows of that table (the earlier n-CLU read, the prototype ceiling, LACE, PoE)
are arms of the same instrument, `nclu.experiments.exp_gen_cell` (`--arm` names listed in that module).

**Tab. lace-poe aggregation.** `python scripts/shapes_tables_lace_poe.py` after `shapes_fit_lace_poe.py --out runs/shapes3d/lace_poe_32k/ladder --nrows 32768` per rung and seed.

**Tab. k5.** `python scripts/shapes_build_nclu.py --where rung:<n> --seed <s> --K 5 --npool 32768`, scored with
`shapes_score_arms.py --corners rung:<n>`, aggregated by `shapes_tables.py`.

**Tab. kcurve, Tab. kcurve3, Fig. crossover.** `python scripts/shapes_sweep_k.py --condition val --K 5 --K 8 --K 10 --K 12 --K 16 --K 20 --K 24 --K 28 --K 32 --seed <s>`
(one seed per invocation; estimated 10-20 min per K and seed). The three-seed aggregation that produced
`CROSSOVER-3SEED.json` is not in this archive (see the report); Fig. crossover is drawn by
`figures/results_figures_a.py` from that file.

**Tab. impl (deletion assertion batteries).** Implementation 1:
`python -m nclu.experiments.deletion_full_square --stage all --condition test --seed <s>`;
implementation 2: `python scripts/shapes_deletion_battery_candidates.py --K 5 --atoms 64 --seed <s>`.

**Tab. ablations (composition-only n-CLU, item memory, learned reference, content ceiling).**
`python -m nclu.experiments.composition_cell --stage pins|prepare|fit|run|table --condition test --seed <s> --out runs/shapes3d/composition_cell/test_seed<s>`.

**Tab. crr (retention-law criteria).** `python -m nclu.experiments.retention_law --stage law --seed <s> --out runs/physics/retention`
then `python scripts/retention_law_criteria.py --out runs/physics/retention --seeds 0,1,2`.
**Zero-dynamics check (Sec. clue, App. clue).** `python scripts/zero_dynamics_check.py` (synthetic, measured below)
and `python -m nclu.experiments.settle_dynamics --stage pins|prepare|fit|run|table --condition test --seed <s> --out runs/physics/settle`
(the zero-step and settling arms).

**Tab. bard-gpu (unfreezing decomposition, longer leg) and Tab. encloop (shortened leg).**
The encoder loop trains the encoder, the store and a decoder jointly. The longer leg ran on a GPU
(pins: 4,000 joint steps, 1,800 decoder-refit steps, 4,000 warm steps, 32,768 training images;
one GPU, several hours per arm and seed, not re-run here); the shortened leg is the CPU pin
(400 joint steps, 8,192 images). Per seed `<s>` and arm `<A>` in alone, loop_nosw, loop_swap, alone_wc:
```bash
python -m nclu.experiments.encoder_loop --stage warm  --seed <s> --out runs/encoder_loop [--joint-steps 4000 --dec-steps 1800 --warm-steps 4000 --n-train 32768]
python -m nclu.experiments.encoder_loop --stage train --seed <s> --arm <A> --out runs/encoder_loop [same pins]
python -m nclu.experiments.encoder_loop --stage probe --seed <s> --arm <A> --out runs/encoder_loop
python -m nclu.experiments.encoder_loop --stage cache --seed <s> --arm <A> --out runs/encoder_loop
python -m nclu.experiments.encoder_loop --stage rivals --seed <s> --arm <A> --out runs/encoder_loop --families mlp_180k
# score each arm through its own cache, decoder and baselines (rows 3-4)
export NCLU_FEATURES_ROOT=runs/encoder_loop/seed_<s>/<A>/cache NCLU_DECODER_DIR=runs/encoder_loop/seed_<s>/<A>/declift NCLU_BASELINES_ROOT=runs/encoder_loop/seed_<s>/<A>/rivals
for st in pins prepare; do python -m nclu.experiments.composition_cell --stage $st --condition test --seed <s> --out runs/encoder_loop_score/<A>; done
for arm in pca_tensor rival_mlp_180k pca_lookup_argmax pca_landscape_removed; do
  python -m nclu.experiments.composition_cell --stage fit --condition test --seed <s> --out runs/encoder_loop_score/<A> --arm $arm
  python -m nclu.experiments.composition_cell --stage run --condition test --seed <s> --out runs/encoder_loop_score/<A> --arm $arm
done
python -m nclu.experiments.composition_cell --stage table --condition test --seed <s> --out runs/encoder_loop_score/<A> --primary pca_tensor
python scripts/encoder_loop_report.py --score-base runs/encoder_loop_score --run-base runs/encoder_loop --condition test
python scripts/encoder_loop_criteria.py --score-base runs/encoder_loop_score --run-base runs/encoder_loop --condition test --out-dir runs/encoder_loop/criteria
```
Toy CPU smoke (seconds to minutes): `--quick` on the warm and train stages.

### CMAZE-T (`<s>` in 0..5)

**Tab. cmazet-sym (symmetric inputs) and Fig. maze-wm.**
```bash
python scripts/cmazet_symmetric.py <s> runs/cmaze_t/symmetric/test --cond BOTH --leg test        # measured: see the proof log (seed 0)
python scripts/cmazet_symmetric.py <s> runs/cmaze_t/symmetric/test --cond WORDS-FULL --leg test  # seeds 0-2
python scripts/cmazet_symmetric.py <s> runs/cmaze_t/symmetric/val  --cond OBS --leg val
python scripts/cmazet_symmetric.py <s> runs/cmaze_t/symmetric/val  --cond WORDS-MAIN --leg val   # seeds 0-2
python scripts/cmazet_symmetric.py <s> runs/cmaze_t/relational/test --cond BOTH --leg test --rel_diag 1 --clu_cfg scripts/configs/nclu_rel.json
python scripts/cmazet_symmetric.py <s> runs/cmaze_t/relational/val  --cond BOTH --leg val  --clu_cfg scripts/configs/nclu_rel.json
python scripts/cmazet_criteria_symmetric.py runs/cmaze_t/symmetric/test BOTH test
python scripts/cmazet_tables_symmetric.py runs/cmaze_t/symmetric
python scripts/cmazet_holdout_support.py runs/cmaze_t/symmetric/diagnosis/SEED-DIAGNOSIS-v2.json --seeds 0,1,2,3,4,5 --legs test,val
python scripts/cmazet_tables_six_seeds.py runs/cmaze_t/symmetric
python scripts/cmazet_tables_relational.py runs/cmaze_t/relational
python scripts/cmazet_content_diagnosis.py runs/cmaze_t/content_diagnosis     # the seed-5 note (reads seed 5's persisted objects)
python figures/results_figures_b.py --out runs/figures --merge runs/figures --panels fig_maze_conditions
```

**Tab. cmazet (asymmetric inputs).**
```bash
python scripts/cmazet_asymmetric.py sweep <s> runs/cmaze_t/asymmetric/t1       # VAL configuration sweep
python scripts/cmazet_select_config.py runs/cmaze_t/asymmetric/t1 none
python scripts/cmazet_asymmetric.py full <s> runs/cmaze_t/asymmetric/t1 --leg test --latent 128 --K 2 --atoms 6
python scripts/cmazet_criteria_asymmetric.py runs/cmaze_t/asymmetric/t1 A1 test
python scripts/cmazet_tables_asymmetric.py runs/cmaze_t/asymmetric
```
Engine self-test and harness check: `python scripts/cmazet_selftest.py 20`, `python scripts/cmazet_harness_check.py <dir>`.

### Figures

| figure | command |
|---|---|
| Fig. task-shapes | `python -m nclu.task_assets.extract_images --h5 data/shapes3d/3dshapes.h5 --out runs/figures/task_assets`, `python -m nclu.task_assets.extract_assets --out runs/figures/task_assets`, then `python figures/task_figures.py --only shapes --assets runs/figures/task_assets --out runs/figures` |
| Fig. task-cmazet | `python figures/task_figures.py --only cmazet --assets runs/figures/task_assets --out runs/figures` |
| Figs. shapes-per-model, shapes-baseline-wins | `python -m nclu.render_shapes.build runs/figures/render_shapes` (needs the Tab. t7 objects at seed 0) |
| Figs. ladder, deletion, maze-wm | `python figures/results_figures_b.py --out runs/figures --merge runs/figures` |
| Fig. crossover | `python figures/results_figures_a.py --out runs/figures ...` (needs `CROSSOVER-3SEED.json`, see the report) |
| Figs. cmazet-families, cmazet-per-arm, cmazet-rel-per-arm, cmazet-door-tint | `python figures/cmazet_extract.py runs/cmaze_t/symmetric/test/persist figures/data/cmazet_items.json`, `python figures/cmazet_extract_relational.py figures/data/cmazet_items_rel.json`, then `python figures/results_figures_b.py --out runs/figures --merge runs/figures` |
