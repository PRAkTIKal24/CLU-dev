"""Name maps for transform.py (tooling only)."""

# ---------------------------------------------------------------- module relocation
MODMAP = {
    'chlu': 'nclu',
    'chlu.exp_w15_cmaze_t': 'nclu.cmaze.task',
    'chlu.exp_w15_cmaze_t_arms': 'nclu.cmaze.arms',
    'w8_envscout': 'nclu.cmaze',
    'w8_envscout.maze_colour': 'nclu.cmaze.maze_colour',
    'w8_envscout.screen': 'nclu.cmaze.screen',
    'w9_harness': 'nclu.cmaze.harness',
    'chlu.eval.rivals': 'nclu.eval.baselines',
    'chlu.eval.rivals.w12_deletion_baselines': 'nclu.eval.baselines.refusal_comparator',
    'chlu.eval.rivals.w4_api': 'nclu.eval.baselines.api',
    'chlu.eval.rivals.w4_cache2': 'nclu.eval.baselines.cache_build',
    'chlu.eval.rivals.w4_common': 'nclu.eval.baselines.common',
    'chlu.eval.rivals.w4_deletion': 'nclu.eval.baselines.deletion',
    'chlu.eval.rivals.w4_final': 'nclu.eval.baselines.final',
    'chlu.eval.rivals.w4_learners': 'nclu.eval.baselines.learners',
    'chlu.eval.rivals.w4_population': 'nclu.eval.baselines.population',
    'chlu.eval.rivals.w4_report': 'nclu.eval.baselines.report',
    'chlu.eval.rivals.w4_rows': 'nclu.eval.baselines.rows',
    'chlu.eval.rivals.w4_specialists': 'nclu.eval.baselines.specialists',
    'chlu.experiments.exp_w4_rivals': 'nclu.experiments.baseline_suite',
    'chlu.experiments.exp_w5_compose_field': 'nclu.experiments.composition_cell',
    'chlu.experiments.exp_w6_compose_physics': 'nclu.experiments.settle_dynamics',
    'chlu.experiments.exp_w6_retention': 'nclu.experiments.retention_law',
    'chlu.experiments.exp_w8_downdate': 'nclu.experiments.deletion_full_square',
    'chlu.experiments.exp_w8_single_design': 'nclu.experiments.covariance_store',
    'chlu.experiments.exp_gen_delete_row7': 'nclu.experiments.gen_deletion',
    'chlu.experiments.exp_w8_encoder_loop': 'nclu.experiments.encoder_loop',
    'chlu.experiments.exp_gen_cell_kstar': 'nclu.experiments.gen_cell_kstar',
    'chlu.experiments.exp_gen_factor_wells': 'nclu.experiments.factor_wells',
    'pw1_render_shapes': 'nclu.render_shapes',
    'w19_task_figures': 'nclu.task_assets',
    'w17_rendered_compositions': 'nclu.rendered_compositions',
    # flat script directories: '<dir>:<module>'
    'scripts:cmazet_symmetric_runner': 'scripts:cmazet_symmetric',
    'scripts:w12_crossover_lib': 'scripts:shapes_nclu_lib',
    'scripts:w15_crossover_seeds': 'scripts:shapes_sweep_k_seeds',
    'scripts:w15_crossover_report': 'scripts:shapes_sweep_k_report',
    'scripts:w15_crossover_figure': 'scripts:shapes_sweep_k_figure',
    'scripts:w15_coverage_leg': 'scripts:shapes_coverage_control',
    'scripts:w15_coverage_report': 'scripts:shapes_coverage_control_report',
    'scripts:w15_lace_poe_gencell': 'scripts:shapes_generation_table',
    'scripts:w15_lace_poe_table': 'scripts:shapes_tables_lace_poe',
    'scripts:w8_encoder_loop_report': 'scripts:encoder_loop_report',
    'scripts:w15_bar_d_report': 'scripts:encoder_loop_criteria',
    'scripts:w12_deletion_baselines': 'scripts:shapes_refusal_comparator',
    'scripts:w12_deletion_rows': 'scripts:shapes_deletion_rows',
    'scripts:w12_fastjudge': 'scripts:shapes_fast_judge',
    'scripts:w12_ladder_leg': 'scripts:shapes_ladder_earlier',
    'scripts:w12_val_sweep': 'scripts:shapes_sweep_k',
    'scripts:w12b_ladder_1x': 'scripts:shapes_ladder_learned',
    'scripts:w12b_ladder_kstar': 'scripts:shapes_ladder_nclu',
    'scripts:w12b_ladder_leg': 'scripts:shapes_ladder_lib',
    'scripts:w12b_scored_leg': 'scripts:shapes_scoring_lib',
    'scripts:w12b_kstar_table': 'scripts:shapes_ladder_nclu_table',
    'scripts:w12_ladder_table': 'scripts:shapes_ladder_earlier_table',
    'scripts:w15_cmaze_t_ready': 'scripts:cmazet_harness_check',
    'scripts:w15_cmaze_t_run': 'scripts:cmazet_lib',
    'scripts:w15_cmaze_t_selftest': 'scripts:cmazet_selftest',
    'scripts:w16_bars_t': 'scripts:cmazet_criteria_asymmetric',
    'scripts:w16_cmaze_t': 'scripts:cmazet_asymmetric',
    'scripts:w16_select_t': 'scripts:cmazet_select_config',
    'scripts:w16_tables': 'scripts:cmazet_tables_asymmetric',
    'scripts:w19_bars': 'scripts:cmazet_criteria_symmetric',
    'scripts:w19_symmetric': 'scripts:cmazet_symmetric_base',
    'scripts:w19_tables': 'scripts:cmazet_tables_symmetric',
    'scripts:w20_delete_store': 'scripts:shapes_delete_nclu',
    'scripts:w20_fit_rival': 'scripts:shapes_fit_baseline',
    'scripts:w20_lace_poe': 'scripts:shapes_fit_lace_poe',
    'scripts:w20_lib': 'scripts:shapes_common',
    'scripts:w20_score': 'scripts:shapes_score_arms',
    'scripts:w20_store': 'scripts:shapes_build_nclu',
    'scripts:w20_tables': 'scripts:shapes_tables',
    'scripts:w21_relnew_replay': 'scripts:cmazet_relational_replay',
    'scripts:w21_relnew_tables': 'scripts:cmazet_tables_relational',
    'scripts:w21_symmetric': 'scripts:cmazet_symmetric_pertuple',
    'scripts:w21_tables': 'scripts:cmazet_tables_six_seeds',
    'scripts:w21_seed_diagnosis': 'scripts:cmazet_holdout_support',
    'scripts:w21C_tables': 'scripts:cmazet_tables_relational_first',
    'scripts:w21C_clu': 'scripts:cmazet_relational_nclu',
    'scripts:w21L_fit_rival': 'scripts:shapes_fit_baseline_content_rows',
    'scripts:w21L_score': 'scripts:shapes_score_content_rows',
    'scripts:w21L_tables': 'scripts:shapes_tables_content_rows',
    'scripts:w21L_lib': 'scripts:shapes_content_rows_lib',
    'scripts:w23_seed5_content': 'scripts:cmazet_content_diagnosis',
    'scripts:w23_seed5_trace': 'scripts:cmazet_content_trace',
    'scripts:w6_retention_bars': 'scripts:retention_law_criteria',
    'scripts:w8_matched_bytes_recut': 'scripts:shapes_matched_bytes',
    'scripts:w8_single_design_dev': 'scripts:shapes_deletion_battery_candidates',
    'clue_checks:check_01_zero_dynamics': 'scripts:zero_dynamics_check',
    'figures:cmazet_door_search': 'figures:cmazet_door_search',
    'figures:cmazet_extract_rel': 'figures:cmazet_extract_relational',
    'figures:cmazet_extract': 'figures:cmazet_extract',
    'figures:cmazet_panels': 'figures:cmazet_panels',
    'figures:figroot': 'figures:fig_paths',
    'figures:make_figures': 'figures:fig_style',
    'figures:make_figures_w21': 'figures:results_figures_a',
    'figures:make_figures_w22': 'figures:results_figures_b',
    'figures:make_task_figures': 'figures:task_figures',
    'figures:maze_six': 'figures:maze_six_seeds',
}

# ---------------------------------------------------------------- identifiers
IDMAP = {
    'R7': 'GD', '_R7_BUILD_ARMS': '_GD_BUILD_ARMS',
    'CHLUConfig': 'NCLUConfig', 'Dfg': 'D_fg', 'dfg': 'd_fg',
    'OUT_W19': 'OUT_SYM', 'PINS_W12': 'PINS_REFUSAL', 'W12': 'RC', 'W12B': 'RCB', 'W16': 'ASYM',
    'W20': 'SM', 'W20C': 'SMC', 'W20_CARRIED_STAGE': 'CARRIED_STAGE', 'W20_EV': 'MATCHED_EV',
    'W20_MANIFEST': 'MATCHED_MANIFEST', 'W20_OUT': 'MATCHED_OUT', 'W20_STAGE': 'MATCHED_STAGE',
    'W5': 'CC', 'W5_EVICT_COMPOSITION_DELTA': 'CELL_EVICT_COMPOSITION_DELTA',
    'W5_HOLE_COMPOSITION_DELTA': 'CELL_HOLE_COMPOSITION_DELTA', 'W5_PINS': 'CELL_PINS',
    'W5_RANDOM_SIZE_MATCHED_DELTA': 'CELL_RANDOM_SIZE_MATCHED_DELTA', 'W6': 'RL', 'W6_PINS': 'SETTLE_PINS',
    'W8': 'FSQ', 'W8_PINS': 'FSQ_PINS', '_claude_root': '_cell_root_dir', '_w20_carried': '_matched_carried',
    'arm_byte_ledger': 'arm_byte_footprint', 'assert_byte_ledger': 'assert_byte_footprint',
    'assert_chlu_in_worktree': 'assert_package_in_root', 'byte_ledger': 'byte_footprint',
    'byte_ledger_expectations': 'byte_footprint_expectations', 'chlu': 'nclu', 'chlu_dim': 'nclu_dim',
    'claude_root': 'cell_root', 'clu_ledger': 'clu_footprint', 'count_ledger': 'count_footprint',
    'coupling_byte_ledger': 'coupling_byte_footprint', 'coupling_ledger': 'coupling_footprint',
    'covariance_ledger': 'covariance_footprint', 'deployed_byte_ledger': 'deployed_byte_footprint',
    'dynread_ledgers': 'dynread_footprints', 'exp_w15_cmaze_t': 'cmaze_task',
    'exp_w15_cmaze_t_arms': 'cmaze_arms', 'exp_w5_compose_field': 'composition_cell',
    'exp_w6_retention': 'retention_law', 'exp_w8_downdate': 'deletion_full_square',
    'exp_w6_compose_physics': 'settle_dynamics', 'exp_w8_single_design': 'covariance_store',
    'exp_w4_rivals': 'baseline_suite', 'exp_gen_delete_row7': 'gen_deletion',
    'exp_gen_factor_wells': 'factor_wells',
    'field_byte_ledger': 'field_byte_footprint', 'four_part_ledger': 'four_part_footprint',
    'honest_byte_ledger': 'honest_byte_footprint', 'honest_ledger': 'honest_footprint',
    'landscape_byte_ledger': 'landscape_byte_footprint', 'ledger': 'footprint', 'ledger_has': 'footprint_has',
    'ledger_matches': 'footprint_matches', 'make_figures_w21': 'results_figures_a', 'n_waves': 'n_periods',
    'perturb_modes_w20': 'perturb_modes_deranged', 'placeholder': 'stand_in',
    'query_set_registry_file': 'query_set_index_file', 'quick_assert_byte_ledger': 'quick_assert_byte_footprint',
    'spokea_steps': 'earlier_set_steps', 'store_byte_ledger': 'store_byte_footprint', 'w1': 'wa', 'w2': 'wb',
    'w2s': 'wbs', 'w12_deletion_baselines': 'refusal_comparator', 'w12_deletion_rows': 'shapes_deletion_rows',
    'w12b_ladder_1x': 'shapes_ladder_learned', 'w12b_ladder_leg': 'shapes_ladder_lib',
    'w15_cmaze_t_run': 'cmazet_lib', 'w16_cmaze_t': 'cmazet_asymmetric', 'w19_baseline': 'sym_baseline',
    'w19_symmetric': 'cmazet_symmetric_base', 'w20_baseline': 'matched_baseline', 'w20_score': 'shapes_score_arms',
    'w20j': 'matched_json', 'w21_clu': 'cmazet_relational_nclu', 'w21_symmetric': 'cmazet_symmetric_pertuple',
    'w21_tables': 'cmazet_tables_six_seeds', 'w4_api': 'baseline_api', 'w4_cache2': 'cache_build',
    'w4_final': 'baseline_final', 'w4_learners': 'baseline_learners', 'w4_population': 'baseline_population',
    'w4_report': 'baseline_report', 'w4_rows': 'baseline_rows', 'w4_common': 'baseline_common',
    'w4_deletion': 'baseline_deletion', 'w4_specialists': 'baseline_specialists', 'w5': 'cell',
    'w20_lib': 'shapes_common', 'w12_crossover_lib': 'shapes_nclu_lib', 'w12_fastjudge': 'shapes_fast_judge',
    'w12b_scored_leg': 'shapes_scoring_lib', 'w12_val_sweep': 'shapes_sweep_k',
    'w8_matched_bytes_recut': 'shapes_matched_bytes', 'w23_seed5_trace': 'cmazet_content_trace',
    'w21L_lib': 'shapes_content_rows_lib', 'w21C_clu': 'cmazet_relational_nclu',
    'w8_envscout': 'cmaze_engine', 'w9_harness': 'cmaze_harness', 'pw1_render_shapes': 'render_shapes',
    'w19_task_figures': 'task_assets',
    'w17_rendered_compositions': 'rendered_compositions',
    'exp_w8_encoder_loop': 'encoder_loop', 'PREREG': 'PROTOCOL_FILE', 'W12_RUNS': 'EARLIER_LADDER_RUNS', 'W12X': 'KSTAR_SCORE_SEED0', 'exp_gen_cell_kstar': 'gen_cell_kstar',
    'w8_encoder_loop_report': 'encoder_loop_report', 'w12_val_sweep_mod': 'sweep_mod',
    'PW1_BANNED': 'FIGURE_TEXT_BANNED', 'PW1_TEXT': 'FIGURE_TEXT_SEEN', 'check_vocab_pw1': 'check_figure_vocabulary',
    'retired_byte_budget': 'prior_byte_budget',
    'QUERY_SET_REGISTRY': 'QUERY_SET_INDEX', 'write_query_set_registry': 'write_query_set_index',
    'registry_path': 'index_path', 'registry': 'index_doc', '_stub_ledger': '_stub_footprint',
    'discover_query_set_registries': 'discover_query_set_indexes',
}

# ---------------------------------------------------------------- strings (ordered)
_mods = sorted(((k, v) for k, v in MODMAP.items() if ':' not in k), key=lambda kv: -len(kv[0]))
STRMAP_MODULES = [(k, v) for k, v in _mods if '.' in k] + [(k + '.', v + '.') for k, v in _mods if '.' not in k and k != 'chlu']
STRMAP_TAIL = [
    ('CHLU_', 'NCLU_'),
    ('chlu_file', 'package_file'),
    ('chlu.__file__', 'nclu.__file__'),
    ('chlu', 'nclu'),
    ('CHLU', 'NCLU'),
]

DOCSTRINGS = {}
FILEMAP_JSON = {
    'chlu/specs/store_plan_shapes3d_recon.json': 'nclu/specs/store_plan_shapes3d_recon.json',
    'figures/data/maze_conditions.json': 'figures/data/maze_conditions.json',
    'figures/data/ladder_rival_sets.json': 'figures/data/ladder_rival_sets.json',
    'figures/data/cmazet_panels.json': 'figures/data/cmazet_panels.json',
}
FILEMAP_EXTRA = {
    'chlu/data_manifest.json': 'nclu/data_manifest.json',
    'scripts/CLU-REL-A10.json': 'scripts/configs/nclu_rel.json',
}

# ---------------------------------------------------------------- output-tree relocation
# old run-output directory (relative to the old outputs root) -> neutral path under NCLU_OUT
PATHMAP = [
    ('w8-encoder-loop', 'encoder_loop'), ('w15-bar-d-score', 'encoder_loop/criteria'),
    ('hub-unseen-valid/w4', 'shapes3d/unseen_valid'),
    ('/Users/batmac/clu/stage/rel-persist/', ''),
    ('_thinkpad/w4/spoke-a-rivals', 'shapes3d/baseline_suite'),
    ('w12b-nr6-and-kstar/KSTAR-LADDER.json', 'shapes3d/ladder_nclu/KSTAR-LADDER.json'),
    ('w12b-nr6-and-kstar/kstar_ladder', 'shapes3d/ladder_nclu'),
    ('w12-composition-ladder/runs', 'shapes3d/ladder_earlier'),
    ('w12b-nr6-and-kstar', 'shapes3d/ladder_nclu'),
    ('w12-composition-ladder', 'shapes3d/ladder_earlier'),
    ('spoke-a-rivals/rivals', 'shapes3d/baseline_suite/rivals'),
    ('spoke-a-rivals/cache_generation2', 'features'),
    ('spoke-a-rivals', 'shapes3d/baseline_suite'),
    ('hub-factor-wells/run-val/partitions.npz', 'partitions/val/partitions.npz'),
    ('hub-factor-wells/run/partitions.npz', 'partitions/test/partitions.npz'),
    ('w12-deletion-baselines/comparators', 'shapes3d/refusal_comparator/comparators'),
    ('w12-deletion-baselines', 'shapes3d/refusal_comparator'),
    ('w12-single-design-crossover/val_sweep', 'shapes3d/k_sweep'),
    ('w12-single-design-crossover', 'shapes3d/k_sweep'),
    ('w20-shapes-data-matched', 'shapes3d'),
    ('w21-ladder-rivals-32k', 'shapes3d/content_rows'),
    ('w8-single-design/dev', 'shapes3d/deletion_battery_candidates'),
    ('w8-downdate', 'shapes3d/deletion_full_square'),
    ('w19-maze-t-symmetric', 'cmaze_t/symmetric'),
    ('w21-words-full-test', 'cmaze_t/symmetric'),
    ('w21-maze-t-seed4', 'cmaze_t/symmetric'),
    ('w21-maze-t-clu-improve', 'cmaze_t/relational'),
    ('w21-rel-newseeds', 'cmaze_t/relational'),
    ('w16-maze-wm', 'cmaze_t/asymmetric'),
    ('w23-seed5-diagnosis', 'cmaze_t/content_diagnosis'),
    ('w15-crossover-seeds', 'shapes3d/k_sweep_seeds'),
    ('w15-lace-poe-rows', 'shapes3d/lace_poe_32k'),
    ('w15-coverage-control', 'shapes3d/coverage_control'),
    ('w13-encoder-adapt', 'shapes3d/encoder_adapt'),
    ('w15-single-design-seeds', 'shapes3d/single_design_seeds'),
    ('pw1-render-shapes', 'figures/render_shapes'),
    ('w19-task-figures', 'figures/task_assets'),
]
OLD_OUT_PREFIXES = ('C:/chlu/evidence/outputs/', 'C:/chlu/outputs/', '/c/chlu/outputs/', 'C:\\chlu\\outputs\\')

# key / label renames inside strings (applied after PATHMAP, before the generic rules)
KEYMAP = [
    ('rival_sets_retired', 'rival_sets_not_drawn'), ('retired_byte_budget', 'prior_byte_budget'),
    ('CHLU_W8_CKPT_REMOTE', 'NCLU_LOOP_CKPT_REMOTE'), ('w8_v1_short', 'loop_v1_short'), ('w8_v1_warm', 'loop_v1_warm'),
    ('w8_arm', 'loop_arm'), ('w8_hub', 'loop_study'), ('NEVER_SHARES_A_TABLE_WITH_W8_CPU', 'NEVER_SHARES_A_TABLE_WITH_CPU_LEG'),
    ('pool_recompute_equals_w12_test_pool', 'pool_recompute_equals_test_pool'), ('source_wave', 'source_round'),
    ('charter_C', 'landscape_C'), ('charter_', 'protocol_'),
    ('chlu/exp_w15_cmaze_t_arms.py', 'nclu/cmaze/arms.py'), ('chlu/exp_w15_cmaze_t.py', 'nclu/cmaze/task.py'),
    ('scripts/w15_cmaze_t_run.py', 'scripts/cmazet_lib.py'), ('scripts/w15_cmaze_t_ready.py', 'scripts/cmazet_harness_check.py'),
    ('scripts/w21_seed_diagnosis.py', 'scripts/cmazet_holdout_support.py'),
    ('scripts/w23_seed5_content.py', 'scripts/cmazet_content_diagnosis.py'),
    ('scripts/w23_seed5_trace.py', 'scripts/cmazet_content_trace.py'),
    ('make_figures_w21.py', 'results_figures_a.py'), ('make_figures_w22.py', 'results_figures_b.py'),
    ('PROVENANCE-w21.json', 'PROVENANCE-results.json'), ('w21_figures_redraw', 'figures_build'),
    ('w17-rendered-compositions', 'rendered-compositions'), ('REQUESTS-pw1.json', 'REQUESTS-render.json'),
    ('w17-plots', 'figures'), ('C:/chlu/worktrees/CHLU-w20-shapes-data-matched', 'scripts'),
    ('CHLU_GEN_DATA_ROOT', 'NCLU_DATA_ROOT'), ('CHLU_GEN_FEATURES_ROOT', 'NCLU_FEATURES_ROOT'),
    ('CHLU_GEN_BRIDGE_ROOT', 'NCLU_ENCODER_ROOT'), ('CHLU_HUB_OUTPUTS', 'NCLU_OUT'),
    ('CHLU_CLAUDE_ROOT', 'NCLU_OUT'), ('CHLU_GEN_STORE_RUNS', 'NCLU_OUT'), ('CHLU_RIVALS_ROOT', 'NCLU_BASELINES_ROOT'),
    ('CHLU_EVIDENCE', 'NCLU_OUT'), ('CHLU_FIG_CODE', 'NCLU_FIG_CODE'), ('CHLU_GEN_CLASSIFIER_ROOT', 'NCLU_WEIGHTS_DIR'),
    ('CHLU_GEN_DECLIFT_DIR', 'NCLU_WEIGHTS_DIR'), ('w12comp', 'priorcomp'), ('cmazet_extract_rel.py', 'cmazet_extract_relational.py'), ('PREREG-w15-bar-d.md', 'protocol.md'), ('w8_encoder_loop_report.py', 'encoder_loop_report.py'), ('w20_convergence', 'matched_convergence'), ('CLU-REL-A10.json', 'nclu_rel.json'),
    ('w21_clu_cfg', 'rel_nclu_cfg'), ('w21_rel_diagnostic', 'rel_diagnostic'), ('w5_pins', 'cell_pins'),
    ('w6_pins', 'settle_pins'), ('w5_measured_mean_delta', 'cell_measured_mean_delta'),
    ('w20_carried_stage', 'matched_carried_stage'), ('w20_carried_steps', 'matched_carried_steps'),
    ('w20_carried', 'matched_carried'), ('w20_stage', 'matched_stage'), ('w20_printed', 'matched_printed'),
    ('setC_w21_32768', 'setC_content_32768'), ('setB_w20_131072', 'setB_matched_131072'),
    ('setA_spokeA_32256', 'setA_earlier_32256'), ('setB_integrity_vs_w20_printed', 'setB_integrity_vs_printed'),
    ('spokeA_', 'earlier_'), ('EXISTING_spokeA_', 'EXISTING_earlier_'), ('integrity_rebuilt_vs_w12b', 'integrity_rebuilt'),
    ('LACE_PoE_from_w15', 'LACE_PoE_32k'), ('how_the_hub_calls_them', 'how_the_scripts_call_them'),
    ('charter_C_control_did_not_collapse', 'landscape_control_did_not_collapse'),
    ('6__PREREG_before_any_number', '6__protocol_before_any_number'), ('byte_ledger_matches_prereg', 'byte_footprint_matches_declared'),
    ('load_without_w12', 'load_without_comparator'), ('Charter-C controls', 'landscape controls'),
    ('charter-C', 'landscape'), ('Charter-C', 'Landscape'),
]
GENERIC_STR = [
    ('LEDGER', 'FOOTPRINT'), ('Ledger', 'Footprint'), ('ledger', 'footprint'),
    ('PLACEHOLDER', 'STAND_IN'), ('placeholder', 'stand_in'), ('TODO', 'OPEN_ITEM'),
    ('PREREG', 'PROTOCOL'), ('prereg', 'protocol'),
]

STRMAP = STRMAP_MODULES + KEYMAP + GENERIC_STR + STRMAP_TAIL

# whole-string replacements (a string constant equal to the key), applied first
EXACT = {
    'R20': 'design constant', 'R0': 'rung_0', 'R2': 'router_2', 'R3': 'router_3',
    'W1': 'Wa', 'W2': 'Wb', 'w19': 'sym3', 'w16': 'asym', 'w21': 'redraw', 'w5': 'cell',
    'spoke': 'study', 'hub': 'study', 'Hub': 'Study', 'w9_reference': 'reference',
    'w12_spoke': 'comparator_study', 'w12/deletion-baselines': 'refusal comparator',
    'hub/factor-wells': 'factor-wells', 'amendment_1': 'change_1',
    '\u26d4_this_spoke_does_not_take_this_trade': 'not_taken_here', 'w21_figures': 'figures',
    'w8_envscout': 'nclu/cmaze', 'w15/cmaze-t-harness': 'cmaze-t harness',
    'chlu-w12b-kstar-ladder/1': 'nclu-ladder-kstar/1', 'w8': 'loop', 'wave': 'round',
    'chlu-w15-bar-d-valgate/1': 'nclu-loop-valgate/1',
    'rc_pw1_shapes_per_model': 'shapes_per_model', 'rc_pw1_shapes_baseline_wins': 'shapes_baseline_wins',
    'retired': 'not_drawn', 'RETIRED': 'NOT_COMPUTED', 'chlu-w15-bar-d/1': 'nclu-loop-criteria/1', 'w9_harness': 'nclu/cmaze/harness', 'registry': 'index',
}

DOCSTRINGS = {
    'scripts/shapes_build_nclu.py': """Build one n-CLU on Shapes3D: wells partition, per-well principal subspaces with the retained
sufficient statistic, and the request-indexed tensor address book (closed-form head).

    python scripts/shapes_build_nclu.py --where test --seed 0 --K 32 --npool 131072

--where test|val|rung:<n> selects the fit pool (the partition file of that condition, or a ladder
rung's pool); the first --npool rows are used. Writes runs/shapes3d/stores/<where>/seed<s>/K<K>_n<npool>/
arm.pkl and info.json (four-part memory footprint).""",
    'scripts/shapes_score_arms.py': """Score arms under the judge (frozen decoder + external classifier) on a combination set.

    python scripts/shapes_score_arms.py --corners rows34 --seed 0 --arms store:<arm.pkl>:<label>
    python scripts/shapes_score_arms.py --corners rows34 --seed 0 --arms rival:test:mlp__declared,table

--corners rows34 (60 pair + 49 six-factor held-out combinations, TEST) | val | rung:<n>.
Arm specs: store:<arm.pkl>:<label> (the four landscape controls are scored beside it unless
--no-controls; the across-wells control is a derangement), rival:<tag>:<name> (a baseline fitted by
shapes_fit_baseline.py), table (the lookup-table floor). Metric: distinct valid novel tuples per
held-out combination. Writes runs/shapes3d/scores/<corners>/seed<s>/<label>.json.""",
    'scripts/shapes_fit_baseline.py': """Fit one read-path-matched baseline family (mlp, lstm, transformer, cont_mlp) on the n-CLU's fit pool.

    python scripts/shapes_fit_baseline.py --where test --seed 0 --family mlp
    python scripts/shapes_fit_baseline.py --where rung:3 --seed 0 --family mlp --hp-from test
    python scripts/shapes_fit_baseline.py --where test --seed 0 --family mlp --delete-scope value

Protocol: six short learning-rate x weight-decay trials and the epoch grid, selected on a fixed 10%
held-out split, then the selected schedule refitted on all rows (declared) and at ten times the
optimiser steps (tenx; --epochs-mult 3 gives the thirty-times diagnostic). --delete-scope
value|conjunction_live|item retrains without the rows the n-CLU deletion removes.""",
    'scripts/shapes_delete_nclu.py': """Deletion on the K*=32 n-CLU (value red, conjunction red sphere, random 1% of items).

    python scripts/shapes_delete_nclu.py --seed 0 [--scope value|conjunction_live|item] [--arm <arm.pkl>]

For each scope: (a) the downdated content against the content refit without the rows, (b) the address
book's statistic and the book refit from it against a book refit without the rows, (c) refusal through
the hole (a display), (d) composition on the surviving combinations before and after, and the downdated
object against the object refit without the rows; with costs. Writes runs/shapes3d/deletion_store/.""",
    'scripts/shapes_fit_lace_poe.py': """LACE and PoE at their own settings in the frozen feature space, on rows 3-4 or a ladder rung.

    python scripts/shapes_fit_lace_poe.py --rung shipped --seed 0 --nrows 131072
    python scripts/shapes_fit_lace_poe.py --rung 3 --seed 0 --nrows 32256""",
    'scripts/shapes_tables.py': """Aggregate the Shapes3D run outputs into runs/shapes3d/SHAPES-MATCHED.json (and .md via the renderer):
composition against the data-matched baselines (rows 3-4), the ladder, the K=5 endpoint, deletion
and the baselines' retrain-without.

    python scripts/shapes_tables.py""",
    'scripts/shapes_sweep_k.py': """Validation sweep of the number of wells K (composition, controls and four-part footprint per K).

    python scripts/shapes_sweep_k.py --K 5 --K 8 --K 10 --K 12 --K 16 --K 20 --K 24 --K 28 --K 32 --seed 0""",
    'scripts/shapes_ladder_earlier.py': """One rung of the n-way ladder: builds the rung's hold-out and fit pool (writes its partitions.npz),
the K=5 label-derived n-CLU, and scores the earlier baseline set and the table floor on the rung.

    python scripts/shapes_ladder_earlier.py --rung 3 --seed 0 --out runs/shapes3d/ladder_earlier""",
    'scripts/shapes_ladder_nclu.py': """The K* n-CLU on one ladder rung (fit pool and dead-channel mask of that rung), with its controls.

    python scripts/shapes_ladder_nclu.py --rung 3 --seed 0 --arms store,controls""",
    'scripts/shapes_ladder_nclu_table.py': """Aggregate the K* ladder legs into KSTAR-LADDER.json / .md.""",
    'scripts/shapes_ladder_earlier_table.py': """Aggregate the earlier ladder legs into LADDER.json / LADDER.csv / LADDER.md.""",
    'scripts/shapes_fit_baseline_content_rows.py': """The n-CLU-content baseline set: the four families fitted on a rung's first 32,768 rows
(exactly the n-CLU's content rows), at the selected schedule, ten times it, or step-matched.""",
    'scripts/shapes_score_content_rows.py': """Score the n-CLU-content baseline set on a rung (as shapes_score_arms.py).""",
    'scripts/shapes_tables_content_rows.py': """Aggregate the n-CLU-content baseline set into LADDER-RIVALS-32K.json / TABLES.md.""",
    'scripts/shapes_refusal_comparator.py': """Read-path-matched comparators and the refuse-token transformer for the refusal display rows.

    python scripts/shapes_refusal_comparator.py --seed 0""",
    'scripts/shapes_deletion_battery_candidates.py': """Deletion assertion battery on covariance-store candidates (implementation 2).

    python scripts/shapes_deletion_battery_candidates.py --K 5 --atoms 64 --seed 0""",
    'scripts/retention_law_criteria.py': """Evaluate the retention law's declared criteria B1-B6 from the retention-law runs.

    python scripts/retention_law_criteria.py --out <run dir of nclu.experiments.retention_law> --seeds 0,1,2""",
    'scripts/zero_dynamics_check.py': """The zero-dynamics identity: the zero-step settle equals the static read (synthetic
checks of conditions C1-C4 and the coercivity certificate).

    python scripts/zero_dynamics_check.py""",
    'scripts/cmazet_symmetric.py': """CMAZE-T transition task with symmetric inputs: the n-CLU, its four landscape controls, the learned
blind arms, matched baselines (8,000 and 80,000 steps; 10x read floats diagnostic), lookup floors,
rollouts and deletion, for one seed, condition and leg.

    python scripts/cmazet_symmetric.py <seed> runs/cmaze_t/symmetric/test --cond BOTH --leg test
    python scripts/cmazet_symmetric.py <seed> runs/cmaze_t/symmetric/val --cond WORDS-MAIN|WORDS-FULL|OBS|BOTH --leg val
    python scripts/cmazet_symmetric.py <seed> runs/cmaze_t/relational/test --cond BOTH --leg test \\
        --rel_diag 1 --clu_cfg scripts/configs/nclu_rel.json        (n-CLU-REL and the +rel baselines)

Options: --smoke 1 (tiny budgets, seconds), --deletion 0|1, --rival_steps N, --diag_mult N.""",
    'scripts/cmazet_asymmetric.py': """CMAZE-T with asymmetric inputs: configuration sweep on VAL and pinned full runs.

    python scripts/cmazet_asymmetric.py sweep <seed> <out_dir>
    python scripts/cmazet_asymmetric.py full <seed> <out_dir> --leg val|test --latent 128 --K 2 --atoms 6""",
    'scripts/cmazet_select_config.py': """Select the CMAZE-T configuration from the VAL sweep.   python scripts/cmazet_select_config.py <dir> <iface>""",
    'scripts/cmazet_criteria_asymmetric.py': """Criteria of the asymmetric-input runs.   python scripts/cmazet_criteria_asymmetric.py <dir> <tag> <leg>""",
    'scripts/cmazet_tables_asymmetric.py': """Tables of the asymmetric-input runs.   python scripts/cmazet_tables_asymmetric.py <out_dir>""",
    'scripts/cmazet_criteria_symmetric.py': """Criteria of the symmetric-input runs.   python scripts/cmazet_criteria_symmetric.py <out_dir> <cond> <leg>""",
    'scripts/cmazet_tables_symmetric.py': """Tables of the symmetric-input runs (seeds 0-2).   python scripts/cmazet_tables_symmetric.py <out_dir>""",
    'scripts/cmazet_tables_six_seeds.py': """Six-seed CMAZE-T tables (words + observation, observation, full word design, n-CLU-REL VAL).

    python scripts/cmazet_tables_six_seeds.py runs/cmaze_t/symmetric""",
    'scripts/cmazet_tables_relational.py': """Six-seed n-CLU-REL tables beside the base n-CLU, with the +rel baselines.

    python scripts/cmazet_tables_relational.py runs/cmaze_t/relational""",
    'scripts/cmazet_tables_relational_first.py': """n-CLU-REL tables for seeds 0-2.   python scripts/cmazet_tables_relational_first.py <out_dir>""",
    'scripts/cmazet_holdout_support.py': """Zero-fit diagnosis of the held-out n-way tuples: training support of each held-out tuple.

    python scripts/cmazet_holdout_support.py <out.json> --seeds 0,1,2,3,4,5 --legs test,val""",
    'scripts/cmazet_content_diagnosis.py': """Zero-fit content diagnosis of the seed-5 n-way misses of the base n-CLU (reads the persisted
seed-5 TEST objects of cmazet_symmetric.py).   python scripts/cmazet_content_diagnosis.py <out_dir>""",
    'scripts/cmazet_selftest.py': """CMAZE-T engine self-test: independent replay of generated transitions.   python scripts/cmazet_selftest.py [n_mazes]""",
    'scripts/cmazet_harness_check.py': """Assemble the harness readiness summary from per-seed validation outputs.   python scripts/cmazet_harness_check.py <out_dir>""",
    'figures/results_figures_a.py': """Result figures (ladder, K-curve, maze conditions) drawn from the run outputs under NCLU_OUT.""",
    'figures/results_figures_b.py': """Result figures: fig_ladder, fig_deletion, fig_maze_conditions and the CMAZE-T panel figures,
drawn from the run outputs under NCLU_OUT.   python figures/results_figures_b.py --out <dir> --merge <dir>""",
    'figures/task_figures.py': """The two task figures (Shapes3D, CMAZE-T); no result is drawn.

    python figures/task_figures.py --only cmazet --assets <dir> --out <dir>
    python figures/task_figures.py --only shapes --assets <dir> --out <dir>""",
    'figures/cmazet_extract.py': """Per-item re-read of the persisted base n-CLU CMAZE-T TEST objects (every arm's n-way score is
asserted equal to the run JSON), writing the item file the panel figures draw.

    python figures/cmazet_extract.py runs/cmaze_t/symmetric/test/persist figures/data/cmazet_items.json""",
    'figures/cmazet_extract_relational.py': """As cmazet_extract.py, for the n-CLU-REL runs.   python figures/cmazet_extract_relational.py figures/data/cmazet_items_rel.json""",
    'figures/cmazet_door_search.py': """Fixed-seed draw of the door-then-tint rows of the CMAZE-T panel figures.""",
    'nclu/render_shapes/build.py': """Render single held-out Shapes3D requests through the frozen decoder for every model (Figs. per-model
and baseline-wins).   python -m nclu.render_shapes.build <out_dir>""",
}
