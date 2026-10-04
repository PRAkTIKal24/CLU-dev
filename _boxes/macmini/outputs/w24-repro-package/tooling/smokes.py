"""Smoke driver for the package's entry points (tooling only).

    smokes.py PKG_COPY PYTHON LOG [--only name1,name2]

Runs every entry point's documented smoke from PKG_COPY with PYTHON (the fresh venv), PYTHONPATH
unset, and appends command / exit status / seconds / last output lines to LOG.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PKG, PY, LOG = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ONLY = None
if '--only' in sys.argv:
    ONLY = set(sys.argv[sys.argv.index('--only') + 1].split(','))

S = 'runs/shapes3d/stores/test/seed0/K32_n2048/arm.pkl'
SMOKES = [
    # name, argv (after python), timeout seconds
    ('split', ['-m', 'nclu.data.gen_value_holdout', '--data-root', 'data/shapes3d', '--out-dir', 'data/shapes3d/value_holdout'], 300),
    ('partition_test', ['-m', 'nclu.experiments.factor_wells', '--stage', 'partition', '--condition', 'test', '--out', 'runs/partitions/test'], 600),
    ('partition_val', ['-m', 'nclu.experiments.factor_wells', '--stage', 'partition', '--condition', 'val', '--out', 'runs/partitions/val'], 600),
    ('build_nclu', ['scripts/shapes_build_nclu.py', '--where', 'test', '--seed', '0', '--K', '32', '--npool', '2048'], 600),
    ('score_nclu', ['scripts/shapes_score_arms.py', '--corners', 'rows34', '--seed', '0', '--arms', 'store:%s:smoke_K32_n2048' % S, '--out-tag', '_smoke'], 900),
    ('fit_baseline', ['scripts/shapes_fit_baseline.py', '--where', 'test', '--seed', '0', '--family', 'mlp', '--npool', '2048', '--no-tenx', '--suffix', '_smoke'], 900),
    ('score_baseline', ['scripts/shapes_score_arms.py', '--corners', 'rows34', '--seed', '0', '--arms', 'rival:test_smoke:mlp__declared', '--out-tag', '_smoke'], 900),
    ('baseline_suite_table', ['-m', 'nclu.experiments.baseline_suite', '--stage', 'table', '--seeds', '0'], 1800),
    ('baseline_suite_learners', ['-m', 'nclu.experiments.baseline_suite', '--stage', 'learners', '--seeds', '0', '--families', 'mlp', '--variants', 'full'], 3600),
    ('score_table_floor', ['scripts/shapes_score_arms.py', '--corners', 'rows34', '--seed', '0', '--arms', 'table', '--out-tag', '_smoke'], 900),
    ('build_nclu_full', ['scripts/shapes_build_nclu.py', '--where', 'test', '--seed', '0', '--K', '32', '--npool', '131072'], 1800),
    ('delete_nclu_item', ['scripts/shapes_delete_nclu.py', '--seed', '0', '--scope', 'item'], 7200),
    ('ladder_earlier_shipped', ['scripts/shapes_ladder_earlier.py', '--rung', 'shipped', '--seed', '0', '--out', 'runs/shapes3d/ladder_earlier', '--arms', 'single_design_K5,rival_table'], 3600),
    ('lace_poe', ['scripts/shapes_fit_lace_poe.py', '--rung', 'shipped', '--seed', '0', '--nrows', '2048', '--max-corners', '2', '--out', 'runs/shapes3d/lacepoe_smoke'], 3600),
    ('sweep_k', ['scripts/shapes_sweep_k.py', '--K', '5', '--seed', '0', '--no-controls', '--no-exactness'], 3600),
    ('ladder_earlier', ['scripts/shapes_ladder_earlier.py', '--rung', '2', '--seed', '0', '--out', 'runs/shapes3d/ladder_earlier', '--arms', 'single_design_K5,pca_tensor,rival_table,rival_mlp'], 3600),
    ('ladder_nclu', ['scripts/shapes_ladder_nclu.py', '--rung', '2', '--seed', '0', '--arms', 'store'], 3600),
    ('refusal_comparator', ['scripts/shapes_refusal_comparator.py', '--seed', '0', '--n-fit', '1024', '--epochs', '1', '--out', 'runs/shapes3d/refusal_comparator_smoke'], 3600),
    ('deletion_battery_candidates', ['scripts/shapes_deletion_battery_candidates.py', '--K', '5', '--seed', '0', '--n-pool', '4096'], 3600),
    ('deletion_full_square_fit', ['-m', 'nclu.experiments.deletion_full_square', '--stage', 'fit', '--out', 'runs/shapes3d/deletion_full_square/smoke'], 3600),
    ('deletion_full_square', ['-m', 'nclu.experiments.deletion_full_square', '--stage', 'assertions', '--out', 'runs/shapes3d/deletion_full_square/smoke'], 3600),
    ('composition_cell_pins', ['-m', 'nclu.experiments.composition_cell', '--stage', 'pins', '--out', 'runs/shapes3d/composition_cell/val_seed0'], 900),
    ('settle_dynamics_pins', ['-m', 'nclu.experiments.settle_dynamics', '--stage', 'pins', '--out', 'runs/shapes3d/settle_dynamics/val_seed0'], 900),
    ('zero_dynamics_check', ['scripts/zero_dynamics_check.py'], 900),
    ('retention_criteria_help', ['scripts/retention_law_criteria.py', '--help'], 120),
    ('retention_law_dev', ['-m', 'nclu.experiments.retention_law', '--stage', 'dev', '--seed', '0', '--out', 'runs/physics/retention_dev', '--max-seconds', '600'], 1800),
    ('shapes_tables', ['scripts/shapes_tables.py'], 900),
    ('generation_table_kstar', ['scripts/shapes_generation_table.py', '--out', 'runs/shapes3d/generation_table', '--seed', '0', '--arm', 'kstar', '--stage', 'all'], 3600),
    ('sweep_k_seeds', ['scripts/shapes_sweep_k_seeds.py', '--K', '32', '--seed', '0', '--no-controls', '--no-exactness'], 3600),
    ('sweep_k_report', ['scripts/shapes_sweep_k_report.py'], 600),
    ('sweep_k_figure', ['scripts/shapes_sweep_k_figure.py'], 600),
    ('coverage_control', ['scripts/shapes_coverage_control.py', '--seed', '0', '--learned', '0'], 3600),
    ('coverage_control_report', ['scripts/shapes_coverage_control_report.py'], 600),
    ('tables_lace_poe', ['scripts/shapes_tables_lace_poe.py'], 600),
    ('encoder_loop_warm_toy', ['-m', 'nclu.experiments.encoder_loop', '--stage', 'warm', '--seed', '0', '--out', 'runs/encoder_loop_smoke', '--quick'], 1800),
    ('encoder_loop_train_toy', ['-m', 'nclu.experiments.encoder_loop', '--stage', 'train', '--arm', 'loop_swap', '--seed', '0', '--out', 'runs/encoder_loop_smoke', '--quick'], 1800),
    ('encoder_loop_report_help', ['scripts/encoder_loop_report.py', '--help'], 120),
    ('encoder_loop_criteria_help', ['scripts/encoder_loop_criteria.py', '--help'], 120),
    # CMAZE-T
    ('cmazet_selftest', ['scripts/cmazet_selftest.py', '3'], 900),
    ('cmazet_symmetric', ['scripts/cmazet_symmetric.py', '0', 'runs/cmaze_t/smoke', '--cond', 'BOTH', '--leg', 'val', '--smoke', '1'], 1800),
    ('cmazet_relational', ['scripts/cmazet_symmetric.py', '0', 'runs/cmaze_t/smoke_rel', '--cond', 'BOTH', '--leg', 'val', '--smoke', '1', '--rel_diag', '1', '--clu_cfg', 'scripts/configs/nclu_rel.json'], 1800),
    ('cmazet_asymmetric', ['scripts/cmazet_asymmetric.py', 'full', '0', 'runs/cmaze_t/asym_smoke', '--leg', 'val', '--latent', '128', '--K', '2', '--atoms', '6', '--smoke', '1'], 1800),
    ('cmazet_holdout_support', ['scripts/cmazet_holdout_support.py', 'runs/cmaze_t/holdout_support_smoke.json', '--seeds', '0', '--legs', 'val'], 1800),
    # figures
    ('task_figure_cmazet', ['figures/task_figures.py', '--only', 'cmazet', '--assets', 'runs/figures/task_assets', '--out', 'runs/figures/task'], 900),
    ('results_figures_help', ['figures/results_figures_b.py', '--help'], 120),
    ('cmazet_extract_help', ['figures/cmazet_extract.py', '--help'], 120),
]


def main():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    with open(LOG, 'a') as log:
        for name, argv, tmo in SMOKES:
            if ONLY and name not in ONLY:
                continue
            t0 = time.time()
            try:
                p = subprocess.run([PY] + argv, cwd=PKG, env=env, capture_output=True, text=True, timeout=tmo)
                rc, out = p.returncode, (p.stdout + p.stderr)
            except subprocess.TimeoutExpired as e:
                rc, out = 'TIMEOUT', ((e.stdout or b'').decode(errors='replace') + (e.stderr or b'').decode(errors='replace'))
            dt = time.time() - t0
            tail = '\n'.join('    | ' + l for l in out.strip().splitlines()[-8:])
            log.write('## %s  rc=%s  %.1fs\n$ python %s\n%s\n\n' % (name, rc, dt, ' '.join(argv), tail))
            log.flush()
            print('%-28s rc=%-7s %7.1fs' % (name, rc, dt), flush=True)


main()
