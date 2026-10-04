#!/bin/bash
# Final standalone proof. Fresh copy of the package, fresh venv (uv), PYTHONPATH unset, no
# environment script sourced. Writes proof/amend3_smoke.log and proof/amend3_repro.log (AMENDMENT-3 rebuild).
set -u
unset PYTHONPATH
O=/Users/batmac/clu/outputs/w24-repro-package
W=$1
S=$O/proof/amend3_smoke.log
R=$O/proof/amend3_repro.log
mkdir -p $W
cp -R $O/nclu-iclr-code $W/pkg
cd $W/pkg
{
echo "# Standalone proof, $(date -u +%FT%TZ)"
echo "copy: $W/pkg (from nclu-iclr-code/, $(find . -type f | wc -l | tr -d ' ') files)"
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
uv venv --python 3.12 $W/venv 2>&1 | tail -1
uv pip install --python $W/venv/bin/python -e . 2>&1 | grep -E "^ \+ (jax|numpy|equinox|nclu)|Installed"
$W/venv/bin/python -c "import nclu, pathlib, jax, numpy, equinox; p=pathlib.Path(nclu.__file__).resolve(); print('nclu.__file__ =', p); print('jax', jax.__version__, 'numpy', numpy.__version__, 'equinox', equinox.__version__); assert str(p).startswith('$W/pkg'), p; print('ASSERT nclu.__file__ inside the copy: OK')"
mkdir -p data/shapes3d
ln -sf /Users/batmac/clu/data/gen/shapes3d/3dshapes.h5 data/shapes3d/3dshapes.h5
echo "data: data/shapes3d/3dshapes.h5 -> dataset file; sha256 $(shasum -a 256 data/shapes3d/3dshapes.h5 | cut -c1-64) (expected 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785)"
export NCLU_FEATURES_ROOT=/Users/batmac/clu/stage/outputs/spoke-a-rivals/cache_generation2
echo "NCLU_FEATURES_ROOT=<feature caches; seed_0 only on this machine>"
} > $S 2>&1
python3 $O/tooling/smokes.py $W/pkg $W/venv/bin/python $S >> $S.progress 2>&1

# ---------------------------------------------------------------- full reproductions
{
echo "# Full reproductions from the fresh copy ($W/pkg), $(date -u +%FT%TZ)"
echo
echo "## Shapes3D: n-CLU K*=32, seed 0, rows 3-4 TEST (Tab. t7, Tab. ledger-kstar)"
T0=$(date +%s)
echo '$ python scripts/shapes_build_nclu.py --where test --seed 0 --K 32 --npool 131072   (built in the smoke pass: build_nclu_full)'
$W/venv/bin/python -c "import json; d=json.load(open('runs/shapes3d/stores/test/seed0/K32_n131072/info.json')); print('build_seconds', d['build_seconds'], 'address_fit_seconds', d['address_fit_seconds']); print('footprint', json.dumps(d['footprint_four_part']))"
echo '$ python scripts/shapes_score_arms.py --corners rows34 --seed 0 --arms store:runs/shapes3d/stores/test/seed0/K32_n131072/arm.pkl:store_K32_n131072'
$W/venv/bin/python scripts/shapes_score_arms.py --corners rows34 --seed 0 --arms store:runs/shapes3d/stores/test/seed0/K32_n131072/arm.pkl:store_K32_n131072 2>&1 | grep -E "novel/corner|DERANGEMENT|Error|Traceback"
T1=$(date +%s); echo "scoring runtime_s=$((T1-T0))"
$W/venv/bin/python - <<'EOF'
import json
b = 'runs/shapes3d/scores/rows34/seed0/'
d = json.load(open(b + 'store_K32_n131072.json'))['summary']
print('n-CLU all-109 %.3f | row3 %.3f | row4 %.3f' % (d['all']['novel_per_corner_mean'], d['row3_pair']['novel_per_corner_mean'], d['row4_nway']['novel_per_corner_mean']))
for c in ('landscape_removed', 'landscape_shuffled', 'landscape_shuffled_across_wells', 'random_atoms'):
    print('control %-34s all-109 %.3f' % (c, json.load(open(b + 'store_K32_n131072__%s.json' % c))['summary']['all']['novel_per_corner_mean']))
EOF
echo
echo "## Shapes3D deletion, random 1% of items, seed 0 (Tab. del item row; run in the smoke pass: delete_nclu_item)"
$W/venv/bin/python - <<'EOF'
import json
try:
    d = json.load(open('runs/shapes3d/deletion_store/seed0/item.json'))
    a = d.get('a_store_equals_refit_without', {})
    print('rows deleted', d['n_deleted'])
    print('(a) max|mode dev|', a.get('max_abs_mode_dev', a))
    dd = d['d_composition_other_values']
    print('(d) survivors before %.3f -> after %.3f (paired %+.3f); downdate - refit-without %+.3f' % (
        dd['before_mean'], dd['after_mean'], dd['paired_after_minus_before']['mean_paired_delta'],
        dd['downdate_minus_refit_without_pipeline']['mean_paired_delta']))
    b = d['b_address_book_statistics_equal_refit_without']['b2_book']
    print('(b) rendered rel dev %.3e; book refit from statistic %.1f s' % (b['rendered_features_rel_dev_on_survivor_requests'], b['book_refit_from_downdated_statistic_seconds']))
except Exception as e:
    print('NOT AVAILABLE:', repr(e))
EOF
echo
echo "## Shapes3D lookup-table floor, seed 0 (Tab. t7 floor row; built and scored in the smoke pass)"
$W/venv/bin/python -c "import json; d=json.load(open('runs/shapes3d/scores/rows34/seed0/table_FLOOR.json'))['summary']; print('table floor all-109 %.3f | row3 %.3f | row4 %.3f' % (d['all']['novel_per_corner_mean'], d['row3_pair']['novel_per_corner_mean'], d['row4_nway']['novel_per_corner_mean']))" 2>&1 | tail -1
echo
echo "## Generation table, K* row and its controls, seed 0 (Tab. gentable-kstar; run in the smoke pass: generation_table_kstar)"
$W/venv/bin/python - <<'EOF2'
import json
try:
    d = json.load(open('runs/shapes3d/generation_table/seed_0/cell_manifest.json'))
    for k, a in d['arms'].items():
        print('%-40s composition-B aggregate %.4f' % (k, a['external']['composition_B']['aggregate']))
    print('deployed bytes (four-part)', d['arms']['kstar']['byte_footprint']['deployed_bytes'])
except Exception as e:
    print('NOT AVAILABLE:', repr(e))
EOF2
echo
echo "## CMAZE-T: words + observation, TEST, seed 0, full run (Tab. cmazet-sym)"
T0=$(date +%s)
echo '$ python scripts/cmazet_symmetric.py 0 runs/cmaze_t/symmetric/test --cond BOTH --leg test'
$W/venv/bin/python scripts/cmazet_symmetric.py 0 runs/cmaze_t/symmetric/test --cond BOTH --leg test > $W/cmazet_full.log 2>&1
echo "exit $? runtime_s=$(( $(date +%s)-T0 ))"
$W/venv/bin/python - <<'EOF'
import json
d = json.load(open('runs/cmaze_t/symmetric/test/BOTH_test_seed0.json'))
r = d['ROWS']['nway']
for k, v in r.items():
    if k != 'n' and not k.endswith(':coverage'):
        print('  %-75s %.4f' % (k, v))
print('rollout', json.dumps(d['ROLLOUT'])[:300])
EOF
echo '$ python scripts/cmazet_tables_six_seeds.py runs/cmaze_t/symmetric   (seed 0 only present)'
$W/venv/bin/python scripts/cmazet_tables_six_seeds.py runs/cmaze_t/symmetric 2>&1 | tail -3
echo "end $(date -u +%FT%TZ)"
} > $R 2>&1
