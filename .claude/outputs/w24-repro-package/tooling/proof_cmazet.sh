#!/bin/bash
# Proof run A: fresh copy of the package, fresh venv (PYTHONPATH unset, no environment script),
# full CMAZE-T words-plus-observation TEST run, seed 0.
set -u
O=/Users/batmac/clu/outputs/w24-repro-package
W=$1
LOG=$O/proof/repro_cmazet_raw.log
unset PYTHONPATH
mkdir -p $W
cp -R $O/nclu-iclr-code $W/pkg
cd $W/pkg
{
echo "=== PROOF A (CMAZE-T) start $(date -u +%FT%TZ) copy=$W/pkg"
uv venv --python 3.12 $W/venv 2>&1 | tail -1
uv pip install --python $W/venv/bin/python -e . 2>&1 | tail -2
$W/venv/bin/python -c "import nclu, pathlib; p=pathlib.Path(nclu.__file__).resolve(); print('nclu.__file__', p); assert str(p).startswith('$W/pkg'), p"
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
shasum -a 256 scripts/cmazet_symmetric.py scripts/cmazet_lib.py scripts/cmazet_asymmetric.py scripts/cmazet_relational_nclu.py nclu/cmaze/*.py nclu/cmaze/harness/*.py nclu/core/gen_request_head.py nclu/experiments/covariance_store.py
echo "CMD: python scripts/cmazet_symmetric.py 0 runs/cmaze_t/symmetric/test --cond BOTH --leg test"
T0=$(date +%s)
$W/venv/bin/python scripts/cmazet_symmetric.py 0 runs/cmaze_t/symmetric/test --cond BOTH --leg test
RC=$?
T1=$(date +%s)
echo "RC=$RC runtime_s=$((T1-T0)) end $(date -u +%FT%TZ)"
$W/venv/bin/python -c "
import json; d=json.load(open('runs/cmaze_t/symmetric/test/BOTH_test_seed0.json'))
print('ROWS.nway.CLU', d['ROWS']['nway']['CLU'])
print('ROWS.nway best matched (rival: / rival@10x-steps:)', max(v for k,v in d['ROWS']['nway'].items() if k.startswith(('rival: ','rival@10x-steps: '))))
print('seconds_total', d.get('seconds_total'))
"
} >> $LOG 2>&1
