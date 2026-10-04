#!/bin/bash
# Re-smoke of the two report entry points (and the steps they read) from a fresh copy + fresh venv.
set -u
unset PYTHONPATH
O=/Users/batmac/clu/outputs/w24-repro-package
W=$1
L=$O/proof/smoke_resmoke.log
mkdir -p $W; cp -R $O/nclu-iclr-code $W/pkg; cd $W/pkg
{
echo "# Re-smoke (AMENDMENT-2), $(date -u +%FT%TZ), fresh copy $W/pkg, PYTHONPATH=${PYTHONPATH:-<unset>}"
uv venv --python 3.12 $W/venv 2>&1 | tail -1
uv pip install --python $W/venv/bin/python -e . 2>&1 | grep -E "Installed|nclu=="
$W/venv/bin/python -c "import nclu,pathlib; p=pathlib.Path(nclu.__file__).resolve(); print('nclu.__file__ =', p); assert str(p).startswith('$W/pkg')"
mkdir -p data/shapes3d; ln -sf /Users/batmac/clu/data/gen/shapes3d/3dshapes.h5 data/shapes3d/3dshapes.h5
} > $L 2>&1
export NCLU_FEATURES_ROOT=/Users/batmac/clu/stage/outputs/spoke-a-rivals/cache_generation2
python3 $O/tooling/smokes.py $W/pkg $W/venv/bin/python $L --only split,partition_test,partition_val,sweep_k,sweep_k_seeds,sweep_k_report,sweep_k_figure,ladder_nclu,coverage_control,coverage_control_report >> $L.progress 2>&1
echo "END $(date -u +%FT%TZ)" >> $L.progress
