#!/bin/sh
# usage: clean_build.sh <bank|pool> <idx>   -- runs INSIDE the clean checkout
O=/c/chlu/outputs/w15-maze-deletion-2b; C=$O/clean_checkout
cd $C; source /c/chlu/env_w3.sh
export PYTHONPATH="$(cygpath -w $C);C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -c "import os,pathlib,chlu; cf=pathlib.Path(chlu.__file__).resolve(); assert str(cf).lower().startswith(str(pathlib.Path(os.environ['CHLU_ROOT']).resolve()).lower()), cf; print('ENVCHECK OK',cf)" || exit 3
if [ "$1" = bank ]; then exec python -u -m w14_request_layer.build_bank after $2 96 $(cygpath -m $O/clean/banks)
else exec python -u -m w14_read_verb.pool $2 90 $(cygpath -m $O/clean/pool); fi
