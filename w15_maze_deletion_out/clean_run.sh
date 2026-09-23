#!/bin/sh
O=/c/chlu/outputs/w15-maze-deletion-2b; C=$O/clean_checkout
cd $C; source /c/chlu/env_w3.sh
export PYTHONPATH="$(cygpath -w $C);C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -c "import os,pathlib,chlu; cf=pathlib.Path(chlu.__file__).resolve(); assert str(cf).lower().startswith(str(pathlib.Path(os.environ['CHLU_ROOT']).resolve()).lower()), cf; print('ENVCHECK OK',cf)" || exit 3
exec python -u -m w15_maze_deletion.run_deletion --seed $1 --encoder train --tag clean --out $(cygpath -m $O/clean/runs) --banks $(cygpath -m $O/clean/banks) --pool $(cygpath -m $O/clean/pool)
