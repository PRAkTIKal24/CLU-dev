#!/bin/sh
# usage: launch.sh <checkout> <seed> <encoder> <tag> <outdir>
cd "$1"
source /c/chlu/env_w3.sh
W=$(cygpath -w "$1")
export PYTHONPATH="$W;C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"
python -c "import os,pathlib,chlu; cf=pathlib.Path(chlu.__file__).resolve(); assert str(cf).lower().startswith(str(pathlib.Path(os.environ['CHLU_ROOT']).resolve()).lower()), cf; print('ENVCHECK OK',cf)" || exit 3
exec python -u -m w15_maze_deletion.run_deletion --seed $2 --encoder $3 --tag $4 --out "$5" $6
