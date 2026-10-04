# w15-crossover-seeds env shim.
# /c/chlu/env_w3.sh builds PYTHONPATH with a POSIX ':' separator; native Windows
# python needs ';' and Windows-form paths, so MSYS glues both entries into one
# broken path and equinox/jax vanish. Re-export explicitly after sourcing.
cd /c/chlu/worktrees/CHLU-w15-crossover-seeds
source /c/chlu/env_w3.sh
export PYTHONPATH="C:/chlu/worktrees/CHLU-w15-crossover-seeds;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"
