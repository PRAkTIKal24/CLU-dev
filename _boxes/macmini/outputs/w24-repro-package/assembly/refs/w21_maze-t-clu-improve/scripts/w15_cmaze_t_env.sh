# w15/cmaze-t-harness environment shim.
#   cd /c/chlu/worktrees/CHLU-w15-cmaze-t-harness && source scripts/w15_cmaze_t_env.sh
#
# env_w3.sh builds PYTHONPATH with a POSIX ':' separator and an msys '/c/...' path.
# Windows CPython splits PYTHONPATH on ';', so that string is one bogus entry and
# equinox/jax are not importable.  This shim sources env_w3.sh unchanged (so every
# CHLU_* variable is exactly what the wave prescribes) and then re-exports
# PYTHONPATH with the Windows separator and Windows paths, worktree FIRST.
source /c/chlu/env_w3.sh
export PYTHONPATH='C:/chlu/worktrees/CHLU-w15-cmaze-t-harness;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages'
export W15_OUT=/c/chlu/outputs/w15-cmaze-t-harness
python -c "import os,pathlib,chlu,jax,numpy,equinox; cf=pathlib.Path(chlu.__file__).resolve(); assert str(cf).lower().startswith(str(pathlib.Path(os.environ['CHLU_ROOT']).resolve()).lower()), cf; print('ENV OK', cf, 'jax', jax.__version__, 'numpy', numpy.__version__, 'eqx', equinox.__version__)"
