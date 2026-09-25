# source from inside the worktree: env_w3.sh then re-join PYTHONPATH with ';' for Windows python
source /c/chlu/env_w3.sh
export PYTHONPATH="$(cygpath -m "$CHLU_ROOT");$(cygpath -m /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages)"
python -c "import os,pathlib,chlu; cf=pathlib.Path(chlu.__file__).resolve(); r=pathlib.Path(os.environ['CHLU_ROOT']).resolve(); assert str(cf).lower().startswith(str(r).lower()), cf; print('ENV OK', cf)"
