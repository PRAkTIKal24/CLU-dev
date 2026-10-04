cd /c/chlu/worktrees/CHLU-w21-rel-newseeds && source /c/chlu/env_w3.sh
export CHLU_ROOT=C:/chlu/worktrees/CHLU-w21-rel-newseeds
export PYTHONPATH="C:/chlu/worktrees/CHLU-w21-rel-newseeds;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages;C:/chlu/worktrees/CHLU-w21-rel-newseeds/scripts"
export PYTHONUTF8=1
python -c "import pathlib,chlu; cf=str(pathlib.Path(chlu.__file__).resolve()).lower(); assert 'chlu-w21-rel-newseeds' in cf, cf; print('ENV OK', cf)" || return 1
