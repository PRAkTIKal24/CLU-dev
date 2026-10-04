cd /c/chlu/worktrees/CHLU-w21-maze-t-clu-improve
source /c/chlu/env_w3.sh
export PYTHONPATH="$(cygpath -w "$CHLU_ROOT");C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages"
python -c "import pathlib,chlu; cf=str(pathlib.Path(chlu.__file__).resolve()).lower(); assert 'chlu-w21-maze-t-clu-improve' in cf, cf" || exit 1
