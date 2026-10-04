cd /c/chlu/worktrees/CHLU-w15-bar-d-score && source /c/chlu/env_w3.sh
# env_w3.sh joins PYTHONPATH with ':' which Windows python mis-parses (one bogus entry) -> equinox not found. Re-export with ';'.
export PYTHONPATH="C:/chlu/worktrees/CHLU-w15-bar-d-score;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"
