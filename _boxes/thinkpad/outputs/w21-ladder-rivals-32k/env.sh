cd /c/chlu/worktrees/CHLU-w21-ladder-rivals-32k
source /c/chlu/env_w3.sh
# env_w3.sh joins PYTHONPATH with ':' -- Windows python reads that as one bogus entry. Re-export with ';'.
export PYTHONPATH="C:/chlu/worktrees/CHLU-w21-ladder-rivals-32k;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"
export CHLU_ROOT_W=C:/chlu/worktrees/CHLU-w21-ladder-rivals-32k
# THE CACHE GENERATION (same value as w12/w12b/w15/w20 launchers): generation 2 only.
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
