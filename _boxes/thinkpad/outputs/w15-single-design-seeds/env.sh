# w15 single-design-seeds env shim.
# env_w3.sh exports PYTHONPATH with MSYS-style paths ("/c/Users/...") which the
# native Windows python does not understand -> equinox/jax invisible.
# This shim sources env_w3.sh unchanged and then re-exports the SAME two entries
# in Windows form. No package set change; site-packages is identical.
source /c/chlu/env_w3.sh
export PYTHONPATH="$(cygpath -w "$CHLU_ROOT");$(cygpath -w /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages)"
# THE CACHE GENERATION. `env_w3.sh` unsets every CHLU_* and this box has no ambient
# value, so a run that forgets it silently reads the frozen bridge -- cache GENERATION 1,
# which w5-common section 3.8 forbids for any number, and which at bridge seeds 1 and 2
# holds only 9,288 rows against the pool's 131,072 indices. Same value as
# scripts/w12_launch.sh and scripts/w12b_launch.sh.
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export PYTHONUNBUFFERED=1
