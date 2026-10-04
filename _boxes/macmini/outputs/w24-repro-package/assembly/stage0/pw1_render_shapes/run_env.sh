# Source from the worktree root.  env_bm.sh first, then the T7 split and the stage roots.
source /Users/batmac/clu/env_bm.sh
export W17_RC_STAGE=/Users/batmac/clu/stage
export CHLU_GEN_DECLIFT_DIR=$W17_RC_STAGE/outputs/gen-decoder-lift/winner
# the value-holdout split (factor_index.npz, value_cells.json) the T7 leg and w20 rows 3-4 were scored on
export CHLU_GEN_DATA_ROOT=$W17_RC_STAGE/outputs/data/gen
export CHLU_GEN_FEATURES_ROOT=$W17_RC_STAGE/outputs/spoke-a-rivals/cache_generation2
# classifier_dir() = BRIDGE_ROOT/../../gen-external-classifier/seed_N
export CHLU_GEN_BRIDGE_ROOT=$W17_RC_STAGE/outputs/_bridge/runs
export PW1_RS_OUT=/Users/batmac/clu/outputs/pw1-render-shapes
