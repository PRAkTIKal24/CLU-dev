cd /c/chlu/worktrees/CHLU-w13-encoder-adapt
source /c/chlu/env_w3.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w13-encoder-adapt.txt)"
