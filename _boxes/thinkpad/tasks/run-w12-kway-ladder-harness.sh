cd /c/chlu/worktrees/CHLU-w12-kway-ladder-harness
source /c/chlu/env_w3.sh
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w12-kway-ladder-harness.txt)"
