cd /Users/batmac/clu/worktrees/CHLU-w17-rendered-compositions
source /Users/batmac/clu/env_bm.sh
export W17_RC_STAGE=/Users/batmac/clu/stage
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w17/launch-rendered-compositions.txt)

--- RESUME INSTRUCTION ---
$(cat /Users/batmac/clu/tasks/w17/relaunch-rendered-compositions.txt)"
