cd /c/chlu/worktrees/CHLU-w4-placement
source /c/chlu/env_w4.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w4/launch-placement.txt)"
