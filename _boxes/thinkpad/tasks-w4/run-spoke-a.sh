cd /c/chlu/worktrees/CHLU-w4-spoke-a
source /c/chlu/env_w4.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w4/launch-spoke-a.txt)"
