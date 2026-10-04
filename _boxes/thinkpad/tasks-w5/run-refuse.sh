cd /c/chlu/worktrees/CHLU-w5-refuse
source /c/chlu/env_w5.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w5/launch-refuse.txt)"
