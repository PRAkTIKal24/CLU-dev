cd /c/chlu/worktrees/CHLU-w5-delete
source /c/chlu/env_w5.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w5/launch-delete.txt)"
