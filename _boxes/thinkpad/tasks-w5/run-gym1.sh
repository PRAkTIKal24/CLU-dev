cd /c/chlu/worktrees/CHLU-w5-gym1
source /c/chlu/env_w5.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w5/launch-gym1.txt)"
