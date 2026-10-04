cd /c/chlu/worktrees/CHLU-w51-gym2
source /c/chlu/env_w5.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w5/launch-w51-gym2.txt)"
