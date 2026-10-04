cd /c/chlu/worktrees/CHLU-w10-seeds-and-tables
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w10-seeds-and-tables.txt)"
