cd /c/chlu/worktrees/CHLU-w5-compose-field
source /c/chlu/env_w5.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w5/launch-compose-field.txt)"
