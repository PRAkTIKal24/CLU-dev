cd /c/chlu/worktrees/CHLU-w10-referee
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w10-referee.txt)"
