cd /c/chlu/worktrees/CHLU-w20-shapes-data-matched
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w20/launch-shapes-data-matched.txt)"
