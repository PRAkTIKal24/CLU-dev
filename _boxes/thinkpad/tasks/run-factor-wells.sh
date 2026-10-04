cd /c/chlu/worktrees/CHLU-hub-factor-wells
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-factor-wells.txt)"
