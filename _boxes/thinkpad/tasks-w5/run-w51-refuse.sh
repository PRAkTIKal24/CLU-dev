cd /c/chlu/worktrees/CHLU-w51-refuse
source /c/chlu/env_w5.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w5/launch-w51-refuse.txt)"
