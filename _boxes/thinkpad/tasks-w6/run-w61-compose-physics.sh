cd /c/chlu/worktrees/CHLU-w61-compose-physics
source /c/chlu/env_w6.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w6/launch-w61-compose-physics.txt)"
