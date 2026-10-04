cd /c/chlu/worktrees/CHLU-w6-compose-physics
source /c/chlu/env_w6.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w6/launch-compose-physics.txt)"
