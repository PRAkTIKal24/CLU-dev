cd /c/chlu/worktrees/CHLU-w9-maze
source /c/chlu/env_w9.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w9/launch-maze.txt)"
