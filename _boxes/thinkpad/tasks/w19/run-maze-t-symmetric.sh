cd /c/chlu/worktrees/CHLU-w19-maze-t-symmetric
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w19/launch-maze-t-symmetric.txt)"
