cd /c/chlu/worktrees/CHLU-w15-maze-deletion-2b
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w15/launch-maze-deletion-2b.txt)"
