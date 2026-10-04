cd /c/chlu/worktrees/CHLU-w16-maze-wm
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w16/launch-maze-wm.txt)"
