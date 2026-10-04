cd /c/chlu/worktrees/CHLU-w21-maze-t-seed4
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w21/launch-maze-t-seed4.txt)"
