cd /c/chlu/worktrees/CHLU-w23-seed5-diagnosis
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w23/launch-seed5-diagnosis.txt)"
