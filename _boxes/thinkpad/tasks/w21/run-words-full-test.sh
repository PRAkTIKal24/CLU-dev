cd /c/chlu/worktrees/CHLU-w21-words-full-test
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w21/launch-words-full-test.txt)"
