cd /Users/batmac/clu/worktrees/CHLU-pw1-draft-final
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/pw1/launch-draft-final.txt)"
