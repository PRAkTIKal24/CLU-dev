cd /Users/batmac/clu/worktrees/CHLU-pw2-draft-methods
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/pw2/launch-draft-methods.txt)"
