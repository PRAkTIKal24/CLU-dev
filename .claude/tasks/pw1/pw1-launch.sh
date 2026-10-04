#!/usr/bin/env bash
# usage: bash ~/clu/tasks/pw1/pw1-launch.sh <slug> <base-ref>
set -uo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
S=$1; BASE=$2; SESS=pw1-$S; T=~/clu/tasks/pw1; WT=/Users/batmac/clu/worktrees/CHLU-pw1-$S; OD=/Users/batmac/clu/outputs/pw1-$S
tmux has-session -t "$SESS" 2>/dev/null && { echo "REFUSING $SESS: exists"; exit 1; }
[ -f "$T/$S.md" ] || { echo "NO BRIEF $T/$S.md"; exit 1; }
cd ~/clu/workspace
if [ ! -d "$WT" ]; then git worktree add -q "$WT" -b "pw1/$S" "$BASE" || { echo "WORKTREE FAIL"; exit 1; }; fi
mkdir -p "$OD"; [ -f "$OD/START_UTC.txt" ] && { echo "STALE START_UTC in $OD"; exit 1; }
sed "s/__SLUG__/$S/g" "$T/launch-template.txt" > "$T/launch-$S.txt"
cat > "$T/run-$S.sh" <<R
cd $WT
export PATH="\$HOME/.local/bin:/opt/homebrew/bin:\$PATH"
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "\$(cat $T/launch-$S.txt)"
R
python3 - "$WT" <<'P'
import json, os, shutil, sys
wt = sys.argv[1]; p = os.path.expanduser('~/.claude.json'); shutil.copy(p, p + '.bak-pw1')
d = json.load(open(p)); pr = d.setdefault('projects', {})
tmpl = {"allowedTools": [], "history": [], "mcpContextUris": [], "mcpServers": {}, "enabledMcpjsonServers": [],
        "disabledMcpjsonServers": [], "hasTrustDialogAccepted": True, "projectOnboardingSeenCount": 1,
        "hasClaudeMdExternalIncludesApproved": False, "hasClaudeMdExternalIncludesWarningShown": False}
pr.setdefault(wt, dict(tmpl))["hasTrustDialogAccepted"] = True
json.dump(d, open(p, 'w'), indent=2)
P
tmux new-session -d -s "$SESS" -c "$WT"
tmux send-keys -t "$SESS" "bash $T/run-$S.sh" Enter
echo "LAUNCHED $SESS on $(git -C "$WT" rev-parse --short HEAD) ($BASE)"
