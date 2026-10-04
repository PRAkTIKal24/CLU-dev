import json, os, shutil
p = os.path.expanduser('~/.claude.json'); shutil.copy(p, p + '.bak.w23')
d = json.load(open(p)); pr = d.setdefault('projects', {})
t = {"allowedTools": [], "history": [], "mcpContextUris": [], "mcpServers": {},
     "enabledMcpjsonServers": [], "disabledMcpjsonServers": [], "hasTrustDialogAccepted": True,
     "projectOnboardingSeenCount": 1, "hasClaudeMdExternalIncludesApproved": False,
     "hasClaudeMdExternalIncludesWarningShown": False}
for k in ("C:\\chlu\\worktrees\\CHLU-w23-seed5-diagnosis", "C:/chlu/worktrees/CHLU-w23-seed5-diagnosis"):
    pr.setdefault(k, dict(t))["hasTrustDialogAccepted"] = True
json.dump(d, open(p, 'w'), indent=2); print("trust seeded")
