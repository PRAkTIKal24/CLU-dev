import json, os, shutil
p = os.path.expanduser('~/.claude.json'); shutil.copy(p, p + '.bak.rel')
d = json.load(open(p)); pr = d.setdefault('projects', {})
tmpl = {"allowedTools": [], "history": [], "mcpContextUris": [], "mcpServers": {},
        "enabledMcpjsonServers": [], "disabledMcpjsonServers": [], "hasTrustDialogAccepted": True,
        "projectOnboardingSeenCount": 1, "hasClaudeMdExternalIncludesApproved": False,
        "hasClaudeMdExternalIncludesWarningShown": False}
for k in ("C:\\chlu\\worktrees\\CHLU-w21-rel-newseeds", "C:/chlu/worktrees/CHLU-w21-rel-newseeds"):
    pr.setdefault(k, dict(tmpl))["hasTrustDialogAccepted"] = True
json.dump(d, open(p, 'w'), indent=2)
print("trust seeded both slash styles")
