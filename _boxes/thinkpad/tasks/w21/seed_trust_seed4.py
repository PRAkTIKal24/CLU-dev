import json, os, shutil
p = os.path.expanduser('~/.claude.json'); shutil.copy(p, p + '.bak.seed4')
d = json.load(open(p)); pr = d.setdefault('projects', {})
tmpl = {"allowedTools": [], "history": [], "mcpContextUris": [], "mcpServers": {},
        "enabledMcpjsonServers": [], "disabledMcpjsonServers": [], "hasTrustDialogAccepted": True,
        "projectOnboardingSeenCount": 1, "hasClaudeMdExternalIncludesApproved": False,
        "hasClaudeMdExternalIncludesWarningShown": False}
for t in ("maze-t-seed4", "ladder-rivals-32k"):
    for k in ("C:\\chlu\\worktrees\\CHLU-w21-%s" % t, "C:/chlu/worktrees/CHLU-w21-%s" % t):
        pr.setdefault(k, dict(tmpl))["hasTrustDialogAccepted"] = True
json.dump(d, open(p, 'w'), indent=2)
print("seeded:", [k for k in pr if 'w21-maze-t-seed4' in k or 'w21-ladder' in k])
