import json, os, shutil
p = os.path.expanduser('~/.claude.json'); shutil.copy(p, p + '.bak.w21')
d = json.load(open(p)); pr = d.setdefault('projects', {})
tmpl = {"allowedTools": [], "history": [], "mcpContextUris": [], "mcpServers": {},
        "enabledMcpjsonServers": [], "disabledMcpjsonServers": [], "hasTrustDialogAccepted": True,
        "projectOnboardingSeenCount": 1, "hasClaudeMdExternalIncludesApproved": False,
        "hasClaudeMdExternalIncludesWarningShown": False}
n = 0
for t in ["words-full-test", "maze-t-clu-improve"]:
    for k in (f"C:\\chlu\\worktrees\\CHLU-w21-{t}", f"C:/chlu/worktrees/CHLU-w21-{t}"):
        pr.setdefault(k, dict(tmpl))["hasTrustDialogAccepted"] = True; n += 1
json.dump(d, open(p, 'w'), indent=2)
print("seeded", n, "keys")
for t in ["words-full-test", "maze-t-clu-improve"]:
    for k in (f"C:\\chlu\\worktrees\\CHLU-w21-{t}", f"C:/chlu/worktrees/CHLU-w21-{t}"):
        print("  ", k, "->", pr[k]["hasTrustDialogAccepted"])
