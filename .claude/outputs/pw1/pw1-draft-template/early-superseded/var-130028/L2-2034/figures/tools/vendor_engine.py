#!/usr/bin/env python3
"""Snapshot the maze engine the figure build imports into <dest>/engine/ (PW-1 item 4).

The CMAZE-T panels and the task figure re-derive every drawn maze from the deterministic engine
(chlu.exp_w15_cmaze_t, scripts/w16_cmaze_t.py, w8_envscout, w9_harness). A flat figure folder is
not a CHLU repo, so it carries its own copy of exactly the engine files the build loads: the import
closure of those four modules, found by importing them in a clean subprocess and keeping every
loaded file under the source root. Files are copied byte for byte; engine/ENGINE.json records the
source commit and each file's SHA-256, and the copy is re-hashed against it.
usage: python vendor_engine.py <CHLU repo root> <dest folder>      (writes <dest>/engine/)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOTS = ("chlu.exp_w15_cmaze_t", "w16_cmaze_t", "w8_envscout.maze_colour", "w9_harness.dynamics")
PROBE = r"""
import os, sys, json
root = sys.argv[1]
sys.path[:0] = [os.path.join(root, "scripts"), root]
for m in %r:
    __import__(m)
out = set()
for mod in list(sys.modules.values()):
    f = getattr(mod, "__file__", None)
    if f and os.path.realpath(f).startswith(os.path.realpath(root) + os.sep):
        out.add(os.path.relpath(os.path.realpath(f), os.path.realpath(root)))
print(json.dumps(sorted(out)))
""" % (ROOTS,)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git(root, *a):
    try:
        return subprocess.run(["git", "-C", root] + list(a), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(root, dest):
    root = os.path.abspath(root)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    rels = json.loads(subprocess.run([sys.executable, "-c", PROBE, root], capture_output=True,
                                     text=True, check=True, env=env).stdout)
    rels = [r for r in rels if r.endswith(".py")]
    assert "chlu/exp_w15_cmaze_t.py" in rels, rels
    eng = os.path.join(dest, "engine")
    want = {r: sha(os.path.join(root, r)) for r in rels}
    try:
        have = json.load(open(os.path.join(eng, "ENGINE.json")))["files"]
        if have == want and all(sha(os.path.join(eng, r)) == h for r, h in have.items()):
            print("engine: unchanged, %d files -> %s" % (len(want), eng))
            return
    except (OSError, ValueError, KeyError):
        pass
    if os.path.exists(eng):
        # never delete: move the old snapshot aside (the caller keeps or discards it)
        os.rename(eng, eng + ".superseded-%d" % os.getpid())
    files = {}
    for r in rels:
        os.makedirs(os.path.join(eng, os.path.dirname(r)), exist_ok=True)
        shutil.copyfile(os.path.join(root, r), os.path.join(eng, r))
        files[r] = sha(os.path.join(root, r))
        assert sha(os.path.join(eng, r)) == files[r], r
    head = git(root, "rev-parse", "HEAD")
    dirty = git(root, "status", "--porcelain", "--", *rels)
    rec = dict(what="maze-engine snapshot for the figure build (figroot.py resolves it)",
               written_by="w21_figures/tools/vendor_engine.py", import_roots=list(ROOTS),
               source_commit=head, source_clean=(dirty == "") if dirty is not None else None,
               n_files=len(files), files=files)
    with open(os.path.join(eng, "ENGINE.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    print("engine: %d files from %s @ %s (clean=%s) -> %s" % (len(files), root, head,
                                                              rec["source_clean"], eng))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
