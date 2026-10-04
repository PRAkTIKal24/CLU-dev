"""Where the figure build finds its code, in either of its two layouts (PW-1 item 4).

Layout "repo":   this file sits in <CHLU repo>/w21_figures/; the task-figure scripts are in
                 <repo>/w19_task_figures/ and the maze engine (chlu, w8_envscout, w9_harness,
                 scripts/w16_cmaze_t.py) is the repo itself.
Layout "folder": this file sits in a flat figure folder (outputs/paper-visuals/, w17-paper/figures/,
                 or any copy of them); every script is next to it and the maze engine is the
                 folder's own snapshot in engine/ (written by tools/vendor_engine.py, pinned by
                 engine/ENGINE.json).
The engine root is, in order: $CHLU_FIG_CODE if set; the repo (layout "repo"); ./engine/.
Nothing here reads evidence; evidence stays at $CHLU_EVIDENCE (default ~/clu/evidence/outputs).
usage (shell): python figroot.py layout|code|script <repo-relative path>|check
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ENGINE_PROBE = os.path.join("chlu", "paths.py")
ENGINE_TOPS = ("chlu",)
def code_root():
    if os.path.isfile(os.path.join(PARENT, ENGINE_PROBE)):
        return PARENT
    raise SystemExit("figroot: the package root was not found above %s" % HERE)


def add_code_to_path():
    root = code_root()
    for p in (os.path.join(root, "scripts"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def check_loaded():
    """Every loaded engine module comes from the engine root; returns the count checked."""
    root = os.path.realpath(code_root()) + os.sep
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] not in ENGINE_TOPS:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        assert os.path.realpath(f).startswith(root), \
            "engine module %s loaded from %s, not from %s" % (name, f, root)
        n += 1
    return n


