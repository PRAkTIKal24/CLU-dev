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
ENGINE_PROBE = os.path.join("chlu", "exp_w15_cmaze_t.py")
#: top-level modules that must come from the engine root (never from another checkout on sys.path,
#: e.g. the venv's editable chlu install)
ENGINE_TOPS = ("chlu", "w8_envscout", "w9_harness", "w15_cmaze_t_run", "w16_cmaze_t")


def layout():
    if os.path.basename(HERE) == "w21_figures" and os.path.isfile(os.path.join(PARENT, ENGINE_PROBE)):
        return "repo"
    return "folder"


def code_root():
    env = os.environ.get("CHLU_FIG_CODE")
    cands = [("$CHLU_FIG_CODE", os.path.abspath(env))] if env else []
    if layout() == "repo":
        cands.append(("repo", PARENT))
    cands.append(("engine/", os.path.join(HERE, "engine")))
    for why, c in cands:
        if os.path.isfile(os.path.join(c, ENGINE_PROBE)):
            return c
        if why == "$CHLU_FIG_CODE":
            break
    raise SystemExit(
        "figroot: no maze engine found (tried %s). The CMAZE-T figures need the engine code: a "
        "folder copy carries it in engine/ (README.md, 'Rebuild'); or set CHLU_FIG_CODE to a CHLU "
        "checkout at f571b2b or later." % ", ".join("%s=%s" % c for c in cands))


def script(rel):
    """A build script named by its repo-relative path (as PROVENANCE.json records it)."""
    if layout() == "repo":
        return os.path.join(PARENT, rel)
    return os.path.join(HERE, os.path.basename(rel))


def engine_file(rel):
    """An engine source named by its repo-relative path (chlu/..., w8_envscout/..., ...)."""
    return os.path.join(code_root(), rel)


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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "code"
    if cmd == "layout":
        print(layout())
    elif cmd == "code":
        print(code_root())
    elif cmd == "script":
        print(script(sys.argv[2]))
    elif cmd == "check":
        add_code_to_path()
        import chlu.exp_w15_cmaze_t  # noqa: F401
        import w16_cmaze_t  # noqa: F401
        import w8_envscout.maze_colour  # noqa: F401
        import w9_harness.dynamics  # noqa: F401
        print("engine OK: %s (%s layout), %d modules from it" % (code_root(), layout(),
                                                                   check_loaded()))
    else:
        raise SystemExit(__doc__)
