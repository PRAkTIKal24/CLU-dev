"""fig paths."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ENGINE_PROBE = os.path.join("nclu", "paths.py")
ENGINE_TOPS = ("nclu",)


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
        assert os.path.realpath(f).startswith(root), (
            "engine module %s loaded from %s, not from %s" % (name, f, root)
        )
        n += 1
    return n
