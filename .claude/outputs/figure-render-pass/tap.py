"""Data tap: record every numeric array handed to a matplotlib plotting call, so a
re-styled figure can be proved to plot exactly the same VALUES as the original.

Digests are order-insensitive (a sorted multiset of per-call hashes), so panels may
be re-ordered or re-labelled; only the numbers must match.  Text/labels/colors are
deliberately NOT recorded.
"""
import hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure

CALLS = []

def _h(obj):
    try:
        a = np.asarray(obj, dtype=float).ravel()
    except Exception:
        return None
    if a.size == 0:
        return "EMPTY"
    b = np.ascontiguousarray(a.astype("<f8")).tobytes()
    return hashlib.sha1(b).hexdigest()[:16]

def _rec(name, args):
    hs = [x for x in (_h(a) for a in args) if x is not None]
    if hs:
        CALLS.append(name.split(".")[-1] + ":" + ",".join(hs))

def _wrap(cls, name):
    orig = getattr(cls, name)
    def f(self, *args, **kw):
        _rec(name, args)
        return orig(self, *args, **kw)
    setattr(cls, name, f)

for _n in ("plot", "loglog", "semilogx", "semilogy", "scatter", "bar", "barh",
           "errorbar", "fill_between", "axhline", "axvline", "axhspan", "axvspan",
           "step", "stairs", "hist", "imshow", "pcolormesh", "contour", "contourf"):
    if hasattr(Axes, _n):
        _wrap(Axes, _n)

# never let a tapped run overwrite a banked artifact
_SAVE = Figure.savefig
def _savefig(self, fname, *a, **kw):
    import os
    tgt = os.path.join(TAP_SAVEDIR, os.path.basename(str(fname)))
    return _SAVE(self, tgt, *a, **kw)
TAP_SAVEDIR = "/tmp"
def redirect_saves(d):
    global TAP_SAVEDIR
    TAP_SAVEDIR = d
    Figure.savefig = _savefig

def digest():
    s = sorted(CALLS)
    return {"n_calls": len(s), "sha1": hashlib.sha1("|".join(s).encode()).hexdigest(), "calls": s}

def dump(path):
    json.dump(digest(), open(path, "w"), indent=1)
    d = digest()
    print(f"[tap] {d['n_calls']} data calls, digest {d['sha1']}  -> {path}")
