"""Shared bits of the round-2 generators that replace LOST generators.
R2_VALIDATE=1: build the figure only (validate.py then re-boxes the axes onto the
original PNG's pixel frames, applies the original's calibrated limits, and checks
every plotted vertex against the original's ink)."""
import os
VAL = os.environ.get("R2_VALIDATE") == "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
TL = os.path.join(_DATA, "t-lever-forgetting")
VG = os.path.join(_DATA, "v5-gate")
OUTDIR = os.environ.get("FRP_OUT", "/tmp")


def finish(fig, name, dpi):
    import style as S
    if VAL:
        return
    os.makedirs(OUTDIR, exist_ok=True)
    S.save(fig, os.path.join(OUTDIR, name), dpi)
    print("wrote", os.path.join(OUTDIR, name))
