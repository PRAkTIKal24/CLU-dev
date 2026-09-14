"""Original V2 Fig 3 generator, isolated: the figure block of
scratch/f1-gmor-condensate/analyze_and_figure.py run verbatim on the banked arrays
(the script's part (1) needs JAX + checkpoints; its only figure-side output is
`tr`, which was banked as angular_tilt_contrast.npz)."""
import os, re
import numpy as np
src = open(".claude/scratch/f1-gmor-condensate/analyze_and_figure.py").read()
block = src[src.index("# ------------------------------------------------------------------ (3) figure"):
            src.index("np.savez(os.path.join(OUT,")]
OUT = ".claude/outputs/f1-gmor-condensate"
DELTAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
EPS = float(np.finfo(np.float64).eps)
d = np.load(os.path.join(OUT, "gmor_condensate.npz"), allow_pickle=True)
tr = dict(np.load(os.path.join(OUT, "angular_tilt_contrast.npz"), allow_pickle=True))
tags = sorted(set(d["tag"].tolist()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(compile(block, "analyze_and_figure.py(figure block)", "exec"), globals())
