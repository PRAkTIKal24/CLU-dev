#!/usr/bin/env python
"""w13 patch 7/7 -- figure polish found by rendering and LOOKING at fig6 and fig8:
 (a) fig6 panel A clipped its ceiling annotations and did not draw `pca_tensor` at all,
     which hides the ladder's strongest single finding (bit-identical at 1/18th the bytes);
 (b) fig8's pin annotation sat on top of a bar.
Exact-string patch; the generator is edited, never the png."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")


def sub(anchor, replacement):
    global t
    n = t.count(anchor)
    assert n == 1, "anchor occurs %d times, expected 1: %r" % (n, anchor[:110])
    t = t.replace(anchor, replacement)


sub('''    ax.set_xticks(ns)
    ax.set_xlabel("n  =  number of jointly composed factors  (the RUNG)")
    ax.set_ylabel("distinct valid novel tuples / corner")
    ax.set_title("A.  score vs n   -   K = 5 wells on every point, n_seeds = 3", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    recessive(ax)
    ax.annotate("ceiling 36\\n(60 corners)", (2, 36), xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=7.0, color=INK3)
    ax.annotate("ceiling 72  (120 corners, and for n>=3 the SAME 120)", (4, 72.6),
                ha="center", fontsize=7.0, color=INK3)
''',
    '''    pc = [[ptr(L, "/rungs/%d/arms/pca_tensor/per_seed/%d/mean" % (n, s)) for s in range(3)] for n in ns]
    ax.plot(ns, [sum(p) / 3 for p in pc], color=SURFACE, linewidth=1.1, linestyle=(0, (2, 2)),
            zorder=6, label="pca_tensor  -  BIT-IDENTICAL to the arm, at 1/18th the bytes")
    ax.plot([1.72, 2.28], [36, 36], color=INK3, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5)
    ax.plot([2.72, 6.28], [72, 72], color=INK3, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5)
    ax.annotate("ceiling 36 (60 corners)", (2, 36), xytext=(6, 3), textcoords="offset points",
                ha="left", fontsize=6.8, color=INK3)
    ax.annotate("ceiling 72 (120 corners, and for n>=3 the SAME 120)", (4.5, 72),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.8, color=INK3)
    ax.set_ylim(0, 80)
    ax.set_xticks(ns)
    ax.set_xlabel("n  =  number of jointly composed factors  (the RUNG)")
    ax.set_ylabel("distinct valid novel tuples / corner")
    ax.set_title("A.  score vs n   -   K = 5 wells on every point, n_seeds = 3", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    recessive(ax)
''')

sub('''    ax.annotate("524,288 B  =  1x  (the RETIRED pin, a reference column this wave)", (len(Ks) - 1, 1.0),
                xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=7.0, color=INK2)''',
    '''    ax.annotate("524,288 B  =  1x  (the RETIRED pin, a reference column this wave)", (-0.45, 1.0),
                xytext=(0, -12), textcoords="offset points", ha="left", fontsize=7.0, color=INK2)''')

SRC.write_text(t, encoding="utf-8")
print("p7 OK ->", SRC)
