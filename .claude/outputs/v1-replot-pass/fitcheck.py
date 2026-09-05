"""Report, in PRINTED points, the axes box and each legend's bbox for a rendered figure."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def report(fig, printed_width_in):
    r = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches()
    s = printed_width_in / fw       # canvas-in -> printed-in
    for i, ax in enumerate(fig.axes):
        ab = ax.get_window_extent(r).transformed(fig.dpi_scale_trans.inverted())
        print(f"  axes[{i}]: {ab.width*s*72:.1f} x {ab.height*s*72:.1f} pt printed")
        lg = ax.get_legend()
        if lg is not None:
            lb = lg.get_window_extent(r).transformed(fig.dpi_scale_trans.inverted())
            print(f"     legend: {lb.width*s*72:.1f} x {lb.height*s*72:.1f} pt "
                  f"({100*lb.width/ab.width:.1f}% of axes width, {100*lb.height/ab.height:.1f}% of height)")


def unclip(fig, margin_pt=2.0, iters=8):
    """Push subplot margins in until no artist is clipped by the canvas edge.
    Changes layout only; touches no data."""
    fw, fh = fig.get_size_inches()
    m = margin_pt / 72.0
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        sp = fig.subplotpars
        dl = max(0.0, m - bb.x0) / fw
        dr = max(0.0, bb.x1 - (fw - m)) / fw
        db = max(0.0, m - bb.y0) / fh
        dt = max(0.0, bb.y1 - (fh - m)) / fh
        if dl + dr + db + dt < 1e-4:
            return True
        fig.subplots_adjust(left=sp.left + dl, right=sp.right - dr,
                            bottom=sp.bottom + db, top=sp.top - dt)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    ok = bb.x0 >= -1e-6 and bb.y0 >= -1e-6 and bb.x1 <= fw + 1e-6 and bb.y1 <= fh + 1e-6
    print(f"  [unclip] residual tightbbox {bb.x0:.4f},{bb.y0:.4f} .. {bb.x1:.4f},{bb.y1:.4f} "
          f"vs canvas {fw:.4f} x {fh:.4f}  -> {'inside' if ok else 'STILL CLIPPED'}")
    return ok
