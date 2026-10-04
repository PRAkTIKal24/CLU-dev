"""measure.py <pdf> [<marker-regex-for-block-end> ...]
Reports: n pages, per-block fractional page split using the figure-render-pass
instrument, printed image boxes, and the body-text end of the last page.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure-render-pass"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagesplit import words, frac_at, find, TOP, BOT
from imgbox import boxes

def run(pdf, marks):
    pgs = words(pdf)
    print(f"{pdf}: {len(pgs)} pages")
    prev = 0.0
    for name, pat in marks:
        i, w = find(pgs, pat)
        if w is None:
            print(f"  !! marker not found: {name} ({pat})"); continue
        cand = [x for x in pgs[i] if x[3] <= w[1] + 1]
        yend = max([x[3] for x in cand], default=TOP)
        f = frac_at(i, yend)
        print(f"  {name:<22} ends at {f:6.2f} pp  (block {f-prev:5.2f} pp)")
        prev = f
    last = [x for x in pgs[-1] if x[3] <= 740]
    ymax = max(x[3] for x in last) if last else TOP
    print(f"  END                    ends at {frac_at(len(pgs)-1, ymax):6.2f} pp   "
          f"(last-page body ends y={ymax:.2f}, slack {720-ymax:.2f} pt)")
    print("  images:")
    for (p, w, h, x, y) in boxes(pdf):
        print(f"    p{p:<3} {w:8.3f} x {h:8.3f} pt = {w/72:6.3f} x {h/72:6.3f} in")

if __name__ == "__main__":
    run(sys.argv[1], [tuple(m.split("=", 1)) for m in sys.argv[2:]])
