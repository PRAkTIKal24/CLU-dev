"""Before/after sheets at PRINTED SIZE (300 dpi): what a reviewer actually sees."""
import sys, os
from PIL import Image
DPI = 300
def at_print(path, w_in):
    im = Image.open(path).convert("RGB")
    h_in = w_in * im.height / im.width
    return im.resize((int(round(w_in * DPI)), int(round(h_in * DPI))), Image.LANCZOS)
def sheet(old, w_old, new, w_new, out, labels=("BEFORE", "AFTER")):
    a, b = at_print(old, w_old), at_print(new, w_new)
    pad = 30
    W = max(a.width, b.width) + 2 * pad
    H = a.height + b.height + 4 * pad
    c = Image.new("RGB", (W, H), (255, 255, 255))
    c.paste(a, (pad, pad)); c.paste(b, (pad, a.height + 3 * pad))
    c.save(out); print("wrote", out, c.size)
if __name__ == "__main__":
    a=sys.argv
    sheet(a[1], float(a[2]), a[3], float(a[4]), a[5])
