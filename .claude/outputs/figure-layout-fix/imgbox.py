"""Printed image boxes, measured from a built PDF via `mutool draw -F trace`.
Reports each raster image's placed width/height in pt and in inches (72 pt/in).
"""
import re, subprocess, sys, math

def boxes(pdf):
    out = subprocess.run(["/opt/homebrew/bin/mutool", "draw", "-F", "trace", "-o", "-", pdf],
                         capture_output=True, text=True).stdout
    res, page = [], 0
    for line in out.splitlines():
        m = re.search(r'<page\b.*?number="(\d+)"', line)
        if m:
            page = int(m.group(1))
        m = re.search(r'<fill_image\b[^>]*transform="([-\d.e ]+)"', line)
        if m:
            a, b, c, d, e, f = [float(x) for x in m.group(1).split()]
            w = math.hypot(a, b); h = math.hypot(c, d)
            res.append((page, w, h, e, f))
    return res

if __name__ == "__main__":
    for pdf in sys.argv[1:]:
        print(pdf)
        for (p, w, h, x, y) in boxes(pdf):
            print(f"  p{p:<3} {w:9.3f} x {h:9.3f} pt  =  {w/72:6.3f} x {h/72:6.3f} in   at ({x:.1f},{y:.1f})")
