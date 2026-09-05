"""Page-split instrument: fractional pages of each block, measured from PDF word
bounding boxes against the text block (top 72 pt, bottom 720 pt, page 792 pt).
Same instrument both variant BUILD-NOTEs used.
"""
import re, subprocess, sys, html

TOP, BOT, PAGEH = 72.0, 720.0, 792.0

def words(pdf):
    out = subprocess.run(["/opt/homebrew/bin/pdftotext", "-bbox", pdf, "-"],
                         capture_output=True, text=True).stdout
    pages, cur = [], None
    for line in out.splitlines():
        if "<page " in line:
            cur = []
            pages.append(cur)
        m = re.search(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*)</word>', line)
        if m and cur is not None:
            cur.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                        float(m.group(4)), html.unescape(m.group(5))))
    return pages

def find(pages, pat, start=0):
    rx = re.compile(pat)
    for i in range(start, len(pages)):
        for w in pages[i]:
            if rx.fullmatch(w[4]):
                return i, w
    return None, None

def frac_at(page_idx, ymax):
    return page_idx + max(0.0, min(1.0, (ymax - TOP) / (BOT - TOP)))

def report(pdf, marks):
    pgs = words(pdf)
    n = len(pgs)
    print(f"{pdf}: {n} pages")
    prev = 0.0
    for name, pat in marks:
        i, w = find(pgs, pat)
        if w is None:
            print(f"  !! marker not found: {name} ({pat})"); continue
        # block ENDS just before this marker: take the last word on the page above it
        cand = [x for x in pgs[i] if x[3] <= w[1] + 1]
        yend = max([x[3] for x in cand], default=TOP)
        f = frac_at(i, yend)
        print(f"  {name:<28} ends at {f:6.2f} pp   (block = {f - prev:5.2f} pp)")
        prev = f
    # last page bottom
    last = max(x[3] for x in pgs[-1]) if pgs[-1] else TOP
    f = frac_at(n - 1, last)
    print(f"  {'END':<28} ends at {f:6.2f} pp   (block = {f - prev:5.2f} pp)")
    return n

if __name__ == "__main__":
    pdf = sys.argv[1]
    marks = [(a, b) for a, b in (m.split("=", 1) for m in sys.argv[2:])]
    report(pdf, marks)
