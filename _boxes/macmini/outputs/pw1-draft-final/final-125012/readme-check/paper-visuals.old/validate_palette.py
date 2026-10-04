"""Python twin of the dataviz skill's scripts/validate_palette.js (categorical mode).

Same thresholds, same Machado-Oliveira-Fernandes (2009) severity-1.0 CVD matrices, same OKLab
Delta E x100. Ported because this machine has no JS runtime; the check must be run, not eyeballed.
usage: python validate_palette.py "#hex,#hex,..." [--mode light|dark] [--surface #hex] [--pairs adjacent|all]
Exit 1 on any hard FAIL (lightness band, chroma floor, CVD < 6, normal-vision < 15).
"""
import argparse
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def cbrt(x):
    return math.copysign(abs(x) ** (1 / 3), x)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l_ = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return [0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [max(0.0, min(1.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    rep, ok = [], True
    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    rep.append(("Lightness band", "PASS" if not off else "FAIL", off or f"all {len(palette)} inside L {lo}-{hi}"))
    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rep.append(("Chroma floor", "PASS" if not lowc else "FAIL", lowc or f"all >= {CHROMA_FLOOR}"))
    n = len(palette)
    pl = ([(i, j) for i in range(n) for j in range(i + 1, n)] if pairs == "all"
          else [(i, i + 1) for i in range(n - 1)])
    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j])
                 for k in ("protan", "deutan") for i, j in pl), default=None)
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pl), default=99)
    wd = worst[0] if worst else 99
    st = "PASS" if wd >= CVD_TARGET else "WARN" if wd >= CVD_FLOOR else "FAIL"
    ok &= st != "FAIL"
    rep.append(("CVD separation", st, f"worst {pairs} {worst[2]}<->{worst[3]} dE {wd:.1f} ({worst[1]}) tritan {tri:.1f}" if worst else "n/a"))
    nw = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pl), default=None)
    nd = nw[0] if nw else 99
    ok &= nd >= NORMAL_FLOOR
    rep.append(("Normal-vision floor", "PASS" if nd >= NORMAL_FLOOR else "FAIL",
                f"worst {pairs} {nw[1]}<->{nw[2]} dE {nd:.1f}" if nw else "n/a"))
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rep.append(("Contrast vs surface", "WARN" if low else "PASS",
                f"below 3:1 - relief required (visible labels or table view): {low}" if low else f"all >= 3:1"))
    return rep, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface")
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    a = ap.parse_args()
    pal = [c.strip() for c in a.palette.split(",") if c.strip()]
    rep, ok = validate(pal, a.mode, a.surface, a.pairs)
    print(f"Palette ({a.mode}, surface {a.surface or DEFAULT_SURFACE[a.mode]}, {a.pairs}): {len(pal)} slots")
    for name, st, det in rep:
        print(f"  [{st:4}] {name:22} {det}")
    print("  ->", "ALL CHECKS PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
