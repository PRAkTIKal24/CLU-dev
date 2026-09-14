"""Exact-footprint canvas helper.

Given the banked figure's pixel size (W0,H0) and the FINAL PRINTED width in inches,
choose figsize+dpi so the regenerated PNG has EXACTLY the banked aspect ratio
(integer pixel dimensions in the same reduced ratio), i.e. LaTeX places it in an
identical box at the same \\linewidth fraction.
"""
from math import gcd

def canvas(W0, H0, w_in, dpi_target=400.0):
    g = gcd(W0, H0)
    w0, h0 = W0 // g, H0 // g
    n = max(1, round(w_in * dpi_target / w0))
    w_px, h_px = n * w0, n * h0
    dpi = w_px / w_in
    return (w_in, h_px / dpi), dpi, (w_px, h_px)

if __name__ == "__main__":
    for name, W0, H0, w in [("v2f1", 1022, 559, 3.740), ("v2f2", 1430, 546, 4.730),
                            ("v2f3", 1856, 1408, 3.300), ("v5f1", 1690, 571, 3.300),
                            ("v5f2", 2400, 672, 4.070), ("v5f3", 3040, 672, 4.400)]:
        fs, dpi, px = canvas(W0, H0, w)
        print(f"{name}: banked {W0}x{H0} aspect {W0/H0:.6f} -> figsize {fs[0]:.4f}x{fs[1]:.4f} in, "
              f"dpi {dpi:.3f}, px {px[0]}x{px[1]}, aspect {px[0]/px[1]:.6f}")
