"""WHY the settle does nothing: two structural diagnostics, no decoding required.

H1  The launch point is ALREADY the valley bottom.  At a row-2 request the two
    bracketing values enter with EQUAL weight, so any energy symmetric in the two
    bracketing atoms has its minimum exactly at the chord midpoint -- which is where
    the linear span read already puts it.  Friction then has nothing to remove.
H2  "The two bracketing atoms" is not a structure this store HAS.  The head's
    coefficients are dense over all 64 modes, so a request does not select a pair of
    atoms to bracket between; it mixes the whole well.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import valley as V  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402

if __name__ == "__main__":
    W.PINS["condition"] = "val"
    from chlu.core.gen_request_head import head_coefficients, head_read, request_design

    row2, grid, train_rows = C.build_row2(8192, seed=0)
    recs = row2["records"][4096:4096 + 512]
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    obj = W.build_arm("factor_aligned_balanced", C.OUT / "objects" / "val", log=print)
    store, heads = obj["store"], obj["heads"]
    Phi, _ = request_design(vals, grid.values, obj["spec"])
    z0 = head_read(heads, store, Phi)

    print("\n=== H2: how concentrated are the head's coefficients? ===")
    Cc = head_coefficients(heads, Phi)
    for k, c in enumerate(Cc):
        a = np.abs(c)
        w = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
        pr = 1.0 / np.maximum((w ** 2).sum(1), 1e-12)     # participation ratio
        top2 = np.sort(w, axis=1)[:, -2:].sum(1)
        print(f"  well {k}: M=64  participation_ratio={pr.mean():6.2f}  "
              f"top-2 mass={top2.mean():.4f}  top-1 mass={np.sort(w,1)[:,-1].mean():.4f}")

    print("\n=== H1: does the settle move at all? (rho=1, lam=0) ===")
    z, resid = V.valley_read(heads, store, Phi, z0, 1.0, lam=0.0)
    d = V.settle_diagnostics(store, Phi, heads, z0, z, None, None)
    for r in d["wells"]:
        print(f"  well {r['well']}: moved/spacing={r['moved_over_spacing']:.5f}  "
              f"d_near/spacing={r['d_near_over_spacing']:.3f}  "
              f"snapped<5%={r['frac_snapped_within_5pct']:.3f}")
    print(f"  max_frac_snapped={d['max_frac_snapped']:.4f}  "
          f"min_d_near_over_spacing={d['min_d_near_over_spacing']:.3f}")

    print("\n=== H1b: is z0 symmetric between the two bracketing requests? ===")
    v_lo = vals.copy(); v_hi = vals.copy()
    k_hue = grid.index_of("object_hue")
    v_lo[:, k_hue] = hues[:, 0]; v_hi[:, k_hue] = hues[:, 1]
    P_lo, _ = request_design(v_lo, grid.values, obj["spec"])
    P_hi, _ = request_design(v_hi, grid.values, obj["spec"])
    z_lo = head_read(heads, store, P_lo); z_hi = head_read(heads, store, P_hi)
    chord = 0.5 * (z_lo + z_hi)
    rel = np.linalg.norm(z0 - chord, axis=1) / np.maximum(
        np.linalg.norm(z_hi - z_lo, axis=1), 1e-12)
    print(f"  ||z0 - chord_midpoint|| / ||z_hi - z_lo|| = {rel.mean():.5f} "
          f"(0 means the span read IS the chord midpoint)")
    C.jdump({"participation_ratio": [float(np.mean(1.0/np.maximum(((np.abs(c)/np.maximum(np.abs(c).sum(1,keepdims=True),1e-12))**2).sum(1),1e-12))) for c in Cc],
             "settle": d, "z0_is_chord_midpoint_rel_error": float(rel.mean()),
             "settle_residual": resid},
            C.OUT / "runs" / "why_null.json")
