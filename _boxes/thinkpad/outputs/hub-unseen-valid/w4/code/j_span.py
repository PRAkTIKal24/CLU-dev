"""STEP 1 (the gate): MEASURE THE SPAN ON THE SECOND JUDGE.  ⛔ Before any arm number.

Three things are measured here, and nothing about any arm:
  A. the ORACLE ANCHOR SPAN -- what the judge reports for the two REAL stored renders
     that bracket each request.  This is the denominator every later number is in.
  B. INTERIOR REACHABILITY -- the positive control.  A feature-space midpoint of the
     two real bracketing features is a genuine interpolation that is not any arm.  If
     the judge puts it at the middle of the span, the span has a usable interior.  If
     it snaps to an end, the judge cannot score this row and this hub STOPS.
  C. the SNAP FLOOR -- a real bracketing feature fed back in.  A perfect lookup.  It
     must land at an end of the span, or the reference coordinate is not a coordinate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()

    out = C.OUT / "runs" / f"span_{a.tag}_s{a.seed}_n{a.n}.json"
    if out.exists():                      # resumable: never redo a finished leg
        print(f"[span] {out.name} exists -- skipping (resumable)")
        return
    t0 = time.time()

    row2, grid, train_rows = C.build_row2(a.n, seed=0)
    recs = row2["records"]
    print(f"[span] row2 regenerated: {len(recs)} requests")
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    cids = C.corner_ids(recs, grid, vals, hues)
    print(f"[span] corners realised: {len(set(cids))}")

    J = C.SecondJudge(a.seed)
    F = W.features_mmap(a.seed)

    # A. oracle anchors -----------------------------------------------------
    A = J.anchors(rows, a.batch)
    a_lo, a_hi = A[:, 0], A[:, 1]
    h_lo, h_hi = hues[:, 0], hues[:, 1]
    span = a_hi - a_lo
    print(f"[span] anchor span: mean {span.mean():+.4f} sd {span.std():.4f} "
          f"min {span.min():+.4f} max {span.max():+.4f} | true gap {(h_hi-h_lo).mean():.4f}")

    # B. interior reachability: the feature-space midpoint ------------------
    f_lo = W.gather(F, rows[:, 0])
    f_hi = W.gather(F, rows[:, 1])
    r_mid = J.readout(0.5 * (f_lo + f_hi), a.batch)
    mid = C.referenced(r_mid, a_lo, a_hi, h_lo, h_hi)

    # C. the snap floor: a real bracketing feature fed straight back --------
    r_lo = J.readout(f_lo, a.batch)
    snap = C.referenced(r_lo, a_lo, a_hi, h_lo, h_hi)

    res = {
        "tag": a.tag, "seed": a.seed, "n_requests": len(recs),
        "n_corners_realised": int(len(set(cids))),
        "anchor_span_readout": {
            "mean": float(span.mean()), "sd": float(span.std()),
            "min": float(span.min()), "max": float(span.max()),
            "frac_positive": float((span > 0).mean()),
            "true_hue_gap": float((h_hi - h_lo).mean()),
        },
        "anchor_calibration": {
            "readout_at_lo_mean": float(a_lo.mean()),
            "readout_at_hi_mean": float(a_hi.mean()),
            "bias_lo": float((a_lo - h_lo).mean()),
            "bias_hi": float((a_hi - h_hi).mean()),
        },
        "B_interior_midpoint": {
            "p_mean": float(np.mean(mid["p"])), "p_sd": float(np.std(mid["p"])),
            "p_median": float(np.median(mid["p"])),
            "hue_hat_err_mean": float(np.mean(np.abs(
                mid["hue_hat"] - 0.5 * (h_lo + h_hi)))),
            "frac_in_interior_20_80": float(np.mean((mid["p"] > 0.2) & (mid["p"] < 0.8))),
        },
        "C_snap_floor": {
            "p_mean": float(np.mean(snap["p"])), "p_sd": float(np.std(snap["p"])),
            "frac_below_0_2": float(np.mean(snap["p"] < 0.2)),
        },
        "seconds": time.time() - t0,
    }
    C.jdump(res, out)
    print(json.dumps(res, indent=1))
    print(f"[span] wrote {out} in {res['seconds']:.1f}s")


if __name__ == "__main__":
    main()
