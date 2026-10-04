"""The INSTRUMENT-ANCHORED SCALE for row 2, both endpoints measured on real images.

The ceiling (j4) showed the judge commits each individual in-between image to one of its
two neighbouring classes, so a per-request distance to the NOMINAL requested value has a
floor of 0.3130 Delta and the snap statistic has a null of 0.6680. Both make the
registered row-2 metrics unusable.

This measures the remedy. For each of the 512 row-2 requests, three images are judged:

  ORACLE   the exact ground-truth in-between render (j3 construction) -- score 0
  LO / HI  the REAL dataset images at the two bracketing stored values -- what a
           PERFECT LOOKUP would render

and the proposed statistic is the arm's readout referenced to the ORACLE's readout on
the SAME request:  HUEMAE_ref = mean cdist(h_hat_arm, h_hat_oracle).

At the oracle it is exactly 0 by construction; at a perfect lookup it is whatever this
script measures. That is a calibrated scale with BOTH endpoints established on real
images, and it is the thing a future row-2 pre-registration needs.

No store, no arm, no fitted object is involved.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402
from j3_bracket import construct, hsv_of, rotational_mask  # noqa: E402

SCALE_TERCILE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
DELTA = hc.DELTA


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def judge_all(judge, imgs, hue_vals, batch=256):
    import equinox as eqx, jax, jax.numpy as jnp
    from chlu.core.gen_bridge import model_dtype
    dt = model_dtype(judge)
    fwd = eqx.filter_jit(jax.vmap(judge))
    hp = []
    for i in range(0, imgs.shape[0], batch):
        lg = fwd(jnp.asarray(imgs[i:i + batch], dt))
        hp.append(np.asarray(jax.nn.softmax(lg[hc.HUE_F], -1), np.float64))
    hp = np.concatenate(hp)
    return hp, hc.circ_mean_hue(hp, hue_vals)


def summ(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std()),
            "p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95))}


def main():
    g, _, _ = hc.grid_from_h5()
    hue_vals = np.asarray(g.values[hc.HUE_F], np.float64)
    judge, jinfo = hc.load_judge(0)

    d = json.load(open(hc.VH / "row2_unseen.json", encoding="utf-8"))
    recs = d["records"]
    n = len(recs)
    lo_rows, hi_rows, h_req, h_lo, h_hi, ck = [], [], [], [], [], []
    for r in recs:
        t = {x["factor"]: float(x["value"]) for x in r["request"]}
        m = re.search(r"stored values ([\d.]+) and ([\d.]+)", r["reason"])
        a, b = float(m.group(1)), float(m.group(2))
        fi = np.array([int(np.argmin(np.abs(np.asarray(g.values[k]) - t[g.names[k]])))
                       for k in range(6)], np.int64)
        fa, fb = fi.copy(), fi.copy()
        fa[hc.HUE_F] = int(np.argmin(np.abs(hue_vals - a)))
        fb[hc.HUE_F] = int(np.argmin(np.abs(hue_vals - b)))
        lo_rows.append(int(hc.rows_of(fa[None, :])[0]))
        hi_rows.append(int(hc.rows_of(fb[None, :])[0]))
        h_req.append(t["object_hue"]); h_lo.append(a); h_hi.append(b)
        ck.append(f"br{a:.1f}-{b:.1f}|sh{int(t['shape'])}|st{SCALE_TERCILE[int(fi[3])]}")
    lo_rows = np.array(lo_rows); hi_rows = np.array(hi_rows)
    h_req = np.array(h_req); h_lo = np.array(h_lo); h_hi = np.array(h_hi)
    ck = np.array(ck)

    both = hc.read_images(np.concatenate([lo_rows, hi_rows]))
    Ia, Ib = both[:n], both[n:]
    ha, hb = hsv_of(Ia), hsv_of(Ib)
    step = (h_hi - h_lo)[:, None, None]
    rot = rotational_mask(ha, hb, step)
    oracle = construct(Ia, Ib, ha, rot, step, 0.5)
    log(f"{n} requests; oracle midpoints built (rot px {rot.mean():.4f})")

    p_or, h_or = judge_all(judge, oracle, hue_vals)
    p_lo, hh_lo = judge_all(judge, Ia, hue_vals)
    p_hi, hh_hi = judge_all(judge, Ib, hue_vals)
    log("all three judged")

    # --- the proposed statistic, and its two calibrated endpoints -----------
    ref_oracle = hc.cdist(h_or, h_or)                       # identically 0
    ref_lo = hc.cdist(hh_lo, h_or)
    ref_hi = hc.cdist(hh_hi, h_or)
    ref_lookup = np.minimum(ref_lo, ref_hi)   # the FRIENDLIEST lookup per request
    ref_lookup_mean = 0.5 * (ref_lo + ref_hi)

    # --- and for comparison, the REGISTERED statistic on the same three -----
    nom_oracle = hc.cdist(h_or, h_req)
    nom_lo = hc.cdist(hh_lo, h_req)
    nom_hi = hc.cdist(hh_hi, h_req)

    sn_or = hc.snap_metrics(h_or, h_lo, h_hi)
    sn_lo = hc.snap_metrics(hh_lo, h_lo, h_hi)

    log("")
    log("REGISTERED statistic  cdist(h_hat, h_REQUESTED) / Delta")
    log(f"  oracle (perfect render)     {nom_oracle.mean()/DELTA:.4f}   <- the FLOOR")
    log(f"  real lo image (lookup)      {nom_lo.mean()/DELTA:.4f}")
    log(f"  real hi image (lookup)      {nom_hi.mean()/DELTA:.4f}")
    log(f"  => usable range             {nom_oracle.mean()/DELTA:.4f} .. "
        f"{0.5*(nom_lo+nom_hi).mean()/DELTA:.4f}  "
        f"(span {0.5*(nom_lo+nom_hi).mean()/DELTA - nom_oracle.mean()/DELTA:.4f} D)")
    log("")
    log("PROPOSED statistic    cdist(h_hat, h_hat_ORACLE) / Delta")
    log(f"  oracle                      {ref_oracle.mean()/DELTA:.4f}   <- 0 by construction")
    log(f"  real lo image (lookup)      {ref_lo.mean()/DELTA:.4f}")
    log(f"  real hi image (lookup)      {ref_hi.mean()/DELTA:.4f}")
    log(f"  mean of the two (lookup)    {ref_lookup_mean.mean()/DELTA:.4f}  <- the CEILING of the scale")
    log(f"  => usable range             0.0000 .. {ref_lookup_mean.mean()/DELTA:.4f} D")
    log("")
    log(f"SNAPFRAC null: oracle {sn_or['snap'].mean():.4f} | perfect lookup "
        f"{sn_lo['snap'].mean():.4f}")
    log(f"mean t: oracle {sn_or['t'].mean():.4f} (sd {sn_or['t'].std():.4f}) | "
        f"lookup-lo {sn_lo['t'].mean():.4f}")

    keys = sorted(set(ck.tolist()))
    pc_or = np.array([float(sn_or["t"][ck == k].mean()) for k in keys])
    pc_n = np.array([int((ck == k).sum()) for k in keys])
    log(f"per-corner mean t at the oracle over {len(keys)} corners: "
        f"mean {pc_or.mean():.4f} sd {pc_or.std():.4f} "
        f"(corner sizes {pc_n.min()}-{pc_n.max()})")

    hc.jdump({
        "what": "the INSTRUMENT-ANCHORED SCALE for row 2; both endpoints on real images",
        "scope": hc.SCOPE, "judge": jinfo, "n_requests": n, "n_corners": len(keys),
        "registered_statistic_cdist_to_requested_value_delta": {
            "oracle_FLOOR": float(nom_oracle.mean() / DELTA),
            "real_lo_lookup": float(nom_lo.mean() / DELTA),
            "real_hi_lookup": float(nom_hi.mean() / DELTA),
            "lookup_mean": float(0.5 * (nom_lo + nom_hi).mean() / DELTA),
            "usable_span_delta": float((0.5 * (nom_lo + nom_hi).mean() - nom_oracle.mean()) / DELTA),
            "verdict": "the floor is 0.313 D and the declared margin was 0.2 D",
        },
        "proposed_statistic_cdist_to_oracle_readout_delta": {
            "oracle": 0.0,
            "real_lo_lookup": float(ref_lo.mean() / DELTA),
            "real_hi_lookup": float(ref_hi.mean() / DELTA),
            "lookup_mean_CEILING": float(ref_lookup_mean.mean() / DELTA),
            "lookup_friendliest": float(ref_lookup.mean() / DELTA),
            "distribution_lookup_mean": summ(ref_lookup_mean / DELTA),
            "note": "0 at a perfect render by construction; the instrument is its own "
                    "control, so the judge's per-image commitment cancels request by "
                    "request instead of becoming a floor",
        },
        "snapfrac_nulls": {"oracle": float(sn_or["snap"].mean()),
                           "perfect_lookup": float(sn_lo["snap"].mean())},
        "t_at_oracle": {"mean": float(sn_or["t"].mean()), "sd": float(sn_or["t"].std())},
        "per_corner_mean_t_at_oracle": {"n_corners": len(keys),
                                        "mean": float(pc_or.mean()),
                                        "sd_over_corners": float(pc_or.std()),
                                        "corner_sizes": [int(x) for x in pc_n]},
    }, hc.OUT / "runs" / "j5_scale.json")
    log("wrote runs/j5_scale.json")


if __name__ == "__main__":
    main()
