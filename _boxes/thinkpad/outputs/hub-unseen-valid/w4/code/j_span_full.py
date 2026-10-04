"""STEP 2: THE MEASURED SPAN, at scale, on 3 seeds, per corner.  ⛔ Still no arm.

The second judge is R2: the RAW LOGIT DIFFERENCE of the two bracket hue classes,
referenced to the two oracle anchors.  p = 0 is the low stored value, p = 1 the high
one, p = 0.5 is the requested point.  The per-request score is

    e = |p - 0.5|      0 = the requested point exactly, 0.5 = a perfect lookup.

Two reference objects bound the span, and NEITHER is an arm:
  * the LOOKUP FLOOR   -- a real bracket feature fed back.  e = 0.5.
  * the REFERENCE INTERPOLATOR -- the feature-space midpoint of the two real
    bracketing features.  A genuine interpolation, available to no store.
The SPAN is floor - interpolator, measured, paired by corner.

A CALIBRATION SWEEP over t in {0, .25, .5, .75, 1} checks the judge resolves POSITION
and not merely "not an endpoint": p(t) must rise monotonically with t.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from j_readout_choice import hue_logits  # noqa: E402

TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def position(L, i_lo, i_hi):
    n = np.arange(L.shape[0])
    return L[n, i_hi] - L[n, i_lo]


def per_corner(cids, e):
    out = []
    for c in sorted(set(cids)):
        m = cids == c
        out.append({"corner_id": c, "n": int(m.sum()), "mean_e": float(e[m].mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--gen2", action="store_true")
    a = ap.parse_args()

    gen = 2 if a.gen2 else 1
    out = C.OUT / "runs" / f"span_full_g{gen}_s{a.seed}_n{a.n}.json"
    if out.exists():
        print(f"[spanfull] {out.name} exists -- skipping (resumable)")
        return
    t0 = time.time()
    prov = C.use_cache_generation2((a.seed,)) if a.gen2 else {"cache_generation": 1}
    print(f"[spanfull] cache generation {prov['cache_generation']}")

    row2, grid, train_rows = C.build_row2(a.n, seed=0)
    recs = row2["records"]
    # the superset assertion, run before any number
    import json as _j
    fr = _j.load(open(W.data_root() / "value_holdout" / "row2_unseen.json",
                      encoding="utf-8"))["records"]
    assert len(recs) >= len(fr), "regenerated row 2 is SMALLER than the frozen file"
    assert all(_j.dumps(recs[i], sort_keys=True) == _j.dumps(fr[i], sort_keys=True)
               for i in range(len(fr))), "regenerated row 2 is NOT a superset of the frozen file"
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    cids = C.corner_ids(recs, grid, vals, hues)
    n_corners = len(set(cids))
    assert n_corners >= 49, f"only {n_corners} corners realised, the estimand needs >= 49"
    print(f"[spanfull] seed {a.seed}: {len(recs)} requests, {n_corners} corners, "
          f"superset of the frozen {len(fr)} ASSERTED")

    J = C.SecondJudge(a.seed)
    F = W.features_mmap(a.seed)
    i_lo = np.round(hues[:, 0] / 0.1).astype(int)
    i_hi = np.round(hues[:, 1] / 0.1).astype(int)
    f_lo = W.gather(F, rows[:, 0])
    f_hi = W.gather(F, rows[:, 1])

    sweep = {}
    for t in TS:
        if time.time() - t0 > a.max_seconds:
            print(f"[spanfull] wall-clock cap hit before t={t}")
            break
        L = hue_logits(J.judge, (1.0 - t) * f_lo + t * f_hi, J.k_hue, a.batch)
        sweep[t] = position(L, i_lo, i_hi)
        print(f"[spanfull]   t={t:.2f} done ({time.time()-t0:.0f}s)")

    v_lo, v_hi = sweep[0.0], sweep[1.0]
    span_logit = v_hi - v_lo

    def p_of(v):
        return (v - v_lo) / span_logit

    cal = {f"{t:.2f}": {"p_mean": float(p_of(v).mean()), "p_sd": float(p_of(v).std()),
                        "p_median": float(np.median(p_of(v)))}
           for t, v in sweep.items()}
    pm = [cal[f"{t:.2f}"]["p_mean"] for t in TS if f"{t:.2f}" in cal]
    monotone = bool(all(pm[i] < pm[i + 1] for i in range(len(pm) - 1)))

    e_mid = np.abs(p_of(sweep[0.5]) - 0.5)
    e_look = np.abs(p_of(v_lo) - 0.5)          # the perfect lookup: exactly 0.5
    pc_mid = per_corner(cids, e_mid)
    pc_look = per_corner(cids, e_look)
    d = np.array([x["mean_e"] for x in pc_look]) - np.array([x["mean_e"] for x in pc_mid])

    res = {
        "seed": a.seed, "cache": prov, "n_requests": len(recs), "n_corners": n_corners,
        "superset_of_frozen": True, "n_frozen": len(fr),
        "readout": "R2 raw logit difference, oracle-referenced",
        "anchor_span_logits": {"mean": float(span_logit.mean()),
                               "sd": float(span_logit.std()),
                               "min": float(span_logit.min()),
                               "frac_positive": float((span_logit > 0).mean())},
        "calibration_sweep": cal, "calibration_monotone": monotone,
        "lookup_floor_e": {"mean": float(e_look.mean()), "sd": float(e_look.std())},
        "reference_interpolator_e": {"mean": float(e_mid.mean()), "sd": float(e_mid.std()),
                                     "median": float(np.median(e_mid))},
        "SPAN_per_corner": {
            "mean": float(d.mean()), "sd": float(d.std()),
            "min": float(d.min()), "max": float(d.max()),
            "n_corners_positive": int((d > 0).sum()),
            "sem": float(d.std(ddof=1) / np.sqrt(d.size)),
        },
        "per_corner_lookup": pc_look, "per_corner_interpolator": pc_mid,
        "seconds": time.time() - t0,
    }
    C.jdump(res, out)
    print(f"[spanfull] seed {a.seed}: anchor span {span_logit.mean():.2f} logits | "
          f"lookup e {e_look.mean():.4f} | interpolator e {e_mid.mean():.4f} | "
          f"SPAN {d.mean():.4f} +/- {res['SPAN_per_corner']['sem']:.4f} "
          f"({(d>0).sum()}/{d.size} corners) | monotone={monotone} | "
          f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
