"""Is the 0.3130 D oracle floor an artefact of the READOUT, or a property of the judge?

Four readouts of the same frozen judge on the same ground-truth in-between images:
  circ   circular posterior mean          (PREREG-0 s3, the registered one)
  lin    linear posterior mean            (the declared diagnostic)
  argmax the top-1 class's grid value     (what probe_accuracy uses)
  bracket  the posterior RESTRICTED to the two bracketing classes and renormalised,
           then a linear mean. j3 measured top1-in-bracket = 1.000 on all 7 brackets,
           so this readout throws away nothing the judge actually used -- it just stops
           the other eight classes from dragging the mean.

If `bracket` recovers the midpoint and the others do not, the judge's information is
present and the floor is a property of the ESTIMATOR, which is a fixable thing and the
Head should be told. If none recovers it, the floor is the judge.
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

DELTA = hc.DELTA


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    g, _, _ = hc.grid_from_h5()
    hv = np.asarray(g.values[hc.HUE_F], np.float64)
    judge, jinfo = hc.load_judge(0)
    recs = json.load(open(hc.VH / "row2_unseen.json", encoding="utf-8"))["records"]
    n = len(recs)

    lo_rows, hi_rows, h_req, h_lo, h_hi, i_lo, i_hi = [], [], [], [], [], [], []
    for r in recs:
        t = {x["factor"]: float(x["value"]) for x in r["request"]}
        m = re.search(r"stored values ([\d.]+) and ([\d.]+)", r["reason"])
        p, q = float(m.group(1)), float(m.group(2))
        fi = np.array([int(np.argmin(np.abs(np.asarray(g.values[k]) - t[g.names[k]])))
                       for k in range(6)], np.int64)
        a_, b_ = fi.copy(), fi.copy()
        ia, ib = int(np.argmin(np.abs(hv - p))), int(np.argmin(np.abs(hv - q)))
        a_[hc.HUE_F], b_[hc.HUE_F] = ia, ib
        lo_rows.append(int(hc.rows_of(a_[None, :])[0]))
        hi_rows.append(int(hc.rows_of(b_[None, :])[0]))
        h_req.append(t["object_hue"]); h_lo.append(p); h_hi.append(q)
        i_lo.append(ia); i_hi.append(ib)
    lo_rows = np.array(lo_rows); hi_rows = np.array(hi_rows)
    h_req = np.array(h_req); h_lo = np.array(h_lo); h_hi = np.array(h_hi)
    i_lo = np.array(i_lo); i_hi = np.array(i_hi)

    both = hc.read_images(np.concatenate([lo_rows, hi_rows]))
    Ia, Ib = both[:n], both[n:]
    ha, hb = hsv_of(Ia), hsv_of(Ib)
    step = (h_hi - h_lo)[:, None, None]
    rot = rotational_mask(ha, hb, step)
    oracle = construct(Ia, Ib, ha, rot, step, 0.5)
    log(f"{n} ground-truth midpoints built")

    import equinox as eqx, jax, jax.numpy as jnp
    from chlu.core.gen_bridge import model_dtype
    dt = model_dtype(judge)
    fwd = eqx.filter_jit(jax.vmap(judge))

    def probs(imgs):
        o = []
        for i in range(0, imgs.shape[0], 256):
            lg = fwd(jnp.asarray(imgs[i:i + 256], dt))
            o.append(np.asarray(jax.nn.softmax(lg[hc.HUE_F], -1), np.float64))
        return np.concatenate(o)

    P = probs(oracle)
    Plo, Phi_ = probs(Ia), probs(Ib)

    ar = np.arange(n)
    def readouts(p):
        w = np.stack([p[ar, i_lo], p[ar, i_hi]], 1)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        return {
            "circ": hc.circ_mean_hue(p, hv),
            "lin": hc.lin_mean_hue(p, hv),
            "argmax": hv[p.argmax(1)],
            "bracket": h_lo + w[:, 1] * (h_hi - h_lo),
        }

    ro, rl, rh = readouts(P), readouts(Plo), readouts(Phi_)
    out = {"what": "is the oracle floor an artefact of the READOUT?",
           "scope": hc.SCOPE, "judge": jinfo, "n_requests": n,
           "mass_on_the_two_bracket_classes_at_the_oracle": float(
               (P[ar, i_lo] + P[ar, i_hi]).mean()),
           "readouts": {}}
    log("")
    log(f"{'readout':<10} {'oracle HUEMAE':>14} {'lookup HUEMAE':>14} "
        f"{'span':>7} {'SNAPFRAC':>9} {'mean t':>8}")
    for k in ("circ", "lin", "argmax", "bracket"):
        eo = hc.cdist(ro[k], h_req).mean() / DELTA
        el = 0.5 * (hc.cdist(rl[k], h_req) + hc.cdist(rh[k], h_req)).mean() / DELTA
        sn = hc.snap_metrics(ro[k], h_lo, h_hi)
        out["readouts"][k] = {
            "oracle_HUEMAE_delta": float(eo), "lookup_HUEMAE_delta": float(el),
            "usable_span_delta": float(el - eo),
            "SNAPFRAC_null": float(sn["snap"].mean()),
            "mean_t": float(sn["t"].mean()), "sd_t": float(sn["t"].std()),
            "exceeds_declared_margin_0.2D": bool((el - eo) >= 0.2),
        }
        log(f"{k:<10} {eo:>14.4f} {el:>14.4f} {el-eo:>7.4f} "
            f"{sn['snap'].mean():>9.4f} {sn['t'].mean():>8.4f}")
    log("")
    log(f"mass the judge puts on the two bracket classes at the oracle: "
        f"{out['mass_on_the_two_bracket_classes_at_the_oracle']:.4f}")
    best = max(out["readouts"], key=lambda k: out["readouts"][k]["usable_span_delta"])
    out["best_readout"] = best
    out["verdict"] = (
        f"the widest usable span is {out['readouts'][best]['usable_span_delta']:.4f} D "
        f"({best}); the declared margin was 0.2 D"
    )
    log(out["verdict"])
    hc.jdump(out, hc.OUT / "runs" / "j7_readouts.json")
    log("wrote runs/j7_readouts.json")


if __name__ == "__main__":
    main()
