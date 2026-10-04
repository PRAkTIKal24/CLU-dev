"""STEP 1b: WHICH continuous readout, if any, has a usable interior.

The softmax posterior mean is near one-hot on real renders (bias ~1e-6), so it is
almost a quantiser and a genuine interpolation mostly falls out of the interior.
Three candidate readouts are compared on the SAME renders, all oracle-referenced:

  R1 softmax posterior mean over all 10 hue classes
  R2 RAW LOGIT DIFFERENCE of the two bracket classes  (unbounded -- cannot saturate)
  R3 two-class softmax over the bracket pair          (bounded, = sigmoid(R2))

The positive control is the feature-space midpoint; the negative control is a real
bracket feature fed back (a perfect lookup).  A readout is USABLE iff the control
separates from the lookup by a lot more than the readout's own noise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from chlu.core.gen_bridge import model_dtype  # noqa: E402
from chlu.eval.gen_probe import decode_features  # noqa: E402


def hue_logits(judge, feats, k_hue, batch=128):
    """(N, C) RAW logits of the hue head -- the softmax is NOT applied."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(W.PINS["feat_shape"]))
    dtype = model_dtype(judge.external)
    fwd = eqx.filter_jit(jax.vmap(judge.external))
    out = []
    for s in range(0, Z.shape[0], batch):
        e = min(s + batch, Z.shape[0])
        imgs = decode_features(judge.decoder, Z[s:e], batch)
        lg = fwd(jnp.asarray(imgs, dtype))[int(k_hue)]
        out.append(np.asarray(lg, np.float64))
    return np.concatenate(out)


def readouts(L, i_lo, i_hi, hue_vals):
    n = np.arange(L.shape[0])
    p = np.exp(L - L.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    r1 = p @ hue_vals
    r2 = L[n, i_hi] - L[n, i_lo]
    r3 = 1.0 / (1.0 + np.exp(-r2))
    return {"R1_softmax_mean": r1, "R2_logit_diff": r2, "R3_two_class": r3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()

    out = C.OUT / "runs" / f"readout_choice_s{a.seed}_n{a.n}.json"
    if out.exists():
        print(f"[readout] {out.name} exists -- skipping")
        return
    t0 = time.time()

    row2, grid, train_rows = C.build_row2(a.n, seed=0)
    recs = row2["records"]
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    J = C.SecondJudge(a.seed)
    F = W.features_mmap(a.seed)
    hue_vals = J.hue_vals
    i_lo = np.round(hues[:, 0] / 0.1).astype(int)
    i_hi = np.round(hues[:, 1] / 0.1).astype(int)

    f_lo = W.gather(F, rows[:, 0])
    f_hi = W.gather(F, rows[:, 1])
    f_mid = 0.5 * (f_lo + f_hi)

    L_lo = hue_logits(J.judge, f_lo, J.k_hue, a.batch)
    L_hi = hue_logits(J.judge, f_hi, J.k_hue, a.batch)
    L_md = hue_logits(J.judge, f_mid, J.k_hue, a.batch)

    # how one-hot is the classifier on REAL renders?
    def maxp(L):
        p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return p.max(1)
    conf = {"real_lo_maxprob_mean": float(maxp(L_lo).mean()),
            "real_hi_maxprob_mean": float(maxp(L_hi).mean()),
            "midpoint_maxprob_mean": float(maxp(L_md).mean())}
    print("[readout] confidence:", json.dumps(conf))

    res = {"seed": a.seed, "n_requests": len(recs), "confidence": conf, "readouts": {}}
    for name in ("R1_softmax_mean", "R2_logit_diff", "R3_two_class"):
        v_lo = readouts(L_lo, i_lo, i_hi, hue_vals)[name]
        v_hi = readouts(L_hi, i_lo, i_hi, hue_vals)[name]
        v_md = readouts(L_md, i_lo, i_hi, hue_vals)[name]
        span = v_hi - v_lo
        p_md = (v_md - v_lo) / span
        p_lo = (v_lo - v_lo) / span
        res["readouts"][name] = {
            "span_mean": float(span.mean()), "span_sd": float(span.std()),
            "span_min": float(span.min()), "frac_span_positive": float((span > 0).mean()),
            "midpoint_p_mean": float(p_md.mean()), "midpoint_p_sd": float(p_md.std()),
            "midpoint_p_median": float(np.median(p_md)),
            "midpoint_frac_interior_20_80": float(((p_md > .2) & (p_md < .8)).mean()),
            "midpoint_frac_interior_05_95": float(((p_md > .05) & (p_md < .95)).mean()),
            "lookup_p_mean": float(p_lo.mean()),
            "interior_quartiles": [float(np.quantile(p_md, q)) for q in (.1, .25, .5, .75, .9)],
        }
        print(f"[readout] {name:18s} span {span.mean():+.4g}±{span.std():.2g} "
              f"| midpoint p {p_md.mean():+.3f}±{p_md.std():.3f} "
              f"interior(.2,.8) {((p_md>.2)&(p_md<.8)).mean():.3f}")
    res["seconds"] = time.time() - t0
    C.jdump(res, out)
    print(f"[readout] wrote {out}")


if __name__ == "__main__":
    main()
