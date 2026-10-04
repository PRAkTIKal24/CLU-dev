"""Cache the ORACLE readout for a row-2 request file, so every arm can be scored on
both the REGISTERED metric (distance to the nominal requested value) and the PROPOSED
instrument-anchored metric (distance to the oracle's readout on the same request).

Also caches the PERFECT-LOOKUP readouts (the two real bracket images), which are the
other calibrated endpoint of the scale.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402
from j3_bracket import construct, hsv_of, rotational_mask  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default=str(hc.OUT / "runs" / "row2_unseen_n4096.json"))
    ap.add_argument("--out", default=str(hc.OUT / "runs" / "oracle_n4096.npz"))
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()

    g, _, _ = hc.grid_from_h5()
    hue_vals = np.asarray(g.values[hc.HUE_F], np.float64)
    judge, jinfo = hc.load_judge(0)
    recs = json.load(open(a.requests, encoding="utf-8"))["records"]
    n = len(recs)

    lo_rows, hi_rows, h_req, h_lo, h_hi = [], [], [], [], []
    for r in recs:
        t = {x["factor"]: float(x["value"]) for x in r["request"]}
        m = re.search(r"stored values ([\d.]+) and ([\d.]+)", r["reason"])
        p, q = float(m.group(1)), float(m.group(2))
        fi = np.array([int(np.argmin(np.abs(np.asarray(g.values[k]) - t[g.names[k]])))
                       for k in range(6)], np.int64)
        fa, fb = fi.copy(), fi.copy()
        fa[hc.HUE_F] = int(np.argmin(np.abs(hue_vals - p)))
        fb[hc.HUE_F] = int(np.argmin(np.abs(hue_vals - q)))
        lo_rows.append(int(hc.rows_of(fa[None, :])[0]))
        hi_rows.append(int(hc.rows_of(fb[None, :])[0]))
        h_req.append(t["object_hue"]); h_lo.append(p); h_hi.append(q)
    lo_rows = np.array(lo_rows); hi_rows = np.array(hi_rows)
    h_req = np.array(h_req); h_lo = np.array(h_lo); h_hi = np.array(h_hi)
    log(f"{n} requests; reading {2*n} real bracket rows")

    import equinox as eqx, jax, jax.numpy as jnp
    from chlu.core.gen_bridge import model_dtype
    dt = model_dtype(judge)
    fwd = eqx.filter_jit(jax.vmap(judge))

    def judge_imgs(imgs):
        out_p, out_pred = [], []
        for i in range(0, imgs.shape[0], 256):
            lg = fwd(jnp.asarray(imgs[i:i + 256], dt))
            out_p.append(np.asarray(jax.nn.softmax(lg[hc.HUE_F], -1), np.float64))
            out_pred.append(np.stack([np.asarray(jnp.argmax(l, -1)) for l in lg], 1))
        return np.concatenate(out_p), np.concatenate(out_pred).astype(np.int32)

    h_or = np.zeros(n); h_ll = np.zeros(n); h_hh = np.zeros(n)
    pred_or = np.zeros((n, 6), np.int32)
    rotfrac = []
    # chunked so peak memory stays bounded (3 x chunk x 3 x 64 x 64 floats)
    for s in range(0, n, a.chunk):
        e = min(n, s + a.chunk)
        both = hc.read_images(np.concatenate([lo_rows[s:e], hi_rows[s:e]]))
        k = e - s
        Ia, Ib = both[:k], both[k:]
        ha, hb = hsv_of(Ia), hsv_of(Ib)
        step = (h_hi[s:e] - h_lo[s:e])[:, None, None]
        rot = rotational_mask(ha, hb, step)
        rotfrac.append(float(rot.mean()))
        orc = construct(Ia, Ib, ha, rot, step, 0.5)
        p, pr = judge_imgs(orc)
        h_or[s:e] = hc.circ_mean_hue(p, hue_vals); pred_or[s:e] = pr
        p, _ = judge_imgs(Ia); h_ll[s:e] = hc.circ_mean_hue(p, hue_vals)
        p, _ = judge_imgs(Ib); h_hh[s:e] = hc.circ_mean_hue(p, hue_vals)
        log(f"  {e}/{n}")

    np.savez_compressed(a.out, h_oracle=h_or, h_lookup_lo=h_ll, h_lookup_hi=h_hh,
                        pred_oracle=pred_or, h_req=h_req, h_lo=h_lo, h_hi=h_hi,
                        lo_rows=lo_rows, hi_rows=hi_rows,
                        rot_frac=np.array(rotfrac))
    D = hc.DELTA
    log(f"oracle HUEMAE (registered metric) {hc.cdist(h_or, h_req).mean()/D:.4f} D")
    log(f"lookup HUEMAE (registered metric) "
        f"{0.5*(hc.cdist(h_ll,h_req)+hc.cdist(h_hh,h_req)).mean()/D:.4f} D")
    log(f"lookup HUEMAE_ref (proposed)      "
        f"{0.5*(hc.cdist(h_ll,h_or)+hc.cdist(h_hh,h_or)).mean()/D:.4f} D")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
