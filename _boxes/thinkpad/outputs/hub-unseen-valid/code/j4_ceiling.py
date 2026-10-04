"""THE ORACLE CEILING for scoreboard row 2, on the real 512-request query set.

Every row-2 request's two bracketing rows are in the store's fit pool, so both exist in
the dataset. For each request I build its EXACT ground-truth in-between image with the
construction validated in j3 (on-grid pixel MAE 0.0037, judge agreement 1.0000) and push
it through the frozen external classifier.

The result is the score a PERFECT arm would get -- one that renders the requested
in-between value exactly. It bounds every arm on this row without running one, and it is
measured per corner on the same 63-corner structure PREREG-0 s4 registered.

It also reports the SNAP (PREREG-0 s3) on ground truth, which is the judge's OWN
contraction toward the stored values with no arm and no store in the room.
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


def main():
    g, _, _ = hc.grid_from_h5()
    hue_vals = np.asarray(g.values[hc.HUE_F], np.float64)
    judge, jinfo = hc.load_judge(0)
    log(f"judge {jinfo['param_hash'][:16]}, gate balacc on real = {jinfo['gate_balacc_on_real']}")

    d = json.load(open(hc.VH / "row2_unseen.json", encoding="utf-8"))
    recs = d["records"]
    n = len(recs)

    # each request: its two bracketing dataset rows (same five other factors)
    lo_rows, hi_rows, h_req, h_lo, h_hi, ck, tgt = [], [], [], [], [], [], []
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
        tgt.append(fi)
        ck.append(f"br{a:.1f}-{b:.1f}|sh{int(t['shape'])}|st{SCALE_TERCILE[int(fi[3])]}")
    lo_rows = np.array(lo_rows); hi_rows = np.array(hi_rows)
    h_req = np.array(h_req); h_lo = np.array(h_lo); h_hi = np.array(h_hi)
    ck = np.array(ck); tgt = np.stack(tgt)
    log(f"{n} requests, {len(set(ck.tolist()))} corners; reading {n*2} real rows in one call")

    both = hc.read_images(np.concatenate([lo_rows, hi_rows]))
    Ia, Ib = both[:n], both[n:]
    log("images read; building ground-truth midpoints")

    ha, hb = hsv_of(Ia), hsv_of(Ib)
    step = (h_hi - h_lo)[:, None, None]
    rot = rotational_mask(ha, hb, step)
    built = construct(Ia, Ib, ha, rot, step, 0.5)
    log(f"ground truth built; rotational pixel fraction {rot.mean():.4f}")

    from chlu.eval.gen_probe import factor_probs
    import equinox as eqx, jax, jax.numpy as jnp
    from chlu.core.gen_bridge import model_dtype
    dt = model_dtype(judge)
    fwd = eqx.filter_jit(jax.vmap(judge))
    preds, hp = [], []
    for i in range(0, n, 256):
        lg = fwd(jnp.asarray(built[i:i + 256], dt))
        preds.append(np.stack([np.asarray(jnp.argmax(l, -1)) for l in lg], 1))
        hp.append(np.asarray(jax.nn.softmax(lg[hc.HUE_F], -1), np.float64))
    preds = np.concatenate(preds); hp = np.concatenate(hp)

    h_hat = hc.circ_mean_hue(hp, hue_vals)
    err = hc.cdist(h_hat, h_req)
    sn = hc.snap_metrics(h_hat, h_lo, h_hi)
    other = [k for k in range(6) if k != hc.HUE_F]
    ok5 = np.ones(n, bool)
    perf = {}
    for k in other:
        c = preds[:, k] == tgt[:, k]
        perf[g.names[k]] = float(c.mean())
        ok5 &= c
    hit = err <= 0.5 * DELTA
    row2acc = ok5 & hit

    keys = sorted(set(ck.tolist()))
    pc = np.array([float(err[ck == k].mean()) for k in keys])

    log("")
    log(f"ORACLE CEILING on row 2 ({n} requests, {len(keys)} corners)")
    log(f"  HUEMAE          {err.mean()/DELTA:.4f} D   (per-corner {pc.min():.4f}-{pc.max():.4f} raw)")
    log(f"  ROW2ACC gated   {row2acc.mean():.4f}")
    log(f"  hue hit ungated {hit.mean():.4f}   (cdist <= 0.5D)")
    log(f"  other-5 correct {ok5.mean():.4f}   {perf}")
    log(f"  SNAPFRAC        {sn['snap'].mean():.4f}")
    log(f"  mean d_req      {sn['d_req'].mean()/DELTA:.4f} D")
    log(f"  mean d_near     {sn['d_near'].mean()/DELTA:.4f} D")
    log(f"  mean t          {sn['t'].mean():.4f}  (0.5 = the requested midpoint)")
    log(f"  top1 in bracket {float(np.isin(preds[:, hc.HUE_F], []).mean() if False else np.mean([p in (int(np.argmin(np.abs(hue_vals-l))), int(np.argmin(np.abs(hue_vals-h)))) for p, l, h in zip(preds[:, hc.HUE_F], h_lo, h_hi)])):.4f}")

    hc.jdump({
        "what": "ORACLE CEILING for scoreboard row 2 -- the score a PERFECT arm would get",
        "scope": hc.SCOPE,
        "how": "each request's exact ground-truth in-between image, built by the j3 "
               "in-bracket construction (on-grid pixel MAE 0.0037, judge agreement "
               "1.0000) from its two REAL bracketing dataset rows, then judged by the "
               "frozen external classifier",
        "judge": jinfo, "n_requests": n, "n_corners": len(keys),
        "rotational_pixel_frac": float(rot.mean()),
        "ceiling": {
            "HUEMAE_delta": float(err.mean() / DELTA),
            "ROW2ACC_gated": float(row2acc.mean()),
            "hue_hit_ungated": float(hit.mean()),
            "other5_all_correct": float(ok5.mean()),
            "per_factor_other5": perf,
            "SNAPFRAC": float(sn["snap"].mean()),
            "mean_d_req_delta": float(sn["d_req"].mean() / DELTA),
            "mean_d_near_delta": float(sn["d_near"].mean() / DELTA),
            "mean_t": float(sn["t"].mean()), "sd_t": float(sn["t"].std()),
        },
        "per_corner_HUEMAE_raw": {k: float(v) for k, v in zip(keys, pc)},
        "declared_margin_delta": 0.2,
        "reading": "this is a CEILING: no arm scoring through this judge can do better. "
                   "It is measured on ground truth with no arm and no store in the room, "
                   "so any part of it that is not 0 is the INSTRUMENT.",
    }, hc.OUT / "runs" / "j4_ceiling.json")
    log("wrote runs/j4_ceiling.json")


if __name__ == "__main__":
    main()
