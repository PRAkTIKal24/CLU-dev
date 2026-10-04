"""AMENDMENT 3: the IN-BRACKET construction, and the judge stop rule re-run on it.

Row 2 asks for a value strictly inside a bracket, so the construction is validated
inside a bracket, between two REAL endpoint images I_a and I_b:

  ROTATIONAL pixels -- those whose measured a->b change IS the hue rotation
      (|dhue - step| <= 0.01, |dS| <= 0.02, |dV| <= 0.02, S >= 0.2 at both ends):
      hue-rotate I_a by f*(b-a). Exact, by the step-1 mechanism measurement.
  every OTHER pixel (blend, reflection, specular):
      RGB-lerp (1-f)*I_a + f*I_b -- the correct first-order behaviour of a blend,
      and exact at f = 0 and f = 1.

VALIDATION, on-grid and therefore checkable: a bracket of width 2*Delta (e.g. 0.0 and
0.2), constructed at f = 0.5, compared PIXEL-WISE to the REAL midpoint image (0.1).
Threshold UNCHANGED at the registered worst whole-image MAE <= 0.02.

Then the PREREG-0 s5.3 resolution criteria, unamended:
  Spearman rho >= 0.9 ; midpoint mean cdist <= 0.25*Delta ; AUROC >= 0.9.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402

rng = np.random.default_rng(0)
N_CTX = 128
DELTA = hc.DELTA
TOL_HUE, TOL_S, TOL_V, S_MIN = 0.01, 0.02, 0.02, 0.2
RETAINED = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]


def build_ctx(n):
    return np.stack([rng.integers(0, 10, n), rng.integers(0, 10, n),
                     np.zeros(n, np.int64), rng.integers(0, 8, n),
                     rng.integers(0, 4, n), rng.integers(0, 15, n)], axis=1).astype(np.int64)


def all_hue_images(ctx):
    rows = []
    for h in range(10):
        fi = ctx.copy(); fi[:, hc.HUE_F] = h
        rows.append(hc.rows_of(fi))
    flat = hc.read_images(np.concatenate(rows))
    n = ctx.shape[0]
    return {h: flat[h * n:(h + 1) * n] for h in range(10)}


def hsv_of(img):
    return hc.rgb_to_hsv(np.transpose(img, (0, 2, 3, 1)))


def rotational_mask(hsv_a, hsv_b, step):
    dh = (hsv_b[..., 0] - hsv_a[..., 0]) % 1.0
    dh = np.minimum(np.abs(dh - step), np.abs(dh - step + 1.0))
    return (
        (dh <= TOL_HUE)
        & (np.abs(hsv_b[..., 1] - hsv_a[..., 1]) <= TOL_S)
        & (np.abs(hsv_b[..., 2] - hsv_a[..., 2]) <= TOL_V)
        & (hsv_a[..., 1] >= S_MIN) & (hsv_b[..., 1] >= S_MIN)
    )


def construct(img_a, img_b, hsv_a, rot, step, f):
    """(N,3,H,W): hue-rotate the rotational pixels, RGB-lerp everything else."""
    h = hsv_a.copy()
    h[..., 0] = np.where(rot, (h[..., 0] + f * step) % 1.0, h[..., 0])
    rotated = np.clip(hc.hsv_to_rgb(h), 0.0, 1.0)
    lerp = np.transpose((1.0 - f) * img_a + f * img_b, (0, 2, 3, 1))
    out = np.where(rot[..., None], rotated, lerp)
    return np.transpose(out, (0, 3, 1, 2)).astype(np.float32)


def hue_readout(judge, imgs, hue_vals, batch=256):
    from chlu.eval.gen_probe import factor_probs
    p = factor_probs(judge, imgs, hc.HUE_F, batch)
    return p, hc.circ_mean_hue(p, hue_vals)


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, np.float64))).astype(np.float64)
    rb = np.argsort(np.argsort(np.asarray(b, np.float64))).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    g, _, _ = hc.grid_from_h5()
    hue_vals = np.asarray(g.values[hc.HUE_F], np.float64)
    judge, jinfo = hc.load_judge(0)
    print(f"judge {jinfo['param_hash'][:16]} ({jinfo['param_count']} params), "
          f"gate balacc on real = {jinfo['gate_balacc_on_real']}", flush=True)

    ctx = build_ctx(N_CTX)
    imgs = all_hue_images(ctx)
    hsv = {h: hsv_of(imgs[h]) for h in range(10)}
    print(f"{N_CTX} contexts, all 10 hues, one h5 call", flush=True)

    # ---- A1.1 re-run: on-grid validation at width 2*Delta, f = 0.5 ---------
    val = []
    for a_i in range(8):
        b_i, m_i = a_i + 2, a_i + 1
        step = float(hue_vals[b_i] - hue_vals[a_i])
        rot = rotational_mask(hsv[a_i], hsv[b_i], step)
        built = construct(imgs[a_i], imgs[b_i], hsv[a_i], rot, step, 0.5)
        mae = float(np.abs(built - imgs[m_i]).mean())
        val.append({"a": float(hue_vals[a_i]), "b": float(hue_vals[b_i]),
                    "real_mid": float(hue_vals[m_i]), "pixel_mae": mae,
                    "rotational_pixel_frac": float(rot.mean())})
        print(f"  bracket ({hue_vals[a_i]:.1f},{hue_vals[b_i]:.1f}) f=0.5 vs real "
              f"{hue_vals[m_i]:.1f}: MAE {mae:.5f}  rot px {rot.mean():.4f}", flush=True)
    worst = max(v["pixel_mae"] for v in val)
    mean_v = float(np.mean([v["pixel_mae"] for v in val]))
    a11 = worst <= 0.02
    print(f"\nA1.1 (in-bracket, on-grid): mean {mean_v:.5f} worst {worst:.5f} "
          f"-> {'PASS' if a11 else 'FAIL'} (registered <= 0.02, unchanged)", flush=True)

    # ---- A1.2 judge agreement on the SAME on-grid construction -------------
    agree, dh = [], []
    for a_i in range(8):
        b_i, m_i = a_i + 2, a_i + 1
        step = float(hue_vals[b_i] - hue_vals[a_i])
        rot = rotational_mask(hsv[a_i], hsv[b_i], step)
        built = construct(imgs[a_i], imgs[b_i], hsv[a_i], rot, step, 0.5)
        pb, hb = hue_readout(judge, built, hue_vals)
        pr, hr = hue_readout(judge, imgs[m_i], hue_vals)
        agree.append(float((pb.argmax(1) == pr.argmax(1)).mean()))
        dh.append(float(np.abs(((hb - hr + 0.5) % 1.0) - 0.5).mean()))
    agr, mdh = float(np.mean(agree)), float(np.mean(dh))
    a12 = (agr >= 0.95) and (mdh <= 0.01)
    print(f"A1.2 judge agreement: top-1 {agr:.4f} (>=0.95), mean |dh| {mdh:.5f} (<=0.01) "
          f"-> {'PASS' if a12 else 'FAIL'}", flush=True)

    # ---- s5.3 RESOLUTION on width-Delta brackets ---------------------------
    from chlu.eval.gen_probe import binary_auroc
    fr = np.arange(0.0, 1.0001, 0.1)
    sweep, rhos, mid_err, aurocs, mid_top1 = [], [], [], [], []
    for (lo, hi) in RETAINED:
        step = float(hue_vals[hi] - hue_vals[lo])
        rot = rotational_mask(hsv[lo], hsv[hi], step)
        ht, hh_all, mid_p = [], [], None
        for f in fr:
            built = construct(imgs[lo], imgs[hi], hsv[lo], rot, step, float(f))
            p, hh = hue_readout(judge, built, hue_vals)
            t = (hue_vals[lo] + f * step) % 1.0
            ht.append(np.full(hh.shape, t)); hh_all.append(hh)
            sweep.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])],
                          "frac": float(f), "h_true": float(t),
                          "mean_h_hat": float(hc.circ_mean_hue(p, hue_vals).mean()),
                          "mean_cdist": float(hc.cdist(hh, t).mean()),
                          "mean_max_prob": float(p.max(1).mean()),
                          "top1_hist": np.bincount(p.argmax(1), minlength=10).tolist()})
            if abs(f - 0.5) < 1e-9:
                mid_p = p
                mid_err.append(float(hc.cdist(hh, t).mean()))
                mid_top1.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])],
                                 "frac_top1_in_bracket": float(
                                     np.isin(p.argmax(1), [lo, hi]).mean()),
                                 "mean_p_lo": float(p[:, lo].mean()),
                                 "mean_p_hi": float(p[:, hi].mean()),
                                 "mean_max_prob": float(p.max(1).mean())})
        # Spearman on the bracket coordinate t, NOT on raw circular hue: a readout
        # near 0.0 wraps to 0.99 and destroys the rank correlation for the (0.0,0.1)
        # bracket. seg_t unwraps onto the segment, which is the quantity meant.
        t_true = np.concatenate([np.full(hh.shape, float(f)) for f, hh in
                                 zip(fr, hh_all)])
        t_hat = hc.seg_t(np.concatenate(hh_all), hue_vals[lo], hue_vals[hi])
        rho = spearman(t_true, t_hat)
        rho_raw = spearman(np.concatenate(ht), np.concatenate(hh_all))
        rhos.append(rho)
        calib = [{"frac": float(f),
                  "mean_t_hat": float(np.mean(hc.seg_t(hh, hue_vals[lo], hue_vals[hi]))),
                  "sd_t_hat": float(np.std(hc.seg_t(hh, hue_vals[lo], hue_vals[hi])))}
                 for f, hh in zip(fr, hh_all)]
        sweep.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])],
                      "spearman_on_t": rho, "spearman_on_raw_circular_hue": rho_raw,
                      "calibration_curve": calib})
        for end in (lo, hi):
            pr, _ = hue_readout(judge, imgs[end], hue_vals)
            s = np.concatenate([mid_p[:, hi] - mid_p[:, lo], pr[:, hi] - pr[:, lo]])
            lb = np.concatenate([np.ones(mid_p.shape[0], bool), np.zeros(pr.shape[0], bool)])
            # SEPARABILITY, not a signed AUROC. An AUROC of 0.0000 is PERFECT
            # separation with the sign reversed; scoring it as 0 would report the
            # judge as blind when it is in fact discriminating perfectly.
            au = float(binary_auroc(s, lb))
            aurocs.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])],
                           "vs_real_hue": float(hue_vals[end]), "auroc_raw": au,
                           "separability": float(max(au, 1.0 - au))})
        print(f"  bracket ({hue_vals[lo]:.1f},{hue_vals[hi]:.1f}): rho {rho:.4f}  "
              f"mid cdist {mid_err[-1]:.5f} ({mid_err[-1]/DELTA:.3f}D)  "
              f"top1-in-bracket {mid_top1[-1]['frac_top1_in_bracket']:.3f}", flush=True)

    min_rho, worst_mid = float(min(rhos)), float(max(mid_err))
    min_au = float(min(x["separability"] for x in aurocs))
    r_ok = (min_rho >= 0.9) and (worst_mid <= 0.25 * DELTA) and (min_au >= 0.9)
    print(f"\ns5.3: min rho {min_rho:.4f} (>=0.9) | midpoint cdist worst {worst_mid:.5f} "
          f"= {worst_mid/DELTA:.3f}D (<=0.25D) | min AUROC {min_au:.4f} (>=0.9) "
          f"-> {'PASS' if r_ok else 'FAIL'}", flush=True)

    verdict = "PROCEED" if (a11 and a12 and r_ok) else "STOP"
    print(f"\n==== JUDGE STOP RULE VERDICT (in-bracket construction): {verdict} ====",
          flush=True)

    hc.jdump({
        "what": "AMENDMENT 3: in-bracket construction + PREREG-0 s5 stop rule",
        "scope": hc.SCOPE, "judge": jinfo, "n_contexts": N_CTX,
        "tolerances": {"hue": TOL_HUE, "S": TOL_S, "V": TOL_V, "s_min": S_MIN},
        "A1_1_on_grid_in_bracket": {"per_bracket": val, "mean": mean_v, "worst": worst,
                                    "threshold": 0.02,
                                    "verdict": "PASS" if a11 else "FAIL"},
        "A1_2_judge_agreement": {"mean_top1": agr, "mean_abs_dh": mdh,
                                 "verdict": "PASS" if a12 else "FAIL"},
        "s5_3_resolution": {"sweep": sweep, "spearman_per_bracket": rhos,
                            "midpoint_cdist": mid_err, "midpoint_top1": mid_top1,
                            "aurocs": aurocs, "min_spearman": min_rho,
                            "midpoint_cdist_worst": worst_mid, "min_auroc": min_au,
                            "verdict": "PASS" if r_ok else "FAIL"},
        "VERDICT": verdict,
    }, hc.OUT / "runs" / "j3_bracket.json")
    print("wrote runs/j3_bracket.json", flush=True)


if __name__ == "__main__":
    main()
