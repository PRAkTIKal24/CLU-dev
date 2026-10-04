"""PREREG-0 section 5 + AMENDMENT 1: the judge stop rule.

A1.1 construction, restricted to INTERIOR object pixels (mask AND S>0.2 in both
     endpoints) -- the domain on which step 1 measured the rotation to be exact.
     Registered pass: worst whole-image pixel MAE <= 0.02.
A1.2 JUDGE AGREEMENT, the operative criterion: the frozen external classifier must
     read a CONSTRUCTED on-grid image the way it reads the REAL one.
     Registered pass: top-1 hue class agreement >= 0.95 AND mean |dh_hat| <= 0.01.
s5.3 RESOLUTION (not amended): Spearman rho >= 0.9 across a sweep; midpoint
     mean cdist <= 0.25*Delta; AUROC(midpoint vs each bracket's REAL images) >= 0.9.

If A1.1 or A1.2 or s5.3 fails -> the hub STOPS and reports row 2 unmeasurable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402

rng = np.random.default_rng(0)
N_CTX = 96
S_MIN = 0.2
DELTA = hc.DELTA


def build_ctx(n):
    return np.stack(
        [
            rng.integers(0, 10, n),
            rng.integers(0, 10, n),
            np.zeros(n, np.int64),
            rng.integers(0, 8, n),
            rng.integers(0, 4, n),
            rng.integers(0, 15, n),
        ],
        axis=1,
    ).astype(np.int64)


def rows_at(ctx, hue_idx):
    fi = ctx.copy()
    fi[:, hc.HUE_F] = int(hue_idx)
    return hc.rows_of(fi)


def interior_masks(ctx):
    """Object mask (union over hue pairs) AND saturated at EVERY grid hue.

    All 10 hue variants of all contexts are fetched in ONE h5 call: the file is
    chunked (15000, 4, 4, 1) + gzip, so a call costs ~768 chunk decompressions per
    distinct 15000-row block and ten separate calls cost ten times that.
    """
    allrows = np.concatenate([rows_at(ctx, h) for h in range(10)])
    flat = hc.read_images(allrows)
    n = ctx.shape[0]
    imgs = {h: flat[h * n:(h + 1) * n] for h in range(10)}
    base = imgs[0]
    moved = np.zeros(base.shape[0:1] + base.shape[2:], bool)
    for h in (1, 3, 5, 7):
        moved |= np.abs(imgs[h] - base).max(axis=1) > (2.0 / 255.0)
    sat = np.ones_like(moved)
    for h in range(10):
        hsv = hc.rgb_to_hsv(np.transpose(imgs[h], (0, 2, 3, 1)))
        sat &= hsv[..., 1] > S_MIN
    return imgs, moved, (moved & sat)


def construct(src_img, mask, delta):
    return np.stack(
        [hc.rotate_object_hue(src_img[n], mask[n], delta) for n in range(src_img.shape[0])]
    )


def hue_readout(judge, imgs, hue_vals, batch=256):
    from chlu.eval.gen_probe import factor_probs

    p = factor_probs(judge, imgs, hc.HUE_F, batch)
    return p, hc.circ_mean_hue(p, hue_vals), hc.lin_mean_hue(p, hue_vals)


def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    g, lab, nchk = hc.grid_from_h5()
    hue_vals = np.asarray(g.values[hc.HUE_F], np.float64)
    judge, jinfo = hc.load_judge(0)
    print(f"judge {jinfo['param_hash'][:16]} ({jinfo['param_count']} params), "
          f"gate balacc on real = {jinfo['gate_balacc_on_real']}")

    ctx = build_ctx(N_CTX)
    imgs, moved, interior = interior_masks(ctx)
    left = float((moved & ~interior).sum() / max(1, moved.sum()))
    print(f"mask: object {moved.mean():.4f} of pixels; interior {interior.mean():.4f}; "
          f"{left:.4f} of object pixels left unrotated (boundary/desaturated)")

    # ---- A1.1 pixel criterion, refined construction -----------------------
    con = []
    for src in (0, 2, 5):
        for dst in range(10):
            if dst == src:
                continue
            d = float(hue_vals[dst] - hue_vals[src])
            built = construct(imgs[src], interior, d)
            real = imgs[dst]
            con.append({
                "src_hue": float(hue_vals[src]), "dst_hue": float(hue_vals[dst]),
                "pixel_mae_all": float(np.abs(built - real).mean()),
            })
    worst_px = max(c["pixel_mae_all"] for c in con)
    mean_px = float(np.mean([c["pixel_mae_all"] for c in con]))
    a11 = worst_px <= 0.02
    print(f"\nA1.1 pixel MAE over {len(con)} pairs: mean {mean_px:.5f} worst {worst_px:.5f} "
          f"-> {'PASS' if a11 else 'FAIL'} (registered <= 0.02)")

    # ---- A1.2 JUDGE AGREEMENT (the operative criterion) -------------------
    agree, dh = [], []
    per_pair = []
    for src in (0, 2, 5):
        for dst in range(10):
            if dst == src:
                continue
            d = float(hue_vals[dst] - hue_vals[src])
            built = construct(imgs[src], interior, d)
            pb, hb, _ = hue_readout(judge, built, hue_vals)
            pr, hrr, _ = hue_readout(judge, imgs[dst], hue_vals)
            a = float((pb.argmax(1) == pr.argmax(1)).mean())
            e = float(np.abs(((hb - hrr + 0.5) % 1.0) - 0.5).mean())
            agree.append(a); dh.append(e)
            per_pair.append({"src_hue": float(hue_vals[src]), "dst_hue": float(hue_vals[dst]),
                             "top1_agreement": a, "mean_abs_dh_hat": e,
                             "top1_correct_real": float((pr.argmax(1) == dst).mean()),
                             "top1_correct_built": float((pb.argmax(1) == dst).mean())})
    agr = float(np.mean(agree)); mdh = float(np.mean(dh))
    a12 = (agr >= 0.95) and (mdh <= 0.01)
    print(f"A1.2 judge agreement: top-1 {agr:.4f} (>=0.95), mean |dh_hat| {mdh:.5f} (<=0.01) "
          f"-> {'PASS' if a12 else 'FAIL'}")
    print(f"     worst per-pair top-1 agreement {min(agree):.4f}; worst |dh_hat| {max(dh):.5f}")

    # ---- s5.3 RESOLUTION ---------------------------------------------------
    # sweep true hue across each retained bracket in steps of 0.1*Delta
    retained = [(0,1),(1,2),(2,3),(3,4),(6,7),(7,8),(8,9)]
    fr = np.arange(0.0, 1.0001, 0.1)
    sweep = []
    mid_err, aurocs = [], []
    for (lo, hi) in retained:
        htrue, hhat = [], []
        mid_p = None
        for f in fr:
            d = float(f) * DELTA
            built = construct(imgs[lo], interior, d)
            p, hh, _ = hue_readout(judge, built, hue_vals)
            t = (hue_vals[lo] + d) % 1.0
            htrue.append(np.full(hh.shape, t)); hhat.append(hh)
            if abs(f - 0.5) < 1e-9:
                mid_p = p
                e = float(hc.cdist(hh, t).mean())
                mid_err.append(e)
            sweep.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])],
                          "frac": float(f), "h_true": float(t),
                          "mean_h_hat": float(hc.circ_mean_hue(p, hue_vals).mean()),
                          "mean_cdist": float(hc.cdist(hh, t).mean()),
                          "top1_hist": np.bincount(p.argmax(1), minlength=10).tolist()})
        htrue = np.concatenate(htrue); hhat = np.concatenate(hhat)
        rho = spearman(htrue, hhat)
        # AUROC: midpoint vs each bracket's REAL images, score = p(hi) - p(lo)
        from chlu.eval.gen_probe import binary_auroc
        for end in (lo, hi):
            pr, _, _ = hue_readout(judge, imgs[end], hue_vals)
            s = np.concatenate([mid_p[:, hi] - mid_p[:, lo], pr[:, hi] - pr[:, lo]])
            lb = np.concatenate([np.ones(mid_p.shape[0], bool), np.zeros(pr.shape[0], bool)])
            au = hc.cdist(0, 0)  # placeholder to keep numpy import honest
            au = binary_auroc(s, lb) if end == hi else binary_auroc(-s, ~lb)
            aurocs.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])],
                           "vs_real_hue": float(hue_vals[end]), "auroc": float(au)})
        print(f"  bracket ({hue_vals[lo]:.1f},{hue_vals[hi]:.1f}): spearman {rho:.4f}  "
              f"midpoint cdist {mid_err[-1]:.5f} ({mid_err[-1]/DELTA:.3f} Delta)")
        sweep.append({"bracket": [float(hue_vals[lo]), float(hue_vals[hi])], "spearman": rho})

    rhos = [s["spearman"] for s in sweep if "spearman" in s]
    min_rho = float(min(rhos)); mean_mid = float(np.mean(mid_err)); worst_mid = float(max(mid_err))
    min_au = float(min(a["auroc"] for a in aurocs))
    r_mono = min_rho >= 0.9
    r_mid = worst_mid <= 0.25 * DELTA
    r_au = min_au >= 0.9
    print(f"\ns5.3 resolution: min spearman {min_rho:.4f} (>=0.9) -> {'PASS' if r_mono else 'FAIL'}")
    print(f"     midpoint cdist mean {mean_mid:.5f} worst {worst_mid:.5f} "
          f"({worst_mid/DELTA:.3f} Delta, registered <= 0.25) -> {'PASS' if r_mid else 'FAIL'}")
    print(f"     min AUROC vs real brackets {min_au:.4f} (>=0.9) -> {'PASS' if r_au else 'FAIL'}")

    verdict = "PROCEED" if (a11 and a12 and r_mono and r_mid and r_au) else "STOP"
    print(f"\n==== JUDGE STOP RULE VERDICT: {verdict} ====")

    hc.jdump({
        "what": "PREREG-0 s5 + AMENDMENT 1: judge stop rule",
        "scope": hc.SCOPE,
        "judge": jinfo, "n_contexts": N_CTX, "s_min": S_MIN,
        "mask": {"object_frac": float(moved.mean()), "interior_frac": float(interior.mean()),
                 "object_pixels_left_unrotated_frac": left},
        "A1_1_pixel": {"pairs": con, "mean": mean_px, "worst": worst_px,
                       "threshold": 0.02, "verdict": "PASS" if a11 else "FAIL"},
        "A1_2_judge_agreement": {"per_pair": per_pair, "mean_top1_agreement": agr,
                                 "worst_top1_agreement": float(min(agree)),
                                 "mean_abs_dh_hat": mdh, "worst_abs_dh_hat": float(max(dh)),
                                 "thresholds": {"top1": 0.95, "dh_hat": 0.01},
                                 "verdict": "PASS" if a12 else "FAIL"},
        "s5_3_resolution": {"sweep": sweep, "aurocs": aurocs,
                            "min_spearman": min_rho, "midpoint_cdist_mean": mean_mid,
                            "midpoint_cdist_worst": worst_mid, "min_auroc": min_au,
                            "thresholds": {"spearman": 0.9, "midpoint_cdist": 0.25 * DELTA,
                                           "auroc": 0.9},
                            "verdict": "PASS" if (r_mono and r_mid and r_au) else "FAIL"},
        "VERDICT": verdict,
    }, hc.OUT / "runs" / "j2_resolution.json")
    print("wrote runs/j2_resolution.json")


if __name__ == "__main__":
    main()
