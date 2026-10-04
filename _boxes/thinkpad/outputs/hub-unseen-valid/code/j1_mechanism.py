"""PREREG-0 section 5, steps 1 and 2: is the dataset's object_hue step an HSV rotation,
and can I therefore build a REAL-DERIVED in-between-hue image?

Step 1 MECHANISM: for rows differing only in object_hue, the object pixels' HSV hue
                  must differ by exactly the grid step, with S and V preserved.
Step 2 CONSTRUCTION: synthesise an ON-GRID hue from a DIFFERENT on-grid hue by that
                  rotation and compare PIXEL-WISE to the real image.
                  Registered pass: mean absolute pixel error <= 0.02.

Nothing about any arm or control is touched here. Read-only on the h5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402

rng = np.random.default_rng(0)
N_CTX = 64  # contexts = (floor, wall, scale, shape, orientation) tuples


def main():
    g, lab, nchk = hc.grid_from_h5()
    print(f"grid verified on {nchk} rows; cards={g.cardinalities}")
    hue_vals = np.asarray(g.values[hc.HUE_F], np.float64)
    print("object_hue grid:", np.round(hue_vals, 4).tolist())

    # sample contexts: every factor but object_hue
    ctx = np.stack(
        [
            rng.integers(0, 10, N_CTX),  # floor
            rng.integers(0, 10, N_CTX),  # wall
            np.zeros(N_CTX, np.int64),  # placeholder for object_hue
            rng.integers(0, 8, N_CTX),  # scale
            rng.integers(0, 4, N_CTX),  # shape
            rng.integers(0, 15, N_CTX),  # orientation
        ],
        axis=1,
    ).astype(np.int64)

    def rows_at(hue_idx):
        fi = ctx.copy()
        fi[:, hc.HUE_F] = int(hue_idx)
        return hc.rows_of(fi)

    # ---- object mask: pixels that move when ONLY object_hue changes -------
    # union over several hue pairs, so a hue whose object happens to match the
    # background at one value cannot shrink the mask.
    base = hc.read_images(rows_at(0))
    diff = np.zeros(base.shape[0:1] + base.shape[2:], bool)
    for hi in (1, 3, 5, 7):
        other = hc.read_images(rows_at(hi))
        diff |= (np.abs(other - base).max(axis=1) > (2.0 / 255.0))
    frac = diff.reshape(diff.shape[0], -1).mean(axis=1)
    print(f"object mask: mean pixel fraction {frac.mean():.4f} "
          f"[{frac.min():.4f}, {frac.max():.4f}]")

    # ---- STEP 1: mechanism -------------------------------------------------
    mech = []
    for hi in range(1, 10):
        cur = hc.read_images(rows_at(hi))
        dh, ds, dv = [], [], []
        for n in range(base.shape[0]):
            m = diff[n]
            if m.sum() < 50:
                continue
            a = hc.rgb_to_hsv(np.transpose(base[n], (1, 2, 0)))[m]
            b = hc.rgb_to_hsv(np.transpose(cur[n], (1, 2, 0)))[m]
            # only saturated pixels carry a meaningful hue
            sel = (a[:, 1] > 0.2) & (b[:, 1] > 0.2)
            if sel.sum() < 20:
                continue
            d = (b[sel, 0] - a[sel, 0]) % 1.0
            dh.append(np.median(d))
            ds.append(np.median(np.abs(b[sel, 1] - a[sel, 1])))
            dv.append(np.median(np.abs(b[sel, 2] - a[sel, 2])))
        step = float(hue_vals[hi] - hue_vals[0])
        mech.append(
            {
                "hue_index": hi,
                "grid_step": step,
                "median_hue_rotation": float(np.median(dh)),
                "abs_err_vs_grid_step": float(abs(np.median(dh) - step)),
                "median_abs_dS": float(np.median(ds)),
                "median_abs_dV": float(np.median(dv)),
                "n_contexts": len(dh),
            }
        )
        print(
            f"  hue 0.0 -> {hue_vals[hi]:.1f}: measured rotation "
            f"{mech[-1]['median_hue_rotation']:+.4f} (grid {step:+.1f}), "
            f"|dS|={mech[-1]['median_abs_dS']:.4f} |dV|={mech[-1]['median_abs_dV']:.4f}"
        )

    max_rot_err = max(m["abs_err_vs_grid_step"] for m in mech)
    max_ds = max(m["median_abs_dS"] for m in mech)
    max_dv = max(m["median_abs_dV"] for m in mech)

    # ---- STEP 2: construction check (on-grid from on-grid, pixel-wise) -----
    con = []
    for src in (0, 2, 5):
        s_img = hc.read_images(rows_at(src))
        for dst in range(10):
            if dst == src:
                continue
            delta = float(hue_vals[dst] - hue_vals[src])
            real = hc.read_images(rows_at(dst))
            built = np.stack(
                [hc.rotate_object_hue(s_img[n], diff[n], delta) for n in range(s_img.shape[0])]
            )
            mae = float(np.abs(built - real).mean())
            mae_obj = float(
                np.abs(
                    (built - real).transpose(0, 2, 3, 1)[diff]
                ).mean()
            )
            con.append(
                {
                    "src_hue": float(hue_vals[src]),
                    "dst_hue": float(hue_vals[dst]),
                    "delta": delta,
                    "pixel_mae_all": mae,
                    "pixel_mae_object": mae_obj,
                }
            )
    worst = max(c["pixel_mae_all"] for c in con)
    worst_obj = max(c["pixel_mae_object"] for c in con)
    mean_mae = float(np.mean([c["pixel_mae_all"] for c in con]))
    print(f"\nconstruction check over {len(con)} (src,dst) hue pairs x {N_CTX} contexts:")
    print(f"  pixel MAE (whole image): mean {mean_mae:.5f}  worst {worst:.5f}")
    print(f"  pixel MAE (object only): worst {worst_obj:.5f}")
    print(f"  REGISTERED PASS: worst whole-image MAE <= 0.02 -> "
          f"{'PASS' if worst <= 0.02 else 'FAIL'}")

    hc.jdump(
        {
            "what": "PREREG-0 s5 steps 1-2: dataset hue mechanism + real-derived construction",
            "scope": hc.SCOPE,
            "n_contexts": N_CTX,
            "object_mask_pixel_fraction_mean": float(frac.mean()),
            "step1_mechanism": mech,
            "step1_summary": {
                "max_abs_rotation_error_vs_grid": max_rot_err,
                "max_median_abs_dS": max_ds,
                "max_median_abs_dV": max_dv,
            },
            "step2_construction": con,
            "step2_summary": {
                "mean_pixel_mae_all": mean_mae,
                "worst_pixel_mae_all": worst,
                "worst_pixel_mae_object": worst_obj,
                "registered_threshold": 0.02,
                "verdict": "PASS" if worst <= 0.02 else "FAIL",
            },
        },
        hc.OUT / "runs" / "j1_mechanism.json",
    )
    print("\nwrote runs/j1_mechanism.json")


if __name__ == "__main__":
    main()
