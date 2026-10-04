"""EXP-R10 - DEMONSTRATE the property the report says justifies the store's +0.027 cost.

The report's central design claim is that a product-quantised store cannot be justified as
compression at this byte scale (it saves 4 % of bytes and costs 0.027 of accuracy), and
must instead be justified by what a dense prototype table cannot do: atoms that are
DECODABLE and EXACTLY DELETABLE.  R4 makes the same distinction the other way round -- a
principal direction is refused as a store precisely because it "cannot be decoded or
deleted".

A claim like that should be shown, not asserted.  This file shows it, on the ARM's own
fitted NMF dictionary, seed 0:

  1. DECODABLE - every atom is pushed through the FROZEN decoder on its own and looked at.
     NMF atoms are non-negative, so each is a point in the data's own orthant and decodes
     to a picture.  A k-means atom decodes too; a centred principal direction does not
     (it is a signed difference, not a feature vector), and the file measures how far
     outside the data orthant each dictionary's entries fall.
  2. EXACTLY DELETABLE - zero ONE atom and re-read.  The claim is not "the output moves";
     it is EXACT: every request whose coefficient on that atom is zero must come back
     BIT-IDENTICAL, and the change in every other request must equal exactly
     `-a_m * atom_m` on that well's channels and be exactly zero everywhere else.

usage:  atoms_are_atoms.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import ctx
    import dev as devmod  # noqa: F401
    from chlu.core.gen_dict_fit import build_store_from_dicts, well_dictionary
    from chlu.core.gen_dream import requests_from_labels
    from chlu.core.gen_shared_basis_head import fit_shared_basis_head
    from chlu.core.gen_store import store_bytes
    from chlu.core.gen_wide_read import wide_vector
    from chlu.experiments.exp_gen_cell import render
    import fast

    SEED, K, A = 0, 64, 1216
    PAIRS = [(0, 3), (2, 3), (4, 5)]
    c = ctx.build_context(SEED)
    D = int(np.prod(c["feat_shape"]))
    pool = c["pool"]
    X = c["gather"](pool)
    lab = c["factor_labels"](c["grid"], pool)
    Z = np.asarray(X, np.float32).reshape(-1, D)
    _, dead, ow, _ = fast.channel_geometry(X, c["cfg"], f"atoms_s{SEED}", log=log)
    labels = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    views = [np.asarray(Z[:, perm[k]], np.float64) for k in range(K)]
    M = A // K

    res = {"seed": SEED, "K": K, "A": A, "M_per_well": M,
           "machine": "FORGIS_14 Windows 11 AMD64"}

    # ---- 1. is a dictionary entry a valid feature vector? ------------------------------
    log("fitting the three dictionaries on well 0 ... (nmf is the arm's)")
    orth = {}
    for method in ("nmf", "kmeans", "atoms_medoid", "pca_affine"):
        neg = tot = 0
        mags = []
        for k in range(0, K, 8):                      # every 8th well: 8 of 64
            Wd = well_dictionary(views[k], M, method, SEED * 1000 + k)
            neg += int((Wd < 0).sum())
            tot += int(Wd.size)
            mags.append(float(np.abs(np.minimum(Wd, 0.0)).max()))
        orth[method] = {"frac_negative_entries": neg / max(tot, 1),
                        "max_negative_magnitude": float(max(mags))}
        log(f"  {method:13s} negative entries {neg/max(tot,1):7.2%}  "
            f"max |negative| {max(mags):.4f}")
    res["outside_the_data_orthant"] = orth
    res["orthant_note"] = (
        "the bridge features are post-ReLU, so a VALID feature vector has no negative "
        "entry. NMF atoms are non-negative by construction; a centred principal "
        "direction is a signed difference and is not a feature vector at all, which is "
        "R4's stated reason for refusing it as a store.")

    # ---- the ARM's own store, and its atoms decoded one at a time ----------------------
    dicts = [well_dictionary(views[k], M, "nmf", SEED * 1000 + k) for k in range(K)]
    st = build_store_from_dicts([tuple(int(i) for i in perm[k]) for k in range(K)],
                                dicts, (D,))
    sb = store_bytes(st)
    res["store_bytes"] = sb["store_bytes"]
    log(f"arm store built: {sb['store_bytes']} B, n_modes_per_well[0]={sb['n_modes_per_well'][0]}")

    # decode 12 single atoms: the feature vector is ZERO everywhere except that atom's
    # own well channels, where it is the atom itself.
    picks = [(k, m) for k in range(0, 24, 6) for m in range(0, M, max(M // 3, 1))][:12]
    feats = np.zeros((len(picks), D), np.float32)
    for i, (k, m) in enumerate(picks):
        feats[i, perm[k]] = np.asarray(dicts[k][m], np.float32)
    imgs = render(c["decoder"], feats.reshape((-1,) + tuple(c["feat_shape"])),
                  c["feat_shape"], 64)
    imgs = np.asarray(imgs, np.float32)
    lo, hi = float(imgs.min()), float(imgs.max())
    res["decoded_single_atoms"] = {
        "n": len(picks), "picks": [[int(k), int(m)] for k, m in picks],
        "image_shape": list(imgs.shape[1:]), "pixel_min": lo, "pixel_max": hi,
        "all_finite": bool(np.isfinite(imgs).all()),
        "per_atom_pixel_std": [float(x) for x in imgs.reshape(len(picks), -1).std(1)]}
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.4))
    for ax, im, (k, m) in zip(axes.ravel(), imgs, picks):
        a = np.asarray(im, np.float32)
        if a.ndim == 3 and a.shape[0] in (1, 3):      # CHW -> HWC
            a = np.transpose(a, (1, 2, 0))
        a = a.squeeze()
        a = a - a.min()
        a = a / max(a.max(), 1e-9)
        ax.imshow(a, cmap=None if a.ndim == 3 else "gray")
        ax.set_title(f"atom (well {k}, m {m})", fontsize=8)
        ax.axis("off")
    fig.suptitle("DECODABLE: single NMF atoms of the arm's store, each pushed through the "
                 "FROZEN decoder on its own\n(the feature vector is zero everywhere "
                 "except that atom's own 32 channels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "atoms_decoded.png", dpi=150)
    log(f"decoded {len(picks)} single atoms -> atoms_decoded.png "
        f"(pixels {lo:.3f}..{hi:.3f}, all finite={res['decoded_single_atoms']['all_finite']})")

    # ---- 2. EXACT deletion -------------------------------------------------------------
    head = fit_shared_basis_head(dicts, perm, Z, lab, c["cards"], PAIRS,
                                 ridge_add=10.0, ridge_pair=100.0, reduced=True)
    req = requests_from_labels(c["factor_labels"](c["grid"], pool[:512]), "full", 0, SEED)
    coef = head.coefficients(req)
    before = np.asarray(wide_vector(st, coef), np.float32).reshape(len(req), D)

    kk, mm = 3, 1
    d2 = [np.array(d, copy=True) for d in dicts]
    d2[kk][mm] = 0.0                                  # DELETE one atom, exactly
    st2 = build_store_from_dicts([tuple(int(i) for i in perm[k]) for k in range(K)],
                                 d2, (D,))
    after = np.asarray(wide_vector(st2, coef), np.float32).reshape(len(req), D)

    delta = after - before
    off = np.ones(D, bool); off[perm[kk]] = False
    predicted = np.zeros_like(delta)
    predicted[:, perm[kk]] = -np.outer(np.asarray(coef[kk][:, mm], np.float64),
                                       np.asarray(dicts[kk][mm], np.float64))
    res["exact_deletion"] = {
        "deleted": {"well": kk, "atom": mm},
        "n_requests": int(len(req)),
        "max_abs_change_OUTSIDE_that_well": float(np.abs(delta[:, off]).max()),
        "n_channels_outside_that_well": int(off.sum()),
        "max_abs_deviation_from_the_predicted_change": float(
            np.abs(delta - predicted.astype(np.float32)).max()),
        "n_requests_with_zero_coefficient_on_the_deleted_atom": int(
            (np.abs(coef[kk][:, mm]) == 0).sum()),
        "max_abs_change_on_those_requests": float(
            np.abs(delta[np.abs(coef[kk][:, mm]) == 0]).max()
            if (np.abs(coef[kk][:, mm]) == 0).any() else 0.0),
        "store_bytes_before": int(sb["store_bytes"]),
        "store_bytes_after": int(store_bytes(st2)["store_bytes"]),
        "rule": "EXACT means: zero change outside the deleted atom's own well, and the "
                "change inside it equals -a_m * atom_m to float32.",
    }
    for k, v in res["exact_deletion"].items():
        log(f"  deletion/{k}: {v}")
    (out_dir / "atoms_are_atoms.json").write_text(json.dumps(res, indent=2),
                                                  encoding="utf-8")
    log(f"wrote {out_dir/'atoms_are_atoms.json'}")


if __name__ == "__main__":
    main()
