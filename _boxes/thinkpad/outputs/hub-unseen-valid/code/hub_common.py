"""hub/unseen-valid — shared helpers. Row 2: a hue requested BETWEEN two stored values.

Nothing here re-fits, re-trains or re-derives a frozen asset. The h5, the grid, the
decoder, the judge and the feature cache are read-only (charter A.4).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

H5 = r"C:/chlu/data/gen/shapes3d/3dshapes.h5"
VH = Path(r"C:/chlu/data/gen/shapes3d/value_holdout")
BUNDLE = Path(r"C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs")
OUT = Path(r"C:/chlu/outputs/hub-unseen-valid")

DELTA = 0.1  # object_hue grid spacing
HUE_F = 2  # object_hue factor index
CARDS = (10, 10, 10, 8, 4, 15)
NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")

SCOPE = (
    "SCOPE: the hold-out is from the STORE'S FIT POOL, not from the encoder, decoder or "
    "judge. Those three are frozen, hash-pinned and were fitted on a split containing "
    "every hue. 'Never seen' here means: the store never held a row carrying that value."
)


# ---------------------------------------------------------------------------
# grid / rows
# ---------------------------------------------------------------------------
def load_grid():
    from chlu.data.gen_splits import FactorGrid

    with_h5 = np.array(
        [
            np.linspace(0.0, 0.9, 10),
            np.linspace(0.0, 0.9, 10),
            np.linspace(0.0, 0.9, 10),
            np.linspace(0.75, 1.25, 8),
            np.arange(4.0),
            np.linspace(-30.0, 30.0, 15),
        ],
        dtype=object,
    )
    return FactorGrid(
        names=NAMES,
        values=tuple(np.asarray(v, np.float64) for v in with_h5),
        dataset="shapes3d",
    )


def grid_from_h5():
    """The grid taken from the FILE's realised labels, never regenerated (FactorGrid doc)."""
    import h5py

    from chlu.data.gen_splits import FactorGrid

    with h5py.File(H5, "r") as f:
        lab = f["labels"][:]
    vals = tuple(np.unique(lab[:, k]).astype(np.float64) for k in range(lab.shape[1]))
    g = FactorGrid(names=NAMES, values=vals, dataset="shapes3d")
    n = g.verify_bijection(lab, n_check=20000)
    return g, lab, int(n)


def rows_of(fi):
    return np.ravel_multi_index(tuple(np.asarray(fi, np.int64).T), CARDS).astype(np.int64)


def read_images(rows, h5=H5):
    """(N,) rows -> (N, 3, 64, 64) float32 in [0,1], channel-first (decoder convention).

    The h5 is chunked (15000, 4, 4, 1) with gzip, so ONE random image row costs
    16*16*3 = 768 chunk decompressions. Rows are therefore gathered in a SINGLE
    fancy-index call (h5py then fetches the union of chunks once) with a large chunk
    cache. Reading the same rows in several calls costs a multiple of this.
    """
    import h5py

    rows = np.asarray(rows, np.int64).ravel()
    order = np.argsort(rows, kind="mergesort")
    srt = rows[order]
    uniq, inv = np.unique(srt, return_inverse=True)
    with h5py.File(h5, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=100003) as f:
        raw = f["images"][uniq]
    raw = raw[inv]
    out = np.empty_like(raw)
    out[order] = raw
    x = np.asarray(out, np.float32) / 255.0
    return np.transpose(x, (0, 3, 1, 2))


# ---------------------------------------------------------------------------
# circular hue arithmetic (hue is a circle; 0.9 and 0.0 are adjacent)
# ---------------------------------------------------------------------------
def cdist(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) % 1.0
    return np.minimum(d, 1.0 - d)


def circ_mean_hue(p, hue_vals):
    """(N, C) posterior -> (N,) circular posterior-mean hue in [0,1)."""
    p = np.asarray(p, np.float64)
    ang = 2.0 * np.pi * np.asarray(hue_vals, np.float64)[None, :]
    z = (p * np.exp(1j * ang)).sum(axis=1)
    return (np.angle(z) / (2.0 * np.pi)) % 1.0


def lin_mean_hue(p, hue_vals):
    p = np.asarray(p, np.float64)
    return (p * np.asarray(hue_vals, np.float64)[None, :]).sum(axis=1)


def seg_t(h_hat, h_lo, h_hi):
    """Project a circular readout onto the bracket segment [h_lo, h_hi]; t in R."""
    h_hat = np.asarray(h_hat, np.float64)
    h_lo = np.asarray(h_lo, np.float64)
    h_hi = np.asarray(h_hi, np.float64)
    d = ((h_hat - h_lo + 0.5) % 1.0) - 0.5  # signed circular offset from h_lo
    return d / (h_hi - h_lo)


def snap_metrics(h_hat, h_lo, h_hi):
    """PREREG-0 section 3: d_req, d_near, SNAPFRAC."""
    t = seg_t(h_hat, h_lo, h_hi)
    d = np.asarray(h_hi, np.float64) - np.asarray(h_lo, np.float64)
    d_req = np.abs(t - 0.5) * d
    d_near = np.minimum(np.abs(t), np.abs(1.0 - t)) * d
    return {
        "t": t,
        "d_req": d_req,
        "d_near": d_near,
        "snap": d_near < d_req,
    }


# ---------------------------------------------------------------------------
# HSV, for the real-derived in-between construction (PREREG-0 section 5)
# ---------------------------------------------------------------------------
def rgb_to_hsv(x):
    """(..., 3) float in [0,1] -> (..., 3) hsv. Vectorised, no colorsys loop."""
    x = np.asarray(x, np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1)
    mn = x.min(-1)
    c = mx - mn
    h = np.zeros_like(mx)
    nz = c > 0
    idx = np.argmax(x, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        hr = ((g - b) / np.where(c == 0, 1, c)) % 6.0
        hg = ((b - r) / np.where(c == 0, 1, c)) + 2.0
        hb = ((r - g) / np.where(c == 0, 1, c)) + 4.0
    h = np.where(idx == 0, hr, np.where(idx == 1, hg, hb)) / 6.0
    h = np.where(nz, h, 0.0) % 1.0
    s = np.where(mx > 0, c / np.where(mx == 0, 1, mx), 0.0)
    return np.stack([h, s, mx], axis=-1)


def hsv_to_rgb(x):
    x = np.asarray(x, np.float64)
    h, s, v = x[..., 0] % 1.0, x[..., 1], x[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).astype(np.int64)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def rotate_object_hue(img_chw, mask, delta):
    """Rotate the hue of the masked pixels by `delta`, S and V untouched.

    img_chw: (3, H, W) float32 in [0,1]; mask: (H, W) bool.
    """
    x = np.transpose(np.asarray(img_chw, np.float64), (1, 2, 0))
    hsv = rgb_to_hsv(x)
    hsv[..., 0] = np.where(mask, (hsv[..., 0] + float(delta)) % 1.0, hsv[..., 0])
    out = np.clip(hsv_to_rgb(hsv), 0.0, 1.0)
    return np.transpose(out, (2, 0, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# frozen instruments
# ---------------------------------------------------------------------------
def load_judge(seed=0):
    from chlu.core.gen_bridge import param_hash
    from chlu.experiments.exp_gen_cell import load_model

    root = BUNDLE / "gen-external-classifier"
    ckpt = root / f"seed_{seed}" / "external_classifier.pkl"
    man = json.load(open(root / f"seed_{seed}" / "classifier_manifest.json", encoding="utf-8"))
    model = load_model(ckpt)
    h = param_hash(model)
    expect = str((man.get("classifier") or {}).get("param_hash"))
    if expect and h != expect:
        raise AssertionError(
            f"external classifier hash {h[:16]} != manifest {expect[:16]} -- "
            "the instrument on disk is not the one that was gated"
        )
    return model, {
        "path": str(ckpt),
        "param_hash": h,
        "param_count": (man.get("classifier") or {}).get("param_count"),
        "gate_balacc_on_real": ((man.get("gate") or {}).get("external") or {}).get(
            "balanced_accuracy"
        ),
    }


def load_decoder(seed=0):
    from chlu.core.gen_bridge import param_hash
    from chlu.experiments.exp_gen_cell import load_model

    d = BUNDLE / "gen-decoder-lift" / "winner"
    ckpt = d / f"decoder_v1_s{seed}.pkl"
    model = load_model(ckpt)
    h = param_hash(model)
    hashes = json.load(open(d / "winner_hashes.json", encoding="utf-8"))
    return model, {"path": str(ckpt), "param_hash": h, "winner_hashes": hashes}


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_np)
    return str(path)


def _np(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
