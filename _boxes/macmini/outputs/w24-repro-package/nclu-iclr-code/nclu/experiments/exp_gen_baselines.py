"""exp gen baselines."""

from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import numpy as np
from nclu.eval.gen_probe import decode_features, probe_accuracy


def score_composition(decoder, probe, feats, labels, names, feat_shape, batch) -> dict:
    """Decode features through THE frozen decoder, read them with THE frozen probe."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(feat_shape))
    imgs = decode_features(decoder, Z, int(batch))
    out = probe_accuracy(probe, imgs, labels, names, int(batch))
    out["decoded_mean"] = float(imgs.mean())
    return out


def _slice_dense(req, n: int):
    from nclu.core.gen_request import DenseRequests

    n = int(n)
    return DenseRequests(
        req.cls_targets[:n],
        req.val_targets[:n],
        req.extra_factor[:n],
        req.extra_cls[:n],
        req.extra_val[:n],
        req.n_factors,
    )
