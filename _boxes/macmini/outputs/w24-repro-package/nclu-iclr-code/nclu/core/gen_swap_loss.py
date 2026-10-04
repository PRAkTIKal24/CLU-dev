"""gen swap loss."""

from __future__ import annotations
from typing import Callable, Optional, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from nclu.core.gen_bridge import quantise_through_store

OWNERSHIP_PAIR_SEED = 20260919


class DecodeJudge:
    def __init__(self, decoder, classifier, feat_shape, info: Optional[dict] = None):
        self.decoder = decoder
        self.classifier = classifier
        self.feat_shape = tuple((int(x) for x in feat_shape))
        self.info = dict(info or {})

    def read_back(self, feats: np.ndarray, batch: int = 256) -> np.ndarray:
        from nclu.eval.gen_probe import decode_features, predict_factors

        Z = np.asarray(feats, np.float32).reshape((-1,) + self.feat_shape)
        n_f = len(self.classifier.heads.linears)
        out = np.empty((Z.shape[0], n_f), np.int64)
        for s in range(0, Z.shape[0], batch):
            e = min(s + batch, Z.shape[0])
            imgs = decode_features(self.decoder, Z[s:e], batch)
            out[s:e] = np.asarray(predict_factors(self.classifier, imgs, batch), np.int64)
        return out


def labels_from_well_channels(well_channels: Sequence, dim: int) -> np.ndarray:
    """``(D,) -> {0..K-1}``: the per-channel well label ``gen_swap_probe`` consumes.

    A channel that belongs to no well is an error, not something to patch: the swap would
    silently leave it at ``x_i`` in every well's intervention and the transfer matrix
    would be reading a different object from the one the store deploys.
    """
    lab = np.full(int(dim), -1, np.int64)
    for w, ch in enumerate(well_channels):
        lab[np.asarray(ch, np.int64)] = int(w)
    if (lab < 0).any():
        raise AssertionError(
            f"{int((lab < 0).sum())} of {int(dim)} channels belong to no well -- a partition with a hole is reported, never silently patched"
        )
    return lab


def derive_fixed_ownership(
    judge: DecodeJudge,
    Zi: np.ndarray,
    Zj: np.ndarray,
    Yi: np.ndarray,
    Yj: np.ndarray,
    well_channels: Sequence,
    dim: int,
    batch: int = 256,
    log: Callable = print,
) -> dict:
    from nclu.core import gen_swap_probe as SP

    K = len(well_channels)
    lab = labels_from_well_channels(well_channels, dim)
    tm = SP.transfer_matrix(judge, Zi, Zj, Yi, Yj, lab, K, batch=batch, log=log)
    own = np.asarray(SP.derive_ownership(tm["T"], vary=None), np.int64)
    if (own < 0).any():
        raise AssertionError(
            f"well(s) {np.flatnonzero(own < 0).tolist()} own no factor -- the swap loss has no target for them and is ill-posed; reported, not patched"
        )
    ident = SP.identity_ceiling(judge, Zi, Yi, batch=batch)
    log(
        f"  ownership FIXED at the warm start: own={own.tolist()} (identity ceiling exact={ident['exact']:.4f})"
    )
    return {
        "own": own,
        "transfer_T": np.asarray(tm["T"]).tolist(),
        "transfer_S": np.asarray(tm["S"]).tolist(),
        "well_sizes": tm["well_sizes"],
        "identity_ceiling": ident,
        "n_pairs": int(np.asarray(Yi).shape[0]),
        "note": "derived ONCE on the shared warm start and held CONSTANT for every arm and every step; gen_swap_probe: 'a refinement loop must fix it, or the objective moves under the optimiser'",
    }


def pad_well_channels(well_channels: Sequence):
    wc = [np.asarray(w, np.int64) for w in well_channels]
    if any((int(w.size) == 0 for w in wc)):
        raise ValueError("an empty well is reported, never silently merged")
    width = max((int(w.size) for w in wc))
    out = np.empty((len(wc), width), np.int64)
    for i, w in enumerate(wc):
        out[i, : w.size] = w
        if w.size < width:
            out[i, w.size :] = w[0]
    return (jnp.asarray(out, jnp.int32), [int(w.size) for w in wc])


def draw_partners(n: int, key):
    """``(idx_i, idx_j)`` -- every row of the batch paired with another row.

    A random partner, not a curated one.  ``gen_swap_probe.make_pairs`` draws pairs that
    differ in EVERY factor because it is reading a transfer matrix off them; a training
    loss does not need that and should not have it -- a pair that happens to agree on the
    owned factor carries the equally real instruction *"then nothing should change"*, and
    filtering those out would train the encoder on only the easy half of the distribution.
    A cyclic shift by a random non-zero amount gives a derangement for free: no row is
    ever paired with itself, and every row appears exactly once as ``i`` and once as ``j``.
    """
    idx_i = jnp.arange(n)
    shift = jax.random.randint(key, (), 1, max(2, n))
    return (idx_i, jnp.mod(idx_i + shift, n))


def intended_tuple(Yi: jnp.ndarray, Yj: jnp.ndarray, f_owned: int) -> jnp.ndarray:
    """``y_i`` with the owned factor's column taken from ``y_j``.  ``(P, F)``."""
    return Yi.at[:, int(f_owned)].set(Yj[:, int(f_owned)])


def swapped_features(
    zf: jnp.ndarray, idx_i: jnp.ndarray, idx_j: jnp.ndarray, channels: jnp.ndarray
) -> jnp.ndarray:
    """``z_i`` with ``channels`` taken from ``z_j``.  ``(P, D)``, traceable.

    The same intervention ``gen_swap_probe.swap_read`` performs in NumPy, so the loss and
    the probe measure one object.  Nothing is normalised, rescaled or blended: a swap is
    a replacement.
    """
    return zf[idx_i].at[:, channels].set(zf[idx_j][:, channels])


def swap_loss_terms(
    model,
    classifier,
    xb,
    Yb,
    idx_i,
    idx_j,
    well: int,
    wells_padded,
    own,
    code_grad: str = "straight_through",
):
    """``(loss, exact_rate)`` for ONE well's swap over the batch.

    ``loss`` is the mean per-factor cross-entropy of the judge's read-back against the
    intended composite tuple -- differentiable into the encoder through
    ``quantise_through_store``'s straight-through estimator.

    ``exact_rate`` is the **non-differentiable** companion: the fraction of swaps the
    judge reads back as exactly the intended tuple, i.e. the very statistic
    ``gen_swap_probe.score_partition`` calls ``exact``.  It is carried alongside so the
    training log shows the quantity the reader cares about and not only a surrogate that
    is falling.

    ``classifier`` is closed over as a constant.  It is not in ``model`` and therefore
    cannot be in the Optax partition; the caller asserts its hash before and after.
    """
    encoder, store, decoder = model
    z = jax.vmap(encoder)(xb)
    zf = z.reshape((z.shape[0], -1))
    zs = swapped_features(zf, idx_i, idx_j, wells_padded[int(well)])
    dec_in, _gather, _codes = quantise_through_store(store, zs, code_grad, 0.0)
    imgs = jax.vmap(decoder)(dec_in.reshape((-1,) + tuple(store.feat_shape)))
    logits = jax.vmap(classifier)(imgs)
    tgt = intended_tuple(Yb[idx_i], Yb[idx_j], int(own[int(well)]))
    loss = jnp.asarray(0.0, jnp.float32)
    hit = jnp.ones(tgt.shape[0], bool)
    for f, lg in enumerate(logits):
        lp = jax.nn.log_softmax(lg, axis=-1)
        loss = loss - jnp.mean(jnp.take_along_axis(lp, tgt[:, f][:, None], axis=1))
        hit = hit & (jnp.argmax(lg, axis=-1) == tgt[:, f])
    return (loss / float(len(logits)), jnp.mean(hit.astype(jnp.float32)))
