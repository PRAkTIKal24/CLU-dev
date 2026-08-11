"""⭐⭐ **The graded-novelty read** — C2W11 VALUE leg ii (§A34.4).

> *"Close to dog, not a dog, snout anomalous"* = **per-particle capture / depth /
> residual diagnostics composed per feature + overlap-as-confidence**
> (§A20.3(c)), trainable via **feature-dropout-as-pseudo-novelty** (loss term
> (e)).

⛔ **Wells are never named semantically** (`PREREG-TierII.md` §2.6): wells,
channels and features carry integer indices and nothing else. The sentence above
is the *shape* of the claim a user would make about an ANSWER, never a claim
that well 7 means "snout".

## ⛔⛔ The feature-set contract, and it is load-bearing

The theorist's §1(e)(v) names a **cross-term conflict that was unflagged
anywhere in the program**: the vacuum reports ``lambda_min ~ 2 alpha`` and depth
``~ 0``, so **if term (c) succeeds** in pulling written sites' ``lambda_min``
toward ``2 alpha`` it makes written and unwritten sites *spectrally
indistinguishable in* ``lambda_min`` — destroying ``lambda_min`` as a novelty
feature. ⇒ **the novelty head keys on DEPTH, ``lambda_2nd`` and the
participation ratio, and NEVER on ``lambda_min``.** That exclusion is enforced
here in code (:data:`NOVELTY_FEATURES`), not in prose.

## ⛔ Designed negatives (pytest-asserted in `tests/test_c2w11_organizer.py`)

1. **permuted payloads** ⇒ AUROC ~ 0.5 — the *registered* negative
   (`PREREG-C2W11.md` §5 V2). ⚠ See :func:`novelty_negatives_note`: on a
   **depth-keyed** channel this negative is measured to be NON-DISCRIMINATING,
   and that is reported as a finding rather than quietly dropped.
2. **blank store** ⇒ AUROC ~ 0.5 (no site is written, so no channel is "known").
3. **shuffled labels** ⇒ AUROC ~ 0.5 (the instrument's own null).

## ⭐ Overlap-as-confidence has a banked shape to honour
**confident ⇒ the ``k`` particles collapse to ``F`` unique wells; unfamiliar ⇒
scattered guesses.** :func:`collapse_statistic` is reported beside every AUROC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "NOVELTY_FEATURES",
    "PSI_FEATURES",
    "particle_descriptors",
    "NoveltyHead",
    "novelty_head_params",
    "auroc",
    "ece",
    "collapse_statistic",
    "novelty_ledger",
    "novelty_negatives_note",
]

#: The psi descriptor ``u_f``, in order. ``q*`` blocks first, then the physics.
PSI_FEATURES = ("q_addr", "q_payload", "V", "grad_norm", "residual",
                "displacement", "capture_w")

#: ⛔ The novelty head's feature set. ``lambda_min`` is **deliberately absent**
#: (the (c)/(e) cross-term contract above); ``q*`` itself is absent so the head
#: cannot become a classifier on the launch geometry.
NOVELTY_FEATURES = ("residual", "grad_norm", "lambda_2nd", "participation_ratio",
                    "displacement", "capture_w")


def _pr_of(vec: jnp.ndarray) -> jnp.ndarray:
    """Participation ratio of a unit eigenvector: 1 (localized) .. D (spread)."""
    u2 = vec ** 2
    return (jnp.sum(u2, axis=-1) ** 2) / (jnp.sum(u2 ** 2, axis=-1) + 1e-30)


def particle_descriptors(V, q_star: np.ndarray, q0: np.ndarray, *,
                         addr_dim: int, confine: float, ball_radius: float,
                         batch: int = 512) -> Dict[str, np.ndarray]:
    """``(B, k, dim) -> the per-particle diagnostics``, computed from the STORE.

    ``residual = V(q*) - alpha ||q*||^2`` is the **vacuum-subtracted depth at the
    settled point**: in an undug region ``V ~ alpha||q||^2`` exactly, so the
    residual is ~0 there and negative at a dug site. That subtraction is what
    makes the descriptor a *store* statistic rather than a distance-to-origin
    statistic.

    ``capture_w = exp(-||grad V|| / (2 alpha R))`` — a smooth capture likelihood
    (the reference scale ``2 alpha R`` is the vacuum gradient at the launch
    shell), used to weight the pooled psi and reported as its own feature.
    """
    q = jnp.asarray(q_star, dtype=jnp.float32)
    B, k, dim = q.shape
    flat = q.reshape(-1, dim)
    a = float(confine)
    grad_ref = 2.0 * a * float(ball_radius)

    @eqx.filter_jit
    def go(z):
        v = jax.vmap(lambda x: jnp.reshape(V(x), ()))(z)
        g = jax.vmap(jax.grad(lambda x: jnp.reshape(V(x), ())))(z)
        H = jax.vmap(jax.hessian(lambda x: jnp.reshape(V(x), ())))(z)
        H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
        w, U = jnp.linalg.eigh(H)
        return v, jnp.linalg.norm(g, axis=-1), w[:, 0], w[:, 1], _pr_of(U[:, :, 0])

    vs, gn, l1, l2, pr = [], [], [], [], []
    for lo in range(0, flat.shape[0], int(batch)):
        out = go(flat[lo:lo + int(batch)])
        for dst, src in zip((vs, gn, l1, l2, pr), out):
            dst.append(np.asarray(src))
    V_ = np.concatenate(vs).reshape(B, k)
    G_ = np.concatenate(gn).reshape(B, k)
    L1 = np.concatenate(l1).reshape(B, k)
    L2 = np.concatenate(l2).reshape(B, k)
    PR = np.concatenate(pr).reshape(B, k)
    qn = np.asarray(q_star, dtype=float)
    res = V_ - a * (qn ** 2).sum(-1)
    disp = np.linalg.norm(qn - np.asarray(q0, dtype=float), axis=-1)
    cap = np.exp(-G_ / grad_ref)
    return {"q_addr": qn[..., :int(addr_dim)], "q_payload": qn[..., int(addr_dim):],
            "V": V_, "grad_norm": G_, "residual": res, "lambda_min": L1,
            "lambda_2nd": L2, "participation_ratio": PR, "displacement": disp,
            "capture_w": cap}


def _stack(d: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    cols = []
    for n in names:
        x = np.asarray(d[n], dtype=np.float32)
        cols.append(x if x.ndim == 3 else x[..., None])
    return np.concatenate(cols, axis=-1)


def psi_input(d: Dict[str, np.ndarray]) -> np.ndarray:
    """``(B, k, u_dim)`` — psi's descriptor set."""
    return _stack(d, PSI_FEATURES)


def novelty_input(d: Dict[str, np.ndarray]) -> np.ndarray:
    """``(B, k, n_dim)`` — the novelty head's set. ⛔ No ``lambda_min``."""
    return _stack(d, NOVELTY_FEATURES)


class NoveltyHead(eqx.Module):
    """``s_f = sigma( g(u_f) )`` per particle — the per-feature novelty channel.

    Trained with **log-loss** (strictly proper, loss term (e)); ⛔ **AUROC is the
    REPORTED statistic and is never trained against**. The head also emits the
    set-level answer confidence used by V2b, as the mean of ``1 - s_f`` weighted
    by capture — *overlap-as-confidence* in its cheapest defensible form.
    """

    mlp: eqx.nn.MLP
    n_dim: int = eqx.field(static=True)

    def __init__(self, n_dim: int, key, *, hidden: int = 16, depth: int = 2):
        self.n_dim = int(n_dim)
        self.mlp = eqx.nn.MLP(int(n_dim), 1, int(hidden), max(int(depth) - 1, 1),
                              activation=jax.nn.tanh, key=key)

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """``(B, k, n_dim) -> (B, k)`` logits (high = NOVEL)."""
        return jax.vmap(jax.vmap(lambda x: jnp.reshape(self.mlp(x), ())))(u)


def novelty_head_params(head: NoveltyHead) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(head, eqx.is_inexact_array))))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUROC with ties handled by mid-rank. Returns ``nan`` if degenerate."""
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # mid-rank for ties
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Expected calibration error of a confidence against a 0/1 correctness."""
    c = np.asarray(conf, dtype=float).ravel()
    k = np.asarray(correct, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(c, edges[1:-1]), 0, int(n_bins) - 1)
    tot, out, bins = len(c), 0.0, []
    for b in range(int(n_bins)):
        m = idx == b
        if not m.any():
            continue
        gap = abs(float(c[m].mean()) - float(k[m].mean()))
        out += m.sum() / tot * gap
        bins.append({"bin": b, "n": int(m.sum()), "conf": float(c[m].mean()),
                     "acc": float(k[m].mean())})
    return {"ece": float(out), "n_bins": int(n_bins), "n": int(tot),
            "brier": float(np.mean((c - k) ** 2)), "bins": bins}


def collapse_statistic(occ: np.ndarray, f_subset: int) -> Dict[str, float]:
    """⭐ **Overlap-as-confidence's banked shape**: *confident ⇒ the ``k``
    particles collapse to ``F`` unique wells; unfamiliar ⇒ scattered guesses.*

    Reported beside every AUROC (the task's explicit requirement).
    """
    o = np.asarray(occ)
    uniq = np.array([len(np.unique(o[i])) for i in range(o.shape[0])], dtype=float)
    return {"mean_unique_wells": float(uniq.mean()),
            "sd_unique_wells": float(uniq.std(ddof=1)) if len(uniq) > 1 else 0.0,
            "frac_collapsed_to_F": float((uniq == float(f_subset)).mean()),
            "F": int(f_subset), "k": int(o.shape[1])}


def novelty_ledger(head: NoveltyHead, *, psi_params: int, extra: Optional[Dict] = None
                   ) -> Dict[str, Any]:
    """The novelty channel's byte ledger + its scoring-rule declaration."""
    n = novelty_head_params(head)
    row = {"novelty_head_params": int(n), "novelty_head_bytes": int(n * 4),
           "psi_params": int(psi_params), "psi_bytes": int(psi_params * 4),
           "novelty_features": list(NOVELTY_FEATURES),
           "lambda_min_excluded": True,
           "lambda_min_exclusion_reason":
               "loss-package §1(e)(v): if term (c) succeeds, written and "
               "unwritten sites become spectrally indistinguishable in "
               "lambda_min, so lambda_min is barred as a novelty feature",
           "objective": "log-loss (strictly proper)",
           "auroc_is_reported_not_trained": True}
    row.update(extra or {})
    return row


def novelty_negatives_note() -> str:
    """⚠ The registered negative, and what it was MEASURED to do (a finding).

    `PREREG-C2W11.md` §5 V2 registers *permuted payloads ⇒ AUROC ~ 0.5*. That
    negative is written for a **payload-keyed** channel. This spoke's channel is
    **depth-keyed** by the (c)/(e) feature-set contract above, and permuting
    ``v_j`` leaves every well written and every depth intact — so the negative is
    **structurally non-discriminating here** (it cannot produce both outcomes,
    §A37's own criterion). ⛔ It is still run and reported; two negatives that DO
    bite are shipped beside it (blank store; shuffled labels), and all three are
    pytest-asserted.
    """
    return novelty_negatives_note.__doc__ or ""
