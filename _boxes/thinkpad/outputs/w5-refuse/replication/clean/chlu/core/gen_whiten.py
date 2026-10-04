"""⭐ **PER-WELL FEATURE EQUALISATION** — the only thing the metric k-means sees.

⭐ **DIAL: feature scaling before the store.**  ⛔ The encoder, the decoder, the
**partition** and ``K`` are all UNCHANGED.  Nothing in this module knows what a
factor is, what a label is or what a decoder is; it takes a channel partition that
was **already fitted on unwhitened features** and an array of features, and it
returns an invertible per-well linear map.

⛔ **ORDER IS RULED AND IT MATTERS (R43.1): the grouping is fitted on UNWHITENED
features FIRST, then the transform is fitted.**  Whitening before grouping would
change the partition — and `gen-loss-attribution` **exonerated** the partition (it
costs 9.5 % of ``scale`` and **0 %** of ``orientation``), so moving it would move the
one part of the path that is not broken.  :func:`assert_grouping_is_unwhitened`
makes that order an assertion rather than a comment.

**The hypothesis this module exists to test** (R40.B, untested until now): k-means
minimises squared error, so it spends its codewords on **high-variance directions**;
if ``scale`` and ``orientation`` live in **low-variance** directions of these
features they are quantised away — not because the information is absent (the frozen
encoder carries them at 0.9352 / 0.9915) but because the objective does not value
them.  ⛔ :func:`well_variance_spectrum` **tests that premise first**; the transforms
below are only worth fitting if it survives.

**Three transforms, one interface** (:class:`WellTransform`):

- ``identity``     — ⭐ the control arm.  The *same* code path, so the unwhitened
  baseline is re-measured through the identical plumbing rather than transcribed.
- ``pca_whiten``   — ⭐ **the primary arm.**  Full PCA whitening **within each well's
  own sub-space**: ``z = diag((λ+ε)^(-1/2)) Vᵀ (x − μ)``.  ⛔ Not across ``D``.
- ``diag_std``     — the secondary arm: per-channel ``(x − μ)/σ``.

⭐ **Why PCA whitening and not ZCA:** Euclidean k-means is invariant under an
orthogonal change of basis, so ZCA (``V diag(...) Vᵀ``) and PCA (``diag(...) Vᵀ``)
whitening induce **exactly the same clustering**; PCA is stored and inverted with the
same ``d_k²`` floats and hands the variance spectrum straight to the Leg-1 analysis.

⭐ **Deployment, priced two ways and both reported** (:func:`transform_byte_ledger`):

1. **fit-time only — 0 deployed bytes.**  Fit the codebook in the whitened metric,
   then map the ``M`` centres per well **back** through the inverse and ship them
   un-whitened.  The request-side read (``request → address book → code →
   read_from_codes``) is then a plain gather of un-whitened centres and ⛔ never
   evaluates the transform.  :func:`unwhiten_store` builds exactly that store and
   :func:`assert_deployment_equivalence` proves the two agree.
2. **deployed — ``Σ_k d_k²·4 + D·4`` bytes.**  Any path that maps a *feature vector*
   to a code at read time (``address_codes``) must evaluate the forward map.
   ⛔ At the shipped cell that is **3,600,672 B = 6.87× the whole 524,288 B store
   budget** — a much larger model, and it is never reported as free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: The arms this module can build.  ⛔ ``identity`` is not a placeholder: it is the
#: control that makes the unwhitened baseline a *measurement* of this code path.
TRANSFORMS = ("identity", "pca_whiten", "diag_std")

#: The store budget every byte ledger in gen-phase-1 is read against (R26.3).
STORE_BYTE_BUDGET = 524_288


# ---------------------------------------------------------------------------
# 0. ⛔ The order guard — the partition must predate the transform
# ---------------------------------------------------------------------------


def assert_grouping_is_unwhitened(well_channels, reference_channels, label: str = "") -> dict:
    """⛔ The partition used here must be **channel-for-channel** the one fitted on
    UNWHITENED features (R43.1).

    ⭐ Sizes are not enough — two partitions can share a size vector and separate
    different channels — so this compares the channel tuples themselves, which is
    the same standard ``exp_gen_attribution.load_attrib_store`` already holds the
    store to.
    """
    a = tuple(tuple(int(i) for i in w) for w in well_channels)
    b = tuple(tuple(int(i) for i in w) for w in reference_channels)
    if a != b:
        raise AssertionError(
            f"{label or 'grouping'}: the well channels are NOT the unwhitened "
            "partition — ⛔ the transform has moved the grouping, which "
            "gen-loss-attribution EXONERATED. Fit the grouping on unwhitened "
            "features FIRST, then whiten (R43.1)."
        )
    return {
        "grouping_fitted_on": "UNWHITENED features (asserted channel-for-channel)",
        "n_wells": len(a),
        "well_sizes": [len(w) for w in a],
        "dim": int(sum(len(w) for w in a)),
        "asserted": True,
    }


# ---------------------------------------------------------------------------
# 1. ⭐ LEG 1 — TEST THE PREMISE BEFORE ACTING ON IT
# ---------------------------------------------------------------------------


def between_class_variance(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column between-class variance ``Σ_v (n_v/N)(mean_v − mean)²``.

    ⭐ Summed over an orthonormal basis this is basis-independent, which is why
    :func:`well_variance_spectrum` can report *both* a share-of-variance number and a
    where-in-the-spectrum number from the same quantity.

    ⚠ **This statistic is POSITIVELY BIASED under the null** — see
    :func:`debias_between_class_variance`, which is what the Leg-1 numbers use.
    """
    T = np.asarray(scores, np.float64)
    yy = np.asarray(y, np.int64)
    n = T.shape[0]
    if n == 0:
        return np.zeros(T.shape[1], np.float64)
    mu = T.mean(axis=0)
    out = np.zeros(T.shape[1], np.float64)
    for v in np.unique(yy):
        sel = yy == v
        nv = int(sel.sum())
        if nv == 0:
            continue
        out += (nv / n) * (T[sel].mean(axis=0) - mu) ** 2
    return out


def debias_between_class_variance(
    b: np.ndarray, eigenvalues: np.ndarray, n: int, n_values: int
) -> np.ndarray:
    """⭐⭐ **Subtract the null.**  ``E[b_j] = ((V−1)/N)·λ_j`` when the labels carry
    nothing, so the raw statistic awards every **high-variance** axis a share of the
    signal **for free** — and "the signal is in the leading axes" would then be a
    trivial truth of the estimator rather than a fact about the features.

    ⛔ This bias is proportional to ``λ_j``, i.e. it is concentrated in exactly the
    directions the hypothesis is about, so it is subtracted rather than bounded.
    Clipped at 0: a negative excess is "no evidence", not negative evidence.
    """
    lam = np.asarray(eigenvalues, np.float64)
    bias = (max(int(n_values) - 1, 0) / max(int(n), 1)) * lam
    return np.clip(np.asarray(b, np.float64) - bias, 0.0, None)


def _median_rank(weights: np.ndarray, frac: float = 0.5) -> int:
    """Smallest ``r`` (1-based) with ``Σ_{j<=r} w_j >= frac·Σ w``."""
    w = np.asarray(weights, np.float64)
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return int(w.size)
    c = np.cumsum(w) / tot
    return int(np.searchsorted(c, float(frac)) + 1)


def well_variance_spectrum(
    X: np.ndarray,
    channels: Sequence[int],
    labels: np.ndarray,
    factor_names: Sequence[str],
    cardinalities: Sequence[int],
    eps: float = 1e-12,
) -> dict:
    """⭐⭐ **THE LEG-1 ROW.**  Where in one well's variance spectrum does each factor's
    discriminative signal sit?

    Returns, per factor:

    ``variance_explained``      ``(Σ_j b_j) / (Σ_j λ_j)`` — the share of the well's
                                total variance the factor's class means account for.
    ``rank_frac_median``        the normalised principal-axis rank ``r/d_k`` at which
                                half the factor's between-class variance has
                                accumulated.  ⭐ **This is the literal "does it live
                                in low-variance directions?" number**, and it is the
                                one the pre-registered premise test adjudicates on.
    ``frac_signal_in_top10pct`` how much of the factor's signal the leading 10 % of
                                the axes carry.
    ``var_frac_at_median_rank`` how much of the WELL's variance you must keep to keep
                                half of this factor's signal.

    ⛔ Every row carries ``n_values_realised`` (R31's fourth strike): a factor
    realising one value has no between-class variance and its numbers are stamped
    ``degenerate`` rather than emitted bare.
    """
    idx = np.asarray(channels, np.int64)
    # ⚠ Slice FIRST, then widen: casting the whole (N, D) block to float64 before
    # gathering one well costs ~270 MB per call on a 16 GB machine.
    Z = np.asarray(np.asarray(X)[:, idx], np.float64)
    n, d = Z.shape
    mu = Z.mean(axis=0)
    Zc = Z - mu
    # ⚠ Covariance via the Gram form on the centred block: d_k <= 744 here, so the
    # (d, d) eigendecomposition is cheap and exact; the (n, n) form would not be.
    C = (Zc.T @ Zc) / max(1, n - 1)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(-lam)
    lam = np.clip(lam[order], 0.0, None)
    V = V[:, order]
    T = Zc @ V
    tot_var = float(lam.sum())
    Y = np.asarray(labels, np.int64)
    per_factor = {}
    for j, nm in enumerate(factor_names):
        nvr = int(np.unique(Y[:, j]).size) if Y.size else 0
        b_raw = between_class_variance(T, Y[:, j])
        # ⭐ the primary statistic is the NULL-SUBTRACTED one (see the docstring of
        # debias_between_class_variance); the raw one ships beside it.
        b = debias_between_class_variance(b_raw, lam, n, nvr)
        bsum = float(b.sum())
        r = _median_rank(b, 0.5)
        top = max(1, int(np.ceil(0.1 * d)))
        per_factor[str(nm)] = {
            "n_values_realised": nvr,
            "cardinality": int(cardinalities[j]),
            "degenerate": bool(nvr <= 1),
            "variance_explained": float(bsum / tot_var) if tot_var > eps else float("nan"),
            "variance_explained_raw": (
                float(b_raw.sum() / tot_var) if tot_var > eps else float("nan")),
            "null_bias_share_of_total_variance": float(
                max(nvr - 1, 0) / max(n, 1)),
            "rank_frac_median_raw": float(_median_rank(b_raw, 0.5) / d),
            "between_class_variance": bsum,
            "rank_median": int(r),
            "rank_frac_median": float(r / d),
            "frac_signal_in_top10pct": (
                float(b[:top].sum() / bsum) if bsum > eps else float("nan")
            ),
            "var_frac_at_median_rank": (
                float(lam[:r].sum() / tot_var) if tot_var > eps else float("nan")
            ),
        }
        if nvr <= 1:
            per_factor[str(nm)]["status"] = "NOT APPLICABLE (1 realised value — R31)"
    return {
        "d_k": int(d),
        "n_rows": int(n),
        "total_variance": tot_var,
        "eigenvalues_head": [float(x) for x in lam[:16]],
        "participation_ratio": (
            float(lam.sum() ** 2 / float((lam**2).sum())) if tot_var > eps else float("nan")
        ),
        "per_factor": per_factor,
    }


def premise_verdict(
    spectra: Sequence[dict],
    factor_names: Sequence[str],
    probe_factors: Sequence[str] = ("scale", "orientation"),
    reference_factors: Sequence[str] = ("floor_hue", "wall_hue"),
    rank_ratio: float = 2.0,
    share_ratio: float = 0.5,
) -> dict:
    """⛔⭐ **PREREG §3, applied mechanically** — does the variance premise survive?

    Each factor is evaluated in **its own headline well** (``argmax_k
    variance_explained``).  The premise is the *direction* claim, so
    ``rank_frac_median`` adjudicates:

    - **HOLDS** iff ``rank_frac_median(f) > rank_ratio · max_ref rank_frac_median``
      for **both** probe factors;
    - **PARTIAL** iff that passes but the supporting share test
      (``variance_explained(f) < share_ratio · min_ref variance_explained``) fails;
    - ⛔ **DEAD** otherwise ⇒ **STOP after Leg 1. Do not whiten.**
    """

    def _headline(nm):
        vals = [s["per_factor"][nm]["variance_explained"] for s in spectra]
        k = int(np.nanargmax(vals))
        return k, spectra[k]["per_factor"][nm]

    head = {str(nm): _headline(str(nm)) for nm in factor_names}
    ref_rank = max(float(head[str(r)][1]["rank_frac_median"]) for r in reference_factors)
    ref_share = min(float(head[str(r)][1]["variance_explained"]) for r in reference_factors)
    rows = {}
    for nm in probe_factors:
        row = head[str(nm)][1]
        rows[str(nm)] = {
            "headline_well": int(head[str(nm)][0]),
            "rank_frac_median": float(row["rank_frac_median"]),
            "variance_explained": float(row["variance_explained"]),
            "direction_test": bool(
                float(row["rank_frac_median"]) > float(rank_ratio) * ref_rank
            ),
            "share_test": bool(
                float(row["variance_explained"]) < float(share_ratio) * ref_share
            ),
        }
    direction = all(r["direction_test"] for r in rows.values())
    share = all(r["share_test"] for r in rows.values())
    verdict = "HOLDS" if (direction and share) else ("PARTIAL" if direction else "DEAD")
    return {
        "verdict": verdict,
        "direction_test_passed": bool(direction),
        "share_test_passed": bool(share),
        "reference_rank_frac_median_max": float(ref_rank),
        "reference_variance_explained_min": float(ref_share),
        "rank_ratio": float(rank_ratio),
        "share_ratio": float(share_ratio),
        "per_probe_factor": rows,
        "headline_well_per_factor": {k: int(v[0]) for k, v in head.items()},
        "stop_after_leg1": bool(verdict == "DEAD"),
        "rule": (
            "PREREG §3 — the premise is the DIRECTION claim: rank_frac_median(f) > "
            "2 x max over {floor_hue, wall_hue}, each in its own headline well, for "
            "BOTH scale and orientation. DEAD => do not whiten."
        ),
    }


# ---------------------------------------------------------------------------
# 2. The transform itself
# ---------------------------------------------------------------------------


@dataclass
class WellTransform:
    """An **invertible, per-well linear** map on the feature vector.

    ``forward(x)[:, w_k] = (x[:, w_k] − mean_k) @ A_k``  and
    ``inverse(z)[:, w_k] = z[:, w_k] @ B_k + mean_k``, with ``B_k = A_k^{-1}``
    held explicitly so the inverse is a matmul and not a solve.

    ⛔ The wells **partition** the coordinates, so the map is block-diagonal by
    construction and can never mix two wells — which is exactly what keeps the
    already-exonerated partition out of the experiment.
    """

    kind: str
    well_channels: tuple
    dim: int
    means: tuple  # per well: (d_k,) float64
    fwd: tuple  # per well: (d_k, d_k) float64  — ⛔ diag stored dense for ONE code path
    inv: tuple  # per well: (d_k, d_k) float64
    eigenvalues: tuple
    eps: float
    n_rows_fit: int
    scale_match: bool = True
    #: ⭐ The per-well uniform scalar of the matched-compute control (1.0 = none).
    #: ⛔ Metric-neutral: k-means is equivariant under a uniform scaling.
    scales: tuple = ()

    # -- construction --------------------------------------------------------
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        well_channels,
        kind: str = "pca_whiten",
        eps: float = 1e-6,
        scale_match: bool = True,
    ) -> "WellTransform":
        """Fit on the **store-training** features, per well, after the grouping exists.

        ``eps`` is **relative**: ``λ + eps·mean(λ)``.  ⭐ An absolute floor would be a
        different regulariser on every well, because the wells have very different
        scales; a relative one whitens each well's own spectrum the same way.

        ⭐⭐ **``scale_match`` is a MATCHED-COMPUTE control, not a second treatment.**
        :func:`chlu.core.gen_store.train_store` anneals a softmax over
        ``−d²/(2τ²)`` with ``τ`` in **absolute distance units**, and Adam's step size
        is absolute too.  Whitening changes a well's overall scale by ~15× at the
        shipped cell, so an unmatched whitened arm would see **a different anneal and
        a different effective learning rate** — and would lose for a reason that has
        nothing to do with the metric.  ⛔ That is a confound, not a result.
        The remedy is a **single scalar per well** chosen so the transformed block has
        the same mean squared norm as the raw one.  ⭐ A uniform scaling is
        **metric-neutral**: Euclidean k-means is equivariant under it, so the induced
        clustering is *identical* — it only restores commensurability with ``τ`` and
        the learning rate.  It is folded into ``A_k`` and ``A_k^{-1}``, so the byte
        ledger and the exact invertibility are unchanged.  For ``identity`` the factor
        is exactly ``1.0`` and the control stays bit-identical to the shipped path.
        """
        if str(kind) not in TRANSFORMS:
            raise ValueError(f"kind must be one of {TRANSFORMS}, got {kind!r}")
        Z = np.asarray(X)
        if Z.ndim > 2:
            Z = Z.reshape((Z.shape[0], -1))
        wc = tuple(tuple(int(i) for i in w) for w in well_channels)
        means, fwd, inv, eig, scales = [], [], [], [], []
        for w in wc:
            idx = np.asarray(w, np.int64)
            # ⚠ Gather in the cache's dtype, widen the (N, d_k) block only.
            B = np.asarray(Z[:, idx], np.float64)
            mu = B.mean(axis=0)
            d = idx.size
            if str(kind) == "identity":
                A = np.eye(d)
                Ai = np.eye(d)
                lam = np.zeros(d)
                mu = np.zeros(d)  # ⛔ the control is the LITERAL identity, not a centring
            elif str(kind) == "diag_std":
                sd = B.std(axis=0)
                floor = float(eps) * float(max(sd.mean(), 1e-30))
                sd = np.maximum(sd, floor)
                A = np.diag(1.0 / sd)
                Ai = np.diag(sd)
                lam = sd**2
            else:  # pca_whiten
                Bc = B - mu
                C = (Bc.T @ Bc) / max(1, Bc.shape[0] - 1)
                lam_raw, V = np.linalg.eigh(C)
                order = np.argsort(-lam_raw)
                lam_raw = np.clip(lam_raw[order], 0.0, None)
                V = V[:, order]
                floor = float(eps) * float(max(lam_raw.mean(), 1e-30))
                lam = lam_raw + floor
                s = 1.0 / np.sqrt(lam)
                A = V * s[None, :]  # (d, d): x_c @ A = s * (Vᵀ x_c)
                Ai = (V * np.sqrt(lam)[None, :]).T  # exact inverse of A
            if bool(scale_match):
                pre = (B - mu) @ A
                num = float(np.mean(np.sum(B * B, axis=1)))
                den = float(np.mean(np.sum(pre * pre, axis=1)))
                alpha = float(np.sqrt(num / den)) if den > 0 else 1.0
                A = A * alpha
                Ai = Ai / alpha
                scales.append(alpha)
            else:
                scales.append(1.0)
            means.append(mu)
            fwd.append(A)
            inv.append(Ai)
            eig.append(lam)
        return cls(
            kind=str(kind),
            well_channels=wc,
            dim=int(sum(len(w) for w in wc)),
            means=tuple(means),
            fwd=tuple(fwd),
            inv=tuple(inv),
            eigenvalues=tuple(eig),
            eps=float(eps),
            n_rows_fit=int(Z.shape[0]),
            scale_match=bool(scale_match),
            scales=tuple(scales),
        )

    # -- application ---------------------------------------------------------
    @property
    def well_sizes(self) -> list:
        return [len(w) for w in self.well_channels]

    def _apply(self, X: np.ndarray, mats, means, add: bool, batch: int = 8192) -> np.ndarray:
        Z = np.asarray(X, np.float32)
        flat = Z.reshape((Z.shape[0], -1))
        if flat.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {flat.shape[1]}")
        out = np.empty_like(flat, dtype=np.float32)
        for i in range(0, flat.shape[0], int(batch)):
            chunk = flat[i : i + int(batch)].astype(np.float64)
            for w, A, mu in zip(self.well_channels, mats, means, strict=True):
                idx = np.asarray(w, np.int64)
                blk = chunk[:, idx]
                blk = (blk @ A + mu) if add else ((blk - mu) @ A)
                out[i : i + int(batch), idx] = blk.astype(np.float32)
        return out

    def forward(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        """Feature space -> the metric k-means will see.  ⛔ Never applied in place."""
        return self._apply(X, self.fwd, self.means, add=False, batch=batch)

    def inverse(self, Z: np.ndarray, batch: int = 8192) -> np.ndarray:
        """⭐ Back to the decoder's usual space.  ⛔ A decoder fed whitened vectors
        measures the transform, not the store."""
        return self._apply(Z, self.inv, self.means, add=True, batch=batch)

    # -- the two things a reviewer checks ------------------------------------
    def round_trip(self, X: np.ndarray, batch: int = 8192) -> dict:
        """⭐ ``inverse(forward(x)) ≈ x``, with the **worst observed** error."""
        Xf = np.asarray(X, np.float32).reshape((np.asarray(X).shape[0], -1))
        back = self.inverse(self.forward(Xf, batch), batch)
        err = np.abs(back.astype(np.float64) - Xf.astype(np.float64))
        scale = float(np.abs(Xf).max()) if Xf.size else 0.0
        return {
            "kind": self.kind,
            "n": int(Xf.shape[0]),
            "max_abs_error": float(err.max()) if err.size else 0.0,
            "mean_abs_error": float(err.mean()) if err.size else 0.0,
            "max_abs_feature": scale,
            "max_rel_error": float(err.max() / scale) if scale > 0 else float("nan"),
        }

    def byte_ledger(self, budget: int = STORE_BYTE_BUDGET) -> dict:
        """⭐ The transform's **deployed** cost, priced both ways (PREREG §9)."""
        return transform_byte_ledger(self.kind, self.well_sizes, budget)


def transform_byte_ledger(kind: str, well_sizes: Sequence[int], budget: int = STORE_BYTE_BUDGET) -> dict:
    """Bytes for a transform over wells of the given sizes — ⛔ in bytes, never in words.

    ``fit_time_only`` is **0** for every kind: the arm can always ship the ``M``
    centres un-whitened and let the request-side read gather them (see
    :func:`unwhiten_store`).  ``deployed`` is what a path that must encode a
    *feature vector* at read time pays.
    """
    d = [int(x) for x in well_sizes]
    D = int(sum(d))
    if str(kind) == "identity":
        deployed = 0
        formula = "0 — the identity arm stores nothing"
    elif str(kind) == "diag_std":
        deployed = 2 * D * 4
        formula = "2·D·4  (per-channel mean and sd)"
    else:
        deployed = sum(x * x for x in d) * 4 + D * 4
        formula = "Σ_k d_k²·4 + D·4  (V with λ^(-1/2) folded in, + the per-well means)"
    inverse_extra = 0 if str(kind) == "identity" else D * 4
    return {
        "kind": str(kind),
        "well_sizes": d,
        "dim": D,
        "sum_d_squared": int(sum(x * x for x in d)),
        "bytes_formula": formula,
        "deployed_bytes": int(deployed),
        "deployed_plus_explicit_inverse_bytes": int(deployed + inverse_extra),
        "fit_time_only_bytes": 0,
        "store_byte_budget": int(budget),
        "deployed_vs_budget": float(deployed / budget) if budget else float("nan"),
        "note": (
            "⭐ TWO PRICES, both reported. fit_time_only = the codebook is FITTED in "
            "the transformed metric and the M centres are stored UN-whitened, so the "
            "request-side read (request → address book → code → read_from_codes) "
            "never evaluates the transform ⇒ 0 deployed bytes. deployed = any path "
            "that maps a FEATURE VECTOR to a code at read time (address_codes) must "
            "evaluate the forward map and pays the bytes above. ⛔ A deployed "
            "transform reported as free is a much larger model reported as a win."
        ),
    }


# ---------------------------------------------------------------------------
# 3. ⭐ The fit-time-only deployment, made a measurement rather than a claim
# ---------------------------------------------------------------------------


def unwhiten_store(store, transform: WellTransform):
    """⭐ Map a store fitted in the transformed metric back to feature space.

    Returns a store whose ``modes[k]`` are the **un-whitened** centres, i.e. exactly
    what a deployment that pays **0 transform bytes** would ship.  ⛔ The channels,
    ``K``, ``M`` and the byte count are untouched — only the numbers in the
    codebook change.
    """
    import equinox as eqx
    import jax.numpy as jnp

    assert_grouping_is_unwhitened(
        store.well_channels, transform.well_channels, "unwhiten_store"
    )
    new = []
    for m, A, mu in zip(store.modes, transform.inv, transform.means, strict=True):
        M = np.asarray(m, np.float64)
        new.append(jnp.asarray(M @ A + mu, jnp.float32))
    return eqx.tree_at(lambda t: t.modes, store, new)


def assert_deployment_equivalence(
    store, store_unwhitened, transform: WellTransform, codes: np.ndarray, tol: float = 1e-3
) -> dict:
    """⛔ Prove the fit-time-only deployment is the same read.

    ``inverse(read_from_codes(store_whitened, c))`` must equal
    ``read_from_codes(store_unwhitened, c)``.  ⭐ If it does, the primary arm costs
    **0 deployed bytes on the request-side path** and the 6.87× ledger applies only
    to a feature-side encode.
    """
    from .gen_store import read_from_codes

    a = np.asarray(read_from_codes(store, np.asarray(codes, np.int64)), np.float32)
    a = transform.inverse(a.reshape((a.shape[0], -1)))
    b = np.asarray(read_from_codes(store_unwhitened, np.asarray(codes, np.int64)), np.float32)
    b = b.reshape((b.shape[0], -1))
    err = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
    scale = float(np.abs(b).max()) if b.size else 0.0
    rel = err / scale if scale > 0 else 0.0
    if rel > float(tol):
        raise AssertionError(
            "the fit-time-only deployment does NOT reproduce the inverted whitened "
            f"read (max rel |Δ| = {rel:.3e} > {tol}) — ⛔ the arm may not be priced "
            "at 0 deployed bytes"
        )
    return {
        "n": int(np.asarray(codes).shape[0]),
        "max_abs_delta": err,
        "max_rel_delta": float(rel),
        "tol": float(tol),
        "verdict": "fit-time-only deployment is EQUIVALENT (0 deployed transform bytes)",
    }


# ---------------------------------------------------------------------------
# 4. ⭐ Leg 4 — the per-well capacity bound, measured the same way in every arm
# ---------------------------------------------------------------------------


def per_well_capacity(
    mi_bits: np.ndarray,
    entropy_bits: Sequence[float],
    cardinalities: Sequence[int],
    factor_names: Sequence[str],
    n_modes: int,
    carry_threshold: float = 0.25,
) -> dict:
    """⭐⭐ How many factor-value combinations must each well resolve, against ``M``?

    A well *carries* a factor iff ``MI(code_k; f)/H(f) >= carry_threshold``; its
    **load** is the product of the carried factors' cardinalities and its **surplus**
    is ``load / M``.

    ⛔ **Aggregate sufficiency is NOT the point and must not be written as the
    cause.**  ``K·log₂M = 30 bits`` exceeds ``log₂(480,000) = 18.87`` — the code is
    sufficient in aggregate.  ⭐ Per well, ``M = 64`` is already exceeded by two hues
    alone (100), by hue × orientation (150), and overwhelmingly by all six (480,000).
    ⭐ Whitening cannot beat this bound; it can only choose which factors spend the
    modes.
    """
    R = np.asarray(mi_bits, np.float64)
    H = np.asarray(entropy_bits, np.float64)
    K = int(R.shape[0])
    # ⭐ ``n_modes`` may be a per-well sequence (``gen-codebook-allocation``); the
    # scalar path is unchanged.
    Ms = ([int(n_modes)] * K if np.isscalar(n_modes) or isinstance(n_modes, (int, np.integer))
          else [int(x) for x in n_modes])
    if len(Ms) != K:
        raise ValueError(f"n_modes has {len(Ms)} entries but there are {K} wells")
    uniform_M = len(set(Ms)) == 1
    M = int(Ms[0])
    wells = []
    for k in range(K):
        M = int(Ms[k])
        frac = np.where(H > 0, R[k] / np.where(H > 0, H, 1.0), 0.0)
        carried = [str(factor_names[j]) for j in range(len(factor_names))
                   if float(frac[j]) >= float(carry_threshold)]
        load = 1
        for nm in carried:
            load *= int(cardinalities[list(factor_names).index(nm)])
        wells.append({
            "well": k,
            "carries": carried,
            "mi_fraction_of_entropy": {str(factor_names[j]): float(frac[j])
                                       for j in range(len(factor_names))},
            "load_combinations": int(load),
            "n_modes": M,
            "surplus_over_M": float(load / M),
            "over_capacity": bool(load > M),
            "bits_needed": float(np.log2(load)) if load > 0 else 0.0,
            "bits_available": float(np.log2(M)),
        })
    n_over = int(sum(1 for w in wells if w["over_capacity"]))
    total = 1
    for c in cardinalities:
        total *= int(c)
    bits_available = float(np.log2(np.asarray(Ms, np.float64)).sum())
    return {
        "carry_threshold": float(carry_threshold),
        # ⛔ ``None`` rather than a lie when the allocation is non-uniform.
        "n_modes": int(Ms[0]) if uniform_M else None,
        "n_modes_per_well": [int(x) for x in Ms],
        "uniform_allocation": bool(uniform_M),
        "n_wells": K,
        "per_well": wells,
        "n_wells_over_capacity": n_over,
        "aggregate_bits_available": bits_available,
        "aggregate_bits_needed": float(np.log2(total)) if total > 0 else 0.0,
        "aggregate_is_sufficient": bool(bits_available >= np.log2(total)),
        "ruling": (
            "⛔ The aggregate code is SUFFICIENT (K·log2(M) = 30 bits vs log2(480000) "
            "= 18.87) — '30 bits' may NOT be written as the cause. ⭐ The binding "
            "constraint is PER WELL: M = 64 is already exceeded by two hues alone "
            "(100) and by one hue x orientation (150)."
        ),
    }


def capacity_reading(total_before: float, total_after: float, spread: float) -> dict:
    """⭐⭐ PREREG §8's reading, applied mechanically.

    *"If the TOTAL-RETAINED line is flat and only the DISTRIBUTION moves, CAPACITY is
    the binding constraint, not the metric."*  ⛔ A first-class result, not a null —
    it sends the repair to ``M`` (or to a different representation), never to the
    metric.
    """
    delta = float(total_after) - float(total_before)
    flat = bool(abs(delta) <= float(spread))
    return {
        "total_retained_before": float(total_before),
        "total_retained_after": float(total_after),
        "delta": delta,
        "seed_spread_used": float(spread),
        "total_is_flat": flat,
        "reading": (
            "CAPACITY IS BINDING — the total is flat, so the metric only chose which "
            "factors spend the modes. ⛔ The repair goes to M (or to a different "
            "representation), NEVER to the metric."
            if flat else
            "NOT the flat-total case — the total moved beyond seed spread, so the "
            "metric changed how much is retained overall, not merely its distribution."
        ),
    }


__all__ = [
    "STORE_BYTE_BUDGET",
    "TRANSFORMS",
    "WellTransform",
    "assert_deployment_equivalence",
    "assert_grouping_is_unwhitened",
    "between_class_variance",
    "capacity_reading",
    "debias_between_class_variance",
    "per_well_capacity",
    "premise_verdict",
    "transform_byte_ledger",
    "unwhiten_store",
    "well_variance_spectrum",
]
