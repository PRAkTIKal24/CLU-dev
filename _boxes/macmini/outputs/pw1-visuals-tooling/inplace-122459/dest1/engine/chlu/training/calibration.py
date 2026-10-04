"""Training-time calibration head + learned threshold tau for CLU memories.

The V1 pivot (Head decision 2026-07-07): the headline mechanism is *calibrated
energy-gated compute allocation on conservative memories*. The write phase of
an energy-based associative memory ends with a **self-test** that fits a
per-model calibration head mapping the retrieval diagnostics of a settled
state — the residual energy ``R = H(settled) - floor`` and (optionally) the
readout margin — to a probability of the answer being wrong. Raw R is NOT
comparable across models (pooled raw AUROC 0.330 in the v1-l0-gate run): this
head is what turns the energy signal into a deployable, cross-model gate.
Per brainstorm Thread 3, calibration is a *training-time learned object*
fitted on a held-out calibration split (probe relaxations generated at write
time), never on evaluation queries.

Simple form (per the v1-pivot task spec): affine/temperature (Platt-style)
calibration

    p_wrong = sigmoid( w . (phi(x) - mu) / sigma + b ),

with per-model feature standardization (mu, sigma) and logistic weights
(w, b) fitted by ridge-regularized logistic regression. The learned threshold
tau is the decision boundary; for the 1-feature energy head the residual
threshold ``tau_R`` (in energy units) is exposed directly.

Also implements a Learn-then-Test wrapper (LTT; Angelopoulos et al. 2021 —
the machinery behind CALM, Schuster et al. 2022): distribution-free selection
of the gate threshold with a selective-risk guarantee, valid under
exchangeability of the calibration split and deployment queries. Our probes
are jittered cues while deployment cues are exact, so the guarantee is only
as good as that shift — it must be *measured* on held-out queries, not
assumed (the exp_v1_calibration experiment reports empirical validity).

Nomenclature (F5 Def-2): no unqualified "mass" appears here; the head
consumes energies only.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Feature sets the head understands: residual energy R, readout margin, or both.
FEATURE_SETS = {
    "r": ("R",),
    "margin": ("margin",),
    "r_margin": ("R", "margin"),
}


def _feature_matrix(features: str, R=None, margin=None) -> np.ndarray:
    """Stack the requested per-query diagnostics into an (n, F) matrix."""
    if features not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{features}' (choose from {list(FEATURE_SETS)})"
        )
    named = {"R": R, "margin": margin}
    cols = []
    for name in FEATURE_SETS[features]:
        if named[name] is None:
            raise ValueError(f"Feature set '{features}' requires '{name}'")
        cols.append(np.asarray(named[name], dtype=float).ravel())
    return np.stack(cols, axis=1)


@dataclass
class CalibrationHead:
    """Per-model calibrated gate p_wrong(features).

    Attributes:
        features: feature set key ("r" | "margin" | "r_margin")
        mu, sigma: per-feature standardization (fitted on the calibration split)
        w, b: logistic weights/intercept in standardized feature space
        degenerate: True if the calibration split had a single class; the head
            then emits the constant class prior ``p_const`` (no ranking within
            this model — an honest "my self-test saw no failures/successes").
        p_const: constant p_wrong used when degenerate.
        n_fit: number of calibration points used for the fit.
    """

    features: str = "r_margin"
    mu: np.ndarray = field(default_factory=lambda: np.zeros(2))
    sigma: np.ndarray = field(default_factory=lambda: np.ones(2))
    w: np.ndarray = field(default_factory=lambda: np.zeros(2))
    b: float = 0.0
    degenerate: bool = False
    p_const: float = 0.5
    n_fit: int = 0

    def p_wrong(self, R=None, margin=None) -> np.ndarray:
        """Calibrated probability that the settled answer is wrong.

        Accepts arrays of any (matching) shape; returns that shape.
        """
        ref = R if R is not None else margin
        shape = np.asarray(ref, dtype=float).shape
        if self.degenerate:
            return np.full(shape, self.p_const)
        X = _feature_matrix(self.features, R=R, margin=margin)
        z = (X - self.mu) / self.sigma @ self.w + self.b
        return (1.0 / (1.0 + np.exp(-z))).reshape(shape)

    def tau_r(self, p_exit: float = 0.5) -> float:
        """Learned residual-energy threshold (energy units), 1-feature head only.

        Solves p_wrong(R) = p_exit for R. NaN if the head is degenerate or
        does not use R as its single feature.
        """
        if self.features != "r" or self.degenerate or self.w[0] == 0.0:
            return float("nan")
        z = math.log(p_exit / (1.0 - p_exit))
        return float(self.mu[0] + self.sigma[0] * (z - self.b) / self.w[0])

    def to_dict(self) -> dict:
        """Plain-python serialization (checkpoint-metadata friendly)."""
        return {
            "features": self.features,
            "mu": np.asarray(self.mu, dtype=float).tolist(),
            "sigma": np.asarray(self.sigma, dtype=float).tolist(),
            "w": np.asarray(self.w, dtype=float).tolist(),
            "b": float(self.b),
            "degenerate": bool(self.degenerate),
            "p_const": float(self.p_const),
            "n_fit": int(self.n_fit),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationHead":
        return cls(
            features=d["features"],
            mu=np.asarray(d["mu"], dtype=float),
            sigma=np.asarray(d["sigma"], dtype=float),
            w=np.asarray(d["w"], dtype=float),
            b=float(d["b"]),
            degenerate=bool(d["degenerate"]),
            p_const=float(d["p_const"]),
            n_fit=int(d.get("n_fit", 0)),
        )


def fit_calibration_head(
    R=None,
    margin=None,
    wrong=None,
    features: str = "r_margin",
    l2: float = 1.0,
    max_iter: int = 1000,
) -> CalibrationHead:
    """Fit a per-model calibration head on a calibration split.

    Ridge-regularized logistic regression (sklearn lbfgs; deterministic) on
    standardized features. If the split is single-class the head degrades to a
    constant prior (see CalibrationHead.degenerate) instead of crashing —
    include probes guaranteed to fail (e.g. impostor cues) to avoid this.

    Args:
        R: residual energies of settled calibration probes
        margin: readout margins of the same probes
        wrong: boolean labels (True = probe answered incorrectly)
        features: which diagnostics the head uses ("r" | "margin" | "r_margin")
        l2: ridge strength (sklearn C = 1 / l2)
        max_iter: lbfgs iteration cap

    Returns:
        CalibrationHead
    """
    y = np.asarray(wrong, dtype=bool).ravel()
    X = _feature_matrix(features, R=R, margin=margin)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features n={X.shape[0]} != labels n={y.shape[0]}")
    mu = X.mean(axis=0)
    sigma = np.maximum(X.std(axis=0), 1e-8)
    if y.min() == y.max():
        p_const = float(np.clip(y.mean(), 1e-3, 1.0 - 1e-3))
        return CalibrationHead(
            features=features,
            mu=mu,
            sigma=sigma,
            w=np.zeros(X.shape[1]),
            b=0.0,
            degenerate=True,
            p_const=p_const,
            n_fit=int(len(y)),
        )
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0 / max(l2, 1e-12), solver="lbfgs", max_iter=max_iter)
    clf.fit((X - mu) / sigma, y.astype(int))
    return CalibrationHead(
        features=features,
        mu=mu,
        sigma=sigma,
        w=clf.coef_[0].astype(float),
        b=float(clf.intercept_[0]),
        degenerate=False,
        p_const=0.5,
        n_fit=int(len(y)),
    )


def ltt_select_threshold(
    p_wrong,
    wrong,
    target_risk: float,
    delta: float = 0.1,
    n_grid: int = 50,
) -> tuple[Optional[float], dict]:
    """Learn-then-Test threshold selection for the answer/abstain gate.

    Deployment rule: ANSWER iff ``p_wrong <= t``. Selects the largest t whose
    selective risk (error rate among answered) is certified <= target_risk
    with confidence 1 - delta, by fixed-sequence testing with exact binomial
    tail p-values: for candidate t, p = P(Bin(n_t, target_risk) <= k_t) where
    n_t = #answered and k_t = #errors among answered on the calibration split.
    Candidates are quantiles of p_wrong (label-independent, so the fixed
    sequence is valid), walked in ascending order (increasing coverage),
    stopping at the first failure to reject.

    The first candidate is placed at the smallest quantile whose answered
    count could possibly certify (n_t >= log(delta)/log(1-target_risk), the
    zero-error requirement) — a power (not validity) optimization that is
    also label-independent.

    Guarantee caveat: valid under exchangeability of calibration probes and
    deployment queries. Report empirical deployment risk alongside.

    Returns:
        (threshold or None, info dict with the walk diagnostics)
    """
    from scipy.stats import binom

    p = np.asarray(p_wrong, dtype=float).ravel()
    y = np.asarray(wrong, dtype=bool).ravel()
    n = len(p)
    if n == 0:
        return None, {"reason": "empty calibration split"}
    n_min = int(math.ceil(math.log(delta) / math.log(1.0 - target_risk)))
    q_min = min(1.0, max(n_min / n, 1.0 / n))
    grid = np.unique(np.quantile(p, np.linspace(q_min, 1.0, n_grid)))
    best = None
    walk = []
    for t in grid:
        sel = p <= t
        n_t = int(sel.sum())
        k_t = int((y & sel).sum())
        pval = float(binom.cdf(k_t, n_t, target_risk)) if n_t > 0 else 1.0
        walk.append((float(t), n_t, k_t, pval))
        if pval <= delta:
            best = float(t)
        else:
            break
    info = {
        "n_cal": n,
        "n_min_zero_error": n_min,
        "grid_size": int(len(grid)),
        "walk": walk,
        "certified": best is not None,
    }
    return best, info
