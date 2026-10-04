"""Metrics and tracking utilities."""

import jax
import jax.numpy as jnp


def compute_mse(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """
    Compute mean squared error.
    
    Args:
        pred: Predictions
        target: Ground truth
    
    Returns:
        MSE (scalar)
    """
    return jnp.mean((pred - target) ** 2)


def track_energy(model, trajectory: jnp.ndarray) -> jnp.ndarray:
    """
    Track energy along a CHLU trajectory.
    
    Args:
        model: CHLU model with H(q, p) method
        trajectory: Trajectory (T, 2*dim) [q, p]
    
    Returns:
        Energy at each timestep (T,)
    """
    dim = trajectory.shape[1] // 2
    
    def compute_energy_single(state):
        q, p = state[:dim], state[dim:]
        return model.H(q, p)
    
    energies = jax.vmap(compute_energy_single)(trajectory)
    
    return energies


def count_params(model) -> int:
    """
    Count number of parameters in an Equinox model.

    Args:
        model: Equinox model

    Returns:
        Total number of parameters
    """
    import equinox as eqx

    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)

    return sum(x.size for x in leaves)


# ---------------------------------------------------------------------------
# Selective-prediction metrics (numpy; used by exp_v1_calibration)
# ---------------------------------------------------------------------------


def risk_coverage_curve(confidence, correct):
    """
    Empirical risk-coverage curve of a selective predictor.

    Queries are answered in order of decreasing confidence; for every prefix
    size k the coverage is k/n and the (selective) risk is the error rate
    within the prefix. Ties are broken by index (stable sort; deterministic).

    Args:
        confidence: (n,) confidence scores (higher = answer sooner)
        correct: (n,) boolean correctness

    Returns:
        (coverage, risk): each (n,), coverage ascending from 1/n to 1.
    """
    import numpy as np

    conf = np.asarray(confidence, dtype=float).ravel()
    cor = np.asarray(correct, dtype=bool).ravel()
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    err = (~cor[order]).astype(float)
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(err) / k


def aurc(coverage, risk) -> float:
    """
    Area under the risk-coverage curve (Geifman & El-Yaniv): the mean of the
    prefix risks, (1/n) * sum_k risk(k/n). Lower is better; 0 = perfect
    ordering with zero errors.
    """
    import numpy as np

    return float(np.mean(np.asarray(risk, dtype=float)))


def coverage_at_risk(coverage, risk, max_risk: float) -> float:
    """
    Largest empirical coverage whose selective risk is <= max_risk
    (e.g. max_risk=0.05 for "95% precision"). 0.0 if no prefix qualifies.
    The empirical curve is not monotone, so this is a max over prefixes.
    """
    import numpy as np

    ok = np.asarray(risk, dtype=float) <= max_risk
    return float(np.max(np.asarray(coverage, dtype=float)[ok])) if ok.any() else 0.0


def expected_calibration_error(
    p_correct, correct, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """
    ECE of a probabilistic confidence signal: sum_b (n_b/n) |acc_b - conf_b|.

    Only meaningful for signals that claim to BE probabilities of correctness
    (calibrated heads, softmax responses); raw margins/energies have no ECE.

    Args:
        p_correct: (n,) predicted probability of being correct, in [0, 1]
        correct: (n,) boolean correctness
        n_bins: number of bins
        strategy: "quantile" (equal-mass, default) or "uniform" (equal-width)

    Returns:
        ECE in [0, 1].
    """
    import numpy as np

    p = np.asarray(p_correct, dtype=float).ravel()
    c = np.asarray(correct, dtype=bool).ravel().astype(float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown ECE binning strategy: {strategy}")
    if len(edges) < 2:  # all-identical predictions -> single bin
        return float(abs(c.mean() - p.mean()))
    edges = edges.copy()
    edges[-1] += 1e-12
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    n = len(p)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(c[m].mean() - p[m].mean())
    return float(ece)


def interpolate_risk_coverage(coverage, risk, grid):
    """
    Interpolate a risk-coverage curve onto a fixed coverage grid (for
    averaging curves across seeds). Left of the first point the risk is
    extended flat (the empirical curve starts at coverage 1/n).
    """
    import numpy as np

    coverage = np.asarray(coverage, dtype=float)
    risk = np.asarray(risk, dtype=float)
    return np.interp(np.asarray(grid, dtype=float), coverage, risk)
