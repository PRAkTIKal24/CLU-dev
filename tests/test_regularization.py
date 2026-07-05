"""Tests for the Lyapunov regularization penalties (handover §7.6, F5 Prop-5).

The historical loss (mean_i log sigma_i of the step Jacobian) is provably
theta-independent: by exact conformal symplecticity it equals
0.5 * ln(1 - gamma) for ANY potential/state, hence ==0 at gamma=0 with ~zero
gradient. The replacements ("max", "sq", "pos") must carry usable theta
gradients; "legacy_degenerate" must not.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chlu.core.chlu_unit import CHLU
from chlu.core.regularization import compute_lyapunov_loss


def _grad_norm(penalty: str) -> float:
    """Global theta-gradient norm of the penalty on a random model/trajectory."""
    model = CHLU(dim=2, hidden=16, key=jax.random.PRNGKey(3))
    # A generic (non-symmetric) probe trajectory in phase space
    trajectory = jax.random.normal(jax.random.PRNGKey(4), (12, 4))

    def loss_fn(model):
        return compute_lyapunov_loss(
            lambda state: model.step(state, dt=0.05),
            trajectory,
            n_samples=5,
            penalty=penalty,
        )

    grads = eqx.filter_grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    return float(jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves if leaf.size > 0)))


def test_new_penalties_have_nonzero_theta_gradients():
    """ "max"/"sq"/"pos" must provide a usable training signal."""
    for penalty in ("max", "sq", "pos"):
        norm = _grad_norm(penalty)
        assert norm > 1e-6, f"penalty={penalty!r} has ~zero grad norm {norm:.3e}"


def test_legacy_penalty_is_degenerate():
    """The legacy mean-log-sigma loss has ~zero theta-gradient (F5 Prop-5)."""
    legacy_norm = _grad_norm("legacy_degenerate")
    max_norm = _grad_norm("max")

    # Absolute smallness (only float32 round-off + the 1e-8 log epsilon)...
    assert legacy_norm < 1e-3, f"legacy grad norm unexpectedly large: {legacy_norm:.3e}"
    # ...and vanishing relative to a usable penalty on the same model/state.
    assert legacy_norm < 1e-2 * max_norm, (
        f"legacy grad norm {legacy_norm:.3e} not << max-penalty norm {max_norm:.3e}"
    )


def test_legacy_penalty_value_is_half_log_one_minus_gamma():
    """Legacy loss == 0.5*ln(1-gamma) independent of theta (here gamma=0 -> 0)."""
    model = CHLU(dim=2, hidden=16, key=jax.random.PRNGKey(5))
    trajectory = jax.random.normal(jax.random.PRNGKey(6), (12, 4))

    loss = compute_lyapunov_loss(
        lambda state: model.step(state, dt=0.05),
        trajectory,
        n_samples=5,
        penalty="legacy_degenerate",
    )
    assert abs(float(loss)) < 1e-4  # 0.5*ln(1-0) = 0 up to float32 round-off


def test_none_penalty_returns_zero():
    loss = compute_lyapunov_loss(
        lambda state: state, jnp.zeros((4, 4)), n_samples=2, penalty="none"
    )
    assert float(loss) == 0.0


def test_unknown_penalty_raises():
    with pytest.raises(ValueError, match="Unknown lyapunov penalty"):
        compute_lyapunov_loss(
            lambda state: state, jnp.zeros((4, 4)), n_samples=2, penalty="bogus"
        )
