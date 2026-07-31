"""Tests for B′'s rival memories and the uniform five-arm audit protocol.

What is under test is **faithfulness to the published equations** and **protocol
uniformity** — the two things the audit paper's claim rests on. Every test that
encodes a paper's equation names it, so a later edit that breaks faithfulness
fails loudly rather than quietly changing what the paper says we measured.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from chlu.eval.rivals import RIVALS, make_rival, rival_arms, table_budget  # noqa: E402
from chlu.eval.rivals.deltanet import DELTA_VARIANTS, DeltaMemory  # noqa: E402
from chlu.eval.rivals.deltanet import metric_native_verdict as delta_verdict  # noqa: E402
from chlu.eval.rivals.ttt import TTTMemory  # noqa: E402


def _model(name, d_in=5, m=1, seed=0, **kw):
    return make_rival(name, d_in, m, key=jax.random.PRNGKey(seed), **kw)


# --------------------------------------------------------------------------
# the state ledgers, against the published conventions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", RIVALS)
def test_every_rival_declares_the_published_state_convention(name):
    """`PREREG-Bprime.md` §2 / `rival-recon` F2:
    ``d_head^2`` (+ b buffer) / ``8 d_head^2`` (+ buffer) / ``n_head d_k d_v``."""
    mo = _model(name)
    d, b = mo.d_head, 16
    if name == "ttt_linear":
        assert mo.declared_state_floats() == d * d + b * d
    elif name == "ttt_mlp":
        assert mo.declared_state_floats() == 8 * d * d + b * d
    else:
        assert mo.declared_state_floats() == mo.n_head * d * d
    led = mo.ledger()
    assert led.state_floats == mo.declared_state_floats()
    assert led.param_floats > 0
    led.check()


def test_gdn2_state_convention_cites_its_own_equation_90():
    """⭐ The -2 revision **preserves** GDN's ``n_head d_k d_v`` accounting; the
    task requires that we verified it rather than inheriting it."""
    led = _model("gdn2").ledger()
    assert "Eq. 90" in led.state_convention
    assert "262,144" in led.state_convention


@pytest.mark.parametrize("name", RIVALS)
def test_learned_initial_state_rule_puts_the_init_in_PARAMETERS(name):
    """``W_0`` / ``S_0`` is parameters, never state (`PREREG-Bprime.md` §4.1)."""
    led = _model(name).ledger()
    init_keys = [k for k in led.param_breakdown if "init" in k]
    assert init_keys, f"{name}: the init must appear in the PARAMETER breakdown"
    assert not any("init" in k for k in led.state_breakdown)
    assert "PARAMETERS" in led.note


# --------------------------------------------------------------------------
# faithfulness: the update equations
# --------------------------------------------------------------------------
def test_deltanet_matches_eq5_step_by_step():
    """Eq. 5: ``S_t = (I - beta k k^T) S_{t-1} + beta k v^T``, computed the long
    way (with the explicit ``d_k x d_k`` matrix) and compared to the shipped
    rank-1 form."""
    mo = DeltaMemory(5, 6, 1, key=jax.random.PRNGKey(0), variant="deltanet")
    xs = np.asarray(np.random.default_rng(0).normal(size=(4, 5)), dtype=np.float32)
    S = np.asarray(mo.init_state())
    for x in xs:
        k, v = (np.asarray(a) for a in mo._kv(jnp.asarray(x)))
        beta = float(jax.nn.sigmoid(np.dot(np.asarray(mo.w_beta), x)))
        S = (np.eye(S.shape[0]) - beta * np.outer(k, k)) @ S + beta * np.outer(k, v)
    assert np.allclose(np.asarray(mo.write(xs)), S, atol=1e-5)


def test_gdn2_matches_eq10_step_by_step():
    """Eq. 10 (boxed): ``S_t = (I - k (b (*) k)^T) D_t S_{t-1} + k (w (*) v)^T``
    with Eq. 11's gates and Eq. 12's channel-wise decay."""
    mo = DeltaMemory(5, 6, 1, key=jax.random.PRNGKey(1), variant="gdn2")
    xs = np.asarray(np.random.default_rng(1).normal(size=(4, 5)), dtype=np.float32)
    S = np.asarray(mo.init_state())
    for x in xs:
        k, v = (np.asarray(a) for a in mo._kv(jnp.asarray(x)))
        b = mo.erase_scale * np.asarray(jax.nn.sigmoid(np.asarray(mo.W_b) @ x))
        w = np.asarray(jax.nn.sigmoid(np.asarray(mo.W_w) @ x))
        g = -np.exp(np.asarray(mo.a_log)) * np.asarray(
            jax.nn.softplus(np.asarray(mo.W_f) @ x + np.asarray(mo.delta)))
        D = np.diag(np.exp(g))
        S = (np.eye(S.shape[0]) - np.outer(k, b * k)) @ (D @ S) + np.outer(k, w * v)
    assert np.allclose(np.asarray(mo.write(xs)), S, atol=1e-5)


def test_gdn2_reduces_to_gated_deltanet_when_the_gates_collapse():
    """⭐ The paper's own reduction (§3.1): *"recovers KDA exactly when
    ``b_t = beta 1_{d_k}`` and ``w_t = beta 1_{d_v}``; recovers Gated DeltaNet by
    further setting ``alpha_t = alpha 1_{d_k}``."*"""
    import equinox as eqx

    d, di = 6, 5
    mo = DeltaMemory(di, d, 1, key=jax.random.PRNGKey(2), variant="gdn2",
                     erase_scale=1.0)
    # collapse both gates to the SAME constant scalar beta = sigmoid(0) = 0.5, and
    # the decay to alpha = exp(-exp(0)*softplus(0)) (a constant), i.e. no x-dependence
    mo = eqx.tree_at(lambda t: [t.W_b, t.W_w, t.W_f],
                     mo, replace=[jnp.zeros((d, di)), jnp.zeros((d, di)),
                                  jnp.zeros((d, di))])
    xs = np.asarray(np.random.default_rng(2).normal(size=(5, di)), dtype=np.float32)
    S = np.asarray(mo.init_state())
    alpha = float(np.exp(-np.exp(0.0) * np.log(2.0)))   # softplus(0) = ln 2
    for x in xs:
        k, v = (np.asarray(a) for a in mo._kv(jnp.asarray(x)))
        Sb = alpha * S
        # the collapsed Eq. 10 IS Eq. 6 with beta = 0.5 acting on both sides
        S = (np.eye(d) - 0.5 * np.outer(k, k)) @ Sb + 0.5 * np.outer(k, v)
    assert np.allclose(np.asarray(mo.write(xs)), S, atol=1e-5)


def test_ttt_linear_one_token_is_exactly_one_gradient_step():
    """Eqs. 2 + 4: with a single token the state must equal
    ``W_0 - eta * grad || f(theta_K x; W) - theta_V x ||^2``."""
    mo = TTTMemory(5, 6, 1, key=jax.random.PRNGKey(3), kind="linear", mini_batch=1)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(1, 5)), dtype=jnp.float32)
    W = mo.write(x)
    g = jax.grad(mo._loss)(mo.init_state(), x[0])
    eta = float(jnp.exp(mo.log_eta) * jax.nn.sigmoid(x[0] @ mo.theta_lr))
    for got, w0, gg in zip(W, mo.init_state(), g, strict=True):
        assert np.allclose(np.asarray(got), np.asarray(w0) - eta * np.asarray(gg),
                           atol=1e-5)


def test_ttt_mini_batch_takes_every_gradient_at_the_chunk_start():
    """§2.4: ``G_t = grad l(W_{t'}; x_t)`` with ``t' = t - mod(t, b)``. With
    ``b >= T`` **every** gradient is taken at ``W_0`` — which is also why the
    arm's ``b`` is in the tuning grid at this stream length."""
    xs = jnp.asarray(np.random.default_rng(4).normal(size=(4, 5)), dtype=jnp.float32)
    big = TTTMemory(5, 6, 1, key=jax.random.PRNGKey(4), kind="linear", mini_batch=16)
    W = big.write(xs)
    W0 = big.init_state()
    eta = jnp.exp(big.log_eta) * jax.nn.sigmoid(xs @ big.theta_lr)
    gs = [jax.grad(big._loss)(W0, x) for x in xs]
    for i, (got, w0) in enumerate(zip(W, W0, strict=True)):
        want = np.asarray(w0) - sum(float(eta[t]) * np.asarray(gs[t][i])
                                    for t in range(4))
        assert np.allclose(np.asarray(got), want, atol=1e-5)
    # ... and b = 1 (online GD) is a DIFFERENT function
    one = TTTMemory(5, 6, 1, key=jax.random.PRNGKey(4), kind="linear", mini_batch=1)
    assert not np.allclose(np.asarray(one.write(xs)[0]), np.asarray(W[0]), atol=1e-4)


def test_ttt_mlp_state_is_eight_d_head_squared():
    mo = TTTMemory(5, 7, 1, key=jax.random.PRNGKey(5), kind="mlp")
    core = sum(int(np.asarray(x).size) for x in mo.init_state()
               if np.asarray(x).ndim == 2)
    assert core == 8 * 7 * 7


# --------------------------------------------------------------------------
# the write mask: padding must not be able to write
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", RIVALS)
def test_masked_tokens_are_a_no_op(name):
    mo = _model(name, seed=6)
    xs = np.asarray(np.random.default_rng(6).normal(size=(6, 5)), dtype=np.float32)
    real = mo.write(xs[:4], np.ones(4, dtype=np.float32))
    padded = mo.write(xs, np.asarray([1, 1, 1, 1, 0, 0], dtype=np.float32))
    for a, b in zip(jax.tree_util.tree_leaves(real),
                    jax.tree_util.tree_leaves(padded), strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


@pytest.mark.parametrize("name", RIVALS)
def test_blank_state_read_equals_the_init_read(name):
    """The blank-store control is *the same system with nothing written* — not a
    re-initialised one."""
    mo = _model(name, seed=7)
    xq = np.asarray(np.random.default_rng(7).normal(size=(3, 5)), dtype=np.float32)
    arms = rival_arms(mo, np.zeros((2, 5), dtype=np.float32),
                      np.zeros(2, dtype=np.float32), xq)
    assert np.allclose(arms["blank"], np.asarray(mo.read(mo.init_state(), xq)),
                       atol=1e-6)
    # with nothing written, `full` IS the blank read, and the empty table must
    # answer with zeros rather than raising (it is a control, not a failure)
    assert np.allclose(arms["full"], arms["blank"], atol=1e-6)
    assert np.all(np.isfinite(arms["launder"]))


# --------------------------------------------------------------------------
# the five-arm protocol
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", RIVALS)
def test_all_five_arms_are_produced_for_every_family(name):
    """Protocol uniformity IS the deliverable (§6.1): every rival, every arm."""
    mo = _model(name, seed=8)
    rng = np.random.default_rng(8)
    xs = np.asarray(rng.normal(size=(7, 5)), dtype=np.float32)
    xq = np.asarray(rng.normal(size=(9, 5)), dtype=np.float32)
    arms = rival_arms(mo, xs, np.ones(7, dtype=np.float32), xq, rng=rng,
                      n_rows=table_budget(mo)["n_rows_affordable"])
    assert set(arms) == {"full", "blank", "launder", "same_keys_null",
                         "knn2_mean_+0B", "knn2_idw_+0B", "table_mean_+0B"}
    for k, v in arms.items():
        assert np.asarray(v).shape == (9, 1), k
        assert np.all(np.isfinite(np.asarray(v))), k


@pytest.mark.parametrize("name", RIVALS)
def test_the_matched_table_is_byte_equal_to_the_state(name):
    """**P5**: ``n_rows = floor(state_floats/(d_k+d_v))``, so the table is inside
    the memory's own state budget by construction — the property the CLU
    provably cannot have (T1's ``ratio >= 2.20x`` floor)."""
    mo = _model(name)
    tb = table_budget(mo)
    used = tb["n_rows_affordable"] * tb["row_floats"]
    assert used <= mo.declared_state_floats()
    assert used + tb["row_floats"] > mo.declared_state_floats()


def test_the_table_launder_only_ever_returns_a_stored_value():
    """The arg-min launder is a *table*: its output must be the decode of one
    stored row, never an interpolation. (This is why an aggregate query has a
    positive error floor against it.)"""
    mo = _model("deltanet", seed=9)
    rng = np.random.default_rng(9)
    xs = np.asarray(rng.normal(size=(5, 5)), dtype=np.float32)
    xq = np.asarray(rng.normal(size=(11, 5)), dtype=np.float32)
    arms = rival_arms(mo, xs, np.ones(5, dtype=np.float32), xq, rng=rng)
    _, vals = mo.kv_table(jnp.asarray(xs))
    stored = np.asarray(mo.decode_values(vals)).ravel()
    for got in np.asarray(arms["launder"]).ravel():
        assert np.min(np.abs(stored - got)) < 1e-5


# --------------------------------------------------------------------------
# the metric-native verdicts, argued at equation level (D3.6)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("v", DELTA_VARIANTS)
def test_every_delta_variant_declares_a_metric_native_verdict(v):
    out = delta_verdict(v)
    assert out["verdict"] == "metric-native"
    assert "Eq." in out["argument"]
    assert "table" in out["measured_against"]


def test_ttt_mlp_is_only_WEAKLY_metric_native():
    from chlu.eval.rivals.ttt import metric_native_verdict

    assert metric_native_verdict("linear")["verdict"] == "metric-native"
    assert metric_native_verdict("mlp")["verdict"] == "weakly metric-native"


# --------------------------------------------------------------------------
# the experiment's own guards
# --------------------------------------------------------------------------
def test_delete_rows_are_skipped_and_declared():
    """No rival family has a deletion verb; the asymmetry is in the RIVALS'
    favour and must be stated, not hidden."""
    from chlu.experiments.exp_bprime_rivals import stream_tokens
    from chlu.experiments.memory_gym import gym_config, make_gym_stream

    g = gym_config("aggregate", "base", seed=0)
    c = g.build_clu()
    st = make_gym_stream(g, c)
    xs, mask, note = stream_tokens(st, c)
    assert note["n_delete_rows_skipped"] >= 1
    assert xs.shape == (note["n_tokens"], c.dim)
    assert mask.sum() == note["n_tokens"]
    assert "RIVALS' FAVOUR" in note["note"]


def test_the_frontier_family_is_labelled_everywhere():
    """⛔ ``overload`` is a byte-frontier column and never a dividend family."""
    from chlu.experiments.exp_bprime_rivals import (
        DEFAULT_PLAN,
        FRONTIER_LABEL,
        BANKED_CLU,
    )

    assert ("recency", "base") not in DEFAULT_PLAN
    assert ("manifold", "base") not in DEFAULT_PLAN
    assert "S_excl = 0.6500" in FRONTIER_LABEL
    assert "BYTE-FRONTIER" in FRONTIER_LABEL
    assert BANKED_CLU["overload/load1x_shipped"]["S_excl"] == 0.65


def test_banked_clu_column_is_quoted_not_recomputed():
    """`PREREG-Bprime.md` §7: the CLU column is banked. These are the numbers the
    C2W1 artefact and `bprime-fb4-gate` §A2 both carry."""
    from chlu.experiments.exp_bprime_rivals import BANKED_CLU

    agg = BANKED_CLU["aggregate/base"]
    assert agg["full"] == [-0.682608, -0.384693, -0.511032]
    assert agg["launder"] == [-0.496261, -0.413103, -0.432255]
    assert agg["byte_ledger"] == {"full_bytes": 5456, "launder_bytes": 100,
                                  "ratio": 54.56}
    ovl = BANKED_CLU["overload/load1x_shipped"]
    assert ovl["accuracy_vs_bytes_curve"]["decode"] == [0.972, 0.097]
    assert ovl["accuracy_vs_bytes_curve"]["ratio"] == [478.0, 2.28]
