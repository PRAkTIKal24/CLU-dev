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


# --------------------------------------------------------------------------
# ⭐ the full-F3 tuning pass (C2W4 rider `bprime-rivals-f3`)
# --------------------------------------------------------------------------
def test_the_F3_grid_is_the_standing_rule_verbatim():
    """`rival-recon` F3 / standing rule 5 (N78's operational form):
    ``lr in {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} x wd in {0, 0.1}``.
    ⚠ Its UPPER edge equals F3-lite's, so widening only adds low-lr points —
    the fact the rider's PREREG reasons from."""
    from chlu.eval.rivals.fit import LR_GRID, LR_GRID_F3, WD_GRID_F3

    assert LR_GRID_F3 == (1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2)
    assert WD_GRID_F3 == (0.0, 0.1)
    assert set(LR_GRID) <= set(LR_GRID_F3)          # the C2W4 sub-grid survives
    assert max(LR_GRID_F3) == max(LR_GRID) == 1e-2  # nothing added above


def _fit_examples(n=2, t=6, nq=5, d_in=5, m=1, seed=0):
    from chlu.eval.rivals import FitExample

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        xs = rng.normal(size=(t, d_in)).astype(np.float32)
        xq = rng.normal(size=(nq, d_in)).astype(np.float32)
        out.append(FitExample(xs=xs, mask=np.ones((t,), np.float32), xq=xq,
                              target=rng.normal(size=(nq, m)).astype(np.float32)))
    return out


def test_widening_the_grid_does_not_perturb_the_incumbent_points():
    """⭐ The property that makes the rider's F3-LITE CONTROL column valid: the
    ``wd = 0`` sub-column of the full F3 grid must be **identical** to the same
    points fitted on their own. It holds because the init is drawn per
    ``(arm, b)`` (not per grid position) and ``wd = 0`` keeps ``optax.adam``."""
    from chlu.eval.rivals.fit import fit_grid

    ex = _fit_examples()
    kw = dict(key=jax.random.PRNGKey(0), b_grid=(16,), steps=3)
    lite, _ = fit_grid("gdn2", 5, 1, ex, lrs=(1e-3, 1e-2), wds=(0.0,), **kw)
    full, _ = fit_grid("gdn2", 5, 1, ex, lrs=(1e-4, 1e-3, 1e-2),
                       wds=(0.0, 0.1), **kw)
    sub = [r for r in full if r["wd"] == 0.0 and r["lr"] in (1e-3, 1e-2)]
    assert [r["lr"] for r in sub] == [r["lr"] for r in lite]
    for a, b in zip(sub, lite, strict=True):
        assert a["final"] == b["final"]      # bit-identical, not approx


def test_the_grid_shares_one_init_per_mini_batch():
    """A duplicated grid point must produce a duplicated result — i.e. the init
    does NOT depend on the grid's length or order (⚠ C2W4's sequential split did,
    which is why the rider prices the redraw with a control column)."""
    from chlu.eval.rivals.fit import fit_grid

    ex = _fit_examples()
    grid, _ = fit_grid("deltanet", 5, 1, ex, key=jax.random.PRNGKey(1),
                       lrs=(1e-3, 1e-3), wds=(0.0,), b_grid=(16,), steps=3)
    assert grid[0]["final"] == grid[1]["final"]


def test_weight_decay_is_decoupled_adamw_and_actually_bites():
    """F3's second axis must be a real axis: ``wd = 0.1`` has to change the fit."""
    from chlu.eval.rivals.fit import fit_grid

    ex = _fit_examples()
    grid, _ = fit_grid("gdn2", 5, 1, ex, key=jax.random.PRNGKey(2), lrs=(1e-2,),
                       wds=(0.0, 0.1), b_grid=(16,), steps=8)
    assert {r["wd"] for r in grid} == {0.0, 0.1}
    assert grid[0]["final"] != grid[1]["final"]


def test_held_out_selection_can_pick_a_point_the_fit_loss_never_would():
    """The declared SECONDARY selection: on a held-out auxiliary stream. ⛔ Neither
    reader ever sees the eval split."""
    from chlu.eval.rivals.fit import select_best

    grid = [{"lr": 1e-3, "wd": 0.0, "final": 0.5, "val_final": 0.1},
            {"lr": 1e-2, "wd": 0.1, "final": 0.2, "val_final": 0.9}]
    models = ["A", "B"]
    assert select_best(grid, models, on="fit")[0] == "B"
    assert select_best(grid, models, on="val")[0] == "A"
    # and a sub-grid restriction (the F3-lite control) selects inside it only
    mo, rec = select_best(grid, models, lrs=(1e-3,), wds=(0.0,), label="ctrl")
    assert mo == "A" and rec["selection"]["n_points"] == 1
    with pytest.raises(ValueError):
        select_best(grid, models, lrs=(7.0,))


def test_the_val_stream_does_not_move_the_training_stream():
    """⛔ One variable moves. Adding held-out streams must leave the outer loop's
    TRAINING examples byte-identical to C2W4's."""
    from chlu.experiments.exp_bprime_rivals import aux_fit_examples
    from chlu.experiments.memory_gym import gym_config

    g = gym_config("aggregate", "base", seed=0)
    c = g.build_clu()
    a, va, na = aux_fit_examples("aggregate", "base", 0, c, g, n_val=0)
    b, vb, nb = aux_fit_examples("aggregate", "base", 0, c, g, n_val=1)
    assert va is None and vb is not None
    assert na["fit_stream_seeds"] == nb["fit_stream_seeds"] == [101, 102]
    assert nb["val_stream_seeds"] == [103]          # a DIFFERENT stream
    assert (na["n_tokens"], na["n_queries"]) == (nb["n_tokens"], nb["n_queries"])
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.xs, y.xs) and np.array_equal(x.target, y.target)


def test_the_before_after_thresholds_are_the_pre_registered_ones():
    """PREREG-f3 §2: T1 rescue flip · T2 R5 sign flip · T3 raw margin positive ·
    T4 the R5 count changes · T5 the P5-vs-raw gap collapses. Adjudication is
    mechanical, so it is tested mechanically."""
    from chlu.experiments.exp_bprime_rivals import C2W4_INCUMBENT, before_after

    def _tab(rows):
        return {"f3": {"aggregate": {"rivals": rows}}}

    same = {n: {"d_head": v["d_head"],
                "RESCUED_above_own_blank_2se": v["rescued"],
                "full": v["full"], "full_se": v["full_se"],
                "dividend_vs_own_table": v["dividend_vs_own_table"],
                "zero_byte_margin": v["zero_byte_margin"],
                "zero_byte_margin_se": v["zero_byte_margin_se"],
                "raw_table_margin": v["raw_table_margin"],
                "raw_table_margin_se": v["raw_table_margin_se"],
                "p5_vs_raw_gap": v["p5_vs_raw_gap"], "p5_vs_raw_gap_se": 0.02,
                "lift_over_own_blank": v["lift_over_own_blank"],
                "lift_se": v["lift_se"]}
           for n, v in C2W4_INCUMBENT.items()}
    out = before_after(_tab(same))
    assert out["OUTCOME"].startswith("UNCHANGED") and not out["thresholds_fired"]
    assert out["R5_count_le_zero"] == {"C2W4": 3, "f3": 3}

    # T3: an arm that now BEATS a raw table of the same bytes by > 2 SE
    beats = {n: dict(v) for n, v in same.items()}
    beats["gdn2"]["raw_table_margin"] = 0.30
    fired = before_after(_tab(beats))
    assert fired["rows"]["gdn2"]["verdict"] == "CHANGED"
    assert "T3_raw_margin_positive" in fired["rows"]["gdn2"]["thresholds_fired"]
    assert "gdn2:T3_raw_margin_positive" in fired["thresholds_fired"]

    # T1 + T4: ttt_mlp rescued and its +0 B margin turning positive
    flip = {n: dict(v) for n, v in same.items()}
    flip["ttt_mlp"]["RESCUED_above_own_blank_2se"] = True
    flip["ttt_mlp"]["zero_byte_margin"] = 0.40
    out2 = before_after(_tab(flip))
    assert "ttt_mlp:T1_rescue_flip" in out2["thresholds_fired"]
    assert "ttt_mlp:T2_R5_sign_flip" in out2["thresholds_fired"]
    assert out2["R5_count_le_zero"]["f3"] == 2
    assert any("T4_R5_count_changed" in f for f in out2["thresholds_fired"])
