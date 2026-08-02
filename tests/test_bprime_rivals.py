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
    elif name == "mamba2":
        # the reference implementation's own ssm_state (nheads, headdim, d_state)
        assert mo.declared_state_floats() == mo.n_head * d * mo.d_state
        assert mo.d_state == d          # the declared iso-state sizing choice
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


# --------------------------------------------------------------------------
# ⭐ Mamba-2 / SSD (C2W5, Head ruling 5) — the selective state-space arm
# --------------------------------------------------------------------------
def _mamba2(d_head=6, d_in=5, m=1, seed=0, **kw):
    from chlu.eval.rivals.mamba2 import Mamba2Memory

    return Mamba2Memory(d_in, d_head, m, key=jax.random.PRNGKey(seed), **kw)


def _stream(t=19, d_in=5, seed=3):
    rng = np.random.default_rng(seed)
    return (np.asarray(rng.normal(size=(t, d_in)), dtype=np.float32),
            np.ones((t,), dtype=np.float32))


def test_mamba2_chunked_ssd_equals_the_sequential_recurrence():
    """⭐ **The SSD property, asserted rather than cited.** The chunked block
    algorithm (Dao & Gu, ICML 2024, §6) is an exact re-association of
    ``h_t = a_t h_{t-1} + B_t (Delta_t v_t)^T`` — so it must agree with the naive
    recurrence to fp32 rounding, not merely approximately."""
    mo = _mamba2()
    xs, mask = _stream()
    a = np.asarray(mo.write(xs, mask))
    b = np.asarray(mo.write_sequential(xs, mask))
    assert np.allclose(a, b, atol=1e-6, rtol=1e-4)
    assert np.abs(a).max() > 0            # not a trivial all-zero agreement


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 16, 256])
def test_mamba2_chunk_length_is_provably_inert(chunk):
    """⛔ **Why the SSD chunk is NOT a tuning axis while TTT's ``b`` is.** TTT's
    mini-batch changes the function (every gradient is taken at the chunk start)
    *and* its state (the in-flight buffer); SSD's chunk changes neither."""
    xs, mask = _stream()
    ref = np.asarray(_mamba2(chunk=1).write_sequential(xs, mask))
    got = np.asarray(_mamba2(chunk=chunk).write(xs, mask))
    assert np.allclose(got, ref, atol=1e-6, rtol=1e-4)


def test_mamba2_dual_quadratic_read_equals_the_recurrent_read():
    """⭐ **State-space duality, measured.** The quadratic form
    ``o_q = sum_j gamma_j (C_q . B_j) Delta_j v_j`` (a 1-semiseparable masked
    attention) must equal the recurrent read of the same stream."""
    mo = _mamba2(seed=4)
    xs, mask = _stream(seed=4)
    xq = np.asarray(np.random.default_rng(5).normal(size=(7, 5)), dtype=np.float32)
    rec = np.asarray(mo.read(mo.write(xs, mask), xq))
    dual = np.asarray(mo.read_dual(xs, xq, mask))
    assert np.allclose(rec, dual, atol=1e-6, rtol=1e-4)


def test_mamba2_with_no_decay_is_plain_linear_attention():
    """``A -> 0`` (no decay, ``a_t = 1``) must collapse the SSD state to the
    unnormalised linear-attention sum ``sum_t B_t (Delta_t v_t)^T``."""
    import equinox as eqx

    mo = _mamba2(seed=6)
    mo = eqx.tree_at(lambda t: t.A_log, mo, jnp.asarray(-30.0))   # exp(A_log) ~ 0
    xs, mask = _stream(seed=6)
    B, v, dt, log_a = mo._stream(jnp.asarray(xs), jnp.asarray(mask))
    want = np.einsum("tn,tp->np", np.asarray(B),
                     np.asarray(dt)[:, None] * np.asarray(v))
    assert np.allclose(np.asarray(mo.write(xs, mask)), want, atol=1e-6, rtol=1e-4)
    assert np.allclose(np.asarray(log_a), 0.0, atol=1e-10)


def test_mamba2_selection_is_input_dependent():
    """Mamba-1 §3.2's selection mechanism, carried over: ``Delta_t``, ``B_t`` and
    ``C_t`` must all depend on the token. A time-invariant SSM would be a
    different (and much weaker) model, so this is worth a test."""
    import equinox as eqx

    mo = _mamba2(seed=7)
    mo = eqx.tree_at(lambda t: t.w_dt, mo, jnp.ones((5,)))
    x1 = jnp.asarray(np.zeros((5,), dtype=np.float32))
    x2 = jnp.asarray(np.ones((5,), dtype=np.float32))
    assert float(mo._dt(x1)) != float(mo._dt(x2))
    b1, _ = mo._kv(x1)
    b2, _ = mo._kv(x2)
    assert not np.allclose(np.asarray(b1), np.asarray(b2))


def test_mamba2_reference_init_ranges():
    """The reference implementation's init (`rival-recon` §1.4): ``A ~ U(1, 16)``
    and ``Delta ~ exp(U(log 1e-3, log 1e-1))`` inverse-softplused into the bias."""
    from chlu.eval.rivals.mamba2 import A_INIT_RANGE, DT_MAX, DT_MIN

    for seed in range(8):
        mo = _mamba2(seed=seed)
        a = float(np.exp(np.asarray(mo.A_log)))
        dt0 = float(jax.nn.softplus(mo.dt_bias))
        assert A_INIT_RANGE[0] <= a <= A_INIT_RANGE[1], a
        assert DT_MIN * 0.99 <= dt0 <= DT_MAX * 1.01, dt0


def test_mamba2_ledger_is_byte_identical_to_the_delta_arms():
    """⭐ The point of the iso-state sizing choice ``d_state = head_dim``: at the
    shipped 1364-float budget the SSD arm and the three delta arms land on the
    **same** state bytes, so the row isolates the update rule and nothing else."""
    mo = make_rival("mamba2", 5, 1, key=jax.random.PRNGKey(0))
    gdn2 = make_rival("gdn2", 5, 1, key=jax.random.PRNGKey(0))
    assert mo.d_head == gdn2.d_head == 36
    assert mo.declared_state_floats() == gdn2.declared_state_floats() == 1296
    led = mo.ledger()
    assert led.state_bytes == 5184 and led.param_floats == 2095
    assert table_budget(mo)["n_rows_affordable"] == 18
    # the conv-state exclusion is DECLARED in the ledger, not left implicit
    assert "conv_state" in led.state_convention
    assert "FAVOUR" in led.state_convention


def test_mamba2_metric_native_verdict_is_weaker_than_the_delta_arms():
    """⚠ Mamba-2 does **not** L2-normalise ``B``/``C`` (GDN-2 §3.5 does), so
    ``arg-min ||q-k||`` and ``arg-max q.k`` do not coincide in its own key space.
    The verdict must say so — it is the mechanism the arm's PREREG predicts from."""
    from chlu.eval.rivals.mamba2 import metric_native_verdict

    out = metric_native_verdict()
    assert out["verdict"] == "metric-native (unnormalised)"
    assert delta_verdict("gdn2")["verdict"] == "metric-native"
    assert "L2-normalise" in out["argument"] and "WEAKLY" in out["argument"]
    assert "2405.21060" in out["citation"]


def test_mamba2_keys_are_not_l2_normalised_but_the_delta_arms_are():
    """The equation-level difference above, checked on the actual projections."""
    xs, _ = _stream(t=6, seed=8)
    k_m, _ = _mamba2(seed=8).kv_table(jnp.asarray(xs))
    k_d, _ = DeltaMemory(5, 6, 1, key=jax.random.PRNGKey(8),
                         variant="gdn2").kv_table(jnp.asarray(xs))
    n_m = np.linalg.norm(np.asarray(k_m), axis=-1)
    n_d = np.linalg.norm(np.asarray(k_d), axis=-1)
    assert np.allclose(n_d, 1.0, atol=1e-3)          # GDN-2 §3.5
    assert n_m.std() > 1e-3                          # Mamba-2: free norms


def test_mamba2_is_appended_last_so_the_banked_fit_keys_do_not_move():
    """⛔ The per-rival fit key is ``RIVALS.index(name)``. Appending `mamba2` at
    the END is what keeps every banked C2W4/C2W5 number reproducible; inserting it
    anywhere else would silently re-draw the later arms' inits."""
    assert RIVALS[:5] == ("ttt_linear", "ttt_mlp", "deltanet", "gdn", "gdn2")
    assert RIVALS[5] == "mamba2" and len(RIVALS) == 6


def test_mamba2_block_parts_are_off_by_default_and_reachable_as_an_ablation():
    """⛔ ``use_D`` / ``gate_z`` are OFF in every audited cell — the same
    minimality every arm carries. They exist so the "you hobbled Mamba-2" attack
    can be answered by a measurement that runs through the SAME outer loop
    (``make_rival(..., **arm_kwargs)``), not a hand-rolled script."""
    from chlu.eval.rivals.fit import fit_grid

    base = make_rival("mamba2", 5, 1, key=jax.random.PRNGKey(0))
    blk = make_rival("mamba2", 5, 1, key=jax.random.PRNGKey(0),
                     use_D=True, gate_z=True)
    assert (base.use_D, base.gate_z) == (False, False)
    assert (blk.use_D, blk.gate_z) == (True, True)
    # the ablation costs NO state bytes (D and W_z are parameters, already counted)
    assert blk.declared_state_floats() == base.declared_state_floats()
    assert blk.ledger().param_floats == base.ledger().param_floats
    xq = np.asarray(np.random.default_rng(0).normal(size=(3, 5)), dtype=np.float32)
    xs, mask = _stream(t=5, seed=11)
    assert not np.allclose(np.asarray(base.read(base.write(xs, mask), xq)),
                           np.asarray(blk.read(blk.write(xs, mask), xq)))
    # and it is reachable through the outer loop, which is the point
    g, mos = fit_grid("mamba2", 5, 1, _fit_examples(), key=jax.random.PRNGKey(0),
                      lrs=(1e-3,), steps=2, arm_kwargs={"use_D": True})
    assert mos[0].use_D is True and len(g) == 1


# --------------------------------------------------------------------------
# ⭐ the CLU's own audit columns, first-class in the harness (C2W5 close-fix 3)
# --------------------------------------------------------------------------
#: The **published** per-seed CLU cells of the `n = 9` column (draft-r4 §4.1.1 /
#: App I.1c(b), aggregated from `.claude/outputs/bprime-rivals-f3/{run400,
#: seeds3to8}` by `.claude/scratch/bprime-referee-closures/n9_clu_column.py`).
#: ⛔ Nothing here is re-measured: these are the banked inputs, verbatim, so the
#: harness's own aggregation is checked against the numbers the paper prints.
_CLU_N9_CELLS = [
    (0, -0.6826079419590398, -0.4962609164502307, -0.4389060366704468,
     -0.8175015073205997, -0.21495899497080173, -0.23013506962922448,
     -0.44860545763514836),
    (1, -0.3846925288903795, -0.41310311933048044, -0.40420145154125486,
     -0.7142438101521984, -0.20116179179316412, -0.1925666698420529,
     -0.34393702982386115),
    (2, -0.5110319980812578, -0.4322550148126372, -0.4230789980543735,
     -0.6261158673872326, -0.250056919965218, -0.21690252112040673,
     -0.361003831918495),
    (3, -0.4215517852106097, -0.250467165074531, -0.3951167329978248,
     -0.4449522066702216, -0.03963758336626879, -0.04693652849822356,
     -0.29979591427361346),
    (4, -0.4583619478966953, -0.514969120249935, -0.3982538069357133,
     -0.5794431140942008, -0.18425168821330332, -0.1783325953454485,
     -0.30512025565243406),
    (5, -0.370921718399806, -0.3797876680493396, -0.3441830029550897,
     -0.6378105047221216, -0.16017945033906936, -0.1690580182440876,
     -0.3582028406090239),
    (6, -0.281720810428571, -0.3321685884996336, -0.4060722253052891,
     -0.7445974305277125, -0.12047228834303533, -0.11695357475373416,
     -0.3544113442043966),
    (7, -0.5261551878754182, -0.40409332807820664, -0.32024955461860694,
     -0.7391920772649633, -0.1819864689365618, -0.16754254351945594,
     -0.31366532075598746),
    (8, -0.29637950543577246, -0.2056928735965694, -0.3852237009912964,
     -0.5567294484975207, -0.044104050947466576, -0.038786031053585246,
     -0.3126934622315224),
]


def _clu_n9_records():
    recs = []
    for (seed, full, lnd, blank, null, knn2_mean, knn2_idw, tbl_mean) in _CLU_N9_CELLS:
        recs.append({
            "cell": f"aggregate/base@s{seed}", "family": "aggregate", "arm": "base",
            "seed": seed, "degenerate": False, "metric": "neg_mae",
            "label": "reader-discrimination family (S = 0.5068)",
            "rivals": {}, "admissible_coverage": {},
            "clu_reproduction": {
                "full": full, "launder": lnd, "blank": blank,
                "all_launder_scores": {
                    "settle_deleted": lnd, "same_keys_null": null,
                    "knn2_mean_+0B": knn2_mean, "knn2_idw_+0B": knn2_idw,
                    "raw_table_mean_+0B": tbl_mean},
            },
        })
    return recs


def test_audit_table_emits_the_published_n9_clu_columns():
    """⭐ **C2W5 editorial 4.** ``audit_table`` used to emit only
    ``clu_reproduced.{full,launder,dividend}`` while the paper's verdict on our
    own arm (blank / same-keys null / ``+0 B`` margin / rescue lift) came out of
    a scratch script. The harness now emits them, and on the SAME banked inputs
    it must reproduce the published `n = 9` column digit-for-digit
    (draft-r4 §4.1.1, App I.1c(b), App L.1a)."""
    from chlu.experiments.exp_bprime_rivals import audit_table

    col = audit_table(_clu_n9_records())["aggregate"]["clu_reproduced"]
    assert col["n_seeds"] == 9 and col["seeds"] == list(range(9))
    published = {
        "full": (-0.4370, 0.0417), "launder": (-0.3810, 0.0345),
        "blank": (-0.3906, 0.0124), "same_keys_null": (-0.6512, 0.0383),
        "dividend_paired": (-0.0561, 0.0315),
        "full_minus_same_keys_null": (+0.2141, 0.0443),
        "zero_byte_margin": (-0.2897, 0.0328),
        "raw_table_margin": (-0.2897, 0.0328),
        "lift_over_own_blank": (-0.0465, 0.0406),
    }
    for key, (mean, se) in published.items():
        se_key = "lift_se" if key == "lift_over_own_blank" else key + "_se"
        se_key = "dividend_se" if key == "dividend_paired" else se_key
        assert round(col[key], 4) == mean, (key, col[key])
        assert round(col[se_key], 4) == se, (key, col[se_key])
    # ⛔ the gate applied to its authors: NOT RESCUED at nine seeds
    assert col["RESCUED_above_own_blank_2se"] is False
    assert col["lift_over_own_blank"] < 2.0 * col["lift_se"]
    # the legacy key is preserved bit-for-bit (difference of the two means)
    assert col["dividend"] == pytest.approx(col["dividend_paired"], abs=1e-12)


def test_clu_zero_byte_and_raw_margins_are_paired_and_argmax_per_seed():
    """The `+0 B` reader is chosen **per seed** by arg-max over the exclusive
    set, and the raw-table candidate set adds the arg-min launder. For the CLU
    the two coincide on 9 of 9 seeds — its launder is already a raw
    ``(key, payload)`` table and never beats the 2-NN readers (draft-r4
    App I.1c(b)'s "float-identical" clause, asserted rather than asserted-in-prose)."""
    from chlu.experiments.exp_bprime_rivals import audit_table

    col = audit_table(_clu_n9_records())["aggregate"]["clu_reproduced"]
    assert col["zero_byte_margin"] == col["raw_table_margin"]
    assert col["zero_byte_substitute_per_seed"] == col["raw_table_reader_per_seed"]
    assert set(col["zero_byte_substitute_per_seed"]) <= {"knn2_mean_+0B",
                                                         "knn2_idw_+0B"}
    assert col["zero_byte_readers"] == ["knn2_idw_+0B", "knn2_mean_+0B",
                                        "raw_table_mean_+0B"]
    # paired, not a difference of column means: recompute the rule by hand
    by_hand = []
    for (_seed, full, _lnd, _blank, _null, km, ki, tm) in _CLU_N9_CELLS:
        by_hand.append(full - max(km, ki, tm))
    assert col["zero_byte_margin"] == pytest.approx(float(np.mean(by_hand)),
                                                    abs=0.0, rel=0.0)


def test_clu_columns_survive_a_record_without_the_launder_set():
    """Banked records that predate ``all_launder_scores`` must contribute
    ``nan`` to the reader-derived columns, not crash and not be silently
    dropped from ``full``/``launder``/``blank``."""
    from chlu.experiments.exp_bprime_rivals import audit_table

    recs = _clu_n9_records()
    recs[0]["clu_reproduction"].pop("all_launder_scores")
    col = audit_table(recs)["aggregate"]["clu_reproduced"]
    assert col["n_seeds"] == 9
    assert np.isfinite(col["full"]) and np.isfinite(col["blank"])
    # the reader columns drop that seed rather than the whole cell
    assert col["zero_byte_margin"] == pytest.approx(
        float(np.mean([full - max(km, ki, tm)
                       for (_s, full, _l, _b, _n, km, ki, tm)
                       in _CLU_N9_CELLS[1:]])))
