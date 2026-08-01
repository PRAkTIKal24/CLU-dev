"""Tier-iii pilot: the plan pass, the swap protocol, and the S2 gradient path.

These are the tests that would fail if the pilot's *claims* stopped being true —
the swap stops being a swap, the controller stops being the real controller, the
blank control stops being blank, or the settled-point arm stops sending zero
gradient to ``phi``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.blocks import StreamMemoryConfig
from chlu.core.clu_system import CluSystemConfig
from chlu.training.train_cluformer import (
    PilotConfig,
    _controller_plan_for_lane,
    allocation_liveness,
    assert_shared_shell_identical,
    build_arm,
    calibrate_phi_gain,
    dynamic_eval,
    evaluate,
    gradient_probe,
    loss_fn,
    monitor_pass,
    plan_pass,
    solve_arms,
    train_arm,
)


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.2).

    ⚠ **Module scope is load-bearing, not tidiness.** Several repo test modules
    (`test_lattice`, `test_goldstone`, `test_twins`, ...) call
    ``jax.config.update("jax_enable_x64", True)`` at MODULE IMPORT, so x64 is
    globally ON in a full-suite run. A *function*-scoped fixture is set up AFTER
    the module-scoped ones, so the store cell would be constructed in float64 and
    then exercised in float32 — 10 tests in this file failed exactly that way in
    the full suite while passing in isolation. An autouse module-scoped fixture
    runs before every other fixture in the module.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def pcfg():
    return PilotConfig(
        d_model=16, n_layers=2, seq_len=32, batch=2, vocab_size=32,
        addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=32,
        steps=2, warmup=1, eval_batches=1, dyneval_batches=1,
        store=dict(min_atoms=64, min_atoms_base=32),
        memory=dict(chunk=8, address_steps=6, read_steps=6, traj_stride=3,
                    psi_hidden=8, write_inner_steps=1, write_n_perturb=4,
                    retry_rounds=1, conv_kernel=3, mlp_mult=2),
    )


@pytest.fixture(scope="module")
def built(pcfg):
    specs, ledger = solve_arms(pcfg, jax.random.PRNGKey(0))
    models = {a: build_arm(a, pcfg, specs, key=jax.random.PRNGKey(5))
              for a in ("clu_store", "gru_matched", "ttt_matched", "none", "echo")}
    return specs, ledger, models


@pytest.fixture(scope="module")
def data(pcfg):
    rng = np.random.default_rng(0)
    return (rng.integers(0, pcfg.vocab_size, (pcfg.batch, pcfg.seq_len)),
            rng.integers(0, pcfg.vocab_size, (pcfg.batch, pcfg.seq_len)))


# ---------------------------------------------------------------------------
# ruling 0.1 — every full-CLU lever is ON
# ---------------------------------------------------------------------------
def test_no_full_clu_feature_is_turned_off(pcfg):
    """⛔ Head ruling 0.1, as a test. If a stage flag ever regresses to the C2W1
    default (``False``) the pilot is no longer running the full store."""
    s = pcfg.store_cfg()
    for flag in ("stage_lifetimes", "stage_admission", "stage_capacity_pressure",
                 "stage_deletion", "stage_basin_interaction", "stage_retry",
                 "stage_trajectory_read"):
        assert getattr(s, flag) is True, flag
    assert s.soft_certificate is True          # the sharing precondition
    assert s.masked_write is True              # C3-local
    assert s.retry_max_rounds >= 1
    assert s.budget < s.capacity               # real capacity pressure
    m = pcfg.memory_cfg()
    assert m.read_mode == "trajectory"         # the only channel that trains phi
    assert m.trainable_gamma and m.trainable_mass
    assert m.retry_rounds >= 1


# ---------------------------------------------------------------------------
# the swap really is a swap
# ---------------------------------------------------------------------------
def test_all_arms_share_a_bit_identical_shell(built):
    _, _, models = built
    led = assert_shared_shell_identical(models)
    assert led["shared_shell_params"] > 0


def test_a_perturbed_shell_is_caught(built):
    """The identity check must actually be able to fail."""
    _, _, models = built
    bad = eqx.tree_at(lambda m: m.head.weight, models["none"],
                      models["none"].head.weight + 1.0)
    with pytest.raises(AssertionError):
        assert_shared_shell_identical({"clu_store": models["clu_store"], "bad": bad})


def test_swap_ledger_publishes_the_gru_two_sided_impossibility(built):
    """⛔ Both GRU columns are published; the state-matched one is arithmetic."""
    _, ledger, _ = built
    assert ledger["gru_matched"]["clu_state_over_arm"] > 10.0
    arith = ledger["gru_matched_state_ARITHMETIC_ONLY"]
    assert arith["state_floats"] == ledger["clu_store"]["state_floats"]
    assert arith["params"] > 100 * ledger["clu_store"]["params"]
    assert "NOT CONSTRUCTED" in arith["note"]
    assert ledger["_byte_law_per_item"] > 2.40      # the errata floor
    assert ledger["_table_rows_at_matched_state"] > ledger["_store_items"]


def test_ttt_arm_matches_both_axes(built):
    _, ledger, _ = built
    t = ledger["ttt_matched"]
    assert abs(t["params_matched_pct"]) < 5.0
    assert 0.9 < t["state_vs_clu"] < 1.1


# ---------------------------------------------------------------------------
# the plan pass runs the REAL controller
# ---------------------------------------------------------------------------
def test_controller_lane_exercises_the_designed_verbs(pcfg):
    scfg = pcfg.store_cfg()
    from chlu.core.monitors import default_registry

    z = np.random.default_rng(1).normal(size=(12, scfg.dim)).astype(np.float32)
    out = _controller_plan_for_lane(z, scfg, default_registry(loud=False))
    st = out["_stats"]
    assert st["offers"] == 12
    assert st["refused"] + int(out["admitted"].sum()) == 12
    assert st["n_live_end"] <= scfg.capacity
    # the admission gate is a real gate: something was decided by a guard
    assert sum(st["guards"].values()) > 0


def test_plan_pass_shapes_and_blank_control(pcfg, built, data):
    _, _, models = built
    x, _ = data
    n_chunks = pcfg.seq_len // pcfg.memory_cfg().chunk
    K, dim = pcfg.capacity, pcfg.store_cfg().dim
    plans, diag = plan_pass(models["clu_store"], jnp.asarray(x), pcfg)
    assert len(plans) == pcfg.n_layers
    assert plans[0].slot.shape == (pcfg.batch, n_chunks)
    assert plans[0].sites.shape == (pcfg.batch, n_chunks, K, dim)
    assert len(diag["layers"]) == pcfg.n_layers
    bplans, _ = plan_pass(models["clu_store"], jnp.asarray(x), pcfg, blank=True)
    assert float(bplans[0].admitted.sum()) == 0.0


def test_blank_store_control_changes_the_loss(pcfg, built, data):
    """⭐ Collapse mode #4: if reading a never-written store gives the identical
    loss, the memory contributes nothing and no dividend claim is admissible.

    ⚠ Driven from a plan that **does** admit, because a real controller plan may
    legitimately admit nothing at this doll's-house scale — the admission gate
    refuses offers that are not ``d_safe``-separated, and 4 chunks of random
    tokens through a 16-wide model rarely are. (That refusal behaviour is itself
    a reported finding of the pilot; here it would silently make the control
    vacuous, so the mechanism is tested directly.)
    """
    from chlu.core.blocks import blank_plan, round_robin_plan

    _, _, models = built
    x, y = data
    m = models["clu_store"]
    n_chunks = pcfg.seq_len // pcfg.memory_cfg().chunk
    K, dim = pcfg.capacity, pcfg.store_cfg().dim

    def batched(pl):
        return [jax.tree_util.tree_map(
            lambda a: jnp.broadcast_to(a, (pcfg.batch,) + a.shape), pl)
            for _ in range(pcfg.n_layers)]

    lv = float(loss_fn(m, jnp.asarray(x), jnp.asarray(y),
                       batched(round_robin_plan(n_chunks, K, dim))))
    bl = float(loss_fn(m, jnp.asarray(x), jnp.asarray(y),
                       batched(blank_plan(n_chunks, K, dim))))
    assert np.isfinite(lv) and np.isfinite(bl)
    assert lv != bl


# ---------------------------------------------------------------------------
# ⭐ S2 — the training path
# ---------------------------------------------------------------------------
def test_gradient_probe_trajectory_beats_settled_point_by_orders(pcfg, built, data):
    """⭐ T3 in-system: ``||dL/dphi||`` through the trajectory read vs the
    settled-point arm's. The settled-point arm is not exactly 0 here — unlike the
    frozen-store probe — because ``phi`` also *writes* the store and the implicit
    gradient reaches store parameters. It must still be orders smaller."""
    _, _, models = built
    x, y = data
    gp = gradient_probe(models["clu_store"], pcfg, x, y)
    assert gp["trajectory"]["grad_phi"] > 0.0
    assert gp["ratio_traj_over_point"] > 10.0
    # gamma and mass are selectors ONLY through the trajectory channel
    assert gp["settled_point"]["grad_gamma"] == 0.0
    assert gp["settled_point"]["grad_mass"] == 0.0
    assert gp["trajectory"]["grad_gamma"] > 0.0


def test_allocation_liveness_is_reported_and_not_collapsed(pcfg, built, data):
    """T3 corollary: the all-one-slot corner is an attractor. Report the anchor."""
    _, _, models = built
    x, y = data
    al = allocation_liveness(models["clu_store"], pcfg, x, y)
    assert al["grad_phi_addr_head"] > 0.0
    assert al["policy_logits"] is None          # rule-based controller, declared
    assert all(0.0 <= e <= 1.0 for e in al["slot_entropy_normalised_per_layer"])


# ---------------------------------------------------------------------------
# training, evaluation, the mandatory dyn-eval column, monitors
# ---------------------------------------------------------------------------
def test_train_eval_and_dyneval_run_for_every_arm(pcfg, built, data):
    _, _, models = built
    x, y = data
    batches = [(x, y)] * pcfg.steps
    for name in ("clu_store", "gru_matched", "none"):
        m, hist = train_arm(name, models[name], pcfg, iter(batches))
        assert len(hist["loss_history"]) == pcfg.steps
        assert all(np.isfinite(v) for v in hist["loss_history"])
        ev = evaluate(m, pcfg, iter([(x, y)]))
        assert np.isfinite(ev["bpc"]) and 0 < ev["bpc"] < 20
        dv = dynamic_eval(m, pcfg, [(x, y)], lrs=[1e-4, 1e-3])
        assert np.isfinite(dv["bpc"])
        assert dv["best_lr"] in (1e-4, 1e-3)
        assert dv["bpc"] == min(dv["per_lr"].values())


def test_dynamic_eval_is_strictly_causal(pcfg, built, data):
    """A batch must never be scored by weights that have already seen it."""
    _, _, models = built
    x, y = data
    base = evaluate(models["clu_store"], pcfg, iter([(x, y)]))
    dv = dynamic_eval(models["clu_store"], pcfg, [(x, y)], lrs=[1e-3])
    assert dv["bpc"] == pytest.approx(base["bpc"], rel=1e-5)


def test_all_thirteen_monitors_are_observed_and_reported(pcfg, built, data):
    """⭐ The inherited acceptance criterion is checkable only if every monitor
    reports. ``inapplicable`` is a legitimate state; silence is not."""
    _, _, models = built
    x, _ = data
    mp = monitor_pass(models["clu_store"], pcfg, x)
    assert mp["applicable"]
    assert len(mp["readings"]) >= 13
    names = {r["name"] for r in mp["readings"]}
    for expected in ("overdamping", "settle_argmin", "vacuous_gate", "blank",
                     "addressing", "objective_divergence", "mass_gauge",
                     "certificates", "lifetimes", "dead_axis", "reach",
                     "starvation", "maturity"):
        assert expected in names, expected
    # the write is re-budgeted to chunk granularity, so maturity CANNOT promote
    assert mp["maturity_trips_by_arithmetic"] is True
    assert mp["maturity_write_steps"] < mp["maturity_floor"]


def test_monitor_pass_declines_on_a_non_store_arm(pcfg, built, data):
    _, _, models = built
    x, _ = data
    mp = monitor_pass(models["gru_matched"], pcfg, x)
    assert mp["applicable"] is False


def test_phi_gain_calibration_fills_the_address_ball(pcfg, data):
    """The declared anti-collapse initialisation: RMS address norm -> ball_radius."""
    x, _ = data
    g = calibrate_phi_gain(pcfg, x, key=jax.random.PRNGKey(2))
    assert g > 1.0                                   # a tanh MLP under-fills
    # IDEMPOTENT: re-calibrating an already-calibrated phi returns the same gain,
    # i.e. the RMS address norm now equals ball_radius.
    p2 = PilotConfig(**{**pcfg.__dict__, "memory": {**pcfg.memory, "phi_gain": g}})
    assert calibrate_phi_gain(p2, x, key=jax.random.PRNGKey(2)) == pytest.approx(g, rel=0.05)


def test_memory_config_defaults_match_the_declared_pilot_budget():
    """The §0.3 declarations must live in code, not only in the report."""
    m = StreamMemoryConfig()
    assert m.chunk == 64
    assert m.address_steps + m.read_steps == 128
    assert m.read_mode == "trajectory"
    _ = CluSystemConfig()
