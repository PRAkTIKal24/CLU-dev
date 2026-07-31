"""Tests for the 13 anti-collapse monitors (C2W1).

These are the executable form of `controller-doctrine`'s table. Three families:

* **the physics the monitors are built on** — the saddle reach criterion (#11) is
  re-derived here from scratch and must reproduce the published ``a_U`` anchors
  to <1 %, and the corrected margin law (#8-N2) must give 99 % at 2.576 sigma,
  **not** at 5 sigma (`controller-doctrine` R1);
* **the gauge test (#7)** — the ONLY monitor that lives in ``pytest``: mass is
  exactly gauge under the Newtonian kinetic term, and the comparison must be over
  the whole **trajectory** (an endpoint-only test passes vacuously once both runs
  settle into the same minimum);
* **the trip predicates themselves**, including the two the doctrine REPLACED
  (#1 must not trip on ``corr(q*, q0)``; #9 must trip on an effect size, not a
  correlation) and the "inapplicable is not passing" state of #2.
"""

import numpy as np
import pytest

from chlu.core.monitors import (
    AddressingMonitor,
    BlankControlMonitor,
    LifetimeMonitor,
    MaturityMonitor,
    MonitorContext,
    OverdampingMonitor,
    SettleArgminMonitor,
    default_registry,
    erf_margin_accuracy,
    gauge_orbit_residual,
    kappa_stat,
    saddle_reach_threshold,
)

# `readout-channel-theory` §1.3, the seven published cells: (s, D, alpha, |c|, a_U)
REACH_ANCHORS = [
    (0.30, 1.0, 0.05, 0.9, 1.023),
    (0.30, 1.0, 0.025, 0.9, 1.092),
    (0.30, 4.0, 0.05, 0.9, 1.155),
    (0.30, 1.0, 0.05, 0.3, 1.195),
    (0.45, 1.0, 0.05, 0.9, 1.522),
    (0.184, 1.0, 0.05, 0.9, 0.638),
    (0.30, 3.2, 0.05, 0.9, 1.135),
]


@pytest.mark.parametrize("s,D,alpha,c,want", REACH_ANCHORS)
def test_saddle_reach_threshold_reproduces_published_anchors(s, D, alpha, c, want):
    """#11 has ZERO free parameters, so an independent re-implementation must
    land on the published numbers, not near them."""
    got = saddle_reach_threshold(D, s, alpha, c)
    assert got == pytest.approx(want, rel=0.01)


def test_kappa_stat_matches_published_values():
    for beta, want in ((1e1, 3.33), (1e2, 4.08), (1e3, 4.67), (1e6, 6.06)):
        assert kappa_stat(beta) == pytest.approx(want, abs=0.01)


def test_kappa_stat_infinite_below_the_spurious_minimum_threshold():
    """``beta <= e^{3/2}/2 = 2.2408`` => no spurious minimum can ever exist."""
    assert np.isinf(kappa_stat(2.0))
    assert np.isfinite(kappa_stat(3.0))


def test_reach_is_logarithmically_unbuyable():
    """Depth buys reach only logarithmically — the ceiling's whole point."""
    a1 = saddle_reach_threshold(1.0, 0.3, 0.05, 0.9)
    a2 = saddle_reach_threshold(100.0, 0.3, 0.05, 0.9)
    assert a2 > a1
    assert a2 / a1 < 2.0  # 100x depth buys well under 2x reach


def test_margin_law_is_2p576_sigma_not_5_sigma():
    """`controller-doctrine` R1: kappa = 5 indexes SPACING/sigma, not margin/sigma."""
    assert erf_margin_accuracy(2.576, 1.0) == pytest.approx(0.99, abs=1e-3)
    assert erf_margin_accuracy(5.0, 1.0) > 0.999999


def test_mass_gauge_is_exact_under_newtonian_kinetics_over_the_trajectory():
    """#7: ``(M, V, p0) -> (lam M, lam V, lam p0)`` leaves the q-trajectory invariant.

    Compared over the WHOLE trajectory, not the endpoint (an endpoint-only test
    passes vacuously once both runs settle: measured 9.1e-2 -> 3.6e-3 by doubling
    the step budget alone).
    """
    import jax.numpy as jnp

    from chlu.core.memory_potentials import AtomStorePotential
    from chlu.experiments.goldstone_harness import clu_with_potential

    store = AtomStorePotential(dim=3, capacity=4, addr_dim=2)
    store = store.with_item(np.array([0.6, 0.2]), 0.4)
    store = store.with_item(np.array([-0.5, 0.3]), -0.3)
    model = clu_with_potential(store, dim=3, kinetic_mode="newtonian_learned",
                               inertia=jnp.ones(3))
    q0 = jnp.array([0.3, 0.1, 0.0])
    p0 = jnp.array([0.05, -0.02, 0.0])
    res = gauge_orbit_residual(model, q0, p0, steps=200, dt=0.05, gamma=0.05, lam=2.0)
    assert res < 1e-5, f"mass is not gauge under Newtonian T: {res}"


def test_overdamping_monitor_does_not_trip_on_the_correlation():
    """#1 REPLACED (doctrine R3): a healthy store has corr(q*, q0) ~ 0.97, which the
    provisional ``> 0.90`` predicate would trip. Only the residual/displacement
    may trip."""
    m = OverdampingMonitor()
    ctx = MonitorContext(
        reads={"grad_norm_q0": np.full(8, 1.0),
               "grad_norm_qstar": np.full(8, 1e-9),
               "displacement": np.full(8, 0.5),
               "corr_q0_qstar": 0.977},
        extras={"sep": 1.0},
    )
    r = m.observe(ctx)
    assert not r.tripped
    assert r.detail["corr_q0_qstar"] == 0.977  # reported, never tripped


def test_overdamping_monitor_trips_on_an_unconverged_settle():
    m = OverdampingMonitor()
    ctx = MonitorContext(
        reads={"grad_norm_q0": np.full(8, 1.0), "grad_norm_qstar": np.full(8, 0.5),
               "displacement": np.full(8, 0.5), "corr_q0_qstar": 0.999},
        extras={"sep": 1.0},
    )
    assert m.observe(ctx).tripped


def test_overdamping_monitor_trips_when_qstar_never_moves():
    """"the last observation": q* ~ q0 even though the gradient is small."""
    m = OverdampingMonitor()
    ctx = MonitorContext(
        reads={"grad_norm_q0": np.full(8, 1.0), "grad_norm_qstar": np.full(8, 1e-12),
               "displacement": np.full(8, 1e-4), "corr_q0_qstar": 1.0},
        extras={"sep": 1.0},
    )
    assert m.observe(ctx).tripped


def test_settle_argmin_is_INAPPLICABLE_not_passing_when_U_is_tiny():
    """#2 at ``U < u_floor`` is 0/0 — inapplicable, which is NOT a pass."""
    m = SettleArgminMonitor(u_floor=0.01)
    n = 100
    ctx = MonitorContext(reads={
        "assign_settle": np.zeros(n, dtype=int),
        "assign_argmin": np.zeros(n, dtype=int),
        "covered": np.ones(n, dtype=bool),
    })
    r = m.observe(ctx)
    assert not r.applicable
    assert not r.tripped
    assert r.verb.startswith("ESCALATE")  # #2 has NO restoring verb, by design


def test_settle_argmin_trips_when_the_settle_agrees_with_argmin_everywhere():
    """D = 0 with mass outside the certified balls => rho_ex = 0 => the dividend is
    structurally non-positive (Prop D2)."""
    m = SettleArgminMonitor()
    n = 100
    cov = np.ones(n, dtype=bool)
    cov[:20] = False
    ctx = MonitorContext(reads={
        "assign_settle": np.zeros(n, dtype=int),
        "assign_argmin": np.zeros(n, dtype=int),
        "covered": cov,
    })
    r = m.observe(ctx)
    assert r.applicable and r.tripped and r.value == 0.0
    assert r.detail["prop_D1_holds"]


def test_settle_argmin_respects_prop_D1():
    """D <= U is a theorem; the monitor reports whether it held."""
    n = 200
    rng = np.random.default_rng(0)
    cov = rng.random(n) > 0.4
    a = rng.integers(0, 4, n)
    b = a.copy()
    b[~cov] = (b[~cov] + 1) % 4  # disagreement only outside the balls
    r = SettleArgminMonitor().observe(
        MonitorContext(reads={"assign_settle": a, "assign_argmin": b, "covered": cov})
    )
    assert r.detail["prop_D1_holds"]


def test_lifetime_monitor_uses_effect_size_not_correlation():
    """#9 REPLACED (doctrine R4): a perfect rank correlation with a TINY effect size
    must not trip — that predicate fired at every excursion measured."""
    ret = np.array([0.90, 0.89, 0.88, 0.87])  # spread 0.03, corr = -1
    amp = np.array([0.1, 0.4, 0.7, 1.0])
    r = LifetimeMonitor(spread_max=0.10).observe(
        MonitorContext(self_probe={"retention": ret, "payload_abs": amp})
    )
    assert not r.tripped
    assert abs(r.detail["corr_direction_only"]) > 0.9  # reported as direction only


def test_lifetime_monitor_trips_on_a_large_effect_size():
    ret = np.array([1.0, 0.95, 0.4, 0.1])
    amp = np.array([0.1, 0.4, 0.7, 1.0])
    assert LifetimeMonitor(spread_max=0.10).observe(
        MonitorContext(self_probe={"retention": ret, "payload_abs": amp})
    ).tripped


def test_blank_control_uses_the_empirical_marginal_bar():
    m = BlankControlMonitor()
    r = m.observe(MonitorContext(blank={"score": 0.99, "chance": 0.25, "se": 0.01}))
    assert r.tripped
    r2 = m.observe(MonitorContext(blank={"score": 0.26, "chance": 0.25, "se": 0.01}))
    assert not r2.tripped


def test_addressing_monitor_trips_at_chance():
    r = AddressingMonitor(acq_min=0.9).observe(
        MonitorContext(self_probe={"acq": 0.042, "chance": 1 / 24})
    )
    assert r.tripped


def test_maturity_is_a_provenance_field_never_a_trip():
    r = MaturityMonitor(min_write_steps=40).observe(MonitorContext(extras={"write_steps": 5}))
    assert not r.tripped
    assert r.provenance["write_steps"] == 5
    assert r.detail["promotable"] is False


def test_registry_logs_trips_and_labels_never_fired_monitors_untested():
    reg = default_registry(loud=False)
    reg.observe(MonitorContext(stage="s0", blank={"score": 0.99, "chance": 0.25,
                                                  "se": 0.01}))
    assert [t.reading.name for t in reg.trips] == ["blank"]
    summary = reg.summary()
    assert summary["blank"]["ever_tripped"] and not summary["blank"]["untested"]
    # a monitor that never fired is UNTESTED, never green
    assert summary["addressing"]["untested"]
    # ...except the two that are not runtime trips by design
    assert summary["mass_gauge"]["never_trips_by_design"]
    assert summary["maturity"]["never_trips_by_design"]
    assert "TRIPPED" in reg.to_markdown()


def test_class_i_trip_is_visible_to_the_controller():
    """The trigger order is enforced through this call: no memory-mutating verb may
    fire while an instrument-validity monitor is tripped."""
    reg = default_registry(loud=False)
    reg.observe(MonitorContext(stage="s0", blank={"score": 1.0, "chance": 0.1,
                                                  "se": 0.01}))
    assert "blank" in reg.class_i_tripped()


def test_every_monitor_declares_its_false_trip_mode():
    """An uncharacterised monitor gets disabled by the next engineer, and then it is
    not a guard."""
    reg = default_registry(loud=False)
    for m in reg.monitors:
        assert getattr(m, "false_trip", None), f"{m.name} has no false-trip mode"


# ==========================================================================
# C2W2 repairs (`phi-particle-head` D4): #6 dead-band, #10 tier (a), #7 scope
# ==========================================================================
def test_monitor6_dead_band_does_not_trip_on_a_numerically_zero_slope():
    """⭐ gym R2: **29 of #6's 58 first-ever trips fired at
    ``slope_write_loss = -5.2e-17``.** A loss that "fell" by 1e-17 has not
    diverged from anything; it has hit the floating-point floor.
    """
    from chlu.core.monitors import ObjectiveDivergenceMonitor

    m = ObjectiveDivergenceMonitor(window=3)
    # a converged write: the loss is constant to float64 round-off
    loss = 0.19260269403457642
    acq = 1.0
    r = None
    for i in range(4):
        r = m.observe(MonitorContext(self_probe={
            "write_loss": loss * (1.0 - 1e-16 * i), "acq": acq * (1.0 - 1e-16 * i)}))
    assert r.applicable
    assert abs(r.detail["slope_write_loss"]) < 1e-15
    assert not r.tripped, "the dead-band did not close the epsilon artefact"
    # ...and the pre-repair predicate is recorded, so the diff is auditable
    assert r.detail["tripped_pre_repair"] is True


def test_monitor6_still_trips_on_a_genuine_divergence():
    """The other ~29 are real (e.g. ``overload/base@s2``: slope_acq -0.214,
    slope_loss -0.055) and must survive the repair."""
    from chlu.core.monitors import ObjectiveDivergenceMonitor

    m = ObjectiveDivergenceMonitor(window=3)
    r = None
    for i in range(4):
        r = m.observe(MonitorContext(self_probe={"write_loss": 1.0 - 0.055 * i,
                                                 "acq": 1.0 - 0.214 * i}))
    assert r.tripped
    assert r.detail["slope_write_loss"] == pytest.approx(-0.055, rel=1e-6)
    assert r.detail["tripped_pre_repair"] is True  # this one was never an artefact


def test_monitor6_eps_zero_reproduces_the_pre_repair_predicate_exactly():
    """``eps_rel = 0`` is the pre-repair monitor — that is how the before/after
    re-score is done without a second stochastic run."""
    from chlu.core.monitors import ObjectiveDivergenceMonitor

    old = ObjectiveDivergenceMonitor(window=3, eps_rel=0.0)
    r = None
    for i in range(4):
        r = old.observe(MonitorContext(self_probe={"write_loss": 0.5 - 1e-17 * i,
                                                   "acq": 1.0 - 1e-17 * i}))
    assert r.tripped  # the artefact, reproduced on demand


def test_objective_divergence_predicate_is_a_pure_function():
    from chlu.core.monitors import objective_divergence_predicate

    assert objective_divergence_predicate(-0.05, -0.2, 0.0)
    assert objective_divergence_predicate(-0.05, -0.2, 1e-9)
    assert objective_divergence_predicate(-5.2e-17, -5.9e-17, 0.0)  # pre-repair
    assert not objective_divergence_predicate(-5.2e-17, -5.9e-17, 1e-9)  # repaired
    assert not objective_divergence_predicate(-0.05, +0.2, 0.0)  # acq rising


# --------------------------------------------------------------------------
# C2W4 repair (`harness-debt` D3): monitor #6's MISSING `+eps_acq` half
# --------------------------------------------------------------------------
def test_monitor6_acq_dead_band_recovers_a_false_negative():
    """⛔ **The half that did not land in C2W2** (`bprime-fb4-gate` R4). With
    ``slope_acq <= 0.0``, a converged run whose acquisition slope is flat *to
    round-off on the positive side* fails the retrieval leg and does **not**
    trip — a false NEGATIVE, the mirror image of the artefact the loss half
    closed. One leg with a dead-band and one without is not a repair.
    """
    from chlu.core.monitors import ObjectiveDivergenceMonitor

    # a genuine write-loss collapse (slope -0.055) with acquisition flat to
    # round-off on the POSITIVE side
    def _run(**kw):
        m = ObjectiveDivergenceMonitor(window=3, **kw)
        r = None
        for i in range(4):
            r = m.observe(MonitorContext(self_probe={
                "write_loss": 1.0 - 0.055 * i, "acq": 0.5 + 1e-15 * i}))
        return r

    r_new = _run()
    assert r_new.detail["slope_write_loss"] == pytest.approx(-0.055, rel=1e-6)
    assert 0.0 < r_new.detail["slope_acq"] < r_new.detail["eps_acq_dead_band"]
    assert r_new.tripped, "the acq dead-band did not recover the false negative"
    # ...and the loss-half-only predicate (the C2W2-C2W3 shipped state) is
    # carried on the reading, so the C2W4 diff is re-scorable offline
    assert r_new.detail["tripped_loss_half_only"] is False
    # ⭐ turning the acq half OFF reproduces that shipped state exactly
    assert _run(eps_acq_rel=0.0).tripped is False


def test_monitor6_acq_dead_band_does_not_swallow_a_real_acquisition_rise():
    """The band is a ROUND-OFF floor, not a resolution floor: an acquisition
    rate that genuinely climbs must still clear the retrieval leg."""
    from chlu.core.monitors import ObjectiveDivergenceMonitor

    m = ObjectiveDivergenceMonitor(window=3)
    r = None
    for i in range(4):
        r = m.observe(MonitorContext(self_probe={"write_loss": 1.0 - 0.055 * i,
                                                 "acq": 0.2 + 0.05 * i}))
    assert r.detail["slope_acq"] == pytest.approx(0.05, rel=1e-6)
    assert r.detail["eps_acq_dead_band"] < 1e-9
    assert not r.tripped


def test_monitor6_eps_acq_zero_reproduces_the_loss_half_only_predicate_exactly():
    """⭐ **Blocking bit-identity gate (task §4.3).** ``eps_acq_rel = 0``
    restores the C2W2-C2W3 shipped predicate exactly — which is how the C2W4
    re-score of the published "27" is done without re-running the store. A
    repair that cannot be turned off is not auditable.
    """
    from chlu.core.monitors import ObjectiveDivergenceMonitor

    rng = np.random.default_rng(0)
    for _ in range(200):
        loss = rng.normal(0.0, 1.0, size=4) * 10.0 ** rng.integers(-18, 2)
        acq = rng.normal(0.0, 1.0, size=4) * 10.0 ** rng.integers(-18, 2)
        off = ObjectiveDivergenceMonitor(window=3, eps_acq_rel=0.0)
        on = ObjectiveDivergenceMonitor(window=3)
        r_off = r_on = None
        for i in range(4):
            ctx = MonitorContext(self_probe={"write_loss": float(loss[i]),
                                             "acq": float(acq[i])})
            r_off = off.observe(ctx)
            r_on = on.observe(ctx)
        # `eps_acq_rel = 0` == the shipped C2W2 state, and the on-state's
        # recorded `tripped_loss_half_only` reproduces it reading-for-reading
        assert r_off.tripped == r_on.detail["tripped_loss_half_only"]
        # ...and the acq band is monotone: it can only ADD trips, never remove
        assert not (r_off.tripped and not r_on.tripped)


def test_monitor6_predicate_is_monotone_in_eps_acq():
    """A dead-band on the acq leg only *widens* the trip condition, so a
    ``TRIP -> no-trip`` flip attributed to it is a contradiction."""
    from chlu.core.monitors import objective_divergence_predicate

    for eps_acq in (0.0, 1e-18, 1e-9, 1e-3, 4.2e-2, 1.0):
        assert objective_divergence_predicate(-0.05, -0.2, 1e-9, eps_acq)
    assert not objective_divergence_predicate(-0.05, +7.84e-4, 1e-9, 1e-9)
    # the theorist's `doctrine-repairs` §2.3 RESOLUTION floor (1/(n_probed*W)
    # ~ 4.2e-2) is what its 2 predicted recoveries needed; the shipped round-off
    # band (1e-9 * max|acq| <= 1e-9) is 7 orders narrower and does not reach it
    assert objective_divergence_predicate(-0.05, +7.84e-4, 1e-9, 4.2e-2)


def test_monitor10_tier_a_catches_a_declared_but_never_read_knob():
    """⭐ doctrine I-8: the O(1) plumbing tier. N19/N20/N58 are all
    "the field is wired to nothing"."""
    from dataclasses import dataclass

    from chlu.core.monitors import (
        ConfigAccessProxy,
        DeadAxisMonitor,
        DeadKnobError,
        assert_knobs_live,
    )

    @dataclass
    class Cfg:
        used: float = 1.0
        never_read: float = 2.0

    proxy = ConfigAccessProxy(Cfg())
    _ = proxy.used  # the only read
    assert proxy.never_read() == ["never_read"]
    with pytest.raises(DeadKnobError):
        assert_knobs_live(proxy)
    # and the monitor sees the same thing through ctx.extras
    r = DeadAxisMonitor().observe(MonitorContext(extras=proxy.knob_extras()))
    assert r.tripped and r.detail["never_read"] == ["never_read"]


def test_monitor10_tier_a_is_clear_when_every_knob_is_read():
    from dataclasses import dataclass

    from chlu.core.monitors import ConfigAccessProxy, DeadAxisMonitor, assert_knobs_live

    @dataclass
    class Cfg:
        a: float = 1.0
        b: float = 2.0

    proxy = ConfigAccessProxy(Cfg())
    _ = (proxy.a, proxy.b)
    assert assert_knobs_live(proxy) == []
    r = DeadAxisMonitor().observe(MonitorContext(extras=proxy.knob_extras()))
    assert not r.tripped
    assert proxy.knob_extras()["knob_tier_a_implemented"] is True


def test_config_access_proxy_delegates_reads_and_writes():
    from chlu.core.clu_system import CluSystemConfig
    from chlu.core.monitors import ConfigAccessProxy

    proxy = ConfigAccessProxy(CluSystemConfig(addr_dim=4))
    assert proxy.addr_dim == 4
    assert proxy.counts["addr_dim"] == 1
    proxy.reset()
    assert proxy.counts["addr_dim"] == 0
    assert proxy.unwrap().addr_dim == 4
    # a derived property still works through the proxy
    assert int(proxy.dim) == 5


@pytest.mark.parametrize("kinetic_mode", ["newtonian_identity", "newtonian_learned",
                                          "relativistic"])
def test_mass_gauge_is_parameterised_by_kinetic_mode(kinetic_mode):
    """⭐ doctrine I-7/R2: the gauge is **Newtonian-only**. The relativistic cell is
    recorded as a SCOPE, not a pass — under the relativistic kinetic the gauge
    breaks as O(1/c^2), and N76 then does not forbid mass as a channel."""
    import jax.numpy as jnp

    from chlu.core.memory_potentials import AtomStorePotential
    from chlu.core.monitors import GAUGE_SCOPE
    from chlu.experiments.goldstone_harness import clu_with_potential

    store = AtomStorePotential(dim=3, capacity=4, addr_dim=2)
    store = store.with_item(np.array([0.6, 0.2]), 0.4)
    store = store.with_item(np.array([-0.5, 0.3]), -0.3)
    model = clu_with_potential(store, dim=3, kinetic_mode=kinetic_mode,
                               inertia=jnp.ones(3))
    q0 = jnp.array([0.3, 0.1, 0.0])
    p0 = jnp.array([0.05, -0.02, 0.0])
    res = gauge_orbit_residual(model, q0, p0, steps=200, dt=0.05, gamma=0.05, lam=2.0)
    assert kinetic_mode in GAUGE_SCOPE
    if kinetic_mode == "newtonian_learned":
        assert res < 1e-5, f"mass is not gauge under {kinetic_mode}: {res}"
    elif kinetic_mode == "newtonian_identity":
        # ⚠ C2W2 SHARPENING of I-7: with T = 0.5 p^2 the mass is not in the
        # dynamics, so (M,V,p0) -> (lam M, lam V, lam p0) rescales V and p0 with
        # NOTHING to compensate them. It is not a gauge orbit here at all.
        assert res > 1e-2, f"unexpectedly gauge under {kinetic_mode}: {res}"
        assert "not a gauge" in GAUGE_SCOPE[kinetic_mode].lower()
    else:
        # SCOPE, not a pass: record the size of the break rather than assert it away
        assert np.isfinite(res)
        assert "BREAKS" in GAUGE_SCOPE[kinetic_mode]


def test_mass_gauge_compares_the_whole_trajectory_not_the_endpoint():
    """doctrine I-7: an endpoint-only comparison passes VACUOUSLY once both runs
    settle (9.1e-2 -> 3.6e-3 by doubling N alone). Under a mass rescaling the
    trajectories differ *in time* even when the endpoints agree, so a
    trajectory-wise residual on a NON-gauge perturbation must be large while the
    endpoint residual is small."""
    import jax.numpy as jnp

    from chlu.core.memory_potentials import AtomStorePotential
    from chlu.experiments.goldstone_harness import clu_with_potential, log_mass_for_inertia

    store = AtomStorePotential(dim=3, capacity=4, addr_dim=2)
    store = store.with_item(np.array([0.6, 0.2]), 0.4)
    model = clu_with_potential(store, dim=3, kinetic_mode="newtonian_learned",
                               inertia=jnp.ones(3))
    q0 = jnp.array([0.3, 0.1, 0.0])
    p0 = jnp.array([0.05, -0.02, 0.0])
    import equinox as eqx

    # M -> 2M WITHOUT rescaling V and p0: not the gauge orbit, so the trajectory
    # must move even though both runs settle into the same minimum.
    other = eqx.tree_at(lambda m: m.log_mass, model,
                        replace=log_mass_for_inertia(jnp.full((3,), 2.0)))
    a = np.asarray(model(q0, p0, 400, 0.05, 0.05))
    b = np.asarray(other(q0, p0, 400, 0.05, 0.05))
    traj_res = float(np.max(np.abs(a[:, :3] - b[:, :3])) / np.max(np.abs(a[:, :3])))
    end_res = float(np.max(np.abs(a[-1, :3] - b[-1, :3])) / np.max(np.abs(a[-1, :3])))
    assert traj_res > 100 * end_res, (
        f"the endpoint comparison is vacuous here: traj {traj_res} vs end {end_res}")
