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
