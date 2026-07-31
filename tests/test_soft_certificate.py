"""Tests for SC-1…SC-7 — the merge certificate as a **monitored soft constraint**.

The spec is `doctrine-repairs.md` §4.4 (Head ruling §A9.8); these tests assert the
three things that make it doctrine rather than a knob:

1. ⛔ **DEFAULT-OFF is bit-identical** to the shipped harness (blocking — the
   C1W27 ``payload_gate`` precedent);
2. **SC-1 breaks the identification** ``d_safe := 2 s_max + kappa' sigma_q``, so
   the admission radius and the certificate radius stop being one object;
3. ⛔ **SC-3 TRIPS, it never refuses** — a soft constraint that refuses is a hard
   constraint with extra steps.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from chlu.core.clu_controller import derived_d_safe
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.monitors import MonitorContext, VacuousGateMonitor
from chlu.core.soft_certificate import (
    BUDGET_DOMAIN,
    C3_ETA,
    C3_KAPPA,
    C3_RHO_BAND,
    SC1_ZETA,
    SC3_BUDGET_B,
    SC5_STATEMENT,
    SC7_FALSIFIER,
    SoftCertificateConfig,
    budget_state,
    c3_calibration,
    capture_radius,
    cert_margin,
    cert_radius,
    expected_separation,
    sc6_state,
    soft_certificate_report,
    soft_d_safe,
)


@pytest.fixture(autouse=True)
def float32_dynamics():
    """The repo-wide x64-at-import ordering guard (handover §7.2)."""
    yield


def _tiny_cfg(**kw):
    base = dict(addr_dim=3, capacity=3, write_steps=25, address_steps=60,
                read_steps=80, n_query_per_item=2, atoms_per_item=8,
                min_atoms=48, min_atoms_base=48, seed=0)
    base.update(kw)
    return CluSystemConfig(**base)


# --------------------------------------------------------------------------
# the spec's constants (derived, not tuned — a test pins them)
# --------------------------------------------------------------------------
def test_the_spec_constants_are_the_theorists_and_are_frozen():
    assert SC1_ZETA == 0.6                      # the harness's own S4 convention
    assert SC3_BUDGET_B == 0.33                 # the measured edge
    assert (C3_KAPPA, C3_ETA) == (3.0, 0.10)    # q95(Delta/B)=4.20 -> 3; 0.062*1.6
    assert C3_RHO_BAND == (1.0 / 3.0, 3.0)      # contains N74's [0.73, 1.63]
    # ⚠ the budget's domain travels with B wherever it appears
    assert "0.15" in BUDGET_DOMAIN and "0.30" in BUDGET_DOMAIN
    assert "NO LONGER HOLDS A PRIORI" in SC5_STATEMENT       # SC-5
    assert "+3.19" in SC7_FALSIFIER and "0.294" in SC7_FALSIFIER  # SC-7


# --------------------------------------------------------------------------
# SC-1 — break the identification
# --------------------------------------------------------------------------
def test_sc1_the_two_radii_are_two_objects():
    """``R_cert`` still exists and is still computed; it is no longer the gate."""
    s_max, sigma_q = 0.30, 0.15
    r = cert_radius(s_max, sigma_q)
    assert r == pytest.approx(derived_d_safe(s_max, sigma_q))   # numerically the same
    d = soft_d_safe(sep_expected=1.20, zeta=SC1_ZETA)
    assert d == pytest.approx(0.72)
    assert d != pytest.approx(r)          # ... but a DIFFERENT object
    # and it does not depend on s_max or sigma_q at all — that is the point
    assert soft_d_safe(1.20, SC1_ZETA) == soft_d_safe(1.20, SC1_ZETA)


def test_sc1_expected_separation_is_a_property_of_the_address_geometry():
    from chlu.core.memory_potentials import designed_sites

    sites = np.asarray(designed_sites(3, 6, R=1.0, seed=0))
    sep = expected_separation(sites)
    assert 0.0 < sep < 2.0
    assert np.isnan(expected_separation(sites[:1]))


def test_sc1_changes_the_admission_radius_in_the_built_system():
    """The gate the controller actually runs on moves — measurably."""
    hard = build_system(_tiny_cfg(), loud=False)
    soft = build_system(_tiny_cfg(soft_certificate=True), loud=False)
    d_hard = float(hard.controller.allocator.d_safe)
    d_soft = float(soft.controller.allocator.d_safe)
    assert d_hard == pytest.approx(derived_d_safe(hard.cfg.atom_width,
                                                  hard.cfg.query_sigma,
                                                  hard.cfg.d_safe_kappa_prime))
    assert d_soft != pytest.approx(d_hard)
    assert d_soft == pytest.approx(SC1_ZETA * expected_separation(
        np.asarray(__import__("chlu.core.memory_potentials", fromlist=["x"])
                   .designed_sites(3, 3, R=soft.cfg.ball_radius, seed=0))))


def test_sc1_an_explicit_d_safe_override_still_wins_and_is_the_legacy_path():
    """⚠ The override WAS the soft certificate, undeclared. It is retired, not
    silently overruled: if a caller sets it, it still wins and is reported."""
    sysm = build_system(_tiny_cfg(soft_certificate=True, d_safe_override=0.58),
                        loud=False)
    assert float(sysm.controller.allocator.d_safe) == pytest.approx(0.58)


# --------------------------------------------------------------------------
# SC-2 / SC-3 — the margin and the budget
# --------------------------------------------------------------------------
def test_sc2_margin_and_relative_deficit():
    m = cert_margin(sep_after=1.0, r_cert=1.2)
    assert m["cert_margin"] == pytest.approx(-0.2)
    assert m["deficit_rel"] == pytest.approx(0.2)
    assert m["violated"] is True
    ok = cert_margin(sep_after=2.0, r_cert=1.2)
    assert ok["deficit_rel"] == 0.0 and ok["violated"] is False


def test_sc3_budget_has_two_legs_and_reports_its_domain():
    assert budget_state([0.10, 0.12], 0.33)["within_budget"] is True
    # the max leg
    assert budget_state([0.40], 0.33)["leg_max_ok"] is False
    # the mean leg: every item inside B but the mean above B/2
    st = budget_state([0.30, 0.30, 0.30], 0.33)
    assert st["leg_max_ok"] is True and st["leg_mean_ok"] is False
    assert st["within_budget"] is False
    assert st["budget_domain"] == BUDGET_DOMAIN
    assert budget_state([], 0.33)["applicable"] is False


def test_sc3_trips_monitor_3_and_does_NOT_refuse_a_write():
    """⛔ The load-bearing half: over budget => a TRIP, and the write still lands."""
    # a deliberately crowded ball: sep_after < R_cert = 2 s_max + kappa' sigma_q,
    # i.e. the certificate is genuinely violated (that is the point of SC-3)
    cfg = _tiny_cfg(ball_radius=0.45, soft_certificate=True,
                    soft_certificate_kwargs={"capture_dirs": 0})
    sysm = build_system(cfg, loud=False)
    from chlu.core.memory_potentials import designed_sites

    sites = np.asarray(designed_sites(cfg.addr_dim, 3, R=cfg.ball_radius, seed=0))
    rep = sysm.write_stream([{"item_id": 0, "address": sites[0], "payload": 0.5},
                             {"item_id": 1, "address": sites[1], "payload": -0.5}],
                            key=jax.random.PRNGKey(0))
    assert rep.admitted == [0, 1]            # NOT refused
    state = sysm.soft_certificate_state()
    assert state["SC3"]["within_budget"] is False
    r = [x for x in sysm.observe(stage="t") if x.name == "vacuous_gate"][0]
    assert r.tripped is True                 # ... but monitor #3 says so
    assert r.detail["soft_certificate_budget"]["B"] == pytest.approx(0.33)
    assert state["SC2"]["violated"] is True


# --------------------------------------------------------------------------
# SC-4(ii) / D3 — the C3 calibration leg
# --------------------------------------------------------------------------
def _pairs(ratios, lam=1.0, B=0.1):
    return [{"B": B, "delta": r * B, "lambda_min": lam} for r in ratios]


def test_c3_leg_passes_a_calibrated_store_and_trips_a_miscalibrated_one():
    ok = c3_calibration(_pairs([0.8, 0.9, 1.0, 1.1]))
    assert ok["applicable"] is True and ok["tripped"] is False
    assert ok["rho_c3"] == pytest.approx(0.95)
    bad = c3_calibration(_pairs([5.0, 6.0, 7.0]))    # rho outside [1/3, 3]
    assert bad["tripped"] is True
    unsound = c3_calibration(_pairs([1.0] * 8 + [10.0] * 4))  # P[Delta>3B] = 1/3
    assert unsound["rho_c3"] == pytest.approx(1.0) and unsound["tripped"] is True


def test_c3_leg_is_INAPPLICABLE_not_passing_below_three_qualifying_pairs():
    out = c3_calibration(_pairs([1.0, 1.0]))
    assert out["applicable"] is False and out["tripped"] is False


def test_c3_leg_disqualifies_non_minima_instead_of_clamping_lambda():
    """⛔ Fix S3: the shipped ``max(lambda, 1e-9)`` clamp sends ``B -> inf`` and
    ``rho_C3 -> 0`` at a non-minimum — a PERFECT certificate exactly where there
    is none. Here such a pair is disqualified."""
    out = c3_calibration(_pairs([1.0, 1.0, 1.0], lam=-0.5))
    assert out["n_pairs"] == 3 and out["n_qualifying"] == 0
    assert out["applicable"] is False
    mixed = c3_calibration(_pairs([1.0, 1.0, 1.0]) + _pairs([1e6], lam=-1e-12))
    assert mixed["n_qualifying"] == 3 and mixed["rho_c3"] == pytest.approx(1.0)


def test_c3_leg_ignores_drift_below_the_settles_numerical_floor():
    out = c3_calibration([{"B": 0.1, "delta": 1e-12, "lambda_min": 1.0}] * 5)
    assert out["n_qualifying"] == 0 and out["applicable"] is False


def test_monitor_3_reports_the_retired_correlation_but_never_trips_on_it():
    """The old leg survives as a diagnostic; only the C3 leg may trip."""
    log = [{"decision": "admit", "gate_margin": m, "post_write_drift": d}
           for m, d in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]]
    #                    ^ perfectly ANTI-valid: larger margin, larger drift
    log.append({"decision": "refuse_spacing"})   # so leg (i) is f in (0,1)
    ctx = MonitorContext(stage="t", t=0, system=None, write_log=log,
                         extras={"c3_pairs": _pairs([1.0, 1.0, 1.0, 1.0])})
    r = VacuousGateMonitor().observe(ctx)
    assert r.detail["validity_corr_RETIRED_DIAGNOSTIC"] == pytest.approx(-1.0)
    assert r.detail["c3_calibration_leg"]["tripped"] is False
    assert r.tripped is False       # ⛔ the retired leg did NOT trip it
    # the same context with the retired leg forced back on DOES trip => the
    # difference is attributable to the repair and to nothing else
    assert VacuousGateMonitor(c3_leg=False).observe(ctx).tripped is True


# --------------------------------------------------------------------------
# SC-6 — the floor that does not relax
# --------------------------------------------------------------------------
def test_sc6_capture_radius_measures_a_basin_and_is_min_over_directions():
    """A quadratic bowl truncated in one direction: the basin is the narrow one."""
    def relax(pts):
        p = np.asarray(pts, dtype=float)
        out = np.zeros_like(p)
        inside = np.abs(p[:, 0]) < 0.25          # captured only within |x| < 0.25
        out[inside] = 0.0
        out[~inside] = 10.0
        return out

    got = capture_radius(relax, np.zeros(2), n_dirs=16, r_hi=1.0, steps=14,
                         tol=0.05, seed=0)
    assert got["capture_radius"] == pytest.approx(0.25, abs=0.05)
    assert got["max_radius"] >= got["capture_radius"]


def test_sc6_lambda_min_positive_does_NOT_certify_a_basin():
    """The measured counter-example: basin 0.000 at ``lambda_min = +0.910``."""
    st = sc6_state([0.910], sigma_q=0.15,
                   capture={"capture_radius": 0.000})
    assert st["leg_lambda_ok"] is True
    assert st["leg_capture_ok"] is False
    assert st["certified"] is False


def test_sc6_is_INAPPLICABLE_never_passed_when_the_basin_was_not_measured():
    st = sc6_state([1.0, 2.0], sigma_q=0.15, capture=None)
    assert st["leg_capture_applicable"] is False and st["certified"] is False


# --------------------------------------------------------------------------
# ⛔ THE BLOCKING REGRESSION: default-off is bit-identical
# --------------------------------------------------------------------------
def test_soft_certificate_defaults_to_off_and_is_absent_from_the_flag_table():
    assert CluSystemConfig().soft_certificate is False
    assert CluSystemConfig().soft_certificate_kwargs == {}
    assert CluSystemConfig().as_flag_table() == {}
    assert SoftCertificateConfig().enabled is False


def test_soft_certificate_off_leaves_the_written_store_BIT_IDENTICAL():
    """⛔ BLOCKING. Explicitly-off vs default-off must agree to the last bit, and
    the admission radius must be the shipped derived one."""
    import equinox as eqx

    def _run(cfg):
        sysm = build_system(cfg, loud=False)
        from chlu.core.memory_potentials import designed_sites

        sites = np.asarray(designed_sites(cfg.addr_dim, 3, R=1.0, seed=0))
        sysm.write_stream([{"item_id": 0, "address": sites[0], "payload": 0.5},
                           {"item_id": 1, "address": sites[1], "payload": -0.5}],
                          key=jax.random.PRNGKey(0))
        return sysm

    a = _run(_tiny_cfg())
    b = _run(_tiny_cfg(soft_certificate=False))
    la = jax.tree_util.tree_leaves(eqx.filter(a.store.V, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(b.store.V, eqx.is_inexact_array))
    for x, y in zip(la, lb, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert float(a.controller.allocator.d_safe) == pytest.approx(
        derived_d_safe(a.cfg.atom_width, a.cfg.query_sigma, a.cfg.d_safe_kappa_prime))
    # and with it off, the monitors see NO soft-certificate block at all
    r = [x for x in a.observe(stage="t") if x.name == "vacuous_gate"][0]
    assert r.detail["soft_certificate_budget"] is None


def test_soft_certificate_report_carries_the_price_and_the_falsifier():
    """SC-5/SC-7 travel with the artifact — the relaxation is a PRECONDITION."""
    rep = soft_certificate_report(
        SoftCertificateConfig(enabled=True), sep_expected=1.2, sep_after=1.0,
        s_max=0.30, sigma_q=0.15, deficits=[0.10],
        c3_pairs=_pairs([1.0, 1.0, 1.0]), lambda_mins=[1.0])
    assert rep["SC1"]["identified_with_R_cert"] is False
    assert rep["SC1"]["d_safe"] == pytest.approx(0.72)
    assert rep["SC2"]["R_cert"] == pytest.approx(0.9864, abs=1e-4)
    assert rep["SC3"]["within_budget"] is True
    assert rep["SC4"]["c3_calibration"]["tripped"] is False
    assert "PRECONDITION" in rep["price"]
    assert rep["SC6"]["certified"] is False      # no capture measured => not passed
