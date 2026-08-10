"""Tests for the C2W8 pass-3 **SPINE** (``exp_capture_strong_phi``).

What must never break silently here:

  * ⛔ **the census stays FROZEN.** The spine substitutes three module-level names
    on ``exp_well_lifecycle`` (``cl_config`` / ``build_phi`` / ``store_config``) and
    must put **all three back** even when a cell raises — otherwise a later arm, or a
    later *experiment*, silently scores a different rig. The substitutions must also
    never see their own replacement (they are bound once, at import);
  * ⛔ **Head ruling R2(b): the launder reads the PROJECTED φ.** ``launder_audit``
    must **raise** on the handicap match (a wide-φ launder against a narrow store)
    and on a launder whose keys are not bit-identical to the store's addresses;
  * ⛔ **the store arm is DECLARED, never inherited.** The banked arm-A census ran at
    ``atom_width_frac_spacing = 1.5``; the shipped default is 0.5 and does not clear
    the pass-2 gate. ``strong_store_config`` must honour **this** experiment's value
    and must not mutate the caller's config while doing it;
  * ⭐ **the joint dial.** ``(addr_dim, atom budget)`` is one dial and ``d = 12`` is
    32 768 atoms. ⛔ ``d = 16`` is a declared NOT-RUN (the store is inert there), and
    the default must say 12;
  * ⭐ **the branch is computed, never argued** — ``daylight_verdict`` must fire on a
    planted positive margin and refuse on a planted negative one, and it must call
    branch **(b)** a *finding*;
  * ⚠ **the D2a diagnostic is two-sided** — ``d2a_cooccurrence`` must flag the
    best-score/lowest-drift co-occurrence and must refuse to claim one from a single
    cell;
  * **the φ term is on the byte ledger of every arm including the launder**, and it
    is the *same number* on both rows.

Everything here runs on tiny synthetic arrays — no download, no encoder fitting.
"""

import copy

import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments import exp_capture_strong_phi as csp
from chlu.experiments import exp_well_lifecycle as ewl
from chlu.experiments.exp_well_lifecycle import PhiAddress
from chlu.experiments.phi_encoders import PhiProjection, ProjectedReadIn


# ---------------------------------------------------------------------------
# a tiny stand-in φ with a parameter count (the ledger needs one)
# ---------------------------------------------------------------------------
class _ToyPhi:
    def __init__(self, dim, k, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(k, dim)).astype(np.float32)
        self.k = int(k)

    def __call__(self, X):
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return X @ self.W.T

    def param_floats(self):
        return int(self.W.size)


def _toy(n=48, dim=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def _projected(dim=40, wide=16, d=6, seed=0):
    X = _toy(dim=dim, seed=seed)
    phi = _ToyPhi(dim, wide, seed=seed)
    return ProjectedReadIn(phi, PhiProjection(phi(X), d, form="pca")), X


def _gaddr(a3a, se_a, a3b=None, se_b=0.0, a1=0.4, drift_pass=True):
    """A minimal G-ADDR block shaped like the shipped one."""
    return {
        "A1": {"correct_basin_rate": float(a1), "pass": True},
        "A2": {"never_addressed_frac": 0.1, "pass": True},
        "A3": {
            "A3a_cue_margin": float(a3a), "A3a_se_paired": float(se_a),
            "A3a_strict_margin": float(a3a), "A3a_store_rate": 0.5,
            "A3a_launder_rate": 0.5 - float(a3a),
            "A3b_stream_margin": (None if a3b is None else float(a3b)),
            "A3b_se_pooled": (None if a3b is None else float(se_b)),
            "A3b_applicable": a3b is not None, "pass": True,
        },
        "gate_addr_pass": bool(drift_pass),
    }


def _legs(drift_ratio, cap=True, dec=True, drift=True):
    return {
        "G_CAP": {"pass": bool(cap)}, "G_DEC": {"pass": bool(dec)},
        "G_DRIFT": {"pass": bool(drift), "ratio": float(drift_ratio)},
    }


# ---------------------------------------------------------------------------
# 1. ⛔ the census stays frozen: the substitutions are scoped and reversible
# ---------------------------------------------------------------------------
def test_the_three_substitutions_are_restored_even_when_a_cell_raises():
    """A leaked substitution silently re-rigs every later arm and experiment."""
    before = (ewl.cl_config, ewl.build_phi, ewl.store_config)

    def boom(*a, **k):
        raise RuntimeError("cell exploded")

    cfg = get_default_config()
    orig_run = ewl.run_census_cell
    ewl.run_census_cell = boom
    try:
        with pytest.raises(RuntimeError, match="cell exploded"):
            csp.run_cell(cfg, 0, "randconv")
    finally:
        ewl.run_census_cell = orig_run
    assert (ewl.cl_config, ewl.build_phi, ewl.store_config) == before


def test_the_frozen_names_are_bound_at_import_so_a_patch_cannot_recurse():
    assert csp._FROZEN_CL_CONFIG is ewl.cl_config
    assert csp._FROZEN_BUILD_PHI is ewl.build_phi
    assert csp._FROZEN_STORE_CONFIG is ewl.store_config


# ---------------------------------------------------------------------------
# 2. ⛔ Head ruling R2(b): the launder reads the PROJECTED φ
# ---------------------------------------------------------------------------
def _stream(X):
    return {"train_X": [X]}


def test_launder_audit_passes_when_the_launder_reads_the_projected_phi():
    pp, X = _projected(d=6)
    embed = PhiAddress(pp, dim=7, addr_dim=6, scale=0.5)
    rep = csp.launder_audit(pp, embed, _stream(X), 6)
    assert rep["launder_key_dim"] == 6 == rep["store_address_dim"]
    assert rep["launder_reads_projected_phi"]
    assert rep["bit_identical_to_store_addresses"]
    assert rep["phi_dim_before_map"] == 16


def test_launder_audit_raises_on_the_handicap_match():
    """A 16-dim launder against a 6-dim store is not a launder (fairness §A4.3)."""
    X = _toy()
    phi = _ToyPhi(40, 16)
    embed = PhiAddress(phi, dim=17, addr_dim=16, scale=0.5)  # the UNPROJECTED φ
    with pytest.raises(AssertionError, match="handicap match"):
        csp.launder_audit(phi, embed, _stream(X), 6)


def test_launder_audit_raises_when_the_keys_are_not_the_stores_own():
    """Same width, different φ ⇒ still not a same-keys launder."""
    pp, X = _projected(d=6, seed=0)
    other, _ = _projected(d=6, seed=7)
    embed = PhiAddress(other, dim=7, addr_dim=6, scale=0.5)
    with pytest.raises(AssertionError, match="bit-identical"):
        csp.launder_audit(pp, embed, _stream(X), 6)


def test_build_projected_phi_refuses_a_map_that_would_truncate(monkeypatch):
    X = _toy()
    phi = _ToyPhi(40, 16)
    monkeypatch.setattr(csp, "_FROZEN_BUILD_PHI",
                        lambda regime, stream, cl, seed: (phi, {"arm": "toy"}))
    stream = {"fit_pool_task1_only": X}
    cl = type("cl", (), {"phi_dim": 16})()
    sink = {}
    pp, prov = csp.build_projected_phi("task1_only", stream, cl, 0, addr_dim=6,
                                       form="pca", sink=sink)
    assert pp.k == 6 and sink["phi"] is pp
    assert prov["map_param_floats"] == 16 + 6 * 16
    assert prov["phi_param_floats_total"] == phi.param_floats() + prov["map_param_floats"]
    # ⛔ an identity "map" at a mismatched width is the shipped defect, not a map
    with pytest.raises(ValueError):
        csp.build_projected_phi("task1_only", stream, cl, 0, addr_dim=6,
                                form="identity")


# ---------------------------------------------------------------------------
# 3. ⛔ the store arm is DECLARED (1.5), never inherited from the 0.5 default
# ---------------------------------------------------------------------------
def test_the_store_width_comes_from_this_experiment_not_the_arm_a_default():
    cfg = get_default_config()
    # the trap: arm A's shipped default is the PILOT cell, not the banked census
    assert cfg.experiment_capture_arm_a.atom_width_frac_spacing == 0.5
    assert cfg.experiment_capture_strong_phi.atom_width_frac_spacing == 1.5
    w = cfg.experiment_well_lifecycle
    d_safe = 0.88 * 0.10  # d_safe_frac x a measured spacing of 0.10
    sc = csp.strong_store_config(cfg, 0, d_safe)
    assert sc.atom_width == pytest.approx(1.5 * 0.10)
    assert sc.atom_kernel == "wendland"
    assert sc.atom_site_local_init is True
    assert sc.addr_dim == int(w.addr_dim)


def test_strong_store_config_does_not_mutate_the_callers_config():
    cfg = get_default_config()
    snapshot = copy.deepcopy(cfg.experiment_capture_arm_a)
    csp.strong_store_config(cfg, 0, 0.088)
    assert cfg.experiment_capture_arm_a == snapshot


def test_the_store_config_accepts_the_overrides_seam():
    """`run_census_cell` ALWAYS passes `overrides=`; a factory that refuses it
    makes the whole arm unrunnable (the bug `c2w8p3-gate-addr` found on main)."""
    cfg = get_default_config()
    sc = csp.strong_store_config(cfg, 0, 0.088, overrides={"write_lr": 5e-3})
    assert sc.write_lr == 5e-3


# ---------------------------------------------------------------------------
# 4. ⭐ the joint dial: (d, atom budget), and d = 16 is a declared NOT-RUN
# ---------------------------------------------------------------------------
def test_the_default_d_is_12_and_carries_its_atom_budget():
    g = get_default_config().experiment_capture_strong_phi
    assert g.addr_dim == 12
    assert csp.atom_budget(12) == 32768
    # ⛔ the geometry-favoured 16 is 4x that and the store is INERT there
    assert csp.atom_budget(16) == 131072


def test_phi_dim_is_the_priced_256_on_conv_arms_and_d_on_the_reference():
    cfg = get_default_config()
    assert csp.phi_dim_for(cfg, "simclr") == 256
    assert csp.phi_dim_for(cfg, "randconv") == 256
    assert csp.phi_dim_for(cfg, "pca") == cfg.experiment_capture_strong_phi.addr_dim


def test_cl_config_applies_the_cifar_preset_before_the_explicit_knobs():
    """The `cl-encoder` §10 preset trap: `apply_cifar10` resets n_fit_pool, so it
    must run FIRST or the run silently uses a different φ fit pool."""
    cfg = get_default_config()
    cl = csp.cl_config_for(cfg, "simclr")
    g = cfg.experiment_capture_strong_phi
    assert cl.dataset == "cifar10"
    assert cl.n_fit_region == g.n_fit_region == 25000
    assert cl.n_fit_pool == g.n_fit_pool == 6000
    assert cl.enc_steps == 8000 and cl.phi_dim == 256
    assert cl.phi_regimes == ["task1_only"]


# ---------------------------------------------------------------------------
# 5. ⭐ the branch is COMPUTED, never argued
# ---------------------------------------------------------------------------
def test_no_daylight_is_reported_as_a_finding_not_a_shortfall():
    v = csp.daylight_verdict([_gaddr(-0.30, 0.05, -0.30, 0.05) for _ in range(3)],
                             min_seeds=3)
    assert v["branch"] == "(b) NO DAYLIGHT" and v["daylight"] is False
    assert v["prereg"]["both_branches_registered_reportable"] is True
    assert "FINDING" in v["prereg"]["status_of_branch_b"]
    assert v["n_seeds_A3a_positive_beyond_2se"] == 0


def test_daylight_fires_on_a_planted_positive_margin():
    v = csp.daylight_verdict([_gaddr(+0.30, 0.02) for _ in range(3)], min_seeds=3)
    assert v["branch"] == "(a) DAYLIGHT" and v["daylight"] is True
    assert v["n_seeds_A3a_positive_beyond_2se"] == 3


def test_a_positive_margin_INSIDE_2se_is_not_daylight():
    """The rule is 'positive beyond 2 SE', not 'positive'."""
    v = csp.daylight_verdict([_gaddr(+0.03, 0.05) for _ in range(3)], min_seeds=3)
    assert v["daylight"] is False


def test_daylight_needs_min_seeds():
    v = csp.daylight_verdict([_gaddr(+0.30, 0.02)], min_seeds=3)
    assert v["daylight"] is False and v["n_seeds"] == 1


def test_the_stream_margin_can_open_daylight_on_its_own():
    v = csp.daylight_verdict([_gaddr(-0.1, 0.2, +0.30, 0.02) for _ in range(3)],
                             min_seeds=3)
    assert v["daylight"] is True and v["n_seeds_A3b_positive_beyond_2se"] == 3


# ---------------------------------------------------------------------------
# 6. ⚠ the D2a diagnostic is TWO-SIDED and refuses to over-claim
# ---------------------------------------------------------------------------
def test_the_best_scoring_cell_being_the_lowest_drift_cell_is_flagged():
    legs = [_legs(0.001), _legs(0.5, cap=False), _legs(0.4, dec=False)]
    gad = [_gaddr(0, 0, a1=0.6), _gaddr(0, 0, a1=0.2, drift_pass=False),
           _gaddr(0, 0, a1=0.3, drift_pass=False)]
    rep = csp.d2a_cooccurrence(legs, gad)
    assert rep["best_is_also_lowest_drift"] is True
    assert "D2a signature" in rep["warning"]


def test_the_cooccurrence_flag_is_not_raised_when_they_differ():
    legs = [_legs(0.5), _legs(0.001, cap=False), _legs(0.4, dec=False)]
    gad = [_gaddr(0, 0, a1=0.6), _gaddr(0, 0, a1=0.2, drift_pass=False),
           _gaddr(0, 0, a1=0.3, drift_pass=False)]
    assert csp.d2a_cooccurrence(legs, gad)["best_is_also_lowest_drift"] is False


def test_one_cell_cannot_co_occur_with_itself():
    rep = csp.d2a_cooccurrence([_legs(0.01)], [_gaddr(0, 0)])
    assert rep["best_is_also_lowest_drift"] is None
    assert "NOT INFORMATIVE" in rep["status"]


# ---------------------------------------------------------------------------
# 7. the byte ledger: φ + map on EVERY arm, launder included
# ---------------------------------------------------------------------------
def _cell(addr_dim=12, n_atoms=32768):
    return {
        "bytes": {"clu_store_bytes": 1000, "clu_codebook_bytes": 96,
                  "emission_head_bytes": 0, "clu_total_bytes": 1096,
                  "knn_launder_bytes": 832, "gamma_phi_hole_bytes": 0,
                  "gamma_phi_enabled": False},
        "flags": {"n_atoms": int(n_atoms)},
    }


def test_the_phi_term_is_the_same_number_on_the_store_row_and_the_launder_row():
    pp, _ = _projected(d=12)
    led = csp.byte_ledger(_cell(), pp, 12)
    assert led["phi_total_param_floats"] == pp.param_floats()
    assert (led["clu_total_bytes_with_phi"] - led["clu_total_bytes"]
            == led["knn_launder_bytes_with_phi"] - led["knn_launder_bytes"]
            == led["phi_total_bytes"])
    assert led["matched_bytes"] is False
    assert "1253" in led["caveat"]


def test_the_ledger_carries_the_joint_dial_and_checks_the_budget():
    pp, _ = _projected(d=12)
    led = csp.byte_ledger(_cell(), pp, 12)
    jd = led["joint_dial_(d, atom_budget)"]
    assert jd["addr_dim"] == 12 and jd["priced_budget"] == 32768
    assert jd["budget_honoured"] is True
    starved = csp.byte_ledger(_cell(n_atoms=8192), pp, 12)
    assert starved["joint_dial_(d, atom_budget)"]["budget_honoured"] is False


# ---------------------------------------------------------------------------
# 8. the completed gate needs ALL FOUR legs
# ---------------------------------------------------------------------------
def test_the_gate_needs_g_addr_too(monkeypatch):
    """Pass 2's three legs alone are the gate that could not fail on the thing
    that matters; the completed gate must refuse them."""
    cells = [{"g_addr": _gaddr(0, 0, drift_pass=False)} for _ in range(3)]
    monkeypatch.setattr(csp.armA, "gate_legs", lambda c: _legs(0.01))
    gate = csp.completed_gate(cells)
    assert gate["G_CAP_pass_seeds"] == 3 and gate["G_DRIFT_pass_seeds"] == 3
    assert gate["all_four_same_seed"] == 0
    assert gate["gate_pass"] is False
    cells_ok = [{"g_addr": _gaddr(0, 0, drift_pass=True)} for _ in range(3)]
    assert csp.completed_gate(cells_ok)["gate_pass"] is True


def test_the_gate_is_explicit_that_g_drift_is_two_sided(monkeypatch):
    monkeypatch.setattr(csp.armA, "gate_legs", lambda c: _legs(0.01))
    cells = [{"g_addr": _gaddr(0, 0)} for _ in range(3)]
    gate = csp.completed_gate(cells)
    assert "NEVER a target" in gate["d_drift_warning"]


# ---------------------------------------------------------------------------
# 9. config round-trip of the new group (the schema is additive)
# ---------------------------------------------------------------------------
def test_the_new_group_round_trips(tmp_path):
    from chlu.config import load_config, save_config

    cfg = get_default_config()
    cfg.experiment_capture_strong_phi.addr_dim = 8
    cfg.experiment_capture_strong_phi.arms = ["pca"]
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.experiment_capture_strong_phi.addr_dim == 8
    assert back.experiment_capture_strong_phi.arms == ["pca"]
