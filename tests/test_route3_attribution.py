"""Tests for ``chlu/experiments/exp_route3_attribution.py`` (Route 3 stage 1).

Stage 1 is a **measurement**, so the properties worth pinning are the ones that
would silently corrupt a verdict: the answer channel each family is scored on,
the admissibility rule (a curve measured on an unwritten store is a measurement
of write failure), and the fact that the runner's four arms share one launch.
"""

import numpy as np
import pytest

from chlu.eval.attribution import SLOT_GRID, THIRDPARTY_SLOT_GRID
from chlu.experiments.exp_route3_attribution import (
    ENDPOINT_LOSS_TOL,
    ESCALATION_WRITE_STEPS,
    FAMILY_ARMS,
    _answer_channel,
    _damped_over_bare,
    _deleted_system,
    _labels,
    _sweep_overrides,
    _t55,
    run_attribution_cell,
    run_thirdparty_cell,
)
from chlu.experiments.memory_gym import gym_config


@pytest.mark.parametrize("family,expected", [("overload", "payload"),
                                             ("aggregate", "payload"),
                                             ("manifold", "spectator")])
def test_the_answer_channel_is_the_one_the_family_is_scored_on(family, expected):
    """The instrument reads the family's OWN answer channel — it is not a new
    read-out smuggled in beside the gym's."""
    ccfg = gym_config(family, "base", seed=0).build_clu()
    idx, name = _answer_channel(family, ccfg)
    assert name == expected
    assert idx == (ccfg.addr_dim + ccfg.payload_dim if family == "manifold"
                   else ccfg.addr_dim)


def test_overload_is_quoted_only_at_the_shipped_atom_budget():
    """⚠ C2W2 reconciliation 6 (Hub-accepted): at the gym's BASE atom budget
    ``overload`` went 0/18 admissible *including the Gaussian control*."""
    assert dict(FAMILY_ARMS)["overload"] == "load1x_shipped"


def test_labels_fall_back_to_pair_ids_where_there_is_no_item_label():
    class _QS:
        label = np.array([-1, -1, -1, -1])
        meta = {"pairs": np.array([[0, 1], [0, 1], [1, 2], [1, 2]])}

    lab = _labels(_QS())
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]


def test_the_admissibility_rule_is_the_ratified_one():
    assert ENDPOINT_LOSS_TOL == 0.05          # Head ruling (i), Route-1 convention
    assert ESCALATION_WRITE_STEPS == 900      # the ONE bounded escalation


def test_a_quick_cell_produces_a_curve_on_both_channels_and_declares_its_write():
    """End-to-end smoke: 4 arms, one launch, both channels, every slot scored."""
    b = run_attribution_cell("overload", "load1x_shipped", 0, quick=True)
    assert b.family == "overload" and b.rows
    got = {(r.slot, r.channel) for r in b.rows}
    assert {c for _, c in got} == {"q", "p"}
    assert all(s in SLOT_GRID for s, _ in got)
    # a quick write does not converge, and the cell says so instead of voting
    assert b.admissible is False and "endpoint write loss" in b.reason
    for r in b.rows:
        assert np.isfinite(r.full) and np.isfinite(r.launder)
        assert r.margin == pytest.approx(r.full - r.launder - r.floor)
    # the §A8.2 curves ride along
    assert set(b.jacobian["channels"]) == {"q", "p"}
    assert len(b.jacobian["channels"]["q"]["contraction"]) == len(b.jacobian["slots"])
    # and the POST-HOC address-block diagnostic is carried, labelled as such
    assert b.flags["address_block_posthoc"][0]["block"] == "address"


# --------------------------------------------------------------------------
# ⭐⭐ C6 (C2W4 `bprime-c6`) — the third-party probe and its two closed forms
# --------------------------------------------------------------------------
def test_the_T5_5_exchange_rate_is_reproduced_TO_THE_DIGIT():
    """⭐ The sweep is priced against `bprime-theory` T5.5, so the closed form has
    to *be* T5.5 rather than resemble it. At its own ``sigma_q = 0.15, s = 0.35``:
    0.80 at ``d/s = 1.9`` (the override-rig label), 0.111 at 2.9 (the soft
    certificate), 7.0e-4 at 4.4 (the designed admission gate)."""
    s, sig = 0.35, 0.15
    assert _t55(1.9 * s, s, sig) == pytest.approx(0.80, abs=5e-3)
    assert _t55(2.9 * s, s, sig) == pytest.approx(0.111, abs=5e-4)
    assert _t55(4.4 * s, s, sig) == pytest.approx(7.0e-4, rel=2e-2)


def test_the_damping_correction_is_derived_not_fitted():
    """The read damps momentum by ``(1-gamma)`` per step, so the bare ballistic
    ``(tau^2/2M)|grad V_j|`` over-predicts. The correction is closed-form:
    **0.864 at n = 10**, **0.744 at n = 20** at ``gamma = 0.05`` — pre-registered
    before the run, and the free-fall residual is what is left over."""
    assert _damped_over_bare(10, 0.05) == pytest.approx(0.8640, abs=1e-4)
    assert _damped_over_bare(20, 0.05) == pytest.approx(0.7439, abs=1e-4)
    assert _damped_over_bare(10, 0.0) == pytest.approx(1.0)      # undamped limit
    assert _damped_over_bare(1, 0.05) == pytest.approx(1.0, abs=1e-9)


def test_the_sweep_moves_the_geometry_and_keeps_the_gate_to_geometry_ratio_fixed():
    """``ball_radius`` scales the designed site set linearly, so it moves ``d`` at
    fixed ``atom_width`` and fixed ``query_sigma`` — the axis T5.5 is a function
    of. The admission gate is scaled with it so the *only* thing changing is the
    geometry, and ``R = 1.0`` is the shipped cell."""
    assert _sweep_overrides(None) == {}
    shipped = _sweep_overrides(1.0)
    assert shipped["d_safe_override"] == pytest.approx(0.58)
    for r in (0.42, 0.64, 1.2):
        o = _sweep_overrides(r)
        assert o["d_safe_override"] / o["ball_radius"] == pytest.approx(0.58)


def test_deleting_an_item_zeroes_ITS_OWN_atoms_and_touches_nothing_else():
    """The deletion is ``A_j = amp_j**2 -> 0`` on the item's own group: exact
    removal of that item's contribution to ``V_theta``, bit-identical elsewhere.
    ⚠ NOT the shipped ``evict`` path, which *re-draws* the freed group (a
    membership-leak repair) — a re-draw substitutes a random row rather than
    deleting one."""
    import jax
    import numpy as _np

    from chlu.core.clu_system import CluSystemConfig, build_system

    cfg = CluSystemConfig(capacity=3, atoms_per_item=4, min_atoms=12,
                          min_atoms_base=12, addr_dim=2, payload_dim=1)
    sysm = build_system(cfg, key=jax.random.PRNGKey(0))
    rows = _np.asarray(sysm.store.group_rows(1), dtype=bool)
    before = _np.asarray(sysm.store.atoms.amp)
    after = _np.asarray(_deleted_system(sysm, 1).store.atoms.amp)
    assert _np.all(after[rows] == 0.0)
    assert _np.array_equal(after[~rows], before[~rows])
    assert _np.any(before[rows] != 0.0)          # the deletion was not a no-op


def test_a_quick_thirdparty_cell_measures_both_channels_and_the_tables_exact_zero():
    """End-to-end smoke: the deletions run, both channels are scored on the C7
    slot grid, and ⛔ the per-slot table's third-party Delta is **exactly 0 by
    construction (Prop T5.4)** — computed, not assumed."""
    c = run_thirdparty_cell(seed=0, quick=True, ball_radius=0.64)
    assert c["n_live"] >= 2 and c["rows"]
    got = {(r["slot"], r["channel"]) for r in c["rows"]}
    assert {ch for _, ch in got} == {"q", "p"}
    assert all(s in THIRDPARTY_SLOT_GRID for s, _ in got)
    assert all(1 <= r["step"] <= 240 for r in c["rows"])       # C7's window
    assert c["table_third_party_max_abs_delta"] == 0.0
    assert c["table_exactly_zero"] is True
    assert c["sel_agrees_with_own_item"] == pytest.approx(1.0)
    # both x-axis conventions travel with every cell (task falsifier 3)
    assert c["s_proxy_atom_width"] > 0 and c["s_fitted_well"] > 0
    assert c["d_over_s_proxy"] > 0 and c["d_over_s_fitted"] > 0
    # a quick write does not converge, and the cell says so instead of voting
    assert c["admissible"] is False and "endpoint write loss" in c["reason"]


def test_cli_exposes_exp_route3_attribution():
    """⛔ Reconciliation 6 (C2W3): the module had no CLI hook and ran only via
    ``python -m``."""
    import argparse

    from chlu.cli.experiment_cmd import setup_experiment_parsers

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_experiment_parsers(sub)
    args = parser.parse_args(["exp-route3-attribution", "--quick"])
    assert args.part == "curve" and args.quick is True and hasattr(args, "func")
    args = parser.parse_args(["exp-route3-attribution", "--part", "thirdparty",
                              "--seeds", "0", "1", "2", "--radii", "0.42", "1.0"])
    assert args.part == "thirdparty" and args.seeds == [0, 1, 2]
    assert args.radii == [0.42, 1.0]
