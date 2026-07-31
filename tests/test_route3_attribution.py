"""Tests for ``chlu/experiments/exp_route3_attribution.py`` (Route 3 stage 1).

Stage 1 is a **measurement**, so the properties worth pinning are the ones that
would silently corrupt a verdict: the answer channel each family is scored on,
the admissibility rule (a curve measured on an unwritten store is a measurement
of write failure), and the fact that the runner's four arms share one launch.
"""

import numpy as np
import pytest

from chlu.eval.attribution import SLOT_GRID
from chlu.experiments.exp_route3_attribution import (
    ENDPOINT_LOSS_TOL,
    ESCALATION_WRITE_STEPS,
    FAMILY_ARMS,
    _answer_channel,
    _labels,
    run_attribution_cell,
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
