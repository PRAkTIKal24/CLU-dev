"""Tests for **B′ dividend family 2** (`triad`) — the three-way superposition family.

What is under test is the family's **construction rules** and the **bound they
prove**, because the paper's entitlement to call `triad` a reader-discrimination
family rests on exactly that: `aggregate` bounds the **arg-min** launder from
below, `triad` bounds **every two-row convex reader** (arg-min, 2-NN mean, 2-NN
IDW). A later edit that weakens the drop rules would silently turn the family
into a second copy of `aggregate`, so the bound is asserted per query rather than
described.

Also under test: that adding the family's own +0 B readers cannot move a
**registered family-1 number** — the `aggregate` reader set must stay bit-for-bit
what it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig
from chlu.core.memory_potentials import designed_payloads, designed_sites
from chlu.eval.dividend import (
    barycentric_launder,
    knn_mean_launder,
    settle_deleted_launder,
)
from chlu.experiments.exp_memory_gym import _launder_predictions
from chlu.experiments.memory_gym import (
    FAMILIES,
    FAMILY_DEFAULTS,
    PRIMARY_METRIC,
    TRIAD_PREREG,
    _triples_within,
    gym_config,
    queries_triad,
    score,
)

SEEDS = (0, 1, 2, 3)


def _cell(seed: int = 0, k_live: int = 5):
    """A synthetic live codebook on the family's own geometry (no store needed)."""
    gcfg = gym_config("triad", "base", seed=seed)
    ccfg: CluSystemConfig = gcfg.build_clu()
    centers = np.asarray(designed_sites(ccfg.addr_dim, int(gcfg.n_offer),
                                        R=ccfg.ball_radius, seed=seed))[:k_live]
    pays = np.asarray(designed_payloads(int(gcfg.n_offer), seed=seed,
                                        lo=gcfg.payload_lo, hi=gcfg.payload_hi)
                      ).reshape(-1, 1)[:k_live]
    rng = np.random.default_rng(seed + 7717)
    return gcfg, ccfg, centers, pays, queries_triad(gcfg, ccfg, centers, pays, rng)


# --------------------------------------------------------------------------
# the family is registered, and at `aggregate`'s own operating point
# --------------------------------------------------------------------------
def test_triad_is_a_registered_family_with_a_registered_prereg():
    assert "triad" in FAMILIES
    assert PRIMARY_METRIC["triad"] == "neg_mae"
    for key in ("predictions", "construction_rules", "n5_rules", "kill_conditions",
                "zero_byte_readers_added", "validation_gate", "seed_plan"):
        assert key in TRIAD_PREREG, key
    # the nine-seed plan is part of the registration, not a runtime choice
    assert tuple(TRIAD_PREREG["seed_plan"]) == tuple(range(9))


def test_triad_shares_the_audited_cells_store_operating_point():
    """Only the QUERY LAW may differ from `aggregate@base` — every store knob
    (capacity, budget, atom budget, admission gate, consolidation) is identical,
    which is what makes the two families attributable to one another."""
    agg = dict(FAMILY_DEFAULTS["aggregate"])
    tri = dict(FAMILY_DEFAULTS["triad"])
    for key in ("n_offer", "capacity", "budget", "reference_capacity",
                "consolidate_every", "clu_overrides"):
        assert tri[key] == agg[key], key


# --------------------------------------------------------------------------
# the two construction rules, per query
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_C1_no_target_is_within_tolerance_of_a_stored_payload(seed):
    """N.5's subsuming rule: the target is absent from the table."""
    _g, ccfg, _c, pays, qs = _cell(seed)
    tol = float(ccfg.payload_tol)
    tgt = np.asarray(qs.target, dtype=float)
    d = np.min(np.abs(tgt[:, None, :] - np.asarray(pays)[None, :, :]).sum(-1), axis=1)
    assert float(np.min(d)) >= tol


@pytest.mark.parametrize("seed", SEEDS)
def test_C2_every_target_sits_outside_the_two_nearest_rows_hull(seed):
    """⭐ The family's own rule, and the thing its bound is derived from."""
    _g, ccfg, centers, pays, qs = _cell(seed)
    tol = float(ccfg.payload_tol)
    pay = np.asarray(pays, dtype=float)
    for n in range(len(qs)):
        near = np.argsort(np.linalg.norm(centers - qs.keys[n][None, :], axis=1))[:2]
        lo, hi = float(np.min(pay[near])), float(np.max(pay[near]))
        t = float(np.asarray(qs.target)[n, 0])
        assert (t > hi + tol - 1e-12) or (t < lo - tol + 1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_bound_every_two_row_convex_reader_errs_by_at_least_the_tolerance(seed):
    """⭐⭐ The family's entitlement, asserted directly: ANY convex combination of
    the two nearest rows is at least ``payload_tol`` away from the target — so the
    arg-min launder, the 2-NN mean and the 2-NN IDW reader are all bounded, and a
    positive dividend against them cannot be an accident of the draw."""
    _g, ccfg, centers, pays, qs = _cell(seed)
    tol = float(ccfg.payload_tol)
    pay = np.asarray(pays, dtype=float)
    lam = np.linspace(0.0, 1.0, 21)
    for n in range(len(qs)):
        near = np.argsort(np.linalg.norm(centers - qs.keys[n][None, :], axis=1))[:2]
        mix = lam[:, None] * pay[near[0]] + (1.0 - lam)[:, None] * pay[near[1]]
        err = np.min(np.abs(mix - np.asarray(qs.target)[n]))
        assert err >= tol - 1e-12
    # and the three shipped two-row readers therefore cannot score above -tol
    for pred in (settle_deleted_launder(centers, pays, qs.keys),
                 knn_mean_launder(centers, pays, qs.keys, k=2),
                 knn_mean_launder(centers, pays, qs.keys, k=2,
                                  weighting="inverse_distance")):
        assert score(qs, pred)["neg_mae"] <= -tol + 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_the_family_meets_its_own_registered_coverage_bar(seed):
    """Kill condition K1, as a test: a change that starves the query law fails."""
    _g, _c, _ce, _p, qs = _cell(seed)
    assert len(qs) >= int(TRIAD_PREREG["predictions"]["n_queries_min_per_seed"])
    assert (float(qs.meta["coverage"])
            >= float(TRIAD_PREREG["predictions"]["coverage_fraction_min"]))


def test_triples_fall_back_to_the_tightest_triple_when_none_qualify():
    centers = np.array([[0.0, 0, 0, 0], [10.0, 0, 0, 0], [0.0, 10, 0, 0],
                        [0.02, 0.01, 0, 0]])
    out = _triples_within(centers, mult=1.0)
    assert len(out) == 1 and set(out[0]) <= {0, 1, 2, 3}


# --------------------------------------------------------------------------
# the family's own +0 B readers
# --------------------------------------------------------------------------
def test_barycentric_reader_is_exact_at_a_noiseless_barycentre():
    """The reader inverts the construction when the query carries no jitter —
    which is why it is the family's declared strongest control."""
    centers = np.array([[0.0, 0, 0, 0], [1.0, 0, 0, 0], [0.0, 1, 0, 0],
                        [5.0, 5, 5, 5]])
    pays = np.array([[-1.0], [0.2], [0.9], [0.0]])
    w = np.array([0.3, 0.45, 0.25])
    q = (w[:, None] * centers[:3]).sum(0)[None, :]
    got = barycentric_launder(centers, pays, q, k=3)
    assert np.allclose(got, (w[:, None] * pays[:3]).sum(0), atol=1e-8)


def test_the_extra_readers_are_added_for_triad_and_for_nothing_else():
    """⛔ The registered `aggregate` reader set must not move: family-1 numbers are
    published, and a silently widened control set would re-price them."""
    _g, _c, centers, pays, qs = _cell(0)
    born = np.arange(centers.shape[0], dtype=float)
    base = _launder_predictions(qs, centers, pays, born,
                                np.random.default_rng(0), None)
    agg = _launder_predictions(qs, centers, pays, born,
                               np.random.default_rng(0), None, family="aggregate")
    tri = _launder_predictions(qs, centers, pays, born,
                               np.random.default_rng(0), None, family="triad")
    assert set(agg) == set(base)
    for k in base:
        assert np.allclose(agg[k], base[k])
        assert np.allclose(tri[k], base[k])
    assert set(tri) - set(base) == set(TRIAD_PREREG["zero_byte_readers_added"])
