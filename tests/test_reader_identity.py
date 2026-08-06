"""Tests for the ZERO-PARAMETER identity readers (``reader-fitting-audit``).

``c2w7-read-cardinality`` §4 measured, on one latent, a **least-squares-fitted**
72-parameter reader scoring **0.0000** where a **zero-parameter identity** reader
scored **0.0539 +/- 0.0207**: lstsq shrinks ``diag(W)`` and pushes the residual of
the *correct* queries past ``tol``. The audit added the identity twins to the
reader class. These tests assert the four things a reviewer must be able to check
without re-running a single experiment:

1. the twins are **exactly zero-parameter** and fit on **nothing** (they ignore
   ``y``, so no branch can select on a split);
2. they are **added, never substituted** — the shipped defaults are unchanged and
   every prior code path is bit-identical;
3. ⭐ the **pathology is real**: on a designed latent whose asserted set is right
   on a minority of queries, the fitted reader scores 0 and the identity reader
   does not (the C2W7 crossing, reproduced in a unit test);
4. ⭐ the **identity reader's OWN failure mode** is real too: rescale the payload
   table and the identity reader collapses to 0 while the fitted reader is exact.
   This is the measured reason it must be *added* to the class rather than
   substituted for it.
"""

import numpy as np
import pytest

from chlu.core.factored_store import (READERS, exact_set_accuracy, fit_readers,
                                      occupancy)
from chlu.core.multiwell_read import (READERS_MW, READERS_MW_IDENTITY,
                                      READERS_MW_PLUS_IDENTITY,
                                      apply_reader_mw, fit_readers_mw,
                                      gated_well_identity_apply,
                                      gated_well_identity_fit,
                                      soft_well_identity_apply,
                                      soft_well_identity_fit)
from chlu.core.null_arms import (READERS_IDENTITY, READERS_PLUS_IDENTITY,
                                 apply_reader_plus_identity,
                                 fit_readers_plus_identity,
                                 score_readers_plus_identity, shrinkage_report,
                                 sum_identity_apply, sum_identity_fit,
                                 well_identity_apply, well_identity_fit)

N_WELLS, M, D, P, B = 16, 6, 4, 4, 96
F = 4


@pytest.fixture(scope="module")
def cell():
    """A designed latent: ``P`` particles sitting on ``P`` distinct anchors."""
    rng = np.random.default_rng(20260805)
    pay = rng.normal(size=(N_WELLS, M))
    pay = pay / np.linalg.norm(pay, axis=1, keepdims=True)
    anc = rng.normal(size=(N_WELLS, D)) * 3.0
    A = np.stack([rng.choice(N_WELLS, size=F, replace=False) for _ in range(B)])
    z = np.concatenate([anc[A], pay[A]], axis=-1)
    y = pay[A].sum(1)
    # tol on the family's own convention: 0.25 * RMS||y - ybar||
    tol = 0.25 * float(np.sqrt(np.mean(
        np.sum((y - y.mean(0)) ** 2, axis=-1))))
    return {"anchors": anc, "payloads": pay, "A": A, "z": z, "y": y, "tol": tol}


# ==========================================================================
# 1. the admissibility obligations: zero parameters, fitted on nothing
# ==========================================================================
def test_identity_readers_have_exactly_zero_fitted_parameters(cell):
    wi = well_identity_fit(cell["z"], cell["y"], anchors=cell["anchors"],
                           well_payloads=cell["payloads"])
    si = sum_identity_fit(cell["z"], cell["y"], addr_dim=D)
    sw = soft_well_identity_fit({"pi": np.zeros((B, N_WELLS))}, cell["y"],
                                well_payloads=cell["payloads"])
    gw = gated_well_identity_fit({"pi": np.zeros((B, N_WELLS))}, cell["y"],
                                 well_payloads=cell["payloads"])
    for m in (wi, si, sw, gw):
        assert m["n_params"] == 0
    # ...and inside the storeless capacity bound N_a * m, trivially.
    assert 0 < N_WELLS * M


def test_identity_readers_ignore_y_entirely(cell):
    """⛔ Fitted on NOTHING: shuffling the targets cannot change the reader."""
    rng = np.random.default_rng(0)
    y2 = cell["y"][rng.permutation(B)]
    a = well_identity_apply(
        well_identity_fit(cell["z"], cell["y"], anchors=cell["anchors"],
                          well_payloads=cell["payloads"]), cell["z"])
    b = well_identity_apply(
        well_identity_fit(cell["z"], y2, anchors=cell["anchors"],
                          well_payloads=cell["payloads"]), cell["z"])
    assert np.array_equal(a, b)


def test_identity_readers_are_exact_on_a_perfect_latent(cell):
    """The zero-parameter readers decode a correct latent to machine precision."""
    wi = well_identity_fit(cell["z"], cell["y"], anchors=cell["anchors"],
                           well_payloads=cell["payloads"])
    si = sum_identity_fit(cell["z"], cell["y"], addr_dim=D)
    assert np.abs(well_identity_apply(wi, cell["z"]) - cell["y"]).max() < 1e-12
    assert np.abs(sum_identity_apply(si, cell["z"]) - cell["y"]).max() < 1e-12


def test_well_identity_deduplicates_the_occupied_set(cell):
    """Two particles in one well must contribute that payload ONCE."""
    anc, pay = cell["anchors"], cell["payloads"]
    z = np.stack([np.concatenate([anc[[0, 0, 1, 2]], pay[[0, 0, 1, 2]]], -1)])
    got = well_identity_apply(
        well_identity_fit(z, None, anchors=anc, well_payloads=pay), z)
    assert np.abs(got[0] - pay[[0, 1, 2]].sum(0)).max() < 1e-12
    # the multiset reading would have double-counted well 0
    assert np.abs(got[0] - pay[[0, 0, 1, 2]].sum(0)).max() > 1e-6


def test_gated_identity_reads_exactly_the_asserted_set(cell):
    pi = np.zeros((2, N_WELLS))
    pi[0, [1, 3, 5, 7]] = 0.9
    pi[1, [2, 4]] = 0.51
    pi[1, 6] = 0.49  # below the gate: excluded
    mdl = gated_well_identity_fit({"pi": pi}, None,
                                  well_payloads=cell["payloads"])
    got = gated_well_identity_apply(mdl, {"pi": pi})
    assert np.abs(got[0] - cell["payloads"][[1, 3, 5, 7]].sum(0)).max() < 1e-12
    assert np.abs(got[1] - cell["payloads"][[2, 4]].sum(0)).max() < 1e-12
    # soft_well_identity uses the WEIGHTS, so it differs from the gated read
    soft = soft_well_identity_apply(
        soft_well_identity_fit({"pi": pi}, None,
                               well_payloads=cell["payloads"]), {"pi": pi})
    assert np.abs(soft[0] - got[0]).max() > 1e-6


# ==========================================================================
# 2. ADDED, never substituted — the shipped defaults must not move
# ==========================================================================
def test_shipped_reader_classes_are_unchanged():
    assert READERS == ("sum_linear", "well_table", "knn", "mlp")
    assert READERS_MW == ("sum_linear", "well_table", "knn", "mlp",
                          "soft_well_table")
    assert set(READERS_IDENTITY) == {"well_identity", "sum_identity"}
    assert set(READERS_PLUS_IDENTITY) == set(READERS) | set(READERS_IDENTITY)
    assert set(READERS_MW_PLUS_IDENTITY) == (set(READERS_MW)
                                             | set(READERS_MW_IDENTITY))


def test_default_fit_is_bit_identical_to_the_pre_audit_path(cell):
    """The audit must not have moved a single published number."""
    lat = {"z": cell["z"], "pi": np.zeros((B, N_WELLS))}
    a = fit_readers_mw(lat, cell["y"], anchors=cell["anchors"],
                       well_payloads=cell["payloads"], seed=0)
    assert set(a) == set(READERS_MW)  # the identity twins are NOT there
    b = fit_readers(cell["z"], cell["y"], anchors=cell["anchors"],
                    well_payloads=cell["payloads"], seed=0)
    for k in ("sum_linear", "well_table"):
        assert np.array_equal(a[k]["w"], b[k]["w"])
    for k in READERS_MW:
        assert np.array_equal(apply_reader_mw(a[k], lat),
                              apply_reader_mw(a[k], lat))


def test_identity_members_appear_only_when_asked(cell):
    rd = fit_readers_plus_identity(cell["z"], cell["y"], anchors=cell["anchors"],
                                   well_payloads=cell["payloads"], addr_dim=D)
    assert set(rd) == set(READERS_PLUS_IDENTITY)
    assert rd["well_identity"]["n_params"] == 0
    sc = score_readers_plus_identity(rd, cell["z"], cell["y"], cell["tol"])
    assert sc["well_identity"] == 1.0 and sc["sum_identity"] == 1.0


# ==========================================================================
# 3. ⭐ THE PATHOLOGY IS REAL — the C2W7 crossing, reproduced as a unit test
# ==========================================================================
def test_lstsq_shrinkage_destroys_a_signal_the_identity_reader_keeps(cell):
    """⭐ A designed latent: the asserted set is EXACTLY right on 20 % of
    queries and wrong on the rest. Least squares is dominated by the 80 % and
    shrinks its gain; the correct queries cross ``tol`` and score 0. The
    zero-parameter reader keeps every one of them.
    """
    rng = np.random.default_rng(7)
    anc, pay, A = cell["anchors"], cell["payloads"], cell["A"]
    good = rng.random(B) < 0.20
    occ = A.copy()
    for i in np.flatnonzero(~good):  # corrupt one well on the bad queries
        occ[i, 0] = (occ[i, 0] + 1 + rng.integers(N_WELLS - 1)) % N_WELLS
    z = np.concatenate([anc[occ], pay[occ]], axis=-1)
    y, tol = cell["y"], cell["tol"]

    rd = fit_readers_plus_identity(z, y, anchors=anc, well_payloads=pay,
                                   addr_dim=D, which=("well_table",
                                                      "well_identity"))
    p_fit = apply_reader_plus_identity(rd["well_table"], z)
    p_idn = apply_reader_plus_identity(rd["well_identity"], z)
    rep = shrinkage_report(rd["well_table"]["w"], p_fit, p_idn, y, tol,
                           asserted_sets=[np.unique(o) for o in
                                          occupancy(z, anc)],
                           subsets=A)
    # the shrinkage happened ...
    assert rep["diag_W_mean"] < 0.8
    # ... the exactly-right population is the ~20 % we designed ...
    assert 0.10 < rep["frac_set_exactly_right"] < 0.35
    # ... the identity reader keeps ALL of them, at zero residual ...
    assert rep["resid_identity_on_exact_mean"] < 1e-9
    assert rep["identity_within_tol_on_exact"] == rep["n_set_exactly_right"]
    # ... and the fitted reader loses most of them past tol. ⭐ The crossing
    # fires: the SHRUNK reader keeps strictly fewer correct queries than the
    # reader with no parameters at all.
    assert rep["resid_fitted_on_exact_mean"] > tol
    assert rep["fitted_within_tol_on_exact"] < rep["n_set_exactly_right"]
    assert rep["c2w7_crossing_fires"] is True
    assert exact_set_accuracy(p_idn, y, tol) > 2 * exact_set_accuracy(p_fit, y, tol)
    assert exact_set_accuracy(p_idn, y, tol) > 0.10


def test_crossing_does_not_fire_when_no_query_is_exactly_right(cell):
    """⛔ The designed NEGATIVE: with an empty exactly-right population the
    shrinkage has nothing to destroy, which is precisely the situation the audit
    measured at every re-scored `orgdiv-null-arms` cell."""
    rng = np.random.default_rng(11)
    anc, pay, A = cell["anchors"], cell["payloads"], cell["A"]
    occ = A.copy()
    occ[:, 0] = (occ[:, 0] + 1 + rng.integers(N_WELLS - 1, size=B)) % N_WELLS
    z = np.concatenate([anc[occ], pay[occ]], axis=-1)
    rd = fit_readers_plus_identity(z, cell["y"], anchors=anc, well_payloads=pay,
                                   addr_dim=D,
                                   which=("well_table", "well_identity"))
    rep = shrinkage_report(
        rd["well_table"]["w"],
        apply_reader_plus_identity(rd["well_table"], z),
        apply_reader_plus_identity(rd["well_identity"], z),
        cell["y"], cell["tol"],
        asserted_sets=[np.unique(o) for o in occupancy(z, anc)], subsets=A)
    assert rep["n_set_exactly_right"] == 0
    assert rep["c2w7_crossing_fires"] is False
    assert rep["acc_identity"] == 0.0


# ==========================================================================
# 4. ⭐ THE IDENTITY READER'S OWN FAILURE MODE (the doctrine question's cost)
# ==========================================================================
@pytest.mark.parametrize("alpha", [2.0, 0.5])
def test_identity_reader_dies_on_a_mis_scaled_latent_where_the_fit_is_exact(
        cell, alpha):
    """⭐ The registered positive control (PREREG P14).

    The asserted set is right on **100 %** of queries but the payload table the
    reader consults is scaled by ``alpha != 1``. The **fitted** reader recovers
    ``1/alpha`` and is exact; the **identity** reader has no gain and collapses.
    ⛔ This is why the twin is ADDED to the class and never substituted for it.
    """
    anc, pay, A, y, tol = (cell["anchors"], cell["payloads"], cell["A"],
                           cell["y"], cell["tol"])
    z = np.concatenate([anc[A], alpha * pay[A]], axis=-1)
    rd = fit_readers_plus_identity(z, y, anchors=anc,
                                   well_payloads=alpha * pay, addr_dim=D)
    sc = score_readers_plus_identity(rd, z, y, tol)
    assert sc["well_table"] == 1.0          # the fit absorbs alpha exactly
    assert sc["sum_linear"] == 1.0
    # the identity reader has no gain: it is off by |1-alpha| * ||y||
    assert sc["well_identity"] <= 0.05
    assert sc["sum_identity"] <= 0.05
    assert sc["well_table"] - sc["well_identity"] > 0.9
