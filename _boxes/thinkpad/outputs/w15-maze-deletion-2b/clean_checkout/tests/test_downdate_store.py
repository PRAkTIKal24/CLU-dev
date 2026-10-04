"""The single-design store's three assertions, on a synthetic corpus -- fast, every commit.

⭐ Why these exist as unit tests and not only as a run artifact: the deliverable of
``w8 spoke/downdate-build`` is three *properties*, and a property that is only ever
checked once, on one corpus, by one script, is an anecdote.  These run in about a second
against no feature cache at all, so anybody -- ``hub/single-design`` included -- can
re-check the exactness claim without rebuilding the instrument.

⛔ Every one of them asserts the exact thing rather than an approximate one, and the
negative tests assert that each check **fires** when it should.  A check that has never
failed is not evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from chlu.core import downdate_store as ds


def _toy(n: int = 4000, seed: int = 0, degenerate: bool = True):
    """3 wells over 14 channels, 3 factors -- and a RANK-DEFICIENT well.

    ⭐ The rank deficiency is the point of the fixture, not decoration: it is where a
    downdate and a refit disagree by O(1) unless the store's definition cuts the rank,
    and it is the defect this object was built around.

    ⚠ MEASURED, and it is why the deficiency is COLLINEARITY and not a zeroed channel:
    a channel that is exactly 0 puts an exactly-zero row and column in the covariance,
    and ``eigh`` resolves that block to the canonical axes deterministically on both
    paths -- the two stores then agree even with no rank cut, and the fixture would be
    testing nothing. Real rank deficiency comes from channels that are linear
    combinations of others, whose null eigenspace is rotated and genuinely arbitrary.
    So well 1 gets one dead channel (the easy case) and well 2 gets two collinear ones,
    giving it a 2-dimensional null eigenspace -- the case that actually bites, since a
    1-dimensional one is pinned by the sign convention anyway.
    """
    rng = np.random.default_rng(seed)
    cards = [4, 3, 5]
    wells = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 14)]
    D = 14
    X = rng.normal(size=(n, D)).astype(np.float32)
    if degenerate:
        X[:, 9] = 0.0                          # well 1: one channel with no variance
        X[:, 11] = 2.0 * X[:, 10]              # well 2: collinear, so the null space
        X[:, 12] = -0.5 * X[:, 10]             # is 2-dimensional and freely rotatable
    idx = np.stack([rng.integers(0, c, n) for c in cards], axis=1)
    values = idx.astype(np.float64)
    grid_values = [np.arange(c, dtype=np.float64) for c in cards]
    return X, values, grid_values, wells, D, cards


def _fit(X, values, grid_values, wells, D, cards, n_atoms=6):
    st = ds.DowndatableStore(wells, D, n_atoms, cards)
    st.absorb(X, values, grid_values)
    st.refresh()
    return st


# ---------------------------------------------------------------------------
# ASSERTION 1: the downdate EQUALS the without-rows refit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_del", [1, 7, 500])
def test_downdate_equals_refit(n_del):
    """⭐⭐ THE DELIVERABLE: downdate(all, rows) == fit(all minus rows), per well."""
    X, values, gv, wells, D, cards = _toy()
    mask = np.zeros(X.shape[0], bool)
    mask[np.random.default_rng(1).permutation(X.shape[0])[:n_del]] = True

    st = _fit(X, values, gv, wells, D, cards)
    st.delete(X[mask], values[mask], gv)
    ref = _fit(X[~mask], values[~mask], gv, wells, D, cards)

    cmp = ds.compare_stores(st, ref)
    # one float32 ulp at unit scale is the finest distinction a float32 store can draw
    # between two unit directions; the downdate is required to be inside it.
    assert cmp["max_dev_in_float32_ulp"] <= 1.0, cmp["max_abs_dev_per_well"]
    assert cmp["ranks_agree_in_every_well"]
    assert cmp["counts_equal"]
    assert cmp["term_counts_equal"] and cmp["pair_counts_equal"]
    assert cmp["coupling_max_abs_dev"] == 0.0


def test_equality_check_fires_on_a_nudged_atom():
    """⛔ THE NEGATIVE CONTROL: the check must fail when the stores really differ."""
    X, values, gv, wells, D, cards = _toy()
    mask = np.zeros(X.shape[0], bool)
    mask[:50] = True
    st = _fit(X, values, gv, wells, D, cards)
    st.delete(X[mask], values[mask], gv)
    bad = _fit(X[~mask], values[~mask], gv, wells, D, cards)
    bad.modes[0][1, 0] = np.float32(float(bad.modes[0][1, 0]) + 1e-3)

    cmp = ds.compare_stores(st, bad)
    assert cmp["max_abs_dev"] >= 1e-4
    assert cmp["max_dev_in_float32_ulp"] > 1.0


def test_rank_cut_is_what_makes_the_store_well_posed():
    """⭐ Without the rank cut a dead-channel well disagrees by O(1); with it, it does not.

    ⛔ This is the test that stops the cut being read as a tolerance chosen to pass:
    it shows the failure the cut removes, on the same rows, in the same code.
    """
    X, values, gv, wells, D, cards = _toy()
    mask = np.zeros(X.shape[0], bool)
    mask[:300] = True

    # ⚠ BOTH cuts disabled -- a negative eigenvalue tolerance and a zero variance
    # share -- so the store asks for min(A-1, d_k) directions exactly as PCASpanStore
    # does, including the ones spanning the null eigenspace. That is the pre-fix
    # object, reproduced. ⭐ Disabling only one of the two no longer reproduces it,
    # which is itself evidence that the declared variance-share cut bites.
    loose = ds.DowndatableStore(wells, D, 6, cards, eig_rel_tol=-1.0,
                                var_share_tol=0.0)
    loose.absorb(X, values, gv)
    loose.refresh()
    loose.delete(X[mask], values[mask], gv)
    loose_ref = ds.DowndatableStore(wells, D, 6, cards, eig_rel_tol=-1.0,
                                    var_share_tol=0.0)
    loose_ref.absorb(X[~mask], values[~mask], gv)
    loose_ref.refresh()

    tight = _fit(X, values, gv, wells, D, cards)
    tight.delete(X[mask], values[mask], gv)
    tight_ref = _fit(X[~mask], values[~mask], gv, wells, D, cards)

    loose_dev = ds.compare_stores(loose, loose_ref)["max_abs_dev"]
    tight_dev = ds.compare_stores(tight, tight_ref)["max_abs_dev"]
    # well 2 spans only 2 dimensions of its 4 channels, so without a cut the store
    # writes 2 directions of a 2-D null eigenspace and the two stores disagree by an
    # O(1) rotation of it -- exactly the failure measured on wells 3 and 4 of the
    # deployed partition, where the raw deviation was 1.0
    assert loose_dev > 1e-3, "the fixture no longer reproduces the defect being fixed"
    assert tight_dev < 1e-6
    assert tight_dev < loose_dev / 1e3


def test_rank_cut_margin_is_reported_on_both_sides():
    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)
    rep = st.rank_report()
    for w in rep["per_well"]:
        assert w["rank_kept"] >= 1
        assert "smallest_kept_over_threshold" in w
        assert "largest_dropped_over_threshold" in w


# ---------------------------------------------------------------------------
# ASSERTION 2: no pass over the data
# ---------------------------------------------------------------------------


def test_delete_consumes_only_the_rows_it_was_handed_at_any_corpus_size():
    """⛔ The number must be the same at two corpus sizes -- small is not the claim."""
    consumed = []
    for n in (2000, 8000):
        X, values, gv, wells, D, cards = _toy(n=n)
        st = _fit(X, values, gv, wells, D, cards)
        before = int(ds.ROWS_CONSUMED["n"])
        st.delete(X[:32], values[:32], gv)
        consumed.append(int(ds.ROWS_CONSUMED["n"]) - before)
    assert consumed[0] == consumed[1] == 32 * 3      # 32 rows x 3 wells
    assert len(set(consumed)) == 1


def test_delete_runs_with_every_route_to_the_corpus_armed_to_raise():
    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)
    out = ds.no_data_pass(st.delete, X[:16], values[:16], gv)
    assert out["rows_consumed"] == 16 * 3
    assert out["n_tripwires"] >= 3
    assert out["result"]["n_rows_deleted"] == 16


def test_the_tripwire_itself_fires():
    """⛔ A tripwire that cannot fire proves nothing."""
    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)

    def _sneaky(*_a, **_k):
        np.load("anything.npy")               # exactly what a refit would do

    with pytest.raises(AssertionError, match="NO PASS OVER THE DATA"):
        ds.no_data_pass(_sneaky)
    assert callable(np.load), "the tripwire did not restore numpy.load"


# ---------------------------------------------------------------------------
# ASSERTION 3: items are never kept
# ---------------------------------------------------------------------------


def test_object_size_is_independent_of_the_corpus():
    """⭐ The decisive form: an object that keeps items grows with the corpus."""
    sizes = []
    for n in (2000, 8000):
        X, values, gv, wells, D, cards = _toy(n=n)
        st = _fit(X, values, gv, wells, D, cards)
        au = ds.retention_audit(st, n)
        sizes.append(au["total_object_bytes"])
        assert au["items_kept"] is False
        assert au["no_array_has_an_n_rows_axis"]
    assert sizes[0] == sizes[1]


def test_retention_audit_fires_on_an_undeclared_array():
    """⛔ The audit must catch an array nobody declared -- that is where a row hides."""
    X, values, gv, wells, D, cards = _toy(n=2000)
    st = _fit(X, values, gv, wells, D, cards)
    st.smuggled_rows = X[:100].copy()          # an item store, hidden on the object
    with pytest.raises(AssertionError, match="not in state_arrays"):
        ds.retention_audit(st, 2000)


def test_retention_audit_fires_when_a_well_has_fewer_rows_than_channels():
    """⛔ n <= d_k is the one case where a covariance can expose its own rows."""
    # ⚠ n = 2, not 4 or 5: those collide with a declared axis length (well width,
    # factor cardinality) and would trip the n_rows-axis check first, which would make
    # this test pass for the wrong reason.
    X, values, gv, wells, D, cards = _toy(n=2)
    st = _fit(X, values, gv, wells, D, cards)
    assert min(st.well_sizes) > 2, "the fixture must have wells wider than the corpus"
    with pytest.raises(AssertionError, match="n <= d_k"):
        ds.retention_audit(st, 2)


# ---------------------------------------------------------------------------
# The other two parts of the object
# ---------------------------------------------------------------------------


def test_counts_downdate_exactly_and_never_go_negative():
    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)
    st.delete(X[:100], values[:100], gv)
    ref = _fit(X[100:], values[100:], gv, wells, D, cards)
    assert np.array_equal(st.term_n, ref.term_n)
    assert np.array_equal(st.pair_n, ref.pair_n)
    assert int(st.term_n.sum()) == (X.shape[0] - 100) * len(cards)
    with pytest.raises(AssertionError, match="count went negative"):
        st._count(values[:100], gv, sign=-1)
        st._count(values[:100], gv, sign=-1)
        st._count(values, gv, sign=-1)


def test_hole_is_one_byte_and_leaves_the_coupling_value_alone():
    """⭐ w5's mechanism: 1 byte, 0 outside, and the value untouched."""
    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)
    before = st.coupling.copy()
    rec = st.punch_hole(0, 1, 2, 3)
    assert rec["permitted_bytes"] == 1
    assert rec["coupling_value_unchanged"]
    assert np.array_equal(st.coupling, before)       # 0 bytes outside the hole
    assert int(st.coupling_hole.sum()) == 1
    assert st.coupling_hole[rec["coupling_index"]] == 1


def test_hole_index_space_is_gen_fields_own():
    """⛔ A hole here must BE w5's hole, not a lookalike at a different index."""
    out = ds.assert_layout_matches_field_store([4, 3, 5])
    assert out["layout_equals_gen_field"]
    assert out["n_pairs"] == 4 * 3 + 4 * 5 + 3 * 5


def test_ledgers_are_separate_and_the_covariance_is_never_netted_in():
    """⛔ Task section 3: the store's bytes and the covariance's bytes, separately."""
    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)
    store = st.byte_ledger(budget_bytes=6 * D * 4)
    cov = st.covariance_ledger()
    honest = st.honest_ledger(budget_bytes=6 * D * 4)
    assert store["deployed_bytes"] == 6 * D * 4
    assert store["matches_budget"]
    assert cov["covariance_bytes"] == sum(w.size ** 2 for w in wells) * 8
    # the covariance is NOT inside the store's number, and the honest sum exceeds it
    assert honest["store_bytes"] == store["deployed_bytes"]
    assert honest["honest_total_bytes"] > honest["store_bytes"]
    assert honest["over_budget"]


def test_the_store_speaks_the_programmes_span_interface():
    """It must read through gen_read unchanged, or no existing arm can deploy it."""
    from chlu.core.gen_read import render_span, span_basis

    X, values, gv, wells, D, cards = _toy()
    st = _fit(X, values, gv, wells, D, cards)
    basis = span_basis(st)
    assert len(basis) == st.K == 3
    coeffs = [np.zeros((7, np.asarray(U).shape[1]), np.float32) for U in basis]
    out = render_span(st, coeffs, basis)
    assert out.shape == (7, D)
