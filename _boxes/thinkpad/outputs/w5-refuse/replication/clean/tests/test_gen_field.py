"""The field store's properties, on a synthetic landscape -- fast, and run on every commit.

The acceptance property of ``hub/field-store`` is a statement about BYTES, so these tests
are statements about bytes.  They are deliberately not "approximately zero": every one of
them asserts an exact count, because the target file is explicit that an approximate
version of this property is worth nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from chlu.core import gen_field as gf


def _toy(n_atoms: int = 16, seed: int = 0):
    """A factor-aligned synthetic landscape: 3 factors, 3 wells, 8 channels each."""
    rng = np.random.default_rng(seed)
    cards = [4, 3, 5]
    F, D, N = 3, 24, 3000
    Y = np.stack([rng.integers(0, c, N) for c in cards], 1)
    base = [rng.normal(size=(c, 8)) * 3.0 for c in cards]
    X = np.concatenate([base[f][Y[:, f]] for f in range(F)], 1).astype(np.float32)
    X += rng.normal(size=X.shape).astype(np.float32) * 0.3
    labels = np.repeat(np.arange(3), 8)
    grid_values = [np.arange(c, dtype=float) for c in cards]
    field = gf.build_field(X, Y, np.arange(N), labels, [[0], [1], [2]], cards,
                           gf.FieldSpec(n_atoms=n_atoms), log=lambda *_: None)
    return field, cards, grid_values, base


def test_ledger_is_atoms_times_dim_times_four():
    field, *_ = _toy(n_atoms=16)
    led = gf.field_byte_ledger(field, 16 * 24 * 4)
    assert led["deployed_bytes"] == 16 * 24 * 4
    assert led["matches_budget"]
    # the address book is reported BESIDE the landscape, never inside it
    assert led["interface_bytes"] > 0
    assert led["landscape_bytes"] + led["interface_bytes"] + led["provenance_bytes"] \
        == led["total_image_bytes"]


def test_atoms_are_training_items_written_into_every_well():
    field, cards, _, _ = _toy()
    assert field.view("atom_alive").sum() == field.A
    # every well holds the same number of atoms, and they are one item each
    for k in range(field.K):
        assert field.atoms_of_well(k).shape == (field.A, field.meta["well_sizes"][k])
    # every value of every factor got at least one atom
    bn = field.view("basin_n")
    assert int(bn.min()) >= 1


def test_read_is_request_only_by_signature_and_numerically():
    field, cards, gvs, _ = _toy()
    vals = np.array([[0., 0., 0.], [1., 2., 3.]])
    rep = gf.assert_read_request_only(field, vals, gvs)
    assert rep["feature_argument_present"] is False
    assert rep["bitwise_stable_under_feature_perturbation"] is True


def test_landscape_removed_collapses_the_read():
    field, cards, gvs, _ = _toy()
    vals = np.array([[0., 0., 0.], [1., 2., 3.], [3., 1., 4.]])
    live = gf.field_read(field, vals, gvs)
    dead = field.copy()
    dead.view("atoms")[:] = 0.0
    out = gf.field_read(dead, vals, gvs)
    assert np.abs(out).max() == 0.0, "zeroing the atoms must collapse the read (charter C.2)"
    assert np.abs(live).max() > 0.0


def test_eviction_changes_zero_bytes_outside_the_evicted_atoms():
    field, cards, gvs, _ = _toy()
    before = field.copy()
    rep = gf.evict_term(field, 0, 2)
    acc = gf.assert_changed_bytes_within(before, field, rep["permitted_ranges"], "evict")
    assert acc["bytes_changed_outside_permitted"] == 0
    assert acc["bytes_changed"] > 0
    assert set(acc["per_segment"]) <= {"atoms", "atom_life", "atom_alive", "atom_row",
                                       "atom_basin"}
    gf.assert_no_refit(before, field)


def test_the_hole_is_exactly_one_byte_and_is_not_a_refit():
    field, cards, gvs, _ = _toy()
    before = field.copy()
    rep = gf.punch_hole(field, 0, 1, 2, 3)
    acc = gf.assert_changed_bytes_within(before, field, rep["permitted_ranges"], "hole")
    assert acc["bytes_changed"] == 1
    assert acc["bytes_changed_outside_permitted"] == 0
    assert acc["per_segment"] == {"coupling_hole": 1}
    gf.assert_no_refit(before, field)


def test_the_hole_refuses_its_conjunction_and_nothing_else():
    field, cards, gvs, _ = _toy()
    gf.punch_hole(field, 0, 1, 2, 3)
    vals = np.array([[1., 0., 3.],      # the conjunction  -> refused
                     [1., 0., 0.],      # marginal A only  -> answered
                     [0., 0., 3.]])     # marginal B only  -> answered
    g = gf.gap_read(field, vals, gvs)
    assert bool(g["hole"][0]) is True
    assert bool(g["hole"][1]) is False
    assert bool(g["hole"][2]) is False


def test_an_unseen_pair_is_answered_and_a_deleted_pair_is_refused():
    """⭐ The trap test.  Wave 3's joint table refused every unseen conjunction; this one
    must refuse only the conjunction that was DELETED."""
    rng = np.random.default_rng(1)
    cards = [4, 3, 5]
    F, D, N = 3, 24, 3000
    Y = np.stack([rng.integers(0, c, N) for c in cards], 1)
    keep = ~((Y[:, 0] == 2) & (Y[:, 2] == 4))       # (0,2)x(2,4) NEVER held by the pool
    Y = Y[keep]
    base = [rng.normal(size=(c, 8)) * 3.0 for c in cards]
    X = np.concatenate([base[f][Y[:, f]] for f in range(F)], 1).astype(np.float32)
    X += rng.normal(size=X.shape).astype(np.float32) * 0.3
    field = gf.build_field(X, Y, np.arange(Y.shape[0]), np.repeat(np.arange(3), 8),
                           [[0], [1], [2]], cards, gf.FieldSpec(n_atoms=16),
                           log=lambda *_: None)
    ci = gf.coupling_index(cards, 0, 2, 2, 4)
    assert float(field.view("coupling")[ci]) == 0.0, \
        "an unseen pair must back off to the INDEPENDENCE prior, pmi exactly 0"
    gvs = [np.arange(c, dtype=float) for c in cards]
    unseen = np.array([[2., 0., 4.]])
    assert bool(gf.gap_read(field, unseen, gvs)["hole"][0]) is False, \
        "an unseen conjunction must be ANSWERED, not refused -- that is the whole point"
    gf.punch_hole(field, 0, 2, 2, 4)
    assert bool(gf.gap_read(field, unseen, gvs)["hole"][0]) is True


def test_lifetime_is_per_item_and_moves_only_lifetime_bytes():
    field, *_ = _toy()
    before = field.copy()
    rep = gf.set_lifetime(field, [5, 6], 0.5)
    acc = gf.assert_changed_bytes_within(before, field, rep["permitted_ranges"], "life")
    assert acc["bytes_changed_outside_permitted"] == 0
    assert set(acc["per_segment"]) == {"atom_life"}
    gf.assert_no_refit(before, field)
    assert float(field.view("atom_life")[5]) == 0.5
    assert float(field.view("atom_life")[0]) == 1.0


def test_decay_to_zero_evicts_and_leaves_nothing_behind():
    field, *_ = _toy()
    before = field.copy()
    ids = [1, 9]
    rep = gf.decay_items(field, ids, 0.0)
    gf.assert_changed_bytes_within(before, field, rep["permitted_ranges"], "decay")
    gf.assert_no_refit(before, field)
    for i in ids:
        for k in range(field.K):
            assert np.abs(field.atoms_of_well(k)[i]).max() == 0.0
        assert float(field.view("atom_life")[i]) == 0.0
        assert int(field.view("atom_alive")[i]) == 0


def test_the_byte_instrument_is_not_blind():
    """The negative control: the smallest possible refit must be visible."""
    field, *_ = _toy()
    before = field.copy()
    field.view("term_mu")[0] += np.float32(1e-3)
    d = gf.byte_diff(before, field)
    assert d["n_changed"] > 0
    with pytest.raises(AssertionError):
        gf.assert_no_refit(before, field)


def test_evicting_a_value_empties_its_basin():
    field, cards, gvs, _ = _toy()
    t = gf.term_index(cards, 0, 2)
    assert int(gf.gap_read(field, np.array([[2., 0., 0.]]), gvs)["term_live_atoms"][t]) > 0
    gf.evict_term(field, 0, 2)
    g = gf.gap_read(field, np.array([[2., 0., 0.]]), gvs)
    assert int(g["term_live_atoms"][t]) == 0
    assert bool(g["empty_basin"][0]) is True
    assert bool(g["refuse"][0]) is True
