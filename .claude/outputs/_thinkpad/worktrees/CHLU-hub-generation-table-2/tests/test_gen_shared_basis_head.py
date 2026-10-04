"""The properties the shared-basis head must have before any number it produces counts."""
import inspect

import numpy as np
import pytest

from chlu.core.gen_dict_fit import fit_factored_head_v2, well_dictionary
from chlu.core.gen_shared_basis_head import (fit_shared_basis_head,
                                             shared_basis_head_bytes)


def _fixture(seed=0, n=400, K=4, d=8, M=5):
    rng = np.random.default_rng(seed)
    D = K * d
    cards = [3, 2, 4, 2, 3, 2]
    labels = np.stack([rng.integers(0, c, n) for c in cards], 1)
    # features that genuinely carry an interaction, so E has something to fit
    X = rng.normal(size=(n, D)) * 0.1
    X += labels[:, [0]] * 0.5
    X += (labels[:, 0] * labels[:, 2])[:, None] * 0.2
    perm = np.stack([np.arange(k * d, (k + 1) * d) for k in range(K)])
    dicts = [well_dictionary(X[:, perm[k]], M, "kmeans", seed * 100 + k)
             for k in range(K)]
    return X, labels, cards, perm, dicts, D, K, M


def test_byte_ledger_matches_its_closed_form_and_does_not_scale_with_A():
    X, labels, cards, perm, dicts, D, K, M = _fixture()
    h = fit_shared_basis_head(dicts, perm, X, labels, cards, [(0, 2)], reduced=True)
    led = h.byte_ledger()
    n_add = 1 + sum(c - 1 for c in cards)
    n_pair = cards[0] * cards[2]
    assert led["bytes"] == shared_basis_head_bytes(n_add, n_pair, K * M)
    assert led["bytes"] == (n_add * K * M + n_pair * n_add) * 4
    # the interaction term is independent of A: doubling the atoms adds only n_add*A
    big = shared_basis_head_bytes(n_add, n_pair, 2 * K * M)
    assert big - led["bytes"] == n_add * K * M * 4


def test_the_read_takes_no_feature_argument():
    """Charter B.3, enforced by SIGNATURE rather than by intention."""
    from chlu.core.gen_shared_basis_head import SharedBasisHead
    assert list(inspect.signature(SharedBasisHead.coefficients).parameters) == [
        "self", "requests"]


def test_coefficients_are_a_function_of_the_request_alone():
    X, labels, cards, perm, dicts, D, K, M = _fixture()
    h = fit_shared_basis_head(dicts, perm, X, labels, cards, [(0, 2)])
    r = labels[:20]
    a1 = h.coefficients(r)
    a2 = h.coefficients(r.copy())
    for u, v in zip(a1, a2):
        assert np.array_equal(u, v)
    assert [a.shape[1] for a in a1] == list(h.modes_per_well)


def test_no_pairs_reduces_to_the_additive_head_exactly():
    X, labels, cards, perm, dicts, D, K, M = _fixture()
    h = fit_shared_basis_head(dicts, perm, X, labels, cards, [], ridge_add=10.0)
    g = fit_factored_head_v2(dicts, perm, X, labels, cards, [], 0, ridge_add=10.0,
                             reduced=True, metric_correct=True)
    for u, v in zip(h.coefficients(labels[:16]), g.coefficients(labels[:16])):
        assert np.allclose(u, v, atol=1e-5)


def test_the_interaction_actually_changes_the_output():
    X, labels, cards, perm, dicts, D, K, M = _fixture()
    h0 = fit_shared_basis_head(dicts, perm, X, labels, cards, [])
    h1 = fit_shared_basis_head(dicts, perm, X, labels, cards, [(0, 2)])
    a0 = np.concatenate(h0.coefficients(labels[:64]), 1)
    a1 = np.concatenate(h1.coefficients(labels[:64]), 1)
    assert not np.allclose(a0, a1), "a fitted pair block that changes nothing is a bug"
    assert 0.0 <= h1.fit_info["interaction_spectral_mass_kept"] <= 1.0 + 1e-9


def test_a_stronger_pair_ridge_shrinks_the_interaction():
    X, labels, cards, perm, dicts, D, K, M = _fixture()
    weak = fit_shared_basis_head(dicts, perm, X, labels, cards, [(0, 2)], ridge_pair=1.0)
    strong = fit_shared_basis_head(dicts, perm, X, labels, cards, [(0, 2)],
                                   ridge_pair=1e6)
    assert np.abs(strong.E).sum() < np.abs(weak.E).sum()


@pytest.mark.parametrize("reduced", [True, False])
def test_the_design_width_is_what_the_ledger_claims(reduced):
    X, labels, cards, perm, dicts, D, K, M = _fixture()
    h = fit_shared_basis_head(dicts, perm, X, labels, cards, [(0, 2)], reduced=reduced)
    want = (1 + sum(c - 1 for c in cards)) if reduced else (1 + sum(cards))
    assert h.byte_ledger()["n_additive_columns"] == want
    assert h.C_add.shape[0] == want
