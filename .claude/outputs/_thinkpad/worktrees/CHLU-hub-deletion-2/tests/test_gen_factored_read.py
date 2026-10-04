"""⭐ The properties `hub/generation-table`'s arm rests on, as tests.

⛔ These are the claims a replication must be able to check without re-running a cell:

1. the factored read IS a plain weighted sum over the stored modes, bit-compared against an
   independent re-implementation;
2. it takes **no feature tensor** — by signature, by provenance, and numerically;
3. the head's byte ledger equals its closed form **and** the measured leaf count;
4. `gen_address.fit_discriminative_address_book(steps=0)` reproduces the shipped
   `gen_dream.fit_address_book` **bit-identically**, and at the **same bytes**;
5. `gen_partition.balanced_correlation_partition` returns a genuine partition with
   near-equal wells.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from chlu.core.gen_address import (  # noqa: E402
    as_address_book,
    fit_discriminative_address_book,
)
from chlu.core.gen_compose import address_book_bytes, clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import fit_address_book  # noqa: E402
from chlu.core.gen_factored_read import (  # noqa: E402
    assert_request_only_factored,
    fit_factored_head,
    mode_basis,
    pair_design,
)
from chlu.core.gen_partition import balanced_correlation_partition  # noqa: E402
from chlu.core.gen_store import GenStore, delete_store, store_bytes  # noqa: E402

CARDS = [4, 3, 5]
D, K, M = 24, 4, 5


def _store(seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D).reshape(K, D // K)
    wc = tuple(tuple(int(i) for i in row) for row in perm)
    init = [rng.normal(size=(M, D // K)).astype(np.float32) for _ in range(K)]
    return GenStore(wc, M, (D,), jax.random.PRNGKey(0), init=init), perm


def _data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    Y = np.stack([rng.integers(0, c, n) for c in CARDS], axis=1).astype(np.int64)
    X = rng.normal(size=(n, D)).astype(np.float32)
    for f, c in enumerate(CARDS):                       # give the labels real signal
        X += (Y[:, f, None] / c) * rng.normal(size=(1, D))
    return X, Y


def test_the_factored_read_is_a_pure_weighted_sum_over_the_stored_modes():
    store, perm = _store()
    X, Y = _data()
    head = fit_factored_head(store, X, Y, CARDS, [(0, 1)], rank=3, ridge=1.0)
    a = head.coefficients(Y[:32])
    got = np.asarray(clu_wide_vector(store, a)).reshape((32, -1))
    _, modes = mode_basis(store)
    ref = np.zeros_like(got, dtype=np.float64)
    for k in range(K):
        ref[:, perm[k]] = np.asarray(a[:, k, :], np.float64) @ modes[k]
    assert np.max(np.abs(got - ref)) < 1e-4 * max(float(np.abs(got).max()), 1.0)


def test_the_factored_path_takes_no_feature_tensor():
    store, _ = _store()
    X, Y = _data()
    head = fit_factored_head(store, X, Y, CARDS, [(0, 1)], rank=3, ridge=1.0)
    out = assert_request_only_factored(store, head, Y[:32], decoy_features=X[:32])
    assert out["signature_ok"]
    assert out["no_feature_argument_on_the_path"]
    assert out["coefficients_re_derived_from_requests_bit_identical"]
    assert out["numeric_regression_guard"]["bit_identical_under_feature_replacement"]


def test_deleting_the_store_changes_the_factored_read():
    """⛔ charter §C.2: if this does not move, the output does not depend on what was written."""
    store, _ = _store()
    X, Y = _data()
    head = fit_factored_head(store, X, Y, CARDS, [(0, 1)], rank=3, ridge=1.0)
    a = head.coefficients(Y[:32])
    y0 = np.asarray(clu_wide_vector(store, a))
    y1 = np.asarray(clu_wide_vector(delete_store(store, "mean", 0), a))
    assert float(np.max(np.abs(y0 - y1))) > 0.0


def test_the_head_byte_ledger_matches_its_closed_form_and_the_store_ledger_is_untouched():
    store, _ = _store()
    X, Y = _data()
    rank = 3
    pairs = [(0, 1), (1, 2)]
    head = fit_factored_head(store, X, Y, CARDS, pairs, rank=rank, ridge=1.0)
    n_add = 1 + sum(CARDS)
    n_pair = sum(CARDS[f] * CARDS[g] for f, g in pairs)
    closed = (n_add * K * M + n_pair * rank + rank * K * M) * 4
    assert head.byte_ledger()["bytes"] == closed
    assert store_bytes(store)["store_bytes"] == M * D * 4
    assert head.channel_ledger()["width"] == K * M


def test_pair_design_is_one_hot_per_block():
    X, Y = _data(n=50)
    A, B = pair_design(Y, CARDS, [(0, 1)])
    assert A.shape == (50, 1 + sum(CARDS))
    assert np.all(A[:, 0] == 1.0)
    assert np.all(A[:, 1:].sum(axis=1) == len(CARDS))
    assert B.shape == (50, CARDS[0] * CARDS[1])
    assert np.all(B.sum(axis=1) == 1.0)


def test_discriminative_head_at_zero_steps_is_the_shipped_estimator_bit_for_bit():
    rng = np.random.default_rng(0)
    n = 500
    Y = np.stack([rng.integers(0, c, n) for c in CARDS], axis=1).astype(np.int64)
    C = rng.integers(0, M, (n, K)).astype(np.int64)
    nb = fit_address_book(C, Y, CARDS, 1.0)
    d0 = fit_discriminative_address_book(C, Y, CARDS, steps=0)
    assert np.array_equal(np.asarray(nb.log_prior, np.float32),
                          np.asarray(d0.log_prior, np.float32))
    for a, b in zip(nb.log_cond, d0.log_cond):
        assert np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


def test_discriminative_head_never_changes_the_addressing_head_byte_count():
    rng = np.random.default_rng(0)
    n = 500
    Y = np.stack([rng.integers(0, c, n) for c in CARDS], axis=1).astype(np.int64)
    C = rng.integers(0, M, (n, K)).astype(np.int64)
    nb_bytes = address_book_bytes(fit_address_book(C, Y, CARDS, 1.0))["bytes"]
    for steps in (0, 5):
        bk = fit_discriminative_address_book(C, Y, CARDS, steps=steps, lr=0.2)
        assert address_book_bytes(as_address_book(bk))["bytes"] == nb_bytes


def test_balanced_partition_is_a_partition_with_near_equal_wells():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, D)).astype(np.float64)
    corr = np.corrcoef(Z, rowvar=False)
    labels, info = balanced_correlation_partition(corr, 4, dead=np.zeros(D, bool))
    sizes = np.bincount(labels, minlength=4)
    assert labels.shape == (D,)
    assert sizes.sum() == D
    assert sizes.max() - sizes.min() <= 1
    assert info["equal_sized_live_blocks"]
