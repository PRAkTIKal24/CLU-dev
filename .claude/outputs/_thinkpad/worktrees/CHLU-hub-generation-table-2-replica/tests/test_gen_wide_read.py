"""⭐ The one property ``gen_wide_read`` must have: on a UNIFORM store it IS the shipped read."""
import jax
import numpy as np
import pytest

from chlu.core.gen_compose import clu_wide_vector
from chlu.core.gen_store import GenStore, store_bytes
from chlu.core.gen_wide_read import is_uniform, stack_coefficients, wide_vector


def _store(sizes, feat_dim, Ms, seed=0):
    wc, s = [], 0
    for d in sizes:
        wc.append(tuple(range(s, s + d)))
        s += d
    rng = np.random.default_rng(seed)
    init = [rng.normal(size=(int(M), int(d))).astype(np.float32) for M, d in zip(Ms, sizes)]
    return GenStore(tuple(wc), int(max(Ms)), (feat_dim,), jax.random.PRNGKey(0), init=init)


def test_uniform_is_bit_identical_to_the_shipped_read():
    st = _store([4, 4, 4], 12, [3, 3, 3])
    rng = np.random.default_rng(1)
    co = [rng.normal(size=(7, 3)).astype(np.float32) for _ in range(3)]
    mine = wide_vector(st, co)
    shipped = np.asarray(clu_wide_vector(st, stack_coefficients(co)))
    assert mine.dtype == shipped.dtype
    assert np.array_equal(mine, shipped), "the uniform path must BE the shipped read"


def test_ragged_is_refused_by_the_stacker_and_handled_by_the_reader():
    st = _store([4, 4, 4], 12, [3, 2, 5])
    rng = np.random.default_rng(2)
    co = [rng.normal(size=(5, M)).astype(np.float32) for M in (3, 2, 5)]
    assert not is_uniform(co)
    with pytest.raises(ValueError):
        stack_coefficients(co)                      # ⛔ never silently padded
    out = wide_vector(st, co)
    assert out.shape == (5, 12)
    # the ragged path is the same scatter, checked well by well
    for w, m, a in zip(st.well_channels, st.modes, co):
        assert np.allclose(out[:, list(w)], a @ np.asarray(m), atol=0, rtol=0)


def test_store_bytes_prices_a_ragged_allocation_honestly():
    st = _store([4, 4, 4], 12, [3, 2, 5])
    sb = store_bytes(st)
    assert sb["store_bytes"] == (3 + 2 + 5) * 4 * 4
    assert sb["uniform_allocation"] is False and sb["n_modes"] is None


def test_no_feature_argument_on_the_read():
    import inspect
    assert list(inspect.signature(wide_vector).parameters) == ["store", "coeffs"]
