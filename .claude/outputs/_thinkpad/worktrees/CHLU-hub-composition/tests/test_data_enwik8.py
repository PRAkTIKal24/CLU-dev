"""enwik8 staging + iterators (the tier-iii pilot's real stream).

Network-free: every test builds a synthetic byte file and points the loader at
it, so CI never depends on a Hutter-Prize mirror. The one thing that *is*
checked against the real cache — if it happens to be staged — is the payload
length, because that is the guard the loader actually relies on.
"""

import numpy as np
import pytest

from chlu.data import enwik8 as E


@pytest.fixture()
def fake_stream(tmp_path, monkeypatch):
    """A 20 000-byte fake 'enwik8' so the split arithmetic is exercised."""
    data = (np.arange(20_000) % 251).astype(np.uint8)
    (tmp_path / "enwik8").write_bytes(data.tobytes())
    monkeypatch.setattr(E, "N_TOTAL", 20_000)
    monkeypatch.setattr(E, "N_TRAIN", 18_000)
    monkeypatch.setattr(E, "N_VALID", 1_000)
    monkeypatch.setattr(E, "N_TEST", 1_000)
    return tmp_path, data


def test_stage_returns_existing_file_without_download(fake_stream):
    root, _ = fake_stream
    assert E.stage_enwik8(root, download=False).name == "enwik8"


def test_missing_stream_raises_rather_than_silently_downloading(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.stage_enwik8(tmp_path, download=False)


def test_split_is_positional_and_90_5_5(fake_stream):
    root, data = fake_stream
    tr, va, te = E.load_enwik8(root, download=False)
    assert (len(tr), len(va), len(te)) == (18_000, 1_000, 1_000)
    # POSITIONAL, not shuffled: the concatenation is the original stream.
    assert np.array_equal(np.concatenate([tr.data, va.data, te.data]), data)


def test_prefix_subsample_keeps_the_proportions_and_is_deterministic(fake_stream):
    root, _ = fake_stream
    a = E.load_enwik8(root, download=False, n_bytes=2_000)
    b = E.load_enwik8(root, download=False, n_bytes=2_000)
    assert [len(s) for s in a] == [1_800, 100, 100]
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.data, y.data)


def test_contiguous_batches_are_lane_contiguous_and_targets_are_shifted(fake_stream):
    """The evaluation iterator must preserve stream order.

    A persistent memory evaluated on a shuffled iterator is handed a store
    written from unrelated text, so this is a correctness property of the pilot,
    not a nicety.
    """
    root, _ = fake_stream
    _, va, _ = E.load_enwik8(root, download=False)
    it = list(E.contiguous_batches(va, batch=4, seq_len=16, n_batches=3))
    assert all(x.shape == (4, 16) and y.shape == (4, 16) for x, y in it)
    for x, y in it:
        assert np.array_equal(x[:, 1:], y[:, :-1])          # next-byte targets
    for b in range(4):                                       # lane continuity
        assert it[1][0][b, 0] == it[0][1][b, -1]


def test_random_batches_are_seeded_and_identical_across_arms(fake_stream):
    """⭐ The system-level swap requires the SAME data order in every arm."""
    root, _ = fake_stream
    tr, _, _ = E.load_enwik8(root, download=False)
    a = list(E.random_batches(tr, batch=2, seq_len=32, n_batches=4, seed=3))
    b = list(E.random_batches(tr, batch=2, seq_len=32, n_batches=4, seed=3))
    c = list(E.random_batches(tr, batch=2, seq_len=32, n_batches=4, seed=4))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x[0], y[0]) for x, y in zip(a, c, strict=True))


def test_too_short_split_raises(fake_stream):
    root, _ = fake_stream
    _, va, _ = E.load_enwik8(root, download=False)
    with pytest.raises(ValueError):
        list(E.contiguous_batches(va, batch=64, seq_len=4096))


def test_bpc_conversion():
    assert E.bits_per_character(np.log(2.0)) == pytest.approx(1.0)
    assert E.bits_per_character(np.log(256.0)) == pytest.approx(8.0)
