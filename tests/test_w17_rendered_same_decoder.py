"""The same-decoder guard of w17 rendered-compositions, exercised without the staged weights.

The real decoder is not on this machine, so the guard is driven with a small equinox
stand-in whose hash is injected as the winner manifest's own seed-0 hash.  What is tested
is the GUARD: it must pass one decoder object with the gated hash, and refuse a wrong
hash, a swapped object, mixed objects, mixed seeds and an empty stamp list.
"""
from pathlib import Path

import numpy as np
import pytest

from w17_rendered_compositions import selection as S
from w17_rendered_compositions.same_decoder import (DecoderMismatch, OneDecoder,
                                                    assert_same_decoder, expected_hash)

WINNER = S.EVIDENCE / S.WINNER_HASHES
needs_evidence = pytest.mark.skipif(not WINNER.exists(), reason="evidence base not mounted")


def _toy_decoder():
    import equinox as eqx
    import jax

    class Toy(eqx.Module):
        lin: eqx.nn.Linear
        out_shape: tuple = eqx.field(static=True, default=(3, 4, 4))

        def __call__(self, z):
            return jax.nn.sigmoid(self.lin(z.reshape(-1))).reshape(self.out_shape)

    return Toy(eqx.nn.Linear(128 * 4 * 4, 3 * 4 * 4, key=jax.random.PRNGKey(0)))


def _hash_as_gated(obj_ok):
    exp = expected_hash(WINNER, 0)
    return lambda m: exp if m is obj_ok else "0" * 64


@needs_evidence
def test_one_decoder_passes_and_stamps_every_arm():
    dec = _toy_decoder()
    od = OneDecoder(dec, external=None, seed=0, winner_hashes=WINNER,
                    hash_fn=_hash_as_gated(dec))
    stamps = []
    for arm in ("cov_tensor", "mlp__as_trained", "table"):
        imgs, st = od.render(np.zeros((2, 2048), np.float32), arm=arm)
        assert imgs.shape == (2, 3, 4, 4)
        stamps += st
    rep = assert_same_decoder(stamps, WINNER, 0)
    assert rep["ASSERTED"] and rep["n_decoder_objects"] == 1 and rep["n_panels"] == 6
    assert rep["decoder_param_hash"] == expected_hash(WINNER, 0)


@needs_evidence
def test_wrong_hash_is_refused_at_load():
    dec = _toy_decoder()
    with pytest.raises(DecoderMismatch):
        OneDecoder(dec, None, 0, WINNER, hash_fn=lambda m: "f" * 64)


@needs_evidence
def test_swapped_decoder_is_refused_at_render():
    dec = _toy_decoder()
    od = OneDecoder(dec, None, 0, WINNER, hash_fn=_hash_as_gated(dec))
    od.decoder = _toy_decoder()
    with pytest.raises(DecoderMismatch):
        od.render(np.zeros((1, 2048), np.float32), arm="mlp__as_trained")


@needs_evidence
def test_mixed_objects_hashes_seeds_and_empty_are_refused():
    exp = expected_hash(WINNER, 0)
    good = {"arm": "cov_tensor", "decoder_param_hash": exp, "decoder_object_id": 1, "seed": 0}
    with pytest.raises(DecoderMismatch):
        assert_same_decoder([good, dict(good, decoder_object_id=2)], WINNER, 0)
    with pytest.raises(DecoderMismatch):
        assert_same_decoder([good, dict(good, decoder_param_hash="0" * 64)], WINNER, 0)
    with pytest.raises(DecoderMismatch):
        assert_same_decoder([good, dict(good, seed=1)], WINNER, 0)
    with pytest.raises(DecoderMismatch):
        assert_same_decoder([], WINNER, 0)


@needs_evidence
def test_selection_is_fixed_by_the_scored_counts():
    sel = S.select()
    s1 = sel["set1_clu_composes_rivals_fail"]
    for cid, counts in s1["counts_n_valid_of_n"].items():
        assert counts["cov_tensor"][0] == counts["cov_tensor"][1]
        assert all(counts[a][0] == 0 for a in S.OTHERS)
    s3 = sel["set3_all_arms_fail"]
    for cid, counts in s3["counts_n_valid_of_n"].items():
        assert counts["cov_tensor"][0] < counts["cov_tensor"][1]
        assert all(counts[a][0] == 0 for a in S.OTHERS)
    assert s1["corners"] and s3["corners"]
