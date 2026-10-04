"""THE SAME FROZEN DECODER FOR EVERY ARM -- asserted, not assumed.

Every panel in every figure is rendered by `OneDecoder.render`.  It holds exactly one
decoder object and exactly one external classifier, both hash-checked on load against
the manifests that gated them, and it stamps every panel it renders with the decoder's
`param_hash` and the Python identity of the object that rendered it.  Before a figure is
written, `assert_same_decoder(panels)` refuses unless every panel carries the SAME hash,
the SAME object identity, and a hash equal to the evidence base's own copy of
`winner_hashes.json` for the rendered seed.

The judge's verdict on a panel is computed by `OneDecoder.verdict`, which is the scorer's
own path (`exp_gen_factor_wells.judge_valid` on the external classifier's read-back of
THIS render) -- and `positive_control` proves, on real features, that this render path
returns the identical tuples `exp_gen_factor_wells.Judge.read_back` returns.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Optional
import numpy as np
class DecoderMismatch(AssertionError):
    """Raised when any panel was not rendered by the one hash-asserted decoder."""


def expected_hash(winner_hashes: Path, seed: int) -> str:
    wh = json.loads(Path(winner_hashes).read_text(encoding="utf-8"))
    return str(wh["decoder"]["files"][str(int(seed))]["param_hash"])


class OneDecoder:
    """One decoder + one classifier, loaded once, hash-asserted, used for every arm.

    `hash_fn` defaults to `chlu.core.gen_bridge.param_hash` (the hash the winner manifest
    was written with); tests inject a stand-in so the guard can be exercised without the
    staged weights.
    """

    def __init__(self, decoder, external, seed: int, winner_hashes: Path,
                 hash_fn: Optional[Callable] = None, feat_shape=(128, 4, 4)):
        if hash_fn is None:
            from chlu.core.gen_bridge import param_hash as hash_fn
        self._hash_fn = hash_fn
        self.decoder = decoder
        self.external = external
        self.seed = int(seed)
        self.feat_shape = tuple(feat_shape)
        self.expected = expected_hash(winner_hashes, seed)
        self.hash = str(hash_fn(decoder))
        if self.hash != self.expected:
            raise DecoderMismatch(
                f"decoder param_hash {self.hash[:16]} != winner_hashes.json seed_{seed} "
                f"{self.expected[:16]} -- this is not the decoder the T7 leg was judged with; "
                "no panel is rendered")
        self.identity = id(decoder)
        self.n_rendered = 0

    @classmethod
    def from_judge(cls, judge, seed: int, winner_hashes: Path) -> "OneDecoder":
        """Wrap the scorer's own `exp_gen_factor_wells.Judge` (which has already
        asserted the decoder against CHLU_GEN_DECLIFT_DIR/winner_hashes.json and the
        classifier against its manifest).  The evidence copy is asserted again here."""
        od = cls(judge.decoder, judge.external, seed, winner_hashes)
        if judge.info["decoder_param_hash"] != od.hash:
            raise DecoderMismatch("the Judge's recorded decoder hash differs from the "
                                  "object it holds")
        od.classifier_hash = judge.info["classifier_param_hash"]
        return od

    # -- the render -------------------------------------------------------------------
    def _check_live(self):
        # the object may not be swapped after construction
        if id(self.decoder) != self.identity or str(self._hash_fn(self.decoder)) != self.hash:
            raise DecoderMismatch("the decoder object changed after it was asserted")

    def render(self, feats: np.ndarray, arm: str, batch: int = 256) -> tuple:
        """(N, D) features -> (N, C, H, W) images + one stamp per image."""
        from chlu.eval.gen_probe import decode_features
        self._check_live()
        Z = np.asarray(feats, np.float32).reshape((-1,) + self.feat_shape)
        imgs = decode_features(self.decoder, Z, batch) if Z.shape[0] else np.zeros((0,))
        self.n_rendered += int(Z.shape[0])
        stamp = {"arm": arm, "decoder_param_hash": self.hash,
                 "decoder_object_id": self.identity, "seed": self.seed}
        return imgs, [dict(stamp) for _ in range(Z.shape[0])]

    def read_back(self, imgs: np.ndarray, batch: int = 256) -> np.ndarray:
        from chlu.eval.gen_probe import predict_factors
        return np.asarray(predict_factors(self.external, imgs, batch), np.int64)

    def verdict(self, values: np.ndarray, accepted: np.ndarray, imgs: np.ndarray,
                grid) -> tuple:
        """The scorer's verdict on THESE rendered images: judge_valid & accept."""
        from chlu.experiments import exp_gen_factor_wells as fw
        tup = self.read_back(imgs)
        return fw.judge_valid(values, tup, grid) & np.asarray(accepted, bool), tup

    def positive_control(self, judge, feats: np.ndarray) -> dict:
        """This render path must return the Judge's read-back EXACTLY, element for element."""
        a = judge.read_back(feats, 256)
        imgs, _ = self.render(feats, arm="__positive_control__")
        b = self.read_back(imgs)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise DecoderMismatch(
                "render->classify disagrees with exp_gen_factor_wells.Judge.read_back on "
                f"{int((a != b).sum()) if a.shape == b.shape else -1} elements -- the "
                "figure would not show what was scored")
        return {"check": "OneDecoder.render+read_back == Judge.read_back",
                "n_requests": int(a.shape[0]), "n_elements": int(a.size),
                "identical": True}


def assert_same_decoder(stamps: list, winner_hashes: Path, seed: int) -> dict:
    """Refuse unless every panel was rendered by the one asserted decoder object."""
    if not stamps:
        raise DecoderMismatch("no panel stamps -- nothing to assert")
    exp = expected_hash(winner_hashes, seed)
    hashes = {s["decoder_param_hash"] for s in stamps}
    ids = {s["decoder_object_id"] for s in stamps}
    seeds = {s["seed"] for s in stamps}
    if hashes != {exp}:
        raise DecoderMismatch(f"panels carry decoder hashes {sorted(h[:16] for h in hashes)}; "
                              f"the only admissible one is {exp[:16]}")
    if len(ids) != 1:
        raise DecoderMismatch(f"panels were rendered by {len(ids)} different decoder objects")
    if seeds != {int(seed)}:
        raise DecoderMismatch(f"panels mix seeds {sorted(seeds)}")
    arms = sorted({s["arm"] for s in stamps})
    return {"ASSERTED": True, "n_panels": len(stamps), "arms": arms,
            "decoder_param_hash": exp, "n_decoder_objects": 1, "seed": int(seed),
            "against": str(winner_hashes)}


