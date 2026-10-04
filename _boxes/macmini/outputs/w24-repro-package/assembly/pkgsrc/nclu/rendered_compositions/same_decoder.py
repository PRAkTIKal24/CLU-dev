"""same decoder."""

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
    def __init__(
        self,
        decoder,
        external,
        seed: int,
        winner_hashes: Path,
        hash_fn: Optional[Callable] = None,
        feat_shape=(128, 4, 4),
    ):
        if hash_fn is None:
            from nclu.core.gen_bridge import param_hash as hash_fn
        self._hash_fn = hash_fn
        self.decoder = decoder
        self.external = external
        self.seed = int(seed)
        self.feat_shape = tuple(feat_shape)
        self.expected = expected_hash(winner_hashes, seed)
        self.hash = str(hash_fn(decoder))
        if self.hash != self.expected:
            raise DecoderMismatch(
                f"decoder param_hash {self.hash[:16]} != winner_hashes.json seed_{seed} {self.expected[:16]} -- this is not the decoder the T7 leg was judged with; no panel is rendered"
            )
        self.identity = id(decoder)
        self.n_rendered = 0

    @classmethod
    def from_judge(cls, judge, seed: int, winner_hashes: Path) -> "OneDecoder":
        od = cls(judge.decoder, judge.external, seed, winner_hashes)
        if judge.info["decoder_param_hash"] != od.hash:
            raise DecoderMismatch(
                "the Judge's recorded decoder hash differs from the object it holds"
            )
        od.classifier_hash = judge.info["classifier_param_hash"]
        return od

    def _check_live(self):
        if id(self.decoder) != self.identity or str(self._hash_fn(self.decoder)) != self.hash:
            raise DecoderMismatch("the decoder object changed after it was asserted")

    def render(self, feats: np.ndarray, arm: str, batch: int = 256) -> tuple:
        """(N, D) features -> (N, C, H, W) images + one stamp per image."""
        from nclu.eval.gen_probe import decode_features

        self._check_live()
        Z = np.asarray(feats, np.float32).reshape((-1,) + self.feat_shape)
        imgs = decode_features(self.decoder, Z, batch) if Z.shape[0] else np.zeros((0,))
        self.n_rendered += int(Z.shape[0])
        stamp = {
            "arm": arm,
            "decoder_param_hash": self.hash,
            "decoder_object_id": self.identity,
            "seed": self.seed,
        }
        return (imgs, [dict(stamp) for _ in range(Z.shape[0])])

    def read_back(self, imgs: np.ndarray, batch: int = 256) -> np.ndarray:
        from nclu.eval.gen_probe import predict_factors

        return np.asarray(predict_factors(self.external, imgs, batch), np.int64)

    def verdict(self, values: np.ndarray, accepted: np.ndarray, imgs: np.ndarray, grid) -> tuple:
        from nclu.experiments import factor_wells as fw

        tup = self.read_back(imgs)
        return (fw.judge_valid(values, tup, grid) & np.asarray(accepted, bool), tup)

    def positive_control(self, judge, feats: np.ndarray) -> dict:
        """This render path must return the Judge's read-back EXACTLY, element for element."""
        a = judge.read_back(feats, 256)
        imgs, _ = self.render(feats, arm="__positive_control__")
        b = self.read_back(imgs)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise DecoderMismatch(
                f"render->classify disagrees with exp_gen_factor_wells.Judge.read_back on {(int((a != b).sum()) if a.shape == b.shape else -1)} elements -- the figure would not show what was scored"
            )
        return {
            "check": "OneDecoder.render+read_back == Judge.read_back",
            "n_requests": int(a.shape[0]),
            "n_elements": int(a.size),
            "identical": True,
        }


def assert_same_decoder(stamps: list, winner_hashes: Path, seed: int) -> dict:
    """Refuse unless every panel was rendered by the one asserted decoder object."""
    if not stamps:
        raise DecoderMismatch("no panel stamps -- nothing to assert")
    exp = expected_hash(winner_hashes, seed)
    hashes = {s["decoder_param_hash"] for s in stamps}
    ids = {s["decoder_object_id"] for s in stamps}
    seeds = {s["seed"] for s in stamps}
    if hashes != {exp}:
        raise DecoderMismatch(
            f"panels carry decoder hashes {sorted((h[:16] for h in hashes))}; the only admissible one is {exp[:16]}"
        )
    if len(ids) != 1:
        raise DecoderMismatch(f"panels were rendered by {len(ids)} different decoder objects")
    if seeds != {int(seed)}:
        raise DecoderMismatch(f"panels mix seeds {sorted(seeds)}")
    arms = sorted({s["arm"] for s in stamps})
    return {
        "ASSERTED": True,
        "n_panels": len(stamps),
        "arms": arms,
        "decoder_param_hash": exp,
        "n_decoder_objects": 1,
        "seed": int(seed),
        "against": str(winner_hashes),
    }
