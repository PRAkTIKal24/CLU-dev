"""hub/store-content — the FROZEN instrument, loaded and hash-asserted.

⛔ Nothing here fits anything.  Encoder, decoder, probe and external classifier are
loaded from the pinned directories and their parameter hashes are checked against
their own manifests, exactly as ``exp_gen_cell`` does it.  A hub that scores against
a re-fitted reader has measured nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from chlu.utils.checkpoints import load_model
from chlu.core.gen_bridge import param_hash
from chlu.eval.gen_probe import decode_features, probe_accuracy, predict_factors

CLAUDE = Path(os.environ.get(
    "CHLU_CLAUDE_ROOT", "C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude"))
BRIDGE = Path(os.environ.get(
    "CHLU_GEN_BRIDGE_ROOT", str(CLAUDE / "outputs/gen-feature-bridge/runs")))
DECLIFT = Path(os.environ.get(
    "CHLU_GEN_DECLIFT_DIR", str(CLAUDE / "outputs/gen-decoder-lift/winner")))

FACTORS = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")


def load_instrument(seed: int = 0, log=print) -> dict:
    bdir = BRIDGE / "shapes3d_recon" / f"seed_{seed}"
    bman = json.load(open(bdir / "bridge_manifest.json"))

    enc = load_model(bdir / "encoder.pkl")
    eh = param_hash(enc)
    assert eh == bman["encoder_param_hash"], "encoder != bridge manifest"

    probe = load_model(bdir / "probe.pkl")
    assert param_hash(probe) == bman["probe_param_hash"], "probe != bridge manifest"

    # THE decoder: the decoder-lift winner for this seed, hash-asserted.
    wh = json.load(open(DECLIFT / "winner_hashes.json"))
    dec = load_model(DECLIFT / f"decoder_v1_s{seed}.pkl")
    dh = param_hash(dec)
    expect = _find_hash(wh, seed)
    if expect and dh != expect:
        raise AssertionError(
            f"decoder hash {dh[:16]} != the winner manifest {str(expect)[:16]}")

    croot = CLAUDE / "outputs" / "gen-external-classifier"
    cman = json.load(open(croot / f"seed_{seed}" / "classifier_manifest.json"))
    clf = load_model(croot / f"seed_{seed}" / "external_classifier.pkl")
    ch = param_hash(clf)
    cexp = str((cman.get("classifier") or {}).get("param_hash") or "")
    if cexp and ch != cexp:
        raise AssertionError(
            f"external classifier {ch[:16]} != its own manifest {cexp[:16]}")

    log(f"  instrument seed {seed}: encoder {eh[:12]} decoder {dh[:12]} "
        f"classifier {ch[:12]}")
    return {"encoder": enc, "decoder": dec, "probe": probe, "classifier": clf,
            "bridge_dir": str(bdir), "bridge_manifest": bman,
            "hashes": {"encoder": eh, "decoder": dh, "classifier": ch,
                       "probe": param_hash(probe)},
            "classifier_manifest": cman, "decoder_winner_hashes": wh}


def _find_hash(wh, seed: int):
    """The winner-hash file's layout is not contractual; look the seed up loosely."""
    if not isinstance(wh, dict):
        return None
    for k in (f"decoder_v1_s{seed}.pkl", f"decoder_v1_s{seed}", str(seed), f"seed_{seed}"):
        v = wh.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for kk in ("param_hash", "hash", "sha256"):
                if isinstance(v.get(kk), str):
                    return v[kk]
    return None


def render(decoder, penult: np.ndarray, batch: int = 256) -> np.ndarray:
    """(n,128,4,4) -> (n,3,64,64) images through THE frozen decoder."""
    return decode_features(decoder, np.asarray(penult, np.float32), int(batch))


def judge(classifier, images: np.ndarray, batch: int = 256) -> np.ndarray:
    """(n,3,64,64) -> (n, F) factor-index tuples read back by the EXTERNAL judge.

    ⛔ ``predict_factors`` is the pin's own reader, called, not re-implemented - so
    this hub's judge is arithmetically the same judge every other arm was scored by.
    """
    return np.asarray(predict_factors(classifier, np.asarray(images, np.float32),
                                      int(batch)), np.int64)


def per_factor_accuracy(pred: np.ndarray, target: np.ndarray, names=FACTORS) -> dict:
    """⭐ The breakdown the target file requires on EVERY row.

    The aggregate is the mean over factors of the per-factor accuracy, which is what
    the bridge manifest's own ``ceiling`` block reports, so the numbers are
    comparable to it.  ``exact_match`` is the all-six-correct rate.
    """
    p = np.asarray(pred, np.int64)
    t = np.asarray(target, np.int64)
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} vs target {t.shape}")
    pf = {n: float((p[:, i] == t[:, i]).mean()) for i, n in enumerate(names)}
    return {"n": int(p.shape[0]), "aggregate": float(np.mean(list(pf.values()))),
            "exact_match": float((p == t).all(axis=1).mean()), "per_factor": pf}
