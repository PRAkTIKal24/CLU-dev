"""weights."""

from __future__ import annotations
import json
from pathlib import Path
import jax
import equinox as eqx
from nclu.paths import weights_dir


def _index() -> dict:
    with open(weights_dir() / "weights.json", encoding="utf-8") as fh:
        return json.load(fh)["files"]


def _classes():
    from nclu.core.gen_bridge import FactorProbe, GenDecoder, GenEncoder
    from nclu.eval.gen_classifier import ExternalFactorCNN

    return {
        "GenDecoder": GenDecoder,
        "GenEncoder": GenEncoder,
        "FactorProbe": FactorProbe,
        "ExternalFactorCNN": ExternalFactorCNN,
    }


def load_eqx(path) -> tuple:
    """Rebuild the network stored at ``path`` (a file under weights/); -> (model, param_hash)."""
    from nclu.core.gen_bridge import param_hash

    path = Path(path).resolve()
    rel = str(path.relative_to(weights_dir().resolve()))
    entry = _index()[rel]
    hp = {k: tuple(v) if k.endswith("shape") else v for k, v in entry["hyperparameters"].items()}
    model = eqx.tree_deserialise_leaves(
        path, _classes()[entry["cls"]](**hp, key=jax.random.PRNGKey(0))
    )
    got = param_hash(model)
    if got != entry["param_hash"]:
        raise AssertionError(
            "%s: parameter hash %s does not match the record %s"
            % (rel, got[:16], entry["param_hash"][:16])
        )
    return (model, got)


def load_decoder(seed: int):
    """The frozen decoder; NCLU_DECODER_DIR selects another decoder directory in the same layout
    (an encoder-loop arm's own refit decoder, written by nclu.experiments.encoder_loop)."""
    import os

    alt = os.environ.get("NCLU_DECODER_DIR")
    if alt:
        from nclu.core.gen_bridge import param_hash
        from nclu.utils.checkpoints import load_model

        d = Path(alt)
        model = load_model(d / ("decoder_v1_s%d.pkl" % int(seed)))
        with open(d / "winner_hashes.json", encoding="utf-8") as fh:
            want = json.load(fh)["seed_%d" % int(seed)]["param_hash"]
        got = param_hash(model)
        if got != want:
            raise AssertionError("decoder in %s: hash %s != %s" % (d, got[:16], want[:16]))
        return (model, got)
    return load_eqx(weights_dir() / "decoder" / ("decoder_v1_s%d.eqx" % int(seed)))


def load_classifier(seed: int):
    return load_eqx(
        weights_dir() / "classifier" / ("seed_%d" % int(seed)) / "external_classifier.eqx"
    )
