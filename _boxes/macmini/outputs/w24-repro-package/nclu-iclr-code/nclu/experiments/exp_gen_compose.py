"""exp gen compose."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
import jax
import matplotlib

matplotlib.use("Agg")
import numpy as np
from nclu.config import NCLUConfig
from nclu.core.gen_dream import assert_dream_decoder, load_winner_hashes
from nclu.utils.checkpoints import load_model


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


def _ms(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    return {
        "mean": float(v.mean()) if v.size else float("nan"),
        "std": float(v.std()) if v.size else float("nan"),
        "n": int(v.size),
        "per_seed": [float(x) for x in v],
    }


def resolve_decoder(config: NCLUConfig, bdir: Path, seed: int, log: Callable = print):
    cfg = config.gen_compose
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    source = str(cfg.decoder_source)
    if source == "winner":
        from nclu.paths import weights_dir

        root = Path(cfg.winner_root).expanduser() if cfg.winner_root else weights_dir() / "decoder"
        hashes = load_winner_hashes(root / "winner_hashes.json")
        rec = hashes[f"seed_{int(seed)}"]
        path = root / cfg.winner_decoder_pattern.format(seed=int(seed))
        dec = load_model(path)
        info = assert_dream_decoder(
            dec, rec["param_hash"], rec["recipe"]["name"], cfg.expected_decoder_recipe
        )
        info["decoder_path"] = str(path)
        info["decoder_source"] = "winner"
        enc_expect = rec.get("encoder_param_hash_it_was_fit_against")
        if enc_expect and enc_expect != bman["encoder_param_hash"]:
            raise AssertionError(
                f"the decoder-lift winner for this seed was fitted against encoder {enc_expect}, but the bridge here is {bman['encoder_param_hash']} — ⛔ these are different feature spaces"
            )
    elif source == "bridge":
        recipe = (bman.get("training") or {}).get("dec_recipe") or {}
        name = recipe.get("name") if isinstance(recipe, dict) else recipe
        if name is None:
            raise AssertionError(
                f"{bdir}/bridge_manifest.json records no decoder recipe, so this decoder cannot be proven to be the gate winner. ⛔ Refusing: a run through the superseded 'baseline' recipe is void."
            )
        dec = load_model(bdir / "decoder.pkl")
        info = assert_dream_decoder(
            dec, bman["decoder_param_hash"], name, cfg.expected_decoder_recipe
        )
        info["decoder_path"] = str(bdir / "decoder.pkl")
        info["decoder_source"] = "bridge"
    else:
        raise ValueError(f"decoder_source must be 'winner' or 'bridge', got {source!r}")
    info["param_count"] = int(sum((np.asarray(x).size for x in jax.tree_util.tree_leaves(dec))))
    log(
        f"  ⭐ decoder: {info['decoder_source']} recipe={info['decoder_recipe']} hash={info['decoder_param_hash'][:16]}… (hash + recipe asserted)"
    )
    return (dec, info, bman)
