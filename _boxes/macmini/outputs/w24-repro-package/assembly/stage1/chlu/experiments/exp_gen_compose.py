"""Experiment: **the B path, through the READ** — ``request → code → decode → probe``.

⭐ DIAL DECLARATION — *isolation*: does composing a request through the store's
READ produce the requested combination?

- **Laundering control:** ⭐ **two**, both full arms on every number —
  ``store_removed`` (the identical read path with the store's content ablated) and
  ``proto`` (a **no-store** additive per-factor prototype composition, R8's
  declared request embedding).
- **Falsifies:** the read path does not beat ``store_removed`` on ``shape`` beyond
  seed spread ⇒ composition is not the store's.  ⭐ A legitimate outcome; it
  returns to Head + Advisor, ⛔ not to tuning.
- **Does NOT falsify:** absolute numbers below the decoder ceiling, or losing to an
  unbounded-channel arithmetic baseline.  ⛔ Every row is read against **both**
  ceilings (R23.5), never absolutely.

⭐ **What changed and why (R26 ruling 3).**  Conditional dreaming is **dropped** as
the composition mechanism — the falsifier fired and the coupling-off control made
it unambiguous (*the chain samples a mode, the clamp does not steer*;
steering-attributable ``+0.0015 ± 0.0204``), and a cell-native request already
specifies 4.42 of 5 wells, so there is nothing to relax.  Composition is the
**k-particle launch read** over the requested modes, then decode.  ⛔ No landscape,
no rollout, no clamp, no MALA/ULD — and the inter-well coupling field is neither
fitted nor deployed, so the deployed store is **524,288 B**, ⛔ **not**
``gen-baselines``' 688,128 B (that target was correct only while dreaming was on
the critical path).

**Metrics (R20/R23, ⛔ none invented here).**  The gate statistic is **balanced
accuracy** of the compositional term *within the corner*: positives = the arm's
renders of the 72 held-out (B) requests, negatives = its renders of the
**matched-seen** requests (same colour corner, a seen shape).  Chance = 0.5.
AUROC is a **mandatory companion and never the gate**.  ``shape`` is reported
**separately from the aggregate, always** — four of six Shapes3D factors sit at
≈1.0 and carry no compositional information.  ⚠ Every comparison metric passes a
**pre-registered invariance check** (same features, different partition, same
value) **before any arm is scored** (R26 standing rule, third strike).

⛔ **THE DECODER TRAP, CHECKED FIRST.**  ``GenBridgeConfig.dec_recipe`` still
defaults to ``"baseline"`` — the decoder that FAILED the gate (0.7191, 0/3 seeds).
The winner is ``v1_capacity_schedule`` (0.9414, 3/3).  A run through the baseline
looks entirely normal in the logs and is **void**, so this module loads
``gen-decoder-lift``'s winner and asserts **both** its parameter hash and its
recipe name before recording a number.

⛔ **No kill-rule verdict is rendered here** — that is ``gen-cell-run``'s, against
its own filed prereg.

Runnable::

    chlu exp-gen-compose [--dataset shapes3d] [--pretext recon] [--quick]
    python -m chlu.experiments.exp_gen_compose \\
        --dataset shapes3d --pretext recon --stage compose --seed 0 --out DIR
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
import jax
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from ..config import CHLUConfig
from ..core.gen_dream import assert_dream_decoder, load_winner_hashes
from ..utils.checkpoints import load_model
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


def resolve_decoder(config: CHLUConfig, bdir: Path, seed: int, log: Callable = print):
    """Load THE shared frozen decoder for this seed and **prove** it is the winner.

    ⛔ ``decoder_source="bridge"`` is refused unless the bridge manifest itself
    records the winning recipe: Montero's bridge decoder is the **superseded**
    baseline and a run through it is void while looking entirely normal.
    """
    cfg = config.gen_compose
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    source = str(cfg.decoder_source)
    if source == "winner":
        from ..paths import weights_dir
        root = Path(cfg.winner_root).expanduser() if cfg.winner_root else (weights_dir() / "decoder")
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
                "the decoder-lift winner for this seed was fitted against encoder "
                f"{enc_expect}, but the bridge here is {bman['encoder_param_hash']} "
                "— ⛔ these are different feature spaces"
            )
    elif source == "bridge":
        recipe = (bman.get("training") or {}).get("dec_recipe") or {}
        name = recipe.get("name") if isinstance(recipe, dict) else recipe
        if name is None:
            raise AssertionError(
                f"{bdir}/bridge_manifest.json records no decoder recipe, so this "
                "decoder cannot be proven to be the gate winner. ⛔ Refusing: a run "
                "through the superseded 'baseline' recipe is void."
            )
        dec = load_model(bdir / "decoder.pkl")
        info = assert_dream_decoder(
            dec, bman["decoder_param_hash"], name, cfg.expected_decoder_recipe
        )
        info["decoder_path"] = str(bdir / "decoder.pkl")
        info["decoder_source"] = "bridge"
    else:
        raise ValueError(f"decoder_source must be 'winner' or 'bridge', got {source!r}")
    info["param_count"] = int(
        sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(dec))
    )
    log(
        f"  ⭐ decoder: {info['decoder_source']} recipe={info['decoder_recipe']} "
        f"hash={info['decoder_param_hash'][:16]}… (hash + recipe asserted)"
    )
    return dec, info, bman


