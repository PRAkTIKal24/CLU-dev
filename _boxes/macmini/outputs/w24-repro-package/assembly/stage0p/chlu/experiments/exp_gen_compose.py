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

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, get_default_config  # noqa: E402
from ..core.gen_bridge import PRETEXTS, load_feature_cache, param_hash  # noqa: E402
from ..core.gen_compose import (  # noqa: E402
    ARMS,
    COMMITMENT_RULES,
    COMPOSITIONAL_FACTOR,
    address_book_bytes,
    assert_request_only,
    channel_ledger,
    clu_wide_vector,
    code_agreement,
    code_overlap,
    commit,
    commitment_channel,
    commitment_read,
    compose_codes,
    deployed_byte_ledger,
    distinct_codes,
    factor_evidence,
    factor_leverage,
    metric_invariance_check,
    posterior_diagnostics,
    read_compose,
    request_posterior,
    store_removed_read,
)
from ..core.gen_dream import (  # noqa: E402
    AddressBook,
    assert_dream_decoder,
    fit_address_book,
    load_winner_hashes,
    requests_from_labels,
)
from ..core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from ..core.gen_store import (  # noqa: E402
    address_codes,
    assert_no_bypass,
    init_store,
    load_store_plan,
    store_read,
    train_store,
    well_channels_from_labels,
)
from ..eval.gen_probe import (  # noqa: E402
    decode_features,
    discrimination_metrics,
    factor_labels,
    factor_probs,
    multiclass_balanced_accuracy,
    probe_accuracy,
)
from ..experiments.exp_gen_bridge import artifacts_tag, condition_of  # noqa: E402
from ..experiments.exp_gen_dream import resolve_grouping  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model, save_model  # noqa: E402

STAGES = ("compose", "aggregate")
#: Kept so the CSF3 launcher's other ``--stage`` values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "store", "dream", "certificate", "baselines", "cell")
#: Request sets in the order every table uses them.  ``MS`` = the matched-seen
#: negatives (R16's permanent companion row).
REQUEST_SETS = ("A", "B", "MS")


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's numbers are NEVER reported."""
    g = config.gen_compose
    g.seeds = list(g.quick_seeds)
    g.n_eval_a = g.quick_n_eval_a
    g.arms = list(g.quick_arms)
    # ⚠ see GenComposeConfig.quick_grouping_source: the per-seed re-derivation is
    # asserted bit-identical to the plan on its own seed and a smoke-sized
    # subsample cannot reproduce it. ⛔ Quick numbers are never reported.
    g.grouping_source = str(g.quick_grouping_source)
    config.gen_store.n_store_train = g.quick_n_store_train
    config.gen_store.train_steps = g.quick_store_train_steps
    return config


def _claude_root(config: CHLUConfig) -> Path:
    r = config.gen_compose.claude_root
    if r:
        return Path(r).expanduser()
    from ..paths import out_root
    return out_root()


def store_plan_path(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    """``gen-store-grouping``'s pinned plan.  ⛔ ``K``/``n_modes``/``D``/``store_bytes``
    are portable; the ``labels`` vector is **not** (R26.5) and is re-derived per seed."""
    p = config.gen_compose.store_plan
    if p:
        return Path(p).expanduser()
    return Path(__file__).resolve().parents[1] / "specs" / f"store_plan_{dataset}_{pretext}.json"


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


# ---------------------------------------------------------------------------
# ⛔ 1. The decoder guard — resolved and asserted BEFORE anything is measured
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. Scoring — ⛔ ``shape`` always separate from the aggregate
# ---------------------------------------------------------------------------


def score_set(decoder, probe, feats, labels, names, feat_shape, batch: int) -> dict:
    """Decode through THE frozen decoder, read with THE frozen probe."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(feat_shape))
    imgs = decode_features(decoder, Z, int(batch))
    out = probe_accuracy(probe, imgs, labels, names, int(batch))
    out["compositional_factor"] = COMPOSITIONAL_FACTOR
    out[f"{COMPOSITIONAL_FACTOR}_accuracy"] = out["per_factor"][COMPOSITIONAL_FACTOR]
    out["decoded_mean"] = float(imgs.mean())
    return out


def corner_discrimination(
    decoder, probe, feats_pos, feats_neg, labels_pos, labels_neg,
    shape_idx: int, n_shape_classes: int, feat_shape, batch: int,
) -> dict:
    """⭐ The gate statistic (R20/R23): DISCRIMINATION within the corner.

    ⛔ Identical in construction to ``gen-baselines``' own reader, because the two
    sets of numbers have to be comparable row by row: positives = renders of the
    held-out requests, negatives = renders of the matched-seen requests.  Balanced
    accuracy is the statistic; AUROC is the mandatory companion; ⛔ chance is 0.5
    and these are **not** comparable in level to a raw multi-class ``shape``
    accuracy.
    """
    Zp = np.asarray(feats_pos, np.float32).reshape((-1,) + tuple(feat_shape))
    Zn = np.asarray(feats_neg, np.float32).reshape((-1,) + tuple(feat_shape))
    pp = factor_probs(probe, decode_features(decoder, Zp, int(batch)), shape_idx, int(batch))
    pn = factor_probs(probe, decode_features(decoder, Zn, int(batch)), shape_idx, int(batch))
    targets = np.unique(np.asarray(labels_pos, np.int64)[:, shape_idx])
    out = {"targets_realised": [int(t) for t in targets]}
    if targets.size == 1:
        out.update(discrimination_metrics(pp, pn, int(targets[0])))
        out["_probs"] = (pp, pn, int(targets[0]))
    else:
        pred = np.concatenate([np.argmax(pp, 1), np.argmax(pn, 1)])
        true = np.concatenate(
            [np.asarray(labels_pos)[:, shape_idx], np.asarray(labels_neg)[:, shape_idx]]
        )
        mc = multiclass_balanced_accuracy(true, pred, int(n_shape_classes))
        out.update({
            "balanced_accuracy": mc["balanced_accuracy"], "auroc": float("nan"),
            "n_pos": int(pp.shape[0]), "n_neg": int(pn.shape[0]),
            "multiclass": mc, "sufficient": True,
            "note": "multi-class held-out cell: macro-recall, ⛔ not the binary in-corner read",
        })
        out["_probs"] = None
    return out


def _strip_probs(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_probs"}


# ---------------------------------------------------------------------------
# 3. The arms — ⛔ every one of them produces a FEATURE-SPACE vector per request
# ---------------------------------------------------------------------------


def arm_features(
    arm: str, store, codes: np.ndarray, posterior: np.ndarray,
    protos: FactorPrototypes, request_dicts, cfg, seed: int,
) -> np.ndarray:
    """``(N, D)`` the arm's rendering input for one request set.

    ⛔ **No arm receives the features of the images it is trying to compose.**  The
    three store arms take integer codes or a request-derived posterior; ``proto``
    takes the request dicts and composes prototypes fitted on *train* rows.
    """
    if arm == "read":
        return np.asarray(read_compose(store, codes), np.float32)
    if arm == "clu_wide":
        return np.asarray(clu_wide_vector(store, posterior), np.float32)
    if arm == "store_removed":
        return np.asarray(
            store_removed_read(store, codes, str(cfg.store_removed_delete_mode), int(seed)),
            np.float32,
        )
    if arm == "proto":
        return np.asarray(protos.embed_many(request_dicts), np.float32)
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


# ---------------------------------------------------------------------------
# 4. One seed
# ---------------------------------------------------------------------------


def run_seed(
    dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
    arms: Optional[list] = None, log: Callable = print,
) -> dict:
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    cfg = config.gen_compose
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = condition_of(config)
    bdir = bridge_dir(config, dataset, pretext, seed)
    tlog(f"seed {seed} · condition={condition} · frozen bridge {bdir}")

    # ---- ⛔ the decoder guard, FIRST -----------------------------------------
    decoder, dec_info, bman = resolve_decoder(config, bdir, seed, log=tlog)
    probe = load_model(bdir / "probe.pkl")
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest hash")
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")

    # ---- rows: A, B and R16's matched-seen companion -------------------------
    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    shape_idx = names.index(COMPOSITIONAL_FACTOR)
    rows = select_store_rows(split, config, seed)
    n_a = int(min(int(cfg.n_eval_a), rows["eval_a"].size))
    row_sets = {
        "A": rows["eval_a"][:n_a],
        "B": rows["eval_b"],
        "MS": matched_seen_rows(dataset, split, values, rows["store_train"]),
    }
    tlog(
        f"rows: store_train={rows['store_train'].size} A={row_sets['A'].size} "
        f"B={row_sets['B'].size} matched_seen={row_sets['MS'].size}"
    )

    # ---- the frozen cache ----------------------------------------------------
    crows, feats, cman = load_feature_cache(bdir, verify=bool(cfg.verify_cache_checksum))
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    tlog(f"cache {cman['n_rows']} rows (scope={cman['scope']}), D={cman['D']}, {cman['dtype']}")
    config.gen_store.encode_chunk = int(cfg.encode_chunk)
    reenc = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=tlog)

    def gather(r):
        return gather_features(dataset, config, encoder, crows, feats, cman, r, log=tlog)

    X_store = gather(rows["store_train"])
    X = {k: gather(v) for k, v in row_sets.items()}
    lab_store = factor_labels(grid, rows["store_train"])
    lab = {k: factor_labels(grid, v) for k, v in row_sets.items()}
    tlog(f"features gathered: store {X_store.shape} " +
         " ".join(f"{k} {v.shape}" for k, v in X.items()))

    # ---- the store: the PINNED plan, the per-seed grouping, train-or-load -----
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    K, D = int(plan["K"]), int(np.prod(feat_shape))
    if len(plan["labels"]) != D:
        raise AssertionError(
            f"the store plan's label vector is {len(plan['labels'])} long but D = {D}"
        )
    # ⭐ Re-uses ``gen-dream-sampler``'s resolver VERBATIM rather than a copy, so the
    # grouping procedure is bit-for-bit the one R0 and the dream cell used and the
    # numbers stay comparable.  ⚠ It reads ``gen_dream.grouping_source``, so this
    # spoke's own knob is forwarded onto it (the only field of that config touched).
    config.gen_dream.grouping_source = str(cfg.grouping_source)
    labels_vec, ginfo = resolve_grouping(
        plan, X_store, config, seed, bman["encoder_param_hash"], log=tlog
    )
    wc = well_channels_from_labels(labels_vec, K)
    tlog(f"⭐ K={K} n_modes={plan['n_modes']} pinned from the plan; grouping "
         f"{ginfo['grouping_source']}; well sizes {[len(w) for w in wc]}")

    ckpt = out_dir / f"store_s{seed}.pkl"
    if ckpt.exists() and bool(cfg.use_store_checkpoint):
        store = load_model(ckpt)
        tlog(f"store loaded from {ckpt}")
        hist = []
    else:
        key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911)
        key, ik, tk = jax.random.split(key, 3)
        store = init_store(X_store, wc, int(plan["n_modes"]), feat_shape, ik, seed=int(seed))
        tlog(f"training the modes ({config.gen_store.train_steps} steps, store-only objective) …")
        store, hist, key = train_store(store, X_store, config.gen_store, tk, log=tlog)
        if bool(cfg.use_store_checkpoint):
            save_model(store, ckpt, metadata={"seed": int(seed), "plan": dict(plan)})

    # ---- ⭐ the two ledgers: bytes (coupling OUT) and the realised channel ----
    codes_train = address_codes(store, np.asarray(X_store, np.float32).reshape(
        (X_store.shape[0], -1)))
    book = fit_address_book(codes_train, lab_store, cards, float(cfg.address_smoothing))
    byte_led = deployed_byte_ledger(store, book)
    if (cfg.expect_deployed_store_bytes is not None
            and int(byte_led["deployed_bytes"]) != int(cfg.expect_deployed_store_bytes)):
        raise AssertionError(
            f"deployed store is {byte_led['deployed_bytes']} B, expected "
            f"{cfg.expect_deployed_store_bytes} B — ⛔ the byte ledger is the one "
            "number a reader will check against gen-baselines'"
        )
    tlog(
        f"⭐ byte ledger: deployed store {byte_led['deployed_bytes']} B "
        f"(⛔ coupling {byte_led['coupling_bytes_excluded']} B EXCLUDED — dreaming is "
        f"off the critical path, R26; ⛔ do NOT inherit 688,128) + addressing head "
        f"{byte_led['addressing_head']['bytes']} B, reported beside, never inside"
    )

    # ---- ⛔ the structural guarantees, before any number ---------------------
    bypass = assert_no_bypass(store, X["A"][:64], rng_seed=int(seed))
    req = {k: requests_from_labels(lab[k], "full", 0, int(seed)) for k in REQUEST_SETS}
    post = {k: request_posterior(book, req[k]) for k in REQUEST_SETS}
    codes = {k: compose_codes(post[k]) for k in REQUEST_SETS}
    req_only = assert_request_only(store, book, req["B"], decoy_features=X["B"])
    tlog(
        f"⭐ no-bypass: pure codebook gather, {bypass['bits_per_read']:.0f} bits/read · "
        "⭐ request-only path asserted (no feature argument on ANY arm; "
        "gen_store.soft_read NOT imported)"
    )
    chan = channel_ledger(store, post["B"], proto_dim=D)
    tlog(
        f"⭐ channel: read {chan['read']['bits']:.0f} bits · clu_wide "
        f"{chan['clu_wide']['n_real_values']} weights (entropy "
        f"{chan['clu_wide']['mean_posterior_entropy_bits']:.2f} bits, dof "
        f"{chan['clu_wide']['degrees_of_freedom']}) · proto {chan['proto']['n_real_values']} "
        "floats (⛔ unbounded)"
    )

    # ---- the no-store request embedding (R8's, ⛔ fitted on TRAIN rows only) --
    protos = FactorPrototypes.fit(grid, X_store, rows["store_train"])
    req_dicts = {k: requests_from_rows(grid, v) for k, v in row_sets.items()}
    tlog(
        f"⭐ R8 prototype embedding: {protos.n_prototypes} prototypes "
        f"({protos.n_bytes} B, reported beside the store by the same convention)"
    )

    # ---- ⭐ THE TWO CEILINGS (R23.5) -----------------------------------------
    ceil = {}
    for name, Xf in (("decoder", X), ("store", {k: store_read(store, X[k], int(cfg.eval_batch))
                                                for k in REQUEST_SETS})):
        comp_a = score_set(decoder, probe, Xf["A"], lab["A"], names, feat_shape, cfg.eval_batch)
        comp_b = score_set(decoder, probe, Xf["B"], lab["B"], names, feat_shape, cfg.eval_batch)
        disc = corner_discrimination(
            decoder, probe, Xf["B"], Xf["MS"], lab["B"], lab["MS"], shape_idx,
            int(cards[shape_idx]), feat_shape, cfg.eval_batch,
        )
        ceil[name] = {"composition_A": comp_a, "composition_B": comp_b,
                      "in_corner_discrimination": disc}
        tlog(
            f"⭐ {name.upper()} ceiling: in-corner {COMPOSITIONAL_FACTOR} balacc="
            f"{disc['balanced_accuracy']:.4f} (AUROC {disc['auroc']:.4f}) · A aggregate="
            f"{comp_a['aggregate']:.4f} {COMPOSITIONAL_FACTOR}="
            f"{comp_a['per_factor'][COMPOSITIONAL_FACTOR]:.4f} · B aggregate="
            f"{comp_b['aggregate']:.4f} {COMPOSITIONAL_FACTOR}="
            f"{comp_b['per_factor'][COMPOSITIONAL_FACTOR]:.4f}"
        )

    # ---- ⚠ the metric invariance check, BEFORE any arm is scored ------------
    inv = None
    probs = ceil["decoder"]["in_corner_discrimination"].get("_probs")
    if probs is not None:
        inv = metric_invariance_check(
            *probs, neg_repeats=int(cfg.invariance_neg_repeats),
            subsample_frac=float(cfg.invariance_subsample_frac),
            seed=int(cfg.invariance_seed),
        )
        tlog(
            f"⚠ metric invariance (R26 standing rule): balacc Δ={inv['delta_balanced_accuracy']:.2e} "
            f"AUROC Δ={inv['delta_auroc']:.2e} under a re-partition, raw accuracy moves "
            f"{inv['delta_raw_accuracy']:+.4f} ⇒ "
            + ("✓ INVARIANT" if inv["invariant"] else "⛔ NOT INVARIANT — the metric is not used")
        )
        if not inv["invariant"]:
            raise AssertionError(
                "⛔ the gate statistic failed its pre-registered invariance check "
                "(same features, different partition, different value) — R26's "
                "standing rule says it is not used"
            )

    # ---- ⭐ a DIAGNOSTIC, ⛔ consumed by no arm ------------------------------
    agree = {
        k: code_agreement(codes[k]["code"], address_codes(
            store, np.asarray(X[k], np.float32).reshape((X[k].shape[0], -1))))
        for k in REQUEST_SETS
    }
    tlog(
        "⭐ request-vs-feature code agreement (diagnostic): "
        + " ".join(f"{k} exact-set={agree[k]['exact_set']:.4f}" for k in REQUEST_SETS)
    )

    manifest = {
        "schema": "chlu-gen-compose/1",
        "dataset": dataset, "pretext": pretext, "condition": condition, "seed": int(seed),
        "mechanism": (
            "R26 ruling 3 — composition is the k-particle launch READ over the "
            "requested modes, then decode. ⛔ No landscape, no rollout, no clamp."
        ),
        "bridge_dir": str(bdir),
        "decoder": dec_info,
        "provenance": {
            "encoder_param_hash": bman["encoder_param_hash"],
            "probe_param_hash": bman["probe_param_hash"],
            "feature_cache_sha256": cman["features_sha256"],
            "feature_cache_scope": cman["scope"],
            "feature_cache_n_rows": int(cman["n_rows"]),
            "cache_reencode_check": reenc,
            "store_plan_path": str(store_plan_path(config, dataset, pretext)),
        },
        "K": K, "n_modes": int(plan["n_modes"]), "grouping": ginfo,
        "byte_ledger": byte_led,
        "prototype_ledger": protos.byte_ledger(),
        "channel_ledger": chan,
        "no_bypass": bypass,
        "request_only_path": req_only,
        "rows": {k: int(v.size) for k, v in row_sets.items()} | {
            "store_train": int(rows["store_train"].size)},
        "store_training": {
            "steps": int(config.gen_store.train_steps),
            "loss_first": float(hist[0]) if hist else None,
            "loss_final": float(hist[-1]) if hist else None,
            "loaded_from_checkpoint": bool(not hist),
        },
        "request_addressing": {
            "mean_confidence": {k: codes[k]["mean_confidence"] for k in REQUEST_SETS},
            "n_distinct_code_configs": {k: codes[k]["n_distinct_code_configs"]
                                        for k in REQUEST_SETS},
            "code_agreement_vs_feature_addressing": agree,
            "address_book_bytes": address_book_bytes(book),
        },
        "metric_invariance_check": inv,
        "ceilings": {
            "decoder": {k: _strip_probs(v) if k == "in_corner_discrimination" else v
                        for k, v in ceil["decoder"].items()},
            "store": {k: _strip_probs(v) if k == "in_corner_discrimination" else v
                      for k, v in ceil["store"].items()},
            "store_ceiling_balacc_quoted": float(cfg.store_ceiling_balacc),
            "knn_proto_reference_balacc": float(cfg.knn_proto_reference_balacc),
            "knn_channel_matched_reference_balacc": float(
                cfg.knn_channel_matched_reference_balacc),
            "note": ("⛔ R23.5: every arm row is read against BOTH ceilings. The quoted "
                     "reference rows are R22/R27's and are ⛔ not re-measured here."),
        },
        "arms": {},
        "NO_KILL_RULE_VERDICT": ("⛔ arm numbers only — gen-cell-run renders every "
                                 "kill-rule verdict against its own filed prereg."),
    }
    _write_json(manifest, out_dir / "compose_manifest.json")

    # ---- the arms -----------------------------------------------------------
    for arm in list(arms or cfg.arms):
        feats_arm = {
            k: arm_features(arm, store, codes[k]["code"], post[k], protos, req_dicts[k],
                            cfg, seed)
            for k in REQUEST_SETS
        }
        comp_a = score_set(decoder, probe, feats_arm["A"], lab["A"], names, feat_shape,
                           cfg.eval_batch)
        comp_b = score_set(decoder, probe, feats_arm["B"], lab["B"], names, feat_shape,
                           cfg.eval_batch)
        disc = corner_discrimination(
            decoder, probe, feats_arm["B"], feats_arm["MS"], lab["B"], lab["MS"],
            shape_idx, int(cards[shape_idx]), feat_shape, cfg.eval_batch,
        )
        cd = ceil["decoder"]["in_corner_discrimination"]["balanced_accuracy"]
        cs = ceil["store"]["in_corner_discrimination"]["balanced_accuracy"]
        manifest["arms"][arm] = {
            "arm": arm,
            "uses_store": bool(arm != "proto"),
            "composition_A": comp_a,
            "composition_B": comp_b,
            "in_corner_discrimination": _strip_probs(disc),
            "fraction_of_decoder_ceiling": float(disc["balanced_accuracy"] / cd) if cd else float("nan"),
            "fraction_of_store_ceiling": float(disc["balanced_accuracy"] / cs) if cs else float("nan"),
            "channel": chan.get(arm if arm in chan else "read"),
            "n_distinct_images_proxy": {
                k: int(np.unique(np.asarray(feats_arm[k]).reshape(
                    (feats_arm[k].shape[0], -1)), axis=0).shape[0]) for k in REQUEST_SETS
            },
        }
        _write_json(manifest, out_dir / "compose_manifest.json")  # ⭐ incremental
        tlog(
            f"  [{arm}] in-corner {COMPOSITIONAL_FACTOR} balacc="
            f"{disc['balanced_accuracy']:.4f} (AUROC {disc['auroc']:.4f}) · "
            f"A aggregate={comp_a['aggregate']:.4f} {COMPOSITIONAL_FACTOR}="
            f"{comp_a['per_factor'][COMPOSITIONAL_FACTOR]:.4f} · "
            f"B aggregate={comp_b['aggregate']:.4f} {COMPOSITIONAL_FACTOR}="
            f"{comp_b['per_factor'][COMPOSITIONAL_FACTOR]:.4f}"
        )

    manifest["falsifier_read_vs_store_removed"] = seed_falsifier(manifest)
    manifest["wall_clock_s"] = float(time.time() - t0)
    _write_json(manifest, out_dir / "compose_manifest.json")
    if cfg.make_plots:
        try:
            _plot_seed(manifest, out_dir / "compose.png")
        except Exception as exc:  # a plot is never allowed to lose a result
            tlog(f"  ⚠ plot failed (results are already on disk): {exc}")
    tlog(f"seed {seed} done")
    return manifest


# ---------------------------------------------------------------------------
# 5. ⭐ The falsifier — read vs store_removed, checked before anything else
# ---------------------------------------------------------------------------


def seed_falsifier(manifest: dict) -> dict:
    """⭐ *"The read path does not beat store-removed on ``shape``"* — the whole claim."""
    arms = manifest.get("arms", {})
    if "read" not in arms or "store_removed" not in arms:
        return {"available": False,
                "note": "both the read arm and its ablation must run for the falsifier"}
    r, s = arms["read"], arms["store_removed"]
    d_bal = float(r["in_corner_discrimination"]["balanced_accuracy"]
                  - s["in_corner_discrimination"]["balanced_accuracy"])
    return {
        "available": True,
        "read_in_corner_balacc": r["in_corner_discrimination"]["balanced_accuracy"],
        "store_removed_in_corner_balacc": s["in_corner_discrimination"]["balanced_accuracy"],
        "delta_in_corner_balacc": d_bal,
        "delta_B_shape_accuracy": float(
            r["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR]
            - s["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR]
        ),
        "delta_A_aggregate": float(r["composition_A"]["aggregate"]
                                   - s["composition_A"]["aggregate"]),
        "note": ("⛔ per-seed only; the seed spread and the verdict live in the "
                 "aggregate. ⛔ No kill-rule verdict is rendered by this spoke."),
    }


# ---------------------------------------------------------------------------
# 6. Aggregate over seeds
# ---------------------------------------------------------------------------


def aggregate_runs(run_dirs, log: Callable = print) -> dict:
    mans = []
    for d in run_dirs:
        p = Path(d) / "compose_manifest.json"
        if p.exists():
            with open(p) as fh:
                mans.append(json.load(fh))
    if not mans:
        raise FileNotFoundError("no per-seed compose manifests found")

    def rows_for(getter):
        return _ms([getter(m) for m in mans])

    table = {}
    for arm in sorted({a for m in mans for a in m.get("arms", {})}):
        got = [m["arms"][arm] for m in mans if arm in m.get("arms", {})]
        table[arm] = {
            "arm": arm,
            "n_seeds": len(got),
            "uses_store": got[0]["uses_store"],
            "in_corner_balacc": _ms([g["in_corner_discrimination"]["balanced_accuracy"] for g in got]),
            "in_corner_auroc": _ms([g["in_corner_discrimination"]["auroc"] for g in got]),
            "in_corner_tpr": _ms([g["in_corner_discrimination"].get("tpr") for g in got]),
            "in_corner_tnr": _ms([g["in_corner_discrimination"].get("tnr") for g in got]),
            "A_aggregate": _ms([g["composition_A"]["aggregate"] for g in got]),
            "A_shape": _ms([g["composition_A"]["per_factor"][COMPOSITIONAL_FACTOR] for g in got]),
            "B_aggregate": _ms([g["composition_B"]["aggregate"] for g in got]),
            "B_shape": _ms([g["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR] for g in got]),
            "fraction_of_decoder_ceiling": _ms([g["fraction_of_decoder_ceiling"] for g in got]),
            "fraction_of_store_ceiling": _ms([g["fraction_of_store_ceiling"] for g in got]),
            "per_factor_A": {
                f: _ms([g["composition_A"]["per_factor"][f] for g in got])
                for f in got[0]["composition_A"]["per_factor"]
            },
            "per_factor_B": {
                f: _ms([g["composition_B"]["per_factor"][f] for g in got])
                for f in got[0]["composition_B"]["per_factor"]
            },
        }

    fals = None
    if "read" in table and "store_removed" in table:
        deltas = [m["falsifier_read_vs_store_removed"]["delta_in_corner_balacc"]
                  for m in mans if m.get("falsifier_read_vs_store_removed", {}).get("available")]
        d = _ms(deltas)
        spread = max(table["read"]["in_corner_balacc"]["std"],
                     table["store_removed"]["in_corner_balacc"]["std"])
        fals = {
            "statistic": ("in-corner shape balanced accuracy, read − store_removed "
                          "(⭐ the whole claim; checked before anything else)"),
            "delta": d,
            "n_seeds_positive": int(sum(1 for x in deltas if x > 0)),
            "largest_arm_seed_spread": float(spread),
            "delta_over_spread": float(d["mean"] / spread) if spread > 0 else float("inf"),
            "beats_store_removed_beyond_seed_spread": bool(
                d["mean"] > 0 and (spread == 0 or d["mean"] > spread)
            ),
            "delta_B_shape": _ms([m["falsifier_read_vs_store_removed"]["delta_B_shape_accuracy"]
                                  for m in mans
                                  if m.get("falsifier_read_vs_store_removed", {}).get("available")]),
            "note": ("⛔ This is the spoke's own falsifier, NOT a kill-rule verdict. "
                     "gen-cell-run renders verdicts against its own filed prereg."),
        }

    agg = {
        "schema": "chlu-gen-compose-aggregate/1",
        "dataset": mans[0]["dataset"], "pretext": mans[0]["pretext"],
        "condition": mans[0]["condition"],
        "seeds": [m["seed"] for m in mans],
        "mechanism": mans[0]["mechanism"],
        "K": mans[0]["K"], "n_modes": mans[0]["n_modes"],
        "byte_ledger": mans[0]["byte_ledger"],
        "channel_ledger": mans[0]["channel_ledger"],
        "decoder": {str(m["seed"]): m["decoder"] for m in mans},
        "metric_invariance_check": {str(m["seed"]): m.get("metric_invariance_check")
                                    for m in mans},
        "ceiling_decoder_in_corner_balacc": rows_for(
            lambda m: m["ceilings"]["decoder"]["in_corner_discrimination"]["balanced_accuracy"]),
        "ceiling_store_in_corner_balacc": rows_for(
            lambda m: m["ceilings"]["store"]["in_corner_discrimination"]["balanced_accuracy"]),
        "ceiling_decoder_A_aggregate": rows_for(
            lambda m: m["ceilings"]["decoder"]["composition_A"]["aggregate"]),
        "ceiling_store_A_aggregate": rows_for(
            lambda m: m["ceilings"]["store"]["composition_A"]["aggregate"]),
        "ceiling_decoder_B_shape": rows_for(
            lambda m: m["ceilings"]["decoder"]["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR]),
        "ceiling_store_B_shape": rows_for(
            lambda m: m["ceilings"]["store"]["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR]),
        "reference_rows": {
            "knn_proto_in_corner_balacc": mans[0]["ceilings"]["knn_proto_reference_balacc"],
            "knn_channel_matched_in_corner_balacc":
                mans[0]["ceilings"]["knn_channel_matched_reference_balacc"],
            "store_ceiling_quoted": mans[0]["ceilings"]["store_ceiling_balacc_quoted"],
            "source": "R22 / R27 — quoted with provenance, ⛔ not re-measured here",
        },
        "code_agreement": {
            k: rows_for(lambda m, k=k: m["request_addressing"]
                        ["code_agreement_vs_feature_addressing"][k]["exact_set"])
            for k in REQUEST_SETS
        },
        "table": table,
        "falsifier_read_vs_store_removed": fals,
        "NO_KILL_RULE_VERDICT": mans[0]["NO_KILL_RULE_VERDICT"],
    }
    log(f"aggregated {len(mans)} seed(s)")
    return agg


# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------


def _plot_seed(man: dict, path: Path) -> None:
    arms = [a for a in ARMS if a in man.get("arms", {})]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    vals = [man["arms"][a]["in_corner_discrimination"]["balanced_accuracy"] for a in arms]
    ax.bar(np.arange(len(arms)), vals, 0.6)
    ax.axhline(man["ceilings"]["decoder"]["in_corner_discrimination"]["balanced_accuracy"],
               ls=":", c="k", lw=0.9, label="decoder ceiling")
    ax.axhline(man["ceilings"]["store"]["in_corner_discrimination"]["balanced_accuracy"],
               ls="--", c="k", lw=0.9, label="store ceiling (measured)")
    ax.axhline(man["ceilings"]["knn_proto_reference_balacc"], ls="-.", c="crimson", lw=0.9,
               label="knn_proto (R27)")
    ax.axhline(0.5, ls="-", c="grey", lw=0.8, label="chance")
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(arms, rotation=20, fontsize=7)
    ax.set_ylabel(f"in-corner {COMPOSITIONAL_FACTOR} balanced accuracy")
    ax.set_title("⭐ the gate statistic (R20/R23)")
    ax.legend(fontsize=6)
    ax = axes[1]
    xs = np.arange(len(arms))
    ax.bar(xs - 0.2, [man["arms"][a]["composition_A"]["aggregate"] for a in arms], 0.4,
           label="A aggregate")
    ax.bar(xs + 0.2, [man["arms"][a]["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR]
                      for a in arms], 0.4, label=f"B {COMPOSITIONAL_FACTOR}")
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, rotation=20, fontsize=7)
    ax.set_ylabel("probe accuracy")
    ax.set_title("⛔ shape is never folded into the aggregate")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------


def run_experiment_gen_compose(
    config: Optional[CHLUConfig] = None,
    dataset: Optional[str] = None,
    pretext: Optional[str] = None,
    out_dir: str = "results/gen_compose",
    seeds: Optional[list] = None,
    stage: str = "compose",
    arms: Optional[list] = None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_compose
    dataset = dataset or cfg.dataset
    pretext = pretext or cfg.pretext
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}, got {pretext!r}")
    if stage in NOT_MINE:
        log(f"[gen-compose] stage {stage!r} belongs to another spoke — nothing to do")
        return {}
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    condition = condition_of(config)
    root = Path(out_dir) / artifacts_tag(dataset, pretext, condition)
    seeds = list(seeds if seeds is not None else cfg.seeds)
    log(
        f"[gen-compose] dataset={dataset} pretext={pretext} condition={condition} "
        f"stage={stage} seeds={seeds} -> {root}"
    )
    run_dirs = [root / f"seed_{s}" for s in seeds]
    if stage == "compose":
        for s, d in zip(seeds, run_dirs, strict=True):
            run_seed(dataset, pretext, int(s), d, config, arms=arms, log=log)
    agg = aggregate_runs(run_dirs, log=log)
    _write_json(agg, root / "compose_results.json")
    return agg


# ---------------------------------------------------------------------------
# 9. CLI  (contract: --dataset --pretext --stage --arm --seed --out)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Composition through the READ: request -> code -> frozen decoder -> probe."
    )
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument("--stage", default="compose",
                   choices=list(STAGES) + list(COMMIT_STAGES) + list(NOT_MINE))
    p.add_argument("--arm", default=None, action="append",
                   help=f"repeatable; default: gen_compose.arms {list(ARMS)}")
    p.add_argument("--seed", type=int, default=None, action="append", help="repeatable")
    p.add_argument("--out", default="results/gen_compose")
    p.add_argument("--condition", default=None,
                   help="hold-out condition (default: gen_data.condition)")
    p.add_argument("--n-eval-a", type=int, default=None)
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--store-plan", default=None)
    p.add_argument("--quick", action="store_true",
                   help="plumbing smoke only — ⛔ NEVER a reported number")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    g = config.gen_compose
    if args.condition:
        config.gen_data.condition = args.condition
    if args.n_eval_a is not None:
        g.n_eval_a = int(args.n_eval_a)
    if args.bridge_root:
        g.bridge_root = args.bridge_root
        config.gen_store.bridge_root = args.bridge_root
    if args.store_plan:
        g.store_plan = args.store_plan
    # ⚠ ⛔ Do NOT copy $CHLU_GEN_DATA_ROOT into ``gen_data.data_root``: the two are
    # read with DIFFERENT conventions (the worktree trap, exp_gen_dream's note).
    return config


def main(argv=None) -> int:
    av = list(sys.argv[1:] if argv is None else argv)
    # ⭐ the commitment stages take their own flags (--config is a shortlist entry),
    # so they are dispatched BEFORE the cell parser sees the command line.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--stage", default="compose")
    if pre.parse_known_args(av)[0].stage in COMMIT_STAGES:
        return main_commit(av)
    args = build_arg_parser().parse_args(av)
    config = _config_from_args(args)
    agg = run_experiment_gen_compose(
        config=config, dataset=args.dataset, pretext=args.pretext, out_dir=args.out,
        seeds=args.seed, stage=args.stage, arms=args.arm, quick=bool(args.quick),
    )
    if agg:
        print(
            f"⭐ ceilings — decoder {agg['ceiling_decoder_in_corner_balacc']['mean']:.4f} · "
            f"store (measured) {agg['ceiling_store_in_corner_balacc']['mean']:.4f} · "
            f"knn_proto {agg['reference_rows']['knn_proto_in_corner_balacc']:.4f} "
            "(⛔ a number quoted against only one ceiling is incomplete)"
        )
        for arm, row in agg["table"].items():
            print(
                f"  {arm:<14} in-corner {COMPOSITIONAL_FACTOR} balacc "
                f"{row['in_corner_balacc']['mean']:.4f} ± {row['in_corner_balacc']['std']:.4f} · "
                f"AUROC {row['in_corner_auroc']['mean']:.4f} · A aggregate "
                f"{row['A_aggregate']['mean']:.4f} · B {COMPOSITIONAL_FACTOR} "
                f"{row['B_shape']['mean']:.4f}"
            )
        f = agg.get("falsifier_read_vs_store_removed")
        if f:
            print(
                f"⭐ falsifier: read − store_removed = {f['delta']['mean']:+.4f} ± "
                f"{f['delta']['std']:.4f} ({f['n_seeds_positive']}/{f['delta']['n']} seeds "
                f"positive, {f['delta_over_spread']:.1f}× the largest arm spread) ⇒ "
                + ("✓ the read beats its ablation"
                   if f["beats_store_removed_beyond_seed_spread"]
                   else "⛔ THE FALSIFIER FIRED — composition is not the store's")
            )
        print("⛔ arm numbers only — no kill-rule verdict is rendered here.")
    return 0



# ---------------------------------------------------------------------------
# 10. ⭐ THE COMMITMENT STAGE (R29 pivot item 1) — 72 requests, 9 codes, why
# ---------------------------------------------------------------------------
#
# ⭐ DIAL: *commitment* — how the request's posterior over modes becomes an integer
# code.  **Laundering control: the matched-seen (MS) corner, on every number.**
# **Falsifies:** ≥60/72 distinct codes on B AND the store→decoder arm within seed
# spread of ``knn_proto`` (0.8817 ± 0.0023) ⇒ PASS; anything else is a FAIL, and
# ⛔ a fail routes to new approaches, never to abandonment or to re-tuning.
# **Does NOT falsify:** losing to the unbounded-channel prototype arm on level
# alone (R23.5's standing reading), or a distinct count that rises with MS — that
# is the corner moving, which is the honest direction.
#
# ⛔ Two traps this stage is written around, both already paid for:
#   1. ``dec_recipe`` defaults to ``"baseline"`` — the decoder that FAILED the gate.
#      :func:`resolve_decoder` asserts hash **and** recipe, and it runs in **both**
#      sub-stages, so a void run cannot look normal in the log.
#   2. ``gen_store.soft_read`` takes FEATURES.  Every weight here comes from
#      :func:`chlu.core.gen_compose.request_posterior` — the request — and
#      ``assert_request_only`` is re-run on this path before any number.

COMMIT_STAGES = ("commit", "score")


def _commit_cfg(config: CHLUConfig):
    from ..config import gen_commit_config

    return gen_commit_config(config)


def commit_out_root(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    c = _commit_cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-commitment-collapse"


def _store_source(config: CHLUConfig, dataset: str, pretext: str, seed: int):
    c = _commit_cfg(config)
    if not c.store_checkpoint_source:
        return None
    p = Path(str(c.store_checkpoint_source).format(
        dataset=dataset, pretext=pretext, seed=int(seed)))
    return p if p.is_absolute() else _claude_root(config) / p


def _cache_path(out_dir: Path, seed: int) -> Path:
    return Path(out_dir) / f"commit_cache_s{seed}.npz"


def build_commit_cache(
    dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
    log: Callable = print,
) -> dict:
    """Everything the commitment sweep needs, **once** per seed, then cached.

    ⭐ The expensive halves are the store-train feature gather (seeds 1–2 hold no
    matched-seen row in their cache and re-encode through the frozen encoder) and
    the per-seed grouping.  Neither depends on any knob, so they are paid once and
    the whole sweep afterwards is ``numpy``.

    ⛔ The A/B/MS **feature** tensors are deliberately NOT gathered by default: the
    commitment path is request-only, so gathering them would cost a 32k-row
    re-encode to produce something no arm may consume.
    """
    cfg = config.gen_compose
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(out_dir, seed)
    bdir = bridge_dir(config, dataset, pretext, seed)

    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_store_rows(split, config, seed)
    n_a = int(min(int(cfg.n_eval_a), rows["eval_a"].size))
    row_sets = {
        "A": rows["eval_a"][:n_a],
        "B": rows["eval_b"],
        "MS": matched_seen_rows(dataset, split, values, rows["store_train"]),
    }
    lab = {k: factor_labels(grid, v) for k, v in row_sets.items()}

    if path.exists():
        z = np.load(path, allow_pickle=False)
        book = AddressBook(z["log_prior"], [z[f"log_cond_{f}"] for f in range(len(cards))])
        log(f"  commit cache HIT {path.name} (⭐ no feature gather, no grouping re-derive)")
        return {"book": book, "labels_vec": z["labels_vec"], "lab": lab, "rows": rows,
                "row_sets": row_sets, "names": names, "cards": cards, "grid": grid,
                "bdir": bdir, "cached": True, "split": split}

    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(cfg.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    config.gen_store.encode_chunk = int(cfg.encode_chunk)
    X_store = gather_features(dataset, config, encoder, crows, feats, cman,
                              rows["store_train"], log=log)
    lab_store = factor_labels(grid, rows["store_train"])

    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    config.gen_dream.grouping_source = str(cfg.grouping_source)
    labels_vec, ginfo = resolve_grouping(
        plan, X_store, config, seed, bman["encoder_param_hash"], log=log)
    store = load_commit_store(dataset, pretext, seed, out_dir, config, labels_vec, plan, log)
    codes_train = address_codes(store, np.asarray(X_store, np.float32).reshape(
        (X_store.shape[0], -1)))
    book = fit_address_book(codes_train, lab_store, cards, float(cfg.address_smoothing))
    payload = {"log_prior": np.asarray(book.log_prior, np.float32),
               "labels_vec": np.asarray(labels_vec, np.int64)}
    for f, lc in enumerate(book.log_cond):
        payload[f"log_cond_{f}"] = np.asarray(lc, np.float32)
    np.savez_compressed(path, **payload)
    log(f"  commit cache WRITTEN {path.name} (grouping {ginfo['grouping_source']})")
    return {"book": book, "labels_vec": np.asarray(labels_vec, np.int64), "lab": lab,
            "rows": rows, "row_sets": row_sets, "names": names, "cards": cards,
            "grid": grid, "bdir": bdir, "cached": False, "grouping": ginfo,
            "split": split}


def load_commit_store(dataset, pretext, seed, out_dir, config, labels_vec, plan, log=print):
    """Load ``gen-read-compose``'s per-seed store checkpoint — ⛔ never retrain one here.

    ⭐ These commitment numbers must sit on **exactly** the store R28/R29 measured,
    otherwise a change in the distinct-code count could be the store moving rather
    than the commitment rule.  The parameter hash is recorded in the manifest.
    """
    src = _store_source(config, dataset, pretext, seed)
    if src is None or not Path(src).exists():
        raise FileNotFoundError(
            f"no store checkpoint at {src} — ⛔ this spoke reads gen-read-compose's "
            "stores rather than training its own, so the commitment rule is the only "
            "thing that changes between it and R28's numbers"
        )
    store = load_model(Path(src))
    K = int(plan["K"])
    wc = well_channels_from_labels(np.asarray(labels_vec, np.int64), K)
    got = [len(w) for w in store.well_channels]
    if got != [len(w) for w in wc]:
        raise AssertionError(
            f"the loaded store's well sizes {got} do not match the re-derived "
            f"grouping {[len(w) for w in wc]} — ⛔ different partitions"
        )
    log(f"  ⭐ store loaded from {src} (hash {param_hash(store)[:16]}…, K={store.n_wells}, "
        f"M={store.n_modes}, {store.bits_per_read:.0f} bits/read)")
    return store


def sweep_configs(config: CHLUConfig) -> list:
    """The pre-registered knob grid: **stopping rule · multiplicity · temperature**."""
    c = _commit_cfg(config)
    out = []
    for tau in c.taus:
        out.append({"rule": "argmax", "tau": float(tau), "margin": 0.0, "k": 1})
    for tau in c.taus:
        out.append({"rule": "sample", "tau": float(tau), "margin": 0.0, "k": 1})
    for tau in c.taus:
        for m in c.margins:
            out.append({"rule": "margin", "tau": float(tau), "margin": float(m), "k": 1})
    for tau in c.taus:
        for k in c.ks:
            out.append({"rule": "topk", "tau": float(tau), "margin": 0.0, "k": int(k)})
    return out


def config_key(cc: dict) -> str:
    return (f"{cc['rule']}|tau={float(cc['tau']):g}|margin={float(cc.get('margin', 0.0)):g}"
            f"|k={int(cc.get('k', 1))}")


def parse_config_key(key: str) -> dict:
    parts = str(key).split("|")
    out = {"rule": parts[0], "tau": 1.0, "margin": 0.0, "k": 1}
    for p in parts[1:]:
        if "=" not in p:
            continue
        a, b = p.split("=", 1)
        out[a] = int(b) if a == "k" else float(b)
    if out["rule"] not in COMMITMENT_RULES:
        raise ValueError(f"rule must be one of {COMMITMENT_RULES}, got {out['rule']!r}")
    return out


def run_commit_seed(
    dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
    log: Callable = print,
) -> dict:
    """Leg 1 (diagnosis) + leg 2 (the cheap sweep) + leg 4 (persist the tuples)."""
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    c = _commit_cfg(config)
    out_dir = Path(out_dir)
    root = commit_out_root(config, dataset, pretext)
    man_path = out_dir / f"commit_manifest_s{seed}.json"
    if man_path.exists():
        with open(man_path) as fh:
            done = json.load(fh)
        if done.get("complete"):
            tlog(f"seed {seed}: commit manifest already complete — ⭐ skipping (resumable)")
            return done

    cache = build_commit_cache(dataset, pretext, seed, out_dir, config, log=tlog)
    book, lab = cache["book"], cache["lab"]
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    store = load_commit_store(dataset, pretext, seed, out_dir, config,
                              cache["labels_vec"], plan, tlog)
    # ⛔ the decoder guard runs here too: a void run must not look normal in the log
    _, dec_info, _ = resolve_decoder(config, cache["bdir"], seed, log=tlog)

    req = {k: requests_from_labels(lab[k], "full", 0, int(seed)) for k in REQUEST_SETS}
    post = {k: request_posterior(book, req[k]) for k in REQUEST_SETS}
    req_only = assert_request_only(store, book, req["B"], decoy_features=None)
    tlog("⭐ request-only path asserted (no feature argument on ANY arm; soft_read not imported)")

    # ---- leg 1: diagnosis — is it the posterior or the argmax? ---------------
    diag = {k: posterior_diagnostics(post[k]) for k in REQUEST_SETS}
    for k in REQUEST_SETS:
        d = diag[k]
        tlog(f"  [{k}] top1={d['mean_top1_confidence']:.4f} margin(top1−top2)="
             f"{d['mean_top1_minus_top2_margin']:.4f} entropy={d['mean_entropy_bits_per_well']:.3f}"
             f"/{d['max_entropy_bits_per_well']:.1f} bits per well ⇒ "
             + ("⛔ DEGENERATE (no stopping rule can move it)" if d["degenerate"]
                else "not degenerate"))
    evid = {k: factor_evidence(book, req[k]) for k in REQUEST_SETS}
    lev = {k: factor_leverage(book, req[k]) for k in REQUEST_SETS}
    for k in REQUEST_SETS:
        tlog(f"  [{k}] ⭐ factor leverage (code flip rate | codes reachable alone): " + " ".join(
            f"{cache['names'][r['factor']]}={r['code_flip_rate']:.2f}|{r['n_codes_from_factor_alone']}"
            for r in lev[k]["per_factor"]))
    # ⭐ the mechanical hypothesis, as a counterfactual: drop the colour factors
    hue_idx = [i for i, nm in enumerate(cache["names"]) if "hue" in nm]
    counterfactual = {}
    for k in REQUEST_SETS:
        r2 = np.array(req[k], np.int64)
        r2[:, hue_idx] = -1
        p2 = request_posterior(book, r2)
        counterfactual[k] = {
            "dropped_factors": [cache["names"][i] for i in hue_idx],
            "n_distinct_argmax": distinct_codes(commit(p2, "argmax")["code"]),
            "n_requests": int(r2.shape[0]),
            "mean_top1_confidence": posterior_diagnostics(p2)["mean_top1_confidence"],
        }
    tlog("⭐ hue-dropped counterfactual (⛔ diagnostic, not an arm): " + " ".join(
        f"{k}={counterfactual[k]['n_distinct_argmax']}/{counterfactual[k]['n_requests']}"
        for k in REQUEST_SETS))

    # ---- leg 4: persist the per-request code tuples --------------------------
    base = {k: commit(post[k], "argmax") for k in REQUEST_SETS}
    for k in REQUEST_SETS:
        _write_json({
            "set": k, "seed": int(seed), "rule": "argmax (shipped)",
            "dataset": dataset, "pretext": pretext,
            "n_requests": int(base[k]["n_requests"]),
            "n_distinct_code_configs": int(base[k]["n_distinct"]),
            "rows": [int(x) for x in cache["row_sets"][k]],
            "factor_names": cache["names"],
            "requests": [[int(x) for x in r] for r in req[k]],
            "codes": [[int(x) for x in r] for r in base[k]["code"]],
            "note": ("⭐ R29 Q2's owed artefact: the per-request integer code TUPLES, "
                     "not just their count."),
        }, root / f"codes_{k}_seed{seed}.json")
    overlap = {
        "B_in_A": code_overlap(base["A"]["code"], base["B"]["code"]),
        "B_in_MS": code_overlap(base["MS"]["code"], base["B"]["code"]),
        "MS_in_A": code_overlap(base["A"]["code"], base["MS"]["code"]),
    }
    tlog(f"⭐ |B ∩ A| = {overlap['B_in_A']['n_intersection']} of "
         f"{overlap['B_in_A']['n_distinct_b']} B codes · |B ∩ MS| = "
         f"{overlap['B_in_MS']['n_intersection']} of {overlap['B_in_MS']['n_distinct_b']}")

    # ---- leg 2: the cheap sweep, ⛔ every row with its MS companion ----------
    sweep = []
    for cc in sweep_configs(config):
        row = {"config": config_key(cc), **cc, "sets": {}}
        for k in REQUEST_SETS:
            draws = int(c.sample_draws) if cc["rule"] == "sample" else 1
            ns, com = [], None
            for d in range(draws):
                com = commit(post[k], cc["rule"], cc["tau"], cc["margin"], cc["k"],
                             str(c.topk_weight),
                             seed=int(c.sample_seed) + 1000 * int(seed)
                             + 37 * REQUEST_SETS.index(k) + d)
                ns.append(com["n_distinct"])
            row["sets"][k] = {
                "n_requests": int(com["n_requests"]),
                "n_distinct": float(np.mean(ns)),
                "n_distinct_draws": [int(x) for x in ns],
                "distinct_fraction": float(np.mean(ns) / com["n_requests"]),
                "abstain_rate": com["abstain_rate"],
                "mean_top1": com["mean_top1"],
            }
            if k == "B":
                row["channel"] = commitment_channel(store, com, float(c.design_constant_bits))
        sweep.append(row)
    _write_json({"seed": int(seed), "dataset": dataset, "pretext": pretext,
                 "design_constant_bits": float(c.design_constant_bits),
                 "sweep": sweep}, out_dir / f"sweep_s{seed}.json")
    tlog(f"⭐ sweep: {len(sweep)} configs · B distinct from "
         f"{min(r['sets']['B']['n_distinct'] for r in sweep):.0f} to "
         f"{max(r['sets']['B']['n_distinct'] for r in sweep):.0f} of "
         f"{sweep[0]['sets']['B']['n_requests']}")

    man = {
        "schema": "chlu-gen-commit/1", "complete": True,
        "dataset": dataset, "pretext": pretext, "seed": int(seed),
        "condition": condition_of(config),
        "dial": ("commitment — how the request's posterior over modes becomes an "
                 "integer code (R29 pivot item 1)"),
        "laundering_control": ("⭐ the matched-seen (MS) corner, on every number: a "
                               "knob that lifts B while MS stays flat has found a "
                               "B-specific artefact, not a fix (R30)"),
        "decoder": dec_info,
        "store_checkpoint": str(_store_source(config, dataset, pretext, seed)),
        "store_param_hash": param_hash(store),
        "K": int(store.n_wells), "n_modes": int(store.n_modes),
        "bits_per_read": float(store.bits_per_read),
        "rows": {k: int(v.size) for k, v in cache["row_sets"].items()},
        "request_only_path": req_only,
        "diagnosis": diag,
        "factor_evidence": evid,
        "factor_leverage": lev,
        "hue_dropped_counterfactual": counterfactual,
        "baseline_argmax_distinct": {k: int(base[k]["n_distinct"]) for k in REQUEST_SETS},
        "code_overlap": overlap,
        "sweep": sweep,
        "wall_clock_s": float(time.time() - t0),
    }
    _write_json(man, man_path)
    tlog(f"seed {seed} commit stage done")
    return man


def score_commit_seed(
    dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
    shortlist: list, log: Callable = print,
) -> dict:
    """Leg 3: accuracy at the **pre-registered shortlist only**, ⛔ never the sweep.

    ⭐ *Both numbers or neither.*  A distinct count is trivially maximised by noise;
    the in-corner accuracy beside it is the only thing that stops noise scoring as a
    win.  Positives = the arm's renders of the 72 held-out (B) requests, negatives =
    its renders of the matched-seen requests — ⛔ identical in construction to
    ``gen-baselines``' reader so the rows stay comparable.
    """
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    cfg, c = config.gen_compose, _commit_cfg(config)
    out_dir = Path(out_dir)
    cache = build_commit_cache(dataset, pretext, seed, out_dir, config, log=tlog)
    book, lab, names, cards = cache["book"], cache["lab"], cache["names"], cache["cards"]
    shape_idx = names.index(COMPOSITIONAL_FACTOR)
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    store = load_commit_store(dataset, pretext, seed, out_dir, config,
                              cache["labels_vec"], plan, tlog)
    decoder, dec_info, bman = resolve_decoder(config, cache["bdir"], seed, log=tlog)
    probe = load_model(cache["bdir"] / "probe.pkl")
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest hash")
    feat_shape = tuple(int(x) for x in json.load(
        open(cache["bdir"] / "feature_cache.json"))["feat_shape"])

    req = {k: requests_from_labels(lab[k], "full", 0, int(seed)) for k in REQUEST_SETS}
    post = {k: request_posterior(book, req[k]) for k in REQUEST_SETS}

    path = out_dir / f"scores_s{seed}.json"
    scored = {}
    if path.exists():
        with open(path) as fh:
            scored = json.load(fh).get("configs", {})
    for key in shortlist:
        if key in scored:
            tlog(f"  [{key}] already scored — ⭐ skipping (per-unit resumable)")
            continue
        cc = parse_config_key(key)
        coms, feats_arm = {}, {}
        for k in REQUEST_SETS:
            coms[k] = commit(post[k], cc["rule"], cc["tau"], cc["margin"], cc["k"],
                             str(c.topk_weight),
                             seed=int(c.sample_seed) + 1000 * int(seed)
                             + 37 * REQUEST_SETS.index(k))
            feats_arm[k] = np.asarray(commitment_read(store, coms[k]), np.float32)
        disc = corner_discrimination(
            decoder, probe, feats_arm["B"], feats_arm["MS"], lab["B"], lab["MS"],
            shape_idx, int(cards[shape_idx]), feat_shape, cfg.eval_batch)
        comp_a = score_set(decoder, probe, feats_arm["A"], lab["A"], names, feat_shape,
                           cfg.eval_batch)
        comp_b = score_set(decoder, probe, feats_arm["B"], lab["B"], names, feat_shape,
                           cfg.eval_batch)
        scored[key] = {
            "config": key, **cc,
            "in_corner_discrimination": _strip_probs(disc),
            "composition_A": comp_a, "composition_B": comp_b,
            "n_distinct": {k: int(coms[k]["n_distinct"]) for k in REQUEST_SETS},
            "n_requests": {k: int(coms[k]["n_requests"]) for k in REQUEST_SETS},
            "abstain_rate": {k: coms[k]["abstain_rate"] for k in REQUEST_SETS},
            "n_distinct_rendered": {
                k: int(np.unique(feats_arm[k].reshape(feats_arm[k].shape[0], -1),
                                 axis=0).shape[0]) for k in REQUEST_SETS},
            "channel": commitment_channel(store, coms["B"], float(c.design_constant_bits)),
            "pass_distinct_B": bool(coms["B"]["n_distinct"] >= int(c.pass_distinct_floor)),
            "pass_accuracy": bool(
                disc["balanced_accuracy"]
                >= float(c.pass_target_balacc) - float(c.pass_target_spread)),
        }
        scored[key]["PASS"] = bool(scored[key]["pass_distinct_B"]
                                   and scored[key]["pass_accuracy"])
        # ⭐ written as each (config, seed) completes: a detached spoke loses its LAST
        # write, never the ones before it.
        _write_json({"seed": int(seed), "dataset": dataset, "pretext": pretext,
                     "decoder": dec_info, "store_param_hash": param_hash(store),
                     "pass_target": {"balacc": float(c.pass_target_balacc),
                                     "spread": float(c.pass_target_spread),
                                     "distinct_floor": int(c.pass_distinct_floor)},
                     "configs": scored}, path)
        tlog(f"  [{key}] B distinct={scored[key]['n_distinct']['B']}/72 "
             f"(MS {scored[key]['n_distinct']['MS']}/{scored[key]['n_requests']['MS']}) · "
             f"in-corner {COMPOSITIONAL_FACTOR} balacc="
             f"{disc['balanced_accuracy']:.4f} (AUROC {disc['auroc']:.4f}) · A aggregate="
             f"{comp_a['aggregate']:.4f} · width={scored[key]['channel']['pointer_bits']:.1f} "
             f"bits{' ⛔ EXCEEDS 30' if scored[key]['channel']['exceeds_design_constant'] else ''}"
             f" ⇒ {'⭐ PASS' if scored[key]['PASS'] else 'FAIL'}")
    return scored


def run_commitment(
    config: Optional[CHLUConfig] = None, dataset: Optional[str] = None,
    pretext: Optional[str] = None, out_dir: Optional[str] = None,
    seeds: Optional[list] = None, stage: str = "commit",
    shortlist: Optional[list] = None, log: Callable = print,
) -> dict:
    config = config or get_default_config()
    cfg, c = config.gen_compose, _commit_cfg(config)
    dataset = dataset or cfg.dataset
    pretext = pretext or cfg.pretext
    root = Path(out_dir) if out_dir else (
        commit_out_root(config, dataset, pretext) / "runs"
        / artifacts_tag(dataset, pretext, condition_of(config)))
    seeds = list(seeds if seeds is not None else c.seeds)
    shortlist = list(shortlist if shortlist is not None else c.shortlist)
    if stage == "score" and len(shortlist) > int(c.max_shortlist):
        raise ValueError(
            f"⛔ the accuracy leg runs at a PRE-REGISTERED shortlist of at most "
            f"{c.max_shortlist} configs; got {len(shortlist)}")
    log(f"[gen-commit] dataset={dataset} pretext={pretext} stage={stage} seeds={seeds} "
        f"-> {root}")
    out = {}
    for s in seeds:
        d = root / f"seed_{s}"
        d.mkdir(parents=True, exist_ok=True)
        if stage == "commit":
            out[str(s)] = run_commit_seed(dataset, pretext, int(s), d, config, log=log)
        elif stage == "score":
            out[str(s)] = score_commit_seed(dataset, pretext, int(s), d, config,
                                            shortlist, log=log)
        else:
            raise ValueError(f"stage must be one of {COMMIT_STAGES}, got {stage!r}")
    return out


def main_commit(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="⭐ The commitment dial: 72 requests, 9 codes — and whether any knob undoes it.")
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument("--stage", default="commit", choices=list(COMMIT_STAGES))
    p.add_argument("--seed", type=int, default=None, action="append", help="repeatable")
    p.add_argument("--config", default=None, action="append",
                   help="shortlist entry, e.g. 'topk|tau=1|margin=0|k=2' (⛔ max 4)")
    p.add_argument("--out", default=None)
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--store-plan", default=None)
    args = p.parse_args(argv)
    config = get_default_config()
    if args.bridge_root:
        config.gen_compose.bridge_root = args.bridge_root
        config.gen_store.bridge_root = args.bridge_root
    if args.store_plan:
        config.gen_compose.store_plan = args.store_plan
    run_commitment(config=config, dataset=args.dataset, pretext=args.pretext,
                   out_dir=args.out, seeds=args.seed, stage=args.stage,
                   shortlist=args.config)
    return 0


# ⛔ kept at the file tail: the commitment stages (section 10) must be defined
# before ``main`` dispatches to them.
if __name__ == "__main__":
    sys.exit(main())
