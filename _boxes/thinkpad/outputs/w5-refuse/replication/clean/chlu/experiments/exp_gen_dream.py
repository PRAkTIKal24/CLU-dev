"""Experiment: **conditional dreaming** by clamped-well Langevin, with an audit
trail that can convict the sampler.

⭐ DIAL DECLARATION — *isolation*: does clamping the wells a request addresses
actually steer what comes out?
- **Laundering control:** ⭐ the **zero-step rollout** — decode the clamp directly,
  no Langevin, same clamp, same seeds.  ⛔ It runs on every reported number.
- **Falsifies:** zero-step matches the full rollout ⇒ the sampler is inert and we
  are decoding the clamp.  Also falsifying: flat energy traces, or a chain that
  never leaves its initialisation.
- **Does NOT falsify:** blurry or low-fidelity samples in absolute terms — the
  frozen decoder's measured ceiling bounds that, and it is read as a ratio (R3.2),
  never absolutely.

⭐ **The acceptance criterion is the CONTRAST, not the accuracy.**  A clamped-well
rollout that simply decodes its own clamp produces respectable accuracy on seen
combinations while performing no inference at all.

⛔ **A-CELL SMOKE ONLY.**  Seen combinations, reported as a fraction of the
bridge's ``ceiling_A`` **and** of the store ceiling (R23.5, two ceilings on every
table).  ⛔ No B, no C, no kill-rule verdict — the cell belongs to
``gen-cell-run``, after the prereg is filed.

⛔ **THE DECODER TRAP, CHECKED FIRST.**  ``chlu/config.py``'s ``dec_recipe``
defaults to ``"baseline"`` — the **superseded** decoder that FAILED the gate
(0.7191 balanced accuracy, 0/3 seeds).  The winner is ``v1_capacity_schedule``
(0.9414, 3/3).  This experiment loads ``gen-decoder-lift``'s winner checkpoint and
**asserts its parameter hash and recipe name** before any number is recorded; an
arm decoded through the baseline looks entirely normal in the logs and is void.

Runnable:
    chlu exp-gen-dream [--dataset shapes3d] [--pretext recon] [--quick]
    uv run python -m chlu.experiments.exp_gen_dream \\
        --dataset shapes3d --pretext recon --stage dream --arm clu --seed 0 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, get_default_config  # noqa: E402
from ..core.gen_bridge import (  # noqa: E402
    PRETEXTS,
    load_feature_cache,
    param_hash,
)
from ..core.gen_dream import (  # noqa: E402
    ARMS,
    DreamLandscape,
    address_requests,
    assert_dream_decoder,
    clamp_from_posterior,
    coupling_bytes,
    dream_rollout,
    fit_address_book,
    fit_code_coupling,
    init_state,
    load_winner_hashes,
    requests_from_labels,
    resolve_rollout_cfg,
    rollout_read,
    save_trace,
)
from ..core.gen_store import (  # noqa: E402
    address_codes,
    assert_no_bypass,
    channel_correlation,
    dead_channels,
    delete_store,
    grouping_labels,
    init_store,
    load_store_plan,
    store_bytes,
    store_read,
    train_store,
    well_channels_from_labels,
)
from ..eval.gen_probe import (  # noqa: E402
    factor_labels,
    multiclass_balanced_accuracy,
    predict_factors,
    decode_features,
)
from ..experiments.exp_gen_bridge import (  # noqa: E402
    artifacts_tag,
    condition_of,
    load_split_and_cells,
)
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    gather_features,
    select_store_rows,
)
from ..utils.checkpoints import load_model, save_model  # noqa: E402

STAGES = ("dream", "aggregate")
#: Kept so the CSF3 launcher's other --stage values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "store", "cell")
#: ⭐ The compositional term (R16).  The aggregate is reported alongside it, never
#: instead of it: four of six Shapes3D factors sit at ~1.0 and carry no
#: compositional information.
COMPOSITIONAL_FACTOR = "shape"


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's numbers are NEVER reported."""
    g = config.gen_dream
    g.seeds = list(g.quick_seeds)
    g.n_dream = g.quick_n_dream
    g.steps = g.quick_steps
    g.tune_rounds = g.quick_tune_rounds
    g.arms = list(g.quick_arms)
    g.samplers = list(g.quick_samplers)
    g.request_protocols = list(g.quick_request_protocols)
    config.gen_store.n_store_train = g.quick_n_store_train
    config.gen_store.train_steps = g.quick_store_train_steps
    return config


def bridge_root(config: CHLUConfig) -> Path:
    """Where the frozen bridge runs live.

    ⚠ A worktree has no ``.claude/``, so ``$CHLU_GEN_BRIDGE_ROOT`` must be set when
    running from one (the same trap as ``$CHLU_GEN_DATA_ROOT``)."""
    r = config.gen_dream.bridge_root or os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    if r:
        return Path(r).expanduser()
    return _claude_root(config) / "outputs" / "gen-feature-bridge" / "runs"


def _claude_root(config: CHLUConfig) -> Path:
    r = config.gen_dream.claude_root or os.environ.get("CHLU_CLAUDE_ROOT")
    if r:
        return Path(r).expanduser()
    return Path(__file__).resolve().parents[2] / ".claude"


def bridge_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    """⭐ Condition-aware: ``recomb2element`` keeps Montero's path, a secondary
    condition gets its own (``artifacts_tag``), so the same code re-points at the
    widened configuration without an edit."""
    tag = artifacts_tag(dataset, pretext, condition_of(config))
    d = bridge_root(config) / tag / f"seed_{seed}"
    if not (d / "bridge_manifest.json").exists():
        raise FileNotFoundError(
            f"no frozen bridge at {d} — this spoke consumes the FROZEN artifacts "
            "and ⛔ never rebuilds them (set gen_dream.bridge_root or "
            "$CHLU_GEN_BRIDGE_ROOT)"
        )
    return d


def store_plan_path(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    """``gen-store-grouping``'s pinned plan — ⛔ loaded verbatim, never re-derived."""
    p = config.gen_dream.store_plan
    if p:
        return Path(p).expanduser()
    return (
        _claude_root(config) / "outputs" / "gen-store-grouping" / "runs"
        / f"{dataset}_{pretext}" / "store_plan.json"
    )


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# ⛔ 1. The decoder — resolved and hash-asserted BEFORE anything is measured
# ---------------------------------------------------------------------------


def resolve_decoder(config: CHLUConfig, bdir: Path, seed: int, log: Callable = print):
    """Load THE shared frozen decoder for this seed and **prove** it is the winner.

    ``decoder_source="winner"`` (default) loads ``gen-decoder-lift``'s
    ``winner/decoder_v1_s<seed>.pkl`` and asserts its parameter hash against that
    spoke's own ``winner_hashes.json``.  ``"bridge"`` loads the bridge directory's
    ``decoder.pkl`` and asserts against ``bridge_manifest.json`` — correct for a
    bridge that was itself fitted at the winning recipe (the wide configuration
    was), ⛔ and refused for one that was not (Montero's bridge decoder is the
    superseded baseline).
    """
    cfg = config.gen_dream
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    source = str(cfg.decoder_source)
    if source == "bridge":
        recipe = (bman.get("training") or {}).get("dec_recipe") or {}
        name = recipe.get("name") if isinstance(recipe, dict) else recipe
        if name is None:
            raise AssertionError(
                f"{bdir}/bridge_manifest.json records no decoder recipe, so this "
                "decoder cannot be proven to be the gate winner. ⛔ Refusing: an "
                "arm decoded through the superseded 'baseline' recipe is void and "
                "looks entirely normal in the logs. Use decoder_source='winner'."
            )
        dec = load_model(bdir / "decoder.pkl")
        info = assert_dream_decoder(
            dec, bman["decoder_param_hash"], name, cfg.expected_decoder_recipe
        )
        info["decoder_path"] = str(bdir / "decoder.pkl")
        info["decoder_source"] = "bridge"
    elif source == "winner":
        root = Path(cfg.winner_root).expanduser() if cfg.winner_root else (
            _claude_root(config) / "outputs" / "gen-decoder-lift" / "winner"
        )
        hashes = load_winner_hashes(root / "winner_hashes.json")
        rec = hashes[f"seed_{int(seed)}"]
        path = root / cfg.winner_decoder_pattern.format(seed=int(seed))
        dec = load_model(path)
        info = assert_dream_decoder(
            dec, rec["param_hash"], rec["recipe"]["name"], cfg.expected_decoder_recipe
        )
        info["decoder_path"] = str(path)
        info["decoder_source"] = "winner"
        # ⛔ the winner was fitted against ONE encoder; decoding another encoder's
        # features through it is meaningless and must not pass silently.
        enc_expect = rec.get("encoder_param_hash_it_was_fit_against")
        if enc_expect and enc_expect != bman["encoder_param_hash"]:
            raise AssertionError(
                "the decoder-lift winner for this seed was fitted against encoder "
                f"{enc_expect}, but the bridge here is {bman['encoder_param_hash']} "
                "— ⛔ these are different feature spaces"
            )
    else:
        raise ValueError(f"decoder_source must be 'winner' or 'bridge', got {source!r}")
    info["param_count"] = int(
        sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(dec))
    )
    log(
        f"  ⭐ decoder: {info['decoder_source']} recipe={info['decoder_recipe']} "
        f"hash={info['decoder_param_hash'][:16]}… (asserted)"
    )
    return dec, info, bman


# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------


def score_requests(
    probe, images: np.ndarray, requests: np.ndarray, truth: np.ndarray,
    names, cardinalities, batch: int,
) -> dict:
    """Per-factor accuracy of the decoded images **against the request**.

    ⭐ A factor only scores where it was actually requested (``request >= 0``), so
    the ``partial`` protocol is scored on what was asked for and never on what was
    deliberately left free.  Balanced accuracy is reported alongside top-1 because
    R23.1 makes balanced accuracy the gate statistic; ⛔ neither is a cell number
    here — this is an A-split smoke.
    """
    preds = predict_factors(probe, np.asarray(images, np.float32), int(batch))
    R = np.asarray(requests, np.int64)
    acc, bal, n_scored = {}, {}, {}
    for f, nm in enumerate(names):
        sel = R[:, f] >= 0
        n_scored[str(nm)] = int(sel.sum())
        if not sel.any():
            acc[str(nm)] = float("nan")
            bal[str(nm)] = float("nan")
            continue
        acc[str(nm)] = float((preds[sel, f] == R[sel, f]).mean())
        bal[str(nm)] = float(
            multiclass_balanced_accuracy(
                R[sel, f], preds[sel, f], int(cardinalities[f])
            )["balanced_accuracy"]
        )
    vals = [v for v in acc.values() if np.isfinite(v)]
    bvals = [v for v in bal.values() if np.isfinite(v)]
    # ⚠ the free factors, reported separately and never pooled into the aggregate
    free = {}
    for f, nm in enumerate(names):
        sel = R[:, f] < 0
        if sel.any():
            free[str(nm)] = float((preds[sel, f] == np.asarray(truth)[sel, f]).mean())
    return {
        "n": int(R.shape[0]),
        "per_factor": acc,
        "per_factor_balanced": bal,
        "n_scored_per_factor": n_scored,
        "aggregate": float(np.mean(vals)) if vals else float("nan"),
        "aggregate_balanced": float(np.mean(bvals)) if bvals else float("nan"),
        "free_factor_accuracy_vs_truth": free,
        "decoded_mean": float(np.asarray(images).mean()),
    }


def _decode_and_score(decoder, probe, feats, requests, truth, names, cards, batch):
    imgs = decode_features(decoder, np.asarray(feats, np.float32), int(batch))
    return score_requests(probe, imgs, requests, truth, names, cards, batch), imgs


# ---------------------------------------------------------------------------
# 3. ⭐ Step-size tuning — on TRAIN requests, to an ACCEPTANCE target
# ---------------------------------------------------------------------------


def tune_step_size(land, x0, coord_clamped, cfg, run_cfg, key, log: Callable = print) -> dict:
    """Pick the MALA step size by targeting an **acceptance rate**, on train rows.

    ⛔ Outcome-independent: the objective is a sampler diagnostic (acceptance), not
    an accuracy, and it is computed on rows drawn from the store's own training
    pool — no eval row, no held-out row and no probe reading enters it.  ⭐ The
    whole search trajectory is returned and written to the manifest.

    ⚠ **Tuned PER PROTOCOL.**  ``full`` and ``partial`` differ several-fold in the
    dimension of the free block, and MALA's acceptance is dimension-sensitive: the
    first run reused one step size across both and measured acceptance 0.592 on
    ``full`` and **0.052** on ``partial`` at the same ``eps``.
    """
    eps = float(cfg.step_size_init)
    target = float(cfg.target_accept)
    tol = float(cfg.accept_tolerance)
    hist = []
    probe_cfg = run_cfg.replace(steps=int(cfg.tune_steps))
    for _r in range(int(cfg.tune_rounds)):
        key, sk = jax.random.split(key)
        out = dream_rollout(
            land, x0, coord_clamped, probe_cfg.replace(step_size=eps), sk, sampler="mala"
        )
        a = float(out["trace"]["summary"]["mean_accept_prob"])
        hist.append({"step_size": float(eps), "mean_accept_prob": a})
        if a > target + tol:
            eps *= float(cfg.tune_factor)
        elif a < target - tol:
            eps /= float(cfg.tune_factor)
        else:
            break
    # ⚠ the acceptance can step straight over the tolerance band (measured:
    # 0.05 -> 0.41 -> 0.82 at successive halvings), in which case the search
    # oscillates and the LAST eps is not the best one. Take the closest to target.
    best = min(hist, key=lambda h: abs(h["mean_accept_prob"] - target))
    eps = float(best["step_size"])
    log(
        f"  ⭐ step size tuned to {eps:.4g} (target accept {target}, "
        f"{len(hist)} round(s): "
        + ", ".join(f"{h['step_size']:.3g}->{h['mean_accept_prob']:.2f}" for h in hist)
        + ")"
    )
    return {"step_size": float(eps), "target_accept": target, "history": hist,
            "tune_steps": int(cfg.tune_steps),
            "n_tune_requests": int(np.asarray(x0).shape[0])}


# ---------------------------------------------------------------------------
# 4. One seed
# ---------------------------------------------------------------------------


def _arm_landscape(store, coupling, arm: str, cfg, seed: int):
    """The landscape each arm runs on.  ⭐ Identical code path; only the store or
    the coupling strength differs."""
    if arm == "clu":
        return DreamLandscape(store, coupling, float(cfg.coupling_strength)), store
    if arm == "no_coupling":
        # ⭐ the SEPARABILITY control: lambda = 0 makes the energy additive over
        # wells, so the clamp provably cannot steer a relaxed well.
        return DreamLandscape(store, coupling, 0.0), store
    if arm == "store_removed":
        # ⭐ the ablation this spoke owns: the identical path with the store's
        # CONTENT deleted. ``mean`` is the fairest deletion (gen-store-grouping §4):
        # the read stays an on-manifold, correctly-scaled feature vector carrying
        # zero information, so the decoder is not merely handed garbage.
        gone = delete_store(store, str(cfg.store_removed_delete_mode), seed=int(seed))
        return DreamLandscape(gone, coupling, float(cfg.coupling_strength)), gone
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def resolve_grouping(plan, X_store, config: CHLUConfig, seed: int,
                     encoder_hash: str, log: Callable = print) -> tuple:
    """⛔⭐ The grouping, resolved per seed — and why it CANNOT be the plan's vector.

    ``store_plan.json``'s ``labels`` vector indexes **encoder channels**, and the
    plan was written from the ``k_choice_seed``'s encoder alone (``i == 0`` in
    ``exp_gen_store``).  Every bridge seed trained its **own** encoder, so channel
    ``c`` of seed 1 has no relation to channel ``c`` of seed 0: applying the plan's
    labels to another seed is a size-matched **random** partition wearing the
    clustered grouping's name.  ⛔ ``gen-store-grouping`` itself did not do this --
    it re-derived the grouping per seed -- but the plan's "load this VERBATIM"
    wording invites exactly that mistake.

    What IS pinned by the plan is what ``assert_plan_matched`` actually guards:
    ``K``, ``n_modes``, ``D`` and ``store_bytes``.  The plan's own note says only
    ``labels`` may change.  So the grouping is re-derived here by the *same*
    procedure, on this seed's own train activations, at the plan's ``K``.

    ⭐ And the re-derivation is **verified, not asserted**: on the seed whose
    encoder hash matches the plan's, the labels must come back **bit-identical to
    the plan's**, which proves this is the same procedure that wrote it.
    """
    cfg = config.gen_dream
    K = int(plan["K"])
    if str(cfg.grouping_source) == "plan":
        labels = np.asarray(plan["labels"], np.int64)
        return labels, {
            "grouping_source": "plan (VERBATIM)",
            "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
            "warning": (
                "⛔ the plan's labels index the k_choice seed's encoder channels; "
                "on any other seed this is a size-matched RANDOM partition"
            ),
        }
    if str(cfg.grouping_source) != "per_seed":
        raise ValueError(
            f"grouping_source must be 'per_seed' or 'plan', got {cfg.grouping_source!r}"
        )
    gs = config.gen_store
    Xc = np.asarray(X_store, np.float32)[: int(gs.n_corr)]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    labels, info = grouping_labels(
        corr, K, "correlation", int(seed), gs.grouping_method,
        dead=dead, min_well_frac=float(gs.min_well_frac),
    )
    out = {
        "grouping_source": "per_seed (re-derived at the plan's K)",
        "n_corr_rows": int(Xc.shape[0]),
        "n_dead_channels": int(dead.sum()),
        "grouping_info": info,
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
        "plan_encoder_param_hash": plan.get("encoder_param_hash"),
        "this_encoder_param_hash": str(encoder_hash),
    }
    if str(encoder_hash) == str(plan.get("encoder_param_hash")):
        same = bool(np.array_equal(labels, np.asarray(plan["labels"], np.int64)))
        out["reproduces_the_plan_bit_identically"] = same
        if not same:
            raise AssertionError(
                "the per-seed grouping does NOT reproduce store_plan.json's labels "
                "on the seed whose encoder wrote it — ⛔ this is not the same "
                "procedure and the ladder comparison would be void"
            )
        log("  ⭐ grouping re-derivation reproduces the plan BIT-IDENTICALLY on its own seed")
    else:
        out["reproduces_the_plan_bit_identically"] = None
        log(
            "  ⚠ this seed's encoder differs from the plan's "
            f"({str(encoder_hash)[:12]}… vs {str(plan.get('encoder_param_hash'))[:12]}…), "
            "so the grouping is re-derived on its own channels — ⛔ the plan's label "
            "vector would be a size-matched RANDOM partition here"
        )
    return labels, out


def run_seed(
    dataset: str,
    pretext: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    arms: Optional[list] = None,
    log: Callable = print,
) -> dict:
    """Fit/load the store, build the landscape, and run every (protocol, arm,
    sampler, variant) cell — writing results **incrementally**."""
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    cfg = config.gen_dream
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces"
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

    # ---- rows, split, cache -------------------------------------------------
    split, cells = load_split_and_cells(dataset, config, log=tlog)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_store_rows(split, config, seed)
    crows, feats, cman = load_feature_cache(bdir, verify=bool(cfg.verify_cache_checksum))
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    tlog(
        f"cache {cman['n_rows']} rows (scope={cman['scope']}), D={cman['D']}, "
        f"{cman['dtype']}"
    )
    reenc = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=tlog)

    n_dream = int(min(int(cfg.n_dream), rows["eval_a"].size))
    dream_rows = rows["eval_a"][:n_dream]
    tlog(
        f"rows: store_train={rows['store_train'].size} eval_a={rows['eval_a'].size} "
        f"-> {n_dream} A-split requests (⛔ A only: no B, no C)"
    )
    X_store = gather_features(dataset, config, encoder, crows, feats, cman,
                              rows["store_train"], log=tlog)
    X_dream = gather_features(dataset, config, encoder, crows, feats, cman,
                              dream_rows, log=tlog)
    lab_store = factor_labels(grid, rows["store_train"])
    lab_dream = factor_labels(grid, dream_rows)
    tlog(f"features gathered: store {X_store.shape} dream {X_dream.shape}")

    # ---- the store: the PINNED plan, then the grouping, then train-or-load ---
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    K = int(plan["K"])
    D = int(np.prod(feat_shape))
    if len(plan["labels"]) != D:
        raise AssertionError(
            f"the store plan's label vector is {len(plan['labels'])} long but D = "
            f"{D} — ⛔ this plan was not written for this cache"
        )
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
    sb = store_bytes(store)
    tlog(f"store: {sb['store_bytes']} B, {sb['bits_per_read']:.0f} bits/read")

    # ⛔ the structural no-bypass proof, on the dream path's own store
    bypass = assert_no_bypass(store, X_dream[:64], rng_seed=int(seed))
    tlog(f"⭐ no-bypass: pure codebook gather, {bypass['bits_per_read']:.0f} bits/read")

    # ---- the landscape's two learned pieces --------------------------------
    codes_train = address_codes(store, np.asarray(X_store, np.float32).reshape(
        (X_store.shape[0], -1)))
    coupling = fit_code_coupling(codes_train, int(plan["n_modes"]),
                                 float(cfg.coupling_smoothing))
    book = fit_address_book(codes_train, lab_store, cards, float(cfg.address_smoothing))
    prior = np.exp(np.asarray(book.log_prior))
    cbytes = coupling_bytes(K, int(plan["n_modes"]))
    tlog(
        f"⭐ coupling fitted from {codes_train.shape[0]} train codes "
        f"(+{cbytes['coupling_bytes']} B, EXTRA to store_bytes; channel unchanged at "
        f"{cbytes['bits_per_read_unchanged']:.0f} bits)"
    )

    # ---- ⭐ THE TWO CEILINGS (R23.5) ----------------------------------------
    ceil_dec, _ = _decode_and_score(
        decoder, probe, X_dream.reshape((-1,) + feat_shape), lab_dream, lab_dream,
        names, cards, cfg.eval_batch,
    )
    read_real = store_read(store, X_dream, int(cfg.eval_batch))
    ceil_store, _ = _decode_and_score(
        decoder, probe, read_real, lab_dream, lab_dream, names, cards, cfg.eval_batch,
    )
    tlog(
        f"⭐ ceilings: decoder aggregate={ceil_dec['aggregate']:.4f} "
        f"{COMPOSITIONAL_FACTOR}={ceil_dec['per_factor'][COMPOSITIONAL_FACTOR]:.4f} | "
        f"STORE aggregate={ceil_store['aggregate']:.4f} "
        f"{COMPOSITIONAL_FACTOR}={ceil_store['per_factor'][COMPOSITIONAL_FACTOR]:.4f}"
    )

    manifest = {
        "schema": "chlu-gen-dream/1",
        "dataset": dataset,
        "pretext": pretext,
        "condition": condition,
        "seed": int(seed),
        "bridge_dir": str(bdir),
        "decoder": dec_info,
        "probe_param_hash": bman["probe_param_hash"],
        "encoder_param_hash": bman["encoder_param_hash"],
        "feature_cache_sha256": cman["features_sha256"],
        "feature_cache_scope": cman["scope"],
        "cache_reencode_check": reenc,
        "store_plan_path": str(store_plan_path(config, dataset, pretext)),
        "K": K,
        "grouping": ginfo,
        "n_modes": int(plan["n_modes"]),
        "store_bytes": sb,
        "coupling_bytes": cbytes,
        "no_bypass": bypass,
        "rows": {
            "store_train": int(rows["store_train"].size),
            "dream_requests": int(n_dream),
            "eval_a": int(rows["eval_a"].size),
        },
        "store_training": {
            "steps": int(config.gen_store.train_steps),
            "loss_first": float(hist[0]) if hist else None,
            "loss_final": float(hist[-1]) if hist else None,
            "loaded_from_checkpoint": bool(not hist),
        },
        "ceiling_decoder": ceil_dec,
        "ceiling_store": ceil_store,
        "cells": [],
    }
    _write_json(manifest, out_dir / "dream_manifest.json")

    # ---- the cells ---------------------------------------------------------
    arms = list(arms or cfg.arms)
    land_tune = DreamLandscape(store, coupling, float(cfg.coupling_strength))
    n_tune = int(min(int(cfg.n_tune_requests), lab_store.shape[0]))
    manifest["step_size_tuning"] = {}
    for protocol in list(cfg.request_protocols):
        # ⭐ the step size is tuned PER PROTOCOL, on TRAIN requests, to an
        # ACCEPTANCE target (⛔ never to an accuracy) — the free block's dimension
        # differs several-fold between the protocols and MALA is dimension-sensitive.
        req_tune = requests_from_labels(
            lab_store[:n_tune], protocol, int(cfg.request_drop_factors), int(seed)
        )
        clamp_tune = clamp_from_posterior(
            address_requests(book, req_tune), float(cfg.clamp_threshold)
        )
        init_tune = init_state(store, clamp_tune, prior, str(cfg.init_mode), int(seed))
        base_cfg = resolve_rollout_cfg(cfg, store, land_tune, init_tune["x0"])
        sat = base_cfg.calibration["saturation_at_tau_start"]["mean_max_pi"]
        tlog(
            f"[{protocol}] calibrated: tau {base_cfg.tau_start:.4g} -> "
            f"{base_cfg.tau_end:.4g} ({cfg.tau_calibration}), T "
            f"{base_cfg.temp_start:.4g} -> {base_cfg.temp_end:.4g} "
            f"({cfg.temp_calibration}); mean_max_pi at tau_start = {sat:.4f}"
            + ("  ⛔ SATURATED" if sat > 0.99 else "")
        )
        if cfg.step_size is None:
            tune = tune_step_size(
                land_tune, init_tune["x0"], init_tune["coord_clamped"], cfg, base_cfg,
                jax.random.PRNGKey(int(seed) * 977 + 3), log=tlog,
            )
        else:
            tune = {"step_size": float(cfg.step_size), "history": [],
                    "note": "pinned by gen_dream.step_size (no tuning)"}
        run_cfg = base_cfg.replace(step_size=tune["step_size"])
        manifest["step_size_tuning"][protocol] = {
            **tune, "calibration": base_cfg.calibration,
            "rollout_cfg": run_cfg.as_dict(),
        }
        _write_json(manifest, out_dir / "dream_manifest.json")

        req = requests_from_labels(
            lab_dream, protocol, int(cfg.request_drop_factors), int(seed)
        )
        clamp = clamp_from_posterior(
            address_requests(book, req), float(cfg.clamp_threshold)
        )
        tlog(
            f"[{protocol}] mean_n_clamped={clamp['mean_n_clamped']:.2f}/{K} "
            f"mean_n_relaxed={clamp['mean_n_relaxed']:.2f} "
            f"mean_confidence={clamp['mean_confidence']:.3f}"
        )
        # ⭐ the two ceilings, scored under THIS protocol's request mask
        ceil_d_p = score_requests(probe, decode_features(
            decoder, X_dream.reshape((-1,) + feat_shape), int(cfg.eval_batch)),
            req, lab_dream, names, cards, cfg.eval_batch)
        ceil_s_p = score_requests(probe, decode_features(
            decoder, read_real, int(cfg.eval_batch)),
            req, lab_dream, names, cards, cfg.eval_batch)

        inits = {
            "full": init_state(store, clamp, prior, str(cfg.init_mode), int(seed)),
            "zero_step": init_state(store, clamp, prior, str(cfg.init_mode), int(seed)),
            "prior_mode": init_state(store, clamp, prior, "prior_mode", int(seed)),
        }
        for arm in arms:
            land, arm_store = _arm_landscape(store, coupling, arm, cfg, seed)
            samplers = list(cfg.samplers) if arm == "clu" else [str(cfg.sampler)]
            for sampler in samplers:
                for variant in list(cfg.variants):
                    if variant != "full" and sampler != str(cfg.sampler):
                        continue  # the controls are sampler-independent by definition
                    ini = inits[variant]
                    steps = int(run_cfg.steps) if variant == "full" else 0
                    # ⛔ NOT Python's hash(): str hashing is randomised per
                    # process (PYTHONHASHSEED), which would make the rollout
                    # irreproducible across runs at a fixed seed.
                    key = jax.random.PRNGKey(
                        int(seed) * 1_000_003
                        + int(zlib.crc32(f"{protocol}|{arm}|{sampler}|{variant}".encode()) % 100_003)
                    )
                    out = dream_rollout(
                        land, ini["x0"], ini["coord_clamped"], run_cfg, key,
                        steps=steps, sampler=sampler,
                    )
                    codes, read = rollout_read(arm_store, out["x"])
                    sc, imgs = _decode_and_score(
                        decoder, probe, read, req, lab_dream, names, cards, cfg.eval_batch
                    )
                    tag = f"{protocol}__{arm}__{sampler}__{variant}"
                    tp = None
                    if bool(cfg.save_traces):
                        tp = save_trace(out["trace"], trace_dir / f"{tag}_s{seed}.npz")
                    s = out["trace"]["summary"]
                    cell = {
                        "protocol": protocol,
                        "arm": arm,
                        "sampler": sampler,
                        "variant": variant,
                        "seed": int(seed),
                        "steps": int(steps),
                        "score": sc,
                        "sampler_summary": s,
                        "clamp": {
                            k: clamp[k] for k in
                            ("mean_n_clamped", "mean_n_relaxed", "mean_confidence", "threshold")
                        },
                        "init_mode": ini["init_mode"],
                        "rollout_cfg": run_cfg.as_dict(),
                        "trace_path": tp,
                        "n_distinct_code_configs": int(
                            np.unique(codes, axis=0).shape[0]
                        ),
                        "ceiling_decoder_this_protocol": ceil_d_p,
                        "ceiling_store_this_protocol": ceil_s_p,
                    }
                    manifest["cells"].append(cell)
                    _write_json(manifest, out_dir / "dream_manifest.json")  # ⭐ incremental
                    tlog(
                        f"  {tag}: aggregate={sc['aggregate']:.4f} "
                        f"bal={sc['aggregate_balanced']:.4f} "
                        f"{COMPOSITIONAL_FACTOR}={sc['per_factor'][COMPOSITIONAL_FACTOR]:.4f} "
                        f"| acc_prob={s['mean_accept_prob']:.3f} "
                        f"ΔE={s['energy_relative_drop']:+.4f} moved={s['frac_rollouts_that_moved']:.2f} "
                        f"maxpi={s['mean_max_pi_first']:.3f} "
                        f"cfgs={cell['n_distinct_code_configs']}"
                    )
    manifest["contrast"] = seed_contrast(manifest)
    _write_json(manifest, out_dir / "dream_manifest.json")
    if cfg.make_plots:
        try:
            _plot_seed(manifest, out_dir / "dream.png")
        except Exception as exc:  # a plot is never allowed to lose a result
            tlog(f"  ⚠ plot failed (results are already on disk): {exc}")
    tlog(f"seed {seed} done")
    return manifest


# ---------------------------------------------------------------------------
# 5. ⭐ The contrast — full rollout vs the zero-step control
# ---------------------------------------------------------------------------


def _cell(manifest: dict, protocol: str, arm: str, sampler: str, variant: str):
    for c in manifest["cells"]:
        if (c["protocol"], c["arm"], c["sampler"], c["variant"]) == (
            protocol, arm, sampler, variant
        ):
            return c
    return None


def _variant_cell(manifest: dict, protocol: str, arm: str, variant: str):
    for c in manifest["cells"]:
        if (c["protocol"], c["arm"], c["variant"]) == (protocol, arm, variant):
            return c
    return None


def seed_contrast(manifest: dict) -> dict:
    """⭐ full − zero_step, per (protocol, arm, sampler).  **The acceptance criterion.**"""
    out = {}
    seen = {(c["protocol"], c["arm"], c["sampler"]) for c in manifest["cells"]}
    for protocol, arm, sampler in sorted(seen):
        full = _cell(manifest, protocol, arm, sampler, "full")
        # ⭐ the controls are sampler-independent by definition (0 steps), so they
        # are stored once per (protocol, arm) under the primary sampler.
        zero = _variant_cell(manifest, protocol, arm, "zero_step")
        pm = _variant_cell(manifest, protocol, arm, "prior_mode")
        if full is None or zero is None:
            continue
        key = f"{protocol}__{arm}__{sampler}"
        ceil_d = full["ceiling_decoder_this_protocol"]["aggregate"]
        ceil_s = full["ceiling_store_this_protocol"]["aggregate"]
        out[key] = {
            "protocol": protocol,
            "arm": arm,
            "sampler": sampler,
            "full_aggregate": full["score"]["aggregate"],
            "zero_step_aggregate": zero["score"]["aggregate"],
            "prior_mode_aggregate": None if pm is None else pm["score"]["aggregate"],
            "delta_aggregate": float(full["score"]["aggregate"] - zero["score"]["aggregate"]),
            "full_compositional": full["score"]["per_factor"][COMPOSITIONAL_FACTOR],
            "zero_step_compositional": zero["score"]["per_factor"][COMPOSITIONAL_FACTOR],
            "delta_compositional": float(
                full["score"]["per_factor"][COMPOSITIONAL_FACTOR]
                - zero["score"]["per_factor"][COMPOSITIONAL_FACTOR]
            ),
            "delta_aggregate_balanced": float(
                full["score"]["aggregate_balanced"] - zero["score"]["aggregate_balanced"]
            ),
            "fraction_of_decoder_ceiling": float(full["score"]["aggregate"] / ceil_d)
            if ceil_d else float("nan"),
            "fraction_of_store_ceiling": float(full["score"]["aggregate"] / ceil_s)
            if ceil_s else float("nan"),
            "mean_n_relaxed": full["clamp"]["mean_n_relaxed"],
            "frac_rollouts_that_moved": full["sampler_summary"]["frac_rollouts_that_moved"],
            "mean_accept_prob": full["sampler_summary"]["mean_accept_prob"],
            "energy_relative_drop": full["sampler_summary"]["energy_relative_drop"],
            "mean_max_pi": full["sampler_summary"].get("mean_max_pi_first"),
            "n_distinct_code_configs_full": full["n_distinct_code_configs"],
            "n_distinct_code_configs_zero_step": zero["n_distinct_code_configs"],
            "SAMPLER_IS_INERT": bool(
                abs(full["score"]["aggregate"] - zero["score"]["aggregate"]) < 0.01
            ),
            "relax_set_is_degenerate": bool(full["clamp"]["mean_n_relaxed"] < 1.0),
        }
    return out


def _ms(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    return {
        "mean": float(v.mean()) if v.size else float("nan"),
        "std": float(v.std()) if v.size else float("nan"),
        "n": int(v.size),
        "per_seed": [float(x) for x in v],
    }


def aggregate_runs(run_dirs, log: Callable = print) -> dict:
    """Pool the per-seed manifests: the contrast with its spread, and the falsifiers."""
    mans = []
    for d in run_dirs:
        p = Path(d) / "dream_manifest.json"
        if p.exists():
            with open(p) as fh:
                mans.append(json.load(fh))
    if not mans:
        raise FileNotFoundError("no per-seed dream manifests found")
    keys = sorted({k for m in mans for k in m.get("contrast", {})})
    table = {}
    for k in keys:
        rows = [m["contrast"][k] for m in mans if k in m.get("contrast", {})]
        table[k] = {
            "protocol": rows[0]["protocol"],
            "arm": rows[0]["arm"],
            "sampler": rows[0]["sampler"],
            "n_seeds": len(rows),
            "full_aggregate": _ms([r["full_aggregate"] for r in rows]),
            "zero_step_aggregate": _ms([r["zero_step_aggregate"] for r in rows]),
            "prior_mode_aggregate": _ms([r["prior_mode_aggregate"] for r in rows]),
            "delta_aggregate": _ms([r["delta_aggregate"] for r in rows]),
            "delta_compositional": _ms([r["delta_compositional"] for r in rows]),
            "delta_aggregate_balanced": _ms([r["delta_aggregate_balanced"] for r in rows]),
            "fraction_of_decoder_ceiling": _ms([r["fraction_of_decoder_ceiling"] for r in rows]),
            "fraction_of_store_ceiling": _ms([r["fraction_of_store_ceiling"] for r in rows]),
            "mean_n_relaxed": _ms([r["mean_n_relaxed"] for r in rows]),
            "n_seeds_positive": int(sum(1 for r in rows if r["delta_aggregate"] > 0)),
            "frac_rollouts_that_moved": _ms([r.get("frac_rollouts_that_moved") for r in rows]),
            "mean_accept_prob": _ms([r.get("mean_accept_prob") for r in rows]),
            "energy_relative_drop": _ms([r.get("energy_relative_drop") for r in rows]),
            "mean_max_pi": _ms([r.get("mean_max_pi") for r in rows]),
            "relax_set_is_degenerate": bool(all(r["relax_set_is_degenerate"] for r in rows)),
        }
    # ⭐ the headline protocol is chosen by a PRE-REGISTERED rule, not by outcome:
    # the protocol whose mean relaxed-well count is >= 1; ties go to ``full``.
    headline_protocol = None
    for p in ("full", "partial"):
        cand = [v for key, v in table.items()
                if v["protocol"] == p and v["arm"] == "clu"]
        if cand and cand[0]["mean_n_relaxed"]["mean"] >= 1.0:
            headline_protocol = p
            break
    falsifier = {}
    if headline_protocol:
        for key, v in table.items():
            if v["protocol"] == headline_protocol and v["arm"] == "clu":
                d = v["delta_aggregate"]
                falsifier[key] = {
                    "delta_aggregate_mean": d["mean"],
                    "delta_aggregate_std": d["std"],
                    "n_seeds_positive": v["n_seeds_positive"],
                    "PREREG_P1_threshold": 0.03,
                    "PREREG_P1_holds": bool(d["mean"] >= 0.03 and v["n_seeds_positive"] >= 2),
                    "SAMPLER_IS_INERT": bool(abs(d["mean"]) < 0.01),
                }
    agg = {
        "schema": "chlu-gen-dream-aggregate/1",
        "dataset": mans[0]["dataset"],
        "pretext": mans[0]["pretext"],
        "condition": mans[0]["condition"],
        "seeds": [m["seed"] for m in mans],
        "K": mans[0]["K"],
        "n_modes": mans[0]["n_modes"],
        "store_bytes": mans[0]["store_bytes"],
        "coupling_bytes": mans[0]["coupling_bytes"],
        "decoder": {m["seed"]: m["decoder"] for m in mans},
        "ceiling_decoder_aggregate": _ms([m["ceiling_decoder"]["aggregate"] for m in mans]),
        "ceiling_store_aggregate": _ms([m["ceiling_store"]["aggregate"] for m in mans]),
        "ceiling_decoder_compositional": _ms(
            [m["ceiling_decoder"]["per_factor"][COMPOSITIONAL_FACTOR] for m in mans]
        ),
        "ceiling_store_compositional": _ms(
            [m["ceiling_store"]["per_factor"][COMPOSITIONAL_FACTOR] for m in mans]
        ),
        "headline_protocol": headline_protocol,
        "headline_protocol_rule": (
            "PRE-REGISTERED (PREREG §4): the protocol whose mean relaxed-well count "
            "is >= 1; ties go to 'full'. ⛔ Not chosen by outcome."
        ),
        "table": table,
        "falsifier_sampler_is_inert": falsifier,
        "step_size": {
            m["seed"]: {
                p: t.get("step_size") for p, t in (m.get("step_size_tuning") or {}).items()
            }
            for m in mans
        },
        "calibration": {
            m["seed"]: {
                p: {
                    k: t.get("calibration", {}).get(k)
                    for k in ("tau_calibration", "tau_start_abs", "tau_end_abs",
                              "temp_calibration", "temp_start_abs", "temp_end_abs",
                              "energy_scale")
                }
                for p, t in (m.get("step_size_tuning") or {}).items()
            }
            for m in mans
        },
        "occupancy_saturation": {
            m["seed"]: {
                f"{c['protocol']}__{c['arm']}__{c['sampler']}__{c['variant']}":
                    c["sampler_summary"].get("mean_max_pi_first")
                for c in m["cells"]
            }
            for m in mans
        },
    }
    log(f"aggregated {len(mans)} seed(s); headline protocol = {headline_protocol}")
    return agg


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------


def _plot_seed(man: dict, path: Path) -> None:
    """Energy traces + the full-vs-zero-step contrast.  ⛔ The figure is a
    convenience; the numbers live in the manifest and the traces on disk."""
    cells = [c for c in man["cells"] if c["variant"] == "full" and c["trace_path"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for c in cells:
        try:
            z = np.load(c["trace_path"])
        except Exception:
            continue
        e = np.asarray(z["energy_ref"]).mean(axis=1)
        ax.plot(e, label=f"{c['protocol']}/{c['arm']}/{c['sampler']}", lw=1.2)
    ax.set_xlabel("Langevin step")
    ax.set_ylabel("energy at fixed tau_end (mean over requests)")
    ax.set_title("energy trace (⛔ never the annealed energy)")
    ax.legend(fontsize=6)
    ax = axes[1]
    con = man.get("contrast", {})
    ks = sorted(con)
    xs = np.arange(len(ks))
    ax.bar(xs - 0.2, [con[k]["zero_step_aggregate"] for k in ks], 0.4, label="zero-step")
    ax.bar(xs + 0.2, [con[k]["full_aggregate"] for k in ks], 0.4, label="full rollout")
    ax.set_xticks(xs)
    ax.set_xticklabels(ks, rotation=90, fontsize=5)
    ax.set_ylabel("A-split aggregate")
    ax.axhline(man["ceiling_store"]["aggregate"], ls="--", c="k", lw=0.8, label="store ceiling")
    ax.axhline(man["ceiling_decoder"]["aggregate"], ls=":", c="k", lw=0.8, label="decoder ceiling")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------


def run_experiment_gen_dream(
    config: Optional[CHLUConfig] = None,
    dataset: Optional[str] = None,
    pretext: Optional[str] = None,
    out_dir: str = "results/gen_dream",
    seeds: Optional[list] = None,
    stage: str = "dream",
    arms: Optional[list] = None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_dream
    dataset = dataset or cfg.dataset
    pretext = pretext or cfg.pretext
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}, got {pretext!r}")
    if stage in NOT_MINE:
        log(f"[gen-dream] stage {stage!r} belongs to another spoke — nothing to do")
        return {}
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    condition = condition_of(config)
    tag = artifacts_tag(dataset, pretext, condition)
    root = Path(out_dir) / tag
    seeds = list(seeds if seeds is not None else cfg.seeds)
    log(
        f"[gen-dream] dataset={dataset} pretext={pretext} condition={condition} "
        f"stage={stage} seeds={seeds} -> {root}"
    )
    run_dirs = [root / f"seed_{s}" for s in seeds]
    if stage == "dream":
        for s, d in zip(seeds, run_dirs, strict=True):
            run_seed(dataset, pretext, int(s), d, config, arms=arms, log=log)
    agg = aggregate_runs(run_dirs, log=log)
    _write_json(agg, root / "dream_results.json")
    return agg


# ---------------------------------------------------------------------------
# 8. CLI  (contract: --dataset --pretext --stage --arm --seed --out)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Conditional dreaming by clamped-well Langevin (A-cell smoke only)."
    )
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument("--stage", default="dream", choices=list(STAGES) + list(NOT_MINE))
    p.add_argument("--arm", default=None, action="append",
                   help="repeatable; default: gen_dream.arms")
    p.add_argument("--seed", type=int, default=None, action="append", help="repeatable")
    p.add_argument("--out", default="results/gen_dream")
    p.add_argument("--condition", default=None,
                   help="hold-out condition (default: gen_data.condition)")
    p.add_argument("--sampler", default=None, choices=["mala", "ula", "uld"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--step-size", type=float, default=None,
                   help="pin the MALA step size (default: tuned to target_accept)")
    p.add_argument("--n-dream", type=int, default=None)
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--store-plan", default=None)
    p.add_argument("--quick", action="store_true",
                   help="plumbing smoke only — ⛔ NEVER a reported number")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    g = config.gen_dream
    if args.condition:
        config.gen_data.condition = args.condition
    if args.sampler:
        g.sampler = args.sampler
        g.samplers = [args.sampler]
    if args.steps is not None:
        g.steps = int(args.steps)
    if args.step_size is not None:
        g.step_size = float(args.step_size)
    if args.n_dream is not None:
        g.n_dream = int(args.n_dream)
    if args.bridge_root:
        g.bridge_root = args.bridge_root
    if args.store_plan:
        g.store_plan = args.store_plan
    # ⚠ ⛔ Do NOT copy $CHLU_GEN_DATA_ROOT into ``gen_data.data_root``: the two
    # are read with DIFFERENT conventions — ``shapes3d_root(root)`` treats an
    # explicit root as the *dataset* directory, while ``default_gen_data_root()``
    # is the *gen* root and appends the dataset name. Leaving it None lets the env
    # variable resolve correctly on both paths (the worktree trap).
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    agg = run_experiment_gen_dream(
        config=config,
        dataset=args.dataset,
        pretext=args.pretext,
        out_dir=args.out,
        seeds=args.seed,
        stage=args.stage,
        arms=args.arm,
        quick=bool(args.quick),
    )
    if agg:
        hp = agg.get("headline_protocol")
        print(f"⭐ headline protocol: {hp}")
        for k, v in agg.get("falsifier_sampler_is_inert", {}).items():
            print(
                f"⭐ {k}: full − zero_step = {v['delta_aggregate_mean']:+.4f} "
                f"± {v['delta_aggregate_std']:.4f} "
                f"({v['n_seeds_positive']} seed(s) positive) "
                + ("⛔ SAMPLER IS INERT" if v["SAMPLER_IS_INERT"] else "✓ not inert")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
