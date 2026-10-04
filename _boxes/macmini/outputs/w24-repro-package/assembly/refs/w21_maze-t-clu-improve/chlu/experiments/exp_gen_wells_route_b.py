"""⭐ **ROUTE B — learn WHERE THE WELLS GO, with the store in the loop.**

⭐ DIAL DECLARATION — *placement*.
- **Dial:** **WHERE THE WELLS GO** — well placement learned with the store in the loop,
  replacing a partition inherited from clustering.  ⛔ Not the read, not the decoder, not
  the encoder.
- **Laundering control:** ⭐⭐ **D-matched RANDOM placement at matched bytes.**  It is
  built and measured **FIRST** (:data:`chlu.core.gen_wells.FLOOR_ARM`), and the harness
  refuses to run a learned arm until the floor artefact exists on disk.
- **Falsifies:** the claim that our wells are semantic.  ⚠ It may show placement was never
  the lever.
- **Does NOT falsify:** failing to recover ``scale``/``orientation`` information that
  ``gen-loss-attribution`` already localised at the **codebook** (90.5 % / 100 %) rather
  than at the partition.  That is a known confirmation of a measured result, not news.

**Why this and not another read.**  The read resolves *inside* the wells it is handed.
Route A takes the wells as given; this asks the prior question — **are the wells in the
right place at all?**

⛔ **What this harness may NOT do, encoded as assertions rather than as good intentions:**

* ⛔ report a sharpness number without its **D-matched random-placement floor** beside it
  (:func:`chlu.core.gen_wells.placement_margin` is the only reporting path);
* ⛔ report a per-factor number without ``n_values_realised`` (every row carries it, and a
  one-valued factor reads ``undefined``, never ``0``);
* ⛔ let the learned arm renegotiate the byte budget (the ledger aborts the run);
* ⛔ quote the **weakly-supervised** arm as the label-free result (separate row, always);
* ⛔ reach a GPU and only then discover it cannot open what it needs (every precondition
  fails at startup or not at all).

⚠ **``--quick`` clobbers explicit CLI flags** — ``apply_quick`` *replaces* the seed list,
so :func:`main` applies quick **first** and explicit flags **last**.  ⛔ A ``--quick``
number is never reported.

Runs through ``chlu experiment gen-wells-route-b`` or as ``python -m
chlu.experiments.exp_gen_wells_route_b``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import jax
import numpy as np

from ..config import CHLUConfig, gen_wells_route_b_config, get_default_config  # noqa: E402
from ..core.gen_bridge import (  # noqa: E402
    dataset_module,
    load_feature_cache,
    param_hash,
)
from ..core.gen_dream import WINNING_DECODER_RECIPE, assert_dream_decoder  # noqa: E402
from ..core.gen_landscape import arm_key  # noqa: E402
from ..core.gen_store import (  # noqa: E402
    address_codes,
    assert_no_bypass,
    channel_correlation,
    correlation_grouping,
    dead_channels,
    init_store,
    mode_usage,
    reconstruction_mse,
    store_bytes,
    train_store,
    well_channels_from_labels,
)
from ..core.gen_wells import (  # noqa: E402
    ARMS,
    FLOOR_ARM,
    LABEL_FREE_ARMS,
    CheckpointCfg,
    PlacementState,
    address_sharpness,
    assert_byte_ledger,
    assert_cache_coverage,
    assert_label_free,
    assert_partition,
    collapse_verdict,
    config_fingerprint,
    dmatched_random_placement,
    harden_to_histogram,
    placement_loss,
    placement_margin,
    sharpness_table,
    soft_memberships,
    train_placement,
    weak_supervision_term,
)
from ..data.gen_splits import default_gen_data_root  # noqa: E402
from ..eval.gen_probe import factor_labels  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402


def _cfg(config: CHLUConfig):
    return gen_wells_route_b_config(config)


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only.  ⛔ A ``--quick`` number is NEVER reported.

    ⛔ The **byte ledger** and the **label-free gate** are deliberately NOT relaxed: the
    ledger is a pre-registered number and a smoke that shrinks it proves nothing about it,
    and a label leak is not less of a leak in a smoke.
    """
    c = _cfg(config)
    c.seeds = list(c.quick_seeds)
    c.n_store_train = int(c.quick_n_store_train)
    c.n_corr = int(c.quick_n_corr)
    c.n_eval = int(c.quick_n_eval)
    c.placement_steps = int(c.quick_placement_steps)
    c.placement_batch = int(c.quick_placement_batch)
    c.store_train_steps = int(c.quick_store_train_steps)
    c.ckpt_every_seconds = float(c.quick_ckpt_every_seconds)
    c.ckpt_every_steps = int(c.quick_ckpt_every_steps)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    c.assert_byte_ledger = bool(c.quick_assert_byte_ledger)
    c.assert_label_free = bool(c.quick_assert_label_free)
    return config


def _claude_root(config: CHLUConfig) -> Path:
    r = _cfg(config).claude_root or os.environ.get("CHLU_CLAUDE_ROOT")
    if r:
        return Path(r).expanduser()
    return Path(__file__).resolve().parents[2] / ".claude"


def route_b_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-wells-route-b"


def _abs(config: CHLUConfig, p) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


def bridge_runs_root(config: CHLUConfig) -> Path:
    """Where ``gen-feature-bridge`` wrote its per-seed runs.

    ⚠ A worktree has no ``.claude/``, so this must be reachable from the environment —
    the same trap as ``$CHLU_GEN_DATA_ROOT``.  ⭐ ``$CHLU_CLAUDE_ROOT`` alone is enough,
    which is exactly what ``scripts/csf3/env.sh``'s export block guarantees a job.
    """
    c = _cfg(config)
    r = c.bridge_root or os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    if r:
        return Path(r).expanduser()
    return _claude_root(config) / "outputs" / "gen-feature-bridge" / "runs"


def _forward_roots(config: CHLUConfig) -> None:
    """⭐ Forward this block's roots into the modules we REUSE rather than re-implement.

    ``exp_gen_store.bridge_dir`` and ``gather_features`` read ``config.gen_store`` /
    ``config.gen_data``; reusing them verbatim (⛔ rather than writing a second loader that
    can drift) means their roots must be told where this block's roots point.
    """
    c = _cfg(config)
    config.gen_store.bridge_root = str(bridge_runs_root(config))
    # ⛔ MEASURED BUG, fixed here: ``select_store_rows`` reads
    # ``config.gen_store.n_store_train``, NOT this block's.  Without this line a
    # ``--quick`` smoke silently gathered the full 32,768 rows and took ~20x as long as
    # it should have, while the log printed a perfectly normal "rows: store_train=32768".
    # ⚠ A quick run that is not quick is a quick run nobody re-runs.
    config.gen_store.n_store_train = int(c.n_store_train)
    config.gen_store.encode_chunk = int(c.encode_chunk)
    # ⚠ ⛔ ``gen_data.data_root`` is deliberately NOT set from the environment.
    # ``shapes3d_root(root)`` treats an EXPLICIT root as the **dataset** directory
    # (…/gen/shapes3d) while ``$CHLU_GEN_DATA_ROOT`` is its **parent** (…/gen).  Leaving
    # it ``None`` lets ``default_gen_data_root()`` read the variable with the right
    # meaning; setting it from the env var costs a FileNotFoundError that looks exactly
    # like a missing dataset.  (exp_gen_navigate §7 — it bit that spoke too, and it bit
    # this one on its first smoke run.)


def store_runs_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    r = c.store_runs or os.environ.get("CHLU_GEN_STORE_RUNS")
    if r:
        return Path(r).expanduser()
    return _claude_root(config) / "outputs" / "gen-store-grouping" / "runs"


def load_plan(config: CHLUConfig, dataset: str, pretext: str) -> dict:
    """⛔ ``K``, ``n_modes``, ``D`` and ``store_bytes`` **loaded VERBATIM** from
    ``gen-store-grouping``'s pinned plan, never re-derived here.

    The plan says so itself: *"Every later factorisation-ladder rung loads this file
    VERBATIM.  K, n_modes, D and store_bytes are pinned here; only the grouping (the
    ``labels`` field) is allowed to change between rungs."*  ⭐ Route B **is** a new
    ``labels`` vector and nothing else.
    """
    c = _cfg(config)
    p = (Path(c.store_plan).expanduser() if c.store_plan
         else store_runs_root(config) / f"{dataset}_{pretext}" / "store_plan.json")
    if not p.exists():
        raise FileNotFoundError(
            f"⛔ no pinned store plan at {p} — K/n_modes/D/store_bytes are loaded from it "
            "VERBATIM and are never invented here (set gen_wells_route_b.store_plan or "
            "$CHLU_GEN_STORE_RUNS)"
        )
    plan = json.loads(p.read_text())
    return {
        "path": str(p),
        "K": int(plan["K"]),
        "n_modes": int(plan["n_modes"]),
        "D": int(plan["D"]),
        "feat_shape": tuple(int(x) for x in plan["feat_shape"]),
        "store_bytes": int(plan["store_bytes"]["store_bytes"]),
        "reference_well_sizes": [int(x) for x in plan["well_sizes"]],
        "grouping_method": str(plan.get("grouping_method", "")),
        "encoder_param_hash": str(plan.get("encoder_param_hash", "")),
        "rung": str(plan.get("rung", "")),
    }


def declift_dir(config: CHLUConfig) -> Path:
    c = _cfg(config)
    r = c.declift_dir or os.environ.get("CHLU_GEN_DECLIFT_DIR")
    if r:
        return Path(r).expanduser()
    return _claude_root(config) / "outputs" / "gen-decoder-lift" / "winner"


def assert_winner_decoder(config: CHLUConfig, seed: int, log: Callable = print) -> dict:
    """⛔⭐ **THE DECODER TRAP**, closed by name AND by per-seed parameter hash.

    ``GenBridgeConfig.dec_recipe`` still defaults to ``"baseline"`` — the decoder that
    **FAILED** the gate (0.7191 balanced accuracy against the winner's 0.9414).  A run
    decoded through it produces perfectly normal-looking logs and is **void**.  ⭐ The
    winner is ``v1_capacity_schedule`` and seed 0's parameter hash is ``01ef91f7…``.
    """
    c = _cfg(config)
    d = declift_dir(config)
    hp = d / "winner_hashes.json"
    if not hp.exists():
        raise FileNotFoundError(
            f"⛔ no winner hashes at {hp} — the decoder recipe cannot be asserted and the "
            "run would be void (set gen_wells_route_b.declift_dir or $CHLU_GEN_DECLIFT_DIR)"
        )
    hashes = json.loads(hp.read_text())
    entry = hashes[f"seed_{int(seed)}"]
    if str(c.dec_recipe) != str(entry["recipe"]["name"]):
        raise AssertionError(
            f"⛔ dec_recipe is {c.dec_recipe!r} but the winner decoder on seed {seed} is "
            f"{entry['recipe']['name']!r} — the default is 'baseline', the recipe that "
            "FAILED the gate, and a run through it looks normal and is VOID"
        )
    dec = load_model(d / f"decoder_v1_s{int(seed)}.pkl")
    rep = assert_dream_decoder(dec, entry["param_hash"], recipe_name=str(c.dec_recipe),
                               expected_recipe=str(c.dec_recipe))
    log(f"  ⭐ decoder: recipe {c.dec_recipe!r} asserted by NAME and by param hash "
        f"{entry['param_hash'][:8]}… (seed {seed})")
    return {"recipe": str(c.dec_recipe), "param_hash": str(entry["param_hash"]),
            "expected_recipe": WINNING_DECODER_RECIPE, "seed": int(seed), "check": rep}


def _store_cfg(c) -> SimpleNamespace:
    """The shim ``gen_store.train_store`` expects.  ⛔ ONE object for every arm — the
    delivered store is fitted by the **identical** call in all four arms, so the only
    thing that differs between them is the label vector."""
    return SimpleNamespace(
        train_batch=int(c.store_train_batch),
        train_steps=int(c.store_train_steps),
        train_lr=float(c.store_train_lr),
        commitment_beta=float(c.store_commitment_beta),
        tau_start=float(c.store_tau_start),
        tau_end=float(c.store_tau_end),
    )


# ---------------------------------------------------------------------------
# 1. ⛔ Preflight — every precondition fails at STARTUP or not at all
# ---------------------------------------------------------------------------


def preflight(config: CHLUConfig, dataset: str, pretext: str, seed: int,
              log: Callable = print) -> dict:
    """⛔⭐ Everything that can refuse this run, refused **before** it does any work.

    ⛔ On a queue an omission fails LATE — inside the loader, after the scheduler has
    already given you the GPU.  That costs a resubmit and a wait, and it will happen.
    """
    c = _cfg(config)
    rep = {"seed": int(seed), "checks": {}}
    _forward_roots(config)

    # (1) the artefact roots, by name, with the missing one printed.
    roots = {
        "bridge_root": bridge_runs_root(config),
        "store_runs": store_runs_root(config),
        "declift_dir": declift_dir(config),
        "data_root": (Path(str(config.gen_data.data_root)).expanduser()
                      if config.gen_data.data_root else default_gen_data_root()),
    }
    missing = {k: str(v) for k, v in roots.items() if not Path(v).exists()}
    if missing:
        raise FileNotFoundError(
            "⛔ artefact root(s) missing, by name: "
            + "; ".join(f"{k} -> {v}" for k, v in missing.items())
            + "\n    ⭐ Set $CHLU_CLAUDE_ROOT (and/or $CHLU_GEN_BRIDGE_ROOT, "
              "$CHLU_GEN_STORE_RUNS, $CHLU_GEN_DECLIFT_DIR, $CHLU_GEN_DATA_ROOT). "
              "A job must never reach the GPU only to discover it cannot open an artefact."
        )
    rep["checks"]["artefact_roots"] = {k: str(v) for k, v in roots.items()}
    log(f"  ⭐ artefact roots present: {', '.join(roots)}")

    # (2) the pinned plan — K/n_modes/D/store_bytes, verbatim.
    plan = load_plan(config, dataset, pretext)
    rep["plan"] = plan
    log(f"  ⭐ plan (VERBATIM): K={plan['K']} n_modes={plan['n_modes']} D={plan['D']} "
        f"store_bytes={plan['store_bytes']:,} (method {plan['grouping_method']!r})")

    # (3) the frozen bridge: encoder hash == manifest == cache.
    bdir = bridge_dir(config, dataset, pretext, seed)
    bman = json.loads((bdir / "bridge_manifest.json").read_text())
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(c.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    log(f"  feature cache verified: {cman['n_rows']} rows (scope={cman['scope']}), "
        f"D={cman['D']}, {cman['dtype']}, encoder {enc_hash[:8]}…")
    rep["checks"]["bridge"] = {
        "dir": str(bdir), "encoder_param_hash": enc_hash,
        "cache_scope": str(cman["scope"]), "cache_n_rows": int(cman["n_rows"]),
        "cache_dtype": str(cman["dtype"]), "D": int(cman["D"]),
    }

    # (4) ⭐ THE CACHE-COVERAGE GATE — names the seed and the shortfall, and dies.
    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    rows = select_store_rows(split, config, int(seed))
    needed = np.concatenate([rows["store_train"], rows["eval_a"], rows["eval_b"]])
    cov = assert_cache_coverage(
        crows, needed, seed=int(seed), scope=str(cman["scope"]),
        allow_reencode=(not bool(c.assert_cache_coverage)) or bool(c.allow_reencode),
        label="store_train + eval_a + eval_b",
    )
    rep["checks"]["cache_coverage"] = cov
    if cov["exercised_untested_path"]:
        log(f"  ⚠⚠ EXERCISED UNTESTED PATH: {cov['n_shortfall']} row(s) of seed {seed} are "
            "NOT cached and will be re-encoded through gather_features' extension path. "
            "⛔ This run is NOT a normal run and is reported as such.")
    else:
        log(f"  ⭐ cache coverage: {cov['n_covered']}/{cov['n_rows_needed']} rows cached "
            f"(seed {seed}, scope={cov['cache_scope']!r}) — 0 shortfall")

    # (5) ⭐ re-encoding CACHED rows must reproduce the cache — BINDING exactly when
    # this run can write new rows into it.  ⛔ The verdict is gate (4)'s OWN answer,
    # three lines above: `cov["n_shortfall"]`.  Coverage is NEVER re-derived here, so
    # the coverage gate and the re-encode gate cannot disagree about the same run.
    # ⚠ `run_seed` gathers `store_train` and a prefix of `eval_a`, both inside gate
    # (4)'s `needed` set — and if that ever stops being true, `gather_features` runs
    # the check itself, ENFORCED, at the point of extension.
    rep["checks"]["cache_reencode"] = _assert_cache_reencodes(
        dataset, config, encoder, crows, feats, cman, log=log,
        will_extend=bool(cov["n_shortfall"] > 0),
        reason=(
            f"cache-coverage gate: {cov['n_covered']}/{cov['n_rows_needed']} rows cached, "
            f"{cov['n_shortfall']} shortfall (seed {seed}, scope={cov['cache_scope']!r}) "
            + ("⇒ this run WILL extend the cache"
               if cov["n_shortfall"] > 0
               else "⇒ this run encodes nothing new and cannot extend the cache")
        ),
    )

    # (6) ⛔ the decoder trap.
    if bool(c.assert_dec_recipe):
        rep["checks"]["decoder"] = assert_winner_decoder(config, int(seed), log=log)

    rep["rows"] = {k: int(np.asarray(v).size) for k, v in rows.items()}
    rep["_runtime"] = {"encoder": encoder, "crows": crows, "feats": feats, "cman": cman,
                       "split": split, "values": values, "rows": rows, "mod": mod,
                       "grid": split.grid}
    return rep


# ---------------------------------------------------------------------------
# 2. One arm — ⛔ the identical code path in all four; only the labels differ
# ---------------------------------------------------------------------------


def run_arm(arm: str, labels: np.ndarray, ctx: dict, config: CHLUConfig, seed: int,
            out_dir: Path, log: Callable = print) -> dict:
    """Fit and score one arm from its **label vector**.

    ⭐ Every arm goes through the same ``init_store`` + ``train_store`` + ``address_codes``
    path with the same budget, so a margin between two arms is attributable to placement
    and to nothing else.
    """
    c = _cfg(config)
    plan = ctx["plan"]
    lab = np.asarray(labels, np.int64)
    part = assert_partition(lab, plan["D"], plan["K"])

    # ⛔ the byte ledger — asserted in code; the run ABORTS on a mismatch.
    ledger = assert_byte_ledger(
        lab, plan["n_modes"], plan["store_bytes"], k=plan["K"], dim=plan["D"],
        rank=int(c.coupling_rank), arm=arm, enforce=bool(c.assert_byte_ledger))
    log(f"  [{arm}] well sizes {part['well_sizes']} · store {ledger['store_bytes']:,} B "
        f"(= M·D·4, matches plan: {ledger['matches_expected']}) · "
        f"{ledger['bits_per_read']:.1f} bits/read")

    wc = well_channels_from_labels(lab, plan["K"])
    # ⭐ arm keys by fold_in: split(k,2)[1] is bit-identical to split(k,3)[1] on JAX 0.9.0
    # here, so lengthening a split would hand two arms the SAME key.
    key = arm_key(jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911), f"routeb:{arm}")
    ik, tk = jax.random.split(key, 2)
    store = init_store(ctx["X_store"], wc, plan["n_modes"], plan["feat_shape"], ik,
                       seed=int(seed))
    log(f"  [{arm}] fitting the delivered store ({c.store_train_steps} steps, "
        "store-only objective, identical call in every arm) …")
    store, hist, _ = train_store(store, ctx["X_store"], _store_cfg(c), tk, log=None)

    sb = store_bytes(store)
    if bool(c.assert_byte_ledger) and int(sb["store_bytes"]) != int(plan["store_bytes"]):
        raise AssertionError(
            f"⛔ the BUILT store on arm {arm!r} is {sb['store_bytes']} B, not the pinned "
            f"{plan['store_bytes']} B — learned placement must cost what inherited "
            "placement costs")

    codes_eval = address_codes(store, ctx["X_eval"].reshape((ctx["X_eval"].shape[0], -1)))
    usage = mode_usage(codes_eval, plan["n_modes"])
    sharp = address_sharpness(
        codes_eval, ctx["lab_eval"], ctx["names"], ctx["cardinalities"],
        n_modes_per_well=store.n_modes_per_well, smoothing=float(c.address_smoothing))
    rec = reconstruction_mse(store, ctx["X_eval"].reshape((ctx["X_eval"].shape[0], -1)))
    bypass = assert_no_bypass(store, ctx["X_eval"][: min(512, ctx["X_eval"].shape[0])]
                              .reshape((-1, plan["D"])))

    return {
        "arm": str(arm),
        "labels": [int(x) for x in lab],
        "partition": part,
        "byte_ledger": ledger,
        "store_bytes": sb,
        "train_loss_first_last": [float(hist[0]), float(hist[-1])] if hist else None,
        "mode_usage": usage,
        "sharpness": sharp,
        "reconstruction": rec,
        "no_bypass": bypass,
        "n_eval_rows": int(ctx["X_eval"].shape[0]),
    }


def _reference_labels(ctx: dict, config: CHLUConfig, log: Callable = print) -> tuple:
    """The inherited partition for THIS seed, and the size histogram every arm is
    D-matched to.

    ⚠ ``K`` is pinned from the plan and ⛔ never re-derived; only the labels are computed
    here, from **train activations alone**.
    """
    plan = ctx["plan"]
    lab, info = correlation_grouping(
        ctx["corr"], plan["K"], "auto", dead=ctx["dead"],
        min_well_frac=0.10, return_info=True)
    sizes = [int(x) for x in np.bincount(lab, minlength=plan["K"])]
    log(f"  ⭐ inherited partition (method {info['method_used']!r}, ⛔ NOT k-means — "
        f"k-means++ seeds the MODES): well sizes {sizes}")
    return np.asarray(lab, np.int64), sizes, info


def run_seed(config: CHLUConfig, dataset: str, pretext: str, seed: int, out_dir: Path,
             resume: bool = False, log: Callable = print) -> dict:
    c = _cfg(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pre = preflight(config, dataset, pretext, int(seed), log=log)
    rt = pre.pop("_runtime")
    plan = pre["plan"]
    grid = rt["grid"]
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = rt["rows"]

    def gather(r):
        return gather_features(dataset, config, rt["encoder"], rt["crows"], rt["feats"],
                               rt["cman"], r, log=log)

    n_eval = int(min(int(c.n_eval), np.asarray(rows["eval_a"]).size))
    eval_rows = np.asarray(rows["eval_a"])[:n_eval]
    log(f"  rows: store_train={rows['store_train'].size} eval={eval_rows.size}")
    X_store = gather(rows["store_train"])
    X_eval = gather(eval_rows)
    lab_store = factor_labels(grid, rows["store_train"])

    Xc = X_store[: int(c.n_corr)]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    log(f"  ⚠ {int(dead.sum())}/{dead.size} coordinates are DEAD (constant post-ReLU)")

    ctx = {
        "plan": plan, "names": names, "cardinalities": cards,
        "X_store": X_store, "X_eval": X_eval,
        "lab_eval": factor_labels(grid, eval_rows), "lab_store": lab_store,
        "corr": corr, "dead": dead,
    }
    ref_labels, ref_sizes, ref_info = _reference_labels(ctx, config, log=log)

    # -----------------------------------------------------------------------
    # ⭐⭐ THE FLOOR, BUILT AND MEASURED FIRST — before the objective exists.
    # -----------------------------------------------------------------------
    floor_dir = out_dir / "floor_first"
    log(f"⭐⭐ [{FLOOR_ARM}] the D-matched random-placement FLOOR — measured FIRST, before "
        "any learned objective runs")
    floor_labels = dmatched_random_placement(ref_labels, int(seed))
    if [int(x) for x in np.bincount(floor_labels, minlength=plan["K"])] != ref_sizes:
        raise AssertionError(
            "⛔ the floor is not D-matched: its well-size histogram differs from the "
            "reference, so any margin over it would be unattributable")
    floor = run_arm(FLOOR_ARM, floor_labels, ctx, config, int(seed), out_dir, log=log)
    _write_json({"seed": int(seed), "arm": FLOOR_ARM, "result": floor,
                 "reference_well_sizes": ref_sizes,
                 "note": ("⭐ the laundering control, measured BEFORE the learned "
                          "objective ran. Every placement number in this run is a margin "
                          "over this.")},
                floor_dir / f"floor_s{int(seed)}.json")
    log(f"  ⭐ floor filed at {floor_dir}/floor_s{int(seed)}.json — learned arms may now run")

    arms = {FLOOR_ARM: floor}
    label_free_reports = {}

    for arm in [a for a in list(c.arms) if a != FLOOR_ARM]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; declared arms are {ARMS}")
        # ⛔ the floor must exist on disk before a learned arm is scored.
        if not (floor_dir / f"floor_s{int(seed)}.json").exists():
            raise AssertionError(
                "⛔ the D-matched random floor has not been measured — refusing to score a "
                "placement arm against nothing")

        if arm == "correlation":
            log(f"  [{arm}] the inherited partition (route A's) — a CONTEXT row")
            arms[arm] = run_arm(arm, ref_labels, ctx, config, int(seed), out_dir, log=log)
            continue

        # --- the learned arms ------------------------------------------------
        weak = arm == "learned_weak_sup"
        log(f"⭐ [{arm}] learning the placement with the store in the loop "
            f"({c.placement_steps} steps)" + ("  ⛔ DECLARED weakly-supervised arm — "
                                              "reported as a SEPARATE row and NEVER as "
                                              "the label-free result" if weak else ""))
        pk = arm_key(jax.random.PRNGKey(int(seed) * 7919 + 20_260_915), f"placement:{arm}")
        lk, mk, bk = jax.random.split(pk, 3)
        D, K, M = plan["D"], plan["K"], plan["n_modes"]
        if bool(c.warm_start_from_reference):
            logits = np.zeros((D, K), np.float32)
            logits[np.arange(D), ref_labels] = 1.0
            logits = logits + float(c.logit_init_scale) * np.asarray(
                jax.random.normal(lk, (D, K)))
        else:
            logits = float(c.logit_init_scale) * np.asarray(jax.random.normal(lk, (D, K)))
        modes = 0.1 * jax.random.normal(mk, (K, M, D), jax.numpy.float32)
        state = PlacementState(logits, modes)

        fp = config_fingerprint({
            "arm": arm, "seed": int(seed), "dataset": dataset, "pretext": pretext,
            "steps": int(c.placement_steps), "batch": int(c.placement_batch),
            "lr": float(c.placement_lr),
            "tau": [float(c.placement_tau_start), float(c.placement_tau_end)],
            "lambda_occ": float(c.lambda_occ), "lambda_fact": float(c.lambda_fact),
            "lambda_sup": float(c.lambda_sup) if weak else 0.0,
            "K": K, "M": M, "D": D, "n_store_train": int(c.n_store_train),
            "warm_start": bool(c.warm_start_from_reference),
            "logit_lr_mult": float(c.placement_logit_lr_mult),
        })
        ckpt = CheckpointCfg(
            Path(c.ckpt_dir).expanduser() / f"seed_{int(seed)}" if c.ckpt_dir
            else out_dir / "checkpoints" / f"seed_{int(seed)}",
            every_seconds=float(c.ckpt_every_seconds),
            every_steps=int(c.ckpt_every_steps), keep_last=int(c.ckpt_keep_last))

        # ⛔ the label-free gate, run on THIS arm's objective before it trains.
        if bool(c.assert_label_free):
            xb = X_store[: min(256, X_store.shape[0])]
            yb = lab_store[: min(256, X_store.shape[0])]
            if weak:
                def _leaky(st, x, y, tau):
                    return (placement_loss(st, x, tau, lambda_occ=c.lambda_occ,
                                           lambda_fact=c.lambda_fact)[0]
                            + float(c.lambda_sup) * weak_supervision_term(
                                st, x, y, tau, cardinalities=cards))
                label_free_reports[arm] = assert_label_free(
                    _leaky, state, xb, yb, tau=float(c.placement_tau_start),
                    seed=int(seed), arm=arm, expect_label_free=False)
                log(f"  ⛔ [{arm}] label-free check FIRED as designed: "
                    f"|Δloss|={label_free_reports[arm]['abs_loss_delta']:.3e}, "
                    f"max|Δgrad|={label_free_reports[arm]['max_abs_grad_delta']:.3e} "
                    "under a label permutation — the check has teeth")
            else:
                def _clean(st, x, tau):
                    return placement_loss(st, x, tau, lambda_occ=c.lambda_occ,
                                          lambda_fact=c.lambda_fact)[0]
                label_free_reports[arm] = assert_label_free(
                    _clean, state, xb, yb, tau=float(c.placement_tau_start),
                    seed=int(seed), arm=arm, expect_label_free=True)
                log(f"  ⭐ [{arm}] LABEL-FREE proven: loss and gradients bit-identical "
                    "under a label permutation, and the signature admits no labels")

        state, hist, info = train_placement(
            state, X_store, steps=int(c.placement_steps), batch=int(c.placement_batch),
            lr=float(c.placement_lr), tau_start=float(c.placement_tau_start),
            tau_end=float(c.placement_tau_end), lambda_occ=float(c.lambda_occ),
            lambda_fact=float(c.lambda_fact), base_key=bk, arm=arm,
            logit_lr_mult=float(c.placement_logit_lr_mult),
            cardinalities=cards,
            labels=(lab_store if weak else None),
            lambda_sup=(float(c.lambda_sup) if weak else 0.0),
            ckpt=ckpt, config_fp=fp, resume=bool(resume), log=log)

        mem = np.asarray(soft_memberships(state.logits))
        learned = harden_to_histogram(mem, ref_sizes)
        moved = int((learned != ref_labels).sum())
        log(f"  [{arm}] hardened to the reference histogram; {moved}/{D} coordinates "
            f"({100.0 * moved / D:.1f} %) moved well")
        res = run_arm(arm, learned, ctx, config, int(seed), out_dir, log=log)
        res["placement"] = {
            "steps_run": info["steps_run"], "resumed_from": info["resumed_from"],
            "total_steps": info["total_steps"],
            "n_checkpoints_written": info.get("n_checkpoints_written"),
            "wall_seconds": info.get("wall_seconds"),
            "loss_first_last": [float(hist[0]), float(hist[-1])] if hist else None,
            "coordinates_moved": moved,
            "fraction_moved": float(moved / D),
            "config_fingerprint": fp,
            "checkpoint_dir": str(ckpt.dir),
            "warm_start_from_reference": bool(c.warm_start_from_reference),
        }
        arms[arm] = res

    # -----------------------------------------------------------------------
    # ⭐ Margins over the floor + the collapse guard.  ⛔ The ONLY reporting path.
    # -----------------------------------------------------------------------
    margins, collapse = {}, {}
    for arm, res in arms.items():
        if arm == FLOOR_ARM:
            continue
        margins[arm] = placement_margin(res["sharpness"], floor["sharpness"], arm=arm)
        collapse[arm] = collapse_verdict(res["mode_usage"], floor["mode_usage"],
                                         frac=float(c.collapse_used_modes_frac))
        mm = margins[arm]["mean_margin_max"]
        log(f"  ⭐ [{arm}] margin over the D-matched random floor: mean ΔS = "
            f"{mm:+.4f}" + (f", best {margins[arm]['best_factor']} "
                            f"{margins[arm]['max_margin']:+.4f}" if mm is not None else "")
            + f" · occupancy {collapse[arm]['verdict']}")

    rec = {
        "schema": "chlu-gen-wells-route-b/1",
        "complete": True,
        "dataset": dataset, "pretext": pretext, "seed": int(seed),
        "preflight": pre,
        "reference": {"well_sizes": ref_sizes, "grouping_info": ref_info,
                      "labels": [int(x) for x in ref_labels]},
        "arms": arms,
        "margins_over_floor": margins,
        "collapse": collapse,
        "label_free_checks": label_free_reports,
        "floor_arm": FLOOR_ARM,
        "label_free_arms": list(LABEL_FREE_ARMS),
        "wall_seconds": float(time.time() - t0),
        "⛔ reporting rule": (
            "every placement number is a MARGIN OVER THE D-MATCHED RANDOM-PLACEMENT "
            "FLOOR, never an absolute; the weakly-supervised arm is a SEPARATE row and is "
            "never the label-free result; a per-factor number without n_values_realised "
            "is void"),
    }
    _write_json(rec, out_dir / f"routeb_s{int(seed)}.json")
    return rec


def verdict(records: list) -> dict:
    """⭐ The pre-registered verdict, in the task's own words.

    ⛔ Rendered only when a *scored* run produced it: a ``--quick``/verification run sets
    ``scored=False`` and no verdict is rendered, because ⛔ no number from a verification
    run enters any table.
    """
    per_seed = []
    for r in records:
        m = r.get("margins_over_floor", {}).get("learned_label_free")
        col = r.get("collapse", {}).get("learned_label_free", {})
        if not m:
            continue
        per_seed.append({
            "seed": r["seed"], "mean_margin_max": m["mean_margin_max"],
            "max_margin": m["max_margin"], "best_factor": m["best_factor"],
            "collapsed": col.get("collapsed"),
        })
    beats = [p for p in per_seed if (p["max_margin"] or 0.0) >= 0.10 and not p["collapsed"]]
    return {
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "n_seeds_clearing_the_bar": len(beats),
        "bar": ("PREREG §6b: learned_label_free beats random_dmatched by ΔS >= 0.10 on at "
                "least one of scale/orientation/shape, 3/3 seeds, occupancy NOT collapsed"),
        "verdict": (
            "⭐ LEARNED PLACEMENT BEATS THE D-MATCHED RANDOM FLOOR"
            if per_seed and len(beats) == len(per_seed) else
            "⛔ LEARNED PLACEMENT DOES NOT BEAT D-MATCHED RANDOM PLACEMENT — placement was "
            "never the lever, the inherited partition was not the problem, and route B "
            "does not go to the cluster"
        ) if per_seed else "NOT RENDERED — no scored label-free arm in these records",
    }


def summarize(records: list, config: CHLUConfig, scored: bool) -> dict:
    c = _cfg(config)
    names = None
    for r in records:
        for a in r.get("arms", {}).values():
            names = list(a["sharpness"]["per_factor"])
            break
        if names:
            break
    tables = {}
    for r in records:
        per_arm = {a: v["sharpness"] for a, v in r.get("arms", {}).items()}
        if per_arm and names:
            tables[str(r["seed"])] = sharpness_table(per_arm, names)
    return {
        "schema": "chlu-gen-wells-route-b-summary/1",
        "dataset": c.dataset, "pretext": c.pretext,
        "seeds": [r["seed"] for r in records],
        "scored": bool(scored),
        "floor_arm": FLOOR_ARM,
        "arms": list(c.arms),
        "sharpness_tables": tables,
        "margins": {str(r["seed"]): r.get("margins_over_floor") for r in records},
        "collapse": {str(r["seed"]): r.get("collapse") for r in records},
        "label_free_checks": {str(r["seed"]): r.get("label_free_checks") for r in records},
        "verdict": verdict(records) if scored else {
            "verdict": ("⛔ NOT RENDERED — this is a verification/--quick run. No number "
                        "from it enters any table."),
        },
    }


def run_experiment_gen_wells_route_b(
    config: Optional[CHLUConfig] = None, out_dir: Optional[str] = None,
    resume: bool = False, scored: bool = False, log: Callable = print,
) -> dict:
    config = config or get_default_config()
    c = _cfg(config)
    root = Path(out_dir) if out_dir else route_b_out_root(config)
    runs = root / "runs" / f"{c.dataset}_{c.pretext}"
    runs.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in list(c.seeds):
        sdir = runs / f"seed_{int(seed)}"
        path = sdir / f"routeb_s{int(seed)}.json"
        if path.exists() and bool(c.skip_if_exists) and not resume:
            rec = json.loads(path.read_text())
            if rec.get("complete"):
                log(f"  seed {seed}: artefact HIT {path.name} — skipped")
                records.append(rec)
                continue
        log(f"⭐ seed {seed}: running")
        records.append(run_seed(config, c.dataset, c.pretext, int(seed), sdir,
                                resume=resume, log=log))
    summary = summarize(records, config, scored)
    _write_json(summary, root / "summary.json")
    if not scored:
        log("⛔ NOT SCORED — no verdict is rendered here and no number from this run "
            "enters any table (pass --scored on the real run).")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=None, choices=list(ARMS) + [None])
    p.add_argument("--steps", type=int, default=None, help="placement steps")
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt-dir", default=None,
                   help="⛔ ~/scratch on the cluster; never $TMPDIR, never $HOME")
    p.add_argument("--ckpt-every-seconds", type=float, default=None)
    p.add_argument("--store-train-steps", type=int, default=None,
                   help="⚠ steps for the DELIVERED store's fit (identical in every arm). "
                        "Lowering it is for plumbing/resume demos ONLY — ⛔ a scored run "
                        "uses the config default. ⚠ gen_store.train_store recompiles its "
                        "jitted step EVERY iteration (tau is a Python float under "
                        "filter_jit), so this knob dominates wall time on a shared box.")
    p.add_argument("--resume", action="store_true",
                   help="⭐ resume from the newest GOOD checkpoint; refuses a checkpoint "
                        "written under a different config")
    p.add_argument("--allow-reencode", action="store_true",
                   help="⚠ take gather_features' UNTESTED extension path DELIBERATELY; "
                        "the run is then reported as an exercised untested path")
    p.add_argument("--scored", action="store_true",
                   help="render the pre-registered verdict (⛔ not for smokes)")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    config = get_default_config()
    c = _cfg(config)
    # ⛔ ORDER MATTERS AND IT WAS A MEASURED BUG ELSEWHERE IN THIS FAMILY: apply_quick
    # REPLACES the seed list, so an explicit flag applied *before* it is silently
    # discarded. ⭐ Quick first, explicit flags last, always.
    if args.quick:
        apply_quick(config)
    if args.dataset:
        c.dataset = args.dataset
    if args.pretext:
        c.pretext = args.pretext
    if args.seeds:
        c.seeds = list(args.seeds)
    if args.arms:
        c.arms = list(args.arms)
    if args.steps:
        c.placement_steps = int(args.steps)
    if args.ckpt_dir:
        c.ckpt_dir = str(args.ckpt_dir)
    if args.ckpt_every_seconds is not None:
        c.ckpt_every_seconds = float(args.ckpt_every_seconds)
    if args.store_train_steps is not None:
        c.store_train_steps = int(args.store_train_steps)
    if args.allow_reencode:
        c.allow_reencode = True
    if args.quick and args.scored:
        raise SystemExit("⛔ --quick --scored is refused: a quick number is never reported")
    run_experiment_gen_wells_route_b(config, out_dir=args.out, resume=bool(args.resume),
                                     scored=bool(args.scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
