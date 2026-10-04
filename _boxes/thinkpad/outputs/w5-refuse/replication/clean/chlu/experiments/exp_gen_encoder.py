"""Experiment: **the encoder trained WITH THE STORE IN THE LOOP** (R29 pivot item 3).

⭐ DIAL DECLARATION — *the encoder*.  In every shipped phase-1 artefact the encoder
is **frozen**: the store trains on a cached feature tensor and the decoder is fitted
downstream of it (R18).  This module is the first configuration in which the
encoder's parameters move while the store sits on the path.

- **Laundering control:** ⭐ **the frozen-encoder arm at matched compute** — the
  *same* stage-2 loop, the *same* warm-started ``(encoder, store, decoder)`` triple,
  the *same* PRNG key and therefore the *same* minibatch sequence, the *same* loss.
  The only difference is whether the encoder's leaves are handed to Optax.  ⛔ An
  unfrozen arm that simply got more gradient steps would be a compute win, not an
  encoder win.
- **Falsifies:** the hypothesis that the commitment collapse (9 distinct codes over
  72 requests, R29 Q2) is caused by the encoder being frozen.  ⛔ If unfreezing does
  not lift the distinct-code count on B, that is a **finding** and it narrows the
  problem — it is not a failed build.
- **Does NOT falsify:** absolute distinct-code counts far below 72 at smoke scale;
  losing on reconstruction MSE to the shipped pipeline (⛔ a different loop).

⛔⛔ **THE GUARANTEE THIS MODULE MUST NOT BREAK.**
``read_from_codes(store, codes)`` takes **no features**.  The only encoder → decoder
channel is ``K·log2(M)`` bits — R20's **declared design constant** — so a bypass is
not merely unmeasured, it is *structurally inexpressible*.  Unfreezing is exactly
the change that can silently create one, because the decoder's loss now flows back
into the encoder.  Every arm therefore runs
:func:`chlu.core.gen_bridge.assert_no_feature_bypass` **before** any number is
recorded, and the **realised channel width** is written into every manifest.

⛔⛔ **WHAT THIS CONFIGURATION VOIDS IF IT IS EVER PROMOTED.**  R18 is explicit: the
store trains on cached features from the frozen encoder and the decoder is
downstream.  ⭐ **A changed encoder changes the feature cache and voids every
downstream number in this phase.**  So this is a **NEW configuration sitting BESIDE
the frozen one, never a replacement**: it writes to its own fresh root
(``.claude/outputs/gen-encoder-unfreeze/runs/…``) and ⛔ re-fits, overwrites and
invalidates **nothing**.  The list of numbers that would have to be re-measured if
it were promoted is in ``.claude/outputs/gen-encoder-unfreeze.md``.

⭐ **Pretext is not ours to choose (R1):** reconstruction, mandatory.  ⛔ The
generative path takes **no factor labels**.  The address book's request → mode
grounding is the one declared, arm-identical supervision at the *interface* (R8),
exactly as in ``gen-read-compose``; the evaluation probe's labels are measurement.

⛔ **Live trap 1** — ``GenBridgeConfig.dec_recipe`` still defaults to ``"baseline"``,
the decoder that FAILED the gate.  This module pins ``v1_capacity_schedule``
explicitly and **asserts the resolved recipe name and the decoder's parameter hash**
into the manifest before recording anything.

⛔ **Live trap 2 (R26.7)** — a silently-defaulted ``data_root`` is *"declared not-run
reported as null"* in code form.  :func:`resolve_data_root` fails loudly here rather
than quietly resolving to a directory that does not exist on a cluster node.

Runnable::

    python -m chlu.experiments.exp_gen_encoder \\
        --dataset shapes3d --pretext recon --stage encoder --arm both --seed 0 \\
        --out DIR [--quick]
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
import jax.numpy as jnp
import numpy as np

from ..config import CHLUConfig, get_default_config  # noqa: E402
from ..core.gen_bridge import (  # noqa: E402
    PRETEXTS,
    assert_no_feature_bypass,
    make_decoder,
    param_count,
    param_hash,
    realised_channel_bits,
    resolve_decoder_cfg,
    stream_images,
    train_encoder,
    train_joint,
)
from ..core.gen_compose import compose_codes, distinct_codes, request_posterior  # noqa: E402
from ..core.gen_dream import fit_address_book  # noqa: E402
from ..core.gen_request import request_labels, requests_from_rows  # noqa: E402
from ..core.gen_store import (  # noqa: E402
    address_codes,
    assert_no_bypass,
    channel_correlation,
    correlation_grouping,
    init_store,
    read_from_codes,
    store_bytes,
    well_channels_from_labels,
)
from ..eval.gen_probe import factor_labels  # noqa: E402
from ..utils.checkpoints import save_model  # noqa: E402

#: ``STAGE=encoder`` is what the CSF3 launcher passes; the others are local aliases.
STAGES = ("encoder", "unfreeze", "aggregate")
#: ⭐ Both, always, in this order.  ⛔ The unfrozen number is not reportable alone.
ARMS = ("frozen", "unfrozen")
#: ⭐ The arm-manifest schema.  ⛔ ``skip_if_exists`` compares against it: an
#: artefact written by an OLDER schema is RE-RUN, loudly, never silently reused —
#: a resumed cluster run that reuses a manifest missing a field the summary reads
#: fails deep inside the aggregate, which is the worst place to find out.
ARM_SCHEMA = "chlu-gen-encoder-unfreeze/2"
#: ⭐ The CSF3 launcher hard-whitelists its ``ARMS`` axis to
#: ``clu store_removed lace knn`` (``job_gpu_gen.sh`` §1), so it **cannot express**
#: this spoke's two arms and ``ARMS="frozen unfrozen"`` is refused at submit time.
#: ⛔ The launcher is read-only to this spoke, so the alias lives here, explicitly:
#: the launcher's ``clu`` label means "run BOTH arms in ONE process".  ⭐ That is
#: the *stronger* control anyway — both arms are handed the **same warm-start
#: object**, not a deterministic reproduction of it.
LAUNCHER_ARM_ALIASES = {"clu": "both"}
#: Request sets every table uses.  ``MS`` = the matched-seen corner companion:
#: ⭐ the collapse is a property of the COLOUR CORNER, not of held-out-ness, so a
#: B number without its MS twin cannot tell those two apart.
REQUEST_SETS = ("B", "MS")


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — ⛔ a quick number is NEVER a reported number."""
    c = config.gen_encoder_unfreeze
    c.warmup_steps = c.quick_warmup_steps
    c.warmup_batch = c.quick_warmup_batch
    c.joint_steps = c.quick_joint_steps
    c.joint_batch = c.quick_joint_batch
    c.n_train = c.quick_n_train
    c.k_wells = c.quick_k_wells
    c.n_modes = c.quick_n_modes
    c.max_matched_seen = c.quick_max_matched_seen
    c.book_rows = c.quick_book_rows
    c.dec_recipe = c.quick_dec_recipe
    c.seeds = list(c.quick_seeds)
    return config


def resolve_data_root(config: CHLUConfig, dataset: str) -> str:
    """⛔ R26.7: the **dataset** root, resolved loudly.  There is no silent default.

    ⚠ **The ambiguity this function exists to close, stated explicitly.**  The
    loaders take a *dataset* directory (``shapes3d_root(root)`` uses an explicit
    ``root`` **verbatim**) while ``gen_data.data_root`` and ``$CHLU_GEN_DATA_ROOT``
    name the *gen* directory that holds one subdirectory per dataset.  The two
    are one level apart, and confusing them fails deep inside a loader — or, in
    the case that R26 ruling 7 named, silently skips a set and reports null.

    Resolution order, and no fallback beyond it:
      1. ``gen_data.data_root`` — taken **verbatim** (the shipped convention), and
         ``<root>/<dataset>`` is accepted too if the verbatim path has no images.
      2. ``$CHLU_GEN_DATA_ROOT`` -> ``<env>/<dataset>``.
      3. ⛔ **raise** — there is no repo-relative default here on purpose: it does
         not exist in a worktree and it does not exist on a cluster node
         (``push_repo.sh`` excludes ``.claude/``).
    """
    from ..core.gen_bridge import IMAGE_KEYS  # noqa: F401  (dataset is known here)

    cand = []
    explicit = getattr(config.gen_data, "data_root", None)
    if explicit:
        cand += [Path(explicit), Path(explicit) / dataset]
    env = os.environ.get("CHLU_GEN_DATA_ROOT")
    if env:
        cand += [Path(env) / dataset, Path(env)]
    if not cand:
        raise ValueError(
            "no gen-data root: set gen_data.data_root or $CHLU_GEN_DATA_ROOT. "
            "⛔ There is no default here on purpose — a silently-skipped data set "
            "is 'declared not-run reported as null' in code form (R26.7)."
        )
    for c in cand:
        if c.is_dir() and any(c.glob("*.h5")) or (c.is_dir() and any(c.glob("*.npz"))):
            return str(c)
    raise FileNotFoundError(
        f"no {dataset} data under any of {[str(c) for c in cand]} — ⛔ refusing to "
        "guess a root (R26.7)"
    )


def out_root(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    """⛔ A FRESH root.  Nothing this spoke writes lands in a bridge/store/cell path."""
    c = config.gen_encoder_unfreeze
    if c.out_root:
        base = Path(c.out_root)
    else:
        base = (
            Path(__file__).resolve().parents[2]
            / ".claude"
            / "outputs"
            / "gen-encoder-unfreeze"
            / "runs"
        )
    return base / f"{dataset}_{pretext}"


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Rows: train (fitting), B (held out), MS (the matched-seen corner companion)
# ---------------------------------------------------------------------------


def select_unfreeze_rows(dataset: str, split, values, config: CHLUConfig, seed: int) -> dict:
    """Train / B / MS rows.  ⛔ Not one held-out row is ever fitted (re-asserted)."""
    from .exp_gen_store import matched_seen_rows

    c = config.gen_encoder_unfreeze
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_914)
    pool = np.asarray(split.train_rows, np.int64)
    n = min(int(c.n_train), pool.size)
    train = np.sort(rng.permutation(pool)[:n])
    # ⭐ the address-book fitting rows: a SUPERSET of the training rows when
    # ``book_rows`` asks for more, so the two arms' books are estimated from the
    # same rows and the count is not an artefact of the fitting budget.
    n_book = min(int(c.book_rows or n), pool.size)
    book = train if n_book <= n else np.sort(
        np.union1d(train, rng.permutation(np.setdiff1d(pool, train))[: n_book - n])
    )
    # ⛔ MS must be rows NO fitting stage touched — exclude the BOOK, not just the
    # training rows, or the matched-seen companion is scored on rows the address
    # book was estimated from.
    ms = matched_seen_rows(dataset, split, values, book)
    if int(c.max_matched_seen) and ms.size > int(c.max_matched_seen):
        ms = np.sort(rng.permutation(ms)[: int(c.max_matched_seen)])
    return {
        "train": train,
        "book": book,
        "B": np.asarray(split.test_rows, np.int64),
        "MS": ms,
    }


# ---------------------------------------------------------------------------
# ⭐ The shared warm start — bit-identical for BOTH arms, by construction
# ---------------------------------------------------------------------------


def build_warm_start(dataset, pretext, seed, imgs, labels, grid, config, log=print):
    """``(encoder, store, decoder, info)``: the triple both arms start from.

    ⭐ Both arms are handed **this same object**, so the comparison is about stage
    2 alone; the manifest records the triple's three parameter hashes and the
    arms' manifests assert them.
    """
    c = config.gen_encoder_unfreeze
    gb = config.gen_bridge
    # the pretext warm start uses the bridge's own encoder knobs + our step budget
    gb.enc_steps, gb.enc_batch = int(c.warmup_steps), int(c.warmup_batch)
    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_914)
    key, ek, sk, dk = jax.random.split(key, 4)

    log(f"  warm start: encoder pretext={pretext}, {gb.enc_steps} steps …")
    encoder, aux, hist = train_encoder(
        imgs, pretext, gb, ek, labels=labels, cardinalities=grid.cardinalities, log=log
    )
    Z = np.asarray(jax.vmap(encoder)(jnp.asarray(imgs, np.float32)))
    Zf = Z.reshape((Z.shape[0], -1)).astype(np.float32)

    log(f"  grouping: K={c.k_wells} over D={Zf.shape[1]} (method={c.grouping_method}) …")
    labels_g, ginfo = correlation_grouping(
        channel_correlation(Zf),
        int(c.k_wells),
        method=str(c.grouping_method),
        min_well_frac=float(c.min_well_frac),
        return_info=True,
    )
    wc = well_channels_from_labels(labels_g, int(c.k_wells))
    store = init_store(Zf, wc, int(c.n_modes), encoder.feat_shape, sk, seed=int(seed))

    recipe = resolve_decoder_cfg(gb, str(c.dec_recipe))
    if recipe.name != str(c.dec_recipe):
        raise AssertionError(f"decoder recipe did not resolve to {c.dec_recipe!r}")
    decoder = make_decoder(encoder.feat_shape, tuple(imgs.shape[1:]), recipe, dk)

    info = {
        "encoder_param_hash": param_hash(encoder),
        "encoder_param_count": param_count(encoder),
        "pretext_head_param_hash": param_hash(aux),
        "pretext_head_discarded": True,
        "warmup_loss_first": float(hist[0]) if hist else None,
        "warmup_loss_final": float(hist[-1]) if hist else None,
        "warmup_steps": int(gb.enc_steps),
        "warmup_batch": int(gb.enc_batch),
        "feat_shape": [int(x) for x in encoder.feat_shape],
        "D": int(encoder.feature_dim),
        "grouping": _jsonable(ginfo),
        "store_param_hash": param_hash(store),
        "store_bytes": store_bytes(store),
        "decoder_param_hash": param_hash(decoder),
        "decoder_param_count": param_count(decoder),
        "decoder_recipe": recipe.as_dict(),
    }
    return encoder, store, decoder, info, key


# ---------------------------------------------------------------------------
# ⭐ Scoring: the distinct-code count on B and its matched-seen companion
# ---------------------------------------------------------------------------


def code_report(book, grid, rows_by_set: dict, requests_by_set: dict) -> dict:
    """Request → posterior → committed code, per request set.

    ⛔ Nothing on this path touches a feature tensor: ``address_requests`` takes a
    **factor assignment**.  The feature-side count is reported separately, as a
    companion, and is a different quantity.
    """
    out = {}
    for name, reqs in requests_by_set.items():
        R = request_labels(reqs, grid)
        if R is None:
            raise AssertionError(f"request set {name!r} contains an invalid request")
        post = request_posterior(book, np.asarray(R, np.int64))
        comm = compose_codes(post)
        out[name] = {
            "n_requests": int(R.shape[0]),
            "n_distinct_code_configs": int(distinct_codes(comm["code"])),
            "distinct_fraction": (
                float(distinct_codes(comm["code"]) / R.shape[0]) if R.shape[0] else float("nan")
            ),
            "mean_confidence": float(comm["mean_confidence"]),
            "_code": np.asarray(comm["code"], np.int64),
        }
    return out


def run_arm(
    arm: str,
    dataset: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    warm,
    warm_info: dict,
    imgs_train: np.ndarray,
    labels_train: np.ndarray,
    imgs_book: np.ndarray,
    labels_book: np.ndarray,
    grid,
    rows: dict,
    imgs_by_set: dict,
    requests_by_set: dict,
    key,
    log: Callable = print,
) -> dict:
    """One arm, end to end.  ⛔ The bypass check runs before any number is kept."""
    c = config.gen_encoder_unfreeze
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    enc0, store0, dec0 = warm
    adir = Path(out_dir) / arm
    adir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    encoder, store, decoder, rep, key = train_joint(
        enc0, store0, dec0, imgs_train, c, key, unfreeze=(arm == "unfrozen"), log=log
    )
    train_seconds = time.time() - t0

    # ---- features + codes, this arm's own encoder ----------------------------
    def encode(x):
        return np.asarray(
            jax.vmap(encoder)(jnp.asarray(np.asarray(x, np.float32)))
        ).reshape((len(x), -1)).astype(np.float32)

    Z_train = encode(imgs_train)

    # ---- ⛔⭐ THE BYPASS CHECK, BEFORE ANY NUMBER IS RECORDED ------------------
    bypass = assert_no_feature_bypass(store, Z_train, code_grad=str(c.code_grad))
    shipped = assert_no_bypass(store, Z_train)
    log(
        f"  ⛔⭐ no-bypass OK: decoder input == read_from_codes to the bit; "
        f"same-codes-different-features delta = "
        f"{bypass['max_abs_delta_same_codes_different_features']}; "
        f"channel = {bypass['design_bits']} bits (K*log2(M))"
    )

    # ---- the address book (R8: the declared, arm-identical interface) ---------
    # ⚠ Fitted on ``rows["book"]`` (a superset of the training rows when
    # ``book_rows`` asks for more).  ⛔ Both arms use the SAME rows, so the
    # request-side count differs only through the arm's own codes.
    Z_book = Z_train if imgs_book is imgs_train else encode(imgs_book)
    codes_book = address_codes(store, Z_book)
    book = fit_address_book(
        codes_book, labels_book, grid.cardinalities, smoothing=float(c.book_smoothing)
    )

    # ---- ⭐ distinct codes per request set, request-side and feature-side ------
    reports = code_report(book, grid, rows, requests_by_set)
    feature_side = {}
    recon = {}
    for name, imgs in imgs_by_set.items():
        Zs = encode(imgs)
        cs = address_codes(store, Zs)
        feature_side[name] = {
            "n_rows": int(Zs.shape[0]),
            "n_distinct_code_configs": int(distinct_codes(cs)),
            "channel": realised_channel_bits(store, cs),
        }
        read = np.asarray(read_from_codes(store, cs))
        px = np.asarray(jax.vmap(decoder)(jnp.asarray(read)))
        recon[name] = {
            "pixel_mse_through_store": float(
                np.mean((px - np.asarray(imgs, np.float32)) ** 2)
            ),
            "feature_mse_of_the_hard_read": float(
                np.mean((read.reshape(Zs.shape) - Zs) ** 2)
            ),
        }
        if bool(c.persist_code_tuples):
            # ⭐ R30's lesson: persist the TUPLES, never only the count.
            np.save(adir / f"codes_features_{name}.npy", cs)

    if bool(c.persist_code_tuples):
        for name, r in reports.items():
            np.save(adir / f"codes_request_{name}.npy", r["_code"])
    reports = {k: {kk: vv for kk, vv in v.items() if kk != "_code"} for k, v in reports.items()}

    save_model(encoder, adir / "encoder.pkl", {"arm": arm, "seed": int(seed)})
    save_model(store, adir / "store.pkl", {"arm": arm, "seed": int(seed)})
    save_model(decoder, adir / "decoder.pkl", {"arm": arm, "seed": int(seed)})

    manifest = {
        "schema": ARM_SCHEMA,
        "status": "ok",
        "dataset": dataset,
        "pretext": str(c.pretext),
        "arm": arm,
        "seed": int(seed),
        "void_as_a_reported_number": bool(getattr(config, "_quick", False)),
        "warm_start": warm_info,
        "compute": {
            **rep["info"],
            "warmup_steps": int(warm_info["warmup_steps"]),
            "warmup_batch": int(warm_info["warmup_batch"]),
            "total_gradient_steps": int(warm_info["warmup_steps"]) + int(rep["info"]["steps"]),
            "train_seconds": float(train_seconds),
            "note": (
                "⭐ MATCHED COMPUTE: both arms share the same warm start and run the "
                "same joint_steps on the same batches from the same key. The unfrozen "
                "arm updates strictly MORE PARAMETERS PER STEP — that is the "
                "treatment, not a budget difference."
            ),
        },
        "no_bypass_joint_path": _jsonable(bypass),
        "no_bypass_shipped_read": _jsonable(shipped),
        "channel": {
            "design_bits": float(store.bits_per_read),
            "formula": "K*log2(M)",
            "k_wells": int(store.n_wells),
            "n_modes": int(store.n_modes),
            "declared_constant": "R20",
            "widened": False,
        },
        "store_bytes": store_bytes(store),
        "param_hashes": {
            "encoder": param_hash(encoder),
            "store": param_hash(store),
            "decoder": param_hash(decoder),
        },
        "rows": {k: int(np.asarray(v).size) for k, v in rows.items()},
        "address_book": {
            "n_fit_rows": int(Z_book.shape[0]),
            "smoothing": float(c.book_smoothing),
            "n_distinct_codes_on_the_fit_rows": int(distinct_codes(codes_book)),
            "note": (
                "⚠ the request-side count is an ESTIMATE over sum_f V_f * K * M "
                "counting cells; report n_fit_rows beside it, always."
            ),
        },
        "distinct_codes_request_side": reports,
        "distinct_codes_feature_side": feature_side,
        "reconstruction": recon,
        "loss_history_last": rep["history"][-1] if rep["history"] else None,
    }
    _write_json(manifest, adir / "manifest.json")
    _write_json({"history": rep["history"]}, adir / "history.json")
    return manifest


def run_seed(
    dataset: str,
    pretext: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    arms: Optional[list] = None,
    log: Callable = print,
) -> dict:
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    c = config.gen_encoder_unfreeze
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}, got {pretext!r}")
    if pretext != "recon":
        tlog(f"⚠ pretext={pretext!r} is the DIAGNOSTIC arm (R1); recon is the headline")
    arms = list(arms or c.arms)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_data_root(config, dataset)
    config.gen_data.data_root = root
    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(root)
    grid = split.grid
    rows = select_unfreeze_rows(dataset, split, values, config, seed)
    for name in ("train", "book"):
        leaked = np.intersect1d(rows[name], split.test_rows)
        if leaked.size:
            raise AssertionError(
                f"{leaked.size} held-out rows are in the {name!r} fitting set"
            )
    if np.intersect1d(rows["book"], rows["MS"]).size:
        raise AssertionError("the matched-seen control overlaps the address book's rows")
    tlog(
        f"rows: train={rows['train'].size} book={rows['book'].size} "
        f"B={rows['B'].size} MS={rows['MS'].size} (0 held-out rows fitted, asserted)"
    )

    tlog("streaming images …")
    imgs_train = stream_images(dataset, rows["train"], root, int(config.gen_bridge.cache_block))
    labels_train = factor_labels(grid, rows["train"])
    same_book = rows["book"].size == rows["train"].size and np.array_equal(
        rows["book"], rows["train"]
    )
    imgs_book = imgs_train if same_book else stream_images(
        dataset, rows["book"], root, int(config.gen_bridge.cache_block)
    )
    labels_book = labels_train if same_book else factor_labels(grid, rows["book"])
    imgs_by_set = {
        k: stream_images(dataset, rows[k], root, int(config.gen_bridge.cache_block))
        for k in REQUEST_SETS
    }
    requests_by_set = {k: requests_from_rows(grid, rows[k]) for k in REQUEST_SETS}

    warm_path = out_dir / "warm_start.json"
    tlog("building the SHARED warm start (both arms start here, bit-identically) …")
    encoder, store, decoder, warm_info, key = build_warm_start(
        dataset, pretext, seed, imgs_train, labels_train, grid, config, log=tlog
    )
    _write_json(warm_info, warm_path)
    tlog(
        f"warm start: D={warm_info['D']} K={c.k_wells} M={c.n_modes} "
        f"channel={float(c.k_wells) * np.log2(float(c.n_modes)):.1f} bits · "
        f"decoder recipe={warm_info['decoder_recipe']['name']}"
    )
    if warm_info["decoder_recipe"]["name"] != str(c.expected_decoder_recipe):
        raise AssertionError(
            f"⛔ decoder recipe is {warm_info['decoder_recipe']['name']!r}, expected "
            f"{c.expected_decoder_recipe!r} — the 'baseline' recipe FAILED the gate"
        )

    out = {}
    for arm in arms:
        man_path = out_dir / arm / "manifest.json"
        if bool(c.skip_if_exists) and man_path.exists():
            man = json.loads(man_path.read_text())
            if man.get("status") == "ok" and man.get("schema") == ARM_SCHEMA:
                tlog(f"arm {arm}: artefact exists ({ARM_SCHEMA}) — SKIPPED")
                out[arm] = man
                continue
            tlog(
                f"⚠ arm {arm}: existing artefact is status={man.get('status')!r} "
                f"schema={man.get('schema')!r} (want {ARM_SCHEMA!r}) — RE-RUNNING"
            )
        tlog(f"arm {arm}: stage 2, {c.joint_steps} steps x batch {c.joint_batch} …")
        out[arm] = run_arm(
            arm, dataset, seed, out_dir, config, (encoder, store, decoder), warm_info,
            imgs_train, labels_train, imgs_book, labels_book, grid, rows,
            imgs_by_set, requests_by_set,
            jax.random.fold_in(key, 20_260_914), log=tlog,
        )
        r = out[arm]["distinct_codes_request_side"]
        tlog(
            f"arm {arm}: distinct codes  B={r['B']['n_distinct_code_configs']}"
            f"/{r['B']['n_requests']}  MS={r['MS']['n_distinct_code_configs']}"
            f"/{r['MS']['n_requests']}"
        )

    summary = summarise_seed(out, warm_info, seed, dataset, pretext)
    _write_json(summary, out_dir / "seed_summary.json")
    return summary


def summarise_seed(arm_manifests: dict, warm_info: dict, seed, dataset, pretext) -> dict:
    """The two arms side by side — ⛔ never the unfrozen arm alone."""
    table = {}
    for arm, man in arm_manifests.items():
        r = man["distinct_codes_request_side"]
        f = man["distinct_codes_feature_side"]
        table[arm] = {
            "distinct_B": r["B"]["n_distinct_code_configs"],
            "n_B": r["B"]["n_requests"],
            "distinct_MS": r["MS"]["n_distinct_code_configs"],
            "n_MS": r["MS"]["n_requests"],
            "distinct_B_feature_side": f["B"]["n_distinct_code_configs"],
            "distinct_MS_feature_side": f["MS"]["n_distinct_code_configs"],
            "mean_confidence_B": r["B"]["mean_confidence"],
            "address_book_fit_rows": man.get("address_book", {}).get("n_fit_rows"),
            "gradient_steps": man["compute"]["gradient_steps"],
            "images_seen": man["compute"]["images_seen"],
            "n_params_optimised": man["compute"]["n_params_optimised"],
            "encoder_changed": man["compute"]["encoder_changed"],
            "train_seconds": man["compute"]["train_seconds"],
            "pixel_mse_B": man["reconstruction"]["B"]["pixel_mse_through_store"],
            "design_bits": man["channel"]["design_bits"],
        }
    delta = None
    if set(ARMS) <= set(table):
        delta = {
            "distinct_B": table["unfrozen"]["distinct_B"] - table["frozen"]["distinct_B"],
            "distinct_MS": table["unfrozen"]["distinct_MS"] - table["frozen"]["distinct_MS"],
            "budgets_matched": (
                table["unfrozen"]["gradient_steps"] == table["frozen"]["gradient_steps"]
                and table["unfrozen"]["images_seen"] == table["frozen"]["images_seen"]
            ),
        }
    return {
        "schema": "chlu-gen-encoder-unfreeze-summary/1",
        "dataset": dataset,
        "pretext": pretext,
        "seed": int(seed),
        "warm_start": {
            k: warm_info[k] for k in ("encoder_param_hash", "store_param_hash",
                                      "decoder_param_hash", "D", "feat_shape")
        },
        "arms": table,
        "unfrozen_minus_frozen": delta,
        "note": (
            "⛔ The unfrozen arm is only readable beside the frozen arm at matched "
            "compute; 'budgets_matched' must be true for the delta to mean anything."
        ),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_experiment_gen_encoder(
    config: Optional[CHLUConfig] = None,
    dataset: Optional[str] = None,
    pretext: Optional[str] = None,
    seeds: Optional[list] = None,
    arms: Optional[list] = None,
    out: Optional[str] = None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
        config._quick = True  # noqa: SLF001  (provenance flag read into the manifest)
    c = config.gen_encoder_unfreeze
    dataset = dataset or c.dataset
    pretext = pretext or c.pretext
    seeds = list(seeds or c.seeds)
    base = Path(out) if out else out_root(config, dataset, pretext)
    results = {}
    for seed in seeds:
        log(f"=== gen-encoder-unfreeze · {dataset}/{pretext} · seed {seed} ===")
        results[str(seed)] = run_seed(
            dataset, pretext, int(seed), base / f"seed_{seed}", config, arms, log=log
        )
    payload = {
        "schema": "chlu-gen-encoder-unfreeze-run/1",
        "dataset": dataset,
        "pretext": pretext,
        "seeds": [int(s) for s in seeds],
        "quick": bool(quick),
        "void_as_a_reported_number": bool(quick),
        "out_root": str(base),
        "per_seed": results,
    }
    _write_json(payload, base / "unfreeze_summary.json")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="encoder unfreeze with the store in the loop")
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None, choices=[None, *PRETEXTS])
    p.add_argument("--stage", default="encoder", choices=list(STAGES))
    p.add_argument(
        "--arm", default="both", choices=["both", *ARMS, *LAUNCHER_ARM_ALIASES]
    )
    p.add_argument("--seed", type=int, action="append", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--data-root", default=None)
    #: ⚠ Sets BOTH the full and the quick value, because ``--quick`` overwrites
    #: the full one; a flag that a later stage silently reverts is worse than no
    #: flag (§7.10 precedent: ``--quick`` was unwired for the dynamics path).
    p.add_argument("--book-rows", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    if args.data_root:
        config.gen_data.data_root = args.data_root
    if args.book_rows is not None:
        config.gen_encoder_unfreeze.book_rows = int(args.book_rows)
        config.gen_encoder_unfreeze.quick_book_rows = int(args.book_rows)
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.stage == "aggregate":
        base = Path(args.out) if args.out else out_root(
            config, args.dataset or config.gen_encoder_unfreeze.dataset,
            args.pretext or config.gen_encoder_unfreeze.pretext,
        )
        print(json.dumps(json.loads((base / "unfreeze_summary.json").read_text()), indent=2))
        return 0
    arm = LAUNCHER_ARM_ALIASES.get(args.arm, args.arm)
    arms = None if arm == "both" else [arm]
    run_experiment_gen_encoder(
        config=config,
        dataset=args.dataset,
        pretext=args.pretext,
        seeds=args.seed,
        arms=arms,
        out=args.out,
        quick=args.quick,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
