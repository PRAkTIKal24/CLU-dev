"""encoder loop."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

PINS = {
    "dataset": "shapes3d",
    "pretext": "recon",
    "warm_steps": 1500,
    "warm_batch": 128,
    "n_train": 8192,
    "joint_steps": 400,
    "joint_batch": 64,
    "joint_lr": 0.001,
    "code_grad": "straight_through",
    "beta_codebook": 1.0,
    "beta_commitment": 0.25,
    "beta_swap": "auto",
    "swap_wells_per_step": 1,
    "k_wells": 5,
    "n_modes": 64,
    "grouping_method": "auto",
    "min_well_frac": 0.1,
    "ownership_pairs": 256,
    "ownership_batch": 256,
    "dec_recipe": "loop_v1_short",
    "dec_recipe_architecture_of": "v1_capacity_schedule",
    "dec_refit_steps": 400,
    "warm_dec_steps": 200,
    "probe_steps": 1500,
    "probe_batch": 256,
    "probe_lr": 0.001,
    "probe_n_fit": 8192,
    "probe_n_eval": 4096,
    "cache_dtype": "float16",
    "cache_block": 8192,
    "eval_batch": 256,
    "rival_families": ["mlp_180k", "cont_mlp", "transformer", "lstm"],
    "rival_val_rows": 512,
    "arms_without_render_decoder": ["alone_wc"],
    "seeds": [0, 1, 2],
}
ARMS = ("alone", "frozen", "loop_nosw", "loop_swap", "alone_wc")
ARM_SPEC = {
    "alone": {"store": False, "unfreeze": True, "swap": False, "wallclock_match": False},
    "frozen": {"store": True, "unfreeze": False, "swap": False, "wallclock_match": False},
    "loop_nosw": {"store": True, "unfreeze": True, "swap": False, "wallclock_match": False},
    "loop_swap": {"store": True, "unfreeze": True, "swap": True, "wallclock_match": False},
    "alone_wc": {"store": False, "unfreeze": True, "swap": False, "wallclock_match": True},
}
ARM_CLASS = {
    "alone": "CONTROL -- the encoder trained ALONE (the target's named control)",
    "frozen": "CONTROL -- the shipped configuration, encoder frozen at matched compute",
    "loop_nosw": "COLUMN -- unfrozen WITHOUT the swap loss (isolates the new ingredient)",
    "loop_swap": "ARM -- the store in the loop, swap loss + reconstruction",
    "alone_wc": "CONTROL -- `alone` matched on WALL CLOCK instead of on steps",
}
ARM_SCHEMA = "nclu-encoder_loop-arm/1"
STAGES = ("warm", "train", "cache", "probe", "rivals", "report", "cloud")
CLOUD_STAGES = ("warm", "train")


def register_short_recipe() -> dict:
    from nclu.core.gen_bridge import DECODER_RECIPES

    base = dict(DECODER_RECIPES[str(PINS["dec_recipe_architecture_of"])])
    base["dec_steps"] = int(PINS["dec_refit_steps"])
    DECODER_RECIPES[str(PINS["dec_recipe"])] = base
    return dict(base)


def _now() -> float:
    return time.time()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def data_root() -> Path:
    from nclu.paths import data_root as _root

    p = _root() / PINS["dataset"]
    if not p.is_dir():
        raise FileNotFoundError(f"no {PINS['dataset']} under {p}")
    return p


def bridge_seed_dir(seed: int) -> Path:
    r = os.environ.get("NCLU_ENCODER_ROOT")
    if r:
        base = Path(r)
    else:
        from nclu.paths import weights_dir

        base = weights_dir() / "encoder"
    return base / f"{PINS['dataset']}_{PINS['pretext']}" / f"seed_{int(seed)}"


def classifier_dir(seed: int) -> Path:
    from nclu.paths import weights_dir

    return weights_dir() / "classifier" / f"seed_{int(seed)}"


def arm_dir(out: Path, seed: int, arm: str) -> Path:
    return Path(out) / f"seed_{int(seed)}" / arm


def seed_dir(out: Path, seed: int) -> Path:
    return Path(out) / f"seed_{int(seed)}"


def cache_root(out: Path, seed: int, arm: str) -> Path:
    return arm_dir(out, seed, arm) / "cache"


def cache_seed_dir(out: Path, seed: int, arm: str) -> Path:
    return cache_root(out, seed, arm) / f"{PINS['dataset']}_{PINS['pretext']}" / f"seed_{int(seed)}"


def declift_dir(out: Path, seed: int, arm: str) -> Path:
    """The arm's decoder, in a decoder-lift-shaped directory the Judge can load.

    The Judge asserts the decoder on disk against ``winner_hashes.json`` *in the same
    directory*.  That assertion is a self-consistency check -- "the file is the one this
    manifest describes" -- and it still does its job here.  What it does NOT do, and must
    never be read as doing, is certify that this is the frozen decoder-lift winner: it is
    not, it cannot be, and the manifest written beside it says so in a field of its own.
    """
    return arm_dir(out, seed, arm) / "declift"


def load_external_classifier(seed: int, log: Callable = print):
    """THE instrument.  Loaded, hash-asserted against its own manifest, never refit."""
    from nclu.core.gen_bridge import param_hash
    from nclu.utils.checkpoints import load_model

    cdir = classifier_dir(seed)
    ckpt = cdir / "external_classifier.pkl"
    if not (ckpt.exists() or ckpt.with_suffix(".eqx").exists()):
        raise FileNotFoundError(
            f"no external classifier at {ckpt} -- this seed's instrument does not exist and its arms are reported NOT RUN, never as zeros"
        )
    man = _read_json(cdir / "classifier_manifest.json")
    model = load_model(ckpt)
    got = param_hash(model)
    exp = str((man.get("classifier") or {}).get("param_hash"))
    if exp and got != exp:
        raise AssertionError(
            f"external classifier hash {got[:16]} does not match its own manifest {exp[:16]} -- the instrument on disk is not the one that was gated"
        )
    info = {
        "path": str(ckpt),
        "param_hash": got,
        "seed": int(seed),
        "gate_balacc_on_real": ((man.get("gate") or {}).get("external") or {}).get(
            "balanced_accuracy"
        ),
        "note": "ONE instrument for every arm and every column; hash-asserted, never fitted here",
    }
    log(f"  judge: external classifier seed {seed} {got[:12]} (hash-asserted)")
    return (model, info)


def select_rows(seed: int, quick: bool = False, log: Callable = print) -> dict:
    from nclu.core.gen_bridge import dataset_module

    mod = dataset_module(PINS["dataset"])
    split, values = mod.build_split(str(data_root()))
    grid = split.grid
    rng = np.random.default_rng(int(seed) * 7919 + 20260920)
    pool = np.asarray(split.train_rows, np.int64)
    perm = rng.permutation(pool)
    n_tr = min(512 if quick else int(PINS["n_train"]), pool.size)
    train = np.sort(perm[:n_tr])
    rest = perm[n_tr:]
    n_pf = min(512 if quick else int(PINS["probe_n_fit"]), rest.size)
    probe_fit = np.sort(rest[:n_pf])
    n_pe = min(256 if quick else int(PINS["probe_n_eval"]), rest.size - n_pf)
    probe_eval = np.sort(rest[n_pf : n_pf + n_pe])
    rows = {"train": train, "probe_fit": probe_fit, "probe_eval": probe_eval}
    test = np.asarray(split.test_rows, np.int64)
    for name, r in rows.items():
        leak = np.intersect1d(r, test)
        if leak.size:
            raise AssertionError(f"{leak.size} held-out rows are in the {name!r} fitting set")
    log(
        f"  rows: train={train.size} probe_fit={probe_fit.size} probe_eval={probe_eval.size} (0 held-out rows fitted, asserted)"
    )
    return {"rows": rows, "grid": grid, "split": split, "values": values}


def stage_warm(out: Path, seed: int, quick: bool, log: Callable = print) -> dict:
    from nclu.config import get_default_config
    from nclu.core.gen_bridge import (
        make_decoder,
        param_count,
        param_hash,
        resolve_decoder_cfg,
        stream_images,
        train_encoder,
    )
    from nclu.core.gen_store import (
        channel_correlation,
        correlation_grouping,
        init_store,
        store_bytes,
        well_channels_from_labels,
    )
    from nclu.core import gen_swap_loss as SL
    from nclu.core import gen_swap_probe as SP
    from nclu.eval.gen_probe import factor_labels
    from nclu.utils.checkpoints import save_model

    sd = seed_dir(out, seed)
    man_p = sd / "warm_start.json"
    if man_p.exists() and _read_json(man_p).get("status") == "ok":
        log(f"  warm start for seed {seed}: already on disk -- SKIPPED")
        return _read_json(man_p)
    sd.mkdir(parents=True, exist_ok=True)
    register_short_recipe()
    cfg = get_default_config()
    gb = cfg.gen_bridge
    gb.enc_steps = 40 if quick else int(PINS["warm_steps"])
    gb.enc_batch = 32 if quick else int(PINS["warm_batch"])
    n_train = 512 if quick else int(PINS["n_train"])
    sel = select_rows(seed, quick=quick, log=log)
    grid, rows = (sel["grid"], sel["rows"])
    root = str(data_root())
    log(f"  streaming {rows['train'].size} training images ...")
    imgs = stream_images(PINS["dataset"], rows["train"], root, int(gb.cache_block))
    labels = factor_labels(grid, rows["train"])
    key = jax.random.PRNGKey(int(seed) * 10007 + 20260920)
    key, ek, sk, dk = jax.random.split(key, 4)
    log(f"  warm start: recon pretext, {gb.enc_steps} steps x {gb.enc_batch} ...")
    t0 = _now()
    encoder, aux, hist = train_encoder(
        imgs, PINS["pretext"], gb, ek, labels=labels, cardinalities=grid.cardinalities, log=log
    )
    warm_seconds = _now() - t0
    Z = np.asarray(jax.vmap(encoder)(jnp.asarray(imgs, np.float32)))
    Zf = Z.reshape((Z.shape[0], -1)).astype(np.float32)
    D = int(Zf.shape[1])
    log(f"  grouping: K={PINS['k_wells']} over D={D} ...")
    labels_g, ginfo = correlation_grouping(
        channel_correlation(Zf),
        int(PINS["k_wells"]),
        method=str(PINS["grouping_method"]),
        min_well_frac=float(PINS["min_well_frac"]),
        return_info=True,
    )
    wc = well_channels_from_labels(labels_g, int(PINS["k_wells"]))
    store = init_store(Zf, wc, int(PINS["n_modes"]), encoder.feat_shape, sk, seed=int(seed))
    _rec = resolve_decoder_cfg(gb, str(PINS["dec_recipe"]))
    _src = resolve_decoder_cfg(gb, str(PINS["dec_recipe_architecture_of"]))
    if (
        _rec.name != str(PINS["dec_recipe"])
        or tuple(_rec.channels) != tuple(_src.channels)
        or int(_rec.blocks_per_stage) != int(_src.blocks_per_stage)
        or (_rec.lr_schedule != _src.lr_schedule)
    ):
        raise AssertionError(
            f"the short recipe {_rec.as_dict()} is not the winner's ARCHITECTURE {_src.as_dict()} with only dec_steps changed -- the one thing this study is allowed to cut is the step count"
        )
    log(
        f"  decoder recipe: {_rec.name} = {PINS['dec_recipe_architecture_of']}'s architecture at {_rec.steps} steps (the gated winner runs {_src.steps}); DECLARED: these renders are NOT comparable with the incumbent's absolutes"
    )
    from nclu.core.gen_bridge import train_decoder_from_features
    from nclu.core.gen_bridge import DECODER_RECIPES

    warm_name = "loop_v1_warm"
    _warm = dict(DECODER_RECIPES[str(PINS["dec_recipe_architecture_of"])])
    _warm["dec_steps"] = 20 if quick else int(PINS["warm_dec_steps"])
    DECODER_RECIPES[warm_name] = _warm
    log(
        f"  warm decoder: {warm_name} ({_warm['dec_steps']} steps, the winner's architecture) -- SHARED bit-identically by every arm ..."
    )
    decoder, wdhist, recipe = train_decoder_from_features(
        encoder.feat_shape, Z, imgs, gb, dk, log=log, recipe=warm_name
    )
    classifier, cls_info = load_external_classifier(seed, log=log)
    judge = SL.DecodeJudge(
        decoder,
        classifier,
        encoder.feat_shape,
        {"provenance": "the SHARED, FITTED warm decoder + THE classifier"},
    )
    n_pairs = 32 if quick else int(PINS["ownership_pairs"])
    Y = np.asarray(labels, np.int64)
    pairs = SP.make_pairs(Y, n_pairs, rng_seed=SL.OWNERSHIP_PAIR_SEED)
    log(f"  deriving FIXED ownership on {pairs.shape[0]} pinned pairs ...")
    own_rec = SL.derive_fixed_ownership(
        judge,
        Zf[pairs[:, 0]],
        Zf[pairs[:, 1]],
        Y[pairs[:, 0]],
        Y[pairs[:, 1]],
        [np.asarray(w) for w in store.well_channels],
        D,
        batch=int(PINS["ownership_batch"]),
        log=log,
    )
    save_model(encoder, sd / "warm_encoder.pkl", {"seed": int(seed)})
    save_model(store, sd / "warm_store.pkl", {"seed": int(seed)})
    save_model(decoder, sd / "warm_decoder.pkl", {"seed": int(seed)})
    np.savez(
        sd / "warm_rows.npz",
        **{k: v for k, v in sel["rows"].items()},
        own=own_rec["own"],
        well_labels=np.asarray(labels_g, np.int64),
    )
    man = {
        "schema": "nclu-encoder_loop-warm/1",
        "status": "ok",
        "seed": int(seed),
        "quick": bool(quick),
        "void_as_a_reported_number": bool(quick),
        "pins": dict(PINS),
        "D": D,
        "feat_shape": [int(x) for x in encoder.feat_shape],
        "encoder_param_hash": param_hash(encoder),
        "encoder_param_count": param_count(encoder),
        "store_param_hash": param_hash(store),
        "store_bytes": store_bytes(store),
        "decoder_param_hash": param_hash(decoder),
        "decoder_param_count": param_count(decoder),
        "decoder_recipe": recipe.as_dict(),
        "grouping": _jsonable(ginfo),
        "well_sizes": [int(np.asarray(w).size) for w in store.well_channels],
        "warm_steps": int(gb.enc_steps),
        "warm_batch": int(gb.enc_batch),
        "warm_seconds": float(warm_seconds),
        "warm_loss_first": float(hist[0]) if hist else None,
        "warm_loss_final": float(hist[-1]) if hist else None,
        "warm_decoder_steps": int(gb.dec_steps),
        "warm_decoder_loss_first": float(wdhist[0]) if wdhist else None,
        "warm_decoder_loss_final": float(wdhist[-1]) if wdhist else None,
        "n_train_rows": int(rows["train"].size),
        "classifier": cls_info,
        "ownership": {k: v for k, v in own_rec.items() if k != "own"},
        "own": own_rec["own"].tolist(),
        "note": "EVERY arm starts from this triple, bit-identically; ownership is derived here ONCE and is a constant for every arm and every step",
    }
    _write_json(man, man_p)
    log(f"  warm start written -> {man_p}")
    return man


def _load_ckpt_module():
    import importlib.util

    here = Path(__file__).resolve().parents[2] / "scripts" / "cloud" / "ckpt.py"
    if not here.exists():
        raise FileNotFoundError(f"no checkpointer at {here}")
    spec = importlib.util.spec_from_file_location("nclu_cloud_ckpt", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_fingerprint(arm: str, seed: int, steps: int, warm: dict) -> dict:
    """The config whose hash a resume is refused against.

    Everything that determines the RESULT is in it, and nothing that does not.  The
    checkpoint cadence and the remote URI are deliberately OUTSIDE it: a relaunch is
    allowed to checkpoint more often or to a different bucket, and refusing that would
    make the mechanism fight the pre-emption it exists to survive.  The warm start's
    three parameter hashes ARE in it, so a checkpoint cannot be resumed onto a different
    warm start -- which is the one substitution that would silently void the arm.
    """
    return {
        "study": "encoder_loop",
        "arm": arm,
        "seed": int(seed),
        "steps": int(steps),
        "arm_spec": ARM_SPEC[arm],
        "joint_batch": PINS["joint_batch"],
        "joint_lr": PINS["joint_lr"],
        "code_grad": PINS["code_grad"],
        "beta_codebook": PINS["beta_codebook"],
        "beta_commitment": PINS["beta_commitment"],
        "beta_swap_mode": PINS["beta_swap"],
        "swap_wells_per_step": PINS["swap_wells_per_step"],
        "n_train": PINS["n_train"],
        "warm_encoder_param_hash": warm["encoder_param_hash"],
        "warm_store_param_hash": warm["store_param_hash"],
        "warm_decoder_param_hash": warm["decoder_param_hash"],
        "own": warm["own"],
    }


def _flatten_for_ckpt(params, opt_state, key, step: int) -> dict:
    """The live training state as a flat dict of arrays ``ckpt.py`` can npz.

    Leaf ORDER is the pytree's own, which is deterministic for a structure rebuilt the
    same way -- and it is, because a resume reconstructs ``params``/``opt_state`` from
    the same warm start under the same fingerprint before restoring into them.
    """
    out = {
        "__step__": np.asarray(int(step), np.int64),
        "__key__": np.asarray(jax.random.key_data(key)),
    }
    for i, leaf in enumerate(jax.tree_util.tree_leaves(params)):
        out[f"p{i:05d}"] = np.asarray(leaf)
    for i, leaf in enumerate(jax.tree_util.tree_leaves(opt_state)):
        out[f"o{i:05d}"] = np.asarray(leaf)
    return out


def _restore_from_ckpt(state: dict, params, opt_state):
    """Pour a restored flat dict back into freshly-built pytrees of the same shape."""
    pl, ptree = jax.tree_util.tree_flatten(params)
    ol, otree = jax.tree_util.tree_flatten(opt_state)
    pn = [state[f"p{i:05d}"] for i in range(len(pl))]
    on = [state[f"o{i:05d}"] for i in range(len(ol))]
    for a, b in zip(pl, pn, strict=True):
        if tuple(np.shape(a)) != tuple(np.shape(b)):
            raise AssertionError(
                f"checkpoint leaf shape {np.shape(b)} != live {np.shape(a)} -- the checkpoint does not belong to this model and is refused"
            )
    params = jax.tree_util.tree_unflatten(ptree, [jnp.asarray(x) for x in pn])
    opt_state = jax.tree_util.tree_unflatten(
        otree, [jnp.asarray(x) if np.ndim(x) or np.asarray(x).dtype != object else x for x in on]
    )
    step = int(state["__step__"])
    key = jax.random.wrap_key_data(np.asarray(state["__key__"], np.uint32))
    return (params, opt_state, key, step)


def _arm_filter_spec(encoder, store, decoder, unfreeze: bool, with_store: bool):
    """Optax filter for ``(encoder, store, decoder)``.

    The ``alone`` arms carry the store in the tuple so the pytree shape is identical
    across arms, but mark every store leaf ``False``: an arm with no store on the loss
    path must not have its store drift, or the object the cache is later read through
    would differ between arms for a reason that is not the treatment.
    """
    enc_leaf = (lambda leaf: bool(eqx.is_inexact_array(leaf))) if unfreeze else lambda _: False
    st_leaf = (lambda leaf: bool(eqx.is_inexact_array(leaf))) if with_store else lambda _: False
    return (
        jax.tree_util.tree_map(enc_leaf, encoder),
        jax.tree_util.tree_map(st_leaf, store),
        jax.tree_util.tree_map(lambda leaf: bool(eqx.is_inexact_array(leaf)), decoder),
    )


def train_arm(
    arm: str,
    encoder,
    store,
    decoder,
    classifier,
    images,
    labels,
    own,
    key,
    steps: int,
    log: Callable = print,
    quick: bool = False,
    ckpt=None,
    sync: Optional[Callable] = None,
):
    """Stage 2 for one arm.  Returns ``(encoder, store, decoder, report, key)``.

    Every arm runs the SAME loop over the SAME batches from the SAME key.  What differs
    is (a) whether the store is on the loss path, (b) whether the encoder's leaves are in
    the optimiser, (c) whether the swap term is added.  Nothing else.
    """
    from nclu.core.gen_bridge import joint_loss_terms, param_hash
    from nclu.core import gen_swap_loss as SL

    spec = ARM_SPEC[arm]
    with_store, unfreeze, with_swap = (spec["store"], spec["unfreeze"], spec["swap"])
    code_grad = str(PINS["code_grad"])
    b_code, b_commit = (float(PINS["beta_codebook"]), float(PINS["beta_commitment"]))
    b_swap = PINS["beta_swap"]
    batch = 16 if quick else int(PINS["joint_batch"])
    lr = float(PINS["joint_lr"])
    K = int(store.n_wells)
    enc_hash_before = param_hash(encoder)
    cls_hash_before = param_hash(classifier)
    model = (encoder, store, decoder)
    filt = _arm_filter_spec(encoder, store, decoder, unfreeze, with_store)
    params, static = eqx.partition(model, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    wells_padded, well_sizes = SL.pad_well_channels([np.asarray(w) for w in store.well_channels])
    own_j = np.asarray(own, np.int64)
    beta_swap_basis = None
    X = jnp.asarray(np.asarray(images, np.float32))
    Y = jnp.asarray(np.asarray(labels, np.int32))
    n = int(X.shape[0])
    batch = min(batch, n)
    if with_swap and (not isinstance(b_swap, (int, float))):
        from nclu.core.gen_bridge import joint_loss_terms as _jlt

        _k0 = jax.random.fold_in(key, 777)
        _idx = jax.random.choice(_k0, n, (min(batch, n),), replace=False)
        _ii, _jj = SL.draw_partners(int(_idx.shape[0]), jax.random.fold_in(_k0, 1))
        _rec0, _, _ = _jlt((encoder, store, decoder), X[_idx], code_grad, 0.0)
        _sw0, _ = SL.swap_loss_terms(
            (encoder, store, decoder),
            classifier,
            X[_idx],
            Y[_idx],
            _ii,
            _jj,
            0,
            wells_padded,
            own_j,
            code_grad,
        )
        beta_swap_basis = {"recon_at_warm_start": float(_rec0), "swap_at_warm_start": float(_sw0)}
        b_swap = float(_rec0) / max(float(_sw0), 1e-08)
        log(
            f"    [{arm}] beta_swap=auto -> {b_swap:.6g} (recon_0={float(_rec0):.5f} / swap_0={float(_sw0):.5f}; the two terms enter the loss at EQUAL magnitude, measured on the shared warm start)"
        )
    b_swap = float(b_swap) if isinstance(b_swap, (int, float)) else 0.0

    def plain_recon(m, xb):
        """The `alone` objective: an autoencoder, NO store on the path."""
        enc, _st, dec = m
        z = jax.vmap(enc)(xb)
        rec = jax.vmap(dec)(z)
        return jnp.mean((rec - jnp.asarray(xb, rec.dtype)) ** 2)

    def loss_fn(p, s, xb, yb, ii, jj, well):
        m = eqx.combine(p, s)
        if with_store:
            l_rec, l_code, l_commit = joint_loss_terms(m, xb, code_grad, 0.0)
        else:
            l_rec = plain_recon(m, xb)
            l_code = jnp.asarray(0.0, l_rec.dtype)
            l_commit = jnp.asarray(0.0, l_rec.dtype)
        total = l_rec + b_code * l_code + b_commit * l_commit
        l_swap = jnp.asarray(0.0, l_rec.dtype)
        swap_exact = jnp.asarray(0.0, l_rec.dtype)
        if with_swap:
            l_swap, swap_exact = SL.swap_loss_terms(
                m, classifier, xb, yb, ii, jj, int(well), wells_padded, own_j, code_grad
            )
            total = total + b_swap * l_swap
        return (total, (l_rec, l_code, l_commit, l_swap, swap_exact))

    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    steppers = {}
    for w in range(K if with_swap else 1):

        def make(wv):
            @eqx.filter_jit
            def step(p, s, o, xb, yb, ii, jj):
                (loss, parts), grads = grad_fn(p, s, xb, yb, ii, jj, wv)
                upd, o2 = opt.update(grads, o, p)
                return (eqx.apply_updates(p, upd), o2, loss, parts)

            return step

        steppers[w] = make(w)
    history = []
    start_step = 0
    if ckpt is not None:
        got = ckpt.latest()
        if got is not None:
            cstep, cstate = got
            params, opt_state, key, start_step = _restore_from_ckpt(cstate, params, opt_state)
            log(
                f"    [{arm}] RESUMED from checkpoint step {cstep}; {int(steps) - start_step} steps remain"
            )
        else:
            log(f"    [{arm}] no checkpoint -- fresh start")
    t0 = _now()
    for i in range(start_step, int(steps)):
        key, sk, pk, wk = jax.random.split(key, 4)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        ii, jj = SL.draw_partners(batch, pk)
        w = int(jax.random.randint(wk, (), 0, K)) if with_swap else 0
        params, opt_state, loss, parts = steppers[w](
            params, static, opt_state, X[idx], Y[idx], ii, jj
        )
        history.append(
            {
                "step": i,
                "well": w,
                "loss": float(loss),
                "recon": float(parts[0]),
                "codebook": float(parts[1]),
                "commitment": float(parts[2]),
                "swap": float(parts[3]),
                "swap_exact": float(parts[4]),
            }
        )
        if log is not None and (i == start_step or (i + 1) % max(1, int(steps) // 6) == 0):
            h = history[-1]
            log(
                f"    [{arm:9s}] step {i + 1}/{steps}  loss={h['loss']:.5f} recon={h['recon']:.5f} code={h['codebook']:.5f} commit={h['commitment']:.5f} swap={h['swap']:.5f} swap_exact={h['swap_exact']:.4f}"
            )
        if ckpt is not None and ckpt.maybe_save(
            i + 1, _flatten_for_ckpt(params, opt_state, key, i + 1)
        ):
            log(
                f"    [{arm}] checkpoint @ step {i + 1} (save #{ckpt.n_saves}) -> {ckpt.remote_uri or ckpt.local_dir}"
            )
            if sync is not None:
                sync()
    if ckpt is not None:
        ckpt.maybe_save(
            int(steps), _flatten_for_ckpt(params, opt_state, key, int(steps)), force=True
        )
    train_seconds = _now() - t0
    encoder, store, decoder = eqx.combine(params, static)
    enc_hash_after = param_hash(encoder)
    if not unfreeze and enc_hash_after != enc_hash_before:
        raise AssertionError(
            f"the FROZEN arm's encoder changed during stage 2 ({enc_hash_before[:12]} -> {enc_hash_after[:12]}) -- the matched-compute control is not a control"
        )
    if param_hash(classifier) != cls_hash_before:
        raise AssertionError(
            "the external classifier's parameters MOVED during training -- the judge was in the optimiser and every number of this arm is void"
        )
    info = {
        "arm": arm,
        "arm_class": ARM_CLASS[arm],
        **spec,
        "code_grad": code_grad,
        "beta_codebook": b_code,
        "beta_commitment": b_commit,
        "beta_swap": b_swap if with_swap else 0.0,
        "beta_swap_mode": PINS["beta_swap"],
        "beta_swap_basis": beta_swap_basis,
        "lr": lr,
        "steps": int(steps),
        "batch": int(batch),
        "n_train_rows": n,
        "gradient_steps": int(steps),
        "images_seen": int(steps) * int(batch),
        "train_seconds": float(train_seconds),
        "seconds_per_step": float(train_seconds / max(1, int(steps))),
        "encoder_param_hash_before": enc_hash_before,
        "encoder_param_hash_after": enc_hash_after,
        "encoder_changed": bool(enc_hash_after != enc_hash_before),
        "classifier_param_hash_before": cls_hash_before,
        "classifier_param_hash_after": param_hash(classifier),
        "classifier_unchanged": True,
        "n_params_optimised": int(
            sum((int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params)))
        ),
        "well_sizes": well_sizes,
        "own": own_j.tolist(),
        "history_first": history[0] if history else None,
        "history_last": history[-1] if history else None,
        "resumed_from_step": int(start_step),
        "checkpoint_saves": int(ckpt.n_saves) if ckpt is not None else 0,
        "seconds_note": "train_seconds covers THIS process's steps only; a resumed arm's wall clock is the sum across attempts and the equal-SECONDS control must not be matched against a resumed arm's partial figure",
    }
    return (encoder, store, decoder, {"info": info, "history": history}, key)


def stage_train(
    out: Path,
    seed: int,
    arm: str,
    quick: bool,
    ckpt_every_s: float = 0.0,
    ckpt_remote: Optional[str] = None,
    log: Callable = print,
) -> dict:
    from nclu.config import get_default_config
    from nclu.core.gen_bridge import (
        assert_no_feature_bypass,
        encode_images,
        param_count,
        param_hash,
        resolve_decoder_cfg,
        stream_images,
        train_decoder_from_features,
    )
    from nclu.core.gen_store import assert_no_bypass, store_bytes
    from nclu.eval.gen_probe import factor_labels
    from nclu.utils.checkpoints import load_model, save_model

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    ad = arm_dir(out, seed, arm)
    man_p = ad / "arm.json"
    if man_p.exists():
        m = _read_json(man_p)
        if m.get("status") == "ok" and m.get("schema") == ARM_SCHEMA:
            log(f"  arm {arm} seed {seed}: already on disk ({ARM_SCHEMA}) -- SKIPPED")
            return m
        log(
            f"  arm {arm}: artefact is status={m.get('status')!r} schema={m.get('schema')!r} (want {ARM_SCHEMA!r}) -- RE-RUNNING, loudly"
        )
    ad.mkdir(parents=True, exist_ok=True)
    sd = seed_dir(out, seed)
    warm = _read_json(sd / "warm_start.json")
    z = np.load(sd / "warm_rows.npz")
    own = np.asarray(z["own"], np.int64)
    rows_train = np.asarray(z["train"], np.int64)
    encoder = load_model(sd / "warm_encoder.pkl")
    store = load_model(sd / "warm_store.pkl")
    decoder = load_model(sd / "warm_decoder.pkl")
    if param_hash(encoder) != warm["encoder_param_hash"]:
        raise AssertionError("the warm-start encoder on disk is not the one the manifest describes")
    classifier, cls_info = load_external_classifier(seed, log=log)
    register_short_recipe()
    cfg = get_default_config()
    gb = cfg.gen_bridge
    root = str(data_root())
    log(f"  streaming {rows_train.size} training images ...")
    imgs = stream_images(PINS["dataset"], rows_train, root, int(gb.cache_block))
    sel = select_rows(seed, quick=quick, log=lambda *_: None)
    labels = factor_labels(sel["grid"], rows_train)
    steps = 60 if quick else int(PINS["joint_steps"])
    if ARM_SPEC[arm]["wallclock_match"]:
        tgt_p = arm_dir(out, seed, "loop_swap") / "arm.json"
        if not tgt_p.exists():
            raise FileNotFoundError(
                f"{arm} is the WALL-CLOCK match for loop_swap and loop_swap has not run on seed {seed} yet -- there is nothing to match, and guessing a step count would make the control a decoration"
            )
        tgt = _read_json(tgt_p)
        base_p = arm_dir(out, seed, "alone") / "arm.json"
        if not base_p.exists():
            raise FileNotFoundError(f"{arm} needs `alone`'s seconds/step; run alone first")
        base = _read_json(base_p)
        sps = float(base["compute"]["seconds_per_step"])
        target_s = float(tgt["compute"]["train_seconds"])
        steps = int(np.ceil(target_s / max(sps, 1e-09)))
        log(
            f"  {arm}: matching loop_swap's {target_s:.1f}s at alone's {sps:.4f}s/step -> {steps} steps (vs the equal-steps arms' {PINS['joint_steps']})"
        )
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed) * 10007 + 20260920), 20260920)
    ckpt = None
    if ckpt_every_s and float(ckpt_every_s) > 0:
        C = _load_ckpt_module()
        fp = C.config_fingerprint(train_fingerprint(arm, seed, steps, warm))
        ckpt = C.Checkpointer(
            str(ad / "ckpt"),
            fp,
            remote_uri=f"{ckpt_remote.rstrip('/')}/seed_{int(seed)}/{arm}/ckpt"
            if ckpt_remote
            else None,
            every_s=float(ckpt_every_s),
            keep=3,
        )
        log(
            f"  checkpoint-first: fingerprint {fp} every {ckpt_every_s}s -> {ckpt.remote_uri or ckpt.local_dir}"
        )
    encoder, store, decoder, rep, key = train_arm(
        arm,
        encoder,
        store,
        decoder,
        classifier,
        imgs,
        labels,
        own,
        key,
        steps,
        log=log,
        quick=quick,
        ckpt=ckpt,
    )
    skip_dec = arm in list(PINS.get("arms_without_render_decoder", []))
    feats = encode_images(encoder, imgs, int(PINS["eval_batch"]))
    key, dk = jax.random.split(key)
    t0 = _now()
    rd_cfg = get_default_config().gen_bridge
    if quick:
        from nclu.core.gen_bridge import DECODER_RECIPES

        DECODER_RECIPES[str(PINS["dec_recipe"])]["dec_steps"] = 20
    if skip_dec:
        log(
            f"  {arm}: NO render-decoder refit (declared in PINS). It is the WALL-CLOCK control; its train_seconds and its probe margin need no renderer, so this changes no reported number. It is therefore NOT SCOREABLE on rows 3/4."
        )
        render_decoder, dhist = (None, [])
        recipe = resolve_decoder_cfg(rd_cfg, str(PINS["dec_recipe"]))
    else:
        log(
            f"  refitting the decoder from {arm}'s own features ({PINS['dec_recipe']}, the gated architecture at a declared shorter budget) ..."
        )
        render_decoder, dhist, recipe = train_decoder_from_features(
            encoder.feat_shape, feats, imgs, rd_cfg, dk, log=log, recipe=str(PINS["dec_recipe"])
        )
    dec_seconds = _now() - t0
    Zf = np.asarray(feats, np.float32).reshape((len(imgs), -1))
    bypass = assert_no_feature_bypass(store, Zf, code_grad=str(PINS["code_grad"]))
    shipped = assert_no_bypass(store, Zf)
    log(
        f"  no-bypass OK: same-codes-different-features delta = {bypass['max_abs_delta_same_codes_different_features']}; channel = {bypass['design_bits']} bits (K*log2(M))"
    )
    save_model(encoder, ad / "encoder.pkl", {"arm": arm, "seed": int(seed)})
    save_model(store, ad / "store.pkl", {"arm": arm, "seed": int(seed)})
    save_model(decoder, ad / "joint_decoder.pkl", {"arm": arm, "seed": int(seed)})
    _write_json({"history": rep["history"], "decoder_refit": dhist}, ad / "history.json")
    if render_decoder is not None:
        save_model(render_decoder, ad / "decoder.pkl", {"arm": arm, "seed": int(seed)})
        _publish_declift(out, seed, arm, render_decoder, recipe, param_hash(encoder), log=log)
    man = {
        "schema": ARM_SCHEMA,
        "status": "ok",
        "arm": arm,
        "arm_class": ARM_CLASS[arm],
        "seed": int(seed),
        "quick": bool(quick),
        "void_as_a_reported_number": bool(quick),
        "pins": dict(PINS),
        "warm_start": {
            k: warm[k]
            for k in (
                "encoder_param_hash",
                "store_param_hash",
                "decoder_param_hash",
                "D",
                "feat_shape",
                "well_sizes",
            )
        },
        "compute": rep["info"],
        "scoreable_on_rows_3_4": not skip_dec,
        "decoder_refit": {
            "skipped": bool(skip_dec),
            "why_skipped": "the WALL-CLOCK control: its train_seconds and probe margin need no renderer, and train_seconds excludes the refit either way, so skipping it changes no reported number -- but it is NOT SCOREABLE on rows 3/4"
            if skip_dec
            else None,
            "recipe": recipe.as_dict(),
            "seconds": float(dec_seconds),
            "param_hash": param_hash(render_decoder) if render_decoder is not None else None,
            "param_count": param_count(render_decoder) if render_decoder is not None else None,
            "loss_first": float(dhist[0]) if dhist else None,
            "loss_final": float(dhist[-1]) if dhist else None,
            "note": "identical recipe and identical steps for EVERY arm, fit downstream of that arm's own FROZEN features (the shape); the one difference between arms is the features",
        },
        "checkpointing": {
            "mechanism": "scripts/cloud/ckpt.py (THE checkpointer; no second mechanism)",
            "every_s": float(ckpt_every_s or 0.0),
            "remote": ckpt_remote,
            "fingerprint": C.config_fingerprint(train_fingerprint(arm, seed, steps, warm))
            if ckpt is not None
            else None,
            "resumed_from_step": rep["info"]["resumed_from_step"],
            "saves": rep["info"]["checkpoint_saves"],
        },
        "no_bypass_joint_path": _jsonable(bypass),
        "no_bypass_shipped_read": _jsonable(shipped),
        "channel": {
            "design_bits": float(store.bits_per_read),
            "formula": "K*log2(M)",
            "k_wells": int(store.n_wells),
            "n_modes": int(store.n_modes),
            "declared_constant": "design constant",
            "widened": False,
        },
        "store_bytes": store_bytes(store),
        "param_hashes": {
            "encoder": param_hash(encoder),
            "store": param_hash(store),
            "joint_decoder": param_hash(decoder),
            "render_decoder": param_hash(render_decoder) if render_decoder is not None else None,
            "classifier": cls_info["param_hash"],
        },
        "classifier": cls_info,
        "n_train_rows": int(rows_train.size),
    }
    _write_json(man, man_p)
    log(f"  arm {arm} seed {seed} written -> {man_p}")
    return man


def _publish_declift(
    out: Path, seed: int, arm: str, decoder, recipe, encoder_hash: str, log: Callable = print
) -> Path:
    """Write ``decoder_v1_s<seed>.pkl`` + ``winner_hashes.json`` for this arm.

    LOUD DECLARATION, because this is the one place a reader could be misled.  The frozen
    Judge asserts the decoder against the ``winner_hashes.json`` beside it.  That check is
    "the file is the one this manifest describes", and it still holds.  It is NOT, and is
    never to be read as, "this is the gated decoder-lift winner" -- it is this arm's own
    refit, and ``is_the_frozen_decoder_lift_winner: false`` says so in the file itself.
    A changed encoder changes the feature space; a decoder that did not follow it would
    be rendering one arm's features through another arm's renderer.
    """
    from nclu.core.gen_bridge import param_count, param_hash
    from nclu.utils.checkpoints import save_model

    dd = declift_dir(out, seed, arm)
    dd.mkdir(parents=True, exist_ok=True)
    p = dd / f"decoder_v1_s{int(seed)}.pkl"
    save_model(decoder, p, {"arm": arm, "seed": int(seed)})
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _write_json(
        {
            f"seed_{int(seed)}": {
                "pkl_sha256": sha,
                "param_hash": param_hash(decoder),
                "param_count": param_count(decoder),
                "recipe": recipe.as_dict(),
                "encoder_param_hash_it_was_fit_against": encoder_hash,
            },
            "is_the_frozen_decoder_lift_winner": False,
            "what_this_is": "an earlier study: arm '"
            + arm
            + "'s OWN decoder, refit by the winner recipe verbatim from this arm's own frozen features. An unfrozen encoder's features are NOT the frozen decoder's input space (rule 1), so the decoder follows the encoder and the JUDGE -- which reads images -- is what stays invariant across arms.",
            "the_invariant_instrument": "the external classifier, hash-asserted, identical for every arm",
        },
        dd / "winner_hashes.json",
    )
    log(f"  decoder published for the Judge -> {dd} (declared NOT the frozen winner)")
    return dd


def stage_cache(
    out: Path,
    seed: int,
    arm: str,
    quick: bool,
    max_seconds: Optional[float] = None,
    log: Callable = print,
) -> dict:
    from nclu.core.gen_bridge import dataset_module, param_hash
    from nclu.eval.baselines.cache_build import build_gen2_cache
    from nclu.utils.checkpoints import load_model

    ad = arm_dir(out, seed, arm)
    csd = cache_seed_dir(out, seed, arm)
    done_p = ad / "cache.json"
    if done_p.exists() and _read_json(done_p).get("status") == "ok":
        log(f"  cache for {arm} seed {seed}: already complete -- SKIPPED")
        return _read_json(done_p)
    encoder = load_model(ad / "encoder.pkl")
    arm_man = _read_json(ad / "arm.json")
    if param_hash(encoder) != arm_man["param_hashes"]["encoder"]:
        raise AssertionError("the arm's encoder on disk is not the one its manifest describes")
    mod = dataset_module(PINS["dataset"])
    split, _values = mod.build_split(str(data_root()))
    rows = np.arange(int(split.n_total), dtype=np.int64)
    if quick:
        rows = rows[: 8 * int(PINS["cache_block"])]
    log(f"  generation-2 cache for {arm} seed {seed}: {rows.size} rows -> {csd}")
    man = build_gen2_cache(
        encoder,
        PINS["dataset"],
        rows,
        csd,
        str(data_root()),
        dtype=str(PINS["cache_dtype"]),
        eval_batch=int(PINS["eval_batch"]),
        block=int(PINS["cache_block"]),
        seed=int(seed),
        source_manifest={"loop_arm": arm, "loop_study": "encoder-loop"},
        max_shard_seconds=max_seconds,
        log=log,
    )
    if not man.get("complete", True):
        log("  cache pass hit its wall-clock cap -- progress flushed, re-run to resume")
        return {"status": "incomplete", "manifest": _jsonable(man)}
    _bridgify(out, seed, arm, log=log)
    payload = {
        "schema": "nclu-encoder_loop-cache/1",
        "status": "ok",
        "arm": arm,
        "seed": int(seed),
        "quick": bool(quick),
        "cache_root": str(cache_root(out, seed, arm)),
        "cache_seed_dir": str(csd),
        "manifest": _jsonable(man),
    }
    _write_json(payload, done_p)
    return payload


def _bridgify(out: Path, seed: int, arm: str, log: Callable = print) -> None:
    import shutil
    from nclu.core.gen_bridge import param_hash
    from nclu.utils.checkpoints import load_model

    ad, csd = (arm_dir(out, seed, arm), cache_seed_dir(out, seed, arm))
    for name in ("encoder.pkl", "decoder.pkl"):
        shutil.copyfile(ad / name, csd / name)
    src_probe = bridge_seed_dir(seed) / "probe.pkl"
    if src_probe.exists():
        shutil.copyfile(src_probe, csd / "probe.pkl")
    enc = load_model(ad / "encoder.pkl")
    dec = load_model(ad / "decoder.pkl")
    cman = _read_json(csd / "feature_cache.json")
    _write_json(
        {
            "schema": "nclu-gen-bridge/1",
            "dataset": PINS["dataset"],
            "pretext": PINS["pretext"],
            "seed": int(seed),
            "D": cman["D"],
            "feature_cache": cman,
            "encoder_param_hash": param_hash(enc),
            "decoder_param_hash": param_hash(dec),
            "loop": {
                "study": "encoder-loop",
                "arm": arm,
                "arm_class": ARM_CLASS[arm],
                "is_the_frozen_bridge": False,
                "note": "this bridge directory describes THIS ARM's encoder and its own refit decoder; the probe.pkl beside it is copied byte-for-byte from the frozen bridge and is NOT used for any number here",
            },
        },
        csd / "bridge_manifest.json",
    )
    log(f"  bridgified -> {csd}")


class LinearFactorProbe(eqx.Module):
    linears: list

    def __init__(self, in_dim: int, cardinalities, key):
        keys = jax.random.split(key, len(cardinalities))
        self.linears = [
            eqx.nn.Linear(int(in_dim), int(c), key=k)
            for c, k in zip(cardinalities, keys, strict=True)
        ]

    def __call__(self, z):
        flat = jnp.asarray(z, self.linears[0].weight.dtype).reshape(-1)
        return [lin(flat) for lin in self.linears]


def fit_feature_probe(
    Zf: np.ndarray,
    Y: np.ndarray,
    Zv: np.ndarray,
    Yv: np.ndarray,
    cardinalities,
    key,
    steps: int,
    batch: int,
    lr: float,
    log: Callable = print,
) -> dict:
    """Fit the linear probe; return per-factor and aggregate accuracy on held-back rows."""
    probe = LinearFactorProbe(int(Zf.shape[1]), cardinalities, key)
    opt = optax.adam(lr)
    params, static = eqx.partition(probe, eqx.is_inexact_array)
    opt_state = opt.init(params)
    X = jnp.asarray(np.asarray(Zf, np.float32))
    Yj = jnp.asarray(np.asarray(Y, np.int32))
    n = int(X.shape[0])
    batch = min(int(batch), n)

    def loss_fn(p, s, xb, yb):
        m = eqx.combine(p, s)
        lg = jax.vmap(m)(xb)
        return sum(
            (
                -jnp.mean(
                    jnp.take_along_axis(jax.nn.log_softmax(l, axis=-1), yb[:, f][:, None], axis=1)
                )
                for f, l in enumerate(lg)
            )
        ) / float(len(lg))

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(p, s, o, xb, yb):
        loss, g = grad_fn(p, s, xb, yb)
        upd, o2 = opt.update(g, o, p)
        return (eqx.apply_updates(p, upd), o2, loss)

    k = key
    first = last = None
    for i in range(int(steps)):
        k, sk = jax.random.split(k)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        params, opt_state, loss = step(params, static, opt_state, X[idx], Yj[idx])
        if i == 0:
            first = float(loss)
        last = float(loss)
    probe = eqx.combine(params, static)
    Xv = jnp.asarray(np.asarray(Zv, np.float32))
    lg = eqx.filter_jit(jax.vmap(probe))(Xv)
    pred = np.stack([np.asarray(jnp.argmax(l, axis=-1)) for l in lg], axis=1)
    Yvn = np.asarray(Yv, np.int64)
    per_factor = (pred == Yvn).mean(axis=0)
    return {
        "aggregate": float(per_factor.mean()),
        "per_factor": [float(x) for x in per_factor],
        "exact_match": float((pred == Yvn).all(axis=1).mean()),
        "chance_per_factor": [float(1.0 / int(c)) for c in cardinalities],
        "loss_first": first,
        "loss_final": last,
        "n_fit": int(n),
        "n_eval": int(Xv.shape[0]),
        "steps": int(steps),
        "batch": int(batch),
        "lr": float(lr),
    }


def stage_probe(out: Path, seed: int, arm: str, quick: bool, log: Callable = print) -> dict:
    """The arm's factor probe AND the random-init floor, fit identically.

    The floor is an UNTRAINED encoder of the identical architecture and the identical
    initialisation stream, carried through the identical probe.  The reported quantity is
    the MARGIN.  An absolute probe number on Shapes3D says how easy Shapes3D is.
    """
    from nclu.core.gen_bridge import GenEncoder, encode_images, param_hash, stream_images
    from nclu.config import get_default_config
    from nclu.eval.gen_probe import factor_labels
    from nclu.utils.checkpoints import load_model

    ad = arm_dir(out, seed, arm)
    p_out = ad / "probe.json"
    if p_out.exists() and _read_json(p_out).get("status") == "ok":
        log(f"  probe for {arm} seed {seed}: already on disk -- SKIPPED")
        return _read_json(p_out)
    register_short_recipe()
    sel = select_rows(seed, quick=quick, log=lambda *_: None)
    grid = sel["grid"]
    rows_fit = sel["rows"]["probe_fit"]
    rows_ev = sel["rows"]["probe_eval"]
    root = str(data_root())
    gb = get_default_config().gen_bridge
    imgs_fit = stream_images(PINS["dataset"], rows_fit, root, int(gb.cache_block))
    imgs_ev = stream_images(PINS["dataset"], rows_ev, root, int(gb.cache_block))
    Yf = factor_labels(grid, rows_fit)
    Yv = factor_labels(grid, rows_ev)
    cards = [int(c) for c in grid.cardinalities]
    steps = 100 if quick else int(PINS["probe_steps"])
    encoder = load_model(ad / "encoder.pkl")
    key = jax.random.PRNGKey(int(seed) * 911 + 20260920)

    def probe_for(enc, tag):
        Zf = np.asarray(encode_images(enc, imgs_fit, int(PINS["eval_batch"]))).reshape(
            (len(imgs_fit), -1)
        )
        Zv = np.asarray(encode_images(enc, imgs_ev, int(PINS["eval_batch"]))).reshape(
            (len(imgs_ev), -1)
        )
        r = fit_feature_probe(
            Zf,
            Yf,
            Zv,
            Yv,
            cards,
            jax.random.fold_in(key, hash(tag) % 10000),
            steps,
            int(PINS["probe_batch"]),
            float(PINS["probe_lr"]),
            log=log,
        )
        log(f"    probe[{tag}] aggregate={r['aggregate']:.4f} exact={r['exact_match']:.4f}")
        return r

    arm_probe = probe_for(encoder, arm)
    floor_p = seed_dir(out, seed) / "probe_random_init.json"
    if floor_p.exists() and _read_json(floor_p).get("status") == "ok":
        floor = _read_json(floor_p)["probe"]
        log("    random-init floor: reused from this seed (identical by construction)")
    else:
        rk = jax.random.PRNGKey(int(seed) * 7717 + 20260920)
        rand_enc = GenEncoder(
            tuple(imgs_fit.shape[1:]),
            tuple(gb.enc_channels),
            int(gb.enc_kernel),
            int(gb.enc_stride),
            int(gb.enc_groups),
            rk,
        )
        floor = probe_for(rand_enc, "random_init")
        _write_json(
            {
                "status": "ok",
                "seed": int(seed),
                "probe": floor,
                "encoder_param_hash": param_hash(rand_enc),
                "note": "an UNTRAINED encoder of the identical architecture, carried through the identical probe at the identical budget; this is the floor every probe number is reported against",
            },
            floor_p,
        )
    margin = {
        "aggregate": float(arm_probe["aggregate"] - floor["aggregate"]),
        "exact_match": float(arm_probe["exact_match"] - floor["exact_match"]),
        "per_factor": [
            float(a - b) for a, b in zip(arm_probe["per_factor"], floor["per_factor"], strict=True)
        ],
    }
    payload = {
        "schema": "nclu-encoder_loop-probe/1",
        "status": "ok",
        "arm": arm,
        "arm_class": ARM_CLASS[arm],
        "seed": int(seed),
        "quick": bool(quick),
        "void_as_a_reported_number": bool(quick),
        "factor_names": list(grid.names),
        "arm_probe": arm_probe,
        "random_init_floor": floor,
        "MARGIN_OVER_RANDOM_INIT": margin,
        "how_to_read_this": "REPORT THE MARGIN. The absolute number says how easy Shapes3D is under a linear readout; only the margin over the random-init floor says anything about the arm. A probe that improves in absolute terms but not over random init is NO CLAIM.",
    }
    _write_json(payload, p_out)
    log(
        f"  probe margin over random init: aggregate {margin['aggregate']:+.4f} exact {margin['exact_match']:+.4f}"
    )
    return payload


def stage_rivals(
    out: Path,
    seed: int,
    arm: str,
    quick: bool,
    families: Optional[list] = None,
    log: Callable = print,
) -> dict:
    import jax as _jax
    from nclu.data.gen_rowsplit import split_fit_val
    from nclu.eval.baselines import api
    from nclu.eval.baselines import learners as L
    from nclu.experiments.baseline_suite import _mini_ctx

    ad = arm_dir(out, seed, arm)
    r_out = ad / "rivals.json"
    fams = list(families or PINS["rival_families"])
    if r_out.exists():
        prev = _read_json(r_out)
        if prev.get("status") == "ok" and set(fams) <= set(prev.get("families_done", [])):
            log(f"  rivals for {arm} seed {seed}: {fams} already fitted -- SKIPPED")
            return prev
    from nclu.experiments import factor_wells as fw

    os.environ["NCLU_FEATURES_ROOT"] = str(cache_root(out, seed, arm))
    fw.PINS["seed"] = int(seed)
    cond = fw.load_condition()
    grid = cond["grid"]
    n_pool = 4096 if quick else int(fw.PINS["n_pool"])
    pool = fw.fit_pool(cond["train_rows"], n_pool, int(fw.PINS["pool_rng_seed"]))
    F = fw.features_mmap(int(seed))
    D = int(np.prod(F.shape[1:]))
    n_val = 128 if quick else int(PINS["rival_val_rows"])
    fit_rows, val_rows, _ = split_fit_val(
        pool,
        grid.factor_indices_of(pool),
        int(pool.size - n_val),
        n_val,
        int(seed),
        mode="stratified",
        log=lambda *_: None,
    )
    log(
        f"  rival pool: {pool.size} rows from THIS ARM's cache (fit {fit_rows.size} / val {val_rows.size}), D={D}"
    )
    Xf = fw.gather(F, fit_rows)
    Xv = fw.gather(F, val_rows)
    ctx = _mini_ctx(grid, fit_rows, val_rows, Xf, Xv, D, seed)
    cman = _read_json(cache_seed_dir(out, seed, arm) / "feature_cache.json")
    root = str(ad / "rivals")
    key = _jax.random.PRNGKey(int(seed) * 5407 + 20260920)
    bs = L.builders(ctx["cardinalities"], D, key)
    done, records = ([], {})
    if r_out.exists():
        prev = _read_json(r_out)
        done = list(prev.get("families_done", []))
        records = dict(prev.get("families", {}))
    for fam in fams:
        if fam in done:
            log(f"    [{fam}] already fitted -- SKIPPED")
            continue
        build, kind = bs[fam]
        target = int(L.FAMILY_BUDGETS.get(fam, 118784))
        w, n_par, err = L.match_width(build, target, log=lambda *_: None)
        log(f"    [{fam}] matched width {w} -> {n_par} params vs target {target} ({err:+.2%})")
        key, fk = _jax.random.split(key)
        t0 = _now()
        r = L.fit_family(fam, build, kind, w, ctx, fk, dropout=False, log=log)
        ck = api.save_learner(
            int(seed), fam, r["_model"], w, r["_in_kind"], grid, D, cache_manifest=cman, root=root
        )
        records[fam] = {
            "param_count": int(n_par),
            "target_params": target,
            "rel_err": float(err),
            "width": int(w),
            "seconds": float(_now() - t0),
            "checkpoint": str(ck),
            "tuning": _jsonable(r.get("tuning")),
            "val_mse_full_schedule": _jsonable(r.get("val_mse_full_schedule")),
            "variant": "full-only (dropout=False) -- the rows 3/4 variant",
        }
        done.append(fam)
        _write_json(
            {
                "schema": "nclu-encoder_loop-rivals/1",
                "status": "ok",
                "arm": arm,
                "seed": int(seed),
                "quick": bool(quick),
                "rivals_root": root,
                "families_done": done,
                "families": records,
                "fit_pool_rows": int(pool.size),
                "D": D,
                "features_sha256": cman.get("features_sha256"),
                "note": "refit on THIS ARM's features with learners' protocol verbatim; the earlier baseline set's checkpoints are untouched",
            },
            r_out,
        )
    return _read_json(r_out)


def stage_report(out: Path, log: Callable = print) -> dict:
    """Every arm, every seed, side by side.  The treatment is never printed alone."""
    out = Path(out)
    table = {}
    for sdir in sorted(out.glob("seed_*")):
        s = int(sdir.name.split("_")[1])
        for adir in sorted(sdir.iterdir()):
            if not adir.is_dir() or adir.name not in ARMS:
                continue
            a = adir.name
            row = {"seed": s, "arm": a, "arm_class": ARM_CLASS[a]}
            if (adir / "arm.json").exists():
                m = _read_json(adir / "arm.json")
                c = m["compute"]
                row.update(
                    {
                        "gradient_steps": c["gradient_steps"],
                        "images_seen": c["images_seen"],
                        "train_seconds": c["train_seconds"],
                        "seconds_per_step": c["seconds_per_step"],
                        "n_params_optimised": c["n_params_optimised"],
                        "encoder_changed": c["encoder_changed"],
                        "swap_exact_final": (c.get("history_last") or {}).get("swap_exact"),
                        "recon_final": (c.get("history_last") or {}).get("recon"),
                        "decoder_refit_loss_final": m["decoder_refit"]["loss_final"],
                        "scoreable_on_rows_3_4": m.get("scoreable_on_rows_3_4", True),
                        "store_bytes": m["store_bytes"],
                    }
                )
            if (adir / "probe.json").exists():
                p = _read_json(adir / "probe.json")
                row["probe_margin_aggregate"] = p["MARGIN_OVER_RANDOM_INIT"]["aggregate"]
                row["probe_margin_exact"] = p["MARGIN_OVER_RANDOM_INIT"]["exact_match"]
                row["probe_absolute_DO_NOT_QUOTE"] = p["arm_probe"]["aggregate"]
            if (adir / "cache.json").exists():
                row["cache"] = _read_json(adir / "cache.json").get("status")
            table[f"seed{s}:{a}"] = row
    payload = {
        "schema": "nclu-encoder_loop-report/1",
        "pins": dict(PINS),
        "arms": {a: ARM_CLASS[a] for a in ARMS},
        "rows": table,
        "how_to_read_this": "EQUAL COMPUTE is `gradient_steps` and `images_seen` equal across the equal-steps arms; `train_seconds` is printed beside them because the swap term costs more per step, and `alone_wc` is the arm matched on seconds instead. The probe is a MARGIN over the random-init floor and the absolute is carried only so a reader can see it is not the claim.",
    }
    _write_json(payload, out / "encoder_loop_report.json")
    log(json.dumps(payload["rows"], indent=2, sort_keys=True, default=float))
    return payload


def stage_cloud(
    out: Path,
    seed: int,
    quick: bool,
    ckpt_every_s: float,
    ckpt_remote: Optional[str],
    arms: Optional[list] = None,
    log: Callable = print,
) -> dict:
    arms = list(arms or ARMS)
    stage_warm(out, seed, quick, log=log)
    done, failed = ([], [])
    for arm in arms:
        log(f"==== unit seed {seed} / arm {arm} ====")
        try:
            stage_train(
                out, seed, arm, quick, ckpt_every_s=ckpt_every_s, ckpt_remote=ckpt_remote, log=log
            )
            done.append(arm)
        except Exception as exc:
            log(f"!!!! ARM FAILED: {arm}: {type(exc).__name__}: {exc}")
            _write_json(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": "the stage raised; this is a FAILURE, not a declared NOT-RUN, and it is never a null",
                },
                arm_dir(out, seed, arm) / "FAILED.json",
            )
            failed.append(arm)
    payload = {
        "schema": "nclu-encoder_loop-cloud/1",
        "seed": int(seed),
        "arms_done": done,
        "arms_failed": failed,
        "stages_run_here": list(CLOUD_STAGES),
        "stages_that_must_run_on_cpu": ["cache", "probe", "rivals", "score"],
        "why": "cache2.device_guard refuses to write a feature cache unless JAX sees exactly one CPU device; every stage downstream of the cache reads it",
    }
    _write_json(payload, seed_dir(out, seed) / "cloud_unit.json")
    if failed:
        raise RuntimeError(f"arms failed on seed {seed}: {failed}")
    return payload


def apply_compute_overrides(
    joint_steps: Optional[int],
    dec_steps: Optional[int],
    warm_steps: Optional[int],
    n_train: Optional[int],
    log: Callable = print,
) -> dict:
    changed = {}
    for name, val in (
        ("joint_steps", joint_steps),
        ("dec_refit_steps", dec_steps),
        ("warm_steps", warm_steps),
        ("n_train", n_train),
    ):
        if val is not None:
            changed[name] = {"from": PINS[name], "to": int(val)}
            PINS[name] = int(val)
    if changed:
        log(f"  COMPUTE OVERRIDES (equal for every arm in this launch): {changed}")
    return changed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="an earlier study: does the store force better features?"
    )
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm", default=None, choices=[None, *ARMS])
    p.add_argument("--out", required=True)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--ckpt-every-s", type=float, default=0.0)
    p.add_argument("--ckpt-remote", default=os.environ.get("NCLU_LOOP_CKPT_REMOTE"))
    p.add_argument("--joint-steps", type=int, default=None)
    p.add_argument("--dec-steps", type=int, default=None)
    p.add_argument("--warm-steps", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    t0 = _now()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"stage={args.stage} seed={args.seed} arm={args.arm} out={out} quick={args.quick}")
    log(f"nclu package: {Path(__import__('nclu').__file__).parent}")
    log(f"jax devices: {jax.devices()}")
    apply_compute_overrides(
        args.joint_steps, args.dec_steps, args.warm_steps, args.n_train, log=log
    )
    if args.quick:
        log("QUICK: plumbing smoke only -- every number this writes is VOID")
    if args.stage == "warm":
        stage_warm(out, args.seed, args.quick, log=log)
    elif args.stage == "train":
        if not args.arm:
            raise SystemExit("--stage train needs --arm")
        stage_train(
            out,
            args.seed,
            args.arm,
            args.quick,
            ckpt_every_s=args.ckpt_every_s,
            ckpt_remote=args.ckpt_remote,
            log=log,
        )
    elif args.stage == "cache":
        if not args.arm:
            raise SystemExit("--stage cache needs --arm")
        stage_cache(out, args.seed, args.arm, args.quick, max_seconds=args.max_seconds, log=log)
    elif args.stage == "probe":
        if not args.arm:
            raise SystemExit("--stage probe needs --arm")
        stage_probe(out, args.seed, args.arm, args.quick, log=log)
    elif args.stage == "rivals":
        if not args.arm:
            raise SystemExit("--stage rivals needs --arm")
        stage_rivals(out, args.seed, args.arm, args.quick, families=args.families, log=log)
    elif args.stage == "cloud":
        stage_cloud(
            out, args.seed, args.quick, args.ckpt_every_s, args.ckpt_remote, arms=args.arms, log=log
        )
    elif args.stage == "report":
        stage_report(out, log=log)
    log(f"done in {_now() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
