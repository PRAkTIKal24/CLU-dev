"""⭐⭐ hub/composition — THE MECHANISM DRIVER.  ⛔ SELECTION ONLY. ⛔ B NEVER ENTERS.

Charter §A.1: *"Iterate freely on MECHANISM."*  §A.2: *"Selection on `val` only."*
⛔ This file is NOT the scoring harness and nothing it prints is a scored number.  The
scored table is `exp_gen_cell` and only `exp_gen_cell`; an arm scored on a cell of its
own is not on that table, which is the whole point of that module.

⛔⛔ THE HARD CONSTRAINT (LAUNCH-COMMON §4), enforced mechanically, not promised:
`assert_cache_covers()` below refuses to run unless EVERY row this driver will touch is
already in the frozen Mac-written feature cache.  ⛔ Nothing is encoded here. Ever.
⭐ That is the same evidence `_assert_cache_reencodes`' docstring names for
`will_extend=False` (`cov["n_shortfall"] == 0`), computed here for this driver's own rows.

⭐ Resumable, per LAUNCH-COMMON §5: one JSON per UNIT, skipped if present.  A kill costs
one unit, never the run.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import jax

sys.path.insert(0, str(Path(os.environ["OUT"]) / "instrument"))
from valid_novel import valid_novel_report                      # noqa: E402

from chlu.config import get_default_config, gen_row_split_config, gen_dynamical_read_config
from chlu.core.gen_bridge import (dataset_module, features_for_rows,
                                  load_feature_cache, param_hash)
from chlu.core.gen_landscape import (
    SettleCfg, arm_key, launch_state, read_settled, settle_chain, convergence_report,
    pairwise_separation, settled_state_spread,
)
from chlu.core.gen_dream import fit_address_book, requests_from_labels
from chlu.core.gen_store import address_codes, load_store_plan, well_channels_from_labels
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior
from chlu.data.gen_rowsplit import split_fit_val
from chlu.eval.gen_probe import decode_features, factor_labels, predict_factors
from chlu.experiments.exp_gen_cell import (
    _cell_root, load_external_classifier, store_plan_path, resolve_store, render,
)
from chlu.experiments.exp_gen_compose import resolve_decoder
from chlu.experiments.exp_gen_dream import resolve_grouping
from chlu.experiments.exp_gen_dynamical_read import build_landscape
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows, matched_seen_rows
from chlu.utils.checkpoints import load_model

DS, PX = "shapes3d", "recon"
OUT = Path(os.environ["OUT"])


def log(*a):
    print(*a, flush=True)


def assert_cache_covers(crows, needed: dict) -> dict:
    """⛔ THE GATE THIS DRIVER ENFORCES ON ITSELF.  Every row must already be cached."""
    have = set(np.asarray(crows, np.int64).tolist())
    short = {k: int(sum(1 for r in np.asarray(v, np.int64).ravel().tolist() if r not in have))
             for k, v in needed.items()}
    total = int(sum(short.values()))
    if total:
        raise AssertionError(
            f"⛔⛔ REFUSING TO RUN: {total} of the rows this driver needs are NOT in the "
            f"frozen feature cache {short}. Encoding them would REBUILD an input on a "
            "machine whose encoder does not reproduce the Mac cache (51 ULP measured). "
            "LAUNCH-COMMON §4. ⛔ Report it; do not proceed.")
    return {"n_shortfall": 0, "by_set": short,
            "evidence": "every row this driver touches is already in the frozen cache; "
                        "nothing is encoded, so no inconsistent feature can be written"}


def build_context(seed: int, quick: bool = False) -> dict:
    cfg = get_default_config()
    g = cfg.gen_cell
    mod = dataset_module(DS)
    split, values = mod.build_split(cfg.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    bdir = bridge_dir(cfg, DS, PX, seed)
    decoder, dec_info, bman = resolve_decoder(cfg, bdir, seed, log=log)
    external, clf_info = load_external_classifier(cfg, seed, log=log)
    encoder = load_model(bdir / "encoder.pkl")
    if param_hash(encoder) != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats_cache, cman = load_feature_cache(bdir, verify=False)
    if cman["encoder_param_hash"] != bman["encoder_param_hash"]:
        raise AssertionError("the feature cache was written by a different encoder")

    rows = select_store_rows(split, cfg, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    rs = gen_row_split_config(cfg)
    fit_rows, val_rows, split_info = split_fit_val(
        pool, grid.factor_indices_of(pool), int(g.n_fit), int(g.n_val), int(seed),
        mode=str(rs.mode), tiebreak_jitter=float(rs.tiebreak_jitter),
        seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset), log=lambda *a: None)
    ms_rows = np.asarray(matched_seen_rows(DS, split, values, rows["store_train"]), np.int64)
    # ⛔ B is deliberately NOT in `needed` and deliberately NOT gathered here.
    needed = {"store_train": pool, "val": val_rows, "MS": ms_rows}
    cov = assert_cache_covers(crows, needed)
    log(f"  ⭐ cache coverage: shortfall {cov['n_shortfall']} — nothing will be encoded")

    # ⭐ THE HARNESS'S OWN GATHER, minus the extension branch it can never take here.
    # `gather_features` does exactly `features_for_rows(...).reshape((-1, D))` for
    # cached rows; calling `features_for_rows` directly makes this driver's features
    # BIT-IDENTICAL to the harness's, and makes it structurally impossible to encode.
    D_cache = int(cman["D"])

    def take(r):
        return features_for_rows(crows, feats_cache,
                                 np.asarray(r, np.int64).ravel()).reshape((-1, D_cache))

    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    X_store = take(pool)
    X_val, X_ms = take(val_rows), take(ms_rows)
    lab_store = factor_labels(grid, pool)
    lab_val, lab_ms = factor_labels(grid, val_rows), factor_labels(grid, ms_rows)

    plan = load_store_plan(store_plan_path(cfg, DS, PX))
    cfg.gen_dream.grouping_source = str(g.grouping_source)
    labels_vec, ginfo = resolve_grouping(plan, X_store, cfg, seed,
                                         bman["encoder_param_hash"], log=log)
    wc = well_channels_from_labels(labels_vec, int(plan["K"]))
    store, store_prov = resolve_store(cfg, DS, PX, seed, X_store, wc, plan, feat_shape, log=log)
    codes_train = address_codes(store, X_store.reshape((X_store.shape[0], -1)))
    book = fit_address_book(codes_train, lab_store, cards, float(g.address_smoothing))
    return dict(cfg=cfg, split=split, grid=grid, names=names, cards=cards,
                decoder=decoder, dec_info=dec_info, external=external, clf_info=clf_info,
                feat_shape=feat_shape, seed=seed, eval_batch=int(g.eval_batch),
                X_store=X_store, lab_store=lab_store, wc=wc, store=store, book=book,
                sets={"val": (val_rows, X_val, lab_val), "MS": (ms_rows, X_ms, lab_ms)},
                coverage=cov, store_prov=store_prov, grouping=ginfo, plan=plan)


def score(ctx, feats, target_lab, atom_tuples) -> dict:
    imgs = render(ctx["decoder"], feats, ctx["feat_shape"], ctx["eval_batch"])
    preds = predict_factors(ctx["external"], imgs, ctx["eval_batch"])
    rep = valid_novel_report(preds, target_lab, atom_tuples, ctx["names"])
    return rep


# ---------------------------------------------------------------------------
# The arms.  ⛔ Every one calls the PINNED functions; nothing is re-implemented.
# ---------------------------------------------------------------------------

def requests_for(ctx, which: str) -> np.ndarray:
    rows, X, lab = ctx["sets"][which]
    return requests_from_labels(lab, "full", 0, int(ctx["seed"]))


def arm_clu(ctx, which: str) -> np.ndarray:
    """The codebook read — `exp_gen_cell`'s `clu` row, same calls, same order."""
    R = requests_for(ctx, which)
    post = request_posterior(ctx["book"], R)
    code = compose_codes(post)["code"]
    return np.asarray(read_compose(ctx["store"], code), np.float32)


def arm_x0(ctx, land, book, which: str, launch_mode: str = "clamp_atom",
           clamp_threshold: float = 0.5) -> dict:
    """⭐ THE ADDRESSED READ — per-well posterior-argmax atom, concatenated. 0 steps."""
    R = requests_for(ctx, which)
    lau = launch_state(land, book, R, launch_mode, float(clamp_threshold))
    return lau


def arm_settle(ctx, land, book, which: str, temp: float, steps: int,
               request_strength: float, seed: int, cfgd) -> dict:
    """⭐ The dynamical read at a DECLARED temperature.  `temp` is `temp_end`; the
    anneal start is `temp` too (a flat schedule), so the declared number IS the
    temperature the chain ran at — ⛔ not the end of a schedule that spent most of its
    steps somewhere else."""
    R = requests_for(ctx, which)
    lnd = land.with_strengths(request_strength=float(request_strength))
    lau = launch_state(lnd, book, R, str(cfgd.launch_mode), float(cfgd.clamp_threshold))
    cfg = SettleCfg(steps=int(steps), step_size=float(cfgd.step_size),
                    sigma_scale_start=float(cfgd.sigma_scale_start),
                    sigma_scale_end=float(cfgd.sigma_scale_end),
                    temp_start=float(temp), temp_end=float(temp),
                    metropolis=bool(cfgd.metropolis))
    key = arm_key(jax.random.PRNGKey(int(seed)),
                  f"settle_T{temp:g}_b{request_strength:g}_{which}")
    res = settle_chain(lnd, lau["x0"], lau.get("log_w"), cfg, key)
    x = res["x"]
    feats = read_settled(lnd, x, str(cfgd.read_mode))
    return {"x0": lau["x0"], "x": x, "feats": feats, "trace": res["trace"],
            "convergence": convergence_report(res["trace"], int(cfgd.descent_window),
                                              float(cfgd.descent_tol)),
            "separation": pairwise_separation(x, 512, int(seed)),
            "separation_launch": pairwise_separation(lau["x0"], 512, int(seed)),
            "spread": settled_state_spread(x, lnd.store),
            "settle_cfg": cfg.as_dict()}
