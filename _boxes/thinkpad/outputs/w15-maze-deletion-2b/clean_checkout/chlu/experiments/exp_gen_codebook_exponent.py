"""Experiment: ⭐⭐ **is the codebook exponent set by the FACTOR MANIFOLD?** (§11b)

⭐ **DIAL DECLARATION — *the dataset*, as a test of WHY the exponent is what it is.**
- **Dial:** the dataset.  ⛔ No arm, no CLU claim, no verdict on any dial.
- **Laundering control:** ⭐ **the identical sweep and the identical fit on Shapes3D,
  re-run here.**  ⛔ ``−0.2631`` is never transcribed into the comparison — three
  errata on this programme came from transcribed table values.
- **Falsifies:** the factor-manifold account.  ⛔ **If dSprites is not steeper, the
  account is REFUTED** and ``−0.26``'s universality needs a different explanation.
  ⭐ That is a result, not a failure.

**The measurement.**  Within each ``(dataset, seed, well)``, OLS of
``log ε_measured`` on ``log M`` over ONE shared ``M`` grid, with ``ε`` the shipped
:func:`chlu.core.gen_alloc.per_well_distortion` on the **SPREAD** rows — the same
function, the same row role and the same fit form as `gen-codebook-allocation` §5d,
which measured ``−0.2631 ± 0.0555`` on Shapes3D from a 3-point, uneven sweep.

⛔ **Exponents are only comparable over comparable ranges.**  One grid,
``M ∈ {4, 8, 16, 32, 64, 128, 186}``, endpoints pinned to Shapes3D's historical
``M ∈ [4, 186]``, used for **both** datasets and asserted.

⛔ **The partition is re-derived, per seed, by the same procedure on both datasets.**
dSprites has no `gen-commitment-collapse` cache; loading one partition and deriving
the other would be two code paths and the comparison would be void.  Where a shipped
``labels_vec`` exists it is read **only** to prove the derivation reproduces it
bit-identically.

⛔ **No decoder, no probe, no rendered image** — so `GenBridgeConfig.dec_recipe`'s
default of ``"baseline"`` (the recipe that FAILED the gate) cannot reach a number
here.  ⛔ And no aggregate ceiling is quoted for dSprites: its probe is NOT
saturated (`gen-bridge-dsprites` §5) and that fact does not carry from Shapes3D.

⚠ **Three environment facts** (a worktree has no ``.claude/``): ``$CHLU_GEN_DATA_ROOT``,
``$CHLU_CLAUDE_ROOT`` **and** the per-dataset bridge roots must resolve, or the run
dies at ``bridge_dir``.

Runnable::

    python -m chlu.experiments.exp_gen_codebook_exponent --quick   # ⚠ plumbing smoke
    python -m chlu.experiments.exp_gen_codebook_exponent
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, gen_codebook_exponent_config, get_default_config  # noqa: E402
from ..core.gen_alloc import per_well_distortion, well_spectra  # noqa: E402
from ..core.gen_bridge import PRETEXTS, load_feature_cache, param_hash  # noqa: E402
from ..core.gen_store import (  # noqa: E402
    address_codes,
    block_diagonality,
    channel_correlation,
    dead_channels,
    grouping_labels,
    init_store,
    store_bytes,
    train_store,
    well_channels_from_labels,
)
from ..eval.gen_attribution import n_values_realised  # noqa: E402
from ..eval.gen_probe import factor_labels  # noqa: E402
from ..experiments.exp_gen_compose import _claude_root, _write_json  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

#: ⭐ Student-t 97.5 % quantile at 2 d.o.f. — the 3-seed CI the PREREG declares.
T_CRIT_2DF = 4.302652729911275


def _cfg(config: CHLUConfig):
    return gen_codebook_exponent_config(config)


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only.  ⛔ A ``--quick`` number is NEVER reported."""
    c = _cfg(config)
    c.seeds = list(c.quick_seeds)
    c.m_grid = list(c.quick_m_grid)
    c.n_store_train = int(c.quick_n_store_train)
    c.n_spread_train = int(c.quick_n_spread_train)
    c.n_spread_test = int(c.quick_n_spread_test)
    c.n_corr = int(c.quick_n_corr)
    c.spectrum_rows = int(c.quick_spectrum_rows)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    # ⛔ The smoke shrinks ``n_corr`` and re-draws ``store_train``, so the partition
    # it derives is NOT the shipped one and CANNOT be — the check is downgraded to a
    # warning here and ⛔ still RAISES in a production run (`gen-bridge-dsprites`
    # §3.1's precedent for the decoder-recipe guard).
    c.assert_partition_matches_shipped = False
    config.gen_store.train_steps = int(c.quick_store_train_steps)
    return config


def exponent_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-dsprites-exponent"


def _abs(config: CHLUConfig, p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


# ---------------------------------------------------------------------------
# 0. The grid — ⛔ one grid, both datasets
# ---------------------------------------------------------------------------


def assert_shared_grid(m_grid, reference_range=(4, 186)) -> dict:
    """⛔ The falsifier the task names: *"your dSprites sweep covers a different M
    range than Shapes3D's"*.  One grid object is used for both datasets, and its
    endpoints are asserted against Shapes3D's historical ``M ∈ [4, 186]``.
    """
    M = [int(x) for x in m_grid]
    if len(set(M)) != len(M) or len(M) < 3:
        raise ValueError(f"⛔ the M grid must hold >=3 distinct values, got {M}")
    if min(M) != int(reference_range[0]) or max(M) != int(reference_range[1]):
        raise AssertionError(
            f"⛔ the M grid spans [{min(M)}, {max(M)}] but Shapes3D's measured range "
            f"is {list(reference_range)} — exponents fitted over different ranges are "
            "NOT comparable, which is this spoke's named falsifier")
    return {"m_grid": sorted(M), "n_points": len(M),
            "m_range": [min(M), max(M)],
            "reference_range": [int(reference_range[0]), int(reference_range[1])],
            "shared_by_both_datasets": True}


# ---------------------------------------------------------------------------
# 1. Rows — ⛔ `gen-codebook-allocation`'s draw, verbatim
# ---------------------------------------------------------------------------


def select_rows(config: CHLUConfig, split, seed: int) -> dict:
    """``store_train`` + ``SPREAD``, with `gen-codebook-allocation`'s own RNG stream.

    ⛔ ``SPREAD_train`` is drawn and discarded rather than skipped: ``SPREAD`` is
    ``take[n_spread_train:]`` there, so not drawing it would silently hand this
    harness a *different* held-out row set and the Shapes3D control would no longer
    be measured on the rows the published ``ε`` was measured on.
    """
    c = _cfg(config)
    config.gen_store.n_store_train = int(c.n_store_train)
    rows = select_store_rows(split, config, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), rows["store_train"])
    rng = np.random.default_rng(int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset))
    n = min(int(c.n_spread_train) + int(c.n_spread_test), pool.size)
    take = rng.permutation(pool)[:n]
    n_tr = min(int(c.n_spread_train), n)
    return {
        "store_train": rows["store_train"],
        "SPREAD": np.sort(take[n_tr:]),
        "n_spread_train_drawn_and_discarded": int(n_tr),
        "eval_a": rows["eval_a"],
    }


# ---------------------------------------------------------------------------
# 2. The partition — ⛔ the SAME procedure on both datasets
# ---------------------------------------------------------------------------


def derive_partition(config: CHLUConfig, X_store: np.ndarray, seed: int, log=print) -> tuple:
    """Per-seed correlation grouping at ``K``, by the shipped procedure.

    ⛔ This is `exp_gen_dream.resolve_grouping`'s ``per_seed`` branch, verbatim in
    content: a seed's encoder channels have no relation to another seed's, and
    **dSprites has no cached ``labels_vec`` at all** — so the partition is derived,
    identically, on both datasets.
    """
    c = _cfg(config)
    gs = config.gen_store
    Xc = np.asarray(X_store, np.float32)[: int(c.n_corr)]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    labels, info = grouping_labels(
        corr, int(c.k_wells), "correlation", int(seed), gs.grouping_method,
        dead=dead, min_well_frac=float(gs.min_well_frac))
    wc = well_channels_from_labels(labels, int(c.k_wells))
    bd = block_diagonality(corr, labels)
    out = {
        "source": "per_seed (re-derived by the shipped procedure — ⛔ identical on both datasets)",
        "n_corr_rows": int(Xc.shape[0]),
        "n_dead_channels": int(dead.sum()),
        "grouping_info": info,
        "well_sizes": [len(w) for w in wc],
        "in_block_fraction": float(bd["in_block_fraction"]),
        "lift_over_size_matched_random": float(bd["lift_over_size_matched_random"]),
    }
    log(f"  ⭐ partition: well sizes {out['well_sizes']} · in-block |corr|^2 = "
        f"{out['in_block_fraction']:.4f} · {out['n_dead_channels']} dead channels")
    return labels, wc, out


def check_partition_against_shipped(config: CHLUConfig, dataset: str, pretext: str,
                                    seed: int, labels: np.ndarray, log=print) -> dict:
    """⭐ Where a shipped ``labels_vec`` exists, the derivation must reproduce it.

    ⛔ Read-only, and **absent on dSprites by construction** — which is the whole
    reason the partition is derived rather than loaded.  A mismatch on Shapes3D
    means this is not the procedure that wrote the shipped wells and the control row
    would not be the published one.
    """
    c = _cfg(config)
    path = _abs(config, c.commit_cache_pattern.format(
        dataset=dataset, pretext=pretext, seed=int(seed))) / f"commit_cache_s{int(seed)}.npz"
    if not path.exists():
        return {"status": "⛔ NO SHIPPED PARTITION (expected on dSprites)",
                "path": str(path), "bit_identical": None}
    with np.load(path, allow_pickle=False) as z:
        shipped = np.asarray(z["labels_vec"], np.int64)
    same = bool(np.array_equal(np.asarray(labels, np.int64), shipped))
    log(f"  ⭐ shipped-partition check: bit_identical={same} ({path.name})")
    if not same and not bool(c.assert_partition_matches_shipped):
        log("  ⚠ the derived partition differs from the shipped one and the assertion "
            "is DISABLED (⚠ expected under --quick, where n_corr and store_train are "
            "both reduced; ⛔ never disabled in a production run)")
    if not same and bool(c.assert_partition_matches_shipped):
        raise AssertionError(
            f"⛔ the per-seed partition does NOT reproduce {path}'s labels_vec — this "
            "is not the procedure that wrote the shipped wells and the Shapes3D "
            "control row would not be the published one")
    return {"status": "shipped partition present", "path": str(path),
            "bit_identical": same, "n_channels": int(shipped.size)}


# ---------------------------------------------------------------------------
# 3. One (dataset, seed): the M sweep
# ---------------------------------------------------------------------------


def run_seed(config: CHLUConfig, dataset: str, seed: int, out_dir: Path,
             log: Callable = print) -> dict:
    c = _cfg(config)
    t0 = time.time()
    pretext = str(c.pretext)
    grid_gate = assert_shared_grid(c.m_grid) if bool(c.assert_grid_is_shared) else {
        "m_grid": [int(x) for x in c.m_grid], "shared_by_both_datasets": None}

    root = c.bridge_root_by_dataset.get(dataset)
    if not root:
        raise KeyError(f"⛔ no bridge root declared for dataset {dataset!r}")
    config.gen_store.bridge_root = str(_abs(config, root))
    bdir = bridge_dir(config, dataset, pretext, int(seed))
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(c.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    log(f"  bridge {bdir} · cache scope={cman.get('scope')!r} rows={crows.size}")

    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    # ⚠ The two loaders' arities differ and always have: ``shapes3d.build_split`` ->
    # (split, labels); ``dsprites.build_split`` -> (split, latent_values,
    # latent_classes).  ⛔ Index, never unpack (`gen-bridge-dsprites` §8.1).
    split = mod.build_split(config.gen_data.data_root)[0]
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    realised = {"n_total": int(split.n_total), "n_test": int(split.n_test),
                "n_train": int(split.n_train)}
    log(f"  ⭐ realised split {dataset}: total={realised['n_total']:,} "
        f"test={realised['n_test']:,} train={realised['n_train']:,}")

    config.gen_store.encode_chunk = int(c.encode_chunk)
    rows = select_rows(config, split, int(seed))
    reencode = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=log)

    X, NVR = {}, {}
    for name in ("store_train", "SPREAD"):
        r = rows[name]
        log(f"    gathering {name}: {r.size} rows")
        X[name] = gather_features(dataset, config, encoder, crows, feats, cman, r, log=log)
        NVR[name] = n_values_realised(factor_labels(grid, r), names, cards)

    labels, wc, ginfo = derive_partition(config, X["store_train"], int(seed), log=log)
    ship = check_partition_against_shipped(config, dataset, pretext, int(seed), labels, log=log)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])

    # ---- ⭐ the label-free per-well spectrum (context; ⛔ nothing is allocated) --
    ns = min(int(c.spectrum_rows) or X["store_train"].shape[0], X["store_train"].shape[0])
    # ⛔ A PREFIX is not a subsample (veq §12 L1): both grids are lexicographic
    # bijections and the rows are sorted.  Seeded permutation only.
    sel = np.sort(np.random.default_rng(
        int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset) + 101
    ).permutation(X["store_train"].shape[0])[:ns])
    spectra = well_spectra(X["store_train"][sel], wc)
    PR = [float(s["participation_ratio"]) for s in spectra]
    log(f"  ⭐ label-free PR = {[round(p, 3) for p in PR]} · d_k = "
        f"{[int(s['d_k']) for s in spectra]}")

    # ---- ⭐⭐ THE SWEEP -----------------------------------------------------
    K = int(c.k_wells)
    sweep = []
    for M in [int(x) for x in sorted(c.m_grid)]:
        tM = time.time()
        # ⛔ The key is the allocation harness's, and is identical at every M: the
        # ONLY thing that changes across the sweep is the number of codewords.
        key = jax.random.PRNGKey(int(seed) * int(c.store_key_mult) + int(c.store_key_offset))
        key, ik, tk = jax.random.split(key, 3)
        store0 = init_store(X["store_train"], wc, [M] * K, feat_shape, ik, seed=int(seed))
        store, hist, _ = train_store(store0, X["store_train"], config.gen_store, tk, log=None)
        if list(store.n_modes_per_well) != [M] * K:
            raise AssertionError(
                f"⛔ M={M}: the fitted store holds {store.n_modes_per_well} modes per well")
        codes = address_codes(store, np.asarray(X["SPREAD"], np.float32))
        dist = per_well_distortion(store, X["SPREAD"], codes)
        sb = store_bytes(store)
        used = [int(np.unique(codes[:, j]).size) for j in range(K)]
        sweep.append({
            "n_modes": int(M),
            "store_bytes": int(sb["store_bytes"]),
            "bits_per_read": float(sb["bits_per_read"]),
            "per_well_relative_distortion": [float(w["relative_distortion"])
                                             for w in dist["per_well"]],
            "per_well_total_variance": [float(w["total_variance"]) for w in dist["per_well"]],
            "variance_weighted_relative_distortion": float(
                dist["variance_weighted_relative_distortion"]),
            "modes_used_per_well": used,
            "all_modes_used": bool(all(u == M for u in used)),
            "train_loss_first": float(hist[0]),
            "train_loss_last": float(hist[-1]),
            "n_spread_rows": int(dist["n_rows"]),
            "wall_clock_s": float(time.time() - tM),
        })
        log(f"    M={M:4d} · eps(var-weighted)={sweep[-1]['variance_weighted_relative_distortion']:.4f}"
            f" · per-well {[round(x, 4) for x in sweep[-1]['per_well_relative_distortion']]}"
            f" · used {sum(used)}/{M * K} modes · {sweep[-1]['wall_clock_s']:.0f}s")

    rec = {
        "schema": "chlu-gen-codebook-exponent/1",
        "dataset": dataset,
        "pretext": pretext,
        "seed": int(seed),
        "dial": ("the DATASET — is the codebook exponent set by the factor manifold? "
                 "⛔ no arm, no CLU claim, no verdict"),
        "grid_gate": grid_gate,
        "realised_split": realised,
        "factor_names": names,
        "cardinalities": [int(x) for x in cards],
        "n_factors": len(names),
        "bridge_dir": str(bdir),
        "encoder_param_hash": enc_hash,
        "cache": {"scope": cman.get("scope"), "n_rows": int(crows.size),
                  "D": int(cman["D"]), "feat_shape": list(feat_shape),
                  "reencode_check": reencode},
        "rows": {"store_train": int(rows["store_train"].size),
                 "SPREAD": int(rows["SPREAD"].size),
                 "n_spread_train_drawn_and_discarded": int(
                     rows["n_spread_train_drawn_and_discarded"])},
        "n_values_realised": NVR,
        "partition": ginfo,
        "shipped_partition_check": ship,
        "spectrum": {"n_rows": int(ns), "per_well": spectra,
                     "note": "⛔ LABEL-FREE; a seeded permutation of store_train"},
        "sweep": sweep,
        "fit": fit_seed_exponent(sweep, PR),
        "wall_clock_s": float(time.time() - t0),
        "complete": True,
    }
    log(f"  ⭐⭐ {dataset} seed {seed}: exponent = {rec['fit']['seed_exponent']:.4f} "
        f"(per-well {[round(x['fitted_exponent'], 3) for x in rec['fit']['per_well']]})")
    _write_json(rec, Path(out_dir) / f"exponent_{dataset}_s{int(seed)}.json")
    return rec


# ---------------------------------------------------------------------------
# 4. The fit — ⛔ ONE function, both datasets
# ---------------------------------------------------------------------------


def fit_log_log(m: np.ndarray, eps: np.ndarray) -> dict:
    """OLS slope of ``log ε`` on ``log M`` + ``R²``.  ⭐ `gen-codebook-allocation`
    §5d's fit form (``np.polyfit(log M, log ε, 1)``), unchanged."""
    M = np.asarray(m, np.float64)
    e = np.asarray(eps, np.float64)
    ok = np.isfinite(M) & np.isfinite(e) & (M > 0) & (e > 0)
    M, e = M[ok], e[ok]
    if np.unique(M).size < 2:
        return {"status": "too few distinct M", "n_points": int(M.size),
                "fitted_exponent": float("nan"), "r_squared": float("nan"),
                "intercept": float("nan")}
    lm, le = np.log(M), np.log(e)
    slope, intercept = np.polyfit(lm, le, 1)
    pred = slope * lm + intercept
    ss_res = float(np.sum((le - pred) ** 2))
    ss_tot = float(np.sum((le - le.mean()) ** 2))
    return {"status": "ok", "n_points": int(M.size),
            "fitted_exponent": float(slope), "intercept": float(intercept),
            "level_at_M64": float(np.exp(slope * np.log(64.0) + intercept)),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def fit_seed_exponent(sweep: list, pr: list) -> dict:
    """Per-well fit over the shared grid, and the seed's exponent = their mean."""
    M = np.asarray([s["n_modes"] for s in sweep], np.float64)
    K = len(sweep[0]["per_well_relative_distortion"])
    per_well = []
    for k in range(K):
        e = np.asarray([s["per_well_relative_distortion"][k] for s in sweep], np.float64)
        f = fit_log_log(M, e)
        f.update({
            "well": int(k),
            "participation_ratio": float(pr[k]),
            "law_exponent_minus_2_over_PR": float(-2.0 / pr[k]) if pr[k] > 0 else float("nan"),
            "eps_by_M": {str(int(mm)): float(x) for mm, x in zip(M, e, strict=True)},
            "monotone_decreasing": bool(np.all(np.diff(e[np.argsort(M)]) <= 0.0)),
        })
        per_well.append(f)
    s = np.asarray([x["fitted_exponent"] for x in per_well], np.float64)
    vw = np.asarray([x["variance_weighted_relative_distortion"] for x in sweep], np.float64)
    return {
        "per_well": per_well,
        "seed_exponent": float(np.mean(s)),
        "seed_exponent_sd_over_wells": float(np.std(s)),
        "n_wells": int(K),
        "n_monotone_wells": int(sum(1 for x in per_well if x["monotone_decreasing"])),
        "variance_weighted_fit": fit_log_log(M, vw),
        "note": ("⭐ the seed's exponent is the MEAN over wells of the per-well "
                 "log-log slope (PREREG §3); the variance-weighted fit is a companion"),
    }


def _mean_sd_ci(vals) -> dict:
    v = np.asarray(list(vals), np.float64)
    n = int(v.size)
    if n == 0:
        # ⛔ A declared absence, never a silent zero (e.g. ``M = 64`` is not on a
        # ``--quick`` grid).
        return {"status": "⛔ NOT MEASURED — no points", "n": 0, "per_seed": []}
    sd_pop = float(np.std(v))
    sd_smp = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    half = T_CRIT_2DF * sd_smp / np.sqrt(n) if n == 3 else float("nan")
    return {"mean": float(v.mean()), "sd_population": sd_pop, "sd_sample": sd_smp,
            "n": n, "per_seed": [float(x) for x in v],
            "min": float(v.min()), "max": float(v.max()),
            "ci95_half_width": float(half),
            "ci95": [float(v.mean() - half), float(v.mean() + half)],
            "ci_note": "⭐ t(2, 0.975) = 4.3027 on 3 seeds (PREREG §3)"}


# ---------------------------------------------------------------------------
# 5. Across datasets — ⛔ the pre-registered rule, applied mechanically
# ---------------------------------------------------------------------------


def summarize(records: dict, config: CHLUConfig) -> dict:
    c = _cfg(config)
    ref, steep = str(c.reference_dataset), str(c.steeper_dataset)
    out = {
        "schema": "chlu-gen-codebook-exponent-summary/1",
        "dial": ("the DATASET.  ⛔ no arm, no CLU claim, no verdict on any dial"),
        "m_grid": [int(x) for x in sorted(c.m_grid)],
        "k_wells": int(c.k_wells),
        "seeds": [int(s) for s in c.seeds],
        "datasets": {},
    }
    for ds, recs in records.items():
        seed_exp = [r["fit"]["seed_exponent"] for r in recs]
        well_rows = [w["fitted_exponent"] for r in recs for w in r["fit"]["per_well"]]
        vw_exp = [r["fit"]["variance_weighted_fit"]["fitted_exponent"] for r in recs]
        out["datasets"][ds] = {
            "n_factors": int(recs[0]["n_factors"]),
            "factor_names": recs[0]["factor_names"],
            "realised_split": recs[0]["realised_split"],
            "n_values_realised_SPREAD": {
                nm: [r["n_values_realised"]["SPREAD"][nm]["n_values_realised"] for r in recs]
                for nm in recs[0]["factor_names"]},
            "well_sizes_per_seed": [r["partition"]["well_sizes"] for r in recs],
            "participation_ratio_per_seed": [
                [w["participation_ratio"] for w in r["fit"]["per_well"]] for r in recs],
            "shipped_partition_bit_identical": [
                r["shipped_partition_check"]["bit_identical"] for r in recs],
            # ⭐⭐ THE HEADLINE
            "exponent_seed_level": _mean_sd_ci(seed_exp),
            "exponent_well_level": {
                "mean": float(np.mean(well_rows)), "sd_population": float(np.std(well_rows)),
                "n": len(well_rows), "min": float(np.min(well_rows)),
                "max": float(np.max(well_rows)),
                "note": "⭐ the form `−0.2631 ± 0.0555` took: mean ± sd over (seed, well)"},
            "variance_weighted_exponent": _mean_sd_ci(vw_exp),
            # ⭐ The cross-check row: at ``M = 64`` the Shapes3D store IS
            # `gen-codebook-allocation`'s uniform arm, whose published
            # variance-weighted ε is 0.2783 ± 0.0064.  ⛔ Declared absent, never
            # zero, if 64 is not on the grid.
            "eps_at_M64_variance_weighted": _mean_sd_ci(
                [s["variance_weighted_relative_distortion"]
                 for r in recs for s in r["sweep"] if int(s["n_modes"]) == 64]),
            "eps_by_M_variance_weighted": {
                str(int(M)): _mean_sd_ci([s["variance_weighted_relative_distortion"]
                                          for r in recs for s in r["sweep"]
                                          if int(s["n_modes"]) == int(M)])
                for M in sorted(c.m_grid)},
            "monotone_wells": {"n_monotone": int(sum(r["fit"]["n_monotone_wells"] for r in recs)),
                               "n_cells": int(sum(r["fit"]["n_wells"] for r in recs))},
            "all_modes_used_every_M": bool(all(s["all_modes_used"]
                                               for r in recs for s in r["sweep"])),
            "wall_clock_s_per_seed": [r["wall_clock_s"] for r in recs],
        }
    out["reference_exponent_shapes3d_HISTORICAL"] = {
        "value": float(c.reference_exponent_shapes3d),
        "source": "gen-codebook-allocation §5d (3-point uneven sweep)",
        "role": ("⛔ HISTORICAL REFERENCE ONLY — ⛔ NEVER the comparator; the re-run "
                 "Shapes3D row above is"),
    }
    if ref in out["datasets"] and steep in out["datasets"]:
        out["comparison"] = compare_datasets(out, ref, steep, c)
    return out


def compare_datasets(out: dict, ref: str, steep: str, c) -> dict:
    """⭐⭐ The PRE-REGISTERED directional rule (PREREG §4), applied mechanically.

    **YES** iff ``max(exponent[steep]) < min(exponent[ref])`` over the 3 seeds AND
    the means order the same way.  ⛔ A NO **refutes the factor-manifold account**
    (§11b's own words) — that is a result, not a failure, and it is ⛔ never rescued
    by a well-level statistic that happens to separate.
    """
    A = out["datasets"][ref]["exponent_seed_level"]
    B = out["datasets"][steep]["exponent_seed_level"]
    non_overlapping = bool(max(B["per_seed"]) < min(A["per_seed"]))
    means_order = bool(B["mean"] < A["mean"])
    # ⛔ The pre-registered rule is stated over THREE seeds.  A 1-seed smoke must
    # not be able to emit "YES" — a ``--quick`` number is never reported, and the
    # cheapest way to keep that true is to make the harness refuse to answer.
    enough = bool(A["n"] >= 3 and B["n"] >= 3)
    if not enough:
        answer = "⛔ NOT ANSWERABLE"
    else:
        answer = "YES" if (non_overlapping and means_order) else "NO"
    comp = {
        "pure_factor_count": float(c.companion_pure_factor_count),
        "carried_offset": float(c.companion_carried_offset),
    }
    dist = {k: abs(B["mean"] - v) for k, v in comp.items()}
    closer = min(dist, key=dist.get)
    return {
        "rule": ("PREREG §4: STEEPER and non-overlapping across 3 seeds iff "
                 f"max(exponent[{steep}]) < min(exponent[{ref}]) AND the means order "
                 "the same way"),
        "reference_dataset": ref, "steeper_dataset": steep,
        f"{ref}_per_seed": A["per_seed"], f"{steep}_per_seed": B["per_seed"],
        f"{ref}_mean_sd": [A["mean"], A["sd_population"]],
        f"{steep}_mean_sd": [B["mean"], B["sd_population"]],
        "delta_mean_steeper_minus_reference": float(B["mean"] - A["mean"]),
        "seed_sets_non_overlapping": non_overlapping,
        "means_order_as_predicted": means_order,
        # ⛔ The answer in the words the acceptance criterion demands.
        "is_the_dsprites_exponent_STEEPER_non_overlapping": answer,
        "n_seeds": {ref: int(A["n"]), steep: int(B["n"])},
        "reading": (
            f"⛔ NOT ANSWERABLE: the pre-registered rule is stated over 3 seeds and "
            f"this run has {A['n']}/{B['n']}. ⛔ No directional answer exists here."
            if not enough else
            f"⭐ YES — the {steep} exponent is STEEPER than {ref}'s and the two "
            "3-seed sets do not overlap; the factor-manifold account SURVIVES this "
            "test." if answer == "YES" else
            f"⛔ NO — the {steep} exponent is NOT steeper (non-overlapping) than "
            f"{ref}'s. ⭐ By §11b's own words the FACTOR-MANIFOLD ACCOUNT IS "
            "REFUTED and −0.26's universality needs a different explanation. ⭐ This "
            "is a result, not a failure."),
        "companions": {
            "values": comp,
            "distance_from_measured": {k: float(v) for k, v in dist.items()},
            "closer": closer,
            "note": ("⭐ REPORTED AS A DISTANCE ONLY. ⛔ Neither companion may be "
                     "promoted to 'the prediction' after the fact (§11b)."),
        },
    }


# ---------------------------------------------------------------------------
# 6. Plot + driver
# ---------------------------------------------------------------------------


def make_plot(summary: dict, records: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    M = np.asarray(summary["m_grid"], np.float64)
    for ds, recs in records.items():
        y = [summary["datasets"][ds]["eps_by_M_variance_weighted"][str(int(m))]["mean"]
             for m in M]
        e = [summary["datasets"][ds]["eps_by_M_variance_weighted"][str(int(m))]["sd_population"]
             for m in M]
        axes[0].errorbar(M, y, yerr=e, marker="o", capsize=2, label=ds)
        for r in recs:
            for w in r["fit"]["per_well"]:
                axes[1].plot(M, [w["eps_by_M"][str(int(m))] for m in M], lw=0.6, alpha=0.5,
                             color="C0" if ds == "shapes3d" else "C1")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("M (codewords per well)")
    axes[0].set_ylabel(r"variance-weighted $\varepsilon$")
    axes[0].set_title("the sweep, ⛔ one shared M grid")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel(r"per-well $\varepsilon$")
    axes[1].set_title("every (seed, well): blue = shapes3d, orange = dsprites")
    dss = list(summary["datasets"])
    for i, ds in enumerate(dss):
        v = summary["datasets"][ds]["exponent_seed_level"]
        axes[2].errorbar([i], [v["mean"]], yerr=[v["sd_population"]], marker="s", capsize=4)
        axes[2].scatter([i] * len(v["per_seed"]), v["per_seed"], s=18, zorder=3)
    for y, lab, st in ((summary["reference_exponent_shapes3d_HISTORICAL"]["value"],
                        "historical −0.2631", ":"),
                       (-0.400, "companion −0.400", "--"),
                       (-0.316, "companion −0.316", "-.")):
        axes[2].axhline(y, ls=st, lw=0.8, color="k")
        axes[2].annotate(lab, (0.02, y), fontsize=6, va="bottom")
    axes[2].set_xticks(range(len(dss)))
    axes[2].set_xticklabels(dss)
    axes[2].set_ylabel("fitted exponent of ε in M")
    cmp_ = summary.get("comparison", {})
    axes[2].set_title("STEEPER, non-overlapping? "
                      f"{cmp_.get('is_the_dsprites_exponent_STEEPER_non_overlapping', '?')}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_experiment_gen_codebook_exponent(config: Optional[CHLUConfig] = None,
                                         out_dir: Optional[str] = None,
                                         log: Callable = print) -> dict:
    config = config or get_default_config()
    c = _cfg(config)
    root = Path(out_dir) if out_dir else exponent_out_root(config)
    records = {}
    for ds in list(c.datasets):
        runs = root / "runs" / f"{ds}_{c.pretext}"
        runs.mkdir(parents=True, exist_ok=True)
        recs = []
        for seed in list(c.seeds):
            path = runs / f"exponent_{ds}_s{int(seed)}.json"
            if path.exists() and bool(c.skip_if_exists):
                with open(path) as fh:
                    rec = json.load(fh)
                if rec.get("complete"):
                    log(f"  {ds} seed {seed}: artefact HIT {path.name} — skipped")
                    recs.append(rec)
                    continue
            log(f"  {ds} seed {seed}: running")
            recs.append(run_seed(config, ds, int(seed), runs, log=log))
        records[ds] = recs
    summary = summarize(records, config)
    _write_json(summary, root / "summary.json")
    if bool(c.make_plots):
        make_plot(summary, records, root / "codebook_exponent.png")
    for ds, d in summary["datasets"].items():
        v = d["exponent_seed_level"]
        log(f"⭐ {ds} ({d['n_factors']} factors): exponent = {v['mean']:.4f} ± "
            f"{v['sd_population']:.4f} (per seed {[round(x, 4) for x in v['per_seed']]}, "
            f"95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}])")
    if "comparison" in summary:
        log("⭐⭐ STEEPER, non-overlapping across 3 seeds? "
            f"{summary['comparison']['is_the_dsprites_exponent_STEEPER_non_overlapping']}")
        log(f"   {summary['comparison']['reading']}")
        log(f"   closer companion: {summary['comparison']['companions']['closer']} "
            f"(⛔ a distance, NOT 'the prediction')")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS) + [None])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--m-grid", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    config = get_default_config()
    c = _cfg(config)
    if args.datasets:
        c.datasets = list(args.datasets)
    if args.pretext:
        c.pretext = args.pretext
    if args.seeds:
        c.seeds = list(args.seeds)
    if args.quick:
        apply_quick(config)
    if args.m_grid:
        c.m_grid = list(args.m_grid)
        c.assert_grid_is_shared = False
    run_experiment_gen_codebook_exponent(config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
