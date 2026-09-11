"""⭐ **The composition certificate, run on the R0 store** — can a request be refused
from the store's own geometry, with zero invalid-example supervision?

⭐ DIAL DECLARATION — *admission*.

* **Laundering control:** a plain distance-to-store threshold on the same features
  (``distance``), and the same thing plus the grid-reach term (``distance_reach``,
  the HARD control).  ⛔ Both run on every number; neither is optional.
* **Falsifies:** certificate AUROC at chance — **or any threshold that touched
  invalid data.**  The second is checked mechanically, not promised
  (:meth:`chlu.eval.gen_cert_eval.ValidOnlyFitter.assert_no_invalid`).
* **Does NOT falsify:** losing to a *label-supervised* post-hoc detector (LACE).
  That comparison is reported from phase 1 onward but is not a kill rule (R7).

**Shape of the run** (config-load -> data -> frozen artifacts -> store -> score ->
report), one job per seed, both groupings inside the job so they share a
byte-identical context:

1. the split and the request sets — A from the bridge's held-back ``eval_a`` rows,
   B from the held-out rows, C read **verbatim** from ``c_invalid.json``;
2. the FROZEN encoder and its feature cache (hashes asserted; coverage extended
   through the same frozen encoder where the bridge wrote a ``used`` cache);
3. the store, **re-materialised** from ``store_plan.json``'s pinned grouping and the
   identical key chain (``gen-store-grouping`` shipped manifests, not a checkpoint),
   with its ``relative_mse`` asserted bit-identical against the shipped manifest
   when one is available;
4. the store's own addressing over **valid train rows** -> per-``(factor, value)``
   mode histograms -> the derived certificate;
5. the three AUROCs, the risk-coverage curve, the AURC, the operating point, and the
   audit.

⛔ **No decoder is loaded and nothing is decoded** — asserted by
:func:`chlu.core.gen_certificate.assert_no_decoder_in_path`.  That is also what makes
this stage immune to the superseded ``dec_recipe="baseline"`` trap.

⚠ The hold-out **condition** is a parameter (``--condition``), not a constant: the
same code re-points at ``recomb2element_wide`` without an edit.
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
import numpy as np

from ..config import CHLUConfig, get_default_config  # noqa: E402
from ..core.gen_bridge import (  # noqa: E402
    PRETEXTS,
    load_feature_cache,
    param_hash,
)
from ..core.gen_certificate import (  # noqa: E402
    COMPONENTS,
    assert_no_decoder_in_path,
    build_term_statistics,
    codes_for,
    factor_support,
    load_c_requests,
    requests_from_rows,
    score_requests,
)
from ..core.gen_store import (  # noqa: E402
    channel_correlation,
    dead_channels,
    grouping_labels,
    init_store,
    load_store_plan,
    mode_usage,
    reconstruction_mse,
    train_store,
    well_channels_from_labels,
)
from ..eval.gen_cert_eval import (  # noqa: E402
    ValidOnlyFitter,
    certificate_metrics,
    operating_point,
    risk_coverage,
)
from ..experiments.exp_gen_bridge import (  # noqa: E402
    DEFAULT_CONDITION,
    artifacts_tag,
    condition_of,
    load_split,
)
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    _write_json,
    gather_features,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

STAGES = ("certificate", "aggregate")
#: Kept so the CSF3 launcher's other --stage values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "store", "cell")


def _log(*a):
    print(*a, flush=True)


def _stamp(t0: float) -> str:
    return f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:7.1f}s]"


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's AUROC is NEVER a reported number."""
    c = config.gen_certificate
    config.gen_store.n_store_train = c.quick_n_store_train
    config.gen_store.n_corr = c.quick_n_corr
    config.gen_store.train_steps = c.quick_train_steps
    config.gen_store.n_modes = c.quick_n_modes
    c.seeds = list(c.quick_seeds)
    c.n_requests_a = c.quick_n_requests_a
    c.assert_store_bit_identity = False
    return config


def bridge_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    """The frozen bridge run for this ``(dataset, pretext, condition, seed)``.

    ⚠ Condition-aware via :func:`chlu.experiments.exp_gen_bridge.artifacts_tag`, so a
    ``recomb2element_wide`` run reads the wide artifacts and can never score against
    an encoder that has seen its held-out rows.
    """
    root = config.gen_certificate.bridge_root or os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    root = (Path(root).expanduser() if root else
            Path(__file__).resolve().parents[2] / ".claude" / "outputs"
            / "gen-feature-bridge" / "runs")
    tag = artifacts_tag(dataset, pretext, condition_of(config))
    d = root / tag / f"seed_{seed}"
    if not (d / "bridge_manifest.json").exists():
        raise FileNotFoundError(
            f"no frozen bridge at {d} — this spoke consumes the FROZEN cache and ⛔ "
            "never rebuilds it (set gen_certificate.bridge_root or $CHLU_GEN_BRIDGE_ROOT)"
        )
    return d


def store_runs_dir(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    """Where ``gen-store-grouping`` wrote ``store_plan.json`` for this condition."""
    root = config.gen_certificate.store_runs or os.environ.get("CHLU_GEN_STORE_RUNS")
    root = (Path(root).expanduser() if root else
            Path(__file__).resolve().parents[2] / ".claude" / "outputs"
            / "gen-store-grouping" / "runs")
    tag = artifacts_tag(dataset, pretext, condition_of(config))
    d = root / tag
    if not (d / "store_plan.json").exists():
        raise FileNotFoundError(
            f"no store plan at {d}/store_plan.json — the certificate reads the "
            "ladder's pinned grouping VERBATIM and ⛔ never re-derives K.  For a "
            "non-Montero condition the store must be fitted for that condition first."
        )
    return d


def c_invalid_path(config: CHLUConfig, dataset: str) -> Path:
    """⛔ The single-sourced C set.  Byte-identical for every arm; never regenerated."""
    p = config.gen_certificate.c_invalid_path
    if p:
        return Path(p).expanduser()
    from ..data.gen_splits import default_gen_data_root

    root = (Path(config.gen_data.data_root).expanduser()
            if config.gen_data.data_root else default_gen_data_root())
    # ⚠ ``gen_data.data_root`` is the DATASET directory for the loaders
    # (``shapes3d_root``) but the PARENT of it for ``default_gen_data_root``; both
    # spellings are in the tree, so both are accepted here rather than failing on a
    # path convention.
    for cand in (root / dataset / "c_invalid.json", root / "c_invalid.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"no c_invalid.json under {root} — ⛔ the C set is single-sourced by "
        "gen-data-splits and is never regenerated here; point gen_data.data_root or "
        "gen_certificate.c_invalid_path at it"
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _standardise(rows: list, key: str, scale: dict) -> np.ndarray:
    v = np.asarray([r[key] for r in rows], np.float64)
    if scale["passthrough"]:
        return v - scale["mean"]
    return (v - scale["mean"]) / scale["std"]


def arm_scores(rows_by_set: dict, fitter: ValidOnlyFitter, tag: str) -> dict:
    """Every arm's score for every request set.

    ⛔ Every normaliser is fitted on the **A** set alone and goes through the audited
    fitter.  The arms:

    * ``clu``            — the derived certificate: ``reach + z(pair) + z(curv) + z(energy)``
    * ``reach``          — the grid-reach term alone (0 on every on-grid request)
    * ``pair``/``curv``/``energy`` — single derived components, reported for attribution
    * ``distance``       — ⭐ the laundering control (naive composition -> distance to store)
    * ``distance_reach`` — ⭐ the HARD control (the control plus the same reach term)
    * ``count``          — the syntactic control: the number of terms in the request
    """
    a_rows = rows_by_set["A"]
    a_ids = [r["id"] for r in a_rows]
    scales = {}
    for key in ("pair", "curv", "energy", "dist"):
        scales[key] = fitter.fit_standardiser(f"{tag}:z[{key}]", a_ids,
                                              [r[key] for r in a_rows])
    out = {}
    for name, rows in rows_by_set.items():
        z = {k: _standardise(rows, k, s) for k, s in scales.items()}
        reach = np.asarray([r["reach"] for r in rows], np.float64)
        out[name] = {
            # ⭐ the PRE-REGISTERED primary, unchanged
            "clu": reach + z["pair"] + z["curv"] + z["energy"],
            # ⚠ declared SECONDARY (prereg amendment A1, filed before any full-scale
            # number): the two composed-point components are functions of a
            # distribution that sharpens with the number of terms, and the C
            # duplicate records carry 7 terms against A/B's 6, so they can inherit the
            # syntactic giveaway.  ``clu_pair_reach`` is the term-count-clean
            # combination.  ⛔ It is reported beside the primary, never instead of it.
            "clu_pair_reach": reach + z["pair"],
            "reach": reach,
            "pair": z["pair"],
            "curv": z["curv"],
            "energy": z["energy"],
            "distance": z["dist"],
            "distance_reach": z["dist"] + reach,
            "count": np.asarray([float(r["n_terms"]) for r in rows], np.float64),
        }
    return out, scales


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def run_seed(dataset: str, pretext: str, seed: int, out_dir: Path,
             config: CHLUConfig, log: Callable = _log) -> dict:
    t0 = time.time()
    cfg = config.gen_certificate
    scfg = config.gen_store
    out_dir = Path(out_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = condition_of(config)

    guard = assert_no_decoder_in_path()
    log(f"{_stamp(t0)} ⭐ structural guard: {guard['note']}")

    split = load_split(dataset, config, log=log)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_store_rows(split, config, seed)
    if cfg.n_requests_a:
        rows["eval_a"] = rows["eval_a"][: int(cfg.n_requests_a)]
    overlap = np.intersect1d(rows["store_train"], rows["eval_b"]).size
    if overlap:
        raise AssertionError(f"{overlap} store-training rows are held-out rows")

    bdir = bridge_dir(config, dataset, pretext, seed)
    bman = json.load(open(bdir / "bridge_manifest.json"))
    encoder = load_model(bdir / "encoder.pkl")
    got = param_hash(encoder)
    if got != bman["encoder_param_hash"]:
        raise AssertionError(f"encoder hash {got} != manifest {bman['encoder_param_hash']}")
    log(f"{_stamp(t0)} frozen encoder {got[:16]} · condition={condition} · bridge={bdir}")

    crows, feats, cman = load_feature_cache(bdir, verify=scfg.verify_cache_checksum)
    reenc = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=log)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])

    X_store = gather_features(dataset, config, encoder, crows, feats, cman,
                              rows["store_train"], log=log)
    log(f"{_stamp(t0)} features: store_train={X_store.shape}")

    plan = load_store_plan(store_runs_dir(config, dataset, pretext) / "store_plan.json")
    k = int(plan["K"])
    shipped_path = store_runs_dir(config, dataset, pretext) / f"seed_{seed}" / "store_manifest.json"
    shipped = json.load(open(shipped_path)) if shipped_path.exists() else None

    # --- the request sets, all three in the SAME representation -------------
    req_a = requests_from_rows(grid, rows["eval_a"], "A")
    req_b = requests_from_rows(grid, rows["eval_b"], "B")
    req_c = load_c_requests(c_invalid_path(config, dataset), names)
    c_ids = [r.id for r in req_c]
    log(f"{_stamp(t0)} requests: A={len(req_a)} B={len(req_b)} C={len(req_c)} "
        f"(C kinds: {sorted(set(r.kind for r in req_c))})")

    support = factor_support(grid, cfg.reach_radius_frac)
    lab_store = np.asarray(grid.factor_indices_of(rows["store_train"]), np.int64)

    Xc = X_store[: scfg.n_corr]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)

    manifest = {
        "schema": "chlu-gen-certificate/1",
        "dataset": dataset, "pretext": pretext, "condition": condition, "seed": int(seed),
        "K": k, "n_modes": int(scfg.n_modes), "D": int(cman["D"]),
        "encoder_param_hash": bman["encoder_param_hash"],
        "feature_cache_sha256": cman["features_sha256"],
        "cache_scope": cman["scope"], "cache_reencode": reenc,
        "decoder_guard": guard,
        "rows": {kk: int(np.asarray(v).size) for kk, v in rows.items()},
        "n_dead_channels": int(dead.sum()),
        "reach_radius": [float(x) for x in support.radius],
        "tau_curvature": float(cfg.tau_curvature if cfg.tau_curvature is not None
                              else scfg.tau_end),
        "requests": {"A": len(req_a), "B": len(req_b), "C": len(req_c)},
        "c_invalid_path": str(c_invalid_path(config, dataset)),
        "arms": {},
    }
    _write_json(manifest, out_dir / "cert_manifest.json")

    tau = float(cfg.tau_curvature if cfg.tau_curvature is not None else scfg.tau_end)
    for grouping in list(cfg.groupings):
        tg = time.time()
        labels_vec, ginfo = grouping_labels(corr, k, grouping, seed, scfg.grouping_method,
                                            dead=dead, min_well_frac=scfg.min_well_frac)
        labels_match = None
        if shipped is not None and grouping in shipped.get("arms", {}):
            ship = shipped["arms"][grouping]
            labels_match = [int(x) for x in labels_vec] == list(ship["labels"])
            if cfg.assert_store_bit_identity and not labels_match:
                raise AssertionError(
                    f"{grouping}: re-derived grouping differs from the shipped plan — "
                    "⛔ the ladder's grouping is pinned, not re-derived")
            if not cfg.assert_store_bit_identity:
                log(f"{_stamp(t0)} ⚠ quick/unasserted mode: shipped-label match = "
                    f"{labels_match} (⛔ a quick run is NEVER a reported number)")
        wc = well_channels_from_labels(labels_vec, k)
        key = jax.random.PRNGKey(seed * 10_007 + 20_260_911
                                 + (0 if grouping == "correlation" else 1))
        key, ik, tk = jax.random.split(key, 3)
        store = init_store(X_store, wc, scfg.n_modes, feat_shape, ik, seed=seed)
        log(f"{_stamp(t0)} [{grouping}] wells {[len(w) for w in wc]} — training the "
            f"store ({scfg.train_steps} steps)")
        store, hist, key = train_store(store, X_store, scfg, tk, log=None)
        rec = reconstruction_mse(store, X_store[: scfg.eval_batch * 4], scfg.eval_batch)
        bit = None
        if shipped is not None and grouping in shipped.get("arms", {}) and cfg.assert_store_bit_identity:
            X_a = gather_features(dataset, config, encoder, crows, feats, cman,
                                  rows["eval_a"], log=log)
            r_a = reconstruction_mse(store, X_a, scfg.eval_batch)
            d = abs(r_a["relative_mse"] - shipped["arms"][grouping]["recon"]["relative_mse"])
            if d != 0.0:
                raise AssertionError(
                    f"{grouping}: store re-materialisation drifted from the shipped "
                    f"manifest, |Δ rel_mse| = {d}")
            bit = {"relative_mse": r_a["relative_mse"], "abs_delta_vs_shipped": d}
            log(f"{_stamp(t0)} [{grouping}] ⭐ BIT-IDENTICAL re-materialisation "
                f"(rel_mse {r_a['relative_mse']!r})")

        codes = codes_for(store, X_store, scfg.eval_batch * 16)
        usage = mode_usage(codes, scfg.n_modes)
        stats = build_term_statistics(codes, lab_store, cards, features=X_store,
                                      alpha=cfg.laplace_alpha,
                                      row_ids=rows["store_train"])
        log(f"{_stamp(t0)} [{grouping}] term statistics over {stats.n_rows} valid train "
            f"rows; mean used modes {usage['mean_used_modes']:.1f}/{scfg.n_modes}")

        rows_by_set = {}
        for nm, reqs in (("A", req_a), ("B", req_b), ("C", req_c)):
            rows_by_set[nm] = score_requests(reqs, store, stats, support, tau,
                                             log=log if nm == "A" else None)
            log(f"{_stamp(t0)} [{grouping}] scored {nm}: {len(rows_by_set[nm])} requests")

        fitter = ValidOnlyFitter()
        fitter.register_fitting_rows(f"{grouping}:store_train_rows",
                                     [f"row:{int(i)}" for i in rows["store_train"]])
        fitter.register_fitting_rows(f"{grouping}:prototype_rows",
                                     [f"row:{int(i)}" for i in rows["store_train"]])
        scores, scales = arm_scores(rows_by_set, fitter, grouping)

        arm_out = {}
        for arm in scores["A"].keys():
            sa, sb, sc = scores["A"][arm], scores["B"][arm], scores["C"][arm]
            m = certificate_metrics(sa, sb, sc, [r["kind"] for r in rows_by_set["C"]])
            thr = fitter.fit_quantile(f"{grouping}:{arm}:threshold",
                                      [r["id"] for r in rows_by_set["A"]], sa,
                                      cfg.accept_quantile)
            m["operating_point"] = operating_point(thr, {"A": sa, "B": sb, "C": sc})
            for kind in ("duplicate_factor", "out_of_range"):
                sel = np.asarray([r["kind"] == kind for r in rows_by_set["C"]])
                if sel.any():
                    m["operating_point"][f"refusal_rate_C[{kind}]"] = float(
                        (sc[sel] > thr).mean())
            m["risk_coverage"] = {
                "B_vs_C": risk_coverage(np.r_[sb, sc],
                                        np.r_[np.zeros(sb.size), np.ones(sc.size)]),
                "A_vs_C": risk_coverage(np.r_[sa, sc],
                                        np.r_[np.zeros(sa.size), np.ones(sc.size)]),
                "balanced_AB_vs_C": risk_coverage(
                    np.r_[sa, sb, sc],
                    np.r_[np.zeros(sa.size + sb.size), np.ones(sc.size)],
                    np.r_[np.full(sa.size, 1.0 / max(sa.size, 1)),
                          np.full(sb.size, 1.0 / max(sb.size, 1)),
                          np.full(sc.size, 1.0 / max(sc.size, 1))]),
            }
            m["aurc"] = {kk: vv["aurc"] for kk, vv in m["risk_coverage"].items()}
            m["aurc_random"] = {kk: vv["aurc_random"] for kk, vv in m["risk_coverage"].items()}
            for kk in m["risk_coverage"]:
                m["risk_coverage"][kk].pop("coverage", None)
                m["risk_coverage"][kk].pop("risk", None)
            arm_out[arm] = m
            log(f"{_stamp(t0)} [{grouping}] {arm:15s} B-vs-C {m['B_vs_C']:.4f} "
                f"(dup {m.get('B_vs_C[duplicate_factor]', float('nan')):.4f} / "
                f"oor {m.get('B_vs_C[out_of_range]', float('nan')):.4f}) · "
                f"A-vs-C {m['A_vs_C']:.4f} · bal {m['AB_vs_C_balanced']:.4f} · "
                f"AURC(B∪C) {m['aurc']['B_vs_C']:.4f}")

        audit = fitter.assert_no_invalid(c_ids)
        log(f"{_stamp(t0)} [{grouping}] ⭐ AUDIT PASSED: {audit['n_fits_audited']} fits, "
            f"{audit['n_ids_audited']} ids, {audit['n_invalid_ids_found_in_any_fit']} of "
            f"{audit['n_invalid_ids_checked']} C ids found in any fit")

        attribution = _attribution(rows_by_set["C"], names)
        manifest["arms"][grouping] = {
            "grouping": grouping,
            "grouping_info": ginfo,
            "labels_match_shipped_plan": labels_match,
            "well_sizes": [len(w) for w in wc],
            "labels": [int(x) for x in labels_vec],
            "store_recon_on_train_slice": rec,
            "bit_identity_vs_shipped": bit,
            "mode_usage": usage,
            "term_counts_min": int(min(min(c) for c in stats.counts)),
            "term_statistics_rows_sha256": stats.source_hash,
            "z_scales": {kk: {"mean": vv["mean"], "std": vv["std"],
                              "passthrough_zero_variance": vv["passthrough"]}
                         for kk, vv in scales.items()},
            "arms": arm_out,
            "no_invalid_supervision_audit": audit,
            "violated_factor_attribution": attribution,
            "component_summary": _component_summary(rows_by_set),
            "seconds": float(time.time() - tg),
        }
        _write_json(manifest, out_dir / "cert_manifest.json")
        _write_json({"A": rows_by_set["A"], "B": rows_by_set["B"], "C": rows_by_set["C"]},
                    out_dir / f"scores_{grouping}.json")
        log(f"{_stamp(t0)} [{grouping}] written -> {out_dir}")

    manifest["seconds"] = float(time.time() - t0)
    _write_json(manifest, out_dir / "cert_manifest.json")
    return manifest


def _attribution(c_rows: list, names) -> dict:
    """⭐ Does the certificate name the RIGHT factor?

    For a ``duplicate_factor`` record the two terms of the max-conflict pair should be
    the duplicated factor; for ``out_of_range`` the max-reach term should be the
    offending one.  ⚠ This is attribution, not accuracy: it is independent of the
    term-count giveaway, so it says whether the barrier fires for the right reason.
    """
    names = list(names)
    per = {}
    for kind, keyfn in (
        ("duplicate_factor", lambda r: (r["pair_argmax_factors"][0]
                                        if r["pair_argmax_factors"][0] == r["pair_argmax_factors"][1]
                                        else -1)),
        ("out_of_range", lambda r: r["reach_argmax_factor"]),
    ):
        sel = [r for r in c_rows if r["kind"] == kind]
        if not sel:
            continue
        hits = 0
        for r in sel:
            j = keyfn(r)
            hits += int(0 <= j < len(names) and names[j] == r.get("violated_factor"))
        per[kind] = {"n": len(sel), "correct": int(hits),
                     "rate": float(hits) / max(len(sel), 1),
                     "chance": 1.0 / len(names)}
    return per


def _component_summary(rows_by_set: dict) -> dict:
    out = {}
    for nm, rows in rows_by_set.items():
        out[nm] = {c: {"mean": float(np.mean([r[c] for r in rows])),
                       "std": float(np.std([r[c] for r in rows])),
                       "median": float(np.median([r[c] for r in rows]))}
                   for c in list(COMPONENTS) + ["dist", "n_terms"]}
    return out


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def _ms(vals) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], np.float64)
    return {"mean": float(v.mean()) if v.size else float("nan"),
            "std": float(v.std()) if v.size else float("nan"),
            "per_seed": [float(x) for x in vals], "n_seeds": int(v.size)}


def aggregate_runs(run_dirs, log: Callable = _log) -> dict:
    mans = []
    for d in run_dirs:
        p = Path(d)
        p = p if p.name == "cert_manifest.json" else p / "cert_manifest.json"
        mans.append(json.load(open(p)))
    if not mans:
        raise ValueError("no run manifests")
    groupings = sorted(set().union(*[set(m["arms"]) for m in mans]))
    keys = ("A_vs_C", "B_vs_C", "AB_vs_C_balanced", "pooled_UNBALANCED_do_not_quote",
            "B_vs_C[duplicate_factor]", "B_vs_C[out_of_range]",
            "A_vs_C[duplicate_factor]", "A_vs_C[out_of_range]")
    agg = {"schema": "chlu-gen-certificate-aggregate/1",
           "seeds": [m["seed"] for m in mans],
           "condition": mans[0].get("condition", DEFAULT_CONDITION),
           "dataset": mans[0]["dataset"], "pretext": mans[0]["pretext"],
           "K": mans[0]["K"], "n_modes": mans[0]["n_modes"],
           "requests": mans[0]["requests"], "arms": {}}
    for g in groupings:
        per_arm = {}
        arm_names = sorted(set().union(*[set(m["arms"][g]["arms"]) for m in mans if g in m["arms"]]))
        for arm in arm_names:
            got = {kk: _ms([m["arms"][g]["arms"][arm].get(kk, float("nan")) for m in mans])
                   for kk in keys}
            got["aurc_B_vs_C"] = _ms([m["arms"][g]["arms"][arm]["aurc"]["B_vs_C"] for m in mans])
            got["refusal_rate_B"] = _ms([m["arms"][g]["arms"][arm]["operating_point"]["refusal_rate_B"]
                                         for m in mans])
            got["refusal_rate_A"] = _ms([m["arms"][g]["arms"][arm]["operating_point"]["refusal_rate_A"]
                                         for m in mans])
            got["refusal_rate_C"] = _ms([m["arms"][g]["arms"][arm]["operating_point"]["refusal_rate_C"]
                                         for m in mans])
            per_arm[arm] = got
        agg["arms"][g] = {
            "per_arm": per_arm,
            "laundering_margin_dup": _ms([
                m["arms"][g]["arms"]["clu"].get("B_vs_C[duplicate_factor]", float("nan"))
                - m["arms"][g]["arms"]["distance_reach"].get("B_vs_C[duplicate_factor]", float("nan"))
                for m in mans]),
            "laundering_margin_overall": _ms([
                m["arms"][g]["arms"]["clu"]["B_vs_C"]
                - m["arms"][g]["arms"]["distance"]["B_vs_C"] for m in mans]),
            "attribution": [m["arms"][g]["violated_factor_attribution"] for m in mans],
            "audit": [m["arms"][g]["no_invalid_supervision_audit"][
                "n_invalid_ids_found_in_any_fit"] for m in mans],
        }
    agg["CERTIFIED_no_invalid_supervision"] = all(
        v == 0 for g in agg["arms"].values() for v in g["audit"])
    if "correlation" in agg["arms"] and "random" in agg["arms"]:
        agg["grouping_contrast_dup"] = _ms([
            m["arms"]["correlation"]["arms"]["clu"].get("B_vs_C[duplicate_factor]", float("nan"))
            - m["arms"]["random"]["arms"]["clu"].get("B_vs_C[duplicate_factor]", float("nan"))
            for m in mans if "correlation" in m["arms"] and "random" in m["arms"]])
    return agg


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_experiment_gen_certificate(config: Optional[CHLUConfig] = None,
                                   dataset: Optional[str] = None,
                                   pretext: Optional[str] = None,
                                   out_dir: Optional[str] = None,
                                   seeds=None, quick: bool = False,
                                   log: Callable = _log) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_certificate
    dataset = dataset or cfg.dataset
    pretext = pretext or cfg.pretext
    out = Path(out_dir or (Path("results") / "gen_certificate"
                           / f"{dataset}_{pretext}"))
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds) if seeds else list(cfg.seeds)
    dirs = []
    for s in seeds:
        run_seed(dataset, pretext, int(s), out, config, log=log)
        dirs.append(out / f"seed_{s}")
    agg = aggregate_runs(dirs, log=log)
    _write_json(agg, out / "certificate_results.json")
    return agg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chlu.experiments.exp_gen_certificate",
        description="the composition certificate: refusal derived from the store, "
                    "with zero invalid-example supervision",
    )
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument("--stage", default="certificate",
                   choices=list(STAGES) + list(NOT_MINE),
                   help="certificate | aggregate (encoder/decoder -> exp_gen_bridge, "
                        "store -> exp_gen_store, cell -> exp_gen_cell)")
    p.add_argument("--arm", default="all",
                   help="recorded in the manifest; ⛔ every arm (including the "
                        "laundering controls) is scored in one job so they share a "
                        "byte-identical context")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="artifacts directory")
    p.add_argument("--condition", default=None,
                   help="hold-out condition (default: gen_data.condition)")
    p.add_argument("--quick", action="store_true", help="plumbing smoke; NOT a result")
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--store-runs", default=None, help="where store_plan.json lives")
    p.add_argument("--data-root", default=None)
    p.add_argument("--groupings", nargs="+", default=None)
    p.add_argument("--n-requests-a", type=int, default=None)
    p.add_argument("--runs", nargs="+", default=None, help="per-seed dirs for --stage aggregate")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    if args.quick:
        config = apply_quick(config)
    c = config.gen_certificate
    if args.data_root:
        config.gen_data.data_root = args.data_root
    if args.condition:
        config.gen_data.condition = args.condition
    if args.bridge_root:
        c.bridge_root = args.bridge_root
    if args.store_runs:
        c.store_runs = args.store_runs
    if args.groupings:
        c.groupings = list(args.groupings)
    if args.n_requests_a is not None:
        c.n_requests_a = int(args.n_requests_a)
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.stage in NOT_MINE:
        print(f"[gen-cert] stage={args.stage} is not this module's "
              "(encoder/decoder -> exp_gen_bridge, store -> exp_gen_store). "
              "Refusing to no-op.", file=sys.stderr)
        return 2
    config = _config_from_args(args)
    cfg = config.gen_certificate
    dataset = args.dataset or cfg.dataset
    pretext = args.pretext or cfg.pretext
    out = Path(args.out or (Path("results") / "gen_certificate" / f"{dataset}_{pretext}"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gen-cert] dataset={dataset} pretext={pretext} stage={args.stage} "
          f"arm={args.arm} seed={args.seed} condition={condition_of(config)} out={out}")
    print("[gen-cert] ⭐ the distance-to-store laundering control runs in this job and "
          "is reported beside every certificate number; ⛔ it is never optional.")
    if args.stage == "certificate":
        run_seed(dataset, pretext, int(args.seed), out, config)
    if args.stage == "aggregate":
        if not args.runs:
            print("[gen-cert] --stage aggregate needs --runs", file=sys.stderr)
            return 2
        agg = aggregate_runs(args.runs)
        _write_json(agg, out / "certificate_results.json")
        print(f"[gen-cert] ⭐ audit certified across seeds: "
              f"{agg['CERTIFIED_no_invalid_supervision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
