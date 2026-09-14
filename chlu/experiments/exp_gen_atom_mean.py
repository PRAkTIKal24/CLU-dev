"""⭐⭐ **ATOM-MEAN LOOKUP — the like-for-like control the dynamical read must beat.**

One arm::

    request -> R8 prototype embedding -> nearest STORED ATOMS (per well) -> mean
            -> THE frozen decoder -> image -> THE external classifier

⭐ **DIAL DECLARATION** — ⛔ *none: this module is a **CONTROL**.*

- **Dial:** none.  ⛔ **No CLU number is produced here and no verdict is rendered.**
- **Laundering control:** ⭐ *this module **is** the control.*  Its falsifier is
  **a control that is weaker than it should be.**
- **Falsifies:** ``gen-dynamical-read``'s claim, if a plain lookup over the same atoms
  does the same job.

⭐ **Why this arm decides that claim.**  The dynamical read's whole case
(``gen-dynamical-read/PREREG.md`` §2) is **a better EXPONENT at matched bytes** —
interpolation beating piecewise-constant lookup **over the same stored points**.
⭐ **Without this arm, a win could be bought by storing atoms instead of centres rather
than by settling at all.**

⛔ **Matched bytes is the whole point.**  ``N`` atoms/well × ``D`` × 4 must **equal**
the store budget: **A = 524,288 B** (64 atoms/well — ⭐ the deployed CLU store exactly,
which is 64 codewords per well) and **B = 4,194,304 B** (512 atoms/well).  The realised
ledger is asserted; the run aborts on any other value.

⛔ **Never weaken this rival.**  The fit rows are ``GenBaselinesConfig``'s
**32,256 / 512** (at 1,024 rows ``knn_proto`` fell 0.8763 → 0.8308 — a rival re-fitted
on a spoke's own row budget is a weakened rival).  The atom selection is **k-medoids**,
the strongest honest member of the family, and the random subsample runs beside it as a
**declared weaker companion** so "as strong as it can be" is measured, not asserted.
The ``(k, metric, normaliser, scope)`` grid is selected on **validation**; ⛔ B never
enters selection.

⚠ **A per-factor statistic is meaningless without ``n_values_realised``** (R31): on the
72 held-out (B) requests ``shape`` realises **one** value, so its ladder row is
``NOT APPLICABLE``.  The ``A`` (seen-spread) row set carries the six-factor ladder where
the factors actually vary.  ⛔ Every ladder row below ships ``n_values_realised``.

⛔ **Live traps this module defends against with assertions, not comments.**

- ``GenBridgeConfig.dec_recipe`` still defaults to ``"baseline"`` — the recipe that
  **FAILED** the gate.  ``v1_capacity_schedule`` is selected explicitly and its
  parameter hash is asserted per arm per seed.
- **The full, unrestricted ``knn_proto`` must not move.**  It is re-run here and its
  probe in-corner number must be **bit-identical** to the published constants
  (``require_knn_proto_bit_identical``); ⛔ a moved ``knn_proto`` voids the comparison.
- ⚠ Feature caches are non-uniform (seed 0 ``scope="full"``, seeds 1–2 ``scope="used"``).
- ⛔ **JAX landmine:** ``jax_threefry_partitionable=True`` makes ``split(key, n)``
  prefix-stable, so ``split(k,2)[1] IS split(k,3)[1]``.  This module derives **no** arm
  key by lengthening a split; the atom streams are NumPy ``SeedSequence`` streams keyed
  on ``[seed, budget_tag, scope_tag, well]``.

Runnable::

    python -m chlu.experiments.exp_gen_atom_mean --dataset shapes3d --pretext recon \\
        --stage atoms --seed 0 --out DIR
    python -m chlu.experiments.exp_gen_atom_mean --stage aggregate --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import CHLUConfig, gen_atom_mean_config, get_default_config  # noqa: E402
from ..core.gen_atoms import (  # noqa: E402
    atom_byte_ledger,
    atom_read,
    atoms_for_budget,
    fitted_exponent,
    relative_read_error,
    save_atoms,
    select_atoms,
)
from ..core.gen_bridge import load_feature_cache, param_hash  # noqa: E402
from ..core.gen_compose import COMPOSITIONAL_FACTOR  # noqa: E402
from ..core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from ..core.gen_store import load_store_plan, well_channels_from_labels  # noqa: E402
from ..eval.gen_probe import factor_labels  # noqa: E402
from ..eval.rivals.gen_knn import fit_normaliser  # noqa: E402
from ..experiments.exp_gen_bridge import condition_of  # noqa: E402
from ..experiments.exp_gen_cell import (  # noqa: E402
    check_preconditions,
    distinct_counts,
    in_corner,
    load_external_classifier,
    render,
    score_images,
)
from ..experiments.exp_gen_compose import _ms, _write_json, resolve_decoder  # noqa: E402
from ..experiments.exp_gen_dream import resolve_grouping  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

STAGES = ("atoms", "aggregate")
#: ⛔ Kept so the launcher's other ``--stage`` values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "store", "dream", "certificate", "baselines",
            "compose", "cell")
#: ⭐ ``knn_proto`` is the REGRESSION CHECK, not a competitor: it is the arm this one is
#: a restriction of, and it must come back bit-identical.
REQUEST_SETS = ("B", "MS", "A")


def _root(config: CHLUConfig) -> Path:
    """``<claude>`` for THIS spoke.

    ⚠ A worktree has no ``.claude/``, so ``$CHLU_CLAUDE_ROOT`` must be set when running
    from one (same trap as ``$CHLU_GEN_DATA_ROOT``).
    """
    cfg = gen_atom_mean_config(config)
    r = cfg.claude_root or os.environ.get("CHLU_CLAUDE_ROOT")
    if r:
        return Path(r).expanduser()
    return Path(__file__).resolve().parents[2] / ".claude"


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — ⛔ a quick number is NEVER reported."""
    g = gen_atom_mean_config(config)
    g.seeds = list(g.quick_seeds)
    g.budgets = dict(g.quick_budgets)
    g.n_fit = int(g.quick_n_fit)
    g.n_val = int(g.quick_n_val)
    g.n_eval_a = int(g.quick_n_eval_a)
    g.tune_n = int(g.quick_tune_n)
    g.knn_ks = list(g.quick_knn_ks)
    g.metrics = list(g.quick_metrics)
    g.normalisers = list(g.quick_normalisers)
    g.scopes = list(g.quick_scopes)
    g.atom_kmeans_iters = int(g.quick_atom_kmeans_iters)
    g.verify_cache_checksum = bool(g.quick_verify_cache_checksum)
    g.run_subsample_companion = bool(g.quick_run_subsample_companion)
    #: ⛔ bit-identity is a 3-seed production claim; a 4,096-row quick prototype table
    #: is a DIFFERENT table and the check is DECLARED SKIPPED rather than relaxed.
    g.require_knn_proto_bit_identical = False
    return config


def store_plan_path(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    cfg = gen_atom_mean_config(config)
    if cfg.store_plan:
        return Path(cfg.store_plan).expanduser()
    return (_root(config) / "outputs" / "gen-store-grouping" / "runs"
            / f"{dataset}_{pretext}" / "store_plan.json")


# ---------------------------------------------------------------------------
# 1. ⭐ The retention ladder — ⛔ never a per-factor row without n_values_realised
# ---------------------------------------------------------------------------


def retention_ladder(model, images: np.ndarray, labels: np.ndarray, names,
                     batch: int) -> dict:
    """Six-factor retention + ⭐ ``n_values_realised`` on **every** row.

    ⚠ R31, fourth strike: a per-factor statistic computed on a row set that realises
    **one** value of that factor is a detection rate wearing an accuracy's name.  Such
    a row is marked ``NOT APPLICABLE`` here rather than quoted.
    """
    sc = score_images(model, images, labels, names, int(batch))
    Y = np.asarray(labels, np.int64)
    ladder = {}
    for j, nm in enumerate(names):
        nvr = int(np.unique(Y[:, j]).size) if Y.size else 0
        row = {
            "accuracy": float(sc["per_factor"][str(nm)]),
            "n_values_realised": nvr,
            "chance": float(1.0 / nvr) if nvr > 0 else float("nan"),
        }
        if nvr <= 1:
            row["status"] = ("NOT APPLICABLE (1 realised value — R31): this is a "
                             "DETECTION rate, ⛔ never a retention accuracy")
        ladder[str(nm)] = row
    return {
        "per_factor": ladder,
        "aggregate": float(sc["aggregate"]),
        "exact_match": float(sc["exact_match"]),
        "n": int(sc["n"]),
        "n_factors_applicable": int(sum(1 for r in ladder.values()
                                        if r["n_values_realised"] > 1)),
    }


# ---------------------------------------------------------------------------
# 2. The arm
# ---------------------------------------------------------------------------


def _trial_grid(cfg) -> list:
    return [
        {"k": int(k), "metric": str(m), "normaliser": str(nz), "scope": str(sc)}
        for sc in list(cfg.scopes)
        for nz in list(cfg.normalisers)
        for m in list(cfg.metrics)
        for k in list(cfg.knn_ks)
    ]


def run_budget(tag: str, budget_bytes: int, ctx: dict, cfg, seed: int, out_dir: Path,
               log: Callable = print) -> dict:
    """One byte budget: build the atoms, select on validation, score, read-error."""
    t0 = time.time()
    D = int(ctx["D"])
    N = atoms_for_budget(int(budget_bytes), D)
    budget_tag = sorted(cfg.budgets).index(tag) if tag in cfg.budgets else 0
    log(f"  ⭐ [{tag}] budget {int(budget_bytes):,} B -> N = {N} atoms/well "
        f"(D={D}, an atom costs {D * 4:,} B)")

    sets, ledgers = {}, {}
    for scope in sorted(set(list(cfg.scopes))):
        aset = select_atoms(ctx["X_store"], ctx["well_channels"], N, int(seed),
                            scope=scope, method=str(cfg.atom_method),
                            iters=int(cfg.atom_kmeans_iters), chunk=int(cfg.atom_chunk),
                            budget_tag=budget_tag, log=log)
        led = atom_byte_ledger(aset.n_atoms, aset.well_sizes, int(budget_bytes),
                               scope=scope)
        led["deployed_bytes_measured"] = aset.deployed_bytes()
        # ⛔ byte EQUALITY, asserted — the whole point of the arm
        if bool(cfg.assert_byte_equality) and (
            led["realised_bytes"] != int(budget_bytes)
            or led["deployed_bytes_measured"] != int(budget_bytes)
        ):
            raise AssertionError(
                f"⛔ atom set ({scope}) realises {led['realised_bytes']} B / measured "
                f"{led['deployed_bytes_measured']} B against a budget of "
                f"{int(budget_bytes)} B — matched bytes means EXACTLY matched"
            )
        sets[scope] = aset
        ledgers[scope] = led
        log(f"    ⭐ [{tag}/{scope}] byte ledger {led['realised_bytes']:,} B == budget "
            f"{int(budget_bytes):,} B (matched_exactly={led['matched_exactly']}), "
            f"sha256 {aset.sha256()[:16]}…")

    # ⭐ persist the canonical (per-well) atom set so the landscape loads THESE atoms
    persisted = {}
    canonical = "per_well" if "per_well" in sets else sorted(sets)[0]
    if bool(cfg.write_atom_npz):
        persisted = save_atoms(sets[canonical],
                               Path(out_dir) / f"atoms_s{int(seed)}_{tag}.npz")
        persisted["canonical_scope"] = canonical
        persisted["what"] = ("⭐ THE stored atoms. gen-dynamical-read must load THIS "
                             "file, ⛔ not re-derive a set that only looks the same.")
        log(f"    ⭐ atoms persisted: {persisted['path']} sha256 "
            f"{persisted['sha256'][:16]}…")

    # ---- tuning: ⛔ VALIDATION only, B never enters -------------------------
    grid = _trial_grid(cfg)
    n_tune = int(min(int(cfg.tune_n), ctx["X_val"].shape[0]))
    Qv = ctx["Q_val"][:n_tune]
    lab_v = ctx["lab_val"][:n_tune]
    log(f"  [{tag}] tuning budget: {len(grid)} trials of "
        f"{int(cfg.tuning_trial_budget)} available; selection = "
        f"{cfg.tuning_select_on} on {n_tune} VALIDATION rows (⛔ B never enters)")
    trials = []
    half = int(n_tune // 2)
    for t_i, tr in enumerate(grid):
        nzt = ctx["normalisers"][tr["normaliser"]]
        r = atom_read(sets[tr["scope"]], Qv, tr["k"], tr["metric"], nzt)
        img = render(ctx["decoder"], r["read"], ctx["feat_shape"], int(cfg.eval_batch))
        sc = score_images(ctx["probe"], img, lab_v, ctx["names"], int(cfg.eval_batch))
        # ⭐ the same render scored on two disjoint halves — the SELECTION-STABILITY
        # diagnostic.  ⚠ ``val`` is the head/tail slice of a SORTED row array
        # (``select_baseline_rows``, inherited verbatim), so it is NOT exchangeable
        # with ``fit``; if the two halves disagree about the winner, the selection is
        # riding that structure and the arm may be WEAKENED by its own tuning.
        h1 = score_images(ctx["probe"], img[:half], lab_v[:half], ctx["names"],
                          int(cfg.eval_batch))["aggregate"] if half else float("nan")
        h2 = score_images(ctx["probe"], img[half:], lab_v[half:], ctx["names"],
                          int(cfg.eval_batch))["aggregate"] if half else float("nan")
        trials.append({**tr, "val_aggregate": float(sc["aggregate"]),
                       "val_compositional": float(sc["per_factor"][COMPOSITIONAL_FACTOR]),
                       "val_aggregate_half1": float(h1),
                       "val_aggregate_half2": float(h2),
                       "bits_per_read": float(r["bits_per_read"])})
        if (t_i + 1) % 6 == 0 or t_i == 0:
            log(f"    [{tag}] trial {t_i + 1}/{len(grid)} {tr}: "
                f"val agg={sc['aggregate']:.4f}")
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    stability = {
        "argmax_half1": {k: trials[int(np.argmax([t["val_aggregate_half1"]
                                                  for t in trials]))][k]
                         for k in ("k", "metric", "normaliser", "scope")},
        "argmax_half2": {k: trials[int(np.argmax([t["val_aggregate_half2"]
                                                  for t in trials]))][k]
                         for k in ("k", "metric", "normaliser", "scope")},
        "selected": {k: best[k] for k in ("k", "metric", "normaliser", "scope")},
        "spread_over_trials": float(np.std([t["val_aggregate"] for t in trials])),
        "gap_best_to_second": float(np.diff(np.sort(
            [t["val_aggregate"] for t in trials])[-2:])[0]) if len(trials) > 1 else 0.0,
        "what": ("⚠ ``val`` is the TAIL slice of a SORTED row index array, so it is "
                 "not exchangeable with ``fit``. ⛔ The row construction is INHERITED "
                 "verbatim from select_baseline_rows and is NOT changed here; this is "
                 "the diagnostic that says whether the selection depends on it."),
    }
    stability["halves_agree"] = bool(
        stability["argmax_half1"] == stability["argmax_half2"] == stability["selected"])
    log(f"  ⭐ [{tag}] SELECTED (validation aggregate, ⛔ C/B never enter): {best}")
    log(f"    [{tag}] selection stability: halves agree={stability['halves_agree']} "
        f"(half1 {stability['argmax_half1']}, half2 {stability['argmax_half2']}); "
        f"best-to-second gap {stability['gap_best_to_second']:.4f}, trial spread "
        f"{stability['spread_over_trials']:.4f}")

    aset = sets[best["scope"]]
    nz = ctx["normalisers"][best["normaliser"]]

    # ---- the scored request sets ------------------------------------------
    reads = {rs: atom_read(aset, ctx[f"Q_{rs}"], best["k"], best["metric"], nz)
             for rs in REQUEST_SETS if ctx.get(f"Q_{rs}") is not None}
    imgs = {rs: render(ctx["decoder"], r["read"], ctx["feat_shape"],
                       int(cfg.eval_batch)) for rs, r in reads.items()}
    shape_idx = int(ctx["names"].index(COMPOSITIONAL_FACTOR))
    scored = {}
    for inst_name, inst in (("external", ctx["external"]), ("probe", ctx["probe"])):
        scored[inst_name] = {
            "in_corner": in_corner(inst, imgs["B"], imgs["MS"], ctx["lab_B"], shape_idx,
                                   int(cfg.eval_batch)),
            "retention_ladder": {
                rs: retention_ladder(inst, imgs[rs], ctx[f"lab_{rs}"], ctx["names"],
                                     int(cfg.eval_batch))
                for rs in imgs
            },
        }
    distinct = {rs: distinct_counts(ctx["external"], reads[rs]["read"], imgs[rs],
                                    int(cfg.eval_batch)) for rs in imgs}
    for rs in imgs:
        distinct[rs]["atom_configurations"] = int(reads[rs]["n_distinct_configs"])
        distinct[rs]["atom_configuration_note"] = (
            "⭐ the tuple of retrieved atom index-sets — the atom arm's analogue of "
            "the store's K-tuple code, and the like-for-like diversity count"
        )
    log(f"  ⭐ [{tag}] IMAGE in-corner {COMPOSITIONAL_FACTOR} balacc="
        f"{scored['external']['in_corner']['balanced_accuracy']:.4f} (probe "
        f"{scored['probe']['in_corner']['balanced_accuracy']:.4f}) · distinct on B: "
        f"{distinct['B']['atom_configurations']} atom-configs / "
        f"{distinct['B']['feature_vectors']} vectors / "
        f"{distinct['B']['factor_tuples']} factor-tuples of {distinct['B']['n']}")

    # ---- ⭐ read error: the EXPONENT quantity -------------------------------
    # ⛔ Measured on EVERY declared row set, and the PRIMARY one is the set the store
    # was never fitted on: ``A`` (``eval_a``) is excluded from ``store_train`` by
    # ``select_store_rows``, which is what makes it comparable with
    # ``gen-variance-equalisation``'s ``SPREAD`` — the set the 0.2783 anchor came from.
    # ⚠ ``val`` rows sit INSIDE the atom pool, so their number is OPTIMISTIC and is
    # reported as a companion, ⛔ never as the headline.
    read_err = {"rows_primary": str(cfg.read_error_primary),
                "rows_measured": list(cfg.read_error_rows)}
    for rows_key in list(cfg.read_error_rows):
        X_t = ctx.get(f"X_{rows_key}")
        Q_t = ctx.get(f"Q_{rows_key}")
        if X_t is None or Q_t is None:
            read_err[rows_key] = {"NOT_RUN": f"no row set {rows_key!r} in this run"}
            continue
        leg = {}
        for label, query in (("oracle_feature_addressed", X_t),
                             ("request_addressed", Q_t)):
            for kname, kk in (("k1", 1), ("k_selected", int(best["k"]))):
                r = atom_read(aset, query, kk, best["metric"], nz)
                leg[f"{label}/{kname}"] = {
                    **relative_read_error(r["read"], X_t),
                    "k": int(kk), "metric": best["metric"], "scope": best["scope"],
                    "normaliser": best["normaliser"],
                    "n_distinct_configs": int(r["n_distinct_configs"]),
                }
        # ⭐ THE CENTROID CONTROL — 0 deployed bytes, ⛔ not an arm.  It isolates the
        # MEDOID PENALTY (data point vs centre at identical N) from "my k-means vs the
        # shipped trained store", which the 0.2783 anchor would otherwise confound.
        if aset.centroids is not None:
            cr = atom_read(aset, X_t, 1, best["metric"], nz, use_centroids=True)
            leg["centroid_control/k1"] = {
                **relative_read_error(cr["read"], X_t),
                "k": 1, "deployed_bytes_on_this_arms_path": 0,
                "role": ("⚠ FIT-TIME CONTROL, ⛔ NOT AN ARM: decoding a centre is the "
                         "shipped codebook's mechanism. It is here so the medoid "
                         "penalty is MEASURED, not assumed."),
            }
            leg["medoid_penalty"] = {
                "ratio": (leg["oracle_feature_addressed/k1"]["relative_mse"]
                          / leg["centroid_control/k1"]["relative_mse"]
                          if leg["centroid_control/k1"]["relative_mse"] > 0
                          else float("nan")),
                "what": ("⭐ medoid(k=1) / centroid(k=1) at IDENTICAL N and an "
                         "identical Lloyd fit — the price of storing DATA POINTS "
                         "rather than centres, which is exactly the trade the "
                         "dynamical read makes."),
            }
        read_err[rows_key] = leg
    read_err["what"] = (
        "⭐ relative_mse = MSE(read, true features) / Var(true features) — identical "
        "in form to exp_gen_variance_eq._hard_read_mse, which is where the centroid "
        "quantiser's 0.2783 anchor came from, so the two are directly comparable. "
        "⭐ 'oracle_feature_addressed' is the QUANTISER-comparable number (the law "
        "eps ~ N^(-2/PR) is about it); 'request_addressed' is the DEPLOYED path, whose "
        "error also carries the R8 embedding's own error."
    )
    pk = str(cfg.read_error_primary)
    if isinstance(read_err.get(pk), dict) and "NOT_RUN" not in read_err[pk]:
        pr = read_err[pk]
        log(f"  ⭐ [{tag}] relative read error on {pk} (⛔ unfitted rows): oracle k=1 "
            f"{pr['oracle_feature_addressed/k1']['relative_mse']:.4f} · oracle "
            f"k={best['k']} "
            f"{pr['oracle_feature_addressed/k_selected']['relative_mse']:.4f} · "
            f"request k={best['k']} "
            f"{pr['request_addressed/k_selected']['relative_mse']:.4f} · ⚠ CENTROID "
            f"control {pr.get('centroid_control/k1', {}).get('relative_mse', float('nan')):.4f} "
            f"(medoid penalty ×{pr.get('medoid_penalty', {}).get('ratio', float('nan')):.3f})")

    # ---- ⚠ the declared WEAK companion: is k-medoids actually the strong one? --
    companion = {"NOT_RUN": "run_subsample_companion=False"}
    if bool(cfg.run_subsample_companion):
        w = select_atoms(ctx["X_store"], ctx["well_channels"], N, int(seed),
                         scope=best["scope"], method="subsample",
                         budget_tag=budget_tag)
        wr = {rs: atom_read(w, ctx[f"Q_{rs}"], best["k"], best["metric"], nz)
              for rs in ("B", "MS")}
        wi = {rs: render(ctx["decoder"], wr[rs]["read"], ctx["feat_shape"],
                         int(cfg.eval_batch)) for rs in wr}
        X_c = ctx[f"X_{cfg.read_error_primary}"]
        wo = atom_read(w, X_c, 1, best["metric"], nz)
        companion = {
            "method": "subsample",
            "byte_ledger": atom_byte_ledger(w.n_atoms, w.well_sizes, int(budget_bytes),
                                            scope=best["scope"]),
            "external_in_corner_balacc": float(
                in_corner(ctx["external"], wi["B"], wi["MS"], ctx["lab_B"], shape_idx,
                          int(cfg.eval_batch))["balanced_accuracy"]),
            "probe_B_aggregate": float(score_images(
                ctx["probe"], wi["B"], ctx["lab_B"], ctx["names"],
                int(cfg.eval_batch))["aggregate"]),
            "oracle_relative_mse_k1": float(
                relative_read_error(wo["read"], X_c)["relative_mse"]),
            "role": ("⚠ THE DECLARED WEAK MEMBER. ⭐ If it beats the shipped k-medoid "
                     "atoms, the shipped selection is NOT the strong one and ⛔ this "
                     "spoke must stop and report."),
        }
        strong = read_err[str(cfg.read_error_primary)][
            "oracle_feature_addressed/k1"]["relative_mse"]
        companion["kmedoid_is_stronger_on_read_error"] = bool(
            strong <= companion["oracle_relative_mse_k1"])
        companion["kmedoid_is_stronger_on_in_corner"] = bool(
            scored["external"]["in_corner"]["balanced_accuracy"]
            >= companion["external_in_corner_balacc"])
        log(f"  ⚠ [{tag}] weak companion (subsample): in-corner "
            f"{companion['external_in_corner_balacc']:.4f} vs shipped "
            f"{scored['external']['in_corner']['balanced_accuracy']:.4f}; oracle "
            f"rel-mse {companion['oracle_relative_mse_k1']:.4f} vs shipped "
            f"{strong:.4f}")

    return {
        "budget_tag": tag,
        "budget_bytes": int(budget_bytes),
        "n_atoms_per_well": int(N),
        "byte_ledgers": ledgers,
        "atoms_persisted": persisted,
        "atom_selection": {s: sets[s].info for s in sets},
        "atom_sha256": {s: sets[s].sha256() for s in sets},
        "tuning": {"budget_available": int(cfg.tuning_trial_budget),
                   "trials_realised": len(grid), "selected": best, "trials": trials,
                   "selection_criterion":
                       "validation aggregate probe accuracy on IMAGES (⛔ B never enters)",
                   "stability": stability},
        "channel_bits": float(reads["B"]["bits_per_read"]),
        "scored": scored,
        "distinct": distinct,
        "read_error": read_err,
        "weak_companion": companion,
        "wall_clock_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 3. ⛔ The regression check: full, unrestricted ``knn_proto`` must NOT move
# ---------------------------------------------------------------------------


def knn_proto_regression(ctx: dict, cfg, seed: int, log: Callable = print) -> dict:
    """⛔ Acceptance item 4 — and the falsifier *"full knn_proto moved"*.

    ``knn_proto`` is R8's composed prototype embedding **decoded directly**: the arm
    this one is a *restriction* of.  It is re-run here on the identical rows, decoder
    and instrument, and its **probe** in-corner number must reproduce the published
    constant **bit-identically**.  ⛔ Probe-vs-probe: an external-vs-probe comparison
    would confound the instrument with the arm.
    """
    shape_idx = int(ctx["names"].index(COMPOSITIONAL_FACTOR))
    imgs = {rs: render(ctx["decoder"], ctx[f"Q_{rs}"], ctx["feat_shape"],
                       int(cfg.eval_batch)) for rs in REQUEST_SETS}
    probe_ic = in_corner(ctx["probe"], imgs["B"], imgs["MS"], ctx["lab_B"], shape_idx,
                         int(cfg.eval_batch))
    ext_ic = in_corner(ctx["external"], imgs["B"], imgs["MS"], ctx["lab_B"], shape_idx,
                       int(cfg.eval_batch))
    refs = list(cfg.knn_proto_probe_reference_per_seed)
    m = float(probe_ic["balanced_accuracy"])
    out = {
        "arm": "knn_proto (full, unrestricted)",
        "role": ("⛔ THE REGRESSION CHECK, ⛔ not a competitor: this is the arm the "
                 "atom-mean lookup is a restriction of. If it moved, I changed "
                 "something out of scope and the comparison is VOID."),
        "probe_in_corner_balacc": m,
        "external_in_corner_balacc": float(ext_ic["balanced_accuracy"]),
        "published_probe": float(refs[seed]) if seed < len(refs) else None,
        "distinct": distinct_counts(ctx["external"], ctx["Q_B"], imgs["B"],
                                    int(cfg.eval_batch)),
        "retention_ladder": {
            rs: retention_ladder(ctx["probe"], imgs[rs], ctx[f"lab_{rs}"], ctx["names"],
                                 int(cfg.eval_batch)) for rs in imgs
        },
        "channel": ("UNBOUNDED — the whole D-float vector reaches the decoder. ⛔ Never "
                    "compared in the same column as a bounded k*log2(N)-bit read."),
    }
    if out["published_probe"] is not None:
        out["delta"] = m - out["published_probe"]
        out["bit_identical"] = bool(m == out["published_probe"])
        log(f"  ⭐ [knn_proto REGRESSION] probe in-corner {m!r} vs published "
            f"{out['published_probe']!r} -> Δ = {out['delta']:+.17g}, "
            f"bit_identical={out['bit_identical']}")
        if bool(cfg.require_knn_proto_bit_identical) and not out["bit_identical"]:
            raise AssertionError(
                f"⛔ FULL knn_proto MOVED: {m!r} vs published {out['published_probe']!r} "
                f"(Δ {out['delta']:+.3e}). Something out of scope changed and the "
                "comparison this spoke exists to make is VOID. ⛔ Stop and report."
            )
    else:
        out["NOT_RUN"] = f"no published constant for seed {seed}"
    return out


# ---------------------------------------------------------------------------
# 4. One seed
# ---------------------------------------------------------------------------


def run_seed(dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
             budgets: Optional[list] = None, log: Callable = print) -> dict:
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    cfg = gen_atom_mean_config(config)
    sdir = Path(out_dir) / f"seed_{int(seed)}"
    sdir.mkdir(parents=True, exist_ok=True)
    man_path = sdir / "atom_manifest.json"
    manifest = {}
    if man_path.exists():
        with open(man_path) as fh:
            manifest = json.load(fh)
    condition = condition_of(config)
    bdir = bridge_dir(config, dataset, pretext, seed)
    tlog(f"seed {seed} · condition={condition} · frozen bridge {bdir}")

    # ---- ⛔ the two guards, FIRST: the decoder and the instrument -----------
    config.gen_compose.claude_root = str(_root(config))
    config.gen_compose.decoder_source = str(cfg.decoder_source)
    config.gen_compose.winner_root = cfg.winner_root
    config.gen_compose.winner_decoder_pattern = str(cfg.winner_decoder_pattern)
    config.gen_compose.expected_decoder_recipe = str(cfg.expected_decoder_recipe)
    # ⛔ the classifier loader reads GenCellConfig's knobs; forward ours so a
    # gen_atom_mean override can never be silently dead (a config knob that does
    # nothing is how a test stops testing anything).
    config.gen_cell.claude_root = str(_root(config))
    config.gen_cell.classifier_root = cfg.classifier_root
    config.gen_cell.classifier_checkpoint_pattern = str(cfg.classifier_checkpoint_pattern)
    config.gen_cell.classifier_manifest_pattern = str(cfg.classifier_manifest_pattern)
    decoder, dec_info, bman = resolve_decoder(config, bdir, seed, log=tlog)
    if str(dec_info["decoder_recipe"]) != str(cfg.expected_decoder_recipe):
        raise AssertionError(
            f"⛔ decoder recipe is {dec_info['decoder_recipe']!r}, not "
            f"{cfg.expected_decoder_recipe!r} — the 'baseline' recipe FAILED the gate "
            "and a run decoded through it is VOID"
        )
    external, clf_info = load_external_classifier(config, seed, log=tlog)
    probe = load_model(bdir / "probe.pkl")
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest hash")
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")

    # ---- rows ---------------------------------------------------------------
    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_store_rows(split, config, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    n_fit = int(min(int(cfg.n_fit), pool.size))
    n_val = int(min(int(cfg.n_val), max(pool.size - n_fit, 0)))
    row_sets = {
        "B": rows["eval_b"],
        "MS": matched_seen_rows(dataset, split, values, rows["store_train"]),
        "A": rows["eval_a"][: int(cfg.n_eval_a)],
        "fit": pool[:n_fit],
        "val": pool[n_fit : n_fit + n_val],
    }
    tlog(f"rows: store_train={pool.size} B={row_sets['B'].size} MS={row_sets['MS'].size} "
         f"A={row_sets['A'].size} fit={n_fit} val={n_val} "
         f"(⛔ fit/val are GenBaselinesConfig's 32,256/512, ⛔ never my own)")

    # ---- the frozen cache ---------------------------------------------------
    crows, feats_cache, cman = load_feature_cache(bdir,
                                                  verify=bool(cfg.verify_cache_checksum))
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    config.gen_store.encode_chunk = int(cfg.encode_chunk)
    reenc = _assert_cache_reencodes(dataset, config, encoder, crows, feats_cache, cman,
                                    log=tlog)
    tlog(f"cache {cman['n_rows']} rows (scope={cman['scope']}), D={cman['D']}; "
         f"re-encode check bit-identical on {reenc.get('n_checked')} rows")

    def gather(r):
        return gather_features(dataset, config, encoder, crows, feats_cache, cman, r,
                               log=tlog)

    X_store = gather(pool)
    X = {k: gather(v) for k, v in row_sets.items()}
    lab = {k: factor_labels(grid, v) for k, v in row_sets.items()}

    # ---- the wells: ⭐ the SAME partition the store (and the landscape) uses --
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    K, D = int(plan["K"]), int(np.prod(feat_shape))
    config.gen_dream.grouping_source = str(cfg.grouping_source)
    labels_vec, ginfo = resolve_grouping(plan, X_store, config, seed,
                                         bman["encoder_param_hash"], log=tlog)
    wc = well_channels_from_labels(labels_vec, K)
    store_budget = int(plan["n_modes"]) * D * 4
    tlog(f"wells K={K} sizes={[len(c) for c in wc]} (Σ={sum(len(c) for c in wc)}=D) · "
         f"the store's own ledger is n_modes({plan['n_modes']})·D·4 = "
         f"{store_budget:,} B")

    # ---- R8's declared request embedding, fitted on the FIT rows only -------
    protos = FactorPrototypes.fit(grid, X["fit"], row_sets["fit"])
    req_dicts = {k: requests_from_rows(grid, row_sets[k])
                 for k in list(REQUEST_SETS) + ["val"]}
    Q = {k: np.asarray(protos.embed_many(req_dicts[k]), np.float32) for k in req_dicts}

    normalisers = {m: fit_normaliser(X["fit"], m) for m in set(list(cfg.normalisers))}
    ctx = {
        "names": names, "cardinalities": cards, "feat_shape": feat_shape, "D": D,
        "decoder": decoder, "probe": probe, "external": external,
        "well_channels": wc, "X_store": X_store,
        "normalisers": normalisers,
        **{f"X_{k}": X[k] for k in X},
        **{f"lab_{k}": lab[k] for k in lab},
        **{f"Q_{k}": Q[k] for k in Q},
    }

    # ---- ⭐ the decoder ceiling (real features) -----------------------------
    shape_idx = int(names.index(COMPOSITIONAL_FACTOR))
    cimg = {rs: render(decoder, X[rs], feat_shape, cfg.eval_batch) for rs in REQUEST_SETS}
    ceiling = {
        inst_name: {
            "in_corner": in_corner(inst, cimg["B"], cimg["MS"], lab["B"], shape_idx,
                                   cfg.eval_batch),
            "retention_ladder": {rs: retention_ladder(inst, cimg[rs], lab[rs], names,
                                                      cfg.eval_batch) for rs in cimg},
        }
        for inst_name, inst in (("external", external), ("probe", probe))
    }
    del cimg
    tlog(f"⭐ DECODER ceiling on IMAGES: external in-corner "
         f"{ceiling['external']['in_corner']['balanced_accuracy']:.4f} · probe "
         f"{ceiling['probe']['in_corner']['balanced_accuracy']:.4f}")

    manifest.update({
        "schema": "chlu-gen-atom-mean/1",
        "dataset": dataset, "pretext": pretext, "condition": condition,
        "seed": int(seed),
        "path": ("request → R8 prototype embedding → nearest STORED ATOMS (per well) "
                 "→ mean → THE frozen decoder → image → THE external classifier"),
        "DIAL_DECLARATION": {
            "dial": "⛔ none — this module is a CONTROL",
            "laundering_control": "⭐ this module IS the control",
            "falsifies": ("gen-dynamical-read's claim, if a plain lookup over the same "
                          "atoms does the same job"),
            "my_own_falsifier": "⭐ a control that is WEAKER than it should be",
            "NO_CLU_NUMBER": "⛔ no CLU number and no verdict is produced here",
        },
        "bridge_dir": str(bdir),
        "decoder": dec_info,
        "external_classifier": clf_info,
        "preconditions": manifest.get("preconditions"),
        "provenance": {
            "encoder_param_hash": bman["encoder_param_hash"],
            "probe_param_hash": bman["probe_param_hash"],
            "feature_cache_sha256": cman["features_sha256"],
            "feature_cache_scope": cman["scope"],
            "cache_reencode_check": reenc,
            "store_plan_path": str(store_plan_path(config, dataset, pretext)),
            "atoms_are": ("⭐ ACTUAL TRAIN DATA POINTS from store_train (the SAME "
                          "stored set the landscape will use), ⛔ never k-means centres"),
        },
        "K": K, "D": D, "n_modes": int(plan["n_modes"]),
        "store_budget_bytes": store_budget,
        "grouping": ginfo, "well_sizes": [len(c) for c in wc],
        "rows": {k: int(v.size) for k, v in row_sets.items()},
        "row_set_diagnostics": {
            k: {
                "n": int(v.size),
                "n_values_realised": {str(nm): int(np.unique(lab[k][:, j]).size)
                                      for j, nm in enumerate(names)},
                "feature_variance": float(np.mean(
                    (X[k] - X[k].mean(axis=0, keepdims=True)) ** 2)),
            } for k, v in row_sets.items()
        },
        "row_construction_note": (
            "⚠ ``fit`` = pool[:n_fit] and ``val`` = pool[n_fit:n_fit+n_val] where "
            "``pool`` is SORTED — this is select_baseline_rows' shipped construction "
            "and is INHERITED VERBATIM here (⛔ never re-chosen on my own budget). "
            "⭐ Consequence, measured above: ``val`` is a contiguous TAIL slice of the "
            "factor grid and is NOT exchangeable with ``fit``; any variance-normalised "
            "statistic on it (relative read error) is distorted, which is why the "
            "PRIMARY read-error rows are ``A`` (``eval_a``, excluded from the pool)."),
        "prototype_byte_ledger": protos.byte_ledger(),
        "ceiling": ceiling,
        "budgets": manifest.get("budgets", {}),
        "NOT_RUN": manifest.get("NOT_RUN", {}),
    })
    _write_json(manifest, man_path)

    # ---- ⛔ the regression check FIRST: if knn_proto moved, nothing else counts
    if "knn_proto_regression" not in manifest:
        manifest["knn_proto_regression"] = knn_proto_regression(ctx, cfg, int(seed),
                                                                log=tlog)
        _write_json(manifest, man_path)

    # ---- the budgets --------------------------------------------------------
    want = list(budgets) if budgets else sorted(cfg.budgets)
    for tag in want:
        if tag in manifest["budgets"]:
            tlog(f"  [{tag}] already on disk for this seed — skipped")
            continue
        if tag not in cfg.budgets:
            raise ValueError(f"unknown budget {tag!r} (have {sorted(cfg.budgets)})")
        try:
            manifest["budgets"][tag] = run_budget(tag, int(cfg.budgets[tag]), ctx, cfg,
                                                  int(seed), sdir, log=tlog)
            manifest.get("NOT_RUN", {}).pop(tag, None)
        except Exception as exc:  # ⛔ NOT RUN, never a zero and never a silent gap
            manifest["NOT_RUN"][tag] = {"reason": f"{type(exc).__name__}: {exc}"}
            tlog(f"  ⛔ [{tag}] NOT RUN — {type(exc).__name__}: {exc}")
        _write_json(manifest, man_path)

    # ---- ⭐ the exponent, across budgets ------------------------------------
    manifest["exponent"] = exponent_across_budgets(manifest, cfg)
    manifest["wall_clock_s"] = float(time.time() - t0)
    _write_json(manifest, man_path)
    tlog(f"seed {seed} done in {manifest['wall_clock_s']:.0f}s")
    return manifest


def exponent_across_budgets(manifest: dict, cfg) -> dict:
    """⭐ **The claim is the exponent** — fitted over the budgets that ran."""
    out = {}
    primary = str(cfg.read_error_primary)
    for leg in ("oracle_feature_addressed/k1", "oracle_feature_addressed/k_selected",
                "request_addressed/k1", "request_addressed/k_selected",
                "centroid_control/k1"):
        pts = []
        for _tag, row in sorted((manifest.get("budgets") or {}).items()):
            e = ((row.get("read_error") or {}).get(primary) or {}).get(leg)
            if e:
                pts.append((float(row["n_atoms_per_well"]), float(e["relative_mse"])))
        out[leg] = fitted_exponent(pts)
    out["rows"] = primary
    a = float(cfg.centroid_anchor_relative_mse)
    n0 = float(cfg.centroid_anchor_n)
    pr = float(cfg.centroid_implied_pr)
    out["centroid_quantiser_law"] = {
        "anchor_relative_mse": a, "anchor_n": n0, "implied_pr": pr,
        "slope": -2.0 / pr,
        "predicted": {
            str(int(row["n_atoms_per_well"])):
                a * (float(row["n_atoms_per_well"]) / n0) ** (-2.0 / pr)
            for row in (manifest.get("budgets") or {}).values()
        },
        "source": ("gen-dynamical-read/PREREG.md §2 — the CENTROID codebook's measured "
                   "anchor. ⛔ Rendered for comparison only; it is NOT a measurement "
                   "made by this module."),
    }
    out["what"] = ("⭐ d ln(relative_mse) / d ln(N_atoms). The dynamical read's case is "
                   "a BETTER EXPONENT at matched bytes; this is the exponent it has to "
                   "beat. ⛔ A win on level without a better exponent is not the claim.")
    return out


# ---------------------------------------------------------------------------
# 5. Aggregate
# ---------------------------------------------------------------------------


def aggregate_runs(run_dirs, log: Callable = print) -> dict:
    mans = []
    for d in run_dirs:
        p = Path(d) / "atom_manifest.json"
        if p.exists():
            with open(p) as fh:
                mans.append(json.load(fh))
    if not mans:
        return {"seeds_present": [], "note": "⛔ nothing on disk — every row is NOT RUN"}
    seeds = sorted(int(m["seed"]) for m in mans)
    tags = sorted({t for m in mans for t in m.get("budgets", {})})
    table = {}
    for tag in tags:
        rows = [m["budgets"][tag] for m in mans if tag in m.get("budgets", {})]
        present = [int(m["seed"]) for m in mans if tag in m.get("budgets", {})]
        names = list((rows[0]["scored"]["probe"]["retention_ladder"]["A"]
                      ["per_factor"]).keys())
        table[tag] = {
            "seeds": present, "n_seeds": len(present),
            "budget_bytes": int(rows[0]["budget_bytes"]),
            "n_atoms_per_well": int(rows[0]["n_atoms_per_well"]),
            "byte_match_exact": bool(all(
                r["byte_ledgers"][r["tuning"]["selected"]["scope"]]["matched_exactly"]
                for r in rows)),
            "selected": [r["tuning"]["selected"] for r in rows],
            "channel_bits": _ms([r["channel_bits"] for r in rows]),
            "external_in_corner_balacc": _ms(
                [r["scored"]["external"]["in_corner"]["balanced_accuracy"] for r in rows]),
            "probe_in_corner_balacc": _ms(
                [r["scored"]["probe"]["in_corner"]["balanced_accuracy"] for r in rows]),
            "retention_ladder": {
                rs: {
                    "aggregate": _ms([r["scored"]["probe"]["retention_ladder"][rs]
                                      ["aggregate"] for r in rows]),
                    "per_factor": {
                        nm: {
                            "accuracy": _ms([r["scored"]["probe"]["retention_ladder"][rs]
                                             ["per_factor"][nm]["accuracy"]
                                             for r in rows]),
                            # ⛔ R31: never a per-factor row without this
                            "n_values_realised": sorted({
                                int(r["scored"]["probe"]["retention_ladder"][rs]
                                    ["per_factor"][nm]["n_values_realised"])
                                for r in rows}),
                            "applicable": bool(all(
                                r["scored"]["probe"]["retention_ladder"][rs]
                                ["per_factor"][nm]["n_values_realised"] > 1
                                for r in rows)),
                        } for nm in names
                    },
                } for rs in ("B", "MS", "A")
                if all(rs in r["scored"]["probe"]["retention_ladder"] for r in rows)
            },
            "distinct_atom_configurations_B": _ms(
                [r["distinct"]["B"]["atom_configurations"] for r in rows]),
            "distinct_feature_vectors_B": _ms(
                [r["distinct"]["B"]["feature_vectors"] for r in rows]),
            "distinct_factor_tuples_B": _ms(
                [r["distinct"]["B"]["factor_tuples"] for r in rows]),
            "n_B": int(rows[0]["distinct"]["B"]["n"]),
            "read_error": {
                rk: {
                    leg: _ms([r["read_error"][rk][leg]["relative_mse"] for r in rows])
                    for leg in ("oracle_feature_addressed/k1",
                                "oracle_feature_addressed/k_selected",
                                "request_addressed/k1", "request_addressed/k_selected",
                                "centroid_control/k1")
                    if all(leg in r["read_error"].get(rk, {}) for r in rows)
                }
                for rk in sorted({k for r in rows for k in r["read_error"]
                                  if isinstance(r["read_error"][k], dict)
                                  and "oracle_feature_addressed/k1"
                                  in r["read_error"][k]})
            },
            "medoid_penalty": {
                rk: _ms([r["read_error"][rk]["medoid_penalty"]["ratio"] for r in rows])
                for rk in sorted({k for r in rows for k in r["read_error"]
                                  if isinstance(r["read_error"][k], dict)
                                  and "medoid_penalty" in r["read_error"][k]})
            },
            "weak_companion": {
                "kmedoid_stronger_on_read_error": [
                    bool(r["weak_companion"].get("kmedoid_is_stronger_on_read_error"))
                    for r in rows],
                "kmedoid_stronger_on_in_corner": [
                    bool(r["weak_companion"].get("kmedoid_is_stronger_on_in_corner"))
                    for r in rows],
                "subsample_in_corner": _ms(
                    [r["weak_companion"].get("external_in_corner_balacc", float("nan"))
                     for r in rows]),
            },
        }
        missing = sorted(set(seeds) - set(present))
        if missing:
            table[tag]["NOT_RUN_seeds"] = missing
    regs = [m.get("knn_proto_regression", {}) for m in mans]
    out = {
        "schema": "chlu-gen-atom-mean-aggregate/1",
        "seeds_present": seeds,
        "budgets": table,
        "knn_proto_regression": {
            "per_seed": {int(m["seed"]): m.get("knn_proto_regression", {})
                         for m in mans},
            "bit_identical_all_seeds": bool(
                all(r.get("bit_identical") for r in regs) and len(regs) == len(mans)),
            "probe_in_corner_balacc": _ms(
                [r["probe_in_corner_balacc"] for r in regs
                 if "probe_in_corner_balacc" in r]),
            "role": ("⛔ the REGRESSION CHECK. ⭐ If this is not bit-identical on every "
                     "seed, something out of scope moved and the comparison is VOID."),
        },
        "exponent": {int(m["seed"]): m.get("exponent") for m in mans},
        "ceiling": {
            "external_in_corner_balacc": _ms(
                [m["ceiling"]["external"]["in_corner"]["balanced_accuracy"]
                 for m in mans]),
            "probe_in_corner_balacc": _ms(
                [m["ceiling"]["probe"]["in_corner"]["balanced_accuracy"] for m in mans]),
        },
        "NOT_RUN": {int(m["seed"]): m.get("NOT_RUN", {}) for m in mans},
        "honesty": [
            "⛔ This module is a CONTROL. No CLU number, no dial verdict, no kill rule.",
            "⭐ Its own falsifier is a control that is WEAKER than it should be — which "
            "is why the weak (subsample) companion runs beside the shipped k-medoid "
            "atoms on every budget.",
            "⛔ knn_proto is the REGRESSION CHECK, never a competitor: the atom-mean "
            "arm is a RESTRICTION of it and cannot beat an unbounded channel.",
            "⛔ No per-factor number above is quotable without its n_values_realised.",
        ],
    }
    # ⭐ the exponent, pooled over seeds
    out["exponent_pooled"] = {}
    for leg in ("oracle_feature_addressed/k1", "oracle_feature_addressed/k_selected",
                "request_addressed/k1", "request_addressed/k_selected",
                "centroid_control/k1"):
        sl = [m["exponent"][leg]["slope"] for m in mans
              if np.isfinite(m.get("exponent", {}).get(leg, {}).get("slope", np.nan))]
        out["exponent_pooled"][leg] = {
            **_ms(sl),
            "implied_pr": _ms([-2.0 / s for s in sl if s != 0]),
            "n_seeds": len(sl),
        }
    log(f"⭐ aggregate: {len(seeds)} seeds, budgets {tags}; knn_proto bit-identical on "
        f"all seeds = {out['knn_proto_regression']['bit_identical_all_seeds']}")
    return out


# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------


def _plot(agg: dict, path: Path) -> None:
    tags = sorted(agg["budgets"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    x = np.arange(len(tags))
    ext = [agg["budgets"][t]["external_in_corner_balacc"]["mean"] for t in tags]
    exs = [agg["budgets"][t]["external_in_corner_balacc"]["std"] for t in tags]
    sub = [agg["budgets"][t]["weak_companion"]["subsample_in_corner"]["mean"]
           for t in tags]
    axes[0].bar(x - 0.2, ext, 0.4, yerr=exs, capsize=3, color="#2b6cb0",
                label="atom-mean (k-medoid, shipped)")
    axes[0].bar(x + 0.2, sub, 0.4, color="#a0aec0", label="⚠ weak companion (subsample)")
    axes[0].axhline(0.5, ls="--", c="k", lw=1, label="chance")
    kp = agg["knn_proto_regression"]["probe_in_corner_balacc"]["mean"]
    axes[0].axhline(kp, ls=":", c="g", lw=1.2, label="knn_proto (unbounded channel)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{t}\n{agg['budgets'][t]['n_atoms_per_well']} atoms/well"
                             for t in tags])
    axes[0].set_ylabel("in-corner `shape` balanced accuracy")
    axes[0].set_title("the control, at matched bytes")
    axes[0].legend(fontsize=7)

    n = [agg["budgets"][t]["n_atoms_per_well"] for t in tags]
    rk = sorted(agg["budgets"][tags[0]]["read_error"])[0]
    for leg, c in (("oracle_feature_addressed/k1", "#c05621"),
                   ("oracle_feature_addressed/k_selected", "#2f855a"),
                   ("request_addressed/k_selected", "#805ad5"),
                   ("centroid_control/k1", "#718096")):
        if leg not in agg["budgets"][tags[0]]["read_error"][rk]:
            continue
        y = [agg["budgets"][t]["read_error"][rk][leg]["mean"] for t in tags]
        e = [agg["budgets"][t]["read_error"][rk][leg]["std"] for t in tags]
        axes[1].errorbar(n, y, yerr=e, marker="o", color=c,
                         ls=":" if leg.startswith("centroid") else "-",
                         label=f"{leg} [{rk}]", capsize=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("atoms per well N (bytes = N·D·4)")
    axes[1].set_ylabel("relative read error (MSE / Var)")
    axes[1].set_title("⭐ the EXPONENT is the claim")
    axes[1].legend(fontsize=6)

    dc = [agg["budgets"][t]["distinct_atom_configurations_B"]["mean"] for t in tags]
    ds = [agg["budgets"][t]["distinct_atom_configurations_B"]["std"] for t in tags]
    axes[2].bar(x, dc, 0.55, yerr=ds, capsize=3, color="#c05621")
    axes[2].axhline(agg["budgets"][tags[0]]["n_B"], ls="--", c="k", lw=1,
                    label=f"all {agg['budgets'][tags[0]]['n_B']} requests distinct")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tags)
    axes[2].set_ylabel("distinct atom configurations on B")
    axes[2].set_title("the column an accuracy must never appear without")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Entry points
# ---------------------------------------------------------------------------


def run_experiment_gen_atom_mean(config: Optional[CHLUConfig] = None,
                                 quick: bool = False, out_dir: Optional[str] = None,
                                 stage: str = "atoms", seeds: Optional[list] = None,
                                 budgets: Optional[list] = None,
                                 log: Callable = print) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = gen_atom_mean_config(config)
    dataset, pretext = str(cfg.dataset), str(cfg.pretext)
    out = Path(out_dir) if out_dir else (
        _root(config) / "outputs" / "gen-atom-mean-control" / "runs"
        / f"{dataset}_{pretext}")
    out.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in (seeds if seeds is not None else cfg.seeds)]

    if stage in ("atoms", "all"):
        pre = check_preconditions(config, log=log)
        for s in seed_list:
            man = run_seed(dataset, pretext, s, out, config, budgets=budgets, log=log)
            man["preconditions"] = pre
            _write_json(man, out / f"seed_{s}" / "atom_manifest.json")
    if stage in ("aggregate", "all"):
        # ⛔ every seed on disk, not only the ones this invocation asked for: a repair
        # run must not silently overwrite a 3-seed table with a 1-seed one.
        on_disk = sorted(int(d.name.split("_")[1]) for d in out.glob("seed_*")
                         if d.is_dir() and (d / "atom_manifest.json").exists())
        agg = aggregate_runs([out / f"seed_{s}"
                              for s in sorted(set(seed_list) | set(on_disk))], log=log)
        agg["seeds_requested_this_invocation"] = seed_list
        agg["seeds_not_run"] = sorted(set(seed_list) - set(agg.get("seeds_present", [])))
        _write_json(agg, out / "atom_results.json")
        if cfg.make_plots and agg.get("budgets"):
            try:
                _plot(agg, out / "atom_table.png")
            except Exception as exc:  # a plot never loses a result
                log(f"  ⚠ plot failed (results are already on disk): {exc}")
        return agg
    return {"stage": stage, "out": str(out)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None)
    p.add_argument("--stage", default="atoms",
                   choices=list(STAGES) + ["all"] + list(NOT_MINE))
    p.add_argument("--seed", type=int, action="append", default=None)
    p.add_argument("--budget", action="append", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--config", default=None)
    return p


def _config_from_args(args) -> CHLUConfig:
    if args.config:
        from ..config import load_config

        config = load_config(Path(args.config))
    else:
        config = get_default_config()
    cfg = gen_atom_mean_config(config)
    if args.dataset:
        cfg.dataset = str(args.dataset)
    if args.pretext:
        cfg.pretext = str(args.pretext)
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.stage in NOT_MINE:
        print(f"⛔ stage={args.stage!r} belongs to another module — this one runs "
              f"{STAGES}. Refusing rather than writing atom artefacts into another "
              "stage's provenance.")
        return 2
    config = _config_from_args(args)
    cfg = gen_atom_mean_config(config)
    out = args.out or os.environ.get("CHLU_GEN_ATOM_OUT")
    t0 = time.time()
    print(f"=== gen-atom-mean-control: {cfg.dataset}/{cfg.pretext} "
          f"stage={args.stage} quick={args.quick} ===")
    print("⛔ DIAL: none — this is a CONTROL. No CLU number, no verdict.")
    run_experiment_gen_atom_mean(config, quick=bool(args.quick), out_dir=out,
                                 stage=args.stage, seeds=args.seed,
                                 budgets=args.budget)
    print(f"=== done in {time.time() - t0:.0f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
