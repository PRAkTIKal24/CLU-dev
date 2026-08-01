"""⭐ THE MATCHED-CAPACITY ORGANIZER AUDIT (C2W5 ``orgdiv-null-arms``).

Harness for :mod:`chlu.core.null_arms`. It runs the five registered null arms
(N1–N5) against ``orgdiv-cat-test``'s **frozen** interfaces and answers the Hub's
re-scoped question:

> **Does ANY matched-capacity organizer clear ``chance + 0.05`` on the rule-4-valid
> unseen split?**

Two decision-grade outcomes were registered before this file ran
(``.claude/outputs/orgdiv-null-arms/PREREG.md``):

* **none clears** ⇒ the family is refuted for every organizer class measured —
  cheaper and stronger than a tier-ii null, and the family (not the physics) is the
  first fix;
* **any clears — especially N1** (identical store parameterisation, non-physics
  training) ⇒ the family is solvable *within the same landscape class* and the
  cat-test's K5 kill becomes attributable to the physics write/read specifically.
  Per the Hub addendum (b), **N1's score is then the revival target** any tuned
  physics arm must beat.

Stages, in the order they run:

1. :func:`stage_grid` — the **full registered tuning grid** (prereg §4.3: >= 5
   optimiser points x 3 capacity points x 3 seeds per arm) on the SEEN split, with
   a held-out-from-seen validation slice. ⛔ Nothing here ever sees ``Q_unseen``.
2. :func:`stage_score` — the selected config per arm, refit on all of SEEN, scored
   on ``Q_unseen`` through the FROZEN reader class, 5 seeds, plus the shuffle-phi
   laundering control and the per-arm byte/compute ledger. ``max_arm`` is the
   **mechanical max over arms x readers**, computed, never estimated.
3. :func:`stage_ceiling` — the declared **out-of-class** phi-decodability
   diagnostic (⛔ reported, never scored as an arm).
4. :func:`stage_oracle` — N3 fitted on the physics arm's own assignments (T5.2
   rider (i)) + the F5 assignment-agreement number.

⛔ **Claim form (prereg §2.6):** no well, code or atom is named semantically here.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import jax
import numpy as np

from chlu.core.factored_store import (
    CatTestConfig,
    build_family,
    build_phi,
    chance_accuracy,
    exact_set_accuracy,
    fit_readers,
    occupancy,
    place_wells,
    reader_bytes,
    score_curve,
    score_reader,
)
from chlu.core.null_arms import (
    ARMS,
    NullArmGrid,
    arm_ledger,
    launch_points,
    n1_apply,
    n1_gradient_placed,
    n2_vq,
    n3_static_geometric,
    n4_keys,
    n4_knn,
    n5_titans,
    phi_decodability_ceiling,
    read_flops,
    shuffle_launches,
)

__all__ = ["run_null_arms", "stage_grid", "stage_score", "stage_ceiling",
           "stage_oracle", "seed_setup"]


def _j(o):
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(obj), indent=2))
    return path


# ==========================================================================
# the per-seed setup — FROZEN family, FROZEN phi, FROZEN launch protocol
# ==========================================================================
def seed_setup(cfg: CatTestConfig, seed: int, n_val: int = 32) -> Dict[str, Any]:
    """Everything an arm is allowed to see, and nothing else.

    ⛔ The train/validation split is **inside SEEN**: ``n_val`` of the ``K`` written
    items are held out of every arm's fit and are the only thing hyperparameters
    may be selected on. ``Q_unseen`` is constructed here and then **not touched**
    until :func:`stage_score`.
    """
    fam = build_family(cfg, seed=int(seed))
    phi = build_phi(cfg)
    ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
    anchors = place_wells(phi, cfg, sep=float(cfg.target_ds * ruler))
    ind_s = fam.indicator(fam.seen, cfg.n_wells)
    ind_u = fam.indicator(fam.unseen, cfg.n_wells)
    # ⭐ the SAME launch keys the physics arm used in `stage_arm` (PRNGKey(2000+seed)
    # for SEEN, fold_in(.,1) for unseen), so the arms start from bit-identical points.
    k = jax.random.PRNGKey(2000 + int(seed))
    q0_s = launch_points(phi, cfg, ind_s, k)
    q0_u = launch_points(phi, cfg, ind_u, jax.random.fold_in(k, 1))
    va, tr, k2val = _rule4_val_split(fam, int(cfg.f_subset), int(n_val), int(seed))
    return {"family": fam, "phi": phi, "anchors": anchors, "ind_s": ind_s,
            "ind_u": ind_u, "q0_s": q0_s, "q0_u": q0_u, "tr": tr, "va": va,
            "chance": chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol),
            "val_rule4": k2val, "seed": int(seed)}


def _rule4_val_split(fam, f_subset: int, n_val: int, seed: int):
    """⭐ A **rule-4-valid** validation split, carved out of SEEN.

    ⚠ The naive "hold out 32 of the 128 written items" split is *not* structurally
    the same problem as ``Q_unseen``: two written items can share ``F-1`` wells, so
    a validation row may sit right next to a training row and selection would then
    reward near-neighbour interpolation — exactly the ability rule 4 exists to
    exclude. We therefore hold out only rows satisfying ``|A_val & A_train| <=
    F-2`` against **every** retained training row, which makes the selection split
    structurally identical to the scoring split. The achieved fraction is returned
    and reported; if too few rows qualify the remainder is filled at random and
    that is recorded rather than hidden.
    """
    seen = np.asarray(fam.seen)
    K = len(seen)
    ind = np.zeros((K, int(seen.max()) + 1), dtype=np.int32)
    np.put_along_axis(ind, seen, 1, axis=1)
    ov = ind @ ind.T
    np.fill_diagonal(ov, 0)
    rng = np.random.default_rng(int(seed) + 4242)
    order = rng.permutation(K)
    va: List[int] = []
    for i in order:
        if len(va) >= int(n_val):
            break
        cand = set(va) | {int(i)}
        rest = np.array([j for j in range(K) if j not in cand], dtype=int)
        if len(rest) and ov[int(i), rest].max() <= f_subset - 2:
            va.append(int(i))
    n_strict = len(va)
    for i in order:  # fill (recorded, never hidden)
        if len(va) >= int(n_val):
            break
        if int(i) not in va:
            va.append(int(i))
    va_a = np.asarray(sorted(va), dtype=int)
    tr_a = np.asarray([j for j in range(K) if j not in set(va)], dtype=int)
    return va_a, tr_a, {"n_val": int(len(va_a)), "n_rule4_valid": int(n_strict),
                        "frac_rule4_valid": float(n_strict / max(len(va_a), 1))}


def _native(pred: np.ndarray, y: np.ndarray, tol: float) -> Dict[str, float]:
    """The arm's OWN read, scored: accuracy (primary) + MSE (the tie-break).

    ⚠ A diverged fit (some N5 momentum/lr corners do diverge) yields ``nan``. It is
    recorded as ``diverged`` and given a finite, enormous MSE so the *selection*
    ordering stays well-defined instead of silently propagating ``nan`` — a
    diverged config must lose, not disappear.
    """
    p = np.asarray(pred)
    mse = float(np.mean(np.sum((p - np.asarray(y)) ** 2, -1)))
    ok = bool(np.isfinite(mse))
    return {"acc": exact_set_accuracy(np.nan_to_num(p, nan=1e9), y, tol),
            "mse": mse if ok else 1e12, "diverged": (not ok)}


def _z_native(z: np.ndarray, cfg: CatTestConfig) -> np.ndarray:
    """``sum_p payload(z_p)`` — the objective every arm and the physics organizer
    is trained on (:func:`~chlu.core.factored_store.organize_physics`)."""
    return np.asarray(z)[:, :, int(cfg.addr_dim):].sum(1)


# ==========================================================================
# the registered grids, enumerated (prereg §4.3)
# ==========================================================================
def _grid_configs(arm: str, g: NullArmGrid) -> List[Dict[str, Any]]:
    if arm == "N1":
        return [{"lr": lr, "atoms_per_well": a, "tau": t, "init": i, "read": r}
                for lr, a, t, i, r in product(g.lrs, g.n1_atoms_per_well, g.n1_taus,
                                              g.n1_inits, ("soft", "hard"))]
    if arm == "N2":
        out = [{"variant": "kmeans", "n_codes": n, "payload_source": ps,
                "commitment": 0.0, "lr": 0.0}
               for n, ps in product(g.n2_codes, ("fitted", "written"))]
        out += [{"variant": "product_vq", "n_codes": n, "payload_source": "fitted",
                 "commitment": 0.0, "lr": 0.0} for n in g.n2_codes]
        out += [{"variant": "vq_ste", "n_codes": n, "payload_source": ps,
                 "commitment": b, "lr": lr}
                for n, b, lr, ps in product(g.n2_codes, g.n2_commitments, g.lrs,
                                            ("fitted",))]
        return out
    if arm == "N3":
        return [{"lr": lr, "level": lv, "payload_source": ps, "tau": t}
                for lr, lv, ps, t in product(g.lrs, g.n3_levels, g.n3_payloads,
                                             (0.05, 0.2))]
    if arm == "N4":
        return [{"key": kk, "k": k, "weight": w}
                for kk, k, w in product(g.n4_keys, g.n4_ks, g.n4_weights)]
    if arm == "N5":
        return [{"lr": lr, "hidden": h, "momentum": mo, "decay": dc, "gate": ga,
                 "chunk": ch}
                for lr, h, mo, dc, ga, ch in product(g.lrs, g.n5_hidden,
                                                     g.n5_momentum, g.n5_decay,
                                                     g.n5_gate, g.n5_chunk)]
    raise ValueError(arm)


# ==========================================================================
# fitting one configuration of one arm  (SEEN only)
# ==========================================================================
def _fit_arm(arm: str, conf: Dict[str, Any], S: Dict[str, Any],
             cfg: CatTestConfig, g: NullArmGrid, *, idx: np.ndarray,
             q0_override: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Fit ``arm`` at ``conf`` on the SEEN rows ``idx``. Returns a predictor.

    The returned dict always carries ``predict(q0, ind) -> yhat`` (the arm's own
    read) and, for the arms that produce a latent, ``z(q0, ind) -> (B, P, dim)``
    plus ``codebook`` for reader R2.
    """
    fam, anchors = S["family"], S["anchors"]
    q0_all = S["q0_s"] if q0_override is None else q0_override
    q0_tr, y_tr = q0_all[idx], fam.y_seen[idx]
    seed = S["seed"]

    if arm == "N1":
        fit = n1_gradient_placed(cfg, fam, anchors, q0_tr, y_tr, lr=conf["lr"],
                                 tau=conf["tau"], init=conf["init"],
                                 steps=g.steps, seed=seed,
                                 atoms_per_well=conf["atoms_per_well"])
        hard = conf["read"] == "hard"
        z = lambda q0, ind=None: n1_apply(fit, q0, hard=hard)  # noqa: E731
        return {"z": z, "codebook": fit["codebook"], "ledger": fit["ledger"],
                "train": {"loss_first": fit["loss_first"],
                          "loss_last": fit["loss_last"]}}
    if arm == "N2":
        fit = n2_vq(cfg, fam, q0_tr, y_tr, variant=conf["variant"],
                    n_codes=conf["n_codes"], commitment=conf["commitment"],
                    lr=max(conf["lr"], 1e-6), steps=g.steps, seed=seed,
                    restarts=g.n2_restarts, payload_source=conf["payload_source"],
                    anchors=anchors)
        return {"z": lambda q0, ind=None: fit["apply"](q0),
                "codebook": fit["codebook"], "ledger": fit["ledger"],
                "assign": fit["assign"],
                "train": {k: fit[k] for k in ("loss_first", "loss_last")
                          if k in fit}}
    if arm == "N3":
        fit = n3_static_geometric(cfg, fam, anchors, q0_tr, y_tr,
                                  level=conf["level"], lr=conf["lr"],
                                  tau=conf["tau"], steps=g.steps, seed=seed,
                                  payload_source=conf["payload_source"])
        return {"z": lambda q0, ind=None: fit["apply"](q0),
                "codebook": fit["codebook"], "ledger": fit["ledger"],
                "assign": fit["assign"],
                "train": {"loss_first": fit["loss_first"],
                          "loss_last": fit["loss_last"]}}
    if arm == "N4":
        phi = S["phi"]
        keys_tr = n4_keys(conf["key"], phi, cfg, S["ind_s"][idx], q0_all[idx])

        def predict(q0, ind):
            kq = n4_keys(conf["key"], phi, cfg, ind, q0)
            return n4_knn(cfg, keys_tr, y_tr, kq, k=conf["k"],
                          weight=conf["weight"])

        return {"predict": predict,
                "ledger": arm_ledger("N4", cfg, n_params=0,
                                     n_state=int(len(idx) * (cfg.addr_dim
                                                             + cfg.payload_dim)),
                                     k=conf["k"], weight=conf["weight"],
                                     key_space=conf["key"],
                                     noiseless_key=bool(conf["key"] == "set_code"),
                                     read_flops=read_flops("N4", cfg)),
                "train": {}}
    if arm == "N5":
        phi = S["phi"]
        keys_tr = n4_keys("launch_mean", phi, cfg, S["ind_s"][idx], q0_all[idx])
        # the per-seed insertion order (rule 1), restricted to the rows in `idx`
        rank = np.empty(len(fam.seen), dtype=int)
        rank[np.asarray(fam.order)] = np.arange(len(fam.seen))
        order = np.argsort(rank[idx])
        fit = n5_titans(cfg, keys_tr, y_tr, hidden=conf["hidden"], lr=conf["lr"],
                        momentum=conf["momentum"], decay=conf["decay"],
                        gate=conf["gate"], chunk=conf["chunk"], passes=g.n5_passes,
                        pretrain_steps=g.n5_pretrain_steps, order=order, seed=seed)

        def predict(q0, ind):
            return fit["apply"](n4_keys("launch_mean", phi, cfg, ind, q0))

        return {"predict": predict, "ledger": fit["ledger"],
                "train": {k: fit[k] for k in ("pre_loss_first", "pre_loss_last",
                                              "stream_loss_first",
                                              "stream_loss_last")}}
    raise ValueError(arm)


def _arm_predict(fitted: Dict[str, Any], q0: np.ndarray, ind: np.ndarray,
                 cfg: CatTestConfig) -> np.ndarray:
    if "predict" in fitted:
        return fitted["predict"](q0, ind)
    return _z_native(fitted["z"](q0, ind), cfg)


# ==========================================================================
# STAGE 1 — the full registered tuning grid (SEEN only)
# ==========================================================================
def stage_grid(cfg: CatTestConfig, grid: NullArmGrid,
               arms: Sequence[str] = ARMS, out: Optional[Path] = None,
               seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """The registered budget, **computed** — every config, every tune seed.

    Selection statistic: the arm's **own read** on the held-out-from-seen
    validation slice — accuracy first, MSE as the tie-break. ⭐ Declared and
    registered in advance: it is *reader-independent*, so no reader is co-adapted
    to an arm through hyperparameter selection (FB4's lesson), and it is the
    objective the physics organizer itself is trained on.
    ⛔ ``Q_unseen`` is not read anywhere in this function.
    """
    seeds = tuple(grid.tune_seeds if seeds is None else seeds)
    res: Dict[str, Any] = {"seeds": list(seeds), "records": {}, "selected": {},
                           "n_configs": {}}
    setups = {s: seed_setup(cfg, s, n_val=grid.n_val) for s in seeds}
    for arm in arms:
        confs = _grid_configs(arm, grid)
        res["n_configs"][arm] = len(confs)
        recs = []
        t0 = time.time()
        for ci, conf in enumerate(confs):
            per_seed = []
            for s in seeds:
                S = setups[s]
                fam = S["family"]
                fitted = _fit_arm(arm, conf, S, cfg, grid, idx=S["tr"])
                pred = _arm_predict(fitted, S["q0_s"][S["va"]],
                                    S["ind_s"][S["va"]], cfg)
                sc = _native(pred, fam.y_seen[S["va"]], fam.tol)
                pred_tr = _arm_predict(fitted, S["q0_s"][S["tr"]],
                                       S["ind_s"][S["tr"]], cfg)
                sc_tr = _native(pred_tr, fam.y_seen[S["tr"]], fam.tol)
                per_seed.append({"seed": s, "val": sc, "train": sc_tr,
                                 "n_params": fitted["ledger"]["n_params"],
                                 "fit": fitted.get("train", {})})
            recs.append({"config": conf,
                         "val_acc": float(np.mean([p["val"]["acc"] for p in per_seed])),
                         "val_mse": float(np.mean([p["val"]["mse"] for p in per_seed])),
                         "train_acc": float(np.mean([p["train"]["acc"]
                                                     for p in per_seed])),
                         "train_mse": float(np.mean([p["train"]["mse"]
                                                     for p in per_seed])),
                         "n_params": per_seed[0]["n_params"],
                         "diverged": bool(any(p["val"]["diverged"]
                                              for p in per_seed)),
                         "per_seed": per_seed})
            if (ci + 1) % 20 == 0 or ci + 1 == len(confs):
                print(f"[grid:{arm}] {ci+1}/{len(confs)} "
                      f"best_val_acc={max(r['val_acc'] for r in recs):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        best = max(recs, key=lambda r: (r["val_acc"], -r["val_mse"]))
        res["records"][arm] = recs
        res["n_diverged"] = res.get("n_diverged", {})
        res["n_diverged"][arm] = int(sum(r["diverged"] for r in recs))
        res["selected"][arm] = {"config": best["config"], "val_acc": best["val_acc"],
                                "val_mse": best["val_mse"],
                                "train_acc": best["train_acc"],
                                "train_mse": best["train_mse"],
                                "n_params": best["n_params"],
                                "wall_s": round(time.time() - t0, 1)}
        print(f"[grid:{arm}] SELECTED {best['config']} val_acc={best['val_acc']:.4f} "
              f"val_mse={best['val_mse']:.4f} train_acc={best['train_acc']:.4f}",
              flush=True)
    if out:
        _dump(res, out / "stage_grid.json")
    return res


# ==========================================================================
# STAGE 2 — score the selected arms on Q_unseen through the FROZEN readers
# ==========================================================================
def stage_score(cfg: CatTestConfig, grid: NullArmGrid, selected: Dict[str, Any],
                arms: Sequence[str] = ARMS, out: Optional[Path] = None,
                seeds: Optional[Sequence[int]] = None,
                launder: bool = True) -> Dict[str, Any]:
    """5 seeds, the frozen reader class, ``max_arm`` by mechanical max.

    Every arm is **refit on all of SEEN** at its selected config (the tuning split
    is discarded), readers are fitted on SEEN, and only then is ``Q_unseen``
    scored. The shuffle-phi laundering control re-runs the whole pipeline with the
    launch blocks permuted across queries.
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    cells: List[Dict[str, Any]] = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=grid.n_val)
        fam = S["family"]
        allidx = np.arange(len(fam.seen))
        for arm in arms:
            t0 = time.time()
            conf = selected[arm]["config"]
            fitted = _fit_arm(arm, conf, S, cfg, grid, idx=allidx)
            row: Dict[str, Any] = {
                "seed": int(s), "arm": arm, "config": conf,
                "chance": S["chance"], "tol": float(fam.tol),
                "ledger": fitted["ledger"], "train": fitted.get("train", {}),
            }
            # -- the arm's own read ----------------------------------------
            pu = _arm_predict(fitted, S["q0_u"], S["ind_u"], cfg)
            ps = _arm_predict(fitted, S["q0_s"], S["ind_s"], cfg)
            row["native_unseen"] = _native(pu, fam.y_unseen, fam.tol)
            row["native_seen"] = _native(ps, fam.y_seen, fam.tol)
            row["native_curve"] = {
                f"x{mult:g}": exact_set_accuracy(pu, fam.y_unseen, fam.tol * mult)
                for mult in (0.25, 0.5, 1.0, 2.0, 4.0)}
            # -- through the FROZEN reader class ---------------------------
            if "z" in fitted:
                z_s = fitted["z"](S["q0_s"], S["ind_s"])
                z_u = fitted["z"](S["q0_u"], S["ind_u"])
                anc, pay = fitted["codebook"]
                rd = fit_readers(z_s, fam.y_seen, anchors=np.asarray(anc),
                                 well_payloads=np.asarray(pay), seed=int(s))
                row["readers_unseen"] = {k: score_reader(v, z_u, fam.y_unseen,
                                                         fam.tol)
                                         for k, v in rd.items()}
                row["readers_seen"] = {k: score_reader(v, z_s, fam.y_seen, fam.tol)
                                       for k, v in rd.items()}
                row["reader_curve"] = {k: score_curve(v, z_u, fam.y_unseen, fam.tol)
                                       for k, v in rd.items()}
                row["reader_params"] = reader_bytes(rd)
                if "assign" in fitted:
                    row["assign_unseen"] = np.asarray(
                        fitted["assign"](S["q0_u"])).astype(int).tolist()
            else:
                row["readers_unseen"] = {"native": row["native_unseen"]["acc"]}
                row["readers_seen"] = {"native": row["native_seen"]["acc"]}
                row["reader_params"] = {"native": 0}
            # -- the laundering control ------------------------------------
            if launder:
                q0_sh = shuffle_launches(S["q0_s"], s)
                f2 = _fit_arm(arm, conf, S, cfg, grid, idx=allidx,
                              q0_override=q0_sh)
                pl = _arm_predict(f2, shuffle_launches(S["q0_u"], s + 1),
                                  S["ind_u"], cfg)
                row["launder_shuffle_phi"] = _native(pl, fam.y_unseen, fam.tol)
            row["wall_s"] = round(time.time() - t0, 1)
            cells.append(row)
            print(f"[score] seed={s} {arm} native_unseen="
                  f"{row['native_unseen']['acc']:.4f} readers="
                  f"{ {k: round(v,4) for k,v in row['readers_unseen'].items()} } "
                  f"({row['wall_s']:.0f}s)", flush=True)

    # -- aggregate: mean, sd(ddof=1), 2SE; clears iff mean - 2SE > bar ------
    chance = float(np.mean([c["chance"] for c in cells]))
    bar = chance + 0.05
    agg: Dict[str, Any] = {}
    for arm in arms:
        sub = [c for c in cells if c["arm"] == arm]
        keys = sorted({k for c in sub for k in c["readers_unseen"]})
        per_reader = {}
        for k in keys:
            v = np.array([c["readers_unseen"][k] for c in sub if k in c["readers_unseen"]])
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            se2 = float(2 * sd / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            per_reader[k] = {"mean": float(v.mean()), "sd": sd, "two_se": se2,
                             "n": int(len(v)),
                             "clears": bool(v.mean() - (se2 if se2 == se2 else 0) > bar),
                             "per_seed": v.tolist()}
        nat = np.array([c["native_unseen"]["acc"] for c in sub])
        nat_s = np.array([c["native_seen"]["acc"] for c in sub])
        lau = np.array([c.get("launder_shuffle_phi", {}).get("acc", np.nan)
                        for c in sub])
        agg[arm] = {"per_reader": per_reader,
                    "native_unseen_mean": float(nat.mean()),
                    "native_unseen_two_se": float(2 * nat.std(ddof=1) / np.sqrt(len(nat)))
                    if len(nat) > 1 else float("nan"),
                    "native_seen_mean": float(nat_s.mean()),
                    "launder_mean": float(np.nanmean(lau)) if len(lau) else float("nan"),
                    "best": float(max([p["mean"] for p in per_reader.values()]
                                      + [float(nat.mean())])),
                    "n_params": sub[0]["ledger"]["n_params"],
                    "total_bytes": sub[0]["ledger"]["total_bytes"],
                    "read_flops": sub[0]["ledger"].get("read_flops", 0)}
    # ⭐ max_arm: the mechanical max over ALL arms and ALL readers
    readers_all = sorted({k for a in agg.values() for k in a["per_reader"]})
    max_arm_by_reader = {}
    for k in readers_all:
        pool = [(a["per_reader"][k]["mean"], arm) for arm, a in agg.items()
                if k in a["per_reader"]]
        if pool:
            v, arm = max(pool)
            max_arm_by_reader[k] = {"value": v, "arg": arm}
    best_overall = max((a["best"], arm) for arm, a in agg.items())
    # the protocol-pure max: excludes N4's noiseless-key (set_code) variant
    pure = []
    for arm, a in agg.items():
        conf = selected[arm]["config"]
        if arm == "N4" and conf.get("key") == "set_code":
            continue
        pure.append((a["best"], arm))
    res = {"cells": cells, "aggregate": agg, "chance": chance, "bar": bar,
           "max_arm_by_reader": max_arm_by_reader,
           "max_arm": {"value": best_overall[0], "arg": best_overall[1]},
           "max_arm_protocol_pure": ({"value": max(pure)[0], "arg": max(pure)[1]}
                                     if pure else None),
           "any_clears": bool(best_overall[0] > bar),
           "seeds": list(seeds)}
    if out:
        _dump(res, out / "stage_score.json")
    return res


# ==========================================================================
# STAGE 2b — ⭐ `null*` as a COMPUTED max over the ENTIRE registered grid
# ==========================================================================
def stage_gridmax(cfg: CatTestConfig, grid: NullArmGrid, arms: Sequence[str] = ARMS,
                  out: Optional[Path] = None,
                  seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """prereg §4.3: *"``null*`` = max over ALL arms AND their entire registered
    grid — **computed, not estimated**"*.

    ⛔ **This is an oracle-selected UPPER BOUND, not a selection protocol.** Every
    configuration in the registered grid is scored on ``Q_unseen`` and the maximum
    is reported. It may not be used as any arm's *score* (that is
    :func:`stage_score`, whose config was chosen on the seen-validation split
    alone); it exists so the audit's verdict reads *"no configuration in the
    registered grid clears"* rather than *"the configuration we picked did not"*.
    The arm's own read is used, because fitting the four readers 3 000 times is not
    affordable and the selected-config cells (which do run all four) show reader
    and native scores agreeing to the last digit.
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    setups = {s: seed_setup(cfg, s, n_val=grid.n_val) for s in seeds}
    res: Dict[str, Any] = {"seeds": list(seeds), "per_arm": {}}
    for arm in arms:
        confs = _grid_configs(arm, grid)
        t0, rows = time.time(), []
        for ci, conf in enumerate(confs):
            accs = []
            for s in seeds:
                S = setups[s]
                fam = S["family"]
                fitted = _fit_arm(arm, conf, S, cfg, grid,
                                  idx=np.arange(len(fam.seen)))
                pu = _arm_predict(fitted, S["q0_u"], S["ind_u"], cfg)
                accs.append(exact_set_accuracy(np.nan_to_num(pu, nan=1e9),
                                               fam.y_unseen, fam.tol))
            rows.append({"config": conf, "mean": float(np.mean(accs)),
                         "max_seed": float(np.max(accs)), "per_seed": accs})
            if (ci + 1) % 40 == 0 or ci + 1 == len(confs):
                print(f"[gridmax:{arm}] {ci+1}/{len(confs)} "
                      f"best={max(r['mean'] for r in rows):.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        best = max(rows, key=lambda r: r["mean"])
        res["per_arm"][arm] = {
            "n_configs": len(confs), "rows": rows,
            "best_mean": best["mean"], "best_config": best["config"],
            "best_single_seed": float(max(r["max_seed"] for r in rows)),
            "wall_s": round(time.time() - t0, 1)}
        print(f"[gridmax:{arm}] MAX over {len(confs)} configs = {best['mean']:.5f} "
              f"at {best['config']}", flush=True)
    chance = float(np.mean([S["chance"] for S in setups.values()]))
    tot = sum(v["n_configs"] for v in res["per_arm"].values())
    gm = max((v["best_mean"], a) for a, v in res["per_arm"].items())
    res.update({"chance": chance, "bar": chance + 0.05,
                "n_configs_total": tot,
                "null_star_gridmax": {"value": gm[0], "arg": gm[1]},
                "any_config_clears": bool(gm[0] > chance + 0.05)})
    if out:
        _dump(res, out / "stage_gridmax.json")
    return res


# ==========================================================================
# STAGE 2c — ⭐ THE MECHANISM: what the P designed launches can even address
# ==========================================================================
def stage_mechanism(cfg: CatTestConfig, grid: NullArmGrid, selected: Dict[str, Any],
                    arms: Sequence[str] = ARMS, out: Optional[Path] = None,
                    seeds: Optional[Sequence[int]] = None,
                    p_sweep: Sequence[int] = (4, 8, 16, 32, 64)) -> Dict[str, Any]:
    """Why every arm fails, measured rather than argued.

    Three statistics per query, for the **raw launch geometry** (no organizer at
    all) and for every arm that produces an assignment:

    * ``distinct`` — how many DISTINCT wells the ``P`` particles occupy. The target
      ``y(x) = sum_{j in A(x)} v_j`` needs ``F`` distinct wells; if the particles
      pile into fewer, the read cannot express the answer **no matter what the
      payloads are**. (This is ``orgdiv-cat-test`` §13 open question 1, answered.)
    * ``precision`` — fraction of particles landing in a well of ``A(x)``.
    * ``exact_set`` — fraction of queries whose distinct occupied set **equals**
      ``A(x)``. This is the arm's own ceiling on the metric.

    ⛔ The ``P``-sweep is a **declared OUT-OF-PROTOCOL diagnostic**: changing ``P``
    re-draws the launch offsets, so it is not a matched arm and never a score. It
    is here because the difference between "more particles fixes it" and "more
    particles does not" is the single most decision-relevant number for any
    revival.
    """
    from chlu.core.factored_store import build_phi as _bphi
    from chlu.core.null_arms import _with as _cfgwith

    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    F = int(cfg.f_subset)
    rows = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=grid.n_val)
        fam, anchors = S["family"], S["anchors"]

        def stats(assign, subsets=fam.unseen):
            a = np.asarray(assign)
            distinct = np.array([len(set(r.tolist())) for r in a])
            prec = np.array([np.isin(r, np.asarray(A)).mean()
                             for r, A in zip(a, subsets)])
            exact = np.array([set(r.tolist()) == set(np.asarray(A).tolist())
                              for r, A in zip(a, subsets)])
            return {"distinct_mean": float(distinct.mean()),
                    "distinct_ge_F": float((distinct >= F).mean()),
                    "precision": float(prec.mean()),
                    "exact_set": float(exact.mean())}

        row = {"seed": int(s), "F": F, "P": int(cfg.n_particles)}
        occ_raw = np.argmin(((S["q0_u"][:, :, None, : cfg.addr_dim]
                              - anchors[None, None]) ** 2).sum(-1), -1)
        row["raw_launch"] = stats(occ_raw)
        for arm in arms:
            if arm not in selected:
                continue
            fitted = _fit_arm(arm, selected[arm]["config"], S, cfg, grid,
                              idx=np.arange(len(fam.seen)))
            if "assign" not in fitted:
                continue
            row[arm] = stats(fitted["assign"](S["q0_u"]))
        # -- declared OUT-OF-PROTOCOL: does more fan-out fix it? -------------
        sweep = {}
        for P2 in p_sweep:
            c2 = _cfgwith(cfg, n_particles=int(P2))
            phi2 = _bphi(c2)
            q2 = launch_points(phi2, c2, S["ind_u"], jax.random.PRNGKey(31 + s))
            occ2 = np.argmin(((q2[:, :, None, : cfg.addr_dim]
                               - anchors[None, None]) ** 2).sum(-1), -1)
            st = stats(occ2)
            # and the combinatorial ceiling at that fan-out (noise ~ sigma_q/sqrt P)
            cl = phi_decodability_ceiling(phi2, c2, fam, q0_unseen=q2)
            st["ceiling_as_launched"] = cl["as_launched_accuracy"]
            sweep[str(P2)] = st
        row["p_sweep_OUT_OF_PROTOCOL"] = sweep
        rows.append(row)
        print(f"[mech] seed={s} raw distinct={row['raw_launch']['distinct_mean']:.2f} "
              f"prec={row['raw_launch']['precision']:.3f} "
              f"exact={row['raw_launch']['exact_set']:.4f}", flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict)
            and k != "p_sweep_OUT_OF_PROTOCOL"]
    agg = {k: {kk: float(np.mean([r[k][kk] for r in rows])) for kk in rows[0][k]}
           for k in keys}
    agg["p_sweep_OUT_OF_PROTOCOL"] = {
        p: {kk: float(np.mean([r["p_sweep_OUT_OF_PROTOCOL"][p][kk] for r in rows]))
            for kk in rows[0]["p_sweep_OUT_OF_PROTOCOL"][p]}
        for p in rows[0]["p_sweep_OUT_OF_PROTOCOL"]}
    res = {"rows": rows, "aggregate": agg}
    if out:
        _dump(res, out / "stage_mechanism.json")
    return res


# ==========================================================================
# STAGE 3 — the declared OUT-OF-CLASS phi-decodability ceiling
# ==========================================================================
def stage_ceiling(cfg: CatTestConfig, grid: NullArmGrid, out: Optional[Path] = None,
                  seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """⛔ Reported, never scored as an arm (the SP-1 precedent)."""
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    rows = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=grid.n_val)
        r = phi_decodability_ceiling(S["phi"], cfg, S["family"], q0_unseen=S["q0_u"])
        r["seed"] = int(s)
        r["chance"] = S["chance"]
        rows.append(r)
        print(f"[ceiling] seed={s} noiseless={r['noiseless_accuracy']:.4f} "
              f"as_launched={r['as_launched_accuracy']:.4f}", flush=True)
    agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
               "two_se": float(2 * np.std([r[k] for r in rows], ddof=1)
                               / np.sqrt(len(rows))) if len(rows) > 1 else float("nan")}
           for k in rows[0] if k not in ("seed", "n_combos")}
    res = {"rows": rows, "aggregate": agg, "n_combos": rows[0]["n_combos"]}
    if out:
        _dump(res, out / "stage_ceiling.json")
    return res


# ==========================================================================
# STAGE 4 — the oracle-imitation row (T5.2 rider (i)) + F5's agreement number
# ==========================================================================
def stage_oracle(cfg: CatTestConfig, grid: NullArmGrid, selected: Dict[str, Any],
                 out: Optional[Path] = None,
                 seeds: Sequence[int] = (0, 1, 2)) -> Dict[str, Any]:
    """N3 fitted **on the physics arm's own assignments**.

    ⚠ The physics arm here is the cat-test's, rebuilt at its registered operating
    point; its K-verdict is **quoted, never re-adjudicated** (it died at K5). The
    row exists because *a physics arm that cannot beat an imitation of itself has
    no organization claim* — and because F5's agreement statistic needs an
    independent reproduction.
    """
    from chlu.core.factored_store import multi_particle_read
    from chlu.experiments.exp_cat_test import build_physics_arm

    rows = []
    for s in seeds:
        t0 = time.time()
        S = seed_setup(cfg, s, n_val=grid.n_val)
        fam, anchors = S["family"], S["anchors"]
        arm = build_physics_arm(cfg, fam, int(s), phi=S["phi"])
        k = jax.random.PRNGKey(2000 + int(s))
        z_s = multi_particle_read(arm["store"], S["phi"], cfg, S["ind_s"], k)
        z_u = multi_particle_read(arm["store"], S["phi"], cfg, S["ind_u"],
                                  jax.random.fold_in(k, 1))
        occ_s = occupancy(z_s, anchors)
        occ_u = occupancy(z_u, anchors)
        phys_native = _native(_z_native(z_u, cfg), fam.y_unseen, fam.tol)
        allidx = np.arange(len(fam.seen))
        # the oracle-imitation arm: N3 trained to REPRODUCE occ_s
        imit = n3_static_geometric(cfg, fam, anchors, S["q0_s"], fam.y_seen,
                                   level="csb", lr=3e-2, tau=0.2, steps=grid.steps,
                                   seed=int(s), payload_source="written",
                                   target_assign=occ_s)
        a_u = imit["assign"](S["q0_u"])
        z_i = imit["apply"](S["q0_u"])
        z_i_s = imit["apply"](S["q0_s"])
        rd = fit_readers(z_i_s, fam.y_seen, anchors=np.asarray(imit["codebook"][0]),
                         well_payloads=np.asarray(imit["codebook"][1]), seed=int(s))
        # the read-objective-fitted N3 (F5's own null), for the agreement number
        conf3 = selected.get("N3", {}).get("config",
                                           {"lr": 1e-2, "level": "sb",
                                            "payload_source": "written", "tau": 0.2})
        f3 = _fit_arm("N3", conf3, S, cfg, grid, idx=allidx)
        rows.append({
            "seed": int(s),
            "physics_native_unseen": phys_native,
            "imitation_agreement_unseen": float((a_u == occ_u).mean()),
            "imitation_agreement_seen": float(
                (imit["assign"](S["q0_s"]) == occ_s).mean()),
            "imitation_unseen_native": _native(_z_native(z_i, cfg), fam.y_unseen,
                                               fam.tol),
            "imitation_readers_unseen": {k2: score_reader(v, z_i, fam.y_unseen,
                                                          fam.tol)
                                         for k2, v in rd.items()},
            "F5_agreement_read_objective": float(
                (f3["assign"](S["q0_u"]) == occ_u).mean()),
            "F5_fires": bool((a_u == occ_u).mean() >= 0.99),
            "physics_occupancy_distinct_wells": float(
                np.mean([len(set(r.tolist())) for r in occ_u])),
            "wall_s": round(time.time() - t0, 1)})
        print(f"[oracle] seed={s} agree_unseen="
              f"{rows[-1]['imitation_agreement_unseen']:.3f} "
              f"F5_read_obj={rows[-1]['F5_agreement_read_objective']:.3f} "
              f"({rows[-1]['wall_s']:.0f}s)", flush=True)
    res = {"rows": rows,
           "imitation_agreement_unseen_mean": float(np.mean(
               [r["imitation_agreement_unseen"] for r in rows])),
           "F5_agreement_read_objective_mean": float(np.mean(
               [r["F5_agreement_read_objective"] for r in rows])),
           "F5_fires_any": bool(any(r["F5_fires"] for r in rows))}
    if out:
        _dump(res, out / "stage_oracle.json")
    return res


# ==========================================================================
# the runner
# ==========================================================================
def run_null_arms(project: Optional[str] = None, seed: int = 0, quick: bool = False,
                  stages: Sequence[str] = ("grid", "score", "gridmax", "mechanism",
                                           "ceiling", "oracle"),
                  arms: Sequence[str] = ARMS, out_dir: Optional[str] = None,
                  overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the organizer audit. Defaults are the FROZEN operating point.

    ⚠ ``atoms_per_well = 32`` and ``payload_dim = 8`` are the cat-test's REGISTERED
    DEVIATIONS (D2, D1) — the cell that actually ran and the one the frozen ledger
    must be read against. The ``a`` capacity axis covers ``{12, 32, 64}``, so both
    readings of the frozen byte row are measured rather than assumed.
    """
    cfg = CatTestConfig(atoms_per_well=32, **dict(overrides or {}))
    g = NullArmGrid()
    if quick:
        cfg = replace(cfg, n_wells=16, f_subset=3, n_items=24, n_unseen=48,
                      atoms_per_well=6, payload_dim=6, write_steps=40,
                      address_steps=60, read_steps=60, organize_steps=5, quick=True)
        g = NullArmGrid(lrs=(3e-3, 1e-2), tune_seeds=(0,), score_seeds=(0, 1),
                        n_val=8, steps=40, n1_atoms_per_well=(4, 6),
                        n1_taus=(0.2,), n1_inits=("shell",), n2_codes=(16,),
                        n2_commitments=(0.0, 0.25), n2_restarts=2,
                        n3_levels=("sb",), n3_payloads=("written",),
                        n4_ks=(1, 3), n5_hidden=(16,), n5_momentum=(0.9,),
                        n5_decay=(0.0,), n5_gate=("surprise",), n5_chunk=(1,),
                        n5_passes=1, n5_pretrain_steps=40)
    out = Path(out_dir) if out_dir else Path("results") / "null_arms"
    out.mkdir(parents=True, exist_ok=True)
    res: Dict[str, Any] = {"config": cfg.as_dict(),
                           "non_default_flags": cfg.as_flag_table(),
                           "grid": g.as_dict(), "quick": bool(quick),
                           "arms": list(arms)}
    sel: Dict[str, Any] = {}

    def _sel():
        nonlocal sel
        if not sel and (out / "stage_grid.json").exists():
            sel = json.loads((out / "stage_grid.json").read_text())["selected"]
        return sel

    if "grid" in stages:
        res["stage_grid"] = stage_grid(cfg, g, arms=arms, out=out)
        sel = res["stage_grid"]["selected"]
    if "score" in stages:
        res["stage_score"] = stage_score(cfg, g, _sel(), arms=arms, out=out)
    if "gridmax" in stages:
        res["stage_gridmax"] = stage_gridmax(cfg, g, arms=arms, out=out)
    if "mechanism" in stages:
        res["stage_mechanism"] = stage_mechanism(cfg, g, _sel(), arms=arms, out=out)
    if "ceiling" in stages:
        res["stage_ceiling"] = stage_ceiling(cfg, g, out=out)
    if "oracle" in stages:
        res["stage_oracle"] = stage_oracle(cfg, g, _sel(), out=out,
                                           seeds=(0,) if quick else (0, 1, 2))
    _dump(res, out / "null_arms_summary.json")
    return res
