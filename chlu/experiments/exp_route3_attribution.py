"""⭐ Experiment ROUTE-3 STAGE 1 — **the store-attribution curve** (charter §A10).

Measurement only. **No new store, no new objective, no claim of a dividend** —
this experiment decides whether Route 3 stage 2 gets built at all, by applying the
Head's pre-registered §A9.4 bar arithmetically:

> **Stage 2 unlocks iff the per-slot store-attributable discriminability (full −
> settle-deleted launder, per slot ``t``) clears the launch-noise floor BEYOND
> 2 SE, at >= 3 seeds, on >= 1 family, at ANY ``t`` — q-slots and p-slots scored
> separately (a live p-channel unlocks even with a dead q-channel).**

⛔ **And it is overridden by §A9.5**, which is built here in stage 1 rather than
deferred: if a **per-slot matched-bytes table launder** reproduces the slotted
read, Route 3 has degenerated into ``K`` time-indexed lookup tables and fails
regardless of dividend (intervention §8.2).

The rig is the shipped one: :func:`chlu.experiments.exp_memory_gym.run_cell`'s own
write path, verbatim (same key chain, same query RNG, same blank-store control),
so the numbers are comparable to `memory-gym-v0` and to both C2W2 routes. Four
reads per cell:

============================  =================================================
``full``                      real store, real launch
``launder`` (settle-deleted)  **store-deleted** system, same launch
``floor`` (launch noise)      store-deleted, launch re-drawn within its own cloud
``perturbed``                 real store, launch re-drawn — the §A8.2 Jacobian
============================  =================================================

Runnable directly::

    uv run python -m chlu.experiments.exp_route3_attribution --quick
    uv run python -m chlu.experiments.exp_route3_attribution --families overload

⚠ **No CLI hook.** ``chlu/cli/experiment_cmd.py`` has no declared owner in C2W3
and two engineer branches are live, so this module is invoked directly rather
than risking the wave's zero-conflict record over a convenience wrapper (the
``exp_ssb_shell`` precedent). The hook is owed and is reported to the Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chlu.eval.attribution import (
    SLOT_GRID,
    CurveBundle,
    a95_verdict,
    address_block_curve,
    apply_a94_bar,
    attribution_curve,
    jacobian_curves,
)
from chlu.experiments.exp_memory_gym import _build_queries
from chlu.experiments.memory_gym import gym_config, make_gym_stream

#: ``(family, gym arm)``. ⚠ ``overload`` is quoted **only** at ``load1x_shipped``
#: (the 478x anchor): at the gym's base atom budget it went 0/18 admissible
#: *including the Gaussian control* (C2W2 reconciliation 6, Hub-accepted).
FAMILY_ARMS: Tuple[Tuple[str, str], ...] = (
    ("overload", "load1x_shipped"),
    ("aggregate", "base"),
    ("manifold", "base"),
)

#: Head ruling (i) admissibility, the Route-1 convention (``race.WriteRecord``).
ENDPOINT_LOSS_TOL = 0.05

#: The one bounded budget escalation a 0-admissible family gets before it may be
#: called "not testable this wave" (C2W2 ruling (i) counterweight).
ESCALATION_WRITE_STEPS = 900


def _answer_channel(family: str, ccfg) -> Tuple[int, str]:
    """The family's own answer channel inside the latent, and its name.

    ``overload``/``aggregate`` answer with the **payload** block; ``manifold``
    answers with the **spectator** block (its target is the launch spectator
    coordinate). The instrument reads exactly the channel the family is scored on
    — it is not a new read-out.
    """
    if family == "manifold":
        return int(ccfg.addr_dim + ccfg.payload_dim), "spectator"
    return int(ccfg.addr_dim), "payload"


def _labels(qs) -> np.ndarray:
    """Item labels for the §A8.2 separation curve (pair id where there is no item)."""
    lab = np.asarray(qs.label).ravel()
    if np.all(lab >= 0):
        return lab
    pairs = np.asarray(qs.meta.get("pairs"))
    if pairs is not None and pairs.size:
        uniq = {tuple(p): i for i, p in enumerate(np.unique(pairs, axis=0))}
        return np.asarray([uniq[tuple(p)] for p in pairs], dtype=int)
    return np.zeros_like(lab)


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
def run_attribution_cell(family: str, arm: str = "base", seed: int = 0, *,
                         quick: bool = False, write_steps: Optional[int] = None,
                         slots: Sequence[int] = SLOT_GRID,
                         loud: bool = False) -> CurveBundle:
    """Write the gym stream on the shipped rig, then measure the four arms."""
    import jax

    from chlu.core.clu_system import build_system

    over: Dict[str, Any] = {}
    clu_over: Dict[str, Any] = {}
    if write_steps is not None:
        clu_over["write_steps"] = int(write_steps)
    if quick:
        over.update(n_offer=6, capacity=6, budget=6, reference_capacity=3,
                    n_query_per_item=2, n_query_per_pair=3, consolidate_every=2,
                    min_consolidations=4, n_manifold_launch=4)
        clu_over.update(write_steps=int(write_steps or 30), address_steps=80,
                        read_steps=120, n_query_per_item=2, quick=True)
    if clu_over:
        over["clu_overrides"] = clu_over
    gcfg = gym_config(family, arm, seed=seed, **over)
    ccfg = gcfg.build_clu()

    # --- the shipped write path, verbatim (same key chain as run_cell) -----
    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=loud)
    stream = make_gym_stream(gcfg, ccfg)
    key = jax.random.PRNGKey(seed + 1)
    prev = 0
    for b in stream.chunks:
        if b > prev:
            key, k_w = jax.random.split(key)
            system.write_stream(stream.items[prev:b], key=k_w)
            prev = b
        system.consolidate()

    ids, centers, pays = system.codebook()
    if len(ids) < 2:
        return CurveBundle(family=family, seed=seed, admissible=False,
                           reason=f"degenerate cell: n_live={len(ids)}")

    eps = [float(x) for x in (system._endpoint_losses or []) if np.isfinite(x)]
    endpoint_loss = float(max(eps)) if eps else float("nan")
    lam_min = float(system.certificates().get("lambda_min", float("nan")))
    admissible = bool(np.isfinite(endpoint_loss) and endpoint_loss <= ENDPOINT_LOSS_TOL
                      and np.isfinite(lam_min) and lam_min >= 0.0)
    reason = "" if admissible else (
        f"endpoint write loss {endpoint_loss:.4f} > {ENDPOINT_LOSS_TOL}"
        if not (np.isfinite(endpoint_loss) and endpoint_loss <= ENDPOINT_LOSS_TOL)
        else f"lambda_min<0 ({lam_min:+.4f})")

    # --- the queries: the family's own, the gym's own RNG -------------------
    rng = np.random.default_rng(seed + 7717)
    born = np.zeros((len(ids),), dtype=float)
    qs = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)

    # --- the four arms ------------------------------------------------------
    # the store-deleted system is the harness's OWN blank/empty-store control
    blank = build_system(replace(ccfg, seed=ccfg.seed + 991),
                         key=jax.random.PRNGKey(seed + 991), loud=False)
    # the launch-noise draw: the query law itself (+N(0, sigma_q) on the address
    # block), an INDEPENDENT draw of the same magnitude
    nrng = np.random.default_rng(seed + 20260731)
    q0 = np.asarray(qs.q0, dtype=np.float32)
    delta = np.zeros_like(q0)
    delta[:, : ccfg.addr_dim] = nrng.normal(
        size=(q0.shape[0], ccfg.addr_dim)) * float(ccfg.query_sigma)
    q0_pert = q0 + delta

    t0 = time.time()
    res_full = system.read(q0)
    res_launder = blank.read(q0)
    res_floor = blank.read(q0_pert)
    res_pert = system.read(q0_pert)
    read_s = time.time() - t0

    idx, chan_name = _answer_channel(family, ccfg)
    rows = attribution_curve(
        res_full, res_launder, res_floor, np.asarray(qs.target, dtype=float),
        idx, slots=slots, keys=np.asarray(qs.keys), centers=np.asarray(centers),
        traj_stride=int(ccfg.traj_stride), dt=float(ccfg.dt),
        address_steps=int(ccfg.address_steps))
    jac = jacobian_curves(res_full, res_pert, delta[:, : ccfg.addr_dim],
                          _labels(qs), slots=slots, dim=int(ccfg.dim))
    # ⚠ POST-HOC secondary diagnostic (declared): the address block, where q0 IS
    # the query — the only place §A8.1's "position at small t is almost pure
    # query" can be tested, because the ANSWER channel of q0 is zeroed by the
    # shipped read. Never enters the §A9.4 bar.
    addr_rows = address_block_curve(res_full, res_launder, res_floor, _labels(qs),
                                    int(ccfg.addr_dim), slots=slots)

    return CurveBundle(
        family=family, seed=seed, admissible=admissible, reason=reason, rows=rows,
        jacobian=jac,
        flags={"address_block_posthoc": addr_rows, "gym_arm": arm, "n_live": int(len(ids)), "n_queries": int(len(qs)),
               "answer_channel": chan_name, "channel_index": int(idx),
               "endpoint_write_loss_max": endpoint_loss, "lambda_min": lam_min,
               "write_steps": int(ccfg.write_steps),
               "address_steps": int(ccfg.address_steps),
               "read_steps": int(ccfg.read_steps),
               "traj_stride": int(ccfg.traj_stride), "dt": float(ccfg.dt),
               "query_sigma": float(ccfg.query_sigma),
               "gamma_address": float(ccfg.gamma_address),
               "gamma_read": float(ccfg.gamma_read),
               "n_traj_points": int(np.asarray(res_full.traj).shape[1]),
               "retries": int(res_full.retries), "read_wall_s": read_s,
               **gcfg.as_flag_table(), **ccfg.as_flag_table()})


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_route3_attribution(
    config=None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    families: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0, 1, 2),
    quick: bool = False,
    escalate: bool = True,
    out_json: Optional[str] = None,
) -> dict:
    """Run the curve on every family, apply §A9.4, then §A9.5 (which overrides)."""
    os.makedirs(save_dir, exist_ok=True)
    fam_arms = [fa for fa in FAMILY_ARMS if not families or fa[0] in set(families)]
    if quick:
        seeds = tuple(seeds)[:1]

    bundles: Dict[str, Dict[int, CurveBundle]] = {}
    coverage: Dict[str, dict] = {}
    t_all = time.time()
    for family, arm in fam_arms:
        bundles[family] = {}
        excl: List[dict] = []
        for s in seeds:
            b = run_attribution_cell(family, arm, int(s), quick=quick)
            bundles[family][int(s)] = b
            if not b.admissible:
                excl.append({"seed": int(s), "reason": b.reason})
            print(f"[{family}/{arm} s{s}] admissible={b.admissible} "
                  f"({b.reason or 'ok'}) n_rows={len(b.rows)}", flush=True)
        adm = [s for s, b in bundles[family].items() if b.admissible]
        escalated = False
        if not adm and escalate and not quick:
            # ⭐ the ONE bounded budget escalation (C2W2 ruling (i) counterweight)
            escalated = True
            for s in seeds:
                b = run_attribution_cell(family, arm, int(s),
                                         write_steps=ESCALATION_WRITE_STEPS)
                bundles[family][int(s)] = b
                excl.append({"seed": int(s), "escalated_write_steps":
                             ESCALATION_WRITE_STEPS, "reason": b.reason})
                print(f"[{family}/{arm} s{s} ESCALATED {ESCALATION_WRITE_STEPS}] "
                      f"admissible={b.admissible} ({b.reason or 'ok'})", flush=True)
            adm = [s for s, b in bundles[family].items() if b.admissible]
        coverage[family] = {
            "gym_arm": arm, "n_cells": len(seeds), "n_admissible": len(adm),
            "coverage": float(len(adm) / max(len(seeds), 1)),
            "admissible_seeds": sorted(int(x) for x in adm),
            "escalated": escalated, "escalation_write_steps": (
                ESCALATION_WRITE_STEPS if escalated else None),
            "excluded": excl,
            "verdict": ("VOTES" if adm else
                        "ABSTAINS (0 admissibly-written cells after the one "
                        "bounded escalation — neither unlocks stage 2 nor blocks it)"),
        }

    a94: Dict[str, Any] = {}
    a95: Dict[str, Any] = {}
    for family, per_seed in bundles.items():
        adm = coverage[family]["admissible_seeds"]
        rows = {s: b.rows for s, b in per_seed.items()}
        a94[family] = apply_a94_bar(rows, family=family, admissible_seeds=adm,
                                    min_seeds=1 if quick else 3)
        clearing = [(r["channel"], r["slot"]) for r in a94[family]["clearing_set"]]
        a95[family] = a95_verdict(rows, admissible_seeds=adm,
                                  clearing_slots=clearing or None)

    unlock = any(v["unlock"] for v in a94.values())
    killed = any(v.get("fires") for f, v in a95.items() if a94[f]["unlock"])
    out = {
        "wall_s": time.time() - t_all,
        "seeds": [int(s) for s in seeds],
        "slot_grid": list(SLOT_GRID),
        "quick": bool(quick),
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); clears iff mean-2SE>0",
        "coverage_per_family": coverage,
        "a94_unlock_bar": a94,
        "a95_table_launder": a95,
        "unlock": bool(unlock),
        "a95_fires_on_clearing_slots": bool(killed),
        "stage2_verdict": (
            "⛔ NO STAGE 2 — §A9.5 fires and overrides the unlock" if killed else
            ("unlock = true" if unlock else "unlock = false")),
        "curves": {f: [b.as_dict() for b in per.values()]
                   for f, per in bundles.items()},
    }
    path = out_json or os.path.join(save_dir, "exp_route3_attribution.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    out["metrics_path"] = path
    try:
        out["figures"] = _plot(out, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        out["figures"] = []
        out["figure_error"] = repr(exc)
    print(f"\nunlock = {out['unlock']}  |  §A9.5 fires on clearing slots = "
          f"{out['a95_fires_on_clearing_slots']}  |  {out['stage2_verdict']}")
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _plot(res: dict, save_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [f for f in res["curves"] if res["coverage_per_family"][f]["admissible_seeds"]]
    if not fams:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"q": "tab:blue", "p": "tab:red"}
    fam = fams[0]
    rows = res["a94_unlock_bar"][fam]["rows"]
    ax = axes[0][0]
    for ch in ("q", "p"):
        sub = sorted([r for r in rows if r["channel"] == ch], key=lambda r: r["t"])
        ax.errorbar([r["t"] for r in sub], [r["margin_mean"] for r in sub],
                    yerr=[2 * (r["margin_se"] or 0) for r in sub], marker="o",
                    ms=3, color=colors[ch], capsize=2,
                    label=f"{ch}-slots  (margin ± 2 SE)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("t (time units, log)")
    ax.set_ylabel("(full − settle-deleted) − launch-noise floor")
    ax.set_title(f"⭐ store-attribution curve — {fam}\n§A9.4: clears iff mean−2SE>0",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for ch in ("q", "p"):
        sub = sorted([r for r in rows if r["channel"] == ch], key=lambda r: r["t"])
        ax.plot([r["t"] for r in sub], [r["full_mean"] for r in sub], "o-", ms=3,
                color=colors[ch], label=f"{ch}: full")
        ax.plot([r["t"] for r in sub], [r["dividend_mean"] for r in sub], "s--",
                ms=3, color=colors[ch], alpha=0.5, label=f"{ch}: full − launder")
    ax.set_xscale("log")
    ax.set_xlabel("t (log)")
    ax.set_ylabel("discriminability |rho|")
    ax.set_title("the curve, not the endpoint", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    a95 = res["a95_table_launder"][fam]["rows"]
    for ch in ("q", "p"):
        sub = [r for r in a95 if r["channel"] == ch]
        ax.plot([r["slot"] for r in sub], [r["table_margin_mean"] for r in sub],
                "o-", ms=3, color=colors[ch], label=f"{ch}: read − table")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("slot index")
    ax.set_ylabel("read − per-slot table launder")
    ax.set_title("⛔ §A9.5 kill-condition (<= 0 ⇒ K time-indexed tables)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    jac = res["curves"][fam][0]["jacobian"]
    for ch in ("q", "p"):
        c = jac["channels"][ch]
        ax.plot(jac["slots"], c["contraction"], "o-", ms=3, color=colors[ch],
                label=f"{ch}: contraction ‖Δs_t‖/‖δ‖")
        ax.plot(jac["slots"], c["fisher"], "s--", ms=3, color=colors[ch], alpha=0.5,
                label=f"{ch}: separation (between/within)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("slot index")
    ax.set_title("§A8.2 flow-map Jacobian: contractive within, separated across",
                 fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(save_dir, "exp_route3_attribution.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def main():
    ap = argparse.ArgumentParser(
        description="Route 3 stage 1: the store-attribution curve (§A10) + the "
                    "§A9.4 unlock bar + the §A9.5 per-slot table launder.")
    ap.add_argument("--project")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    config, save_dir, models_dir = None, "results", None
    if a.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(a.project)
        paths = pm.get_paths(a.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        os.makedirs(save_dir, exist_ok=True)
    run_experiment_route3_attribution(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=a.seed,
        families=a.families, seeds=a.seeds, quick=a.quick,
        escalate=not a.no_escalate, out_json=a.out)


if __name__ == "__main__":
    main()
