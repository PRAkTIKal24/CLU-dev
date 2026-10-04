"""Experiment TRAJ-WRITE (C2W2 Route 1): **ask the write to put information in
the trajectory**, then measure honestly whether it buys anything.

**The diagnosis this experiment tests** (charter §A2.1, ratified). C2W1 measured
the dividend at ≈0 or negative on every family, the trajectory channel not
measurably live, and the substitute audit 0-for-4. The Advisor's diagnosis is
that ``write_loss`` constrains only **isolated settled endpoints**, so every read
that touches the in-between regions loses *by construction* — i.e. **the
charter's actual hypothesis has never been tested**. Route 1 adds two terms to
the write objective (:func:`~chlu.training.train_memory.trajectory_margin_penalty`
and :func:`~chlu.training.train_memory.path_equal_depth_penalty`, both default
0.0) and re-runs the gym on the store they produce.

**Everything is emitted as :class:`~chlu.eval.race.RaceCell`** — the frozen C2W2
schema shared with Route 2 (``ssb-shell-atoms``); nothing else is comparable
across the two branches, and the C2W2 gate is evaluated on the merged card.

⚠ **The gate's verdict is the Hub's.** This module computes the arithmetic
(sample sd ``ddof=1``, ``SE = sd/sqrt(n)``, "clears 0 beyond 2 SE"), the
admissible-cell coverage per family, and every excluded cell **with its reason**.
It does not decide anything.

Runnable directly::

    uv run python -m chlu.experiments.exp_traj_write --quick
    uv run python -m chlu.experiments.exp_traj_write --families overload --seeds 0 1 2

or via the CLI: ``chlu exp-traj-write [--project N] [--seed I] [--quick]``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chlu.eval.race import (
    ByteLedger,
    Liveness,
    RaceCell,
    TrajectoryLaunder,
    WriteRecord,
    coverage_table,
    gate_summary,
    save_cells,
    score_card,
    verdicts_to_markdown,
)
from chlu.experiments.exp_memory_gym import run_cell

#: ⛔ PRE-REGISTERED coefficient grid (PREREG §1.3, declared before any run):
#: >= 3 non-zero values spanning >= 2 decades (0.03 -> 3.0), plus the perturbing
#: liveness anchor at the top and the mandatory zero point (the D1 regression
#: gate). Never chosen after seeing a result.
COEFF_GRID: Tuple[float, ...] = (0.03, 0.3, 3.0, 30.0)

#: The registered trajectory-term rollout (PREREG §1.1). ``rollout_steps=60`` at
#: ``dt=0.05``/``gamma=0.05`` is 3 time units of the read's own dynamics; the
#: write-side rollout is asserted equal to the shipped Verlet step for identity
#: mass in ``tests/test_traj_write_objective.py``.
TRAJ_KWARGS = {"rollout_steps": 60, "stride": 6, "n_launch": 4}
PATH_KWARGS = {"n_interp": 7}

#: ``(family, gym-arm)``. ⭐ ``overload`` is run at the **shipped atom budget**
#: (``load1x_shipped``, the 478x anchor) because the gym measured that as the
#: only cell in which the store actually works — a liveness verdict taken on an
#: unwritten store measures write failure, not the term.
FAMILY_ARMS: Tuple[Tuple[str, str], ...] = (
    ("overload", "load1x_shipped"),
    ("aggregate", "base"),
    ("manifold", "base"),
)


def objective_spec(arm: str, lam: float) -> Optional[dict]:
    """The write-objective spec for one race arm (``None`` = the control)."""
    if arm == "endpoint_write":
        return None
    lk: Dict[str, Any] = {}
    if arm in ("traj_write", "traj+path"):
        lk.update(lambda_traj=float(lam), traj_kwargs=dict(TRAJ_KWARGS))
    if arm in ("path_write", "traj+path"):
        lk.update(lambda_path=float(lam), path_kwargs=dict(PATH_KWARGS))
    if not lk:
        raise ValueError(f"unknown race arm {arm!r}")
    return {"loss_kwargs": lk}


def plan(coeffs: Sequence[float] = COEFF_GRID,
         combined: Sequence[float] = (0.3,)) -> List[Tuple[str, float]]:
    """``(arm, coefficient)`` pairs. The zero point is the ``endpoint_write``
    control; ``traj+path`` is run at a single declared coefficient (cost)."""
    out: List[Tuple[str, float]] = [("endpoint_write", 0.0)]
    out += [("traj_write", c) for c in coeffs]
    out += [("path_write", c) for c in coeffs]
    out += [("traj+path", c) for c in combined]
    return out


# --------------------------------------------------------------------------
# gym record -> race cell
# --------------------------------------------------------------------------
def _write_record(rec: dict, loss_tol: float = 0.05) -> WriteRecord:
    """Gate ruling (i): a cell may only vote if its write **converged**.

    ``converged`` = the endpoint part of the final write loss is below
    ``loss_tol`` **and** the loss is still moving (not plateaued at a high
    value). ⚠ For a Route-1 arm the recorded loss INCLUDES ``lambda * L_term``,
    which does not go to zero — so the endpoint part is recovered by subtracting
    the term's own (measured constant) contribution where it is known, and
    otherwise the sub-shipped-budget signature the gym measured
    (``final loss 0.20-0.24``, ``lambda_min -0.21..-1.20``) is used.
    """
    losses = list(rec.get("write_losses") or [])
    final = float(losses[-1]) if losses else float("nan")
    ep = rec.get("endpoint_write_loss", None)
    endpoint_part = float(ep) if ep is not None and np.isfinite(ep) else final
    # the WORST site, not the last one: a stream converges per item
    eps = [float(x) for x in (rec.get("endpoint_write_losses") or [])
           if x is not None and np.isfinite(x)]
    if eps:
        endpoint_part = float(max(eps))
    lam = (rec.get("certificates") or {}).get("lambda_min", float("nan"))
    lam = float(lam) if lam is not None else float("nan")
    plateaued = bool(len(losses) >= 3 and np.isfinite(endpoint_part)
                     and endpoint_part > loss_tol)
    converged = bool(np.isfinite(endpoint_part) and endpoint_part <= loss_tol)
    return WriteRecord(
        steps=int((rec.get("clu_config_non_default") or {}).get("write_steps", 300)),
        final_loss=final, lambda_min_min=lam, converged=converged,
        plateaued=plateaued,
        reason=("" if converged else
                f"endpoint write loss {endpoint_part:.4f} > {loss_tol}"),
    )


def to_cell(rec: dict, arm: str, lam: float, liveness: Optional[Liveness] = None,
            route: str = "route1") -> RaceCell:
    """Convert one gym record into the frozen race-card schema."""
    d = rec["dividend"]
    ctrl = dict(d.get("controls") or {})
    tl = dict(rec.get("trajectory_launder") or {})
    ledger = rec.get("byte_ledger") or {}
    audit = rec.get("trivial_substitute_audit") or {}
    mon = {m["name"]: {"tripped": m["tripped"], "applicable": m["applicable"],
                       "value": m["value"], "mode": m["mode"]}
           for m in rec.get("monitors", [])}
    cell = RaceCell(
        route=route, arm=(arm if lam == 0.0 else f"{arm}@{lam:g}"),
        family=rec["family"], seed=int(rec["seed"]),
        metric_name=rec["primary_metric"],
        full=float(d["full"]), settle_deleted_launder=float(d["launder"]),
        same_keys_null=float(ctrl.get("same_keys_null", float("nan"))),
        blank=float(ctrl.get("blank_store", float("nan"))),
        plus_zero_byte_substitute=float(audit.get("best_zero_byte", float("nan"))),
        trajectory_launder=TrajectoryLaunder(
            full=float(tl.get("full", float("nan"))),
            q0_only=float(tl.get("q0_only", float("nan"))),
            endpoints=float(tl.get("endpoints", float("nan"))),
            blank_store=float(tl.get("blank_store", float("nan"))),
            chance=float(tl.get("chance", float("nan"))),
            bar=float(tl.get("bar", float("nan")))),
        bytes=ByteLedger(full=int(ledger.get("full_bytes", 0)),
                         launder=int(ledger.get("launder_bytes", 0)),
                         breakdown={k: int(v) for k, v in
                                    (ledger.get("breakdown") or {}).items()}),
        phi_id="identity (gym embeds the address directly; phi is not learned "
               "in this family set)",
        phi_bytes=0,
        write=_write_record(rec),
        liveness=liveness or Liveness(),
        monitors=mon,
        flags={"gym_arm": rec["arm"], "coefficient": lam,
               "traj_kwargs": TRAJ_KWARGS if "traj" in arm else None,
               "path_kwargs": PATH_KWARGS if "path" in arm else None,
               **(rec.get("gym_config_non_default") or {}),
               **(rec.get("clu_config_non_default") or {})},
        notes=("byte ratio >= 2.20x is ARCHITECTURAL (one atom group per item); "
               "this cell is NOT quotable as a byte-matched dividend"
               if float(ledger.get("ratio", 0.0)) >= 2.20 else ""),
    )
    cell.resolve_admissibility()
    return cell


# --------------------------------------------------------------------------
# ⭐ liveness / the perturbing anchor — gate ruling (ii) and its counterweight
# --------------------------------------------------------------------------
#: PRE-REGISTERED liveness bars (PREREG §1.3 / §2 / P3), fixed before any run.
ANCHOR_METRIC_DROP = 0.20   # "visibly perturbs the write, even destructively"
LIVENESS_DELTA = 0.05       # trajectory decodability gain over the lambda=0 store


def _term_of(arm: str) -> Optional[str]:
    base = arm.split("@")[0]
    return None if base == "endpoint_write" else base


def annotate_liveness(cells: Sequence[RaceCell]) -> List[RaceCell]:
    """Fill each cell's :class:`~chlu.eval.race.Liveness` from the measured card.

    ⭐ **This is the counterweight to gate ruling (ii), and it is the difference
    between a legitimate <=0 vote and an under-powered grid.** The bars are the
    PRE-REGISTERED ones, applied to the card; nothing here is chosen after the
    fact:

    * **the perturbing anchor** — the grid must contain at least one coefficient
      at which the term *visibly perturbs the write, even destructively*. Met
      when the family's primary metric drops by ``ANCHOR_METRIC_DROP`` against
      the ``endpoint_write`` control, **or** when the term drives the write
      itself inadmissible (``lambda_min < 0`` / non-convergence) — the most
      visible perturbation there is. It is a **grid-level** property, so it is
      stamped on every cell of that ``(family, term)``.
    * **liveness** — the P-A bar: the trajectory-written store's trajectory must
      carry more decodable information than the ``lambda=0`` store's, by
      ``LIVENESS_DELTA``, against the **capacity-matched** ``endpoints``
      baseline and the blank store.

    ⚠ The ``endpoint_write`` control has no term, so liveness is **not
    applicable** to it; it is stamped ``perturbing_anchor=True`` with
    ``detail["applicable"]=False`` so it grades on its dividend alone rather
    than being mislabelled an under-powered grid.
    """
    cells = list(cells)
    ctrl_full: Dict[str, List[float]] = {}
    ctrl_traj: Dict[str, List[float]] = {}
    for c in cells:
        if _term_of(c.arm) is None:
            ctrl_full.setdefault(c.family, []).append(c.full)
            ctrl_traj.setdefault(c.family, []).append(c.trajectory_launder.full)

    def _m(d, fam):
        v = [x for x in d.get(fam, []) if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    groups: Dict[Tuple[str, str], List[RaceCell]] = {}
    for c in cells:
        t = _term_of(c.arm)
        if t is not None:
            groups.setdefault((c.family, t), []).append(c)

    anchors: Dict[Tuple[str, str], Tuple[bool, List[float], float]] = {}
    for (fam, term), grp in groups.items():
        base = _m(ctrl_full, fam)
        by_lam: Dict[float, List[RaceCell]] = {}
        for c in grp:
            by_lam.setdefault(float(c.arm.split("@")[1]), []).append(c)
        hit, anchor_lam = False, float("nan")
        for lam in sorted(by_lam):
            vals = [c.full for c in by_lam[lam] if np.isfinite(c.full)]
            drop = (base - float(np.mean(vals))) if (vals and np.isfinite(base)) else 0.0
            broke = any(not c.write.admissible() for c in by_lam[lam])
            if drop >= ANCHOR_METRIC_DROP or broke:
                hit, anchor_lam = True, lam
                break
        anchors[(fam, term)] = (hit, sorted(by_lam), anchor_lam)

    for c in cells:
        term = _term_of(c.arm)
        if term is None:
            c.liveness = Liveness(
                passed=False, perturbing_anchor=True, grid=(0.0,),
                detail={"applicable": False,
                        "why": "control arm (lambda=0): no objective term to be live"})
            continue
        hit, grid, anchor_lam = anchors[(c.family, term)]
        lam = float(c.arm.split("@")[1])
        tl = c.trajectory_launder
        base_tl = _m(ctrl_traj, c.family)
        bar = float(np.nanmax([tl.endpoints, tl.blank_store, tl.bar]))
        gain = (tl.full - base_tl) if np.isfinite(base_tl) else float("nan")
        c.liveness = Liveness(
            passed=bool(np.isfinite(gain) and gain >= LIVENESS_DELTA
                        and np.isfinite(tl.full) and tl.full > bar),
            coefficient=lam, value=tl.full, baseline=base_tl, bar=bar,
            perturbing_anchor=bool(hit), grid=tuple(grid),
            detail={"anchor_coefficient": anchor_lam,
                    "anchor_rule": (f"family metric drops >= {ANCHOR_METRIC_DROP} "
                                    "vs endpoint_write, or the term drives the "
                                    "write inadmissible"),
                    "liveness_rule": (f"trajectory-launder `full` gain >= "
                                      f"{LIVENESS_DELTA} over the lambda=0 store, "
                                      "AND above max(endpoints, blank, bar)"),
                    "gain_over_lambda0": gain})
    return cells


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_traj_write(
    config=None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    families: Optional[Sequence[str]] = None,
    arms: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0, 1, 2),
    coeffs: Sequence[float] = COEFF_GRID,
    quick: bool = False,
    out_json: Optional[str] = None,
) -> dict:
    """Run the Route-1 race card and score it."""
    os.makedirs(save_dir, exist_ok=True)
    fam_arms = [fa for fa in FAMILY_ARMS if not families or fa[0] in set(families)]
    steps = [s for s in plan(coeffs) if not arms or s[0] in set(arms)]
    if quick:
        seeds = (0,)
        steps = [s for s in steps if s[1] in (0.0, 3.0)]

    cells: List[RaceCell] = []
    records: List[dict] = []
    t_all = time.time()
    for family, gym_arm in fam_arms:
        for arm, lam in steps:
            for s in seeds:
                t0 = time.time()
                try:
                    rec = run_cell(family, gym_arm, int(s), quick=quick, loud=False,
                                   write_objective=objective_spec(arm, lam))
                except Exception as exc:  # a failed cell is REPORTED, never silent
                    import traceback

                    traceback.print_exc()
                    cells.append(RaceCell(
                        route="route1", arm=f"{arm}@{lam:g}", family=family,
                        seed=int(s), metric_name="?", gate_admissible=False,
                        exclusion_reason=f"cell raised: {exc!r}"))
                    continue
                rec["wall_s"] = time.time() - t0
                records.append(rec)
                cell = to_cell(rec, arm, lam)
                cells.append(cell)
                print(f"[{family}/{arm}@{lam:g} s{s}] {cell.metric_name} "
                      f"full={cell.full:.4f} launder={cell.settle_deleted_launder:.4f} "
                      f"div={cell.dividend:+.4f} +0B_margin={cell.substitute_margin:+.4f} "
                      f"adm={cell.gate_admissible} ({rec['wall_s']:.0f}s)", flush=True)
                _dump(save_dir, cells, records, out_json)

    cells = annotate_liveness(cells)
    verdicts = score_card(cells)
    out = {
        "wall_s": time.time() - t_all,
        "n_cells": len(cells),
        "coefficient_grid": list(coeffs),
        "seeds": list(seeds),
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); clears iff mean-2SE>0",
        "coverage_per_family": coverage_table(verdicts),
        "gate": gate_summary(verdicts),
        "verdicts": [v.as_dict() for v in verdicts],
        "markdown": verdicts_to_markdown(verdicts),
    }
    _dump(save_dir, cells, records, out_json, summary=out)
    print("\n" + out["markdown"])
    return out


def _dump(save_dir, cells, records, out_json, summary=None):
    path = out_json or os.path.join(save_dir, "exp_traj_write_race_card.json")
    save_cells(path, cells)
    with open(str(path).replace(".json", "_records.json"), "w") as fh:
        json.dump(records, fh, indent=2, default=_json_default)
    if summary is not None:
        with open(str(path).replace(".json", "_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser(
        description="C2W2 Route 1: the trajectory/path write objective, on the "
                    "frozen race card.")
    ap.add_argument("--project")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--coeffs", nargs="+", type=float, default=list(COEFF_GRID))
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
    run_experiment_traj_write(config=config, save_dir=save_dir,
                              models_dir=models_dir, seed=a.seed,
                              families=a.families, arms=a.arms, seeds=a.seeds,
                              coeffs=a.coeffs, quick=a.quick, out_json=a.out)


if __name__ == "__main__":
    main()
