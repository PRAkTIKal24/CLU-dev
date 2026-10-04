#!/usr/bin/env python
"""⭐ The C3 store-geometry sweep — smoke scale, REAL stream, ⛔ NEVER a claim venue.

############################################################################
⛔⛔ THIS IS NEVER A CLAIM VENUE. ⛔⛔
The shapes here (d_model 32, 2 layers, 60 optimisation steps, 0.6 MB of enwik8)
are chosen so a geometry can be *compared to another geometry* in minutes, not
so any number means anything on its own. A bpc from this script is not evidence
for or against the CLU, a rival, or anything else, in either direction. Claims
live at 26-47 M on CSF3 with >=3 seeds and the tier's own control (charter §2).
############################################################################

WHAT IT IS FOR — the one question this script exists to answer.

`CluSystemConfig.n_atoms` is **not** ``atoms_per_item * capacity``; it is

    n_atoms = max(atoms_per_item*K, min_atoms, round(min_atoms_base * min_atoms_c**addr_dim))

rounded up to a multiple of ``capacity``. At the ruled ``addr_dim = 8`` the
geometric term is ``512 * sqrt(2)**8 = 8192`` and the pilot's ``K*A = 32*256``
**ties** it — so at the pilot geometry ``atoms_per_item`` and ``capacity`` are
byte-inert and **the CLU store cannot be shrunk at all** through them. The only
way under a ~2 MiB matched-state-byte ceiling at ``addr_dim=8``/``n_layers=12``
is to go **below the w23 dimension-aware atom floor**, which is precisely the
regime that floor exists to forbid ("a starved cell reads as a capacity result
when it is an optimizer artefact", ``chlu/core/clu_system.py``).

So this sweep measures **write efficacy vs atom budget at d_addr = 8 on real
text**: is the w23 floor's declared margin real, and how far below 8192 can the
writer still dig a well? Predictions and falsifiers were filed BEFORE it ran, in
``.claude/outputs/c3-rival-ladder-prereg/PREREG-GEOMETRY-SWEEP.md``.

⛔ It trains **no ladder arm** and writes **no checkpoint**: two arms
(``clu_store`` and the ``none`` null), a few dozen steps, static bpc on the test
split, and the store-health watch series that ``train_arm`` already emits.

USAGE
    python scripts/c3_geometry_sweep.py OUT.json [--seeds 0 1] [--steps 60]
                                        [--axes A,B,C,D] [--quick]
⚠ enwik8 must already be staged (serial, once):
    python -c "from chlu.data.corpora import stage_corpus; stage_corpus('enwik8')"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any, Dict, List, Optional

BANNER = "⛔ smoke-scale geometry sweep — NEVER a claim venue (charter §2)."

#: ⭐ The deploy shape the sweep is *about*. The sweep runs at smoke shapes, but
#: every byte number it reports is projected onto THIS geometry, because that is
#: the object being frozen. ⛔ ``addr_dim`` is pinned at 8 (Hub ruling R2,
#: 2026-08-13): flipping it refuses every banked journal and re-rolls the TTT
#: arm's ``eta*n/d`` stability criterion.
DEPLOY = dict(d_model=512, n_layers=12, addr_dim=8, payload_dim=4, capacity=32)


def store_state_floats(n_atoms: int, dim: int, capacity: int) -> int:
    """``CluStoreCell.cell_ledger()``'s ``state_floats``, from the config alone.

    ``V_theta`` deviation (centers ``n_atoms*dim`` + log_width + amp) plus the
    retained codebook ``capacity*dim``. Asserted against the built cell in
    ``tests/test_c3_geometry_freeze.py``.
    """
    return int(n_atoms * (dim + 2) + capacity * dim)


def deploy_state_bytes(n_atoms: int, *, dim: Optional[int] = None,
                       n_layers: Optional[int] = None,
                       capacity: Optional[int] = None) -> int:
    """Total (all layers) fp32 inference state at the DEPLOY shape. ⛔ Total, not
    per-layer, and no dtype normalisation — the convention ruled 2026-08-13."""
    dim = int(DEPLOY["addr_dim"] + DEPLOY["payload_dim"]) if dim is None else int(dim)
    n_layers = int(DEPLOY["n_layers"]) if n_layers is None else int(n_layers)
    capacity = int(DEPLOY["capacity"]) if capacity is None else int(capacity)
    return int(4 * n_layers * store_state_floats(int(n_atoms), dim, capacity))


def geometry_points(axes: str = "ABCD") -> List[Dict[str, Any]]:
    """The pre-registered grid (PREREG-GEOMETRY-SWEEP.md §1)."""
    pts: List[Dict[str, Any]] = []
    K = 32
    if "A" in axes:      # the byte / starvation ladder
        for n in (512, 1024, 2048, 3072, 4096, 8192):
            pts.append(dict(axis="A", n_atoms=n, capacity=K, payload_dim=4, n_layers=2))
    if "B" in axes:      # iso-byte spend: items vs atoms-per-item
        for k in (16, 64, 128):
            pts.append(dict(axis="B", n_atoms=2048, capacity=k, payload_dim=4, n_layers=2))
    if "C" in axes:      # dim (addr_dim HELD at 8)
        pts.append(dict(axis="C", n_atoms=2048, capacity=K, payload_dim=8, n_layers=2))
    if "D" in axes:      # depth linearity
        pts.append(dict(axis="D", n_atoms=2048, capacity=K, payload_dim=4, n_layers=4))
    return pts


#: ⭐ The write levers the **landed CSF3 runs 1/2/3 actually used**, read off
#: ``.claude/outputs/cluformer-pilot/csf3_outs/run2/pilot_pilot_seed0_S4.json``'s
#: own ``flags`` block (``atom_place_radius 0.3``, ``write_margin 0.6``), not off a
#: report. ⚠ ``write_inner_steps`` is **40** there and is scaled to 8 here purely
#: for wall-clock; that deviation is declared rather than silent.
#:
#: ⛔ **Why this arm exists and why it is not the default.** The w23 dimension-aware
#: atom floor compensates for *scattered* atom init: "the fraction landing near any
#: stored site DECAYS roughly geometrically per added dimension"
#: (``chlu/config.py`` L1518-1540). ``atom_place_radius > 0`` (H1b) re-draws the
#: **written slot's own atoms** into a ball around the incoming chunk's address at
#: write time, so the near-site atom count becomes ``atoms_per_item`` — a
#: *dimension-free* quantity — and the floor's mechanism is structurally absent.
#: The sweep therefore has to be run **both ways**: the pre-registered arm (as
#: filed, placement OFF) and this one (the config the ladder would actually run).
DEPLOYED_WRITE = dict(atom_place_radius=0.3, write_inner_steps=8)
DEPLOYED_STORE = dict(write_margin=0.6)


def overrides_for(pt: Dict[str, Any], *, steps: int, seed: int,
                  deployed_write: bool = False) -> Dict[str, Any]:
    """Smoke shapes + the geometry under test.

    ⭐ ``n_atoms`` is set by ``min_atoms_base = n_atoms/16`` (so the w23 geometric
    term ``base*sqrt(2)**8`` lands exactly on the target) **and** by
    ``atoms_per_item = n_atoms/K``, so both terms of the ``max`` agree and the
    resulting geometry is unambiguous. ``min_atoms_c`` is left at sqrt(2): the
    floor's *shape* is not what is being questioned, its *height* is.

    ``deployed_write`` adds :data:`DEPLOYED_WRITE` / :data:`DEPLOYED_STORE`.
    """
    n, K = int(pt["n_atoms"]), int(pt["capacity"])
    if n % K:
        raise ValueError(f"n_atoms {n} must be a multiple of capacity {K}")
    if n % 16:
        raise ValueError(f"n_atoms {n} must be a multiple of 16 (min_atoms_base)")
    store = dict(min_atoms_base=n // 16, min_atoms=1)
    memory = dict(chunk=32, address_steps=8, read_steps=8, traj_stride=4,
                  psi_hidden=16, write_inner_steps=2, write_n_perturb=4)
    if deployed_write:
        # ⛔ `update`, not `**` — DEPLOYED_WRITE deliberately OVERRIDES
        # `write_inner_steps`, and a duplicate-kwarg TypeError is a silly way to
        # lose an arm of the sweep (this fired in the test suite first).
        store.update(DEPLOYED_STORE)
        memory.update(DEPLOYED_WRITE)
    return dict(
        d_model=32, n_layers=int(pt["n_layers"]), seq_len=256, batch=2,
        steps=int(steps), warmup=max(1, int(steps) // 10),
        eval_batches=4, dyneval_batches=2, data_bytes=600_000,
        monitor_every=max(1, int(steps) - 1),      # watch at step 0 and the last step
        addr_dim=int(DEPLOY["addr_dim"]), payload_dim=int(pt["payload_dim"]),
        capacity=K, atoms_per_item=n // K,
        arms=("clu_store", "none"),
        # ⛔ DECLARED non-compliant by construction: the point of the sweep is to
        # measure geometries on both sides of the ceiling, so the ledger records
        # `enforced: false` rather than refusing to run.
        enforce_state_byte_budget=False,
        store=store,
        memory=memory,
        seed=int(seed),
    )


def _last_watch(hist: Dict[str, Any]) -> Dict[str, Any]:
    w = [r for r in (hist.get("store_health") or []) if r.get("tag") == "trained"]
    return w[-1] if w else {}


def _untrained_watch(hist: Dict[str, Any]) -> Dict[str, Any]:
    w = [r for r in (hist.get("store_health") or []) if r.get("tag") == "untrained"]
    return w[0] if w else {}


def run_point(pt: Dict[str, Any], seed: int, *, steps: int,
              deployed_write: bool = False) -> Dict[str, Any]:
    """One (geometry, seed): train clu_store + none, evaluate, read the watch."""
    import jax

    from chlu.experiments.exp_cluformer_pilot import (
        _data, _eval_batches, _train_batches, make_config,
    )
    from chlu.training.train_cluformer import (
        build_arm, calibrate_phi_gain, evaluate, solve_arms, train_arm,
    )

    ov = overrides_for(pt, steps=steps, seed=seed, deployed_write=deployed_write)
    pcfg = make_config("toy", seed, ov)
    scfg = pcfg.store_cfg()
    if int(scfg.n_atoms) != int(pt["n_atoms"]):
        raise AssertionError(
            f"geometry did not resolve: asked n_atoms={pt['n_atoms']}, got "
            f"{scfg.n_atoms} (capacity={scfg.capacity}, "
            f"atoms_per_item={scfg.atoms_per_item}, "
            f"min_atoms_base={scfg.min_atoms_base})")

    tr, _va, te = _data(pcfg)
    key = jax.random.PRNGKey(1000 + seed)
    k_cal, k_solve, k_model = jax.random.split(key, 3)
    calib = _train_batches(tr, pcfg)[0][0]
    pcfg.memory = dict(pcfg.memory)
    pcfg.memory["phi_gain"] = calibrate_phi_gain(pcfg, calib, key=k_cal)

    specs, ledger = solve_arms(pcfg, k_solve)
    batches = _train_batches(tr, pcfg)
    ev = _eval_batches(te, pcfg, pcfg.eval_batches)

    row: Dict[str, Any] = {
        "axis": pt["axis"], "seed": seed, "steps": steps,
        "deployed_write": bool(deployed_write),
        "atom_place_radius": float(pcfg.memory_cfg().atom_place_radius),
        "write_inner_steps": int(pcfg.memory_cfg().write_inner_steps),
        "n_atoms": int(scfg.n_atoms), "capacity": int(scfg.capacity),
        "atoms_per_item": int(scfg.atoms_per_item),
        "addr_dim": int(scfg.addr_dim), "payload_dim": int(scfg.payload_dim),
        "dim": int(scfg.dim), "n_layers": int(pcfg.n_layers),
        "smoke_cell_state_floats": int(ledger["clu_store"]["state_floats"]),
        "smoke_cell_state_bytes": int(ledger["clu_store"]["state_bytes"]),
        "smoke_cell_params": int(ledger["clu_store"]["params"]),
        "ttt_k_n": list(specs["ttt_matched"].ttt_shape or ()),
        # ⭐ the number the freeze is actually about: this geometry AT DEPLOY.
        "deploy_total_state_bytes": deploy_state_bytes(
            int(scfg.n_atoms), dim=int(scfg.dim)),
    }
    for arm in ("clu_store", "none"):
        m = build_arm(arm, pcfg, specs, key=k_model)
        t0 = time.time()
        m, hist = train_arm(arm, m, pcfg, iter(batches), log_every=10 ** 9)
        train_s = time.time() - t0
        ev_row = evaluate(m, pcfg, iter(ev))
        row[f"{arm}_bpc"] = float(ev_row["bpc"])
        row[f"{arm}_train_s"] = float(train_s)
        row[f"{arm}_s_per_step"] = float(train_s / max(steps, 1))
        row[f"{arm}_plan_frac"] = float(hist.get("plan_pass_frac", float("nan")))
        row[f"{arm}_final_nll"] = float(hist["loss_history"][-1])
        if arm == "clu_store":
            w0, w1 = _untrained_watch(hist), _last_watch(hist)
            row["depth_untrained"] = w0.get("depth_median")
            row["depth_trained"] = w1.get("depth_median")
            row["depth_ratio_vs_untrained"] = w1.get("depth_ratio_vs_untrained")
            row["spread_untrained"] = w0.get("qstar_payload_spread")
            row["spread_trained"] = w1.get("qstar_payload_spread")
            row["spread_ratio_vs_untrained"] = w1.get("spread_ratio_vs_untrained")
            row["n_live_untrained"] = w0.get("n_live")
            row["n_live_trained"] = w1.get("n_live")
    row["dividend_bpc"] = row["none_bpc"] - row["clu_store_bpc"]
    print(f"[sweep] axis {row['axis']} n_atoms {row['n_atoms']:>5} K {row['capacity']:>4} "
          f"dim {row['dim']:>2} L {row['n_layers']} seed {seed} | "
          f"deploy {row['deploy_total_state_bytes']:>10,} B | "
          f"bpc clu {row['clu_store_bpc']:.4f} none {row['none_bpc']:.4f} "
          f"(div {row['dividend_bpc']:+.4f}) | s/step {row['clu_store_s_per_step']:.2f} | "
          f"depth x{row.get('depth_ratio_vs_untrained')} n_live "
          f"{row.get('n_live_trained')}", flush=True)
    return row


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-collapsed view + the two pre-registered ratios (G2, G3, G4)."""
    def key(r):
        return (r["axis"], r["n_atoms"], r["capacity"], r["dim"], r["n_layers"])

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    def agg(rs, field):
        vals = [r[field] for r in rs
                if isinstance(r.get(field), (int, float))
                and not (isinstance(r[field], float) and math.isnan(r[field]))]
        if not vals:
            return {"mean": None, "spread": None, "n": 0}
        return {"mean": statistics.fmean(vals),
                "spread": (max(vals) - min(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals)}

    per_geom = {}
    for k, rs in groups.items():
        per_geom["|".join(map(str, k))] = {
            "axis": k[0], "n_atoms": k[1], "capacity": k[2], "dim": k[3],
            "n_layers": k[4],
            "deploy_total_state_bytes": rs[0]["deploy_total_state_bytes"],
            "clu_bpc": agg(rs, "clu_store_bpc"), "none_bpc": agg(rs, "none_bpc"),
            "dividend_bpc": agg(rs, "dividend_bpc"),
            "s_per_step": agg(rs, "clu_store_s_per_step"),
            "depth_ratio": agg(rs, "depth_ratio_vs_untrained"),
            "spread_ratio": agg(rs, "spread_ratio_vs_untrained"),
            "n_live_trained": agg(rs, "n_live_trained"),
            "seeds": sorted(r["seed"] for r in rs),
        }
    return {"banner": BANNER, "per_geometry": per_geom}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output JSON path")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--axes", default="ABCD")
    ap.add_argument("--deployed-write", action="store_true",
                    help="add the landed run-1/2/3 write levers (atom_place_radius "
                         "0.3, write_margin 0.6) — see DEPLOYED_WRITE")
    ap.add_argument("--quick", action="store_true",
                    help="one seed, 6 steps, axis A only — a plumbing check")
    a = ap.parse_args(argv)
    if a.quick:
        a.seeds, a.steps, a.axes = [0], 6, "A"

    print("#" * 74)
    print(BANNER)
    print("#" * 74, flush=True)
    pts = geometry_points(a.axes)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for pt in pts:
        for seed in a.seeds:
            rows.append(run_point(pt, seed, steps=a.steps,
                                  deployed_write=a.deployed_write))
            with open(a.out, "w") as fh:      # incremental: a kill loses nothing
                json.dump({"banner": BANNER, "deploy": DEPLOY, "steps": a.steps,
                           "seeds": a.seeds, "axes": a.axes,
                           "deployed_write": bool(a.deployed_write),
                           "rows": rows,
                           "summary": summarise(rows),
                           "wall_s": time.time() - t0}, fh, indent=2)
    print(f"wrote {a.out} ({len(rows)} rows, {time.time() - t0:.0f}s)")
    print(BANNER)
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
