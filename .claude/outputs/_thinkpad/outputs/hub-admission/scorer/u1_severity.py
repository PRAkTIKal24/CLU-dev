"""UNIT 1 -- SS3 step 2 / falsifier SS5.10.  Does the request cross the zero level set?

Symbol-mode ``out_of_range`` at R8 severities w in {2,3,4,7,13}, ONE FACTOR AT A TIME,
on val corners.  Reported at both live gate points, per well, with the per-term residual.
Plus the two other symbol-mode invalid kinds that the FULL store can express:
``unseen_value`` (w = 0.5, interpolated between adjacent grid values) and
``duplicate_factor`` (interface grammar, reported as such).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-admission"
sys.path.insert(0, str(OUT / "scorer"))
sys.path.insert(0, os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer"))

import admission_lib as AL          # noqa: E402
import refusal_lib as R             # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, BUDGET = 0, "A"
N_PROBE = int(os.environ.get("N_PROBE", "256"))
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]

ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, book, grid = built["land"], built["book"], ctx["grid"]
log(f"land: K={land.store.n_wells} N={land.store.n_atoms} sigma={np.asarray(land.sigma)}")

protos = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])
val_rows = np.asarray(ctx["val_rows"], np.int64)[:N_PROBE]
base = requests_from_rows(grid, val_rows)
names = list(grid.names)

rec = {"schema": "admission-u1-severity/1", "seed": SEED, "budget": BUDGET,
       "n_probe": int(len(base)), "severities": SEVERITIES, "names": names,
       "sigma": [float(s) for s in np.asarray(land.sigma)],
       "certified_radius": [float(x) for x in AL.certified_radius(land)],
       "atom_cloud_spread": AL.atom_cloud_spread(land),
       "rows": []}
log("atom spread vs sigma:")
for w in rec["atom_cloud_spread"]["wells"]:
    log(f"   well {w['well']}: sigma={w['sigma_nn_median']:.3f} cloudRMS={w['cloud_rms_radius']:.3f} "
        f"(={w['rms_over_sigma']:.2f} sigma)  perChanStd={w['per_channel_std']:.4f}")


def offgrid_value(f: int, w: float) -> float:
    """The R8 severity-w value on factor f: v = vals[-2] + w*(vals[-1]-vals[-2])."""
    v = np.asarray(grid.values[f], np.float64)
    return float(v[-2] + w * (v[-1] - v[-2]))


def midgrid_value(f: int, req) -> float:
    """``unseen value``: halfway between the requested grid value and its neighbour."""
    v = np.asarray(grid.values[f], np.float64)
    cur = float([t for t in req["request"] if int(t["factor_index"]) == f][0]["value"])
    i = int(np.argmin(np.abs(v - cur)))
    j = min(i + 1, len(v) - 1) if i + 1 < len(v) else i - 1
    return float(0.5 * (v[i] + v[j]))


def perturb(reqs, f: int, newval_fn):
    out = []
    for r in reqs:
        q = copy.deepcopy(r)
        for t in q["request"]:
            if int(t["factor_index"]) == f:
                t["value"] = float(newval_fn(f, q))
        q.pop("class_indices", None)
        out.append(q)
    return out


def duplicate(reqs, f: int):
    out = []
    for r in reqs:
        q = copy.deepcopy(r)
        term = [t for t in q["request"] if int(t["factor_index"]) == f][0]
        v = np.asarray(grid.values[f], np.float64)
        other = float(v[(int(np.argmin(np.abs(v - float(term["value"])))) + 1) % len(v)])
        q["request"] = list(q["request"]) + [{"factor_index": int(f), "value": other}]
        q.pop("class_indices", None)
        out.append(q)
    return out


def measure(tag, kind, factor, sev, reqs):
    log_w, n_terms, off = R.term_experts(book, grid, reqs)
    res = R.per_term_residuals(land, AL.launch_from_terms_mode(
        land, book, log_w, n_terms, "posterior_mean")["x0"], log_w, n_terms)
    row = {"tag": tag, "kind": kind, "factor": (names[factor] if factor is not None else None),
           "factor_index": factor, "severity": sev, "n": int(len(reqs)),
           "n_offgrid_terms": int(off["n_offgrid_terms"])}
    with np.errstate(invalid="ignore"):
        row["residual_max_mean"] = float(np.nanmean(np.nanmax(res, axis=1)))
        row["residual_requested_term_mean"] = (
            float(np.nanmean(res[:, factor])) if factor is not None and factor < res.shape[1] else None)
    pts = {"posterior_mean": AL.launch_from_terms_mode(land, book, log_w, n_terms, "posterior_mean")["x0"],
           "clamp_atom": AL.launch_from_terms_mode(land, book, log_w, n_terms, "clamp_atom")["x0"],
           "r8_affine": protos.embed_many(reqs).reshape(len(reqs), -1)}
    for m, X in pts.items():
        row[m] = AL.gate_summary(land, X)
    rec["rows"].append(row)
    log(f"  {tag:34s} pm: acc={row['posterior_mean']['accept_rate']:.3f} "
        f"Emax={row['posterior_mean']['E_max_over_wells_mean']:+9.3g} "
        f"d/s={row['posterior_mean']['d_over_sigma_min_mean']:.2f} | "
        f"r8: acc={row['r8_affine']['accept_rate']:.3f} "
        f"Emax={row['r8_affine']['E_max_over_wells_mean']:+9.3g} "
        f"d/s={row['r8_affine']['d_over_sigma_min_mean']:.2f} | "
        f"clamp acc={row['clamp_atom']['accept_rate']:.3f} | res={row['residual_max_mean']:.4g}")


log("")
log("=== BASELINE (valid) ===")
measure("valid", "valid", None, None, base)

log("")
log("=== OUT OF RANGE, one factor at a time, R8 severity w ===")
for f in range(len(names)):
    for w in SEVERITIES:
        measure(f"oor:{names[f]}:w={w:g}", "out_of_range", f, w,
                perturb(base, f, lambda ff, q, w=w: offgrid_value(ff, w)))

log("")
log("=== UNSEEN VALUE (interpolated, w = 0.5) ===")
for f in range(len(names)):
    measure(f"unseen:{names[f]}", "unseen_value", f, 0.5, perturb(base, f, midgrid_value))

log("")
log("=== DUPLICATE FACTOR (interface grammar) ===")
for f in range(len(names)):
    measure(f"dup:{names[f]}", "duplicate_factor", f, None, duplicate(base, f))

json.dump(rec, open(OUT / "U1_SEVERITY_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'U1_SEVERITY_seed0_A.json'}")
