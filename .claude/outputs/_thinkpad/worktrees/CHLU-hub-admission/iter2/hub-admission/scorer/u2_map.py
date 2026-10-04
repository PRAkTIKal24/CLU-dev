"""UNIT 2 -- SS3 step 1.  THE FACTOR-TO-WELL MAP, over shuffled-label floors.

Method inherited from hub/refusal: mutual information against a SHUFFLED-LABEL FLOOR,
so a well that carries nothing reports ~0 rather than the positive bias finite-sample MI
always has.

The per-well observable is the store's OWN discrete address: the index of the nearest
ATOM of well k, for every store-train row.  That is the quantity the address book is a
distribution over, so "well k carries factor f" is measured on exactly the variable the
interface uses -- not on a re-encoded or re-projected one.

Output: FACTOR_WELL_MAP_seed0.json.  Without it, SS3 says exemplar mode and term
localisation are undefined and the run proceeds in symbol and free modes only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-admission"
sys.path.insert(0, str(OUT / "scorer"))
sys.path.insert(0, os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer"))

import refusal_lib as R             # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, BUDGET = 0, "A"
N_SHUFFLE = int(os.environ.get("N_SHUFFLE", "32"))
RNG = np.random.default_rng(20260917)


def mi_bits(y: np.ndarray, f: np.ndarray, ny: int, nf: int) -> float:
    """Plug-in MI in bits between two discrete vectors."""
    J = np.zeros((ny, nf), np.float64)
    np.add.at(J, (y, f), 1.0)
    J /= J.sum()
    py, pf = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    nz = J > 0
    return float((J[nz] * np.log2(J[nz] / (py @ pf)[nz])).sum())


def entropy_bits(f: np.ndarray, nf: int) -> float:
    p = np.bincount(f, minlength=nf).astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, grid = built["land"], built["store"], ctx["grid"]
names = list(grid.names)
cards = [int(c) for c in grid.cardinalities]
log(f"land: K={store.n_wells} N={store.n_atoms}  factors={names} cards={cards}")

X = np.asarray(ctx["X_store"], np.float32).reshape(ctx["pool"].size, -1)
F = np.asarray(grid.factor_indices_of(ctx["pool"]), np.int64)          # (n, 6)
n = X.shape[0]
log(f"pool rows = {n}")

# the per-well discrete address: nearest atom index, for every store-train row
addr = np.zeros((n, store.n_wells), np.int64)
for k, (w, a) in enumerate(zip(store.well_channels, store.atoms, strict=True)):
    A = np.asarray(a, np.float32)
    Z = X[:, np.asarray(w, np.int64)]
    best = np.empty(n, np.int64)
    for i in range(0, n, 4096):
        z = Z[i: i + 4096]
        d2 = ((z * z).sum(1)[:, None] - 2.0 * (z @ A.T) + (A * A).sum(1)[None, :])
        best[i: i + z.shape[0]] = d2.argmin(1)
    addr[:, k] = best
    log(f"  well {k}: |channels|={len(w)}  distinct atoms used = {np.unique(best).size}/{A.shape[0]}")

rec = {"schema": "admission-factor-well-map/1", "seed": SEED, "budget": BUDGET,
       "K_wells": int(store.n_wells), "n_atoms": int(store.n_atoms),
       "well_sizes": [int(len(w)) for w in store.well_channels],
       "names": names, "cards": cards, "n_rows": int(n), "n_shuffle": N_SHUFFLE,
       "observable": "index of the nearest ATOM of well k, per store-train row",
       "wells": [], "map": {}, "excess_mi_norm": []}

NA = int(store.n_atoms)
log("")
log("=== MI(nearest-atom index ; factor) over shuffled-label floors, bits ===")
hdr = "  well  " + "".join(f"{nm[:11]:>13s}" for nm in names)
log(hdr)
Mnorm = np.zeros((store.n_wells, len(names)), np.float64)
raw = np.zeros_like(Mnorm)
floor = np.zeros_like(Mnorm)
for k in range(store.n_wells):
    cells = []
    for f in range(len(names)):
        m = mi_bits(addr[:, k], F[:, f], NA, cards[f])
        fl = np.mean([mi_bits(addr[:, k], RNG.permutation(F[:, f]), NA, cards[f])
                      for _ in range(N_SHUFFLE)])
        H = entropy_bits(F[:, f], cards[f])
        raw[k, f], floor[k, f] = m, fl
        Mnorm[k, f] = (m - fl) / H
        cells.append(f"{Mnorm[k, f]:13.4f}")
    log(f"  {k:4d}  " + "".join(cells))
    rec["wells"].append({"well": k, "raw_mi_bits": [float(x) for x in raw[k]],
                         "shuffled_floor_bits": [float(x) for x in floor[k]],
                         "excess_mi_over_H": [float(x) for x in Mnorm[k]]})

rec["excess_mi_norm"] = [[float(x) for x in row] for row in Mnorm]
log("")
log("=== THE MAP: factor -> well(s) ===")
for f, nm in enumerate(names):
    order = np.argsort(-Mnorm[:, f])
    top = int(order[0])
    share = Mnorm[:, f] / max(Mnorm[:, f].sum(), 1e-12)
    # a well is ASSIGNED to the factor if it carries >= 50% of the top well's excess MI
    assigned = [int(k) for k in range(store.n_wells) if Mnorm[k, f] >= 0.5 * Mnorm[top, f]]
    rec["map"][nm] = {"factor_index": f, "argmax_well": top,
                      "argmax_excess_mi_over_H": float(Mnorm[top, f]),
                      "assigned_wells": assigned,
                      "share_per_well": [float(x) for x in share],
                      "margin_over_second": float(Mnorm[top, f] - Mnorm[int(order[1]), f])}
    log(f"  {nm:12s} -> well {top}  (excess MI/H = {Mnorm[top, f]:.4f}, "
        f"margin over 2nd = {rec['map'][nm]['margin_over_second']:+.4f}, "
        f"assigned = {assigned})")

# is the map a FUNCTION (each factor localised) or a smear?
tops = [rec["map"][nm]["argmax_well"] for nm in names]
rec["map_is_injective_on_argmax"] = bool(len(set(tops)) == len(tops))
rec["n_distinct_argmax_wells"] = int(len(set(tops)))
rec["localised_factors"] = [nm for nm in names
                            if len(rec["map"][nm]["assigned_wells"]) == 1]
log("")
log(f"distinct argmax wells = {rec['n_distinct_argmax_wells']} of {store.n_wells} "
    f"(factors = {len(names)}); singly-assigned factors = {rec['localised_factors']}")

json.dump(rec, open(OUT / "FACTOR_WELL_MAP_seed0.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'FACTOR_WELL_MAP_seed0.json'}")
