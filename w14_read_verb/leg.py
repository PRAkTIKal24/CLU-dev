"""Run one CMAZE-2b read-verb leg.

    python -m w14_read_verb.leg --seed 0 --which val --tag dev1 [--grid rebased|w9]

``--which val`` is a DEV leg and is free.  ``--which test`` is a SCORED leg: it refuses
to run unless BOTH gate files are present and a ledger row for this hub's target has
already been written, because the standing rule is that the row is written BEFORE the
run, not after it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

from w14_read_verb import pipeline

BANKS = "C:/chlu/outputs/w14-request-layer-fix/banks"
POOL = "C:/chlu/outputs/w14-read-verb/pool"
OUT = "C:/chlu/outputs/w14-read-verb/runs"
GATE_A = "C:/chlu/outputs/w14-request-layer-fix/REQUEST-LAYER-READY.json"
GATE_B = "C:/chlu/outputs/w14-refusal-displays-closeout/CLOSEOUT.md"
LEDGER = "C:/chlu/SCORED-LEDGER.md"

#: GRID_W9  -- w9's unchanged absolute words.  One grid for all three seeds.
GRID_W9 = {0: ((30, 45, 60), (38, 53)), 1: ((30, 45, 60), (38, 53)),
           2: ((30, 45, 60), (38, 53))}
#: GRID_REBASED -- the words re-derived per seed by ``w14_request_layer.budget`` so they
#: sit at the same percentiles of the AFTER bank's route-length distribution that w9's
#: words sat at in the BEFORE bank's.  Quoted from the gate file's BUDGET_GRID block,
#: never recomputed here.
GRID_REBASED = {0: ((55, 85, 105), (70, 95)), 1: ((60, 85, 100), (72, 92)),
                2: ((55, 75, 90), (65, 82))}
GRIDS = {"w9": GRID_W9, "rebased": GRID_REBASED}


def load_pool(max_sites=None):
    out = []
    for p in sorted(glob.glob(os.path.join(POOL, "pool_shard*.pkl"))):
        with open(p, "rb") as fh:
            out.extend(pickle.load(fh)["sites"])
    return out if max_sites is None else out[:max_sites]


def load_bank(seed, n_sites=None):
    with open(os.path.join(BANKS, "bank_after_seed%d.pkl" % seed), "rb") as fh:
        bank = pickle.load(fh)["sites"]
    return bank if n_sites is None else bank[:n_sites]


def check_gates(which, tag):
    for p in (GATE_A, GATE_B):
        if not os.path.exists(p):
            raise SystemExit("GATED: missing %s" % p)
    with open(GATE_A) as fh:
        g = json.load(fh)
    if g.get("status") != "ready":
        raise SystemExit("GATED: %s is not ready" % GATE_A)
    if not all(v.get("pass") for v in g["ACCEPTANCE"].values()):
        raise SystemExit("GATED: an ACCEPTANCE item does not pass")
    if which != "test":
        return
    if not os.path.exists(LEDGER):
        raise SystemExit("SCORED run refused: no %s" % LEDGER)
    with open(LEDGER, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    if "read-verb" not in txt or tag not in txt:
        raise SystemExit("SCORED run refused: no ledger row for read-verb / %s" % tag)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", default="val")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--grid", default="rebased", choices=("w9", "rebased"))
    ap.add_argument("--n-atoms", type=int, default=31)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--ae-steps", type=int, default=12000)
    ap.add_argument("--bl-steps", type=int, default=4000)
    ap.add_argument("--lowrank-atoms", type=int, default=7)
    ap.add_argument("--pool-sites", type=int, default=None)
    ap.add_argument("--n-sites", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args(argv)

    check_gates(a.which, a.tag)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "ae"), exist_ok=True)
    budgets, interp = GRIDS[a.grid][a.seed]
    bank = load_bank(a.seed, a.n_sites)
    pool = load_pool(a.pool_sites)
    print("leg %s  seed %d  which %s  grid %s %s/%s  bank %d  pool %d  %s"
          % (a.tag, a.seed, a.which, a.grid, budgets, interp, len(bank), len(pool),
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), flush=True)

    # the encoder depends only on (seed, grid, split, pool, steps) -- never on a store
    # configuration and never on the leg -- so it is cached and shared across legs.
    cache = os.path.join(OUT, "ae", "ae_seed%d_%s_%d_%d.eqx"
                         % (a.seed, a.grid, a.ae_steps, len(pool)))
    res, _art = pipeline.run(bank, a.seed, budgets, interp, pool_sites=pool,
                             n_atoms=a.n_atoms, K=a.K, ridge=a.ridge,
                             ae_steps=a.ae_steps, bl_steps=a.bl_steps,
                             which=a.which, ae_cache=cache,
                             do_baselines=not a.no_baselines,
                             do_controls=not a.no_controls,
                             lowrank_atoms=a.lowrank_atoms)
    res["tag"] = a.tag
    res["grid_name"] = a.grid
    path = os.path.join(OUT, "%s_seed%d_%s.json" % (a.tag, a.seed, a.which))
    pipeline.to_json(res, path)
    print("wrote", path, "in %.0fs" % res["seconds"], flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
