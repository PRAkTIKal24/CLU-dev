"""The environment's own columns on a CMAZE-2 bank -- BEFORE or AFTER the repair.

Usage:
  python -m w14_request_layer.measure <before|after> <seed_index> <pool.pkl|-> \
         <bank_dir> <out.json> [<before_bank.pkl for the rebased grid|->]

Every column is ``w9_maze.envcols``'s, called unmodified, so the numbers are like-for-like
with wave 9's.  The planners are ``w8_envscout.screen``'s.  The judge is
``w8_envscout.maze_colour.judge`` -- imported, not wrapped, not reimplemented -- and the
run records the sha256 of the file it lives in.

Three budget conditions are reported and they are three different questions:

  GRID_W9        w9's UNCHANGED absolute budget words 30/45/60, interp 38/53.  This is
                 the primary column: nothing but the acceptance rule differs from w9.
  GRID_REBASED   the same words re-derived by ``w14_request_layer.budget`` so they sit at
                 the same percentiles of the route-length distribution that w9's words sat
                 at on w9's bank.  Published beside the primary, not instead of it.
  SLACK10        every kept pair at CMAZE-1's own rule, ``opt + ceil(0.10 * opt)``.  This
                 removes the budget grid from the comparison entirely and is the column
                 that is directly comparable with the published 0.4980 and with w9's own
                 sentence that the same measurement left blind-sequential at 1.0000.

Two fragment-lookup quantities are computed and they are DIFFERENT quantities:
  FRAG-r1 / r2         SAME-MAZE -- the table holds routes through the very maze being
                       evaluated.  w9 measured 0.0741 / 0.5105.
  FRAG-cross-r1 / r2   CROSS-MAZE -- the table is built from a POOL of sites from a
                       disjoint generator-seed range.  This is the like-for-like with
                       CMAZE-1's published 0.0006 / 0.0000.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

from w9_maze import envcols, splits
from w9_maze.task import BUDGET_GRID, BUDGET_INTERP, Request
from w14_request_layer.budget import rebase
from w14_request_layer.task2 import key_count_histogram

ROWS = ("seen", "unseen_pair", "nway", "budget_interp")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _rows_for(bank, split, row, budgets, interp):
    """``splits.row_requests`` with the interpolation words made explicit."""
    if row != "budget_interp":
        return splits.row_requests(bank, split, row, budgets)
    held = {m: (split.unseen_pair.get(m, []) + split.nway.get(m, []))
            for m in split.train_pairs}
    return splits.requests_for(bank, held, tuple(interp), True)


def measure(bank, pool_sites, seed, which_split, budgets, interp, tag):
    val, test = splits.make_split(bank, seed)
    split = val if which_split == "val" else test
    train_rows = splits.train_items(bank, split, tuple(budgets))
    row_sets = {r: _rows_for(bank, split, r, tuple(budgets), tuple(interp))
                for r in ROWS}
    out = {"tag": tag, "which_split": which_split, "n_mazes": len(bank),
           "budgets": list(budgets), "interp": list(interp),
           "n_train_requests": len(train_rows),
           "n_rows": {r: len(v) for r, v in row_sets.items()}}
    for r in ROWS:
        rows = row_sets[r]
        if not rows:
            out[r] = {"n": 0, "note": "DECLARED EMPTY: no request of this row is "
                                      "writable at these budget words"}
            continue
        t0 = time.time()
        out[r] = {"blind": envcols.blind_columns(bank, rows),
                  "bend": envcols.bend_stats(bank, rows),
                  "seconds": time.time() - t0}
        print("   [%s/%s] %-14s n=%-5d %.0fs" % (tag, which_split, r, len(rows),
                                                 time.time() - t0), flush=True)
    eval_rows = row_sets["unseen_pair"] + row_sets["nway"]
    out["n_fragment_eval_rows"] = len(eval_rows)
    if not eval_rows or not train_rows:
        out["fragment_same_maze"] = {"DECLARED_NOT_RUN": "empty train or eval row set"}
        out["fragment_cross_maze"] = {"DECLARED_NOT_RUN": "empty eval row set"}
        return out
    t0 = time.time()
    out["fragment_same_maze"] = envcols.fragment_columns(bank, train_rows, eval_rows)
    print("   [%s/%s] fragment same-maze %.0fs" % (tag, which_split, time.time() - t0),
          flush=True)
    if pool_sites:
        t0 = time.time()
        out["fragment_cross_maze"] = envcols.fragment_cross_maze(bank, pool_sites,
                                                                 eval_rows)
        print("   [%s/%s] fragment cross-maze %.0fs" % (tag, which_split,
                                                        time.time() - t0), flush=True)
    else:
        out["fragment_cross_maze"] = {"DECLARED_NOT_RUN":
                                      "no disjoint pool bank was given"}
    return out


def all_pairs_slack10(bank):
    """Every kept pair at CMAZE-1's OWN budget rule, ``opt + ceil(0.10 * opt)``."""
    rows = []
    for m, site in enumerate(bank):
        for (i, j) in site.pairs:
            o = site.opt[(i, j)]
            rows.append((Request(m, i, j, int(o + int(np.ceil(0.10 * o)))), m, i, j))
    return {"n": len(rows), "blind": envcols.blind_columns(bank, rows),
            "bend": envcols.bend_stats(bank, rows)}


def budget_fit(bank, budgets):
    """How many kept pairs are WRITABLE at each budget word -- a request-layer statistic."""
    opt = np.array([s.opt[p] for s in bank for p in s.pairs], float)
    return {"n_pairs": int(opt.size), "opt_mean": float(opt.mean()),
            "opt_median": float(np.median(opt)), "opt_max": float(opt.max()),
            "fraction_writable": {str(b): float((opt <= b).mean()) for b in budgets}}


def main(which, seed_index, pool_path, bank_dir, out_path, before_path="-"):
    import w8_envscout.maze_colour as mc
    import w9_maze.envcols as ec
    from w8_envscout.screen import PLANNERS
    bank_path = os.path.join(bank_dir, "bank_%s_seed%d.pkl" % (which, int(seed_index)))
    blob = load(bank_path)
    bank = blob["sites"]
    pool = load(pool_path)["sites"] if pool_path not in ("-", "") else None
    t0 = time.time()
    res = {"which": which, "seed_index": int(seed_index), "bank": bank_path,
           "bank_seed0": blob["seed0"], "bank_n_tried": blob["n_tried"],
           "bank_acceptance": len(bank) / blob["n_tried"],
           "pool_bank": pool_path, "n_pool_sites": (len(pool) if pool else 0),
           "planners": list(PLANNERS),
           "key_count_histogram": {str(k): v for k, v in
                                   key_count_histogram(bank).items()},
           "n_routes_in_bank": sum(len(s.pairs) for s in bank),
           "judge": {"module": mc.judge.__module__, "qualname": mc.judge.__qualname__,
                     "source_file": os.path.relpath(mc.__file__, os.getcwd()),
                     "sha256": _sha256(mc.__file__)},
           "envcols_source_sha256": _sha256(ec.__file__),
           #: the exact integer judge does not move.
           "tolerance_used": None}

    grids = {"GRID_W9": (BUDGET_GRID, BUDGET_INTERP)}
    if before_path not in ("-", ""):
        g, i, diag = rebase(load(before_path)["sites"], bank)
        res["rebase"] = diag
        grids["GRID_REBASED"] = (g, i)
    res["budget_fit"] = {tag: budget_fit(bank, tuple(b) + tuple(v))
                         for tag, (b, v) in grids.items()}
    res["all_pairs_slack10"] = all_pairs_slack10(bank)
    print("   SLACK10 blind-sequential %.4f"
          % res["all_pairs_slack10"]["blind"][
              "BLIND-SEQUENTIAL(key order from A)"]["validity"], flush=True)
    res["measurements"] = {}
    for tag, (b, v) in grids.items():
        for w in ("test", "val"):
            res["measurements"]["%s/%s" % (tag, w)] = measure(
                bank, pool, int(seed_index), w, b, v, tag)
    res["seconds"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out_path, "%.0fs" % res["seconds"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
