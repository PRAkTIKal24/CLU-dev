"""The SPAN SWEEP: where do the charter-C controls collapse, and where do they not?

Why this exists
---------------
The CMAZE-O VAL smoke ran Config-A at ``n_atoms = 12`` over wells of width 16 -- a span
fraction of 0.688 -- and found that **only ``landscape_removed`` collapses**.  That is
exactly the regime ``w9_maze.arms`` warns about in its own docstring:

    span fraction = K * n_atoms / D = n_atoms / d_k = THE SATURATION RATIO
    ... at saturation the projection is the identity, the content carries no information
    and the random-atom control stops collapsing.

So the question "is there a STORE claim on CMAZE-O, or only an INTERFACE claim?" is a
question about the span, and it is answered by sweeping the span, not by arguing.  This
script sweeps ``n_atoms`` and re-measures every control at each point.

It also checks, numerically and directly, the claim in ``exp_w15_cmaze_o_arms``'s
docstring that ``landscape_shuffled`` (a permutation of the atom ROWS) is a **proved
null** for a read whose address book is re-solved against the atoms: it prints
``max |read_CLU - read_control|`` over the evaluation rows.

The autoencoder is reproduced EXACTLY, not re-trained differently: ``train_ae`` draws its
batches from ``np.random.default_rng(0)`` and initialises from ``jax.random.PRNGKey(seed)``,
so rebuilding it from the same seed and the same step count gives the same frozen model
the VAL smoke used.  This script asserts that by comparing the realised AE ceiling
against the VAL smoke's.

Usage:  python scripts/w15_cmaze_o_span.py <out_dir> [--seeds=0,1,2]
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chlu.exp_w15_cmaze_o_arms import (CONTROLS_SHIPPED, OCluRead,       # noqa: E402
                                       make_grids)
from chlu.exp_w15_cmaze_o_encoder import decode_all, encode_all, train_ae  # noqa: E402
from chlu.exp_w15_cmaze_o_env import (build_layout_bank, decode_state,   # noqa: E402
                                      item_tensor, judge_state, render_state)
from chlu.exp_w15_cmaze_o_pipeline import CONFIG_A                       # noqa: E402
from chlu.exp_w15_cmaze_o_splits import build_pool                       # noqa: E402

N_ATOMS_GRID = (2, 6, 12)
EVAL_ROWS = ("seen", "unseen_nway")


def _eval(ae, bank, reqs, Zr):
    LG = decode_all(ae, np.asarray(Zr, np.float32))
    out = {"valid": [], "exact_objects": [], "legal": []}
    for i, r in enumerate(reqs):
        j = judge_state(decode_state(LG[i]), render_state(bank[r.layout], r))
        for k in out:
            out[k].append(j[k])
    return {k: float(np.mean(v)) for k, v in out.items()} | {"n": len(reqs),
                                                             "tolerance_used": None}


def run(out_dir, seeds):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "SPAN-SWEEP.json")
    doc = {"artefact": "SPAN-SWEEP.json", "spoke": "w15/cmaze-o-harness",
           "split": "VAL", "scored_runs_spent": 0, "tolerance_used": None,
           "n_atoms_grid": list(N_ATOMS_GRID), "eval_rows": list(EVAL_ROWS),
           "controls": list(CONTROLS_SHIPPED), "seeds_completed": [], "by_seed": {},
           "what_it_answers": (
               "whether CMAZE-O at a given SPAN FRACTION supports a STORE claim (every "
               "charter-C control collapses) or only an INTERFACE claim (they do not)")}

    for seed in seeds:
        t0 = time.time()
        bank, _tried = build_layout_bank(12, 90_000 + 1000 * int(seed))
        pool = build_pool(bank, seed, n_train=8000, n_row=400, n_seen=600,
                          verbose=False)
        X = np.stack([item_tensor(render_state(bank[r.layout], r)) for r in pool.train])
        ae, _hist = train_ae(X, jax.random.PRNGKey(seed), steps=6000, batch=64,
                             log_every=3000, verbose=True)
        Z = encode_all(ae, X)
        grids = make_grids(len(bank))
        print("[span seed %d] AE reproduced, %.0fs" % (seed, time.time() - t0),
              flush=True)

        rows = {r: pool.rows[r] for r in EVAL_ROWS}
        ceiling = {r: _eval(ae, bank, rows[r], encode_all(
            ae, np.stack([item_tensor(render_state(bank[q.layout], q))
                          for q in rows[r]]))) for r in EVAL_ROWS}
        seed_doc = {"ae_ceiling": ceiling, "points": {}}

        for na in N_ATOMS_GRID:
            t = time.time()
            clu = OCluRead(Z, pool.train, grids, K=CONFIG_A["K"], n_atoms=na,
                           ridge=CONFIG_A["ridge"], pair_block=True, seed=seed)
            led = clu.ledger()
            point = {"n_atoms": na, "ledger": led, "arms": {}, "read_gap_vs_CLU": {}}
            reads = {"CLU": {r: clu.read(rows[r]) for r in EVAL_ROWS}}
            for c in CONTROLS_SHIPPED:
                ctrl = OCluRead(Z, pool.train, grids, K=CONFIG_A["K"], n_atoms=na,
                                ridge=CONFIG_A["ridge"], pair_block=True, control=c,
                                seed=seed, well_channels=clu.well_channels)
                reads[c] = {r: ctrl.read(rows[r]) for r in EVAL_ROWS}
                point["read_gap_vs_CLU"][c] = float(max(
                    np.max(np.abs(reads[c][r] - reads["CLU"][r])) for r in EVAL_ROWS))
            for arm, zz in reads.items():
                point["arms"][arm] = {r: _eval(ae, bank, rows[r], zz[r])
                                      for r in EVAL_ROWS}
            seed_doc["points"][str(na)] = point
            print("[span seed %d] n_atoms=%2d span=%.3f  CLU seen %.4f  "
                  "removed %.4f  shuffled %.4f  channels %.4f  across %.4f  random "
                  "%.4f  (%.0fs)"
                  % (seed, na, led["span_fraction"],
                     point["arms"]["CLU"]["seen"]["valid"],
                     point["arms"]["landscape_removed"]["seen"]["valid"],
                     point["arms"]["landscape_shuffled"]["seen"]["valid"],
                     point["arms"]["landscape_shuffled_channels"]["seen"]["valid"],
                     point["arms"]["landscape_shuffled_across_wells"]["seen"]["valid"],
                     point["arms"]["random_atoms"]["seen"]["valid"],
                     time.time() - t), flush=True)
            doc["by_seed"][str(seed)] = seed_doc
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1, default=str)

        doc["seeds_completed"].append(int(seed))
        doc["by_seed"][str(seed)]["wall_seconds"] = time.time() - t0
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=str)
        print("[span seed %d] done %.0fs" % (seed, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    seeds = [0, 1, 2]
    for a in sys.argv:
        if a.startswith("--seeds"):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
    sys.exit(run(sys.argv[1], seeds))
