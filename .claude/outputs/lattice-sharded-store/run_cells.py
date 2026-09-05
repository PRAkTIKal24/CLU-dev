"""Staged runner for the w25 sharded-store 2x2 discriminator.

One (cell, arm, seed) unit of work per line, appended to a per-worker JSONL as it
completes, so a long sweep survives interruption and partial results are always
readable. Each worker uses ~1 core (the read is a 1200-step sequential lax.scan,
latency-bound), so 4 workers run near-linearly on this 8-core box.

    PYTHONPATH=/Users/user/Desktop/CHLU-shard python run_cells.py \
        --out w1.jsonl --work "6:64:2/monolithic/0,6:64:2/sharded_matched/0"
"""

import argparse
import json
import os
import sys
import time

from chlu.config import get_default_config
from chlu.experiments.exp_designed_mechanism import _replace
from chlu.experiments.exp_sharded_store import evaluate_cell, parse_cell

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True,
                    help="comma-separated cell/arm/seed triples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-local", action="store_true")
    ap.add_argument("--nq", type=int, default=None,
                    help="override queries per item (declared read-cost cut)")
    args = ap.parse_args()

    cfg = get_default_config()
    ss = cfg.experiment_sharded_store
    ss.atom_init_local = bool(args.init_local)
    dm = _replace(
        cfg.experiment_designed_mechanism,
        n_query_per_item=args.nq or ss.n_query_per_item,
    )
    out = os.path.join(HERE, args.out)

    for spec in args.work.split(","):
        cell, arm, seed = spec.strip().split("/")
        d, K, n = parse_cell(cell)
        t0 = time.time()
        rec = evaluate_cell(
            arm, d, K, n, int(seed), dm, ss,
            # the blank makes a PASS leak-immune; a cell that misses the bar is a
            # fail whatever the blank does (the w24 "if_pass" rule)
            with_blank="if_pass",
        )
        rec["cell"] = cell
        rec["atom_init_local"] = bool(args.init_local)
        rec["n_query_per_item"] = int(dm.n_query_per_item)
        rec["seconds_total"] = time.time() - t0
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        w = rec["written"]
        print(
            f"[{cell}] {arm} s={seed} local={args.init_local} "
            f"strict={w['strict_success_rate']:.4f} "
            f"basin={w['basin_success_rate']:.4f} "
            f"route={w['route_accuracy']:.4f} "
            f"payerr={w['payload_abs_err_mean']:.4f} "
            f"atoms={rec['atoms_total']} N={rec['n_shards']} "
            f"sepU={rec['union_separation']:.3f} "
            f"sepS={rec['within_shard_separation_min']:.3f} "
            f"wloss={rec['write_loss_final']:.2e} "
            f"blank={rec.get('value_blank_ok')} "
            f"[{rec['seconds_total']:.0f}s]",
            flush=True,
        )
    print("worker complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
