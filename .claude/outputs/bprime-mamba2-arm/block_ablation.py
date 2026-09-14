"""⭐ The DECLARED Mamba-2 block-level ablation (the "you hobbled it" answer).

Runs the SAME outer loop, the SAME F3 grid and the SAME scorer on
``aggregate@base`` for two configurations of the arm:

* **minimal** (the audited one) — SSD update only; no ``D`` skip, no ``z`` gate,
  no conv. The same minimality caption every arm in this rig carries.
* **+block** — ``use_D = True, gate_z = True`` (the block-level parts Mamba-2's
  published block has around the SSM). ⚠ Both cost **zero extra state bytes**:
  ``D`` and ``W_z`` are parameters and are already in F1.

⚠ Declared asymmetry of the +block reading: the ``D`` skip adds a query-dependent
path to ``full`` that the table launder cannot have, so its `full` is NOT strictly
comparable to its own launder. It IS comparable to the **raw-metric +0 B table**,
which is the load-bearing control, and that is the number this script reports.

Run from a checkout of `agent/experiment-engineer/bprime-mamba2-arm`:
    PYTHONPATH=. python .claude/outputs/bprime-mamba2-arm/block_ablation.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

# the worktree was removed after the branch ref was verified; point this at
# whichever checkout has `agent/experiment-engineer/bprime-mamba2-arm` out.
sys.path.insert(0, "/Users/user/Desktop/CHLU")

import jax  # noqa: E402

from chlu.eval.rivals import matched_table_rows, rival_arms, select_best  # noqa: E402
from chlu.eval.rivals.fit import LR_GRID_F3, RIVALS, WD_GRID_F3, fit_grid  # noqa: E402
from chlu.experiments.exp_bprime_rivals import aux_fit_examples, stream_tokens  # noqa: E402
from chlu.experiments.exp_memory_gym import _build_queries, _insertion_order  # noqa: E402
from chlu.experiments.memory_gym import (  # noqa: E402
    PRIMARY_METRIC,
    gym_config,
    make_gym_stream,
    score,
)
from chlu.core.clu_system import build_system  # noqa: E402

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
CONFIGS = {"minimal": {}, "+block(D,z)": {"use_D": True, "gate_z": True}}


def cell(seed: int) -> dict:
    gcfg = gym_config("aggregate", "base", seed=seed)
    ccfg = gcfg.build_clu()
    metric = PRIMARY_METRIC["aggregate"]
    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=False)
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
    order_map = _insertion_order(system)
    born = np.asarray([order_map.get(int(i), -1) for i in ids], dtype=float)
    rng = np.random.default_rng(seed + 7717)
    qs = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)
    xs, mask, _ = stream_tokens(stream, ccfg)
    ex, val, _ = aux_fit_examples("aggregate", "base", seed, ccfg, gcfg, n_val=1)
    d_in, m = int(ccfg.dim), int(ccfg.payload_dim)
    k_fit = jax.random.PRNGKey(seed * 1000 + 7 * (RIVALS.index("mamba2") + 1))

    # the raw-metric +0 B control: identical to the harness's (the CLU's own
    # launder set on these queries), recomputed here so the script is standalone
    from chlu.experiments.exp_memory_gym import _launder_predictions

    lnd = _launder_predictions(qs, centers, pays, born, rng, None, None)
    raw = {k: float(score(qs, p)[metric]) for k, p in lnd.items()}
    raw["raw_table_mean_+0B"] = float(score(qs, np.broadcast_to(
        np.asarray(pays, float).mean(axis=0, keepdims=True),
        (len(qs), int(pays.shape[1]))))[metric])
    best_raw = max(raw, key=lambda n: raw[n])

    out = {"seed": seed, "raw_best": best_raw, "raw_best_score": raw[best_raw]}
    for label, kw in CONFIGS.items():
        grid, models = fit_grid("mamba2", d_in, m, ex, key=k_fit,
                                lrs=LR_GRID_F3, wds=WD_GRID_F3, steps=400,
                                val_examples=val, arm_kwargs=kw)
        model, rec = select_best(grid, models, label=label)
        n_rows = matched_table_rows(int(model.declared_state_floats()),
                                    model.d_k, model.d_v)
        arms = rival_arms(model, xs, mask, np.asarray(qs.q0, dtype=np.float32),
                          rng=np.random.default_rng(seed + 31337), n_rows=n_rows)
        sc = {k: float(score(qs, v)[metric]) for k, v in arms.items()}
        out[label] = {"full": sc["full"], "blank": sc["blank"],
                      "launder": sc["launder"],
                      "raw_margin": sc["full"] - raw[best_raw],
                      "fit": rec["best"]["final"], "lr": rec["best"]["lr"],
                      "wd": rec["best"]["wd"],
                      "state_bytes": int(model.ledger().state_bytes),
                      "param_bytes": int(model.ledger().param_bytes)}
    return out


def _ms(v):
    a = np.asarray(v, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)


if __name__ == "__main__":
    recs = [cell(s) for s in SEEDS]
    print(json.dumps(recs, indent=1, default=float))
    print("\n=== Mamba-2 block ablation, aggregate@base, n =", len(SEEDS), "===")
    for label in CONFIGS:
        for q in ("full", "raw_margin", "blank", "fit"):
            m, se = _ms([r[label][q] for r in recs])
            print(f"{label:14s} {q:11s} {m:+.4f} ± {se:.4f}")
    with open("/Users/user/Desktop/CHLU/.claude/outputs/bprime-mamba2-arm/"
              "block_ablation.json", "w") as fh:
        json.dump(recs, fh, indent=1, default=float)
