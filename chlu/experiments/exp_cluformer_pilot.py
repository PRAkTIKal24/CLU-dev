"""⭐ TIER-III PILOT — the full C2W1 CLU as a streaming block's memory, on enwik8.

⛔ **What is being claimed and what is not.** The memory in the block is the
**C2W1 full store** (learned ``V_theta``, admission, per-item lifetimes, masked
local write, permitted basin interaction under the soft certificate, learned
``phi`` in, learned **trajectory** ``psi`` out, two-phase relaxation, mass and
friction as trainable selectors, confidence-gated retry, the controller's verb
set, all 13 monitors). It is **not** ``CLUBlock``, the w20/w21 driven-Hamiltonian
recurrence with no store — that object is ruled out as a tier-iii arm.

**The control is the SYSTEM-LEVEL SWAP**: the same block with the CLU replaced by
a matched-state GRU cell and by a matched-state TTT-class cell — matched
parameters AND matched state-bytes, same embedding, same depth, same norms, same
residual, same optimiser, same data order, same seeds, same chunk granularity.
⛔ The tier-i settle-deleted / matched-bytes launder is **not** this task's
control and is not run here.

**The acceptance criterion is inherited verbatim from ``full-clu-harness``:** the
system runs the stream **without tripping a silent collapse mode**. *"Does not
collapse", not "wins".* Every monitor's trip-state is a reported artifact.

Staged acceptance (task §3) — the runner reports which stage it reached:

* **S1** the block exists and does not collapse (``--stage s1``);
* **S2** the training path is real — ``||dL/dphi||`` end to end through the
  trajectory read vs the settled-point arm's 0.0 (``--stage s2``);
* **S3** the swap control is defined and matched, ledgers published, all arms
  trained on the same data order and seeds (``--stage s3``);
* **S4** the 26-47 M CSF3 run with the dynamic-evaluation substitute column and
  the pre-registered directional falsifier adjudicated (``--stage s4``).

Run::

    PYTHONPATH=. python -m chlu.experiments.exp_cluformer_pilot --stage s3 --seed 0

⚠ **No CLI hook.** ``chlu/cli/experiment_cmd.py`` is read-only to this task
(owned by ``bprime-c6`` this wave), so the runner is invoked as a module. Noted
in the report.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np

from chlu.core.blocks import _count
from chlu.data.enwik8 import contiguous_batches, load_enwik8, random_batches
from chlu.training.train_cluformer import (
    PilotConfig,
    anytime_curve,
    allocation_liveness,
    assert_shared_shell_identical,
    build_arm,
    calibrate_phi_gain,
    dynamic_eval,
    evaluate,
    gradient_probe,
    monitor_pass,
    save_json,
    solve_arms,
    train_arm,
)

#: Toy scale — the LOCAL configuration. ⛔ Not a 26-47 M number and never
#: reported as one (task §0.2: no 26-47 M run happens on the laptop).
TOY = dict(d_model=64, n_layers=2, seq_len=512, batch=4,
           addr_dim=2, payload_dim=1, capacity=8, atoms_per_item=128,
           steps=60, warmup=10, eval_batches=4, dyneval_batches=4,
           data_bytes=4_000_000,
           memory=dict(chunk=32, address_steps=24, read_steps=24, traj_stride=8,
                       psi_hidden=32, write_inner_steps=4, write_n_perturb=8,
                       retry_rounds=1, conv_kernel=4, mlp_mult=4))

#: ⭐ The PILOT scale — 26-47 M, **CSF3 only** (Head ruling §0.2). Declared here
#: so the job script and the report quote the same object.
PILOT = dict(d_model=512, n_layers=12, seq_len=1024, batch=8,
             addr_dim=8, payload_dim=4, capacity=32, atoms_per_item=256,
             steps=4000, warmup=200, eval_batches=40, dyneval_batches=40,
             data_bytes=None,
             memory=dict(chunk=64, address_steps=64, read_steps=64, traj_stride=8,
                         psi_hidden=128, write_inner_steps=4, write_n_perturb=8,
                         retry_rounds=1, conv_kernel=4, mlp_mult=4))


def make_config(scale: str, seed: int, overrides: Optional[dict] = None) -> PilotConfig:
    base = dict(TOY if scale == "toy" else PILOT)
    base["seed"] = int(seed)
    base.update(dict(overrides or {}))
    return PilotConfig.from_mapping(base)


def _data(pcfg: PilotConfig):
    tr, va, te = load_enwik8(pcfg.data_root, n_bytes=pcfg.data_bytes)
    return tr, va, te


def _train_batches(split, pcfg: PilotConfig) -> List:
    """⭐ Materialised ONCE and reused by every arm — identical data order."""
    return list(random_batches(split, batch=pcfg.batch, seq_len=pcfg.seq_len,
                               n_batches=pcfg.steps, seed=pcfg.seed))


def _eval_batches(split, pcfg: PilotConfig, n: int) -> List:
    return list(contiguous_batches(split, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                   n_batches=n))


def run_pilot(scale: str = "toy", seed: int = 0, stage: str = "s3",
              out_dir: str = ".claude/outputs/cluformer-pilot",
              overrides: Optional[dict] = None,
              with_d5: bool = False) -> Dict[str, Any]:
    """Run the pilot to ``stage``; write one JSON artifact; return the record."""
    t_all = time.time()
    pcfg = make_config(scale, seed, overrides)
    out = Path(out_dir)
    rec: Dict[str, Any] = {
        "scale": scale, "seed": seed, "stage_requested": stage,
        "flags": {
            "pilot": pcfg.as_flag_table(),
            "memory": asdict(pcfg.memory_cfg()),
            "store": pcfg.store_cfg().as_flag_table(),
            "store_dim": int(pcfg.store_cfg().dim),
            "store_n_atoms": int(pcfg.store_cfg().n_atoms),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "stages_reached": [],
        "not_run": [],
    }
    tr, va, te = _data(pcfg)
    rec["data"] = {"train_B": len(tr), "valid_B": len(va), "test_B": len(te),
                   "n_bytes_staged": pcfg.data_bytes or 100_000_000}

    key = jax.random.PRNGKey(1000 + seed)
    k_cal, k_solve, k_model = jax.random.split(key, 3)
    calib_x = next(iter(random_batches(tr, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                       n_batches=1, seed=seed)))[0]
    gain = calibrate_phi_gain(pcfg, calib_x, key=k_cal)
    pcfg.memory = dict(pcfg.memory)
    pcfg.memory["phi_gain"] = gain
    rec["phi_gain_calibrated"] = gain
    rec["flags"]["memory"]["phi_gain"] = gain

    specs, ledger = solve_arms(pcfg, k_solve)
    rec["swap_ledger"] = ledger

    arms = [a for a in pcfg.arms]
    models = {a: build_arm(a, pcfg, specs, key=k_model) for a in arms}
    rec["shell"] = assert_shared_shell_identical(models)
    rec["total_params"] = {a: _count(m) for a, m in models.items()}

    # ---------------- S1: the block runs the stream, monitors reported --------
    x0, y0 = _eval_batches(va, pcfg, 1)[0]
    t = time.time()
    rec["monitors_init"] = monitor_pass(models["clu_store"], pcfg, x0)
    rec["monitors_init"]["wall_s"] = time.time() - t
    rec["allocation_liveness_init"] = allocation_liveness(models["clu_store"], pcfg,
                                                          x0, y0)
    rec["stages_reached"].append("S1")
    if stage == "s1":
        return _finish(rec, out, t_all, pcfg)

    # ---------------- S2: the training path is real ---------------------------
    t = time.time()
    rec["gradient_probe_init"] = gradient_probe(models["clu_store"], pcfg, x0, y0)
    rec["gradient_probe_init"]["wall_s_total"] = time.time() - t
    rec["stages_reached"].append("S2")
    if stage == "s2":
        return _finish(rec, out, t_all, pcfg)

    # ---------------- S3: the swap control, trained on identical data ---------
    batches = _train_batches(tr, pcfg)
    ev = _eval_batches(te, pcfg, pcfg.eval_batches)
    dv = _eval_batches(te, pcfg, pcfg.dyneval_batches)
    rec["train_log"] = []
    rec["arms"] = {}
    for a in arms:
        t = time.time()
        m, hist = train_arm(a, models[a], pcfg, iter(batches), log=rec["train_log"])
        row = {"train": hist}
        row["static"] = evaluate(m, pcfg, iter(ev))
        row["dyneval"] = dynamic_eval(m, pcfg, iter(dv))
        if a == "clu_store":
            row["blank_store"] = evaluate(m, pcfg, iter(ev), blank=True)
            if with_d5:
                mc = pcfg.memory_cfg()
                base = (mc.address_steps, mc.read_steps)
                row["anytime_curve"] = anytime_curve(
                    m, pcfg, ev, [(max(2, base[0] // f), max(2, base[1] // f))
                                  for f in (8, 4, 2, 1)]
                    + [(base[0] * 2, base[1] * 2)])
            row["monitors_final"] = monitor_pass(m, pcfg, x0)
            row["gradient_probe_final"] = gradient_probe(m, pcfg, x0, y0)
            row["selectors_final"] = _selectors(m)
        row["wall_s"] = time.time() - t
        rec["arms"][a] = row
        models[a] = m
        print(f"[{a}] static bpc {row['static']['bpc']:.4f} | "
              f"dyneval bpc {row['dyneval']['bpc']:.4f} | {row['wall_s']:.0f}s", flush=True)
    rec["swap_table"] = _swap_table(rec)
    rec["stages_reached"].append("S3")
    if stage in ("s3",):
        rec["not_run"].append(
            "S4 (26-47 M on CSF3): NOT RUN at this scale. See report.")
        return _finish(rec, out, t_all, pcfg)

    rec["stages_reached"].append("S4")
    return _finish(rec, out, t_all, pcfg)


def _selectors(model) -> Dict[str, Any]:
    """The trainable friction/mass selectors after training (§A13 rule 3, P8)."""
    import jax.numpy as jnp
    ga = [float(jnp.exp(b.cell.log_gamma_addr)) for b in model.blocks]
    gr = [float(jnp.exp(b.cell.log_gamma_read)) for b in model.blocks]
    mm = [float(jnp.mean(jax.nn.softplus(b.cell.clu.log_mass))) for b in model.blocks]
    return {"gamma_address": ga, "gamma_read": gr, "mean_mass": mm}


def _swap_table(rec: Dict[str, Any]) -> Dict[str, Any]:
    """⭐ The swap table with the dynamic-evaluation column **in it**, not a footnote."""
    ref = rec["arms"].get("clu_store")
    rows = {}
    for a, r in rec["arms"].items():
        row = {"bpc_static": r["static"]["bpc"], "bpc_dyneval": r["dyneval"]["bpc"],
               "params_total": rec["total_params"][a],
               "cell_params": rec["swap_ledger"].get(a, {}).get("params"),
               "cell_state_bytes": rec["swap_ledger"].get(a, {}).get("state_bytes"),
               "wall_s": r["wall_s"]}
        if ref is not None:
            row["margin_vs_clu_static"] = ref["static"]["bpc"] - r["static"]["bpc"]
            row["margin_vs_clu_dyneval"] = ref["dyneval"]["bpc"] - r["dyneval"]["bpc"]
        rows[a] = row
    return rows


def _finish(rec, out: Path, t0: float, pcfg: PilotConfig) -> Dict[str, Any]:
    rec["wall_s_total"] = time.time() - t0
    p = save_json(out / f"pilot_{rec['scale']}_seed{rec['seed']}_"
                        f"{rec['stages_reached'][-1]}.json", rec)
    rec["artifact"] = str(p)
    print(f"wrote {p} ({rec['wall_s_total']:.0f}s, stages "
          f"{'+'.join(rec['stages_reached'])})", flush=True)
    _ = pcfg
    return rec


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-mean +- s.e. of every arm's bpc, and the falsifier's adjudication.

    ⛔ **DF1 as pre-registered**: tier iii is alive only if the CLU's seed-mean
    bpc is at least 0.02 below BOTH matched swap arms', with the gap exceeding
    the sum of the two +-1 s.e. bars. ⛔ **DF3**: if the CLU's advantage does not
    survive dynamic evaluation, the primary is dead.
    """
    arms = sorted({a for r in records for a in r.get("arms", {})})
    out: Dict[str, Any] = {"n_seeds": len(records), "seeds": [r["seed"] for r in records],
                           "arms": {}}
    for a in arms:
        for col in ("static", "dyneval"):
            v = [r["arms"][a][col]["bpc"] for r in records if a in r.get("arms", {})]
            out["arms"].setdefault(a, {})[f"bpc_{col}_mean"] = float(np.mean(v))
            out["arms"][a][f"bpc_{col}_se"] = (
                float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
            out["arms"][a][f"bpc_{col}_per_seed"] = [float(x) for x in v]
    if "clu_store" in out["arms"]:
        c = out["arms"]["clu_store"]
        verdict = {}
        for opp in ("gru_matched", "ttt_matched"):
            if opp not in out["arms"]:
                continue
            o = out["arms"][opp]
            for col in ("static", "dyneval"):
                m = o[f"bpc_{col}_mean"] - c[f"bpc_{col}_mean"]   # >0 => CLU better
                se = c[f"bpc_{col}_se"] + o[f"bpc_{col}_se"]
                verdict[f"{opp}_{col}"] = {
                    "clu_advantage_bpc": m, "se_sum": se,
                    "passes_0.02_and_se": bool(m >= 0.02 and m > se),
                }
        out["DF1_alive"] = all(v["passes_0.02_and_se"]
                               for k, v in verdict.items() if k.endswith("static"))
        out["DF3_primary_dead"] = any(
            verdict.get(f"{o}_dyneval", {}).get("clu_advantage_bpc", -1e9)
            < verdict.get(f"{o}_static", {}).get("clu_advantage_bpc", 1e9) - 0.02
            for o in ("gru_matched", "ttt_matched") if f"{o}_static" in verdict)
        out["verdict"] = verdict
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", choices=("toy", "pilot"), default="toy")
    ap.add_argument("--stage", choices=("s1", "s2", "s3", "s4"), default="s3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="run several seeds and aggregate (>=3 for any reported number)")
    ap.add_argument("--out", default=".claude/outputs/cluformer-pilot")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--d5", action="store_true",
                    help="also run the anytime shape curve (secondary; §A3 shape only)")
    a = ap.parse_args(argv)
    ov: Dict[str, Any] = {}
    if a.steps is not None:
        ov["steps"] = a.steps
        ov["warmup"] = max(1, a.steps // 10)
    if a.arms:
        ov["arms"] = tuple(a.arms)
    if a.quick:
        ov.update(steps=6, warmup=2, eval_batches=2, dyneval_batches=2,
                  data_bytes=1_000_000)
    seeds = a.seeds if a.seeds else [a.seed]
    recs = [run_pilot(a.scale, s, a.stage, a.out, ov, a.d5) for s in seeds]
    if len(recs) > 1:
        agg = aggregate(recs)
        save_json(Path(a.out) / f"pilot_{a.scale}_aggregate.json", agg)
        print(json.dumps(agg, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
