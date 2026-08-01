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
    from chlu.core.monitors import default_registry
    for a in arms:
        t = time.time()
        # ⭐ ONE persistent registry per arm, so monitor #6's (loss, acq) window
        # accumulates across the run instead of restarting at every observation.
        reg = default_registry(loud=False) if a == "clu_store" else None
        during: List[Dict[str, Any]] = []
        m, hist = train_arm(a, models[a], pcfg, iter(batches), log=rec["train_log"],
                            monitor_registry=reg,
                            monitor_tokens=(x0 if reg is not None else None),
                            monitor_out=during)
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
            row["monitors_during"] = [
                {k: v for k, v in d.items() if k != "readings"} for d in during]
            row["monitors_final"] = monitor_pass(
                m, pcfg, x0, registry=reg,
                write_loss_now=float(hist["loss_history"][-1]))
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


# ==========================================================================
# plots (artifacts live under .claude/, never in the repo)
# ==========================================================================
def plot_pilot(records: List[Dict[str, Any]], out_dir: str) -> Optional[str]:
    """Four panels: training curves - the swap table - the anytime curve - monitors.

    Self-contained rather than routed through ``chlu/utils/plotting.py``: that
    module is shared across every campaign and this task owns none of it, so a
    new helper there is a merge-conflict surface for no benefit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [r for r in records if r.get("arms")]
    if not recs:
        return None
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"clu_store": "tab:red", "gru_matched": "tab:blue",
              "ttt_matched": "tab:green", "none": "0.5", "echo": "tab:orange"}

    # (a) training curves, seed 0
    r0 = recs[0]
    for arm in r0["arms"]:
        h = r0["arms"][arm]["train"]["loss_history"]
        b = [v / np.log(2.0) for v in h]
        k = max(1, len(b) // 60)
        sm = [float(np.mean(b[max(0, i - k):i + 1])) for i in range(len(b))]
        ax[0, 0].plot(sm, label=arm, color=colors.get(arm))
    ax[0, 0].set(xlabel="optimisation step", ylabel="train bpc",
                 title=f"(a) training, seed {r0['seed']} ({r0['scale']} scale)")
    ax[0, 0].legend(fontsize=8)

    # (b) the swap table: static vs dyn-eval, seed-mean +- se
    arms = list(r0["arms"])
    xs = np.arange(len(arms))
    lo, hi = np.inf, -np.inf
    for j, col in enumerate(("static", "dyneval")):
        mu = [np.mean([r["arms"][a][col]["bpc"] for r in recs]) for a in arms]
        se = [(np.std([r["arms"][a][col]["bpc"] for r in recs], ddof=1)
               / np.sqrt(len(recs))) if len(recs) > 1 else 0.0 for a in arms]
        ax[0, 1].bar(xs + 0.36 * j - 0.18, mu, 0.34, yerr=se, capsize=3,
                     label=("static" if col == "static" else "dynamic eval"))
        lo = min(lo, min(np.array(mu) - np.array(se)))
        hi = max(hi, max(np.array(mu) + np.array(se)))
    pad = 0.25 * (hi - lo) + 1e-6
    ax[0, 1].set_ylim(lo - pad, hi + pad)     # zoomed: a 0-based axis hides 0.01 bpc
    ax[0, 1].set_xticks(xs)
    ax[0, 1].set_xticklabels(arms, rotation=20, fontsize=8)
    ax[0, 1].set(ylabel="held-out bpc",
                 title=f"(b) system-level swap, {len(recs)} seeds "
                       "(dyn-eval IN the table)")
    ax[0, 1].legend(fontsize=8)

    # (c) the anytime curve — SHAPE only
    any_c = [r["arms"]["clu_store"].get("anytime_curve") for r in recs
             if "clu_store" in r["arms"]]
    if any(any_c):
        for r, c in zip(recs, any_c, strict=False):
            if not c:
                continue
            ax[1, 0].plot([p["verlet_per_read"] for p in c],
                          [p["bpc"] for p in c], "o-", label=f"seed {r['seed']}")
        ax[1, 0].set(xscale="log", xlabel="Verlet steps per read",
                     ylabel="held-out bpc",
                     title="(c) anytime shape (SHAPE claim only, §A3)")
        ax[1, 0].legend(fontsize=8)
    else:
        ax[1, 0].text(0.5, 0.5, "D5 NOT RUN", ha="center", transform=ax[1, 0].transAxes)
        ax[1, 0].set_axis_off()

    # (d) monitors
    m = r0["arms"].get("clu_store", {}).get("monitors_final") or r0["monitors_init"]
    # ⚠ THREE states, not two: `inapplicable` is NOT `clear`, and colouring it
    # green would be exactly the silent pass the acceptance criterion forbids.
    names = [x["name"] for x in m["readings"]]
    state = ["TRIPPED" if x.get("tripped") else
             ("clear" if x.get("applicable", True) else "inapplicable")
             for x in m["readings"]]
    cmap = {"TRIPPED": "tab:red", "clear": "tab:green", "inapplicable": "0.75"}
    ax[1, 1].barh(names, [1.0] * len(names), color=[cmap[s_] for s_ in state])
    for yi, s_ in enumerate(state):
        ax[1, 1].text(0.5, yi, s_, ha="center", va="center", fontsize=7,
                      color="white" if s_ != "inapplicable" else "black")
    ax[1, 1].set(xlim=(0, 1), xticks=[],
                 title=f"(d) 13 monitors + M14, seed {r0['seed']} — "
                       f"{m['n_tripped']} TRIPPED, "
                       f"{len(names) - m['n_applicable']} inapplicable")
    ax[1, 1].tick_params(labelsize=7)
    fig.suptitle("Tier-iii pilot: the FULL C2W1 CLU store as a streaming block's "
                 "memory (enwik8)  —  acceptance = DOES NOT COLLAPSE, not WINS")
    fig.tight_layout()
    p = Path(out_dir) / f"pilot_{r0['scale']}_panels.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def _parse_kv(pair: str):
    """``key=value`` with int/float/bool/None inference (strings pass through).

    Exists so the CSF3 job script can carry a recommended config without the
    module being edited on the cluster — an edited module is a provenance hole
    (the artifact would not say which config produced it), whereas a flag is
    recorded verbatim in ``rec['flags']``.
    """
    if "=" not in pair:
        raise SystemExit(f"expected KEY=VALUE, got {pair!r}")
    k, v = pair.split("=", 1)
    low = v.strip().lower()
    if low in ("true", "false"):
        return k.strip(), low == "true"
    if low in ("none", "null"):
        return k.strip(), None
    for cast in (int, float):
        try:
            return k.strip(), cast(v)
        except ValueError:
            pass
    return k.strip(), v


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
    ap.add_argument("--plot-only", action="store_true",
                    help="re-aggregate + re-plot from artifacts already on disk")
    ap.add_argument("--d5", action="store_true",
                    help="also run the anytime shape curve (secondary; §A3 shape only)")
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                    help="top-level PilotConfig overrides, e.g. monitor_every=25")
    ap.add_argument("--mem", nargs="*", default=None, metavar="KEY=VALUE",
                    help="StreamMemoryConfig overrides, e.g. atom_place_radius=0.3 "
                         "write_inner_steps=40 (⭐ the `pilot-placement-probe` "
                         "recommendation block sets the submitted config here, so "
                         "the scale run never needs the module edited)")
    ap.add_argument("--store", nargs="*", default=None, metavar="KEY=VALUE",
                    help="CluSystemConfig overrides")
    a = ap.parse_args(argv)
    ov: Dict[str, Any] = {}
    for flag, key in (("set", None), ("mem", "memory"), ("store", "store")):
        pairs = getattr(a, flag)
        if not pairs:
            continue
        parsed = dict(_parse_kv(p) for p in pairs)
        if key is None:
            ov.update(parsed)
        else:
            ov[key] = dict((TOY if a.scale == "toy" else PILOT).get(key, {}),
                           **parsed)
    if a.steps is not None:
        ov["steps"] = a.steps
        ov["warmup"] = max(1, a.steps // 10)
    if a.arms:
        ov["arms"] = tuple(a.arms)
    if a.quick:
        ov.update(steps=6, warmup=2, eval_batches=2, dyneval_batches=2,
                  data_bytes=1_000_000)
    if a.plot_only:
        recs = [json.loads(p.read_text())
                for p in sorted(Path(a.out).glob(f"pilot_{a.scale}_seed*_S*.json"))]
        agg = aggregate(recs)
        save_json(Path(a.out) / f"pilot_{a.scale}_aggregate.json", agg)
        print("plot:", plot_pilot(recs, a.out), flush=True)
        print(json.dumps(agg, indent=2, default=float), flush=True)
        return 0
    seeds = a.seeds if a.seeds else [a.seed]
    recs = [run_pilot(a.scale, s, a.stage, a.out, ov, a.d5) for s in seeds]
    if len(recs) > 1:
        agg = aggregate(recs)
        save_json(Path(a.out) / f"pilot_{a.scale}_aggregate.json", agg)
        print("plot:", plot_pilot(recs, a.out), flush=True)
        print(json.dumps(agg, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
