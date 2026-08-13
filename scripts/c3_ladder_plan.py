#!/usr/bin/env python
"""⭐ The C3 Track-A ladder JOB PLAN — (arm × seed) jobs, costed off MEASURED walls.

⛔ This script **trains nothing**. It emits the plan that
``scripts/csf3/job_gpu_c3_seeds.sh`` executes, together with the arithmetic that
says whether it fits the **2×A100 / 4-day** per-job envelope (charter §4), so the
envelope claim in ``PREREG-C3-LADDER.md`` is a computed number a reviewer can
re-run rather than an assurance.

⭐ **The cost model is measured, not assumed.** Every phase second in
:data:`PILOT_PHASE_WALL_S` is read off the landed CSF3 artifact
``.claude/outputs/cluformer-pilot/csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json``
(``host_rss`` phase marks, 2×A100, JAX 0.9.0/gpu, ``steps=4000``,
``eval_batches=dyneval_batches=40``, ``write_inner_steps=40``,
``plan_workers=8``) — i.e. **the same model, the same code path, on the target
hardware**. ⛔ Those runs are the pre-C3, over-budget geometry (C3 Addendum 1 §2:
never quotable as budget-compliant ladder rows); they are used here **only** as a
timing basis, and no bpc of theirs enters this file.

The store-bearing phases are then rescaled to a candidate geometry by the
measured **atom-linearity** of the CLU cell's per-step cost (see
:data:`STORE_COST_MODEL` and ``scripts/c3_geometry_sweep.py``).

USAGE
    python scripts/c3_ladder_plan.py OUT.json [--n-atoms 2048] [--steps 20000]
                                     [--seeds 3] [--slice-batches 10]
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

#: The five swap arms the ladder trains. ⛔ The six *pinned rivals* of
#: ``chlu.eval.byte_ledger.RIVAL_SPECS`` are **ledgered, not built** — training
#: them is a separate engineering task and is a declared NOT-RUN of this ladder.
LADDER_ARMS = ("clu_store", "gru_matched", "ttt_matched", "none", "echo")

#: ⭐ MEASURED phase walls, seconds, 2×A100, pilot geometry (n_atoms 8192).
#: Source: run2/seed0 ``host_rss`` marks. Phases under ~150 s are not resolved by
#: the mark spacing and are entered as their measured upper bound.
PILOT_PHASE_WALL_S: Dict[str, Dict[str, float]] = {
    "_init": {"monitors_init": 164.0, "allocation_liveness_init": 449.0,
              "gradient_probe_init": 2077.0, "enter": 33.0},
    "clu_store": {"train": 58025.0, "static": 319.0, "dyneval": 59673.0,
                  "blank_store": 476.0, "gradient_probe_final": 2273.0,
                  "hygiene": 422.0},
    "gru_matched": {"train": 4265.0, "static": 150.0, "dyneval": 976.0},
    "ttt_matched": {"train": 2643.0, "static": 150.0, "dyneval": 808.0},
    "none": {"train": 2968.0, "static": 150.0, "dyneval": 333.0},
    "echo": {"train": 3822.0, "static": 150.0, "dyneval": 501.0},
}

#: The pilot leg those walls were measured at.
PILOT_STEPS = 4000
PILOT_N_ATOMS = 8192
PILOT_EVAL_BATCHES = 40
#: batch 8 x seq_len 1024 (both PilotConfig defaults, hence absent from the
#: landed artifact's non-default flag table).
TOKENS_PER_STEP = 8 * 1024
#: Landed artifact's ``total_params['clu_store']``.
PILOT_PARAMS = 28_556_792
#: 2xA100 bf16 peak, scout §1.4.
PEAK_FLOPS = 6.24e14
#: enwik8 canonical train split.
TRAIN_SPLIT_BYTES = 90_000_000
#: CSF3 per-job limit (charter §4).
JOB_LIMIT_S = 4 * 24 * 3600

#: ⭐ The store-cost model, MEASURED at smoke scale by
#: ``scripts/c3_geometry_sweep.py`` (axis A): the CLU arm's per-step wall is
#: ``a + b*n_atoms`` — an **atom-LINEAR** store term on a constant shell term.
#: Only ``store_fraction_at_pilot`` is taken from the A100 legs: the null arm
#: costs 2968/4000 = 0.742 s/step against the CLU arm's 14.51, so 94.9 % of the
#: CLU arm's step is the store, and that is the part that rescales with atoms.
STORE_COST_MODEL = {
    "form": "t_step(n_atoms) = shell + store * (n_atoms / PILOT_N_ATOMS)",
    "store_fraction_at_pilot": 1.0 - (2968.0 / 4000.0) / (58025.0 / 4000.0),
    "exponent": 1.0,
    "provenance": "shape from scripts/c3_geometry_sweep.py axis A (smoke, CPU); "
                  "the shell/store split from the landed 2xA100 legs (none "
                  "0.742 s/step vs clu_store 14.51 s/step at n_atoms 8192).",
}


def scale_store_phase(seconds: float, *, n_atoms: int, steps: int,
                      eval_batches: Optional[int] = None) -> float:
    """Rescale one measured store-bearing phase to a candidate geometry/length.

    Train phases scale with ``steps``; eval phases with ``eval_batches``. Both
    scale with the atom-linear store term while the shell term does not.
    """
    frac = float(STORE_COST_MODEL["store_fraction_at_pilot"])
    atom_factor = frac * (n_atoms / PILOT_N_ATOMS) + (1.0 - frac)
    length = ((steps / PILOT_STEPS) if eval_batches is None
              else (eval_batches / PILOT_EVAL_BATCHES))
    return float(seconds * atom_factor * length)


def job_cost(arm: str, *, n_atoms: int, steps: int, eval_batches: int,
             slice_batches: int, with_slices: bool = True,
             with_d5: bool = False) -> Dict[str, Any]:
    """Seconds for one (arm, seed) job at a candidate geometry.

    ⚠ **The slice phase is NOT measured** — it did not exist when the pilot legs
    ran. It is priced as one ``static``-shaped pass plus one ``dyneval``-shaped
    pass at ``slice_batches`` (``chlu/experiments/exp_cluformer_pilot._arm_slices``
    computes both columns), which is the shape of the call, not a guess about its
    constant. ⛔ It is the single largest unmeasured item in this plan and is
    flagged as such rather than buried.
    """
    store_bearing = arm == "clu_store"
    ph = dict(PILOT_PHASE_WALL_S[arm])
    init = sum(PILOT_PHASE_WALL_S["_init"].values())
    rows: Dict[str, float] = {}

    def sc(sec, *, eb=None):
        if store_bearing:
            return scale_store_phase(sec, n_atoms=n_atoms, steps=steps,
                                     eval_batches=eb)
        return float(sec * ((steps / PILOT_STEPS) if eb is None
                            else (eb / PILOT_EVAL_BATCHES)))

    rows["init"] = sc(init, eb=PILOT_EVAL_BATCHES)
    rows["train"] = sc(ph["train"])
    rows["static"] = sc(ph["static"], eb=eval_batches)
    rows["dyneval"] = sc(ph["dyneval"], eb=eval_batches)
    for k in ("blank_store", "gradient_probe_final", "hygiene"):
        if k in ph:
            rows[k] = sc(ph[k], eb=PILOT_EVAL_BATCHES)
    if with_slices:
        rows["slices_static_UNMEASURED"] = sc(ph["static"], eb=slice_batches)
        rows["slices_dyneval_UNMEASURED"] = sc(ph["dyneval"], eb=slice_batches)
    if with_d5 and arm == "clu_store":
        # 5 anytime budgets, each a `static`-shaped pass (exp_cluformer_pilot).
        rows["anytime_curve"] = 5.0 * sc(ph["static"], eb=eval_batches)
    total = sum(rows.values())
    return {"arm": arm, "phases": rows, "total_s": total,
            "total_h": total / 3600.0,
            "fits_job_limit": total <= JOB_LIMIT_S,
            "headroom_x": JOB_LIMIT_S / total if total else float("inf")}


def mfu(seconds_per_step: float, params: int = PILOT_PARAMS) -> float:
    """Model-FLOP utilisation under the field's ``C ~ 6ND`` convention."""
    return (6.0 * params * TOKENS_PER_STEP / seconds_per_step) / PEAK_FLOPS


def build_plan(*, n_atoms: int, steps: int, seeds: int, eval_batches: int,
               slice_batches: int, concurrency: int = 4,
               with_slices: bool = True, with_d5: bool = False) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    for arm in LADDER_ARMS:
        c = job_cost(arm, n_atoms=n_atoms, steps=steps, eval_batches=eval_batches,
                     slice_batches=slice_batches, with_slices=with_slices,
                     with_d5=with_d5)
        for seed in range(seeds):
            jobs.append({**c, "seed": seed,
                         "task_id": len(jobs),
                         "out": f"$OUT_BASE/{arm}_s{seed}"})
    # %concurrency scheduling: greedy longest-first over `concurrency` lanes.
    lanes = [0.0] * max(1, concurrency)
    for j in sorted(jobs, key=lambda r: -r["total_s"]):
        i = min(range(len(lanes)), key=lambda k: lanes[k])
        lanes[i] += j["total_s"]
    makespan = max(lanes)

    clu_step_s = scale_store_phase(PILOT_PHASE_WALL_S["clu_store"]["train"],
                                   n_atoms=n_atoms, steps=steps) / steps
    tokens = steps * TOKENS_PER_STEP
    return {
        "banner": "⛔ a PLAN, not a run — this script trains nothing.",
        "geometry": {"n_atoms": n_atoms, "addr_dim": 8, "payload_dim": 4,
                     "capacity": 32, "n_layers": 12, "d_model": 512},
        "schedule": {"arms": list(LADDER_ARMS), "seeds": seeds,
                     "n_jobs": len(jobs), "concurrency": concurrency,
                     "makespan_s": makespan, "makespan_h": makespan / 3600.0,
                     "makespan_days": makespan / 86400.0},
        "envelope": {
            "job_limit_s": JOB_LIMIT_S,
            "worst_job_h": max(j["total_h"] for j in jobs),
            "worst_job_arm": max(jobs, key=lambda j: j["total_s"])["arm"],
            "all_jobs_fit": all(j["fits_job_limit"] for j in jobs),
            "min_headroom_x": min(j["headroom_x"] for j in jobs),
        },
        "throughput": {
            "clu_store_s_per_step": clu_step_s,
            "clu_store_mfu": mfu(clu_step_s),
            "null_arm_s_per_step": PILOT_PHASE_WALL_S["none"]["train"] / PILOT_STEPS,
            "null_arm_mfu": mfu(PILOT_PHASE_WALL_S["none"]["train"] / PILOT_STEPS),
            "measured_clu_mfu_at_pilot_geometry": mfu(
                PILOT_PHASE_WALL_S["clu_store"]["train"] / PILOT_STEPS),
            "convention": "C = 6*N*D, N = total_params(clu_store) = "
                          f"{PILOT_PARAMS:,}, D = {TOKENS_PER_STEP} tokens/step, "
                          f"peak = {PEAK_FLOPS:.3g} FLOP/s (2xA100 bf16, scout §1.4)",
        },
        "token_budget": {
            "steps": steps, "tokens": tokens,
            "epochs_of_enwik8_train_split": tokens / TRAIN_SPLIT_BYTES,
            "note": "⛔ byte-level: 1 token = 1 byte, so tokens == bytes seen.",
        },
        "cost_model": STORE_COST_MODEL,
        "measured_basis": {
            "source": ".claude/outputs/cluformer-pilot/csf3_outs/run2/"
                      "pilot_pilot_seed0_PARTIAL.json (host_rss phase marks)",
            "hardware": "2xA100, JAX 0.9.0/gpu, plan_workers=8",
            "pilot_phase_wall_s": PILOT_PHASE_WALL_S,
            "caveat": "runs 1/2 are the pre-C3, OVER-BUDGET geometry and are a "
                      "TIMING basis only (C3 Addendum 1 §2: never quotable as "
                      "budget-compliant ladder rows). No bpc of theirs is used.",
        },
        "jobs": jobs,
        "not_run": [
            "the six PINNED RIVALS (ttt_linear, ttt_mlp, gated_deltanet2, mamba2, "
            "transformer_xl, sliding_window) are LEDGERED, not BUILT — no "
            "implementation of them exists in this repo, so they are a declared "
            "NOT-RUN of this ladder and a separate engineering task.",
            "the slice phase's constant is UNMEASURED (priced by call shape).",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--n-atoms", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--slice-batches", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--d5", action="store_true")
    a = ap.parse_args(argv)
    plan = build_plan(n_atoms=a.n_atoms, steps=a.steps, seeds=a.seeds,
                      eval_batches=a.eval_batches, slice_batches=a.slice_batches,
                      concurrency=a.concurrency, with_d5=a.d5)
    with open(a.out, "w") as fh:
        json.dump(plan, fh, indent=2)
    e, s, t = plan["envelope"], plan["schedule"], plan["throughput"]
    print(f"n_atoms {a.n_atoms} | steps {a.steps} | {s['n_jobs']} jobs "
          f"({len(LADDER_ARMS)} arms x {a.seeds} seeds)")
    print(f"  worst job: {e['worst_job_arm']} {e['worst_job_h']:.1f} h "
          f"(limit {JOB_LIMIT_S / 3600:.0f} h, headroom {e['min_headroom_x']:.2f}x, "
          f"all fit={e['all_jobs_fit']})")
    print(f"  makespan at %{a.concurrency}: {s['makespan_h']:.1f} h "
          f"({s['makespan_days']:.2f} days)")
    print(f"  clu_store {t['clu_store_s_per_step']:.2f} s/step, MFU "
          f"{100 * t['clu_store_mfu']:.4f}% (measured at pilot geometry: "
          f"{100 * t['measured_clu_mfu_at_pilot_geometry']:.4f}%)")
    tb = plan["token_budget"]
    print(f"  tokens {tb['tokens']:,} = {tb['epochs_of_enwik8_train_split']:.2f} "
          f"epochs of the enwik8 train split")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
