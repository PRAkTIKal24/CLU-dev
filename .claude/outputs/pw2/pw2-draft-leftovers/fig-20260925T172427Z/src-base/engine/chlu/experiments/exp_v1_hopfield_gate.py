"""Experiment V1.1: the IDENTICAL gate stack on a non-CLU (Hopfield) memory.

Critique P7/V1.1: the v1-pivot gate stack (training-time Platt calibration head
on retrieval diagnostics -> learned per-instance gate; escalation-ladder
compute allocation; Learn-then-Test risk certificate) may be **memory-agnostic**
— energy ~= readout margin on the CLU (v1-pivot Finding 4). This runs the same
stack on a modern-Hopfield associative memory, using ITS natural scalars
(Hopfield energy as R; nearest-neighbour readout margin), on the identical MQAR
task / seeds / difficulty levels as v1-pivot, and reports the same three
headline metrics side-by-side:

  1. calibration-transfer AUROC  (pooled raw R -> wrong  vs  calibrated p_wrong)
  2. allocation compute-savings  (learned-point cost vs always-full, on an
     iteration ladder — Hopfield's only compute lever)
  3. LTT validity count          (distribution-free selective-risk certificate)

The verdict: if Hopfield's numbers track the CLU's, the gate stack is
memory-agnostic (V1 = "gate mechanism + certificates", physics is not the gate
signal); if not, a CLU-specific advantage is measured. Hopfield needs NO
training, so this runs at full scale cheaply.

Reuses the v1-pivot self-test (_probe_cues), decode (_decode_values), policy
simulator (_simulate_tau_policy), AUROC (_auroc), and the calibration.py head /
LTT verbatim — only the memory changes.

Nomenclature (F5 Def-2): consumes energies and probabilities only.
"""

import json
import math
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.data.mqar import generate_mqar, make_token_embeddings
from chlu.experiments.exp_v1_calibration import _clustered_embeddings, _probe_cues
from chlu.experiments.exp_v1_gate import _auroc, _decode_values, _simulate_tau_policy
from chlu.training.calibration import fit_calibration_head, ltt_select_threshold
from chlu.utils.metrics import aurc, coverage_at_risk, risk_coverage_curve

#: Hopfield compute ladder = fixed-point iterations (its only compute lever).
HOPFIELD_LADDER = (1, 2, 4)


def _hopfield_energy(z, patterns, beta):
    """Modern-Hopfield energy E = -lse(beta, X z)/beta + 0.5||z||^2 (per query)."""
    s = beta * (z @ patterns.T)  # (T, kv)
    lse = jax.scipy.special.logsumexp(s, axis=1) / beta
    return -lse + 0.5 * jnp.sum(z * z, axis=1)


def _hopfield_ladder(q0, patterns, beta, e, val_embeds, val_tokens, true_tok):
    """Iterated Hopfield retrieval at the compute ladder -> stage records.

    z_{t+1} = softmax(beta * X z_t) X (full-state fixed-point iteration).
    Records, at each ladder stage (n_iter in HOPFIELD_LADDER): the Hopfield
    energy (R feature), the nearest-neighbour readout margin, the decoded
    prediction and its correctness, plus cumulative cost (= n_iter).
    """
    Rs, margins, preds, cost = [], [], [], []
    for n_iter in HOPFIELD_LADDER:
        z = q0
        for _ in range(n_iter):
            attn = jax.nn.softmax(beta * (z @ patterns.T), axis=1)
            z = attn @ patterns
        E = _hopfield_energy(z, patterns, beta)
        pred, margin = _decode_values(z[:, e:], val_embeds, val_tokens)
        Rs.append(np.asarray(E))
        margins.append(np.asarray(margin))
        preds.append(np.asarray(pred))
        cost.append(n_iter)
    pred_arr = np.stack(preds, axis=1)
    return {
        "R": np.stack(Rs, axis=1),
        "margin": np.stack(margins, axis=1),
        "pred": pred_arr,
        "correct": pred_arr == np.asarray(true_tok)[:, None],
        "cost": np.asarray(cost),
    }


def _fit_hopfield_heads(probe_rec, cfg):
    """Fit the r / margin / r_margin heads on the Hopfield self-test split."""
    if cfg.calib_fit_all_stages:
        R, mg = probe_rec["R"].ravel(), probe_rec["margin"].ravel()
        wrong = (~probe_rec["correct"]).ravel()
    else:
        R, mg = probe_rec["R"][:, 0], probe_rec["margin"][:, 0]
        wrong = ~probe_rec["correct"][:, 0]
    return {
        name: fit_calibration_head(R=R, margin=mg, wrong=wrong, features=name, l2=cfg.calib_l2)
        for name in ("r", "margin", "r_margin")
    }


def run_experiment_v1_hopfield_gate(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: Optional[bool] = None,
) -> dict:
    """Run the gate stack on a Hopfield memory (see module docstring)."""
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed

    cfg = config.experiment_v1_gate
    if quick:
        cfg.calib_difficulty_levels = [[64, 8], [128, 16]]
        cfg.calib_n_seeds = 2
        cfg.calib_min_trials_per_level = 32
        cfg.calib_max_episodes_per_level = 3
        cfg.calib_probes_per_key = 4
        cfg.calib_cue_noise_scales = [0.1, 0.3]
        cfg.calib_n_impostors = 8

    save_dir = config.project.save_dir or "results/"
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    base_seed = config.project.seed
    e = cfg.embed_dim
    beta = cfg.hopfield_beta
    rho = cfg.hopfield_gate_correlation
    eval_noise = cfg.hopfield_gate_eval_noise
    levels = [tuple(lv) for lv in cfg.calib_difficulty_levels]
    seeds = [base_seed + i for i in range(cfg.calib_n_seeds)]

    print("\n" + "=" * 64)
    print("EXPERIMENT V1.1: gate stack on a HOPFIELD memory (memory-agnostic?)")
    print("=" * 64)
    print(f"levels={levels} seeds={seeds} beta={beta} ladder={HOPFIELD_LADDER} "
          f"| stress: correlation={rho} eval_noise={eval_noise}")

    cells = []
    for seed_i in seeds:
        master = jax.random.PRNGKey(seed_i)
        embed_key, run_key = jax.random.split(master)
        if rho > 0.0:
            embeds = _clustered_embeddings(
                embed_key, cfg.vocab_size, e, cfg.embed_scale, rho, cfg.regime_n_clusters
            )
        else:
            embeds = make_token_embeddings(embed_key, cfg.vocab_size, e, scale=cfg.embed_scale)
        val_tokens = jnp.arange(cfg.vocab_size // 2, cfg.vocab_size)
        val_embeds = embeds[val_tokens]

        for li, (N, kv) in enumerate(levels):
            n_eps = int(min(cfg.calib_max_episodes_per_level,
                            max(1, math.ceil(cfg.calib_min_trials_per_level / kv))))
            R_d, mg_d, cor_d, pw_d = [], [], [], []
            probe_pw0, probe_wrong0 = [], []
            per_ep_stage0 = []
            cost = None

            for ep in range(n_eps):
                ep_key = jax.random.fold_in(jax.random.fold_in(run_key, li), ep)
                k_data, k_probe, k_pn, k_dn = jax.random.split(ep_key, 4)
                mq = generate_mqar(k_data, 1, N, kv, vocab_size=cfg.vocab_size,
                                   gap_distribution=cfg.gap_distribution,
                                   powerlaw_alpha=cfg.powerlaw_alpha)
                keys_tok, vals_tok = mq["keys"][0], mq["values"][0]
                qk_idx = mq["query_key_idx"][0]
                stored = jnp.concatenate([embeds[keys_tok], embeds[vals_tok]], axis=1)

                # --- write-time self-test -> per-model heads (identical to CLU) ---
                q0p, true_p, _imp = _probe_cues(k_probe, embeds, keys_tok, vals_tok, cfg, e)
                if eval_noise > 0.0:
                    q0p = q0p.at[:, :e].add(jax.random.normal(k_pn, (q0p.shape[0], e)) * eval_noise)
                probe_rec = _hopfield_ladder(q0p, stored, beta, e, val_embeds, val_tokens, true_p)
                heads = _fit_hopfield_heads(probe_rec, cfg)
                deployed = heads[cfg.calib_features]
                probe_pw0.append(deployed.p_wrong(R=probe_rec["R"][:, 0], margin=probe_rec["margin"][:, 0]))
                probe_wrong0.append(~probe_rec["correct"][:, 0])

                # --- deployment (exact cues) on the iteration ladder ---
                true_tok = vals_tok[qk_idx]
                cue = embeds[keys_tok[qk_idx]]
                if eval_noise > 0.0:
                    cue = cue + jax.random.normal(k_dn, cue.shape) * eval_noise
                q0 = jnp.concatenate([cue, jnp.zeros((qk_idx.shape[0], e))], axis=1)
                rec = _hopfield_ladder(q0, stored, beta, e, val_embeds, val_tokens, true_tok)
                R_d.append(rec["R"])
                mg_d.append(rec["margin"])
                cor_d.append(rec["correct"])
                pw_d.append(deployed.p_wrong(R=rec["R"], margin=rec["margin"]))
                per_ep_stage0.append((rec["R"][:, 0], rec["correct"][:, 0]))
                cost = rec["cost"]

            cell = {
                "seed": seed_i, "N": N, "kv": kv,
                "R": np.concatenate(R_d, 0), "margin": np.concatenate(mg_d, 0),
                "correct": np.concatenate(cor_d, 0), "pw": np.concatenate(pw_d, 0),
                "probe_pw0": np.concatenate(probe_pw0, 0),
                "probe_wrong0": np.concatenate(probe_wrong0, 0),
                "per_ep_stage0": per_ep_stage0, "cost": cost,
            }
            cell["metrics"] = _cell_metrics(cell, cfg)
            cells.append(cell)
            mm = cell["metrics"]
            print(f"  [seed {seed_i}] N={N} kv={kv}: trials={mm['trials']} "
                  f"base_acc={mm['base_acc']:.3f} full_acc={mm['full_acc']:.3f} "
                  f"n_wrong0={mm['n_wrong0']} rawAUROC={mm['pooled_raw']:.3f} "
                  f"calibAUROC={mm['pooled_calib']:.3f} "
                  f"gate={mm['gate_acc']:.3f}@{mm['gate_cost']:.2f}({mm['savings']:.2f}x)")

    summary = _summarize(cells, levels, seeds, cfg, base_seed)
    with open(os.path.join(results_dir, "exp_v1_hopfield_gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _print_verdict(summary)
    return {"cells": cells, "summary": summary}


def _cell_metrics(cell, cfg):
    R, cor, pw, cost = cell["R"], cell["correct"], cell["pw"], cell["cost"].astype(float)
    wrong0 = ~cor[:, 0]
    per_ep = [_auroc((~c0).astype(int), r0) for r0, c0 in cell["per_ep_stage0"]]
    m = {
        "trials": int(cor.shape[0]),
        "n_wrong0": int(wrong0.sum()),
        "base_acc": float(cor[:, 0].mean()),
        "full_acc": float(cor[:, -1].mean()),
        "within_model_auroc": float(np.nanmean(per_ep)) if per_ep else float("nan"),
        "pooled_raw": _auroc(wrong0.astype(int), R[:, 0]),
        "pooled_calib": _auroc(wrong0.astype(int), pw[:, 0]),
    }
    acc_pt, cost_pt = _simulate_tau_policy(pw, cor, cost, [cfg.calib_p_exit], mode="le")
    m["gate_acc"] = float(acc_pt[0])
    m["gate_cost"] = float(cost_pt[0])
    m["always_full_acc"] = float(cor[:, -1].mean())
    m["savings"] = float(cost[-1] / cost_pt[0]) if cost_pt[0] > 0 else float("nan")
    # abstention frontier (calibrated gate at full budget)
    conf = 1.0 - pw[:, -1]
    covv, riskv = risk_coverage_curve(conf, cor[:, -1])
    m["aurc"] = aurc(covv, riskv)
    m["cov_at_risk"] = {f"{eps}": coverage_at_risk(covv, riskv, eps)
                        for eps in cfg.calib_risk_targets}
    # LTT: calibrate on probe stage-0, validate on deployment stage-0
    m["ltt"] = {}
    for eps in cfg.calib_risk_targets:
        thr, info = ltt_select_threshold(cell["probe_pw0"], cell["probe_wrong0"],
                                         target_risk=eps, delta=cfg.calib_ltt_delta)
        entry = {"certified": info["certified"]}
        if thr is not None:
            sel = pw[:, 0] <= thr
            entry["eval_coverage"] = float(sel.mean())
            entry["eval_risk"] = float(wrong0[sel].mean()) if sel.any() else float("nan")
            entry["valid"] = bool((not sel.any()) or wrong0[sel].mean() <= eps)
        else:
            entry.update(eval_coverage=0.0, eval_risk=float("nan"), valid=True)
        m["ltt"][f"{eps}"] = entry
    return m


def _agg(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    return {"mean": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std()) if a.size else float("nan"),
            "per_seed": [float(v) for v in vals]}


def _summarize(cells, levels, seeds, cfg, base_seed):
    def of(lv):
        return [c["metrics"] for c in cells if (c["N"], c["kv"]) == lv]

    summary = {"base_seed": base_seed, "seeds": list(seeds),
               "levels": [list(lv) for lv in levels], "gate_features": cfg.calib_features,
               "ladder": list(HOPFIELD_LADDER), "per_level": {}, "pooled": {}}
    for lv in levels:
        ms = of(lv)
        blk = {k: _agg([m[k] for m in ms]) for k in
               ("base_acc", "full_acc", "within_model_auroc", "pooled_raw",
                "pooled_calib", "gate_acc", "gate_cost", "savings", "aurc")}
        blk["n_wrong0"] = [m["n_wrong0"] for m in ms]
        blk["ltt"] = {}
        for eps in cfg.calib_risk_targets:
            entries = [m["ltt"][f"{eps}"] for m in ms]
            blk["ltt"][f"{eps}"] = {
                "certified_frac": float(np.mean([e["certified"] for e in entries])),
                "valid_frac": float(np.mean([e["valid"] for e in entries])),
                "n_valid": int(sum(e["valid"] for e in entries)),
                "n": len(entries),
                "eval_coverage": _agg([e["eval_coverage"] for e in entries]),
            }
        summary["per_level"][f"N{lv[0]}_kv{lv[1]}"] = blk
    # pooled across levels per seed
    praw, pcal = [], []
    for s in seeds:
        cs = [c for c in cells if c["seed"] == s]
        w = np.concatenate([~c["correct"][:, 0] for c in cs])
        R0 = np.concatenate([c["R"][:, 0] for c in cs])
        pw0 = np.concatenate([c["pw"][:, 0] for c in cs])
        praw.append(_auroc(w.astype(int), R0))
        pcal.append(_auroc(w.astype(int), pw0))
    summary["pooled"] = {"auroc_raw_R": _agg(praw), "auroc_calibrated": _agg(pcal)}
    return summary


def _print_verdict(summary):
    p = summary["pooled"]
    print("\n" + "=" * 64)
    print("V1.1 VERDICT — Hopfield gate stack vs the v1-pivot CLU numbers")
    print("=" * 64)
    print(f"  Hopfield pooled AUROC(->wrong): raw R "
          f"{p['auroc_raw_R']['mean']:.3f}±{p['auroc_raw_R']['std']:.3f}  "
          f"vs calibrated {p['auroc_calibrated']['mean']:.3f}±{p['auroc_calibrated']['std']:.3f}")
    print("  (v1-pivot CLU reference: raw 0.431±0.038 -> calibrated 0.869±0.015)")
    for lv, blk in summary["per_level"].items():
        n_valid = sum(v["n_valid"] for v in blk["ltt"].values())
        n_tot = sum(v["n"] for v in blk["ltt"].values())
        print(f"  {lv}: base={blk['base_acc']['mean']:.3f} full={blk['full_acc']['mean']:.3f} "
              f"n_wrong0={blk['n_wrong0']} | savings={blk['savings']['mean']:.2f}x "
              f"@ {blk['gate_cost']['mean']:.2f} iters | AURC={blk['aurc']['mean']:.3f} "
              f"| LTT valid {n_valid}/{n_tot}")
    print("=" * 64 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V1.1: gate stack on Hopfield")
    parser.add_argument("--project")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rho", type=float, help="embedding correlation stress")
    parser.add_argument("--noise", type=float, help="cue eval-noise stress")
    args = parser.parse_args()
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        config = get_default_config()
        save_dir, models_dir = "results", None
        os.makedirs(save_dir, exist_ok=True)
    if args.rho is not None:
        config.experiment_v1_gate.hopfield_gate_correlation = args.rho
    if args.noise is not None:
        config.experiment_v1_gate.hopfield_gate_eval_noise = args.noise
    run_experiment_v1_hopfield_gate(config=config, save_dir=save_dir,
                                    models_dir=models_dir, seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
