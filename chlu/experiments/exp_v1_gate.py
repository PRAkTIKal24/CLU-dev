"""Experiment V1-Gate: L0 boost-retry cascade on MQAR associative recall.

The empirical half of the V1 (adaptive-compute) gate. A CHLU is trained as an
energy-based associative memory on a per-episode MQAR dictionary (generative
PCD, no MSE); retrieval = initialize q at the query-key embedding, relax under
the energy governor, decode the value half of the settled position against the
value vocabulary. Two questions:

  Q1  Does the residual energy R = H(settled) - floor of a relaxed state
      calibrate with retrieval correctness? (AUROC + reliability bins.)
  Q2  Do mass-weighted squeeze retries S^(M) (F5 Def-6/7, Prop-12) recover
      answers plain relaxation misses, at matched compute vs relax-longer?

Arms (per query):
  - "mass":   cascade with mass-weighted squeezes S^(M), candidates selected
              by lowest post-relax residual energy (the V1 mechanism).
  - "raw":    same but mass-blind raw squeezes (comparison flag; F5 §5.4).
  - "kick":   kinetically-matched random momentum kicks (same T-injection as
              the paired squeeze candidate, no structured q-reframe) —
              "is it the squeeze, or just any perturbation?"
  - "margin": same actions as "mass" but gated/selected on the readout
              confidence margin instead of energy (entropy-gated baseline).
  - control:  always-relax-longer at matched total Verlet steps (no boosts).
  - Hopfield: modern-Hopfield retrieval (softmax attention over the stored
              patterns) — the mandatory associative-memory baseline.

Cascades run ungated to the full retry budget while recording per-stage
(residual, margin, prediction); every threshold policy tau is then simulated
post-hoc, which yields the entire calibration/compute curve from one run.

Compute accounting: a governed rollout of `n` output rows executes n-1 Verlet
steps (the initial state is prepended); each line-search candidate costs one
retry relaxation. Squeezes are O(d) and counted as free. Hopfield cost is one
(kv x 2e) matvec — incommensurable with Verlet steps; reported separately.
"""

import json
import math
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.transforms import effective_mass, mass_weighted_squeeze, squeeze
from chlu.data.mqar import generate_mqar, make_token_embeddings
from chlu.training.train_generative import train_generative
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.plotting import (
    plot_v1_gate_calibration,
    plot_v1_gate_compute_curves,
    plot_v1_gate_mass_scatter,
)


# ---------------------------------------------------------------------------
# Jitted primitives
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _settle_batch(model, q0, p0, steps, dt, floor, sensitivity):
    """Governed relaxation of a batch of states; returns final (q, p, H).

    Executes steps-1 Verlet steps per trial (governed_rollout prepends the
    initial state).
    """

    def one(q, p):
        traj = model.governed_rollout(q, p, steps, dt, floor, sensitivity)
        return traj[-1, : model.dim], traj[-1, model.dim :]

    qf, pf = jax.vmap(one)(q0, p0)
    Hf = jax.vmap(model.H)(qf, pf)
    return qf, pf, Hf


@eqx.filter_jit
def _relax_checkpoints_batch(model, q0, p0, steps, dt, floor, sensitivity, idx):
    """Long governed relaxation, returning (q, H) at the requested row indices.

    Row t of a governed rollout is the state after t Verlet steps.
    """

    def one(q, p):
        traj = model.governed_rollout(q, p, steps, dt, floor, sensitivity)
        qs = traj[idx, : model.dim]
        ps = traj[idx, model.dim :]
        Hs = jax.vmap(model.H)(qs, ps)
        return qs, Hs

    return jax.vmap(one)(q0, p0)


def _decode_values(q_value_half, val_embeds, val_tokens):
    """Nearest-neighbor decode of the settled value half against the value
    sub-vocabulary. Returns (predicted tokens, confidence margin d2 - d1)."""
    d2_all = jnp.sum((q_value_half[:, None, :] - val_embeds[None, :, :]) ** 2, axis=-1)
    dist = jnp.sqrt(d2_all)
    neg_top2, top2_idx = jax.lax.top_k(-dist, 2)
    pred = val_tokens[top2_idx[:, 0]]
    margin = -neg_top2[:, 1] - (-neg_top2[:, 0])  # d2 - d1 >= 0
    return pred, margin


def _hopfield_retrieve(q0, patterns, beta, embed_dim, val_embeds, val_tokens):
    """One modern-Hopfield update: z = patterns^T softmax(beta patterns q).

    The query's value half is zero, so the score is the key-half inner
    product; matched stored content (same [key||value] patterns as the CLU).
    """
    scores = beta * (q0 @ patterns.T)
    attn = jax.nn.softmax(scores, axis=1)
    z = attn @ patterns
    return _decode_values(z[:, embed_dim:], val_embeds, val_tokens)


# ---------------------------------------------------------------------------
# The cascade (F5 Def-7, single shell)
# ---------------------------------------------------------------------------


def _run_cascade(
    model,
    m_eff,
    q0,
    p0,
    true_tok,
    val_embeds,
    val_tokens,
    embed_dim,
    floor,
    cfg,
    arm: str,
    select_by: str = "energy",
    kick_key=None,
):
    """Run the ungated boost-retry cascade for one batch of queries.

    Returns a dict of per-stage records (stage 0 = after base relaxation,
    stage b = best-so-far after b retry rounds):
        R:      (T, B+1) residual energy of the best-so-far state
        margin: (T, B+1) readout margin of the best-so-far state
        pred:   (T, B+1) predicted value token
        correct:(T, B+1)
        cost:   (B+1,) cumulative Verlet steps per trial
        scatter: (mass arm only) per-mode displacement records of retry 1
    """
    dt = cfg.dt
    sens = cfg.governor_sensitivity
    B = cfg.retry_budget
    zeta_grid = list(cfg.zeta_grid)
    G = len(zeta_grid)
    floor_j = jnp.asarray(floor)

    q, p, H = _settle_batch(model, q0, p0, cfg.relax_steps, dt, floor_j, sens)
    R = H - floor_j
    pred, margin = _decode_values(q[:, embed_dim:], val_embeds, val_tokens)

    best = {"q": q, "p": p, "R": R, "margin": margin, "pred": pred}
    rec_R = [R]
    rec_margin = [margin]
    rec_pred = [pred]
    cost = [cfg.relax_steps - 1]
    scatter = None

    for b in range(B):
        zetas = [z * (cfg.zeta_scale_per_retry**b) for z in zeta_grid]
        q_entry, p_entry = best["q"], best["p"]

        cand_q, cand_p, cand_R, cand_margin, cand_pred, cand_qboost = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for z in zetas:
            if arm == "raw":
                qb, pb = squeeze(q_entry, p_entry, z)
            elif arm == "kick":
                # Kinetically-matched random kick: same p^T M^-1 p as the
                # paired squeeze candidate would inject, random direction
                # (isotropic in the M^-1 metric), q unchanged.
                _, p_sq = mass_weighted_squeeze(q_entry, p_entry, z, m_eff)
                kick_key, sub = jax.random.split(kick_key)
                u = jax.random.normal(sub, p_entry.shape) * jnp.sqrt(m_eff)
                target = jnp.sqrt(jnp.sum(p_sq**2 / m_eff, axis=1, keepdims=True))
                unorm = jnp.sqrt(jnp.sum(u**2 / m_eff, axis=1, keepdims=True)) + 1e-12
                qb, pb = q_entry, u * target / unorm
            else:  # "mass" (and "margin" uses the same mass-weighted action)
                qb, pb = mass_weighted_squeeze(q_entry, p_entry, z, m_eff)

            qs, ps, Hs = _settle_batch(
                model, qb, pb, cfg.retry_relax_steps, dt, floor_j, sens
            )
            Rs = Hs - floor_j
            pr, mg = _decode_values(qs[:, embed_dim:], val_embeds, val_tokens)
            cand_q.append(qs)
            cand_p.append(ps)
            cand_R.append(Rs)
            cand_margin.append(mg)
            cand_pred.append(pr)
            cand_qboost.append(qb)

        cand_q = jnp.stack(cand_q)  # (G, T, d)
        cand_p = jnp.stack(cand_p)
        cand_R = jnp.stack(cand_R)  # (G, T)
        cand_margin = jnp.stack(cand_margin)
        cand_pred = jnp.stack(cand_pred)
        cand_qboost = jnp.stack(cand_qboost)

        if select_by == "margin":
            chosen = jnp.argmax(cand_margin, axis=0)  # (T,)
        else:
            chosen = jnp.argmin(cand_R, axis=0)
        t_idx = jnp.arange(q0.shape[0])
        ch_q = cand_q[chosen, t_idx]
        ch_p = cand_p[chosen, t_idx]
        ch_R = cand_R[chosen, t_idx]
        ch_margin = cand_margin[chosen, t_idx]
        ch_pred = cand_pred[chosen, t_idx]

        if select_by == "margin":
            improved = ch_margin > best["margin"]
        else:
            improved = ch_R < best["R"]
        imp_col = improved[:, None]
        best = {
            "q": jnp.where(imp_col, ch_q, best["q"]),
            "p": jnp.where(imp_col, ch_p, best["p"]),
            "R": jnp.where(improved, ch_R, best["R"]),
            "margin": jnp.where(improved, ch_margin, best["margin"]),
            "pred": jnp.where(improved, ch_pred, best["pred"]),
        }

        if b == 0 and arm == "mass" and select_by == "energy":
            # Thread-5 falsifiable (ii): per-mode displacement of retry 1.
            ch_qboost = cand_qboost[chosen, t_idx]
            zeta_arr = jnp.asarray(zetas)[chosen]
            scatter = {
                "p_before": np.asarray(p_entry),
                "q_before": np.asarray(q_entry),
                "dq_instant": np.asarray(ch_qboost - q_entry),
                "dq_total": np.asarray(ch_q - q_entry),
                "zeta_chosen": np.asarray(zeta_arr),
            }

        rec_R.append(best["R"])
        rec_margin.append(best["margin"])
        rec_pred.append(best["pred"])
        cost.append(cost[-1] + G * (cfg.retry_relax_steps - 1))

    pred_arr = np.stack([np.asarray(x) for x in rec_pred], axis=1)  # (T, B+1)
    out = {
        "R": np.stack([np.asarray(x) for x in rec_R], axis=1),
        "margin": np.stack([np.asarray(x) for x in rec_margin], axis=1),
        "pred": pred_arr,
        "correct": pred_arr == np.asarray(true_tok)[:, None],
        "cost": np.asarray(cost),
    }
    if scatter is not None:
        out["scatter"] = scatter
    return out


def _simulate_tau_policy(score_stages, correct_stages, cost, taus, mode="le"):
    """Simulate the gated cascade for each threshold from ungated records.

    A query exits at the first stage b whose gate signal passes the threshold
    (score <= tau for energy residuals, score >= tau for margins); otherwise
    it runs the full budget. Returns (accuracy, mean cost) per tau.
    """
    T, S = score_stages.shape
    accs, costs = [], []
    for tau in taus:
        passes = score_stages <= tau if mode == "le" else score_stages >= tau  # (T, S)
        padded = np.concatenate([passes, np.ones((T, 1), bool)], axis=1)
        exit_b = np.argmax(padded, axis=1)
        exit_b = np.minimum(exit_b, S - 1)
        accs.append(float(np.mean(correct_stages[np.arange(T), exit_b])))
        costs.append(float(np.mean(cost[exit_b])))
    return np.asarray(accs), np.asarray(costs)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of `scores` ranking `labels` (1 = positive). NaN if one class."""
    if labels.min() == labels.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def run_experiment_v1_gate(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: Optional[bool] = None,
):
    """Run the V1 L0 gate experiment (see module docstring).

    Args:
        config: CHLUConfig (defaults if None); knobs in config.experiment_v1_gate
        save_dir: plots directory (metrics go to save_dir/../results)
        models_dir: checkpoint directory (defaults to save_dir/../models)
        seed: overrides config.project.seed
        quick: shrink the grid/training for a smoke run

    Returns:
        results dict (per-level records + summary).
    """
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed

    cfg = config.experiment_v1_gate
    if quick:
        cfg.difficulty_levels = [[64, 4], [128, 16]]
        cfg.train_epochs = min(cfg.train_epochs, 120)
        cfg.min_trials_per_level = 16
        cfg.max_episodes_per_level = 2
        cfg.relax_steps = min(cfg.relax_steps, 120)
        cfg.retry_relax_steps = min(cfg.retry_relax_steps, 60)
        cfg.retry_budget = min(cfg.retry_budget, 2)
        cfg.zeta_grid = [-0.4, -0.2, 0.2, 0.4]

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "..", "models")
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    seed = config.project.seed
    e = cfg.embed_dim
    dim = 2 * e
    G = len(cfg.zeta_grid)

    print("\n" + "=" * 60)
    print("EXPERIMENT V1-GATE: L0 boost-retry cascade on MQAR")
    print("=" * 60)
    print(
        f"vocab={cfg.vocab_size} (Zoology default 8192, scaled down), "
        f"embed_dim={e} -> CLU dim={dim}"
    )
    print(
        f"levels (N, kv): {cfg.difficulty_levels} | kinetic="
        f"{cfg.kinetic_energy_mode} | potential={cfg.potential_type}"
    )
    print(
        f"cascade: relax={cfg.relax_steps}, retry={cfg.retry_relax_steps} x "
        f"G={G} candidates x B={cfg.retry_budget} retries"
    )

    master = jax.random.PRNGKey(seed)
    embed_key, run_key = jax.random.split(master)
    embeds = make_token_embeddings(embed_key, cfg.vocab_size, e, scale=cfg.embed_scale)
    val_tokens = jnp.arange(cfg.vocab_size // 2, cfg.vocab_size)
    val_embeds = embeds[val_tokens]

    arms = ["mass"]
    if cfg.compare_raw_squeeze:
        arms.append("raw")
    if cfg.compare_noise_kick:
        arms.append("kick")
    arms.append("margin")

    results = {"levels": [], "config_seed": seed}
    scatter_pool = []

    for li, (N, kv) in enumerate(cfg.difficulty_levels):
        n_eps = int(
            min(
                cfg.max_episodes_per_level,
                max(1, math.ceil(cfg.min_trials_per_level / kv)),
            )
        )
        print(f"\n[Level {li}] N={N}, kv={kv}, episodes={n_eps}")

        lvl = {
            "N": N,
            "kv": kv,
            "R0": [],
            "margin0": [],
            "correct0": [],
            "fidelity": [],
            "hop_correct": [],
            "ctrl_correct": [],  # (T, B+1) checkpointed control
            "arms": {a: {"R": [], "margin": [], "correct": []} for a in arms},
            "cost": None,
            "ctrl_cost": None,
            "m_eff": [],
        }

        for ep in range(n_eps):
            ep_key = jax.random.fold_in(jax.random.fold_in(run_key, li), ep)
            k_data, k_model, k_train, k_kick = jax.random.split(ep_key, 4)

            mq = generate_mqar(
                k_data,
                1,
                N,
                kv,
                vocab_size=cfg.vocab_size,
                gap_distribution=cfg.gap_distribution,
                powerlaw_alpha=cfg.powerlaw_alpha,
            )
            keys_tok = mq["keys"][0]
            vals_tok = mq["values"][0]
            qk_idx = mq["query_key_idx"][0]
            stored = jnp.concatenate(
                [embeds[keys_tok], embeds[vals_tok]], axis=1
            )  # (kv, dim)

            model = CHLU(
                dim=dim,
                hidden=cfg.hidden_dim,
                rest_mass=config.model.rest_mass,
                c=config.model.speed_of_causality,
                kinetic_mode=cfg.kinetic_energy_mode,
                potential_type=cfg.potential_type,
                key=k_model,
            )

            ckpt = os.path.join(models_dir, f"v1gate_N{N}_kv{kv}_ep{ep}_seed{seed}.pkl")
            if cfg.use_pretrained and os.path.exists(ckpt):
                model, meta = load_checkpoint(ckpt)
                floor = float(meta["target_energy"])
                print(f"  ep{ep}: loaded {ckpt} (floor={floor:.4f})")
            else:
                model, _losses, floor = train_generative(
                    model,
                    stored,
                    key=k_train,
                    config=config,
                    epochs=cfg.train_epochs,
                    lr=cfg.train_lr,
                    batch_size=cfg.train_batch_size,
                    dt=cfg.dt,
                    buffer_capacity=cfg.train_buffer_capacity,
                    k_steps=cfg.train_k_steps,
                    sleep_friction=cfg.train_friction,
                    sleep_temperature=cfg.train_temperature,
                    input_noise_sigma=cfg.train_input_noise_sigma,
                )
                save_checkpoint(
                    model,
                    ckpt,
                    epoch=cfg.train_epochs,
                    loss=float(_losses["total"][-1]),
                    config=None,
                    target_energy=floor,
                )

            m_eff = effective_mass(model)
            lvl["m_eff"].append(np.asarray(m_eff))
            floor_j = jnp.asarray(floor)

            # --- trials: one per query instance in the MQAR sequence ---
            true_tok = vals_tok[qk_idx]
            q0 = jnp.concatenate(
                [embeds[keys_tok[qk_idx]], jnp.zeros((qk_idx.shape[0], e))],
                axis=1,
            )
            p0 = jnp.zeros_like(q0)

            # Storage fidelity: relax from the stored pattern itself.
            qs_f, _, _ = _settle_batch(
                model,
                stored,
                jnp.zeros_like(stored),
                cfg.relax_steps,
                cfg.dt,
                floor_j,
                cfg.governor_sensitivity,
            )
            pred_f, _ = _decode_values(qs_f[:, e:], val_embeds, val_tokens)
            lvl["fidelity"].append(np.asarray(pred_f == vals_tok))

            # --- arms ---
            for arm in arms:
                sel = "margin" if arm == "margin" else "energy"
                rec = _run_cascade(
                    model,
                    m_eff,
                    q0,
                    p0,
                    true_tok,
                    val_embeds,
                    val_tokens,
                    e,
                    floor,
                    cfg,
                    arm="mass" if arm == "margin" else arm,
                    select_by=sel,
                    kick_key=k_kick if arm == "kick" else None,
                )
                lvl["arms"][arm]["R"].append(rec["R"])
                lvl["arms"][arm]["margin"].append(rec["margin"])
                lvl["arms"][arm]["correct"].append(rec["correct"])
                if arm == "mass":
                    lvl["cost"] = rec["cost"]
                    lvl["R0"].append(rec["R"][:, 0])
                    lvl["margin0"].append(rec["margin"][:, 0])
                    lvl["correct0"].append(rec["correct"][:, 0])
                    if "scatter" in rec:
                        sc = rec["scatter"]
                        sc["m_eff"] = np.asarray(m_eff)
                        sc["correct0"] = np.asarray(rec["correct"][:, 0])
                        sc["level"] = li
                        scatter_pool.append(sc)

            # --- always-relax-longer control at matched budgets ---
            total = (cfg.relax_steps - 1) + cfg.retry_budget * G * (
                cfg.retry_relax_steps - 1
            )
            ckpts = np.asarray(
                [
                    (cfg.relax_steps - 1) + b * G * (cfg.retry_relax_steps - 1)
                    for b in range(cfg.retry_budget + 1)
                ]
            )
            qs_c, _H_c = _relax_checkpoints_batch(
                model,
                q0,
                p0,
                total + 1,
                cfg.dt,
                floor_j,
                cfg.governor_sensitivity,
                jnp.asarray(ckpts),
            )
            pred_c = []
            for bi in range(len(ckpts)):
                pr, _ = _decode_values(qs_c[:, bi, e:], val_embeds, val_tokens)
                pred_c.append(np.asarray(pr == true_tok))
            lvl["ctrl_correct"].append(np.stack(pred_c, axis=1))
            lvl["ctrl_cost"] = ckpts

            # --- modern-Hopfield baseline (matched stored content) ---
            pred_h, _ = _hopfield_retrieve(
                q0, stored, cfg.hopfield_beta, e, val_embeds, val_tokens
            )
            lvl["hop_correct"].append(np.asarray(pred_h == true_tok))

        # concat episodes
        for k in ["R0", "margin0", "correct0", "fidelity", "hop_correct"]:
            lvl[k] = np.concatenate(lvl[k])
        lvl["ctrl_correct"] = np.concatenate(lvl["ctrl_correct"], axis=0)
        for a in arms:
            for k in ["R", "margin", "correct"]:
                lvl["arms"][a][k] = np.concatenate(lvl["arms"][a][k], axis=0)
        results["levels"].append(lvl)
        print(
            f"  trials={len(lvl['R0'])}, base acc={lvl['correct0'].mean():.3f},"
            f" fidelity={lvl['fidelity'].mean():.3f},"
            f" Hopfield acc={lvl['hop_correct'].mean():.3f}"
        )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    summary = {"levels": [], "seed": seed, "arms": arms}
    for lvl in results["levels"]:
        wrong0 = ~lvl["correct0"]
        n_wrong = int(wrong0.sum())
        B1 = lvl["arms"]["mass"]["correct"].shape[1]
        entry = {
            "N": lvl["N"],
            "kv": lvl["kv"],
            "trials": int(len(lvl["R0"])),
            "base_acc": float(lvl["correct0"].mean()),
            "fidelity": float(lvl["fidelity"].mean()),
            "hopfield_acc": float(lvl["hop_correct"].mean()),
            "auroc_R_vs_incorrect": _auroc(wrong0.astype(int), lvl["R0"]),
            "auroc_margin_vs_correct": _auroc(
                lvl["correct0"].astype(int), lvl["margin0"]
            ),
            "n_initially_wrong": n_wrong,
            "m_eff_spread": float(np.std(np.log(np.concatenate(lvl["m_eff"])))),
        }
        for a in lvl["arms"]:
            c = lvl["arms"][a]["correct"]
            entry[f"{a}_final_acc"] = float(c[:, -1].mean())
            entry[f"{a}_recovery"] = (
                float(c[wrong0, -1].mean()) if n_wrong else float("nan")
            )
            entry[f"{a}_flips"] = int(np.sum(lvl["correct0"] & ~c[:, -1]))
        cc = lvl["ctrl_correct"]
        entry["control_final_acc"] = float(cc[:, -1].mean())
        entry["control_recovery"] = (
            float(cc[wrong0, -1].mean()) if n_wrong else float("nan")
        )
        entry["cost_per_stage"] = lvl["cost"].tolist()
        assert B1 == cc.shape[1]
        summary["levels"].append(entry)

    # pooled Q1/Q2
    R0_all = np.concatenate([lv["R0"] for lv in results["levels"]])
    c0_all = np.concatenate([lv["correct0"] for lv in results["levels"]])
    wrong_all = ~c0_all
    summary["pooled"] = {
        "auroc_R_vs_incorrect": _auroc(wrong_all.astype(int), R0_all),
        "base_acc": float(c0_all.mean()),
        "n_trials": int(len(c0_all)),
        "n_initially_wrong": int(wrong_all.sum()),
    }
    for a in arms:
        c = np.concatenate(
            [lv["arms"][a]["correct"] for lv in results["levels"]], axis=0
        )
        summary["pooled"][f"{a}_recovery"] = (
            float(c[wrong_all, -1].mean()) if wrong_all.any() else float("nan")
        )
        summary["pooled"][f"{a}_final_acc"] = float(c[:, -1].mean())
    cc = np.concatenate([lv["ctrl_correct"] for lv in results["levels"]], axis=0)
    summary["pooled"]["control_recovery"] = (
        float(cc[wrong_all, -1].mean()) if wrong_all.any() else float("nan")
    )
    summary["pooled"]["control_final_acc"] = float(cc[:, -1].mean())

    # per-mode displacement vs 1/M (Thread-5 falsifiable (ii))
    if scatter_pool:
        inv_m = np.concatenate(
            [
                np.tile(1.0 / s["m_eff"], (s["dq_total"].shape[0], 1)).ravel()
                for s in scatter_pool
            ]
        )
        dq = np.concatenate([np.abs(s["dq_total"]).ravel() for s in scatter_pool])
        from scipy.stats import spearmanr

        rho, pval = spearmanr(np.log(inv_m), np.log(dq + 1e-12))
        summary["mass_scatter"] = {
            "spearman_log_dq_vs_log_invM": float(rho),
            "p_value": float(pval),
            "n_points": int(len(dq)),
        }

    # -----------------------------------------------------------------
    # Save + plots
    # -----------------------------------------------------------------
    npz = {}
    for li, lvl in enumerate(results["levels"]):
        pre = f"L{li}_"
        npz[pre + "R0"] = lvl["R0"]
        npz[pre + "margin0"] = lvl["margin0"]
        npz[pre + "correct0"] = lvl["correct0"]
        npz[pre + "fidelity"] = lvl["fidelity"]
        npz[pre + "hop_correct"] = lvl["hop_correct"]
        npz[pre + "ctrl_correct"] = lvl["ctrl_correct"]
        npz[pre + "ctrl_cost"] = lvl["ctrl_cost"]
        npz[pre + "cost"] = lvl["cost"]
        npz[pre + "m_eff"] = np.stack(lvl["m_eff"])
        for a in lvl["arms"]:
            for k in ["R", "margin", "correct"]:
                npz[pre + f"{a}_{k}"] = lvl["arms"][a][k]
    for si, sc in enumerate(scatter_pool):
        for k in ["p_before", "dq_instant", "dq_total", "zeta_chosen", "m_eff"]:
            npz[f"scatter{si}_{k}"] = sc[k]
    metrics_path = os.path.join(results_dir, "exp_v1_gate_metrics.npz")
    np.savez(metrics_path, **npz)
    summary_path = os.path.join(results_dir, "exp_v1_gate_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}\nSaved summary to {summary_path}")

    # Q1 calibration plot
    plot_v1_gate_calibration(
        [lv["R0"] for lv in results["levels"]],
        [lv["correct0"] for lv in results["levels"]],
        [f"N={lv['N']}, kv={lv['kv']}" for lv in results["levels"]],
        [e["auroc_R_vs_incorrect"] for e in summary["levels"]],
        os.path.join(save_dir, "exp_v1_gate_calibration.png"),
    )

    # compute-vs-accuracy curves (pooled over levels), tau swept post-hoc
    taus = np.quantile(R0_all, np.linspace(0.05, 0.95, cfg.n_tau))
    curves = {}
    for a in arms:
        R = np.concatenate([lv["arms"][a]["R"] for lv in results["levels"]], axis=0)
        M = np.concatenate(
            [lv["arms"][a]["margin"] for lv in results["levels"]], axis=0
        )
        C = np.concatenate(
            [lv["arms"][a]["correct"] for lv in results["levels"]], axis=0
        )
        cost = results["levels"][0]["cost"]
        if a == "margin":
            m_taus = np.quantile(
                np.concatenate([lv["margin0"] for lv in results["levels"]]),
                np.linspace(0.05, 0.95, cfg.n_tau),
            )
            acc, cst = _simulate_tau_policy(M, C, cost, m_taus, mode="ge")
        else:
            acc, cst = _simulate_tau_policy(R, C, cost, taus, mode="le")
        curves[a] = (cst, acc)
    ctrl_acc = cc.mean(axis=0)
    curves["relax-longer"] = (
        results["levels"][0]["ctrl_cost"].astype(float),
        ctrl_acc,
    )
    hop_acc = float(
        np.concatenate([lv["hop_correct"] for lv in results["levels"]]).mean()
    )
    plot_v1_gate_compute_curves(
        curves,
        hop_acc,
        os.path.join(save_dir, "exp_v1_gate_compute_curves.png"),
        title="V1 gate: accuracy vs compute (pooled; tau swept post-hoc)",
    )

    # mass scatter
    if scatter_pool:
        plot_v1_gate_mass_scatter(
            scatter_pool,
            os.path.join(save_dir, "exp_v1_gate_mass_scatter.png"),
        )

    # -----------------------------------------------------------------
    # Gate read
    # -----------------------------------------------------------------
    aurocs = [e["auroc_R_vs_incorrect"] for e in summary["levels"]]
    print("\n" + "=" * 60)
    print("V1 GATE READ (heuristics; final read is the Hub's)")
    print("=" * 60)
    for e_lvl, a in zip(summary["levels"], aurocs, strict=True):
        print(
            f"  N={e_lvl['N']:>3} kv={e_lvl['kv']:>3}: base={e_lvl['base_acc']:.3f}"
            f" fid={e_lvl['fidelity']:.3f} AUROC(R->wrong)={a:.3f}"
            f" | recov mass={e_lvl['mass_recovery']:.3f}"
            f" ctrl={e_lvl['control_recovery']:.3f}"
            f" hop={e_lvl['hopfield_acc']:.3f}"
        )
    p = summary["pooled"]
    print(
        f"  POOLED: AUROC={p['auroc_R_vs_incorrect']:.3f}"
        f" | recovery mass={p['mass_recovery']:.3f} vs control="
        f"{p['control_recovery']:.3f} (n_wrong={p['n_initially_wrong']})"
    )
    print("=" * 60 + "\n")

    results["summary"] = summary
    return results
