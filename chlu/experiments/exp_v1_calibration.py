"""Experiment V1-Calibration: calibrated energy-gated compute allocation.

The pivoted V1 headline (Head decision 2026-07-07): squeeze retries are
PARKED (kept importable/tested in chlu.core.transforms, not exercised here);
the mechanism under test is the **learned per-instance gate**. A CLU trained
as an energy-based associative memory (generative PCD on a per-episode MQAR
dictionary, as in exp_v1_gate) finishes its write phase with a *self-test*:
probe relaxations from jittered cues + impostor cues fit a per-model
CalibrationHead mapping (residual energy R = H(settled) - floor, readout
margin) -> p_wrong. Deployment queries then run a staged-relaxation
escalation ladder (the gate run's exact cost ladder), gated per stage by the
head; abstention uses the same signal.

Powered protocol (v1-pivot task): kv in {16, 24, 32} x >= 5 seeds; enough
episodes per level that recovery/abstention deltas carry error bars.

Measurements:
  1. Calibration: does the training-time head generalize to exact-cue eval
     queries? Headline: pooled cross-model AUROC of raw R (0.330 in the gate
     run) vs calibrated p_wrong.
  2. Abstention head-to-head: risk-coverage / AURC / coverage@risk / ECE for
     CLU + learned gate vs modern Hopfield with naive (max-softmax, logit
     margin) and Platt-calibrated confidences (same probe data - fair), and
     the entropy-gated CLU (readout margin, raw + calibrated).
  3. Compute allocation: tau-gated staged relaxation cost-vs-accuracy versus
     always-small / always-full / entropy-gated, mean +/- std over seeds;
     the *learned* operating point (p_exit), not just post-hoc sweeps.
  4. Deployment wrapper: Learn-then-Test threshold on the calibrated gate
     (risk targets from config); empirical validity on eval queries reported
     (probes are jittered cues, eval cues exact -> exchangeability is an
     approximation to be measured, not assumed).

ECE is reported only for probability-valued signals (calibrated heads,
Hopfield max-softmax); raw margins/energies are not probabilities and get
NaN there (documented deviation from "ECE of each confidence signal").

Nomenclature (F5 Def-2): inertial/spectral mass language does not appear;
this experiment consumes energies and probabilities only.
"""

import json
import math
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.data.mqar import generate_mqar, make_token_embeddings
from chlu.experiments.exp_v1_gate import (
    _auroc,
    _decode_values,
    _settle_batch,
    _simulate_tau_policy,
)
from chlu.training.calibration import (
    fit_calibration_head,
    ltt_select_threshold,
)
from chlu.training.train_generative import train_generative
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.metrics import (
    aurc,
    coverage_at_risk,
    expected_calibration_error,
    interpolate_risk_coverage,
    risk_coverage_curve,
)
from chlu.utils.plotting import (
    plot_v1_calib_compute,
    plot_v1_calib_reliability,
    plot_v1_calib_risk_coverage,
    plot_v1_regime_map,
)

#: methods whose confidence IS a probability of correctness (ECE-eligible)
PROBABILISTIC_METHODS = (
    "clu_calib_r",
    "clu_calib_margin",
    "clu_calib_rm",
    "hop_msp",
    "hop_calib",
)


# ---------------------------------------------------------------------------
# Episode-level pieces
# ---------------------------------------------------------------------------


def _probe_cues(key, embeds, keys_tok, vals_tok, cfg, e):
    """Build the write-time self-test cue set for one episode.

    Jittered cues: each stored key repeated calib_probes_per_key times with
    Gaussian jitter at the configured noise scales (cycled), so the split
    contains both easy and hard retrievals. Impostor cues: unbound key-vocab
    tokens (exact embeddings); the memory holds no binding for them, so any
    decoded answer is wrong by definition (their true token is the -1
    sentinel, which never matches a value token). Impostors guarantee the
    wrong class is populated even when the memory self-tests perfectly.

    Note the leakage rule: probes must be jittered (never the exact stored
    key with p=0), because retrieval is deterministic and the exact cue IS
    the eval query - fitting on it would fit on eval outcomes.

    Returns:
        (q0, true_tok, is_impostor): (T, 2e) cue||zeros states, (T,) value
        tokens (-1 for impostors), (T,) bool.
    """
    P = cfg.calib_probes_per_key
    scales = [
        cfg.calib_cue_noise_scales[j % len(cfg.calib_cue_noise_scales)]
        for j in range(P)
    ]
    kv = int(keys_tok.shape[0])
    k_jit, k_imp = jax.random.split(key)

    base = jnp.repeat(embeds[keys_tok], P, axis=0)  # (kv*P, e)
    sig = jnp.tile(jnp.asarray(scales), kv)[:, None]  # (kv*P, 1)
    cues = base + jax.random.normal(k_jit, base.shape) * sig
    true_tok = jnp.repeat(vals_tok, P)
    is_imp = np.zeros(kv * P, dtype=bool)

    n_imp = int(cfg.calib_n_impostors)
    if n_imp > 0:
        allowed = np.setdiff1d(np.arange(1, cfg.vocab_size // 2), np.asarray(keys_tok))
        imp_tok = jax.random.choice(
            k_imp, jnp.asarray(allowed), shape=(n_imp,), replace=False
        )
        cues = jnp.concatenate([cues, embeds[imp_tok]], axis=0)
        true_tok = jnp.concatenate([true_tok, -jnp.ones(n_imp, dtype=true_tok.dtype)])
        is_imp = np.concatenate([is_imp, np.ones(n_imp, dtype=bool)])

    q0 = jnp.concatenate([cues, jnp.zeros((cues.shape[0], e))], axis=1)
    return q0, true_tok, is_imp


def _ladder_records(model, q0, p0, true_tok, val_embeds, val_tokens, e, floor, cfg, cd):
    """Staged governed relaxation on the escalation cost ladder.

    Chains _settle_batch: base relax_steps, then calib_n_stages blocks of
    calib_stage_steps each (defaults reproduce the v1-l0-gate cost ladder
    300/1200/2100/3000). The cue clamp invariant is maintained across calls
    (each call re-clamps to its own initial key half, which is already the
    cue). Records (R, margin, pred, correct) at every checkpoint.

    Returns:
        dict with R/margin/pred/correct as (T, S) arrays (S = n_stages + 1)
        and cost (S,) cumulative Verlet steps.
    """
    dt = cfg.dt
    sens = cfg.governor_sensitivity
    floor_j = jnp.asarray(floor)

    q, p, H = _settle_batch(model, q0, p0, cfg.relax_steps, dt, floor_j, sens, cd)
    pred, margin = _decode_values(q[:, e:], val_embeds, val_tokens)
    Rs = [np.asarray(H - floor_j)]
    preds = [np.asarray(pred)]
    margins = [np.asarray(margin)]
    cost = [cfg.relax_steps]

    for _s in range(cfg.calib_n_stages):
        q, p, H = _settle_batch(
            model, q, p, cfg.calib_stage_steps, dt, floor_j, sens, cd
        )
        pred, margin = _decode_values(q[:, e:], val_embeds, val_tokens)
        Rs.append(np.asarray(H - floor_j))
        preds.append(np.asarray(pred))
        margins.append(np.asarray(margin))
        cost.append(cost[-1] + cfg.calib_stage_steps)

    pred_arr = np.stack(preds, axis=1)  # (T, S)
    return {
        "R": np.stack(Rs, axis=1),
        "margin": np.stack(margins, axis=1),
        "pred": pred_arr,
        "correct": pred_arr == np.asarray(true_tok)[:, None],
        "cost": np.asarray(cost),
    }


def _hopfield_confidences(q0, patterns, beta, e, val_embeds, val_tokens):
    """Modern-Hopfield retrieval with its native confidence signals.

    One softmax-attention update over the stored [key||value] patterns (same
    content the CLU was trained on). Confidences: max softmax probability
    (MSP) and the top-2 logit margin beta*(s1 - s2) - the latter does not
    saturate at high beta and is the stronger naive signal.

    Returns:
        (pred, msp, logit_margin) as numpy arrays.
    """
    scores = beta * (q0 @ patterns.T)  # (T, kv)
    attn = jax.nn.softmax(scores, axis=1)
    msp = jnp.max(attn, axis=1)
    top2, _ = jax.lax.top_k(scores, 2)
    logit_margin = top2[:, 0] - top2[:, 1]
    z = attn @ patterns
    pred, _rm = _decode_values(z[:, e:], val_embeds, val_tokens)
    return np.asarray(pred), np.asarray(msp), np.asarray(logit_margin)


def _fit_episode_heads(probe_rec, hop_wrong, hop_logit_margin, cfg):
    """Fit this episode's calibration heads on its self-test split.

    CLU heads ("r", "margin", "r_margin") are fitted on the probe ladder
    records - all stages pooled if calib_fit_all_stages (the head then models
    p(wrong | R, margin) regardless of how much compute produced the state),
    else stage 0 only. The Hopfield head is a 1-feature Platt fit on its
    logit margin from the same probe cues (the fitter is feature-agnostic;
    the score is passed through the single-feature slot).
    """
    if cfg.calib_fit_all_stages:
        R = probe_rec["R"].ravel()
        mg = probe_rec["margin"].ravel()
        wrong = (~probe_rec["correct"]).ravel()
    else:
        R = probe_rec["R"][:, 0]
        mg = probe_rec["margin"][:, 0]
        wrong = ~probe_rec["correct"][:, 0]

    heads = {
        name: fit_calibration_head(
            R=R, margin=mg, wrong=wrong, features=name, l2=cfg.calib_l2
        )
        for name in ("r", "margin", "r_margin")
    }
    heads["hopfield"] = fit_calibration_head(
        R=hop_logit_margin, wrong=hop_wrong, features="r", l2=cfg.calib_l2
    )
    return heads


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _agg(values):
    """Aggregate a per-seed list of scalars into mean/std/per_seed (NaN-safe)."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "per_seed": [float(v) for v in arr],
    }


def _stage0_confidences(cell):
    """Per-method (confidence, correct) pairs at the base stage."""
    out = {
        "clu_R_raw": (-cell["R"][:, 0], cell["correct"][:, 0]),
        "clu_margin_raw": (cell["margin"][:, 0], cell["correct"][:, 0]),
        "clu_calib_r": (1.0 - cell["pw_r"][:, 0], cell["correct"][:, 0]),
        "clu_calib_margin": (1.0 - cell["pw_margin"][:, 0], cell["correct"][:, 0]),
        "clu_calib_rm": (1.0 - cell["pw_rm"][:, 0], cell["correct"][:, 0]),
        "hop_msp": (cell["hop_msp"], cell["hop_correct"]),
        "hop_logit_margin": (cell["hop_lm"], cell["hop_correct"]),
        "hop_calib": (1.0 - cell["pw_hop"], cell["hop_correct"]),
    }
    return out


def _cell_metrics(cell, cfg):
    """Selective-prediction + compute + LTT metrics for one (level, seed) cell."""
    m = {}
    conf = _stage0_confidences(cell)
    risk_targets = list(cfg.calib_risk_targets)

    m["auroc"], m["aurc"], m["cov_at_risk"], m["ece"] = {}, {}, {}, {}
    for name, (c, cor) in conf.items():
        cov, risk = risk_coverage_curve(c, cor)
        m["auroc"][name] = _auroc(np.asarray(cor).astype(int), np.asarray(c))
        m["aurc"][name] = aurc(cov, risk)
        for eps in risk_targets:
            m["cov_at_risk"][f"{name}@{eps}"] = coverage_at_risk(cov, risk, eps)
        if name in PROBABILISTIC_METHODS:
            m["ece"][name] = expected_calibration_error(c, cor)
        else:
            m["ece"][name] = float("nan")

    # post-escalation abstention (deployed head at the final ladder stage)
    conf_full = 1.0 - cell["pw_deployed"][:, -1]
    cov_f, risk_f = risk_coverage_curve(conf_full, cell["correct"][:, -1])
    m["aurc"]["clu_calib_deployed_full"] = aurc(cov_f, risk_f)
    for eps in risk_targets:
        m["cov_at_risk"][f"clu_calib_deployed_full@{eps}"] = coverage_at_risk(
            cov_f, risk_f, eps
        )

    # calibration transfer: within-model vs pooled raw vs pooled calibrated
    wrong0 = ~cell["correct"][:, 0]
    per_ep = [_auroc((~c0).astype(int), r0) for r0, c0 in cell["per_episode_stage0"]]
    m["within_model_auroc_R"] = float(np.nanmean(per_ep)) if per_ep else float("nan")
    m["pooled_raw_R_auroc"] = _auroc(wrong0.astype(int), cell["R"][:, 0])
    m["pooled_calibrated_auroc"] = _auroc(wrong0.astype(int), cell["pw_deployed"][:, 0])
    m["energy_over_margin_auroc_delta"] = (
        m["auroc"]["clu_calib_rm"] - m["auroc"]["clu_calib_margin"]
    )

    # compute-allocation policies on the escalation ladder
    cost = cell["cost"].astype(float)
    correct = cell["correct"]
    acc_pt, cost_pt = _simulate_tau_policy(
        cell["pw_deployed"], correct, cost, [cfg.calib_p_exit], mode="le"
    )
    m["compute"] = {
        "learned_point": {"acc": float(acc_pt[0]), "cost": float(cost_pt[0])},
        "always_small": {"acc": float(correct[:, 0].mean()), "cost": float(cost[0])},
        "always_full": {"acc": float(correct[:, -1].mean()), "cost": float(cost[-1])},
        "hopfield_acc": float(cell["hop_correct"].mean()),
    }
    n_taus = cfg.calib_n_policy_taus
    p_grid = np.linspace(0.02, 0.98, n_taus)
    sweeps = {}
    for name, (stages, mode, taus) in {
        "calib_deployed": (cell["pw_deployed"], "le", p_grid),
        "calib_margin": (cell["pw_margin"], "le", p_grid),
        "margin_raw": (
            cell["margin"],
            "ge",
            np.quantile(cell["margin"][:, 0], np.linspace(0.02, 0.98, n_taus)),
        ),
        "R_raw": (
            cell["R"],
            "le",
            np.quantile(cell["R"][:, 0], np.linspace(0.02, 0.98, n_taus)),
        ),
    }.items():
        acc, cst = _simulate_tau_policy(stages, correct, cost, taus, mode=mode)
        sweeps[name] = {"acc": acc.tolist(), "cost": cst.tolist()}
    m["compute"]["sweeps"] = sweeps

    # Learn-then-Test deployment wrapper (calibration pool = stage-0 probes)
    m["ltt"] = {}
    pw_cal = cell["probe_pw0"]
    wrong_cal = cell["probe_wrong0"]
    pw_eval = cell["pw_deployed"][:, 0]
    for eps in risk_targets:
        thr, info = ltt_select_threshold(
            pw_cal, wrong_cal, target_risk=eps, delta=cfg.calib_ltt_delta
        )
        entry = {"certified": info["certified"], "threshold": thr}
        if thr is not None:
            sel = pw_eval <= thr
            entry["eval_coverage"] = float(sel.mean())
            entry["eval_risk"] = (
                float(wrong0[sel].mean()) if sel.any() else float("nan")
            )
            entry["valid"] = bool((not sel.any()) or wrong0[sel].mean() <= eps)
        else:
            entry["eval_coverage"] = 0.0
            entry["eval_risk"] = float("nan")
            entry["valid"] = True  # abstain-everything is vacuously safe
        m["ltt"][f"eps={eps}"] = entry

    m["base_acc"] = float(correct[:, 0].mean())
    m["full_acc"] = float(correct[:, -1].mean())
    m["hop_acc"] = float(cell["hop_correct"].mean())
    m["fidelity"] = float(cell["fidelity"].mean())
    m["trials"] = int(correct.shape[0])
    m["n_wrong0"] = int(wrong0.sum())
    m["probe_wrong_frac"] = float(cell["probe_wrong0"].mean())
    m["degenerate_heads"] = int(cell["n_degenerate"])
    return m


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_experiment_v1_calibration(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: Optional[bool] = None,
):
    """Run the pivoted V1 experiment (see module docstring).

    Args:
        config: CHLUConfig (defaults if None); knobs in config.experiment_v1_gate
        save_dir: plots directory (metrics go to save_dir/../results)
        models_dir: checkpoint directory (defaults to save_dir/../models)
        seed: base seed (replicates are seed + 0 .. seed + calib_n_seeds - 1)
        quick: shrink grids/training for a smoke run

    Returns:
        results dict with per-cell records and the aggregated summary.
    """
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
        cfg.calib_min_trials_per_level = 16
        cfg.calib_max_episodes_per_level = 2
        cfg.train_epochs = min(cfg.train_epochs, 120)
        cfg.relax_steps = min(cfg.relax_steps, 120)
        cfg.calib_n_stages = 2
        cfg.calib_stage_steps = 240
        cfg.calib_probes_per_key = 4
        cfg.calib_cue_noise_scales = [0.1, 0.3]
        cfg.calib_n_impostors = 8

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "..", "models")
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    base_seed = config.project.seed
    e = cfg.embed_dim
    dim = 2 * e
    levels = [tuple(lv) for lv in cfg.calib_difficulty_levels]
    seeds = [base_seed + i for i in range(cfg.calib_n_seeds)]

    print("\n" + "=" * 60)
    print("EXPERIMENT V1-CALIBRATION: learned energy gate on MQAR")
    print("=" * 60)
    print(
        f"levels (N, kv): {levels} | seeds: {seeds} | "
        f"kinetic={cfg.kinetic_energy_mode} | potential={cfg.potential_type}"
    )
    print(
        f"ladder: {cfg.relax_steps} + {cfg.calib_n_stages} x "
        f"{cfg.calib_stage_steps} steps | gate features={cfg.calib_features} "
        f"| p_exit={cfg.calib_p_exit} | clamp_key={cfg.clamp_key}"
    )
    print(
        f"self-test: {cfg.calib_probes_per_key} probes/key @ sigma="
        f"{cfg.calib_cue_noise_scales} + {cfg.calib_n_impostors} impostors "
        f"(fit_all_stages={cfg.calib_fit_all_stages})"
    )

    cells = []
    for seed_i in seeds:
        master = jax.random.PRNGKey(seed_i)
        embed_key, run_key = jax.random.split(master)
        embeds = make_token_embeddings(
            embed_key, cfg.vocab_size, e, scale=cfg.embed_scale
        )
        val_tokens = jnp.arange(cfg.vocab_size // 2, cfg.vocab_size)
        val_embeds = embeds[val_tokens]

        for li, (N, kv) in enumerate(levels):
            n_eps = int(
                min(
                    cfg.calib_max_episodes_per_level,
                    max(1, math.ceil(cfg.calib_min_trials_per_level / kv)),
                )
            )
            cell = {
                "seed": seed_i,
                "N": N,
                "kv": kv,
                "R": [],
                "margin": [],
                "correct": [],
                "pw_r": [],
                "pw_margin": [],
                "pw_rm": [],
                "hop_correct": [],
                "hop_msp": [],
                "hop_lm": [],
                "pw_hop": [],
                "probe_pw0": [],
                "probe_wrong0": [],
                "probe_is_imp": [],
                "fidelity": [],
                "per_episode_stage0": [],
                "n_degenerate": 0,
                "cost": None,
            }

            for ep in range(n_eps):
                ep_key = jax.random.fold_in(jax.random.fold_in(run_key, li), ep)
                k_data, k_model, k_train, k_probe = jax.random.split(ep_key, 4)

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
                stored = jnp.concatenate([embeds[keys_tok], embeds[vals_tok]], axis=1)

                model = CHLU(
                    dim=dim,
                    hidden=cfg.hidden_dim,
                    rest_mass=config.model.rest_mass,
                    c=config.model.speed_of_causality,
                    kinetic_mode=cfg.kinetic_energy_mode,
                    potential_type=cfg.potential_type,
                    key=k_model,
                )

                ckpt = os.path.join(
                    models_dir, f"v1calib_N{N}_kv{kv}_ep{ep}_seed{seed_i}.pkl"
                )
                if cfg.use_pretrained and os.path.exists(ckpt):
                    model, meta = load_checkpoint(ckpt)
                    floor = float(meta["target_energy"])
                    print(f"  s{seed_i} ep{ep}: loaded {ckpt}")
                    final_loss = float(meta.get("loss", float("nan")))
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
                    final_loss = float(_losses["total"][-1])

                cd = e if cfg.clamp_key else 0

                # --- write-time self-test: probes -> heads (deterministic
                #     given k_probe, so refitting on reload is exact) ---
                q0p, true_p, is_imp = _probe_cues(
                    k_probe, embeds, keys_tok, vals_tok, cfg, e
                )
                probe_rec = _ladder_records(
                    model,
                    q0p,
                    jnp.zeros_like(q0p),
                    true_p,
                    val_embeds,
                    val_tokens,
                    e,
                    floor,
                    cfg,
                    cd,
                )
                hp_pred, _hp_msp, hp_lm = _hopfield_confidences(
                    q0p, stored, cfg.hopfield_beta, e, val_embeds, val_tokens
                )
                hop_probe_wrong = hp_pred != np.asarray(true_p)
                heads = _fit_episode_heads(probe_rec, hop_probe_wrong, hp_lm, cfg)
                cell["n_degenerate"] += sum(1 for h in heads.values() if h.degenerate)

                # ship the gate with the checkpoint (training-time artifact)
                if not (cfg.use_pretrained and os.path.exists(ckpt)):
                    save_checkpoint(
                        model,
                        ckpt,
                        epoch=cfg.train_epochs,
                        loss=final_loss,
                        config=None,
                        target_energy=floor,
                        calib_heads={k: h.to_dict() for k, h in heads.items()},
                    )

                deployed = heads[cfg.calib_features]
                probe_pw0 = deployed.p_wrong(
                    R=probe_rec["R"][:, 0], margin=probe_rec["margin"][:, 0]
                )
                cell["probe_pw0"].append(probe_pw0)
                cell["probe_wrong0"].append(~probe_rec["correct"][:, 0])
                cell["probe_is_imp"].append(is_imp)

                # --- storage fidelity (relax from the stored patterns) ---
                qs_f, _, _ = _settle_batch(
                    model,
                    stored,
                    jnp.zeros_like(stored),
                    cfg.relax_steps,
                    cfg.dt,
                    jnp.asarray(floor),
                    cfg.governor_sensitivity,
                    cd,
                )
                pred_f, _ = _decode_values(qs_f[:, e:], val_embeds, val_tokens)
                cell["fidelity"].append(np.asarray(pred_f == vals_tok))

                # --- deployment queries: escalation ladder ---
                true_tok = vals_tok[qk_idx]
                q0 = jnp.concatenate(
                    [embeds[keys_tok[qk_idx]], jnp.zeros((qk_idx.shape[0], e))],
                    axis=1,
                )
                rec = _ladder_records(
                    model,
                    q0,
                    jnp.zeros_like(q0),
                    true_tok,
                    val_embeds,
                    val_tokens,
                    e,
                    floor,
                    cfg,
                    cd,
                )
                cell["R"].append(rec["R"])
                cell["margin"].append(rec["margin"])
                cell["correct"].append(rec["correct"])
                cell["cost"] = rec["cost"]
                for hname, key_ in (
                    ("r", "pw_r"),
                    ("margin", "pw_margin"),
                    ("r_margin", "pw_rm"),
                ):
                    cell[key_].append(
                        heads[hname].p_wrong(R=rec["R"], margin=rec["margin"])
                    )
                cell["per_episode_stage0"].append(
                    (rec["R"][:, 0], rec["correct"][:, 0])
                )

                # --- modern-Hopfield baseline on the same queries ---
                h_pred, h_msp, h_lm = _hopfield_confidences(
                    q0, stored, cfg.hopfield_beta, e, val_embeds, val_tokens
                )
                cell["hop_correct"].append(h_pred == np.asarray(true_tok))
                cell["hop_msp"].append(h_msp)
                cell["hop_lm"].append(h_lm)
                cell["pw_hop"].append(heads["hopfield"].p_wrong(R=h_lm))

            for k in (
                "R",
                "margin",
                "correct",
                "pw_r",
                "pw_margin",
                "pw_rm",
                "hop_correct",
                "hop_msp",
                "hop_lm",
                "pw_hop",
                "probe_pw0",
                "probe_wrong0",
                "probe_is_imp",
                "fidelity",
            ):
                cell[k] = np.concatenate(cell[k], axis=0)
            cell["pw_deployed"] = {
                "r": cell["pw_r"],
                "margin": cell["pw_margin"],
                "r_margin": cell["pw_rm"],
            }[cfg.calib_features]
            cell["metrics"] = _cell_metrics(cell, cfg)
            cells.append(cell)
            mm = cell["metrics"]
            print(
                f"  [seed {seed_i}] N={N} kv={kv}: trials={mm['trials']} "
                f"base={mm['base_acc']:.3f} full={mm['full_acc']:.3f} "
                f"hop={mm['hop_acc']:.3f} fid={mm['fidelity']:.3f} "
                f"pooledAUROC raw={mm['pooled_raw_R_auroc']:.3f} "
                f"calib={mm['pooled_calibrated_auroc']:.3f} "
                f"gate acc={mm['compute']['learned_point']['acc']:.3f}@"
                f"{mm['compute']['learned_point']['cost']:.0f} steps"
            )

    summary = _summarize(cells, levels, seeds, cfg, base_seed)
    _save_outputs(cells, summary, cfg, levels, seeds, save_dir, results_dir)
    _print_read(summary, cfg)
    return {"cells": cells, "summary": summary}


# ---------------------------------------------------------------------------
# Summary / outputs
# ---------------------------------------------------------------------------


def _summarize(cells, levels, seeds, cfg, base_seed):
    """Aggregate per-cell metrics into per-level and pooled mean/std blocks."""
    summary = {
        "base_seed": base_seed,
        "seeds": list(seeds),
        "levels": [list(lv) for lv in levels],
        "gate_features": cfg.calib_features,
        "p_exit": cfg.calib_p_exit,
        "note": (
            "squeeze retries parked (Head 2026-07-07); escalation = staged "
            "governed relaxation on the v1-l0-gate cost ladder"
        ),
        "per_level": {},
        "pooled": {},
    }

    def cells_of(level=None, seed=None):
        return [
            c
            for c in cells
            if (level is None or (c["N"], c["kv"]) == level)
            and (seed is None or c["seed"] == seed)
        ]

    method_names = list(_stage0_confidences(cells[0]).keys())
    risk_targets = list(cfg.calib_risk_targets)

    for lv in levels:
        key = f"N{lv[0]}_kv{lv[1]}"
        per_seed = [cells_of(lv, s)[0]["metrics"] for s in seeds]
        blk = {
            "trials": [m["trials"] for m in per_seed],
            "n_wrong0": [m["n_wrong0"] for m in per_seed],
            "base_acc": _agg([m["base_acc"] for m in per_seed]),
            "full_acc": _agg([m["full_acc"] for m in per_seed]),
            "hop_acc": _agg([m["hop_acc"] for m in per_seed]),
            "fidelity": _agg([m["fidelity"] for m in per_seed]),
            "probe_wrong_frac": _agg([m["probe_wrong_frac"] for m in per_seed]),
            "within_model_auroc_R": _agg([m["within_model_auroc_R"] for m in per_seed]),
            "pooled_raw_R_auroc": _agg([m["pooled_raw_R_auroc"] for m in per_seed]),
            "pooled_calibrated_auroc": _agg(
                [m["pooled_calibrated_auroc"] for m in per_seed]
            ),
            "energy_over_margin_auroc_delta": _agg(
                [m["energy_over_margin_auroc_delta"] for m in per_seed]
            ),
            "auroc": {n: _agg([m["auroc"][n] for m in per_seed]) for n in method_names},
            "aurc": {
                n: _agg([m["aurc"][n] for m in per_seed])
                for n in method_names + ["clu_calib_deployed_full"]
            },
            "cov_at_risk": {},
            "ece": {
                n: _agg([m["ece"][n] for m in per_seed]) for n in PROBABILISTIC_METHODS
            },
            "compute": {
                "learned_point_acc": _agg(
                    [m["compute"]["learned_point"]["acc"] for m in per_seed]
                ),
                "learned_point_cost": _agg(
                    [m["compute"]["learned_point"]["cost"] for m in per_seed]
                ),
                "always_small_acc": _agg(
                    [m["compute"]["always_small"]["acc"] for m in per_seed]
                ),
                "always_full_acc": _agg(
                    [m["compute"]["always_full"]["acc"] for m in per_seed]
                ),
                "full_cost": per_seed[0]["compute"]["always_full"]["cost"],
                "savings_ratio": _agg(
                    [
                        m["compute"]["always_full"]["cost"]
                        / m["compute"]["learned_point"]["cost"]
                        for m in per_seed
                    ]
                ),
            },
            "ltt": {},
        }
        for n in method_names + ["clu_calib_deployed_full"]:
            for eps in risk_targets:
                k = f"{n}@{eps}"
                blk["cov_at_risk"][k] = _agg([m["cov_at_risk"][k] for m in per_seed])
        for eps in risk_targets:
            k = f"eps={eps}"
            entries = [m["ltt"][k] for m in per_seed]
            blk["ltt"][k] = {
                "certified_frac": float(np.mean([e_["certified"] for e_ in entries])),
                "valid_frac": float(np.mean([e_["valid"] for e_ in entries])),
                "eval_coverage": _agg([e_["eval_coverage"] for e_ in entries]),
                "eval_risk": _agg([e_["eval_risk"] for e_ in entries]),
            }
        summary["per_level"][key] = blk

    # pooled across levels, per seed (the cross-difficulty deployment view)
    pooled_raw, pooled_cal, pooled_aurc = [], [], {n: [] for n in method_names}
    for s in seeds:
        cs = cells_of(seed=s)
        wrong0 = np.concatenate([~c["correct"][:, 0] for c in cs])
        R0 = np.concatenate([c["R"][:, 0] for c in cs])
        pw0 = np.concatenate([c["pw_deployed"][:, 0] for c in cs])
        pooled_raw.append(_auroc(wrong0.astype(int), R0))
        pooled_cal.append(_auroc(wrong0.astype(int), pw0))
        for n in method_names:
            conf = np.concatenate([_stage0_confidences(c)[n][0] for c in cs])
            cor = np.concatenate([_stage0_confidences(c)[n][1] for c in cs])
            cov, risk = risk_coverage_curve(conf, cor)
            pooled_aurc[n].append(aurc(cov, risk))
    summary["pooled"] = {
        "auroc_raw_R_to_wrong": _agg(pooled_raw),
        "auroc_calibrated_to_wrong": _agg(pooled_cal),
        "aurc": {n: _agg(v) for n, v in pooled_aurc.items()},
    }
    return summary


def _save_outputs(cells, summary, cfg, levels, seeds, save_dir, results_dir):
    """Write metrics npz + summary json + the three figures."""
    npz = {}
    for c in cells:
        pre = f"S{c['seed']}_N{c['N']}_kv{c['kv']}_"
        for k in (
            "R",
            "margin",
            "correct",
            "pw_r",
            "pw_margin",
            "pw_rm",
            "hop_correct",
            "hop_msp",
            "hop_lm",
            "pw_hop",
            "probe_pw0",
            "probe_wrong0",
            "probe_is_imp",
            "fidelity",
            "cost",
        ):
            npz[pre + k] = np.asarray(c[k])
    metrics_path = os.path.join(results_dir, "exp_v1_calibration_metrics.npz")
    np.savez(metrics_path, **npz)
    summary_path = os.path.join(results_dir, "exp_v1_calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}\nSaved summary to {summary_path}")

    grid = np.linspace(0.02, 1.0, 50)
    plot_methods = (
        "clu_calib_rm",
        "clu_calib_r",
        "clu_margin_raw",
        "clu_R_raw",
        "hop_msp",
        "hop_calib",
    )

    def rc_panel(cell_lists, title):
        methods = {}
        for name in plot_methods:
            curves = []
            for cs in cell_lists:  # one entry per seed
                conf = np.concatenate([_stage0_confidences(c)[name][0] for c in cs])
                cor = np.concatenate([_stage0_confidences(c)[name][1] for c in cs])
                cov, risk = risk_coverage_curve(conf, cor)
                curves.append(interpolate_risk_coverage(cov, risk, grid))
            curves = np.stack(curves)
            methods[name] = (grid, curves.mean(0), curves.std(0))
        return {"title": title, "methods": methods}

    panels = []
    for lv in levels:
        per_seed = [
            [c for c in cells if (c["N"], c["kv"]) == lv and c["seed"] == s]
            for s in seeds
        ]
        panels.append(rc_panel(per_seed, f"N={lv[0]}, kv={lv[1]}"))
    panels.append(
        rc_panel(
            [[c for c in cells if c["seed"] == s] for s in seeds],
            "pooled levels",
        )
    )
    plot_v1_calib_risk_coverage(
        panels,
        os.path.join(save_dir, "exp_v1_calibration_risk_coverage.png"),
        risk_line=min(cfg.calib_risk_targets) if cfg.calib_risk_targets else 0.05,
    )

    # compute panels
    comp_panels = []
    for lv in levels:
        lv_key = f"N{lv[0]}_kv{lv[1]}"
        blk = summary["per_level"][lv_key]
        cs_by_seed = [
            [c for c in cells if (c["N"], c["kv"]) == lv and c["seed"] == s][0]
            for s in seeds
        ]
        cost_lo = float(cs_by_seed[0]["cost"][0])
        cost_hi = float(cs_by_seed[0]["cost"][-1])
        cost_grid = np.linspace(cost_lo, cost_hi, 24)
        curves = {}
        for sw_name in ("calib_deployed", "margin_raw", "R_raw"):
            accs = []
            for c in cs_by_seed:
                sw = c["metrics"]["compute"]["sweeps"][sw_name]
                cst = np.asarray(sw["cost"])
                acc = np.asarray(sw["acc"])
                order = np.argsort(cst)
                accs.append(np.interp(cost_grid, cst[order], acc[order]))
            accs = np.stack(accs)
            curves[sw_name] = (cost_grid, accs.mean(0), accs.std(0))
        points = {
            "learned gate (p_exit)": (
                blk["compute"]["learned_point_cost"]["mean"],
                blk["compute"]["learned_point_cost"]["std"],
                blk["compute"]["learned_point_acc"]["mean"],
                blk["compute"]["learned_point_acc"]["std"],
            ),
            "always-small": (
                cost_lo,
                0.0,
                blk["compute"]["always_small_acc"]["mean"],
                blk["compute"]["always_small_acc"]["std"],
            ),
            "always-full": (
                cost_hi,
                0.0,
                blk["compute"]["always_full_acc"]["mean"],
                blk["compute"]["always_full_acc"]["std"],
            ),
        }
        comp_panels.append(
            {
                "title": f"N={lv[0]}, kv={lv[1]}",
                "curves": curves,
                "points": points,
                "hopfield_acc": blk["hop_acc"]["mean"],
            }
        )
    plot_v1_calib_compute(
        comp_panels, os.path.join(save_dir, "exp_v1_calibration_compute.png")
    )

    # reliability panel (pooled over all cells; ECE mean±std across seeds)
    rel_methods = ("clu_calib_rm", "hop_msp", "hop_calib")
    rel = {}
    for name in rel_methods:
        conf = np.concatenate([_stage0_confidences(c)[name][0] for c in cells])
        cor = np.concatenate([_stage0_confidences(c)[name][1] for c in cells])
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, 11)))
        idx = np.clip(np.digitize(conf, edges) - 1, 0, len(edges) - 2)
        cb, ab = [], []
        for b in range(len(edges) - 1):
            msk = idx == b
            if msk.any():
                cb.append(float(conf[msk].mean()))
                ab.append(float(cor[msk].mean()))
        eces = [
            expected_calibration_error(
                np.concatenate(
                    [_stage0_confidences(c)[name][0] for c in cells if c["seed"] == s]
                ),
                np.concatenate(
                    [_stage0_confidences(c)[name][1] for c in cells if c["seed"] == s]
                ),
            )
            for s in seeds
        ]
        rel[name] = {
            "bins": (np.asarray(cb), np.asarray(ab)),
            "ece": (float(np.mean(eces)), float(np.std(eces))),
        }
    plot_v1_calib_reliability(
        rel, os.path.join(save_dir, "exp_v1_calibration_reliability.png")
    )


def _print_read(summary, cfg):
    """Console digest of the pivot's headline numbers."""
    print("\n" + "=" * 60)
    print("V1-CALIBRATION READ (heuristics; final read is the Hub's)")
    print("=" * 60)
    p = summary["pooled"]
    print(
        f"  pooled AUROC(->wrong): raw R "
        f"{p['auroc_raw_R_to_wrong']['mean']:.3f}"
        f"±{p['auroc_raw_R_to_wrong']['std']:.3f}  vs  calibrated "
        f"{p['auroc_calibrated_to_wrong']['mean']:.3f}"
        f"±{p['auroc_calibrated_to_wrong']['std']:.3f}"
    )
    for lv_key, blk in summary["per_level"].items():
        cp = blk["compute"]
        print(
            f"  {lv_key}: base={blk['base_acc']['mean']:.3f} "
            f"full={blk['full_acc']['mean']:.3f} "
            f"gate={cp['learned_point_acc']['mean']:.3f}"
            f"±{cp['learned_point_acc']['std']:.3f} @ "
            f"{cp['learned_point_cost']['mean']:.0f} steps "
            f"({cp['savings_ratio']['mean']:.1f}x saved) | "
            f"AURC clu_rm={blk['aurc']['clu_calib_rm']['mean']:.3f} "
            f"hop_calib={blk['aurc']['hop_calib']['mean']:.3f} | "
            f"E-over-M dAUROC="
            f"{blk['energy_over_margin_auroc_delta']['mean']:+.3f}"
        )
        for eps_key, ltt in blk["ltt"].items():
            print(
                f"      LTT {eps_key}: certified={ltt['certified_frac']:.2f} "
                f"valid={ltt['valid_frac']:.2f} "
                f"cov={ltt['eval_coverage']['mean']:.2f} "
                f"risk={ltt['eval_risk']['mean']:.3f}"
            )
    print("=" * 60 + "\n")


# ===========================================================================
# v1-hopfield-stress: CLU-vs-Hopfield regime map (Head decision 1b)
# ===========================================================================
#
# Goal (NOT to hide Hopfield's dominance): chart WHERE the CLU-gate-vs-Hopfield
# trade lives so a reader knows when to build with which. The v1-pivot run
# proved Hopfield is near-perfect on vanilla MQAR at kv<=32 (abstention
# head-to-head unwinnable). Here we stress both systems and classify each grid
# cell. Two fair stress mechanics (CLU and Hopfield see identical stressed
# cues/embeddings):
#   - correlation: key/value embeddings pulled toward shared centroids (reduced
#     separation = the classic Hopfield failure mode; retrains the memory).
#   - eval_noise: Gaussian sigma on the deployment cue only (degrades
#     retrieval, not storage; the memory is written from clean patterns).
# "Hopfield wins everywhere we could reach" is a valid, reportable outcome
# (trend lines are then the result).

#: regime-cell classification codes (used by the map figure)
REGIME_CATEGORIES = ("hopfield_dominant", "comparable", "clu_gate_advantage")


def _clustered_embeddings(key, vocab_size, embed_dim, scale, rho, n_clusters):
    """Token embeddings with tunable correlation (reduced separation stress).

    rho=0 reproduces make_token_embeddings (iid Gaussian). rho>0 pulls each
    token toward one of n_clusters shared centroids while preserving the
    marginal norm::

        emb = sqrt(1 - rho^2) * iid + rho * centroid[assign]

    so E[<e_i, e_j>] ~= rho^2 * ||centroid||^2 for two same-cluster tokens
    (cosine ~ rho^2) and ~0 across clusters. Keys (lower half-vocab) and values
    (upper half) cluster independently, so both the Hopfield key-match and the
    CLU/Hopfield value decode face the classic reduced-separation stress.
    """
    k_c, k_i, k_a = jax.random.split(key, 3)
    sd = scale / jnp.sqrt(embed_dim)
    iid = jax.random.normal(k_i, (vocab_size, embed_dim)) * sd
    if rho <= 0.0:
        return iid
    centroids = jax.random.normal(k_c, (n_clusters, embed_dim)) * sd
    assign = jax.random.randint(k_a, (vocab_size,), 0, n_clusters)
    mix = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    return mix * iid + rho * centroids[assign]


def _regime_cell(config, cfg, e, dim, seed_i, N, kv, correlation, eval_noise,
                 n_clusters, models_dir):
    """One (capacity, stress, seed) cell of the regime map.

    Trains the per-episode CLU memory (generative PCD) on possibly-correlated
    embeddings, ends the write with the standard self-test -> per-model heads,
    then runs the deployment ladder under the cell's stress (noisy cue if
    eval_noise>0) and the modern-Hopfield baseline on the *same* stressed cue
    with its Platt-calibrated logit-margin head (fair). Reuses the episode
    helpers (_probe_cues, _ladder_records, _hopfield_confidences,
    _fit_episode_heads) verbatim.

    Returns per-cell arrays pooled over episodes: correct (T,S), pw_deployed
    (T,S), R/margin (T,S), hop_correct (T,), hop_pw (T,), cost (S,), plus
    fidelity and degenerate-head counts.
    """
    master = jax.random.PRNGKey(seed_i)
    embed_key, run_key = jax.random.split(master)
    embeds = _clustered_embeddings(
        embed_key, cfg.vocab_size, e, cfg.embed_scale, correlation, n_clusters
    )
    val_tokens = jnp.arange(cfg.vocab_size // 2, cfg.vocab_size)
    val_embeds = embeds[val_tokens]

    acc = {k: [] for k in ("correct", "pw_deployed", "R", "margin",
                           "hop_correct", "hop_pw")}
    cost = None
    fidelity, n_degenerate = [], 0

    for ep in range(cfg.regime_episodes_per_cell):
        ep_key = jax.random.fold_in(run_key, ep)
        k_data, k_model, k_train, k_probe, k_cue = jax.random.split(ep_key, 5)

        mq = generate_mqar(
            k_data, 1, N, kv, vocab_size=cfg.vocab_size,
            gap_distribution=cfg.gap_distribution, powerlaw_alpha=cfg.powerlaw_alpha,
        )
        keys_tok = mq["keys"][0]
        vals_tok = mq["values"][0]
        qk_idx = mq["query_key_idx"][0]
        stored = jnp.concatenate([embeds[keys_tok], embeds[vals_tok]], axis=1)

        model = CHLU(
            dim=dim, hidden=cfg.hidden_dim, rest_mass=config.model.rest_mass,
            c=config.model.speed_of_causality, kinetic_mode=cfg.kinetic_energy_mode,
            potential_type=cfg.potential_type, key=k_model,
        )
        model, _losses, floor = train_generative(
            model, stored, key=k_train, config=config, epochs=cfg.train_epochs,
            lr=cfg.train_lr, batch_size=cfg.train_batch_size, dt=cfg.dt,
            buffer_capacity=cfg.train_buffer_capacity, k_steps=cfg.train_k_steps,
            sleep_friction=cfg.train_friction, sleep_temperature=cfg.train_temperature,
            input_noise_sigma=cfg.train_input_noise_sigma,
        )
        cd = e if cfg.clamp_key else 0

        # write-time self-test -> heads (probes = jittered clean cues)
        q0p, true_p, _is_imp = _probe_cues(k_probe, embeds, keys_tok, vals_tok, cfg, e)
        probe_rec = _ladder_records(
            model, q0p, jnp.zeros_like(q0p), true_p, val_embeds, val_tokens,
            e, floor, cfg, cd,
        )
        hp_pred, _hp_msp, hp_lm = _hopfield_confidences(
            q0p, stored, cfg.hopfield_beta, e, val_embeds, val_tokens
        )
        heads = _fit_episode_heads(
            probe_rec, hp_pred != np.asarray(true_p), hp_lm, cfg
        )
        n_degenerate += sum(1 for h in heads.values() if h.degenerate)
        deployed = heads[cfg.calib_features]

        # storage fidelity (relax from the stored pattern itself)
        qs_f, _, _ = _settle_batch(
            model, stored, jnp.zeros_like(stored), cfg.relax_steps, cfg.dt,
            jnp.asarray(floor), cfg.governor_sensitivity, cd,
        )
        pred_f, _ = _decode_values(qs_f[:, e:], val_embeds, val_tokens)
        fidelity.append(np.asarray(pred_f == vals_tok))

        # --- deployment under stress: (optionally noisy) cue on the key half ---
        true_tok = vals_tok[qk_idx]
        cue = embeds[keys_tok[qk_idx]]
        if eval_noise > 0.0:
            cue = cue + jax.random.normal(k_cue, cue.shape) * eval_noise
        q0 = jnp.concatenate([cue, jnp.zeros((qk_idx.shape[0], e))], axis=1)
        rec = _ladder_records(
            model, q0, jnp.zeros_like(q0), true_tok, val_embeds, val_tokens,
            e, floor, cfg, cd,
        )
        acc["correct"].append(rec["correct"])
        acc["R"].append(rec["R"])
        acc["margin"].append(rec["margin"])
        acc["pw_deployed"].append(deployed.p_wrong(R=rec["R"], margin=rec["margin"]))
        cost = rec["cost"]

        # modern-Hopfield on the SAME stressed cue + its Platt-calibrated margin
        h_pred, _h_msp, h_lm = _hopfield_confidences(
            q0, stored, cfg.hopfield_beta, e, val_embeds, val_tokens
        )
        acc["hop_correct"].append(h_pred == np.asarray(true_tok))
        acc["hop_pw"].append(heads["hopfield"].p_wrong(R=h_lm))

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["cost"] = np.asarray(cost)
    out["fidelity"] = float(np.concatenate(fidelity).mean())
    out["n_degenerate"] = int(n_degenerate)
    return out


def _regime_metrics(out, cfg):
    """Scalar CLU-gate-vs-Hopfield comparison for one cell (per seed)."""
    correct = out["correct"]
    cost = out["cost"].astype(float)

    # CLU learned operating point (tau fixed at write time, p_exit)
    acc_pt, cost_pt = _simulate_tau_policy(
        out["pw_deployed"], correct, cost, [cfg.calib_p_exit], mode="le"
    )
    clu_gate_acc = float(acc_pt[0])
    clu_gate_cost = float(cost_pt[0])
    savings = float(cost[-1] / clu_gate_cost) if clu_gate_cost > 0 else float("nan")

    # CLU abstention frontier at full budget (deployed head, most compute)
    clu_conf = 1.0 - out["pw_deployed"][:, -1]
    cov_c, risk_c = risk_coverage_curve(clu_conf, correct[:, -1])
    clu_aurc = aurc(cov_c, risk_c)

    # Hopfield: single matvec + Platt-calibrated logit margin
    hop_conf = 1.0 - out["hop_pw"]
    cov_h, risk_h = risk_coverage_curve(hop_conf, out["hop_correct"])
    hop_aurc = aurc(cov_h, risk_h)

    eps = min(cfg.calib_risk_targets) if cfg.calib_risk_targets else 0.05
    return {
        "clu_gate_acc": clu_gate_acc,
        "clu_gate_cost": clu_gate_cost,
        "clu_base_acc": float(correct[:, 0].mean()),
        "clu_full_acc": float(correct[:, -1].mean()),
        "savings": savings,
        "clu_aurc": clu_aurc,
        "hop_acc": float(out["hop_correct"].mean()),
        "hop_aurc": hop_aurc,
        "clu_cov_at_risk": coverage_at_risk(cov_c, risk_c, eps),
        "hop_cov_at_risk": coverage_at_risk(cov_h, risk_h, eps),
        "fidelity": out["fidelity"],
        "n_wrong": int((~correct[:, -1]).sum()),
        "hop_n_wrong": int((~out["hop_correct"]).sum()),
    }


def _classify_regime(d_acc, d_aurc, margin):
    """Classify a cell from mean CLU-minus-Hopfield deltas.

    d_acc  = clu_gate_acc - hop_acc          (>0 => CLU better accuracy)
    d_aurc = hop_aurc - clu_aurc             (>0 => CLU better abstention)
    Returns (category, advantage_dim) where category in REGIME_CATEGORIES.
    """
    clu_acc, clu_abs = d_acc > margin, d_aurc > margin
    hop_acc, hop_abs = d_acc < -margin, d_aurc < -margin
    if hop_acc and hop_abs:
        return "hopfield_dominant", None
    if (clu_acc or clu_abs) and not (hop_acc or hop_abs):
        dim = "both" if (clu_acc and clu_abs) else ("accuracy" if clu_acc else "abstention")
        return "clu_gate_advantage", dim
    return "comparable", None


def _agg_regime(per_seed):
    """Mean/std over seeds of each scalar in a list of _regime_metrics dicts."""
    keys = per_seed[0].keys()
    return {
        k: {
            "mean": float(np.nanmean([m[k] for m in per_seed])),
            "std": float(np.nanstd([m[k] for m in per_seed])),
        }
        for k in keys
    }


def run_v1_hopfield_regime_map(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: Optional[bool] = None,
):
    """Chart the CLU-gate-vs-Hopfield regime map over capacity x a stress axis.

    Sweeps regime_capacity_levels (kv) x regime_stress_grid under
    regime_stress_axis ("correlation" | "eval_noise"), regime_n_seeds seeds per
    cell. Each cell reuses the exp_v1_calibration episode machinery. Emits a
    regime-map figure (per-cell classification + deltas + compute savings), a
    summary json, and a metrics npz.

    Args mirror run_experiment_v1_calibration.

    Returns:
        results dict {"cells": ..., "summary": ...}.
    """
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed

    cfg = config.experiment_v1_gate
    if quick:
        cfg.regime_capacity_levels = [[64, 8], [128, 32]]
        cfg.regime_stress_grid = [0.0, 0.6]
        cfg.regime_n_seeds = 2
        cfg.regime_episodes_per_cell = 1
        cfg.train_epochs = min(cfg.train_epochs, 120)
        cfg.relax_steps = min(cfg.relax_steps, 120)
        cfg.calib_n_stages = 2
        cfg.calib_stage_steps = 240
        cfg.calib_probes_per_key = 4
        cfg.calib_cue_noise_scales = [0.1, 0.3]
        cfg.calib_n_impostors = 8

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "..", "models")
    results_dir = os.path.join(save_dir, "..", "results")
    for d in (save_dir, models_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    axis = cfg.regime_stress_axis
    if axis not in ("correlation", "eval_noise"):
        raise ValueError(f"regime_stress_axis must be correlation|eval_noise, got {axis}")
    caps = [tuple(lv) for lv in cfg.regime_capacity_levels]
    grid = [float(s) for s in cfg.regime_stress_grid]
    base_seed = config.project.seed
    seeds = [base_seed + i for i in range(cfg.regime_n_seeds)]
    e = cfg.embed_dim
    dim = 2 * e
    margin = cfg.regime_comparable_margin

    print("\n" + "=" * 64)
    print("EXPERIMENT V1-HOPFIELD-STRESS: CLU-gate vs Hopfield regime map")
    print("=" * 64)
    print(
        f"capacity (N,kv): {caps} | stress axis: {axis} = {grid} | seeds: {seeds}"
    )
    print(
        f"gate features={cfg.calib_features} p_exit={cfg.calib_p_exit} | "
        f"ladder {cfg.relax_steps}+{cfg.calib_n_stages}x{cfg.calib_stage_steps} | "
        f"hopfield_beta={cfg.hopfield_beta}"
    )

    n_cap, n_str = len(caps), len(grid)
    fields = ("clu_gate_acc", "hop_acc", "clu_full_acc", "clu_base_acc",
              "savings", "clu_aurc", "hop_aurc", "clu_cov_at_risk",
              "hop_cov_at_risk", "fidelity")
    arr = {f: np.full((n_cap, n_str), np.nan) for f in fields}
    arr_std = {f: np.full((n_cap, n_str), np.nan) for f in fields}
    d_acc = np.full((n_cap, n_str), np.nan)
    d_aurc = np.full((n_cap, n_str), np.nan)
    cat = np.zeros((n_cap, n_str), dtype=int)
    cells = {}

    for ci, (N, kv) in enumerate(caps):
        for sj, s in enumerate(grid):
            corr = s if axis == "correlation" else cfg.regime_base_correlation
            noise = s if axis == "eval_noise" else cfg.regime_base_eval_noise
            per_seed = []
            for seed_i in seeds:
                out = _regime_cell(
                    config, cfg, e, dim, seed_i, N, kv, corr, noise,
                    cfg.regime_n_clusters, models_dir,
                )
                per_seed.append(_regime_metrics(out, cfg))
            agg = _agg_regime(per_seed)
            for f in fields:
                arr[f][ci, sj] = agg[f]["mean"]
                arr_std[f][ci, sj] = agg[f]["std"]
            da = agg["clu_gate_acc"]["mean"] - agg["hop_acc"]["mean"]
            dr = agg["hop_aurc"]["mean"] - agg["clu_aurc"]["mean"]
            d_acc[ci, sj] = da
            d_aurc[ci, sj] = dr
            category, adv = _classify_regime(da, dr, margin)
            cat[ci, sj] = REGIME_CATEGORIES.index(category)
            cells[f"cap{N}_kv{kv}__{axis}{s}"] = {
                "N": N, "kv": kv, "stress": s, "category": category,
                "advantage_dim": adv, "d_acc": da, "d_aurc": dr,
                "per_seed": per_seed, "agg": agg,
            }
            print(
                f"  N={N} kv={kv} {axis}={s}: "
                f"clu_gate={agg['clu_gate_acc']['mean']:.3f} "
                f"hop={agg['hop_acc']['mean']:.3f} (dacc={da:+.3f}) | "
                f"AURC clu={agg['clu_aurc']['mean']:.3f} "
                f"hop={agg['hop_aurc']['mean']:.3f} (daurc={dr:+.3f}) | "
                f"save={agg['savings']['mean']:.1f}x | fid={agg['fidelity']['mean']:.3f}"
                f" => {category}"
            )

    cap_labels = [f"N{N}/kv{kv}" for (N, kv) in caps]
    regime = {
        "axis": axis,
        "cap_labels": cap_labels,
        "stress_grid": grid,
        "categories": list(REGIME_CATEGORIES),
        "cat": cat,
        "d_acc": d_acc,
        "d_aurc": d_aurc,
        "clu_acc": arr["clu_gate_acc"],
        "hop_acc": arr["hop_acc"],
        "clu_aurc": arr["clu_aurc"],
        "hop_aurc": arr["hop_aurc"],
        "savings": arr["savings"],
        "fidelity": arr["fidelity"],
    }

    plot_v1_regime_map(
        regime, os.path.join(save_dir, "exp_v1_regime_map.png")
    )

    # summary json (JSON-safe: drop the raw per-seed metric dicts into lists)
    summary = {
        "axis": axis,
        "capacities": [list(c) for c in caps],
        "stress_grid": grid,
        "seeds": seeds,
        "comparable_margin": margin,
        "categories": {
            k: v.tolist() for k, v in
            {"cat_codes": cat, "d_acc": d_acc, "d_aurc": d_aurc}.items()
        },
        "fields_mean": {f: arr[f].tolist() for f in fields},
        "fields_std": {f: arr_std[f].tolist() for f in fields},
        "cells": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
            for k, v in cells.items()
        },
    }
    summary_path = os.path.join(results_dir, "exp_v1_regime_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    npz_path = os.path.join(results_dir, "exp_v1_regime_metrics.npz")
    np.savez(npz_path, cat=cat, d_acc=d_acc, d_aurc=d_aurc, **arr)
    print(f"\nSaved regime map to {summary_path}\nSaved metrics to {npz_path}")

    _print_regime_read(regime)
    return {"cells": cells, "summary": summary, "regime": regime}


def _print_regime_read(regime):
    """Console digest of the regime map."""
    print("\n" + "=" * 64)
    print("V1-HOPFIELD-STRESS READ (heuristics; final read is the Hub's)")
    print("=" * 64)
    caps, grid = regime["cap_labels"], regime["stress_grid"]
    cats = regime["categories"]
    n_adv = int((regime["cat"] == cats.index("clu_gate_advantage")).sum())
    n_comp = int((regime["cat"] == cats.index("comparable")).sum())
    n_dom = int((regime["cat"] == cats.index("hopfield_dominant")).sum())
    print(
        f"  cells: {regime['cat'].size} | Hopfield-dominant {n_dom}, "
        f"comparable {n_comp}, CLU-gate-advantage {n_adv}"
    )
    for ci, cl in enumerate(caps):
        row = " ".join(
            f"{regime['axis'][:4]}{grid[sj]}:{cats[regime['cat'][ci, sj]][:4]}"
            f"(da{regime['d_acc'][ci, sj]:+.2f})"
            for sj in range(len(grid))
        )
        print(f"  {cl:>10}: {row}")
    print("=" * 64 + "\n")
