"""Experiment PHI-READ-IN: the learned read-in ``φ`` around a DESIGNED store (w23).

⭐ **The phase-doctrine flagship** — *learn around a designed core*. Attention is the
precedent: ``softmax(QKᵀ)V`` is fixed, the projections are learned. Here the
relaxation physics + the wells are DESIGNED; the **read-in ``φ`` is learned** (off the
CLU side of the interface, w20's law). Every "CLU on raw data" loss traced to the
missing ``φ``; this builds it and re-fights the one external benchmark we already ran
(w22 Hopfield/U-Hop) in **feature space**, where the trivial baseline becomes
kNN-in-φ — the comparison we can actually contest.

**Store = key–value.** address = ``φ(x)`` written as a designed Gaussian well; payload
= the raw ``x``. Read-out ``ψ`` = settle the query in φ-space → return the payload of the
well it lands in. So a "retrieval" produces a raw image (the payload), and success is
measured by mean ``sqdiff`` in **pixel space** on that payload — identical to w22.

**Four lines (Item 2):**
  1. **CLU-in-φ** — settle ``φ(query)`` in a ``GaussianMemoryPotential`` over ``φ(store)``
     (damped Verlet), then read the payload of the nearest well.
  2. **kNN-in-φ** — argmin_i ‖φ(query)−φ_i‖ → payload. The trivial baseline, now FAIR.
  3. **closed-form Hopfield-in-φ** — the U-Hop/Ramsauer softmax update over stored φ,
     decoded to the nearest stored φ → payload. (A ``φ`` is exactly what fixes
     closed-form Hopfield's CIFAR chance-collapse: DC-dominated inner products.)
  4. **raw-space CLU** — the w22 pixel-space line (continuity control).

**Two ``φ`` arms (Item 1), never trained through the store:**
  - **φ-A ``pca``** — frozen PCA-k (unsupervised, linear, cheap).
  - **φ-B ``ae``** — a small autoencoder trained on a DISJOINT data-distribution pool
    with a reconstruction loss ONLY; never sees the store, wells, or a retrieval loss.

**Item 3 — the laundering control (mandatory):** same ``φ``, trivial store swap. If
kNN-in-φ matches CLU-in-φ everywhere, the win is ``φ``'s, not the store's (the C17-3
lesson). A CLU margin that exists ONLY with the designed store is the result the program
needs. The CLU-in-φ vs kNN-in-φ comparison is reported in exactly those terms.

**Item 4 — does the retry hook survive ``φ``?** We probe whether distance-to-nearest-well
at settle still separates correct vs incorrect first-pass reads in φ-space (AUROC).

Runnable: ``uv run python -m chlu.experiments.exp_phi_read_in --quick`` or via the CLI
``chlu exp-phi-read-in [--project N] [--seed I] [--quick]``.
"""

import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.config import CHLUConfig, get_default_config
from chlu.core.memory_potentials import GaussianMemoryPotential
from chlu.experiments.exp_hopfield_capacity import (
    ACTIVATIONS,
    RAMSAUER_COMMIT,
    UHOP_COMMIT,
    _median_nn_distance,
    _settle_read,
    dropout_query,
    hopfield_retrieve,
    load_patterns,
    noise_query,
    score_retrieval,
)
from chlu.experiments.goldstone_harness import clu_with_potential
from chlu.experiments.phi_encoders import ENCODER_ARMS, build_encoder_read_in


# ---------------------------------------------------------------------------
# Data — draw a store pool and a DISJOINT fit pool (φ never sees the store)
# ---------------------------------------------------------------------------


def load_store_and_fit_pools(dataset, n_store, n_fit, seed):
    """Return ``(store_pool, fit_pool)`` — two DISJOINT [0,1] image sets. The store
    is drawn from ``store_pool``; ``φ`` (PCA or AE) is fit ONLY on ``fit_pool`` so it
    never sees the stored patterns (the fairness guarantee of Item 1)."""
    # over-draw a combined pool with a single seed, then split by index
    combined = load_patterns(dataset, n_store + n_fit, seed)
    combined = np.asarray(combined)
    n_have = combined.shape[0]
    n_store = min(n_store, n_have)
    store = combined[:n_store]
    fit = combined[n_store:] if n_have > n_store else combined[:n_store]
    return jnp.asarray(store, jnp.float32), jnp.asarray(fit, jnp.float32)


# ---------------------------------------------------------------------------
# φ-A — frozen PCA-k (fit on the disjoint pool; zero learning on the CLU side)
# ---------------------------------------------------------------------------


class PCAReadIn:
    """Frozen PCA-k read-in. Fit (mean + top-k components) on the disjoint fit pool.

    ``φ(x) = (x − mean) @ Wᵀ`` with ``W`` the top-k principal directions. Linear,
    unsupervised, and frozen at retrieval — never sees the store or a retrieval loss.
    """

    def __init__(self, fit_pool, k):
        X = np.asarray(fit_pool, np.float64)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        # economy SVD of the centered data; components = right singular vectors
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = Vt[:k].astype(np.float32)  # (k, D)
        self.k = int(k)

    def __call__(self, X):
        X = np.asarray(X, np.float32)
        return jnp.asarray((X - self.mean.astype(np.float32)) @ self.components.T)


# ---------------------------------------------------------------------------
# φ-B — a small autoencoder trained on the disjoint pool (reconstruction only)
# ---------------------------------------------------------------------------


class _AE(eqx.Module):
    enc: eqx.nn.MLP
    dec: eqx.nn.MLP

    def __init__(self, dim, hidden, k, key):
        ke, kd = jax.random.split(key)
        self.enc = eqx.nn.MLP(
            in_size=dim, out_size=k, width_size=hidden, depth=1,
            activation=jax.nn.tanh, key=ke,
        )
        self.dec = eqx.nn.MLP(
            in_size=k, out_size=dim, width_size=hidden, depth=1,
            activation=jax.nn.tanh, key=kd,
        )

    def encode(self, x):
        return self.enc(x)

    def __call__(self, x):
        return self.dec(self.enc(x))


class AEReadIn:
    """A small autoencoder read-in trained on the disjoint fit pool with a
    reconstruction MSE loss ONLY (the w20 rule: learning stays OFF the CLU side of
    the interface — the AE never sees the store, the wells, or a retrieval loss).
    ``φ(x)`` = the encoder output. Frozen at retrieval time."""

    def __init__(self, fit_pool, dim, hidden, k, epochs, lr, batch, seed):
        key = jax.random.PRNGKey(seed + 777)
        key, mk = jax.random.split(key)
        model = _AE(dim, hidden, k, mk)
        opt = optax.adam(lr)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        X = jnp.asarray(fit_pool, jnp.float32)
        n = X.shape[0]
        bs = min(batch, n)

        @eqx.filter_value_and_grad
        def loss_fn(m, xb):
            recon = jax.vmap(m)(xb)
            return jnp.mean((recon - xb) ** 2)

        @eqx.filter_jit
        def step(m, opt_state, xb):
            loss, grads = loss_fn(m, xb)
            updates, opt_state = opt.update(grads, opt_state, m)
            m = eqx.apply_updates(m, updates)
            return m, opt_state, loss

        self.final_loss = None
        for _e in range(epochs):
            key, sk = jax.random.split(key)
            idx = jax.random.choice(sk, n, (bs,), replace=False)
            model, opt_state, loss = step(model, opt_state, X[idx])
            self.final_loss = float(loss)
        self._encode = eqx.filter_jit(jax.vmap(model.encode))
        self.k = int(k)

    def __call__(self, X):
        return self._encode(jnp.asarray(X, jnp.float32))


def build_read_in(arm, dataset, store_pool, fit_pool, cfg, seed):
    dim = int(store_pool.shape[1])
    if arm == "pca":
        phi = PCAReadIn(fit_pool, cfg.phi_dim)
        return phi, {"arm": "pca", "k": cfg.phi_dim, "frozen": True}
    elif arm == "ae":
        phi = AEReadIn(
            fit_pool, dim, cfg.ae_hidden, cfg.phi_dim,
            cfg.ae_epochs, cfg.ae_lr, cfg.ae_batch, seed,
        )
        return phi, {
            "arm": "ae", "k": cfg.phi_dim, "hidden": cfg.ae_hidden,
            "epochs": cfg.ae_epochs, "recon_mse": phi.final_loss, "frozen": True,
        }
    elif arm in ENCODER_ARMS:
        # w26 (cl-encoder): the CL-capable conv arms — additive, see phi_encoders.py
        return build_encoder_read_in(arm, dataset, store_pool, fit_pool, cfg, seed)
    raise ValueError(f"unknown φ arm {arm!r}")


# ---------------------------------------------------------------------------
# The store: key = φ(pattern), payload = raw pattern. Read-out = payload lookup.
# ---------------------------------------------------------------------------


def _payload_from_index(patterns, idx):
    """Read-out ψ: return the raw payload (image) associated with a store index."""
    return np.asarray(patterns)[np.asarray(idx)]


def _nearest_store_index(feat_points, keys):
    """argmin_i ‖feat − key_i‖ over the stored φ-keys."""
    F = np.asarray(feat_points)
    K = np.asarray(keys)
    d2 = np.sum((F[:, None, :] - K[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=1), d2


def build_clu_phi_memory(keys, s, cfg):
    V = GaussianMemoryPotential(keys, s=s, b=cfg.clu_b, alpha=cfg.clu_alpha)
    dim = int(keys.shape[1])
    return clu_with_potential(V, dim=dim, kinetic_mode=cfg.clu_kinetic_mode)


def clu_in_phi(keys, feat_q, s, cfg):
    """Settle φ(query) in the designed Gaussian store (damped Verlet) → φ*; return
    ``(store_index, distance_to_nearest_well_at_settle, dt)``. The read-out then maps
    the index to the raw payload. This is the DESIGNED-store retrieval."""
    model = build_clu_phi_memory(keys, s, cfg)
    dt = cfg.clu_dt if cfg.clu_dt > 0 else 0.5 * s / np.sqrt(cfg.clu_b)
    tail = int(max(1, cfg.clu_tail_frac * cfg.clu_steps))
    feat_star = _settle_read(
        model, jnp.asarray(feat_q), cfg.clu_steps, dt, cfg.clu_gamma, tail,
        cfg.rollout_chunk,
    )
    idx, d2 = _nearest_store_index(feat_star, keys)
    dist = np.sqrt(d2[np.arange(len(idx)), idx])  # distance to landed well
    return idx, dist, float(dt)


def hopfield_in_phi(keys, feat_q, cfg, act_name):
    """Closed-form modern-Hopfield update over the stored φ-keys, decoded to the
    nearest stored key → store index (then payload). A ``φ`` is what lets closed-form
    Hopfield escape the raw-pixel DC-overlap collapse (w22 CIFAR)."""
    act_fn = ACTIVATIONS[act_name]
    feat_hat = hopfield_retrieve(
        jnp.asarray(keys), jnp.asarray(feat_q), 1.0, cfg.hopfield_beta,
        cfg.hopfield_steps, act_fn,
    )
    idx, _ = _nearest_store_index(feat_hat, keys)
    return idx


def knn_in_phi(keys, feat_q):
    """The trivial baseline, now fair: nearest stored φ-key to the query φ."""
    idx, _ = _nearest_store_index(feat_q, keys)
    return idx


# ---------------------------------------------------------------------------
# raw-space CLU (the w22 pixel-space continuity line)
# ---------------------------------------------------------------------------


def raw_clu_retrieve(patterns, Q, cfg):
    s = cfg.clu_s_frac * _median_nn_distance(patterns)
    V = GaussianMemoryPotential(patterns, s=s, b=cfg.clu_b, alpha=cfg.clu_alpha)
    dim = int(patterns.shape[1])
    model = clu_with_potential(V, dim=dim, kinetic_mode=cfg.clu_kinetic_mode)
    dt = cfg.clu_dt if cfg.clu_dt > 0 else 0.5 * s / np.sqrt(cfg.clu_b)
    tail = int(max(1, cfg.clu_tail_frac * cfg.clu_steps))
    xhat = _settle_read(model, Q, cfg.clu_steps, dt, cfg.clu_gamma, tail,
                        cfg.rollout_chunk)
    return np.asarray(xhat)


# ---------------------------------------------------------------------------
# The φ-space sweeps (capacity + noise) with all four lines
# ---------------------------------------------------------------------------


def _score_payload(patterns, xhat_pixels, true_idx, success_cosine):
    """All lines are scored identically: produce a pixel-space image (payload or
    settled state), then use the w22 scorer (identity via argmin over stored
    patterns + mean sqdiff vs the true pattern in pixel space)."""
    m, _, _ = score_retrieval(patterns, xhat_pixels, true_idx, success_cosine)
    return m


def _run_all_lines(patterns, Q, phi, cfg, want_confidence=False):
    """Retrieve with every line for one (store, query) pair. Returns a dict of
    pixel-space payload images keyed by line name (+ optional CLU confidence)."""
    keys = np.asarray(phi(patterns))  # (M, k) store addresses
    feat_q = np.asarray(phi(Q))  # (Nq, k) query features
    s_phi = cfg.clu_s_frac * _median_nn_distance(keys)

    idx_clu, dist_clu, dt = clu_in_phi(keys, feat_q, s_phi, cfg)
    idx_knn = knn_in_phi(keys, feat_q)

    out = {
        "clu_in_phi": _payload_from_index(patterns, idx_clu),
        "knn_in_phi": _payload_from_index(patterns, idx_knn),
    }
    for act in cfg.activations:
        idx_h = hopfield_in_phi(keys, feat_q, cfg, act)
        out[f"hopfield_{act}_in_phi"] = _payload_from_index(patterns, idx_h)

    meta = {"s_phi": float(s_phi), "phi_dim": int(keys.shape[1]), "clu_dt": dt}
    if want_confidence:
        meta["clu_confidence"] = np.asarray(dist_clu)  # dist-to-nearest-well
        meta["clu_index"] = np.asarray(idx_clu)
    return out, meta


def capacity_sweep_phi(cfg, dataset, arm, phi, store_pool, seed):
    key = jax.random.PRNGKey(seed)
    rows = []
    packing = None
    for M in cfg.load_grid:
        if M > store_pool.shape[0]:
            continue
        patterns = store_pool[:M]
        key, kq = jax.random.split(key)
        Q = dropout_query(patterns, cfg.mask_p, kq)
        true_idx = np.arange(M)

        lines, meta = _run_all_lines(patterns, Q, phi, cfg)
        row = {"M": int(M), **meta}
        for name, xhat in lines.items():
            row[name] = _score_payload(patterns, xhat, true_idx, cfg.success_cosine)
        # raw-space CLU continuity line (pixel space, φ-independent)
        xhat_raw = raw_clu_retrieve(patterns, Q, cfg)
        row["raw_space_clu"] = _score_payload(
            patterns, xhat_raw, true_idx, cfg.success_cosine
        )
        # packing-law occupancy for this feature dim (matrix v2.1 §1)
        if packing is None:
            packing = _packing_report(np.asarray(phi(patterns)), meta["s_phi"])
        rows.append(row)
    return {"dataset": dataset, "arm": arm, "rows": rows, "packing": packing}


def noise_sweep_phi(cfg, dataset, arm, phi, store_pool, seed):
    key = jax.random.PRNGKey(seed + 100)
    M = min(cfg.noise_fixed_load, store_pool.shape[0])
    patterns = store_pool[:M]
    true_idx = np.arange(M)
    rows = []
    for sigma in cfg.noise_levels:
        key, kq = jax.random.split(key)
        Q = noise_query(patterns, sigma, kq)
        lines, meta = _run_all_lines(patterns, Q, phi, cfg)
        row = {"sigma": float(sigma), **meta}
        for name, xhat in lines.items():
            row[name] = _score_payload(patterns, xhat, true_idx, cfg.success_cosine)
        xhat_raw = raw_clu_retrieve(patterns, Q, cfg)
        row["raw_space_clu"] = _score_payload(
            patterns, xhat_raw, true_idx, cfg.success_cosine
        )
        rows.append(row)
    return {"dataset": dataset, "arm": arm, "fixed_load": int(M), "rows": rows}


def _packing_report(keys, s_phi):
    """Report d, well width w=s_phi, query scale, and the packing-law occupancy
    Δ_req ≈ 3.1·max(w, σ_query) vs the achieved median NN spacing (matrix v2.1 §1)."""
    d = int(keys.shape[1])
    spacing = _median_nn_distance(keys)
    delta_req = 3.1 * s_phi  # σ_query enters at query time; report the w-bound form
    return {
        "d": d,
        "well_width_s": float(s_phi),
        "median_nn_spacing": float(spacing),
        "delta_req_3p1_w": float(delta_req),
        "spacing_over_delta_req": float(spacing / (delta_req + 1e-12)),
        "note": "occupancy adequate iff median_nn_spacing >= Δ_req (≈3.1·w); "
        "σ_query adds to the max() at query time",
    }


# ---------------------------------------------------------------------------
# Item 3 — the laundering control (CLU-in-φ vs kNN-in-φ, in the task's words)
# ---------------------------------------------------------------------------


def laundering_control(cap_results, tie_band=0.03):
    """Compare CLU-in-φ vs kNN-in-φ on the capacity axis for each arm. If kNN-in-φ
    matches CLU-in-φ everywhere (within the tie band), the win is φ's, not the
    store's (Item 3, the C17-3 lesson)."""
    out = []
    for cap in cap_results:
        rows = cap["rows"]
        if not rows:
            continue
        deltas = []
        for r in rows:
            d = r["clu_in_phi"]["identity_acc"] - r["knn_in_phi"]["identity_acc"]
            deltas.append({"M": r["M"], "clu_minus_knn_acc": float(d)})
        arr = np.array([d["clu_minus_knn_acc"] for d in deltas])
        n_tie = int(np.sum(np.abs(arr) <= tie_band))
        n_clu_win = int(np.sum(arr > tie_band))
        n_knn_win = int(np.sum(arr < -tie_band))
        laundered = n_clu_win == 0  # CLU never beats kNN with the designed store
        out.append({
            "dataset": cap["dataset"],
            "arm": cap["arm"],
            "tie_band": tie_band,
            "per_M": deltas,
            "n_points": len(deltas),
            "n_tie": n_tie,
            "n_clu_wins": n_clu_win,
            "n_knn_wins": n_knn_win,
            "max_clu_margin": float(np.max(arr)) if len(arr) else None,
            "laundered": laundered,
            "verdict": (
                "LAUNDERED — CLU-in-φ never beats kNN-in-φ with the designed "
                "store: the win is φ's, not ours."
                if laundered else
                "CLU-in-φ beats kNN-in-φ with the designed store on "
                f"{n_clu_win}/{len(deltas)} loads (max +{float(np.max(arr)):.3f} "
                "identity-acc) — a margin that exists ONLY with the designed store."
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Item 4 — does the retry hook (confidence = dist-to-nearest-well) survive φ?
# ---------------------------------------------------------------------------


def retry_confidence_probe(cfg, dataset, arm, phi, store_pool, seed):
    """Does distance-to-nearest-well at settle still separate correct vs incorrect
    first-pass reads in φ-space? Report AUROC of (−distance) vs correctness."""
    M = min(cfg.noise_fixed_load, store_pool.shape[0])
    patterns = store_pool[:M]
    true_idx = np.arange(M)
    key = jax.random.PRNGKey(seed + 300)
    key, kq = jax.random.split(key)
    Q = dropout_query(patterns, cfg.mask_p, kq)
    _, meta = _run_all_lines(patterns, Q, phi, cfg, want_confidence=True)
    dist = meta["clu_confidence"]  # smaller = more confident
    correct = (meta["clu_index"] == true_idx).astype(np.float64)
    auroc = _auroc(-dist, correct)  # higher confidence (−dist) should predict correct
    return {
        "dataset": dataset,
        "arm": arm,
        "load": int(M),
        "n_correct": int(correct.sum()),
        "n_total": int(len(correct)),
        "confidence_auroc": float(auroc),
        "mean_dist_correct": float(np.mean(dist[correct == 1]))
        if correct.sum() > 0 else None,
        "mean_dist_incorrect": float(np.mean(dist[correct == 0]))
        if (correct == 0).sum() > 0 else None,
        "note": "confidence = distance-to-nearest-well at settle; AUROC>0.5 means "
        "the retry trigger still separates correct/incorrect reads in φ-space.",
    }


def _auroc(scores, labels):
    """AUROC via the rank-sum (Mann–Whitney) identity. Degenerate → 0.5."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.float64)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_all(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    for cap in results.get("capacity", []):
        rows = cap["rows"]
        if not rows:
            continue
        arm_names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        Ms = [r["M"] for r in rows]
        for name in arm_names:
            ax.plot(Ms, [r[name]["identity_acc"] for r in rows], "o-",
                    label=name, lw=1.3, ms=4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("stored memories M")
        ax.set_ylabel("identity-retrieval accuracy (payload)")
        ax.set_title(f"φ-space capacity — {cap['dataset']} / φ={cap['arm']} "
                     "(50%-masked)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        p = os.path.join(save_dir, f"phi_capacity_{cap['dataset']}_{cap['arm']}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    for nz in results.get("noise", []):
        rows = nz["rows"]
        if not rows:
            continue
        arm_names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sg = [r["sigma"] for r in rows]
        for name in arm_names:
            ax.plot(sg, [r[name]["identity_acc"] for r in rows], "o-",
                    label=name, lw=1.3, ms=4)
        ax.set_xlabel("Gaussian noise σ")
        ax.set_ylabel("identity-retrieval accuracy (payload)")
        ax.set_title(f"φ-space noise — {nz['dataset']} / φ={nz['arm']} "
                     f"(load {nz['fixed_load']})")
        ax.legend(fontsize=6)
        fig.tight_layout()
        p = os.path.join(save_dir, f"phi_noise_{nz['dataset']}_{nz['arm']}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_phi_read_in(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_phi_read_in
    seed = cfg.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "doctrine": "learn around a designed core (φ learned, store designed)",
        "protocol": {
            "uhop_commit": UHOP_COMMIT,
            "ramsauer_commit": RAMSAUER_COMMIT,
            "mask": f"torch.dropout(x, p={cfg.mask_p}) (zero + ×2 survivors)",
            "primary_metric": "mean sqdiff in PIXEL space on the payload (w22-comparable)",
            "store": "key = φ(pattern) as a Gaussian well; payload = raw pattern; "
            "read-out ψ = settle → payload of nearest well",
            "phi_fit": "φ fit on a DISJOINT pool (never sees the store/wells/loss)",
        },
        "config": {
            k: getattr(cfg, k)
            for k in (
                "datasets", "phi_arms", "load_grid", "noise_levels",
                "noise_fixed_load", "n_data_pool", "n_fit_pool", "mask_p",
                "phi_dim", "ae_hidden", "ae_epochs", "activations",
                "clu_s_frac", "clu_b", "clu_alpha", "clu_gamma", "clu_steps",
                "clu_tail_frac", "clu_kinetic_mode", "success_cosine",
            )
        },
        "phi_provenance": [],
    }

    cap, noise, retry_probe = [], [], []
    for ds in cfg.datasets:
        try:
            store_pool, fit_pool = load_store_and_fit_pools(
                ds, cfg.n_data_pool, cfg.n_fit_pool, seed
            )
        except FileNotFoundError as e:
            cap.append({"dataset": ds, "arm": None, "rows": [], "skipped": str(e)})
            continue
        for arm in cfg.phi_arms:
            phi, prov = build_read_in(arm, ds, store_pool, fit_pool, cfg, seed)
            prov["dataset"] = ds
            results["phi_provenance"].append(prov)
            cap.append(capacity_sweep_phi(cfg, ds, arm, phi, store_pool, seed))
            noise.append(noise_sweep_phi(cfg, ds, arm, phi, store_pool, seed))
            if cfg.probe_retry_confidence:
                retry_probe.append(
                    retry_confidence_probe(cfg, ds, arm, phi, store_pool, seed)
                )

    results["capacity"] = cap
    results["noise"] = noise
    results["laundering_control"] = laundering_control(
        [c for c in cap if c["rows"]]
    )
    results["retry_confidence_probe"] = retry_probe
    results["figures"] = _plot_all(results, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_phi_read_in_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_phi_read_in
    cfg.load_grid = [8, 16, 32]
    cfg.noise_levels = [0.0, 0.4, 0.8]
    cfg.noise_fixed_load = 16
    cfg.n_data_pool = 200
    cfg.n_fit_pool = 400
    cfg.phi_dim = 16
    cfg.ae_hidden = 64
    cfg.ae_epochs = 60
    cfg.clu_steps = 60
    cfg.activations = ["softmax", "sparsemax"]
    cfg.rollout_chunk = 64


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="φ read-in around a designed store: Hopfield protocol in φ-space"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--dataset", help="Override datasets (comma-separated)")
    parser.add_argument("--arms", help="Override φ arms (comma-separated: pca,ae)")
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

    if args.quick:
        apply_quick(config)
    if args.dataset:
        config.experiment_phi_read_in.datasets = args.dataset.split(",")
    if args.arms:
        config.experiment_phi_read_in.phi_arms = args.arms.split(",")

    res = run_experiment_phi_read_in(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k in ("laundering_control", "retry_confidence_probe",
                         "phi_provenance")
            },
            indent=2,
        )[:6000]
    )


if __name__ == "__main__":
    main()
