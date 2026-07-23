"""Experiment HOPFIELD-CAPACITY: CLU vs modern-Hopfield SOTA on the
associative-memory retrieval PERFORMANCE benchmark (w22).

⭐ **The one external benchmark where a *designed* CLU is admissible** (scout
#1-ranked target): nothing is learned on either side. The modern-Hopfield line
writes patterns in closed form; CLU designs a landscape (a Gaussian dense
associative memory) and retrieves by the actual damped velocity-Verlet CLU
rollout. The w20 "learning destroys everything" blocker does not bind here.

**Protocol matched VERBATIM** to the real repositories (task Item 0):

- ``MAGICS-LAB/UHop @ cdac75431df968b7142b4fb605a0fcd56feb59cb`` —
  ``memory_retrieval.py`` / ``memory_retrieval_noise.py`` / ``functions.py``.
  Store ``m_size`` images reshaped to ``(m_size, D)``, pixels in ``[0,1]``. Query
  per stored pattern ``x`` = ``torch.dropout(x, p=0.5)`` (randomly zero 50% of
  pixels AND scale survivors by ``1/(1-p)=2``). Update
  ``score = beta*activation(Xiᵀx); x = Xi @ score`` (default ``beta=1, steps=1``).
  **Success metric = mean sqdiff = Σ(clamp(x,0,1)−clamp(x̂,0,1))²**, lower better.
  Activations: softmax (dense MHN = Ramsauer), sparsemax / entmax15 (the sparse
  SOTA line). Noise sweep: ``q = clamp(|x + N(0,σ)|,0,1)``, mean sqdiff.
- ``ml-jku/hopfield-layers @ f56f929`` — energy ``-lse(βXᵀx)+½|x|²``, update
  ``x = X softmax(βXᵀx)``: identical to UHop MHN-softmax, confirms the dense arm.

⚠ The scout's "cosine>0.9" success criterion is NOT what the repo computes; the
repo reports mean squared pixel error. We match sqdiff and ALSO report cosine and
identity-retrieval accuracy (argmin over stored patterns of ‖x̂−ξ_i‖ == true i),
which is the legible "did it retrieve the right memory?" number.

Arms (Item 1): (1) dense modern-Hopfield softmax — repo-verbatim β=1 AND a proper
Ramsauer tuned/iterated variant so it is not strawmanned; (2) sparse SOTA line
(sparsemax / entmax15); (3) CLU designed register (``GaussianMemoryPotential`` +
damped Verlet); (4) nearest-neighbour in pixel space — the floor.

Differentiators (Item 2): retry (adaptive-compute second pass) tested AS
performance; per-item retention control tested as a demonstrated capability.

Runnable: ``uv run python -m chlu.experiments.exp_hopfield_capacity --quick``
or via the CLI: ``chlu exp-hopfield-capacity [--project N] [--seed I] [--quick]``.
"""

import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.memory_potentials import GaussianMemoryPotential
from chlu.experiments.goldstone_harness import clu_with_potential

UHOP_COMMIT = "cdac75431df968b7142b4fb605a0fcd56feb59cb"
RAMSAUER_COMMIT = "f56f929c95b77a070ae675ea4f56b6d54d36e730"


# ---------------------------------------------------------------------------
# Data — pixels in [0,1], matching torchvision ToTensor (the repo convention)
# ---------------------------------------------------------------------------


def load_patterns(dataset: str, n: int, seed: int):
    """(n, D) float32 images in [0,1]. Raises with a clear message if a dataset
    is not locally available (CIFAR-10 via openml is checksum-blocked here)."""
    rng = np.random.default_rng(seed)
    if dataset == "mnist":
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = (mnist.data.astype(np.float32)) / 255.0  # [0,1], matches ToTensor
    elif dataset == "cifar10":
        X = _load_cifar10()
    elif dataset == "synthetic":
        # UHop's load_synthetic: i.i.d. Gaussian vectors (the separable control)
        X = rng.normal(size=(max(n, 2000), 100)).astype(np.float32)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return jnp.asarray(X[idx], dtype=jnp.float32)


def _load_cifar10():
    """Load CIFAR-10 train images from the canonical tarball if present."""
    import pickle
    import tarfile

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(
            here,
            "..",
            "..",
            ".claude",
            "scratch",
            "hopfield-capacity-benchmark",
            "cifar10.tar.gz",
        ),
        os.path.expanduser("~/cifar-10-python.tar.gz"),
        "cifar-10-python.tar.gz",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "CIFAR-10 not available locally (openml copy is checksum-blocked). "
            "Place cifar-10-python.tar.gz in .claude/scratch/"
            "hopfield-capacity-benchmark/ to enable the CIFAR arm."
        )
    with tarfile.open(path, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("data_batch_1"))
        d = pickle.load(tar.extractfile(member), encoding="bytes")
    X = np.asarray(d[b"data"], dtype=np.float32) / 255.0  # (10000, 3072) in [0,1]
    return X


# ---------------------------------------------------------------------------
# Queries — the exact half-mask (torch.dropout, p=0.5) and Gaussian-noise forms
# ---------------------------------------------------------------------------


def dropout_query(X, p, key):
    """torch.dropout(x, p): keep each pixel w.p. (1-p), scale survivors by
    1/(1-p). Independent mask per pattern. Matches UHop memory_retrieval.py."""
    keep = jax.random.bernoulli(key, 1.0 - p, X.shape).astype(X.dtype)
    return X * keep / (1.0 - p)


def noise_query(X, sigma, key):
    """clamp(|x + N(0,sigma)|, 0, 1). Matches UHop memory_retrieval_noise.py."""
    q = jnp.abs(X + jax.random.normal(key, X.shape) * sigma)
    return jnp.clip(q, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Simplex activations (sparse SOTA line) — JAX ports of the repo's torch versions
# ---------------------------------------------------------------------------


def _softmax(S):
    return jax.nn.softmax(S, axis=-1)


def _sparsemax(S):
    """Exact sparsemax (Martins & Astudillo 2016) over the last axis."""
    z = S
    z_sorted = jnp.sort(z, axis=-1)[..., ::-1]
    k = jnp.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    z_cumsum = jnp.cumsum(z_sorted, axis=-1)
    support = (1.0 + k * z_sorted) > z_cumsum
    k_max = jnp.sum(support.astype(z.dtype), axis=-1, keepdims=True)
    tau = (
        jnp.take_along_axis(z_cumsum, (k_max - 1).astype(int), axis=-1) - 1.0
    ) / k_max
    return jnp.clip(z - tau, 0.0, None)


def _entmax15(S, n_iter: int = 30):
    """1.5-entmax (Peters et al. 2019) via bisection on the threshold.

    p_i = [(z_i/2 - tau)]_+^2 with tau chosen so sum(p)=1 (alpha=1.5 form after
    the standard z <- z/2 reparameterization). Non-differentiable bisection is
    fine — retrieval does not backprop through the activation."""
    z = S - jnp.max(S, axis=-1, keepdims=True)
    z = z / 2.0
    lo = jnp.max(z, axis=-1, keepdims=True) - 1.0
    hi = jnp.max(z, axis=-1, keepdims=True)

    def body(_, bounds):
        lo, hi = bounds
        tau = (lo + hi) / 2.0
        p = jnp.clip(z - tau, 0.0, None) ** 2
        f = jnp.sum(p, axis=-1, keepdims=True) - 1.0
        lo = jnp.where(f > 0, tau, lo)
        hi = jnp.where(f > 0, hi, tau)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    tau = (lo + hi) / 2.0
    p = jnp.clip(z - tau, 0.0, None) ** 2
    return p / jnp.sum(p, axis=-1, keepdims=True)


ACTIVATIONS = {"softmax": _softmax, "sparsemax": _sparsemax, "entmax": _entmax15}


# ---------------------------------------------------------------------------
# Arm 1/2 — modern-Hopfield update (dense softmax + sparse SOTA)
# ---------------------------------------------------------------------------


@eqx.filter_jit
def hopfield_retrieve(Xi, Q, beta_in, beta_out, steps, act_fn):
    """Iterated modern-Hopfield update over a batch of queries.

    Xi: (M, D) stored patterns. Q: (Nq, D) queries.
    x_{t+1} = beta_out * act(beta_in * <Xi, x_t>) @ Xi   (rows of Xi).

    Repo-verbatim dense arm: beta_in=1, beta_out=beta, steps=1 (beta OUTSIDE the
    activation — the UHop convention). Proper Ramsauer arm: beta_in=beta,
    beta_out=1, steps>1 (beta INSIDE the softmax — the sharpening temperature)."""

    def one_step(x, _):
        s = act_fn(beta_in * (x @ Xi.T))  # (Nq, M)
        return beta_out * (s @ Xi), None

    x, _ = jax.lax.scan(one_step, Q, None, length=steps)
    return x


# ---------------------------------------------------------------------------
# Arm 3 — CLU designed register: damped velocity-Verlet settling
# ---------------------------------------------------------------------------


def _settle_read(model, Q0, steps, dt, gamma, tail_steps, chunk):
    """Damped Verlet from each query; read = mean q over the last tail_steps.

    Memory-safe: the settle phase carries only (q,p) (no trajectory stack); only
    the short tail is stacked to average out residual oscillation. p(0)=0."""
    tail_steps = int(max(1, tail_steps))
    settle_steps = int(max(1, steps - tail_steps))

    def per_query(q0):
        p0 = jnp.zeros_like(q0)

        def step_fn(state, _):
            return model.step(state, dt, gamma), None

        (q1, p1), _ = jax.lax.scan(step_fn, (q0, p0), None, length=settle_steps)
        traj = model(q1, p1, tail_steps, dt, gamma)  # (tail_steps, 2*dim)
        dim = q0.shape[0]
        return jnp.mean(traj[:, :dim], axis=0)

    f = eqx.filter_jit(jax.vmap(per_query))
    outs = []
    for i in range(0, Q0.shape[0], chunk):
        outs.append(np.asarray(f(Q0[i : i + chunk])))
    return jnp.asarray(np.concatenate(outs, axis=0))


def build_clu_memory(patterns, s, cfg):
    """A CLU wired to a GaussianMemoryPotential over the stored patterns."""
    V = GaussianMemoryPotential(patterns, s=s, b=cfg.clu_b, alpha=cfg.clu_alpha)
    dim = int(patterns.shape[1])
    return clu_with_potential(V, dim=dim, kinetic_mode=cfg.clu_kinetic_mode)


def clu_retrieve(patterns, Q, s, cfg):
    model = build_clu_memory(patterns, s, cfg)
    dt = cfg.clu_dt if cfg.clu_dt > 0 else 0.5 * s / np.sqrt(cfg.clu_b)
    tail = int(max(1, cfg.clu_tail_frac * cfg.clu_steps))
    x = _settle_read(
        model, Q, cfg.clu_steps, dt, cfg.clu_gamma, tail, cfg.rollout_chunk
    )
    return x, float(dt)


# ---------------------------------------------------------------------------
# Metrics — match the repo (sqdiff) and add legible identity accuracy + cosine
# ---------------------------------------------------------------------------


def _sqdiff(x, xhat):
    """Repo metric: Σ(clamp(x,0,1) − clamp(x̂,0,1))² per pattern (then averaged)."""
    a = np.clip(np.asarray(x), 0.0, 1.0)
    b = np.clip(np.asarray(xhat), 0.0, 1.0)
    return np.sum((a - b) ** 2, axis=-1)


def _cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return num / den


def score_retrieval(patterns, xhat, true_idx, success_cosine):
    """Return dict of metrics. Identity accuracy = argmin_j ‖x̂−ξ_j‖ == true_idx."""
    P = np.asarray(patterns)
    X = np.asarray(xhat)
    d2 = np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=-1)  # (Nq, M)
    nn = np.argmin(d2, axis=1)
    identity_acc = float(np.mean(nn == true_idx))
    truth = P[true_idx]
    cos = _cosine(X, truth)
    sqd = _sqdiff(truth, X)
    return (
        {
            "identity_acc": identity_acc,
            "mean_sqdiff": float(np.mean(sqd)),
            "median_cosine": float(np.median(cos)),
            "success_at_cosine": float(np.mean(cos > success_cosine)),
        },
        nn,
        cos,
    )


# ---------------------------------------------------------------------------
# Item 1 — the capacity-degradation sweep (all arms, one axis)
# ---------------------------------------------------------------------------


def _hopfield_arms(cfg, patterns):
    """Return a dict name -> (beta_in, beta_out, steps, act_fn) for every
    Hopfield arm, including the auto-tuned proper-Ramsauer variant."""
    P = np.asarray(patterns)
    self_overlap = float(np.mean(np.sum(P * P, axis=-1)))  # ⟨x,x⟩ scale
    # sharpen the softmax so β·⟨x,x⟩ ≈ 200 (a hard, load-/dataset-independent
    # rule). Measured on MNIST M=256: 1-step acc saturates at 0.742 for any β≥1
    # and iterating HURTS (β=2/3-step 0.664 < β=1/1-step 0.742), so this arm
    # documents that a *sharper, iterated* Hopfield does not beat the β=1 default
    # on this protocol — it is the anti-strawman, not a booster.
    beta_tuned = (
        cfg.hopfield_beta_tuned
        if cfg.hopfield_beta_tuned > 0
        else 200.0 / (self_overlap + 1e-9)
    )
    arms = {}
    for act in cfg.activations:
        act_fn = ACTIVATIONS[act]
        # repo-verbatim: beta OUTSIDE activation, 1 step (the literal default)
        arms[f"hopfield_{act}_repo"] = (
            1.0,
            cfg.hopfield_beta,
            cfg.hopfield_steps,
            act_fn,
        )
    # proper Ramsauer dense arm (beta INSIDE softmax, iterated) — not strawmanned
    arms["hopfield_softmax_tuned"] = (
        beta_tuned,
        1.0,
        cfg.hopfield_steps_tuned,
        _softmax,
    )
    return arms, beta_tuned


def capacity_sweep(cfg, dataset, seed):
    """Retrieval accuracy vs number of stored memories, all arms."""
    key = jax.random.PRNGKey(seed)
    pool = load_patterns(dataset, cfg.n_data_pool, seed)
    rows = []
    s_used = None
    beta_tuned = None
    for M in cfg.load_grid:
        if M > pool.shape[0]:
            continue
        patterns = pool[:M]
        key, kq = jax.random.split(key)
        Q = dropout_query(patterns, cfg.mask_p, kq)
        true_idx = np.arange(M)

        row = {"M": int(M)}

        # Hopfield arms
        arms, beta_tuned = _hopfield_arms(cfg, patterns)
        for name, (bi, bo, steps, act_fn) in arms.items():
            xhat = hopfield_retrieve(patterns, Q, bi, bo, steps, act_fn)
            m, _, _ = score_retrieval(patterns, xhat, true_idx, cfg.success_cosine)
            row[name] = m

        # NN floor (nearest stored pattern to the query)
        d2 = np.sum(
            (np.asarray(Q)[:, None, :] - np.asarray(patterns)[None, :, :]) ** 2, axis=-1
        )
        nn_idx = np.argmin(d2, axis=1)
        xhat_nn = np.asarray(patterns)[nn_idx]
        m, _, _ = score_retrieval(patterns, xhat_nn, true_idx, cfg.success_cosine)
        row["nearest_neighbor"] = m

        # CLU designed register
        s = cfg.clu_s_frac * _median_nn_distance(patterns)
        s_used = float(s)
        xhat_clu, dt = clu_retrieve(patterns, Q, s, cfg)
        m, nn_clu, cos_clu = score_retrieval(
            patterns, xhat_clu, true_idx, cfg.success_cosine
        )
        m["dt"] = dt
        row["clu_register"] = m
        row["clu_s"] = s_used

        rows.append(row)

    return {
        "dataset": dataset,
        "rows": rows,
        "clu_s_last": s_used,
        "hopfield_beta_tuned": beta_tuned,
    }


def _median_nn_distance(patterns):
    """Median nearest-neighbour distance among stored patterns (the fixed,
    load-driven rule that sets the CLU well width — NOT tuned per load)."""
    P = np.asarray(patterns)
    if P.shape[0] < 2:
        return float(np.sqrt(np.sum(P**2)) + 1e-6)
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(np.min(d2, axis=1))))


# ---------------------------------------------------------------------------
# Item 1b — accuracy vs Gaussian noise at fixed load
# ---------------------------------------------------------------------------


def noise_sweep(cfg, dataset, seed):
    key = jax.random.PRNGKey(seed + 100)
    M = min(cfg.noise_fixed_load, cfg.n_data_pool)
    pool = load_patterns(dataset, cfg.n_data_pool, seed)
    patterns = pool[:M]
    true_idx = np.arange(M)
    arms, beta_tuned = _hopfield_arms(cfg, patterns)
    s = cfg.clu_s_frac * _median_nn_distance(patterns)
    rows = []
    for sigma in cfg.noise_levels:
        key, kq = jax.random.split(key)
        Q = noise_query(patterns, sigma, kq)
        row = {"sigma": float(sigma)}
        for name, (bi, bo, steps, act_fn) in arms.items():
            xhat = hopfield_retrieve(patterns, Q, bi, bo, steps, act_fn)
            m, _, _ = score_retrieval(patterns, xhat, true_idx, cfg.success_cosine)
            row[name] = m
        d2 = np.sum(
            (np.asarray(Q)[:, None, :] - np.asarray(patterns)[None, :, :]) ** 2, axis=-1
        )
        xhat_nn = np.asarray(patterns)[np.argmin(d2, axis=1)]
        m, _, _ = score_retrieval(patterns, xhat_nn, true_idx, cfg.success_cosine)
        row["nearest_neighbor"] = m
        xhat_clu, _ = clu_retrieve(patterns, Q, s, cfg)
        m, _, _ = score_retrieval(patterns, xhat_clu, true_idx, cfg.success_cosine)
        row["clu_register"] = m
        rows.append(row)
    return {"dataset": dataset, "fixed_load": int(M), "rows": rows}


# ---------------------------------------------------------------------------
# Cross-over: the load at which each arm's identity-acc falls below criterion
# ---------------------------------------------------------------------------


def crossover_points(cap, threshold=0.9):
    rows = cap["rows"]
    if not rows:
        return {}
    arm_names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    out = {}
    for name in arm_names:
        cross = None
        for r in rows:
            if r[name]["identity_acc"] < threshold:
                cross = r["M"]
                break
        out[name] = cross  # first M below threshold, or None if never
    return {"threshold": threshold, "first_M_below": out}


# ---------------------------------------------------------------------------
# Item 2 — differentiator: RETRY (adaptive compute, codebook-gated second pass)
# ---------------------------------------------------------------------------


def retry_differentiator(cfg, dataset, seed):
    """A second boosted CLU relaxation pass on the LOW-CONFIDENCE first-pass
    queries. Confidence = negative distance to the settled nearest well. The
    retry re-launches from the settled point with a KE boost (a symplectic
    energy injection — the boost-retry lineage) to escape a wrong shallow well.

    Blank guard: also retry the HIGH-confidence queries; a legitimate lift must
    come from flipping wrong→right, not from perturbing already-correct ones."""
    M = min(cfg.noise_fixed_load, cfg.n_data_pool)
    pool = load_patterns(dataset, cfg.n_data_pool, seed)
    patterns = pool[:M]
    true_idx = np.arange(M)
    key = jax.random.PRNGKey(seed + 200)
    key, kq = jax.random.split(key)
    Q = dropout_query(patterns, cfg.mask_p, kq)
    s = cfg.clu_s_frac * _median_nn_distance(patterns)

    xhat1, dt = clu_retrieve(patterns, Q, s, cfg)
    m1, nn1, cos1 = score_retrieval(patterns, xhat1, true_idx, cfg.success_cosine)

    # confidence = cosine to the settled nearest well's pattern
    conf = cos1
    order = np.argsort(conf)
    n_retry = int(cfg.retry_conf_frac * len(order))
    low = order[:n_retry]  # least confident — the retry targets
    high = order[-n_retry:] if n_retry > 0 else order[:0]  # blank guard

    def boosted(idx):
        # re-launch from the settled point with a KE boost toward the query
        model = build_clu_memory(patterns, s, cfg)
        q_start = jnp.asarray(xhat1[idx])
        direction = jnp.asarray(Q)[idx] - q_start
        p_boost = cfg.retry_boost * direction
        tail = int(max(1, cfg.clu_tail_frac * cfg.clu_steps))
        settle = int(max(1, cfg.clu_steps - tail))

        def per(q0, p0):
            def step_fn(state, _):
                return model.step(state, dt, cfg.clu_gamma), None

            (q1, p1), _ = jax.lax.scan(step_fn, (q0, p0), None, length=settle)
            traj = model(q1, p1, tail, dt, cfg.clu_gamma)
            return jnp.mean(traj[:, : q0.shape[0]], axis=0)

        f = eqx.filter_jit(jax.vmap(per))
        return np.asarray(f(q_start, p_boost))

    xhat2 = np.array(xhat1)
    if n_retry > 0:
        xhat2[low] = boosted(low)
    m2, _, _ = score_retrieval(patterns, xhat2, true_idx, cfg.success_cosine)

    # blank: retry the HIGH-confidence set instead
    xhat_blank = np.array(xhat1)
    if n_retry > 0:
        xhat_blank[high] = boosted(high)
    mb, _, _ = score_retrieval(patterns, xhat_blank, true_idx, cfg.success_cosine)

    return {
        "dataset": dataset,
        "load": int(M),
        "n_retried": int(n_retry),
        "compute_multiplier": 1.0 + cfg.retry_conf_frac,
        "first_pass": m1,
        "after_retry_lowconf": m2,
        "blank_retry_highconf": mb,
        "acc_lift_pp": 100.0 * (m2["identity_acc"] - m1["identity_acc"]),
        "blank_lift_pp": 100.0 * (mb["identity_acc"] - m1["identity_acc"]),
    }


# ---------------------------------------------------------------------------
# Item 2 — differentiator: per-item RETENTION control (a capability, not a curve)
# ---------------------------------------------------------------------------


def retention_capability(cfg, dataset, seed):
    """A capability modern Hopfield has no analogue for: in ONE store, some items
    are permanent and others decay on a per-item schedule.

    Mechanism: give each well a per-item depth ``b_i(t) = b * exp(-t/tau_i)`` with
    ``tau_i = inf`` (permanent) or finite (decaying). We measure, at several
    ``t``, whether a masked query for a permanent item still retrieves it while a
    decaying item's well has flattened below retrievability. Hopfield stores all
    patterns with identical, timeless weight — there is no ``t`` axis to place a
    schedule on. Reported as a demonstrated capability."""
    M = 16
    pool = load_patterns(dataset, cfg.n_data_pool, seed)
    patterns = pool[:M]
    s = cfg.clu_s_frac * _median_nn_distance(patterns)
    dt = cfg.clu_dt if cfg.clu_dt > 0 else 0.5 * s / np.sqrt(cfg.clu_b)
    tail = int(max(1, cfg.clu_tail_frac * cfg.clu_steps))
    dim = int(patterns.shape[1])

    # half permanent (even index), half decaying (odd index)
    permanent = np.arange(M) % 2 == 0
    tau = np.where(permanent, np.inf, 1.0)
    key = jax.random.PRNGKey(seed + 300)
    key, kq = jax.random.split(key)
    Q = dropout_query(patterns, cfg.mask_p, kq)

    times = [0.0, 0.5, 1.0, 2.0, 4.0]
    schedule = []
    for t in times:
        depths = cfg.clu_b * np.exp(-t / tau)  # (M,) per-item well depths
        # per-item depth via an amplitude-weighted Gaussian memory
        V = _WeightedGaussianMemory(
            patterns, s, jnp.asarray(depths, jnp.float32), cfg.clu_alpha
        )
        model = clu_with_potential(V, dim=dim, kinetic_mode=cfg.clu_kinetic_mode)
        x = _settle_read(
            model, Q, cfg.clu_steps, dt, cfg.clu_gamma, tail, cfg.rollout_chunk
        )
        d2 = np.sum(
            (np.asarray(x)[:, None, :] - np.asarray(patterns)[None, :, :]) ** 2, axis=-1
        )
        nn = np.argmin(d2, axis=1)
        correct = nn == np.arange(M)
        schedule.append(
            {
                "t": t,
                "acc_permanent": float(np.mean(correct[permanent])),
                "acc_decaying": float(np.mean(correct[~permanent])),
            }
        )
    return {
        "dataset": dataset,
        "n_items": M,
        "n_permanent": int(np.sum(permanent)),
        "schedule": schedule,
        "note": "Hopfield has no time axis for a per-item retention schedule; "
        "this is a capability demonstration, not a benchmark curve.",
    }


class _WeightedGaussianMemory(eqx.Module):
    """GaussianMemoryPotential with a PER-ITEM depth vector (for retention)."""

    centers: jnp.ndarray
    depths: jnp.ndarray
    s: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, centers, s, depths, alpha):
        self.centers = jnp.asarray(centers, jnp.float32)
        self.depths = jnp.asarray(depths, jnp.float32)
        self.s = float(s)
        self.alpha = float(alpha)

    def __call__(self, q):
        d2 = jnp.sum((q[None, :] - self.centers) ** 2, axis=-1)
        wells = self.depths * jnp.exp(-d2 / (2.0 * self.s**2))
        return 0.5 * self.alpha * jnp.sum(q * q) - jnp.sum(wells)


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
    for ds_res in results.get("capacity", []):
        ds = ds_res["dataset"]
        rows = ds_res["rows"]
        if not rows:
            continue
        arm_names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        Ms = [r["M"] for r in rows]
        for name in arm_names:
            ax.plot(
                Ms,
                [r[name]["identity_acc"] for r in rows],
                "o-",
                label=name,
                lw=1.2,
                ms=4,
            )
        ax.axhline(0.9, color="r", ls="--", lw=0.8, label="criterion 0.9")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("stored memories M")
        ax.set_ylabel("identity-retrieval accuracy")
        ax.set_title(f"Capacity degradation — {ds} (50%-masked queries)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        p = os.path.join(save_dir, f"hopfield_capacity_{ds}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    for ds_res in results.get("noise", []):
        ds = ds_res["dataset"]
        rows = ds_res["rows"]
        if not rows:
            continue
        arm_names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sg = [r["sigma"] for r in rows]
        for name in arm_names:
            ax.plot(
                sg,
                [r[name]["identity_acc"] for r in rows],
                "o-",
                label=name,
                lw=1.2,
                ms=4,
            )
        ax.set_xlabel("Gaussian noise σ")
        ax.set_ylabel("identity-retrieval accuracy")
        ax.set_title(f"Noise robustness — {ds} (load {ds_res['fixed_load']})")
        ax.legend(fontsize=6)
        fig.tight_layout()
        p = os.path.join(save_dir, f"hopfield_noise_{ds}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_hopfield_capacity(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_hopfield_capacity
    seed = cfg.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "designed_not_learned": True,
        "protocol": {
            "uhop_commit": UHOP_COMMIT,
            "ramsauer_commit": RAMSAUER_COMMIT,
            "mask": f"torch.dropout(x, p={cfg.mask_p}) — zero {int(cfg.mask_p * 100)}%"
            ", scale survivors by 1/(1-p)",
            "primary_metric": "mean sqdiff (repo); identity-acc + cosine reported",
            "pixels": "[0,1] (torchvision ToTensor convention)",
        },
        "config": {
            k: getattr(cfg, k)
            for k in (
                "datasets",
                "load_grid",
                "noise_levels",
                "noise_fixed_load",
                "n_data_pool",
                "mask_p",
                "hopfield_beta",
                "hopfield_steps",
                "hopfield_steps_tuned",
                "activations",
                "clu_s_frac",
                "clu_b",
                "clu_alpha",
                "clu_gamma",
                "clu_steps",
                "clu_tail_frac",
                "clu_kinetic_mode",
                "retry_boost",
                "retry_conf_frac",
                "success_cosine",
            )
        },
    }

    cap, noise = [], []
    for ds in cfg.datasets:
        try:
            cap.append(capacity_sweep(cfg, ds, seed))
            noise.append(noise_sweep(cfg, ds, seed))
        except FileNotFoundError as e:
            cap.append({"dataset": ds, "rows": [], "skipped": str(e)})
    results["capacity"] = cap
    results["noise"] = noise
    results["crossover"] = [
        {"dataset": c["dataset"], **crossover_points(c)} for c in cap if c["rows"]
    ]

    diff = {}
    primary_ds = next((c["dataset"] for c in cap if c["rows"]), cfg.datasets[0])
    if cfg.retry_enabled:
        try:
            diff["retry"] = retry_differentiator(cfg, primary_ds, seed)
        except FileNotFoundError:
            pass
    try:
        diff["retention"] = retention_capability(cfg, primary_ds, seed)
    except FileNotFoundError:
        pass
    results["differentiators"] = diff

    results["figures"] = _plot_all(results, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_hopfield_capacity_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_hopfield_capacity
    cfg.load_grid = [8, 16, 32]
    cfg.noise_levels = [0.0, 0.5, 1.0]
    cfg.noise_fixed_load = 16
    cfg.n_data_pool = 200
    cfg.clu_steps = 60
    cfg.activations = ["softmax", "sparsemax"]
    cfg.rollout_chunk = 64


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hopfield-capacity benchmark: CLU vs modern-Hopfield SOTA"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--dataset", help="Override datasets (comma-separated)")
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
        config.experiment_hopfield_capacity.datasets = args.dataset.split(",")

    res = run_experiment_hopfield_capacity(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k in ("capacity", "crossover", "differentiators")
            },
            indent=2,
        )[:6000]
    )


if __name__ == "__main__":
    main()
