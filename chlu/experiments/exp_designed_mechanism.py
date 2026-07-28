"""Experiment DESIGNED-MECHANISM-LEARNED-CONTENT (w22): is the K=8 wall GEOMETRY
or LEARNING?

Every "primitive" is a **designed mechanism with learned content** (attention's
``softmax(QK)V`` is fixed; ``W_{Q,K,V}`` are learned). The fair CLU configuration is
a fixed atom-dictionary MECHANISM (:class:`AtomDictionaryPotential`) with **learned**
amplitudes/centers/widths, written by the static objective
(:func:`chlu.training.train_memory`). In ``potential-function-class`` that arm scored
0.980 @K=4 and **broke** at 0.741 @K=8 — but K=8 sits at the 2-D ring's own capacity
ceiling (``K_max≈8.4``), so we cannot tell whether the wall is:

* **H-GEOMETRY** — the ring ran out of room; the wall moves up with ``d`` (designed
  capacity is ``4·2^d``). Primitive claim alive.
* **H-LEARNING** — gradient descent cannot fill a landscape past ~8 items regardless
  of room; the wall stays near 8 at every ``d``. Primitive claim in trouble.

This module discriminates them by sweeping the **address dimension** ``d`` (a
``d``-ball address space, sites farthest-point-packed by :func:`designed_sites`,
payload channel at index ``d``) and measuring, at each ``d``, ``K_learned`` = the
largest item count a LEARNED atom dictionary clears at strict 0.9 (leak-immune value
criterion, blank control on every cell), overlaid on ``K_designed`` re-measured on the
IDENTICAL harness with a hand-built :class:`BallRegisterPotential`.

⚠ **The parameter ceiling is a confound (theorist §4.3, ``B_total ≤ P·b_θ``).** The
atom count is **scaled with K** (``n_atoms = atoms_per_item·K``, ``n_groups = K``) so a
plateau is a *learning* failure, not a *parameterization-capacity* failure. ``P`` is
reported per cell.

Items:

1. ``item1_discriminator``  ⭐ ``K_learned`` vs ``d`` and ``K_designed`` vs ``d`` on
    one axis, ≥5 seeds on the learned cells, with the fitted growth of ``K_learned``.
2. ``item2_mass``           per-item learned/assigned mass vs uniform, WITH the
    address-coupling check (Prop F1: mass helps only if ``∂_i∂_j V ≠ 0``).
3. ``item3_interference``   masked vs global cross-write corruption at each ``d``.
4. ``item4_frontier``       the (d,K) performance frontier: where learned content
    matches the designed ceiling and where it falls away.

Runnable directly:
    uv run python -m chlu.experiments.exp_designed_mechanism --quick
or via the CLI: ``chlu exp-designed-mechanism [--project N] [--seed I] [--quick]``.
"""

import copy
import json
import os
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.memory_potentials import (
    BallRegisterPotential,
    DesignFreedomPotential,
    atom_write_mask_fn,
    designed_payloads,
    designed_sites,
    site_separation,
)
from chlu.experiments.exp_retrieval import (
    linear_codebook_read,
    nearest_centroid_read,
)
from chlu.experiments.goldstone_harness import clu_with_potential
from chlu.training.train_memory import train_memory_landscape

# ---------------------------------------------------------------------------
# Geometry: the d-ball address space (address q[:d], payload q[d])
# ---------------------------------------------------------------------------


def n_pay_channels(cfg) -> int:
    """``m`` — the number of read-out (payload) channels (w26 arm (a)); 1 = shipped."""
    return max(1, int(getattr(cfg, "n_payload_channels", 1)))


def payload_codebook(K: int, cfg):
    """(K, m) stored codewords. ``m=1`` + ``"linspace"`` = the shipped codebook.

    ⭐ w26 arm (a) — the **min-separation-preserving grid code** (fairness condition
    1). The 1-channel codebook is ``linspace(-1, 1, K)``, whose minimum pairwise
    separation is ``delta = 2/(K-1)`` and whose maximum excursion is ``1``. The
    ``m``-channel code places the K codewords on the ``delta``-spaced integer lattice
    in ``R^m``, keeping the K lattice points of SMALLEST NORM (ties broken
    deterministically), and then applies the same fixed permutation
    :func:`designed_payloads` uses, so the code stays non-monotone in the site index.

    Consequences, and they are the whole point:

    * minimum codeword separation is **exactly delta at every m** => at a given
      per-channel read noise the K codewords are exactly as distinguishable, i.e.
      the item carries the same ``log2 K`` bits (condition 1);
    * the maximum excursion falls from ``1`` to ``~delta*sqrt(m)*(K^(1/m)-1)/2``,
      i.e. the *reach demand* drops while the *precision* does not;
    * the total noise entering the value test grows as ``sqrt(m)`` at fixed per-axis
      sigma, so the multi-channel arm is if anything penalised, never flattered.
    """
    m = n_pay_channels(cfg)
    code = getattr(cfg, "payload_code", "linspace")
    if m == 1 and code == "linspace":
        return np.asarray(designed_payloads(K, seed=cfg.payload_seed))[:, None]
    if code not in ("linspace", "grid"):
        raise ValueError(f"payload_code must be 'linspace' or 'grid', got {code!r}")
    delta = 2.0 / max(K - 1, 1)
    n_side = int(np.ceil(K ** (1.0 / m)))
    while n_side**m < K:
        n_side += 1
    axis = (np.arange(n_side) - (n_side - 1) / 2.0) * delta
    grid = np.stack(
        np.meshgrid(*([axis] * m), indexing="ij"), axis=-1
    ).reshape(-1, m)
    # keep the K smallest-norm lattice points (deterministic tie-break on the
    # lexicographic index, which np.argsort(kind="stable") supplies)
    order = np.argsort(np.sum(grid**2, axis=1), kind="stable")
    words = grid[order[:K]]
    rng = np.random.default_rng(cfg.payload_seed)
    return np.asarray(rng.permutation(words), dtype=np.float32)


def ball_setup(d: int, K: int, cfg, payloads=None):
    """Centers (K,d), payloads, targets (K,d+m), min site separation.

    ``payloads`` keeps the shipped **(K,)** shape at ``m = 1`` (so every w20-w25
    caller is unaffected) and is **(K,m)** for the w26 multi-channel code.
    """
    centers = designed_sites(d, K, R=cfg.R, seed=cfg.site_seed)
    m = n_pay_channels(cfg)
    if payloads is None:
        payloads = payload_codebook(K, cfg)
    payloads = jnp.asarray(payloads, dtype=jnp.float32)
    if m == 1:
        payloads = payloads.reshape(-1)
    elif payloads.ndim == 1:
        payloads = payloads[:, None]
    dim = d + m
    targets = jnp.zeros((K, dim))
    targets = targets.at[:, :d].set(centers).at[:, d : d + m].set(
        payloads.reshape(K, m)
    )
    return centers, payloads, targets, site_separation(centers)


def _floor_atoms(cfg, d: int) -> int:
    """Dimension-aware atom FLOOR ``max(min_atoms, round(min_atoms_base·c^d))``.

    ⚠ w23 (dimension-aware-budget). w22 used a FIXED floor (``min_atoms``), which is
    inadequate at high ``d``: the flat-start atoms init ``N(0, atom_init_scale)`` in
    the ``(d+1)``-ball, and the fraction landing near any stored site (radius ~R)
    DECAYS roughly geometrically per added dimension, so a fixed count starves the
    write at high ``d`` (measured w22: d=8 K=2 stalled at strict 0.400 with site
    separation 1.838 — geometrically trivial). A geometric floor ``c^d`` holds the
    atoms-near-each-site count ~constant across ``d``. ``c = min_atoms_c = 2.0``
    matches the designed capacity growth ``4·2^d`` and compensates the per-dimension
    near-site thinning; ``base = min_atoms_base`` pins the low-``d`` floor.
    """
    geo = int(round(cfg.min_atoms_base * cfg.min_atoms_c**d))
    return max(cfg.min_atoms, geo)


def _atoms_for(cfg, K: int, d: int) -> int:
    """Atom count = ``max(atoms_per_item·K, floor(d))``.

    The ``·K`` term scales the parameter budget with K so a ``K_learned`` plateau is
    a LEARNING failure, not a parameterization-capacity one (theorist §4.3). The
    dimension-aware FLOOR (:func:`_floor_atoms`) is load-bearing and separate: a
    large over-complete dictionary *smooths the write optimization*, and it must
    scale with the address DIMENSION (not only with K) or the ladder walk terminates
    on a starved low-K high-``d`` cell (w22 §7 confound, w23 fix). The floor keeps
    every cell over-complete; the ·K term dominates once K is large.
    """
    return max(cfg.atoms_per_item * K, _floor_atoms(cfg, d))


def build_designed_model(centers, payloads, cfg) -> CHLU:
    """CLU wired to a hand-built d-ball register (the designed ceiling arm)."""
    V = BallRegisterPotential(
        payloads,
        centers,
        R=cfg.R + cfg.wall_margin,
        w=cfg.well_width,
        b=cfg.well_depth,
        kappa=cfg.payload_kappa,
        c_conf=cfg.c_conf,
    )
    dim = V.dim
    return clu_with_potential(
        V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )


def build_learned_V(d: int, K: int, cfg, key, centers=None) -> DesignFreedomPotential:
    """A LEARNED atom dictionary on the (d+1)-dim latent, K groups, K-scaled atoms.

    Returned as a ``DesignFreedomPotential(rung="free_mlp", learned_family="atoms")``
    whose designed part is ``None`` — i.e. a PURE ``AtomDictionaryPotential`` wrapped
    in the ``.learned`` container that ``train_memory.trainable_filter`` and
    ``atom_write_mask_fn`` require (a bare atom dictionary has no ``.learned`` subtree,
    so ``train_memory_landscape`` would train it to a silent no-op).

    Starts FLAT (``depth_init`` tiny, ``A = amp**2``) so the writer digs the wells;
    partitioned into ``K`` contiguous atom blocks so a masked write is local in
    parameter space (one block per item slot). ``.learned`` is the atom dictionary.

    ⭐ w26 Stage A: with ``cfg.atom_init_local`` and ``centers`` supplied, group j's
    atoms are initialised in a ball of radius ``atom_init_local_mult *
    atom_init_width`` around item j's ADDRESS site (the N98 localized init, ported
    from ``exp_sharded_store``). Address axes only — see the N46 note on
    :class:`AtomDictionaryPotential`. ``centers=None`` or the flag off reproduces the
    historical scatter bit-for-bit.
    """
    n_atoms = _atoms_for(cfg, K, d)
    local = bool(getattr(cfg, "atom_init_local", False)) and centers is not None
    radius = (
        float(getattr(cfg, "atom_init_local_mult", 2.0)) * float(cfg.atom_init_width)
        if local
        else 0.0
    )
    return DesignFreedomPotential(
        rung="free_mlp",
        dim=d + n_pay_channels(cfg),
        payloads=jnp.zeros((K,)),  # unused: the free_mlp rung has designed=None
        key=key,
        learned_family="atoms",
        n_atoms=n_atoms,
        rbf_init_width=cfg.atom_init_width,
        confine=cfg.learned_confine,
        atom_depth_init=cfg.atom_depth_init,
        atom_groups=K,
        atom_init_scale=cfg.atom_init_scale,
        atom_group_centers=jnp.asarray(centers) if local else None,
        atom_local_radius=radius,
    )


def trained_well_widths(V, targets, top_frac: float = 0.9) -> dict:
    """⭐ w25 (`lattice-capacity-theory` §5.0): the TRAINED well width at the stored sites.

    ``atom_init_width`` is only an *initialisation* — ``log_width`` is trainable, so the
    width that actually sets the geometry (and hence the ``minsep/width`` ratio the
    geometric account of the capacity ceiling is built on) is a **measured**, not a
    configured, quantity. This returns it, for one written landscape.

    Per stored site ``x*`` the atoms are ranked by their contribution to the well there,
    ``A_j·exp(-|x*-c_j|²/2s_j²)`` with ``A_j = amp_j²``; the atoms supplying the top
    ``top_frac`` of the summed contribution are the ones that *form* the well, and the
    reported width is the contribution-weighted median of their ``s_j``. Reported per
    site and as the median over sites.

    Args:
        V: a written ``DesignFreedomPotential`` with an atom-dictionary ``.learned``
            (or a bare :class:`AtomDictionaryPotential`).
        targets: (K, dim) stored sites in the latent (address + payload channel).
        top_frac: contribution mass that defines "the atoms that form the well".

    Returns:
        ``{"w_atom_per_site", "w_atom", "n_keep_per_site", "all_atom_width_median",
        "init_width_atoms"}`` — ``w_atom`` is the median over sites (the headline).
    """
    atoms = V.learned if hasattr(V, "learned") else V
    C = np.asarray(atoms.centers)
    S = np.exp(np.asarray(atoms.log_width))
    A = np.asarray(atoms.amp) ** 2
    T = np.asarray(targets)

    per_site, n_keep = [], []
    for i in range(T.shape[0]):
        d2 = ((C - T[i][None, :]) ** 2).sum(-1)
        contrib = A * np.exp(-d2 / (2.0 * S**2))
        order = np.argsort(-contrib)
        cum = np.cumsum(contrib[order]) / max(float(contrib.sum()), 1e-30)
        sel = order[: int(np.searchsorted(cum, top_frac) + 1)]
        w, c = S[sel], contrib[sel]
        o = np.argsort(w)
        cw = np.cumsum(c[o]) / max(float(c.sum()), 1e-30)
        per_site.append(float(w[o][int(np.searchsorted(cw, 0.5))]))
        n_keep.append(int(sel.size))
    return {
        "w_atom_per_site": per_site,
        "w_atom": float(np.median(per_site)),
        "n_keep_per_site": n_keep,
        "all_atom_width_median": float(np.median(S)),
        "n_atoms": int(S.size),
    }


def _n_params(V) -> int:
    return int(
        sum(
            x.size
            for x in jax.tree_util.tree_leaves(eqx.filter(V, eqx.is_inexact_array))
        )
    )


def _loss_kwargs(cfg, d: int) -> dict:
    m = n_pay_channels(cfg)
    # payload channel(s) are q[d:d+m] in the ball geometry. An ARRAY index leaves the
    # write objective untouched (`scale.at[idx].set` / `q_jit.at[..., idx].set` are
    # index-agnostic); m=1 keeps the plain int so the shipped path is bit-identical.
    idx = d if m == 1 else jnp.arange(d, d + m)
    return dict(
        n_perturb=cfg.write_n_perturb,
        sigma_addr=cfg.write_sigma_addr,
        sigma_pay=cfg.write_sigma_pay,
        margin=cfg.write_margin,
        barrier=cfg.write_barrier,
        payload_index=idx,
    )


def write_learned(V, targets, cfg, key, d: int, mode: str = "local"):
    """Write ``targets`` into a learned atom dictionary.

    * ``"global"`` — one Adam run over ALL atoms, jointly on all targets (the w20
      write; interferes across items via the shared gradient step).
    * ``"local"``  — one masked single-item write per item; every OTHER item's
      atoms come out bit-identical (the MVC-0 C3-local write, the atom mechanism's
      best operator per ``potential-function-class``).
    """
    targets = jnp.asarray(targets)
    K = targets.shape[0]
    lk = _loss_kwargs(cfg, d)
    if mode == "global":
        V, hist = train_memory_landscape(
            V,
            targets,
            key,
            steps=cfg.write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=lk,
        )
        return V, hist
    if mode != "local":
        raise ValueError(f"unknown write mode {mode!r}")
    hist = []
    for i in range(K):
        key, k = jax.random.split(key)
        mask = V.learned.group_rows(i) if hasattr(V, "learned") else V.group_rows(i)
        V, h = train_memory_landscape(
            V,
            targets[i : i + 1],
            k,
            steps=cfg.local_write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=lk,
            update_mask_fn=atom_write_mask_fn(mask),
        )
        hist.extend(h)
    return V, hist


# ---------------------------------------------------------------------------
# Queries + two-phase retrieval, d-ball geometry
# ---------------------------------------------------------------------------


def make_ball_queries(key, centers, n_per_item: int, cfg):
    """Jittered addresses around each site; payload channel launched at 0 (guard).

    ``fixed_norm`` jitter (``sigma/sqrt(d)`` per axis) so the query NORM is
    ``query_sigma`` at every ``d`` — precision held constant, only the address
    dimension varies (the apples-to-apples generalization of the ring).

    ⭐ w26 fairness condition 3: with ``payload_launch_sigma > 0`` the payload
    channels launch at ``N(0, sigma)`` instead of *exactly* 0 — the anti-decoration
    guard is preserved (the launch distribution is item-independent and centred on
    the payload-zero manifold), but the value channel is no longer noise-free, which
    is what made shrinking the excursion free in w25.
    """
    K, d = centers.shape
    m = n_pay_channels(cfg)
    dim = d + m
    k_x, k_p, k_y = jax.random.split(key, 3)
    n = K * n_per_item
    labels = np.repeat(np.arange(K), n_per_item)
    scale = cfg.query_sigma / np.sqrt(d)
    x0 = jnp.repeat(jnp.asarray(centers), n_per_item, axis=0)
    x0 = x0 + jax.random.normal(k_x, (n, d)) * scale
    Q0 = jnp.zeros((n, dim)).at[:, :d].set(x0)
    s_launch = float(getattr(cfg, "payload_launch_sigma", 0.0))
    if s_launch > 0.0:
        Q0 = Q0.at[:, d : d + m].set(jax.random.normal(k_y, (n, m)) * s_launch)
    P0 = jnp.zeros((n, dim)).at[:, :d].set(
        jax.random.normal(k_p, (n, d)) * cfg.query_sigma_p
    )
    return Q0, P0, labels


# ---------------------------------------------------------------------------
# ⭐ w26 arm (b): the annealed / continuation read
# ---------------------------------------------------------------------------


def anneal_widths(cfg):
    """The read schedule ``[s_extra(0), ..., s_extra(L-1)]``, always ending at 0."""
    L = max(1, int(getattr(cfg, "read_anneal_stages", 1)))
    s0 = float(getattr(cfg, "read_anneal_s0", 0.0))
    if L == 1 or s0 <= 0.0:
        return [0.0]
    p = float(getattr(cfg, "read_anneal_power", 1.0))
    return [s0 * ((L - 1 - i) / (L - 1)) ** p for i in range(L)]


def inflate_potential(V, s_extra: float, mode: str = "amplitude"):
    """Blur a stored landscape: ``s_j -> sqrt(s_j^2 + s_extra^2)`` (Gaussian x Gaussian).

    ``mode="amplitude"`` keeps the well depth ``A_j`` (the force at the payload-zero
    launch manifold then *grows*, which is the reach lever); ``mode="mass"`` is the
    exact convolution, ``A_j *= (s_j/s_eff)^dim``, which preserves the integral of
    the well and lets the depth fall. Handles both the learned
    :class:`AtomDictionaryPotential` (inside a ``DesignFreedomPotential``) and the
    designed :class:`BallRegisterPotential`, so the baseline arm is annealed by the
    SAME operator (fairness condition 4).
    """
    if s_extra <= 0.0:
        return V
    if isinstance(V, BallRegisterPotential):
        w_eff = float(np.sqrt(V.w**2 + s_extra**2))
        b = float(V.b) * (
            (float(V.w) / w_eff) ** V.dim if mode == "mass" else 1.0
        )
        return BallRegisterPotential(
            V.payloads, V.centers, R=V.R, w=w_eff, b=b, kappa=V.kappa,
            c_conf=V.c_conf, dim=V.dim, spectator_k=V.spectator_k,
        )
    atoms = V.learned if hasattr(V, "learned") else V
    s = jnp.exp(atoms.log_width)
    s_eff = jnp.sqrt(s**2 + s_extra**2)
    new = eqx.tree_at(lambda a: a.log_width, atoms, replace=jnp.log(s_eff))
    if mode == "mass":
        dim = int(atoms.centers.shape[1])
        new = eqx.tree_at(
            lambda a: a.amp, new, replace=atoms.amp * (s / s_eff) ** (dim / 2.0)
        )
    elif mode != "amplitude":
        raise ValueError(f"read_anneal_mode must be amplitude|mass, got {mode!r}")
    if hasattr(V, "learned"):
        return eqx.tree_at(lambda p: p.learned, V, replace=new)
    return new


def anneal_stage_models(model, cfg):
    """``[model_0, ..., model_{L-1}]`` — the same CLU with progressively sharper wells."""
    mode = str(getattr(cfg, "read_anneal_mode", "amplitude"))
    out = []
    for s_extra in anneal_widths(cfg):
        V = inflate_potential(model.potential_net, s_extra, mode=mode)
        out.append(eqx.tree_at(lambda m: m.potential_net, model, replace=V))
    return out


def _two_phase(model, Q0, P0, cfg, d: int, masses=None, stage_models=None):
    """query -> [gamma_address relax] -> address -> [gamma_read rollout] -> traj.

    Returns ``(addr_x (n,d), payload_tail (n, n_subsample, m))``, chunked so the jit
    compiles once and the full trajectory is never materialized outside it.
    ``masses`` (n, dim) supplies a per-query inertial mass (the address-side key);
    ``None`` uses identity mass.

    ⭐ w26 arm (b): ``stage_models`` (from :func:`anneal_stage_models`) runs the read
    as ``L`` consecutive segments on progressively sharper landscapes.
    ``address_steps`` and ``read_steps`` are **split** across the segments, so the
    annealed read integrates exactly as many Verlet steps as the baseline (equal
    compute by construction), and the value tail is always sampled inside the FINAL
    segment, whose landscape is the stored one (``s_extra = 0``).
    """
    m_pay = n_pay_channels(cfg)
    dim = d + m_pay
    stages = list(stage_models) if stage_models else [model]
    L = len(stages)
    anneal_addr = str(getattr(cfg, "read_anneal_phases", "both")) == "both"
    a_steps = max(1, cfg.address_steps // L) if anneal_addr else cfg.address_steps
    addr_stages = stages if anneal_addr else [stages[-1]]
    r_steps = max(1, cfg.read_steps // L)
    start = int((1.0 - cfg.tail_frac) * r_steps)
    tail_idx = jnp.asarray(np.linspace(start, r_steps - 1, cfg.n_subsample).astype(int))
    ones = jnp.ones(dim)

    @eqx.filter_jit
    def _addr(mdl, q, p, m):
        def one(q, p, m):
            tr = mdl(q, p, a_steps, cfg.dt, cfg.gamma_address, m)
            return tr[-1, :dim], tr[-1, dim:]

        return jax.vmap(one)(q, p, m)

    @eqx.filter_jit
    def _read(mdl, q, p, m):
        def one(q, p, m):
            tr = mdl(q, p, r_steps, cfg.dt, cfg.gamma_read, m)
            return tr[-1, :dim], tr[-1, dim:], tr[tail_idx, d : d + m_pay]

        return jax.vmap(one)(q, p, m)

    def run(q, p, m):
        for mdl in addr_stages:
            q, p = _addr(mdl, q, p, m)
        f = None
        for mdl in stages:
            q, p, f = _read(mdl, q, p, m)
        return q[:, :d], f

    n = Q0.shape[0]
    chunk = min(cfg.rollout_chunk, n)
    xs, feats = [], []
    for i in range(0, n, chunk):
        q, p = Q0[i : i + chunk], P0[i : i + chunk]
        m = ones[None, :].repeat(q.shape[0], axis=0) if masses is None else masses[i : i + chunk]
        pad = chunk - q.shape[0]
        if pad > 0:
            q = jnp.concatenate([q, jnp.zeros((pad,) + q.shape[1:])], axis=0)
            p = jnp.concatenate([p, jnp.zeros((pad,) + p.shape[1:])], axis=0)
            m = jnp.concatenate([m, jnp.ones((pad,) + m.shape[1:])], axis=0)
        x, f = run(q, p, m)
        if pad > 0:
            x, f = x[: chunk - pad], f[: chunk - pad]
        xs.append(np.asarray(x))
        feats.append(np.asarray(f))
    return np.concatenate(xs, axis=0), np.concatenate(feats, axis=0)


def score_cell(model, centers, payloads, cfg, d: int, seed: int, masses_fn=None,
               stage_models=None):
    """Run the loop on ONE landscape and score it (the unit of measurement).

    ``masses_fn(labels) -> (n, dim)`` optionally supplies a per-query mass keyed by
    the query's item label (the per-item mass arm); ``None`` = identity mass.
    ``stage_models`` runs the w26 annealed read (:func:`anneal_stage_models`).

    Returns strict/basin/payload/selectivity + the classification reads (for the
    blank control). Under ``pass_metric="decode"`` the headline ``strict`` is
    nearest-CODEWORD decoding rather than the absolute ``payload_tol`` test; both are
    always reported (``strict_tol`` / ``strict_decode``).
    """
    K = len(payloads)
    m_pay = n_pay_channels(cfg)
    dim = d + m_pay
    key = jax.random.PRNGKey(seed)
    n_per = int(
        np.clip(cfg.max_total_queries // K, cfg.min_query_per_item, cfg.n_query_per_item)
    )
    Q0, P0, labels = make_ball_queries(key, centers, n_per, cfg)
    masses = None
    if masses_fn is not None:
        masses = masses_fn(labels, dim)

    addr_x, feat = _two_phase(
        model, Q0, P0, cfg, d, masses=masses, stage_models=stage_models
    )
    finite = bool(np.all(np.isfinite(addr_x)) and np.all(np.isfinite(feat)))

    c = np.asarray(centers)
    d2 = ((addr_x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    basin = np.argmin(d2, axis=1)
    basin_ok = basin == labels

    read_val = feat.mean(axis=1)  # (n, m)
    pay = np.asarray(payloads)
    if pay.ndim == 1:
        pay = pay[:, None]
    # ⭐ w26 condition 3: observation noise on the read-out value. The store cannot
    # denoise this one, so it is what makes a codebook spacing (hence an excursion)
    # cost something.
    s_obs = float(getattr(cfg, "payload_obs_sigma", 0.0))
    if s_obs > 0.0:
        rng_obs = np.random.default_rng(seed + 104729)
        read_val = read_val + rng_obs.normal(scale=s_obs, size=read_val.shape)
    err = np.linalg.norm(read_val - pay[labels], axis=-1)
    strict_tol = basin_ok & (err < cfg.payload_tol)
    # nearest-codeword decode (blind to the item index: it only sees the value)
    dec = np.argmin(
        ((read_val[:, None, :] - pay[None, :, :]) ** 2).sum(-1), axis=1
    )
    decode_ok = np.all(np.isclose(pay[dec], pay[labels]), axis=-1)
    strict_dec = basin_ok & decode_ok
    metric = str(getattr(cfg, "pass_metric", "tol"))
    strict_ok = strict_dec if metric == "decode" else strict_tol

    flat = feat.reshape(feat.shape[0], -1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.shape[0])
    half = flat.shape[0] // 2
    tr, te = perm[:half], perm[half:]
    acc_nc = nearest_centroid_read(flat[tr], labels[tr], flat[te], labels[te], K)
    if m_pay == 1:
        acc_cb, _ = linear_codebook_read(
            flat[tr], labels[tr], flat[te], labels[te], pay[:, 0]
        )
    else:
        # the scalar-codebook linear read is undefined for a vector code; the
        # nearest-centroid read (also linear) carries the classification blank.
        acc_cb = acc_nc

    return {
        "K": K,
        "finite": finite,
        "basin_success_rate": float(np.mean(basin_ok)),
        "strict_success_rate": float(np.mean(strict_ok)),
        "strict_tol": float(np.mean(strict_tol)),
        "strict_decode": float(np.mean(strict_dec)),
        "decode_rate": float(np.mean(decode_ok)),
        "selectivity": float(np.mean(basin_ok)),
        "payload_abs_err_mean": float(np.mean(err)),
        "acc_payload_codebook_read": float(acc_cb),
        "acc_payload_nearest_centroid": float(acc_nc),
        "chance": 1.0 / K,
        "n_queries": int(flat.shape[0]),
    }


# ---------------------------------------------------------------------------
# One (arm, d, K, seed) cell: written + blank + pass/fail
# ---------------------------------------------------------------------------


def evaluate_arm_cell(arm: str, d: int, K: int, seed: int, cfg):
    """Train (learned) or build (designed) the WRITTEN + BLANK landscape and score.

    Value criterion (leak-immune): mean strict >= pass_strict AND the value blank
    (blank strict <= blank_strict_max) passes. A learned V couples the payload
    channel to the address, so a *classification* read leaks the address on a blank
    landscape — only the value-recovery strict metric is used to gate.
    """
    centers, payloads, targets, sep = ball_setup(d, K, cfg)
    blank_pay = jnp.zeros_like(payloads)
    _, _, blank_targets, _ = ball_setup(d, K, cfg, payloads=blank_pay)
    n_params = 0
    t0 = time.perf_counter()

    dim = d + n_pay_channels(cfg)
    if arm == "designed":
        mw = build_designed_model(centers, payloads, cfg)
        mb = build_designed_model(centers, blank_pay, cfg)
    elif arm in ("learned_local", "learned_global"):
        mode = "local" if arm == "learned_local" else "global"
        k_w, k_b = jax.random.split(jax.random.PRNGKey(seed + 7919), 2)
        Vw = build_learned_V(d, K, cfg, k_w, centers=centers)
        Vw, hist = write_learned(Vw, targets, cfg, k_w, d, mode=mode)
        Vb = build_learned_V(d, K, cfg, k_b, centers=centers)
        Vb, _ = write_learned(Vb, blank_targets, cfg, k_b, d, mode=mode)
        n_params = _n_params(Vw)
        mw = clu_with_potential(
            Vw, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
        )
        mb = clu_with_potential(
            Vb, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    write_seconds = time.perf_counter() - t0

    sw = anneal_stage_models(mw, cfg) if len(anneal_widths(cfg)) > 1 else None
    sb = anneal_stage_models(mb, cfg) if len(anneal_widths(cfg)) > 1 else None
    written = score_cell(mw, centers, payloads, cfg, d, seed, stage_models=sw)
    blank = score_cell(mb, centers, payloads, cfg, d, seed, stage_models=sb)
    # ⚠ A blank landscape returns ~0 for every item, so it LEGITIMATELY "retrieves"
    # any item whose real payload happens to lie within payload_tol of 0. At large K
    # the linspace(-1,1,K) codebook has several such near-zero values, so a valid
    # blank scores up to this trivial ceiling on strict — gating at a flat
    # blank_strict_max would spuriously disqualify the DESIGNED arm at large K.
    pay_np = np.asarray(payloads)
    if pay_np.ndim == 1:
        pay_np = pay_np[:, None]
    if str(getattr(cfg, "pass_metric", "tol")) == "decode":
        # a blank landscape reads ~0 for every item, so it trivially "decodes" every
        # item whose codeword is the one nearest the origin
        near0 = int(np.argmin((pay_np**2).sum(-1)))
        trivial_ceiling = float(
            np.mean(np.all(np.isclose(pay_np, pay_np[near0]), axis=-1))
        )
    else:
        trivial_ceiling = float(np.mean(np.linalg.norm(pay_np, axis=-1) < cfg.payload_tol))
    blank_ceiling = max(cfg.blank_strict_max, trivial_ceiling + 0.02)
    value_blank_ok = bool(blank["strict_success_rate"] <= blank_ceiling)
    class_blank = max(
        blank["acc_payload_codebook_read"], blank["acc_payload_nearest_centroid"]
    )
    class_blank_ok = bool(class_blank <= blank["chance"] + cfg.blank_margin)
    return {
        "arm": arm,
        "d": d,
        "K": K,
        "seed": seed,
        "site_sep": float(sep),
        "n_learned_params": n_params,
        "param_bits_budget": int(n_params * cfg.bits_per_param),
        "written": written,
        "blank": blank,
        "value_blank_ok": value_blank_ok,
        "classification_blank_ok": class_blank_ok,
        "write_seconds": float(write_seconds),
    }


def _agg(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def k_star_for_arm(arm: str, d: int, cfg, seeds, verbose=True):
    """Walk the K ladder; K_star = largest K clearing strict at every seed's blank.

    A cell passes iff mean strict over seeds >= pass_strict AND the value blank
    passes for EVERY seed (a leaking cell is not a measurement). Stops at the first
    failing K. A cell that runs off the ladder / cap without failing is CENSORED
    (K_star a lower bound).
    """
    ladder = [K for K in cfg.k_ladder if K <= cfg.k_cap]
    per_K, censored = [], False
    for K in ladder:
        cells = [evaluate_arm_cell(arm, d, K, s, cfg) for s in seeds]
        strict = [c["written"]["strict_success_rate"] for c in cells]
        blanks_ok = all(c["value_blank_ok"] for c in cells)
        mean_strict = float(np.mean(strict))
        passes = bool(mean_strict >= cfg.pass_strict and blanks_ok)
        rec = {
            "K": K,
            "strict": _agg(strict),
            "selectivity": _agg([c["written"]["selectivity"] for c in cells]),
            "payload_abs_err": _agg(
                [c["written"]["payload_abs_err_mean"] for c in cells]
            ),
            "n_value_blank_pass": int(sum(c["value_blank_ok"] for c in cells)),
            "blank_strict": _agg([c["blank"]["strict_success_rate"] for c in cells]),
            "n_learned_params": cells[0]["n_learned_params"],
            "site_sep": cells[0]["site_sep"],
            "passes": passes,
            "write_seconds": _agg([c["write_seconds"] for c in cells]),
        }
        per_K.append(rec)
        if verbose:
            print(
                f"  [{arm}] d={d} K={K:5d} strict={mean_strict:.3f}"
                f" blankOK={blanks_ok} P={rec['n_learned_params']}"
                f" sep={rec['site_sep']:.3f} -> {'PASS' if passes else 'fail'}",
                flush=True,
            )
        if not passes:
            break
    else:
        censored = True

    passing = [r["K"] for r in per_K if r["passes"]]
    k_star = max(passing, default=0)
    k_top = max(r["K"] for r in per_K)
    return {
        "arm": arm,
        "d": d,
        "k_star": k_star,
        "censored": bool(censored and k_star == k_top),
        "per_K": per_K,
        "seeds": list(seeds),
    }


def _fit_growth(ds, ks, censored=None):
    """Exponential (A^d) vs polynomial (d^alpha) fit; censored points excluded."""
    ds = np.asarray(ds, float)
    ks = np.asarray(ks, float)
    ok = ks > 1
    fit = {}
    if censored is not None:
        ok = ok & ~np.asarray(censored, dtype=bool)
        fit["n_censored_excluded"] = int(np.sum(censored))
    if ok.sum() >= 2:
        A = np.stack([ds[ok], np.ones(ok.sum())], axis=1)
        coef, *_ = np.linalg.lstsq(A, np.log(ks[ok]), rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((np.log(ks[ok]) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ks[ok]) - np.log(ks[ok]).mean()) ** 2))
        fit["exponential_base_A"] = float(np.exp(coef[0]))
        fit["exponential_intercept"] = float(coef[1])
        fit["exponential_r2"] = 1.0 - ss_res / (ss_tot + 1e-12)
        B = np.stack([np.log(ds[ok]), np.ones(ok.sum())], axis=1)
        cp, *_ = np.linalg.lstsq(B, np.log(ks[ok]), rcond=None)
        rp = float(np.sum((np.log(ks[ok]) - B @ cp) ** 2))
        fit["polynomial_exponent_alpha"] = float(cp[0])
        fit["polynomial_r2"] = 1.0 - rp / (ss_tot + 1e-12)
        fit["exponential_beats_polynomial"] = bool(ss_res < rp)
        fit["n_points_fitted"] = int(ok.sum())
    return fit


# ---------------------------------------------------------------------------
# Item 1 -- ⭐ the discriminator: K_learned vs d, K_designed overlaid
# ---------------------------------------------------------------------------


def item1_discriminator(cfg):
    """K_learned vs d and K_designed vs d on one axis, with the fitted growth."""
    learned_rows, designed_rows = [], []
    for d in cfg.dims:
        designed_rows.append(
            k_star_for_arm("designed", d, cfg, cfg.designed_seeds)
        )
        learned_rows.append(
            k_star_for_arm(cfg.learned_arm, d, cfg, cfg.discriminator_seeds)
        )

    def _fit(rows):
        return _fit_growth(
            [r["d"] for r in rows],
            [max(r["k_star"], 1) for r in rows],
            [r["censored"] for r in rows],
        )

    fit_l = _fit(learned_rows)
    fit_d = _fit(designed_rows)

    # ratio K_learned / K_designed per d (falls under H-GEOMETRY-WEAK, ~const under
    # H-GEOMETRY-STRONG, and collapses hardest under H-LEARNING).
    ratios = []
    for lr, dr in zip(learned_rows, designed_rows, strict=True):
        kd = dr["k_star"]
        ratios.append(
            {
                "d": lr["d"],
                "k_learned": lr["k_star"],
                "k_designed": kd,
                "k_designed_4x2d": int(4 * 2**lr["d"]),
                "ratio_learned_over_designed": (
                    float(lr["k_star"] / kd) if kd > 0 else None
                ),
                "learned_censored": lr["censored"],
                "designed_censored": dr["censored"],
            }
        )

    # Verdict per the pre-registered decision rule.
    A = fit_l.get("exponential_base_A", 1.0)
    kl = [r["k_star"] for r in learned_rows]
    grows = bool(max(kl) - min(kl) >= 4)  # >= +2 ladder rungs over the sweep
    if A >= 1.9:
        verdict = "H-GEOMETRY-STRONG"
    elif A >= 1.3 and fit_l.get("exponential_r2", 0) >= 0.8 and grows:
        verdict = "H-GEOMETRY-WEAK"
    elif A < 1.3 and not grows:
        verdict = "H-LEARNING"
    else:
        verdict = "AMBIGUOUS"

    return {
        "dims": list(cfg.dims),
        "learned_arm": cfg.learned_arm,
        "pass_strict": cfg.pass_strict,
        "discriminator_seeds": list(cfg.discriminator_seeds),
        "designed_seeds": list(cfg.designed_seeds),
        "atoms_per_item": cfg.atoms_per_item,
        "k_learned_vs_d": [
            {"d": r["d"], "k_star": r["k_star"], "censored": r["censored"]}
            for r in learned_rows
        ],
        "k_designed_vs_d": [
            {"d": r["d"], "k_star": r["k_star"], "censored": r["censored"]}
            for r in designed_rows
        ],
        "ratios": ratios,
        "fit_k_learned": fit_l,
        "fit_k_designed": fit_d,
        "verdict": verdict,
        "learned_detail": learned_rows,
        "designed_detail": designed_rows,
    }


# ---------------------------------------------------------------------------
# Item 2 -- does per-item MASS help? (folds in mass-visible-objective + Prop F1)
# ---------------------------------------------------------------------------


def _hessian_coupling(V, centers, d: int):
    """Address-coupling ratio mean|off-diag| / mean|diag| of Hess V at stored sites.

    Prop F1 (relaxation-fiber-capacity): mass is address-side and worth ~0 bits in a
    SEPARABLE well (``∂_i∂_j V = 0``). An isotropic Gaussian atom has a diagonal
    Hessian at its own center, so this ratio being ~0 PREDICTS a mass null.
    Evaluated on the address block ``[:d, :d]`` at each stored site.
    """
    H = eqx.filter_jit(jax.hessian(lambda q: V(q)))
    ratios = []
    for c in np.asarray(centers):
        q = np.zeros(d + 1, dtype=np.float32)
        q[:d] = c
        h = np.asarray(H(jnp.asarray(q)))[:d, :d]
        diag = np.abs(np.diag(h))
        off = np.abs(h - np.diag(np.diag(h)))
        md = float(np.mean(diag)) if diag.size else 0.0
        mo = float(np.sum(off) / max(off.size - d, 1))
        ratios.append(mo / (md + 1e-12))
    return float(np.mean(ratios))


def item2_mass(cfg):
    """Uniform mass (a) vs per-item mass spread (b), at fixed d, WITH the coupling
    check. Per Prop F1, expect ~0 gain unless the atom wells couple coordinates.

    ⚠ Honesty: the write objective is mass-BLIND by construction (kinetic terms
    cancel in a static minimum-digging loss), so the per-item masses are ASSIGNED
    (a geometric spread), not gradient-learned. Whether *any* mass value changes
    retrieval is the load-bearing test — a separable well is mass-invariant at its
    fixed point, so the coupling ratio determines the ceiling on what mass can buy.
    """
    d, K = cfg.mass_dim, cfg.mass_K
    out = []
    for seed in cfg.mass_seeds:
        centers, payloads, targets, _ = ball_setup(d, K, cfg)
        k_w = jax.random.PRNGKey(seed + 111)
        V = build_learned_V(d, K, cfg, k_w)
        V, _ = write_learned(V, targets, cfg, k_w, d, mode="local")
        dim = d + n_pay_channels(cfg)
        model = clu_with_potential(
            V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
        )
        coupling = _hessian_coupling(V, centers, d)

        # arm (a): uniform mass
        a = score_cell(model, centers, payloads, cfg, d, seed, masses_fn=None)

        # arm (b): per-item geometric mass spread, keyed by the query's item label
        spread = np.geomspace(1.0 / cfg.mass_spread, cfg.mass_spread, K).astype(
            np.float32
        )

        def masses_fn(labels, dim, spread=spread):
            m = np.ones((len(labels), dim), dtype=np.float32)
            m[:, :] = spread[labels][:, None]
            return jnp.asarray(m)

        b = score_cell(model, centers, payloads, cfg, d, seed, masses_fn=masses_fn)
        out.append(
            {
                "seed": seed,
                "coupling_ratio_offdiag_over_diag": coupling,
                "uniform_strict": a["strict_success_rate"],
                "permass_strict": b["strict_success_rate"],
                "uniform_payload_err": a["payload_abs_err_mean"],
                "permass_payload_err": b["payload_abs_err_mean"],
                "delta_strict": b["strict_success_rate"] - a["strict_success_rate"],
            }
        )
    return {
        "d": d,
        "K": K,
        "mass_spread": cfg.mass_spread,
        "seeds": list(cfg.mass_seeds),
        "note": "masses ASSIGNED (write objective is mass-blind); coupling ratio "
        "bounds what mass can buy (Prop F1).",
        "rows": out,
        "coupling_ratio": _agg([r["coupling_ratio_offdiag_over_diag"] for r in out]),
        "delta_strict": _agg([r["delta_strict"] for r in out]),
        "mass_helps": bool(
            np.mean([r["delta_strict"] for r in out]) > cfg.mass_help_threshold
        ),
    }


# ---------------------------------------------------------------------------
# Item 3 -- interference across d: masked vs global write
# ---------------------------------------------------------------------------


def _corruption(arm_mode: str, d: int, seed: int, cfg):
    """Write A (K-1 items), read A; write B into a fresh site; re-read A.

    ``arm_mode`` in {"local", "global"}; corruption = change in A's read error.
    """
    K = cfg.interference_K
    centers, payloads, targets, _ = ball_setup(d, K, cfg)
    pay_A = payloads[: K - 1]
    c_A = centers[: K - 1]
    k_w, k_a, k_b = jax.random.split(jax.random.PRNGKey(seed + 333), 3)
    V = build_learned_V(d, K, cfg, k_w)

    # write items 0..K-2
    if arm_mode == "global":
        V, _ = train_memory_landscape(
            V, targets[: K - 1], k_a, steps=cfg.write_steps, lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay, loss_kwargs=_loss_kwargs(cfg, d),
        )
    else:
        for i in range(K - 1):
            k_a, kk = jax.random.split(k_a)
            V, _ = train_memory_landscape(
                V, targets[i : i + 1], kk, steps=cfg.local_write_steps, lr=cfg.write_lr,
                weight_decay=cfg.write_weight_decay, loss_kwargs=_loss_kwargs(cfg, d),
                update_mask_fn=atom_write_mask_fn(V.learned.group_rows(i)),
            )
    V_A = V
    dim = d + n_pay_channels(cfg)
    m_A = clu_with_potential(
        V_A, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )
    before = score_cell(m_A, c_A, pay_A, cfg, d, seed)

    # write item K-1 (B) into its own site/block
    if arm_mode == "global":
        V_B, _ = train_memory_landscape(
            V_A, targets[K - 1 : K], k_b, steps=cfg.interference_write_steps,
            lr=cfg.write_lr, weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg, d),
        )
    else:
        V_B, _ = train_memory_landscape(
            V_A, targets[K - 1 : K], k_b, steps=cfg.interference_write_steps,
            lr=cfg.write_lr, weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg, d),
            update_mask_fn=atom_write_mask_fn(V_A.learned.group_rows(K - 1)),
        )
    m_B = clu_with_potential(
        V_B, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )
    after = score_cell(m_B, c_A, pay_A, cfg, d, seed)

    # bit-level check: did the mask freeze the other atoms?
    la = jax.tree_util.tree_leaves(eqx.filter(V_A, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(V_B, eqx.is_inexact_array))
    delt = [np.asarray(y) - np.asarray(x) for x, y in zip(la, lb, strict=False)]
    n = int(sum(x.size for x in delt))
    nz = int(sum(int(np.count_nonzero(x)) for x in delt))
    return {
        "mode": arm_mode,
        "d": d,
        "seed": seed,
        "read_err_A_before": before["payload_abs_err_mean"],
        "read_err_A_after": after["payload_abs_err_mean"],
        "corruption": abs(
            after["payload_abs_err_mean"] - before["payload_abs_err_mean"]
        ),
        "strict_A_before": before["strict_success_rate"],
        "strict_A_after": after["strict_success_rate"],
        "frac_params_moved_by_B": float(nz / n) if n else 0.0,
    }


def item3_interference(cfg):
    """Masked vs global cross-write corruption at each d (does the write-operator
    advantage survive higher dimensions?)."""
    rows, summary = [], []
    for d in cfg.interference_dims:
        for mode in ("local", "global"):
            got = [_corruption(mode, d, s, cfg) for s in cfg.interference_seeds]
            rows.extend(got)
            summary.append(
                {
                    "d": d,
                    "mode": mode,
                    "corruption": _agg([g["corruption"] for g in got]),
                    "frac_params_moved_by_B": _agg(
                        [g["frac_params_moved_by_B"] for g in got]
                    ),
                    "strict_A_after": _agg([g["strict_A_after"] for g in got]),
                }
            )
    # local-advantage ratio per d
    adv = []
    for d in cfg.interference_dims:
        loc = next(s for s in summary if s["d"] == d and s["mode"] == "local")
        glo = next(s for s in summary if s["d"] == d and s["mode"] == "global")
        lc = max(loc["corruption"]["mean"], 1e-12)
        adv.append(
            {
                "d": d,
                "local_corruption": loc["corruption"]["mean"],
                "global_corruption": glo["corruption"]["mean"],
                "local_advantage_ratio": glo["corruption"]["mean"] / lc,
            }
        )
    return {
        "dims": list(cfg.interference_dims),
        "interference_K": cfg.interference_K,
        "seeds": list(cfg.interference_seeds),
        "summary": summary,
        "local_advantage_by_d": adv,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Item 4 -- the honest performance frontier
# ---------------------------------------------------------------------------


def item4_frontier(item1):
    """Where does learned content MATCH the designed ceiling, and where fall away?

    Derived from item 1: the (d, K) grid of learned strict, with the K where learned
    == designed and the largest K where learned still clears the bar (= K_learned).
    """
    frontier = []
    for lr in item1["learned_detail"]:
        d = lr["d"]
        dr = next(r for r in item1["designed_detail"] if r["d"] == d)
        frontier.append(
            {
                "d": d,
                "k_learned": lr["k_star"],
                "k_designed": dr["k_star"],
                "learned_strict_by_K": [
                    {"K": r["K"], "strict": r["strict"]["mean"], "passes": r["passes"]}
                    for r in lr["per_K"]
                ],
                "matches_designed_up_to_K": (
                    lr["k_star"] if lr["k_star"] >= dr["k_star"] else lr["k_star"]
                ),
                "falls_away_at_K": next(
                    (r["K"] for r in lr["per_K"] if not r["passes"]), None
                ),
            }
        )
    best = max(frontier, key=lambda f: f["k_learned"], default=None)
    return {
        "frontier": frontier,
        "best_learned_cell": (
            {"d": best["d"], "k_learned": best["k_learned"]} if best else None
        ),
    }


# ---------------------------------------------------------------------------
# Figure (local, per the exp_dim_scaling / exp_potential_class precedent)
# ---------------------------------------------------------------------------


def _plot(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    it1 = results.get("item1_discriminator")
    if not it1:
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    dl = [r["d"] for r in it1["k_learned_vs_d"]]
    kl = [max(r["k_star"], 0.5) for r in it1["k_learned_vs_d"]]
    kd = [max(r["k_star"], 0.5) for r in it1["k_designed_vs_d"]]
    a1.semilogy(dl, kl, "o-", lw=2, label="$K_{learned}$ (atom dict, trained)")
    a1.semilogy(dl, kd, "s-", lw=2, label="$K_{designed}$ (re-measured)")
    a1.semilogy(dl, [4 * 2**x for x in dl], "k--", alpha=0.5, label=r"$4\cdot2^d$")
    fit = it1.get("fit_k_learned", {})
    if fit.get("exponential_base_A"):
        A, b = fit["exponential_base_A"], fit["exponential_intercept"]
        xs = np.linspace(min(dl), max(dl), 40)
        a1.semilogy(
            xs, np.exp(b) * A**xs, "-", color="C0", alpha=0.4,
            label=f"fit $A^d$, A={A:.2f} ($R^2$={fit.get('exponential_r2', 0):.2f})",
        )
    a1.axhline(8, color="r", ls=":", lw=0.9, label="ring ceiling (~8)")
    a1.set_xlabel("address dimension $d$")
    a1.set_ylabel("$K$ cleared at strict 0.9")
    a1.set_title(f"Discriminator: {it1['verdict']}")
    a1.legend(fontsize=7)

    it3 = results.get("item3_interference")
    if it3:
        ds = [a["d"] for a in it3["local_advantage_by_d"]]
        a2.semilogy(
            ds, [max(a["global_corruption"], 1e-9) for a in it3["local_advantage_by_d"]],
            "o-", label="global write",
        )
        a2.semilogy(
            ds, [max(a["local_corruption"], 1e-9) for a in it3["local_advantage_by_d"]],
            "s-", label="masked (local) write",
        )
        a2.set_xlabel("address dimension $d$")
        a2.set_ylabel("corruption of A by writing B")
        a2.set_title("Interference: write operator across $d$")
        a2.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(save_dir, "designed_mechanism_fig1.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_designed_mechanism(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_designed_mechanism
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "R", "wall_margin", "well_width", "well_depth", "payload_kappa",
                "c_conf", "site_seed", "payload_seed", "dt", "gamma_address",
                "gamma_read", "address_steps", "read_steps", "tail_frac",
                "n_subsample", "n_query_per_item", "query_sigma", "query_sigma_p",
                "payload_tol", "pass_strict", "blank_strict_max", "blank_margin",
                "atoms_per_item", "min_atoms", "min_atoms_base", "min_atoms_c",
                "atom_init_scale", "atom_init_width",
                "atom_depth_init", "learned_confine", "bits_per_param",
                "atom_init_local", "atom_init_local_mult", "n_payload_channels",
                "payload_code", "payload_launch_sigma", "payload_obs_sigma",
                "pass_metric", "read_anneal_stages", "read_anneal_s0",
                "read_anneal_power", "read_anneal_mode", "read_anneal_phases",
                "write_steps", "local_write_steps", "write_lr", "write_n_perturb",
                "write_sigma_addr", "write_sigma_pay", "write_margin",
                "write_barrier", "dims", "k_ladder", "k_cap", "learned_arm",
            )
        },
    }
    print("[item 1] discriminator: K_learned vs d", flush=True)
    results["item1_discriminator"] = item1_discriminator(cfg)
    print("[item 3] interference across d", flush=True)
    results["item3_interference"] = item3_interference(cfg)
    print("[item 2] mass arm + coupling check", flush=True)
    results["item2_mass"] = item2_mass(cfg)
    results["item4_frontier"] = item4_frontier(results["item1_discriminator"])

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_designed_mechanism_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover
        results["figures"] = []
        results["figure_error"] = repr(exc)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke: same code path, tiny sweeps. NOT shorter on the rollout budget
    than the designed reference needs (a smoke run where the reference fails prints
    a fake scientific negative — w20 lesson)."""
    cfg = config.experiment_designed_mechanism
    cfg.dims = [2, 3]
    cfg.k_ladder = [2, 4, 8]
    cfg.k_cap = 8
    cfg.discriminator_seeds = [0, 1]
    cfg.designed_seeds = [0]
    cfg.atoms_per_item = 8
    # keep the dimension-aware floor tiny for the smoke path (dims=[2,3])
    cfg.min_atoms = 32
    cfg.min_atoms_base = 16
    cfg.min_atoms_c = 2.0
    cfg.address_steps = 300
    cfg.read_steps = 200
    cfg.write_steps = 60
    cfg.local_write_steps = 40
    cfg.write_n_perturb = 8
    cfg.n_query_per_item = 8
    cfg.interference_dims = [2]
    cfg.interference_seeds = [0]
    cfg.interference_write_steps = 30
    cfg.mass_dim = 2
    cfg.mass_K = 4
    cfg.mass_seeds = [0]


def _replace(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment DESIGNED-MECHANISM-LEARNED-CONTENT: is the K=8 wall "
        "geometry or learning?"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
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

    res = run_experiment_designed_mechanism(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(json.dumps(res["item1_discriminator"]["k_learned_vs_d"], indent=2))
    print(json.dumps(res["item1_discriminator"]["k_designed_vs_d"], indent=2))
    print("verdict:", res["item1_discriminator"]["verdict"])
    print("metrics ->", res["metrics_path"])


if __name__ == "__main__":
    main()
