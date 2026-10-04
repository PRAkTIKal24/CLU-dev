"""C2W2 `phi-particle-head` — the read-in that parametrizes the particle.

Charter Addendum-1 **§A4.3** (phi policy + the ``(d, atom-budget)`` joint dial)
and **§A2.2** (*"Prop F1's mass-gauge dissolves under trajectory reads — 'mass as
selector' becomes live for the first time"*), which has never been tested.

Three parts, run by module invocation (this wave's CLI file belongs to
``traj-write-objective``; **no CLI hook is added**)::

    PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part phi
    PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part grad --seed 0
    PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part monitors

``--part phi``
    **D1/D2.** Every phi family built, byte-ledgered (``phi_id``/``phi_bytes``
    for the race card), the **identical-phi invariant** exercised in both
    directions (it must RAISE on a mismatch), the ``(d, n_atoms)`` joint dial
    printed, the particle head's **default-off bit-identity** checked against the
    shipped launch, one **strong encoder (CNN) smoke-run end to end**, and the
    declared friction band probed for where it sits against monitor #1's edge.

``--part grad``
    ⭐ **D3, the one real experiment.** ``||dL/dlog_mass||`` and
    ``||dL/dfriction||`` for a settled-point psi and a trajectory psi, matched
    parameters (:func:`~chlu.core.psi_readout.matched_pair`), on the real store.
    Prop Q1.1 says the fixed-point set contains no ``M`` and no ``gamma``, so the
    point arm must be zero (**exactly**, by the implicit path) and the trajectory
    arm must not. If the trajectory arm is numerically zero too, **§A2.2 is
    refuted**. Plus the full gradcheck (implicit vs unroll vs finite differences)
    and the wall-clock against the 30 s budget.

``--part monitors``
    **D4 acceptance.** Re-runs the C2W1 shipped anchor (``overload/load1x_shipped``,
    the 478x cell) on three seeds and diffs the monitor trip states **before vs
    after** the repairs — pre-repair states are re-scored from the *same*
    readings (every monitor reading carries ``tripped_pre_repair``), so the diff
    is exact rather than a second stochastic run, and is then checked against the
    C2W1 artifact on disk.

⚠ **Truncation direction is load-bearing** (spike R-2): tail truncation enters
through a ``stop_gradient``, so every particle/phi gradient here is taken at
``retain=None`` (full backprop). A truncated gradient would be exactly 0 and
would look like a refutation.
⚠ **Monitors are guards, never losses** — nothing here differentiates a monitor.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np

OUT_DIR = os.environ.get(
    "PHI_PARTICLE_OUT",
    os.path.join(os.path.expanduser("~"), "Desktop", "CHLU", ".claude", "outputs",
                 "phi-particle-head"),
)

#: the C2W1 anchor whose artefacts are on disk (`memory-gym-v0`, the 478x cell)
ANCHOR_FAMILY, ANCHOR_ARM = "overload", "load1x_shipped"
ANCHOR_ARTIFACT = os.path.join(os.path.expanduser("~"), "Desktop", "CHLU", ".claude",
                               "outputs", "memory-gym-v0", "exp_memory_gym_metrics.json")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _save(name: str, payload: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    return path


def _bind(loss_of, psi_):
    """``(log_mass, friction) -> loss`` with ``psi`` bound (no loop-variable capture)."""
    def f(lm, gf):
        return loss_of(psi_, lm, gf)
    return f


def _tree_norm(tree) -> float:
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    if not leaves:
        return 0.0
    return float(np.sqrt(sum(float(np.sum(np.asarray(x) ** 2)) for x in leaves)))


# ==========================================================================
# the particle read — the shipped two-phase read with a PER-PARTICLE (M, gamma)
# ==========================================================================
def particle_read(model, launch, cfg, *, retain1=None, retain2=None,
                  stride: Optional[int] = None, use_particle: Optional[bool] = None,
                  dtype=None):
    """``CluSystem.read``'s two phases with the particle attributes phi emitted.

    Identical dynamics to ``exp_trajectory_read.differentiable_read`` (which is
    verified against the frozen ``CluSystem.read`` to float32 round-off) with two
    differences, both from charter §A4.3:

    * ``mass_override = softplus(launch.log_mass)`` per query — the shipped
      ``CHLU.__call__`` already carries this argument (w20 Prop 6), so nothing is
      forked;
    * ``gamma`` per query. Phase 1 runs at ``launch.friction``; phase 2 runs at
      ``launch.friction * (gamma_read / gamma_address)``, i.e. the **shipped
      ratio is preserved**, so at ``friction = gamma_address`` the read is the
      shipped read exactly. (Phase 2 REQUIRES dissipation — payload error 0.57 at
      ``gamma_read = 0`` — so the two phases cannot share one gamma.)

    ``use_particle=None`` (the default) reads it off the launch itself: a head-off
    phi emits ``log_mass = friction = None`` and the read falls back to the
    model's own global ``M`` and the config's ``gamma`` — i.e. the shipped read,
    **bit-for-bit**, not merely numerically close. ``use_particle=False`` forces
    that path.
    """
    import jax
    import jax.numpy as jnp

    from chlu.core.clu_system import ReadState
    from chlu.core.implicit_grad import truncated_rollout

    st = int(cfg.traj_stride if stride is None else stride)
    dtype = jnp.float32 if dtype is None else dtype
    q0 = jnp.asarray(launch.q0, dtype=dtype)
    p0 = jnp.zeros_like(q0)
    ratio = float(cfg.gamma_read) / float(cfg.gamma_address)
    n = q0.shape[0]
    if use_particle is None:
        use_particle = bool(getattr(launch, "has_particle", False))
    if use_particle:
        mass = jax.nn.softplus(jnp.asarray(launch.log_mass, dtype=dtype))
        fric = jnp.asarray(launch.friction, dtype=dtype)
    else:
        mass = jnp.zeros((n, 0), dtype=dtype)  # unused; uniform vmap signature
        fric = jnp.zeros((n,), dtype=dtype)

    def one(qa, pa, M, g1):
        m_ov = M if use_particle else None
        ga = g1 if use_particle else float(cfg.gamma_address)
        gr = g1 * ratio if use_particle else float(cfg.gamma_read)
        tr1, q_addr, p_addr = truncated_rollout(
            model, qa, pa, int(cfg.address_steps), float(cfg.dt), ga,
            retain=retain1, stride=st, return_endpoint=True, mass_override=m_ov)
        tr2, q_star, p_star = truncated_rollout(
            model, q_addr, p_addr, int(cfg.read_steps), float(cfg.dt), gr,
            retain=retain2, stride=st, return_endpoint=True, mass_override=m_ov)
        return jnp.concatenate([tr1, tr2], axis=0), q_addr, p_addr, q_star, p_star

    traj, q_addr, p_addr, q_star, p_star = jax.vmap(one)(q0, p0, mass, fric)
    state = ReadState(q0=q0, p0=p0, q_addr=q_addr, p_addr=p_addr,
                      q_star=q_star, p_star=p_star)
    return traj, state


# ==========================================================================
# PART "phi" — D1/D2: plumbing, ledger, the invariant, the smoke run
# ==========================================================================
def part_phi(seed: int = 0, quick: bool = False) -> dict:
    """Strong-phi plumbing, byte ledger, fairness invariant, particle head."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chlu.core.psi_readout import (
        PhiMismatchError,
        PhiSpec,
        PsiSpec,
        SharedPhi,
        assert_identical_phi,
        joint_dial,
        make_phi,
        make_psi,
        phi_ledger,
    )
    from chlu.experiments.exp_trajectory_read import build_store

    out: Dict[str, Any] = {"seed": int(seed)}
    key = jax.random.PRNGKey(seed + 4242)

    system, cfg, info = build_store(seed=seed, quick=quick)
    out["store"] = info
    model = system.model()
    dim, d, m = system.store.dim, cfg.addr_dim, cfg.payload_dim

    # ---- D1: the families, each with its byte-ledger row -------------------
    rng = np.random.default_rng(seed)
    proj = np.linalg.qr(rng.normal(size=(d, d)))[0]
    variants = [
        ("identity", dict(family="identity")),
        ("pca", dict(family="pca")),
        ("mlp", dict(family="mlp")),
        ("mlp+head", dict(family="mlp", particle_head=True)),
        ("cnn", dict(family="cnn", image_shape=(1, 8, 8), particle_head=True)),
        ("gru", dict(family="gru", seq_shape=(4, 2), particle_head=True)),
    ]
    ledger_rows = []
    phis = {}
    for name, kw in variants:
        in_dim = 64 if kw.get("family") == "cnn" else (8 if kw.get("family") == "gru" else d)
        spec = PhiSpec(in_dim=in_dim, dim=dim, addr_dim=d, payload_dim=m,
                       capacity=cfg.capacity, n_atoms=cfg.n_atoms, **kw)
        k_i = jax.random.fold_in(key, abs(hash(name)) % 1000)
        phi = make_phi(spec, k_i, proj=(proj if kw["family"] == "pca" else None))
        phis[name] = phi
        row = phi_ledger(phi, spec, arm=name)
        row["variant"] = name
        row["flags"] = spec.as_flags()
        ledger_rows.append(row)
    out["phi_ledger"] = ledger_rows

    # ---- the (d, atom-budget) JOINT DIAL — declared, not swept -------------
    out["joint_dial"] = joint_dial(d, capacity=cfg.capacity, n_atoms=cfg.n_atoms)
    out["joint_dial"]["note"] = (
        "DECLARATION only (task §3): no d-sweep this wave. K_learned(8) is "
        "LOWER-BOUNDED (K=2048 was never run), never bracketed."
    )

    # ---- ⭐ the identical-phi invariant, both directions -------------------
    spec_m = PhiSpec(in_dim=d, dim=dim, addr_dim=d, payload_dim=m, family="mlp",
                     capacity=cfg.capacity, n_atoms=cfg.n_atoms)
    shared = SharedPhi(make_phi(spec_m, jax.random.fold_in(key, 7)), spec_m)
    for arm in ("clu", "baseline_knn", "trajectory_launder", "blank_store"):
        shared.for_arm(arm)
    ok_id = shared.assert_invariant()
    # a DIFFERENT phi for one arm must RAISE, not warn
    other = make_phi(spec_m, jax.random.fold_in(key, 8))
    try:
        shared.check("baseline_knn_rebuilt", other)
    except PhiMismatchError as exc:
        raised = True
        msg = str(exc)[:160]
    else:
        raised = False
        msg = ""
    # and the same phi handed to a second arm must NOT raise
    same_ok = True
    try:
        assert_identical_phi({"a": shared.phi, "b": shared.phi})
    except PhiMismatchError:
        same_ok = False
    out["identical_phi_invariant"] = {
        "shared_phi_id": ok_id, "raises_on_mismatch": raised,
        "message": msg, "accepts_identical": same_ok,
        "enforced": bool(raised and same_ok),
    }

    # ---- D2: default-off must be BIT-IDENTICAL to the shipped launch -------
    x = jnp.asarray(system.codebook()[1][:4], dtype=jnp.float32)
    spec_off = replace(spec_m, particle_head=False)
    spec_on = replace(spec_m, particle_head=True)
    phi_off = make_phi(spec_off, jax.random.fold_in(key, 11))
    phi_on = make_phi(spec_on, jax.random.fold_in(key, 11))
    l_off = phi_off.launch(x)
    l_on = phi_on.launch(x)
    from chlu.experiments.exp_trajectory_read import differentiable_read

    tr_auto, st_auto = particle_read(model, l_off, cfg)  # auto => falls back
    tr_ship, st_ship = particle_read(model, l_off, cfg, use_particle=False)
    # ...and against the C1/C2W1 read path itself (implicit q* off, so the two
    # are the same dynamics with the same gradient plumbing)
    tr_ref, _ph, st_ref = differentiable_read(model, l_off.q0, cfg,
                                              implicit_q_star=False)
    # the head ON but its outputs pinned to the shipped values: the *numerical*
    # form of default-off, which is NOT bit-identical because the store model's
    # own inertia is softplus(N(0, 0.1)) != 1 (kinetic_mode = newtonian_learned)
    l_pin = replace_launch(l_on, friction=float(cfg.gamma_address),
                           log_mass=float(spec_on.log_mass_center))
    tr_pin, st_pin = particle_read(model, l_pin, cfg, use_particle=True)
    out["default_off"] = {
        "head_off_emits_none": bool(l_off.log_mass is None and l_off.friction is None),
        "max_abs_traj_diff_auto_vs_forced": float(
            np.max(np.abs(np.asarray(tr_auto) - np.asarray(tr_ship)))),
        "max_abs_traj_diff_vs_differentiable_read": float(
            np.max(np.abs(np.asarray(tr_auto) - np.asarray(tr_ref)))),
        "max_abs_qstar_diff_vs_differentiable_read": float(
            np.max(np.abs(np.asarray(st_auto.q_star) - np.asarray(st_ref.q_star)))),
        "max_abs_traj_diff_pinned_head_vs_shipped": float(
            np.max(np.abs(np.asarray(tr_pin) - np.asarray(tr_ship)))),
        "model_mass_vector": _jsonable(np.asarray(model.effective_inertia())),
        "gamma_address": float(cfg.gamma_address),
        "note": ("default-off is STRUCTURAL: a head-off phi emits log_mass = "
                 "friction = None and the read falls back to the model's own M "
                 "and the config's gamma => bit-identical. The 'pinned head' row "
                 "is the numerical form and is NOT zero, because M=softplus("
                 "center)=1 while the store's own inertia is softplus(N(0,0.1))."),
        "log_mass_spread_head_on": float(np.std(np.asarray(l_on.log_mass))),
        "friction_spread_head_on": float(np.std(np.asarray(l_on.friction))),
        "friction_mean_head_on": float(np.mean(np.asarray(l_on.friction))),
    }

    # ---- the friction band vs monitor #1's edge ---------------------------
    # rho_conv = med|grad V(q*)| / med|grad V(q0)| — monitor #1 trips above 1e-6;
    # doctrine R2 measured the shipped read at 4.3e-7 (2.3x inside the edge).
    from chlu.core.clu_system import _grad_norms

    band_rows = []
    for g in (spec_on.friction_lo, 0.03, 0.04, cfg.gamma_address, 0.10,
              spec_on.friction_hi):
        lau = replace_launch(l_on, friction=float(g),
                             log_mass=float(spec_on.log_mass_center))
        _tr, stt = particle_read(model, lau, cfg, use_particle=True)
        g0 = np.asarray(_grad_norms(model, stt.q0))
        gs = np.asarray(_grad_norms(model, stt.q_star))
        band_rows.append({"friction": float(g),
                          "rho_conv": float(np.median(gs) / max(np.median(g0), 1e-9)),
                          "monitor1_edge": 1e-6})
    for r in band_rows:
        r["would_trip_monitor_1"] = bool(r["rho_conv"] > r["monitor1_edge"])
    hot = [r["friction"] for r in band_rows if r["would_trip_monitor_1"]]
    out["friction_band"] = {"band": [spec_on.friction_lo, spec_on.friction_hi],
                            "rows": band_rows,
                            "shipped_gamma_address": float(cfg.gamma_address),
                            "monitor1_hot_gammas": hot,
                            "doctrine_R2_shipped_rho_conv": 4.3e-7,
                            "note": ("the declared band's LOW end is monitor-#1 "
                                     "hot on this harness — reported, not hidden "
                                     "(the band is harness-specific)")}

    # ---- one STRONG encoder, wired end to end and smoke-run ---------------
    import optax

    spec_cnn = PhiSpec(in_dim=64, dim=dim, addr_dim=d, payload_dim=m, family="cnn",
                       image_shape=(1, 8, 8), particle_head=True, cnn_channels=(8, 16),
                       cnn_pool=1, cnn_groups=4, capacity=cfg.capacity,
                       n_atoms=cfg.n_atoms)
    phi_cnn = make_phi(spec_cnn, jax.random.fold_in(key, 21))
    ids, centers, pays = system.codebook()
    K = len(ids)
    nb = 4 if quick else 8
    lab = np.asarray(jax.random.randint(jax.random.fold_in(key, 31), (nb,), 0, K))
    # a toy 8x8 "image" per item: the address broadcast into a patch + noise
    imgs = np.zeros((nb, 64), dtype=np.float32)
    for i, li in enumerate(lab):
        imgs[i, : centers.shape[1]] = centers[li]
        imgs[i] += 0.05 * np.asarray(jax.random.normal(
            jax.random.fold_in(key, 100 + i), (64,)))
    xin = jnp.asarray(imgs)
    y = jnp.asarray(pays[lab], dtype=jnp.float32)
    psi = make_psi("deepsets", PsiSpec(dim=dim, addr_dim=d, payload_dim=m,
                                       hidden=16, depth=2, input_mode="trajectory",
                                       stride=4), jax.random.fold_in(key, 41))

    def loss_fn(phi_, psi_):
        lau = phi_.launch(xin)
        tr, stt = particle_read(model, lau, cfg, use_particle=True, stride=8)
        return jnp.mean((psi_(tr, stt) - y) ** 2)

    t0 = time.time()
    val, grads = eqx.filter_value_and_grad(lambda pr: loss_fn(*pr))((phi_cnn, psi))
    compile_s = time.time() - t0
    opt = optax.adam(1e-3)
    params = (phi_cnn, psi)
    ostate = opt.init(eqx.filter(params, eqx.is_inexact_array))
    t0 = time.time()
    upd, ostate = opt.update(grads, ostate, eqx.filter(params, eqx.is_inexact_array))
    params = eqx.apply_updates(params, upd)
    val2 = loss_fn(*params)
    step_s = time.time() - t0
    out["strong_phi_smoke"] = {
        "family": "cnn", "trunk": "phi_encoders.ConvTrunk (imported, not forked)",
        "phi_bytes": phi_ledger(phi_cnn, spec_cnn)["phi_bytes"],
        "phi_params": phi_ledger(phi_cnn, spec_cnn)["phi_params"],
        "loss_before": float(val), "loss_after_1_step": float(val2),
        "grad_norm_phi": _tree_norm(grads[0]), "grad_norm_psi": _tree_norm(grads[1]),
        "compile_and_grad_s": compile_s, "step_s": step_s,
        "batch": int(nb),
        "gradient_reaches_phi": bool(_tree_norm(grads[0]) > 0.0),
    }
    return out


def replace_launch(launch, *, friction=None, log_mass=None):
    """A :class:`ParticleLaunch` with one field replaced (broadcast scalars)."""
    import jax.numpy as jnp

    from chlu.core.psi_readout import ParticleLaunch

    n, dim = launch.q0.shape
    f = launch.friction if friction is None else jnp.full((n,), float(friction),
                                                          dtype=launch.q0.dtype)
    lm = launch.log_mass if log_mass is None else jnp.full((n, dim), float(log_mass),
                                                           dtype=launch.q0.dtype)
    return ParticleLaunch(q0=launch.q0, log_mass=lm, friction=f)


# ==========================================================================
# ⭐ PART "grad" — D3: does the mass gauge dissolve under a trajectory read?
# ==========================================================================
def part_grad(seed: int = 0, quick: bool = False, batch: int = 16) -> dict:
    """The pre-registered mass/friction gradient measurement (charter §A2.2).

    Prop Q1.1: ``Fix(T_theta) = {(q, 0) : grad V = 0}`` contains neither ``M``
    nor ``gamma``, so a **settled-point** read-out cannot see either. The
    trajectory is not a fixed point. This measures both, on the real store, with
    a matched-parameter psi pair.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chlu.core.implicit_grad import SettleSpec, implicit_settle
    from chlu.core.psi_readout import PhiSpec, PsiSpec, make_phi, matched_pair, psi_param_count
    from chlu.experiments.exp_trajectory_read import build_store

    t_all = time.time()
    out: Dict[str, Any] = {"seed": int(seed), "batch": int(batch), "quick": bool(quick)}
    system, cfg, info = build_store(seed=seed, quick=quick)
    out["store"] = info
    model = system.model()
    dim, d, m = system.store.dim, cfg.addr_dim, cfg.payload_dim
    ids, centers, pays = system.codebook()
    K = len(ids)
    batch = 8 if quick else int(batch)

    key = jax.random.PRNGKey(seed + 909)
    k_q, k_psi, k_phi = jax.random.split(key, 3)
    lab = np.asarray(jax.random.randint(k_q, (batch,), 0, K))
    x = jnp.asarray(centers[lab] + np.asarray(
        jax.random.normal(jax.random.fold_in(k_q, 1), (batch, d))) * cfg.query_sigma,
        dtype=jnp.float32)
    y = jnp.asarray(pays[lab], dtype=jnp.float32)

    spec_phi = PhiSpec(in_dim=d, dim=dim, addr_dim=d, payload_dim=m, family="mlp",
                       particle_head=True, capacity=cfg.capacity, n_atoms=cfg.n_atoms)
    phi = make_phi(spec_phi, k_phi)
    launch0 = phi.launch(x)
    log_mass0 = jnp.asarray(launch0.log_mass)
    fric0 = jnp.asarray(launch0.friction)
    q0 = jnp.asarray(launch0.q0)

    base_psi = PsiSpec(dim=dim, addr_dim=d, payload_dim=m, hidden=32, depth=2,
                       stride=1)
    psi_point, psi_traj = matched_pair("deepsets", base_psi, k_psi)
    out["psi"] = {"family": "deepsets", "params_point": psi_param_count(psi_point),
                  "params_trajectory": psi_param_count(psi_traj),
                  "matched": psi_param_count(psi_point) == psi_param_count(psi_traj),
                  "bit_identical_params": bool(all(
                      np.array_equal(np.asarray(a), np.asarray(b))
                      for a, b in zip(
                          jax.tree_util.tree_leaves(eqx.filter(psi_point, eqx.is_inexact_array)),
                          jax.tree_util.tree_leaves(eqx.filter(psi_traj, eqx.is_inexact_array)),
                          strict=True,
                      )))}

    # ---- the losses -------------------------------------------------------
    from chlu.core.psi_readout import ParticleLaunch

    def loss_of(psi_, lm, gf, *, use_particle=True):
        lau = ParticleLaunch(q0=q0, log_mass=lm, friction=gf)
        tr, stt = particle_read(model, lau, cfg, use_particle=use_particle)
        return jnp.mean((psi_(tr, stt) - y) ** 2)

    arms = {}
    for name, psi_ in (("settled_point", psi_point), ("trajectory", psi_traj)):
        f = _bind(loss_of, psi_)
        t0 = time.time()
        val, (g_lm, g_gf) = jax.value_and_grad(f, argnums=(0, 1))(log_mass0, fric0)
        val.block_until_ready()
        wall = time.time() - t0
        arms[name] = {
            "loss": float(val),
            "grad_log_mass_norm": float(jnp.linalg.norm(g_lm)),
            "grad_friction_norm": float(jnp.linalg.norm(g_gf)),
            "grad_log_mass_max_abs": float(jnp.max(jnp.abs(g_lm))),
            "grad_friction_max_abs": float(jnp.max(jnp.abs(g_gf))),
            "wall_s": wall,
        }
        # psi-gradient scale, for the relative "numerically zero" test
        gpsi = eqx.filter_grad(lambda p_: loss_of(p_, log_mass0, fric0))(psi_)
        arms[name]["grad_psi_norm"] = _tree_norm(gpsi)
    out["arms_unrolled"] = arms

    # ---- P1: the IMPLICIT point arm — exactly zero by Prop Q1.1 -----------
    # Mass enters here as the model's own `log_mass` leaf, so the implicit VJP
    # (theta_bar = -VJP_theta[grad V]) can be inspected directly: grad V does not
    # contain M, so the mass cotangent is exactly 0 — not approximately.
    spec_s = SettleSpec(steps=int(cfg.read_steps), dt=float(cfg.dt),
                        gamma=float(cfg.gamma_read), ridge=0.0)
    p0 = jnp.zeros_like(q0)

    def loss_implicit(model_):
        qs = jax.vmap(lambda a, b: implicit_settle(model_, a, b, spec_s))(q0, p0)
        stt_traj = jnp.concatenate([qs, jnp.zeros_like(qs)], axis=-1)[:, None, :]
        from chlu.core.clu_system import ReadState

        stt = ReadState(q0=q0, p0=p0, q_addr=q0, p_addr=p0, q_star=qs,
                        p_star=jnp.zeros_like(qs))
        return jnp.mean((psi_point(stt_traj, stt) - y) ** 2)

    def loss_unrolled_model(model_):
        lau = ParticleLaunch(q0=q0, log_mass=log_mass0, friction=fric0)
        tr, stt = particle_read(model_, lau, cfg, use_particle=False)
        return jnp.mean((psi_point(tr, stt) - y) ** 2)

    g_impl = eqx.filter_grad(loss_implicit)(model)
    g_unr = eqx.filter_grad(loss_unrolled_model)(model)
    out["model_log_mass_grad"] = {
        "implicit_settle": float(np.linalg.norm(np.asarray(g_impl.log_mass))),
        "implicit_settle_exact_zero": bool(np.all(np.asarray(g_impl.log_mass) == 0.0)),
        "full_unroll": float(np.linalg.norm(np.asarray(g_unr.log_mass))),
        "note": ("implicit: theta_bar = -VJP_theta[grad V(.,q*)] and grad V has no "
                 "M in it => exactly 0. unroll: the geometric-death remnant."),
    }

    # ---- the headline ratio ----------------------------------------------
    pt, tj = arms["settled_point"], arms["trajectory"]
    zero_bar = 1e-10
    out["mass_gauge"] = {
        "point_grad_log_mass": pt["grad_log_mass_norm"],
        "traj_grad_log_mass": tj["grad_log_mass_norm"],
        "ratio_log_mass": (tj["grad_log_mass_norm"] / pt["grad_log_mass_norm"]
                           if pt["grad_log_mass_norm"] > 0 else float("inf")),
        "point_grad_friction": pt["grad_friction_norm"],
        "traj_grad_friction": tj["grad_friction_norm"],
        "ratio_friction": (tj["grad_friction_norm"] / pt["grad_friction_norm"]
                           if pt["grad_friction_norm"] > 0 else float("inf")),
        "numerically_zero_bar": zero_bar,
        "traj_mass_numerically_zero": bool(
            tj["grad_log_mass_norm"] < zero_bar
            or tj["grad_log_mass_norm"] < zero_bar * max(tj["grad_psi_norm"], 1e-30)),
        "traj_friction_numerically_zero": bool(tj["grad_friction_norm"] < zero_bar),
        "verdict": None,  # filled below
    }
    mg = out["mass_gauge"]
    mg["verdict"] = ("A2.2 REFUTED (trajectory mass gradient is numerically zero)"
                     if mg["traj_mass_numerically_zero"]
                     else "A2.2 SUPPORTED (mass gauge dissolves under the trajectory read)")

    # ---- P7: finite-difference cross-check on the trajectory arm ----------
    def scalar_loss(psi_, dm: float, dg: float):
        return float(loss_of(psi_, log_mass0 + dm, fric0 * (1.0 + dg)))

    fd = {}
    for name, psi_ in (("settled_point", psi_point), ("trajectory", psi_traj)):
        f = _bind(loss_of, psi_)
        _, (g_lm, g_gf) = jax.value_and_grad(f, argnums=(0, 1))(log_mass0, fric0)
        # directional derivative along the all-ones log-mass direction
        h = 1e-2
        dplus = scalar_loss(psi_, +h, 0.0)
        dminus = scalar_loss(psi_, -h, 0.0)
        fd_mass = (dplus - dminus) / (2 * h)
        ad_mass = float(jnp.sum(g_lm))
        # relative multiplicative perturbation of friction
        hg = 1e-2
        gplus = scalar_loss(psi_, 0.0, +hg)
        gminus = scalar_loss(psi_, 0.0, -hg)
        fd_fric = (gplus - gminus) / (2 * hg)
        ad_fric = float(jnp.sum(g_gf * fric0))
        fd[name] = {
            "ad_dL_dlogmass_sum": ad_mass, "fd_dL_dlogmass_sum": fd_mass,
            "rel_err_mass": (abs(fd_mass - ad_mass) / max(abs(ad_mass), 1e-30)
                             if abs(ad_mass) > 0 else float("nan")),
            "ad_dL_dfriction_dir": ad_fric, "fd_dL_dfriction_dir": fd_fric,
            "rel_err_friction": (abs(fd_fric - ad_fric) / max(abs(ad_fric), 1e-30)
                                 if abs(ad_fric) > 0 else float("nan")),
            "h_logmass": h, "h_friction_rel": hg,
        }
    out["finite_differences"] = fd
    out["finite_differences"]["_note"] = (
        "float32 (the store's own precision). The FD NOISE FLOOR on a loss of "
        "order 0.5 is eps_f32*L/(2h) ~ 3e-6 at h=1e-2, which is 400x the point "
        "arm's true gradient => FD cannot resolve the point arm at this "
        "precision. See finite_differences_float64."
    )

    # ---- the SAME cross-check in float64, on the SAME store ---------------
    # (post-hoc, added after the float32 pass showed an FD noise floor 400x
    # above the point arm's gradient; declared as unregistered in the report)
    out["finite_differences_float64"] = real_store_fd_float64(
        model, cfg, q0, log_mass0, fric0, y, psi_point, psi_traj)

    # ---- P8: the float64 toy, where FD is tight ---------------------------
    out["toy_float64"] = toy_mass_gradcheck()

    # ---- wall clock -------------------------------------------------------
    out["budget_s"] = 30.0
    out["falsifier_s"] = 300.0
    out["budget_met"] = bool(max(a["wall_s"] for a in arms.values()) <= 30.0)
    out["wall_s"] = time.time() - t_all
    return out


def real_store_fd_float64(model, cfg, q0, log_mass0, fric0, y, psi_point, psi_traj,
                          hs=(1e-2, 1e-3, 1e-4)) -> dict:
    """The mass/friction FD cross-check on the **real store**, in float64.

    Everything (store potential, psi, launch, particle attributes) is upcast to
    float64 and the identical read is re-run, so the only thing that changes is
    precision. Reported at several ``h`` because the trajectory loss is *curved*
    in ``log M`` — a single ``h`` cannot separate "the AD is wrong" from "the FD
    step is too large".
    """
    import jax
    import jax.numpy as jnp

    from chlu.core.psi_readout import ParticleLaunch

    jax.config.update("jax_enable_x64", True)
    try:
        def up(tree):
            return jax.tree_util.tree_map(
                lambda x: (jnp.asarray(x, dtype=jnp.float64)
                           if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
                           else x), tree)

        m64 = up(model)
        q064, lm64, gf64, y64 = (jnp.asarray(v, dtype=jnp.float64)
                                 for v in (q0, log_mass0, fric0, y))
        rows = {}
        for name, psi_ in (("settled_point", psi_point), ("trajectory", psi_traj)):
            p64 = up(psi_)

            def loss(lm, gf, p64=p64):
                lau = ParticleLaunch(q0=q064, log_mass=lm, friction=gf)
                tr, stt = particle_read(m64, lau, cfg, use_particle=True,
                                        dtype=jnp.float64)
                return jnp.mean((p64(tr, stt) - y64) ** 2)

            _, (g_lm, g_gf) = jax.value_and_grad(loss, argnums=(0, 1))(lm64, gf64)
            ad_m, ad_g = float(jnp.sum(g_lm)), float(jnp.sum(g_gf * gf64))
            per_h = []
            for h in hs:
                fdm = float((loss(lm64 + h, gf64) - loss(lm64 - h, gf64)) / (2 * h))
                fdg = float((loss(lm64, gf64 * (1 + h)) - loss(lm64, gf64 * (1 - h)))
                            / (2 * h))
                per_h.append({
                    "h": h, "fd_mass": fdm, "fd_friction": fdg,
                    "rel_err_mass": (abs(fdm - ad_m) / abs(ad_m)
                                     if abs(ad_m) > 0 else float("nan")),
                    "rel_err_friction": (abs(fdg - ad_g) / abs(ad_g)
                                         if abs(ad_g) > 0 else float("nan")),
                })
            rows[name] = {
                "grad_log_mass_norm": float(jnp.linalg.norm(g_lm)),
                "grad_friction_norm": float(jnp.linalg.norm(g_gf)),
                "ad_dL_dlogmass_sum": ad_m, "ad_dL_dfriction_dir": ad_g,
                "fd_by_h": per_h,
                "best_rel_err_mass": min(r["rel_err_mass"] for r in per_h),
                "best_rel_err_friction": min(r["rel_err_friction"] for r in per_h),
            }
        rows["ratio_log_mass"] = (rows["trajectory"]["grad_log_mass_norm"]
                                  / max(rows["settled_point"]["grad_log_mass_norm"],
                                        1e-300))
        rows["ratio_friction"] = (rows["trajectory"]["grad_friction_norm"]
                                  / max(rows["settled_point"]["grad_friction_norm"],
                                        1e-300))
        rows["dtype"] = "float64 (store upcast; dynamics otherwise identical)"
        return rows
    finally:
        jax.config.update("jax_enable_x64", False)


def toy_mass_gradcheck() -> dict:
    """The controlled toy with a known answer, in **float64** (spike Part A).

    ``GaussianWellsPotential`` + the shipped damped Verlet: the mass/friction
    gradient of an endpoint loss vs the same gradient of a **trajectory** loss,
    each cross-checked against central finite differences at the spike's
    registered 1e-5 bar.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    try:
        from chlu.core.implicit_grad import GaussianWellsPotential, truncated_rollout
        from chlu.experiments.goldstone_harness import clu_with_potential

        centers = jnp.asarray(np.stack([
            [np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 5)[:4]]),
            dtype=jnp.float64)
        V = GaussianWellsPotential(centers, jnp.asarray([1.0, 0.9, 1.1, 0.95],
                                                        dtype=jnp.float64),
                                   s=0.35, alpha=0.05)
        model = clu_with_potential(V, dim=2, kinetic_mode="newtonian_learned",
                                   inertia=jnp.ones(2, dtype=jnp.float64))
        q0 = centers[0] + jnp.asarray([0.2, -0.15], dtype=jnp.float64)
        p0 = jnp.zeros(2, dtype=jnp.float64)
        N, dt, gam = 1500, 0.05, 0.05

        def endpoint_loss(log_m, g):
            M = jax.nn.softplus(log_m)
            tr = truncated_rollout(model, q0, p0, N, dt, g, retain=None, stride=1,
                                   mass_override=M)
            return 0.5 * jnp.sum(tr[-1, :2] ** 2)

        def traj_loss(log_m, g):
            M = jax.nn.softplus(log_m)
            tr = truncated_rollout(model, q0, p0, N, dt, g, retain=None, stride=10,
                                   mass_override=M)
            return jnp.mean(jnp.sum(tr[:, :2] ** 2, axis=-1))

        lm0 = jnp.asarray([0.5413248546129181] * 2, dtype=jnp.float64)
        g0 = jnp.asarray(gam, dtype=jnp.float64)
        rows = {}
        for name, fn in (("endpoint", endpoint_loss), ("trajectory", traj_loss)):
            val, (gm, gg) = jax.value_and_grad(fn, argnums=(0, 1))(lm0, g0)
            h = 1e-4
            fdm = float((fn(lm0 + h, g0) - fn(lm0 - h, g0)) / (2 * h))
            fdg = float((fn(lm0, g0 + h) - fn(lm0, g0 - h)) / (2 * h))
            adm, adg = float(jnp.sum(gm)), float(gg)
            rows[name] = {
                "loss": float(val),
                "ad_dlogmass": adm, "fd_dlogmass": fdm,
                "rel_err_mass": (abs(fdm - adm) / max(abs(adm), 1e-30)
                                 if abs(adm) > 1e-30 else float("nan")),
                "ad_dfriction": adg, "fd_dfriction": fdg,
                "rel_err_friction": (abs(fdg - adg) / max(abs(adg), 1e-30)
                                     if abs(adg) > 1e-30 else float("nan")),
                "grad_logmass_norm": float(jnp.linalg.norm(gm)),
                "grad_friction_abs": float(jnp.abs(gg)),
            }
        rows["ratio_traj_over_endpoint_mass"] = (
            rows["trajectory"]["grad_logmass_norm"]
            / max(rows["endpoint"]["grad_logmass_norm"], 1e-300))
        rows["dtype"] = "float64"
        rows["settle"] = {"N": N, "dt": dt, "gamma": gam}
        return rows
    finally:
        jax.config.update("jax_enable_x64", False)


# ==========================================================================
# PART "monitors" — D4 acceptance: the before/after trip diff on the anchor
# ==========================================================================
def _trip_tables(readings) -> Dict[str, Dict[str, bool]]:
    """``{"post": {name: tripped}, "pre": {...}}`` from a registry's readings.

    The pre-repair state is re-scored from the **same** readings
    (``detail["tripped_pre_repair"]``), so the diff is exact and needs no second
    stochastic run.
    """
    post: Dict[str, bool] = {}
    pre: Dict[str, bool] = {}
    n_post: Dict[str, int] = {}
    n_pre: Dict[str, int] = {}
    for r in readings:
        d = r.detail or {}
        tp = bool(d.get("tripped_pre_repair", r.tripped))
        post[r.name] = post.get(r.name, False) or bool(r.tripped)
        pre[r.name] = pre.get(r.name, False) or tp
        n_post[r.name] = n_post.get(r.name, 0) + int(bool(r.tripped))
        n_pre[r.name] = n_pre.get(r.name, 0) + int(tp)
    return {"post": post, "pre": pre, "n_post": n_post, "n_pre": n_pre}


def part_monitors(seeds=(0, 1, 2), quick: bool = False) -> dict:
    """Re-run the C2W1 anchor and diff monitor trip-states, before vs after."""
    import chlu.experiments.exp_memory_gym as gymx

    out: Dict[str, Any] = {"anchor": f"{ANCHOR_FAMILY}/{ANCHOR_ARM}",
                           "seeds": list(seeds), "cells": []}
    # the C2W1 artifact on disk — the thing we must diff against (never a fresh
    # baseline we generate ourselves)
    ref = {}
    if os.path.exists(ANCHOR_ARTIFACT):
        with open(ANCHOR_ARTIFACT) as fh:
            art = json.load(fh)
        ref = art.get("monitor_table", {})
        out["reference_artifact"] = ANCHOR_ARTIFACT
    else:
        out["reference_artifact"] = None

    captured = []
    orig_build = gymx.build_system

    def _spy(*a, **k):
        s = orig_build(*a, **k)
        captured.append(s)
        return s

    for seed in seeds:
        captured.clear()
        gymx.build_system = _spy  # read-only file: instrumented at runtime, not edited
        t0 = time.time()
        try:
            rec = gymx.run_cell(ANCHOR_FAMILY, ANCHOR_ARM, seed=int(seed),
                                quick=quick, loud=False)
        finally:
            gymx.build_system = orig_build
        # ⚠ the blank-store control builds a SECOND system that never observes;
        # the cell's own system is the one carrying the readings.
        system = max(captured, key=lambda s: len(s.registry.readings))
        tables = _trip_tables(system.registry.readings)
        label = f"{ANCHOR_FAMILY}/{ANCHOR_ARM}@s{seed}"

        # every #6 reading, with both predicates, so the re-score is auditable
        obj6 = [{"stage": r.stage,
                 "slope_write_loss": r.detail.get("slope_write_loss"),
                 "slope_acq": r.detail.get("slope_acq"),
                 "eps": r.detail.get("eps_dead_band"),
                 "loss_scale": r.detail.get("loss_scale"),
                 "tripped_post": bool(r.tripped),
                 "tripped_pre": bool(r.detail.get("tripped_pre_repair", r.tripped))}
                for r in system.registry.readings
                if r.name == "objective_divergence" and r.applicable]

        changed = sorted(k for k in tables["post"]
                         if bool(tables["post"][k]) != bool(tables["pre"][k]))
        # ...and against the C2W1 artifact
        ref_states = {name: row["cells"].get(label)
                      for name, row in ref.items() if label in row.get("cells", {})}
        post_states = {}
        for m in rec.get("monitors", []):
            post_states[m["name"]] = ("TRIP" if m["tripped"]
                                      else ("inapplicable" if not m["applicable"]
                                            else "clear"))
        art_diff = {k: {"c2w1": ref_states.get(k), "now": post_states.get(k)}
                    for k in sorted(set(ref_states) | set(post_states))
                    if ref_states.get(k) != post_states.get(k)}
        out["cells"].append({
            "cell": label, "seed": int(seed), "wall_s": time.time() - t0,
            "n_live": rec.get("n_live"), "primary": rec.get("dividend", {}),
            "trips_now": rec.get("trips"),
            "trip_state_pre_repair": tables["pre"],
            "trip_state_post_repair": tables["post"],
            "n_trips_pre": tables["n_pre"], "n_trips_post": tables["n_post"],
            "changed_monitors": changed,
            "monitor6_readings": obj6,
            "artifact_diff_vs_c2w1": art_diff,
            "artifact_states_c2w1": ref_states,
            "artifact_states_now": post_states,
        })
        print(f"[anchor {label}] changed={changed} artifact_diff={list(art_diff)} "
              f"({time.time() - t0:.0f}s)")

    # the whole-gym #6 re-score, from the C2W1 log (the only place the 58
    # first-ever trips are recorded)
    out["monitor6_rescore_from_log"] = rescore_monitor6_log()
    return out


def part_gym_rescore(quick: bool = False, limit: Optional[int] = None) -> dict:
    """⭐ D4 acceptance, gym-wide: re-run **every C2W1 cell** and diff trip states.

    The plan is read from the C2W1 artifact itself (``family``/``arm``/``seed`` of
    each of its 28 cells), so the comparison is against the same cells and not a
    plan of my own choosing. Each cell is run with a ``build_system`` spy so the
    registry's **complete** reading list (consolidation windows included) is
    available, and every monitor-#6 reading carries both predicates — which makes
    the "how many of the 58 first-ever trips survive the dead-band" question
    answerable **exactly**, not by inference from the log's ``slope_acq``.
    """
    import chlu.experiments.exp_memory_gym as gymx

    with open(ANCHOR_ARTIFACT) as fh:
        art = json.load(fh)
    ref_table = art.get("monitor_table", {})
    plan = [(c["family"], c["arm"], int(c["seed"])) for c in art["cells"]]
    if limit:
        plan = plan[: int(limit)]

    out: Dict[str, Any] = {"n_cells": len(plan), "cells": [],
                           "reference_artifact": ANCHOR_ARTIFACT}
    all6: list = []
    changed_cells: list = []
    artifact_diffs: Dict[str, Any] = {}
    captured: list = []
    orig_build = gymx.build_system

    def _spy(*a, **k):
        s = orig_build(*a, **k)
        captured.append(s)
        return s

    for family, arm, seed in plan:
        captured.clear()
        label = f"{family}/{arm}@s{seed}"
        gymx.build_system = _spy
        t0 = time.time()
        try:
            rec = gymx.run_cell(family, arm, seed=seed, quick=quick, loud=False)
        except Exception as exc:  # a failed cell is reported, never skipped
            gymx.build_system = orig_build
            out["cells"].append({"cell": label, "error": repr(exc)})
            print(f"[gym-rescore] {label} ERROR {exc!r}")
            continue
        finally:
            gymx.build_system = orig_build
        system = max(captured, key=lambda s: len(s.registry.readings))
        tables = _trip_tables(system.registry.readings)
        changed = sorted(k for k in tables["post"]
                         if bool(tables["post"][k]) != bool(tables["pre"][k]))
        rows6 = [{"cell": label, "stage": r.stage,
                  "slope_write_loss": r.detail.get("slope_write_loss"),
                  "slope_acq": r.detail.get("slope_acq"),
                  "eps": r.detail.get("eps_dead_band"),
                  "pre": bool(r.detail.get("tripped_pre_repair", r.tripped)),
                  "post": bool(r.tripped)}
                 for r in system.registry.readings
                 if r.name == "objective_divergence" and r.applicable]
        all6 += rows6
        post_states = {m["name"]: ("TRIP" if m["tripped"]
                                   else ("inapplicable" if not m["applicable"] else "clear"))
                       for m in rec.get("monitors", [])}
        post_states_a = {m["name"]: ("TRIP" if m["tripped"]
                                     else ("inapplicable" if not m["applicable"] else "clear"))
                         for m in rec.get("monitors_annealed", [])}
        diff = {}
        for name, row in ref_table.items():
            for key, states in ((label, post_states), (label + "/annealed", post_states_a)):
                if key in row.get("cells", {}) and name in states:
                    if row["cells"][key] != states[name]:
                        diff[f"{name}@{key}"] = {"c2w1": row["cells"][key],
                                                 "now": states[name]}
        if diff:
            artifact_diffs[label] = diff
        if changed:
            changed_cells.append({"cell": label, "changed": changed})
        out["cells"].append({
            "cell": label, "wall_s": time.time() - t0,
            "degenerate": bool(rec.get("degenerate")),
            "n_trips_pre": {k: v for k, v in tables["n_pre"].items() if v},
            "n_trips_post": {k: v for k, v in tables["n_post"].items() if v},
            "changed_monitors": changed, "artifact_diff": diff,
            "monitor6_readings": rows6,
        })
        print(f"[gym-rescore] {label:34s} changed={changed} diff={list(diff)} "
              f"({time.time() - t0:.0f}s)")

    pre_trips = [r for r in all6 if r["pre"]]
    post_trips = [r for r in all6 if r["post"]]
    killed = [r for r in all6 if r["pre"] and not r["post"]]
    born = [r for r in all6 if r["post"] and not r["pre"]]
    out["monitor6"] = {
        "n_readings_applicable": len(all6),
        "n_trips_pre_repair": len(pre_trips),
        "n_trips_post_repair": len(post_trips),
        "n_killed_by_dead_band": len(killed),
        "n_new_trips": len(born),  # MUST be 0: a dead-band can only remove trips
        "killed_slope_loss_max_abs": (max(abs(r["slope_write_loss"]) for r in killed)
                                      if killed else None),
        "surviving_slope_loss_min_abs": (min(abs(r["slope_write_loss"])
                                             for r in post_trips)
                                         if post_trips else None),
        "c2w1_reported_first_ever_trips": 58,
        "c2w1_reported_epsilon_artefacts": 29,
    }
    out["changed_cells"] = changed_cells
    out["artifact_diffs"] = artifact_diffs
    out["monitors_changed_other_than_6"] = sorted(
        {m for c in changed_cells for m in c["changed"] if m != "objective_divergence"})
    return out


def rescore_monitor6_log(path: Optional[str] = None) -> dict:
    """Re-score monitor #6's 58 first-ever trips from the C2W1 gym log.

    The loud trip line carries ``value = slope_acq``. A trip whose ``slope_acq``
    is at the floating-point floor is the R2 artefact; ``slope_write_loss`` is not
    in the log, so this is scored on ``|slope_acq| < tol`` and cross-checked
    against the anchor cells (where both slopes are measured directly).
    """
    import re

    path = path or os.path.join(os.path.expanduser("~"), "Desktop", "CHLU", ".claude",
                                "outputs", "memory-gym-v0", "run_full.log")
    if not os.path.exists(path):
        return {"log": None}
    pat = re.compile(r"MONITOR TRIP \[#6 objective_divergence\] stage='([^']*)' "
                     r"t=(\d+) value=([-\d.e+]+)")
    rows = [(m.group(1), int(m.group(2)), float(m.group(3)))
            for m in pat.finditer(open(path).read())]
    vals = np.asarray([v for _, _, v in rows], dtype=float)
    tol = 1e-12
    tiny = np.abs(vals) < tol
    return {
        "log": path, "n_trips_total": int(vals.size),
        "n_slope_acq_below_1e-12": int(np.sum(tiny)),
        "n_survive_predicted": int(vals.size - np.sum(tiny)),
        "tiny_values": sorted(set(float(v) for v in vals[tiny])),
        "smallest_surviving_abs": (float(np.min(np.abs(vals[~tiny])))
                                   if np.any(~tiny) else None),
        "stages_tiny": sorted({s for (s, _, v) in rows if abs(v) < tol}),
        "caveat": ("the log records slope_acq only; the dead-band is on "
                   "slope_loss. Cross-checked on the anchor cells, where both "
                   "slopes are measured."),
    }


# ==========================================================================
def main(argv: Optional[list] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="phi",
                    choices=["phi", "grad", "monitors", "gym-rescore", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="first N cells only (gym-rescore)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds for --part monitors")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    out: Dict[str, Any] = {"part": args.part, "seed": args.seed, "quick": args.quick}
    t0 = time.time()
    if args.part in ("phi", "all"):
        out["PHI"] = part_phi(seed=args.seed, quick=args.quick)
        print(json.dumps(_jsonable({k: v for k, v in out["PHI"].items()
                                    if k != "phi_ledger"}), indent=2))
        print("\n  phi byte ledger:")
        for row in out["PHI"]["phi_ledger"]:
            print(f"    {row['variant']:10s} {row['phi_family']:9s} head="
                  f"{str(row['phi_particle_head']):5s} params={row['phi_params']:7d} "
                  f"bytes={row['phi_bytes']:8d}  phi_id={row['phi_id']}")
    if args.part in ("grad", "all"):
        out["GRAD"] = part_grad(seed=args.seed, quick=args.quick, batch=args.batch)
        print(json.dumps(_jsonable(out["GRAD"]), indent=2))
    if args.part in ("gym-rescore", "all"):
        out["GYM"] = part_gym_rescore(quick=args.quick, limit=args.limit)
        print(json.dumps(_jsonable({k: v for k, v in out["GYM"].items()
                                    if k != "cells"}), indent=2))
    if args.part in ("monitors", "all"):
        out["MONITORS"] = part_monitors(seeds=tuple(args.seeds), quick=args.quick)
        print(json.dumps(_jsonable({k: v for k, v in out["MONITORS"].items()
                                    if k != "cells"}), indent=2))
        for c in out["MONITORS"]["cells"]:
            print(f"  {c['cell']:34s} changed={c['changed_monitors']} "
                  f"artifact_diff={list(c['artifact_diff_vs_c2w1'])}")
    out["wall_s"] = time.time() - t0
    path = _save(f"exp_phi_particle_{args.part}_seed{args.seed}.json", out)
    print(f"\n[exp_phi_particle] wrote {path}  ({out['wall_s']:.1f} s)")
    return out


if __name__ == "__main__":
    main()
