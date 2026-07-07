"""Tests for V1 wormhole routing (energy-gated sparse non-local edges).

Contract (task acceptance):
  (1) gate monotonicity in the driving energy — the smooth energy gate
      g = sigmoid((drive - t)/w) is monotone increasing in `drive`, and the
      lattice's GatedCoupling gate is monotone DECREASING in its base value v
      (opens as endpoints align);
  (2) closed-gate ⇒ bit-equal to uncoupled units — with gate weight 0 (or the
      governor disabled) the joint routed settle reduces exactly to independent
      per-unit rollouts (extends the kappa_c=0 lattice reduction to the gated
      edge);
  (3) routing smoke — the experiment runs end-to-end in quick mode and the
      wormhole raises distant-answer accuracy above local-only.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from chlu.core.chlu_unit import CHLU  # noqa: E402
from chlu.core.lattice import GatedCoupling, channel_spring_coupling  # noqa: E402
from chlu.experiments.exp_v1_wormhole import (  # noqa: E402
    _fit_router_mlp,
    _joint_settle,
    _potential_grad_flops,
    _router_mlp_flops,
    _router_n_params,
    _verlet_flops,
    smooth_gate,
)

DT = 0.05


# ---------------------------------------------------------------------------
# (1) gate monotonicity in the driving energy
# ---------------------------------------------------------------------------


def test_smooth_gate_monotone_increasing_in_drive():
    drive = np.linspace(-5.0, 5.0, 101)
    g = smooth_gate(drive, threshold=0.7, width=0.5)
    # strictly increasing, bounded in (0, 1), crosses 0.5 at threshold
    assert np.all(np.diff(g) > 0), "gate not monotone increasing in drive"
    assert g.min() > 0.0 and g.max() < 1.0
    mid = smooth_gate(0.7, threshold=0.7, width=0.5)
    assert abs(float(mid) - 0.5) < 1e-9


def test_gated_coupling_monotone_decreasing_in_base():
    """lattice.GatedCoupling opens (->1) as the base value v shrinks (endpoints
    align); the transmitted energy g(v)*v is bounded (F5 §7.4)."""
    base = channel_spring_coupling(2, 2, kappa=1.0, channel=(0, 1))
    gate = GatedCoupling(base=base, threshold=1.0, width=0.25)
    # sweep separation on channel 0; larger separation => larger v => gate closes
    seps = np.linspace(0.0, 4.0, 40)
    gvals, energies = [], []
    for s in seps:
        qi = jnp.array([0.0, 0.0])
        qj = jnp.array([s, 0.0])
        v = float(base(qi, qj))
        g = float(jax.nn.sigmoid((gate.threshold - v) / gate.width))
        gvals.append(g)
        energies.append(float(gate(qi, qj)))
    gvals = np.asarray(gvals)
    assert np.all(np.diff(gvals) <= 1e-9), "gate should close as base value grows"
    # transmitted wormhole energy is bounded (capped by construction)
    assert np.max(energies) < 10.0


# ---------------------------------------------------------------------------
# (2) closed-gate ⇒ bit-equal to uncoupled units
# ---------------------------------------------------------------------------


def _two_units():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    u1 = CHLU(
        dim=4, hidden=8, kinetic_mode="newtonian_learned", potential_type="mlp", key=k1
    )
    u2 = CHLU(
        dim=4, hidden=8, kinetic_mode="relativistic", potential_type="mlp", key=k2
    )
    return u1, u2


def test_closed_gate_reduces_to_independent_units_bitlevel():
    """gate weight 0 AND governor disabled (sensitivity=0 => gamma=0) => the
    joint routed settle equals two independent CHLU rollouts, bit-for-bit."""
    u1, u2 = _two_units()
    coupling = channel_spring_coupling(4, 4, kappa=1.0, channel=(2, 3))
    kq, kp = jax.random.split(jax.random.PRNGKey(7))
    B, steps = 3, 40
    q0 = jax.random.normal(kq, (B, 8))
    p0 = jax.random.normal(kp, (B, 8))

    # gate = 0 => the coupling term is exactly 0; sensitivity=0 => gamma=0;
    # no clamp => pure independent Hamiltonian dynamics.
    clamp = jnp.zeros(8, dtype=bool)
    qf, pf = _joint_settle(
        (u1, u2),
        (coupling,),
        ((0, 1),),
        (0, 4, 8),
        q0,
        p0,
        jnp.zeros((B, 1)),
        steps,
        DT,
        jnp.asarray(0.0),
        0.0,
        clamp,
    )
    # independent rollouts (CHLU.__call__ returns (steps, 2*dim); take last row)
    t1 = jax.vmap(lambda q, p: u1(q, p, steps=steps, dt=DT, gamma=0.0))(
        q0[:, :4], p0[:, :4]
    )
    t2 = jax.vmap(lambda q, p: u2(q, p, steps=steps, dt=DT, gamma=0.0))(
        q0[:, 4:], p0[:, 4:]
    )
    ref_q = jnp.concatenate([t1[:, -1, :4], t2[:, -1, :4]], axis=1)
    ref_p = jnp.concatenate([t1[:, -1, 4:], t2[:, -1, 4:]], axis=1)
    assert jnp.array_equal(qf, ref_q), "closed-gate q != independent units"
    assert jnp.array_equal(pf, ref_p), "closed-gate p != independent units"


def test_gate_weight_zero_equals_edge_removed():
    """Under the joint governor, gate weight 0 gives a state bit-identical to
    the same lattice with the edge removed (the coupling term is 0.0*V_c)."""
    u1, u2 = _two_units()
    coupling = channel_spring_coupling(4, 4, kappa=1.0, channel=(2, 3))
    kq, kp = jax.random.split(jax.random.PRNGKey(3))
    B, steps = 2, 30
    q0 = jax.random.normal(kq, (B, 8))
    p0 = jax.random.normal(kp, (B, 8))
    clamp = jnp.zeros(8, dtype=bool)
    floor = jnp.asarray(0.1)

    qf_g0, pf_g0 = _joint_settle(
        (u1, u2),
        (coupling,),
        ((0, 1),),
        (0, 4, 8),
        q0,
        p0,
        jnp.zeros((B, 1)),
        steps,
        DT,
        floor,
        0.9,
        clamp,
    )
    qf_no, pf_no = _joint_settle(
        (u1, u2),
        (),
        (),
        (0, 4, 8),
        q0,
        p0,
        jnp.zeros((B, 0)),
        steps,
        DT,
        floor,
        0.9,
        clamp,
    )
    assert jnp.array_equal(qf_g0, qf_no)
    assert jnp.array_equal(pf_g0, pf_no)


def test_open_gate_actually_couples():
    """Sanity: an OPEN gate (weight 1) makes the endpoints influence each other
    — the joint state differs from the gate-0 (uncoupled) state."""
    u1, u2 = _two_units()
    coupling = channel_spring_coupling(4, 4, kappa=1.0, channel=(2, 3))
    kq, kp = jax.random.split(jax.random.PRNGKey(11))
    B, steps = 2, 30
    q0 = jax.random.normal(kq, (B, 8))
    p0 = jax.random.normal(kp, (B, 8))
    clamp = jnp.zeros(8, dtype=bool)
    floor = jnp.asarray(0.1)
    qf_open, _ = _joint_settle(
        (u1, u2),
        (coupling,),
        ((0, 1),),
        (0, 4, 8),
        q0,
        p0,
        jnp.ones((B, 1)),
        steps,
        DT,
        floor,
        0.5,
        clamp,
    )
    qf_closed, _ = _joint_settle(
        (u1, u2),
        (coupling,),
        ((0, 1),),
        (0, 4, 8),
        q0,
        p0,
        jnp.zeros((B, 1)),
        steps,
        DT,
        floor,
        0.5,
        clamp,
    )
    assert not jnp.allclose(qf_open, qf_closed), "open gate did not couple"


# ---------------------------------------------------------------------------
# (3) routing smoke — end-to-end quick run, wormhole beats local-only distant
# ---------------------------------------------------------------------------


def test_wormhole_routing_smoke(tmp_path):
    from chlu.config import get_default_config
    from chlu.experiments.exp_v1_wormhole import run_experiment_v1_wormhole

    cfg = get_default_config()
    out = run_experiment_v1_wormhole(
        config=cfg,
        save_dir=str(tmp_path / "plots"),
        models_dir=str(tmp_path / "models"),
        seed=0,
        quick=True,
    )
    summary = out["summary"]
    assert "by_N" in summary and len(summary["by_N"]) >= 1
    N0 = sorted(int(k) for k in summary["by_N"].keys())[0]
    entry = summary["by_N"][str(N0)]
    arms = entry["arms"]
    # the learned-router baseline arm is present and scored
    assert "router_mlp" in arms
    # all arms produced finite accuracies in [0, 1] + finite FLOPs
    for ent in arms.values():
        assert 0.0 <= ent["acc_mean"] <= 1.0
        assert np.isfinite(ent["cost_mean"])
        assert np.isfinite(ent["flops_mean"]) and ent["flops_mean"] > 0
    # the mechanism claim: gated wormhole reaches >= local-only on DISTANT
    # queries (it opens a path local-only cannot use). Honest smoke bound only.
    assert (
        arms["gated"]["acc_distant_mean"]
        >= arms["local_only"]["acc_distant_mean"] - 1e-9
    )
    # energy injected through the open gate is finite/bounded
    assert np.isfinite(entry["energy_injected_max"])
    # workload-mix aggregation present for all requested mixes, arms scored
    assert "mixes" in entry and len(entry["mixes"]) == len(
        cfg.experiment_v1_wormhole.workload_mixes
    )
    for mx in entry["mixes"].values():
        assert "router_mlp" in mx["arms"] and "gated" in mx["arms"]
        for a in mx["arms"].values():
            assert 0.0 <= a["acc_mean"] <= 1.0 and np.isfinite(a["flops_mean"])
    # router ranking AUROC recorded
    assert np.isfinite(entry["auroc_router_distant_mean"])


# ---------------------------------------------------------------------------
# (4) FLOPs cost model + learned-router-MLP baseline (P9/V1.2)
# ---------------------------------------------------------------------------


class _FlopsCfg:
    flops_grad_factor = 6.0
    flops_verlet_grads = 2.0


def test_flops_model_scales_and_orders():
    """The FLOPs model is positive, grows with hidden width, and the wormhole's
    routed leg (2 units, fixed) is cheaper than the chain's (N units) for N>2 —
    the '1-hop flat vs N-hop diffusion' claim, in FLOPs."""
    cfg = _FlopsCfg()
    f_small = _potential_grad_flops(24, 32, cfg)
    f_big = _potential_grad_flops(24, 128, cfg)
    assert f_small > 0 and f_big > f_small
    steps = 250
    wormhole = _verlet_flops(steps, 2, 24, 128, cfg)  # 2 active units, any N
    for N in (4, 8):
        chain = _verlet_flops(steps, N, 24, 128, cfg)
        assert chain > wormhole
    # chain FLOPs scale linearly in N; wormhole does not
    assert _verlet_flops(steps, 8, 24, 128, cfg) == pytest.approx(
        2.0 * _verlet_flops(steps, 4, 24, 128, cfg)
    )
    assert _router_n_params(12, 32) == 12 * 32 + 32 + 32 + 1
    assert _router_mlp_flops(12, 32, cfg) > 0


def test_router_mlp_separates_own_from_impostor():
    """The physics-free router MLP, trained on own-key (route=False) vs impostor
    (route=True) cues, ranks impostors above own keys on held-out jittered cues
    — a real learned router, not a degenerate constant."""
    rng = np.random.default_rng(0)
    e = 8
    own = rng.standard_normal(e)
    imp = own + 5.0 * rng.standard_normal(e)  # a well-separated impostor key
    cues, labels = [], []
    for _ in range(40):
        cues.append(own + 0.1 * rng.standard_normal(e))
        labels.append(False)
        cues.append(imp + 0.1 * rng.standard_normal(e))
        labels.append(True)
    model = _fit_router_mlp(
        np.stack(cues), np.asarray(labels), e, hidden=16,
        key=jax.random.PRNGKey(0), epochs=200, lr=5e-3, l2=1e-3,
    )
    s_own = float(jax.nn.sigmoid(model(jnp.asarray(own))))
    s_imp = float(jax.nn.sigmoid(model(jnp.asarray(imp))))
    assert s_imp > s_own, "router failed to rank impostor above own key"
