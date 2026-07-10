"""CLU-lattice correctness tests (V3 first build; F5 §7).

The contract (task acceptance): the joint-Hamiltonian lattice must
  (1) reduce EXACTLY (bit-level) to N independent CHLUs at kappa_c = 0,
  (2) be jointly symplectic at gamma = 0 (||J^T Omega J - Omega|| ~ eps),
  (3) conserve the SIMULTANEOUS-rotation Noether charge at gamma = 0 and
      decay it exactly as (1-gamma)^n under friction,
  (4) reproduce the F5 §7.2 quadratic-order pricing law on a hand-built
      lattice (mu_rel^2 = 4 kappa_c / M; overdamped n_1/2 ∝ 1/kappa_c;
      shared channel = exact latch),
  (5) enforce the designed per-band causal caps v_max,i = c/sqrt(M_i)
      (F5 Prop-1).

Per the harness-test convention (test_goldstone.py): if these fail, the
lattice (or integrator) is wrong — not F5. x64 enabled at import,
process-global (same pattern and caveat as test_goldstone.py).
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import pytest  # noqa: E402

from chlu.core.chlu_unit import CHLU  # noqa: E402
from chlu.core.lattice import (  # noqa: E402
    CLULattice,
    GatedCoupling,
    MLPCoupling,
    SpringCoupling,
    build_lattice,
    chain_edges,
    channel_spring_coupling,
    scale_inertial_mass,
    spring_coupling,
    torus_edges,
)
from chlu.data.two_timescale_orbits import generate_two_timescale_orbits  # noqa: E402
from chlu.experiments.goldstone_harness import (  # noqa: E402
    MexicanHatPotential,
    classify_mode,
    clu_with_potential,
    half_life_first_crossing,
    latch_prediction,
    noether_charge,
    perturb_and_track,
    predicted_half_life,
    rollout_from,
    spectrum_probe,
    step_jacobian,
)

DT = 0.05


def _hat_pair(kappa: float, lam=1.0, f=1.0, inertia=1.0) -> CLULattice:
    """Hand-built 2-unit SO(2) lattice (the F5 §7.2 pricing geometry)."""
    hat = MexicanHatPotential(lam=lam, f=f, k_spec=None)
    units = [
        clu_with_potential(
            hat,
            dim=2,
            kinetic_mode="newtonian_learned",
            inertia=(inertia, inertia),
            key=jax.random.PRNGKey(i),
        )
        for i in range(2)
    ]
    coupling = channel_spring_coupling(2, 2, kappa, channel=(0, 1))
    return CLULattice(units=units, edges=((0, 1),), couplings=(coupling,))


def _pair_state(f: float, th1: float, th2: float) -> jnp.ndarray:
    return jnp.array(
        [f * math.cos(th1), f * math.sin(th1), f * math.cos(th2), f * math.sin(th2)]
    )


# ---------------------------------------------------------------------------
# (1) kappa_c = 0 reduction: bit-level identity with independent units
# ---------------------------------------------------------------------------


def test_kappa_zero_reduction_bitlevel():
    key = jax.random.PRNGKey(0)
    k1, k2, kc, kq, kp = jax.random.split(key, 5)
    # Heterogeneous dims + potentials to make the reduction non-trivial
    u1 = CHLU(
        dim=2, hidden=16, kinetic_mode="newtonian_learned", potential_type="mlp", key=k1
    )
    u2 = CHLU(
        dim=3, hidden=16, kinetic_mode="relativistic", potential_type="mlp", key=k2
    )

    q0 = jax.random.normal(kq, (5,))
    p0 = jax.random.normal(kp, (5,))
    steps = 50

    for label, lattice in [
        ("empty edge list", CLULattice(units=(u1, u2))),
        (
            "kappa=0 spring",
            CLULattice(
                units=(u1, u2),
                edges=((0, 1),),
                couplings=(spring_coupling(2, 3, kappa=0.0, coupling_dim=2, key=kc),),
            ),
        ),
    ]:
        for gamma in (0.0, 0.05):
            joint = lattice(q0, p0, steps=steps, dt=DT, gamma=gamma)
            t1 = u1(q0[:2], p0[:2], steps=steps, dt=DT, gamma=gamma)
            t2 = u2(q0[2:], p0[2:], steps=steps, dt=DT, gamma=gamma)
            # trajectory rows are [q_1, q_2, p_1, p_2]
            ref = jnp.concatenate([t1[:, :2], t2[:, :3], t1[:, 2:], t2[:, 3:]], axis=1)
            assert jnp.array_equal(joint, ref), (
                f"{label}, gamma={gamma}: joint step != independent units (bit-level)"
            )

    # H reduces to the sum of unit Hamiltonians (fp-additivity tolerance only)
    lattice = CLULattice(units=(u1, u2))
    h_joint = float(lattice.H(q0, p0))
    h_sum = float(u1.H(q0[:2], p0[:2]) + u2.H(q0[2:], p0[2:]))
    assert abs(h_joint - h_sum) < 1e-12 * max(1.0, abs(h_sum))


def test_chlu_T_delegation_bitlevel():
    """CHLU.H must equal CHLU.T + potential bit-level (the extraction that
    the lattice's separable T_net relies on)."""
    for mode in ("newtonian_identity", "newtonian_learned", "relativistic"):
        m = CHLU(dim=3, hidden=8, kinetic_mode=mode, key=jax.random.PRNGKey(7))
        q = jnp.array([0.3, -0.5, 0.9])
        p = jnp.array([1.5, -0.2, 0.4])
        assert float(m.H(q, p)) == float(m.T(p) + m.potential_net(q))


# ---------------------------------------------------------------------------
# (2) joint symplecticity at gamma = 0; per-unit-gamma exact volume law
# ---------------------------------------------------------------------------


def _omega(D: int) -> np.ndarray:
    return np.block([[np.zeros((D, D)), np.eye(D)], [-np.eye(D), np.zeros((D, D))]])


def test_joint_symplecticity_chain():
    for n in (2, 4):
        key = jax.random.PRNGKey(n)
        kb, kq, kp = jax.random.split(key, 3)
        lattice = build_lattice(
            kb,
            unit_dims=[2] * n,
            hidden=8,
            potential_type="mlp",
            kinetic_mode="newtonian_learned",
            edges=chain_edges(n),
            kappa_c=0.05,
        )
        D = lattice.dim
        q = 0.5 * jax.random.normal(kq, (D,))
        p = 0.5 * jax.random.normal(kp, (D,))
        J = np.asarray(step_jacobian(lattice, q, p, DT, gamma=0.0))
        err = np.max(np.abs(J.T @ _omega(D) @ J - _omega(D)))
        assert err < 1e-12, f"N={n}: ||J^T O J - O|| = {err:.3e}"


def test_per_unit_gamma_volume_law():
    """Heterogeneous damping (flagged path): det J = prod_i (1-gamma_i)^{d_i}
    exactly (F5 §7.2 condition 3), while uniform gamma keeps conformality."""
    lattice = _hat_pair(kappa=0.05)
    q = _pair_state(1.0, 0.3, -0.2)
    p = jnp.array([0.1, -0.2, 0.05, 0.15])

    gammas = (0.1, 0.3)
    gvec = lattice.gamma_vector(gammas)
    J = np.asarray(step_jacobian(lattice, q, p, DT, gamma=gvec))
    det_pred = (1 - gammas[0]) ** 2 * (1 - gammas[1]) ** 2
    assert abs(np.linalg.det(J) - det_pred) < 1e-12

    # uniform gamma: det J = (1-gamma)^D (conformal case)
    J_u = np.asarray(step_jacobian(lattice, q, p, DT, gamma=0.2))
    assert abs(np.linalg.det(J_u) - (1 - 0.2) ** 4) < 1e-12


# ---------------------------------------------------------------------------
# (3) Noether charge of the SIMULTANEOUS rotation
# ---------------------------------------------------------------------------


def _joint_charge(traj, dim):
    return np.asarray(
        noether_charge(traj, dim, channel=(0, 1))
        + noether_charge(traj, dim, channel=(2, 3))
    )


def test_joint_noether_simultaneous_rotation():
    lattice = _hat_pair(kappa=0.08)
    q0 = _pair_state(1.0, 0.4, -0.1)
    p0 = jnp.array([0.15, 0.3, -0.1, 0.2])

    # gamma = 0: joint charge conserved to machine precision
    traj = rollout_from(lattice, q0, p0, steps=2000, dt=DT, gamma=0.0)
    Q = _joint_charge(traj, lattice.dim)
    assert np.max(np.abs(Q - Q[0])) / abs(Q[0]) < 1e-10

    # ... while the SINGLE-unit charge is NOT conserved at kappa > 0:
    # the coupling moves charge between units (communication is charge flow)
    Q1 = np.asarray(noether_charge(traj, lattice.dim, channel=(0, 1)))
    assert np.max(np.abs(Q1 - Q1[0])) / max(abs(Q1[0]), 1e-12) > 1e-3

    # gamma > 0: exact geometric decay of the joint charge
    gamma = 0.05
    traj_g = rollout_from(lattice, q0, p0, steps=1500, dt=DT, gamma=gamma)
    Qg = _joint_charge(traj_g, lattice.dim)
    n = np.arange(len(Qg))
    err = np.max(np.abs(Qg - (1 - gamma) ** n * Qg[0])) / abs(Qg[0])
    assert err < 1e-9, f"joint Noether decay-law error {err:.3e}"


# ---------------------------------------------------------------------------
# (4) pricing-law smoke on the hand-built lattice (F5 §7.2 quadratic order)
# ---------------------------------------------------------------------------


def test_pricing_law_quadratic_order():
    lam, f, M = 1.0, 1.0, 1.0
    gamma = 0.2  # h*(0.2) = 0.111; both kappas far below => register band
    kappas = (0.01, 0.04)
    half_lives = {}

    for kappa in kappas:
        lattice = _hat_pair(kappa, lam=lam, f=f, inertia=M)
        q_star = _pair_state(f, 0.0, 0.0)
        probe = spectrum_probe(lattice, q_star)
        mu_sq = np.asarray(probe.mu_sq)

        # Exact joint channel spectrum at the synchronized vacuum:
        # {0, 4k/M, 8*lam*f^2/M, 8*lam*f^2/M + 4k/M}
        pred = np.array(
            [0.0, 4 * kappa / M, 8 * lam * f**2 / M, 8 * lam * f**2 / M + 4 * kappa / M]
        )
        assert np.allclose(mu_sq, pred, atol=1e-9), f"mu^2 = {mu_sq} vs {pred}"
        assert classify_mode(float(mu_sq[1]), DT, gamma) == "register"

        # Overdamped retention of the relative mode
        res = perturb_and_track(
            lattice,
            probe,
            mode_idx=1,
            kick=0.05,
            kick_type="position",
            steps=int(6 * predicted_half_life(float(mu_sq[1]), DT, gamma)) + 200,
            dt=DT,
            gamma=gamma,
        )
        n_meas = half_life_first_crossing(res["retention"])
        n_pred = predicted_half_life(float(mu_sq[1]), DT, gamma)
        assert abs(n_meas - n_pred) / n_pred < 0.02, (
            f"kappa={kappa}: n_1/2 measured {n_meas} vs exact {n_pred:.1f}"
        )
        half_lives[kappa] = n_meas

    # n_1/2 ∝ 1/kappa: kappa ratio 4 => half-life ratio 1/4
    ratio = half_lives[0.01] / half_lives[0.04]
    assert abs(ratio - 4.0) < 0.05, f"pricing ratio {ratio} != 4.0"


def test_shared_channel_latch_at_any_kappa():
    """(c) of the pricing claim: the diagonal channel is an exact latch at
    every kappa — momentum write freezes to d_inf = d0 + eps*pc0/gamma."""
    gamma, kick = 0.2, 0.05
    for kappa in (0.01, 0.3):
        lattice = _hat_pair(kappa)
        probe = spectrum_probe(lattice, _pair_state(1.0, 0.0, 0.0))
        assert abs(float(probe.mu_sq[0])) < 1e-10  # shared channel exactly flat
        res = perturb_and_track(
            lattice,
            probe,
            mode_idx=0,
            kick=kick,
            kick_type="momentum",
            steps=2000,
            dt=DT,
            gamma=gamma,
        )
        d = np.asarray(res["d"][:, 0])
        # THE claim: the written displacement FREEZES (deadbeat memory)
        assert abs(d[-1] - d[1000]) < 1e-12, f"kappa={kappa}: latch not frozen"
        # The linear latch prediction d_inf = d0 + eps*pc0/gamma holds up to
        # the curved-vacuum-manifold projection bias: the canonical mode
        # coordinate is a LINEAR projection onto the tangent eigendirection,
        # while the true latch lives on the circle — bias ~ |d_inf|^3/6
        # (~4e-7 here; the machine-exact latch on a genuinely linear flat
        # mode is asserted in test_goldstone.py::test_goldstone_latch_exact).
        pred = latch_prediction(d[0], kick, DT, gamma)
        cubic_bias = abs(pred) ** 3 / 6.0
        assert abs(d[-1] - pred) < 10.0 * cubic_bias, (
            f"kappa={kappa}: latch {d[-1]} vs {pred} "
            f"(err {abs(d[-1] - pred):.2e}, cubic scale {cubic_bias:.2e})"
        )


# ---------------------------------------------------------------------------
# (5) designed mass banding + per-band causal caps (F5 Prop-1)
# ---------------------------------------------------------------------------


def test_mass_banding_exact_and_causal_caps():
    key = jax.random.PRNGKey(3)
    scales = (4.0, 0.25)
    c = 2.0
    lattice = build_lattice(
        key,
        unit_dims=[2, 2],
        hidden=8,
        potential_type="mlp",
        kinetic_mode="relativistic",
        mass_scales=scales,
        kappa_c=0.05,
        c=c,
    )

    # Exact softplus-space banding: masses scaled by exactly the design ratio
    raw = build_lattice(
        jax.random.PRNGKey(3),
        unit_dims=[2, 2],
        hidden=8,
        potential_type="mlp",
        kinetic_mode="relativistic",
        mass_scales=None,
        kappa_c=0.05,
        c=c,
    )
    for u_b, u_r, s in zip(lattice.units, raw.units, scales, strict=True):
        assert np.allclose(
            np.asarray(u_b.mass_vector()),
            s * np.asarray(u_r.mass_vector()),
            rtol=1e-12,
        )

    # Anisotropic causal caps v_max,i = c / sqrt(M_i), per band (F5 Prop-1):
    # velocity = grad_p H saturates at the cap coordinate-wise
    caps = np.asarray(lattice.causal_caps())
    M = np.asarray(lattice.mass_vector())
    assert np.allclose(caps, c / np.sqrt(M + 1e-6), rtol=1e-12)

    q = jnp.zeros(lattice.dim)
    grad_p = jax.grad(lattice.H, argnums=1)
    for i in range(lattice.dim):
        p_huge = jnp.zeros(lattice.dim).at[i].set(1e8)
        v = np.asarray(grad_p(q, p_huge))
        assert abs(v[i]) <= caps[i] * (1 + 1e-9), (
            f"coord {i}: |v|={v[i]} > cap {caps[i]}"
        )
        assert abs(v[i]) > 0.999 * caps[i], (
            f"coord {i} not saturated: {v[i]} vs {caps[i]}"
        )
    # heavy band strictly slower than light band
    assert caps[0] < caps[2]

    # banding an identity-kinetic unit must fail loudly, not silently no-op
    u_id = CHLU(
        dim=2, hidden=4, kinetic_mode="newtonian_identity", key=jax.random.PRNGKey(0)
    )
    try:
        scale_inertial_mass(u_id, 2.0)
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# wormhole slot (skeleton) + coupling variants + training viability
# ---------------------------------------------------------------------------


def test_wormhole_distant_pair_couples():
    n = 4
    lattice = build_lattice(
        jax.random.PRNGKey(1),
        unit_dims=[2] * n,
        hidden=8,
        potential_type="mlp",
        kinetic_mode="newtonian_learned",
        edges=chain_edges(n),
        kappa_c=0.05,
        wormhole_edges=((0, n - 1),),
        wormhole_gate_threshold=1.0,
        wormhole_gate_width=0.25,
    )
    assert isinstance(lattice.couplings[-1], GatedCoupling)
    s0, s3 = lattice.unit_slice(0), lattice.unit_slice(n - 1)
    grad_V = jax.grad(lattice.V)

    # Aligned endpoints: moving unit 0 changes the force on distant unit 3
    q_a = (
        jnp.zeros(lattice.dim)
        .at[s0]
        .set(jnp.array([0.3, -0.2]))
        .at[s3]
        .set(jnp.array([0.3, -0.2]))
    )
    q_b = q_a.at[s0].set(jnp.array([0.8, 0.4]))
    dforce = float(jnp.max(jnp.abs(grad_V(q_a)[s3] - grad_V(q_b)[s3])))
    assert dforce > 1e-6, f"distant pair does not couple (dF = {dforce:.2e})"

    # Far apart: the smooth gate closes (sigmoid tail — exponentially small,
    # not exactly zero) and the wormhole energy is negligible vs the open
    # regime (~1e-2 here): assert several orders of suppression.
    gate = lattice.couplings[-1]
    v_open = float(gate(q_a[s0], q_a[s3] + 0.5))  # moderately displaced, open
    v_far = float(gate(jnp.array([50.0, 0.0]), jnp.array([-50.0, 0.0])))
    assert abs(v_far) < 1e-5, f"gate not closed at distance: V_wh = {v_far:.2e}"
    assert abs(v_far) < 1e-3 * max(abs(v_open), 1e-12), (
        f"insufficient gate suppression: far {v_far:.2e} vs open {v_open:.2e}"
    )


def test_mlp_coupling_and_gradient_flow():
    """Both coupling types must be position-only-callable and pass gradients
    into unit potentials, log_mass, AND coupling parameters (training
    viability of the joint Hamiltonian)."""
    for coupling_type in ("spring", "mlp"):
        lattice = build_lattice(
            jax.random.PRNGKey(5),
            unit_dims=[2, 2],
            hidden=8,
            potential_type="mlp",
            kinetic_mode="newtonian_learned",
            coupling_type=coupling_type,
            kappa_c=0.1,
        )
        if coupling_type == "mlp":
            assert isinstance(lattice.couplings[0], MLPCoupling)
        q0 = jnp.array([0.5, -0.3, 0.2, 0.4])
        p0 = jnp.array([0.1, 0.2, -0.1, 0.05])

        def loss_fn(m, q0=q0, p0=p0):  # bind loop vars (B023)
            traj = m(q0, p0, steps=10, dt=DT)
            return jnp.sum(traj**2)

        grads = eqx.filter_grad(loss_fn)(lattice)
        # Coupling parameters receive gradient. Exception that is correct
        # physics: the MLP coupling's FINAL bias adds a constant to H (gauge
        # — no force), so its gradient is legitimately zero; require every
        # weight matrix (ndim >= 2 leaf) to be touched instead.
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads.couplings, eqx.is_array))
        assert any(bool(jnp.any(g != 0)) for g in leaves), (
            f"{coupling_type}: no coupling gradient at all"
        )
        for g in leaves:
            if g.ndim >= 2:
                assert jnp.any(g != 0), f"{coupling_type}: zero weight gradient"
        # unit log_mass receives gradient
        for gu in grads.units:
            assert jnp.any(gu.log_mass != 0)


def test_lattice_duck_types_trainer_surfaces():
    """The surfaces train_chlu relies on: H, step, stochastic_step, __call__
    shapes; plus the two-timescale data generator layout."""
    lattice = build_lattice(
        jax.random.PRNGKey(2),
        unit_dims=[2, 2],
        hidden=8,
        potential_type="mlp",
        kinetic_mode="newtonian_learned",
        mass_scales=(4.0, 0.25),
        kappa_c=0.01,
    )
    D = lattice.dim
    data = generate_two_timescale_orbits(
        jax.random.PRNGKey(0),
        n_traj=3,
        seq_len=8,
        dt=DT,
        omegas=(0.5, 2.0),
        masses=(4.0, 0.25),
        radius=1.0,
    )
    assert data.shape == (3, 8, 2 * D)
    # p = M * dq/dt for the reference system: slow unit |p| = M R omega = 2.0
    p_slow = np.asarray(data[0, 0, D : D + 2])
    assert abs(np.linalg.norm(p_slow) - 4.0 * 1.0 * 0.5) < 1e-5

    q0, p0 = data[0, 0, :D], data[0, 0, D:]
    assert jnp.isfinite(lattice.H(q0, p0))
    qn, pn = lattice.step((q0, p0), DT, gamma=0.1)
    assert qn.shape == (D,) and pn.shape == (D,)
    qs, ps, _ = lattice.stochastic_step(
        (q0, p0), DT, gamma=0.1, temperature=0.5, key=jax.random.PRNGKey(1)
    )
    assert qs.shape == (D,) and jnp.all(jnp.isfinite(ps))
    traj = lattice(q0, p0, steps=7, dt=DT)
    assert traj.shape == (7, 2 * D)


def test_lattice_validation_errors():
    u = CHLU(
        dim=2, hidden=4, kinetic_mode="newtonian_learned", key=jax.random.PRNGKey(0)
    )
    cpl = channel_spring_coupling(2, 2, 0.1)
    for bad_edges, bad_couplings in [
        (((0, 0),), (cpl,)),  # self-loop
        (((0, 2),), (cpl,)),  # out of range
        (((0, 1),), ()),  # edge/coupling misalignment
    ]:
        try:
            CLULattice(units=(u, u), edges=bad_edges, couplings=bad_couplings)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"no error for edges={bad_edges}"


# ---------------------------------------------------------------------------
# (6) the annealed-Ising gate: FORCE, not just energy (xy-lattice-theory §6.2)
#
# v3-lattice-build asserted only that the gate SUPPRESSES ENERGY at distance.
# That is exactly why the legacy `<sigma> * v` form's sign-reversing force went
# unnoticed. These tests assert the force.
# ---------------------------------------------------------------------------


def _gate_force_vs_v(gate: GatedCoupling, vs):
    """dV_wh/dv along the channel-0 separation ray (v = kappa * s^2)."""

    def V_of_v(v):
        s = jnp.sqrt(v)
        return gate(jnp.zeros(2), jnp.array([s, 0.0]))

    dV = jax.jit(jax.grad(V_of_v))
    return np.array([float(dV(v)) for v in vs])


def test_gate_free_energy_force_monotone_and_bounded():
    """The DEFAULT (free-energy) gate: V_wh is monotone increasing in v, its
    force dV/dv equals the annealed occupancy <sigma> and is bounded in [0, 1]
    (always attractive, never stronger than the ungated coupling), and V_wh is
    bounded below by -w*ln(1 + e^{t/w})."""
    t, w = 1.0, 0.25
    base = channel_spring_coupling(2, 2, kappa=1.0, channel=(0, 1))
    gate = GatedCoupling(base=base, threshold=t, width=w)  # default mode
    assert gate.energy_mode == "free_energy"

    vs = np.linspace(1e-9, 5.0, 501)
    force = _gate_force_vs_v(gate, vs)

    # (a) force is the transmitted fraction: bounded in [0, 1], never negative
    assert force.min() >= 0.0, f"gate force went negative: {force.min():.6f}"
    assert force.max() <= 1.0 + 1e-12, f"gate force exceeds 1: {force.max():.6f}"

    # (b) force == <sigma> exactly (the annealed mean force)
    occ = np.array(
        [float(gate.occupancy(jnp.zeros(2), jnp.array([np.sqrt(v), 0.0]))) for v in vs]
    )
    assert np.max(np.abs(force - occ)) < 1e-12

    # (c) V_wh monotone increasing in v (attraction: energy grows with
    #     separation) and bounded in [-w*ln(1+e^{t/w}), 0)
    V = np.array([float(gate(jnp.zeros(2), jnp.array([np.sqrt(v), 0.0]))) for v in vs])
    assert np.all(np.diff(V) > 0), "free-energy gate potential is not monotone"
    lower = -w * math.log1p(math.exp(t / w))
    assert V.min() >= lower - 1e-12 and V.max() < 0.0
    assert abs(float(gate(jnp.zeros(2), jnp.zeros(2))) - lower) < 1e-12


def test_gate_mean_energy_legacy_force_reverses_sign():
    """The LEGACY mode is retained (nothing silently deleted) — and it is
    retained BROKEN, on purpose. Pin the defect so nobody re-defaults to it:
    its force reverses sign at v = 0.8020 (inside the nominally-open region
    v < t = 1.0), reaching -0.7181."""
    gate = GatedCoupling(
        base=channel_spring_coupling(2, 2, kappa=1.0, channel=(0, 1)),
        threshold=1.0,
        width=0.25,
        energy_mode="mean_energy",
    )
    vs = np.linspace(1e-9, 3.0, 3001)
    force = _gate_force_vs_v(gate, vs)
    assert force.min() < 0.0, "legacy gate force should (still) reverse sign"
    assert abs(force.min() - (-0.718078)) < 1e-4, force.min()
    # first sign change strictly inside the open region v < threshold
    first_neg = vs[np.argmax(force < 0.0)]
    assert 0.79 < first_neg < 0.81, first_neg
    assert first_neg < gate.threshold

    # legacy expression is preserved bit-for-bit
    for v in (0.1, 0.802, 2.5):
        q = jnp.array([np.sqrt(v), 0.0])
        expect = jax.nn.sigmoid((1.0 - v) / 0.25) * v
        assert abs(float(gate(jnp.zeros(2), q)) - float(expect)) < 1e-12


def test_gate_on_xy_bond_stays_ferromagnetic():
    """On an XY bond v(dtheta) = J(1 - cos dtheta) with (J, t, w) =
    (0.1, 0.05, 0.02): the free-energy gate keeps the exchange FERROMAGNETIC
    (J1 = +0.0223); the legacy gate flips it ANTIFERROMAGNETIC (J1 = -0.0072),
    i.e. a wormhole array on the legacy gate frustrates itself
    (xy-lattice-theory §6.2)."""
    kappa = 0.05  # J = 2 kappa r*^2 = 0.1 at r* = 1
    base = channel_spring_coupling(2, 2, kappa=kappa, channel=(0, 1))
    n_grid = 2048
    th = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)

    def harmonics(fn):
        V = np.array(
            [
                float(fn(jnp.array([1.0, 0.0]), jnp.array([np.cos(a), np.sin(a)])))
                for a in th
            ]
        )
        # V = const - sum_n J_n cos(n dtheta)  =>  J_n = -(2/N) sum_k V_k cos(n th_k)
        return [-(2.0 / n_grid) * float(np.sum(V * np.cos(n * th))) for n in (1, 2, 3)]

    j_ungated = harmonics(base)
    assert abs(j_ungated[0] - 0.1) < 1e-12  # pure first harmonic, J = 2*kappa
    assert max(abs(j_ungated[1]), abs(j_ungated[2])) < 1e-12

    kw = dict(base=base, threshold=0.05, width=0.02)
    j_free = harmonics(GatedCoupling(**kw, energy_mode="free_energy"))
    j_legacy = harmonics(GatedCoupling(**kw, energy_mode="mean_energy"))

    assert j_free[0] > 0.0, f"free-energy gate is antiferromagnetic: {j_free[0]}"
    assert abs(j_free[0] - 0.0223) < 5e-4, j_free[0]
    assert j_legacy[0] < 0.0, "legacy gate should (still) be antiferromagnetic"
    assert abs(j_legacy[0] - (-0.0072)) < 5e-4, j_legacy[0]


def test_build_lattice_gate_energy_mode_flag():
    """build_lattice defaults to the free-energy gate and can still construct
    the legacy one; unknown modes fail loudly."""

    def build(**kw):
        return build_lattice(
            jax.random.PRNGKey(1),
            unit_dims=[2] * 4,
            hidden=8,
            potential_type="mlp",
            kinetic_mode="newtonian_learned",
            edges=chain_edges(4),
            kappa_c=0.05,
            wormhole_edges=((0, 3),),
            **kw,
        )

    assert build().couplings[-1].energy_mode == "free_energy"
    assert build(gate_energy_mode="mean_energy").couplings[-1].energy_mode == (
        "mean_energy"
    )
    with pytest.raises(ValueError):
        build(gate_energy_mode="mean")


# ---------------------------------------------------------------------------
# (7) U(1)-preserving coupling (xy-lattice-theory P5)
# ---------------------------------------------------------------------------


def test_conformal_init_equals_channel_spring_and_is_trainable():
    """`spring_coupling(init_mode="conformal")` starts at W = 1_k — exactly the
    channel_spring geometry (so the reduction is exactly XY at init) — but its
    W are still LEARNABLE leaves that receive gradient."""
    conf = spring_coupling(2, 2, kappa=0.05, coupling_dim=2, init_mode="conformal")
    chan = channel_spring_coupling(2, 2, kappa=0.05, channel=(0, 1))
    assert np.array_equal(np.asarray(conf.W_i), np.asarray(chan.W_i))
    assert np.array_equal(np.asarray(conf.W_j), np.asarray(chan.W_j))

    # identical energies on random states
    k = jax.random.PRNGKey(0)
    for _ in range(5):
        k, k1, k2 = jax.random.split(k, 3)
        qi, qj = jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))
        assert abs(float(conf(qi, qj)) - float(chan(qi, qj))) < 1e-15

    # spectator coords: conformal embeds the CHANNEL only (eye(k, d))
    conf4 = spring_coupling(4, 4, kappa=0.05, coupling_dim=2, init_mode="conformal")
    assert np.allclose(np.asarray(conf4.W_i), np.eye(2, 4))

    # still trainable: W receives gradient
    def loss(c):
        return c(jnp.array([0.3, -0.2]), jnp.array([0.9, 0.4])) ** 2

    g = eqx.filter_grad(loss)(conf)
    assert jnp.any(g.W_i != 0) and jnp.any(g.W_j != 0)

    # too-wide coupling_dim fails loudly; bad mode fails loudly
    with pytest.raises(ValueError):
        spring_coupling(2, 2, 0.05, coupling_dim=3, init_mode="conformal")
    with pytest.raises(ValueError):
        spring_coupling(2, 2, 0.05, init_mode="identity")


def test_coupling_type_auto_defaults_by_potential_type():
    """ "auto" = channel_spring for so2_invariant (U(1)-preserving design rule),
    spring otherwise. The non-so2 path must be BIT-IDENTICAL to the old
    default (`coupling_type="spring"`, random W) — no shipped lattice moves."""
    kw = dict(
        unit_dims=[2, 2], hidden=8, kinetic_mode="newtonian_learned", kappa_c=0.05
    )

    # so2_invariant + auto -> fixed identity-on-channel projections
    so2 = build_lattice(jax.random.PRNGKey(4), potential_type="so2_invariant", **kw)
    c = so2.couplings[0]
    assert isinstance(c, SpringCoupling)
    assert np.array_equal(np.asarray(c.W_i), np.eye(2, 2))
    assert np.array_equal(np.asarray(c.W_j), np.eye(2, 2))

    # mlp + auto == mlp + explicit "spring", bit-for-bit (legacy default)
    auto = build_lattice(jax.random.PRNGKey(4), potential_type="mlp", **kw)
    legacy = build_lattice(
        jax.random.PRNGKey(4), potential_type="mlp", coupling_type="spring", **kw
    )
    assert np.array_equal(
        np.asarray(auto.couplings[0].W_i), np.asarray(legacy.couplings[0].W_i)
    )
    assert np.array_equal(
        np.asarray(auto.couplings[0].W_j), np.asarray(legacy.couplings[0].W_j)
    )
    q = jnp.array([0.4, -0.1, 0.2, 0.7])
    p = jnp.array([0.1, 0.0, -0.3, 0.2])
    assert float(auto.H(q, p)) == float(legacy.H(q, p))

    with pytest.raises(ValueError):
        build_lattice(
            jax.random.PRNGKey(4), potential_type="mlp", coupling_type="bogus", **kw
        )


def test_channel_spring_lattice_is_exactly_xy_on_the_vacuum_torus():
    """coupling_type="channel_spring" gives V_c = 2 kappa r*^2 (1 - cos dtheta):
    a PURE first harmonic with J = 2 kappa r*^2 — no p=2 anisotropy, no
    U(1)-breaking cos(theta_i + theta_j) term (xy-lattice-theory Prop-1)."""
    kappa, r = 0.05, 1.0
    lat = build_lattice(
        jax.random.PRNGKey(11),
        unit_dims=[2, 2],
        hidden=8,
        potential_type="so2_invariant",
        kinetic_mode="newtonian_learned",
        kappa_c=kappa,
    )
    coupling = lat.couplings[0]

    n_grid = 64
    th = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    Vc = np.array(
        [
            [
                float(
                    coupling(
                        jnp.array([r * np.cos(a), r * np.sin(a)]),
                        jnp.array([r * np.cos(b), r * np.sin(b)]),
                    )
                )
                for b in th
            ]
            for a in th
        ]
    )
    F = np.fft.fft2(Vc) / n_grid**2
    # exchange J from the (1, -1) mode: V_c = const - J cos(th_i - th_j)
    J = -2.0 * float(np.real(F[1, -1]))
    assert abs(J - 2.0 * kappa * r**2) < 1e-12, J
    # p=2 single-site anisotropy and the U(1)-breaking (1, 1) mode are ZERO
    assert abs(F[2, 0]) < 1e-14 and abs(F[0, 2]) < 1e-14  # h_2
    assert abs(F[1, 1]) < 1e-14  # cos(th_i + th_j)


def test_so2_with_random_spring_warns_about_u1_breaking():
    with pytest.warns(UserWarning, match="break"):
        build_lattice(
            jax.random.PRNGKey(6),
            unit_dims=[2, 2],
            hidden=8,
            potential_type="so2_invariant",
            kinetic_mode="newtonian_learned",
            coupling_type="spring",  # random W on an SO(2) lattice
            kappa_c=0.05,
        )
    # conformal init, channel_spring, and auto are all silent
    import warnings as _warnings

    for kw in (
        dict(coupling_type="spring", proj_init_mode="conformal"),
        dict(coupling_type="channel_spring"),
        dict(coupling_type="auto"),
    ):
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            build_lattice(
                jax.random.PRNGKey(6),
                unit_dims=[2, 2],
                hidden=8,
                potential_type="so2_invariant",
                kinetic_mode="newtonian_learned",
                kappa_c=0.05,
                **kw,
            )
    # non-so2 lattices keep the old default silently
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        build_lattice(
            jax.random.PRNGKey(6),
            unit_dims=[2, 2],
            hidden=8,
            potential_type="mlp",
            kinetic_mode="newtonian_learned",
            coupling_type="spring",
            kappa_c=0.05,
        )


# ---------------------------------------------------------------------------
# (8) 2-D torus edge list (xy-lattice-theory §7.2)
# ---------------------------------------------------------------------------


def test_torus_edges_degree_four_no_duplicates():
    for L in (3, 4, 8):
        edges = torus_edges(L)
        n = L * L
        assert len(edges) == 2 * n, f"L={L}: {len(edges)} edges, want {2 * n}"
        assert all(i != j for i, j in edges), f"L={L}: self-loop"
        undirected = {frozenset(e) for e in edges}
        assert len(undirected) == 2 * n, f"L={L}: duplicate bonds"
        deg = [0] * n
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        assert set(deg) == {4}, f"L={L}: degrees {sorted(set(deg))} != {{4}}"
        assert all(0 <= i < n and 0 <= j < n for i, j in edges)

    # L = 2 is a genuine MULTIgraph (the +x and -x neighbour coincide): the
    # honest periodic lattice has 8 bonds over 4 distinct pairs. Raise unless
    # the caller opts in, so nobody silently builds a degree-2 "torus".
    with pytest.raises(ValueError, match="degenerate"):
        torus_edges(2)
    doubled = torus_edges(2, allow_double_bonds=True)
    assert len(doubled) == 8
    assert len({frozenset(e) for e in doubled}) == 4
    assert all(i != j for i, j in doubled)
    with pytest.raises(ValueError):
        torus_edges(1)


def test_torus_lattice_builds_and_steps():
    """The torus edge list drops straight into build_lattice (no new lattice
    code): an L=3 (N=9) SO(2) torus builds, is finite, and steps."""
    L = 3
    lat = build_lattice(
        jax.random.PRNGKey(9),
        unit_dims=[2] * (L * L),
        hidden=8,
        potential_type="so2_invariant",
        kinetic_mode="newtonian_learned",
        edges=torus_edges(L),
        kappa_c=0.05,
    )
    assert lat.n_units == L * L and lat.dim == 2 * L * L
    assert len(lat.edges) == 2 * L * L
    # auto -> channel_spring on so2 units (no U(1) breaking anywhere)
    assert all(np.array_equal(np.asarray(c.W_i), np.eye(2, 2)) for c in lat.couplings)
    kq, kp = jax.random.split(jax.random.PRNGKey(0))
    q0 = 0.5 * jax.random.normal(kq, (lat.dim,))
    p0 = 0.5 * jax.random.normal(kp, (lat.dim,))
    assert jnp.isfinite(lat.H(q0, p0))
    traj = lat(q0, p0, steps=5, dt=DT)
    assert traj.shape == (5, 2 * lat.dim) and bool(jnp.all(jnp.isfinite(traj)))
