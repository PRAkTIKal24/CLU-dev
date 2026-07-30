"""Tests for the C2W2 Route-2 shell atom (charter §A4.1/§A4.2).

The load-bearing one is :func:`test_r0_gate_value_grad_hessian_bit_identical`
and its written-store sibling: the charter makes the ``r = 0 ⇒ exact Gaussian
reduction`` a **mandatory regression gate**, and the task file makes it blocking
— "assert bit-identical ``V``, ``∇V``, ``Hess V`` and a bit-identical written
store ... as a **test**, not as a claim in prose". So these assert ``== 0.0``
exactly, with no ``allclose`` anywhere in the gate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.memory_potentials import (
    AtomDictionaryPotential,
    atom_write_mask_fn,
)
from chlu.core.shell_atoms import (
    ShellAtomDictionaryPotential,
    shell_hessian_spectrum,
    shell_potential_from,
    shell_write_mask_fn,
)
from chlu.training.train_memory import train_memory_landscape


@pytest.fixture(autouse=True)
def float32_dynamics():
    """Pin float32 for the module, restoring the global flag after (handover §7.2:
    other modules enable x64 at import, so the flag leaks in a full-suite run)."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


DIM, N_ATOMS, N_GROUPS = 6, 24, 4


def _pair(radius_scale=0.0, tilt_eps=0.0, seed=0, dim=DIM, n_atoms=N_ATOMS,
          r_init=0.5, spread=1.0):
    """A Gaussian store and the shell store built from the SAME arrays."""
    key = jax.random.PRNGKey(seed)
    g = AtomDictionaryPotential(
        dim, n_atoms, key, init_scale=spread, init_width=0.3,
        confine=0.05, depth_init=1e-4, n_groups=N_GROUPS,
    )
    # give the atoms real depth so the gate is not tested on a flat landscape
    g = eqx.tree_at(lambda t: t.amp, g, jnp.linspace(0.2, 1.1, n_atoms))
    s = ShellAtomDictionaryPotential.from_gaussian(
        g, radius_scale=radius_scale, r_init=r_init, tilt_eps=tilt_eps,
    )
    return g, s


def _wrap(atoms):
    """Wrap bare atoms in the ``DesignFreedomPotential`` shell the writer needs.

    ⚠ ``train_memory.trainable_filter`` keys off ``V.learned`` and returns
    ``None`` (i.e. trains NOTHING and returns ``V`` unchanged) for a bare atom
    module — so a write test on bare atoms is vacuously green. Every write test
    below goes through this wrapper for that reason.
    """
    from chlu.core.memory_potentials import DesignFreedomPotential

    V = DesignFreedomPotential(
        rung="free_mlp", dim=int(atoms.centers.shape[1]),
        payloads=jnp.zeros((int(atoms.n_groups),)), key=jax.random.PRNGKey(0),
        learned_family="atoms", n_atoms=int(atoms.centers.shape[0]),
        atom_groups=int(atoms.n_groups), atom_depth_init=1e-4,
    )
    return eqx.tree_at(lambda t: t.learned, V, replace=atoms)


# ----------------------------------------------------------------------
# ⛔ the r = 0 regression gate (blocking)
# ----------------------------------------------------------------------
def test_r0_gate_value_grad_hessian_bit_identical():
    g, s = _pair(radius_scale=0.0)
    qs = jax.random.normal(jax.random.PRNGKey(7), (64, DIM)) * 1.2
    qs = jnp.concatenate([qs, g.centers[:4]], axis=0)  # include q == c exactly

    vg = jax.vmap(g)(qs)
    vs = jax.vmap(s)(qs)
    assert np.max(np.abs(np.asarray(vs) - np.asarray(vg))) == 0.0

    gg = jax.vmap(jax.grad(lambda q: g(q)))(qs)
    gs = jax.vmap(jax.grad(lambda q: s(q)))(qs)
    assert np.all(np.isfinite(np.asarray(gs)))
    assert np.max(np.abs(np.asarray(gs) - np.asarray(gg))) == 0.0

    hg = jax.vmap(jax.hessian(lambda q: g(q)))(qs)
    hs = jax.vmap(jax.hessian(lambda q: s(q)))(qs)
    assert np.all(np.isfinite(np.asarray(hs)))
    assert np.max(np.abs(np.asarray(hs) - np.asarray(hg))) == 0.0


def test_r0_gate_holds_in_float64():
    """The gate is an IEEE754 statement, so it must not depend on the dtype."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        g, s = _pair(radius_scale=0.0, seed=3)
        qs = jax.random.normal(jax.random.PRNGKey(11), (32, DIM))
        assert np.max(np.abs(np.asarray(jax.vmap(s)(qs))
                             - np.asarray(jax.vmap(g)(qs)))) == 0.0
    finally:
        jax.config.update("jax_enable_x64", was)


def test_r0_gate_written_store_bit_identical():
    """A masked write must leave the two stores bit-identical (the charter's
    "bit-identical written store" leg of the gate)."""
    g, s = _pair(radius_scale=0.0, seed=1, spread=0.35)
    targets = jnp.asarray(
        np.stack([np.linspace(-0.6, 0.6, DIM) * (1 + i) * 0.3 for i in range(2)])
    ).astype(jnp.float32)
    key = jax.random.PRNGKey(5)
    rows = g.group_rows(0)
    kw = dict(steps=60, lr=3e-3, weight_decay=1e-4,
              loss_kwargs=dict(n_perturb=8, payload_index=4, barrier_pairs="nn"))
    gt, hg = train_memory_landscape(
        _wrap(g), targets, key, update_mask_fn=atom_write_mask_fn(rows), **kw)
    st, hs = train_memory_landscape(
        _wrap(s), targets, key, update_mask_fn=shell_write_mask_fn(rows), **kw)

    assert len(hg) == 60 and hg[-1] != hg[0]  # the write actually did something
    for name in ("centers", "log_width", "amp"):
        a = np.asarray(getattr(gt.learned, name))
        b = np.asarray(getattr(st.learned, name))
        assert np.max(np.abs(a - b)) == 0.0, name
        assert np.max(np.abs(a - np.asarray(getattr(g, name)))) > 0.0, name
    assert np.max(np.abs(np.asarray(hg) - np.asarray(hs))) == 0.0


def test_r0_gate_holds_on_the_ARITHMETIC_path_too():
    """The strong form of the gate: not the static short-circuit, but the shell
    arithmetic *running* with a numerically exact zero radius.

    ``softplus(−1000) == 0.0`` exactly in float32 (``logaddexp`` underflows), so
    this exercises ``d2 − 2ρr + r²`` with ``r ≡ 0`` and asserts the IEEE754
    claim rather than the code path.
    """
    g, _ = _pair(radius_scale=0.0, seed=1)
    s = ShellAtomDictionaryPotential.from_gaussian(g, radius_scale=1.0, r_init=0.5)
    s = eqx.tree_at(lambda t: t.radius_raw, s,
                    jnp.full_like(s.radius_raw, -1000.0))
    assert float(jnp.max(s.radii())) == 0.0  # exactly, not approximately
    qs = jnp.concatenate(
        [jax.random.normal(jax.random.PRNGKey(21), (32, DIM)), g.centers[:4]])
    for fn in (lambda m: jax.vmap(m),
               lambda m: jax.vmap(jax.grad(lambda q: m(q))),
               lambda m: jax.vmap(jax.hessian(lambda q: m(q)))):
        a, b = np.asarray(fn(g)(qs)), np.asarray(fn(s)(qs))
        assert np.all(np.isfinite(b))
        assert np.max(np.abs(a - b)) == 0.0


def test_softplus_radius_recovers_gaussian_in_the_limit():
    """P1e: with radius_scale = 1 a tiny radius is *near*, not exactly, Gaussian."""
    key = jax.random.PRNGKey(2)
    g = AtomDictionaryPotential(DIM, N_ATOMS, key, n_groups=N_GROUPS)
    s = ShellAtomDictionaryPotential.from_gaussian(g, radius_scale=1.0, r_init=1e-4)
    q = jax.random.normal(jax.random.PRNGKey(9), (DIM,))
    assert float(s(q)) == pytest.approx(float(g(q)), rel=1e-5)
    assert float(np.max(np.asarray(s.radii()))) == pytest.approx(1e-4, rel=1e-3)


def test_shell_minimum_set_is_a_sphere():
    """The designed degeneracy: V is constant on ‖q − c‖ = r, and that is the
    property the Gaussian atom does not have."""
    dim, r = 3, 0.5
    key = jax.random.PRNGKey(0)
    g = AtomDictionaryPotential(dim, 1, key, n_groups=1, confine=0.0)
    g = eqx.tree_at(lambda t: [t.centers, t.amp, t.log_width], g,
                    replace=[jnp.zeros((1, dim)), jnp.ones((1,)),
                             jnp.full((1,), float(np.log(0.3)))])
    s = ShellAtomDictionaryPotential.from_gaussian(g, radius_scale=1.0, r_init=r)
    th = np.linspace(0, 2 * np.pi, 17)
    pts = jnp.asarray(np.stack([r * np.cos(th), r * np.sin(th),
                                np.zeros_like(th)], axis=-1), dtype=jnp.float32)
    vals = np.asarray(jax.vmap(s)(pts))
    assert np.ptp(vals) < 1e-6
    assert vals.mean() == pytest.approx(-1.0, abs=1e-5)
    # The Gaussian's minimum is the single point q = c; the shell's is the whole
    # sphere and q = c is a local MAXIMUM (V(c) = −A e^{−r²/2s²} ≈ 0).
    assert float(g(jnp.zeros((dim,)))) == pytest.approx(-1.0, abs=1e-5)
    assert float(s(jnp.zeros((dim,)))) == pytest.approx(
        -float(np.exp(-(r**2) / (2 * 0.3**2))), rel=1e-4)  # = −0.2494
    assert float(s(jnp.zeros((dim,)))) > float(vals.mean()) + 0.5
    # and the settled manifold has dimension 2 (a 2-sphere in 3-D)
    w, _, _ = shell_hessian_spectrum(s, pts[0])
    assert abs(float(w[0])) < 1e-3 and abs(float(w[1])) < 1e-3
    assert float(w[2]) > 1.0


# ----------------------------------------------------------------------
# the tilt dial (charter §A4.2)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("eps", [1e-3, 1e-2, 1e-1, 1.0])
def test_tilt_gives_lambda_soft_equal_to_epsilon(eps):
    """⭐ P2a: ``Hess V_tilt = ε ûûᵀ`` at an on-shell point with ``û·(q−c) = 0``,
    so the constant of proportionality between λ_soft and ε is 1 BY CONSTRUCTION
    — which is what makes charter §A4.2's ``λ_min = ε`` falsifiable."""
    dim, r = 3, 0.5
    key = jax.random.PRNGKey(0)
    g = AtomDictionaryPotential(dim, 1, key, n_groups=1, confine=0.0)
    g = eqx.tree_at(lambda t: [t.centers, t.amp, t.log_width], g,
                    replace=[jnp.zeros((1, dim)), jnp.ones((1,)),
                             jnp.full((1,), float(np.log(0.3)))])
    s = ShellAtomDictionaryPotential.from_gaussian(
        g, radius_scale=1.0, r_init=r, tilt_eps=eps)
    # tilt direction defaults to the LAST axis; sit on the shell perpendicular
    # to it, i.e. û·(q − c) = 0.
    z = jnp.asarray([r, 0.0, 0.0], dtype=jnp.float32)
    w, _, info = shell_hessian_spectrum(s, z, tilt_dir=np.asarray(s.tilt_dir[0]),
                                        axis=dim - 1)
    # the tilted axis carries exactly eps; the OTHER tangent stays flat
    assert float(w[1]) == pytest.approx(eps, rel=0.05, abs=1e-6)
    assert abs(float(w[0])) < 1e-4
    assert info["lambda_max"] > 1.0  # the radial mode is massive


def test_tilt_eps_zero_is_bit_identical_to_no_tilt():
    g, s0 = _pair(radius_scale=1.0, tilt_eps=0.0, seed=4)
    assert s0.tilt_dir is None
    q = jax.random.normal(jax.random.PRNGKey(13), (DIM,))
    s1 = ShellAtomDictionaryPotential.from_gaussian(
        g, radius_scale=1.0, r_init=0.5, tilt_eps=0.0)
    assert float(s0(q)) == float(s1(q))
    assert s0.byte_ledger()["tilt_bytes"] == 0


# ----------------------------------------------------------------------
# locality, ledger, plumbing
# ----------------------------------------------------------------------
def test_write_mask_keeps_foreign_groups_bit_identical():
    """C3 locality: the shipped mask covers 3 leaves; a shell store has 5, and
    leaving the extra two unmasked would let a write for item i move EVERY
    item's radius."""
    _, s = _pair(radius_scale=1.0, tilt_eps=1e-2, seed=6, spread=0.35)
    targets = jnp.asarray(np.linspace(-0.5, 0.5, DIM)[None, :], dtype=jnp.float32)
    rows = np.asarray(s.group_rows(1), dtype=bool)
    out, _ = train_memory_landscape(
        _wrap(s), targets, jax.random.PRNGKey(3), steps=25,
        loss_kwargs=dict(n_perturb=4, payload_index=4),
        update_mask_fn=shell_write_mask_fn(jnp.asarray(rows)),
    )
    out = out.learned
    for name in ("centers", "log_width", "amp", "radius_raw"):
        a = np.asarray(getattr(s, name))
        b = np.asarray(getattr(out, name))
        assert np.max(np.abs(a[~rows] - b[~rows])) == 0.0, f"{name} leaked"
        assert np.max(np.abs(a[rows] - b[rows])) > 0.0, f"{name} did not move"
    # only group 1's tilt direction may move
    td_a, td_b = np.asarray(s.tilt_dir), np.asarray(out.tilt_dir)
    moved = np.max(np.abs(td_a - td_b), axis=-1) > 0.0
    assert moved[1] and not moved[[0, 2, 3]].any()


def test_shipped_mask_would_leak_the_radius():
    """⚠ The reason :func:`shell_write_mask_fn` exists: the SHIPPED mask covers
    three leaves, so with it a write for item 1 moves every item's radius."""
    _, s = _pair(radius_scale=1.0, seed=6, spread=0.35)
    targets = jnp.asarray(np.linspace(-0.5, 0.5, DIM)[None, :], dtype=jnp.float32)
    rows = np.asarray(s.group_rows(1), dtype=bool)
    out, _ = train_memory_landscape(
        _wrap(s), targets, jax.random.PRNGKey(3), steps=25,
        loss_kwargs=dict(n_perturb=4, payload_index=4),
        update_mask_fn=atom_write_mask_fn(jnp.asarray(rows)),
    )
    leak = np.max(np.abs(np.asarray(out.learned.radius_raw)[~rows]
                         - np.asarray(s.radius_raw)[~rows]))
    assert leak > 0.0


def test_freeze_radius_pins_the_designed_degeneracy():
    """The ``shell_fixed`` arm: designed radius, learned placement (w20 doctrine)."""
    _, s = _pair(radius_scale=1.0, seed=8, spread=0.35)
    targets = jnp.asarray(np.linspace(-0.5, 0.5, DIM)[None, :], dtype=jnp.float32)
    rows = jnp.asarray(np.asarray(s.group_rows(0), dtype=bool))
    out, _ = train_memory_landscape(
        _wrap(s), targets, jax.random.PRNGKey(3), steps=25,
        loss_kwargs=dict(n_perturb=4, payload_index=4),
        update_mask_fn=shell_write_mask_fn(rows, freeze_radius=True),
    )
    assert np.max(np.abs(np.asarray(out.learned.radius_raw)
                         - np.asarray(s.radius_raw))) == 0.0
    assert np.max(np.abs(np.asarray(out.learned.centers)
                         - np.asarray(s.centers))) > 0.0


def test_byte_ledger_matches_the_declared_overhead():
    """P7a: the shell surcharge is exactly 1/(dim+2) of the atom bytes."""
    _, s = _pair(radius_scale=1.0, tilt_eps=0.0)
    led = s.byte_ledger()
    assert led["gauss_bytes"] == N_ATOMS * (DIM + 2) * 4
    assert led["shell_bytes"] == N_ATOMS * 4
    assert led["overhead_frac"] == pytest.approx(1.0 / (DIM + 2), rel=1e-9)
    _, st = _pair(radius_scale=1.0, tilt_eps=1e-2)
    assert st.byte_ledger()["tilt_bytes"] == N_GROUPS * DIM * 4


def test_n_bytes_counts_the_shell_and_the_tilt():
    """The harness's byte ledger walks inexact leaves, so the surcharge must be
    visible to it automatically — no arm may under-report its bytes."""
    _, s = _pair(radius_scale=1.0, tilt_eps=1e-2)
    leaves = jax.tree_util.tree_leaves(eqx.filter(s, eqx.is_inexact_array))
    total = int(sum(int(np.asarray(x).size) for x in leaves) * 4)
    assert total == s.byte_ledger()["total_bytes"]


def test_group_rows_matches_the_shipped_partition():
    g, s = _pair(radius_scale=1.0)
    for gi in range(N_GROUPS + 1):
        assert np.array_equal(np.asarray(g.group_rows(gi)),
                              np.asarray(s.group_rows(gi)))


def test_shell_potential_from_swaps_the_learned_subtree():
    from chlu.core.memory_potentials import DesignFreedomPotential

    V = DesignFreedomPotential(
        rung="free_mlp", dim=DIM, payloads=jnp.zeros((N_GROUPS,)),
        key=jax.random.PRNGKey(0), learned_family="atoms", n_atoms=N_ATOMS,
        atom_groups=N_GROUPS, atom_depth_init=1e-4,
    )
    Vs = shell_potential_from(V, radius_scale=0.0)
    assert isinstance(Vs.learned, ShellAtomDictionaryPotential)
    q = jax.random.normal(jax.random.PRNGKey(1), (DIM,))
    assert float(Vs(q)) == float(V(q))  # r=0 gate, through the wrapper
