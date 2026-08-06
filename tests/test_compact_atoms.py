"""⭐ **C2W8 PASS 2, ARM A** — K7 (the capture instrument must be provable) and
K6 (OFF is bit-identical), plus the compact-atom mechanism's own properties.

**K7 is why this file exists and it is asserted before any arm number is
reported.** Pass 1's gate legs were *forced false by construction* and nobody
noticed until review: `capture_radius` was exactly 0.000 on 47 of 48 wells while
`lambda_min > 0` everywhere, and a single non-zero reading is far too thin to
license the instrument by observation. So, before `capture_radius > 0` is
trusted as a **success** criterion:

* a store with an **analytically known** capture radius must be recovered
  (`test_k7_planted_two_well_basin_is_recovered_two_sided` — two identical
  planted wells at `(+-0.30, 0)` put the separatrix exactly on the plane `x = 0`
  by symmetry, so the true radius at either site is **0.30**);
* a **planted flat** site must return **exactly 0.0**
  (`test_k7_planted_flat_site_returns_exactly_zero`).

Two-sided, both directions, on the shipped instrument
(`chlu.core.soft_certificate.capture_radius` via `chlu.core.well_lifecycle`,
both **read-only** for this arm).

⚠ Two instrument properties are pinned here because they are *not* what a reader
assumes and they change how a positive G-CAP must be read:

1. **The reading has a FLOOR of `tol / expansion_rate`**, not 0: the bisection
   asks whether the relaxed point is within `tol` of the site, so any site whose
   relaxation *barely moves* reports a positive radius with no basin at all
   (`test_k7_capture_radius_floor_is_tol_over_expansion_rate`). At the census's
   operating point `tol = sigma_q = 0.15`.
2. **A flat site sitting at the confinement minimum is a FALSE POSITIVE** of
   nearly `r_hi` (`test_k7_flat_site_at_the_bowl_minimum_is_a_declared_false_positive`).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.memory_potentials import (
    ATOM_KERNELS,
    AtomDictionaryPotential,
    atom_profile,
    localize_group_atoms,
)
from chlu.core.soft_certificate import capture_radius
from chlu.core.well_lifecycle import (
    capture_radii,
    flatten_unused_groups,
    plant_item,
)

SITE = np.array([0.2, -0.1, 0.05])


def _tiny_system(capacity=2, d_safe=0.01, seed=0, **over):
    """The `tests/test_well_lifecycle.py` rig, so K7 measures the shipped path."""
    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=capacity, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=float(d_safe), address_steps=40, n_query_per_item=2,
        **over,
    )
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)


# ==========================================================================
# K7 — the capture instrument can report a POSITIVE, and a ZERO
# ==========================================================================
def test_k7_capture_radius_recovers_an_analytically_known_radius():
    """Synthetic `relax_fn` with basin `R_true` -> recovered to one bisection cell."""
    r_true = 0.37

    def relax(pts):
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        d = np.linalg.norm(p - SITE[None, :], axis=1, keepdims=True)
        return np.where(d <= r_true, SITE[None, :], SITE[None, :] + 10.0 * (p - SITE[None, :]))

    out = capture_radius(relax, SITE, n_dirs=32, r_hi=1.0, steps=12, tol=0.01, seed=0)
    cell = 1.0 / 2**12
    assert abs(out["capture_radius"] - r_true) <= cell, out


def test_k7_capture_radius_is_zero_when_nothing_returns():
    """A relaxation that lands far away from every launch -> **exactly** 0.0."""

    def relax(pts):
        n = len(np.atleast_2d(np.asarray(pts)))
        return np.repeat((SITE + np.array([9.0, 0.0, 0.0]))[None, :], n, axis=0)

    out = capture_radius(relax, SITE, n_dirs=32, r_hi=1.0, steps=12, tol=0.01, seed=0)
    assert out["capture_radius"] == 0.0, out


def test_k7_capture_radius_floor_is_tol_over_expansion_rate():
    """⚠ The instrument's floor is `tol / lambda`, NOT 0 — pinned so a positive
    G-CAP is never read as "a basin exists" without checking the relaxation moved.

    `relax(x) = z + lambda (x - z)` has **no** basin (it is expanding everywhere),
    yet every launch within `tol / lambda` of the site still lands within `tol`,
    so the bisection reports that radius.
    """
    lam, tol = 5.0, 0.01

    def relax(pts):
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        return SITE[None, :] + lam * (p - SITE[None, :])

    out = capture_radius(relax, SITE, n_dirs=32, r_hi=1.0, steps=12, tol=tol, seed=0)
    assert abs(out["capture_radius"] - tol / lam) <= 1.0 / 2**12, out
    assert out["capture_radius"] > 0.0


def test_k7_planted_two_well_basin_is_recovered_two_sided():
    """⭐ **THE K7 POSITIVE.** Two identical planted wells at `(+-0.30, 0)`.

    Both wells have the same depth and width and the confinement bowl is centred
    at the origin, so by symmetry the separatrix is **exactly** the plane
    `x = 0` and the analytic capture radius at either site is `a/2 = 0.30`.
    `a / (2 s) = 2 > 1`, so the two minima are distinct (not one merged well).
    Finite direction sampling can only raise the reading (`0.30 / max|u_x|`);
    inertial overshoot under `gamma_read` can only lower it. Registered band
    (PREREG, filed before this ran): **[0.20, 0.40]** around **0.30**.
    """
    sysm = _tiny_system(capacity=2, read_steps=400)
    sites = np.array([[0.30, 0.0], [-0.30, 0.0]])
    for i in range(2):
        plant_item(sysm, i, sites[i], payload=0.0, depth=1.0, width=0.15, leak=0.0)
    flatten_unused_groups(sysm)

    z = np.array([0.30, 0.0, 0.0])
    out = capture_radius(sysm._relax_points, z, n_dirs=64, r_hi=1.0, steps=10,
                         tol=float(sysm.cfg.query_sigma), seed=0)
    r = float(out["capture_radius"])
    assert 0.20 <= r <= 0.40, out          # two-sided, registered band
    # and the planted site really is (essentially) the fixed point it claims
    settled = np.asarray(sysm._relax_points(z[None, :].astype(np.float32)))[0]
    assert np.linalg.norm(settled - z) < 0.02, settled
    # the census wrapper must agree with the instrument it wraps
    via_census = capture_radii(sysm, z[None, :], n_dirs=64, steps=10, seed=0)
    assert float(via_census[0]) > 0.0


def test_k7_planted_flat_site_returns_exactly_zero():
    """⭐ **THE K7 NEGATIVE.** A planted site with no well -> exactly 0.0.

    With `depth = 1e-9` and every unused group flattened the landscape is
    `V = alpha |q|^2`, so every launch runs to the origin — 0.6 away from the
    site, i.e. further than `tol`. Even `r = 0` fails, so `lo` never leaves 0.
    """
    sysm = _tiny_system(capacity=2, read_steps=400)
    plant_item(sysm, 0, np.array([0.6, 0.0]), payload=0.0, depth=1e-9,
               width=0.15, leak=0.0)
    flatten_unused_groups(sysm)
    z = np.array([0.6, 0.0, 0.0])
    out = capture_radius(sysm._relax_points, z, n_dirs=32, r_hi=1.0, steps=10,
                         tol=float(sysm.cfg.query_sigma), seed=0)
    assert out["capture_radius"] == 0.0, out


def test_k7_flat_site_at_the_bowl_minimum_is_a_declared_false_positive():
    """⚠ The one configuration in which the instrument's positive is NOT a well.

    A *flat* store whose site sits at the confinement minimum captures the whole
    ball: `capture_radius ~ r_hi` with no atom depth anywhere. Asserted (not
    fixed — `well_lifecycle.py` / `soft_certificate.py` are read-only for this
    arm) so the limitation travels with every G-CAP number. Benign in the census
    only because real `phi`-sites sit at `||z|| ~ 0.5 - 1.0`.
    """
    sysm = _tiny_system(capacity=2, read_steps=400)
    plant_item(sysm, 0, np.array([0.0, 0.0]), payload=0.0, depth=1e-9,
               width=0.15, leak=0.0)
    flatten_unused_groups(sysm)
    out = capture_radius(sysm._relax_points, np.zeros(3), n_dirs=32, r_hi=1.0,
                         steps=10, tol=float(sysm.cfg.query_sigma), seed=0)
    assert out["capture_radius"] >= 0.9, out


# ==========================================================================
# K6 — OFF is bit-identical AND parameter-count-identical
# ==========================================================================
def _n_params(m) -> int:
    return int(sum(np.asarray(x).size
                   for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))))


def test_k6_gaussian_kernel_is_the_literal_pre_pass2_expression():
    """`atom_kernel = "gaussian"` reproduces `-sum A_j exp(-d2/(2 s^2 + 1e-9))` **bitwise**."""
    a = AtomDictionaryPotential(4, 24, jax.random.PRNGKey(0), n_groups=2)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = jnp.asarray(rng.normal(size=4), dtype=jnp.float32)
        s = jnp.exp(a.log_width)
        d2 = jnp.sum((q[None, :] - a.centers) ** 2, axis=-1)
        ref = (-jnp.sum(a.amp**2 * jnp.exp(-d2 / (2.0 * s**2 + 1e-9)))
               + a.confine * jnp.sum(q**2))
        assert np.asarray(a(q)) == np.asarray(ref)
        # and the profile helper itself is the literal expression
        assert np.array_equal(np.asarray(atom_profile(d2, s, "gaussian")),
                              np.asarray(jnp.exp(-d2 / (2.0 * s**2 + 1e-9))))


def test_k6_every_kernel_is_parameter_count_identical():
    """The kernel is STATIC: no arm may buy bytes with it (K5's ledger)."""
    base = None
    for k in ATOM_KERNELS:
        a = AtomDictionaryPotential(4, 24, jax.random.PRNGKey(0), n_groups=2, kernel=k)
        n = _n_params(a)
        base = n if base is None else base
        assert n == base, (k, n, base)
        assert np.array_equal(  # the init draw is untouched by the kernel choice
            np.asarray(a.centers),
            np.asarray(AtomDictionaryPotential(4, 24, jax.random.PRNGKey(0),
                                               n_groups=2).centers))


def test_k6_flags_off_is_bit_identical_end_to_end():
    """A full admit+write+read with the flags explicitly OFF == the shipped path.

    Bits **and** parameter count, on the store leaves and on the settled read —
    the K2 / P1 / psires precedent. Reddening this test un-ships the flag.
    """
    def run(**over):
        sysm = _tiny_system(capacity=2, write_steps=20, read_steps=60, **over)
        sysm.write_stream([{"item_id": 0, "address": np.array([0.4, 0.1]),
                            "payload": 0.2}], key=jax.random.PRNGKey(3))
        q0 = np.zeros((2, 3), dtype=np.float32)
        q0[:, :2] = np.array([[0.4, 0.1], [-0.3, 0.2]])
        return sysm, np.asarray(sysm.read(q0).state.q_star)

    shipped, q_ship = run()
    explicit, q_expl = run(atom_kernel="gaussian", atom_kernel_cutoff=2.5,
                           atom_site_local_init=False, atom_site_local_radius=0.0)
    assert _n_params(shipped.model()) == _n_params(explicit.model())
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(shipped.store, eqx.is_inexact_array)),
                    jax.tree_util.tree_leaves(eqx.filter(explicit.store, eqx.is_inexact_array)),
                    strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(q_ship, q_expl)
    # the defaults are genuinely default: nothing new shows up in the flag table
    assert CluSystemConfig().as_flag_table() == {}


# ==========================================================================
# the mechanism itself — what "compact" buys and what it costs
# ==========================================================================
def test_compact_kernels_are_exactly_zero_beyond_their_support():
    """Value AND gradient are **exactly** 0 beyond `R = cutoff * s` — no tail.

    This is the K2 lesson one level down: a sigmoid tail makes a "local" change
    global, so the atom's influence must be identically zero, not 1e-30.
    """
    s, cutoff = jnp.asarray(0.1), 2.5
    R = float(cutoff * 0.1)
    for k in ("wendland", "truncated_gaussian"):
        for r in (R + 1e-6, R + 0.05, 1.0):
            d2 = jnp.asarray(r**2)
            assert float(atom_profile(d2, s, k, cutoff)) == 0.0
            g = jax.grad(lambda x, kk=k: atom_profile(x, s, kk, cutoff))(d2)
            assert float(g) == 0.0, (k, r, float(g))
        assert float(atom_profile(jnp.asarray(0.0), s, k, cutoff)) == 1.0


def test_wendland_is_c1_at_its_boundary_and_truncated_gaussian_is_not():
    """The declared smoothness difference, measured rather than asserted in prose.

    `wendland` has `phi'(R) = 0` (C^2 across the boundary, so the FORCE is
    continuous); `truncated_gaussian` is only C^0 and its force jumps.
    """
    s, cutoff = 0.1, 2.5
    R = cutoff * s

    def dprofile(kernel, r):
        return float(jax.grad(lambda x: atom_profile(x**2, jnp.asarray(s), kernel, cutoff))(r))

    assert abs(dprofile("wendland", R - 1e-4)) < 1e-3
    assert abs(dprofile("truncated_gaussian", R - 1e-4)) > 0.5


def test_site_local_init_moves_only_its_own_group_rows():
    """C3 parameter-space locality survives the companion lever (own rows only)."""
    a = AtomDictionaryPotential(3, 32, jax.random.PRNGKey(0), n_groups=4)
    rows = np.asarray(a.group_rows(1), dtype=bool)
    b = localize_group_atoms(a, rows, np.array([0.5, -0.2, 0.1]), 0.07,
                             jax.random.PRNGKey(1))
    c0, c1 = np.asarray(a.centers), np.asarray(b.centers)
    assert np.array_equal(c0[~rows], c1[~rows])          # foreign rows: bit-identical
    assert not np.array_equal(c0[rows], c1[rows])
    d = np.linalg.norm(c1[rows] - np.array([0.5, -0.2, 0.1])[None, :], axis=1)
    assert float(d.max()) <= 0.07 + 1e-6                 # inside the declared ball
    assert _n_params(a) == _n_params(b)


def test_site_local_init_is_wired_and_only_touches_the_admitted_slot():
    """The flag ON must actually relocate the admitted slot's atoms, and nothing else."""
    off = _tiny_system(capacity=2, write_steps=5, read_steps=60)
    on = _tiny_system(capacity=2, write_steps=5, read_steps=60,
                      atom_site_local_init=True, atom_site_local_radius=0.05)
    addr = np.array([0.4, 0.1])
    foreign_before = np.asarray(on.store.atoms.centers)[
        ~np.asarray(on.store.group_rows(0), dtype=bool)].copy()
    for s in (off, on):
        s.write_stream([{"item_id": 0, "address": addr, "payload": 0.2}],
                       key=jax.random.PRNGKey(3))
    rows = np.asarray(on.store.group_rows(0), dtype=bool)
    z = np.array([0.4, 0.1, 0.2])
    d_on = np.linalg.norm(np.asarray(on.store.atoms.centers)[rows] - z[None, :], axis=1)
    d_off = np.linalg.norm(np.asarray(off.store.atoms.centers)[rows] - z[None, :], axis=1)
    assert float(d_on.max()) < float(d_off.max())
    assert np.array_equal(
        np.asarray(on.store.atoms.centers)[~rows], foreign_before)
