"""Tests for the K=8-wall discriminator (exp_designed_mechanism, w22).

These pin the load-bearing pieces of the "designed mechanism, learned content"
experiment:

  * the **d-ball geometry** — the payload channel lives at index ``d`` (NOT the
    ring's index 2), so the write objective and the eval must both address it
    there; the generalized ``write_loss(payload_index=d)`` is the fix that makes
    the atom dictionary trainable in a d-dimensional address space;
  * the **masked (local) write is bit-local in parameter space** at any ``d`` —
    writing one item's atom block leaves every other block bit-identical (the
    C3-local claim the interference arm rests on);
  * the **parameter budget scales with K** (``n_atoms = atoms_per_item*K``), so a
    plateau in ``K_learned`` is a learning failure, not a capacity-of-parameters
    one;
  * the growth fit **excludes censored points** (a lower bound is not a
    measurement) and the discriminator verdict follows the pre-registered rule.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.memory_potentials import atom_write_mask_fn
from chlu.experiments.exp_designed_mechanism import (
    _atoms_for,
    _floor_atoms,
    _fit_growth,
    apply_quick,
    ball_setup,
    build_learned_V,
    score_cell,
    write_learned,
    build_designed_model,
)
from chlu.training.train_memory import write_loss


@pytest.fixture
def cfg():
    c = get_default_config()
    dm = c.experiment_designed_mechanism
    apply_quick(c)
    return dm


def test_ball_setup_payload_at_index_d():
    c = get_default_config().experiment_designed_mechanism
    for d in (2, 3, 4):
        centers, payloads, targets, sep = ball_setup(d, 4, c)
        assert centers.shape == (4, d)
        assert targets.shape == (4, d + 1)
        # payload lives at index d, address at [:d]
        np.testing.assert_allclose(np.asarray(targets[:, d]), np.asarray(payloads), atol=1e-6)
        np.testing.assert_allclose(np.asarray(targets[:, :d]), np.asarray(centers), atol=1e-6)
        assert sep > 0.0


def test_write_loss_payload_index_generalizes():
    """The generalized write_loss must pin the payload channel at ``payload_index``,
    and default to the ring's index 2 (backward compatible)."""
    key = jax.random.PRNGKey(0)

    def V(q):
        return jnp.sum(q**2)

    # dim=5 (d=4 ball): payload at index 4. Loss must be finite and differentiable.
    targets = jax.random.normal(jax.random.PRNGKey(1), (3, 5))
    lo = write_loss(V, targets, key, n_perturb=4, payload_index=4)
    assert np.isfinite(float(lo))
    # default index 2 still works (ring geometry, dim=3)
    t3 = jax.random.normal(jax.random.PRNGKey(2), (3, 3))
    l3 = write_loss(V, t3, key, n_perturb=4)
    assert np.isfinite(float(l3))


def test_param_budget_scales_with_K():
    c = get_default_config().experiment_designed_mechanism
    d = 3
    # The dimension-aware floor pins small-K cells (an over-complete write regime);
    # the atoms_per_item*K term dominates once K passes floor(d)/atoms_per_item.
    floor = _floor_atoms(c, d)
    k_floor = floor // c.atoms_per_item
    assert _atoms_for(c, 2, d) == floor  # floored
    big1, big2 = 2 * k_floor, 4 * k_floor
    assert _atoms_for(c, big1, d) == c.atoms_per_item * big1  # ·K term dominates
    assert _atoms_for(c, big2, d) == 2 * _atoms_for(c, big1, d)  # and scales with K
    V1 = build_learned_V(d, big1, c, jax.random.PRNGKey(0))
    V2 = build_learned_V(d, big2, c, jax.random.PRNGKey(0))
    n1 = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(V1, eqx.is_inexact_array)))
    n2 = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(V2, eqx.is_inexact_array)))
    assert n2 == 2 * n1
    # the learned atom dictionary lives in the .learned container (so the tested
    # trainable_filter / atom_write_mask_fn machinery applies)
    assert V1.learned.n_groups == big1 and V2.learned.n_groups == big2
    assert V1.designed is None


def test_dimension_aware_floor_grows_geometrically():
    """The atom floor must scale with the address dimension (w23): floor(d+1) =
    c * floor(d) in the geometric regime, and it must strictly exceed the fixed
    hard floor at the sweep dimensions so a high-d low-K cell is not budget-starved.
    """
    c = get_default_config().experiment_designed_mechanism
    # geometric law floor(d) = round(base * c**d) once it clears the hard floor
    for d in (2, 3, 4, 6, 8):
        assert _floor_atoms(c, d) == max(
            c.min_atoms, round(c.min_atoms_base * c.min_atoms_c**d)
        )
    # strictly increasing in d, and the high-d floor is much larger than low-d
    floors = [_floor_atoms(c, d) for d in (2, 3, 4, 6, 8)]
    assert all(b > a for a, b in zip(floors, floors[1:], strict=False))
    assert floors[-1] >= c.min_atoms_c**6 * floors[0] * 0.99  # d=8 vs d=2 ~ c^6
    # at a fixed low K the atom count IS the dimension-aware floor (the ·K term is
    # dominated), so the write budget grows with d exactly as the floor does.
    assert _atoms_for(c, 2, 8) == _floor_atoms(c, 8)
    assert _atoms_for(c, 2, 8) > _atoms_for(c, 2, 2)


def test_masked_write_is_bit_local_in_d_ball():
    """Writing item 0's atom block must leave every OTHER block bit-identical, at a
    d>2 address dimension (the interference arm's C3-local claim)."""
    c = get_default_config().experiment_designed_mechanism
    d, K = 3, 4
    centers, payloads, targets, _ = ball_setup(d, K, c)
    V = build_learned_V(d, K, c, jax.random.PRNGKey(3))
    mask0 = V.learned.group_rows(0)
    V2, _ = _single_write(V, targets[:1], mask0, c, d)
    # atoms OUTSIDE block 0 are bit-identical; block-0 atoms moved.
    m = np.asarray(mask0)
    for attr in ("centers", "amp", "log_width"):
        before = np.asarray(getattr(V.learned, attr))
        after = np.asarray(getattr(V2.learned, attr))
        if before.ndim == 2:
            np.testing.assert_array_equal(before[~m], after[~m])
            assert np.any(before[m] != after[m])
        else:
            np.testing.assert_array_equal(before[~m], after[~m])


def _single_write(V, target, mask, c, d):
    from chlu.training.train_memory import train_memory_landscape

    return train_memory_landscape(
        V,
        target,
        jax.random.PRNGKey(9),
        steps=20,
        lr=3e-3,
        loss_kwargs=dict(n_perturb=4, payload_index=d),
        update_mask_fn=atom_write_mask_fn(mask),
    )


def test_fit_growth_excludes_censored():
    """A censored (lower-bound) point must not enter the fit; excluding it must
    keep the exponential base honest."""
    ds = [2, 3, 4, 6, 8]
    ks = [16, 32, 64, 256, 256]  # last is censored at cap 256
    cens = [False, False, False, False, True]
    fit = _fit_growth(ds, ks, cens)
    assert fit["n_censored_excluded"] == 1
    assert fit["n_points_fitted"] == 4
    # 4*2^d over the 4 non-censored points -> base ~ 2.0
    assert 1.8 <= fit["exponential_base_A"] <= 2.2
    assert fit["exponential_r2"] > 0.95


def test_designed_arm_retrieves_small_K(cfg):
    """Gate: the DESIGNED ball register must actually retrieve at a small cell, or
    the harness (not learning) is at fault and no learned number is reportable."""
    d, K = 2, 4
    centers, payloads, _, _ = ball_setup(d, K, cfg)
    model = build_designed_model(centers, payloads, cfg)
    out = score_cell(model, centers, payloads, cfg, d, seed=0)
    assert out["finite"]
    # designed selectivity (addressing) should be high even in quick mode
    assert out["selectivity"] >= 0.75


def test_blank_learned_scores_low_on_value(cfg):
    """A learned landscape trained with ZERO payloads must not return stored values
    (the value blank control is what makes strict success leak-immune)."""
    d, K = 2, 4
    centers, payloads, targets, _ = ball_setup(d, K, cfg)
    blank_pay = jnp.zeros_like(payloads)
    _, _, blank_targets, _ = ball_setup(d, K, cfg, payloads=blank_pay)
    k = jax.random.PRNGKey(0)
    Vb = build_learned_V(d, K, cfg, k)
    Vb, _ = write_learned(Vb, blank_targets, cfg, k, d, mode="local")
    from chlu.experiments.goldstone_harness import clu_with_potential

    mb = clu_with_potential(
        Vb, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    blank = score_cell(mb, centers, payloads, cfg, d, seed=0)
    # nothing stored -> strict success against the REAL payloads must be ~0
    assert blank["strict_success_rate"] <= cfg.blank_strict_max + 1e-9


def test_trained_well_widths_recovers_planted_widths():
    """⭐ w25 §5.0: the width dump must report the width of the atoms that FORM the
    well at a site, not the population median.

    Planted landscape: at each site one deep atom of width 0.11 (the well), plus a
    large background of shallow far-away atoms of width 0.9. The population median is
    0.9; the measured well width must be the planted 0.11.
    """
    from chlu.core.memory_potentials import AtomDictionaryPotential
    from chlu.experiments.exp_designed_mechanism import trained_well_widths

    d, K, dim = 3, 4, 4
    c = get_default_config().experiment_designed_mechanism
    _, _, targets, _ = ball_setup(d, K, c)
    n_bg = 40
    atoms = AtomDictionaryPotential(dim, K + n_bg, jax.random.PRNGKey(0))
    centers = jnp.concatenate(
        [jnp.asarray(targets), jax.random.normal(jax.random.PRNGKey(1), (n_bg, dim)) * 8.0]
    )
    widths = jnp.concatenate([jnp.full((K,), 0.11), jnp.full((n_bg,), 0.9)])
    amps = jnp.concatenate([jnp.full((K,), 1.0), jnp.full((n_bg,), 0.05)])
    atoms = eqx.tree_at(
        lambda a: (a.centers, a.log_width, a.amp),
        atoms,
        (centers, jnp.log(widths), amps),
    )
    out = trained_well_widths(atoms, targets)

    assert out["n_atoms"] == K + n_bg
    assert np.isclose(out["all_atom_width_median"], 0.9, atol=1e-3)  # the wrong answer
    assert np.isclose(out["w_atom"], 0.11, rtol=1e-3)  # the right one
    assert len(out["w_atom_per_site"]) == K
    assert all(n >= 1 for n in out["n_keep_per_site"])


def test_trained_well_widths_accepts_wrapped_potential(cfg):
    """It must accept the harness's ``DesignFreedomPotential`` wrapper (``.learned``)
    as well as a bare atom dictionary, and report the INIT width on an unwritten V."""
    from chlu.experiments.exp_designed_mechanism import trained_well_widths

    d, K = 2, 4
    _, _, targets, _ = ball_setup(d, K, cfg)
    V = build_learned_V(d, K, cfg, jax.random.PRNGKey(0))
    out = trained_well_widths(V, targets)
    assert np.isclose(out["w_atom"], cfg.atom_init_width, rtol=1e-5)
    assert np.isclose(out["all_atom_width_median"], cfg.atom_init_width, rtol=1e-5)
