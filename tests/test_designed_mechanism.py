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


# ---------------------------------------------------------------------------
# w26 (r2-excursion-reach): the read-out excursion levers
# ---------------------------------------------------------------------------


def test_payload_codebook_holds_min_separation_and_cuts_excursion():
    """⭐ Fairness condition 1: the m-channel code keeps the codeword MINIMUM
    SEPARATION of the 1-channel codebook (same K, same delta => same bits at the same
    per-axis noise) while cutting the per-item EXCURSION (the reach demand)."""
    import copy

    from chlu.experiments.exp_designed_mechanism import payload_codebook

    base = get_default_config().experiment_designed_mechanism
    for K in (16, 32):
        delta = 2.0 / (K - 1)
        prev_max = None
        for m in (1, 2, 4):
            c = copy.copy(base)
            c.n_payload_channels = m
            c.payload_code = "grid"
            w = np.asarray(payload_codebook(K, c))
            assert w.shape == (K, m)
            dd = np.sqrt(((w[:, None, :] - w[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dd, np.inf)
            assert np.isclose(dd.min(), delta, rtol=1e-5)  # precision preserved
            mx = float(np.linalg.norm(w, axis=1).max())
            if prev_max is not None:
                assert mx < prev_max  # excursion strictly falls with m
            prev_max = mx
        assert prev_max < 0.2  # m=4 cuts the reach demand by >5x


def test_multichannel_payload_geometry_and_designed_arm():
    """m payload channels live at q[d:d+m]; the DESIGNED arm reads the same code
    (fairness condition 4) and still retrieves it."""
    import copy

    from chlu.experiments.exp_designed_mechanism import build_designed_model, score_cell

    c = copy.copy(get_default_config().experiment_designed_mechanism)
    c.n_payload_channels, c.payload_code = 2, "grid"
    d, K = 2, 4
    centers, payloads, targets, _ = ball_setup(d, K, c)
    assert payloads.shape == (K, 2)
    assert targets.shape == (K, d + 2)
    np.testing.assert_allclose(np.asarray(targets[:, d:]), np.asarray(payloads), atol=1e-6)
    model = build_designed_model(centers, payloads, c)
    assert model.potential_net.dim == d + 2
    r = score_cell(model, centers, payloads, c, d, seed=0)
    assert r["strict_success_rate"] > 0.9  # the designed arm reads the vector code


def test_anneal_schedule_ends_sharp_and_inflation_is_a_gaussian_blur(cfg):
    """s_eff^2 = s^2 + s_extra^2, the schedule always ends at s_extra = 0, and
    s_extra = 0 is the identity."""
    import copy

    from chlu.experiments.exp_designed_mechanism import (
        anneal_widths,
        inflate_potential,
    )

    c = copy.copy(cfg)
    assert anneal_widths(c) == [0.0]  # default = the shipped single-stage read
    c.read_anneal_stages, c.read_anneal_s0 = 4, 0.3
    sch = anneal_widths(c)
    assert len(sch) == 4 and sch[0] == 0.3 and sch[-1] == 0.0
    assert all(a >= b for a, b in zip(sch[:-1], sch[1:], strict=True))

    V = build_learned_V(2, 4, c, jax.random.PRNGKey(0))
    assert inflate_potential(V, 0.0) is V
    s = np.exp(np.asarray(V.learned.log_width))
    for mode in ("amplitude", "mass"):
        Vi = inflate_potential(V, 0.4, mode=mode)
        s_eff = np.exp(np.asarray(Vi.learned.log_width))
        np.testing.assert_allclose(s_eff**2, s**2 + 0.4**2, rtol=1e-5)
        A0 = np.asarray(V.learned.amp) ** 2
        A1 = np.asarray(Vi.learned.amp) ** 2
        dim = V.learned.centers.shape[1]
        if mode == "amplitude":
            np.testing.assert_allclose(A1, A0, rtol=1e-6)  # depth held
        else:  # mass mode preserves the integral A * s^dim
            np.testing.assert_allclose(A1 * s_eff**dim, A0 * s**dim, rtol=1e-4)


def test_staged_read_is_equal_compute(cfg):
    """⭐ The annealed read must integrate exactly as many Verlet steps as the
    baseline: splitting the SAME landscape into L stages reproduces the one-stage
    address read bit-for-bit (deterministic Verlet, restarted from the exact state)."""
    from chlu.experiments.exp_designed_mechanism import _two_phase, make_ball_queries

    d, K = 2, 4
    centers, payloads, _, _ = ball_setup(d, K, cfg)
    model = build_designed_model(centers, payloads, cfg)
    Q0, P0, _ = make_ball_queries(jax.random.PRNGKey(0), centers, 4, cfg)
    x1, _ = _two_phase(model, Q0, P0, cfg, d)
    x2, _ = _two_phase(model, Q0, P0, cfg, d, stage_models=[model, model])
    np.testing.assert_allclose(x1, x2, atol=1e-5)


def test_localized_atom_init_is_address_only_and_off_by_default(cfg):
    """Stage A's lever: group j's atoms start near item j's ADDRESS site; the payload
    axis keeps the scattered init (N46 -- localizing it would hand the writer the
    answer). Default OFF reproduces the historical scatter bit-for-bit."""
    import copy

    d, K = 3, 4
    c = copy.copy(cfg)
    centers, _, _, _ = ball_setup(d, K, c)
    key = jax.random.PRNGKey(0)
    V_off = build_learned_V(d, K, c, key, centers=centers)
    V_none = build_learned_V(d, K, c, key)
    np.testing.assert_array_equal(
        np.asarray(V_off.learned.centers), np.asarray(V_none.learned.centers)
    )
    c.atom_init_local, c.atom_init_local_mult = True, 2.0
    V_on = build_learned_V(d, K, c, key, centers=centers)
    A = np.asarray(V_on.learned.centers)
    radius = c.atom_init_local_mult * c.atom_init_width
    owner = np.asarray(V_on.learned.n_groups)
    assert int(owner) == K
    n_at = A.shape[0]
    for g in range(K):
        rows = np.asarray(V_on.learned.group_rows(g))
        dist = np.linalg.norm(A[rows, :d] - np.asarray(centers)[g][None, :], axis=1)
        assert dist.max() <= radius + 1e-5
    # the payload axis is NOT localized (it keeps the N(0, init_scale) spread)
    assert np.std(A[:, d]) > 0.5
    assert n_at == _atoms_for(c, K, d)


def test_payload_read_noise_and_decode_metric(cfg):
    """⭐ Fairness condition 3. Launch noise perturbs the payload channel at t=0;
    observation noise perturbs the read value; the decode metric is nearest-codeword
    and is strictly harsher than the absolute-tolerance metric once the codebook
    spacing falls below payload_tol."""
    import copy

    from chlu.experiments.exp_designed_mechanism import make_ball_queries, score_cell

    d, K = 2, 4
    c = copy.copy(cfg)
    centers, payloads, _, _ = ball_setup(d, K, c)
    Q0, _, _ = make_ball_queries(jax.random.PRNGKey(0), centers, 4, c)
    assert float(np.abs(np.asarray(Q0[:, d])).max()) == 0.0  # the shipped guard
    c.payload_launch_sigma = 0.2
    Q1, _, _ = make_ball_queries(jax.random.PRNGKey(0), centers, 4, c)
    assert float(np.abs(np.asarray(Q1[:, d])).max()) > 0.0
    np.testing.assert_allclose(np.asarray(Q0[:, :d]), np.asarray(Q1[:, :d]), atol=1e-6)

    c.payload_launch_sigma = 0.0
    model = build_designed_model(centers, payloads, c)
    clean = score_cell(model, centers, payloads, c, d, seed=0)
    assert clean["strict_tol"] == clean["strict_decode"]  # exact read: metrics agree
    c.payload_obs_sigma = 5.0  # swamp the value channel
    c.pass_metric = "decode"
    noisy = score_cell(model, centers, payloads, c, d, seed=0)
    assert noisy["strict_success_rate"] == noisy["strict_decode"]
    assert noisy["strict_decode"] < clean["strict_decode"]
    assert noisy["basin_success_rate"] == clean["basin_success_rate"]  # address intact


def test_anisotropic_anneal_widens_only_the_payload_axis(cfg):
    """⭐ The isotropic blur cannot buy reach without merging neighbouring wells in
    the ADDRESS space. ``read_anneal_axes="payload"`` widens only the payload
    channels, ends at multiplier exactly 1 (the stored landscape), is invisible to
    the write, and strictly increases the force a far-payload well exerts at the
    payload-zero launch manifold."""
    import copy

    import jax.tree_util as tu

    from chlu.experiments.exp_designed_mechanism import (
        anneal_axis_mults,
        anneal_stage_models,
    )
    from chlu.experiments.goldstone_harness import clu_with_potential
    from chlu.training.train_memory import trainable_filter

    d, K = 2, 4
    c = copy.copy(cfg)
    assert anneal_axis_mults(c, d, d + 1) is None  # default off
    c.read_anneal_stages, c.read_anneal_axes, c.read_anneal_payload_mult = 4, "payload", 3.0
    mults = anneal_axis_mults(c, d, d + 1)
    assert len(mults) == 4
    for mu in mults:
        assert mu[:d] == (1.0,) * d  # address axes untouched
    assert mults[0][d] == 3.0 and mults[-1][d] == 1.0  # ends on the stored landscape

    centers, payloads, _, _ = ball_setup(d, K, c)
    V = build_learned_V(d, K, c, jax.random.PRNGKey(0))
    # plant one deep atom at (site_0, payload=1) so there is a far-payload well
    V = eqx.tree_at(
        lambda p: (p.learned.centers, p.learned.amp),
        V,
        (
            V.learned.centers.at[0, :d].set(centers[0]).at[0, d].set(1.0),
            V.learned.amp.at[0].set(1.0),
        ),
    )
    model = clu_with_potential(
        V, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    stages = anneal_stage_models(model, c, d=d)
    assert len(stages) == 4
    # the write can never see the knob (it is not an inexact array)
    spec = trainable_filter(stages[0].potential_net)
    assert sum(1 for x in tu.tree_leaves(spec) if x is True) == 3  # centers/width/amp

    q = jnp.zeros(d + 1).at[:d].set(centers[0])  # ON the payload-zero manifold
    f0 = float(jnp.abs(jax.grad(lambda x: model.potential_net(x))(q)[d]))
    f1 = float(jnp.abs(jax.grad(lambda x: stages[0].potential_net(x))(q)[d]))
    assert f1 > 3.0 * f0  # reach along the payload axis, bought at the launch point


def test_value_blank_is_rejected_at_m_gt_1_only_under_decode(cfg):
    """⭐ w27 stage 1 (Head ruling: a CORRECTNESS fix, not a lever promotion).

    ``pass_metric="tol"`` is VACUOUS at ``m > 1``: the whole grid codebook lives inside
    ``payload_tol`` (m=4, K=32: ``max||a|| = 0.0912 < 0.1``), so a landscape written with
    ZERO payloads -- which stores no value at all -- scores ``strict = 1.0000`` on ``tol``
    *and* slips past the value-blank gate, because the trivial ceiling
    ``mean(||a|| < payload_tol)`` is itself 1.0. Nearest-codeword ``decode`` scores the same
    blank at exactly chance ``1/K``. Hence the w27 default.
    """
    import copy

    from chlu.experiments.exp_designed_mechanism import build_designed_model, score_cell

    assert get_default_config().experiment_designed_mechanism.pass_metric == "decode"

    c = copy.copy(cfg)
    c.n_payload_channels, c.payload_code = 4, "grid"
    c.max_total_queries = 128  # keep the unit test cheap; the designed arm is exact here
    d, K = 4, 32
    centers, payloads, _, _ = ball_setup(d, K, c)
    assert float(np.linalg.norm(np.asarray(payloads), axis=1).max()) < c.payload_tol

    blank = jnp.zeros_like(payloads)
    r_blank = score_cell(build_designed_model(centers, blank, c), centers, payloads, c, d, 0)
    r_true = score_cell(build_designed_model(centers, payloads, c), centers, payloads, c, d, 0)

    # the blank store settles into the right BASIN (the addresses are still there) ...
    assert r_blank["basin_success_rate"] > 0.99
    # ... and the absolute-tolerance criterion then hands it a perfect score
    assert r_blank["strict_tol"] == pytest.approx(1.0)
    # while the decode criterion puts it at exactly chance
    assert r_blank["strict_decode"] == pytest.approx(1.0 / K, abs=1e-6)

    # the DEFAULT metric is the one that rejects it
    c.pass_metric = "decode"
    r = score_cell(build_designed_model(centers, blank, c), centers, payloads, c, d, 0)
    assert r["strict_success_rate"] == r_blank["strict_decode"]
    assert r["strict_success_rate"] < c.pass_strict  # a blank store does NOT pass
    assert r_true["strict_decode"] > c.pass_strict  # a real store still does

    # "tol" stays selectable for pre-w27 reproduction -- and is shown to be vacuous
    c.pass_metric = "tol"
    r = score_cell(build_designed_model(centers, blank, c), centers, payloads, c, d, 0)
    assert r["strict_success_rate"] >= c.pass_strict  # ⛔ the blank "passes" under tol
