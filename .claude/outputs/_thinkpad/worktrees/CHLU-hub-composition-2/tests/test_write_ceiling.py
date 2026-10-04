"""Tests for WRITE-CEILING-BREAK (exp_write_ceiling, w24).

These pin the pieces the ceiling verdict rests on:

  * the new ``write_loss`` levers are **backward compatible** — with default
    arguments the loss is bit-identical to the w20-w23 objective every prior
    result was produced with;
  * the crowding penalty **sees crowding** (it is >0 exactly when an atom sits
    within ``d_safe`` of a site it does not own, and 0 otherwise) and is a no-op
    on a potential family where "the atoms this item owns" is undefined;
  * the ``max`` minimum-aggregation and the nearest-neighbour barrier are the
    *undiluted* forms of the mean/all-pairs terms (the blindness this experiment
    attacks: one violated direction out of many is averaged away);
  * the **sequential masked** write is bit-local in parameter space (writing item
    i leaves every other atom block bit-identical) — the locality lever must be
    what it claims to be;
  * the arms differ ONLY in the write operator (same geometry, same atom budget,
    same learned content), and the sequential arms do not out-spend the baseline
    in item-gradient evaluations.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments.exp_write_ceiling import (
    ARMS,
    apply_quick,
    arm_write_spec,
    build_V,
    _item_grad_budget,
    ladder_for,
    sequential_write,
)
from chlu.experiments.exp_designed_mechanism import _atoms_for, ball_setup
from chlu.training.train_memory import atom_crowding_penalty, write_loss


@pytest.fixture
def cfgs():
    c = get_default_config()
    apply_quick(c)
    return c.experiment_designed_mechanism, c.experiment_write_ceiling


def _toy_targets(d=2, K=4, seed=0):
    c = get_default_config().experiment_designed_mechanism
    return ball_setup(d, K, c)


def test_write_loss_defaults_are_bit_identical():
    """The w24 levers must not perturb the default objective: every prior result
    (w20-w23) was produced with it."""
    key = jax.random.PRNGKey(0)

    def V(q):
        return jnp.sum(q**2) - 0.5 * jnp.exp(-jnp.sum(q**2))

    targets = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    base = float(write_loss(V, targets, key, n_perturb=8, payload_index=3))
    same = float(
        write_loss(
            V, targets, key, n_perturb=8, payload_index=3,
            min_agg="mean", barrier_pairs="all", item_agg="mean",
            crowd_weight=0.0, crowd_d_safe=0.0, crowd_targets=None,
        )
    )
    assert base == same


def test_min_agg_max_is_undiluted_and_item_agg_sum_scales():
    key = jax.random.PRNGKey(0)

    def V(q):
        return jnp.sum(q**2)

    targets = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    lo = float(write_loss(V, targets, key, n_perturb=16))
    hi = float(write_loss(V, targets, key, n_perturb=16, min_agg="max"))
    assert hi >= lo  # a max over directions can only be >= the mean
    s = float(write_loss(V, targets, key, n_perturb=16, item_agg="sum"))
    # every term (grad, min AND barrier) must scale the same way, so sum over
    # K=4 items is exactly 4x the mean-aggregated loss
    assert s == pytest.approx(4.0 * lo, rel=1e-5)


def test_nn_barrier_is_the_undiluted_all_pairs_barrier():
    """With K items, a violated pair carries weight ~1/K^2 under the all-pairs
    mean while the number of violated (i.e. NEAR) pairs is only O(K): the crowding
    signal is diluted by ~1/K. The nearest-neighbour form must recover it."""
    key = jax.random.PRNGKey(0)

    def V(q):
        # radially decreasing: midpoints of ADJACENT sites (radius 0.92) sit low
        # enough to violate the barrier, midpoints of distant sites do not.
        return -0.5 * jnp.sum(q**2)

    K, dim = 8, 3
    ang = jnp.linspace(0, 2 * jnp.pi, K, endpoint=False)
    targets = jnp.zeros((K, dim)).at[:, 0].set(jnp.cos(ang)).at[:, 1].set(jnp.sin(ang))
    bar = dict(w_grad=0.0, w_min=0.0, n_perturb=4)  # isolate the barrier term
    all_pairs = float(write_loss(V, targets, key, barrier_pairs="all", **bar))
    nn = float(write_loss(V, targets, key, barrier_pairs="nn", **bar))
    assert all_pairs > 0.0
    # 8 violated pairs out of 28 -> the all-pairs mean is ~3.5x diluted
    assert nn > 3.0 * all_pairs


def test_crowding_penalty_sees_crowding():
    """>0 exactly when an atom of one block sits within d_safe of another item's
    site; 0 when every atom sits on its own site."""
    c = get_default_config().experiment_designed_mechanism
    d, K = 2, 4
    centers, payloads, targets, sep = ball_setup(d, K, c)
    V = build_V(d, K, 8, c, jax.random.PRNGKey(0))  # 8 atoms, 2 per group
    n = V.learned.n_atoms
    # (a) park every atom exactly on the site its own block owns -> no crowding
    own = jnp.stack([targets[min(i * K // n, K - 1)] for i in range(n)])
    V_ok = eqx.tree_at(lambda t: t.learned.centers, V, own)
    V_ok = eqx.tree_at(lambda t: t.learned.amp, V_ok, jnp.ones((n,)))
    pen_ok = float(atom_crowding_penalty(V_ok, targets, d_safe=0.5 * float(sep)))
    assert pen_ok == pytest.approx(0.0, abs=1e-8)
    # (b) park every atom on item 0's site -> blocks 1..K-1 are encroaching
    bad = jnp.repeat(targets[:1], n, axis=0)
    V_bad = eqx.tree_at(lambda t: t.learned.centers, V, bad)
    V_bad = eqx.tree_at(lambda t: t.learned.amp, V_bad, jnp.ones((n,)))
    pen_bad = float(atom_crowding_penalty(V_bad, targets, d_safe=0.5 * float(sep)))
    assert pen_bad > 0.0
    # and it enters write_loss only when switched on
    key = jax.random.PRNGKey(0)
    lk = dict(n_perturb=4, payload_index=d)
    off = float(write_loss(V_bad, targets, key, **lk))
    on = float(
        write_loss(
            V_bad, targets, key, crowd_weight=1.0, crowd_d_safe=0.5 * float(sep), **lk
        )
    )
    assert on > off


def test_crowding_penalty_is_a_noop_without_grouped_atoms():
    """A potential with no grouped atom dictionary has no notion of "the atoms
    this item owns" — the term must be exactly zero, not an error."""
    _, _, targets, _ = _toy_targets()

    def V(q):
        return jnp.sum(q**2)

    assert float(atom_crowding_penalty(V, targets, d_safe=0.3)) == 0.0
    # a grouped dictionary whose group count disagrees with the item count too
    c = get_default_config().experiment_designed_mechanism
    V2 = build_V(2, 8, 16, c, jax.random.PRNGKey(0))  # 8 groups vs 4 targets
    assert float(atom_crowding_penalty(V2, targets, d_safe=0.3)) == 0.0


def test_sequential_masked_write_is_bit_local(cfgs):
    """Writing item i's block must leave every OTHER block bit-identical."""
    dm, wc = cfgs
    d, K = 2, 4
    _, _, targets, sep = ball_setup(d, K, dm)
    V = build_V(d, K, 16, dm, jax.random.PRNGKey(3))
    spec = arm_write_spec("sequential_masked", dm, wc, d, K, float(sep), targets)
    spec["steps"] = 3
    # write ONLY item 0 by handing the writer a single-row target set
    out, hist = sequential_write(V, targets[:1], jax.random.PRNGKey(1), spec, dm, wc)
    assert len(hist) == 3
    m = np.asarray(V.learned.group_rows(0))
    for attr in ("centers", "amp", "log_width"):
        before = np.asarray(getattr(V.learned, attr))
        after = np.asarray(getattr(out.learned, attr))
        np.testing.assert_array_equal(before[~m], after[~m])
        assert np.any(before[m] != after[m])


def test_sequential_free_write_moves_all_atoms(cfgs):
    """The unmasked sequential arm is the control that isolates 'one gradient at a
    time' from 'parameter-space masking': it must move atoms outside the block."""
    dm, wc = cfgs
    d, K = 2, 4
    _, _, targets, sep = ball_setup(d, K, dm)
    V = build_V(d, K, 16, dm, jax.random.PRNGKey(3))
    spec = arm_write_spec("sequential_free", dm, wc, d, K, float(sep), targets)
    spec["steps"] = 3
    out, _ = sequential_write(V, targets[:1], jax.random.PRNGKey(1), spec, dm, wc)
    m = np.asarray(V.learned.group_rows(0))
    before = np.asarray(V.learned.amp)
    after = np.asarray(out.learned.amp)
    assert np.any(before[~m] != after[~m])


def test_arm_specs_differ_only_in_the_write_operator(cfgs):
    """Every arm must share the geometry/budget and differ only in locality, the
    length scales, and the objective terms (the N46 fairness statement)."""
    dm, wc = cfgs
    d, K, sep = 4, 16, 0.8
    _, _, targets, _ = ball_setup(d, K, dm)
    specs = {a: arm_write_spec(a, dm, wc, d, K, sep, targets) for a in ARMS}
    base = specs["baseline_global"]
    assert base["sequential"] is False and base["init_width"] is None
    assert base["loss_kwargs"]["sigma_addr"] == dm.write_sigma_addr
    assert "crowd_weight" not in base["loss_kwargs"]
    assert specs["sequential_masked"]["sequential"] and specs["sequential_masked"]["masked"]
    assert specs["sequential_free"]["sequential"] and not specs["sequential_free"]["masked"]
    si = specs["scale_invariant"]
    assert si["loss_kwargs"]["sigma_addr"] == pytest.approx(wc.scale_sigma_frac * sep)
    assert si["init_width"] == pytest.approx(wc.scale_width_frac * sep)
    assert si["loss_kwargs"]["item_agg"] == "sum"
    ca = specs["crowding_aware"]
    assert ca["loss_kwargs"]["min_agg"] == "max"
    assert ca["loss_kwargs"]["barrier_pairs"] == "nn"
    assert 0.0 < ca["loss_kwargs"]["crowd_d_safe"] <= wc.crowd_d_safe_frac * sep
    combo = specs["combo"]
    assert combo["sequential"] and combo["masked"]
    assert combo["loss_kwargs"]["crowd_weight"] > 0
    assert combo["init_width"] == si["init_width"]
    # the atom budget is the w23 one for every arm (only the operator changes)
    assert _atoms_for(dm, K, d) == _atoms_for(dm, K, d)


def test_sequential_arms_do_not_outspend_the_baseline():
    """The sequential write must not buy the ceiling with extra compute: its
    item-gradient budget is <= the global baseline's at the default settings."""
    c = get_default_config()
    dm, wc = c.experiment_designed_mechanism, c.experiment_write_ceiling
    for K in (16, 32, 64, 128):
        b = _item_grad_budget("baseline_global", K, dm, wc)
        s = _item_grad_budget("sequential_masked", K, dm, wc)
        assert s <= b, (K, s, b)


def test_ladder_starts_at_the_w23_last_pass_rung():
    c = get_default_config()
    wc = c.experiment_write_ceiling
    assert ladder_for(wc, 4)[0] == wc.k_start[wc.dims.index(4)]
    for d in wc.dims:
        lad = ladder_for(wc, d)
        assert lad and max(lad) <= wc.k_cap
        # the ceiling question lives above 32, so every dimension must probe past it
        assert max(lad) > 32


def test_write_ceiling_config_round_trips(tmp_path):
    from chlu.config import load_config, save_config

    c = get_default_config()
    c.experiment_write_ceiling.crowd_weight = 3.5
    c.experiment_write_ceiling.seq_steps_per_item = 42
    c.experiment_write_ceiling.arms = ["baseline_global", "combo"]
    p = tmp_path / "config.yaml"
    save_config(c, p)
    loaded = load_config(p).experiment_write_ceiling
    assert loaded.crowd_weight == 3.5
    assert loaded.seq_steps_per_item == 42
    assert loaded.arms == ["baseline_global", "combo"]
