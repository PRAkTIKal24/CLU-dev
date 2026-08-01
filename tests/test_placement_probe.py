"""`pilot-placement-probe` — the three placement/write levers, as tests.

These are the tests that would fail if the probe's *claims* stopped being true:
the levers stop being bit-identical when off, the localized init stops matching
the shipped N98 construction, the write-time placement stops being C3-local, a
refused offer stops leaving ``V_theta`` untouched, or the success-signal
adjudication stops matching the pre-registration.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.blocks import CluStoreCell, StreamMemoryConfig, localize_atom_init
from chlu.core.clu_system import CluSystemConfig, LearnedVStore
from chlu.core.memory_potentials import AtomDictionaryPotential, DesignFreedomPotential
from chlu.experiments.exp_placement_probe import (
    CELLS,
    PILOT_SATURATED_DEPTH,
    SHIPPED_DEPTH_BAND,
    aggregate,
)
from chlu.training.train_cluformer import PilotConfig, calibrate_atom_group_centers


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.2 + the pilot's x64 lesson).

    ⚠ Module scope is load-bearing: several repo test modules enable
    ``jax_enable_x64`` at MODULE IMPORT, and a function-scoped fixture is set up
    *after* the module-scoped ones — the store would be built in float64 and
    exercised in float32.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def scfg():
    return CluSystemConfig.from_mapping(dict(
        addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=16,
        min_atoms=64, min_atoms_base=32, seed=0,
        stage_lifetimes=True, stage_admission=True, stage_capacity_pressure=True,
        stage_deletion=True, stage_basin_interaction=True, stage_retry=True,
        stage_trajectory_read=True, soft_certificate=True, budget=3))


def _mcfg(**kw):
    base = dict(chunk=8, address_steps=6, read_steps=6, traj_stride=3,
                psi_hidden=8, write_inner_steps=1, write_n_perturb=4,
                retry_rounds=1, conv_kernel=3, mlp_mult=2)
    base.update(kw)
    return StreamMemoryConfig.from_mapping(base)


def _plan_c(cell, slot=0, admitted=1.0):
    """A one-chunk plan slice for the cell's write."""
    K, dim = int(cell.cfg.capacity), int(cell.cfg.dim)
    rng = np.random.default_rng(3)
    sites = jnp.asarray(rng.normal(size=(K, dim)), dtype=jnp.float32)
    return type("P", (), dict(
        slot=jnp.asarray(slot, jnp.int32),
        admitted=jnp.asarray(admitted, jnp.float32),
        group_scale=jnp.ones((K,), jnp.float32),
        reset=jnp.zeros((K,), jnp.float32),
        sites=sites, live=jnp.ones((K,), jnp.float32),
        retry=jnp.asarray(0, jnp.int32)))()


# ---------------------------------------------------------------------------
# the config stays usable as a STATIC field
# ---------------------------------------------------------------------------
def test_stream_memory_config_stays_hashable_with_the_new_levers():
    """``StreamMemoryConfig`` is an equinox STATIC field: an unhashable knob
    (e.g. a raw ndarray of group centres) would break every ``jit`` cache."""
    m = _mcfg(atom_local_radius=0.6, atom_place_radius=0.3,
              write_lambda_traj=0.3, atom_group_centers=((0.1, 0.2), (0.3, 0.4),
                                                        (0.5, 0.6), (0.7, 0.8)))
    assert isinstance(hash(m), int)
    assert m == _mcfg(atom_local_radius=0.6, atom_place_radius=0.3,
                      write_lambda_traj=0.3,
                      atom_group_centers=((0.1, 0.2), (0.3, 0.4),
                                          (0.5, 0.6), (0.7, 0.8)))


def test_every_new_lever_defaults_to_off():
    """⛔ The regression gate: the probe's levers must ship OFF, or every
    pre-probe number in the repo silently changes."""
    m = StreamMemoryConfig()
    assert m.atom_local_radius == 0.0
    assert m.atom_place_radius == 0.0
    assert m.write_lambda_traj == 0.0
    assert m.atom_group_centers is None


# ---------------------------------------------------------------------------
# H1 — the N98 localized init
# ---------------------------------------------------------------------------
def test_localize_atom_init_is_a_noop_when_disabled(scfg):
    store = LearnedVStore(scfg, jax.random.PRNGKey(1))
    same = localize_atom_init(store, ((0.0, 0.0),) * 4, 0.0,
                              key=jax.random.PRNGKey(1))
    assert np.array_equal(np.asarray(store.V.learned.centers),
                          np.asarray(same.V.learned.centers))


def test_cell_construction_is_bit_identical_at_the_default(scfg):
    a = CluStoreCell(scfg, _mcfg(), key=jax.random.PRNGKey(2))
    b = CluStoreCell(scfg, _mcfg(atom_local_radius=0.0,
                                 atom_group_centers=((0.0, 0.0),) * 4),
                     key=jax.random.PRNGKey(2))
    assert np.array_equal(np.asarray(a._atoms.centers),
                          np.asarray(b._atoms.centers))


def test_localize_atom_init_reproduces_the_shipped_n98_construction(scfg):
    """⭐ The claim this file exists for: the block's localization is the SAME
    mechanism as ``AtomDictionaryPotential(group_centers, local_radius)``, not a
    look-alike. ``LearnedVStore`` never forwards ``cfg.atom_local_radius``, so
    the shipped lever is unreachable through the store's own config — this
    asserts the outside-in reconstruction is bit-for-bit."""
    key = jax.random.PRNGKey(7)
    gc = ((0.4, -0.2), (-0.5, 0.1), (0.0, 0.7), (-0.3, -0.6))
    store = LearnedVStore(scfg, key)
    got = localize_atom_init(store, gc, 0.6, key=key)
    direct = DesignFreedomPotential(
        rung="free_mlp", dim=int(scfg.dim), payloads=jnp.zeros((scfg.capacity,)),
        key=key, learned_family="atoms", n_atoms=int(scfg.n_atoms),
        rbf_init_width=float(scfg.atom_width), confine=float(scfg.confine),
        atom_depth_init=float(scfg.atom_depth_init),
        atom_groups=int(scfg.capacity), atom_init_scale=float(scfg.atom_init_scale),
        atom_group_centers=jnp.asarray(gc), atom_local_radius=0.6)
    assert np.array_equal(np.asarray(got.V.learned.centers),
                          np.asarray(direct.learned.centers))


def test_localization_touches_address_axes_only(scfg):
    """N46: localizing the PAYLOAD axis would hand the write the value it is
    supposed to learn (and destroy the basin-reach property)."""
    key = jax.random.PRNGKey(9)
    store = LearnedVStore(scfg, key)
    gc = ((0.4, -0.2), (-0.5, 0.1), (0.0, 0.7), (-0.3, -0.6))
    loc = localize_atom_init(store, gc, 0.6, key=key)
    d = int(scfg.addr_dim)
    before = np.asarray(store.V.learned.centers)
    after = np.asarray(loc.V.learned.centers)
    assert np.array_equal(before[:, d:], after[:, d:])          # payload untouched
    assert not np.array_equal(before[:, :d], after[:, :d])      # address moved


def test_localized_atoms_land_inside_the_declared_ball(scfg):
    key = jax.random.PRNGKey(11)
    gc = np.array([(0.4, -0.2), (-0.5, 0.1), (0.0, 0.7), (-0.3, -0.6)])
    loc = localize_atom_init(LearnedVStore(scfg, key), tuple(map(tuple, gc)),
                             0.25, key=key)
    c = np.asarray(loc.V.learned.centers)[:, :2]
    n = c.shape[0]
    owner = np.minimum((np.arange(n) * 4) // n, 3)
    assert np.all(np.linalg.norm(c - gc[owner], axis=-1) <= 0.25 + 1e-5)


def test_calibrated_group_centers_are_hashable_and_address_only():
    pcfg = PilotConfig(d_model=16, n_layers=1, seq_len=32, batch=2, vocab_size=32,
                       addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=16,
                       store=dict(min_atoms=64, min_atoms_base=32),
                       memory=dict(chunk=8, address_steps=4, read_steps=4,
                                   traj_stride=2, psi_hidden=8,
                                   write_inner_steps=1, write_n_perturb=2,
                                   conv_kernel=3, mlp_mult=2))
    rng = np.random.default_rng(0)
    tok = rng.integers(0, pcfg.vocab_size, (pcfg.batch, pcfg.seq_len))
    gc = calibrate_atom_group_centers(pcfg, tok, key=jax.random.PRNGKey(0))
    assert isinstance(hash(gc), int)
    assert len(gc) == pcfg.capacity
    assert all(len(r) == pcfg.addr_dim for r in gc)


# ---------------------------------------------------------------------------
# H1b — localized placement AT WRITE
# ---------------------------------------------------------------------------
def test_place_radius_is_bit_identical_when_off(scfg):
    cell = CluStoreCell(scfg, _mcfg(), key=jax.random.PRNGKey(4))
    cell2 = CluStoreCell(scfg, _mcfg(atom_place_radius=0.0),
                         key=jax.random.PRNGKey(4))
    z = jnp.asarray(np.random.default_rng(1).normal(size=(int(scfg.dim),)),
                    dtype=jnp.float32)
    a = cell.write(cell.init_state(), z, _plan_c(cell))
    b = cell2.write(cell2.init_state(), z, _plan_c(cell2))
    assert np.array_equal(np.asarray(a.centers), np.asarray(b.centers))
    assert np.array_equal(np.asarray(a.amp), np.asarray(b.amp))


def test_write_placement_is_c3_local_and_lands_at_the_incoming_address(scfg):
    """⭐ The two properties that make H1b legitimate: only the written slot's
    rows move (C3 locality in parameter space), and they land in the declared
    ball around the INCOMING chunk's address."""
    r = 0.2
    cell = CluStoreCell(scfg, _mcfg(atom_place_radius=r), key=jax.random.PRNGKey(6))
    d = int(scfg.addr_dim)
    z = jnp.asarray([0.9, -0.7, 0.3][: int(scfg.dim)], dtype=jnp.float32)
    st0 = cell.init_state()
    st = cell.write(st0, z, _plan_c(cell, slot=1))
    rows = np.asarray(cell.group_matrix[1], dtype=bool)
    moved = np.asarray(st.centers)[rows][:, :d]
    # inside the ball, up to the 1 inner sign-SGD step of lr 0.05
    assert np.all(np.linalg.norm(moved - np.asarray(z)[:d], axis=-1)
                  <= r + 0.05 * cell.mcfg.write_inner_steps * np.sqrt(d) + 1e-4)
    # every OTHER group's rows are untouched by the placement
    other = ~rows
    assert np.array_equal(np.asarray(st.centers)[other],
                          np.asarray(st0.centers)[other])


def test_a_refused_offer_leaves_the_landscape_bit_identical_under_placement(scfg):
    """The admission gate's contract survives H1b: refusing must not move a
    single atom, or 'the store did not admit this chunk' becomes a lie."""
    cell = CluStoreCell(scfg, _mcfg(atom_place_radius=0.3),
                        key=jax.random.PRNGKey(8))
    z = jnp.asarray(np.random.default_rng(2).normal(size=(int(scfg.dim),)),
                    dtype=jnp.float32)
    st0 = cell.init_state()
    st = cell.write(st0, z, _plan_c(cell, slot=2, admitted=0.0))
    for a, b in ((st0.centers, st.centers), (st0.amp, st.amp),
                 (st0.log_width, st.log_width)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_placement_deepens_the_well_at_the_written_site(scfg):
    """The mechanism claim in one assertion: with the atoms placed at the site
    the same write digs a deeper well than with the scattered init."""
    from chlu.training.train_cluformer import cell_group_depth

    z = jnp.asarray([0.6, -0.4, 0.2][: int(scfg.dim)], dtype=jnp.float32)
    depths = []
    for r in (0.0, 0.3):
        cell = CluStoreCell(scfg, _mcfg(atom_place_radius=r, write_inner_steps=4),
                            key=jax.random.PRNGKey(12))
        st = cell.write(cell.init_state(), z, _plan_c(cell, slot=0))
        depths.append(cell_group_depth(cell, st, 0, np.asarray(z))[0])
    assert depths[1] > depths[0]


# ---------------------------------------------------------------------------
# H2 — the trajectory write term
# ---------------------------------------------------------------------------
def test_lambda_traj_zero_is_bit_identical(scfg):
    """C2W2's coefficient-zero regression gate, re-asserted at the block level:
    at ``write_lambda_traj = 0`` not one extra op is traced and the written
    ``V_theta`` is bit-identical to the shipped objective's."""
    z = jnp.asarray(np.random.default_rng(4).normal(size=(int(scfg.dim),)),
                    dtype=jnp.float32)
    a = CluStoreCell(scfg, _mcfg(), key=jax.random.PRNGKey(13))
    b = CluStoreCell(scfg, _mcfg(write_lambda_traj=0.0), key=jax.random.PRNGKey(13))
    sa = a.write(a.init_state(), z, _plan_c(a))
    sb = b.write(b.init_state(), z, _plan_c(b))
    assert np.array_equal(np.asarray(sa.centers), np.asarray(sb.centers))
    assert np.array_equal(np.asarray(sa.amp), np.asarray(sb.amp))


def test_lambda_traj_is_wired_but_its_gradient_is_vanishing_at_the_flat_init(scfg):
    """⭐⭐ **H2's measured result, pinned as a test.** The term is genuinely
    wired — it evaluates to an ``O(0.1)`` penalty — but its gradient into the
    atoms is ``~1e-13`` at ``atom_depth_init = 1e-4``, because the read path is
    then dominated by the confinement term and carries essentially nothing back.
    Consequence: with the shipped **sign-SGD** inner write (which needs a SIGN
    FLIP, not a magnitude) the written ``V_theta`` is **bit-identical** with and
    without the term. This is the same arithmetic-inertness mechanism the pilot
    found in §6.2, on the other side of the objective.

    ⛔ If this test ever starts failing, H2's null in the probe report no longer
    holds and the finding must be re-measured — that is exactly what it is for.
    """
    from chlu.training.train_memory import trajectory_margin_penalty

    cell = CluStoreCell(scfg, _mcfg(), key=jax.random.PRNGKey(14))
    V = cell.clu.potential_net
    z = jnp.asarray(np.random.default_rng(5).normal(size=(int(scfg.dim),)),
                    dtype=jnp.float32)
    kw = dict(payload_index=int(scfg.addr_dim), payload_dim=int(scfg.payload_dim),
              crowd_targets=_plan_c(cell).sites, sigma_addr=0.25, margin=0.15,
              rollout_steps=6, stride=3, gamma=0.02, dt=0.05, n_launch=2)
    val = float(trajectory_margin_penalty(V, z[None, :], jax.random.PRNGKey(0), **kw))
    assert val > 1e-3, "the term is not wired"
    g = eqx.filter_grad(
        lambda v: trajectory_margin_penalty(v, z[None, :], jax.random.PRNGKey(0),
                                            **kw))(V)
    gmax = max(float(jnp.max(jnp.abs(x)))
               for x in jax.tree_util.tree_leaves(g) if x is not None)
    assert gmax < 1e-8, gmax

    # ...and therefore the write is bit-identical, at every budget and both
    # inner optimisers.
    for ws in (2, 8):
        for sign in (True, False):
            a = CluStoreCell(scfg, _mcfg(write_inner_steps=ws, write_sign=sign),
                             key=jax.random.PRNGKey(14))
            b = CluStoreCell(scfg, _mcfg(write_inner_steps=ws, write_sign=sign,
                                         write_lambda_traj=3.0),
                             key=jax.random.PRNGKey(14))
            sa = a.write(a.init_state(), z, _plan_c(a))
            sb = b.write(b.init_state(), z, _plan_c(b))
            assert np.array_equal(np.asarray(sa.centers), np.asarray(sb.centers))
            assert np.array_equal(np.asarray(sa.amp), np.asarray(sb.amp))


def test_the_cell_ledger_is_unchanged_by_every_probe_lever(scfg):
    """⛔ The byte ledger must not move: the placement jig is key-free, the
    localization is an initialisation, and the trajectory term is an objective —
    none of them is a parameter or a state byte."""
    base = CluStoreCell(scfg, _mcfg(), key=jax.random.PRNGKey(15)).cell_ledger()
    for kw in (dict(atom_local_radius=0.6,
                    atom_group_centers=((0.1, 0.1),) * int(scfg.capacity)),
               dict(atom_place_radius=0.3), dict(write_lambda_traj=0.3)):
        led = CluStoreCell(scfg, _mcfg(**kw), key=jax.random.PRNGKey(15)).cell_ledger()
        assert led == base, kw


def test_gradient_still_flows_to_the_read_under_every_lever(scfg):
    """The probe must not silently sever the trajectory channel it is meant to
    inform (T3's channel is the whole reason the block trains)."""
    z = jnp.asarray(np.random.default_rng(6).normal(size=(int(scfg.dim),)),
                    dtype=jnp.float32)
    for kw in ({}, dict(atom_place_radius=0.3), dict(write_lambda_traj=0.3)):
        cell = CluStoreCell(scfg, _mcfg(**kw), key=jax.random.PRNGKey(16))
        st = cell.write(cell.init_state(), z, _plan_c(cell))

        def f(zz, cell=cell, st=st):
            return jnp.sum(cell.read(st, zz) ** 2)

        g = jax.grad(f)(z)
        assert np.isfinite(np.asarray(g)).all()
        assert float(jnp.linalg.norm(g)) > 0.0, kw


# ---------------------------------------------------------------------------
# the probe's own adjudication
# ---------------------------------------------------------------------------
def test_registered_cells_cover_both_hypotheses_and_their_controls():
    assert "baseline" in CELLS
    assert any(c["mem"].get("atom_local_radius") == 0.6 for c in CELLS.values())
    assert any(c["mem"].get("atom_place_radius") for c in CELLS.values())
    assert any(c["mem"].get("write_lambda_traj") for c in CELLS.values())
    # the N94 40-step floor cells exist (the only non-demoted readings)
    assert any(c["mem"].get("write_inner_steps") == 40 for c in CELLS.values())


def test_success_signal_adjudication_matches_the_prereg():
    """(a) acq off chance by 2 SE, (b) live != blank, (c) depth off 0.045."""
    def rec(acq, chance, blank, depth, gap):
        return {"cell": "baseline", "seed": 0, "tier": "screen",
                "acq": acq, "acq_chance": chance, "blank_acq": blank,
                "acq_minus_blank": acq - blank, "depth_median": depth,
                "bpc_live_minus_blank": gap,
                "read_output": {"read_delta_median": 0.0,
                                "amp_max_deviation": 0.0}}
    # an inert store: at chance, no gap, saturated depth
    inert = aggregate([rec(0.30, 0.30, 0.30, PILOT_SATURATED_DEPTH, 0.0),
                       rec(0.31, 0.30, 0.31, PILOT_SATURATED_DEPTH, 0.0),
                       rec(0.29, 0.30, 0.29, PILOT_SATURATED_DEPTH, 0.0)])
    row = inert["cells"]["screen/baseline"]
    assert not row["signal_a_acq_off_chance"]
    assert not row["signal_b_live_ne_blank"]
    assert not row["signal_c_depth_off_saturation"]
    # a woken store
    d = 0.5 * (SHIPPED_DEPTH_BAND[0] + SHIPPED_DEPTH_BAND[1])
    woke = aggregate([rec(0.80, 0.30, 0.32, d, 0.05),
                      rec(0.79, 0.30, 0.31, d, 0.05),
                      rec(0.81, 0.30, 0.30, d, 0.05)])
    row = woke["cells"]["screen/baseline"]
    assert row["signal_a_acq_off_chance"]
    assert row["signal_a_write_effect"]
    assert row["signal_b_live_ne_blank"]
    assert row["signal_c_depth_off_saturation"]
    assert row["signal_c_depth_in_shipped_band"]


def test_the_blank_control_can_launder_an_acquisition_number():
    """⭐⭐ The laundering control, as a test: an arm whose live AND blank probes
    both sit high is NOT evidence of a write, and the adjudication must say so."""
    def rec(acq, blank):
        return {"cell": "baseline", "seed": 0, "tier": "screen",
                "acq": acq, "acq_chance": 0.30, "blank_acq": blank,
                "acq_minus_blank": acq - blank, "depth_median": 0.6,
                "bpc_live_minus_blank": 0.01,
                "read_output": {"read_delta_median": 0.0,
                                "amp_max_deviation": 0.0}}
    row = aggregate([rec(0.80, 0.80), rec(0.79, 0.79),
                     rec(0.81, 0.81)])["cells"]["screen/baseline"]
    assert row["signal_a_acq_off_chance"]        # off chance...
    assert not row["signal_a_write_effect"]      # ...but bought by the INIT


def test_equinox_filter_sees_no_new_parameters(scfg):
    """The placement jig must never become a leaf (it would be trained, counted
    and shipped)."""
    a = CluStoreCell(scfg, _mcfg(), key=jax.random.PRNGKey(17))
    b = CluStoreCell(scfg, _mcfg(atom_place_radius=0.3), key=jax.random.PRNGKey(17))
    na = len(jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_inexact_array)))
    nb = len(jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_inexact_array)))
    assert na == nb


def test_atom_dictionary_localization_helper_rejects_bad_shapes(scfg):
    store = LearnedVStore(scfg, jax.random.PRNGKey(18))
    with pytest.raises(ValueError):
        localize_atom_init(store, ((0.0, 0.0),) * (int(scfg.capacity) + 1), 0.5,
                           key=jax.random.PRNGKey(18))
    with pytest.raises(ValueError):
        localize_atom_init(store, ((0.0,) * (int(scfg.dim) + 1),) * int(scfg.capacity),
                           0.5, key=jax.random.PRNGKey(18))
    assert isinstance(AtomDictionaryPotential, type)
