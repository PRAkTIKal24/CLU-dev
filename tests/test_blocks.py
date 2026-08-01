"""Tier-iii block: the memory cells, the shared shell, and the matched swap.

⚠ The pre-existing ``CLUBlock``/``SequenceModel`` surface is covered by
``tests/test_primitive_harness.py``; this module tests the C2W4 additions and
one **regression guard** on the shared history (``CLUBlock``'s 3-way key split,
which a 4-way split would silently re-randomise for every published w20 cell).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.blocks import (
    CLU_READ_MODES,
    MEMORY_CELLS,
    CluStoreCell,
    CLUBlock,
    EchoMemoryCell,
    MatchedGRUCell,
    MatchedTTTCell,
    NullMemoryCell,
    StreamMemoryConfig,
    StreamModel,
    _count,
    blank_plan,
    make_memory_cell,
    round_robin_plan,
    solve_matched_gru,
    solve_matched_ttt,
    store_byte_law,
    swap_ledger,
)
from chlu.core.clu_system import CluSystemConfig


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.2).

    ⚠ **Module scope is load-bearing, not tidiness.** Several repo test modules
    (`test_lattice`, `test_goldstone`, `test_twins`, ...) call
    ``jax.config.update("jax_enable_x64", True)`` at MODULE IMPORT, so x64 is
    globally ON in a full-suite run. A *function*-scoped fixture is set up AFTER
    the module-scoped ones, so the store cell would be constructed in float64 and
    then exercised in float32 — 10 tests in this file failed exactly that way in
    the full suite while passing in isolation. An autouse module-scoped fixture
    runs before every other fixture in the module.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def cfgs():
    scfg = CluSystemConfig(addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=32,
                           min_atoms=64, min_atoms_base=32)
    mcfg = StreamMemoryConfig(chunk=8, address_steps=6, read_steps=6, traj_stride=3,
                              psi_hidden=8, write_inner_steps=1, write_n_perturb=4,
                              retry_rounds=1, conv_kernel=3, mlp_mult=2)
    return scfg, mcfg


@pytest.fixture(scope="module")
def cell(cfgs):
    scfg, mcfg = cfgs
    return CluStoreCell(scfg, mcfg, key=jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# regression guard on the shared history
# ---------------------------------------------------------------------------
def test_clublock_key_split_is_still_three_way():
    """⛔ ``CLUBlock`` uses a 3-way split; a 4-way split re-randomises every
    published w20 CLU cell. Asserted against an explicit reconstruction."""
    key = jax.random.PRNGKey(11)
    blk = CLUBlock(6, 4, key=key)
    k1, k2, k3 = jax.random.split(key, 3)
    assert jnp.allclose(blk.w_in.weight, eqx.nn.Linear(6, 4, key=k2).weight)
    assert jnp.allclose(blk.w_out.weight, eqx.nn.Linear(8, 6, key=k3).weight)
    assert CLU_READ_MODES == ("endpoint", "trajectory")
    _ = k1


# ---------------------------------------------------------------------------
# the store cell IS the full store, used not edited
# ---------------------------------------------------------------------------
def test_store_cell_holds_a_learned_v_theta_not_an_array(cell, cfgs):
    """The items live in ``V_theta``; the cell's state IS the atom leaves."""
    scfg, _ = cfgs
    st = cell.init_state()
    assert st.centers.shape == (scfg.n_atoms, scfg.dim)
    assert st.amp.shape == (scfg.n_atoms,)
    assert st.codebook.shape == (scfg.capacity, scfg.dim)
    led = cell.cell_ledger()
    assert led["v_theta_floats"] == scfg.n_atoms * (scfg.dim + 2)
    assert led["state_floats"] == led["v_theta_floats"] + scfg.capacity * scfg.dim


def test_write_is_masked_and_local_in_parameter_space(cell, cfgs):
    """⭐ C3 locality: writing slot ``s`` leaves every other group BIT-IDENTICAL."""
    scfg, _ = cfgs
    st = cell.init_state()
    plan = round_robin_plan(2, scfg.capacity, scfg.dim)
    p0 = jax.tree_util.tree_map(lambda a: a[0], plan)
    z = jnp.linspace(0.1, 0.5, scfg.dim)
    st2 = cell.write(st, z, p0)
    own = np.asarray(cell.group_matrix[int(p0.slot)], dtype=bool)
    assert not np.allclose(np.asarray(st2.centers[own]), np.asarray(st.centers[own]))
    assert np.array_equal(np.asarray(st2.centers[~own]), np.asarray(st.centers[~own]))
    assert np.array_equal(np.asarray(st2.amp[~own]), np.asarray(st.amp[~own]))


def test_a_refused_offer_leaves_the_landscape_bit_identical(cell, cfgs):
    """Admission is a real gate: ``admitted = 0`` must not move ``V_theta``."""
    scfg, _ = cfgs
    st = cell.init_state()
    plan = round_robin_plan(1, scfg.capacity, scfg.dim)
    p0 = jax.tree_util.tree_map(lambda a: a[0], plan)
    refused = p0._replace(admitted=jnp.asarray(0.0))
    st2 = cell.write(st, jnp.ones(scfg.dim) * 0.3, refused)
    for a, b in ((st.centers, st2.centers), (st.amp, st2.amp),
                 (st.log_width, st2.log_width), (st.codebook, st2.codebook)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_eviction_redraws_the_freed_group_it_does_not_zero_it(cell, cfgs):
    """⚠ Zeroing starves the next item in the slot AND leaks membership
    (``LearnedVStore.reinit_group``). A reset must restore the INIT draw."""
    scfg, _ = cfgs
    st = cell.init_state()
    plan = round_robin_plan(2, scfg.capacity, scfg.dim)
    p0 = jax.tree_util.tree_map(lambda a: a[0], plan)
    dirty = cell.write(st, jnp.ones(scfg.dim) * 0.4, p0)
    reset = p0._replace(admitted=jnp.asarray(0.0),
                        reset=jnp.zeros((scfg.capacity,)).at[int(p0.slot)].set(1.0))
    clean = cell.write(dirty, jnp.zeros(scfg.dim), reset)
    own = np.asarray(cell.group_matrix[int(p0.slot)], dtype=bool)
    assert np.allclose(np.asarray(clean.centers[own]), np.asarray(st.centers[own]))
    assert not np.allclose(np.asarray(clean.amp[own]), 0.0)   # NOT zeroed


def test_lifetimes_shallow_only_the_targeted_group(cell, cfgs):
    scfg, _ = cfgs
    st = cell.init_state()
    plan = round_robin_plan(1, scfg.capacity, scfg.dim)
    p0 = jax.tree_util.tree_map(lambda a: a[0], plan)
    decayed = p0._replace(admitted=jnp.asarray(0.0),
                          group_scale=jnp.ones((scfg.capacity,)).at[1].set(0.5))
    st2 = cell.write(st, jnp.zeros(scfg.dim), decayed)
    m1 = np.asarray(cell.group_matrix[1], dtype=bool)
    assert np.allclose(np.asarray(st2.amp[m1]), 0.5 * np.asarray(st.amp[m1]))
    assert np.array_equal(np.asarray(st2.amp[~m1]), np.asarray(st.amp[~m1]))


def test_read_returns_the_latent_and_the_payload_channel_is_launched_at_zero(cell, cfgs):
    """The read must RECOVER the payload from ``V_theta``, never be handed it."""
    scfg, _ = cfgs
    st = cell.init_state()
    z = jnp.arange(scfg.dim, dtype=jnp.float32) + 1.0
    r = cell.read(st, z)
    assert r.shape == (scfg.dim,)
    # two queries differing ONLY in the payload block must read identically
    z2 = z.at[scfg.addr_dim:].set(-99.0)
    assert np.allclose(np.asarray(r), np.asarray(cell.read(st, z2)), atol=1e-6)


# ---------------------------------------------------------------------------
# ⭐ the design rule that makes the whole build shaped the way it is
# ---------------------------------------------------------------------------
def test_settled_point_read_sends_no_gradient_to_the_query(cell, cfgs):
    """⛔ ``d q*/d q0 = 0`` EXACTLY, so a settled-point read cannot train phi.

    Measured here on the read in isolation (no write path), which is the clean
    form: the trajectory read's query-gradient is O(1e-2), the settled point's
    is exactly 0.0.
    """
    scfg, _ = cfgs
    st = cell.init_state()
    z = jnp.linspace(-0.4, 0.4, scfg.dim)

    def L(zz, mode):
        return jnp.sum(cell.read(st, zz, None, mode) ** 2)

    g_traj = jax.grad(L)(z, "trajectory")
    g_point = jax.grad(L)(z, "settled_point")
    assert float(jnp.linalg.norm(g_point)) == 0.0
    assert float(jnp.linalg.norm(g_traj)) > 0.0


def test_gamma_and_mass_selectors_are_trainable_only_through_the_trajectory(cell, cfgs):
    """γ / M are selectors ONLY through the trajectory channel (§A13 rule 3)."""
    scfg, _ = cfgs
    st = cell.init_state()
    z = jnp.linspace(-0.4, 0.4, scfg.dim)

    def L(c, mode):
        return jnp.sum(c.read(st, z, None, mode) ** 2)

    for mode, expect_zero in (("settled_point", True), ("trajectory", False)):
        g = eqx.filter_grad(lambda c, m=mode: L(c, m))(cell)
        n = float(jnp.abs(g.log_gamma_addr) + jnp.abs(g.log_gamma_read))
        nm = float(jnp.linalg.norm(g.clu.log_mass))
        assert (n == 0.0) is expect_zero
        assert (nm == 0.0) is expect_zero


# ---------------------------------------------------------------------------
# ⭐ the system-level swap
# ---------------------------------------------------------------------------
def test_matched_gru_hits_params_and_provably_misses_state(cell, cfgs):
    """⛔ The D2 finding, as a test: a GRU cannot match BOTH axes."""
    scfg, _ = cfgs
    led = cell.cell_ledger()
    h = solve_matched_gru(led["params"], scfg.dim)
    gru = MatchedGRUCell(scfg.dim, h, key=jax.random.PRNGKey(0))
    g = gru.cell_ledger()
    assert abs(g["params"] - led["params"]) / led["params"] < 0.05
    assert led["state_floats"] / g["state_floats"] > 10.0
    # and matching the STATE would blow the parameter budget by orders
    hs = led["state_floats"]
    params_at_matched_state = 3 * hs * hs + (4 * scfg.dim + 5) * hs + scfg.dim ** 2
    assert params_at_matched_state > 100 * led["params"]


def test_matched_ttt_hits_both_axes(cell, cfgs):
    """⭐ A TTT-class cell CAN match both, because its params ARE its state."""
    scfg, _ = cfgs
    led = cell.cell_ledger()
    k, n = solve_matched_ttt(led["params"], led["state_floats"], scfg.dim)
    ttt = MatchedTTTCell(scfg.dim, k, n, key=jax.random.PRNGKey(0))
    t = ttt.cell_ledger()
    assert abs(t["params"] - led["params"]) / led["params"] < 0.05
    assert abs(t["state_floats"] - led["state_floats"]) / led["state_floats"] < 0.05


def test_zero_byte_controls_have_no_state_and_no_parameters(cfgs):
    scfg, _ = cfgs
    for c in (NullMemoryCell(latent_dim=scfg.dim), EchoMemoryCell(latent_dim=scfg.dim)):
        led = c.cell_ledger()
        assert led == {"params": 0, "state_floats": 0, "state_bytes": 0}
        assert _count(c) == 0


def test_swap_ledger_reports_both_axes_against_the_clu(cell, cfgs):
    scfg, _ = cfgs
    led = cell.cell_ledger()
    h = solve_matched_gru(led["params"], scfg.dim)
    L = swap_ledger({"clu_store": cell,
                     "gru_matched": MatchedGRUCell(scfg.dim, h,
                                                   key=jax.random.PRNGKey(0))})
    assert L["clu_store"]["params_matched_pct"] == 0.0
    assert L["gru_matched"]["clu_state_over_arm"] > 10.0


def test_byte_law_matches_the_corrected_form_and_its_floor():
    """``[A(D+2)+d]/(d+m)``; floor **2.40x at n_spec = 1** (errata 24/28)."""
    assert store_byte_law(1, 6, 4, 1) == pytest.approx(2.40)
    assert store_byte_law(256, 5, 4, 1) == pytest.approx((256 * 7 + 4) / 5)


# ---------------------------------------------------------------------------
# the block and the model
# ---------------------------------------------------------------------------
def _model(name, cfgs, key=jax.random.PRNGKey(3), n_layers=2, d_model=16, T=16):
    scfg, mcfg = cfgs
    led = CluStoreCell(scfg, mcfg, key=jax.random.PRNGKey(0)).cell_ledger()
    h = solve_matched_gru(led["params"], scfg.dim)
    kn = solve_matched_ttt(led["params"], led["state_floats"], scfg.dim)
    ks = jax.random.split(key, n_layers + 4)
    cells = [make_memory_cell(name, latent_dim=scfg.dim, clu_cfg=scfg, mcfg=mcfg,
                              hidden=h, ttt_shape=kn, key=ks[i])
             for i in range(n_layers)]
    return StreamModel(vocab_size=32, d_model=d_model, n_layers=n_layers, max_len=T,
                       cells=cells, mcfg=mcfg, latent_dim=scfg.dim,
                       addr_dim=scfg.addr_dim, payload_dim=scfg.payload_dim,
                       key=ks[-1]), scfg, mcfg


@pytest.mark.parametrize("name", MEMORY_CELLS)
def test_every_arm_runs_the_stream_and_shares_a_bit_identical_shell(name, cfgs):
    from chlu.training.train_cluformer import assert_shared_shell_identical

    m, scfg, mcfg = _model(name, cfgs)
    T = 16
    plans = [round_robin_plan(T // mcfg.chunk, scfg.capacity, scfg.dim)
             for _ in range(m.n_layers)]
    out = m(jnp.arange(T) % 32, plans)
    assert out.shape == (T, 32)
    assert bool(jnp.all(jnp.isfinite(out)))
    ref, _, _ = _model("none", cfgs)
    assert_shared_shell_identical({"none": ref, name: m})


def test_the_block_is_causal_in_the_chunk_sense(cfgs):
    """⭐ Read-before-write + the one-chunk shift ⇒ no token sees its own chunk's
    pooled summary through the memory, and no later token can affect an earlier
    one. Perturbing the LAST chunk must not move the FIRST chunk's logits."""
    T = 24
    m, scfg, mcfg = _model("clu_store", cfgs, T=T)
    C = mcfg.chunk
    plans = [round_robin_plan(T // C, scfg.capacity, scfg.dim) for _ in range(m.n_layers)]
    a = jnp.arange(T, dtype=jnp.int32) % 32
    b = a.at[-1].set((int(a[-1]) + 7) % 32)
    ya, yb = m(a, plans), m(b, plans)
    assert np.allclose(np.asarray(ya[:C]), np.asarray(yb[:C]), atol=1e-5)
    assert not np.allclose(np.asarray(ya[-1]), np.asarray(yb[-1]), atol=1e-5)


def test_blank_plan_never_writes_the_store(cfgs):
    """The blank/leak control (collapse mode #4) must really store nothing."""
    scfg, mcfg = cfgs
    c = CluStoreCell(scfg, mcfg, key=jax.random.PRNGKey(0))
    st = c.init_state()
    bp = blank_plan(3, scfg.capacity, scfg.dim)
    for i in range(3):
        st = c.write(st, jnp.ones(scfg.dim) * 0.5,
                     jax.tree_util.tree_map(lambda a, j=i: a[j], bp))
    ref = c.init_state()
    assert np.array_equal(np.asarray(st.centers), np.asarray(ref.centers))
    assert np.array_equal(np.asarray(st.amp), np.asarray(ref.amp))


def test_gradients_reach_phi_the_store_and_psi_through_the_block(cfgs):
    """⭐ S2 in miniature: the end-to-end path ``token -> phi -> store ->
    trajectory psi -> loss`` carries gradient to all three."""
    m, scfg, mcfg = _model("clu_store", cfgs)
    T = 16
    plans = [round_robin_plan(T // mcfg.chunk, scfg.capacity, scfg.dim)
             for _ in range(m.n_layers)]
    toks = jnp.arange(T, dtype=jnp.int32) % 32

    def loss(mm):
        return jnp.mean(mm(toks, plans) ** 2)

    g = eqx.filter_grad(loss)(m)
    for sub in (g.blocks[0].phi, g.blocks[0].cell.psi,
                g.blocks[0].cell.clu.potential_net):
        n = sum(float(jnp.sum(x ** 2))
                for x in jax.tree_util.tree_leaves(eqx.filter(sub, eqx.is_inexact_array)))
        assert n > 0.0
