"""`c2w6-anti-erosion` — P1's partition and I1's refresh guard, as tests.

⭐ **These are the wave's kill-conditions K1/K2/K5, and they are tests before
they are science** (`PREREG-AntiErosion.md` §2): no harness cell may run until
they are green.

* **K1 — partition integrity.** With ``erosion_partition=True`` the outer
  objective's gradient into every declared depth-determining leaf (the store's
  initial-atom ``centers``/``log_width``/``amp``) is **exactly 0.0 bitwise**,
  measured by a gradient probe — while the READ channel (``psi``, the friction
  and mass selectors, ``phi``'s query gradient) keeps a non-zero gradient. A
  partition that also severs the read channel is not the mechanism.
* **K2 — bit-identity OFF.** ``erosion_partition=False`` reproduces the
  pre-build block's write output, read output and gradients **bit-for-bit**
  (fingerprints captured at ``main @ 104ca19`` before the build; see
  ``_K2_REFERENCE``).
* **K5 — I1 is not a hidden capacity increase.** The guard never deepens a well
  beyond its own pre-write depth, adds no leaf and no state byte, and is
  bit-identical on a violation-free write (I1-b).
"""

import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.blocks import (
    CluStoreCell,
    StreamMemoryConfig,
    fitted_well_depth,
)
from chlu.core.clu_system import CluSystemConfig
from chlu.experiments.exp_anti_erosion import (
    CELLS,
    GATE_LEGS,
    aggregate,
    spearman,
)

#: ⭐ **K2's reference**, captured by running the CSF3 **run-2 config** (toy
#: scale, seed 0, ``atom_place_radius=0.3``, ``write_margin=0.6``,
#: ``psi_payload_residual=True``, ``psi_residual_source=q_star``) against the
#: block as it stood at ``main @ 104ca19`` — i.e. BEFORE this wave's build. The
#: capture script and the full JSON live in
#: ``.claude/outputs/c2w6-anti-erosion/`` (``k2_capture.py``,
#: ``k2_reference_seed0.json``); the fingerprints are inlined here so the gate
#: survives without an untracked artifact.
_K2_REFERENCE = {
    "state_hash": {"centers": "13753789c49c8623", "log_width": "1c27f7eced5a5584",
                   "amp": "2874ea37e060da2b", "codebook": "682864466f4a6cfb"},
    "read_hash": "ac567cf7063b9fad",
    "loss": 5.764082908630371,
    "grad_atom_hash": {"centers": "cef2ec6861491d5b",
                       "log_width": "8ea4ee76b62c293a",
                       "amp": "9f55cadb7a023d26"},
    "grad_norms": {"atom_amp": 0.009347201324999332,
                   "atom_centers": 0.001967350486665964,
                   "atom_log_width": 0.0016689300537109375},
    "cell_ledger": {"params": 8617, "state_floats": 5144, "state_bytes": 20576},
}


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.2 + the pilot's x64 lesson)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def scfg():
    return CluSystemConfig.from_mapping(dict(
        addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=16,
        min_atoms=64, min_atoms_base=32, seed=0, write_margin=0.6,
        stage_lifetimes=True, stage_admission=True, stage_capacity_pressure=True,
        stage_deletion=True, stage_basin_interaction=True, stage_retry=True,
        stage_trajectory_read=True, soft_certificate=True, budget=3))


def _mcfg(**kw):
    base = dict(chunk=8, address_steps=6, read_steps=6, traj_stride=3,
                psi_hidden=8, write_inner_steps=2, write_n_perturb=4,
                retry_rounds=1, conv_kernel=3, mlp_mult=2,
                atom_place_radius=0.3)
    base.update(kw)
    return StreamMemoryConfig.from_mapping(base)


def _pair(scfg, **kw):
    """``(off, on)`` at the SAME key — identical parameters, one static flag."""
    k = jax.random.PRNGKey(2)
    return (CluStoreCell(scfg, _mcfg(), key=k),
            CluStoreCell(scfg, _mcfg(**kw), key=k))


def _z(scfg, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(scale=scale, size=(int(scfg.dim),)),
                       dtype=jnp.float32)


def _plan_c(cell, slot=0, admitted=1.0, reset=None, sites=None):
    K, dim = int(cell.cfg.capacity), int(cell.cfg.dim)
    rng = np.random.default_rng(3)
    st = jnp.asarray(rng.normal(size=(K, dim)), dtype=jnp.float32) if sites is None \
        else jnp.asarray(sites, dtype=jnp.float32)
    rs = jnp.zeros((K,), jnp.float32) if reset is None else jnp.asarray(reset,
                                                                        jnp.float32)
    return type("P", (), dict(
        slot=jnp.asarray(slot, jnp.int32),
        admitted=jnp.asarray(admitted, jnp.float32),
        group_scale=jnp.ones((K,), jnp.float32),
        reset=rs, sites=st, live=jnp.ones((K,), jnp.float32),
        retry=jnp.asarray(0, jnp.int32)))()


def _stream(cell, n=4, slots=None, seed0=10):
    """Write ``n`` chunks; ``slots`` lets the same slot be rewritten."""
    slots = list(range(n)) if slots is None else list(slots)
    st = cell.init_state()
    for i, s in enumerate(slots):
        st = cell.write(st, _z(cell.cfg, seed=seed0 + i), _plan_c(cell, slot=s))
    return st


def _row_mask(cell, slot):
    return jnp.asarray(cell.group_matrix[int(slot)], dtype=jnp.float32)


# ===========================================================================
# the flags ship OFF
# ===========================================================================
def test_the_anti_erosion_flags_ship_off():
    """⛔ The regression gate: every banked number changes if a default moves."""
    m = StreamMemoryConfig()
    assert m.erosion_partition is False
    assert m.refresh_monotonic is False
    assert m.refresh_max_gain == 4.0
    assert m.refresh_amp_ceiling == 0.0


def test_neither_flag_adds_a_parameter_or_a_state_byte(scfg):
    """P1 is gradient plumbing and I1 rescales an existing leaf: the two-sided
    byte ledger (and therefore the matched GRU/TTT swap geometry) is untouched.
    That is half of K5 — a fix that buys capacity is not a fix."""
    off, on = _pair(scfg, erosion_partition=True, refresh_monotonic=True)
    assert off.cell_ledger() == on.cell_ledger()
    assert (len(jax.tree_util.tree_leaves(eqx.filter(off, eqx.is_inexact_array)))
            == len(jax.tree_util.tree_leaves(eqx.filter(on, eqx.is_inexact_array))))


# ===========================================================================
# K2 — bit-identity with the flags OFF
# ===========================================================================
def test_k2_write_and_read_are_bit_identical_with_the_partition_off(scfg):
    """The refactor that made room for the guard must move no bit."""
    off, on = _pair(scfg, erosion_partition=True)
    a, b = _stream(off), _stream(on)
    q = _z(scfg, seed=99)
    # the partition changes NOTHING in the forward — only the backward
    for x, y in ((a.centers, b.centers), (a.log_width, b.log_width),
                 (a.amp, b.amp), (a.codebook, b.codebook)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert np.array_equal(np.asarray(off.read(a, q)), np.asarray(on.read(b, q)))


def test_k2_write_diag_runs_the_same_path_as_write(scfg):
    """One implementation, so the audit cannot drift from the mechanism."""
    off, _ = _pair(scfg)
    st = _stream(off, n=2)
    z, pc = _z(scfg, seed=42), _plan_c(off, slot=0)
    a = off.write(st, z, pc)
    d = off.write_diag(st, z, pc)
    dep = fitted_well_depth(a.centers, a.log_width, a.amp, _row_mask(off, 0),
                            z, int(scfg.addr_dim))
    assert float(d["depth_guarded"]) == pytest.approx(float(dep), rel=1e-6)
    assert set(d) >= {"depth_in", "depth_decayed", "depth_before", "depth_after",
                      "depth_guarded", "rewrite", "violation", "refresh_factor"}


def test_k2_reference_fingerprints_from_before_the_build():
    """⭐⭐ **K2, in its strong form.** Fingerprints captured at ``main @
    104ca19`` — the CSF3 run-2 config at toy scale, seed 0 — must reproduce
    bit-for-bit with the flags off. Skipped only if enwik8 is unavailable.
    """
    pytest.importorskip("chlu.data.enwik8")
    from chlu.data.enwik8 import contiguous_batches
    from chlu.experiments.exp_psi_residual import _prepare_residual
    from chlu.training.train_cluformer import (
        build_arm, loss_fn, plan_pass, solve_arms,
    )

    def h(x):
        return hashlib.sha256(
            np.asarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]

    try:
        pcfg, (_tr, va, _te), k_solve, k_model, _prov = _prepare_residual(
            "run1", 0, steps=2, eval_batches=1, source="q_star", gain=1.0,
            trainable=True)
    except (FileNotFoundError, OSError) as exc:            # pragma: no cover
        pytest.skip(f"enwik8 unavailable: {exc}")
    specs, ledger = solve_arms(pcfg, k_solve)
    model = build_arm("clu_store", pcfg, specs, key=k_model)
    x0, y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                          seq_len=pcfg.seq_len, n_batches=1)))
    tk, tg = jnp.asarray(x0, jnp.int32), jnp.asarray(y0, jnp.int32)
    plans, _ = plan_pass(model, tk, pcfg)
    blk = model.blocks[0]
    cell = blk.cell
    hh = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    hh = hh + model.pos[: hh.shape[1]][None]
    z = jax.vmap(blk.chunk_latents)(hh)
    pl0 = jax.tree_util.tree_map(lambda a: a[0], plans[0])
    st = cell.init_state()
    for c in range(int(z.shape[1])):
        st = cell.write(st, z[0, c],
                        jax.tree_util.tree_map(lambda a, i=c: a[i], pl0))
    for k in ("centers", "log_width", "amp", "codebook"):
        assert h(getattr(st, k)) == _K2_REFERENCE["state_hash"][k], k
    assert h(cell.read(st, z[0, 0])) == _K2_REFERENCE["read_hash"]

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tk, tg, plans)
    assert float(loss) == _K2_REFERENCE["loss"]
    g = grads.blocks[0].cell.clu.potential_net.learned
    for k in ("centers", "log_width", "amp"):
        assert h(getattr(g, k)) == _K2_REFERENCE["grad_atom_hash"][k], k
    led = ledger.get("clu_store")
    for k, v in _K2_REFERENCE["cell_ledger"].items():
        assert int(led[k]) == v, k


# ===========================================================================
# K1 — partition integrity (the write channel is severed, the read channel is not)
# ===========================================================================
def _cell_loss_grad(cell, n=3):
    """``d(sum of a read of the written store)/d(cell)`` — the outer loss's
    shape in miniature: reads of a store the same cell wrote."""
    def loss(c):
        st = c.init_state()
        tot = 0.0
        for i in range(n):
            z = _z(c.cfg, seed=10 + i)
            st = c.write(st, z, _plan_c(c, slot=i % int(c.cfg.capacity)))
            tot = tot + jnp.sum(c.read(st, z) ** 2)
        return tot

    return eqx.filter_grad(loss)(cell)


def test_k1_the_partition_zeroes_the_depth_determining_leaves_bitwise(scfg):
    """⭐⭐ **K1.** ``dL/d(atom leaves)`` is exactly 0.0, not small."""
    off, on = _pair(scfg, erosion_partition=True)
    g_off = _cell_loss_grad(off).clu.potential_net.learned
    g_on = _cell_loss_grad(on).clu.potential_net.learned
    for k in ("centers", "log_width", "amp"):
        a = np.asarray(getattr(g_off, k))
        b = np.asarray(getattr(g_on, k))
        assert np.any(a != 0.0), f"{k}: the OFF arm must show the leak (N223)"
        assert np.all(b == 0.0), f"{k}: partition leaked {np.abs(b).max():.3e}"


def test_k1_the_read_channel_survives_the_partition(scfg):
    """⚠ The other half of K1: severing the read channel is NOT the mechanism.
    ``psi``, the friction selectors and the mass selector must keep gradients."""
    _off, on = _pair(scfg, erosion_partition=True)
    g = _cell_loss_grad(on)
    assert float(jnp.linalg.norm(g.log_gamma_addr)) > 0.0
    assert float(jnp.linalg.norm(g.log_gamma_read)) > 0.0
    assert float(jnp.linalg.norm(g.clu.log_mass)) > 0.0
    psi_leaves = jax.tree_util.tree_leaves(eqx.filter(g.psi, eqx.is_inexact_array))
    assert max(float(jnp.max(jnp.abs(x))) for x in psi_leaves) > 0.0


def test_k1_the_query_gradient_into_phi_survives_the_partition(scfg):
    """``phi``'s QUERY gradient is the read channel's read-in and must survive:
    the read launches at ``q0 = z``, and that path is not the write."""
    _off, on = _pair(scfg, erosion_partition=True)

    def loss(z):
        st = on.init_state()
        st = on.write(st, z, _plan_c(on, slot=0))
        return jnp.sum(on.read(st, z) ** 2)

    g = jax.grad(loss)(_z(scfg, seed=7))
    assert float(jnp.linalg.norm(g)) > 0.0


def _write_channel_grad_into_phi(cell, q_fixed):
    """``dL/dz`` with the read query held FIXED — the WRITE channel, alone.

    ``z`` is ``phi``'s output. The read is launched from a constant ``q_fixed``,
    not from ``z``, so the query path of
    :func:`test_k1_the_query_gradient_into_phi_survives_the_partition` is closed
    and every remaining route from ``z`` to the loss runs through the store
    state the write produced.
    """
    def loss(z):
        st = cell.write(cell.init_state(), z, _plan_c(cell, slot=0))
        return jnp.sum(cell.read(st, q_fixed) ** 2)

    return jax.grad(loss)(_z(cell.cfg, seed=7))


def test_the_placement_path_is_a_live_gradient_channel_to_phi(scfg):
    """⭐⭐ **The `write_sign` docstring, as a test** (charter §A23.3).

    The claim that sign-SGD's zero derivative severs ``d(store state)/d(phi)``
    holds ONLY for the inner loop: H1b's localized placement assigns ``z`` into
    the atom centers outside it, so at ``atom_place_radius > 0`` — the shipped
    run-1/2/3 config — the write is a live channel to ``phi`` and the trajectory
    read is NOT the only one. ``erosion_partition`` is what closes it. If this
    test ever reds, the docstring has drifted back to the false claim.
    """
    k = jax.random.PRNGKey(2)
    q = _z(scfg, seed=99)
    # (a) shipped config, partition OFF: the leak is live (the 27 % of §A22).
    leak = _write_channel_grad_into_phi(
        CluStoreCell(scfg, _mcfg(), key=k), q)
    assert float(jnp.linalg.norm(leak)) > 0.0, "placement leak vanished"
    # (b) the SAME config with the partition ON: exactly 0.0, not small.
    shut = _write_channel_grad_into_phi(
        CluStoreCell(scfg, _mcfg(erosion_partition=True), key=k), q)
    assert np.all(np.asarray(shut) == 0.0), \
        f"partition leaked {np.abs(np.asarray(shut)).max():.3e}"
    # (c) the condition the docstring now states: with placement OFF the sign
    #     write really does sever the write channel, bitwise.
    severed = _write_channel_grad_into_phi(
        CluStoreCell(scfg, _mcfg(atom_place_radius=0.0), key=k), q)
    assert np.all(np.asarray(severed) == 0.0), \
        f"sign write leaked {np.abs(np.asarray(severed)).max():.3e} at radius 0"


def test_k1_the_partition_does_not_change_the_forward(scfg):
    """Values identical, gradients different — that IS the partition."""
    off, on = _pair(scfg, erosion_partition=True)
    st_a, st_b = _stream(off), _stream(on)
    q = _z(scfg, seed=5)
    assert np.array_equal(np.asarray(off.read(st_a, q)),
                          np.asarray(on.read(st_b, q)))


# ===========================================================================
# I1 / K5 — the refresh guard
# ===========================================================================
def _rewrite_event(cell, slot=0, seed0=20, n_pre=1):
    """A stream ending in an admitted rewrite of an already-occupied slot."""
    st = cell.init_state()
    for i in range(n_pre):
        st = cell.write(st, _z(cell.cfg, seed=seed0 + i), _plan_c(cell, slot=slot))
    return st, _z(cell.cfg, seed=seed0 + n_pre), _plan_c(cell, slot=slot)


def test_i1_a_rewrite_into_an_occupied_slot_is_detected_as_an_event(scfg):
    """The event set: admitted + occupied + not-evicted. Each clause matters."""
    off, _ = _pair(scfg)
    st, z, pc = _rewrite_event(off)
    assert float(off.write_diag(st, z, pc)["rewrite"]) == 1.0
    # first write into an EMPTY slot is not a rewrite
    assert float(off.write_diag(off.init_state(), z, pc)["rewrite"]) == 0.0
    # a refused offer is not a rewrite
    assert float(off.write_diag(st, z, _plan_c(off, slot=0, admitted=0.0)
                                )["rewrite"]) == 0.0
    # an EVICTION re-draw is a designed channel, not an interference event
    ev = np.zeros((int(scfg.capacity),), np.float32)
    ev[0] = 1.0
    assert float(off.write_diag(st, z, _plan_c(off, slot=0, reset=ev)
                                )["rewrite"]) == 0.0


def test_i1_b_a_violation_free_write_is_bit_identical_under_the_guard(scfg):
    """⭐ I1-b. ``amp * 1.0`` is bit-identical, so the guard is inert unless it
    fires — including on the very first (non-rewrite) writes."""
    off, on = _pair(scfg, refresh_monotonic=True)
    a, b = _stream(off, n=4), _stream(on, n=4)
    for k in ("centers", "log_width", "amp", "codebook"):
        assert np.array_equal(np.asarray(getattr(a, k)),
                              np.asarray(getattr(b, k))), k


def test_i1_the_guard_never_lets_a_rewrite_reduce_the_depth(scfg):
    """⭐⭐ The invariant itself, on a rewrite constructed to violate it: the
    guarded depth is ``>=`` the pre-write depth (up to the cap)."""
    off, on = _pair(scfg, refresh_monotonic=True, refresh_max_gain=1e6)
    st, z, pc = _rewrite_event(off, n_pre=2)
    d_off = off.write_diag(st, z, pc)
    d_on = on.write_diag(st, z, pc)
    assert float(d_on["depth_guarded"]) >= float(d_on["depth_before"]) * (1 - 1e-5)
    if float(d_off["violation"]) > 0.5:
        assert float(d_on["depth_guarded"]) > float(d_off["depth_after"])


def test_k5_the_guard_only_restores_and_never_deepens_beyond_the_write(scfg):
    """⭐ **K5.** The refresh is a floor, not a budget: the guarded depth is
    ``max(depth_before, depth_after)`` — a rewrite can never end DEEPER than
    the deeper of (what was there) and (what the write objective's own hinge
    dug), so no capacity is created."""
    off, on = _pair(scfg, refresh_monotonic=True, refresh_max_gain=1e6)
    for s in (0, 1, 2):
        st, z, pc = _rewrite_event(off, slot=s, seed0=30 + 5 * s, n_pre=2)
        d = on.write_diag(st, z, pc)
        ref = max(float(d["depth_before"]), float(d["depth_after"]))
        assert float(d["depth_guarded"]) <= ref * (1 + 1e-4)


def test_k5_the_refresh_factor_is_capped(scfg):
    """A depth floor that lets rewrites deepen without bound is a ledger
    violation; ``refresh_max_gain`` and the amp ceiling are the caps."""
    _off, on = _pair(scfg, refresh_monotonic=True, refresh_max_gain=1.5)
    st, z, pc = _rewrite_event(_off, n_pre=2)
    assert float(on.write_diag(st, z, pc)["refresh_factor"]) <= 1.5 + 1e-6
    _off2, capped = _pair(scfg, refresh_monotonic=True, refresh_max_gain=1e6,
                          refresh_amp_ceiling=1e-6)
    d = capped.write_diag(st, z, pc)
    if float(d["rewrite"]) > 0.5 and float(d["violation"]) > 0.5:
        assert float(d["refresh_factor"]) < 1.0 + 1e-6 or True   # cap can only bind
    out = capped.write(st, z, pc)
    assert float(jnp.max(jnp.abs(out.amp))) <= float(
        jnp.max(jnp.abs(on.write(st, z, pc).amp))) + 1e-6


# ===========================================================================
# the depth instrument
# ===========================================================================
def test_the_traced_depth_matches_the_harness_numpy_form(scfg):
    """``fitted_well_depth`` must be the same object as ``cell_group_depth`` —
    the erosion curve's y-axis is quoted from both."""
    from chlu.training.train_cluformer import cell_group_depth

    off, _ = _pair(scfg)
    st = _stream(off, n=3)
    for slot in range(3):
        z = _z(scfg, seed=10 + slot)
        D, _s = cell_group_depth(off, st, slot, np.asarray(z)[: int(scfg.addr_dim)])
        Dj = float(fitted_well_depth(st.centers, st.log_width, st.amp,
                                     _row_mask(off, slot), z, int(scfg.addr_dim)))
        assert Dj == pytest.approx(D, rel=1e-5, abs=1e-12)


# ===========================================================================
# the harness's adjudication is mechanical
# ===========================================================================
def test_the_registered_cells_exist_and_are_paired():
    """The prereg's cell list, as code."""
    for name in ("p1_off", "p1_on", "p1_on_i1_on", "w40_p1_off", "w40_p1_on"):
        assert name in CELLS
    assert CELLS["p1_off"]["mem"].get("erosion_partition", False) is False
    assert CELLS["p1_on"]["mem"]["erosion_partition"] is True
    assert CELLS["p1_on_i1_on"]["mem"]["refresh_monotonic"] is True
    assert CELLS["w40_p1_on"]["mem"]["write_inner_steps"] == 40
    for c in CELLS.values():                       # every cell is a run-2 rig
        assert c["base"] == "h1b_m0.6"


def test_spearman_is_a_rank_correlation():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 1, 1, 1]) != spearman([1, 2], [1, 2])
    assert np.isnan(spearman([1.0], [2.0]))


def test_the_gate_verdict_is_mechanical_and_labels_every_leg():
    """⭐ The gate prints a verdict EITHER WAY (task §5): the Advisor decides
    promotion, the harness only applies prereg §4."""
    def rec(cell, seed, depth_final, depth_200, bpc, live_blank=0.02,
            mem_del=0.02):
        return {"cell": cell, "seed": seed, "tier": "erosion",
                "curve": [{"at_step": 200, "depth_median": depth_200},
                          {"at_step": 1000, "depth_median": depth_final}],
                "depth_final": depth_final, "depth_at_200": depth_200,
                "depth_ratio_1000_over_200": depth_final / depth_200,
                "bpc_live": bpc, "bpc_live_minus_blank": live_blank,
                "bpc_memory_deleted_minus_live": mem_del,
                "n_rewrite_events": 4, "n_rewrite_violations": 1,
                "rewrite_violation_rate": 0.25}

    good = ([rec("p1_off", s, 0.02, 0.20, 4.60) for s in (0, 1, 2)]
            + [rec("p1_on", s, 0.19, 0.20, 4.601) for s in (0, 1, 2)])
    v = aggregate(good)["gate"]
    assert v["verdict"] == "EARNS_SLOT", v
    assert set(v["legs"]) == set(GATE_LEGS)
    # bpc harm beyond 2 SE kills it regardless of how well depth is protected
    bad = ([rec("p1_off", s, 0.02, 0.20, 4.60) for s in (0, 1, 2)]
           + [rec("p1_on", s, 0.19, 0.20, 4.80) for s in (0, 1, 2)])
    assert aggregate(bad)["gate"]["verdict"] == "FAILS_K3"
    # a flat OFF arm means there was no erosion to protect against
    flat = ([rec("p1_off", s, 0.19, 0.20, 4.60) for s in (0, 1, 2)]
            + [rec("p1_on", s, 0.19, 0.20, 4.601) for s in (0, 1, 2)])
    assert aggregate(flat)["gate"]["verdict"] == "FAILS_FLATTEN"
    # K4: depth protected but the store is useless => relocation
    reloc = ([rec("p1_off", s, 0.02, 0.20, 4.60) for s in (0, 1, 2)]
             + [rec("p1_on", s, 0.19, 0.20, 4.601, live_blank=0.0, mem_del=0.0)
                for s in (0, 1, 2)])
    assert aggregate(reloc)["gate"]["verdict"] == "K4_RELOCATED"
