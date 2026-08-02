"""`psi-payload-residual` — the payload-carrying read-out residual, as tests.

These are the tests that would fail if the build's *claims* stopped being true:
the residual stops being bit-identical (and parameter-count-identical) when off,
it stops being additive-and-linear in the gate (which is what makes the spread
ledger and the gate sweep arithmetic rather than a grid), it starts touching a
coordinate that is not the payload (the N68 leak guard), or the blank-store read
starts carrying something.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.blocks import (
    CluStoreCell,
    StreamMemoryConfig,
    psi_residual_sources,
)
from chlu.core.clu_system import CluSystemConfig
from chlu.experiments.exp_psi_residual import (
    ACCEPTANCE_RATIO,
    CELLS,
    GATE_GRID,
    PROBE_BEFORE_RATIO,
    _acq,
    _std,
    aggregate,
)


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


def _cells(scfg, **kw):
    """``(off, on)`` at the SAME key — they differ by one leaf and nothing else."""
    k = jax.random.PRNGKey(2)
    return (CluStoreCell(scfg, _mcfg(), key=k),
            CluStoreCell(scfg, _mcfg(psi_payload_residual=True, **kw), key=k))


def _z(scfg, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(scale=0.5, size=(int(scfg.dim),)),
                       dtype=jnp.float32)


def _plan_c(cell, slot=0, admitted=1.0):
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


def _written(cell, n=3):
    st = cell.init_state()
    for i in range(n):
        st = cell.write(st, _z(cell.cfg, seed=10 + i), _plan_c(cell, slot=i))
    return st


# ---------------------------------------------------------------------------
# ships OFF, and off it is bit-identical
# ---------------------------------------------------------------------------
def test_the_residual_ships_off():
    """⛔ The regression gate: every pre-`psi-payload-residual` number in the
    repo silently changes if this default moves."""
    m = StreamMemoryConfig()
    assert m.psi_payload_residual is False
    assert m.psi_residual_source == "q_star"
    assert m.psi_residual_gain == 1.0
    assert m.psi_residual_trainable is True


def test_the_gate_leaf_does_not_exist_when_the_residual_is_off(scfg):
    """``None``, not ``zeros`` — the parameter count and the byte ledger (and
    therefore the matched GRU/TTT swap geometry) must be untouched."""
    off, on = _cells(scfg)
    assert off.psi_res_gate is None
    assert on.psi_res_gate is not None
    assert off.cell_ledger()["params"] < on.cell_ledger()["params"]
    assert (on.cell_ledger()["params"] - off.cell_ledger()["params"]
            == int(scfg.payload_dim))
    # the STATE column is untouched: a read-out gate is parameters, not state
    assert off.cell_ledger()["state_floats"] == on.cell_ledger()["state_floats"]


def test_the_read_is_bit_identical_when_the_residual_is_off(scfg):
    off, on = _cells(scfg)
    st_off, st_on = _written(off), _written(on)
    for i in range(3):
        q = _z(scfg, seed=10 + i)
        a = np.asarray(off.read(st_off, q))
        b = np.asarray(on.read(st_on, q))
        # the two cells' non-gate parameters are identical, so the ONLY
        # difference is the residual; at gate 0 they must agree bit-for-bit
        z = eqx.tree_at(lambda c: c.psi_res_gate, on,
                        jnp.zeros_like(on.psi_res_gate))
        assert np.array_equal(a, np.asarray(z.read(st_on, q)))
        assert not np.array_equal(a, b)          # at gate 1 it does something


def test_the_write_is_bit_identical_with_and_without_the_residual(scfg):
    """N46 / the read-side-only contract: the residual must not reach the write."""
    off, on = _cells(scfg)
    a, b = _written(off), _written(on)
    for x, y in ((a.centers, b.centers), (a.log_width, b.log_width),
                 (a.amp, b.amp), (a.codebook, b.codebook)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# ---------------------------------------------------------------------------
# the mechanism: additive, linear in the gate, payload-only
# ---------------------------------------------------------------------------
def test_the_residual_is_exactly_linear_in_the_gate(scfg):
    """⭐ The property the whole instrument rests on: ``decode(g) = decode(0) +
    g * source``. If this fails, every arithmetically-derived spread and every
    swept acquisition in the report is void."""
    _off, on = _cells(scfg)
    st = _written(on)
    q = _z(scfg, seed=11)
    base = np.asarray(eqx.tree_at(lambda c: c.psi_res_gate, on,
                                  jnp.zeros_like(on.psi_res_gate)).read(st, q))
    unit = np.asarray(on.read(st, q)) - base          # gate = 1
    for g in (0.5, 2.0, 3.0, -1.0):
        cg = eqx.tree_at(lambda c: c.psi_res_gate, on,
                         jnp.full_like(on.psi_res_gate, g))
        got = np.asarray(cg.read(st, q))
        assert np.allclose(got, base + g * unit, atol=1e-5, rtol=0)


def test_the_residual_touches_the_payload_block_and_nothing_else(scfg):
    """⛔ The N68 leak guard, structural form: an ADDRESS-block residual would be
    a query bypass, so the residual must be provably zero off the payload."""
    _off, on = _cells(scfg)
    st = _written(on)
    a, m = int(scfg.addr_dim), int(scfg.payload_dim)
    q = _z(scfg, seed=12)
    base = np.asarray(eqx.tree_at(lambda c: c.psi_res_gate, on,
                                  jnp.zeros_like(on.psi_res_gate)).read(st, q))
    got = np.asarray(on.read(st, q))
    d = got - base
    assert np.all(d[:a] == 0.0)
    assert np.all(d[a + m:] == 0.0)
    assert np.abs(d[a: a + m]).max() > 0.0


def test_the_residual_source_is_the_settled_points_payload(scfg):
    """The residual carries ``q*[payload]`` — the channel §6 measured — and is
    checked against the independently-coded ``read_diag`` settle."""
    _off, on = _cells(scfg)
    st = _written(on)
    a, m = int(scfg.addr_dim), int(scfg.payload_dim)
    q = _z(scfg, seed=13)
    base = np.asarray(eqx.tree_at(lambda c: c.psi_res_gate, on,
                                  jnp.zeros_like(on.psi_res_gate)).read(st, q))
    unit = np.asarray(on.read(st, q))[a: a + m] - base[a: a + m]
    qs = np.asarray(on.read_diag(st, q)["q_star"])[a: a + m]
    assert np.allclose(unit, qs, atol=1e-5, rtol=0)


def test_the_blank_store_carries_essentially_nothing_through_the_residual(scfg):
    """⭐⭐ The laundering control, as a test. The read launches on the
    payload-zero manifold, so on a blank store the residual's source is ~0 — it
    cannot become a query bypass. (The experiment's blank ARM is the measured
    form; this is the structural one.)"""
    _off, on = _cells(scfg)
    blank, live = on.init_state(), _written(on)
    a, m = int(scfg.addr_dim), int(scfg.payload_dim)
    zero = eqx.tree_at(lambda c: c.psi_res_gate, on,
                       jnp.zeros_like(on.psi_res_gate))
    src_b, src_l = [], []
    for i in range(3):
        q = _z(scfg, seed=10 + i)
        src_b.append(float(np.abs(np.asarray(on.read(blank, q))[a: a + m]
                                  - np.asarray(zero.read(blank, q))[a: a + m]).max()))
        src_l.append(float(np.abs(np.asarray(on.read(live, q))[a: a + m]
                                  - np.asarray(zero.read(live, q))[a: a + m]).max()))
    assert max(src_b) < 0.05
    assert max(src_l) > 3.0 * max(max(src_b), 1e-9)


def test_the_traj_mean_source_and_the_two_row_gate(scfg):
    """``both`` gives one gate row per source, and the rows are separable."""
    _off, on = _cells(scfg, psi_residual_source="both")
    assert on.psi_res_gate.shape == (2, int(scfg.payload_dim))
    # the init gain is SPLIT across the rows, so `both` at gain 1 is still a
    # unit-total pass-through
    assert np.allclose(np.asarray(on.psi_res_gate), 0.5)
    st = _written(on)
    q = _z(scfg, seed=14)
    m = int(scfg.payload_dim)
    g0 = np.zeros((2, m), np.float32)
    e_q, e_t = g0.copy(), g0.copy()
    e_q[0] = 1.0
    e_t[1] = 1.0
    rd = lambda g: np.asarray(eqx.tree_at(  # noqa: E731
        lambda c: c.psi_res_gate, on, jnp.asarray(g)).read(st, q))
    base = rd(g0)
    only_q = rd(e_q) - base
    only_t = rd(e_t) - base
    both = rd(e_q + e_t) - base
    assert np.allclose(both, only_q + only_t, atol=1e-5, rtol=0)
    assert not np.allclose(only_q, only_t, atol=1e-4)


def test_a_frozen_gate_carries_no_gradient(scfg):
    """``psi_residual_trainable=False`` is the designed-mechanism control: the
    gate must be ``stop_gradient``-ed, not merely 'expected to stay put'."""
    for trainable, expect_zero in ((True, False), (False, True)):
        _off, on = _cells(scfg, psi_residual_trainable=trainable)
        st = _written(on)
        q = _z(scfg, seed=15)

        def loss(c, st=st, q=q):
            return jnp.sum(c.read(st, q) ** 2)

        g = eqx.filter_grad(loss)(on)
        gg = float(np.abs(np.asarray(g.psi_res_gate)).max())
        assert (gg == 0.0) if expect_zero else (gg > 0.0)


def test_an_unknown_residual_source_raises():
    with pytest.raises(ValueError):
        psi_residual_sources("qstar")
    assert psi_residual_sources("both") == ("q_star", "traj_mean")
    assert psi_residual_sources("q_star") == ("q_star",)


# ---------------------------------------------------------------------------
# the instrument's own arithmetic
# ---------------------------------------------------------------------------
def test_acq_reproduces_the_probes_exact_chance_arithmetic():
    """⭐ §6's mechanism, as a unit test: a decode that under-shoots uniformly
    assigns EVERY item to the smallest-magnitude stored payload, which yields
    exactly one hit ⇒ acq = 1/n = chance, identically. This is why restoring the
    spread is necessary and not sufficient (PREREG P3)."""
    true = np.array([[-0.622], [-0.779], [-0.850]])
    q_star = np.array([[-0.411], [-0.297], [-0.371]])        # probe §6.2, h1b_r0.3
    assert _acq(q_star, true) == pytest.approx(1.0 / 3.0)
    # ... and at the scale-corrected gate it comes off chance (PREREG P4)
    assert _acq(2.5 * q_star, true) > 1.0 / 3.0


def test_the_spread_convention_reproduces_the_probes_before_column():
    """⚠ PREREG ADDENDUM 1: the before-column, recomputed CONSISTENTLY off §6's
    own raw numbers, is 0.160 / 0.058 / 0.191 — not the 0.04-0.15 the task file
    quotes, which is the reciprocal of §6's 7-25x compression band. The ratio is
    ddof-invariant; the mixed-ddof arithmetic is what produced the discrepancy."""
    q = np.array([-0.23329377, -0.18051672, -0.21142307])     # baseline q*
    dec = np.array([-0.06861678, -0.06392013, -0.06014941])   # baseline decode
    assert _std(dec) / _std(q) == pytest.approx(0.1600, abs=0.002)
    assert PROBE_BEFORE_RATIO["baseline"] == pytest.approx(
        _std(dec) / _std(q), abs=0.002)


def test_the_acceptance_bar_is_adjudicated_mechanically():
    """No hand-grading: ``aggregate`` decides §A20.3(a) from the numbers."""
    def rec(cell, seed, ratio, blank_qstar=1e-4, acq_blank=0.25, chance=0.25,
            blank_spread=0.01, psi_blank=0.01):
        pooled = {k: float("nan") for k in
                  ("ratio_psi_only_over_qstar", "ratio_qstar_over_true",
                   "frac_of_true_median", "spread_true", "spread_traj_mean",
                   "spread_psi_only", "linearity_maxabs",
                   "qstar_source_maxabs_vs_read_diag")}
        pooled.update({"ratio_decoded_over_qstar": ratio, "spread_decoded": 0.05,
                       "spread_q_star": 0.05, "spread_q_star_blank": blank_qstar,
                       "spread_decoded_blank": blank_spread,
                       "spread_psi_only_blank": psi_blank,
                       "acq_by_gate": {f"{g:g}": 0.25 for g in GATE_GRID},
                       "acq_blank_by_gate": {f"{g:g}": 0.25 for g in GATE_GRID}})
        fwd = {g: {"acq": 0.25, "acq_blank": acq_blank, "acq_minus_blank": 0.0,
                   "chance": chance, "bpc_live_minus_blank": 1e-5,
                   "depth_median": 0.05} for g in ("0", "1")}
        return {"cell": cell, "seed": seed, "tier": "ledger",
                "ledger": {"pooled": pooled, "lanes": []}, "forward": fwd}

    agg = aggregate([rec("baseline", s, 0.98) for s in (0, 1, 2)]
                    + [rec("h1b_r0.3", s, 0.2) for s in (0, 1, 2)]
                    + [rec("h1b_m1.0", s, 0.9, blank_qstar=0.9,
                           acq_blank=0.9, blank_spread=0.9) for s in (0, 1, 2)])
    assert agg["cells"]["ledger/baseline"]["ACCEPTANCE_MET"] is True
    assert agg["cells"]["ledger/h1b_r0.3"]["ACCEPTANCE_MET"] is False
    assert agg["cells"]["ledger/baseline"]["LEAK_CHECK_GREEN"] is True
    # ⛔ a blank arm whose q* carries as much as the live one, and whose decode
    # goes off chance, is a LEAK — whatever the acceptance ratio says
    assert agg["cells"]["ledger/h1b_m1.0"]["LEAK_CHECK_GREEN"] is False
    assert agg["cells"]["ledger/h1b_m1.0"]["leak_blank_acq_at_chance"] is False
    assert agg["cells"]["ledger/h1b_m1.0"]["leak_residual_blank_share"] > 0.05
    # ⚠ psi's OWN blank-store spread is shipped behaviour and is NOT charged to
    # the residual: a wide blank decode that psi_only already explains is green
    agg2 = aggregate([rec("baseline", s, 0.98, blank_spread=0.2, psi_blank=0.2)
                      for s in (0, 1, 2)])
    assert agg2["cells"]["ledger/baseline"]["LEAK_CHECK_GREEN"] is True
    assert ACCEPTANCE_RATIO == 0.5


def test_every_cell_inherits_a_declared_probe_cell():
    """The cells are the probe's own, so the shell/gain/data order are paired
    with §6 rather than re-drawn."""
    from chlu.experiments.exp_placement_probe import CELLS as PROBE

    for name, spec in CELLS.items():
        assert spec["base"] in PROBE, name


def test_cli_exposes_exp_psi_residual():
    """§8's declared NOT-RUN, discharged (C2W5 close-fix 5): the module had no
    CLI hook and ran only via ``python -m``. The hook forwards the module's own
    argv contract, so a bad cell is rejected by `main`'s validator, not by the
    parser."""
    import argparse

    from chlu.cli.experiment_cmd import setup_experiment_parsers

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_experiment_parsers(sub)
    args = parser.parse_args(["exp-psi-residual"])
    assert args.tier == "ledger" and hasattr(args, "func")
    args = parser.parse_args(["exp-psi-residual", "--tier", "trained",
                              "--cells", "run1", "--seeds", "0", "1",
                              "--steps", "5", "--out-dir", "/tmp/psires",
                              "--tag", "smoke"])
    assert args.tier == "trained" and args.cells == ["run1"]
    assert args.seeds == [0, 1] and args.steps == 5
    # the forwarded argv reaches `exp_psi_residual.main`'s own validation
    bad = parser.parse_args(["exp-psi-residual", "--cells", "not_a_cell"])
    with pytest.raises(SystemExit, match="unknown cells"):
        bad.func(bad)
