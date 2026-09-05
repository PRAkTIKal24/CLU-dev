"""Tests for the C3 store-geometry freeze and the ladder job plan
(`c3-rival-ladder-prereg`).

Five things are pinned here, each because it is a number the C3 ladder's whole
matched-state-byte control rests on, and each of which was a *surprise* when it
was measured rather than assumed:

1. ⚠⚠ **`atoms_per_item` and `capacity` are BYTE-INERT at the pilot geometry.**
   ``CluSystemConfig.n_atoms`` is a ``max`` whose w23 dimension-aware term
   ``512*sqrt(2)**addr_dim`` equals **8192** at the ruled ``addr_dim = 8`` and
   *ties* the pilot's ``K*A = 32*256``. Shrinking either knob moves **zero
   bytes** — the harness's "shrink the store (capacity / atoms_per_item / ...)"
   advice is unachievable through those two knobs, and a future agent must not
   rediscover that on the cluster.
2. **No sub-2 MiB CLU geometry exists at `addr_dim=8, n_layers=12` while the w23
   floor stands** — the conflict is structural, not a config oversight.
3. **The frozen geometry resolves exactly**, its total state bytes are the
   pre-registered number, and it fits the ruled ceiling with the two-sided swap
   still byte-honest.
4. ⭐ **The shrink moves the `ttt_matched` arm's inner-loop stability product**
   (``eta*n/d``) from outside to inside the non-expansive region — *and the
   worst-direction criterion is STILL above 2*, so the cure is partial and the
   caveat is asserted so it cannot be quietly dropped.
5. **The ladder job plan fits the 2xA100 / 4-day envelope at the frozen geometry
   and does NOT fit it at the pilot geometry** at the same step budget.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "scripts" / "c3_geometry_sweep.py"
PLAN = REPO / "scripts" / "c3_ladder_plan.py"

#: ⭐ THE FROZEN C3 CLU GEOMETRY (PREREG-C3-LADDER.md §2). One place, so a change
#: to it breaks a test rather than drifting through the codebase.
FROZEN = dict(addr_dim=8, payload_dim=4, capacity=32, atoms_per_item=64,
              n_layers=12, min_atoms_base=128)
FROZEN_N_ATOMS = 2048
FROZEN_TOTAL_STATE_BYTES = 1_394_688


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load(SWEEP, "_c3_geometry_sweep")


@pytest.fixture(scope="module")
def plan():
    return _load(PLAN, "_c3_ladder_plan")


def _store_cfg(**over):
    from chlu.experiments.exp_cluformer_pilot import make_config

    base = dict(FROZEN)
    mab = base.pop("min_atoms_base")
    base.update(over)
    store = dict(min_atoms_base=over.pop("min_atoms_base", mab))
    return make_config("pilot", 0, dict(base, store=store)).store_cfg()


# ==========================================================================
# 1. the trap: atoms_per_item / capacity are byte-inert at addr_dim = 8
# ==========================================================================
@pytest.mark.parametrize("atoms_per_item", [16, 32, 64, 96, 128, 256])
def test_atoms_per_item_is_byte_inert_at_the_shipped_w23_floor(atoms_per_item):
    """⚠⚠ Every one of these gives the SAME 8192 atoms — the floor dominates."""
    from chlu.core.clu_system import CluSystemConfig

    cfg = CluSystemConfig(addr_dim=8, payload_dim=4, capacity=32,
                          atoms_per_item=atoms_per_item)
    assert cfg.n_atoms == 8192, (atoms_per_item, cfg.n_atoms)


@pytest.mark.parametrize("capacity", [16, 32, 64, 128])
def test_capacity_cannot_shrink_the_store_either(capacity):
    from chlu.core.clu_system import CluSystemConfig

    cfg = CluSystemConfig(addr_dim=8, payload_dim=4, capacity=capacity,
                          atoms_per_item=16)
    assert cfg.n_atoms == 8192


def test_the_w23_geometric_term_is_what_pins_it():
    """The floor is ``round(min_atoms_base * min_atoms_c**addr_dim)``."""
    from chlu.core.clu_system import CluSystemConfig

    cfg = CluSystemConfig(addr_dim=8)
    assert round(cfg.min_atoms_base * cfg.min_atoms_c ** cfg.addr_dim) == 8192
    assert cfg.min_atoms_base == 512


# ==========================================================================
# 2. the structural conflict with the ruled ceiling
# ==========================================================================
def test_no_sub_2MiB_clu_geometry_exists_while_the_w23_floor_stands(sweep):
    """⛔ At ``addr_dim=8, n_layers=12`` the floor alone busts the ceiling 2.6x."""
    from chlu.eval.byte_ledger import MATCHED_STATE_BYTE_BUDGET

    floor_only = sweep.deploy_state_bytes(8192)          # dim 12, 12 layers
    assert floor_only == 5_523_456
    assert floor_only > 2.6 * MATCHED_STATE_BYTE_BUDGET
    # ...and the *smallest* the store can be with the floor intact is that same
    # number: capacity contributes only K*dim = 384 floats.
    assert sweep.store_state_floats(8192, 12, 1) * 4 * 12 > MATCHED_STATE_BYTE_BUDGET


# ==========================================================================
# 3. the frozen geometry
# ==========================================================================
def test_the_frozen_geometry_resolves_to_exactly_the_requested_atoms():
    cfg = _store_cfg()
    assert cfg.n_atoms == FROZEN_N_ATOMS
    assert cfg.addr_dim == 8 and cfg.dim == 12
    # both terms of the max AGREE, so the geometry is unambiguous
    assert cfg.atoms_per_item * cfg.capacity == FROZEN_N_ATOMS
    assert round(cfg.min_atoms_base * cfg.min_atoms_c ** cfg.addr_dim) == FROZEN_N_ATOMS


def test_the_helper_reproduces_the_built_cell_byte_for_byte(sweep):
    """⛔ The plan's arithmetic must equal the model's, or the ledger is fiction."""
    import jax

    from chlu.core.blocks import CluStoreCell
    from chlu.experiments.exp_cluformer_pilot import make_config

    pcfg = make_config("pilot", 0, dict(FROZEN, store={"min_atoms_base": 128}))
    scfg, mcfg = pcfg.store_cfg(), pcfg.memory_cfg()
    led = CluStoreCell(scfg, mcfg, key=jax.random.PRNGKey(0)).cell_ledger()
    assert led["state_floats"] == sweep.store_state_floats(
        int(scfg.n_atoms), int(scfg.dim), int(scfg.capacity))
    assert 4 * 12 * led["state_floats"] == FROZEN_TOTAL_STATE_BYTES
    assert sweep.deploy_state_bytes(FROZEN_N_ATOMS) == FROZEN_TOTAL_STATE_BYTES


def test_the_frozen_geometry_fits_the_ruled_ceiling_with_room(sweep):
    from chlu.eval.byte_ledger import MATCHED_STATE_BYTE_BUDGET

    occ = sweep.deploy_state_bytes(FROZEN_N_ATOMS) / MATCHED_STATE_BYTE_BUDGET
    assert 0.66 < occ < 0.67
    # ⭐ "no arm sitting exactly on the ceiling" — the frozen store does not.
    assert occ < 0.95


def test_the_admissible_ceiling_window_is_what_the_prereg_claims(sweep):
    """ceiling >= max(both swap members, TTT-Linear)  AND  ceiling < GDN-2."""
    from chlu.eval.byte_ledger import (
        MATCHED_STATE_BYTE_BUDGET as CEIL, RIVAL_SPECS,
    )

    lo = max(sweep.deploy_state_bytes(FROZEN_N_ATOMS),
             RIVAL_SPECS["ttt_linear"].state_bytes())
    hi = RIVAL_SPECS["gated_deltanet2"].state_bytes()
    assert lo == 1_597_440 and hi == 3_145_728
    assert lo <= CEIL < hi
    # every other pinned rival is SHRUNK, never grown
    for name, spec in RIVAL_SPECS.items():
        if name in ("ttt_linear",):
            assert spec.state_bytes() <= CEIL
        else:
            assert spec.state_bytes() >= CEIL


# ==========================================================================
# 4. the shrink moves the TTT arm's inner-loop stability — partially
# ==========================================================================
#: (params, state_floats, dim) of the CLU cell at each geometry, measured by
#: ``solve_arms`` at PILOT shapes.
PILOT_LEDGER = (168_986, 115_072, 12)
FROZEN_LEDGER = (82_970, 29_056, 12)


def test_the_shrink_pulls_the_ttt_mean_criterion_inside_the_stable_region():
    """⭐ ``eta*n/d`` is a pure function of the solved geometry (7.30)."""
    import jax
    import jax.nn

    from chlu.core.blocks import MatchedTTTCell, solve_matched_ttt

    kp, np_ = solve_matched_ttt(*PILOT_LEDGER)
    kf, nf = solve_matched_ttt(*FROZEN_LEDGER)
    assert (kp, np_) == (2197, 52) and (kf, nf) == (2235, 13)
    eta = float(jax.nn.softplus(
        MatchedTTTCell(12, kf, nf, key=jax.random.PRNGKey(0)).log_eta))
    assert eta * np_ / 12 > 2.0          # pilot: divergent
    assert eta * nf / 12 < 1.0           # frozen: comfortably inside
    assert (eta * np_ / 12) / (eta * nf / 12) == pytest.approx(np_ / nf, rel=1e-6)


def test_the_cure_is_PARTIAL_the_worst_direction_still_exceeds_2():
    """⚠ The caveat is asserted so it cannot be lost: a coherent chunk stream
    along ``theta_K``'s top right-singular vector still amplifies."""
    import jax
    import jax.nn
    import jax.numpy as jnp
    import numpy as np

    from chlu.core.blocks import MatchedTTTCell, solve_matched_ttt

    k, n = solve_matched_ttt(*FROZEN_LEDGER)
    cell = MatchedTTTCell(12, k, n, key=jax.random.PRNGKey(0))
    _u, _s, vt = np.linalg.svd(np.asarray(cell.theta_K), full_matrices=False)
    z = jnp.asarray(vt[0] / np.linalg.norm(vt[0]))
    crit = float(jax.nn.softplus(cell.log_eta)) * float(jnp.sum((cell.theta_K @ z) ** 2))
    assert crit > 2.0, crit          # ⛔ still expansive on the worst direction
    assert crit < 3.0, crit          # ...but far below the pilot's 6.18


# ==========================================================================
# 5. the job plan and the envelope
# ==========================================================================
def test_the_plan_fits_the_envelope_at_the_frozen_geometry_and_not_at_pilot(plan):
    frozen = plan.build_plan(n_atoms=FROZEN_N_ATOMS, steps=20_000, seeds=3,
                             eval_batches=40, slice_batches=10)
    pilot = plan.build_plan(n_atoms=8192, steps=20_000, seeds=3,
                            eval_batches=40, slice_batches=10)
    assert frozen["envelope"]["all_jobs_fit"] is True
    assert frozen["envelope"]["min_headroom_x"] > 3.0
    assert pilot["envelope"]["all_jobs_fit"] is False
    assert frozen["schedule"]["n_jobs"] == 15
    assert frozen["schedule"]["makespan_days"] < 4.0


def test_G_B_is_byte_compute_and_envelope_EQUIVALENT_to_the_descent(plan):
    """⭐ The whole reason G-B is recommended over G-A: the store's bytes AND its
    compute are both per store-bearing layer, so 8192 atoms in 3 of 12 layers buys
    exactly what 2048 atoms in 12 layers buys — without descending below the w23
    floor. If this equivalence ever breaks, the recommendation must be re-argued."""
    ga = plan.build_plan(n_atoms=2048, n_store_layers=12, steps=20_000, seeds=3,
                         eval_batches=40, slice_batches=10)
    gb = plan.build_plan(n_atoms=8192, n_store_layers=3, steps=20_000, seeds=3,
                         eval_batches=40, slice_batches=10)
    assert gb["geometry"]["clu_total_state_bytes"] == 1_380_864
    assert ga["geometry"]["clu_total_state_bytes"] == FROZEN_TOTAL_STATE_BYTES
    # within 1 % on bytes, and IDENTICAL on compute
    assert abs(gb["geometry"]["occupancy_of_2MiB"]
               - ga["geometry"]["occupancy_of_2MiB"]) < 0.01
    assert gb["throughput"]["clu_store_s_per_step"] == pytest.approx(
        ga["throughput"]["clu_store_s_per_step"], rel=1e-9)
    assert gb["envelope"]["worst_job_h"] == pytest.approx(
        ga["envelope"]["worst_job_h"], rel=1e-9)
    # ...and only ONE of them descends below the floor. That is the whole point.
    assert ga["geometry"]["descends_below_w23_floor"] is True
    assert gb["geometry"]["descends_below_w23_floor"] is False


def test_G_B_reproduces_the_pilots_DIVERGENT_ttt_cell():
    """⛔ G-B's price, pinned: it shrinks the number of cells, not the cell, so
    `solve_matched_ttt` sees the pilot ledger and the NaN geometry returns. A
    choice of G-B therefore FORCES `ttt_normalized_write=True`."""
    import jax
    import jax.nn

    from chlu.core.blocks import MatchedTTTCell, solve_matched_ttt

    k, n = solve_matched_ttt(*PILOT_LEDGER)          # unchanged per-layer cell
    assert (k, n) == (2197, 52)
    eta = float(jax.nn.softplus(
        MatchedTTTCell(12, k, n, key=jax.random.PRNGKey(0)).log_eta))
    assert eta * n / 12 > 2.0


def test_the_plan_reproduces_the_measured_pilot_mfu(plan):
    """⛔ The MFU is MEASURED (2xA100 legs), not assumed at 3 % like the scout's."""
    p = plan.build_plan(n_atoms=8192, steps=4000, seeds=3, eval_batches=40,
                        slice_batches=10)
    t = p["throughput"]
    assert t["measured_clu_mfu_at_pilot_geometry"] == pytest.approx(1.551e-4, rel=0.02)
    assert t["null_arm_mfu"] == pytest.approx(3.032e-3, rel=0.02)
    # ⚠ two orders of magnitude below the scout's most pessimistic 3 % assumption
    assert t["measured_clu_mfu_at_pilot_geometry"] < 0.03 / 100


def test_the_ladder_trains_five_arms_and_declares_the_rivals_NOT_RUN(plan):
    p = plan.build_plan(n_atoms=FROZEN_N_ATOMS, steps=20_000, seeds=3,
                        eval_batches=40, slice_batches=10)
    assert list(plan.LADDER_ARMS) == ["clu_store", "gru_matched", "ttt_matched",
                                      "none", "echo"]
    assert any("NOT-RUN" in s or "not BUILT" in s for s in p["not_run"])
    # the slice phase is priced but flagged as unmeasured, never silently folded in
    clu = next(j for j in p["jobs"] if j["arm"] == "clu_store")
    assert any(k.endswith("UNMEASURED") for k in clu["phases"])


# ==========================================================================
# 6. the sweep grid, and the banner both scripts must carry
# ==========================================================================
@pytest.mark.parametrize("n_atoms", [512, 1024, 2048, 3072, 4096, 8192])
def test_every_swept_geometry_resolves_to_the_requested_atom_count(sweep, n_atoms):
    from chlu.experiments.exp_cluformer_pilot import make_config

    pt = dict(axis="A", n_atoms=n_atoms, capacity=32, payload_dim=4, n_layers=2)
    cfg = make_config("toy", 0, sweep.overrides_for(pt, steps=1, seed=0)).store_cfg()
    assert int(cfg.n_atoms) == n_atoms


def test_the_deployed_write_arm_carries_the_landed_run_levers(sweep):
    """⛔ Read off the landed artifact's own flags, not off a report."""
    assert sweep.DEPLOYED_WRITE["atom_place_radius"] == 0.3
    assert sweep.DEPLOYED_STORE["write_margin"] == 0.6
    pt = dict(axis="A", n_atoms=2048, capacity=32, payload_dim=4, n_layers=2)
    ov = sweep.overrides_for(pt, steps=1, seed=0, deployed_write=True)
    assert ov["memory"]["atom_place_radius"] == 0.3
    assert ov["store"]["write_margin"] == 0.6
    off = sweep.overrides_for(pt, steps=1, seed=0)
    assert "atom_place_radius" not in off["memory"]


@pytest.mark.parametrize("path", [SWEEP, PLAN])
def test_neither_script_can_be_read_as_a_claim_venue(path):
    head = path.read_text(encoding="utf-8")[:2200]
    assert "NEVER a claim venue" in head or "trains nothing" in head
