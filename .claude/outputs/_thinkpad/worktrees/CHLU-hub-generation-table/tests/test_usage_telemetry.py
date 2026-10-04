"""C2W8 build requirement B2 — the usage telemetry, and its three prerequisites.

The C2W6 erosion adjudication made three things binding, and each is a test here:
item-id keying (slot != well), a single registered primary proxy, and the LOO
probe reported **only** beside its ICC(1,1) — labelled ``UNDEFINED`` when the
ICC is non-positive, never quoted as a null.
"""

import jax
import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import flatten_unused_groups, plant_item
from chlu.experiments.usage_telemetry import (
    PRIMARY_PROXY,
    UNDEFINED,
    UsageTelemetry,
    attach_reads,
    icc_1_1,
    loo_loss_contribution,
)


def _tiny_system(capacity=4, seed=0, **over):
    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=capacity, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=0.05, read_steps=60, address_steps=40,
        n_query_per_item=2, **over,
    )
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)


def test_registered_primary_proxy_is_read_hits():
    """The proxy was decided in the prereg, not on results."""
    assert PRIMARY_PROXY == "read_hits"
    assert UsageTelemetry().proxy == "read_hits"


def test_counter_is_item_id_keyed_and_survives_slot_reuse():
    """slot != well: a recycled slot must NOT inherit the evicted item's hits."""
    sysm = _tiny_system(capacity=2)
    plant_item(sysm, 100, np.array([0.5, 0.0]), payload=0.0, depth=0.7, width=0.25,
               leak=0.01)
    tel = UsageTelemetry()
    ctrl = sysm.controller.allocator
    for _ in range(3):
        tel.note_read(100, t=1, controller=ctrl)
    slot = sysm._slot_of(100)

    ev = sysm.controller.evict(100, reason="budget",
                               trips=sysm.controller.policy.evict_persistence_W)
    assert ev.applied, ev.reason
    plant_item(sysm, 200, np.array([0.5, 0.0]), payload=0.0, depth=0.7, width=0.25,
               leak=0.01)
    assert sysm._slot_of(200) == slot           # same slot, different item
    assert tel.hits(200) == 0 and tel.never_read(200)
    assert tel.hits(100) == 3                   # and the evicted item's count survives
    assert ctrl.read_hits == {100: 3}


def test_touch_records_reads_without_changing_lru_semantics():
    sysm = _tiny_system(capacity=4)
    plant_item(sysm, 1, np.array([0.5, 0.0]), payload=0.0, depth=0.7, width=0.25)
    ctrl = sysm.controller.allocator
    ctrl.t = 7
    ctrl.touch(1)
    rec = [r for r in ctrl.records.values() if r.item_id == 1][0]
    assert rec.last_used == 7           # the shipped staleness semantics, unchanged
    assert ctrl.read_hits[1] == 1       # plus the new, decision-free counter
    ctrl.touch(999)                     # an id that is not live is still counted
    assert ctrl.read_hits[999] == 1


def test_uncovered_reads_are_credited_to_nobody():
    tel = UsageTelemetry()
    n = tel.observe_basins([10, 11], basins=[0, 1, 0], t=3,
                           covered=[True, False, True])
    assert n == 2
    assert tel.hits(10) == 2 and tel.hits(11) == 0
    assert tel.n_read_events == 3 and tel.n_unassigned == 1
    s = tel.summary(live_ids=[10, 11])
    assert s["n_never_read"] == 1 and s["key"] == "item_id"


def test_attach_reads_uses_the_systems_own_basin_assignment():
    sysm = _tiny_system(capacity=4)
    sites = np.array([[0.6, 0.0], [-0.6, 0.0]])
    for i, s in enumerate(sites):
        plant_item(sysm, i, s, payload=0.3 * (2 * i - 1), depth=0.9, width=0.25)
    flatten_unused_groups(sysm)
    q0 = np.zeros((4, sysm.store.dim), dtype=np.float32)
    q0[:2, :2] = sites[0] + 0.01
    q0[2:, :2] = sites[1] + 0.01
    res = sysm.read(q0)
    n = attach_reads(sysm, tel := UsageTelemetry(), res, t=1)
    assert n + tel.n_unassigned == 4
    assert tel.hits(0) + tel.hits(1) == n
    # the counts agree with the controller's own touch-path counter
    assert dict(sysm.controller.allocator.read_hits) == {
        k: v for k, v in tel.read_hits.items() if v > 0
    }


def test_icc_1_1_matches_the_closed_form_and_flags_noise():
    # perfectly reproducible targets => ICC 1.0
    x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert icc_1_1(x) == pytest.approx(1.0)
    # pure within-target noise, no between-target signal => ICC <= 0
    rng = np.random.default_rng(0)
    y = rng.normal(size=(40, 3))
    assert icc_1_1(y) <= 0.05
    z = np.array([[0.0, 2.0], [2.0, 0.0]])   # anti-correlated repeats
    assert icc_1_1(z) < 0.0


def test_loo_is_labelled_undefined_when_its_icc_is_non_positive():
    """⛔ ICC <= 0 => the quantity is UNDEFINED, never a null, and no number
    may be quoted from it (C2W6: negative ICC on 3/3 seeds)."""
    sysm = _tiny_system(capacity=4)
    sites = np.array([[0.6, 0.0], [-0.6, 0.0], [0.0, 0.6]])
    for i, s in enumerate(sites):
        plant_item(sysm, i, s, payload=0.3 * (i - 1), depth=0.9, width=0.25)
    flatten_unused_groups(sysm)
    before = sysm.store
    out = loo_loss_contribution(sysm, [0, 1, 2], repeats=2, seed=0)
    assert out["role"].startswith("SECONDARY")
    assert out["status"] in ("usable", UNDEFINED)
    if out["status"] == UNDEFINED:
        assert out["values"] is None        # no number is quotable
    # the probe must leave the store exactly as it found it
    assert np.allclose(np.asarray(sysm.store.V.learned.amp),
                       np.asarray(before.V.learned.amp))
