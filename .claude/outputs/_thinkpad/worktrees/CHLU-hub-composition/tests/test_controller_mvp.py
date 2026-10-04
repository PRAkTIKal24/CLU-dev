"""Tests for the MVC-0 controller and the N75-rematch harness (w23).

These pin the controller's three decision rules and the geometry it lives on —
the things whose silent breakage would flip the rematch verdict:

  * **admission** refuses a too-close offer and admits a far one, and
    ``refuse-and-relocate`` actually relocates to an admissible site;
  * **eviction** removes the least-recently-used non-permanent item when the
    budget is full, NEVER a permanent one, and raises a capacity alarm (refuse)
    when the full store is all-permanent — deletion is the one irreversible verb;
  * **scheduled decay** shallows leaky wells by ``exp(-leak)`` per tick, leaves
    permanent (``leak==0``) wells untouched, and self-evicts a well below floor;
  * the **packing bound** the admitted fraction is checked against reproduces
    N74's disk value (6.1 at R=2, d_safe=1.54), and ``radius_for_capacity``
    inverts it;
  * the store's ``evict`` / ``with_amps`` primitives are functional (return a new
    frozen PyTree) and change ``V`` the way the controller assumes.
"""

import numpy as np
import pytest

from chlu.core.controller import (
    Controller,
    ItemRecord,
    packing_bound_disk,
    radius_for_capacity,
)
from chlu.core.memory_potentials import AtomStorePotential


def _store(capacity=8, s=0.35):
    return AtomStorePotential(dim=3, capacity=capacity, alpha=0.02, s=s, kappa=1.0)


# ---------------------------------------------------------------------------
# geometry: the packing bound the admitted fraction is checked against (N74)
# ---------------------------------------------------------------------------
def test_packing_bound_matches_n74():
    """⭐ Disk radius 2, d_safe 1.54 => 6.1, which N74 measured as 6.0 ± 0.9."""
    assert packing_bound_disk(2.0, 1.54) == pytest.approx(6.12, abs=0.05)


def test_radius_for_capacity_inverts_the_bound():
    for k in (4, 16, 64):
        R = radius_for_capacity(k, 1.54)
        assert packing_bound_disk(R, 1.54) == pytest.approx(k, rel=1e-6)


# ---------------------------------------------------------------------------
# AtomStorePotential.evict / with_amps (the controller's store primitives)
# ---------------------------------------------------------------------------
def test_store_evict_clears_slot():
    V = _store().with_item([0.0, 0.0], 0.5, amp=1.0)
    assert V.n_stored == 1
    slot = int(np.argmax(np.asarray(V.active)))
    V2 = V.evict(slot)
    assert V2.n_stored == 0
    assert float(np.asarray(V2.amps)[slot]) == 0.0
    assert V.n_stored == 1  # functional: original untouched


def test_store_with_amps_replaces_depths():
    V = _store().with_item([0.0, 0.0], 0.5, amp=1.0)
    amps = np.asarray(V.amps).copy()
    amps *= 0.5
    V2 = V.with_amps(amps)
    np.testing.assert_allclose(np.asarray(V2.amps), amps, atol=1e-6)
    with pytest.raises(ValueError):
        V.with_amps(np.zeros(V.capacity + 1))


# ---------------------------------------------------------------------------
# admission (C5-A1/A2)
# ---------------------------------------------------------------------------
def test_admission_refuses_close_admits_far():
    import jax

    c = Controller(_store(), d_safe=1.0)
    r0 = c.offer(0, [0.0, 0.0], 0.5, key=jax.random.PRNGKey(0))
    assert r0["decision"] == "admit" and c.n_live == 1
    # a second offer inside d_safe with NO proposer => refuse
    r1 = c.offer(1, [0.3, 0.0], -0.5, key=jax.random.PRNGKey(1))
    assert r1["decision"] == "refuse_spacing" and c.n_live == 1
    # a far offer => admit
    r2 = c.offer(2, [2.0, 0.0], 0.2, key=jax.random.PRNGKey(2))
    assert r2["decision"] == "admit" and c.n_live == 2


def test_refuse_and_relocate_finds_admissible_site():
    import jax

    from chlu.core.admission import disk_proposer

    c = Controller(_store(), d_safe=1.0, n_candidates=400)
    c.offer(0, [0.0, 0.0], 0.5, key=jax.random.PRNGKey(0))

    def proposer(k, n):
        return np.asarray(disk_proposer(3.0, 3)(k, n))[:, :2]

    r = c.offer(1, [0.2, 0.0], -0.5, key=jax.random.PRNGKey(3), proposer=proposer)
    assert r["decision"] == "relocate"
    # the relocated site is genuinely >= d_safe from the stored one
    assert r["d_min_written"] >= 1.0 - 1e-6
    assert c.n_live == 2


# ---------------------------------------------------------------------------
# eviction / budget (C5 budget, §3.C trash)
# ---------------------------------------------------------------------------
def test_budget_evicts_least_recently_used():
    import jax

    c = Controller(_store(capacity=4), d_safe=1.0, budget=2, evict_policy="staleness")
    c.offer(0, [0.0, 0.0], 0.1, key=jax.random.PRNGKey(0))
    c.offer(1, [3.0, 0.0], 0.2, key=jax.random.PRNGKey(1))
    assert c.n_live == 2
    c.touch(1)  # item 1 is now the more-recently-used
    # a third admissible offer must evict item 0 (the stalest)
    r = c.offer(2, [0.0, 3.0], 0.3, key=jax.random.PRNGKey(2))
    assert r["decision"] == "admit"
    assert r["evicted_item"] == 0
    live = {rec.item_id for rec in c.records.values()}
    assert live == {1, 2}
    assert c.stats["evicted"] == 1


def test_permanent_item_is_never_evicted_and_full_permanent_alarms():
    import jax

    c = Controller(_store(capacity=4), d_safe=1.0, budget=2)
    c.offer(0, [0.0, 0.0], 0.1, key=jax.random.PRNGKey(0), permanent=True)
    c.offer(1, [3.0, 0.0], 0.2, key=jax.random.PRNGKey(1), permanent=True)
    # store full of permanents => a new admissible offer is a CAPACITY ALARM
    r = c.offer(2, [0.0, 3.0], 0.3, key=jax.random.PRNGKey(2))
    assert r["decision"] == "refuse_full"
    assert c.stats["refused_full"] == 1
    live = {rec.item_id for rec in c.records.values()}
    assert live == {0, 1}


# ---------------------------------------------------------------------------
# scheduled decay (leaky vs permanent wells; per-item retention, w22)
# ---------------------------------------------------------------------------
def test_tick_decays_leaky_leaves_permanent():
    import jax

    c = Controller(_store(capacity=4), d_safe=1.0, amp=1.0, leak=0.5, amp_floor=0.05)
    c.offer(0, [0.0, 0.0], 0.1, key=jax.random.PRNGKey(0), permanent=True)  # leak forced 0
    c.offer(1, [3.0, 0.0], 0.2, key=jax.random.PRNGKey(1))  # leaky
    amps0 = np.asarray(c.store.amps).copy()
    perm_slot = next(r.slot for r in c.records.values() if r.item_id == 0)
    leak_slot = next(r.slot for r in c.records.values() if r.item_id == 1)
    c.tick()
    amps1 = np.asarray(c.store.amps)
    assert amps1[perm_slot] == pytest.approx(amps0[perm_slot])  # permanent untouched
    assert amps1[leak_slot] == pytest.approx(amps0[leak_slot] * np.exp(-0.5), rel=1e-4)


def test_leaky_well_self_evicts_below_floor():
    import jax

    c = Controller(_store(capacity=4), d_safe=1.0, amp=1.0, leak=1.0, amp_floor=0.1)
    c.offer(0, [0.0, 0.0], 0.1, key=jax.random.PRNGKey(0), permanent=True)
    c.offer(1, [3.0, 0.0], 0.2, key=jax.random.PRNGKey(1))
    # amp 1.0 * exp(-1*t) < 0.1 => t >= 3 ticks
    for _ in range(3):
        c.tick()
    live = {rec.item_id for rec in c.records.values()}
    assert live == {0}  # the leaky item decayed out; the permanent one remains
    assert c.stats["decayed_out"] == 1


def test_item_record_is_a_plain_record():
    r = ItemRecord(item_id=1, slot=0, center=np.zeros(2), payload=0.0,
                   base_amp=1.0, leak=0.0, permanent=True, born=0, last_used=0)
    assert r.permanent and r.leak == 0.0
