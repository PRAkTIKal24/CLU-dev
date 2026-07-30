"""Tests for controller v0 — the designed guards (C2W1).

The point of this file is that **a guard is not a preference**. Each test below
asks the controller to do something outside the designed feasible set and asserts
it *cannot*: either the action is refused (and the refusal is logged and
attributed) or the call raises. If any of these can be talked into succeeding,
the w20 failure ("free learning erases design") reappears at the controller
level, which is exactly what the projection formalism exists to prevent.
"""

import numpy as np
import pytest

from chlu.core.clu_controller import (
    GUARDS,
    VERBS,
    CluControllerV0,
    ControllerBands,
    ControllerPolicy,
    GuardViolation,
    assert_d_safe_consistent,
    derived_d_safe,
)
from chlu.core.controller import Controller
from chlu.core.memory_potentials import AtomStorePotential
from chlu.core.monitors import MonitorContext, default_registry


def _controller(budget=4, capacity=4, registry=None, **policy_kw):
    store = AtomStorePotential(dim=3, capacity=capacity, s=0.3, addr_dim=2)
    alloc = Controller(store, d_safe=derived_d_safe(0.3, 0.15), budget=capacity,
                       evict_policy="depth", allow_relocation=False)
    return CluControllerV0(alloc, policy=ControllerPolicy(**policy_kw),
                           registry=registry, budget=budget)


def test_verb_set_is_the_designed_seven_plus_the_doctrine_two():
    assert set(VERBS) == {"admit", "place", "evict", "decay", "route", "retry",
                          "stop", "anneal", "expand"}


def test_d_safe_is_the_merge_plus_margin_form_not_4p4_s():
    """`controller-doctrine` R5: ``4.4 s`` exceeds the store's own spacing at working
    widths, so the store fails its own gate."""
    s, sigma = 0.35, 0.15
    assert derived_d_safe(s, sigma) == pytest.approx(2 * s + 2.576 * sigma)
    assert derived_d_safe(s, sigma) < 4.4 * s
    assert_d_safe_consistent(1.0, 1.2)
    with pytest.raises(GuardViolation):
        assert_d_safe_consistent(1.5, 1.2)


def test_policy_is_PROJECTED_onto_the_designed_bands():
    """Guards are constraints, not penalties: no Theta maps to an infeasible action."""
    c = _controller()
    proposed = ControllerPolicy(evict_persistence_W=99, retry_max_rounds=-3,
                                anneal_payload_mult=1e6, expand_growth=0.01,
                                decay_leak=17.0)
    p = c.project(proposed)
    b = ControllerBands()
    assert p.evict_persistence_W == b.evict_W[1]
    assert p.retry_max_rounds == b.retry_rounds[0]
    assert p.anneal_payload_mult == b.anneal_mult[1]
    assert p.expand_growth == b.expand_growth[0]
    assert p.decay_leak == b.leak[1]


def test_admit_refuses_an_unreachable_item():
    """The reach certificate is designed: a payload beyond ``a_U`` is refused at
    write time, not diagnosed after the fact."""
    c = _controller()
    r = c.admit(0, np.array([0.5, 0.0]), 9.0, reach_margin=-7.5)
    assert not r.applied and r.guard == "admit.reach"
    assert c.guard_fire_counts()["admit.reach"] == 1


def test_admit_refuses_a_merge_violation():
    c = _controller()
    assert c.admit(0, np.array([0.5, 0.0]), 0.3).applied
    r = c.admit(1, np.array([0.5, 0.0]), -0.3)  # same address
    assert not r.applied and r.guard == "admit.merge"


def test_admit_raises_a_capacity_alarm_rather_than_overwriting():
    """A full store of permanent items must alarm, never silently overwrite."""
    c = _controller(budget=2, capacity=2)
    assert c.admit(0, np.array([0.9, 0.0]), 0.3, permanent=True).applied
    assert c.admit(1, np.array([-0.9, 0.0]), -0.3, permanent=True).applied
    r = c.admit(2, np.array([0.0, 0.9]), 0.5)
    assert not r.applied and r.guard == "admit.budget"


def test_evict_is_forbidden_while_a_class_I_monitor_is_tripped():
    """doctrine §5 consequence 1: the store must not irreversibly delete on the
    basis of readings known to be invalid."""
    reg = default_registry(loud=False)
    c = _controller(registry=reg)
    assert c.admit(0, np.array([0.9, 0.0]), 0.3).applied
    reg.observe(MonitorContext(stage="t", blank={"score": 1.0, "chance": 0.1,
                                                 "se": 0.01}))
    r = c.evict(0, reason="budget", trips=99)
    assert not r.applied and r.guard == "evict.class_i"


def test_evict_requires_persistence():
    c = _controller(evict_persistence_W=3)
    assert c.admit(0, np.array([0.9, 0.0]), 0.3).applied
    assert not c.evict(0, reason="budget", trips=1).applied
    assert c.evict(0, reason="budget", trips=3).applied


def test_evict_refuses_LRU_outright():
    """LRU is query history => the store stops being order-independent and the
    exact deletion claim no longer covers it (N99)."""
    c = _controller()
    c.admit(0, np.array([0.9, 0.0]), 0.3)
    with pytest.raises(GuardViolation):
        c.evict(0, reason="lru", trips=9)


def test_route_may_not_be_wired_to_post_settle_energy():
    """N97: post-settle energy is not a routing/confidence signal."""
    c = _controller()
    assert c.route(np.array([0.1, 0.1]), signal="address").applied
    with pytest.raises(GuardViolation):
        c.route(np.array([0.1, 0.1]), signal="energy")


def test_anneal_must_return_to_the_stored_landscape():
    """N109's ``static`` control reaches basin 0.9993 and reads the WRONG value."""
    c = _controller()
    assert c.anneal([4.0, 2.0, 1.0]).applied
    with pytest.raises(GuardViolation):
        c.anneal([4.0, 2.0, 1.5])


def test_expand_may_not_shrink_the_space_while_items_are_live():
    c = _controller()
    assert c.expand(1.5).applied
    c.admit(0, np.array([0.9, 0.0]), 0.3)
    with pytest.raises(GuardViolation):
        c.expand(0.5)


def test_retry_cannot_exceed_the_declared_compute_budget():
    c = _controller(retry_max_rounds=1, retry_confidence_tau=0.9)
    assert c.retry(0.1, round_index=0).applied
    r = c.retry(0.1, round_index=1)
    assert not r.applied and r.guard == "retry.budget"


def test_place_refuses_a_non_minimum_and_a_newton_derivation():
    """A Newton re-derivation once wrote a SADDLE into the codebook and the deadband
    preserved it for 150 epochs (clu-controller-spec §C4.3/P12)."""
    c = _controller()
    c.admit(0, np.array([0.9, 0.0]), 0.3)
    r = c.place(0, np.array([0.8, 0.1]), lambda_min=-0.2)
    assert not r.applied and r.guard == "place.lambda_min"
    with pytest.raises(GuardViolation):
        c.place(0, np.array([0.8, 0.1]), lambda_min=1.0, derived_by="newton")
    assert c.place(0, np.array([0.8, 0.1]), lambda_min=1.0).applied


def test_decay_never_touches_a_permanent_item():
    c = _controller(decay_leak=0.5)
    c.allocator.leak = 0.5
    assert c.admit(0, np.array([0.9, 0.0]), 0.3, permanent=True).applied
    assert c.admit(1, np.array([-0.9, 0.0]), -0.3).applied
    before = np.asarray(c.allocator.store.amps).copy()
    c.decay(3)
    after = np.asarray(c.allocator.store.amps)
    perm_slot = [r.slot for r in c.allocator.records.values() if r.permanent][0]
    assert after[perm_slot] == before[perm_slot]
    leaky = [r.slot for r in c.allocator.records.values() if not r.permanent]
    assert all(after[s] < before[s] for s in leaky)


def test_guard_fire_counts_cover_every_declared_guard():
    """M14's input: the count dict must have a slot for every designed guard, or a
    dead guard is invisible."""
    c = _controller()
    counts = c.guard_fire_counts()
    assert set(counts) == set(GUARDS)


def test_stop_is_always_available():
    c = _controller()
    r = c.stop("blank control passed")
    assert r.applied and c.stopped == "blank control passed"
