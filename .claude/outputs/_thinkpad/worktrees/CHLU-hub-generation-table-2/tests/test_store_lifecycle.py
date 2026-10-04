"""⭐ C2W10 — the THREE-STATE lifecycle's **designed negatives**, kill-conditions first.

*A guard that cannot be shown to FAIL is not a guard* — the defect class C2W8
caught twice (the vacuous ``M``, the addressability-blind gate). So every
designed negative in this file comes in a **pair**:

* the negative itself (the bad behaviour must not happen), and
* its ``*_can_fail`` twin, which either disables the guard through a **declared
  config knob** or feeds the criterion the discriminating input with the opposite
  value, and asserts the bad behaviour **does** then appear.

Legs covered here: **L1** promotion (hysteresis), **L2** demotion (the
rich-get-richer negative), **L3** trash (three negatives incl. the censoring
guard), **L4** the protected fraction + its monitor, **L5/L5-b** I1
refresh-monotonicity incl. the cross-implementation validation against C2W6's
recorded rewrite events, and **L6** netting (three assertions).

⛔ Depth never enters the usefulness criterion (§A28.3(ii)).
⛔ Demotion is PROTECTED -> ACTIVE, never to trash (§A34.3).
"""

import json
import math
import os

import numpy as np
import pytest

from chlu.core.store_lifecycle import (
    ACTIVE,
    PROTECTED,
    TRASH,
    LifecycleParams,
    ProtectedSaturationMonitor,
    StoreLifecycle,
    cumulative_decay,
    net_depth,
    promotion_dwell,
    refresh_factor,
    replay_rewrite_events,
    should_demote,
    should_promote,
    should_trash,
)

# the registered operating parameters (PREREG §P2: d_dwell > window is DERIVED)
REG = dict(h_hi=2, h_lo=1, window=2, d_dwell=3, d_demote=2, k_streams=3, f_max=0.25)


def params(**over):
    kw = dict(REG)
    kw.update(over)
    return LifecycleParams(lifecycle=True, **kw)


def _burst(n_chunks: int, at: int, size: int):
    """A single burst of ``size`` hits in chunk ``at``, nothing anywhere else."""
    h = [0] * n_chunks
    h[at] = size
    return h


def _sustained(n_chunks: int, per_chunk: int):
    return [per_chunk] * n_chunks


# ==========================================================================
# L1 PROMOTION — sustained usage, with hysteresis
# ==========================================================================
def test_l1_sustained_usage_promotes():
    p = params()
    h = _sustained(6, REG["h_hi"])
    assert should_promote(h, 5, p)


def test_l1_single_burst_does_not_promote():
    """⭐ DESIGNED NEGATIVE: one burst reaching ``h_hi`` must NOT promote.

    A burst in chunk ``c`` holds the trailing-window test for exactly ``window``
    consecutive chunks, so the dwell requirement binds iff ``d_dwell > window``
    (PREREG §P2). At the registered values the burst reaches dwell 2 < 3.
    """
    p = params()
    h = _burst(8, at=2, size=10 * REG["h_hi"])
    assert promotion_dwell(h, 3, p) == REG["window"]
    assert not any(should_promote(h, c, p) for c in range(8))


def test_l1_single_burst_negative_can_fail():
    """The same burst DOES promote once the hysteresis is disabled."""
    p = params(d_dwell=REG["window"])  # d_dwell == window: the dwell stops binding
    h = _burst(8, at=2, size=10 * REG["h_hi"])
    assert any(should_promote(h, c, p) for c in range(8))


def test_l1_below_h_lo_never_promotes():
    p = params()
    h = [0] * 10
    assert not any(should_promote(h, c, p) for c in range(10))


def test_l1_below_h_lo_negative_can_fail():
    """With ``h_hi = 0`` the threshold stops binding and a never-read well promotes."""
    p = params(h_hi=0)
    h = [0] * 10
    assert any(should_promote(h, c, p) for c in range(10))


def test_l1_construction_refuses_a_non_binding_hysteresis():
    """``d_dwell <= window`` makes the burst negative arithmetically incapable of
    failing — the vacuous-guard defect class. It must raise, not ship."""
    with pytest.raises(ValueError, match="d_dwell"):
        params(d_dwell=REG["window"])._assert_hysteresis_binds()


# ==========================================================================
# L2 DEMOTION — the rich-get-richer negative
# ==========================================================================
def test_l2_demotion_predicate():
    """⚠ Demotion is scored on the chunk's OWN hits, not on the trailing window:
    a window of length ``W`` would carry the last pre-abandonment hits forward and
    delay demotion to ``W - 1 + d_demote``, breaking the registered "within
    ``d_demote`` chunks". Promotion sticky, demotion prompt."""
    p = params()
    h = _sustained(4, 3) + [0] * REG["d_demote"]
    assert should_demote(h, len(h) - 1, p)
    assert not should_demote(h, 3, p)
    assert not should_demote(h, 4, p)  # one chunk below h_lo is not yet d_demote


def test_l2_early_popular_then_abandoned_well_demotes():
    """⭐⭐ THE rich-get-richer negative (the leg the Head named).

    A well with high stream-1 usage and zero usage thereafter must be ACTIVE
    (not PROTECTED) within ``d_demote`` chunks — and must **NOT** be trashed by
    the demotion.
    """
    lc = StoreLifecycle(params(), budget=8)
    lc.note_admitted(7, chunk=0, stream=0)
    for c in range(4):  # stream 1: sustained usage -> PROTECTED
        lc.observe_chunk(c, {7: REG["h_hi"]}, stream=0)
    assert lc.state(7) == PROTECTED
    promoted_at = max(e["chunk"] for e in lc.events if e["verb"] == "promote")
    for c in range(4, 4 + REG["d_demote"]):  # abandoned
        lc.observe_chunk(c, {}, stream=1)
    assert lc.state(7) == ACTIVE
    demoted_at = max(e["chunk"] for e in lc.events if e["verb"] == "demote")
    assert demoted_at - promoted_at <= REG["d_demote"] + 1
    assert lc.state(7) != TRASH
    assert not [e for e in lc.events if e["verb"] == "trash" and e["item_id"] == 7]


def test_l2_demotion_negative_can_fail():
    """With the demote verb OFF the abandoned well keeps its protection forever."""
    lc = StoreLifecycle(params(demote=False), budget=8)
    lc.note_admitted(7, chunk=0, stream=0)
    for c in range(4):
        lc.observe_chunk(c, {7: REG["h_hi"]}, stream=0)
    for c in range(4, 12):
        lc.observe_chunk(c, {}, stream=1)
    assert lc.state(7) == PROTECTED


def test_l2_demotion_restores_the_designed_decay():
    """The demoted well's depth follows the designed law again (L6 tolerance)."""
    from chlu.core.controller import Controller
    from chlu.core.memory_potentials import AtomStorePotential

    store = AtomStorePotential(dim=3, capacity=4, addr_dim=2)
    ctl = Controller(store, d_safe=0.1, budget=4, leak=0.02, amp_floor=1e-9)
    ctl.offer(7, np.array([0.0, 0.0]), 0.1, permanent=True)
    lc = StoreLifecycle(params(leak=0.02), budget=4, controller=ctl)
    lc.note_admitted(7, chunk=0, stream=0)
    lc.set_state(7, PROTECTED)
    a0 = float(np.asarray(ctl.store.amps)[ctl.live_slots()[0]])
    for _ in range(3):
        ctl.tick()
    a1 = float(np.asarray(ctl.store.amps)[ctl.live_slots()[0]])
    assert a1 == pytest.approx(a0, abs=0.0)  # PROTECTED: no decay at all
    for c in range(REG["d_demote"]):
        lc.observe_chunk(c, {}, stream=1)
    assert lc.state(7) == ACTIVE
    for _ in range(5):
        ctl.tick()
    a2 = float(np.asarray(ctl.store.amps)[ctl.live_slots()[0]])
    # the designed law, to the shipped store's float32 floor (see the L6 pair)
    assert a2 == pytest.approx(a1 * math.exp(-0.02 * 5), rel=1e-6)


# ==========================================================================
# L3 TRASH — the §A20.6 Head addition, three designed negatives
# ==========================================================================
def test_l3_useful_in_stream_one_only_is_trashed_at_k():
    """(a) the intended positive."""
    p = params()
    hits = {0: 5, 1: 0, 2: 0, 3: 0}
    ok, why = should_trash(hits, 0, 3, p)
    assert ok, why


def test_l3_useful_in_every_stream_is_never_trashed():
    """(b) useful in every stream => NEVER trashed."""
    p = params()
    hits = {0: 1, 1: 1, 2: 1, 3: 1}
    ok, _ = should_trash(hits, 0, 3, p)
    assert not ok


def test_l3_useful_in_every_stream_negative_can_fail():
    """The discriminating input, flipped: zero the hits and it IS trashed."""
    p = params()
    assert not should_trash({0: 1, 1: 1, 2: 1, 3: 1}, 0, 3, p)[0]
    assert should_trash({0: 1, 1: 0, 2: 0, 3: 0}, 0, 3, p)[0]


def test_l3_censoring_guard_spares_a_young_well():
    """(c) a well admitted in the LAST stream is never trashed: never-useful-YET
    is not never-useful."""
    p = params()
    ok, why = should_trash({3: 0}, 3, 3, p)
    assert not ok
    assert "age" in why or "censor" in why


def test_l3_censoring_guard_can_fail():
    """With the guard OFF the young well IS trashed — the guard is load-bearing."""
    p = params(censoring_guard=False)
    assert should_trash({3: 0}, 3, 3, p)[0]


def test_l3_depth_never_enters_the_usefulness_criterion():
    """§A28.3(ii): depth != usefulness. The criterion's signature cannot take one."""
    import inspect

    sig = inspect.signature(should_trash)
    assert not [n for n in sig.parameters if "depth" in n or "amp" in n]


def test_l3_the_two_readings_of_the_criterion_differ_and_both_ship():
    """The registered wording ("never useful since first appearance over k stream
    boundaries") and designed negative (a) ("useful in stream 1 only => trashed at
    k") are only jointly satisfiable under the last-k-streams reading. Both are
    implemented; the default is declared; the difference is asserted, not hidden."""
    hits = {0: 5, 1: 0, 2: 0, 3: 0}
    assert should_trash(hits, 0, 3, params())[0]
    assert not should_trash(hits, 0, 3, params(trash_criterion="since_first_seen"))[0]
    assert should_trash({0: 0, 1: 0, 2: 0, 3: 0}, 0, 3,
                        params(trash_criterion="since_first_seen"))[0]


def test_l3_trash_routes_through_the_system_verb_and_is_never_a_demotion():
    routed = []
    lc = StoreLifecycle(params(), budget=8,
                        trash_route=lambda iid, center: routed.append(int(iid)))
    lc.note_admitted(3, chunk=0, stream=0, center=np.zeros(2))
    lc.observe_chunk(0, {3: 4}, stream=0)
    for s in (1, 2, 3):
        lc.end_stream(s - 1)
        lc.observe_chunk(s, {}, stream=s)
    lc.end_stream(3)
    assert routed == [3]
    assert lc.state(3) == TRASH
    assert not [e for e in lc.events if e["verb"] == "demote" and e["item_id"] == 3]


def test_l3_a_protected_well_is_never_trashed():
    """⛔ trash is the never-useful route; a PROTECTED well is out of its scope."""
    routed = []
    lc = StoreLifecycle(params(demote=False), budget=8,
                        trash_route=lambda iid, c: routed.append(int(iid)))
    lc.note_admitted(3, chunk=0, stream=0, center=np.zeros(2))
    for c in range(4):
        lc.observe_chunk(c, {3: REG["h_hi"]}, stream=0)
    assert lc.state(3) == PROTECTED
    for s in (1, 2, 3, 4):
        lc.end_stream(s - 1)
        lc.observe_chunk(4 + s, {}, stream=s)
    lc.end_stream(4)
    assert routed == []
    assert lc.state(3) == PROTECTED


# ==========================================================================
# L4 PROTECTED FRACTION — the anti-collapse leg
# ==========================================================================
def test_l4_forcing_every_item_high_trips_the_monitor_and_refuses():
    """⭐ DESIGNED NEGATIVE: forcing every item's usage high must trip
    ``protected_saturation`` and REFUSE, never silently protect everything."""
    lc = StoreLifecycle(params(), budget=8)
    for iid in range(8):
        lc.note_admitted(iid, chunk=0, stream=0)
    for c in range(6):
        lc.observe_chunk(c, {iid: 10 for iid in range(8)}, stream=0)
    assert lc.n_protected == int(REG["f_max"] * 8) == 2
    assert lc.protected_fraction() <= REG["f_max"] + 1e-12
    refused = [e for e in lc.events if e["verb"] == "promote_refused"]
    assert len(refused) >= 1
    assert lc.monitor_state()["tripped"] is True
    assert lc.monitor_state()["name"] == "protected_saturation"


def test_l4_negative_can_fail():
    """At ``f_max = 1.0`` the bound stops binding: everything protects, no trip."""
    lc = StoreLifecycle(params(f_max=1.0), budget=8)
    for iid in range(8):
        lc.note_admitted(iid, chunk=0, stream=0)
    for c in range(6):
        lc.observe_chunk(c, {iid: 10 for iid in range(8)}, stream=0)
    assert lc.n_protected == 8
    assert not [e for e in lc.events if e["verb"] == "promote_refused"]
    assert lc.monitor_state()["tripped"] is False


def test_l4_monitor_is_a_registry_monitor_and_fails_loudly_not_as_a_loss():
    from chlu.core.monitors import MonitorContext

    lc = StoreLifecycle(params(), budget=4)
    mon = ProtectedSaturationMonitor(lc)
    r = mon.observe(MonitorContext(stage="test"))
    assert r.name == "protected_saturation"
    assert not r.tripped
    for iid in range(4):
        lc.note_admitted(iid, chunk=0, stream=0)
    for c in range(6):
        lc.observe_chunk(c, {iid: 10 for iid in range(4)}, stream=0)
    r = mon.observe(MonitorContext(stage="test"))
    assert r.tripped
    assert r.severity_class in ("I", "II", "III", "IV", "V")


# ==========================================================================
# L5 / L5-b I1 REFRESH-MONOTONICITY
# ==========================================================================
def test_l5_refresh_factor_is_one_on_a_non_violating_write():
    """A violation-free write multiplies the amplitude by exactly 1.0 => the
    guarded write is BIT-IDENTICAL to the unguarded one (blocks.py's I1-b)."""
    assert refresh_factor(1.0, 2.0, params(refresh_monotonic=True)) == 1.0
    assert refresh_factor(1.0, 1.0, params(refresh_monotonic=True)) == 1.0


def test_l5_refresh_factor_restores_a_destructive_rewrite():
    p = params(refresh_monotonic=True)
    f = refresh_factor(4.0, 1.0, p)
    assert f == pytest.approx(2.0, rel=1e-12)
    assert (1.0 * f**2) == pytest.approx(4.0, rel=1e-12)


def test_l5_refresh_factor_is_capped():
    p = params(refresh_monotonic=True, refresh_max_gain=4.0)
    assert refresh_factor(1e6, 1e-6, p) == pytest.approx(4.0, rel=1e-12)


def test_l5_guard_off_a_destructive_rewrite_reduces_depth():
    """⭐ DESIGNED NEGATIVE: with the guard OFF the planted destructive rewrite
    must reduce the depth. A guard that cannot be shown to fail is not a guard."""
    ev = [{"depth_before": 4.0, "depth_after": 1.0, "rewrite": 1.0}]
    off = replay_rewrite_events(ev, params(refresh_monotonic=False))
    assert off["n_violations_pre"] == 1
    assert off["n_violations_post"] == 1
    assert off["depth_guarded"][0] == pytest.approx(1.0)
    on = replay_rewrite_events(ev, params(refresh_monotonic=True))
    assert on["n_violations_pre"] == 1
    assert on["n_violations_post"] == 0
    assert on["depth_guarded"][0] >= 4.0 * (1.0 - 1e-6)


C2W6_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "CHLU", ".claude", "outputs", "c2w6-anti-erosion",
)


def _c2w6_records(cell):
    """C2W6's raw artifact (gitignored `.claude/` tree) or ``None``."""
    for base in (C2W6_DIR, os.environ.get("CHLU_C2W6_DIR", "")):
        if not base:
            continue
        path = os.path.join(base, f"erosion_{cell}_records.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["records"]
    return None


@pytest.mark.parametrize("cell,expected_pre", [
    ("p1_off", [16, 2, 3]),
    ("p1_on_i1_on", [6, 0, 0]),
])
def test_l5b_cross_implementation_against_c2w6_recorded_events(cell, expected_pre):
    """⭐ L5-b: the store-level guard is validated against the BLOCK-level one on
    C2W6's **own recorded rewrite events** (PREREG §P1, tolerances E1/E2/E3).

    ⚠ The events live in the gitignored `.claude/` artifact tree, so this test
    SKIPS where that tree is absent; the measured numbers are in the spoke report.
    """
    recs = _c2w6_records(cell)
    if recs is None:
        pytest.skip("C2W6 artifact tree absent (gitignored); see the spoke report")
    for rec, exp in zip(recs, expected_pre, strict=True):
        ev = [e for t in rec.get("telemetry", []) for e in (t.get("rewrite_events") or [])
              if float(e.get("rewrite", 0.0)) > 0.5]
        rep = replay_rewrite_events(ev, params(refresh_monotonic=True))
        assert rep["n_events"] == int(rec["n_rewrite_events"])
        # E1: the violation flag reproduces the block-level one exactly
        assert rep["n_flag_mismatch"] == 0
        assert rep["n_violations_pre"] == exp
        assert rep["rate_pre"] == pytest.approx(rec["rewrite_violation_rate"],
                                                abs=1e-12)
        # E3: no post-guard violation survives
        assert rep["n_violations_post"] == 0
        # ...and the guard is load-bearing ON THESE REAL EVENTS: with it off, the
        # recorded destructive rewrites reduce the depth, exactly `exp` of them.
        off = replay_rewrite_events(ev, params(refresh_monotonic=False))
        assert off["n_violations_post"] == exp


def test_l5b_refresh_factor_matches_the_recorded_block_level_factor():
    """E2: my factor equals ``blocks.py``'s recorded ``refresh_factor`` to 1e-6."""
    recs = _c2w6_records("p1_on_i1_on")
    if recs is None:
        pytest.skip("C2W6 artifact tree absent (gitignored); see the spoke report")
    dev = 0.0
    n = 0
    for rec in recs:
        for t in rec.get("telemetry", []):
            for e in t.get("rewrite_events") or []:
                if float(e.get("rewrite", 0.0)) <= 0.5:
                    continue
                mine = refresh_factor(e["depth_before"], e["depth_after"],
                                      params(refresh_monotonic=True))
                theirs = float(e["refresh_factor"])
                dev = max(dev, abs(mine - theirs) / max(abs(theirs), 1e-30))
                n += 1
    assert n > 0
    assert dev <= 1e-6, f"max relative factor deviation {dev:.3e} over {n} events"


# ==========================================================================
# L6 NETTING — the build requirement (Add.9 §A27.1)
# ==========================================================================
def test_l6_netted_is_bitwise_raw_at_leak_zero():
    for raw in (0.0, 1e-30, 0.3333333333333333, 1.0, 1e30):
        assert net_depth(raw, 1.0).hex() == float(raw).hex()


def test_l6_netted_exceeds_raw_under_decay():
    raw = 0.75
    for cum in (0.999, 0.9, 0.5, 1e-3):
        assert net_depth(raw, cum) > raw


def _no_write_cum_factor(leak, dt):
    from chlu.core.clu_controller import CluControllerV0
    from chlu.core.controller import Controller
    from chlu.core.memory_potentials import AtomStorePotential

    store = AtomStorePotential(dim=3, capacity=4, addr_dim=2)
    alloc = Controller(store, d_safe=0.1, budget=4, leak=leak, amp_floor=1e-12)
    ctl = CluControllerV0(alloc)
    ctl.admit(11, np.array([0.0, 0.0]), 0.1, leak=leak)
    for _ in range(dt):
        ctl.decay(1)
    return cumulative_decay(ctl)[11]


def test_l6_a_well_with_no_writes_nets_to_the_analytic_law_x64():
    """⭐ The registered L6 assertion, **1e-9**, which needs float64 amplitudes.

    Function-scoped x64 (module-scoped is the §7.23/N211 hazard). The shipped
    store holds its amplitudes in float32, whose eps is 1.2e-7 — see the
    companion test for the achievable floor there. The netting law itself is
    exact; the tolerance is a dtype fact and is reported as one.
    """
    from jax import config as jax_config

    was = bool(jax_config.read("jax_enable_x64"))
    jax_config.update("jax_enable_x64", True)
    try:
        assert _no_write_cum_factor(0.02, 7) == pytest.approx(
            math.exp(-0.02 * 7), abs=1e-9)
    finally:
        jax_config.update("jax_enable_x64", was)


def test_l6_a_well_with_no_writes_nets_to_the_analytic_law_float32_floor():
    """The same law at the SHIPPED dtype: exact to the float32 floor, not to 1e-9.

    ⚠ **§7.23's ordering hazard, and the full suite caught it on this very test.**
    The dtype must be PINNED, not inherited: run alone the store is float32 and
    the 1e-9 bound fails as asserted, but after any module that enables
    ``jax_enable_x64`` the same store is float64 and the bound *holds*, turning a
    correct negative assertion red. Green in isolation, red in the suite — which
    is the whole defect class. Function-scoped, because a module-scoped x64
    fixture is itself the N211 hazard.
    """
    from jax import config as jax_config

    was = bool(jax_config.read("jax_enable_x64"))
    jax_config.update("jax_enable_x64", False)
    try:
        got = _no_write_cum_factor(0.02, 7)
        assert got == pytest.approx(math.exp(-0.02 * 7), abs=1e-6)
        assert got != pytest.approx(math.exp(-0.02 * 7), abs=1e-9)
    finally:
        jax_config.update("jax_enable_x64", was)


def test_l6_every_emitted_curve_carries_both_forms():
    lc = StoreLifecycle(params(), budget=4)
    lc.note_admitted(1, chunk=0, stream=0)
    lc.note_depth(1, chunk=0, depth_raw=1.0, cum_factor=1.0)
    lc.note_depth(1, chunk=1, depth_raw=0.5, cum_factor=math.exp(-0.02))
    curves = lc.depth_curves()
    assert set(curves[1]) >= {"chunk", "depth_raw", "depth_netted"}
    assert curves[1]["depth_netted"][0] == 1.0
    assert curves[1]["depth_netted"][1] > curves[1]["depth_raw"][1]
