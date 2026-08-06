"""C2W8 stage 1 — the census instrument and its **designed negatives**.

*"A census instrument that cannot see a planted population cannot license a
kill"* (`PREREG-C2W8.md` §5 K1). The two assertions this wave's stage-2 gate
rests on are here:

* a hand-built store with **4 known never-read attractors** must read
  ``P >= 4 / n_live``;
* a hand-built store with **3 known near-duplicate pairs** must read
  ``M >= 3 / n_pairs``.

Plus the rules that keep the instrument honest: the two well populations stay
separate (mechanic 1), the prune leg is usage and not depth (mechanic 2), the
decay netting is exact (B1), and the K1 arithmetic is mechanical.
"""

import jax
import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import (
    UNLOCK_THRESHOLD,
    census,
    designed_decay_factors,
    flatten_unused_groups,
    measure_theta_att,
    mergeable_pairs,
    own_foreign_site_depth,
    plant_item,
    unlock_verdict,
    well_states,
)
from chlu.experiments.usage_telemetry import UsageTelemetry


def _tiny_system(capacity=8, d_safe=0.05, seed=0, **over):
    """A small CluSystem whose atoms are cheap and whose gate admits neighbours."""
    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=capacity, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=float(d_safe), read_steps=60, address_steps=40,
        n_query_per_item=2, **over,
    )
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)


def _ring(n, r=0.6):
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


# --------------------------------------------------------------------------
# DESIGNED NEGATIVE 1 (K1): four planted never-read attractors must be seen
# --------------------------------------------------------------------------
def test_census_sees_four_planted_unread_attractors():
    sysm = _tiny_system(capacity=8)
    sites = _ring(8)
    for i in range(8):
        plant_item(sysm, i, sites[i], payload=0.1 * (i - 4), depth=0.8,
                   width=0.25, leak=0.01)
    flatten_unused_groups(sysm)

    tel = UsageTelemetry()
    for i in range(8):
        tel.note_admitted(i, t=i)
    # four wells are read, four are NEVER read -> the planted prunable population
    for i in (0, 1, 2, 3):
        tel.note_read(i, t=10, controller=sysm.controller.allocator)

    out = census(sysm, tel, well_budget=4, n_admitted=8, measure_capture=False)
    assert out["n_live"] == 8
    assert out["P"] >= 4.0 / out["n_live"], out
    # every planted unread well is an attractor, none is counted as eroded
    assert set(out["population_live_attractor_never_read"]) >= {4, 5, 6, 7}
    assert not set(out["population_eroded_not_attractor"]) & {4, 5, 6, 7}
    # and the counters really are item-id keyed and survive the census
    assert sysm.controller.allocator.read_hits == {0: 1, 1: 1, 2: 1, 3: 1}


# --------------------------------------------------------------------------
# DESIGNED NEGATIVE 2 (K1): three planted near-duplicate pairs must be seen
# --------------------------------------------------------------------------
def test_census_sees_three_planted_near_duplicate_pairs():
    sysm = _tiny_system(capacity=8, d_safe=0.001)
    base = _ring(3, r=0.7)
    ids = 0
    for k in range(3):  # three near-duplicate PAIRS: same payload, close centers
        for j in range(2):
            plant_item(sysm, ids, base[k] + np.array([0.004 * j, 0.0]),
                       payload=0.3 * k, depth=0.7, width=0.25, leak=0.01)
            ids += 1
    # two far-apart, distinct-payload wells that must NOT be called mergeable
    plant_item(sysm, 6, np.array([-0.9, 0.9]), payload=-0.9, depth=0.7,
               width=0.25, leak=0.01)
    plant_item(sysm, 7, np.array([0.9, -0.9]), payload=0.9, depth=0.7,
               width=0.25, leak=0.01)
    flatten_unused_groups(sysm)

    out = census(sysm, UsageTelemetry(), well_budget=4, n_admitted=8,
                 measure_capture=False)
    n_pairs = out["n_pairs"]
    assert n_pairs == 28
    assert out["M"] >= 3.0 / n_pairs, out
    found = {tuple(sorted((p["item_i"], p["item_j"]))) for p in out["mergeable_pairs"]}
    assert {(0, 1), (2, 3), (4, 5)} <= found
    assert (6, 7) not in found  # far apart AND payload-distinct


# --------------------------------------------------------------------------
# mechanic 1 — an eroded well is NOT prunable, and is counted separately
# --------------------------------------------------------------------------
def test_eroded_well_is_not_prunable_and_is_counted_separately():
    sysm = _tiny_system(capacity=4)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25, leak=0.01)
    plant_item(sysm, 1, sites[1], payload=0.4, depth=0.9, width=0.25, leak=0.01)
    plant_item(sysm, 2, sites[2], payload=-0.4, depth=0.0, width=0.25, leak=0.01)
    flatten_unused_groups(sysm)

    tel = UsageTelemetry()
    states, theta = well_states(sysm, tel, measure_capture=False)
    by_id = {s.item_id: s for s in states}
    assert by_id[2].depth_raw == pytest.approx(0.0, abs=1e-6)
    # theta_att is 0.0 when capture is not measured, so the zero-depth well is
    # excluded by "depth > theta_att" alone: it is eroded, hence not prunable.
    assert not by_id[2].is_attractor
    assert by_id[2].eroded and not by_id[2].prunable
    assert by_id[0].is_attractor and by_id[0].prunable  # unread live attractor

    out = census(sysm, tel, well_budget=2, n_admitted=3, measure_capture=False)
    assert 2 in out["population_eroded_not_attractor"]
    assert 2 not in out["population_live_attractor_never_read"]


# --------------------------------------------------------------------------
# mechanic 2 — the prune leg is USAGE, not depth
# --------------------------------------------------------------------------
def test_prunable_is_usage_not_depth():
    """A deep-but-unread well is prunable; a shallow-but-read one is not.

    ⛔ This is the census-level form of K3 (`PREREG-C2W8.md` §5): if depth could
    substitute for usage here, the prune criterion would be a depth policy
    wearing a usage costume. The stage-2 verb's own K3 assertion rides on top of
    this and does not replace it.
    """
    sysm = _tiny_system(capacity=4)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25, leak=0.01)  # deep
    plant_item(sysm, 1, sites[1], payload=0.4, depth=0.05, width=0.25, leak=0.01)  # shallow
    flatten_unused_groups(sysm)

    tel = UsageTelemetry()
    tel.note_admitted(0, 0)
    tel.note_admitted(1, 0)
    for _ in range(5):  # the SHALLOW well is the frequently-read one
        tel.note_read(1, t=1, controller=sysm.controller.allocator)

    states, _ = well_states(sysm, tel, measure_capture=False)
    by_id = {s.item_id: s for s in states}
    assert by_id[0].depth_raw > by_id[1].depth_raw
    assert by_id[0].prunable is True     # deep, never read
    assert by_id[1].prunable is False    # shallow, read 5x


# --------------------------------------------------------------------------
# protection — the leak == 0 cohort is excluded from P
# --------------------------------------------------------------------------
def test_permanent_cohort_is_excluded_from_P():
    sysm = _tiny_system(capacity=4)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25,
               permanent=True, leak=0.0)
    plant_item(sysm, 1, sites[1], payload=0.4, depth=0.9, width=0.25, leak=0.01)
    flatten_unused_groups(sysm)
    out = census(sysm, UsageTelemetry(), well_budget=2, n_admitted=2,
                 measure_capture=False)
    assert 0 in out["population_protected"]
    assert out["n_prunable"] == 1  # only the non-protected unread attractor
    assert out["P"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# B1 — the decay netting is exact, and is reported beside the raw curve
# --------------------------------------------------------------------------
def test_decay_netting_is_exact_against_the_designed_law():
    sysm = _tiny_system(capacity=4, leak=0.05, stage_lifetimes=True)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25, leak=0.05)
    flatten_unused_groups(sysm)
    d0, _ = sysm.well_fits()

    n_ticks = 4
    for _ in range(n_ticks):
        sysm.controller.decay(1)
        sysm._sync_decay()

    factors = designed_decay_factors(sysm.controller)
    predicted = float(np.exp(-0.05 * n_ticks))
    assert factors[0] == pytest.approx(predicted, rel=1e-4)

    states, _ = well_states(sysm, UsageTelemetry(), measure_capture=False)
    s0 = states[0]
    assert s0.depth_raw < float(d0[0])                       # the raw curve fell
    assert s0.depth_netted == pytest.approx(s0.depth_raw / factors[0], rel=1e-9)
    assert s0.depth_netted == pytest.approx(float(d0[0]), rel=0.05)  # netting restores it


def test_own_and_foreign_site_depth_split():
    sysm = _tiny_system(capacity=4, d_safe=0.001)
    plant_item(sysm, 0, np.array([0.0, 0.0]), payload=0.0, depth=0.6, width=0.25)
    plant_item(sysm, 1, np.array([0.02, 0.0]), payload=0.0, depth=0.6, width=0.25)
    flatten_unused_groups(sysm)
    z = np.zeros((sysm.store.dim,))
    own, foreign = own_foreign_site_depth(sysm.store, sysm._slot_of(0), z)
    assert own == pytest.approx(0.6, rel=1e-5)     # exactly what was planted
    assert foreign > 0.0                            # the near-duplicate is felt
    assert foreign < own


# --------------------------------------------------------------------------
# theta_att is MEASURED, and K1's arithmetic is mechanical
# --------------------------------------------------------------------------
def test_theta_att_is_measured_not_guessed():
    # two wells whose capture radius is below sigma_q => the floor sits at the
    # deepest of THOSE, not at a constant.
    out = measure_theta_att(depths=[0.9, 0.2, 0.05], capture=[0.4, 0.02, 0.01],
                            sigma_q=0.15)
    assert out["theta_att"] == pytest.approx(0.2)
    assert out["n_capturing"] == 1 and out["n_non_capturing"] == 2
    allcap = measure_theta_att([0.9, 0.8], [0.4, 0.5], 0.15)
    assert allcap["theta_att"] == 0.0


def test_unlock_verdict_is_mechanical():
    assert unlock_verdict([0.2, 0.1, 0.0], [0.0, 0.0, 0.0])["stage2_unlock"] is True
    assert unlock_verdict([0.0, 0.0], [0.09, 0.09])["stage2_unlock"] is True
    v = unlock_verdict([0.0, 0.01, 0.0], [0.0, 0.02, 0.0])
    assert v["stage2_unlock"] is False and v["kill"] is True
    # a single seed above threshold blocks KILL but need not UNLOCK on the mean
    v2 = unlock_verdict([0.0, 0.0, 0.12], [0.0, 0.0, 0.0])
    assert v2["kill"] is False
    assert UNLOCK_THRESHOLD == 0.05


def test_mergeable_pairs_use_certificate_radius_and_payload_tol():
    sysm = _tiny_system(capacity=4, d_safe=0.001)
    plant_item(sysm, 0, np.array([0.0, 0.0]), payload=0.0, depth=0.6, width=0.25)
    plant_item(sysm, 1, np.array([0.01, 0.0]), payload=0.0, depth=0.6, width=0.25)
    flatten_unused_groups(sysm)
    states, _ = well_states(sysm, UsageTelemetry(), measure_capture=False)
    pairs, meta = mergeable_pairs(sysm, states)
    assert len(pairs) == 1
    # payload distance beyond the read tolerance disqualifies the same geometry
    pairs2, _ = mergeable_pairs(sysm, states, payload_thresh=-1.0)
    assert pairs2 == []
    # so does a certificate radius below the separation
    pairs3, _ = mergeable_pairs(sysm, states, r_cert=1e-6)
    assert pairs3 == []
    assert meta["r_cert"] > 0.0
