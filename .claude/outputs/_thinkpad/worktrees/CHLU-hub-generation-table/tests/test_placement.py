"""Tests for canonical placement — PGCP — and the exact store-level deletion verb (w26).

These are the executable form of the theorist's Theorems 1–4
(``order-independent-placement`` §2/§3). The pass criterion for the exactness family is
**bit-identity** (``tobytes()`` equality), not a tolerance: the whole point of the rule is
that ``Store(S)`` is a *set function*, so two histories reaching the same live set must
produce byte-equal ``centers/payloads/amps/active``.

  * **T1/T2 (order-independence)** — every write order of a set gives the same store
    (exhaustive at n=4, random orders at n ∈ {8,16,40,64});
  * **T3 (interleavings)** — any write/delete interleaving reaching live set ``S`` equals a
    fresh build of ``S``;
  * **T4 (decay commutes)** — deleting a decaying item mid-schedule leaves every survivor
    bit-identical to the history in which it was never written;
  * **T5** — write-then-delete == never-written;
  * **the lattice invariant** — achieved min spacing is exactly ``d_safe`` (Theorem 3), so
    admission needs no per-write spacing test;
  * **packing** — ``N_cells(R(K)·1.05) >= K`` (the engineer-facing sizing rule);
  * **cascade cost** — deleting from a full lattice really does move survivors (the
    priced-in churn), and never moves a *higher*-priority item;
  * **the scrub (mia D1)** — ``evict`` now clears ``centers``/``payloads`` too, and that
    moves **no physics number** (``V`` is bit-identical);
  * **the scope guards** — LRU eviction, a missing lattice radius, ``allow_relocation=
    False`` and ``delete`` under refuse-and-relocate all hard-error.
"""

import itertools

import numpy as np
import pytest

from chlu.core.controller import Controller, radius_for_capacity
from chlu.core.memory_potentials import AtomStorePotential
from chlu.core.placement import (
    CanonicalPlacer,
    canonical_layout,
    hash_point,
    hex_cells,
    n_cells_for,
    prio,
    radius_for_cells,
    splitmix64,
)

D_SAFE = 4.4 * 0.35  # 1.54 — the shipped admission radius


def _store(capacity):
    return AtomStorePotential(dim=3, capacity=capacity, alpha=0.02, s=0.35, kappa=1.0)


def _ctrl(capacity, radius, leak=0.0, budget=None, waitlist=True):
    return Controller(
        _store(capacity), d_safe=D_SAFE, budget=budget, amp=1.0, leak=leak,
        evict_policy="depth", placement="canonical", lattice_radius=radius,
        waitlist=waitlist,
    )


def _items(n, seed=0, radius=3.0):
    """n (id, anchor, payload) records with distinct ids — the record SET."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    anchors = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    pays = np.linspace(-1.0, 1.0, n)
    return [(i, anchors[i], float(pays[i])) for i in range(n)]


def _arrays(ctrl):
    s = ctrl.store
    return tuple(np.asarray(a) for a in (s.centers, s.payloads, s.amps, s.active))


def _same(a, b):
    return all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


def _build(items, order, radius, capacity=None, leak=0.0, deletes=(), ticks=0,
           waitlist=True):
    """Offer ``items`` in ``order``, optionally tick/delete, return the controller."""
    cap = len(items) if capacity is None else capacity
    c = _ctrl(cap, radius, leak=leak, waitlist=waitlist)
    for i in order:
        item_id, anchor, pay = items[i]
        c.offer(item_id=item_id, q_new=anchor, payload=pay)
    for _ in range(ticks):
        c.tick()
    for d in deletes:
        c.delete(d)
    return c


# ---------------------------------------------------------------------------
# the deterministic per-key functions
# ---------------------------------------------------------------------------
def test_splitmix_and_prio_are_deterministic_and_history_free():
    assert splitmix64(0) == splitmix64(0)
    assert splitmix64(1) != splitmix64(0)
    assert prio(7) == prio(7) and prio(7) != prio(8)
    p = hash_point(42, 2.0)
    np.testing.assert_array_equal(p, hash_point(42, 2.0))
    assert np.hypot(*p) <= 2.0


def test_lattice_spacing_is_exactly_d_safe():
    """⭐ Theorem 3: the spacing certificate is free — a lattice invariant."""
    cells = hex_cells(3.0, D_SAFE)
    d = np.sqrt(((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1))
    d[np.diag_indices(len(cells))] = np.inf
    assert d.min() == pytest.approx(D_SAFE, abs=1e-12)
    assert np.hypot(cells[:, 0], cells[:, 1]).max() <= 3.0 + 1e-9


# ---------------------------------------------------------------------------
# T1/T2 — order-independence (bit-identity, zero tolerance)
# ---------------------------------------------------------------------------
def test_T1_all_write_orders_of_four_items_agree_bit_identically():
    """⭐ n=4, all 24 permutations => one store."""
    R = radius_for_cells(4, D_SAFE) * 1.35
    items = _items(4, seed=1, radius=R * 0.8)
    ref = _arrays(_build(items, (0, 1, 2, 3), R))
    n_ok = 0
    for order in itertools.permutations(range(4)):
        assert _same(_arrays(_build(items, order, R)), ref)
        n_ok += 1
    assert n_ok == 24


@pytest.mark.parametrize("n", [8, 16, 40])
def test_T2_random_write_orders_agree_and_spacing_holds(n):
    R = radius_for_cells(n, D_SAFE)
    assert n_cells_for(R, D_SAFE) >= n
    items = _items(n, seed=n, radius=R * 0.9)
    rng = np.random.default_rng(0)
    ref_ctrl = _build(items, list(range(n)), R)
    ref = _arrays(ref_ctrl)
    for _ in range(3):
        order = list(rng.permutation(n))
        assert _same(_arrays(_build(items, order, R)), ref)
    # the lattice invariant on the LIVE sites (assert, never re-check per write)
    c = np.stack([r.center for r in ref_ctrl.records.values()])
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    d[np.diag_indices(len(c))] = np.inf
    assert d.min() == pytest.approx(D_SAFE, abs=1e-9)
    assert ref_ctrl.n_live == n


def test_T2_at_the_rematch_point_admits_the_whole_lattice():
    """K=64: admission is deterministic — 61 cells at R(K), 64/64 at R(K)·1.05."""
    K = 64
    R = radius_for_capacity(K, D_SAFE)
    items = _items(K, seed=64, radius=R * 0.95)
    c = _build(items, list(range(K)), R, capacity=K)
    assert n_cells_for(R, D_SAFE) == 61
    assert c.n_live == 61  # == n_cells: the lattice IS the capacity
    c105 = _build(items, list(range(K)), R * 1.05, capacity=K)
    assert c105.n_live == 64


# ---------------------------------------------------------------------------
# T3/T5 — deletion is set-minus, bit-exactly (Theorem 2)
# ---------------------------------------------------------------------------
def test_T5_write_then_delete_equals_never_written():
    R = radius_for_cells(8, D_SAFE)
    items = _items(8, seed=5, radius=R * 0.8)
    with_v = _build(items, range(8), R, deletes=(3,))
    without = _build([it for it in items if it[0] != 3], range(7), R, capacity=8)
    assert _same(_arrays(with_v), _arrays(without))


def test_T3_write_delete_interleavings_agree_with_a_fresh_build():
    """⭐ 20 random interleavings reaching the same live set => one store."""
    R = radius_for_cells(12, D_SAFE)
    items = _items(12, seed=3, radius=R * 0.85)
    doomed = {2, 5, 9}
    survivors = [it for it in items if it[0] not in doomed]
    ref = _arrays(_build(survivors, range(len(survivors)), R, capacity=12))
    rng = np.random.default_rng(7)
    for _ in range(20):
        c = _ctrl(12, R)
        order = list(rng.permutation(12))
        pending = set(doomed)
        for i in order:
            item_id, anchor, pay = items[i]
            c.offer(item_id=item_id, q_new=anchor, payload=pay)
            # delete some doomed items early (interleaved), the rest at the end
            for d in list(pending):
                if d in {r.item_id for r in c.records.values()} and rng.random() < 0.4:
                    c.delete(d)
                    pending.discard(d)
        for d in pending:
            c.delete(d)
        assert _same(_arrays(c), ref)
        assert c.n_live == len(survivors)


def test_T4_mid_decay_delete_leaves_survivors_bit_identical():
    """⭐ Theorem 4: deletion and scheduled decay commute."""
    R = radius_for_cells(8, D_SAFE)
    items = _items(8, seed=4, radius=R * 0.8)
    victim = 6
    for t_del in (1, 3):
        c = _build(items, range(8), R, leak=0.35, ticks=t_del)
        c.delete(victim)
        for _ in range(5 - t_del):
            c.tick()
        never = _build([it for it in items if it[0] != victim], range(7), R,
                       capacity=8, leak=0.35, ticks=5)
        assert _same(_arrays(c), _arrays(never))
        # and the amplitude law itself is untouched by the re-pack
        amps = np.asarray(c.store.amps)[: c.n_live]
        np.testing.assert_allclose(amps, np.exp(-0.35 * 5), rtol=1e-5)


def test_delete_moves_survivors_but_never_a_higher_priority_item():
    """The cascade is bounded: only lower-priority keys can move (suffix stability)."""
    R = radius_for_capacity(16, D_SAFE)
    n = n_cells_for(R, D_SAFE)
    items = _items(n, seed=11, radius=R * 0.95)  # a FULL lattice
    c = _build(items, range(n), R)
    assert c.n_live == n
    victim = items[n // 2][0]
    before = {r.item_id: r.center.copy() for r in c.records.values()}
    row = c.delete(victim)
    after = {r.item_id: r.center.copy() for r in c.records.values()}
    moved = [k for k in after if not np.allclose(before[k], after[k])]
    assert row["moves"] == len(moved)
    for k in moved:
        assert prio(k) < prio(victim)  # strictly lower priority only


def test_cascade_cost_smoke_at_full_load():
    """Delete-time churn is the real price — pin its order of magnitude."""
    R = radius_for_capacity(64, D_SAFE)
    n = n_cells_for(R, D_SAFE)  # 61
    items = _items(n, seed=21, radius=R * 0.95)
    c = _build(items, range(n), R)
    moves = []
    for item_id in [it[0] for it in items[:12]]:
        moves.append(c.delete(item_id)["moves"])
        # re-insert to keep the lattice full
        anchor, pay = items[item_id][1], items[item_id][2]
        c.offer(item_id=item_id, q_new=anchor, payload=pay)
    assert max(moves) <= 15
    assert 0.0 < float(np.mean(moves)) < 6.0


# ---------------------------------------------------------------------------
# ⭐ P2 — the waitlist: exactness AT OVERFLOW (w27)
# ---------------------------------------------------------------------------
def _overflow_case(n_over=2, seed=31):
    """A lattice deliberately smaller than the offered set."""
    R = radius_for_capacity(8, D_SAFE)          # 7 cells, the mia geometry
    n_cells = n_cells_for(R, D_SAFE)
    items = _items(n_cells + n_over, seed=seed, radius=R * 0.8)
    return R, n_cells, items


def test_P2_waitlist_makes_deletion_exact_at_overflow():
    """⭐ The acceptance mechanism: a key refused while the victim was resident RETURNS.

    Without the waitlist the post-delete store is NOT the store that never held the
    victim (w26 rung P1); with it, byte-identity is restored at overflow.
    """
    R, n_cells, items = _overflow_case()
    victim = max((it[0] for it in items), key=prio)      # highest priority => seated
    survivors = [it for it in items if it[0] != victim]
    for waitlist, expect in ((True, True), (False, False)):
        with_v = _build(items, range(len(items)), R, capacity=len(items),
                        deletes=(victim,), waitlist=waitlist)
        never = _build(survivors, range(len(survivors)), R, capacity=len(items),
                       waitlist=waitlist)
        assert _same(_arrays(with_v), _arrays(never)) is expect
        if waitlist:
            assert with_v.n_live == n_cells        # the freed cell was re-taken
        else:
            assert with_v.n_live == n_cells - 1    # ... and stayed empty (the defect)


def test_P2_waitlisted_key_is_reseated_by_priority_not_by_arrival():
    R, n_cells, items = _overflow_case(n_over=2)
    c = _build(items, range(len(items)), R, capacity=len(items))
    waiting = sorted(c.waiting)
    assert len(waiting) == 2 and c.n_live == n_cells
    # the waiting keys are the LOWEST-priority offers, whatever the arrival order
    live = [r.item_id for r in c.records.values()]
    assert max(prio(k) for k in waiting) < min(prio(k) for k in live)
    victim = max(live, key=prio)
    c.delete(victim)
    reseated = [k for k in waiting if k not in c.waiting]
    assert reseated == [max(waiting, key=prio)]          # priority order, not FIFO
    assert c.n_live == n_cells


def test_P2_delete_of_a_waitlisted_item_is_a_noop_on_the_store():
    """The counterfactual for an unseated offer: the store never held it."""
    R, n_cells, items = _overflow_case(n_over=1)
    c = _build(items, range(len(items)), R, capacity=len(items))
    waiting_id = next(iter(c.waiting))
    before = _arrays(c)
    row = c.delete(waiting_id)
    assert row["was_waiting"] and row["moves"] == 0
    assert _same(_arrays(c), before) and c.n_waiting == 0


def test_P2_exactness_survives_decay_of_a_waiting_item():
    """⭐ Theorem 4 with the waitlist: a re-seated item carries the depth the history
    that never refused it would have produced — bit-identically, not approximately."""
    R, n_cells, items = _overflow_case(n_over=1)
    victim = max((it[0] for it in items), key=prio)
    survivors = [it for it in items if it[0] != victim]
    c = _build(items, range(len(items)), R, capacity=len(items), leak=0.35, ticks=3)
    c.delete(victim)
    for _ in range(2):
        c.tick()
    never = _build(survivors, range(len(survivors)), R, capacity=len(items),
                   leak=0.35, ticks=5)
    assert _same(_arrays(c), _arrays(never))
    amps = np.asarray(c.store.amps)[: c.n_live]
    np.testing.assert_allclose(amps, np.float32(np.exp(-0.35 * 5)), rtol=1e-6)


def test_P2_write_delete_interleavings_agree_under_overflow():
    """T3 at overflow: any interleaving reaching the same OFFERED set gives one store."""
    R, n_cells, items = _overflow_case(n_over=3, seed=77)
    doomed = {items[0][0], items[3][0]}
    survivors = [it for it in items if it[0] not in doomed]
    ref = _arrays(_build(survivors, range(len(survivors)), R, capacity=len(items)))
    rng = np.random.default_rng(5)
    for _ in range(10):
        c = _ctrl(len(items), R)
        pending = set(doomed)
        for i in rng.permutation(len(items)):
            item_id, anchor, pay = items[i]
            c.offer(item_id=item_id, q_new=anchor, payload=pay)
            for d in list(pending):
                if (d in {r.item_id for r in c.records.values()} or d in c.waiting) \
                        and rng.random() < 0.4:
                    c.delete(d)
                    pending.discard(d)
        for d in pending:
            c.delete(d)
        assert _same(_arrays(c), ref)
        assert c.n_live == n_cells


def test_P2_below_capacity_nothing_waits_and_nothing_changes():
    """The flag is inert below capacity — every w26 number is untouched there."""
    R = radius_for_cells(8, D_SAFE)
    items = _items(8, seed=5, radius=R * 0.8)
    on = _build(items, range(8), R, deletes=(3,), waitlist=True)
    off = _build(items, range(8), R, deletes=(3,), waitlist=False)
    assert on.n_waiting == 0 and _same(_arrays(on), _arrays(off))


# ---------------------------------------------------------------------------
# packing regression (the sizing rule the engineer must honour)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [16, 32, 64, 128])
def test_packing_sizing_rule_gives_enough_cells(K):
    R = radius_for_capacity(K, D_SAFE)
    assert n_cells_for(R * 1.05, D_SAFE) >= K
    assert n_cells_for(R, D_SAFE) <= n_cells_for(R * 1.05, D_SAFE)


def test_canonical_layout_matches_the_incremental_placer():
    R = 3.0
    keys = [3, 1, 4, 1_000, 7]
    anchors = [hash_point(k, R) for k in keys]
    pl = CanonicalPlacer(R, D_SAFE)
    for k, a in zip(keys, anchors, strict=True):
        pl.insert(k, a)
    ref = canonical_layout(R, D_SAFE, list(zip(keys, anchors, strict=True)))
    assert [k for k, _ in pl.layout()] == [k for k, _ in ref]
    for (_, a), (_, b) in zip(pl.layout(), ref, strict=True):
        assert a.tobytes() == b.tobytes()
    assert pl.min_spacing() == pytest.approx(D_SAFE, abs=1e-12)


# ---------------------------------------------------------------------------
# the array scrub (mia-decay-measurement D1)
# ---------------------------------------------------------------------------
def test_evict_scrubs_the_row():
    """⭐ ``evict`` clears centers/payloads too — "the slot is freed" now means erased."""
    V = _store(4).with_item([0.7, -0.3], 0.5, amp=1.0)
    slot = int(np.argmax(np.asarray(V.active)))
    V2 = V.evict(slot)
    assert float(np.asarray(V2.active)[slot]) == 0.0
    assert float(np.asarray(V2.amps)[slot]) == 0.0
    np.testing.assert_array_equal(np.asarray(V2.centers)[slot], np.zeros(2))
    assert float(np.asarray(V2.payloads)[slot]) == 0.0
    # functional: the original PyTree is untouched
    np.testing.assert_allclose(np.asarray(V.centers)[slot], [0.7, -0.3], atol=1e-6)


def test_evict_scrub_does_not_move_V():
    """⭐ No physics number moves: verify it, don't assert it (task item 4)."""
    import equinox as eqx
    import jax.numpy as jnp

    V = _store(4).with_item([0.7, -0.3], 0.5, amp=1.0).with_item([-1.9, 0.4], -0.5)
    masked = eqx.tree_at(lambda t: t.active, V, V.active.at[0].set(0.0))
    masked = eqx.tree_at(lambda t: t.amps, masked, masked.amps.at[0].set(0.0))
    scrubbed = V.evict(0)  # masked AND scrubbed
    rng = np.random.default_rng(0)
    qs = rng.normal(size=(64, 3)) * 1.5
    dv = [float(masked(jnp.asarray(q))) - float(scrubbed(jnp.asarray(q))) for q in qs]
    assert max(abs(x) for x in dv) == 0.0


# ---------------------------------------------------------------------------
# ⭐ option (d) — the gated-stiffness payload channel (w27, mia-D3)
# ---------------------------------------------------------------------------
def _gated(g0=0.05, capacity=4):
    return AtomStorePotential(dim=3, capacity=capacity, alpha=0.02, s=0.35, kappa=1.0,
                              payload_gate=True, payload_g0=g0)


def test_payload_gate_is_off_by_default_and_inert_when_off():
    """⛔ B1.4: the flag defaults OFF and the shipped V is bit-identical."""
    import jax.numpy as jnp

    def fill(V):
        return V.with_item([0.7, -0.3], 0.5, amp=1.0).with_item([-1.9, 0.4], -0.5)

    V = fill(_store(4))                                        # shipped default
    off = fill(AtomStorePotential(dim=3, capacity=4, alpha=0.02, s=0.35, kappa=1.0,
                                  payload_gate=False, payload_g0=0.5))
    on = fill(_gated(g0=0.05))
    assert V.payload_gate is False
    rng = np.random.default_rng(0)
    n_diff = 0
    for q in rng.normal(size=(32, 3)) * 1.5:
        q = jnp.asarray(q)
        assert float(V(q)) == float(off(q))                    # bit-identical when off
        n_diff += float(V(q)) != float(on(q))
    assert n_diff == 32                                        # ... and live when on


@pytest.mark.parametrize("A", [1.0, 0.2, 0.06, 0.001])
def test_payload_gate_returns_a_i_exactly_at_every_amplitude(A):
    """⭐ The whole point of NOT flooring the normaliser: abar(c_i) = a_i for all A.

    (The theorist's first implementation floored it, destroying the value at small A and
    refuting their own proposal — this test pins the corrected form.)
    """
    import equinox as eqx
    import jax.numpy as jnp

    V = _gated().with_item([0.7, -0.3], 0.5, amp=A).with_item([-1.9, 0.4], -0.5, amp=A)
    q_at = jnp.array([0.7, -0.3, 0.5])                 # y == a_i at the site
    grad = float(eqx.filter_grad(lambda q: V(q))(q_at)[2])
    assert abs(grad) < 1e-4                            # the payload spring is at rest
    # ... and the payload hill scales with the gated stiffness kappa*(g0 + A), so it
    # SHRINKS with the well instead of standing over it (that is the D3 fix)
    hill = float(V(jnp.array([0.7, -0.3, 0.0]))) - float(V(q_at))
    assert hill == pytest.approx(0.5 * (0.05 + A) * 0.5 ** 2, rel=0.02)


def test_payload_gate_fixes_the_D3_hill_over_well_inversion():
    """⭐ mia-D3 in one assertion, for the worst codeword |a| = 1 at the amp floor.

    Baseline: the payload hill is ``0.5*kappa*a^2 = 0.5`` at EVERY amplitude, against an
    address well of depth ``A`` — at ``A = 0.051`` the site is a net *maximum* of ``V``
    on the payload axis (measured mia §5: s5 = -0.113 at the floor, retention 0.25 for
    ``a = +1``). Gated: the hill decays with the well, so it never inverts.
    """
    import jax.numpy as jnp

    def hill(V, A):
        W = V.with_item([0.7, -0.3], 1.0, amp=A)
        at = jnp.array([0.7, -0.3, 1.0])
        return float(W(jnp.array([0.7, -0.3, 0.0]))) - float(W(at))

    A = 0.051
    assert hill(_store(4), A) == pytest.approx(0.5, rel=0.02)     # >> A: inverted
    assert hill(_store(4), A) > 9 * A
    assert hill(_gated(g0=0.05), A) == pytest.approx(0.5 * (0.05 + A), rel=0.02)
    assert hill(_gated(g0=0.05), A) <= A                          # never inverts


def test_payload_gate_survives_the_canonical_rebuild():
    """`Controller._canonical_sync` rebuilds the store — the gate must travel with it."""
    R = radius_for_cells(4, D_SAFE)
    c = Controller(_gated(g0=0.005, capacity=4), d_safe=D_SAFE, evict_policy="depth",
                   placement="canonical", lattice_radius=R)
    c.offer(item_id=0, q_new=np.array([0.3, 0.1]), payload=0.5)
    c.offer(item_id=1, q_new=np.array([-1.4, 0.6]), payload=-0.5)
    assert c.store.payload_gate is True
    assert c.store.payload_g0 == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# scope guards (theorist §4c — what the deletion claim does NOT cover)
# ---------------------------------------------------------------------------
def test_lru_eviction_is_forbidden_under_canonical_placement():
    """⭐ LRU is query history: a store that can LRU-evict is not order-independent."""
    with pytest.raises(ValueError, match="staleness"):
        Controller(_store(8), d_safe=D_SAFE, evict_policy="staleness",
                   placement="canonical", lattice_radius=3.0)
    # "depth" is allowed: amp = base*exp(-leak*age) is item-intrinsic
    Controller(_store(8), d_safe=D_SAFE, evict_policy="depth",
               placement="canonical", lattice_radius=3.0)


def test_canonical_requires_a_lattice_radius_and_refuses_incompatible_modes():
    with pytest.raises(ValueError, match="lattice_radius"):
        Controller(_store(8), d_safe=D_SAFE, evict_policy="depth", placement="canonical")
    with pytest.raises(ValueError, match="allow_relocation"):
        Controller(_store(8), d_safe=D_SAFE, evict_policy="depth", placement="canonical",
                   lattice_radius=3.0, allow_relocation=False)
    with pytest.raises(ValueError, match="peer_addresses_fn"):
        Controller(_store(8), d_safe=D_SAFE, evict_policy="depth", placement="canonical",
                   lattice_radius=3.0, peer_addresses_fn=lambda: np.zeros((0, 2)))
    with pytest.raises(ValueError, match="placement"):
        Controller(_store(8), d_safe=D_SAFE, placement="nonsense")


def test_delete_is_refused_under_refuse_and_relocate():
    """⛔ There is no exact deletion verb for a history-dependent allocator."""
    import jax

    c = Controller(_store(4), d_safe=1.0)
    c.offer(0, [0.0, 0.0], 0.5, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="canonical"):
        c.delete(0)
    with pytest.raises(KeyError):
        _ctrl(4, 3.0).delete(99)


def test_duplicate_item_ids_are_rejected_under_canonical():
    c = _ctrl(4, 3.0)
    c.offer(1, [0.0, 0.0], 0.5)
    with pytest.raises(ValueError, match="already live"):
        c.offer(1, [2.0, 0.0], 0.5)


def test_relocate_placement_is_untouched_by_the_canonical_path():
    """The default arm must behave exactly as in w23 (a proposal at its own site)."""
    import jax

    c = Controller(_store(4), d_safe=1.0)
    r = c.offer(0, [0.5, 0.5], 0.25, key=jax.random.PRNGKey(0))
    assert r["decision"] == "admit" and c.placement == "relocate" and c.placer is None
    np.testing.assert_allclose(np.asarray(c.store.centers)[0], [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# budget / decay under canonical placement
# ---------------------------------------------------------------------------
def test_budget_eviction_under_canonical_uses_depth_and_repacks():
    c = _ctrl(4, 3.0, budget=2)
    c.offer(0, [0.0, 0.0], 0.1)
    c.offer(1, [2.0, 0.0], 0.2)
    amps = np.asarray(c.store.amps).copy()
    shallow = min(c.records.values(), key=lambda r: amps[r.slot]).item_id
    r = c.offer(2, [0.0, 2.0], 0.3)
    assert r["decision"] in ("admit", "relocate")
    live = {rec.item_id for rec in c.records.values()}
    assert shallow not in live and live == {2, 0 if shallow == 1 else 1}
    assert c.n_live == 2 and c.stats["evicted"] == 1


def test_leaky_well_self_evicts_and_the_store_stays_canonical():
    R = radius_for_cells(8, D_SAFE)
    items = _items(6, seed=8, radius=R * 0.8)
    c = _ctrl(8, R, leak=1.0)
    for item_id, anchor, pay in items:
        c.offer(item_id=item_id, q_new=anchor, payload=pay,
                permanent=(item_id == 0))
    for _ in range(4):  # 1.0*exp(-4) = 0.018 < amp_floor 0.05
        c.tick()
    assert {r.item_id for r in c.records.values()} == {0}
    assert c.stats["decayed_out"] == 5
    ref = _build([items[0]], [0], R, capacity=8)
    assert _same(_arrays(c), _arrays(ref))
