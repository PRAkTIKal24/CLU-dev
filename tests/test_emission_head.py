"""C2W8 PASS 2, ARM B — the emission head, and **K7**: the capture instrument
must be proven able to report a POSITIVE.

Four kill-conditions live here, all mechanical:

* **K7** (`test_k7_*`) — pass 2's most important one, and it exists because of
  pass 1's own failure: before ``capture_radius > 0`` is trusted as a *success*
  criterion, the instrument must fire on a **planted** basin and return 0 on a
  **planted flat** site. Two-sided, plus a third cell that shows it recovers the
  planted *geometry* (doubling the separation doubles the radius), so a green
  G-CAP is not "an untested instrument agreeing with us".
  ⚠ Predictions were registered in ``PREREG.md`` before this file was run.
* **K6** (`test_emission_head_off_is_*`) — the flag OFF is bit-identical AND
  parameter-count-identical to the pre-arm path.
* **K8** (`test_k8_*`) — ``wells_per_item`` and ``vocabulary_shared`` are
  declared, and a private-well arm is labelled ``NO_TIER_II_CLAIM``.
* **the pinning prohibition** (`test_reach_penalty_is_not_a_pin_*`) — the only
  term coupling the emitted center to ``phi`` is a hinge with a free width, and
  the two properties that distinguish it from a pin are asserted, not promised.

⛔ ``chlu/core/well_lifecycle.py`` is READ-ONLY this wave: K7 is asserted here,
against that instrument, and nothing in it is modified.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.emission_head import (
    NO_TIER_II_CLAIM,
    EmissionHead,
    WellParams,
    apply_emitted_well,
    compose_wells,
    emission_ledger,
    emitted_potential,
    private_vocabulary,
    reach_penalty,
)
from chlu.core.well_lifecycle import (
    capture_radii,
    flatten_unused_groups,
    own_foreign_site_depth,
    plant_item,
)

R_HI = 1.0
BISECT_STEPS = 8
N_DIRS = 16
#: `lo` saturates one bisection resolution below `r_hi` when every launch returns
SATURATED = R_HI * (1.0 - 2.0 ** -BISECT_STEPS)  # 0.99609375


def _k7_system(capacity=4, seed=0, **over):
    """A 2-D-address store cheap enough to bisect, at the census's read horizon."""
    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=capacity, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=0.05, read_steps=800, address_steps=40,
        n_query_per_item=2, **over,
    )
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)


def _measure(system, sites):
    return capture_radii(system, np.atleast_2d(sites), n_dirs=N_DIRS,
                         steps=BISECT_STEPS, seed=0)


# ==========================================================================
# K7 — the capture instrument, two-sided, PLUS the planted geometry
# ==========================================================================
def test_k7_positive_isolated_planted_well_saturates_the_search_bound():
    """K7-a. PREREG: exactly ``r_hi * (1 - 2^-steps) = 0.99609375``.

    With one Gaussian well and the coercive bowl as the ONLY features, the well's
    minimum sits ``2*alpha*s^2*|z0|/D = 2.7e-3`` from the planted site (three
    orders below ``tol``), so every launch inside ``r_hi`` returns and the
    bisection's lower bound saturates.
    """
    sysm = _k7_system()
    plant_item(sysm, 0, np.array([0.5, 0.0]), payload=0.1, depth=1.2, width=0.25)
    flatten_unused_groups(sysm)

    site = np.array([0.5, 0.0, 0.1])
    own, foreign = own_foreign_site_depth(sysm.store, 0, site)
    assert own == pytest.approx(1.2, rel=1e-4) and foreign == 0.0

    r = _measure(sysm, site)[0]
    assert r == pytest.approx(SATURATED, abs=1e-9), r


def test_k7_negative_planted_flat_site_returns_exactly_zero():
    """K7-b. A flat site 0.51 from the bowl's minimum: nothing returns."""
    sysm = _k7_system()
    plant_item(sysm, 0, np.array([0.5, 0.0]), payload=0.1, depth=0.0, width=0.25)
    flatten_unused_groups(sysm)

    site = np.array([0.5, 0.0, 0.1])
    assert own_foreign_site_depth(sysm.store, 0, site) == (0.0, 0.0)
    r = _measure(sysm, site)[0]
    assert r == 0.0, r


@pytest.mark.parametrize("R,predicted", [(0.30, 0.31000), (0.60, 0.62001)])
def test_k7_positive_finite_recovers_the_planted_separatrix(R, predicted):
    """K7-c. Two identical wells at ``+-R e0``: by exact reflection symmetry the
    separatrix IS the hyperplane ``q0 = 0``, so along direction ``u`` the crossing
    is ``R / (-u0)``.

    PREREG (directions are deterministic, ``default_rng(0)``, 16 x 3, normalised;
    ``max_k(-u_k0) = 0.9677334``): ``0.31000`` at ``R = 0.30`` and ``0.62001`` at
    ``R = 0.60``, to +-25 % — the slack is the bisection resolution (one-sided
    DOWN) plus finite-horizon damped relaxation (also one-sided down).
    """
    sysm = _k7_system(capacity=4)
    plant_item(sysm, 0, np.array([+R, 0.0]), payload=0.0, depth=1.2, width=0.25)
    plant_item(sysm, 1, np.array([-R, 0.0]), payload=0.0, depth=1.2, width=0.25)
    flatten_unused_groups(sysm)

    r = _measure(sysm, np.array([R, 0.0, 0.0]))[0]
    assert 0.0 < r < SATURATED, r                      # finite, not saturating
    assert r == pytest.approx(predicted, rel=0.25), r  # and where it was planted


def test_k7_recovers_the_planted_geometry_not_a_constant():
    """K7-c, the ratio: doubling the planted separation doubles the radius.

    PREREG: ratio = 2.000, registered acceptance [1.6, 2.4]. This is what makes
    K7 a statement about the *instrument* rather than about one lucky store.
    """
    out = {}
    for R in (0.30, 0.60):
        sysm = _k7_system(capacity=4)
        plant_item(sysm, 0, np.array([+R, 0.0]), payload=0.0, depth=1.2, width=0.25)
        plant_item(sysm, 1, np.array([-R, 0.0]), payload=0.0, depth=1.2, width=0.25)
        flatten_unused_groups(sysm)
        out[R] = _measure(sysm, np.array([R, 0.0, 0.0]))[0]
    ratio = out[0.60] / max(out[0.30], 1e-12)
    assert 1.6 <= ratio <= 2.4, out


# ==========================================================================
# K6 — OFF is bit-identical AND parameter-count-identical
# ==========================================================================
def _n_params(m):
    return int(sum(np.asarray(x).size for x in
                   jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))))


def test_emission_head_off_is_the_shipped_default():
    """The flag ships OFF and does not appear in any default flag table."""
    base = CluSystemConfig()
    assert base.emission_head is False
    assert base.as_flag_table() == {}
    assert CluSystemConfig(emission_head=True).as_flag_table() == {
        "emission_head": True}


def test_emission_head_off_is_bit_identical_and_parameter_count_identical():
    """⛔ K6. OFF means the head is never constructed, so the read model is the
    pre-arm tree and the write takes the shipped ``train_memory_landscape``
    branch. ON adds **no leaf to the read model at all** — the head is a WRITE
    mechanism — but it IS on the byte ledger.
    """
    off = _k7_system(emission_head=False)
    ref = _k7_system()  # the flag absent entirely
    on = _k7_system(emission_head=True)

    assert off.emitter is None and ref.emitter is None and on.emitter is not None
    # (i) parameter-count-identical, both ways
    assert _n_params(off.model()) == _n_params(ref.model())
    assert _n_params(on.model()) == _n_params(off.model())
    assert off.emission_bytes() == 0 and ref.emission_bytes() == 0
    assert on.emission_bytes() > 0
    # (ii) bit-identical reads
    q0 = np.zeros((3, off.store.dim), dtype=np.float32)
    q0[:, 0] = np.array([0.1, -0.3, 0.6])
    assert np.array_equal(np.asarray(off.read(q0).state.q_star),
                          np.asarray(ref.read(q0).state.q_star))
    # (iii) bit-identical WRITES (the write_stream / _write_item edits are no-ops)
    for s in (off, ref):
        s.write_stream([{"item_id": 0, "address": np.array([0.4, 0.1]),
                         "payload": 0.2}])
    assert np.array_equal(np.asarray(off.store.V.learned.amp),
                          np.asarray(ref.store.V.learned.amp))
    assert np.array_equal(np.asarray(off.store.V.learned.centers),
                          np.asarray(ref.store.V.learned.centers))
    # and the byte ledger is unchanged with the flag off
    assert off.n_bytes() == ref.n_bytes()


# ==========================================================================
# K8 — the declared trap, machine-checkable
# ==========================================================================
def test_k8_ledger_declares_wells_per_item_and_vocabulary_shared():
    head = EmissionHead(addr_dim=4, payload_dim=1, key=jax.random.PRNGKey(0),
                        hidden=8, layers=1)
    led = emission_ledger(head, n_items=16)
    for k in ("wells_per_item", "vocabulary_shared", "vocabulary_size",
              "tier_ii_status", "head_param_count", "head_bytes"):
        assert k in led, k
    assert led["wells_per_item"] == 1
    assert led["vocabulary_shared"] is False
    assert led["tier_ii_status"] == NO_TIER_II_CLAIM
    assert led["head_bytes"] == 4 * led["head_param_count"] > 0
    assert led["factored_store_built"] is False


def test_k8_a_shared_vocabulary_arm_is_not_auto_labelled():
    """The label is earned by the CONFIGURATION, not stamped on the arm's name."""
    led = emission_ledger(None, n_items=64, vocabulary_shared=True,
                          vocabulary_size=8)
    assert led["vocabulary_shared"] is True
    assert led["tier_ii_status"] != NO_TIER_II_CLAIM
    assert led["vocabulary_size"] == 8


def test_the_private_well_case_is_the_degenerate_shared_vocabulary_case():
    """⭐ The interface does not foreclose the factored store — bitwise.

    ``compose_wells(private_vocabulary(p))`` must return ``p`` exactly, i.e. the
    private-well configuration this arm ships IS the ``V = n_items,
    coefficients = one-hot`` special case of the shared-vocabulary interface.
    """
    head = EmissionHead(addr_dim=3, payload_dim=1, key=jax.random.PRNGKey(1),
                        hidden=8, layers=1)
    phi = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)), dtype=jnp.float32)
    a = jnp.asarray(np.linspace(-0.4, 0.4, 5).reshape(5, 1), dtype=jnp.float32)
    params = head.emit(phi, a)

    vocab, coeffs = private_vocabulary(params)
    assert vocab.shared is False and vocab.size == 5
    assert np.array_equal(np.asarray(coeffs), np.eye(5, dtype=np.float32))
    got = compose_wells(vocab, coeffs)
    assert np.array_equal(np.asarray(got.centers), np.asarray(params.centers))
    assert np.array_equal(np.asarray(got.log_widths), np.asarray(params.log_widths))
    assert np.array_equal(np.asarray(got.log_depths), np.asarray(params.log_depths))


# ==========================================================================
# the pinning prohibition — the reach hinge is NOT a pin
# ==========================================================================
def _params(center, log_width):
    return WellParams(centers=jnp.asarray([center], dtype=jnp.float32),
                      log_widths=jnp.asarray([log_width], dtype=jnp.float32),
                      log_depths=jnp.asarray([0.0], dtype=jnp.float32))


def test_reach_penalty_is_not_a_pin_zero_value_and_zero_gradient_when_reachable():
    """⛔ Property 1. Inside ``rho*s`` the hinge is bitwise 0 with a bitwise-0
    gradient in the center. An L2 pin has neither: it is never 0 and always pulls.
    """
    phi = jnp.asarray([[0.5, 0.0]], dtype=jnp.float32)
    # |q_launch - z| = 0.4  <  rho*s = 2 * 0.3 = 0.6  => inside
    p = _params([0.9, 0.0, 0.0], np.log(0.3))

    val = float(reach_penalty(p, phi, addr_dim=2, rho=2.0))
    assert val == 0.0

    g = jax.grad(lambda c: reach_penalty(
        WellParams(centers=c, log_widths=p.log_widths, log_depths=p.log_depths),
        phi, addr_dim=2, rho=2.0))(p.centers)
    assert np.array_equal(np.asarray(g), np.zeros_like(np.asarray(g)))

    # the pin it is NOT: |c - phi|^2 at the same point
    q = jnp.zeros_like(p.centers).at[:, :2].set(phi)
    pin = jax.value_and_grad(lambda c: jnp.sum((c - q) ** 2))(p.centers)
    assert float(pin[0]) == pytest.approx(0.16, rel=1e-5)
    assert float(np.max(np.abs(np.asarray(pin[1])))) > 0.5


def test_reach_penalty_is_satisfiable_by_the_WIDTH_alone():
    """⛔ Property 2. Its zero set is a manifold, not the point ``c = phi``: the
    same (unmoved) center satisfies it once the emitted width grows."""
    phi = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)
    center = [0.9, 0.0, 0.0]  # |q_launch - z| = 0.9, unchanged throughout

    narrow = float(reach_penalty(_params(center, np.log(0.3)), phi,
                                 addr_dim=2, rho=2.0))
    wide = float(reach_penalty(_params(center, np.log(0.5)), phi,
                               addr_dim=2, rho=2.0))
    assert narrow == pytest.approx(0.09, rel=1e-4)  # relu(0.9 - 0.6)^2
    assert wide == 0.0                              # rho*s = 1.0 > 0.9, center same


# ==========================================================================
# the emission itself
# ==========================================================================
def test_emitted_potential_is_the_shipped_atom_dictionary_form():
    """The well's functional form is unchanged: the emitted landscape IS an
    ``AtomDictionaryPotential``, evaluated against its closed form."""
    p = WellParams(centers=jnp.asarray([[0.2, -0.1, 0.05]], dtype=jnp.float32),
                   log_widths=jnp.asarray([np.log(0.4)], dtype=jnp.float32),
                   log_depths=jnp.asarray([np.log(1.5)], dtype=jnp.float32))
    V = emitted_potential(p, dim=3, confine=0.05)
    q = jnp.asarray([0.3, 0.0, 0.0], dtype=jnp.float32)
    d2 = float(np.sum((np.asarray(q) - np.asarray(p.centers)[0]) ** 2))
    want = 0.05 * float(np.sum(np.asarray(q) ** 2)) - 1.5 * np.exp(
        -d2 / (2.0 * 0.4 ** 2 + 1e-9))
    assert float(V(q)) == pytest.approx(want, rel=1e-5)


def test_head_emits_inside_its_declared_bands_and_cannot_invent_a_payload():
    """Width/depth are squashed into the declared bands, and the emitted payload
    coordinate is the item's OWN value plus a bounded correction — the head
    encodes the content it is handed, it never predicts one from ``phi``."""
    head = EmissionHead(addr_dim=4, payload_dim=1, key=jax.random.PRNGKey(3),
                        hidden=16, layers=2, width_min=0.15, width_max=0.8,
                        depth_min=0.05, depth_max=3.0, payload_delta_max=0.05)
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.normal(size=(32, 4)) * 3.0, dtype=jnp.float32)
    a = jnp.asarray(rng.uniform(-0.5, 0.5, size=(32, 1)), dtype=jnp.float32)
    p = head.emit(phi, a)
    w, d = np.asarray(p.widths), np.asarray(p.depths)
    assert np.all((w > 0.15) & (w < 0.80))
    assert np.all((d > 0.05) & (d < 3.0))
    emitted_pay = np.asarray(p.centers)[:, 4]
    assert np.all(np.abs(emitted_pay - np.asarray(a)[:, 0]) <= 0.05 + 1e-6)
    # the same phi with a DIFFERENT payload emits a different payload coordinate
    p2 = head.emit(phi, a + 0.3)
    assert np.all(np.asarray(p2.centers)[:, 4] > emitted_pay)


def test_apply_emitted_well_puts_exactly_the_emitted_depth_at_the_site():
    sysm = _k7_system()
    sysm.controller.admit(7, np.array([0.3, -0.2]), 0.15)
    sysm._payloads[7] = np.array([0.15])
    slot = sysm._slot_of(7)
    site = np.array([0.3, -0.2, 0.15])
    sysm.store = apply_emitted_well(sysm.store, slot, site, width=0.42, depth=1.7)
    own, _ = own_foreign_site_depth(sysm.store, slot, site)
    assert own == pytest.approx(1.7, rel=1e-4)
    rows = np.asarray(sysm.store.group_rows(slot), dtype=bool)
    s = np.exp(np.asarray(sysm.store.atoms.log_width)[rows])
    assert np.allclose(s, 0.42, rtol=1e-5)


def test_emission_write_is_a_forward_pass_and_places_the_codebook_at_the_well():
    """End-to-end: with the head on, a write digs the emitted well AND the
    codebook records the site the well is at (so ``site_drift`` is measured
    against the well, not against a stale ``phi`` key)."""
    sysm = _k7_system(emission_head=True, capacity=4)
    keys = np.array([[0.4, 0.1], [-0.5, 0.3], [0.1, -0.6]])
    for i, k in enumerate(keys):
        sysm.write_stream([{"item_id": i, "address": k,
                            "payload": 0.2 * (i - 1)}])
    ids, centers, pays = sysm.codebook()
    assert len(ids) == 3
    for i, iid in enumerate(ids):
        z = np.concatenate([centers[i], pays[i]])
        own, _ = own_foreign_site_depth(sysm.store, sysm._slot_of(int(iid)), z)
        assert own > 0.05, (iid, own)          # inside the emitted-depth band
        # the codebook center IS the emitted center, not the raw phi key
        assert np.allclose(centers[i],
                           sysm.emitter.emit_center(
                               sysm._emitted_from_phi[int(iid)], pays[i]),
                           atol=1e-6)
    # ...and |c - phi| is a real, reported, NON-ZERO quantity (never pinned)
    gaps = [float(np.linalg.norm(centers[i] - sysm._emitted_from_phi[int(iid)]))
            for i, iid in enumerate(ids)]
    assert min(gaps) > 0.0
    assert sysm.emission_bytes() > 0
    assert sysm.n_bytes() > sysm.store.n_bytes() + sysm.emission_bytes() - 1


def test_pretraining_moves_the_head_through_the_designed_write_objective():
    """The amortised cost is real: the shipped write objective (imported, not
    re-implemented) decreases and the head's parameters actually move."""
    from chlu.core.emission_head import emission_write_loss, pretrain_emission_head

    head = EmissionHead(addr_dim=2, payload_dim=1, key=jax.random.PRNGKey(2),
                        hidden=16, layers=2)
    rng = np.random.default_rng(5)
    phi = rng.normal(size=(24, 2)) * 0.5
    pay = rng.uniform(-0.5, 0.5, size=(24, 1))
    k = jax.random.PRNGKey(11)
    before = float(emission_write_loss(head, jnp.asarray(phi, dtype=jnp.float32),
                                       jnp.asarray(pay, dtype=jnp.float32), k,
                                       dim=3, confine=0.05, addr_dim=2,
                                       loss_kwargs={"n_perturb": 8}))
    trained, hist = pretrain_emission_head(
        head, phi, pay, jax.random.PRNGKey(12), dim=3, confine=0.05, addr_dim=2,
        steps=40, batch=8, loss_kwargs={"n_perturb": 8})
    after = float(emission_write_loss(trained, jnp.asarray(phi, dtype=jnp.float32),
                                      jnp.asarray(pay, dtype=jnp.float32), k,
                                      dim=3, confine=0.05, addr_dim=2,
                                      loss_kwargs={"n_perturb": 8}))
    assert after < before, (before, after)
    assert hist["objective_evals"] == 40 * 8
    assert not np.array_equal(
        np.asarray(head.mlp.layers[0].weight),
        np.asarray(trained.mlp.layers[0].weight))
