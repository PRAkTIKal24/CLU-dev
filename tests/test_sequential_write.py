"""Tests for sequential-write interference and the MVC-0 admission gate (w21).

These pin the things whose silent breakage would turn the w21 result upside
down:

  * the **spacing gate is arithmetically vacuous on the w20 ring geometry**
    (site spacing 1.414 vs d_safe 1.10) — this is the headline of item 1, and it
    is a property of two config numbers, so it gets a regression test;
  * ``refuse-and-relocate`` must actually relocate to an admissible site, and
    must **refuse** (not silently accept) when no admissible site exists —
    refusal is a correct controller output, and a gate that never refuses is not
    a gate;
  * an **atom write is C3-local** (its induced fixed-point drift at the admission
    radius is ~5 orders below its drift at half a well width) whereas a learned
    global-support write is not — the whole w21 finding turns on this contrast;
  * the first-order law ``||H^-1 grad dV(q*)||`` must track the MEASURED drift
    for a small local write, else the C3 check is measuring nothing;
  * the designed store must actually store (payload recovered) and must fail its
    own blank control (the w20 leak-immune method rule);
  * the key->value dataset used for the cross-primitive arm must have distinct
    keys and distinct values, else "retention" is ill-defined.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.admission import (
    D_SAFE_MULT,
    admit_site,
    c3_admissible,
    c3_drift,
    disk_proposer,
    min_separation,
    spacing_ok,
)
from chlu.core.memory_potentials import AtomDictionaryPotential, designed_payloads
from chlu.experiments.exp_learned_memory import model_for
from chlu.experiments.exp_sequential_write import (
    evaluate_items,
    kv_dataset,
    sequential_run,
)


# ---------------------------------------------------------------------------
# The spacing gate (C5-A1/A2)
# ---------------------------------------------------------------------------
def test_min_separation_and_spacing_ok():
    stored = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert min_separation([1.0, 0.0], stored) == pytest.approx(1.0)
    assert min_separation([0.0, 0.0], np.zeros((0, 2))) == float("inf")
    assert spacing_ok([1.0, 0.0], stored, 0.9)
    assert not spacing_ok([1.0, 0.0], stored, 1.1)


def test_spacing_gate_is_vacuous_on_the_w20_ring_geometry():
    """⭐ The headline of item 1, pinned as arithmetic.

    w20 §3 writes on a K=4 ring of radius f=1: nearest-neighbour spacing is
    2 f sin(pi/4) = 1.4142. MVC-0's gate is d_safe = 4.4 * write_sigma_addr =
    1.10. Since 1.4142 > 1.10 the gate can NEVER fire on that geometry, so it
    cannot possibly change any w20 number. If a future config edit made this
    false, the gated-vs-ungated comparison would silently start measuring
    something else.
    """
    cfg = get_default_config().experiment_sequential_write
    spacing = 2.0 * cfg.f * np.sin(np.pi / cfg.interference_K)
    d_safe = cfg.d_safe_mult * cfg.write_sigma_addr
    assert spacing == pytest.approx(1.41421, abs=1e-4)
    assert d_safe == pytest.approx(1.10, abs=1e-9)
    assert spacing > d_safe  # => every w20 write is already admissible


def test_admit_relocate_refuse():
    stored = np.array([[0.0, 0.0]])
    d_safe = 1.0
    # far away -> admit, untouched
    a = admit_site(np.array([3.0, 0.0]), stored, d_safe)
    assert a["decision"] == "admit"
    assert np.allclose(a["site"], [3.0, 0.0])
    # too close, no proposer -> refuse (never a silent accept)
    r = admit_site(np.array([0.2, 0.0]), stored, d_safe)
    assert r["decision"] == "refuse" and r["site"] is None
    # too close, with a proposer -> relocate to an ADMISSIBLE site
    rel = admit_site(
        np.array([0.2, 0.0]),
        stored,
        d_safe,
        key=jax.random.PRNGKey(0),
        proposer=lambda k, n: disk_proposer(2.0, 2)(k, n)[:, :2],
        n_candidates=200,
    )
    assert rel["decision"] == "relocate"
    assert min_separation(rel["site"], stored) >= d_safe


def test_gate_refuses_when_the_store_is_saturated():
    """A gate that cannot refuse is not a gate (theorist A2: 7/20 refused)."""
    grid = np.stack(
        np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9)), -1
    ).reshape(-1, 2)
    out = admit_site(
        np.array([0.0, 0.0]),
        grid,
        1.0,
        key=jax.random.PRNGKey(1),
        proposer=lambda k, n: disk_proposer(2.0, 2)(k, n)[:, :2],
        n_candidates=300,
    )
    assert out["decision"] == "refuse"
    assert out["n_candidates_examined"] == 300


# ---------------------------------------------------------------------------
# C3: locality of the write operator
# ---------------------------------------------------------------------------
def _store(sites, payloads, s=0.35):
    V = AtomDictionaryPotential(dim=3, capacity=8, alpha=0.02, s=s)
    for c, a in zip(sites, payloads, strict=True):
        V = V.with_item(c, a)
    return V


def test_atom_write_is_local_at_the_admission_radius():
    """⭐ The 5 orders come from exp(-d^2/2s^2), not from the gate itself."""
    s = 0.35
    V0 = _store([[0.0, 0.0]], [0.5], s=s)
    q_star = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)

    far = V0.with_item([D_SAFE_MULT * s, 0.0], -0.5)  # d = d_safe
    near = V0.with_item([0.5 * s, 0.0], -0.5)  # deep inside the merger band

    d_far = float(c3_drift(V0, far, q_star)[0])
    d_near = float(c3_drift(V0, near, q_star)[0])
    assert d_far < 1e-3
    assert d_near > 1e-1
    assert d_near / max(d_far, 1e-12) > 1e3


def test_c3_admissible_uses_the_budget():
    V0 = _store([[0.0, 0.0]], [0.5])
    q_star = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    near = V0.with_item([0.2, 0.0], -0.5)
    ok, d = c3_admissible(V0, near, q_star, delta_budget=0.1)
    assert not ok and d.shape == (1,)
    ok2, _ = c3_admissible(V0, near, np.zeros((0, 3)), delta_budget=0.1)
    assert ok2  # nothing stored => trivially admissible


def test_first_order_drift_law_tracks_the_measured_drift():
    """``dq* = -H^-1 grad dV`` must predict where the minimum actually moves."""
    V0 = _store([[0.0, 0.0]], [0.5], s=0.35)
    q_star = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    V1 = V0.with_item([1.2, 0.0], -0.5)  # small but non-negligible perturbation

    predicted = float(c3_drift(V0, V1, q_star)[0])
    # measured: gradient-descend V1 from the stored minimum
    grad = jax.grad(lambda q: V1(q))
    q = jnp.asarray(q_star[0])
    for _ in range(4000):
        q = q - 0.02 * grad(q)
    measured = float(jnp.linalg.norm(q - jnp.asarray(q_star[0])))
    assert predicted > 0 and measured > 0
    assert 0.5 < predicted / measured < 2.0


# ---------------------------------------------------------------------------
# The designed store
# ---------------------------------------------------------------------------
def test_atom_dictionary_stores_and_reads_back_the_payload():
    cfg = get_default_config().experiment_sequential_write
    cfg.address_steps, cfg.read_steps, cfg.n_query_sequential = 400, 400, 4
    sites = np.array([[1.5, 0.0, 0.0], [-1.5, 0.6, 0.0]])
    pay = np.array([0.6, -0.4])
    V = _store(sites[:, :2], pay)
    sites[:, 2] = pay
    ev = evaluate_items(
        model_for(V, 3), sites, pay, cfg, seed=0, n_query=cfg.n_query_sequential
    )
    assert ev["finite"]
    assert ev["mean_strict"] >= 0.9


def test_blank_designed_store_fails_the_value_read():
    """The leak-immune blank control (w20 method finding), as a regression."""
    cfg = get_default_config().experiment_sequential_write
    cfg.address_steps, cfg.read_steps = 400, 400
    sites = np.array([[1.5, 0.0, 0.0], [-1.5, 0.6, 0.0]])
    pay = np.array([0.6, -0.4])
    V_blank = _store(sites[:, :2], [0.0, 0.0])  # identical geometry, NOTHING stored
    ev = evaluate_items(model_for(V_blank, 3), sites, pay, cfg, seed=0, n_query=4)
    assert ev["mean_strict"] <= 0.1


def test_atom_dictionary_full_raises():
    V = AtomDictionaryPotential(dim=3, capacity=2)
    V = V.with_item([0.0, 0.0], 0.1).with_item([2.0, 0.0], 0.2)
    assert V.n_stored == 2
    with pytest.raises(RuntimeError):
        V.with_item([4.0, 0.0], 0.3)


def test_atom_dictionary_rejects_low_dim():
    with pytest.raises(ValueError):
        AtomDictionaryPotential(dim=2)


# ---------------------------------------------------------------------------
# The sequential protocol
# ---------------------------------------------------------------------------
def test_sequential_run_designed_gated_never_loses_item_one():
    cfg = get_default_config().experiment_sequential_write
    cfg.n_sequential_items = 5
    cfg.address_steps, cfg.read_steps = 400, 300
    cfg.n_query_sequential = 4
    run = sequential_run("designed_gated", cfg, seed=0)
    assert run["n_admitted"] <= cfg.n_sequential_items
    assert len(run["history"]) == cfg.n_sequential_items
    assert all(h["item1_strict"] >= 0.9 for h in run["history"])


def test_sequential_run_is_paired_across_arms():
    """Both arms must see the SAME proposal sequence at a given seed, else the
    gated/ungated contrast is confounded with a different set of sites."""
    from chlu.experiments.exp_sequential_write import _propose_sequence

    cfg = get_default_config().experiment_sequential_write
    a = _propose_sequence(jax.random.split(jax.random.PRNGKey(3), 4)[0], 6, cfg, 3)
    b = _propose_sequence(jax.random.split(jax.random.PRNGKey(3), 4)[0], 6, cfg, 3)
    assert np.allclose(a, b)


def test_ungated_designed_run_admits_everything():
    cfg = get_default_config().experiment_sequential_write
    cfg.n_sequential_items = 6
    cfg.address_steps, cfg.read_steps = 200, 200
    cfg.n_query_sequential = 2
    run = sequential_run("designed_ungated", cfg, seed=0)
    assert run["n_admitted"] == 6
    assert all(d["decision"] == "admit" for d in run["decisions"])


# ---------------------------------------------------------------------------
# The cross-primitive key/value store
# ---------------------------------------------------------------------------
def test_kv_dataset_keys_and_values_are_distinct():
    keys, vals = kv_dataset(jax.random.PRNGKey(0), 16, 4, 64)
    assert keys.shape == (16, 4) and vals.shape == (16,)
    assert len(set(map(tuple, keys.tolist()))) == 16
    assert len(set(vals.tolist())) == 16
    assert vals.min() >= 1  # 0 is reserved


def test_value_tolerance_is_capped_by_the_codebook_spacing():
    """A tolerance wider than half the codebook spacing makes "the stored value
    came back" ambiguous between neighbouring codewords. At K=16 on [-1,1] the
    spacing is 0.133 while ``payload_tol`` is 0.1, so the cap MUST bind."""
    from chlu.experiments.exp_sequential_write import effective_payload_tol

    cfg = get_default_config().experiment_sequential_write
    pay16 = np.asarray(designed_payloads(16, seed=cfg.payload_seed))
    gap16 = float(np.min(np.diff(np.sort(pay16))))
    tol16 = effective_payload_tol(cfg, pay16)
    assert gap16 < 2 * cfg.payload_tol  # the naive tolerance WOULD be ambiguous
    assert tol16 < 0.5 * gap16
    # at K=8 the w20 absolute tolerance is already inside the cap
    pay8 = np.asarray(designed_payloads(8, seed=cfg.payload_seed))
    assert effective_payload_tol(cfg, pay8) == pytest.approx(cfg.payload_tol)
    # a single item has no codebook to be confused with
    assert effective_payload_tol(cfg, np.array([0.3])) == cfg.payload_tol


def test_config_group_is_registered_everywhere():
    from chlu.config import CHLUConfig

    cfg = get_default_config()
    assert hasattr(cfg, "experiment_sequential_write")
    assert isinstance(cfg, CHLUConfig)
