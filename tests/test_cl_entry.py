"""Tests for the w25 continual-learning entry (`exp_cl_entry` + `cl_baselines`).

These pin the things whose silent breakage would flip the entry's verdict:

  * the **high-dimensional address block** — ``AtomStorePotential`` and the MVC-0
    controller must work with ``addr_dim = phi_dim`` (the store is addressed by
    ``φ(x)``, not by a 2-D plane), and the default ``addr_dim = 2`` behaviour must
    be bit-unchanged;
  * ``allow_relocation=False`` — in a **content-addressed** store a refused item is
    refused, never moved (a relocated address is an address no query can reach);
  * per-well amplitudes in ``GaussianMemoryPotential`` (default = the old uniform
    store), because that is what makes scheduled decay physical rather than
    bookkeeping;
  * the **Class-IL protocol**: the read-out is masked to the classes seen so far and
    the accuracy matrix is lower-triangular; the GEM metric formulas;
  * the **φ protocol** (PREREG_CL_PHI): only ``task1_only``/``generic_frozen`` are
    runnable, the task-1 fit pool contains task-1 classes only, and the fit region is
    disjoint from the stream region;
  * the **R1 naming rule** (CM-22 m/n/o): the retention result never says
    "certified"/"unlearning"/"deletion";
  * the whole pipeline runs end-to-end on tiny synthetic labelled data.
"""

import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.controller import Controller
from chlu.core.memory_potentials import AtomStorePotential, GaussianMemoryPotential
from chlu.experiments import exp_cl_entry as cle
from chlu.experiments.cl_baselines import cl_metrics


# ---------------------------------------------------------------------------
# synthetic labelled data: 10 well-separated Gaussian class blobs
# ---------------------------------------------------------------------------
def _toy_data(n_per_class=40, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, dim)) * 3.0
    X, y = [], []
    for c in range(10):
        X.append(centers[c] + rng.normal(size=(n_per_class, dim)) * 0.35)
        y.append(np.full(n_per_class, c))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])


def _toy_cfg():
    cfg = get_default_config().experiment_cl_entry
    cfg.seeds = [0]
    cfg.n_train_per_task = 20
    cfg.n_test_per_task = 12
    cfg.n_fit_region = 120
    cfg.n_fit_pool = 60
    cfg.phi_dim = 16
    cfg.memory_items = 16
    cfg.clu_steps = 25
    cfg.rollout_chunk = 32
    cfg.baseline_iters = 12
    cfg.baselines = ["finetune", "er"]
    cfg.tune_baselines = False
    cfg.retry_ladder = [0, 1]
    cfg.retry_mask_levels = [0.5]
    cfg.retry_tau_grid = [1.0]
    cfg.ticks_per_task = 2
    cfg.permanent_per_task = 3
    cfg.mlp_width = 32
    cfg.fisher_samples = 10
    return cfg


# ---------------------------------------------------------------------------
# the store: a φ-dimensional address block
# ---------------------------------------------------------------------------
def test_atom_store_defaults_to_the_2d_address_plane():
    """The w23/w24 behaviour must be untouched by the addr_dim generalization."""
    V = AtomStorePotential(dim=3, capacity=4, alpha=0.02, s=0.35)
    assert V.addr_dim == 2
    assert np.asarray(V.centers).shape == (4, 2)
    V = V.with_item([0.3, -0.2], 0.7, amp=1.0)
    assert np.allclose(np.asarray(V.centers)[0], [0.3, -0.2])
    assert float(V(np.array([0.3, -0.2, 0.7]))) < 0.0  # sits in the well


def test_atom_store_supports_a_high_dimensional_address():
    d = 16
    V = AtomStorePotential(dim=d + 1, capacity=4, alpha=1e-3, s=0.5, addr_dim=d)
    c = np.linspace(-1, 1, d)
    V = V.with_item(c, 0.25, amp=1.0)
    assert np.asarray(V.centers).shape == (4, d)
    q_at = np.concatenate([c, [0.25]])
    q_far = np.concatenate([c + 5.0, [0.25]])
    assert float(V(q_at)) < float(V(q_far))  # the well is where we wrote it
    assert np.asarray(V.sites()).shape == (1, d + 1)


def test_atom_store_rejects_too_small_a_latent_dim():
    with pytest.raises(ValueError):
        AtomStorePotential(dim=8, capacity=2, addr_dim=8)  # needs dim >= addr_dim+1


def test_gaussian_memory_amps_default_to_the_uniform_store():
    C = np.array([[0.0, 0.0], [2.0, 0.0]], np.float32)
    v_plain = GaussianMemoryPotential(C, s=0.3, b=1.0)
    v_ones = GaussianMemoryPotential(C, s=0.3, b=1.0, amps=np.ones(2))
    q = np.array([0.1, 0.05], np.float32)
    assert float(v_plain(q)) == pytest.approx(float(v_ones(q)), rel=1e-6)


def test_gaussian_memory_amps_shallow_a_decayed_well():
    """A decayed well is physically shallower — that is what makes the retention
    schedule a property of the landscape, not of a bookkeeping table."""
    C = np.array([[0.0, 0.0], [2.0, 0.0]], np.float32)
    q = np.array([0.05, 0.0], np.float32)
    full = float(GaussianMemoryPotential(C, s=0.3, amps=[1.0, 1.0])(q))
    decayed = float(GaussianMemoryPotential(C, s=0.3, amps=[0.1, 1.0])(q))
    assert decayed > full  # shallower well ⇒ higher energy at the same point


# ---------------------------------------------------------------------------
# the controller in a content-addressed store
# ---------------------------------------------------------------------------
def _phi_controller(d=8, d_safe=1.0, budget=4):
    store = AtomStorePotential(dim=d + 1, capacity=budget, alpha=1e-3, s=0.2, addr_dim=d)
    return Controller(store, d_safe=d_safe, budget=budget, allow_relocation=False)


def test_controller_refuses_never_relocates_when_content_addressed():
    ctrl = _phi_controller()
    a = np.zeros(8)
    assert ctrl.offer(0, a, 1.0)["decision"] == "admit"
    near = a.copy()
    near[0] = 0.1  # inside d_safe
    row = ctrl.offer(1, near, 2.0)
    assert row["decision"] == "refuse_spacing"
    assert row["n_candidates_examined"] == 0  # no relocation was even attempted
    far = a.copy()
    far[0] = 5.0
    assert ctrl.offer(2, far, 3.0)["decision"] == "admit"
    assert ctrl.stored_addresses().shape == (2, 8)


def test_controller_evict_item_is_public_and_removes_by_id():
    ctrl = _phi_controller()
    for i in range(3):
        v = np.zeros(8)
        v[0] = 3.0 * i
        ctrl.offer(i, v, float(i))
    assert ctrl.n_live == 3
    assert ctrl.evict_item(1) is True
    assert ctrl.n_live == 2
    assert 1 not in [r.item_id for r in ctrl.records.values()]
    assert ctrl.evict_item(99) is False


def test_live_amps_tracks_scheduled_decay():
    ctrl = _phi_controller(budget=3)
    for i in range(2):
        v = np.zeros(8)
        v[0] = 3.0 * i
        ctrl.offer(i, v, float(i), permanent=(i == 0), leak=0.4)
    ctrl.tick()
    amps = ctrl.live_amps()
    assert amps[0] == pytest.approx(1.0)  # permanent ⇔ leak 0
    assert amps[1] == pytest.approx(np.exp(-0.4), rel=1e-5)


# ---------------------------------------------------------------------------
# CL metrics (GEM formulas) + the Class-IL protocol
# ---------------------------------------------------------------------------
def test_cl_metrics_match_the_gem_formulas():
    A = np.array([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.5, 0.8]])
    m = cl_metrics(A)
    assert m["ACC"] == pytest.approx((0.2 + 0.5 + 0.8) / 3)
    assert m["BWT"] == pytest.approx(((0.2 - 1.0) + (0.5 - 0.9)) / 2)
    assert m["forgetting"] == pytest.approx(((1.0 - 0.2) + (0.9 - 0.5)) / 2)


def test_stream_regions_are_disjoint_and_task1_pool_is_task1_only():
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    assert len(st["train_X"]) == cfg.n_tasks == len(st["test_X"])
    # the task-1-only pool may only contain task-1 classes
    (Xtr, ytr), _ = _toy_data()
    pool = st["fit_pool_task1_only"]
    rows = {tuple(np.round(r, 5)) for r in pool}
    t1_rows = {
        tuple(np.round(r, 5))
        for r in Xtr[np.isin(ytr, st["task_classes"][0])]
    }
    assert rows <= t1_rows and len(rows) > 0
    # no stored (stream) item is ever in a φ fit pool
    stream_rows = {
        tuple(np.round(r, 5)) for X in st["train_X"] for r in X
    }
    assert rows.isdisjoint(stream_rows)
    generic_rows = {tuple(np.round(r, 5)) for r in st["fit_pool_generic_frozen"]}
    assert generic_rows.isdisjoint(stream_rows)


def test_out_of_protocol_phi_regime_is_refused():
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    with pytest.raises(ValueError):
        cle.build_phi("online", st, cfg, 0)


# ---------------------------------------------------------------------------
# the entry itself
# ---------------------------------------------------------------------------
def test_clu_entry_runs_and_produces_a_lower_triangular_matrix():
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    res = cle.run_clu_entry(cfg, st, cle.PHI_PRIMARY, seed=0)
    A = np.array(res["A_clu"])
    assert A.shape == (cfg.n_tasks, cfg.n_tasks)
    assert np.all(A[np.triu_indices(cfg.n_tasks, k=1)] == 0.0)  # never evaluated ahead
    assert 0.0 <= res["metrics_clu"]["ACC"] <= 1.0
    # the store never exceeds its item budget and stores no exemplar
    assert res["memory_items"] <= cfg.memory_items
    assert res["memory_floats"] == res["memory_items"] * cfg.phi_dim
    # the laundering lines are always computed alongside (N89 is not optional)
    assert "metrics_knn_same_keys" in res and "metrics_knn_ringbuffer" in res
    # every task reports its admission accounting (per-admitted needs the fraction)
    for row in res["per_task"]:
        assert 0.0 <= row["admitted_fraction"] <= 1.0
        assert row["well_width_s"] > 0.0


def test_permanent_items_survive_the_whole_stream_under_decay():
    """⛔ *scheduled per-item retention* — not deletion, not unlearning."""
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    res = cle.run_clu_entry(cfg, st, cle.PHI_PRIMARY, seed=0, decay_on=True)
    law = cle.retention_law_check(res["retention_rows"], cfg)
    assert law["permanent"]["n_points"] > 0
    assert all(a == pytest.approx(1.0) for a in law["permanent"]["measured_amp"])
    for cohort in ("slow", "fast"):
        d = law.get(cohort, {})
        if d.get("n_points"):
            # the measured amplitude follows the SCHEDULE exp(−leak·t)
            assert d["max_abs_error"] < 1e-4
            assert d["measured_amp"][-1] < 1.0


def test_r1_naming_rule_is_enforced_in_the_reported_strings():
    """CM-22 m/n/o: the forbidden words must not appear in what we emit."""
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    res = cle.run_clu_entry(cfg, st, cle.PHI_PRIMARY, seed=0, decay_on=True)
    blob = str(cle.retention_law_check(res["retention_rows"], cfg)).lower()
    for word in cle.FORBIDDEN_R1_WORDS:
        assert f" {word}" not in blob.replace("not unlearning", "").replace(
            "not a privacy claim", ""
        ) or f"not {word}" in blob
    assert cle.R1_NAME in blob


def test_phi_space_feedforward_floor_is_a_floor_not_a_collapse():
    """⚠ The w24 ladder augments with ``clip(|q+noise|,0,1)`` — valid for pixels in
    [0,1], DESTRUCTIVE for signed φ vectors (it collapsed the 'floor' to ~0.005).
    The φ-space version must (a) equal exact 1-NN at k=0 and (b) not collapse."""
    rng = np.random.default_rng(0)
    C = rng.normal(size=(20, 8)) * 3.0
    Q = C + rng.normal(size=C.shape) * 0.2  # queries near their own address
    true_idx = np.arange(20)
    cfg = _toy_cfg()
    cfg.retry_ladder = [0, 1, 2]
    out = cle._feedforward_ladder_phi(Q, C, true_idx, cfg, np.random.default_rng(1))
    exact = float(np.mean(((Q[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1) == true_idx))
    assert out[0][0] == pytest.approx(exact)
    assert out[0][1] == 1.0
    assert out[2][0] >= exact - 0.15  # votes may not help, but must not destroy


def test_eviction_cause_is_attributed_schedule_vs_budget():
    """Scheduled forgetting (the dial) must never be confused with budget pressure
    (the capacity policy) — they are counted separately, per cohort."""
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    res = cle.run_clu_entry(cfg, st, cle.PHI_PRIMARY, seed=0, decay_on=True)
    law = cle.retention_law_check(res["retention_rows"], cfg)
    ev = law["evictions"]
    assert "evicted_by_schedule_per_cohort" in ev and "evicted_by_budget_per_cohort" in ev
    # permanent items are exempt from BOTH causes
    assert ev["evicted_by_schedule_per_cohort"].get("permanent", 0) == 0
    assert ev["evicted_by_budget_per_cohort"].get("permanent", 0) == 0
    # a scheduled eviction can only be a leaky cohort
    assert set(ev["evicted_by_schedule_per_cohort"]) <= {"slow", "fast"}


def test_end_to_end_driver_writes_metrics_and_the_verdict(tmp_path):
    cfg_all = get_default_config()
    cfg_all.experiment_cl_entry = _toy_cfg()
    res = cle.run_experiment_cl_entry(
        config=cfg_all, save_dir=str(tmp_path / "plots"), seed=0,
        items=["entry", "retention"], data=_toy_data(),
    )
    assert res["baseline_table"], "the mandatory baseline table must not be empty"
    methods = {r["method"] for r in res["baseline_table"]}
    assert "clu_entry_task1_only" in methods
    assert any(m.startswith("knn_phi_") for m in methods)  # the launder always runs
    v = res["verdict"]
    assert "wins_rehearsal_free_class" in v and "laundered" in v
    assert "beats replay" not in str(v).lower().replace("'beats replay' is never claimed", "")
