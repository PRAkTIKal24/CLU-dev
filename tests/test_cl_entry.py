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

import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.controller import Controller
from chlu.core.memory_potentials import AtomStorePotential, GaussianMemoryPotential
from chlu.experiments import exp_cl_entry as cle
from chlu.experiments.cl_baselines import (
    _train_task,
    cl_metrics,
    fixed_state_floats,
    floats_per_stored_item,
    items_for_budget,
    make_net,
    run_baseline_stream,
)


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
    # ⚠ w26: memory_floats uses the PINNED per-item accounting (address + payload
    # + amp + active + the controller's record scalars), not the address alone
    assert res["memory_floats"] == res["memory_items"] * (cfg.phi_dim + 3 + 6)
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


# ---------------------------------------------------------------------------
# ⭐ w26 — the matched-BYTES forgetting frontier (`matched-bytes-frontier`)
#
# What must never break silently here: the BYTE ACCOUNTING (a referee checks it
# first), the item budget every method is actually held to (an over-running
# baseline makes the frontier a lie), and the anti-degeneracy clause (a method
# that never learned never forgets).
# ---------------------------------------------------------------------------
def test_byte_accounting_is_pinned():
    """PREREG §1 — the floats-per-stored-item table, held to its published values."""
    cfg = _toy_cfg()
    cfg.phi_dim = 32
    per = floats_per_stored_item(cfg, dim=784, n_classes=10)
    # CLU: 32 address + payload + amp + active + 6 controller record scalars
    assert per["clu_entry"] == 41
    assert per["knn_phi_same_keys"] == 41
    # the launder is deliberately CHEAPER per item ⇒ it gets MORE keys than the
    # store gets wells at the same bytes. Under-resourcing it would rig the control.
    assert per["knn_phi_ringbuffer"] == 34
    assert per["er"] == per["gdumb"] == per["icarl"] == 785
    assert per["derpp"] == 795  # + the stored logit vector it distils against
    assert all(per[m] == 0 for m in ("finetune", "ewc", "si", "lwf", "joint"))
    # the landscape-only (secondary) accounting drops the controller's scalars
    cfg.count_controller_record_floats = False
    assert floats_per_stored_item(cfg, 784, 10)["clu_entry"] == 35


def test_matched_byte_budget_converts_to_the_published_item_counts():
    cfg = _toy_cfg()
    cfg.phi_dim = 32
    n = items_for_budget(cfg, 157000, dim=784, n_classes=10)
    assert n["er"] == 200 and n["gdumb"] == 200 and n["icarl"] == 200
    assert n["derpp"] == 197
    assert n["clu_entry"] == 3829  # 19.1x the raw-exemplar item count
    assert n["knn_phi_ringbuffer"] == 4617  # 1.21x MORE keys than CLU has wells
    # nobody may exceed the budget they were given
    per = floats_per_stored_item(cfg, 784, 10)
    for m, cnt in n.items():
        assert cnt * per[m] <= 157000


def test_fixed_state_is_counted_and_the_entry_has_no_backbone():
    """The CLU entry runs zero gradient steps: its only fixed state is φ."""
    cfg = _toy_cfg()
    cfg.phi_dim = 32
    cfg.mlp_width, cfg.mlp_depth, cfg.backbone = 400, 2, "mlp"
    fx = fixed_state_floats(cfg, dim=784, n_classes=10)
    assert fx["er"] == 478410  # 784-400-400-10 MLP
    assert fx["ewc"] == 3 * 478410 and fx["lwf"] == 2 * 478410
    assert fx["clu_entry"] == 32 * 784 + 784  # PCA components + mean
    assert fx["clu_entry"] < fx["er"]


def test_icarl_never_exceeds_its_item_budget():
    """Regression (w26): the old ``max(1, budget // n_classes)`` rule kept ONE
    exemplar per class even when the budget was smaller than the class count, so
    iCaRL silently ran 2.5x over budget at the smallest frontier point."""
    cfg = _toy_cfg()
    cfg.baselines = ["icarl"]
    cfg.memory_items = 4  # < the 10 classes of the stream
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    _, diag = run_baseline_stream("icarl", st, cfg, seed=0)
    assert diag["memory_items"] <= cfg.memory_items
    assert diag["memory_floats"] <= cfg.memory_items * (st["dim"] + 1)


def test_derpp_runs_stays_in_budget_and_stores_logits():
    cfg = _toy_cfg()
    cfg.memory_items = 8
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    A, diag = run_baseline_stream(
        "derpp", st, cfg, seed=0, hyper={"derpp_alpha": 0.5, "derpp_beta": 0.5}
    )
    assert A.shape == (cfg.n_tasks, cfg.n_tasks)
    assert diag["memory_items"] <= cfg.memory_items
    # DER++ pays for the logit vector it distils against — 10 extra floats/item
    assert diag["floats_per_item"] == st["dim"] + 1 + cfg.n_tasks * cfg.classes_per_task


def test_lwf_distils_on_the_current_minibatch():
    """w26 retune: the regularizer is handed the CURRENT minibatch as a third
    argument, instead of one fixed sub-batch stored in ``aux`` and reused for every
    step of the task (the w25 bug that cost LwF ~4 pp against the published value).

    Pins the *contract*, which is what a silent revert would break: ``extra_loss``
    is called as ``f(model, aux, xb)`` with ``xb`` the live minibatch — a two-arg
    regularizer no longer runs at all.
    """
    import jax

    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    shapes = []

    def spy(m, aux, xb):
        shapes.append(tuple(xb.shape))
        return 0.0 * jnp.sum(xb)

    model = make_net(cfg, st["dim"], 10, jax.random.PRNGKey(0))
    mask = np.zeros(10, bool)
    mask[:2] = True
    _train_task(model, st["train_X"][0], st["train_y"][0], mask, cfg,
                jax.random.PRNGKey(1), extra_loss=spy, aux={})
    bs = min(cfg.baseline_batch, len(st["train_X"][0]))
    assert shapes and shapes[0] == (bs, st["dim"])

    with pytest.raises(TypeError):
        _train_task(model, st["train_X"][0], st["train_y"][0], mask, cfg,
                    jax.random.PRNGKey(1), extra_loss=lambda m, aux: 0.0, aux={})

    # and LwF itself still runs end-to-end under the new contract
    A, _ = run_baseline_stream("lwf", st, cfg, seed=0, hyper={"lwf_alpha": 2.0})
    assert A.shape == (cfg.n_tasks, cfg.n_tasks)


def test_constant_predictor_is_the_zero_forgetting_control():
    """A method that never learns never forgets — the frontier carries this line
    so that a low-forgetting number cannot be read as a merit on its own."""
    cfg = _toy_cfg()
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    m = cle.constant_predictor_row(st, cfg)
    assert m["forgetting"] == pytest.approx(0.0, abs=1e-12)
    assert m["BWT"] == pytest.approx(0.0, abs=1e-12)
    assert m["ACC"] < 0.5  # and it is useless, which is the point


def test_byte_frontier_runs_and_holds_every_method_to_the_same_bytes():
    cfg = _toy_cfg()
    cfg.frontier_budgets_floats = [2000, 8000]
    cfg.frontier_seeds = [0]
    cfg.frontier_methods = ["er", "derpp", "icarl", "gdumb"]
    cfg.frontier_fixed_methods = ["finetune"]
    cfg.tune_baselines = False
    fr = cle.run_byte_frontier(cfg, verbose=False, data=_toy_data())
    assert fr["table"] and fr["store_saturation"]
    for r in fr["rows"]:
        if r["budget_floats"] is None:
            continue
        assert r["memory_floats"] <= r["budget_floats"], (
            f"{r['method']} exceeded its byte budget"
        )
    # the launder gets MORE keys than the store gets wells at the same bytes
    for B in cfg.frontier_budgets_floats:
        clu = [r for r in fr["rows"] if r["method"] == "clu_entry"
               and r["budget_floats"] == B][0]
        ring = [r for r in fr["rows"] if r["method"] == "knn_phi_ringbuffer"
                and r["budget_floats"] == B][0]
        assert ring["memory_items"] >= clu["memory_items"]
    v = fr["verdict"]
    assert "reading" in v and "per_budget" in v
    assert set(v["per_budget"]) == {str(b) for b in cfg.frontier_budgets_floats}


def test_frontier_dominance_requires_the_anti_degeneracy_clause():
    """A degenerate method with zero forgetting must NOT be scored as dominant."""
    cfg = _toy_cfg()
    tab = [
        {"budget_floats": 100, "method": "clu_entry", "class": "clu",
         "forgetting": 0.0, "forgetting_sd": 0.0, "ACC": 0.2, "LA": 0.2},
        {"budget_floats": 100, "method": "er", "class": "replay",
         "forgetting": 0.30, "forgetting_sd": 0.01, "ACC": 0.8, "LA": 0.9},
        {"budget_floats": 100, "method": "knn_phi_ringbuffer", "class": "launder",
         "forgetting": 0.25, "forgetting_sd": 0.01, "ACC": 0.75, "LA": 0.85},
    ]
    v = cle.frontier_verdict(tab, cfg)
    assert v["per_budget"]["100"]["la_within_band"] is False
    assert v["dominates_at_budgets"] == []
    # the same numbers with a healthy LA DO count as a dominance region
    for r in tab:
        if r["method"] == "clu_entry":
            r["LA"] = 0.88
    v2 = cle.frontier_verdict(tab, cfg)
    assert v2["dominates_at_budgets"] == ["100"]
    assert "CONTESTED WIN" in v2["reading"]


def test_lwf_ce_scope_is_a_convention_not_a_hyperparameter():
    """⚠ w26 diagnostic: restricting LwF's training cross-entropy to the CURRENT
    task's classes (rather than every class seen so far) is a different loss
    decomposition, not a tuning knob — it moves the Class-IL score far more than
    the whole alpha grid does. The flag exists so the measurement is reproducible;
    the DEFAULT stays on the convention under which EWC/SI/finetune reproduce their
    published Split-MNIST values."""
    cfg = _toy_cfg()
    assert cfg.lwf_ce_scope == "seen"
    st = cle.build_cl_stream(cfg, seed=0, data=_toy_data())
    A_seen, _ = run_baseline_stream("lwf", st, cfg, seed=0, hyper={"lwf_alpha": 1.0})
    cfg.lwf_ce_scope = "current_task"
    A_cur, _ = run_baseline_stream("lwf", st, cfg, seed=0, hyper={"lwf_alpha": 1.0})
    assert not np.allclose(A_seen, A_cur), "the flag must actually change training"
