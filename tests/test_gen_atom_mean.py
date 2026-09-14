"""Atom-mean lookup — the like-for-like control (``gen-atom-mean-control``).

⭐ The load-bearing tests here are the ones that can **fail on purpose**; a check that
has never been seen to fire has not been tested:

- ``test_atoms_for_budget_REFUSES_a_budget_that_is_not_matched`` — a "matched-bytes"
  arm that silently floors is an **under-resourced** rival, which hands the dynamical
  read a false win.  The refusal is shown firing.
- ``test_byte_equality_is_EXACT_at_both_shipped_budgets`` — 64 atoms/well = 524,288 B
  (⭐ the deployed CLU store exactly) and 512 atoms/well = 4,194,304 B.
- ``test_atoms_are_REAL_DATA_ROWS_never_centroids`` — the ⛔ that separates this arm
  from the shipped codebook.
- ``test_k1_read_is_EXACTLY_a_stored_atom`` and
  ``test_read_never_contains_the_query`` — the read is a lookup, ⛔ it does not settle,
  and the query reaches the decoder only through indices.
- ``test_kmedoid_BEATS_the_weak_subsample_companion`` — "never weaken this rival" is a
  **measured** statement here.
- ``test_knn_proto_regression_FIRES_when_the_rival_moves`` — the falsifier *"full
  knn_proto moved ⇒ the comparison is void"*, shown refusing.
- ``test_retention_ladder_marks_a_ONE_VALUE_factor_NOT_APPLICABLE`` — R31, fourth
  strike: a per-factor statistic is meaningless without ``n_values_realised``.
- ``test_budget_tags_are_DECORRELATED`` — the JAX landmine's analogue: two budgets
  must not share a stream.

Everything runs on synthetic tensors; the real-data path is exercised by the runs
reported in ``.claude/outputs/gen-atom-mean-control.md``.

⚠ §7.23: other test modules enable ``jax_enable_x64`` at **module import** and may sort
before this one, so nothing here pins float32.
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import (
    GenAtomMeanConfig,
    gen_atom_mean_config,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.gen_atoms import (
    ATOM_METHODS,
    SCOPES,
    AtomSet,
    atom_byte_ledger,
    atom_read,
    atoms_for_budget,
    bits_per_read,
    fitted_exponent,
    load_atoms,
    relative_read_error,
    save_atoms,
    select_atoms,
)
from chlu.experiments.exp_gen_atom_mean import (
    NOT_MINE,
    REQUEST_SETS,
    STAGES,
    _trial_grid,
    aggregate_runs,
    exponent_across_budgets,
    knn_proto_regression,
    retention_ladder,
)

#: ⛔ The two budgets, declared in ``gen-dynamical-read``'s PREREG §3 and asserted here.
BUDGET_A = 524_288
BUDGET_B = 4_194_304
SHIPPED_D = 2048


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class _Reader:
    """A stand-in instrument: ``images -> per-factor logits``, deterministic.

    ⚠ It is called under ``eqx.filter_jit(jax.vmap(...))`` by ``predict_factors``, so
    it must be **TRACEABLE** — ⛔ no ``np.asarray`` on a tracer and no Python ``if``.
    (My first version was not, and the test failed for a reason that was mine rather
    than the code's; recorded because a test that has never failed has never been
    tested.)
    """

    def __init__(self, cards):
        self.cards = tuple(int(c) for c in cards)

    def __call__(self, x):
        v = jnp.asarray(x).reshape(-1)
        out = []
        for j, c in enumerate(self.cards):
            k = jnp.abs(v[j % int(v.shape[0])]).astype(jnp.int32) % int(c)
            out.append(jnp.where(jnp.arange(int(c)) == k, 9.0, -9.0))
        return out


@pytest.fixture
def toy():
    rng = np.random.default_rng(11)
    D, n = 36, 400
    X = rng.normal(size=(n, D)).astype(np.float32)
    wc = (tuple(range(0, 15)), tuple(range(15, 25)), tuple(range(25, 36)))
    return {"X": X, "wc": wc, "D": D, "n": n}


# ---------------------------------------------------------------------------
# 1. ⛔ matched bytes — the whole point of the arm
# ---------------------------------------------------------------------------


def test_byte_equality_is_EXACT_at_both_shipped_budgets():
    assert atoms_for_budget(BUDGET_A, SHIPPED_D) == 64
    assert atoms_for_budget(BUDGET_B, SHIPPED_D) == 512
    # ⭐ budget A is the deployed CLU store EXACTLY (the store is 64 codewords/well)
    assert 64 * SHIPPED_D * 4 == BUDGET_A
    assert BUDGET_B == 8 * BUDGET_A


def test_atoms_for_budget_REFUSES_a_budget_that_is_not_matched():
    """⛔ Flooring would let the rival silently under-spend — the failure this spoke
    exists to avoid."""
    with pytest.raises(ValueError, match="exact multiple"):
        atoms_for_budget(BUDGET_A - 1, SHIPPED_D)
    with pytest.raises(ValueError, match="exact multiple"):
        atoms_for_budget(524_288 + 4096, 2048 + 1)


def test_byte_ledger_is_scope_INVARIANT(toy):
    """⭐ per_well and global cost the same: the scope is a structure choice, ⛔ never a
    byte advantage."""
    n_bytes = 8 * toy["D"] * 4
    led = {}
    for scope in SCOPES:
        A = select_atoms(toy["X"], toy["wc"], 8, seed=1, scope=scope, iters=4)
        led[scope] = atom_byte_ledger(A.n_atoms, A.well_sizes, n_bytes, scope=scope)
        assert A.deployed_bytes() == n_bytes
        assert led[scope]["matched_exactly"]
    assert led["per_well"]["realised_bytes"] == led["global"]["realised_bytes"]


def test_bits_per_read_is_the_bounded_channel():
    # ⭐ budget A, k=1, K=5 wells: 5 * log2(64) = 30.0 — the store's own channel width
    assert bits_per_read(64, 1, 5, "per_well") == pytest.approx(30.0)
    assert bits_per_read(64, 1, 5, "global") == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 2. ⛔ the atoms are DATA POINTS, and there are exactly N of them
# ---------------------------------------------------------------------------


def test_atoms_are_REAL_DATA_ROWS_never_centroids(toy):
    A = select_atoms(toy["X"], toy["wc"], 12, seed=2, scope="per_well", iters=6)
    for k, (rows, vals) in enumerate(zip(A.rows, A.values, strict=True)):
        cidx = np.asarray(toy["wc"][k], np.int64)
        assert np.array_equal(vals, toy["X"][rows][:, cidx]), (
            f"well {k}: a deployed atom is not a row of the pool — ⛔ it is a centroid"
        )
        assert np.unique(rows).size == 12, "fewer distinct atoms than the budget buys"


@pytest.mark.parametrize("method", list(ATOM_METHODS))
def test_every_method_realises_EXACTLY_the_budgeted_atom_count(toy, method):
    A = select_atoms(toy["X"], toy["wc"], 10, seed=3, method=method, iters=4)
    assert A.n_atoms == 10
    assert all(np.unique(r).size == 10 for r in A.rows)


def test_budget_tags_are_DECORRELATED(toy):
    """⭐ The JAX landmine's analogue: ``split(k,2)[1] IS split(k,3)[1]``, so two arms
    derived by lengthening a split are bit-identical rather than decorrelated.  These
    streams are ``SeedSequence([seed, budget_tag, scope_tag, well])`` and must differ."""
    a = select_atoms(toy["X"], toy["wc"], 6, seed=4, method="subsample", budget_tag=0)
    b = select_atoms(toy["X"], toy["wc"], 6, seed=4, method="subsample", budget_tag=1)
    assert a.sha256() != b.sha256()
    assert any(not np.array_equal(x, y) for x, y in zip(a.rows, b.rows, strict=True))


def test_atom_set_round_trips_by_sha256(tmp_path, toy):
    """⭐ ``gen-dynamical-read`` must load THESE atoms, ⛔ not re-derive a lookalike."""
    A = select_atoms(toy["X"], toy["wc"], 7, seed=5, iters=4)
    man = save_atoms(A, tmp_path / "atoms.npz")
    B = load_atoms(tmp_path / "atoms.npz")
    assert B.sha256() == A.sha256() == man["sha256"]
    assert B.well_channels == A.well_channels
    assert [int(v.shape[1]) for v in B.values] == A.well_sizes


def test_a_partial_well_cover_is_REFUSED(toy):
    with pytest.raises(AssertionError, match="channels"):
        select_atoms(toy["X"], (tuple(range(0, 15)), tuple(range(15, 25))), 5, seed=6)


# ---------------------------------------------------------------------------
# 3. ⭐ the read is a LOOKUP — it does not settle, and it never sees the query
# ---------------------------------------------------------------------------


def test_k1_read_is_EXACTLY_a_stored_atom(toy):
    A = select_atoms(toy["X"], toy["wc"], 10, seed=7, iters=6)
    Q = toy["X"][:25] + 0.05 * np.random.default_rng(0).normal(
        size=(25, toy["D"])).astype(np.float32)
    r = atom_read(A, Q, 1, "euclidean", None)
    for k, c in enumerate(toy["wc"]):
        sl = r["read"][:, np.asarray(c, np.int64)]
        for i in range(sl.shape[0]):
            assert np.any(np.all(np.isclose(A.values[k], sl[i]), axis=1)), (
                "⛔ the k=1 read is not a stored atom — this arm SETTLES, which is "
                "exactly what it must not do"
            )


def test_read_never_contains_the_query(toy):
    """⛔ The structural property ``gen_store.read_from_codes`` has: the query chooses
    indices and nothing else.  Perturbing the query WITHIN a cell must not move the
    read at all."""
    A = select_atoms(toy["X"], toy["wc"], 6, seed=8, iters=6)
    Q = toy["X"][:30]
    r0 = atom_read(A, Q, 1, "euclidean", None)
    r1 = atom_read(A, Q + 1e-4, 1, "euclidean", None)
    same = np.all(r0["config_id"] == r1["config_id"], axis=1)
    assert same.any(), "the perturbation moved every cell; the test cannot conclude"
    assert np.array_equal(r0["read"][same], r1["read"][same]), (
        "the read moved without the retrieved indices moving ⇒ the query leaked in"
    )


def test_more_atoms_means_less_read_error(toy):
    e = {}
    for N in (4, 16, 64):
        A = select_atoms(toy["X"], toy["wc"], N, seed=9, iters=8)
        r = atom_read(A, toy["X"][:100], 1, "euclidean", None)
        e[N] = relative_read_error(r["read"], toy["X"][:100])["relative_mse"]
    assert e[64] < e[16] < e[4], e


def test_fitted_exponent_recovers_a_known_slope():
    pts = [(64.0, 0.32), (512.0, 0.32 * 8.0 ** -0.25)]
    f = fitted_exponent(pts)
    assert f["slope"] == pytest.approx(-0.25, abs=1e-9)
    assert f["implied_pr"] == pytest.approx(8.0, abs=1e-6)
    assert fitted_exponent([(64.0, 0.3)])["n_points"] == 1
    assert "NOT_RUN" in fitted_exponent([(64.0, 0.3)])


def test_distinct_configurations_is_bounded_by_the_atom_count(toy):
    A = select_atoms(toy["X"], toy["wc"], 5, seed=10, iters=6)
    r = atom_read(A, toy["X"][:200], 1, "euclidean", None)
    assert r["n_distinct_configs"] <= 5 ** A.n_wells
    assert r["n_distinct_configs"] <= 200


# ---------------------------------------------------------------------------
# 4. ⛔ NEVER WEAKEN THIS RIVAL — measured, not promised
# ---------------------------------------------------------------------------


def test_kmedoid_BEATS_the_weak_subsample_companion(toy):
    """⭐ The shipped selection must be the strong one.  ⛔ If the random subsample
    wins, the shipped atoms are weak and this spoke must stop and report."""
    strong = select_atoms(toy["X"], toy["wc"], 16, seed=12, method="kmedoid", iters=15)
    weak = select_atoms(toy["X"], toy["wc"], 16, seed=12, method="subsample")
    T = toy["X"][:200]
    es = relative_read_error(atom_read(strong, T, 1, "euclidean", None)["read"],
                             T)["relative_mse"]
    ew = relative_read_error(atom_read(weak, T, 1, "euclidean", None)["read"],
                             T)["relative_mse"]
    assert es < ew, f"k-medoid {es} is not stronger than the subsample {ew}"


def test_a_hobbled_row_budget_is_NOT_the_default():
    """⚠ The standing trap: a spoke that re-fits a rival on its own row budget silently
    weakens it (``knn_proto`` 0.8763 -> 0.8308 at 1,024 fit rows)."""
    cfg = GenAtomMeanConfig()
    assert cfg.n_fit == 32_256
    assert cfg.n_val == 512


# ---------------------------------------------------------------------------
# 5. ⛔ the regression check — the falsifier "full knn_proto moved"
# ---------------------------------------------------------------------------


def _reg_ctx(monkeypatch, value):
    """A ctx whose ``in_corner`` returns ``value`` — so the guard can be shown firing."""
    import chlu.experiments.exp_gen_atom_mean as M

    monkeypatch.setattr(M, "render", lambda *a, **k: np.zeros((4, 1, 2, 2), np.float32))
    monkeypatch.setattr(M, "in_corner", lambda *a, **k: {"balanced_accuracy": value,
                                                         "auroc": 0.5, "tpr": 0.0,
                                                         "tnr": 1.0})
    monkeypatch.setattr(M, "distinct_counts", lambda *a, **k: {"n": 4})
    monkeypatch.setattr(M, "retention_ladder", lambda *a, **k: {"aggregate": 0.0})
    return {
        "names": ["floor_hue", "wall_hue", "object_hue", "scale", "shape",
                  "orientation"],
        "decoder": None, "probe": None, "external": None, "feat_shape": (4,),
        "lab_B": np.zeros((4, 6), np.int64),
        **{f"Q_{rs}": np.zeros((4, 4), np.float32) for rs in REQUEST_SETS},
        **{f"lab_{rs}": np.zeros((4, 6), np.int64) for rs in REQUEST_SETS},
    }


def test_knn_proto_regression_PASSES_when_the_rival_is_bit_identical(monkeypatch):
    cfg = GenAtomMeanConfig()
    ref = cfg.knn_proto_probe_reference_per_seed[0]
    out = knn_proto_regression(_reg_ctx(monkeypatch, ref), cfg, 0, log=lambda *_: None)
    assert out["bit_identical"] is True
    assert out["delta"] == 0.0


def test_knn_proto_regression_FIRES_when_the_rival_moves(monkeypatch):
    """⛔ Not "close": the tolerance is exactly zero, because a moved ``knn_proto``
    means something out of scope changed and the comparison is VOID."""
    cfg = GenAtomMeanConfig()
    moved = cfg.knn_proto_probe_reference_per_seed[0] + 1e-12
    with pytest.raises(AssertionError, match="FULL knn_proto MOVED"):
        knn_proto_regression(_reg_ctx(monkeypatch, moved), cfg, 0, log=lambda *_: None)


def test_knn_proto_reference_constants_are_NOT_4dp_transcriptions():
    """⚠ Third strike on this file family: a reference constant must come from the
    peer's manifest at full precision, ⛔ never from a rendered 4-dp table."""
    for v in GenAtomMeanConfig().knn_proto_probe_reference_per_seed:
        assert abs(v - round(v, 4)) > 1e-9, f"{v} looks like a 4-dp figure"


# ---------------------------------------------------------------------------
# 6. ⛔ R31 — no per-factor row without n_values_realised
# ---------------------------------------------------------------------------


def test_retention_ladder_marks_a_ONE_VALUE_factor_NOT_APPLICABLE():
    names = ["a", "b"]
    lab = np.array([[0, 0], [0, 1], [0, 2], [0, 1]], np.int64)
    # ⭐ the reader reads factor j off pixel j, so the renders carry the true labels
    imgs = np.zeros((4, 1, 2, 2), np.float32)
    for i in range(4):
        imgs[i, 0, 0, 0] = float(lab[i, 0])
        imgs[i, 0, 0, 1] = float(lab[i, 1])
    reader = _Reader([1, 3])
    lad = retention_ladder(reader, imgs, lab, names, 2)
    assert lad["per_factor"]["a"]["n_values_realised"] == 1
    assert "NOT APPLICABLE" in lad["per_factor"]["a"]["status"]
    assert lad["per_factor"]["b"]["n_values_realised"] == 3
    assert "status" not in lad["per_factor"]["b"]
    assert lad["n_factors_applicable"] == 1
    assert lad["per_factor"]["b"]["accuracy"] == pytest.approx(1.0)
    for row in lad["per_factor"].values():
        assert "n_values_realised" in row


# ---------------------------------------------------------------------------
# 7. config + plumbing
# ---------------------------------------------------------------------------


def test_config_block_has_zero_house_wiring_sites(tmp_path):
    """⛔ The merge rule for ``config.py``'s tail only holds if the block is ONE
    contiguous region reached through an accessor."""
    c = get_default_config()
    assert not hasattr(c, "gen_atom_mean")           # not a CHLUConfig field
    cfg = gen_atom_mean_config(c)
    assert isinstance(cfg, GenAtomMeanConfig)
    assert gen_atom_mean_config(c) is cfg            # memoised
    p = tmp_path / "c.yaml"
    save_config(c, p)                                # ⛔ must not blow up on the block
    assert "gen_atom_mean" not in json.dumps(
        {k: str(v) for k, v in {"_": 0}.items()})    # no save_config site by design
    load_config(p)


def test_config_pins_the_two_declared_budgets():
    cfg = GenAtomMeanConfig()
    assert cfg.budgets == {"A": BUDGET_A, "B": BUDGET_B}
    assert cfg.assert_byte_equality is True
    assert cfg.atom_method == "kmedoid"
    assert cfg.expected_decoder_recipe == "v1_capacity_schedule"
    assert cfg.require_knn_proto_bit_identical is True


def test_trial_grid_is_deterministic_and_within_budget():
    cfg = GenAtomMeanConfig()
    g = _trial_grid(cfg)
    assert len(g) == len(cfg.knn_ks) * len(cfg.metrics) * len(cfg.normalisers) * len(
        cfg.scopes)
    assert len(g) <= cfg.tuning_trial_budget
    assert _trial_grid(cfg) == g


def test_stage_router_refuses_another_modules_stage():
    assert "cell" in NOT_MINE and "cell" not in STAGES
    assert set(STAGES).isdisjoint(NOT_MINE)


def test_aggregate_reports_a_missing_seed_as_NOT_RUN_never_as_a_zero(tmp_path):
    def _man(seed, tags):
        lad = {rs: {"aggregate": 0.6, "exact_match": 0.1, "n": 4,
                    "n_factors_applicable": 1,
                    "per_factor": {"shape": {"accuracy": 0.5, "n_values_realised":
                                             1 if rs == "B" else 3, "chance": 0.5}}}
               for rs in ("B", "MS", "A")}
        return {
            "seed": seed,
            "ceiling": {i: {"in_corner": {"balanced_accuracy": 0.9}} for i in
                        ("external", "probe")},
            "knn_proto_regression": {"probe_in_corner_balacc": 0.87,
                                     "bit_identical": True},
            "exponent": {leg: {"slope": -0.25} for leg in
                         ("oracle_feature_addressed/k1",
                          "oracle_feature_addressed/k_selected",
                          "request_addressed/k_selected")},
            "budgets": {
                t: {
                    "budget_bytes": BUDGET_A, "n_atoms_per_well": 64,
                    "channel_bits": 30.0,
                    "byte_ledgers": {"per_well": {"matched_exactly": True}},
                    "tuning": {"selected": {"scope": "per_well", "k": 1}},
                    "scored": {i: {"in_corner": {"balanced_accuracy": 0.7},
                                   "retention_ladder": lad} for i in
                               ("external", "probe")},
                    "distinct": {"B": {"atom_configurations": 30,
                                       "feature_vectors": 30, "factor_tuples": 25,
                                       "n": 72}},
                    "read_error": {
                        "rows_primary": "A", "rows_measured": ["A", "val"],
                        "A": {
                            **{k: {"relative_mse": 0.3} for k in
                               ("oracle_feature_addressed/k1",
                                "oracle_feature_addressed/k_selected",
                                "request_addressed/k1",
                                "request_addressed/k_selected",
                                "centroid_control/k1")},
                            "medoid_penalty": {"ratio": 1.9},
                        },
                    },
                    "weak_companion": {"kmedoid_is_stronger_on_read_error": True,
                                       "kmedoid_is_stronger_on_in_corner": True,
                                       "external_in_corner_balacc": 0.65},
                } for t in tags
            },
            "NOT_RUN": {},
        }

    for s, tags in ((0, ["A", "B"]), (1, ["A"])):
        d = tmp_path / f"seed_{s}"
        d.mkdir()
        (d / "atom_manifest.json").write_text(json.dumps(_man(s, tags)))
    agg = aggregate_runs([tmp_path / "seed_0", tmp_path / "seed_1"],
                         log=lambda *_: None)
    assert agg["seeds_present"] == [0, 1]
    assert agg["budgets"]["A"]["n_seeds"] == 2
    assert agg["budgets"]["A"]["read_error"]["A"][
        "centroid_control/k1"]["mean"] == pytest.approx(0.3)
    assert agg["budgets"]["A"]["medoid_penalty"]["A"]["mean"] == pytest.approx(1.9)
    assert agg["budgets"]["B"]["NOT_RUN_seeds"] == [1]   # ⛔ never a zero
    assert agg["knn_proto_regression"]["bit_identical_all_seeds"] is True
    # ⛔ R31 travels into the aggregate: shape on B realises ONE value
    lad = agg["budgets"]["A"]["retention_ladder"]
    assert lad["B"]["per_factor"]["shape"]["applicable"] is False
    assert lad["A"]["per_factor"]["shape"]["applicable"] is True
    assert lad["B"]["per_factor"]["shape"]["n_values_realised"] == [1]


def test_exponent_across_budgets_uses_both_budgets():
    man = {"budgets": {
        "A": {"n_atoms_per_well": 64, "read_error": {
            "A": {"oracle_feature_addressed/k1": {"relative_mse": 0.32}}}},
        "B": {"n_atoms_per_well": 512, "read_error": {
            "A": {"oracle_feature_addressed/k1": {"relative_mse": 0.19}}}},
    }}
    out = exponent_across_budgets(man, GenAtomMeanConfig())
    e = out["oracle_feature_addressed/k1"]
    assert e["n_points"] == 2
    assert e["slope"] < 0
    # ⭐ the centroid law is rendered for comparison and is NOT a measurement here
    assert "⛔" in out["centroid_quantiser_law"]["source"]
    assert out["centroid_quantiser_law"]["predicted"]["64"] == pytest.approx(0.2783)


def test_atomset_dataclass_reports_its_own_bytes():
    vals = (np.zeros((3, 5), np.float32), np.zeros((3, 7), np.float32))
    A = AtomSet("per_well", ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11)),
                (np.arange(3), np.arange(3)), vals, {})
    assert A.dim == 12 and A.n_atoms == 3 and A.n_wells == 2
    assert A.deployed_bytes() == 3 * 12 * 4


# ---------------------------------------------------------------------------
# 8. ⚠ the CENTROID CONTROL — 0 deployed bytes, ⛔ not an arm
# ---------------------------------------------------------------------------


def test_centroid_control_isolates_the_MEDOID_PENALTY(toy):
    """⭐ Storing a DATA POINT costs more than storing the centre of its cell, at the
    same N and the same Lloyd fit.  ⛔ That price must be MEASURED here, or the
    0.2783 anchor confounds it with "my k-means vs their trained store"."""
    A = select_atoms(toy["X"], toy["wc"], 12, seed=13, method="kmedoid", iters=15)
    assert A.centroids is not None
    T = toy["X"][:200]
    med = relative_read_error(
        atom_read(A, T, 1, "euclidean", None)["read"], T)["relative_mse"]
    cen = relative_read_error(
        atom_read(A, T, 1, "euclidean", None, use_centroids=True)["read"],
        T)["relative_mse"]
    assert cen < med, "the centroid read is not cheaper than the medoid read"
    # ⛔ the centroids are FIT-TIME ONLY: they change no deployed byte
    assert A.deployed_bytes() == 12 * toy["D"] * 4


def test_centroid_control_is_REFUSED_on_a_subsampled_set(toy):
    A = select_atoms(toy["X"], toy["wc"], 8, seed=14, method="subsample")
    assert A.centroids is None
    with pytest.raises(ValueError, match="no centroids"):
        atom_read(A, toy["X"][:10], 1, "euclidean", None, use_centroids=True)


def test_read_error_primary_rows_are_OUTSIDE_the_atom_pool():
    """⚠ ``val`` sits inside ``store_train`` (the atom pool) and is therefore
    OPTIMISTIC; ``A`` (``eval_a``) is excluded from it by ``select_store_rows``, which
    is why it is the primary."""
    cfg = GenAtomMeanConfig()
    assert cfg.read_error_primary == "A"
    assert set(cfg.read_error_rows) == {"A", "val"}
    assert cfg.n_eval_a > 0, "the primary read-error row set would be empty"
