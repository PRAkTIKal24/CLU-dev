"""A7 (the term-count control) and the NO-MODEL DETECTOR that is now a suite test.

Why this file exists
--------------------
``build_c_invalid`` makes a ``duplicate_factor`` record by appending a seventh term
to a six-term request.  ``gen-certificate`` measured the consequence: a function that
does nothing but **count terms** scores AUROC **1.0000** on C[dup]-vs-B and **0.7500**
pooled, with no model in it, and every arm in the generative cell inherited that
number invisibly.  The cause was not the C set; it was that the C spec never required
C records to be *structurally indistinguishable* from the valid ones.

R26 ruling 1 made the no-model detector a standing rule.  A convention would not have
helped -- "the convention that has to be remembered is the one that failed" -- so the
detector lives here, and three things make it inherit future query-set definitions
instead of restating today's bug:

1. it runs on the **registered artifact discovered from disk** (``query_sets.json``),
   never on a hardcoded list of sets;
2. every *invalid kind* must name the valid set that is its structural control, and a
   kind without one **fails**, so a new C kind cannot ship uncontrolled;
3. any record key outside the declared allowlist **fails**, so a new field cannot
   become the next giveaway silently.

The detector may use every field except the factor **values** (which are masked before
a single feature is computed): term count, term ordering, the repeated-factor pattern
and its position, factor-index arithmetic, and the length of the masked serialisation.
"""

import json

import numpy as np
import pytest

from chlu.cli.data_cmd import _build_gen_artifacts
from chlu.config import GenDataConfig
from chlu.data import gen_splits as G

# ---------------------------------------------------------------------------
# THE TOLERANCE, declared here (R26: "tolerance declared in the test")
# ---------------------------------------------------------------------------
# Every feature below is a small integer statistic of the term list, so a matched
# control makes each one tie EXACTLY: the pre-registered prediction was 0.5000, and
# the realised value on the shipped Shapes3D and dSprites artifacts is 0.5000 on all
# 11 features, per kind and pooled.  The 0.05 is therefore pure headroom for a future
# query set whose control is only statistically matched (e.g. a sampled control),
# not a budget today's sets spend.  ⛔ Raising it is not a way to make this test pass.
DETECTOR_TOLERANCE = 0.05

CARDS = (10, 10, 10, 8, 4, 15)


def _grid(cards=CARDS, dataset="toy"):
    return G.FactorGrid(
        dataset=dataset,
        names=tuple(f"f{i}" for i in range(len(cards))),
        values=tuple(np.arange(c, dtype=float) for c in cards),
    )


def _split(grid):
    """A toy split of the grid, so the production emitter can be called as-is."""
    cells = [G.FactorCell("c0", {grid.names[0]: (float(grid.values[0][0]),)})]
    return G.build_cell_split(grid, cells, grid.dataset, "toy_cond", "leave1out")


def _emit(tmp_path, n_per_kind=36, **cfg_kw):
    """Run the PRODUCTION emitter (``chlu data <dataset>``'s inner function)."""
    grid = _grid()
    split = _split(grid)
    cfg = GenDataConfig(c_invalid_n_per_kind=n_per_kind, **cfg_kw)
    lines = _build_gen_artifacts(None, split, cfg, tmp_path)
    return grid, split, lines


# ---------------------------------------------------------------------------
# 1. A7 itself
# ---------------------------------------------------------------------------


def test_a7_is_one_term_longer_and_still_satisfiable():
    grid = _grid()
    rows = G.select_a7_rows(np.arange(4096), 72, seed=0)
    a7 = G.build_a7_redundant(grid, rows, seed=0)
    assert G.validate_a7_redundant(a7, grid) == 72
    for rec in a7["records"]:
        assert len(rec["request"]) == grid.n_factors + 1
        # the repeated factor is repeated at the value it ALREADY has
        by_factor = {}
        for t in rec["request"]:
            by_factor.setdefault(t["factor"], set()).add(t["value"])
        assert all(len(v) == 1 for v in by_factor.values())
        assert len(by_factor) == grid.n_factors


def test_a7_denotes_the_same_image_so_it_cannot_be_a_c_record():
    """⭐ The supervision claim, mechanically: A7 is built from valid requests alone.

    Every A7 record is *satisfiable*, so the C validator -- the function that decides
    what counts as structurally invalid -- rejects all of them.  A control built from
    (or fitted to) an invalid example would not survive this.
    """
    grid = _grid()
    a7 = G.build_a7_redundant(grid, G.select_a7_rows(np.arange(4096), 12, seed=0))
    payload = {"records": [dict(r, kind="duplicate_factor") for r in a7["records"]]}
    with pytest.raises(AssertionError, match="claims duplicate_factor but has none"):
        G.validate_c_invalid(payload, grid)


def test_a7_is_deterministic_and_one_per_valid_request():
    """The convention is ``|A7| = |A|``: the construction is a 1:1 map."""
    grid = _grid()
    rows = G.select_a7_rows(np.arange(4096), 30, seed=3)
    a, b = G.build_a7_redundant(grid, rows), G.build_a7_redundant(grid, rows)
    assert a == b
    assert len(a["records"]) == len(rows) == 30
    assert [r["row"] for r in a["records"]] == rows.tolist()
    assert G.select_a7_rows(np.arange(4096), 30, seed=3).tolist() == rows.tolist()


def test_a7_never_spends_a_held_out_row():
    """B is the evaluation set; a control built from it would leak the hold-out."""
    grid = _grid()
    split = _split(grid)
    rows = G.select_a7_rows(split.train_rows, 72, seed=0)
    assert not np.intersect1d(rows, split.test_rows).size


def test_the_on_disk_a7_matches_the_in_memory_rule_used_by_the_certificate():
    """⛔ Two implementations of one rule must not drift.

    ``chlu.core.gen_certificate.redundant_requests`` builds A7 in memory from the
    certificate's own A; this module builds the shipped artifact.  If they ever
    disagree, the file and the run mean different things by "A7".
    """
    from chlu.core.gen_certificate import redundant_requests, requests_from_rows

    grid = _grid()
    rows = np.arange(24)
    mem = redundant_requests(requests_from_rows(grid, rows, "A"), grid.n_factors)
    disk = G.build_a7_redundant(grid, rows)["records"]
    assert len(mem) == len(disk) == len(rows)
    for m, d in zip(mem, disk, strict=True):
        assert m.terms == tuple(
            (int(t["factor_index"]), float(t["value"])) for t in d["request"]
        )


# ---------------------------------------------------------------------------
# 2. The detector's own machinery
# ---------------------------------------------------------------------------


def test_auroc_ties_give_exactly_one_half_and_separation_gives_one():
    assert G.auroc([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert G.auroc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert G.auroc([0.0, 1.0], [2.0, 3.0]) == 0.0
    # one tied pair (0.5), one win, two losses -> (0.5 + 1 + 0 + 0) / 4
    assert G.auroc([1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.375)
    with pytest.raises(ValueError, match="non-empty class"):
        G.auroc([], [1.0])


def test_values_are_masked_before_a_single_feature_is_computed():
    grid = _grid()
    rec = G.build_base_requests(grid, [12345])[0]
    assert {t["value"] for t in G.mask_request(rec["request"])} == {G.MASKED_VALUE}
    other = G.build_base_requests(grid, [54321])[0]
    assert G.structural_features(rec) == G.structural_features(other)


def test_the_detector_reaches_more_than_the_term_count():
    """⭐ Being adversarial is the point: the next giveaway will not be term count."""
    feats = set(G.structural_features(G.build_base_requests(_grid(), [0])[0]))
    assert {
        "n_terms",
        "n_distinct_factors",
        "max_factor_multiplicity",
        "n_redundant_terms",
        "is_factor_order_sorted",
        "n_order_inversions",
        "first_duplicate_position",
        "sum_factor_indices",
        "first_factor_index",
        "last_factor_index",
        "masked_request_json_len",
    } <= feats


def test_the_definitional_giveaway_is_reported_not_hidden(tmp_path):
    """⭐ The residual A7 cannot close, measured and named rather than left implicit.

    ``duplicate_factor`` means "one factor, two different values", which is decidable
    from the request by **value equality alone** -- no values, no grid, no model.  It
    is the dup counterpart of the grid lookup that decides ``out_of_range``.  ⛔ It is
    not a structural leak and no control removes it: it is what makes the record
    invalid.  The claim it leaves standing is the one the cell actually makes -- that
    the barrier is computed from the store's geometry -- so the number is reported.
    """
    _emit(tmp_path)
    report = G.audit_query_sets(
        tmp_path / "query_sets.json", tolerance=DETECTOR_TOLERANCE
    )
    dup = report["definitional_vs_base"]["c_invalid[duplicate_factor]"]
    oor = report["definitional_vs_base"]["c_invalid[out_of_range]"]
    assert dup["has_conflicting_repeat"] == 1.0  # decidable, by definition
    assert oor["has_conflicting_repeat"] == 0.5  # oor's counterpart needs the grid
    # ⛔ and it does NOT reach the asserted view, which stays at chance
    assert report["per_kind"]["c_invalid[duplicate_factor]"][
        "max_auroc"
    ] == pytest.approx(0.5)
    assert not report["violations"]


# ---------------------------------------------------------------------------
# 3. The detector is CAPABLE OF FAILING -- the positive control
# ---------------------------------------------------------------------------


def test_the_detector_catches_the_leak_it_was_built_for(tmp_path):
    """⭐ Point the dup kind's control back at the uncontrolled pool: it must FAIL.

    This is the bug as it shipped.  If this test ever stops raising, the detector has
    become decoration.
    """
    _emit(tmp_path)
    reg = tmp_path / "query_sets.json"
    payload = json.loads(reg.read_text())
    payload["sets"]["c_invalid"]["controls"]["duplicate_factor"] = "base"
    reg.write_text(json.dumps(payload))

    report = G.audit_query_sets(reg, tolerance=DETECTOR_TOLERANCE)
    dup = report["per_kind"]["c_invalid[duplicate_factor]"]
    assert dup["max_auroc"] == 1.0
    assert dup["features"]["n_terms"] == 1.0
    assert report["pooled"]["c_invalid"]["max_auroc"] == pytest.approx(0.75)
    with pytest.raises(AssertionError, match="no-model detector audit FAILED"):
        G.assert_no_model_detector_at_chance(reg, tolerance=DETECTOR_TOLERANCE)


def test_an_invalid_kind_with_no_declared_control_fails_loudly(tmp_path):
    """⭐ THE inheritance mechanism: a NEW C kind cannot ship uncontrolled."""
    _emit(tmp_path)
    c = json.loads((tmp_path / "c_invalid.json").read_text())
    c["records"][0]["kind"] = "some_future_kind"
    (tmp_path / "c_invalid.json").write_text(json.dumps(c))
    with pytest.raises(AssertionError, match="some_future_kind.*NO declared"):
        G.assert_no_model_detector_at_chance(
            tmp_path / "query_sets.json", tolerance=DETECTOR_TOLERANCE
        )


def test_a_new_record_field_fails_loudly(tmp_path):
    """A field nobody declared is a candidate giveaway; it must be declared first."""
    _emit(tmp_path)
    c = json.loads((tmp_path / "c_invalid.json").read_text())
    for rec in c["records"]:
        rec["difficulty"] = 3
    (tmp_path / "c_invalid.json").write_text(json.dumps(c))
    with pytest.raises(AssertionError, match="difficulty.*allowlist"):
        G.assert_no_model_detector_at_chance(
            tmp_path / "query_sets.json", tolerance=DETECTOR_TOLERANCE
        )


def test_term_key_order_is_single_sourced(tmp_path):
    """Key ORDER is a structural property too -- so it has exactly one definition."""
    _emit(tmp_path)
    c = json.loads((tmp_path / "c_invalid.json").read_text())
    for rec in c["records"]:
        rec["request"] = [
            {
                "value": t["value"],
                "factor": t["factor"],
                "factor_index": t["factor_index"],
            }
            for t in rec["request"]
        ]
    (tmp_path / "c_invalid.json").write_text(json.dumps(c))
    with pytest.raises(AssertionError, match="key order is itself a giveaway"):
        G.assert_no_model_detector_at_chance(
            tmp_path / "query_sets.json", tolerance=DETECTOR_TOLERANCE
        )


def test_a_registered_set_missing_from_disk_is_not_a_silent_pass(tmp_path):
    _emit(tmp_path)
    (tmp_path / "a7_redundant.json").unlink()
    with pytest.raises(FileNotFoundError, match="registered at"):
        G.audit_query_sets(tmp_path / "query_sets.json", tolerance=DETECTOR_TOLERANCE)


# ---------------------------------------------------------------------------
# 4. The emitter, end to end
# ---------------------------------------------------------------------------


def test_the_emitter_registers_a7_beside_the_c_set(tmp_path):
    grid, split, lines = _emit(tmp_path)
    reg = json.loads((tmp_path / "query_sets.json").read_text())
    assert set(reg["sets"]) == set(G.QUERY_SET_REGISTRY)
    assert reg["sets"]["c_invalid"]["controls"]["duplicate_factor"] == "a7_redundant"
    a7 = json.loads((tmp_path / "a7_redundant.json").read_text())
    c = json.loads((tmp_path / "c_invalid.json").read_text())
    # |A7| = |C| in the shipped file, so the control is class-balanced against it
    assert a7["n_requests"] == len(c["records"]) == 72
    assert (
        reg["sets"]["a7_redundant"]["sha256"]
        == G.write_json_with_checksum(a7, tmp_path / "roundtrip.json")[1]
    )


def test_emitting_a7_leaves_the_c_set_bit_identical(tmp_path):
    """⛔ A7 is ADDITIVE.  The C set is single-sourced and must not move."""
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    _emit(tmp_path / "with", emit_a7=True)
    _emit(tmp_path / "without", emit_a7=False)
    assert (tmp_path / "with" / "c_invalid.json").read_bytes() == (
        tmp_path / "without" / "c_invalid.json"
    ).read_bytes()
    assert (tmp_path / "with" / "factor_index.npz").read_bytes() == (
        tmp_path / "without" / "factor_index.npz"
    ).read_bytes()


def test_the_emitted_sets_pass_the_detector(tmp_path):
    """The pre-registered prediction: every feature exactly 0.5, per kind and pooled."""
    _emit(tmp_path)
    report = G.assert_no_model_detector_at_chance(
        tmp_path / "query_sets.json", tolerance=DETECTOR_TOLERANCE
    )
    for key, res in report["per_kind"].items():
        assert res["max_auroc"] == pytest.approx(0.5), (key, res["features"])
    assert report["pooled"]["c_invalid"]["max_auroc"] == pytest.approx(0.5)
    # and the leak is still visible in the uncontrolled column, by design
    unc = report["uncontrolled_vs_base"]["c_invalid[duplicate_factor]"]
    assert unc["features"]["n_terms"] == 1.0
    assert report["pooled_uncontrolled_vs_base"]["c_invalid"][
        "max_auroc"
    ] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 5. ⭐ THE SUITE TEST: the artifacts that are actually on disk
# ---------------------------------------------------------------------------


def _registries():
    """Discovered from disk, ⛔ never a hardcoded list (that is the whole point).

    An empty parametrisation would make this test vanish silently, so no data root
    yields one explicitly SKIPPED case instead of nothing.
    """
    found = G.discover_query_set_registries()
    if found:
        return found
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason=f"no generative data root at {G.default_gen_data_root()}"
            ),
        )
    ]


@pytest.mark.parametrize(
    "registry", _registries(), ids=lambda p: getattr(p, "parent", p)
)
def test_registered_query_sets_on_disk_are_at_chance(registry):
    report = G.assert_no_model_detector_at_chance(
        registry, tolerance=DETECTOR_TOLERANCE
    )
    assert report["per_kind"], f"{registry}: audited nothing -- a vacuous pass"
    for key, res in report["per_kind"].items():
        assert res["max_auroc"] <= 0.5 + DETECTOR_TOLERANCE, (key, res)


def test_every_query_set_on_disk_is_registered():
    """An unregistered query set is invisible to the audit, which is the failure mode.

    Skips only when there is no generative data root at all (a clone or a worktree);
    it does NOT skip when a dataset directory exists but forgot its registry.
    """
    root = G.default_gen_data_root()
    if not root.exists():
        pytest.skip(f"no generative data root at {root}")
    orphans = [
        p
        for p in sorted(root.glob("**/c_invalid.json"))
        if not (p.parent / "query_sets.json").exists()
    ]
    assert not orphans, (
        f"query set(s) on disk with no query_sets.json beside them: {orphans} — "
        "they escape the no-model detector audit entirely"
    )


# ---------------------------------------------------------------------------
# 6. ``data_root`` must fail loudly (R26 ruling 7)
# ---------------------------------------------------------------------------
#
# ``gen_data.data_root`` means the DATASET directory for ``shapes3d_root()`` and its
# PARENT for ``default_gen_data_root()``.  ``exp_gen_store.load_c_records`` uses the
# parent spelling, so a ``--data-root`` pointing at the dataset directory used to log
# "⚠ no C set … SKIPPED" and carry on: a run with no C mapping in it, reported as a
# run with a null C result.  ⛔ That is "declared not-run reported as null" in code
# form, and it must raise.


def _config_with_root(root):
    from chlu.config import get_default_config

    cfg = get_default_config()
    cfg.gen_data.data_root = str(root)
    return cfg


def test_load_c_records_raises_on_a_wrong_data_root(tmp_path, monkeypatch):
    from chlu.experiments.exp_gen_store import load_c_records

    monkeypatch.delenv("CHLU_GEN_DATA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="no C set at"):
        load_c_records("shapes3d", _config_with_root(tmp_path), log=lambda *a: None)


def test_the_raise_names_the_dataset_dir_spelling_when_that_is_the_mistake(
    tmp_path, monkeypatch
):
    """The ambiguity is the bug, so the message resolves it instead of guessing."""
    from chlu.experiments.exp_gen_store import load_c_records

    monkeypatch.delenv("CHLU_GEN_DATA_ROOT", raising=False)
    (tmp_path / "c_invalid.json").write_text("{}")  # root IS the dataset dir
    with pytest.raises(FileNotFoundError, match="DOES exist.*PARENT"):
        load_c_records("shapes3d", _config_with_root(tmp_path), log=lambda *a: None)


def test_load_c_records_still_loads_a_correct_root(tmp_path, monkeypatch):
    """⛔ The raise must not be reachable when the C set IS where it should be."""
    from chlu.experiments.exp_gen_store import load_c_records

    monkeypatch.delenv("CHLU_GEN_DATA_ROOT", raising=False)
    _emit(tmp_path / "shapes3d")
    recs = load_c_records("shapes3d", _config_with_root(tmp_path), log=lambda *a: None)
    assert len(recs) == 72
