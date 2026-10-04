"""The seven-row scoreboard, and the falsifiers for the METRIC ITSELF.

Why this file exists
--------------------
A refusal metric is the easiest thing in this programme to get free.  Three failure
modes are silent unless they are tested:

1. **an arm that refuses everything looks perfect** on ``reject_invalid`` — so the
   refuse-everything stub is shipped and must score exactly **0**;
2. **the two halves of the pair drift apart** — ``accept_unseen_valid`` measured at a
   different threshold, a different setting or a different deployed object is not a
   pair at all, and the row must score **0** and say why;
3. ⛔⛔ **the pair is winnable with no store in it.**  ``grid_syntax_rule`` reads the
   request and the factor grid and nothing else, and it scores a perfect pair on BOTH
   refusal rows.  That is measured here, not asserted, and it is why the rule is a
   mandatory row rather than a sentence in a paper.

⛔ ``basis_head`` (the wave-2 fitted basis + linear head) is the like-for-like and
``knn_proto`` is the second row; :func:`build_scoreboard` refuses to emit a table
missing either on any row, and that refusal is tested.
"""

import json

import numpy as np
import pytest

from chlu.data import gen_splits as G
from chlu.data import gen_value_holdout as V
from chlu.experiments import gen_scoreboard as S

BUDGET = 524_288
N = 32
F = 6


@pytest.fixture(scope="module")
def grid():
    return G.FactorGrid(
        names=("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"),
        values=(
            np.arange(10) / 10.0,
            np.arange(10) / 10.0,
            np.arange(10) / 10.0,
            np.linspace(0.75, 1.25, 8),
            np.arange(4, dtype=float),
            np.linspace(-30.0, 30.0, 15),
        ),
        dataset="toy3d",
    )


def _ledger(deployed=BUDGET, note=None):
    led = {"deployed_bytes": deployed, "budget_bytes": BUDGET, "what": "test stub"}
    if note:
        led["budget_note"] = note
    return led


def _outcome(accept, valid, seed=0, threshold=0.5, setting=None, ledger=None):
    rng = np.random.default_rng(seed)
    accept = np.asarray(accept, bool)
    valid = np.asarray(valid, bool) & accept
    tuples = rng.integers(0, 4, size=(accept.size, F))
    tuples[~valid] = -1
    return S.make_outcome(
        accept=accept,
        valid=valid,
        tuples=tuples,
        threshold=threshold,
        setting=setting if setting is not None else {"arm": "stub"},
        byte_ledger=ledger or _ledger(),
    )


def _full_arm(seed=0, accept_rate=1.0, valid_rate=0.8, **kw):
    """All seven rows for one arm, at one threshold and one setting."""
    rng = np.random.default_rng(seed)
    rows = {}
    for row_id in S.ROW_IDS:
        a = rng.random(N) < accept_rate
        v = a & (rng.random(N) < valid_rate)
        rows[row_id] = _outcome(a, v, seed=seed, **kw)
    return rows


# ---------------------------------------------------------------------------
# 1. ⛔ the falsifier for the metric: refuse everything
# ---------------------------------------------------------------------------


def test_an_arm_that_refuses_everything_scores_ZERO_on_the_refusal_rows():
    """⛔ THE falsifier.  Encoded in the metric, ⛔ not left to prose."""
    refuse_row5 = S.refuse_everything_outcome(N)
    refuse_row2 = S.refuse_everything_outcome(N)
    scored = S.score_refusal_row(refuse_row5, refuse_row2, row_id="r5_comp_refusal")
    assert scored["reject_invalid"] == 1.0
    assert scored["accept_unseen_valid"] == 0.0
    assert scored["pair"] == [1.0, 0.0]
    assert scored["score"] == 0.0


def test_an_arm_that_accepts_everything_also_scores_ZERO():
    """The mirror: a permissive arm buys nothing either."""
    scored = S.score_refusal_row(
        S.accept_everything_outcome(N), S.accept_everything_outcome(N)
    )
    assert scored["pair"] == [0.0, 1.0]
    assert scored["score"] == 0.0


def test_the_refuse_everything_stub_scores_zero_on_the_WHOLE_table():
    """End to end: the stub fills all seven rows and wins nothing anywhere."""
    stub = {r: S.refuse_everything_outcome(N) for r in S.ROW_IDS}
    outcomes = {
        "refuse_everything": stub,
        S.PRIMARY_CONTROL: _full_arm(1),
        "knn_proto": _full_arm(2, ledger=_ledger(475_136, note="⚠ DECLARED: ~31 %")),
        S.LAUNDERING_ARM: {r: S.refuse_everything_outcome(N) for r in S.ROW_IDS},
    }
    table = S.build_scoreboard("refuse_everything", outcomes, seen_tuples=[])
    for row_id in S.REFUSAL_ROWS:
        assert table["rows"][row_id]["arms"]["refuse_everything"]["score"] == 0.0
    for row_id in ("r1_seen", "r2_unseen_valid"):
        assert table["rows"][row_id]["arms"]["refuse_everything"]["score"] == 0.0
    for row_id in ("r3_comp_2way", "r4_comp_nway"):
        assert table["rows"][row_id]["arms"]["refuse_everything"]["score"] == 0.0


# ---------------------------------------------------------------------------
# 2. ⛔ the pair must BE a pair
# ---------------------------------------------------------------------------


def test_a_different_threshold_makes_the_row_score_zero_and_says_why():
    row5 = _outcome(np.zeros(N, bool), np.zeros(N, bool), threshold=0.9)
    row2 = _outcome(np.ones(N, bool), np.ones(N, bool), threshold=0.5)
    scored = S.score_refusal_row(row5, row2)
    assert scored["refusal_gmean"] == 1.0  # ⚠ the naive number looks perfect
    assert scored["score"] == 0.0          # ⛔ and the metric refuses to pay it
    assert any("same threshold" in f for f in scored["faults"])


def test_a_different_setting_or_a_different_deployed_object_scores_zero():
    row2 = _outcome(np.ones(N, bool), np.ones(N, bool), setting={"tau": 1.0})
    row5 = _outcome(np.zeros(N, bool), np.zeros(N, bool), setting={"tau": 2.0})
    assert S.score_refusal_row(row5, row2)["score"] == 0.0
    row5b = _outcome(
        np.zeros(N, bool), np.zeros(N, bool), setting={"tau": 1.0},
        ledger=_ledger(2 * BUDGET, note="declared"),
    )
    scored = S.score_refusal_row(row5b, row2)
    assert scored["score"] == 0.0
    assert any("deployed_bytes" in f for f in scored["faults"])


def test_accept_unseen_valid_is_always_measured_on_row_2():
    row2 = _outcome(np.r_[np.ones(24, bool), np.zeros(8, bool)], np.zeros(N, bool))
    row5 = _outcome(np.r_[np.zeros(28, bool), np.ones(4, bool)], np.zeros(N, bool))
    scored = S.score_refusal_row(row5, row2)
    assert scored["accept_unseen_valid_source"] == "r2_unseen_valid"
    assert scored["accept_unseen_valid"] == pytest.approx(24 / 32)
    assert scored["reject_invalid"] == pytest.approx(28 / 32)
    assert scored["refusal_gmean"] == pytest.approx(np.sqrt(24 / 32 * 28 / 32))


# ---------------------------------------------------------------------------
# 3. ⛔⛔ the laundering control — the refusal rows are winnable with NO store
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def query_sets(tmp_path_factory, grid):
    """The real emitted rows 2 / 5 / 6, built by the shipped emitter."""
    d = tmp_path_factory.mktemp("qs")
    cfg = V.ValueHoldoutConfig(
        dataset="toy3d", row2_n_requests=128, row5_n_per_kind=64,
        c_invalid_n_per_kind=64, a7_n_requests=128,
    )
    V.emit_value_holdout(grid, d, cfg, run_detector=False)
    return {
        "row2": json.loads((d / "row2_unseen.json").read_text())["records"],
        "row5": json.loads((d / "row5_held_out.json").read_text())["records"],
        "row6": json.loads((d / "c_invalid.json").read_text())["records"],
    }


def test_a_no_model_grid_rule_scores_a_PERFECT_pair_on_both_refusal_rows(
    grid, query_sets
):
    """⛔⛔ The finding this harness exists to surface.

    ``accept iff (some term is off-grid but in range) and (no duplicate factor)``
    reads the request and the grid and nothing else.  It accepts every row-2 request
    and refuses every row-5 and row-6 request ⇒ both pairs are ``(1.0, 1.0)``.
    """
    row2 = S.grid_syntax_outcome(query_sets["row2"], grid)
    row5 = S.grid_syntax_outcome(query_sets["row5"], grid)
    row6 = S.grid_syntax_outcome(query_sets["row6"], grid)
    assert row2["accept"].all(), "row 2 is off-grid-in-range by definition"
    assert not row5["accept"].any(), "row 5 is ON-grid by definition"
    assert not row6["accept"].any(), "row 6 is out-of-range or duplicated"
    for inv in (row5, row6):
        scored = S.score_refusal_row(inv, row2)
        assert scored["pair"] == [1.0, 1.0]
        assert scored["score"] == 1.0


def test_the_no_model_rule_scores_ZERO_on_the_generation_rows(grid, query_sets):
    """⭐ The contrast that keeps the finding honest: it produces no image."""
    row2 = S.grid_syntax_outcome(query_sets["row2"], grid)
    assert S.score_accuracy_row(row2)["judge_accuracy"] == 0.0
    assert S.score_novelty_row(row2, seen_tuples=[])[
        "distinct_valid_novel_tuples"
    ] == 0


def test_the_no_model_rule_deploys_zero_bytes_and_declares_it(grid, query_sets):
    out = S.grid_syntax_outcome(query_sets["row2"], grid)
    assert out["byte_ledger"]["deployed_bytes"] == 0
    assert "DECLARED" in out["byte_ledger"]["budget_note"]


def test_a_scoreboard_without_the_laundering_control_is_refused():
    outcomes = {
        "clu": _full_arm(0),
        S.PRIMARY_CONTROL: _full_arm(1),
        "knn_proto": _full_arm(2),
    }
    with pytest.raises(S.ScoreboardError, match="laundering control"):
        S.build_scoreboard("clu", outcomes, seen_tuples=[])


# ---------------------------------------------------------------------------
# 4. both controls, on every row, always
# ---------------------------------------------------------------------------


def _ok_outcomes():
    return {
        "clu": _full_arm(0),
        S.PRIMARY_CONTROL: _full_arm(1),
        "knn_proto": _full_arm(2, ledger=_ledger(475_136, note="⚠ DECLARED: ~31 %")),
        S.LAUNDERING_ARM: _full_arm(3, ledger=_ledger(0, note="⛔ DECLARED: no store")),
    }


@pytest.mark.parametrize("drop", [S.PRIMARY_CONTROL, "knn_proto"])
def test_a_missing_control_is_refused(drop):
    outcomes = _ok_outcomes()
    del outcomes[drop]
    with pytest.raises(S.ScoreboardError, match="missing arm"):
        S.build_scoreboard("clu", outcomes, seen_tuples=[])


def test_a_missing_ROW_is_refused():
    outcomes = _ok_outcomes()
    del outcomes["knn_proto"]["r7_deletion"]
    with pytest.raises(S.ScoreboardError, match="missing row"):
        S.build_scoreboard("clu", outcomes, seen_tuples=[])


def test_every_row_carries_every_arm_and_both_controls():
    table = S.build_scoreboard("clu", _ok_outcomes(), seen_tuples=[])
    assert set(table["rows"]) == set(S.ROW_IDS)
    for row in table["rows"].values():
        assert {"clu", S.PRIMARY_CONTROL, "knn_proto", S.LAUNDERING_ARM} <= set(
            row["arms"]
        )


def test_the_like_for_like_is_basis_head_and_knn_proto_is_the_second_row():
    table = S.build_scoreboard("clu", _ok_outcomes(), seen_tuples=[])
    assert table["controls"]["like_for_like"] == "basis_head"
    assert table["controls"]["second_row"] == "knn_proto"


def test_a_tie_with_basis_head_is_reported_as_decoration_IN_THOSE_WORDS():
    """⛔⛔ w3-hub-common §1, verbatim."""
    arm = _full_arm(7)
    outcomes = _ok_outcomes()
    outcomes["clu"] = arm
    outcomes[S.PRIMARY_CONTROL] = {r: _outcome(
        o["accept"], o["valid"], seed=7, threshold=o["threshold"],
    ) for r, o in arm.items()}
    table = S.build_scoreboard("clu", outcomes, seen_tuples=[])
    v = table["rows"]["r1_seen"]["verdict"]
    assert v["delta_vs_basis_head"] == 0.0
    assert v["beats_control"] is False
    assert "decoration and is reported as such, in those words" in v["verdict"]


# ---------------------------------------------------------------------------
# 5. the byte ledger — a column, per row
# ---------------------------------------------------------------------------


def test_an_unmatched_ledger_without_a_budget_note_is_refused():
    outcomes = _ok_outcomes()
    outcomes["knn_proto"] = _full_arm(2, ledger=_ledger(475_136))  # ⛔ no note
    with pytest.raises(S.ScoreboardError, match="DECLARED, not silent"):
        S.build_scoreboard("clu", outcomes, seen_tuples=[])


def test_an_unmatched_like_for_like_is_refused_even_WITH_a_note():
    """⭐ arm-vs-basis_head at unmatched bytes is a different experiment."""
    outcomes = _ok_outcomes()
    outcomes[S.PRIMARY_CONTROL] = _full_arm(
        1, ledger=_ledger(2 * BUDGET, note="declared, but not good enough")
    )
    with pytest.raises(S.ScoreboardError, match="THE like-for-like"):
        S.build_scoreboard("clu", outcomes, seen_tuples=[])


def test_a_ledger_that_moves_between_rows_is_refused():
    """⛔ One deployed object answers all seven rows, or it is seven experiments."""
    outcomes = _ok_outcomes()
    outcomes["clu"]["r5_comp_refusal"] = _outcome(
        np.zeros(N, bool), np.zeros(N, bool),
        ledger=_ledger(2 * BUDGET, note="declared"),
    )
    with pytest.raises(S.ScoreboardError, match="not constant across rows"):
        S.build_scoreboard("clu", outcomes, seen_tuples=[])


def test_the_ledger_is_a_field_on_every_scored_cell():
    table = S.build_scoreboard("clu", _ok_outcomes(), seen_tuples=[])
    for row in table["rows"].values():
        for arm, cell in row["arms"].items():
            assert cell["byte_ledger"]["deployed_bytes"] is not None, arm


def test_an_outcome_without_a_measured_ledger_is_refused():
    with pytest.raises(S.ScoreboardError, match="deployed_bytes"):
        S.make_outcome([True], [True], None, 0.5, {}, {"deployed_bytes": None})


# ---------------------------------------------------------------------------
# 6. rows 3-4 (novelty) and row 7 (deletion)
# ---------------------------------------------------------------------------


def test_replaying_a_stored_tuple_scores_nothing_on_the_compositional_rows():
    """Wave-2 fact 3: density descent regresses to seen combinations."""
    tuples = np.tile(np.array([[1, 1, 1, 1, 1, 1]]), (N, 1))
    out = S.make_outcome(
        np.ones(N, bool), np.ones(N, bool), tuples, 0.5, {}, _ledger()
    )
    scored = S.score_novelty_row(out, seen_tuples=[(1, 1, 1, 1, 1, 1)])
    assert scored["distinct_valid_tuples"] == 1
    assert scored["distinct_valid_novel_tuples"] == 0
    assert scored["replayed_seen_tuples"] == 1
    assert scored["score"] == 0.0


def test_row_7_needs_BOTH_halves_and_no_retraining():
    seen = [(1, 1, 1, 1, 1, 1)]
    novel = np.tile(np.array([[2, 2, 2, 2, 2, 2]]), (N, 1))
    survivor = S.make_outcome(
        np.ones(N, bool), np.ones(N, bool), novel, 0.5, {"t": 1}, _ledger()
    )
    deleted = S.make_outcome(
        np.zeros(N, bool), np.zeros(N, bool), None, 0.5, {"t": 1}, _ledger()
    )
    row2 = S.make_outcome(
        np.ones(N, bool), np.ones(N, bool), novel, 0.5, {"t": 1}, _ledger()
    )
    ok = S.score_deletion_row(
        survivor, deleted, row2, seen, fit_fingerprint=("sha", "sha")
    )
    assert ok["both_hold"] is True and ok["score"] == 1.0
    assert ok["no_retraining_verified"] is True
    # ⛔ a moved fit fingerprint IS retraining
    bad = S.score_deletion_row(
        survivor, deleted, row2, seen, fit_fingerprint=("sha", "OTHER")
    )
    assert bad["score"] == 0.0 and any("RETRAINED" in f for f in bad["faults"])
    # ⛔ survivors that stop composing kill the row even if refusal is perfect
    dead = S.score_deletion_row(
        S.refuse_everything_outcome(N), deleted, row2, seen,
        fit_fingerprint=("sha", "sha"),
    )
    assert dead["score"] == 0.0 and dead["both_hold"] is False


def test_delta_reject_is_the_non_free_half(grid, query_sets):
    """⭐ The grid does not move when a value is deleted from the store."""
    row2 = S.grid_syntax_outcome(query_sets["row2"], grid)
    before = S.grid_syntax_outcome(query_sets["row5"], grid)
    after = S.grid_syntax_outcome(query_sets["row5"], grid)
    scored = S.score_deletion_row(
        row2, after, row2, seen_tuples=[], pre_deletion_refusal=before,
        fit_fingerprint=("sha", "sha"),
    )
    assert scored["delta_reject"] == 0.0


# ---------------------------------------------------------------------------
# 7. the scope statement + the artifact
# ---------------------------------------------------------------------------


def test_the_scope_statement_is_a_FIELD_on_every_row():
    """⛔ §2: no hub can quote a row without it."""
    table = S.build_scoreboard("clu", _ok_outcomes(), seen_tuples=[])
    assert table["value_holdout_scope"] == V.VALUE_HOLDOUT_SCOPE
    for row_id, row in table["rows"].items():
        assert row["value_holdout_scope"] == V.VALUE_HOLDOUT_SCOPE, row_id
        assert "STORE'S FIT POOL" in row["value_holdout_scope"]


def test_the_written_artifact_round_trips_with_a_checksum(tmp_path):
    table = S.build_scoreboard("clu", _ok_outcomes(), seen_tuples=[])
    path, sha = S.write_scoreboard(table, tmp_path / "scoreboard.json")
    back = json.loads(path.read_text())
    assert len(sha) == 64
    assert back["value_holdout_scope"] == V.VALUE_HOLDOUT_SCOPE
    assert set(back["rows"]) == set(S.ROW_IDS)
    assert [back["rows"][r]["index"] for r in S.ROW_IDS] == [1, 2, 3, 4, 5, 6, 7]


def test_config_lives_next_to_the_code_and_a_typo_raises():
    """⛔ §5: not a block in chlu/config.py."""
    assert S.ScoreboardConfig.from_mapping({"byte_budget": 8}).byte_budget == 8
    with pytest.raises(ValueError, match="unknown key"):
        S.ScoreboardConfig.from_mapping({"byte_budge": 8})


def test_a_valid_tuple_on_a_refused_request_is_refused():
    """A refused request produces no output for the judge to score."""
    with pytest.raises(S.ScoreboardError, match="marked VALID while refused"):
        S.make_outcome([False], [True], None, 0.5, {}, _ledger())
