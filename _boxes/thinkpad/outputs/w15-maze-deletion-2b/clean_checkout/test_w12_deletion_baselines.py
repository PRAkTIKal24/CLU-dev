"""The properties of the w12 deletion comparators, as checks rather than prose.

Hermetic: builds its own 4-factor grid and its own random targets, so it needs no
feature cache, no decoder and no judge.  Run::

    python test_w12_deletion_baselines.py

Every check below corresponds to a sentence the FINDINGS report makes.  If one of them
ever fails, that sentence is wrong.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chlu.data.gen_splits import FactorGrid                      # noqa: E402
from chlu.eval.rivals import w12_deletion_baselines as W12       # noqa: E402

FAILED = []


def check(name, fn):
    try:
        detail = fn()
        print("  PASS  %-58s %s" % (name, detail if detail else ""))
    except Exception as exc:                                     # noqa: BLE001
        FAILED.append((name, exc))
        print("  FAIL  %-58s %s" % (name, exc))
        traceback.print_exc()


def toy_grid():
    return FactorGrid(
        names=("a", "b", "c", "d"),
        values=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        dataset="toy")


def toy_rows(grid):
    return np.arange(grid.n_rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. the worktree assertion
# ---------------------------------------------------------------------------


def t_worktree():
    wt = W12.assert_chlu_in_worktree()
    assert wt["inside"] is True
    bad = False
    try:
        W12.assert_chlu_in_worktree(expected_root=tempfile.gettempdir())
    except AssertionError:
        bad = True
    assert bad, "assert_chlu_in_worktree accepted a root chlu is NOT inside"
    return Path(wt["chlu_file"]).parent.name


# ---------------------------------------------------------------------------
# 2. a no-op deletion is REFUSED, never returned as a clean zero
# ---------------------------------------------------------------------------


def t_noop_refused():
    g = toy_grid()
    rows = toy_rows(g)
    fi = g.factor_indices_of(rows)
    keep = rows[fi[:, 0] != 2]                # factor a never takes index 2 here
    spec = W12.DeletionSpec(kind="value", terms=((0, 2),), label="noop")
    try:
        W12.partition_fit_rows(g, keep, spec)
    except AssertionError as exc:
        assert "NO-OP" in str(exc)
        return "refused with the realised counts in the message"
    raise AssertionError("a zero-row deletion was NOT refused")


def t_partition_counts():
    g = toy_grid()
    rows = toy_rows(g)
    spec = W12.DeletionSpec(kind="conjunction", terms=((0, 0), (2, 1)), label="cj")
    p = W12.partition_fit_rows(g, rows, spec)
    fi = g.factor_indices_of(rows)
    want = int(((fi[:, 0] == 0) & (fi[:, 2] == 1)).sum())
    assert p["n_deleted"] == want, (p["n_deleted"], want)
    assert p["n_fit_after"] + p["n_deleted"] == p["n_fit_before"]
    assert np.intersect1d(p["kept_rows"], p["deleted_rows"]).size == 0
    return "%d deleted / %d kept, disjoint" % (p["n_deleted"], p["n_fit_after"])


# ---------------------------------------------------------------------------
# 3. matched compute
# ---------------------------------------------------------------------------


def t_matched_compute():
    ref = W12.retrain_compute(16_384)
    b = W12.matched_compute_budget(16_384)
    assert b["gradient_row_visits"] == ref["gradient_row_visits"], (b, ref)
    assert b["fraction"] == 1.0
    return "%s row-visits, matched to the exact reference" % format(
        b["gradient_row_visits"], ",")


def t_matched_compute_refuses_unmeetable():
    try:
        W12.matched_compute_budget(16_384, fraction=1e-9)
    except AssertionError as exc:
        assert "CANNOT be matched" in str(exc)
        return "refuses rather than silently shrinking"
    raise AssertionError("an unmeetable matched-compute budget was accepted")


def t_assert_compute_matched():
    b = {"mlp": W12.matched_compute_budget(1000),
         "lstm": W12.matched_compute_budget(1000)}
    out = W12.assert_compute_matched(b)
    assert out["matched"] is True
    bad = {"mlp": W12.matched_compute_budget(1000),
           "lstm": W12.matched_compute_budget(9000)}
    try:
        W12.assert_compute_matched(bad)
    except AssertionError as exc:
        assert "matched compute FAILED" in str(exc)
        return "equal budgets pass, unequal budgets raise"
    raise AssertionError("mismatched budgets were accepted as matched")


# ---------------------------------------------------------------------------
# 4. the cost line is part of the deliverable
# ---------------------------------------------------------------------------


def t_cost_line():
    c = W12.CostLine(seconds=1.5, data_rows_read=10, bytes_rewritten=20,
                     gradient_row_visits=30).as_dict()
    W12.assert_cost_line_complete(c, "toy")
    for f in W12.COST_FIELDS:
        broken = dict(c)
        broken[f] = None
        try:
            W12.assert_cost_line_complete(broken, "toy")
        except AssertionError:
            continue
        raise AssertionError("a cost line missing %r was accepted" % f)
    return "seconds / data_rows_read / bytes_rewritten all required"


# ---------------------------------------------------------------------------
# 5. the impossible certifier is EXACT
# ---------------------------------------------------------------------------


def t_certifier():
    g = toy_grid()
    rows = toy_rows(g)
    spec = W12.DeletionSpec(kind="value", terms=((0, 1),), label="v")
    p = W12.partition_fit_rows(g, rows, spec)
    from chlu.experiments.exp_gen_factor_wells import value_presence_bitmap
    bm = value_presence_bitmap(g, p["kept_rows"])
    terms = [(int(f), int(v)) for f, v in p["spec"]["terms"]]

    Vd = W12.values_from_rows(g, p["deleted_rows"])
    Vk = W12.values_from_rows(g, p["kept_rows"])
    cd = W12.certify_impossible(Vd, g, bm, terms)
    ck = W12.certify_impossible(Vk, g, bm, terms)
    assert cd["impossible"].all(), "a deleted-value request was not certified impossible"
    assert not ck["impossible"].any(), "a surviving request was certified impossible"
    assert set(cd["reason"].tolist()) == {"deleted_value"}, set(cd["reason"].tolist())

    # out of range and off grid, decided by the factor grid alone
    Voor = Vk[:4].copy(); Voor[:, 2] = 99.0
    Voff = Vk[:4].copy(); Voff[:, 0] = 0.5
    ro = W12.certify_impossible(Voor, g, bm, terms)
    rf = W12.certify_impossible(Voff, g, bm, terms)
    assert set(ro["reason"].tolist()) == {"out_of_range"}, set(ro["reason"].tolist())
    assert set(rf["reason"].tolist()) == {"off_grid_value"}, set(rf["reason"].tolist())
    return "deleted / surviving / out-of-range / off-grid all decided exactly"


def t_certifier_conjunction_precedence():
    """A conjunction deletion leaves both VALUES held; only the pair is refused."""
    g = toy_grid()
    rows = toy_rows(g)
    spec = W12.DeletionSpec(kind="conjunction", terms=((0, 1), (1, 1)), label="cj")
    p = W12.partition_fit_rows(g, rows, spec)
    from chlu.experiments.exp_gen_factor_wells import value_presence_bitmap
    bm = value_presence_bitmap(g, p["kept_rows"])
    terms = [(int(f), int(v)) for f, v in p["spec"]["terms"]]
    assert bool(bm[0][1]) and bool(bm[1][1]), (
        "the conjunction deletion removed a whole VALUE; then it is not a conjunction "
        "test at all")
    cd = W12.certify_impossible(W12.values_from_rows(g, p["deleted_rows"]), g, bm,
                                terms)
    ck = W12.certify_impossible(W12.values_from_rows(g, p["kept_rows"]), g, bm, terms)
    assert cd["impossible"].all() and set(cd["reason"].tolist()) == {
        "deleted_conjunction"}
    assert not ck["impossible"].any()
    return "both values still held; only the deleted PAIR refused"


def t_refusal_sets_disjoint():
    g = toy_grid()
    rows = toy_rows(g)
    spec = W12.DeletionSpec(kind="value", terms=((0, 1),), label="v")
    p = W12.partition_fit_rows(g, rows, spec)
    from chlu.experiments.exp_gen_factor_wells import value_presence_bitmap
    bm = value_presence_bitmap(g, p["kept_rows"])
    s = W12.build_refusal_sets(g, p, bm, n_per_class=6)
    assert s["train_eval_overlap"] == 0, s["train_eval_overlap"]
    assert s["include_out_of_range"] is False
    s2 = W12.build_refusal_sets(g, p, bm, n_per_class=6, include_out_of_range=True)
    assert s2["n_impossible_out_of_range"] > 0
    return "train/eval overlap 0; out-of-range leg off by default, on by request"


# ---------------------------------------------------------------------------
# 6. the separation statistic
# ---------------------------------------------------------------------------


def t_auroc():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = (rng.random(60) > 0.5).astype(float)
        s = rng.random(60)
        pos, neg = s[y > 0.5], s[y < 0.5]
        brute = float(np.mean((pos[:, None] > neg[None, :]).astype(float)
                              + 0.5 * (pos[:, None] == neg[None, :])))
        got = W12._auroc(s, y)
        assert abs(got - brute) < 1e-12, (got, brute)
    return "matches the brute-force rank statistic to 1e-12"


def t_separation():
    s = np.array([0.9, 0.8, 0.1, 0.2])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    r = W12.separation_report(s, y)
    assert abs(r["separation"] - 0.7) < 1e-12, r["separation"]
    assert abs(r["auroc"] - 1.0) < 1e-12
    assert r["refuse_rate_on_impossible"] == 1.0
    assert r["false_refusal_on_possible"] == 0.0
    return "separation, AUROC and both rates agree with the definition"


# ---------------------------------------------------------------------------
# 7. the shipped-trainer regression that had to be fixed
# ---------------------------------------------------------------------------


def t_token_dtype_preserved():
    from chlu.eval.rivals import w4_learners as L
    ints = np.arange(12, dtype=np.int32).reshape(3, 4)
    out = L._as_model_input(ints)
    assert np.issubdtype(out.dtype, np.integer), out.dtype
    flo = np.zeros((3, 4), np.float64)
    assert L._as_model_input(flo).dtype == np.float32
    return "integer token inputs survive; float inputs still become float32"


# ---------------------------------------------------------------------------
# 8. the wiring: a saved comparator loads BY NAME through w4_api, unchanged
# ---------------------------------------------------------------------------


def t_roundtrip_by_name():
    import jax

    from chlu.eval.rivals.w4_api import load_rival

    g = toy_grid()
    rows = toy_rows(g)
    spec = W12.DeletionSpec(kind="value", terms=((0, 1),), label="v")
    part = W12.partition_fit_rows(g, rows, spec)
    cards = [int(c) for c in g.cardinalities]
    D = 8
    key = jax.random.PRNGKey(0)
    arm = W12.matched_arm("mlp", cards, D, key, param_target=2000,
                          log=lambda *_: None)
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((rows.size, D)).astype(np.float32)

    def gfn(r):
        return Y[np.asarray(r, np.int64)]

    res = W12.retrain_without(arm, g, part, gfn, cards, key, epochs=1,
                              log=lambda *_: None)
    W12.assert_cost_line_complete(res["cost"], "roundtrip")
    with tempfile.TemporaryDirectory() as td:
        W12.save_comparator(res, 0, g, {"features_sha256": "0" * 64}, td, spec.name(),
                            part=part)
        name = W12.comparator_name("mlp", "retrain_without", spec.name())
        r = load_rival(0, name, root=td)
        v = W12.values_from_rows(g, part["kept_rows"][:5])
        f, oor = r(v)
        assert f.shape == (5, D), f.shape
        assert not oor.any()
        assert r.meta["base_family"] == "mlp"
        assert r.meta["cost_line"]["data_rows_read"] == part["n_fit_after"]
        r2 = W12.load_comparator(0, name, td)
        assert np.array_equal(f, r2(v)[0]), "load_comparator and load_rival disagree"
    return "%s loads through w4_api with no w12 import" % name


def t_registry_complete():
    for k in W12.COMPARATOR_KINDS:
        assert k in W12.COMPARATORS, k
        e = W12.COMPARATORS[k]
        for f in ("role", "entry_point", "arms"):
            assert e.get(f), (k, f)
    return "%d comparator kinds, each with role / entry_point / arms" % len(
        W12.COMPARATOR_KINDS)


def main() -> int:
    print("w12 deletion-baselines -- property checks")
    for name, fn in [
        ("chlu resolves inside this worktree", t_worktree),
        ("a no-op deletion is REFUSED", t_noop_refused),
        ("partition counts are exact and disjoint", t_partition_counts),
        ("matched compute == the exact reference", t_matched_compute),
        ("an unmeetable matched budget raises", t_matched_compute_refuses_unmeetable),
        ("assert_compute_matched checks equality", t_assert_compute_matched),
        ("a cost line missing a field is refused", t_cost_line),
        ("the impossible certifier is exact", t_certifier),
        ("conjunction deletion keeps both values", t_certifier_conjunction_precedence),
        ("refusal train/eval sets are disjoint", t_refusal_sets_disjoint),
        ("AUROC matches brute force", t_auroc),
        ("separation report matches its definition", t_separation),
        ("token dtype survives the shipped trainer", t_token_dtype_preserved),
        ("a saved comparator loads BY NAME", t_roundtrip_by_name),
        ("the registry is complete", t_registry_complete),
    ]:
        check(name, fn)
    print()
    if FAILED:
        print("FAILED: %d of the properties above do not hold" % len(FAILED))
        return 1
    print("ALL PROPERTIES HOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
