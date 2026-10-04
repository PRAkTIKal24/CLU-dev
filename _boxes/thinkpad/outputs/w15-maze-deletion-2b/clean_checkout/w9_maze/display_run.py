"""The DISPLAY section: refusal, deletion, and composition after deletion.

Run after :func:`w9_maze.pipeline.run`, on the artefacts it returns, so the object
being displayed is EXACTLY the object the RESULT rows were measured on.

Wave-9 rule 3 governs everything here: a DISPLAY row is never quoted as a RESULT, its
only comparator is a supervised refuser trained on impossible examples, and our line
is that no impossible example was ever seen.

Four refusal conditions, and the last two are the controls that make the first two
mean something:

  budget_infeasible  IMPOSSIBLE.  The same three words -- maze, agent, target -- that
                     the object answers correctly at a larger budget word.  There is
                     no novelty here at all, so a refusal cannot be novelty detection.
  walled             IMPOSSIBLE.  The target is sealed behind walls in this maze.  The
                     (maze, target) combination was never written.
  mt_unseen          POSSIBLE.  A (maze, target) combination that was also never
                     written, but is REACHABLE.  If the object refuses these as often
                     as the walled ones, then the walled refusal is novelty detection
                     and the row says so.
  grounded_possible  POSSIBLE.  The row-2 and row-3 requests.  Refusing these is the
                     false-refusal rate, and accept-unseen-valid is exactly this
                     number being low.
"""

from __future__ import annotations

import numpy as np

from w9_maze import display, evaluate, splits
from w9_maze.baselines import token_inputs
from w9_maze.display import SupervisedRefuser, delete_word_content, downdate_vs_refit
from w9_maze.task import BUDGET_GRID, TARGET_WORDS

GREEN = 1                       # the target word "green"
MOVE_LEFT = "W"


def _run_rows(bank, render, reader, rows, use_cert=True, forbid=()):
    if not rows:
        return {"n": 0}
    Zh = reader.read([r for (r, _m, _i, _j) in rows])
    plans, scenes = render.plans(Zh)
    recs = evaluate.answers(bank, rows, plans, scenes, use_certificate=use_cert,
                            forbid=forbid)
    agg = evaluate.aggregate(recs)
    agg["recs"] = recs
    return agg


def conditions(bank, split, row_sets):
    return {
        "budget_infeasible": splits.infeasible_requests(bank, split),
        "walled": splits.walled_requests(bank),
        "mt_unseen": row_sets.get("mt_unseen", []),
        "grounded_possible": row_sets["unseen_pair"] + row_sets["nway"],
    }


def run(bank, art, res, seed=0, refuser_steps=3000, verbose=True,
        exact_head_delete=False):
    ae = art["render"]
    clu = art["readers"]["clu"]
    split, row_sets = art["split"], dict(art["row_sets"])
    row_sets["mt_unseen"] = splits.row_requests(bank, split, "mt_unseen")
    conds = conditions(bank, split, row_sets)
    out = {"n_conditions": {k: len(v) for k, v in conds.items()},
           "target_words": list(TARGET_WORDS)}

    # ---------------------------------------------------------------- refusal
    out["refusal"] = {}
    for name, rows in conds.items():
        agg = _run_rows(bank, ae, clu, rows)
        out["refusal"][name] = {k: v for k, v in agg.items() if k != "recs"}
    if verbose:
        print("  DISPLAY refusal:", {k: round(v.get("refusal", float("nan")), 4)
                                     for k, v in out["refusal"].items()}, flush=True)

    # -- the supervised refuser: the ONLY comparator, and it HAS seen the impossible
    grids, main_spec = art["grids"], art["main_spec"]
    from w9_maze.arms import design_of
    M = len(bank)
    half = M // 2

    def tok(rows):
        Phi = design_of([r for (r, _m, _i, _j) in rows], grids, main_spec)[0]
        return token_inputs(Phi, main_spec.layout)

    sup_train_rows, sup_y = [], []
    for rows, lab in ((art["train_rows"], 0.0),
                      (conds["budget_infeasible"], 1.0), (conds["walled"], 1.0)):
        for r in rows:
            if r[1] < half:                    # impossible examples from HALF the bank
                sup_train_rows.append(r)
                sup_y.append(lab)
    sup = None
    if sup_train_rows and 0 < sum(sup_y) < len(sup_y):
        import jax
        sup = SupervisedRefuser(jax.random.PRNGKey(seed * 7 + 3),
                                tok(sup_train_rows[:1]).shape[2],
                                res["ledger"]["read_floats"], seed=seed)
        sup.fit(tok(sup_train_rows), np.asarray(sup_y), steps=refuser_steps)
        out["supervised_refuser"] = {
            "trained_on": {"n": len(sup_train_rows),
                           "n_impossible": int(sum(sup_y)),
                           "mazes": "0..%d (half the bank)" % (half - 1)},
            "params": sup.arm.n_params, "matched": bool(sup.arm.matched),
            "evaluated_on": "held-out mazes %d..%d only" % (half, M - 1),
            "refusal_rate": {},
        }
        for name, rows in conds.items():
            held = [r for r in rows if r[1] >= half]
            if not held:
                out["supervised_refuser"]["refusal_rate"][name] = None
                continue
            out["supervised_refuser"]["refusal_rate"][name] = float(
                sup.refuses(tok(held)).mean())
        out["clu_refusal_rate_heldout_mazes"] = {}
        for name, rows in conds.items():
            held = [r for r in rows if r[1] >= half]
            agg = _run_rows(bank, ae, clu, held)
            out["clu_refusal_rate_heldout_mazes"][name] = {
                k: v for k, v in agg.items() if k != "recs"}
        if verbose:
            print("  DISPLAY supervised refuser:",
                  out["supervised_refuser"]["refusal_rate"], flush=True)

    # ------------------------------------------------- interface deletion: move-left
    needs_w = {name: display.route_uses(bank, rows, MOVE_LEFT)
               for name, rows in conds.items() if name == "grounded_possible"}
    rows = conds["grounded_possible"]
    before = _run_rows(bank, ae, clu, rows, use_cert=True)
    after = _run_rows(bank, ae, clu, rows, use_cert=True, forbid=(MOVE_LEFT,))
    w = needs_w["grounded_possible"]
    out["delete_move_left"] = {
        "kind": ("INTERFACE deletion -- the action word is removed from the emission "
                 "interface and from the certificate's planning. NO ATOM IS TOUCHED, "
                 "so this row carries none of the exactness claim of the content "
                 "deletion below."),
        "n_routes_needing_move_left": int(w.sum()), "n": int(w.size),
        "before": {"refusal": before["refusal"], "validity": before["validity"]},
        "after_needs_W": _sub(after["recs"], w),
        "after_no_W": _sub(after["recs"], ~w),
    }
    if verbose:
        print("  DISPLAY delete move-left: refusal on routes that need it %.4f, "
              "on routes that do not %.4f"
              % (out["delete_move_left"]["after_needs_W"]["refusal"],
                 out["delete_move_left"]["after_no_W"]["refusal"]), flush=True)

    # -------------------------------------------------- content deletion: the green target
    Ztrain = art["Ztrain"]
    reqs_train = [r for (r, _m, _i, _j) in art["train_rows"]]
    keys = np.array([[r.m, r.a, r.t, r.b] for r in reqs_train], np.int64)
    keep = keys[:, 2] != GREEN
    out["delete_green_exactness"] = downdate_vs_refit(
        Ztrain, keep, clu.well_channels, clu.n_atoms, clu.dim)
    green_rows = [r for r in rows if r[3] == GREEN]
    other_rows = [r for r in rows if r[3] != GREEN]
    pre_green = _run_rows(bank, ae, clu, green_rows, use_cert=True)
    pre_other = _run_rows(bank, ae, clu, other_rows, use_cert=False)
    receipt = delete_word_content(clu, Ztrain, reqs_train, "target", GREEN,
                                  exact_head=exact_head_delete)
    post_green = _run_rows(bank, ae, clu, green_rows, use_cert=True)
    post_other = _run_rows(bank, ae, clu, other_rows, use_cert=False)
    out["delete_green"] = {
        "kind": ("CONTENT deletion -- every item written under the word is removed "
                 "from the sufficient statistic by a rank-m subtract, the atoms are "
                 "re-derived and the address book is carried across the basis change. "
                 "No refit, no pass over the pool, the items were never kept."),
        "receipt": receipt,
        "green_before": _strip(pre_green), "green_after": _strip(post_green),
        "other_before": _strip(pre_other), "other_after": _strip(post_other),
        "composition_without_retraining": {
            "validity_delta_on_other_targets":
                float(post_other["validity"] - pre_other["validity"]),
            "score_delta_on_other_targets":
                float(post_other["score"] - pre_other["score"]),
            "retrained": False,
        },
    }
    if verbose:
        print("  DISPLAY delete green: refusal %.4f -> %.4f; other targets validity "
              "%.4f -> %.4f (no retraining)"
              % (pre_green["refusal"], post_green["refusal"],
                 pre_other["validity"], post_other["validity"]), flush=True)
    return out


def _strip(agg):
    return {k: v for k, v in agg.items() if k != "recs"}


def _sub(recs, mask):
    r = [x for x, m in zip(recs, mask) if m]
    if not r:
        return {"n": 0, "refusal": float("nan"), "validity": float("nan")}
    return {"n": len(r), "refusal": float(np.mean([x["refused"] for x in r])),
            "validity": float(np.mean([x["valid"] for x in r]))}
