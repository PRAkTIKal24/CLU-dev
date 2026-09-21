"""Tests for the w9 maze harness.  Run: python test_w9_harness.py

These are the properties the instrument has to have for anything scored through it
to mean what it says.  They are cheap enough to run every time the harness changes.
"""

from __future__ import annotations

import numpy as np

from w8_envscout.maze_colour import generate, solve
from w9_harness import arms, columns, data, dynamics, encoder, features, render, tabular
from w9_harness.judge_maze import (ACTIONS, JUDGE, Contamination, audit,
                                   audit_oracle, audit_redundant, audit_source,
                                   audit_types, judge_ref, random_routes,
                                   reason_class)

FAILED = []


def check(name, cond, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if cond else "FAIL", "  " + detail if detail else ""))
    if not cond:
        FAILED.append(name)


def instances(n, seed0=900_000):
    out, s = [], seed0
    while len(out) < n:
        i = generate(s)
        s += 1
        if i is not None:
            out.append(i)
    return out


def test_judge(insts):
    print("judge -- the dial")
    check("no float machinery in the judge source", not audit_source())
    rng = np.random.default_rng(1)
    pairs = [(i, r) for i in insts for r in random_routes(i, rng)]
    check("judge never touches a float (tracing ints)", audit_types(pairs))
    check("judge == independent replay on every route", not audit_redundant(pairs),
          "%d routes" % len(pairs))
    check("optimum / truncation / budget / mutants crisp", not audit_oracle(insts))
    # the tracing int really does fire when a float gets in
    caught = False
    try:
        from w9_harness.judge_maze import _StrictInt
        _StrictInt(3) < 3.5
    except Contamination:
        caught = True
    check("the float tracer is not vacuous (it fires on a float)", caught)
    # budget is a hard integer boundary, not a soft one
    hard = True
    for inst in insts:
        L, route = solve(inst)
        tight = data._with(inst, budget=L)
        loose = data._with(inst, budget=L - 1)
        hard &= JUDGE(tight, route)[0] and not JUDGE(loose, route)[0]
    check("budget is an exact integer boundary (L ok, L-1 rejected)", hard)


def test_stepper(insts):
    print("stepper -- the judge's twin")
    r = dynamics.audit_stepper(insts, n_routes=6, verbose=False)
    check("stepper verdict == judge verdict on random routes", r["pass"],
          "%d routes" % r["routes"])
    ok = all(dynamics.replay(i, solve(i)[1])[2] for i in insts)
    check("every oracle route replays to the goal", ok)


def test_encoder(insts):
    print("encoder -- shared, and shared means shared")
    imgs = render.render_batch(insts)
    check("image is binary in every channel",
          bool(np.all((imgs == 0) | (imgs == 1))))
    enc, rep = encoder.train(imgs, feat_dim=8, steps=60, verbose=False)
    ind = encoder.independence_report(enc)
    check("encoder loss sees no actions/labels", not ind["sees_actions"])
    check("encoder loss sees no judge", not ind["sees_judge"])
    check("encoder loss takes exactly (model, images)",
          ind["loss_arguments"] == ["ae", "X"])
    f1 = encoder.features(enc, imgs)
    f2 = encoder.features(enc, imgs)
    check("frozen encoder is deterministic", bool(np.array_equal(f1, f2)))
    return enc, f1


def test_observation(insts, feats):
    print("observation -- one vector, every arm")
    k = int(insts[0].n_colours)
    d = features.obs_dim(8, k)
    st = dynamics.initial_state(insts[0])
    v = features.obs(insts[0], feats[0], st)
    check("obs_dim matches the vector actually produced", v.shape[0] == d,
          "%d" % d)
    check("obs is finite", bool(np.all(np.isfinite(v))))
    # the pocket must change the observation -- otherwise rule B is invisible
    st2 = dynamics.State(st.pos, tuple([1] + list(st.ks[1:])), 0)
    check("pocket state changes the observation",
          not np.array_equal(features.obs(insts[0], feats[0], st2), v))


def test_head_and_params(obs_dim):
    print("arms -- one head, matched counts")
    ws = arms.match_widths(obs_dim, arms.N_ACTIONS, 30000, tmax=64, verbose=False)
    pols = {k: arms.build(k, obs_dim, w, arms.N_ACTIONS, **kw)
            for k, (w, p, kw) in ws.items()}
    rep = arms.head_identity_report(pols)
    check("every arm's action head has the same shape",
          rep["identical_shape_across_arms"], str(rep["head_params"]))
    import equinox as eqx
    import jax
    leaves = [jax.tree_util.tree_leaves(eqx.filter(p.head, eqx.is_array))
              for p in pols.values()]
    same = all(all(np.array_equal(a, b) for a, b in zip(leaves[0], L)) for L in leaves)
    check("every arm's action head starts from identical weights", same)
    counts = [p for (w, p, kw) in ws.values()]
    spread = (max(counts) - min(counts)) / max(counts)
    check("trunk parameter counts matched within 2%", spread < 0.02,
          "spread %.3f%%  %s" % (100 * spread, counts))


def test_impossible():
    print("impossible examples -- exactly certified, and unseen")
    insts, fams = data.make_impossible(12, 700_000, rng_seed=0)
    bad = data.certify_impossible(insts, fams)
    check("every impossible instance is certified impossible", not bad, str(bad[:2]))
    # no route at all can be accepted: brute force short random routes as a sanity net
    rng = np.random.default_rng(5)
    accepted = 0
    for inst in insts:
        for _ in range(200):
            L = int(rng.integers(1, int(inst.budget) + 1))
            r = "".join(ACTIONS[int(rng.integers(5))] for _ in range(L))
            accepted += JUDGE(inst, r)[0]
    check("random search finds no accepted route on impossible set", accepted == 0)
    poss, seeds = data.make_instances(8, 800_000)
    check("impossible seeds disjoint from the arms' seeds",
          data.assert_unseen(seeds, insts)["clean"])


def test_columns(insts):
    print("reported columns -- reported, never gates")
    cols = columns.all_columns(insts[:6])
    check("every blind planner produced a column", len(cols) == len(columns.BLIND))
    from w9_harness.rollout import score
    orc = score(insts[:6], cols["oracle (full composition)"])
    check("the oracle column is 1.0 under the exact judge", orc["validity"] == 1.0,
          "%.4f" % orc["validity"])
    src = open("w9_harness/columns.py").read() + open("exp_w9_harness.py").read()
    check("no abort/raise keyed on a column value",
          "raise" not in src.split("REPORTED COLUMNS")[-1])


def main():
    insts = instances(8)
    test_judge(insts)
    test_stepper(insts)
    enc, feats = test_encoder(insts)
    test_observation(insts, feats)
    k = int(insts[0].n_colours)
    test_observation(insts, feats)
    test_head_and_params(features.obs_dim(8, k))
    test_impossible()
    test_columns(insts)
    print("\n%d checks failed%s" % (len(FAILED), (": " + ", ".join(FAILED)) if FAILED else ""))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
