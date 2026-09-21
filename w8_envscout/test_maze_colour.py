"""Evidence for the dial: the CMAZE-1 validity check is EXACT.

Run:  python -m w8_envscout.test_maze_colour
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np

from w8_envscout import maze_colour as mc


def test_judge_uses_no_floating_point():
    """The judge must not contain a float literal or a float-producing operator.

    Checked against the executable body only: the docstring is allowed to say the
    word "tolerance", the code is not allowed to introduce one.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(mc.judge)))
    fn = tree.body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]                      # drop the docstring
    src = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
    for bad in (".0", "1e-", "1e+", "abs(", "isclose", "tol", "float", "/", "**"):
        assert bad not in src, "judge() body contains %r -- validity would inherit a tolerance" % bad
    return "judge() executable body has no float, no division, no tolerance"


def test_optimal_route_is_always_accepted(n=200):
    for s in range(n):
        inst = mc.generate(s)
        if inst is None:
            continue
        L, route = mc.solve(inst)
        ok, reason, steps = mc.judge(inst, route)
        assert ok, (s, reason)
        assert steps == L == len(route) == inst.opt_len, (s, steps, L, inst.opt_len)
    return "every exact optimum replays to 'goal' at exactly opt_len steps"


def test_judge_is_deterministic_and_repeatable(n=100):
    for s in range(n):
        inst = mc.generate(s)
        if inst is None:
            continue
        _, route = mc.solve(inst)
        first = mc.judge(inst, route)
        for _ in range(3):
            assert mc.judge(inst, route) == first
    return "judge is a pure function of (instance, route)"


def test_each_rule_actually_refuses(n=300):
    """Both rules must be observed rejecting routes -- otherwise neither is load-bearing."""
    seen = set()
    rng = np.random.default_rng(0)
    for s in range(n):
        inst = mc.generate(s)
        if inst is None:
            continue
        for kw in (dict(rule_b=False), dict(rule_a=False), dict(colour_blind=True)):
            _, route = mc.solve(inst, **kw)
            ok, reason, _ = mc.judge(inst, route)
            if not ok:
                seen.add(reason.split("@")[0].split("-c")[0])
        _, route = mc.solve(inst)
        ok, reason, _ = mc.judge(inst, route[: len(route) // 2] + "NNNNNNNNNN")
        if not ok:
            seen.add(reason.split("@")[0].split("-c")[0])
        ok, reason, _ = mc.judge(inst, route + "N" * inst.budget)
        if not ok:
            seen.add(reason.split("@")[0].split("(")[0])
    for need in ("locked-door", "forbidden-tint", "over-budget"):
        assert need in seen, "rule never fired: %s (saw %s)" % (need, sorted(seen))
    return "RULE A, RULE B and the budget all observed refusing routes: %s" % sorted(seen)


def test_generator_contract(n=200):
    """Every accepted instance satisfies the unconditional acceptance conditions.

    At the adopted defaults `require_bend` is OFF -- adversarial placement is supposed
    to make rule B bend the route without a filter discarding the instances where it
    does not.  So the bend is measured as a RATE and asserted to be high, not asserted
    per instance; if it ever collapses, this test says so with the number.
    """
    n_ok = bends = 0
    for s in range(n):
        inst = mc.generate(s)
        if inst is None:
            continue
        n_ok += 1
        assert not mc._reachable_without_keys(inst), s      # doors are load-bearing
        assert mc.solve(inst)[0] == inst.opt_len, s         # feasible, and opt recorded
        assert mc.solve(inst, rule_b=False)[0] == inst.opt_len_a_only, s
        assert inst.budget >= inst.opt_len, s
        bends += inst.opt_len > inst.opt_len_a_only
    rate = bends / n_ok
    assert rate > 0.5, "rule B bends the route in only %.3f of instances" % rate
    return ("%d/%d seeds accepted; contract holds; rule B bends the route in %.3f"
            % (n_ok, n, rate))


def test_rule_b_alone_is_not_a_cost(n=200):
    """A blind planner's failure must be ILLEGALITY, not expense.

    If every blind failure were 'over-budget' the rules would only be changing cost,
    which is exactly the GYM-1/GYM-2 death.  Assert that illegality dominates.
    """
    illegal = overbudget = 0
    for s in range(n):
        inst = mc.generate(s)
        if inst is None:
            continue
        _, route = mc.solve(inst, rule_b=False)
        ok, reason, _ = mc.judge(inst, route)
        if ok:
            continue
        if reason.startswith("over-budget"):
            overbudget += 1
        else:
            illegal += 1
    assert illegal > 0 and illegal > overbudget, (illegal, overbudget)
    return "blind failures: %d by ILLEGALITY, %d by budget" % (illegal, overbudget)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print("PASS  %-42s %s" % (name, fn()))
