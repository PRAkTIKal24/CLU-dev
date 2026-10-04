"""The ONE exact judge, plus the audit that shows it is exact.

The judge itself is `w8_envscout.maze_colour.judge`: integer replay of RULES A+B,
re-exported here UNCHANGED so that every arm in this harness is scored by literally
the same function object.  Nothing in this module wraps it in a tolerance, a metric
or a soft score.

`audit()` is the evidence for the `judge_exact` field of HARNESS-READY.json.  It is
four independent checks:

  A. SOURCE    -- the judge's source (and its one helper) contains no float literal,
                  no true division, no `isclose`/`allclose`/tolerance name, no call
                  into float machinery.
  B. TYPES     -- every comparison the judge performs is between integers.  Enforced
                  by replaying with a tracing int subclass that raises the moment it
                  is compared with, or coerced to, a float.
  C. REDUNDANT -- a second, independently written replay (`judge_ref`, written from
                  the RULES in the docstring rather than from the judge's code)
                  agrees with the judge on every (instance, route) pair, including
                  the failure reason class and the step count.
  D. ORACLE    -- the exact BFS optimum is accepted at exactly its own length, the
                  optimum truncated by one step is rejected, a route one over budget
                  is rejected AS over-budget, and every single-action mutation is
                  decided identically by both judges -- never "almost".

A judge that needed a tolerance would fail A or B.  A judge that was exact but wrong
would fail C or D.
"""

from __future__ import annotations

import ast
import inspect
from collections import Counter

import numpy as np

from w8_envscout import maze_colour as mc
from w8_envscout.maze_colour import WALL, judge, solve

# The single judge every arm is scored by.  Do not shadow, do not wrap.
JUDGE = judge

ACTIONS = "NESWP"
REFUSE = "R"


# ----------------------------------------------------------------- A. source
_FORBIDDEN_NAMES = {"isclose", "allclose", "approx", "atol", "rtol", "eps",
                    "epsilon", "tolerance", "tol", "round", "float", "isfinite"}
_FORBIDDEN_ATTRS = {"isclose", "allclose", "float32", "float64", "sqrt", "exp",
                    "log", "norm", "linalg"}


def _dedent(src):
    lines = src.splitlines()
    pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[pad:] for l in lines)


def _source_findings(fn):
    tree = ast.parse(_dedent(inspect.getsource(fn)))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            bad.append("float literal %r" % (node.value,))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            bad.append("true division -- produces a float")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            bad.append("name %s" % node.id)
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            bad.append("attribute .%s" % node.attr)
    return bad


def audit_source():
    """Check A.  The judge and its one helper carry no float machinery."""
    findings = {}
    for fn in (mc.judge, mc._pocket):
        f = _source_findings(fn)
        if f:
            findings[fn.__name__] = f
    return findings


# ------------------------------------------------------------------ B. types
class Contamination(AssertionError):
    """Raised when the judge touches a float."""


class _StrictInt(int):
    """An int that refuses to touch a float.  Traces the judge's arithmetic."""

    def _chk(self, other):
        if isinstance(other, (float, np.floating)):
            raise Contamination("judge compared an int with a float %r" % (other,))

    def __eq__(self, o):
        self._chk(o)
        return int(self) == o

    def __ne__(self, o):
        self._chk(o)
        return int(self) != o

    def __lt__(self, o):
        self._chk(o)
        return int(self) < o

    def __gt__(self, o):
        self._chk(o)
        return int(self) > o

    def __le__(self, o):
        self._chk(o)
        return int(self) <= o

    def __ge__(self, o):
        self._chk(o)
        return int(self) >= o

    def __hash__(self):
        return int.__hash__(self)

    def __float__(self):
        raise Contamination("judge coerced a state value to float")


def _strict_instance(inst):
    """The same instance with budget and coordinates carried as tracing ints."""
    si = lambda v: _StrictInt(int(v))
    return mc.Instance(inst.seed, inst.grid, inst.tint, inst.door,
                       tuple((si(r), si(c)) for (r, c) in inst.key_at),
                       (si(inst.start[0]), si(inst.start[1])),
                       (si(inst.goal[0]), si(inst.goal[1])),
                       inst.n_colours, si(inst.budget),
                       inst.opt_len, inst.opt_len_a_only)


def audit_types(pairs):
    """Check B.  Replay under tracing ints; any float contact raises."""
    for inst, route in pairs:
        if route is None:
            continue
        JUDGE(_strict_instance(inst), route)
    return True


# -------------------------------------------------------------- C. redundant
_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


def judge_ref(inst, route):
    """A second replay, written from the RULES, not from `judge`'s code.

    RULE A  entering a door cell of colour c requires the pocket to hold key c;
            the key is consumed and the door is open from then on.
    RULE B  entering a cell tinted colour c is forbidden while the pocket holds c.
    Plus: a pickup needs an empty pocket and an as-yet-untaken key on the current
    cell, the goal must be entered, and the route may not exceed the budget.

    Returns (valid, reason, steps) using the same reason vocabulary as `judge`,
    colour suffixes included, so the agreement check is on the colour too.
    """
    if route is None:
        return False, "no-route", 0
    if int(len(route)) > int(inst.budget):
        return False, "over-budget", int(len(route))
    grid, tint, door = inst.grid, inst.tint, inst.door
    H, W = int(grid.shape[0]), int(grid.shape[1])
    homes = {inst.key_at[i]: i + 1 for i in range(int(inst.n_colours))}
    carried = 0                                   # 0 = empty pocket
    used = set()                                  # colours whose door is open
    taken = set()                                 # colours picked up at some point
    r, c = int(inst.start[0]), int(inst.start[1])
    for t in range(len(route)):
        a = route[t]
        if a == "P":
            col = homes.get((r, c), 0)
            if col == 0:
                return False, "pickup-on-empty-cell", t
            if col in taken:
                return False, "key-not-there", t
            if carried != 0:
                return False, "pocket-full", t
            taken.add(col)
            carried = col
            continue
        if a not in _DELTA:
            return False, "bad-action", t
        dr, dc = _DELTA[a]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W or int(grid[nr, nc]) == WALL:
            return False, "into-wall", t
        dcol = int(door[nr, nc])
        if dcol != 0 and dcol not in used:
            if carried != dcol:
                return False, "locked-door-c%d" % dcol, t
            used.add(dcol)
            carried = 0                            # consumed by the door
        tcol = int(tint[nr, nc])
        if tcol != 0 and carried == tcol:
            return False, "forbidden-tint-c%d" % tcol, t
        r, c = nr, nc
        if (r, c) == (int(inst.goal[0]), int(inst.goal[1])):
            return True, "goal", t + 1
    return False, "no-goal", int(len(route))


def reason_class(reason):
    return reason.split("@")[0].split("(")[0]


def audit_redundant(pairs):
    """Check C.  `judge` and `judge_ref` agree on verdict, reason class and steps."""
    diffs = []
    for inst, route in pairs:
        ok1, why1, n1 = JUDGE(inst, route)
        ok2, why2, n2 = judge_ref(inst, route)
        if (ok1, reason_class(why1), n1) != (ok2, reason_class(why2), n2):
            diffs.append((inst.seed, route, (ok1, why1, n1), (ok2, why2, n2)))
    return diffs


# ------------------------------------------------------------------ D. oracle
def audit_oracle(insts, n_mutations=3):
    """Check D.  Optimum accepted; truncation, over-budget and single-action
    perturbations all decided crisply and identically by both judges."""
    rng = np.random.default_rng(0)
    bad = []
    for inst in insts:
        L, route = solve(inst)
        if route is None:
            bad.append((inst.seed, "instance infeasible", "", 0, 0))
            continue
        ok, why, n = JUDGE(inst, route)
        if not ok or n != L:
            bad.append((inst.seed, "optimum rejected", why, L, n))
            continue
        if JUDGE(inst, route[:-1])[0]:
            bad.append((inst.seed, "truncated optimum accepted", "", L, 0))
        pad = route + "N" * (int(inst.budget) - L + 1)
        ok3, why3, _ = JUDGE(inst, pad)
        if ok3 or reason_class(why3) != "over-budget":
            bad.append((inst.seed, "over-budget route not rejected as such", why3, L, 0))
        for _ in range(n_mutations):                          # single-action edits
            i = int(rng.integers(len(route)))
            for a in ACTIONS:
                if a == route[i]:
                    continue
                mutant = route[:i] + a + route[i + 1:]
                vm = JUDGE(inst, mutant)
                vr = judge_ref(inst, mutant)
                if (vm[0], reason_class(vm[1]), vm[2]) != (vr[0], reason_class(vr[1]), vr[2]):
                    bad.append((inst.seed, "mutant disagreement", mutant, vm, vr))
    return bad


# ------------------------------------------------------------------- fuzzing
def random_routes(inst, rng, n=8):
    """Random, truncated and optimal routes -- enough to exercise every verdict."""
    out = []
    for _ in range(n):
        L = int(rng.integers(1, int(inst.budget) + 3))
        out.append("".join(ACTIONS[int(rng.integers(5))] for _ in range(L)))
    L, opt = solve(inst)
    if opt is not None:
        out.append(opt)
        if L > 2:
            i = int(rng.integers(L))
            out.append(opt[:i] + opt[i + 1:])
    return out


def audit(insts, verbose=True):
    """Run all four checks.  Returns a dict; `exact` is True iff all four pass."""
    rng = np.random.default_rng(7)
    pairs = [(i, r) for i in insts for r in random_routes(i, rng)]
    src = audit_source()
    try:
        audit_types(pairs)
        typ = []
    except Contamination as e:                                # pragma: no cover
        typ = [str(e)]
    red = audit_redundant(pairs)
    orc = audit_oracle(insts)
    res = {"n_instances": len(insts), "n_route_pairs": len(pairs),
           "source_findings": src, "type_findings": typ,
           "redundant_disagreements": len(red), "oracle_failures": len(orc),
           "verdict_classes": dict(Counter(reason_class(JUDGE(i, r)[1]) for i, r in pairs)),
           "exact": (not src) and (not typ) and (not red) and (not orc)}
    if verbose:
        print("JUDGE AUDIT  (%d instances, %d judged routes)" % (len(insts), len(pairs)))
        print("  A source   : no float machinery in judge/_pocket ....... %s"
              % ("PASS" if not src else "FAIL %s" % src))
        print("  B types    : integer-only comparisons under tracing .... %s"
              % ("PASS" if not typ else "FAIL %s" % typ))
        print("  C redundant: judge == independently written replay ..... %s (%d/%d disagree)"
              % ("PASS" if not red else "FAIL", len(red), len(pairs)))
        print("  D oracle   : optimum / truncation / budget / mutants ... %s (%d bad)"
              % ("PASS" if not orc else "FAIL", len(orc)))
        print("  verdict classes exercised: %s" % res["verdict_classes"])
        print("  JUDGE EXACT: %s    tolerance used: None" % res["exact"])
    return res
