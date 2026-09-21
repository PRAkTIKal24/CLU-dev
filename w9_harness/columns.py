"""The composition-blind REPORTED COLUMN.

A composition-blind planner is env-scout's mechanical definition: a planner that is
OPTIMAL within a RELAXED legality relation, and is then graded by the UNRELAXED
exact judge.  Because each blind planner is optimal in its own view of the world, it
cannot be dismissed as a straw man, and because the relaxation is a flag on the
solver rather than a hand-written heuristic, it cannot drift.

These are COLUMNS, not gates.  If composition-blind scores high on a run, that is a
number about the distribution which goes in the table; nothing in this harness
aborts on it, and nothing in this harness filters instances to push it down.  The
same goes for fragment lookup in `tabular.py`.
"""

from __future__ import annotations

from w8_envscout.maze_colour import solve
from w8_envscout.screen import key_order_of

# env-scout's relaxations, unchanged.
BLIND = {
    "oracle (full composition)": dict(),
    "blind-A (rule B dropped)": dict(rule_b=False),
    "blind-B (rule A dropped)": dict(rule_a=False),
    "blind-colour (colours collapsed)": dict(colour_blind=True),
    "blind-factor (A x B local only)": dict(tint_local_only=True),
    "blind-sequential (key order from A)": "sequential",
}

# The single headline "composition-blind" number: the strongest blind planner.
HEADLINE = "blind-factor (A x B local only)"


def plan(inst, spec):
    if spec == "sequential":
        _, ra = solve(inst, rule_b=False)
        if ra is None:
            return None
        return solve(inst, key_order=key_order_of(inst, ra))[1]
    return solve(inst, **spec)[1]


def routes_for(insts, name):
    return [plan(i, BLIND[name]) or "" for i in insts]


def all_columns(insts):
    return {name: [plan(i, spec) or "" for i in insts] for name, spec in BLIND.items()}
