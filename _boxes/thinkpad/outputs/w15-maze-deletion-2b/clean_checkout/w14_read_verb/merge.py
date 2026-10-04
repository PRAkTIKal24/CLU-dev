"""THE DIAL: the READ VERB.  Two half-addresses, and a FIXED plan-space merge.

What is held fixed
------------------
The same store (``w9_maze.arms.CluRead``, unmodified), the same request-only read to
per-address latents (``chlu.core.gen_request_head.head_read``, unmodified), the frozen
encoder and the shared decoder (``w9_maze.encoder``, unmodified), the parameterless
executor and the exact integer judge (``w9_maze.render.execute``,
``w8_envscout.maze_colour.judge``, unmodified).

What moves
----------
WHERE the two halves are composed.

* ``MERGE``  -- decode each half-address SEPARATELY through the shared decoder, then
  overlay the two decoded plan maps.  Composition happens in PLAN space.
* ``AVG``    -- average the two halves' latents and decode ONCE.  This is wave 9's
  read, restated on the identical two halves.  Composition happens in LATENT space.

⭐ Both verbs consume exactly the same two latents and are rendered by exactly the same
decoder, so a margin between them is a margin on the composition operator and on
nothing else.

The two half-addresses
----------------------
They are not a new mechanism: they are the head's OWN documented "not requested"
encoding.  ``one_hot_interpolated`` reads ``NaN`` as *this factor is not requested* and
gives the factor an all-zero weight row -- "the head's version of ``-1``", in that
function's words -- which zeroes that factor's main-effect block and every pair block it
enters.  So for a request ``(m, a, t, b)``:

    half A (the AGENT half)   = read at ``(m, a, NaN, b)``
    half B (the TARGET half)  = read at ``(m, NaN, t, b)``

Half A is what the store says when it is told the maze, the agent and the budget and is
NOT told the target; half B is its mirror.  This is precisely the factorisation wave 9
measured to be correct: same-agent plus same-target training plans cover 0.997 of a
held-out route (``outputs/w9-advisor-report.md`` section 1).

⛔ Nothing here is fitted, tuned or selected per instance.  ``NaN`` is applied to a
fixed column of the request; the merge is a fixed function of two plan maps with no
free parameter, no threshold and no tolerance.  The exact integer judge does not move.

⛔ EVERY ARM gets both verbs.  A merge only the CLU is given is not a result, so the
learned baselines, the matched-bytes lookup and every charter-C control are read at the
same two half-addresses and composed by the same two operators.
"""

from __future__ import annotations

import numpy as np

__all__ = ["half_values", "overlay", "MERGE_RULES", "fold_overlay"]


def half_values(requests):
    """``(N, 4)`` value arrays for the two half-addresses of each request.

    Returns ``(V_agent_half, V_target_half)``.  Column 2 (the target word) is ``NaN``
    in the agent half; column 1 (the agent word) is ``NaN`` in the target half.  The
    maze word and the budget word are in BOTH halves -- they are not part of the
    agent/target factorisation, and dropping the maze would address a different maze.
    """
    V = np.array([r.values() for r in requests], dtype=np.float64)
    Va = V.copy()
    Va[:, 2] = np.nan
    Vt = V.copy()
    Vt[:, 1] = np.nan
    return Va, Vt


def overlay(plan_a, plan_b, rule="target_wins"):
    """THE FIXED PLAN-SPACE MERGE.  Two ``(..., H, W, PHASE)`` class maps -> one.

    Class 0 is "no action", so a plan map is a set of CONSTRAINTS on (cell, phase)
    slots and the overlay is the union of two constraint sets with a fixed,
    parameter-free conflict rule.  Wave 9's measured rule is ``target_wins``; the other
    two are its controls and are reported beside it, never instead of it.

    * ``target_wins`` -- where the target half names an action, take it; elsewhere take
      the agent half's.
    * ``agent_wins``  -- the mirror.  Wave 9 measured this at 0.42-0.57 against the
      1.0000 of ``target_wins``, which is what says the rule is load-bearing.
    * ``agent_only`` / ``target_only`` -- one half alone, so the reader can see whether
      the overlay does anything at all.
    """
    A = np.asarray(plan_a, np.int8)
    B = np.asarray(plan_b, np.int8)
    if rule == "target_wins":
        return np.where(B != 0, B, A).astype(np.int8)
    if rule == "agent_wins":
        return np.where(A != 0, A, B).astype(np.int8)
    if rule == "agent_only":
        return A
    if rule == "target_only":
        return B
    raise KeyError(rule)


#: the merge rule the arm deploys, and the controls printed beside it
MERGE_RULES = ("target_wins", "agent_wins", "agent_only", "target_only")


def fold_overlay(plans, rule="target_wins"):
    """Fold several plan maps into one by the same fixed rule, in the order given.

    Used only by the plan-space REFERENCE (whole written training items), never by an
    arm: an arm reads exactly two half-addresses.
    """
    out = None
    for p in plans:
        out = np.asarray(p, np.int8) if out is None else overlay(out, p, rule)
    return out
