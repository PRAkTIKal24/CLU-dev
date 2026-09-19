"""GYM-1: one agent on a line, with momentum, a speed limit and an uncrossable wall.

INSTRUMENT ONLY. This package is a measuring device for a later wave. Nothing in it
makes, or is evidence for, any claim about the CLU.
"""

from chlu.gym1.line1d import (
    ACTIONS,
    A_MAX,
    Episode,
    FAULT_NAMES,
    FAULT_SPEED,
    FAULT_TELEPORT,
    FAULT_WALL,
    L,
    N_ACTIONS,
    RolloutResult,
    Rules,
    V_MAX_MAX,
    can_park,
    is_free_cell,
    plan,
    positions_of,
    replay_policy,
    rollout,
    segment_hits_wall,
    step,
    step_faults,
    wall_is_uncrossable,
)

__all__ = [
    "ACTIONS", "A_MAX", "Episode", "FAULT_NAMES", "FAULT_SPEED", "FAULT_TELEPORT",
    "FAULT_WALL", "L", "N_ACTIONS", "RolloutResult", "Rules", "V_MAX_MAX",
    "can_park", "is_free_cell", "plan", "positions_of", "replay_policy", "rollout",
    "segment_hits_wall", "step", "step_faults", "wall_is_uncrossable",
]
