"""⭐ **C2W10 — the THREE-STATE store lifecycle.** STUB (kill-conditions first).

**This file is intentionally unimplemented at this commit.** Standing doctrine
§A12: *build the kill-condition before the thing it can kill.* The designed
negatives of L1-L5 land in ``tests/test_store_lifecycle.py`` **first**, RED,
against this stub; the verbs are implemented in the next commit and the same
tests go green without being edited. C2W8 killed a stage-2 build for hours of
measurement by doing exactly this, twice (the vacuous ``M``, the
addressability-blind gate), which is why the order is part of the acceptance.

The states are **PROTECTED <-> ACTIVE -> TRASH** (charter Add.12 §A34.3):

``PROTECTED``
    no decay (``leak = 0``, the allocator's existing permanent flag).
``ACTIVE``
    the designed decay applies.
``TRASH``
    routed to ``gamma_phi(q)`` via ``CluSystem.trash_route`` (C2W8's K2 region,
    whose first experimental use ON this is).

⛔ **Demotion is PROTECTED -> ACTIVE, NEVER to trash** (§A34.3). Trash is the
never-useful / spurious route only.
⛔ **Depth never enters the usefulness criterion** (§A28.3(ii): depth != useful).
"""

from __future__ import annotations

_NOT_YET = (
    "chlu.core.store_lifecycle is a STUB at this commit: the designed negatives "
    "land before the verbs they can kill (doctrine §A12)."
)

#: the three lifecycle states
PROTECTED = "PROTECTED"
ACTIVE = "ACTIVE"
TRASH = "TRASH"
STATES = (PROTECTED, ACTIVE, TRASH)


class LifecycleParams:  # pragma: no cover - stub
    def __init__(self, *a, **k):
        raise NotImplementedError(_NOT_YET)

    @classmethod
    def from_config(cls, group):
        raise NotImplementedError(_NOT_YET)


class StoreLifecycle:  # pragma: no cover - stub
    def __init__(self, *a, **k):
        raise NotImplementedError(_NOT_YET)


class ProtectedSaturationMonitor:  # pragma: no cover - stub
    name = "protected_saturation"

    def __init__(self, *a, **k):
        raise NotImplementedError(_NOT_YET)


def promotion_dwell(hits_by_chunk, chunk, params):  # pragma: no cover - stub
    raise NotImplementedError(_NOT_YET)


def should_promote(hits_by_chunk, chunk, params):  # pragma: no cover - stub
    raise NotImplementedError(_NOT_YET)


def should_demote(hits_by_chunk, chunk, params):  # pragma: no cover - stub
    raise NotImplementedError(_NOT_YET)


def should_trash(hits_by_stream, first_seen_stream, stream, params):  # pragma: no cover
    raise NotImplementedError(_NOT_YET)


def refresh_factor(depth_before, depth_after, params=None, max_gain=4.0):  # pragma: no cover
    raise NotImplementedError(_NOT_YET)


def guarded_rewrite(system, item_id, address, payload, key, params):  # pragma: no cover
    raise NotImplementedError(_NOT_YET)


def replay_rewrite_events(events, params):  # pragma: no cover - stub
    raise NotImplementedError(_NOT_YET)


def cumulative_decay(controller):  # pragma: no cover - stub
    raise NotImplementedError(_NOT_YET)


def net_depth(raw, cum_factor):  # pragma: no cover - stub
    raise NotImplementedError(_NOT_YET)


__all__ = [
    "PROTECTED", "ACTIVE", "TRASH", "STATES",
    "LifecycleParams", "StoreLifecycle", "ProtectedSaturationMonitor",
    "promotion_dwell", "should_promote", "should_demote", "should_trash",
    "refresh_factor", "guarded_rewrite", "replay_rewrite_events",
    "cumulative_decay", "net_depth",
]
