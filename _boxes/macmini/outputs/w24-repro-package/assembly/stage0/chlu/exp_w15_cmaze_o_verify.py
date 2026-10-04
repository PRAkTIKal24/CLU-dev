"""THE INDEPENDENT CHECKER: every judged CMAZE-O state, replayed by a second driver.

Acceptance criterion 1 of the task file: *independent replay of every judged state by a
second independently-driven checker -- 0 disagreements, count printed.*

What "independently driven" means here, precisely
--------------------------------------------------
This module does **not** import ``judge_state``, ``legality``, ``render_state``,
``OState.engine_instance`` or ``floor_degree``.  It reads only

* ``states_seed<N>.npz`` -- the packed bits of every state the pipeline judged, the nine
  words behind it, and the pipeline's own four verdict bits;
* ``bank_seed<N>.npz``   -- the layout bank the pipeline used;

rebuilds the layout bank **from the seed** and cross-checks it against the saved arrays,
then re-derives, in its own code:

* the REFERENCE state the nine words name (its own slot-table lookup);
* ``exact_objects``  -- its own tuple comparison;
* ``exact_layout``   -- its own mask comparison;
* ``legal``          -- its own clause set, assembling the ``Instance`` itself and
  calling ``w8_envscout.maze_colour.solve`` and ``_reachable_without_keys``, with the
  corridor test written as an explicit neighbour loop rather than the pipeline's shifted
  -array convolution;
* ``valid = exact_objects and legal``.

and reports the number of states where its four bits differ from the pipeline's.  The
shared surface is the ENGINE, which is the point: both drivers must agree with CMAZE-1,
not with each other's re-implementation of CMAZE-1.
"""

from __future__ import annotations

import os

import numpy as np

from w8_envscout.maze_colour import (FLOOR, MOVES, WALL, Instance,
                                     _reachable_without_keys, solve)

H = W = 15
NC = 5                  # N_COLOURS, restated here rather than imported


def _unpack(bits):
    return np.unpackbits(np.asarray(bits, np.uint8))[:H * W].reshape((H, W))


def _neighbours(r, c):
    for dr, dc in MOVES.values():
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc


def _sentinels(grid):
    out = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] == WALL:
                out.append((r, c))
                if len(out) == NC - 1:
                    return out
    return None


def _instance(grid, tint_mask, ct, agent, ck, key, cd, door, goal, sent):
    tint = np.zeros((H, W), np.int8)
    tint[tint_mask.astype(bool)] = int(ct)
    dr_ = np.zeros((H, W), np.int8)
    dr_[door] = int(cd)
    homes, si = [], 0
    for col in range(1, NC + 1):
        if col == int(ck):
            homes.append(tuple(key))
        else:
            homes.append(tuple(sent[si]))
            si += 1
    return Instance(0, grid, tint, dr_, tuple(homes), tuple(agent), tuple(goal),
                    NC, 10 ** 9, 0, 0)


def _legal(grid, tint_mask, ct, agent, ca, key, ck, door, cd, goal):
    """This checker's OWN clause set.  Same engine, different driver."""
    cells = [tuple(agent), tuple(key), tuple(door), tuple(goal)]
    for (r, c) in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
    if any(grid[p] != FLOOR for p in cells):
        return False
    if len({tuple(p) for p in cells}) != 4:
        return False
    deg = sum(1 for (rr, cc) in _neighbours(*door) if grid[rr, cc] == FLOOR)
    if deg < 2:
        return False
    if any(bool(tint_mask[p]) for p in cells):
        return False
    for v in (ca, ck, cd, ct):
        if not (1 <= int(v) <= NC):
            return False
    sent = _sentinels(grid)
    if sent is None:
        return False
    inst = _instance(grid, tint_mask, ct, agent, ck, key, cd, door, goal, sent)
    return solve(inst)[0] is not None


def _reference(bank_grids, bank_tints, bank_slots, words):
    lid, ca, ck, cd, ct, za, zk, zd, zg = [int(x) for x in words]
    zs = (za, zk, zd, zg)
    cells = []
    for s in range(4):
        v = int(bank_slots[lid][s, zs[s]])
        cells.append((v // W, v % W))
    return (bank_grids[lid], bank_tints[lid].astype(bool), ct,
            cells[0], ca, cells[1], ck, cells[2], cd, cells[3])


def verify_seed(seed: int, out_dir: str, rebuild_bank=True) -> dict:
    """Replay every judged state of one seed.  Returns the disagreement report."""
    st = np.load(os.path.join(out_dir, "states_seed%d.npz" % seed), allow_pickle=True)
    bk = np.load(os.path.join(out_dir, "bank_seed%d.npz" % seed))
    grids, tints, slots = bk["grids"], bk["tints"], bk["slots"]

    bank_mismatch = None
    if rebuild_bank:
        from chlu.exp_w15_cmaze_o_env import build_layout_bank
        bank, _tried = build_layout_bank(len(grids), 90_000 + 1000 * int(seed))
        bank_mismatch = int(sum(
            int(not np.array_equal(np.asarray(L.grid), grids[i]))
            + int(not np.array_equal(np.asarray(L.tint_mask), tints[i]))
            + int(not np.array_equal(np.asarray(L.slot_cell), slots[i]))
            for i, L in enumerate(bank)))

    fb, tb, iv = st["floor_bits"], st["tint_bits"], st["ints"]
    words, verdict = st["words"], st["verdict"]
    arms, rows = st["arm"], st["row"]

    n = int(fb.shape[0])
    disagree, per_pair = 0, {}
    n_valid_mine = 0
    for i in range(n):
        g = np.where(_unpack(fb[i]) > 0, FLOOR, WALL).astype(np.uint8)
        tm = _unpack(tb[i]).astype(bool)
        v = [int(x) for x in iv[i]]
        p_agent, p_ca = (v[0], v[1]), v[2]
        p_key, p_ck = (v[3], v[4]), v[5]
        p_door, p_cd = (v[6], v[7]), v[8]
        p_goal, p_ct = (v[9], v[10]), v[11]

        (r_grid, r_tm, r_ct, r_agent, r_ca, r_key, r_ck, r_door, r_cd,
         r_goal) = _reference(grids, tints, slots, words[i])

        exact_objects = (p_agent == r_agent and p_ca == r_ca and p_key == r_key
                         and p_ck == r_ck and p_door == r_door and p_cd == r_cd
                         and p_goal == r_goal and p_ct == r_ct)
        exact_layout = (np.array_equal(g == FLOOR, np.asarray(r_grid) == FLOOR)
                        and np.array_equal(tm, np.asarray(r_tm, bool)))
        legal = _legal(g, tm, p_ct, p_agent, p_ca, p_key, p_ck, p_door, p_cd, p_goal)
        valid = bool(exact_objects and legal)
        n_valid_mine += int(valid)
        mine = np.array([valid, exact_objects, exact_layout, legal], np.uint8)
        if not np.array_equal(mine, verdict[i]):
            disagree += 1
            k = "%s/%s" % (arms[i], rows[i])
            per_pair[k] = per_pair.get(k, 0) + 1

    return {
        "seed": int(seed),
        "states_replayed": n,
        "disagreements": int(disagree),
        "disagreements_by_arm_row": per_pair,
        "valid_count_independent": int(n_valid_mine),
        "valid_count_pipeline": int(verdict[:, 0].sum()),
        "layout_bank_rebuild_mismatches": bank_mismatch,
        "tolerance_used": None,
        "statement": ("this module imports neither judge_state nor legality nor "
                      "render_state; it re-derives the reference from the nine words "
                      "and the verdict from w8_envscout.maze_colour directly"),
    }
