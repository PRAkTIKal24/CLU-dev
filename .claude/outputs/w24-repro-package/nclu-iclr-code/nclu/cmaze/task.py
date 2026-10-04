"""task."""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
from nclu.cmaze.maze_colour import (
    CARRIED,
    FLOOR,
    MOVES,
    MOVE_ORDER,
    UNTAKEN,
    USED,
    Instance,
    generate,
    solve,
)
from nclu.cmaze.harness.dynamics import State, initial_state, key_homes, step
from nclu.cmaze.harness.judge_maze import judge_ref, reason_class

ACTIONS = "NESWP"
N_DIR = 5
WORD_NAMES = ("w_agent", "w_key", "w_door", "w_tint", "w_dir")


def alphabets(k):
    return [k + 1, k + 1, k + 3, k + 1, N_DIR]


def door_word(inst, cell, ks, k):
    """``w_door`` for a target cell, with WALL/off-grid as the top symbol."""
    H, W = inst.grid.shape
    r, c = cell
    if not (0 <= r < H and 0 <= c < W) or inst.grid[r, c] != FLOOR:
        return k + 2
    d = int(inst.door[r, c])
    if d == 0:
        return 0
    return k + 1 if ks[d - 1] == USED else d


def words_of(inst, st, action, homes=None):
    """``(inst, state, action) -> the five words``.  Reads the grid, never a rule."""
    k = int(inst.n_colours)
    homes = homes if homes is not None else key_homes(inst)
    a_i = ACTIONS.index(action)
    kc = homes.get(st.pos, 0)
    w_key = kc if kc and st.ks[kc - 1] == UNTAKEN else 0
    if action == "P":
        return (st.pocket, w_key, 0, 0, a_i)
    dr, dc = MOVES[action]
    tgt = (st.pos[0] + dr, st.pos[1] + dc)
    wd = door_word(inst, tgt, st.ks, k)
    H, W = inst.grid.shape
    wt = 0
    if 0 <= tgt[0] < H and 0 <= tgt[1] < W and (inst.grid[tgt] == FLOOR):
        wt = int(inst.tint[tgt])
    return (st.pocket, w_key, wd, wt, a_i)


def key_event_alphabet(k):
    """0 = nothing, 1..k = picked up colour c, k+1..2k = consumed colour c at a door."""
    return 2 * k + 1


def outcome_of(inst, st, action, homes=None):
    """The TRUE outcome, taken from the engine's stepper.  No rule is written here."""
    homes = homes if homes is not None else key_homes(inst)
    nxt, legal, _goal = step(inst, st, action, homes)
    k = int(inst.n_colours)
    if not legal:
        return (0, 0, 0, st.pocket, st)
    disp = 0 if action == "P" else 1 + ACTIONS.index(action)
    ev = 0
    for c in range(k):
        if st.ks[c] == UNTAKEN and nxt.ks[c] == CARRIED:
            ev = 1 + c
        elif st.ks[c] == CARRIED and nxt.ks[c] == USED:
            ev = 1 + k + c
    return (1, disp, ev, nxt.pocket, nxt)


N_DISP = 5


def code_dim(k):
    return 2 + N_DISP + key_event_alphabet(k) + (k + 1)


def code_blocks(k):
    """Offsets of ``[legal | displacement | key event | pocket_next]``."""
    o = 0
    out = {}
    for name, card in (
        ("legal", 2),
        ("disp", N_DISP),
        ("event", key_event_alphabet(k)),
        ("pocket", k + 1),
    ):
        out[name] = (o, card)
        o += card
    return out


def encode_outcome(k, legal, disp, ev, pocket_next):
    v = np.zeros(code_dim(k), np.float32)
    b = code_blocks(k)
    v[b["legal"][0] + int(legal)] = 1.0
    v[b["disp"][0] + int(disp)] = 1.0
    v[b["event"][0] + int(ev)] = 1.0
    v[b["pocket"][0] + int(pocket_next)] = 1.0
    return v


def enumerate_code_alphabet(k):
    out = []
    for legal in range(2):
        for disp in range(N_DISP):
            for ev in range(key_event_alphabet(k)):
                for pk in range(k + 1):
                    out.append(encode_outcome(k, legal, disp, ev, pk))
    return np.stack(out).astype(np.float32)


def decode_codes(k, V):
    b = code_blocks(k)
    A = np.asarray(V)
    cols = []
    for name in ("legal", "disp", "event", "pocket"):
        o, c = b[name]
        cols.append(np.argmax(A[:, o : o + c], axis=1).astype(np.int64))
    return np.stack(cols, axis=1)


def apply_outcome(inst, st, legal, disp, ev):
    """Build the GENERATED ``s_{t+1}`` from an emitted outcome.  Engine-free.

    An emission whose key event is impossible against ``s_t`` (picking up a key that is
    not untaken or into a full pocket, consuming a key that is not carried) is
    INCOHERENT; the caller is told and the judge rejects.  Nothing is silently repaired.
    """
    k = int(inst.n_colours)
    if legal == 0:
        return (State(st.pos, st.ks, st.t + 1), True)
    ks = list(st.ks)
    coherent = True
    if ev != 0:
        if ev <= k:
            c = ev - 1
            if ks[c] != UNTAKEN or st.pocket != 0:
                coherent = False
            ks[c] = CARRIED
        else:
            c = ev - k - 1
            if ks[c] != CARRIED:
                coherent = False
            ks[c] = USED
    if disp == 0:
        pos = st.pos
    else:
        dr, dc = MOVES[MOVE_ORDER[disp - 1]]
        pos = (st.pos[0] + dr, st.pos[1] + dc)
    return (State(pos, tuple(ks), st.t + 1), coherent)


def judge_transition(inst, st, action, emitted, homes=None):
    """EXACT.  Integer comparison of the generated ``s_{t+1}`` against the engine's.

    ``emitted = (legal, disp, ev, pocket_next)``.  Returns
    ``(correct, reason, generated_state)``.  There is no tolerance and no metric here;
    every comparison is between small integers or tuples of them.
    """
    homes = homes if homes is not None else key_homes(inst)
    t_legal, t_disp, t_ev, t_pocket, t_state = outcome_of(inst, st, action, homes)
    e_legal, e_disp, e_ev, e_pocket = (int(x) for x in emitted)
    gen, coherent = apply_outcome(inst, st, e_legal, e_disp, e_ev)
    if not coherent:
        return (False, "incoherent-key-event", gen)
    if e_legal != t_legal:
        return (False, "legality", gen)
    if e_legal == 0:
        ok = e_pocket == t_pocket
        return (ok, "ok" if ok else "pocket", gen)
    if e_disp != t_disp:
        return (False, "displacement", gen)
    if gen.pos != t_state.pos:
        return (False, "position", gen)
    if tuple(gen.ks) != tuple(t_state.ks):
        return (False, "keystate", gen)
    if e_pocket != t_pocket:
        return (False, "pocket", gen)
    return (True, "ok", gen)


_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


def step_ref(inst, pos, carried, used, taken, action):
    """A second transition function, written from the RULES, not from `step`'s code.

    RULE A  entering a door cell of colour c requires the pocket to hold key c; the key
            is CONSUMED and the door is open from then on.
    RULE B  entering a cell tinted colour c is forbidden while the pocket holds c.
    PICKUP  needs an empty pocket and an as-yet-untaken key on the CURRENT cell.

    State is carried as ``(pos, carried, used:set, taken:set)`` -- a scalar and two sets,
    not the array-of-trits the stepper uses -- so a shared bug cannot hide in a shared
    data structure.  Returns ``(legal, pos', carried', used', taken')``.
    """
    homes = {inst.key_at[i]: i + 1 for i in range(int(inst.n_colours))}
    H, W = (int(inst.grid.shape[0]), int(inst.grid.shape[1]))
    if action == "P":
        col = homes.get(pos, 0)
        if col == 0 or col in taken or carried != 0:
            return (False, pos, carried, used, taken)
        return (True, pos, col, used, taken | {col})
    if action not in _DELTA:
        return (False, pos, carried, used, taken)
    dr, dc = _DELTA[action]
    nr, nc = (pos[0] + dr, pos[1] + dc)
    if nr < 0 or nr >= H or nc < 0 or (nc >= W) or (int(inst.grid[nr, nc]) != FLOOR):
        return (False, pos, carried, used, taken)
    ncar, nused = (carried, used)
    dcol = int(inst.door[nr, nc])
    if dcol != 0 and dcol not in used:
        if carried != dcol:
            return (False, pos, carried, used, taken)
        nused = used | {dcol}
        ncar = 0
    tcol = int(inst.tint[nr, nc])
    if tcol != 0 and ncar == tcol:
        return (False, pos, carried, used, taken)
    return (True, (nr, nc), ncar, nused, taken)


def ks_to_sets(ks):
    carried = 0
    used, taken = (set(), set())
    for i, v in enumerate(ks):
        if v == CARRIED:
            carried = i + 1
            taken.add(i + 1)
        elif v == USED:
            used.add(i + 1)
            taken.add(i + 1)
    return (carried, used, taken)


_NEAR_CACHE = {}


def _near_door(inst, cell):
    key = id(inst)
    m = _NEAR_CACHE.get(key)
    if m is None:
        m = np.zeros(inst.door.shape, bool)
        H, W = inst.door.shape
        for r, c in np.argwhere(inst.door > 0):
            m[r, c] = True
            for dr, dc in MOVES.values():
                if 0 <= r + dr < H and 0 <= c + dc < W:
                    m[r + dr, c + dc] = True
        if len(_NEAR_CACHE) > 2048:
            _NEAR_CACHE.clear()
        _NEAR_CACHE[key] = m
    return bool(m[cell])


def blind_outcome(inst, st, action, mode, homes=None):
    """A composition-blind transition predictor: OPTIMAL in a RELAXED legality
    relation, then graded by the exact transition judge.

    ``oracle``            the full composition (correct by construction)
    ``blind-A``           rule B dropped
    ``blind-B``           rule A dropped
    ``blind-colour``      colour identity collapsed: any key opens any door and a tint
                          blocks whatever is carried
    ``blind-sequential``  the two rules treated as independent STAGES: rule B is
                          evaluated against the pocket BEFORE the door consumed the key
    ``blind-factor``      rule B applied only next to a door (A x B local only)

    Because each is optimal in its own view of the world and the relaxation is a named
    flag rather than a hand-written heuristic, none of them is a straw man.
    """
    homes = homes if homes is not None else key_homes(inst)
    k = int(inst.n_colours)
    if action == "P":
        return outcome_of(inst, st, action, homes)[:4]
    if mode == "oracle":
        return outcome_of(inst, st, action, homes)[:4]
    dr, dc = MOVES[action]
    nr, nc = (st.pos[0] + dr, st.pos[1] + dc)
    H, W = inst.grid.shape
    if not (0 <= nr < H and 0 <= nc < W) or inst.grid[nr, nc] != FLOOR:
        return (0, 0, 0, st.pocket)
    ks = list(st.ks)
    pre = st.pocket
    dcol = int(inst.door[nr, nc])
    ev = 0
    if dcol and mode != "blind-B":
        if ks[dcol - 1] == USED:
            pass
        elif mode == "blind-colour":
            if pre == 0:
                return (0, 0, 0, st.pocket)
            ks[pre - 1] = USED
            ev = 1 + k + (pre - 1)
        elif pre == dcol:
            ks[dcol - 1] = USED
            ev = 1 + k + (dcol - 1)
        else:
            return (0, 0, 0, st.pocket)
    post = 0
    for i, v in enumerate(ks):
        if v == CARRIED:
            post = i + 1
    tcol = int(inst.tint[nr, nc])
    if tcol and mode != "blind-A":
        probe = pre if mode == "blind-sequential" else post
        if mode == "blind-factor" and (not _near_door(inst, (nr, nc))):
            probe = 0
        if mode == "blind-colour":
            if probe != 0:
                return (0, 0, 0, st.pocket)
        elif probe == tcol:
            return (0, 0, 0, st.pocket)
    return (1, 1 + ACTIONS.index(action), ev, post)


BLIND_MODES = ("oracle", "blind-A", "blind-B", "blind-colour", "blind-sequential", "blind-factor")


def rule_b_bites(inst, st, action, homes=None):
    """Does RULE B render THIS action ILLEGAL (rather than merely slower)?

    True iff the action is illegal under A+B and legal under A alone.  The
    transition-level reading of the two-gym law's first clause.  REPORTED COLUMN.
    """
    homes = homes if homes is not None else key_homes(inst)
    true_legal = outcome_of(inst, st, action, homes)[0]
    a_only = blind_outcome(inst, st, action, "blind-A", homes)[0]
    return bool(a_only == 1 and true_legal == 0)


def tinted_door_instance(inst, rng, p_door_tint=1.0, self_tint_prob=0.6, slack=0.1):
    k = int(inst.n_colours)
    tint = inst.tint.copy()
    doors = [tuple(map(int, p)) for p in np.argwhere(inst.door > 0)]
    rng.shuffle(doors)
    added, reverted, self_n = (0, 0, 0)
    for cell in doors:
        if rng.random() > p_door_tint:
            continue
        dcol = int(inst.door[cell])
        if rng.random() < self_tint_prob:
            col = dcol
        else:
            others = [c for c in range(1, k + 1) if c != dcol]
            col = int(others[int(rng.integers(len(others)))])
        tint[cell] = col
        probe = inst._replace_tint(tint)
        if solve(probe)[0] is None:
            tint[cell] = 0
            reverted += 1
            continue
        added += 1
        self_n += col == dcol
    probe = inst._replace_tint(tint)
    lj, _ = solve(probe)
    if lj is None:
        return (
            None,
            {"door_cells": len(doors), "tinted": 0, "reverted": reverted, "infeasible": True},
        )
    la, _ = solve(probe, rule_b=False)
    budget = lj + int(np.ceil(slack * lj))
    out = Instance(
        inst.seed,
        inst.grid,
        tint,
        inst.door,
        inst.key_at,
        inst.start,
        inst.goal,
        inst.n_colours,
        int(budget),
        int(lj),
        int(la if la is not None else lj),
    )
    return (
        out,
        {
            "door_cells": len(doors),
            "tinted": int(added),
            "self_tinted": int(self_n),
            "reverted": int(reverted),
            "opt_len_before": int(inst.opt_len),
            "opt_len_after": int(lj),
            "infeasible": False,
        },
    )


def reachable_states(inst, cap=20000):
    """BFS over ``(cell, keystate)`` through the AUDITED stepper.

    Returns ``[(prefix, State), ...]``.  Every state carries an action prefix from
    ``inst.start`` that reaches it, so every transition sampled from this set can be
    re-judged by the whole-prefix ``judge_ref`` driver as well as by ``step_ref`` --
    which is what makes acceptance check 1 cover 100 % of the bank rather than the
    reachable fraction of it.
    """
    from collections import deque

    homes = key_homes(inst)
    st0 = initial_state(inst)
    key0 = (st0.pos, tuple(st0.ks))
    seen = {key0: ""}
    order = [("", st0)]
    q = deque([(st0, "")])
    while q and len(order) < cap:
        st, pref = q.popleft()
        for a in ACTIONS:
            nxt, legal, goal = step(inst, st, a, homes)
            if not legal:
                continue
            key = (nxt.pos, tuple(nxt.ks))
            if key in seen:
                continue
            seen[key] = pref + a
            order.append((pref + a, nxt))
            if not goal:
                q.append((nxt, pref + a))
    return order


@dataclass
class TRow:
    """One transition: everything the harness needs, and nothing learned."""

    maze: int
    prefix: str
    pos: tuple
    ks: tuple
    action: str
    words: tuple
    legal: int
    disp: int
    ev: int
    pocket_next: int
    bites_b: bool

    def state(self):
        return State(self.pos, self.ks, 0)


def row_of(inst, m_idx, prefix, st, action, homes):
    w = words_of(inst, st, action, homes)
    legal, disp, ev, pn, _ = outcome_of(inst, st, action, homes)
    return TRow(
        m_idx,
        prefix,
        st.pos,
        tuple(st.ks),
        action,
        w,
        legal,
        disp,
        ev,
        pn,
        rule_b_bites(inst, st, action, homes),
    )


def transitions_for_maze(inst, m_idx, rng, per_tuple=2, cap=20000):
    homes = key_homes(inst)
    states = reachable_states(inst, cap=cap)
    cand = [(p, st, a) for p, st in states for a in ACTIONS]
    order = rng.permutation(len(cand))
    kept, seen = ([], Counter())
    for i in order:
        p, st, a = cand[int(i)]
        w = words_of(inst, st, a, homes)
        if seen[w] >= per_tuple:
            continue
        seen[w] += 1
        kept.append(row_of(inst, m_idx, p, st, a, homes))
    return (kept, len(cand))


def natural_rows_for_maze(inst, m_idx, rng, n=400, cap=20000):
    """A uniform sample of ``(reachable state, action)`` pairs -- the UNSTRATIFIED
    distribution, used only for the reported natural-frequency columns."""
    homes = key_homes(inst)
    states = reachable_states(inst, cap=cap)
    cand = [(p, st, a) for p, st in states for a in ACTIONS]
    idx = rng.permutation(len(cand))[:n]
    return [row_of(inst, m_idx, *cand[int(i)], homes) for i in idx]


def build_bank(
    seed0, n_mazes, per_tuple=2, n_natural=400, p_door_tint=1.0, self_tint_prob=0.6, verbose=False
):
    """``n_mazes`` CMAZE-T instances from consecutive generator seeds, plus transitions.

    ``generate`` is called at ITS OWN DEFAULTS and nothing is filtered on the outcome;
    the ONE change is ``tinted_door_instance`` (LOG.md D7), whose receipt is returned.
    """
    insts, rows, nat = ([], [], [])
    rng = np.random.default_rng(int(seed0) + 5551)
    s, tried, tint_rec = (int(seed0), 0, [])
    while len(insts) < n_mazes:
        base = generate(s)
        s += 1
        tried += 1
        if base is None:
            continue
        inst, rec = tinted_door_instance(base, rng, p_door_tint, self_tint_prob)
        if inst is None:
            continue
        m = len(insts)
        insts.append(inst)
        tint_rec.append(rec)
        kept, n_cand = transitions_for_maze(inst, m, rng, per_tuple=per_tuple)
        rows.extend(kept)
        nat.extend(natural_rows_for_maze(inst, m, rng, n=n_natural))
        if verbose and m % 10 == 0:
            print("    bank %d/%d mazes, %d rows" % (m, n_mazes, len(rows)), flush=True)
    meta = {
        "generator_seeds_tried": tried,
        "seed0": int(seed0),
        "n_mazes": len(insts),
        "n_rows": len(rows),
        "n_natural_rows": len(nat),
        "per_tuple": int(per_tuple),
        "door_tint": {
            "p_door_tint": float(p_door_tint),
            "self_tint_prob": float(self_tint_prob),
            "door_cells": int(sum((r["door_cells"] for r in tint_rec))),
            "tinted": int(sum((r["tinted"] for r in tint_rec))),
            "self_tinted": int(sum((r["self_tinted"] for r in tint_rec))),
            "reverted_to_preserve_the_oracle": int(sum((r["reverted"] for r in tint_rec))),
            "opt_len_mean_before": float(np.mean([r["opt_len_before"] for r in tint_rec])),
            "opt_len_mean_after": float(np.mean([r["opt_len_after"] for r in tint_rec])),
        },
    }
    return (insts, rows, nat, meta)


def rollouts_for_maze(inst, m_idx, L=None):
    _len, route = solve(inst)
    if route is None:
        return None
    acts = route if L is None else route[:L]
    return {
        "maze": m_idx,
        "actions": acts,
        "full_len": len(route),
        "budget": int(inst.budget),
        "complete": len(acts) == len(route),
    }


@dataclass
class HoldOut:
    """The hold-out specification.  Disjointness is BY CONSTRUCTION, and checked."""

    value_colour: int
    pairs: tuple
    nway: tuple


NWAY_FAMILIES = ("interaction", "rule_b", "rule_a", "open_door_tint", "free")


def _nway_candidates(k, value_colour):
    cols = [c for c in range(1, k + 1) if c != value_colour]
    OPEN = k + 1
    fam = {f: [] for f in NWAY_FAMILIES}
    for di in range(4):
        for c in cols:
            fam["interaction"].append((c, c, c, di))
            fam["rule_b"].append((c, 0, c, di))
            fam["open_door_tint"].append((c, OPEN, c, di))
            for b in cols:
                if b != c:
                    fam["rule_a"].append((c, b, 0, di))
                    fam["free"].append((c, 0, b, di))
    return fam


def make_holdout(k, seed=0, n_pairs=8, n_nway=20):
    rng = np.random.default_rng(seed + 424242)
    value_colour = int(k)
    cols = [c for c in range(1, k + 1) if c != value_colour]
    cand_pairs = []
    for i, j in ((0, 2), (0, 3)):
        for a in cols:
            for b in cols:
                cand_pairs.append((i, a, j, b))
    rng.shuffle(cand_pairs)
    pairs = tuple((tuple((int(x) for x in p)) for p in cand_pairs[:n_pairs]))
    fam = _nway_candidates(k, value_colour)
    per = max(1, n_nway // len(NWAY_FAMILIES))
    nway = []
    for f in NWAY_FAMILIES:
        c = list(fam[f])
        rng.shuffle(c)
        nway.extend(c[:per])
    nway = tuple((tuple((int(x) for x in c)) for c in nway[:n_nway]))
    return HoldOut(value_colour, pairs, nway)


def _has_colour(words, k, c):
    """Does this transition mention colour ``c`` in ANY rule-bearing role?"""
    wa, wk, wd, wt, _ = words
    return wa == c or wk == c or wt == c or (wd == c)


def row_class(row, ho, k):
    """Which split a transition belongs to.  Priority: value > n-way > 2-way > seen."""
    w = row.words
    if _has_colour(w, k, ho.value_colour):
        return "value"
    key4 = (w[0], w[2], w[3], w[4])
    if key4 in ho.nway:
        return "nway"
    for i, a, j, b in ho.pairs:
        if w[i] == a and w[j] == b:
            return "pair"
    return "seen"


def split_rows(rows, ho, k):
    out = defaultdict(list)
    for r in rows:
        out[row_class(r, ho, k)].append(r)
    return dict(out)


def disjointness_report(train_rows, eval_splits, ho, k):
    """CHECK 3.  The hold-outs are disjoint from train BY CONSTRUCTION -- measured."""
    seen_colour = sum((1 for r in train_rows if _has_colour(r.words, k, ho.value_colour)))
    train_nway = set(((r.words[0], r.words[2], r.words[3], r.words[4]) for r in train_rows))
    nway_leak = [t for t in ho.nway if t in train_nway]
    pair_leak = []
    for i, a, j, b in ho.pairs:
        n = sum((1 for r in train_rows if r.words[i] == a and r.words[j] == b))
        if n:
            pair_leak.append(((i, a, j, b), n))
    mis = {
        name: int(sum((1 for r in rs if row_class(r, ho, k) != name)))
        for name, rs in eval_splits.items()
    }
    return {
        "value_holdout_colour": int(ho.value_colour),
        "n_train_rows": len(train_rows),
        "train_rows_mentioning_the_held_out_colour": int(seen_colour),
        "n_nway_combos_held_out": len(ho.nway),
        "nway_combos_present_in_train": len(nway_leak),
        "n_pair_combos_held_out": len(ho.pairs),
        "pair_combos_present_in_train": len(pair_leak),
        "pair_leak_detail": [[list(p), int(n)] for p, n in pair_leak],
        "rows_misclassified_into_their_own_split": mis,
        "distinct_word_tuples_in_train": len(set((r.words for r in train_rows))),
        "pass": bool(
            seen_colour == 0
            and (not nway_leak)
            and (not pair_leak)
            and all((v == 0 for v in mis.values()))
        ),
    }


def replay_audit(insts, rows):
    n, bad = (0, [])
    for r in rows:
        inst = insts[r.maze]
        st = r.state()
        n += 1
        car, used, taken = ks_to_sets(r.ks)
        ok_b, pos_b, car_b, used_b, taken_b = step_ref(inst, r.pos, car, used, taken, r.action)
        if int(ok_b) != int(r.legal):
            bad.append([r.maze, r.prefix, r.action, "step_ref legality"])
            continue
        if r.legal:
            true_state = outcome_of(inst, st, r.action)[4]
            car2, used2, taken2 = ks_to_sets(true_state.ks)
            if (pos_b, car_b, used_b, taken_b) != (true_state.pos, car2, used2, taken2):
                bad.append([r.maze, r.prefix, r.action, "step_ref state"])
                continue
            if car_b != r.pocket_next:
                bad.append([r.maze, r.prefix, r.action, "step_ref pocket"])
                continue
        elif car != r.pocket_next:
            bad.append([r.maze, r.prefix, r.action, "illegal pocket moved"])
            continue
        lifted = Instance(
            inst.seed,
            inst.grid,
            inst.tint,
            inst.door,
            inst.key_at,
            inst.start,
            inst.goal,
            inst.n_colours,
            10**9,
            inst.opt_len,
            inst.opt_len_a_only,
        )
        route = r.prefix + r.action
        okj, whyj, stepj = judge_ref(lifted, route)
        blames_here = not okj and reason_class(whyj) != "no-goal" and (stepj == len(r.prefix))
        reached_goal_earlier = okj and stepj <= len(r.prefix)
        if r.legal and blames_here:
            bad.append([r.maze, r.prefix, r.action, "judge_ref says illegal"])
        if not r.legal and (not blames_here) and (not reached_goal_earlier):
            bad.append([r.maze, r.prefix, r.action, "judge_ref says legal"])
    return {
        "transitions_replayed": n,
        "disagreements": len(bad),
        "detail": bad[:20],
        "pass": not bad,
        "drivers": [
            "step_ref (rules-written, set-based state, no array of trits)",
            "judge_ref (the second replay, driven from inst.start over prefix+action; it never sees s_t)",
        ],
    }


def words_determine_transition(rows):
    """CHECK 3b.  The five words determine ``(legal, disp, event, pocket)`` exactly."""
    m, clash = ({}, 0)
    for r in rows:
        key = r.words
        val = (r.legal, r.disp, r.ev, r.pocket_next)
        if key in m and m[key] != val:
            clash += 1
        m[key] = val
    return {
        "distinct_word_tuples": len(m),
        "rows": len(rows),
        "word_tuples_with_two_outcomes": int(clash),
        "pass": clash == 0,
    }


def impossible_example_audit(insts, rows):
    n, bad = (0, 0)
    for r in rows:
        ok = judge_transition(
            insts[r.maze], r.state(), r.action, (r.legal, r.disp, r.ev, r.pocket_next)
        )[0]
        n += 1
        bad += not ok
    return {
        "rows_checked": int(n),
        "rows_with_no_accepting_emission": int(bad),
        "pass": bad == 0,
        "definition": "an example is IMPOSSIBLE iff no emission exists that the exact transition judge accepts; this audit constructs the accepting emission for every row and checks the judge takes it",
    }


def rule_b_column(rows):
    """CHECK 4.  Reported, never a gate."""
    n = len(rows)
    bites = sum((1 for r in rows if r.bites_b))
    moves = [r for r in rows if r.action != "P"]
    illegal = sum((1 for r in rows if r.legal == 0))
    inter = sum(
        (
            1
            for r in rows
            if r.words[4] < 4
            and r.words[0] != 0
            and (r.words[2] == r.words[0])
            and (r.words[3] == r.words[0])
        )
    )
    return {
        "n_transitions": int(n),
        "rule_b_renders_the_action_ILLEGAL": int(bites),
        "fraction": bites / max(n, 1),
        "fraction_of_move_actions": bites / max(len(moves), 1),
        "n_move_actions": len(moves),
        "illegal_fraction_overall": illegal / max(n, 1),
        "n_door_eq_pocket_eq_tint": int(inter),
        "note": "fraction of transitions the engine makes ILLEGAL under RULES A+B that would be LEGAL under RULE A alone.  REPORTED COLUMN, never an abort condition.",
    }
