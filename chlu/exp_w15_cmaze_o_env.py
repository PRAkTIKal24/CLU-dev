"""CMAZE-O -- the compositional OBSERVATION-GENERATION instrument.

What this is
------------
CMAZE-1 (``w8_envscout.maze_colour``) is a colour-keyed maze with an exact integer
judge.  CMAZE-2 / CMAZE-2b (``w9_maze.task``, ``w14_request_layer.task2``) are REQUEST
layers over it whose answer is a ROUTE.  **CMAZE-O is the world model's OBSERVATION
MODEL**: the answer is not a route, it is *the rendered grid state itself*.

A request is nine words::

    layout id  x  agent colour  x  key colour  x  door colour  x  tinted-floor colour
               x  agent cell    x  key cell    x  door cell    x  goal cell

and the answer is the grid state those words name.  The judge is **exact placement of
every object AND legality of the state under the engine's own rules**.  There is no
tolerance anywhere: every comparison in this file is between small integers or between
boolean masks.  ``tolerance_used`` is ``None`` on every row the harness emits.

Engine and judge are IMPORTED UNMODIFIED from CMAZE-1
-----------------------------------------------------
``_carve_maze``, ``_braid``, ``MOVES``, ``WALL``, ``FLOOR``, ``Instance``, ``solve``
and ``_reachable_without_keys`` come from ``w8_envscout.maze_colour``.  This module adds
no rule to the engine and changes none.  Legality of a CMAZE-O state is decided by
calling the engine, never by re-implementing it.

The vocabulary, and why the cells are ZONE words
------------------------------------------------
A cell cannot be a raw ``(r, c)`` pair: a coordinate is not a word, it does not compose,
and "unseen 2-way over the vocabulary" would be meaningless.  So the 15x15 grid carries
a fixed 3x3 arrangement of 5x5 **zones**, and a cell word is a zone word ``z0..z8``.
Inside a layout each zone designates **four distinct cells, one per object slot**
(agent / key / door / goal), chosen deterministically by distance to the zone centre.
The door's designated cell is additionally required to be a **corridor** cell (>= 2 floor
neighbours), which is the engine-side reading of "a door on a corridor".

That makes ``layout x zone`` a genuine interaction -- the same zone word resolves to a
different cell in every layout -- which is exactly the composition a blind arm cannot do
and is what the BLIND columns in ``exp_w15_cmaze_o_arms`` are blind to.

The five-colour vocabulary and the VALUE hold-out
--------------------------------------------------
``N_COLOURS = 5``.  Four colours are trained; **colour 5 is the VALUE hold-out** -- it
never appears in any training request in any of the four colour slots.  The engine is
therefore run at ``n_colours = 5``.

THE SENTINEL-KEY DEVICE, declared
----------------------------------
A CMAZE-O state carries exactly ONE key, but ``w8_envscout.maze_colour.Instance``
requires a ``key_at`` home cell for every one of the ``n_colours`` colours.  The four
colours that are not the state's key colour are given homes on **WALL cells**.  This is
inert and provably so: ``solve`` and ``judge`` consult ``key_cell`` only at positions the
walker occupies, a walker never occupies a WALL cell (both functions refuse
``g[nr, nc] == WALL`` before moving), so a sentinel key can never be picked up.
:func:`sentinels_are_inert` asserts it for real, on real states.

A CMAZE-O state's legality never mentions the sentinels: the state has one key and
"a key never on a wall" is a statement about THAT key.

THE ONE DECLARED DEVIATION FROM CMAZE-1's ADOPTED CONFIGURATION
---------------------------------------------------------------
``braid = 0.08`` instead of CMAZE-1's adopted ``0.40``, and the door's designated cell
in each zone is the **balanced articulation point** of the floor graph rather than just
any corridor cell.  Both changes exist for one reason and it is measured, not asserted:

    braid   zones with a balanced-articulation door    agent/goal separated by the door
    0.40    0.31 / 1                                   0.0208
    0.25    0.56 / 1                                   0.0693
    0.15    0.77 / 1                                   0.1621
    0.08    0.93 / 1                                   0.2914

(``$CHLU_HUB_OUTPUTS/w15-cmaze-o-harness/artefacts/braid_probe.json``.)  CMAZE-1 adopted
0.40 because its task emits a ROUTE and its adversarial tint placement needs route
MULTIPLICITY to work.  CMAZE-O emits a STATE, never a route, and uses no adversarial
placement; what it needs instead is that the door is capable of being **load-bearing**,
which is CMAZE-1's own adoption condition ("the doors are load-bearing").  At 0.40 the
door cuts the agent from the goal in 2 % of requests and RULES A and B are wallpaper.
Nothing in the engine, the solver or the judge is touched by this: ``braid`` is an
argument of ``_braid``, called with a different number.

Notation (COMMON.md): **n = factors jointly composed** (here n = 9), **K = wells**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collections import deque

from w8_envscout.maze_colour import (FLOOR, MOVES, WALL, Instance, _braid,
                                     _carve_maze, _reachable_without_keys, solve)

# ------------------------------------------------------------------ constants
H = W = 15                      # the adopted CMAZE-1 geometry: h = w = 7 -> 2*7+1
#: CMAZE-O's braid fraction.  CMAZE-1 adopted 0.40; see THE ONE DECLARED DEVIATION.
BRAID = 0.08
BRAID_CMAZE1_ADOPTED = 0.40
P_TINT = 0.22                   # the adopted tint density
N_COLOURS = 5                   # 4 trained + 1 VALUE hold-out
VALUE_HOLDOUT_COLOUR = 5        # the colour word that never appears in training
N_ZONES = 9
ZONE_SIDE = 5
SLOTS = ("agent", "key", "door", "goal")
N_SLOTS = 4

#: item-tensor channel layout.  COLOUR IS A SHARED FIELD, NOT A PER-SLOT BLOCK.
#:
#: A per-slot colour block (5 channels per object type) would make "a colour never seen
#: in THIS slot" an output channel that was never once positive, so no arm -- not even
#: the autoencoder reading the true picture -- could ever render it, and the whole
#: family of slot-value hold-outs would be identically zero for structural reasons that
#: have nothing to do with composition.  Factoring colour OUT of the slot fixes that:
#: the five colour channels are ONE field shared by the tint, the door, the key and the
#: agent, so a colour trained on a key can be rendered on an agent, and
#: ``value_slot_holdout`` becomes a real composition question.
CH_FLOOR = 0
CH_TINT = 1                     # the tinted-floor MASK
CH_DOOR = 2
CH_KEY = 3
CH_AGENT = 4
CH_GOAL = 5
CH_COL = 6                      # 6 .. 10, the shared colour field, one-hot
N_CH = 6 + N_COLOURS            # 11

#: the nine words, in the order every design matrix and every arm sees them
FACTORS = ("layout", "agent_colour", "key_colour", "door_colour", "tint_colour",
           "agent_zone", "key_zone", "door_zone", "goal_zone")
COLOUR_FACTORS = (1, 2, 3, 4)
ZONE_FACTORS = (5, 6, 7, 8)


def zone_of(r: int, c: int) -> int:
    return 3 * (r // ZONE_SIDE) + (c // ZONE_SIDE)


def _zone_centre(z: int):
    return (ZONE_SIDE * (z // 3) + ZONE_SIDE // 2, ZONE_SIDE * (z % 3) + ZONE_SIDE // 2)


def floor_degree(grid: np.ndarray) -> np.ndarray:
    """Number of FLOOR neighbours of every cell, using the engine's own MOVES."""
    g = (np.asarray(grid) == FLOOR)
    deg = np.zeros(g.shape, np.int8)
    for dr, dc in MOVES.values():
        sh = np.zeros_like(g)
        r0, r1 = max(0, -dr), min(g.shape[0], g.shape[0] - dr)
        c0, c1 = max(0, -dc), min(g.shape[1], g.shape[1] - dc)
        sh[r0:r1, c0:c1] = g[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        deg += sh.astype(np.int8)
    return deg * g.astype(np.int8)


def floor_components(grid, cut=None):
    """Connected components of the FLOOR with one cell optionally removed."""
    g = np.asarray(grid)
    hh, ww = g.shape
    seen = set() if cut is None else {tuple(cut)}
    out = []
    for r in range(hh):
        for c in range(ww):
            if g[r, c] != FLOOR or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cur = {(r, c)}
            while q:
                y, x = q.popleft()
                for dr, dc in MOVES.values():
                    ny, nx = y + dr, x + dc
                    if (0 <= ny < hh and 0 <= nx < ww and g[ny, nx] == FLOOR
                            and (ny, nx) not in seen):
                        seen.add((ny, nx))
                        q.append((ny, nx))
                        cur.add((ny, nx))
            out.append(cur)
    return out


def cut_balance(grid, cell) -> int:
    """Size of the SECOND-largest floor component after removing ``cell``.

    0 means the cell is not an articulation point at all; larger means its removal
    splits the maze more evenly, i.e. a door there separates more of the world.
    """
    cs = floor_components(grid, cell)
    if len(cs) < 2:
        return 0
    return int(sorted((len(c) for c in cs), reverse=True)[1])


# ------------------------------------------------------------------ the layout bank
@dataclass
class Layout:
    """One maze in the FIXED BANK OF LAYOUTS SEEN IN TRAINING."""

    lid: int
    seed: int
    grid: np.ndarray                 # (15, 15) uint8 WALL/FLOOR
    tint_mask: np.ndarray            # (15, 15) bool -- WHICH cells are tinted (fixed)
    slot_cell: np.ndarray            # (4, 9) int32, [slot, zone] -> r*W+c
    sentinels: tuple                 # 4 WALL cells, the inert key homes

    def cell(self, slot: int, zone: int):
        v = int(self.slot_cell[slot, zone])
        return (v // W, v % W)


def build_layout(lid: int, seed: int):
    """One layout, or ``None`` if the carved maze cannot furnish every zone.

    ``_carve_maze`` and ``_braid`` are CMAZE-1's, called at CMAZE-1's adopted
    configuration.  Nothing about the maze itself is new here.
    """
    rng = np.random.default_rng(seed)
    g = _carve_maze(7, 7, rng)
    _braid(g, rng, BRAID)
    deg = floor_degree(g)
    slot_cell = np.zeros((N_SLOTS, N_ZONES), np.int32)
    chosen = []
    for z in range(N_ZONES):
        r0, c0 = ZONE_SIDE * (z // 3), ZONE_SIDE * (z % 3)
        cr, cc = _zone_centre(z)
        cells = [(r, c) for r in range(r0, r0 + ZONE_SIDE)
                 for c in range(c0, c0 + ZONE_SIDE) if g[r, c] == FLOOR]
        if len(cells) < N_SLOTS:
            return None
        cells.sort(key=lambda p: (abs(p[0] - cr) + abs(p[1] - cc), p[0], p[1]))
        corridor = [p for p in cells if deg[p] >= 2]
        if not corridor:
            return None
        #: the door goes on the BALANCED ARTICULATION POINT of the floor graph -- the
        #: corridor cell whose removal leaves the largest second component.  That is
        #: the cell at which a door can actually be load-bearing.
        door = max(corridor, key=lambda p: (cut_balance(g, p),
                                            -(abs(p[0] - cr) + abs(p[1] - cc)),
                                            -p[0], -p[1]))
        rest = [p for p in cells if p != door][:N_SLOTS - 1]
        if len(rest) < N_SLOTS - 1:
            return None
        assign = {"door": door, "agent": rest[0], "key": rest[1], "goal": rest[2]}
        for si, s in enumerate(SLOTS):
            p = assign[s]
            slot_cell[si, z] = p[0] * W + p[1]
            chosen.append(p)
    # the tint pattern: CMAZE-1's own Bernoulli scatter, with every designated slot
    # cell excluded exactly as generate() clears start/goal/doors/keys.
    reserved = set(chosen)
    tint_mask = np.zeros((H, W), bool)
    for r in range(H):
        for c in range(W):
            if g[r, c] == FLOOR and (r, c) not in reserved and rng.random() < P_TINT:
                tint_mask[r, c] = True
    walls = [(r, c) for r in range(H) for c in range(W) if g[r, c] == WALL]
    if len(walls) < N_COLOURS - 1:
        return None
    return Layout(lid, seed, g, tint_mask, slot_cell, tuple(walls[:N_COLOURS - 1]))


def build_layout_bank(n_layouts: int, seed0: int):
    """``n_layouts`` layouts from consecutive seeds, plus how many seeds were tried."""
    out, s, tried = [], int(seed0), 0
    while len(out) < n_layouts:
        L = build_layout(len(out), s)
        s += 1
        tried += 1
        if L is not None:
            out.append(L)
    return out, tried


# ------------------------------------------------------------------ the request
@dataclass(frozen=True)
class ORequest:
    """Nine words.  No cell, no colour channel, no image."""

    layout: int
    ca: int          # agent colour, 1..C
    ck: int          # key colour
    cd: int          # door colour
    ct: int          # tinted-floor colour
    za: int          # agent zone, 0..8
    zk: int
    zd: int
    zg: int

    def values(self):
        return (float(self.layout), float(self.ca), float(self.ck), float(self.cd),
                float(self.ct), float(self.za), float(self.zk), float(self.zd),
                float(self.zg))

    def words(self):
        return (self.layout, self.ca, self.ck, self.cd, self.ct,
                self.za, self.zk, self.zd, self.zg)


# ------------------------------------------------------------------ the state
@dataclass
class OState:
    """A rendered CMAZE-O grid state.  Everything here is an integer or a mask."""

    grid: np.ndarray            # (15,15) uint8 WALL/FLOOR
    tint_mask: np.ndarray       # (15,15) bool
    ct: int                     # the tint colour carried by every tinted cell
    agent: tuple
    ca: int
    key: tuple
    ck: int
    door: tuple
    cd: int
    goal: tuple

    def engine_instance(self, sentinels, budget=10 ** 9) -> Instance:
        """The CMAZE-1 ``Instance`` for this state.  See the sentinel-key device."""
        tint = np.zeros((H, W), np.int8)
        tint[self.tint_mask] = self.ct
        door = np.zeros((H, W), np.int8)
        door[self.door] = self.cd
        spare = list(sentinels)[:N_COLOURS - 1]
        homes, si = [], 0
        for c in range(1, N_COLOURS + 1):
            if c == self.ck:
                homes.append(tuple(self.key))
            else:
                homes.append(tuple(spare[si]))
                si += 1
        return Instance(0, self.grid, tint, door, tuple(homes), tuple(self.agent),
                        tuple(self.goal), N_COLOURS, int(budget), 0, 0)

    def key_tuple(self):
        """The judged content: every object's placement and every colour word."""
        return (tuple(self.agent), int(self.ca), tuple(self.key), int(self.ck),
                tuple(self.door), int(self.cd), tuple(self.goal), int(self.ct))


def render_state(layout: Layout, req: ORequest) -> OState:
    """THE REFERENCE STATE the nine words name.  Deterministic; no randomness."""
    return OState(grid=layout.grid, tint_mask=layout.tint_mask, ct=int(req.ct),
                  agent=layout.cell(0, req.za), ca=int(req.ca),
                  key=layout.cell(1, req.zk), ck=int(req.ck),
                  door=layout.cell(2, req.zd), cd=int(req.cd),
                  goal=layout.cell(3, req.zg))


# ------------------------------------------------------------------ legality
def _wall_sentinels(grid: np.ndarray):
    """The first ``N_COLOURS-1`` WALL cells in row-major order of a grid, or None."""
    out = []
    g = np.asarray(grid)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == WALL:
                out.append((r, c))
                if len(out) == N_COLOURS - 1:
                    return tuple(out)
    return None


def legality(st: OState) -> dict:
    """Is this a LEGAL CMAZE-O state under the engine's rules?  Exact, all-integer.

    Clauses, in the task file's own words:

    * ``on_floor``           a key never on a wall -- and neither is any other object;
    * ``door_on_corridor``   a door on a corridor: >= 2 FLOOR neighbours;
    * ``distinct``           the four objects occupy four different cells;
    * ``tint_clear``         no object stands on a tinted cell (colours consistent:
                             the engine's generator clears the tint under every object,
                             so an object on a tint is not a state the engine emits);
    * ``colours_in_range``   every colour word is in ``1..N_COLOURS``;
    * ``feasible``           ``w8_envscout.maze_colour.solve`` finds a route from the
                             agent to the goal under RULES A+B.  The engine decides
                             this, not this module.

    ``door_load_bearing`` (``not _reachable_without_keys``) is COMPUTED and REPORTED,
    never required: forcing it would force ``door colour == key colour`` and collapse
    two words of the vocabulary into one.
    """
    g = np.asarray(st.grid)
    cells = [tuple(st.agent), tuple(st.key), tuple(st.door), tuple(st.goal)]
    in_bounds = all(0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1] for p in cells)
    on_floor = bool(in_bounds and all(g[p] == FLOOR for p in cells))
    deg = floor_degree(g)
    door_on_corridor = bool(in_bounds and g[tuple(st.door)] == FLOOR
                            and deg[tuple(st.door)] >= 2)
    distinct = len(set(cells)) == 4
    tint_clear = bool(in_bounds and not any(bool(st.tint_mask[p]) for p in cells))
    colours_in_range = all(1 <= int(x) <= N_COLOURS
                           for x in (st.ca, st.ck, st.cd, st.ct))
    sent = _wall_sentinels(g)
    feasible = False
    load_bearing = False
    if on_floor and distinct and colours_in_range and sent is not None:
        inst = st.engine_instance(sent)
        feasible = solve(inst)[0] is not None
        load_bearing = not _reachable_without_keys(inst)
    clauses = {
        "on_floor": on_floor,
        "door_on_corridor": door_on_corridor,
        "distinct": distinct,
        "tint_clear": tint_clear,
        "colours_in_range": colours_in_range,
        "sentinel_homes_available": sent is not None,
        "feasible": bool(feasible),
    }
    return {"legal": bool(all(clauses.values())), "clauses": clauses,
            "door_load_bearing": bool(load_bearing)}


def admissible(layout: Layout, req: ORequest):
    """``(ok, info)`` -- is the reference state this request names LEGAL?

    Only admissible requests are ever written or ever shown to an arm, which is how
    "no impossible request is shown to a non-refuser arm" holds **by construction**
    rather than by a filter applied after the fact.
    """
    st = render_state(layout, req)
    lg = legality(st)
    return bool(lg["legal"]), lg


def rule_b_bites(layout: Layout, req: ORequest) -> bool:
    """Does RULE B lengthen this state's optimal route?  Reported, never required."""
    st = render_state(layout, req)
    inst = st.engine_instance(_wall_sentinels(st.grid))
    lb, _ = solve(inst)
    la, _ = solve(inst, rule_b=False)
    if lb is None or la is None:
        return False
    return bool(lb > la)


def sentinels_are_inert(layouts, reqs, max_check: int = 64) -> dict:
    """The sentinel-key device, asserted on real states rather than argued.

    For each state the engine's own optimal route is replayed and every cell the walker
    stands on is checked against the sentinel homes.  A walker never stands on a WALL,
    so the count must be 0.
    """
    hits, n = 0, 0
    for (lay, req) in list(zip(layouts, reqs))[:max_check]:
        st = render_state(lay, req)
        sent = _wall_sentinels(st.grid)
        inst = st.engine_instance(sent)
        sent = set(sent)
        _L, route = solve(inst)
        if route is None:
            continue
        n += 1
        pos = tuple(st.agent)
        if pos in sent:
            hits += 1
        for a in route:
            if a == "P":
                continue
            dr, dc = MOVES[a]
            pos = (pos[0] + dr, pos[1] + dc)
            if pos in sent:
                hits += 1
    return {"states_replayed": n, "sentinel_cells_occupied": hits,
            "inert": bool(hits == 0),
            "statement": ("solve/judge read key homes only at occupied cells; a "
                          "sentinel home is a WALL cell and is never occupied")}


# ------------------------------------------------------------------ the item tensor
def item_tensor(st: OState) -> np.ndarray:
    """``(15, 15, 11)`` uint8 -- the rendered grid state, as the picture the AE sees."""
    x = np.zeros((H, W, N_CH), np.uint8)
    x[:, :, CH_FLOOR] = (np.asarray(st.grid) == FLOOR)
    x[st.tint_mask, CH_TINT] = 1
    x[st.door[0], st.door[1], CH_DOOR] = 1
    x[st.key[0], st.key[1], CH_KEY] = 1
    x[st.agent[0], st.agent[1], CH_AGENT] = 1
    x[st.goal[0], st.goal[1], CH_GOAL] = 1
    x[st.tint_mask, CH_COL + int(st.ct) - 1] = 1
    x[st.door[0], st.door[1], CH_COL + int(st.cd) - 1] = 1
    x[st.key[0], st.key[1], CH_COL + int(st.ck) - 1] = 1
    x[st.agent[0], st.agent[1], CH_COL + int(st.ca) - 1] = 1
    return x


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def decode_state(logits: np.ndarray) -> OState:
    """``(15, 15, 11)`` logits -> a CMAZE-O state.  Deterministic; no threshold tuned.

    The only number in here is ``0.5`` on a probability, which is ``logit > 0``: the
    sign of the decoder's own output, not a tolerance.  Nothing is fitted to it and it
    is identical for every arm and every row.  An object is placed at the argmax of its
    own mask channel and takes its colour from the SHARED colour field at that cell.
    """
    p = _sigmoid(logits)
    grid = np.where(p[:, :, CH_FLOOR] > 0.5, FLOOR, WALL).astype(np.uint8)
    tint_mask = p[:, :, CH_TINT] > 0.5
    col = p[:, :, CH_COL:CH_COL + N_COLOURS]

    def cell(ch):
        r, c = np.unravel_index(int(np.argmax(p[:, :, ch])), (H, W))
        return (int(r), int(c))

    def colour_at(cl):
        return int(np.argmax(col[cl[0], cl[1]])) + 1

    door, key, agent, goal = (cell(CH_DOOR), cell(CH_KEY), cell(CH_AGENT),
                              cell(CH_GOAL))
    #: the tint colour is the colour field averaged under the tint MASK PROBABILITY,
    #: so it is defined even when the decoded mask is empty.
    ct = int(np.argmax((col * p[:, :, CH_TINT][:, :, None]).sum(axis=(0, 1)))) + 1
    return OState(grid=grid, tint_mask=tint_mask, ct=ct, agent=agent,
                  ca=colour_at(agent), key=key, ck=colour_at(key),
                  door=door, cd=colour_at(door), goal=goal)


def state_record(st: OState) -> dict:
    """A compact, replayable record of a state -- what the independent checker reads."""
    return {
        "floor_bits": np.packbits((np.asarray(st.grid) == FLOOR).ravel()),
        "tint_bits": np.packbits(np.asarray(st.tint_mask, bool).ravel()),
        "ints": np.array([st.agent[0], st.agent[1], st.ca,
                          st.key[0], st.key[1], st.ck,
                          st.door[0], st.door[1], st.cd,
                          st.goal[0], st.goal[1], st.ct], np.int16),
    }


def state_from_record(floor_bits, tint_bits, ints) -> OState:
    """Rebuild a state from :func:`state_record`.  Used by the independent checker."""
    fl = np.unpackbits(np.asarray(floor_bits, np.uint8))[:H * W].reshape((H, W))
    tm = np.unpackbits(np.asarray(tint_bits, np.uint8))[:H * W].reshape((H, W))
    v = [int(x) for x in np.asarray(ints).ravel()]
    return OState(grid=np.where(fl > 0, FLOOR, WALL).astype(np.uint8),
                  tint_mask=tm.astype(bool), ct=v[11],
                  agent=(v[0], v[1]), ca=v[2], key=(v[3], v[4]), ck=v[5],
                  door=(v[6], v[7]), cd=v[8], goal=(v[9], v[10]))


# ------------------------------------------------------------------ the judge
def judge_state(pred: OState, ref: OState) -> dict:
    """THE CMAZE-O JUDGE.  Exact placement of every object AND legality.  No tolerance.

    ``valid = exact_objects and legal``.

    * ``exact_objects`` -- the eight request-borne facts: the four object cells, the
      four colour words.  These are what the nine words NAME.
    * ``legal``         -- the rendered world passes :func:`legality`, i.e. it is a
      world the engine would accept.
    * ``exact_layout``  -- the floor mask and the tint pattern, which the LAYOUT word
      names.  Reported beside, and ``exact_all`` is the conjunction.

    Every comparison below is integer or boolean.  ``tolerance_used`` is ``None``.
    """
    lg = legality(pred)
    exact_objects = bool(pred.key_tuple() == ref.key_tuple())
    exact_floor = bool(np.array_equal(np.asarray(pred.grid) == FLOOR,
                                      np.asarray(ref.grid) == FLOOR))
    exact_tintmask = bool(np.array_equal(np.asarray(pred.tint_mask, bool),
                                         np.asarray(ref.tint_mask, bool)))
    exact_layout = bool(exact_floor and exact_tintmask)
    per = {
        "agent_cell": tuple(pred.agent) == tuple(ref.agent),
        "key_cell": tuple(pred.key) == tuple(ref.key),
        "door_cell": tuple(pred.door) == tuple(ref.door),
        "goal_cell": tuple(pred.goal) == tuple(ref.goal),
        "agent_colour": int(pred.ca) == int(ref.ca),
        "key_colour": int(pred.ck) == int(ref.ck),
        "door_colour": int(pred.cd) == int(ref.cd),
        "tint_colour": int(pred.ct) == int(ref.ct),
    }
    return {
        "valid": bool(exact_objects and lg["legal"]),
        "exact_objects": exact_objects,
        "exact_layout": exact_layout,
        "exact_all": bool(exact_objects and exact_layout),
        "legal": bool(lg["legal"]),
        "clauses": lg["clauses"],
        "door_load_bearing": lg["door_load_bearing"],
        "per_fact": {k: bool(v) for k, v in per.items()},
        "floor_cells_wrong": int(np.sum((np.asarray(pred.grid) == FLOOR)
                                        != (np.asarray(ref.grid) == FLOOR))),
        "tint_cells_wrong": int(np.sum(np.asarray(pred.tint_mask, bool)
                                       != np.asarray(ref.tint_mask, bool))),
        "tolerance_used": None,
    }
