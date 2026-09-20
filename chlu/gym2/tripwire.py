"""The matched-bytes trajectory-lookup TRIPWIRE for GYM-2.

The question, which has to be answered before the gym is used for anything:

    if a table of stored trajectories, given the same number of bytes as the neural
    baselines have parameters, already solves the held-out pairs, then the held-out pairs
    are not testing composition and this gym is not an instrument.

⛔ Inherited lesson from w5/gym1: **T1 exact-key alone is not a tripwire.** It scored 0.0000
there and would have certified a dead gym as sound. The headline is the best over the whole
grid of strategies, and fragment stitching is run as a curve over BOTH the fragment
allowance `K` and the minimum fragment length `min_frag`.

  T1  exact key lookup
  T2  nearest neighbour on the absolute key, replay
  T3  nearest neighbour on the RELATIVE key, replay -- prefers the SAME layout (in 2-D a
      plan replayed into a different obstacle field simply collides, so same-layout is the
      strongest honest retrieval), then a stored speed limit no HIGHER than the query's (a
      plan built under a tighter limit is automatically legal under a looser one), then
      nearest relative geometry.
  T4  fragment stitching: shortest-path search over translation-invariant macro-actions
      harvested from the training trajectories.

Everything is scored by REPLAYING what is retrieved in the real environment, under exactly
the metrics a neural model gets. Byte accounting is explicit and the fraction of the budget
actually consumed is reported -- if the budget is slack, the tripwire was never byte-limited
and the result is the strongest possible form of the test.

EXACTNESS OF THE FRAGMENT LEGALITY TEST. In 1-D a fragment's swept footprint is an
interval, so the parent could test it with two numbers. In 2-D it is a *set of cells* and a
bounding box would be merely conservative -- which would weaken the tripwire, and a weak
tripwire is a weak instrument. So the footprint is carried exactly, as the full set of
swept cell offsets, and legality at every start cell of a layout is evaluated with 20-bit
bitboards: `legal[f] = AND over o in footprint(f) of (free shifted by o)`. Exact, and cheap
enough to run per layout. The search's verdict is then checked against reality anyway,
because every stitched plan is replayed.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

from chlu.gym2.grid2d import (
    ACTIONS, G, SWEEP, V_MAX_MAX, Episode, get_layout, replay_policy, rollout,
)

_AIDX = {a: i for i, a in enumerate(ACTIONS)}
VM = V_MAX_MAX
NVC = (2 * VM + 1) ** 2          # 81 velocity codes


def vcode(vx, vy) -> int:
    return (int(vx) + VM) * (2 * VM + 1) + (int(vy) + VM)


def vdec(c: int):
    return c // (2 * VM + 1) - VM, c % (2 * VM + 1) - VM


# --- byte accounting --------------------------------------------------------

KEY_BYTES_ABS = 8      # lid, V, sx, sy, vx0, vy0, gx, gy
KEY_BYTES_REL = 8      # lid, V, vx0, vy0, dx, dy, sx, sy
LEN_BYTES = 1
FRAG_HEAD_BYTES = 2    # start-velocity code + length


def action_bytes(n: int) -> int:
    """9 actions per step; two steps (81 combinations) pack into one byte."""
    return math.ceil(n / 2)


def entry_bytes_abs(n):
    return KEY_BYTES_ABS + LEN_BYTES + action_bytes(n)


def entry_bytes_rel(n):
    return KEY_BYTES_REL + LEN_BYTES + action_bytes(n)


def frag_bytes(n):
    return FRAG_HEAD_BYTES + action_bytes(n)


# ---------------------------------------------------------------------------

class TripwireScore(NamedTuple):
    name: str
    n: int
    valid_and_reached: float
    reached: float
    all_valid: float
    held_at_goal: float
    hit_rate: float
    bytes_used: int
    bytes_budget: int
    detail: dict


def _score(name, eps, act_seqs, horizon, budget, bytes_used, detail=None):
    n = len(eps)
    vr = rc = av = hg = hit = 0
    for ep, acts in zip(eps, act_seqs):
        if acts is None:
            continue
        hit += 1
        r = rollout(ep, replay_policy(acts), horizon=horizon, continue_after_fault=True)
        rc += r.reached_goal
        av += r.all_steps_valid
        hg += r.held_at_goal
        vr += (r.all_steps_valid and r.reached_goal)
    n1 = max(n, 1)
    return TripwireScore(name, n, vr / n1, rc / n1, av / n1, hg / n1, hit / n1,
                         int(bytes_used), int(budget), detail or {})


# ---------------------------------------------------------------------------
# T1 / T2 / T3 -- stored whole trajectories
# ---------------------------------------------------------------------------

def _fit_entries(records, budget, cost_fn):
    """Greedily store training trajectories cheapest-first until the budget is spent."""
    order = sorted(range(len(records)), key=lambda i: cost_fn(len(records[i].actions)))
    kept, used = [], 0
    for i in order:
        c = cost_fn(len(records[i].actions))
        if used + c > budget:
            break
        kept.append(i)
        used += c
    return kept, used


def t1_exact(train, held, horizon, budget):
    kept, used = _fit_entries(train, budget, entry_bytes_abs)
    table = {train[i].ep.key(): train[i].actions for i in kept}
    outs = [table.get(r.ep.key()) for r in held]
    return _score("T1_exact_key", [r.ep for r in held], outs, horizon, budget, used,
                  {"entries": len(table)})


def _feat_abs(rec):
    e = rec.ep
    m = get_layout(e.lid).meta
    return [e.lid, e.V, m["cx"], m["tw"], m["hb"], m["tunnel_len"],
            e.sx, e.sy, e.vx0, e.vy0, e.gx, e.gy]


def t2_nn_absolute(train, held, horizon, budget):
    kept, used = _fit_entries(train, budget, entry_bytes_abs)
    arr = np.array([_feat_abs(train[i]) for i in kept], dtype=np.int32)
    acts = [train[i].actions for i in kept]
    # Rule fields dominate: a trajectory from a different rule setting is a worse answer
    # than one from a different place.
    W = np.array([0, 40, 6, 6, 6, 3, 1, 1, 8, 8, 1, 1], dtype=np.int32)
    same_layout_bonus = 200
    outs = []
    for r in held:
        q = np.array(_feat_abs(r), dtype=np.int32)
        d = np.abs(arr - q) @ W + same_layout_bonus * (arr[:, 0] != q[0])
        outs.append(acts[int(np.argmin(d))])
    return _score("T2_nn_absolute", [r.ep for r in held], outs, horizon, budget, used,
                  {"entries": len(kept)})


def t3_nn_relative(train, held, horizon, budget):
    """Nearest neighbour on the relative key, with the strongest honest backoff.

    Same layout first (in 2-D a plan replayed into a different obstacle field just
    collides), then a stored speed limit no higher than the query's -- undershooting V is
    cheap because a tighter plan stays legal, overshooting is expensive -- then nearest
    relative geometry (start offset, displacement, start velocity).
    """
    kept, used = _fit_entries(train, budget, entry_bytes_rel)
    rows = []
    for i in kept:
        e = train[i].ep
        rows.append([e.lid, e.V, e.vx0, e.vy0, e.gx - e.sx, e.gy - e.sy, e.sx, e.sy])
    arr = np.array(rows, dtype=np.int32)
    acts = [train[i].actions for i in kept]
    outs = []
    for r in held:
        e = r.ep
        q = np.array([e.lid, e.V, e.vx0, e.vy0, e.gx - e.sx, e.gy - e.sy, e.sx, e.sy],
                     dtype=np.int32)
        dV = arr[:, 1] - q[1]
        cV = np.where(dV > 0, 60 * dV, -6 * dV)
        cost = (400 * (arr[:, 0] != q[0])                       # different layout: last resort
                + cV
                + 30 * (np.abs(arr[:, 2] - q[2]) + np.abs(arr[:, 3] - q[3]))
                + 12 * (np.abs(arr[:, 4] - q[4]) + np.abs(arr[:, 5] - q[5]))
                + 2 * (np.abs(arr[:, 6] - q[6]) + np.abs(arr[:, 7] - q[7])))
        outs.append(acts[int(np.argmin(cost))])
    return _score("T3_nn_relative", [r.ep for r in held], outs, horizon, budget, used,
                  {"entries": len(kept)})


# ---------------------------------------------------------------------------
# T4 -- fragment stitching
# ---------------------------------------------------------------------------

class Frags(NamedTuple):
    v_start: np.ndarray     # velocity code
    v_end: np.ndarray       # velocity code
    dx: np.ndarray
    dy: np.ndarray
    length: np.ndarray
    maxabsv: np.ndarray
    foot_ptr: np.ndarray    # CSR pointers into foot_off
    foot_off: np.ndarray    # (total, 2) int16 cell offsets, exact swept footprint
    seqs: list
    bytes_used: int
    n_before_prune: int


def _footprint(vx0, vy0, acts):
    """Exact set of cell offsets swept by a fragment, relative to its start cell."""
    cells = {(0, 0)}
    x = y = 0
    vx, vy = int(vx0), int(vy0)
    mav = max(abs(vx), abs(vy))
    for ax, ay in acts:
        vx += ax
        vy += ay
        mav = max(mav, abs(vx), abs(vy))
        for oi, oj in SWEEP[(vx, vy)]:
            cells.add((x + oi, y + oj))
        x += vx
        y += vy
    return sorted(cells), (x, y), (vx, vy), mav


def harvest_fragments(train, min_frag=3, max_frag=12, budget=None, harvest_from=None,
                      seed=0, pool_factor=3):
    """Every contiguous piece of every training trajectory, deduplicated by BEHAVIOUR.

    A fragment's entire effect -- displacement, velocity profile and swept footprint -- is
    determined by its start velocity plus its action string, so two pieces with the same
    (start velocity, actions) are the same macro-action and only one needs storing.

    Order of operations, chosen so memory stays bounded without weakening the table:
    dedup -> cheapest-first truncation to a candidate POOL a few times larger than the
    byte budget could hold -> exact footprints -> dominance pruning -> byte budget. The
    pool is the only approximation and it only ever discards fragments LONGER than ones
    already kept, which are the expensive ones the budget would have rejected anyway.

    Dominance pruning keeps, for each (v_start, v_end, dx, dy), only pieces Pareto-optimal
    in (length, footprint size): a piece both longer and wider-footprinted than another
    with the same endpoints and displacement costs more steps AND is more likely to hit an
    obstacle, so it can never be part of a better plan.
    """
    src = train if harvest_from is None else train[:harvest_from]
    seen = set()
    by_len = {}
    for rec in src:
        acts = rec.actions
        ep = rec.ep
        vs = [(ep.vx0, ep.vy0)]
        vx, vy = ep.vx0, ep.vy0
        for ax, ay in acts:
            vx += ax
            vy += ay
            vs.append((vx, vy))
        n = len(acts)
        for i in range(n):
            for ell in range(min_frag, min(max_frag, n - i) + 1):
                key = (vs[i], acts[i:i + ell])
                if key in seen:
                    continue
                seen.add(key)
                by_len.setdefault(ell, []).append(key)
    n_raw = len(seen)
    del seen

    pool_cap = None
    if budget is not None:
        pool_cap = pool_factor * (budget // max(frag_bytes(min_frag), 1))
    pool = []
    for ell in sorted(by_len):                       # cheapest-first
        for key in by_len[ell]:
            pool.append((ell, key))
            if pool_cap is not None and len(pool) >= pool_cap:
                break
        if pool_cap is not None and len(pool) >= pool_cap:
            break
    del by_len

    items = []
    for ell, (v0, sub) in pool:
        cells, disp, vend, mav = _footprint(v0[0], v0[1], sub)
        items.append((ell, len(cells), v0, sub, cells, disp, vend, mav))
    del pool

    by = {}
    for it in items:
        by.setdefault((vcode(*it[2]), vcode(*it[6]), it[5][0], it[5][1]), []).append(it)
    keep = []
    for group in by.values():
        group.sort(key=lambda z: (z[0], z[1]))
        best_nf = None
        for it in group:
            if best_nf is not None and it[1] >= best_nf:
                continue
            keep.append(it)
            best_nf = it[1] if best_nf is None else min(best_nf, it[1])
    del by, items

    keep.sort(key=lambda z: frag_bytes(z[0]))
    used = 0
    vs_, ve_, dx_, dy_, ln_, mv_, seqs = [], [], [], [], [], [], []
    offs, ptr = [], [0]
    for ell, nf, v0, sub, cells, disp, vend, mav in keep:
        c = frag_bytes(ell)
        if budget is not None and used + c > budget:
            break
        used += c
        vs_.append(vcode(*v0))
        ve_.append(vcode(*vend))
        dx_.append(disp[0])
        dy_.append(disp[1])
        ln_.append(ell)
        mv_.append(mav)
        seqs.append(sub)
        offs.extend(cells)
        ptr.append(len(offs))
    A = lambda z, t=np.int32: np.array(z, dtype=t)
    return Frags(A(vs_), A(ve_), A(dx_), A(dy_), A(ln_), A(mv_),
                 A(ptr, np.int64), np.array(offs, dtype=np.int16).reshape(-1, 2),
                 seqs, used, n_raw)


# --- exact per-layout legality via bitboards --------------------------------

_MASK_G = (1 << G) - 1


def _shift_table(lid: int, P: int):
    """SH[ox+P, oy+P][x] = bitboard row of "cell (x+ox, y+oy) is free", exactly.

    Out-of-grid is treated as blocked, so the table folds the bounds check in.
    """
    free = ~get_layout(lid).blocked
    rows = np.zeros(G, dtype=np.uint32)
    for x in range(G):
        r = 0
        for y in range(G):
            if free[x, y]:
                r |= (1 << y)
        rows[x] = r
    n = 2 * P + 1
    SH = np.zeros((n, n, G), dtype=np.uint32)
    for ox in range(-P, P + 1):
        src = np.zeros(G, dtype=np.uint32)
        lo, hi = max(0, -ox), min(G, G - ox)
        if lo < hi:
            src[lo:hi] = rows[lo + ox:hi + ox]
        for oy in range(-P, P + 1):
            if oy >= 0:
                v = (src >> np.uint32(oy)) if oy < 32 else np.zeros(G, dtype=np.uint32)
            else:
                v = (src << np.uint32(-oy)) & np.uint32(_MASK_G)
            SH[ox + P, oy + P] = v & np.uint32(_MASK_G)
    return SH


def fragment_legality(F: Frags, lid: int, chunk: int = 20000):
    """(n_frag, G) uint32 bitboards: is fragment f legal when started at cell (x, y)?

    `legal[f][x] >> y & 1` iff every cell of f's exact swept footprint, translated to
    (x, y), is inside the grid and free. This is the full footprint, not a bounding box.
    """
    if len(F.seqs) == 0:
        return np.zeros((0, G), dtype=np.uint32)
    P = int(max(1, np.abs(F.foot_off).max())) if F.foot_off.size else 1
    SH = _shift_table(lid, P).reshape((2 * P + 1) ** 2, G)
    nf = len(F.seqs)
    sizes = np.diff(F.foot_ptr)
    maxs = int(sizes.max())
    out = np.full((nf, G), np.uint32(_MASK_G), dtype=np.uint32)
    flat = (F.foot_off[:, 0].astype(np.int64) + P) * (2 * P + 1) + \
           (F.foot_off[:, 1].astype(np.int64) + P)
    for lo in range(0, nf, chunk):
        hi = min(nf, lo + chunk)
        m = hi - lo
        idx = np.zeros((m, maxs), dtype=np.int64)
        # pad short footprints by repeating the first offset (idempotent under AND)
        for j in range(m):
            a, b = F.foot_ptr[lo + j], F.foot_ptr[lo + j + 1]
            k = b - a
            idx[j, :k] = flat[a:b]
            if k < maxs:
                idx[j, k:] = flat[a]
        out[lo:hi] = np.bitwise_and.reduce(SH[idx], axis=1)
    return out


class _LayoutIndex(NamedTuple):
    legal: np.ndarray           # (nf, G) uint32
    legal_b: np.ndarray         # (nf, G*G) bool, unpacked for the search
    by_vs: dict                 # v_start code -> fragment indices
    by_ve: dict                 # v_end code   -> fragment indices


def build_layout_index(F: Frags, lid: int) -> _LayoutIndex:
    legal = fragment_legality(F, lid)
    nf = legal.shape[0]
    legal_b = np.empty((nf, G, G), dtype=bool)
    for yy in range(G):      # column-at-a-time keeps the temporary at (nf, G) uint32
        legal_b[:, :, yy] = ((legal >> np.uint32(yy)) & np.uint32(1)).astype(bool)
    legal_b = legal_b.reshape(nf, G * G)
    by_vs, by_ve = {}, {}
    for c in range(NVC):
        i = np.nonzero(F.v_start == c)[0]
        if i.size:
            by_vs[c] = i
        j = np.nonzero(F.v_end == c)[0]
        if j.size:
            by_ve[c] = j
    return _LayoutIndex(legal, legal_b, by_vs, by_ve)


BIG = 1 << 29


def _expand(ep: Episode, F: Frags, LI: _LayoutIndex, cost, horizon):
    """One layer of the fragment search, vectorised over (source cell x fragment)."""
    V = ep.V
    new = cost.copy()
    for c, idx in LI.by_vs.items():
        vx, vy = vdec(c)
        if abs(vx) > V or abs(vy) > V:
            continue
        srcflat = np.nonzero(cost[:, c] < BIG)[0]
        if srcflat.size == 0:
            continue
        idx = idx[F.maxabsv[idx] <= V]
        if idx.size == 0:
            continue
        sc = cost[srcflat, c]
        ok = LI.legal_b[np.ix_(idx, srcflat)]                # (nf, S)
        tot = sc[None, :] + F.length[idx][:, None]
        sel = ok & (tot <= horizon)
        if not sel.any():
            continue
        fi, si = np.nonzero(sel)
        sflat = srcflat[si]
        nx = sflat // G + F.dx[idx][fi]
        ny = sflat % G + F.dy[idx][fi]
        tgt = (nx * G + ny) * NVC + F.v_end[idx][fi]
        np.minimum.at(new.reshape(-1), tgt, tot[fi, si])
    return new


def _stitch_one(ep: Episode, F: Frags, LI: _LayoutIndex, K: int, horizon: int):
    """Fewest-fragment plan from (s, v0) to (g, 0), or None.

    Fragment legality is the exact footprint test, and the plan is then REPLAYED in the
    real environment by the caller -- so a mistake in the fast test would show up as a
    search that claims success and a rollout that faults. That equality is checked and
    reported.
    """
    cost = np.full((G * G, NVC), BIG, dtype=np.int64)
    cost[ep.sx * G + ep.sy, vcode(ep.vx0, ep.vy0)] = 0
    layers = [cost]
    gflat = ep.gx * G + ep.gy
    gc = vcode(0, 0)
    for _ in range(K):
        nxt = _expand(ep, F, LI, layers[-1], horizon)
        layers.append(nxt)
        if nxt[gflat, gc] < BIG:
            break
    if layers[-1][gflat, gc] >= BIG:
        return None

    pieces = []
    xf, vc_, c = gflat, gc, int(layers[-1][gflat, gc])
    for k in range(len(layers) - 1, 0, -1):
        prev = layers[k - 1]
        if prev[xf, vc_] == c:
            continue
        idx = LI.by_ve.get(vc_)
        if idx is None:
            return None
        idx = idx[F.maxabsv[idx] <= ep.V]
        if idx.size == 0:
            return None
        px = xf // G - F.dx[idx]
        py = xf % G - F.dy[idx]
        good = (px >= 0) & (px < G) & (py >= 0) & (py < G)
        if not good.any():
            return None
        idx2, px, py = idx[good], px[good], py[good]
        pflat = px * G + py
        pc = prev[pflat, F.v_start[idx2]]
        legal = LI.legal_b[idx2, pflat]
        hit = np.nonzero(legal & (pc + F.length[idx2] == c))[0]
        if hit.size == 0:
            return None
        j = int(hit[0])
        f = int(idx2[j])
        pieces.append(F.seqs[f])
        xf, vc_, c = int(pflat[j]), int(F.v_start[f]), int(pc[j])
        if xf == ep.sx * G + ep.sy and vc_ == vcode(ep.vx0, ep.vy0) and c == 0:
            break
    if not (xf == ep.sx * G + ep.sy and vc_ == vcode(ep.vx0, ep.vy0) and c == 0):
        return None
    seq = []
    for p in reversed(pieces):
        seq.extend(p)
    return seq or None


def t4_stitch(train, held, horizon, budget, ks=(1, 2, 3, 4, 6), min_frag=3, max_frag=12,
              harvest_from=None, sample=None, seed=0, F=None, verbose=False):
    """Fragment-stitching tripwire, reported as a curve over the fragment allowance K."""
    if F is None:
        F = harvest_fragments(train, min_frag=min_frag, max_frag=max_frag, budget=budget,
                              harvest_from=harvest_from)
    if sample is not None and sample < len(held):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(held), size=sample, replace=False)
        held = [held[int(i)] for i in sel]

    # group by layout so the exact legality table is built once per layout
    by_lay = {}
    for i, r in enumerate(held):
        by_lay.setdefault(r.ep.lid, []).append(i)

    Kmax = max(ks)
    plans = {K: [None] * len(held) for K in ks}
    for li, (lid, ids) in enumerate(sorted(by_lay.items())):
        LI = build_layout_index(F, lid)
        for i in ids:
            ep = held[i].ep
            for K in ks:
                plans[K][i] = _stitch_one(ep, F, LI, K, horizon)
        if verbose and li % 20 == 0:
            print(f"    T4 min_frag={min_frag} layout {li+1}/{len(by_lay)}", flush=True)

    out = []
    for K in ks:
        out.append(_score(f"T4_stitch_mf{min_frag}_K{K}", [r.ep for r in held],
                          plans[K], horizon, budget, F.bytes_used,
                          {"fragments_before_prune": F.n_before_prune,
                           "fragments": len(F.seqs), "min_frag": min_frag,
                           "max_frag": max_frag, "K": K,
                           "harvested_from_episodes": harvest_from or len(train),
                           "held_subsample": len(held),
                           "n_layouts": len(by_lay)}))
    return out, F
