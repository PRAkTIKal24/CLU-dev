"""hub/placement, task items (a) + (b): TWO-STAGE -- MI-partition init, then refine by the
judge-scored SWAP LOSS.

The task file asks for these two in this order, and they compose into one object: (b) is
the loss, (a) is the schedule.  The init is the label-MI partition; the refinement moves
channels between wells and pays for a move only if the judge says the resulting store
partition composes better.

THREE PROPERTIES THAT MAKE IT A PLACEMENT RESULT AND NOT A FITTING ARTEFACT
  1. **Every move is an EXCHANGE.**  A channel leaves well ``w`` only in exchange for a
     channel entering it from well ``w'``.  The well-size histogram is therefore invariant
     along the whole trajectory, so the size-matched laundering floor stays exactly
     matched, ``K`` stays 5, and ``store_bytes = n_modes*D*4 = 524,288`` never moves.
  2. **The refinement never sees a held-out row.**  The pairs are drawn from the store's
     own fit pool, which ``assert_relevance_is_train_only`` certifies disjoint from the
     hold-out.  No corner, no cell, no arm outcome enters the loss.
  3. **The loss is scored on one pair set and the winner is re-scored on a DISJOINT one.**
     A refinement that only moves the pairs it was fitted on is an overfit to the probe
     and this reports it as one, in the same table.

THE COST MODEL, which is why the schedule looks like this.  An exchange between wells
``w`` and ``w'`` changes only those two wells' swap outcomes, so a proposal costs
``2 * n_pairs`` decodes (~3 s at 256 pairs) instead of ``5 * n_pairs``.  A leave-chunk-out
credit sweep over all five wells costs ``K * n_chunks * n_pairs`` decodes (~80 s at
10 chunks and 256 pairs) and tells the proposal generator where to look, so the hub does
not spend its budget on the 8,000 single-channel moves that a blind greedy would.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import exp_gen_factor_wells as W
from . import exp_gen_placement as P

#: Chunk schedule.  Coarse first, then finer: an early sweep should move a lot of channels
#: cheaply, a late one should not be able to.  Declared here, not tuned per run.
CHUNK_SCHEDULE = (41, 41, 20, 20, 10, 10)

#: The refinement pair set and the DISJOINT audit pair set draw from different streams.
#: The audit stream is never used to accept a move.
AUDIT_RNG_SEED = 20_260_920

#: THE ACCEPTANCE RULE, declared before the loop runs and never tuned inside it.
#: MEASURED on the 64-pair smoke run: an unpaired scalar comparison accepted a move
#: that gained +0.0062 on the pairs it was fitted on and LOST 0.0157 on the disjoint
#: audit set -- the rule was accepting probe noise.  The fix is not more pairs, it is
#: a PAIRED test: the same pairs score the partition before and after a move, most of
#: them are unaffected by it, and the paired difference has far smaller variance than
#: either level.  A move is accepted only when the mean paired gain clears
#: ``ACCEPT_MARGIN`` AND the 2.5% quantile of a paired bootstrap is above zero.
ACCEPT_MARGIN = 0.010
ACCEPT_BOOTSTRAP = 2_000


def _now() -> float:
    return time.time()


def _chunks(idx: np.ndarray, size: int, rng: np.random.Generator) -> list:
    """Split a well's channels into contiguous chunks of a shuffled order."""
    o = np.asarray(idx, np.int64)[rng.permutation(np.asarray(idx).size)]
    return [o[s:s + int(size)] for s in range(0, o.size, int(size))]


def _well_exact(judge, Xi, Xj, Yi, Yj, channels, own_f: int, batch: int) -> tuple:
    """``(exact, T_row, per_pair_hit)`` for ONE well's swap set.

    ``per_pair_hit`` is what makes the acceptance test PAIRED: the same pairs score
    the partition before and after a move, so the difference is a within-pair contrast
    and the pairs a move does not touch contribute exactly zero variance to it.
    """
    from ..core import gen_swap_probe as SP
    t = SP.swap_read(judge, Xi, Xj, channels, batch)
    Yi_ = np.asarray(Yi, np.int64)
    Yj_ = np.asarray(Yj, np.int64)
    intended = Yi_.copy()
    if own_f >= 0:
        intended[:, own_f] = Yj_[:, own_f]
    hit = (t == intended).all(axis=1)
    return float(hit.mean()), (t == Yj_).mean(axis=0), hit.astype(np.float64)


def paired_gain(before: np.ndarray, after: np.ndarray,
                rng: np.random.Generator) -> dict:
    """Mean paired gain plus a paired bootstrap interval.  The acceptance statistic."""
    d = np.asarray(after, np.float64) - np.asarray(before, np.float64)
    n = int(d.size)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(ACCEPT_BOOTSTRAP)])
    lo = float(np.quantile(boot, 0.025))
    return {"mean_gain": float(d.mean()), "ci95_lo": lo,
            "ci95_hi": float(np.quantile(boot, 0.975)),
            "n_pairs_changed": int((d != 0).sum()),
            "accept": bool(d.mean() > ACCEPT_MARGIN and lo > 0.0)}


def credit_sweep(judge, Xi, Xj, Yi, Yj, labels: np.ndarray, own: np.ndarray, k: int,
                 chunk_size: int, batch: int, rng: np.random.Generator,
                 log: Callable = print) -> list:
    """Leave-chunk-out credit for every chunk of every well.

    For well ``w`` with owned factor ``f``, removing chunk ``g`` from its swap set changes
    two things: how much of ``f`` still transfers (``control_drop``) and how much of
    everything else still comes along (``leak_drop``).  A chunk with
    ``harm = leak_drop - control_drop > 0`` is carrying more contamination than signal and
    is the thing an exchange should offer away.  This is a CAUSAL credit, read through the
    frozen decoder and the frozen classifier, not a correlation on encoder activations.
    """
    lab = np.asarray(labels, np.int64)
    out = []
    for w in range(int(k)):
        ch = np.flatnonzero(lab == w)
        f = int(own[w])
        base_exact, base_T, _ = _well_exact(judge, Xi, Xj, Yi, Yj, ch, f, batch)
        for g in _chunks(ch, chunk_size, rng):
            keep = np.setdiff1d(ch, g, assume_unique=False)
            if keep.size == 0:
                continue
            ex, T, _ = _well_exact(judge, Xi, Xj, Yi, Yj, keep, f, batch)
            control_drop = float(base_T[f] - T[f]) if f >= 0 else 0.0
            off = [x for x in range(base_T.size) if x != f]
            leak_drop = float(np.mean(base_T[off] - T[off]))
            out.append({"well": w, "chunk": g, "size": int(g.size),
                        "control_drop": control_drop, "leak_drop": leak_drop,
                        "harm": leak_drop - control_drop,
                        "exact_without": ex, "exact_base": base_exact})
        log(f"    well {w} (own={f}) base exact={base_exact:.4f}; "
            f"{int(np.ceil(ch.size / chunk_size))} chunks probed")
    return out


def refine(out_dir: Path, init: str = "mi_balanced", n_pairs: int = P.REFINE_PAIRS,
           sweeps: Optional[int] = None, max_seconds: float = 7200.0,
           log: Callable = print) -> dict:
    """Task items (a)+(b).  Resumable: the trajectory and the labels are checkpointed."""
    from ..core import gen_swap_probe as SP

    art = out_dir / f"refine_{init}.json"
    lab_p = out_dir / f"refine_{init}.npz"
    cands = P.build_candidates(out_dir, log=log)
    K = int(W.PINS["K"])
    batch = int(W.PINS["eval_batch"])
    grid, Xi, Xj, Yi, Yj, pairs = P._probe_inputs(out_dir, n_pairs, log=log)
    judge = W._JUDGE[0] if W._JUDGE else W.Judge(int(W.PINS["seed"]), log=log)
    if not W._JUDGE:
        W._JUDGE.append(judge)

    state: dict = {"init": init, "trajectory": [], "n_pairs": int(n_pairs),
                   "accept_margin": ACCEPT_MARGIN, "accept_rule":
                   "mean paired gain > margin AND paired-bootstrap 2.5% > 0",
                   "accepted": 0, "proposed": 0, "sweep": 0}
    labels = np.asarray(cands["labels"][init], np.int64).copy()
    if art.exists() and lab_p.exists():
        with open(art, encoding="utf-8") as fh:
            state = json.load(fh)
        labels = np.asarray(np.load(lab_p)["labels"], np.int64)
        log(f"  refine: resuming at sweep {state['sweep']}, "
            f"{state['accepted']}/{state['proposed']} accepted")

    # OWNERSHIP IS FIXED ONCE, from the INIT's measured transfer.  If it were re-derived
    # every step the objective would move under the optimiser and an "improvement" could
    # be a relabelling rather than a better placement.
    if "own" not in state:
        s0 = SP.score_partition(judge, Xi, Xj, Yi, Yj, labels, K, batch)
        state["own"] = s0["own"]
        state["init_score"] = s0
        log(f"  init {init}: exact={s0['exact']:.4f} control={s0['control']:.4f} "
            f"leak={s0['leak']:.4f} own={s0['own']} (ownership now FIXED)")
    own = np.asarray(state["own"], np.int64)

    # per-well exact, maintained incrementally
    ex = np.zeros(K, np.float64)
    hits = [None] * K
    for w in range(K):
        ex[w], _, hits[w] = _well_exact(judge, Xi, Xj, Yi, Yj,
                                        np.flatnonzero(labels == w), int(own[w]),
                                        batch)
    log(f"  current per-well exact {np.round(ex, 4).tolist()} mean={ex.mean():.4f}")
    arng = np.random.default_rng(AUDIT_RNG_SEED + 1)

    rng = np.random.default_rng(int(W.PINS["seed"]) * 131 + 20_260_919 + state["sweep"])
    t0 = _now()
    n_sweeps = len(CHUNK_SCHEDULE) if sweeps is None else int(sweeps)
    for si in range(int(state["sweep"]), n_sweeps):
        if _now() - t0 > float(max_seconds):
            log("  refine: wall-clock cap hit -- state saved, rerun to continue")
            break
        csize = CHUNK_SCHEDULE[min(si, len(CHUNK_SCHEDULE) - 1)]
        log(f"  sweep {si}: chunk size {csize}")
        cred = credit_sweep(judge, Xi, Xj, Yi, Yj, labels, own, K, csize, batch, rng,
                            log=log)
        # the most harmful chunk each well would like to give away
        worst = {}
        for c in cred:
            w = c["well"]
            if w not in worst or c["harm"] > worst[w]["harm"]:
                worst[w] = c
        order = sorted(worst, key=lambda w: -worst[w]["harm"])
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                if _now() - t0 > float(max_seconds):
                    break
                wa, wb = order[a], order[b]
                ga, gb = worst[wa]["chunk"], worst[wb]["chunk"]
                m = int(min(np.asarray(ga).size, np.asarray(gb).size))
                if m == 0:
                    continue
                ga, gb = np.asarray(ga)[:m], np.asarray(gb)[:m]
                trial = labels.copy()
                trial[ga] = wb
                trial[gb] = wa
                exa, _, ha = _well_exact(judge, Xi, Xj, Yi, Yj,
                                         np.flatnonzero(trial == wa), int(own[wa]),
                                         batch)
                exb, _, hb = _well_exact(judge, Xi, Xj, Yi, Yj,
                                         np.flatnonzero(trial == wb), int(own[wb]),
                                         batch)
                state["proposed"] += 1
                # the PAIRED statistic: the two touched wells, summed pair by pair
                g = paired_gain((hits[wa] + hits[wb]) / 2.0, (ha + hb) / 2.0, arng)
                before = float(ex[wa] + ex[wb])
                after = float(exa + exb)
                acc = bool(g["accept"])
                state["trajectory"].append({
                    "sweep": si, "chunk_size": int(csize), "wells": [wa, wb],
                    "n_moved_each_way": m, "before": before / 2.0,
                    "after": after / 2.0, "accepted": acc, "paired": g,
                    "mean_exact_after": float(
                        (ex.sum() - before + after) / K) if acc else float(ex.mean()),
                })
                if acc:
                    labels = trial
                    ex[wa], ex[wb] = exa, exb
                    hits[wa], hits[wb] = ha, hb
                    state["accepted"] += 1
                    log(f"    ACCEPT wells {wa}<->{wb} x{m}: paired gain "
                        f"{g['mean_gain']:+.4f} lo={g['ci95_lo']:+.4f}  "
                        f"{before / 2:.4f} -> {after / 2:.4f}  mean={ex.mean():.4f}")
                else:
                    log(f"    reject wells {wa}<->{wb} x{m}: paired gain "
                        f"{g['mean_gain']:+.4f} lo={g['ci95_lo']:+.4f} "
                        f"(margin {ACCEPT_MARGIN:+.3f})")
        state["sweep"] = si + 1
        state["mean_exact"] = float(ex.mean())
        state["well_sizes"] = np.bincount(labels, minlength=K).tolist()
        np.savez_compressed(lab_p, labels=labels)
        W._jdump(state, art)

    # -- the winner, scored properly: full screen pairs AND a DISJOINT audit set ------
    final = SP.score_partition(judge, Xi, Xj, Yi, Yj, labels, K, batch)
    state["final_on_refine_pairs"] = final
    Ya = grid.factor_indices_of(np.asarray(np.load(out_dir / "candidates.npz")["pool"],
                                           np.int64))
    ap = SP.make_pairs(Ya, int(P.SCREEN_PAIRS), rng_seed=AUDIT_RNG_SEED)
    z = np.load(out_dir / "candidates.npz")
    pool = np.asarray(z["pool"], np.int64)
    Fm = W.features_mmap(int(W.PINS["seed"]))
    Ai = W.gather(Fm, pool[ap[:, 0]])
    Aj = W.gather(Fm, pool[ap[:, 1]])
    state["audit_disjoint_pairs"] = {
        "rng_seed": AUDIT_RNG_SEED, "n_pairs": int(ap.shape[0]),
        "refined": SP.score_partition(judge, Ai, Aj, Ya[ap[:, 0]], Ya[ap[:, 1]],
                                      labels, K, batch, own=own),
        "init": SP.score_partition(judge, Ai, Aj, Ya[ap[:, 0]], Ya[ap[:, 1]],
                                   np.asarray(cands["labels"][init], np.int64), K, batch,
                                   own=own),
        "note": ("the acceptance rule never saw these pairs; if the refined partition "
                 "beats the init here too, the gain is not an overfit to the probe"),
    }
    state["ARI_refined_vs_init"] = float(_ari(labels, cands["labels"][init]))
    state["n_channels_moved_vs_init"] = int(
        (labels != np.asarray(cands["labels"][init], np.int64)).sum())
    np.savez_compressed(lab_p, labels=labels)
    W._jdump(state, art)
    a_ref = state["audit_disjoint_pairs"]["refined"]["exact"]
    a_ini = state["audit_disjoint_pairs"]["init"]["exact"]
    log(f"  DONE {init}: refine-pair exact {state['init_score']['exact']:.4f} -> "
        f"{final['exact']:.4f}; AUDIT (disjoint pairs) {a_ini:.4f} -> {a_ref:.4f}; "
        f"{state['n_channels_moved_vs_init']} channels moved, ARI "
        f"{state['ARI_refined_vs_init']:.4f}")
    return state


def _ari(a, b) -> float:
    from ..core.gen_store import adjusted_rand_index
    return float(adjusted_rand_index(np.asarray(a, np.int64), np.asarray(b, np.int64)))


# ---------------------------------------------------------------------------
# export -- hand the winning label vectors to exp_gen_factor_wells
# ---------------------------------------------------------------------------


def export(out_dir: Path, dest: Path, names: list, log: Callable = print) -> dict:
    """Write chosen label vectors to ``dest/extra_partitions.npz``.

    ``exp_gen_factor_wells.build_partitions`` merges that file if it exists, so a new
    placement enters the scoreboard through the SAME store fit, the SAME Gram, the SAME
    head, the SAME read verb and the SAME judge as every inherited arm.  Nothing about the
    scoring path is re-implemented here.
    """
    dest.mkdir(parents=True, exist_ok=True)
    z = np.load(out_dir / "candidates.npz")
    have = P.labels_from_npz(z)
    extra = {}
    for nm in names:
        if nm in have:
            extra[nm] = np.asarray(have[nm], np.int64)
            continue
        rp = out_dir / f"refine_{nm.replace('refined_', '')}.npz"
        if nm.startswith("refined_") and rp.exists():
            extra[nm] = np.asarray(np.load(rp)["labels"], np.int64)
        else:
            raise KeyError(f"no label vector for {nm!r} in {out_dir}")
    p = dest / "extra_partitions.npz"
    if p.exists():
        extra = {**P.labels_from_npz(np.load(p)), **extra}
    np.savez_compressed(p, **extra)
    meta = {k: {"well_sizes": np.bincount(v, minlength=int(W.PINS["K"])).tolist(),
                "source": str(out_dir)} for k, v in extra.items()}
    W._jdump(meta, dest / "extra_partitions.json")
    log(f"  exported {sorted(extra)} -> {p}")
    return meta


def dispatch(a, out: Path, log: Callable) -> None:
    if a.stage == "refine":
        refine(out, init=a.init, n_pairs=int(a.pairs or P.REFINE_PAIRS),
               max_seconds=float(a.max_seconds), log=log)
    elif a.stage == "export":
        names = (a.only.split(",") if a.only else [])
        export(out, Path(a.out).parent / "export", names, log=log)
