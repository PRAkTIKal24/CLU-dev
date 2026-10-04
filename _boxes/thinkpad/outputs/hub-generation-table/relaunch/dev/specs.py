"""RELAUNCH - the variant tables.  A variant is a DATA ROW, never a code path.

Every configuration is priced from the closed byte formula BEFORE it runs and the price
is re-checked against the LIVE objects afterwards (``arm2.run_variant`` returns
``total_bytes`` from ``gen_store.store_bytes`` + ``RaggedHead.byte_ledger``).

    store = (D / K) * A * 4                       A = sum_k M_k
    head  = (n_add * A  +  n_pair * r  +  r * A) * 4
    budget = knn_proto's deployed 475,136 B

The rank is never chosen: it is the largest one the budget leaves (``arm2.budget_rank``).
"""
from __future__ import annotations

import arm2

KNN_BYTES = arm2.KNN_BYTES        # 475,136

#: the two pair terms iteration 1 selected.  INHERITED, not re-selected here.
PAIRS_INHERITED = ["floor_huexscale", "object_huexscale"]
#: the pair set the 0.9734 pairwise BOUND was carried by (launch addendum).
PAIRS_BOUND = ["floor_huexscale", "object_huexscale", "shapexorientation"]

FACTORS = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")


def _pairs(names, spec):
    from chlu.core.gen_factored_read import default_pairs
    return default_pairs(names, spec)


def pair_index(names, spec):
    """``["floor_huexscale", ...] -> [(f, g), ...]`` by factor NAME, order-independent."""
    idx = []
    for s in spec:
        a, b = s.split("x")
        idx.append((list(names).index(a), list(names).index(b)))
    return idx


def n_add_of(cards, reduced):
    return (1 + sum(int(x) - 1 for x in cards)) if reduced else (1 + sum(int(x) for x in cards))


def n_pair_of(cards, pairs):
    return int(sum(int(cards[f]) * int(cards[g]) for f, g in pairs))


def make(cards, D, *, K, A, pairs, method="kmeans", reduced=True, metric_correct=True,
         ridge_add=10.0, ridge_pair=100.0, alloc="uniform", rank=None,
         budget=KNN_BYTES):
    n_add = n_add_of(cards, reduced)
    n_pair = n_pair_of(cards, pairs)
    d_well = D // K
    r = arm2.budget_rank(K, A, n_add, n_pair, d_well, budget) if rank is None else int(rank)
    r = min(r, n_pair, A)                       # rank cannot exceed either dimension
    pred = (d_well * A + n_add * A + n_pair * r + r * A) * 4
    return dict(K=K, A=A, rank=r, pairs=pairs, ridge_add=ridge_add, ridge_pair=ridge_pair,
                reduced=reduced, metric_correct=metric_correct, method=method,
                alloc=alloc, n_add=n_add, n_pair=n_pair, _bytes_pred=pred)


def make_shared(cards, D, *, K, pairs, method="kmeans", reduced=True, ridge_add=10.0,
                ridge_pair=100.0, alloc="uniform", A=None, budget=KNN_BYTES):
    """The SHARED-BASIS head: the interaction reuses C_add, so it costs n_pair*n_add*4
    whatever A is, and every remaining byte goes into ATOMS -- which is what the measured
    frontier says is the valuable end."""
    n_add = n_add_of(cards, reduced)
    n_pair = n_pair_of(cards, pairs)
    d_well = D // K
    Amax = arm2.budget_atoms_shared(K, n_add, n_pair, d_well, budget)
    A = int(Amax if A is None else min(int(A), Amax))
    A -= A % K                                  # a uniform allocation needs K | A
    A = min(A, K * d_well)                      # M_k cannot exceed d_k
    pred = (d_well * A + n_add * A + n_pair * n_add) * 4
    return dict(K=K, A=A, rank=0, pairs=pairs, ridge_add=ridge_add, ridge_pair=ridge_pair,
                reduced=reduced, metric_correct=True, method=method, alloc=alloc,
                n_add=n_add, n_pair=n_pair, head_kind="shared", _bytes_pred=pred)


def table(name, cards, D):
    names = FACTORS
    P2 = pair_index(names, PAIRS_INHERITED)
    P3 = pair_index(names, PAIRS_BOUND)

    if name == "levers":
        # ---- EXP-R1: one lever at a time, off the predecessor's inherited arm ----
        base = dict(K=32, A=832, pairs=P2, method="kmeans", alloc="uniform")
        return {
            # the predecessor's DEV-24 arm, reproduced: full 58-col design, coefficient
            # -metric truncation, one shared ridge.  The reference row.
            "base":      make(cards, D, **base, reduced=False, metric_correct=False,
                              ridge_add=10.0, ridge_pair=10.0),
            "reduced":   make(cards, D, **base, reduced=True, metric_correct=False,
                              ridge_add=10.0, ridge_pair=10.0),
            "metric":    make(cards, D, **base, reduced=False, metric_correct=True,
                              ridge_add=10.0, ridge_pair=10.0),
            "ridgepair": make(cards, D, **base, reduced=False, metric_correct=False,
                              ridge_add=10.0, ridge_pair=100.0),
            "nmf":       make(cards, D, K=32, A=832, pairs=P2, method="nmf",
                              alloc="uniform", reduced=False, metric_correct=False,
                              ridge_add=10.0, ridge_pair=10.0),
            "all3":      make(cards, D, **base, reduced=True, metric_correct=True,
                              ridge_add=10.0, ridge_pair=100.0),
            # "all3_nmf" DROPPED 11:19Z on COST, not on taste: NMF costs 200 s a unit
            # against 45 s and lost on both SELECT corners measured (c0 0.8241 vs base
            # 0.8843; c6 0.8125 vs 0.8287).  The single-lever "nmf" row stays so the
            # isolation is complete; the combined one is not worth 13 min of the window.
            "all3_p3":   make(cards, D, K=32, A=832, pairs=P3, method="kmeans",
                              alloc="uniform", reduced=True, metric_correct=True,
                              ridge_add=10.0, ridge_pair=100.0),
        }

    if name == "frontier":
        # ---- EXP-R2: the ATOMS-vs-RANK frontier at a FIXED byte budget ----
        # store and additive head BOTH scale with A; the interaction rank is what the
        # freed bytes buy.  The predecessor fixed A = 832 and never swept it.
        out = {}
        for A in (832, 704, 576, 448, 384, 320, 256):
            out[f"K32_A{A}"] = make(cards, D, K=32, A=A, pairs=P2, reduced=True,
                                    metric_correct=True, ridge_add=10.0,
                                    ridge_pair=100.0)
        return out

    if name == "frontier3":
        # the same frontier with the THIRD pair term (shape x orientation) added -
        # the pair set the 0.9734 pairwise bound was carried by.
        out = {}
        for A in (832, 704, 576, 448, 384, 320, 256):
            out[f"P3_K32_A{A}"] = make(cards, D, K=32, A=A, pairs=P3, reduced=True,
                                       metric_correct=True, ridge_add=10.0,
                                       ridge_pair=100.0)
        return out

    if name == "grid":
        # ---- EXP-R2: the ATOMS-vs-RANK frontier at a FIXED byte budget, x K x pair set.
        # store AND the additive head both scale with A; the interaction rank is what the
        # freed bytes buy.  Iteration 1 fixed A = 832 and never swept it.
        # K = 64 halves d = D/K, so an atom costs 336 B instead of 464 B.
        # K = 128 is EXCLUDED as an arm: C.2 is measured to stop collapsing there.
        out = {}
        for K in (32, 64):
            for A in (320, 448, 576, 704, 832):
                for tag, P in (("P2", P2), ("P3", P3)):
                    out[f"{tag}_K{K}_A{A}"] = make(
                        cards, D, K=K, A=A, pairs=P, reduced=True, metric_correct=True,
                        ridge_add=10.0, ridge_pair=100.0)
        return out

    if name == "decomp":
        # ---- the DECOMPOSITION rows: how much of the arm is additive and how much is
        # the interaction block.  rank = 0 is the arm's additive part alone, pushed
        # through the SAME store span -- the like-for-like against knn_proto, which IS
        # an unrestricted additive map into the full D = 2048.
        out = {}
        for K in (32, 64):
            for A in (448, 832):
                out[f"r0_K{K}_A{A}"] = make(cards, D, K=K, A=A, pairs=P2, rank=0,
                                            reduced=True, metric_correct=True,
                                            ridge_add=10.0, ridge_pair=100.0)
        return out

    if name == "shared":
        # ---- EXP-R4: the SHARED-BASIS interaction head.  The measured frontier says the
        # arm's score is monotone increasing in A, and the low-rank head charges r*A for
        # every unit of interaction.  Taking U := C_add makes the interaction cost
        # n_pair*n_add*4 INDEPENDENT of A, so every freed byte becomes an atom.
        out = {}
        for K in (32, 64):
            for tag, P in (("P2", P2), ("P3", P3)):
                out[f"SH_{tag}_K{K}"] = make_shared(cards, D, K=K, pairs=P)
            # the additive-only reference at the SAME K: the arm with no interaction at
            # all, so the interaction's contribution is a paired difference, not a guess.
            out[f"SH_ADD_K{K}"] = make_shared(cards, D, K=K, pairs=[])
        # and the same head at the INHERITED atom count, so A and the head change
        # one at a time rather than together.
        for K in (32, 64):
            out[f"SH_P2_K{K}_A832"] = make_shared(cards, D, K=K, pairs=P2, A=832)
        return out

    if name == "screen":
        # ---- THE DISQUALIFICATION SCREEN, run BEFORE any confirm run is committed.
        # charter C.2: a linear/soft read over K wells gets K FREE SCALARS from the
        # partition alone, and iteration 1 measured that the store-removed control STOPS
        # COLLAPSING at K >~ 128.  The arm that won the SELECT set is at K = 64, so
        # whether C.2 still collapses there is a DISQUALIFIER, and a configuration is
        # disqualified BEFORE it is scored, never after.
        # K = 128 is run beside it ONLY to place the boundary -- never as a candidate.
        out = {}
        for K in (32, 64, 128):
            for m in ("mean", "atoms_medoid", "pca_affine", "random"):
                sp = make_shared(cards, D, K=K, pairs=P2, method=m)
                out[f"SCR_K{K}_{m}"] = sp
            if K == 128:
                out["SCR_K128_kmeans"] = make_shared(cards, D, K=128, pairs=P2,
                                                     method="kmeans")
        return out

    if name == "klow":
        # ---- EXP-R5: THE OTHER END OF K, which I never looked at and my own data points
        # straight at.  Measured: K=32 > K=64 > K=128 (the last disqualified by C.2).
        # The trend says SMALLER, and the arithmetic says why it should keep going:
        #
        #   the deployed additive map is  Phi_add . C_add . B   =  (52, D)
        #   unfactorised that is 52*D*4 = 425,984 B;  the store+head factorisation costs
        #   (D/K)*A*4 + 52*A*4.  The factorisation only SAVES bytes while A is small, and
        #   a small A is exactly what destroys the span -- unless the atoms are WIDE.
        #   At K=1 an atom spans all D channels, so A ~ 51 atoms already span the whole
        #   rank-52 additive map: the factorisation loss goes to ~zero.
        #
        # C.2 is not at risk at the small-K end (a partition hands the read K free
        # scalars, and K <= 16 gives it at most 16), but it is measured anyway.
        out = {}
        for K in (1, 2, 4, 8, 16):
            out[f"KL_K{K}"] = make_shared(cards, D, K=K, pairs=P3, method="kmeans")
            out[f"KL_K{K}_mean"] = make_shared(cards, D, K=K, pairs=P3, method="mean")
        return out

    if name == "klow_nmf":
        out = {}
        for K in (1, 2, 4, 8, 16):
            out[f"KLN_K{K}"] = make_shared(cards, D, K=K, pairs=P3, method="nmf")
        return out

    if name == "part":
        # ---- EXP-R7: the last two levers the launch addendum names that I have not
        # spent -- the PARTITION and the per-well ATOM ALLOCATION.  Both are free at a
        # fixed byte count, both were built by the stopped predecessor and NEITHER was
        # ever measured.  k-means atoms throughout (190 s -> 35 s a unit): the question
        # is the partition, and the NMF-vs-k-means effect is orthogonal and already
        # measured at +0.0149 [+0.0080, +0.0219].
        #   equal : ward dendrogram leaf order cut into K EQUAL blocks  (the arm's)
        #   ward  : the SHIPPED gen_store.correlation_grouping, UNEQUAL wells
        #   uniform / waterfill : atoms per well, at a fixed total
        # A waterfill allocation is RAGGED, so it cannot use the pinned rectangular
        # clu_wide_vector; those rows are DEV-only and say so in their own record.
        out = {}
        for part in ("equal", "ward"):
            for alloc in ("uniform", "waterfill"):
                sp = make_shared(cards, D, K=64, pairs=P3, method="kmeans", A=1216,
                                 alloc=alloc)
                sp["partition"] = part
                out[f"PT_{part}_{alloc}"] = sp
        return out

    if name == "best":
        # ---- EXP-R8: the best DEV configuration the whole run points at, measured
        # DIRECTLY rather than assumed additive: NMF atoms + the shipped ward grouping
        # + the water-filled allocation.  It is a RECOMMENDATION FOR THE NEXT ITERATION,
        # not an arm of this one: the scored budget is spent, the frozen arm is frozen,
        # and a water-filled allocation is RAGGED, so it cannot use the pinned
        # rectangular clu_wide_vector until the harness owner generalises it.
        out = {}
        sp = make_shared(cards, D, K=64, pairs=P3, method="nmf", A=1216,
                         alloc="waterfill")
        sp["partition"] = "ward"
        out["BEST_nmf_ward_wf"] = sp
        sp2 = make_shared(cards, D, K=64, pairs=P3, method="nmf", A=1216,
                          alloc="waterfill")
        sp2["partition"] = "equal"
        out["BEST_nmf_equal_wf"] = sp2
        return out

    if name == "legal":
        # ---- The screen says the R4 `pca_affine` CONTROL beats the k-means arm at the
        # winning configuration.  R4 forbids adopting a principal direction as a store
        # ("it cannot be decoded or deleted, and the deletion target needs atoms"), so
        # that is a control and stays one.  NMF is the CHARTER-LEGAL version of the same
        # intuition: the SPAN objective (min ||X - A W||, the read verb's own objective)
        # with the atoms constrained to the data's own non-negative orthant, so every
        # atom is still a valid, decodable, deletable feature vector.
        # The question this table asks: how much of the pca_affine gap is the OBJECTIVE
        # (legal to chase) rather than the unconstrained basis (not legal to adopt)?
        out = {}
        for tag, P in (("P2", P2), ("P3", P3)):
            out[f"NMF_{tag}_K64"] = make_shared(cards, D, K=64, pairs=P, method="nmf")
        return out

    if name == "confirm":
        # ---- the frozen arm and EVERY control it must carry, run on the 43 CONFIRM
        # corners.  The configuration is frozen at 12:33Z on the SELECT set and nothing
        # in it is re-chosen after a CONFIRM number is seen.
        # FROZEN 13:10Z on the SELECT set, before one CONFIRM corner was looked at:
        #   K = 64 - shared-basis head - A = 1,216 atoms - NMF dictionary - 3 pair terms
        #   454,336 B = 0.956 x knn_proto's 475,136 B
        A0 = 1216
        out = {"ARM": make_shared(cards, D, K=64, pairs=P3, method="nmf", A=A0)}
        # the dictionary-objective comparison, paired on the same corners
        out["ARM_KMEANS"] = make_shared(cards, D, K=64, pairs=P3, method="kmeans", A=A0)
        # the interaction's own contribution: same K, same ATOM COUNT, no pair block
        out["ADD_ONLY"] = make_shared(cards, D, K=64, pairs=[], method="kmeans", A=A0)
        # every control the launch addendum requires, at the SAME configuration and the
        # SAME atom count, so no control differs from the arm in more than one thing
        for m in ("mean", "atoms_medoid", "atoms_kpp", "atoms_random", "pca_affine",
                  "random"):
            out[f"CTRL_{m}"] = make_shared(cards, D, K=64, pairs=P3, method=m, A=A0)
        # iteration 1's design at its own atom count, as the inherited reference row
        out["ITER1"] = make(cards, D, K=32, A=832, pairs=P3, reduced=True,
                            metric_correct=True, ridge_add=10.0, ridge_pair=100.0)
        return out

    if name == "kfront":
        # ---- EXP-R3: K at its own best A.  Larger K => cheaper atoms (d = D/K) =>
        # more rank per byte, but each atom covers fewer channels.
        # C.2 is measured to stop collapsing at K >~ 128, so 128 is the last rung and
        # is run ONLY to place the disqualification boundary, never as a candidate.
        out = {}
        for K in (16, 32, 64, 128):
            for A in (448, 832):
                out[f"K{K}_A{A}"] = make(cards, D, K=K, A=A, pairs=P3, reduced=True,
                                         metric_correct=True, ridge_add=10.0,
                                         ridge_pair=100.0)
        return out

    raise ValueError(f"unknown variant table {name!r}")
