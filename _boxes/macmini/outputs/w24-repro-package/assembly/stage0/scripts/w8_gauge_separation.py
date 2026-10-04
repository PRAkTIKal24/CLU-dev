"""w8 hub/single-design -- SEPARATE THE GAUGE FROM THE DELETION.  ⛔ DEV only, NOT scored.

The peer spoke's pinned-basis smoke reads back after deleting 11,024 red rows with no
refit of anything and collapses **0.8941 -> 0.0584**.  This script asks the only
question that makes that number interpretable:

    How much of that collapse is a SLOT MISMATCH, and how much is the deletion?

A pinned eigenvalue-ORDERED basis is well defined only if the deletion does not change
the ORDER.  Measured (LOG 19:40Z), it changes it for 11-35 % of directions per well.
So slot ``j`` of a book fitted before the deletion multiplies a direction that is no
longer direction ``j``.  That is gauge, not forgetting.

Four reads, all through the SAME post-deletion content, differing only in how the
pre-deletion coefficients are put into correspondence with the post-deletion directions:

    before   c @ U0^T                    the pre-deletion answer
    naive    c @ U1^T                    slot j meets post-direction j   <- the smoke
    perm     c[:, pi] * s  @ U1^T        slots matched by |overlap|, signs fixed
    carry    c @ (U0^T U1) @ U1^T        the full linear correspondence

``carry`` is provably the correct post-deletion answer: it equals the pre-deletion
ambient answer projected onto the post-deletion span, so the deleted content is
projected out and nothing else moves.  Therefore

    REAL deletion effect = ||before - carry||
    GAUGE error in the naive read = ||naive - carry||
    what a permutation recovers  = ||naive - carry|| - ||perm - carry||

⚠ Coefficients are ORACLE span coordinates ``c = X @ U0`` -- what a perfect address book
would emit before the deletion.  ⛔ That makes this a DIAGNOSTIC, not an arm: a read
handed the true feature is disqualified as a mechanism by charter B.3 forever.  It is
used here only because the question is about coordinate correspondence, and oracle
coefficients are the most realistic possible distribution for it.

⚠ This is FEATURE space, not the judge.  It bounds what the judge can see; it does not
predict a valid-fraction.  ⛔ The two are not interchangeable and no number here is
quoted as a composition score.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chlu.experiments.exp_w8_single_design import CovContentStore   # noqa: E402
from w8_single_design_dev import red_mask                           # noqa: E402

D = 2048
A = 64
# the VAL leg, to be directly comparable with the peer's 0.8941 -> 0.0584
PART = "C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
N_PROBE = 4096


def log(m=""):
    print(m, flush=True)


def directions(store, k):
    """The pinned basis's direction block for well k: (d, r), already orthonormal."""
    r = store.deployed_rank[k]
    return np.asarray(store.modes[k][1:1 + r], np.float64).T


def main() -> int:
    z = dict(np.load(PART, allow_pickle=True))
    lab = np.asarray(z["factor"], np.int64)
    pool = np.asarray(z["pool"], np.int64)
    K = int(lab.max()) + 1
    wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
    log(f"val-leg partition K={K} wells {[int(c.size) for c in wc]}")

    from chlu.experiments import exp_gen_factor_wells as fw
    F = fw.features_mmap(0)
    X = np.asarray(fw.gather(F, pool), np.float32)
    red = red_mask(pool)
    log(f"pool {pool.size:,} rows, red {int(red.sum()):,}")

    # probe rows: held OUT of the pool, so the diagnostic is not read off the fit set
    allr = np.arange(int(F.shape[0]), dtype=np.int64)
    out = np.setdiff1d(allr, pool, assume_unique=False)
    rng = np.random.default_rng(20_260_920)
    probe = np.sort(rng.choice(out, size=min(N_PROBE, out.size), replace=False))
    Xp = np.asarray(fw.gather(F, probe), np.float32)
    log(f"probe {probe.size:,} rows, drawn from the {out.size:,} rows OUTSIDE the pool")

    t0 = time.time()
    st = CovContentStore(X, wc, A, D, allow_saturated=True)
    U0 = [directions(st, k) for k in range(K)]
    rank0 = list(st.deployed_rank)
    st.delete_rows(X[red])
    U1 = [directions(st, k) for k in range(K)]
    log(f"build+delete {time.time() - t0:.1f}s; ranks {rank0} -> {st.deployed_rank}")

    rows = []
    tot = {"before_carry": 0.0, "naive_carry": 0.0, "perm_carry": 0.0, "scale": 0.0}
    log("\n== per well, feature-space L2 (relative to the well's own signal scale) ==")
    log("  well   r   reordered   REAL deletion   NAIVE gauge err   AFTER perm match")
    for k in range(K):
        r = min(U0[k].shape[1], U1[k].shape[1])
        if r == 0:
            continue
        A0, A1 = U0[k][:, :r], U1[k][:, :r]
        Xw = np.asarray(Xp[:, wc[k]], np.float64)
        c = Xw @ A0                                   # oracle pre-deletion coefficients

        M = A0.T @ A1                                 # correspondence
        # ⚠ CORRECTED. My first version used a per-row argmax, which can send two
        # pre-directions to ONE post-direction -- so what I called a "permutation" was
        # not guaranteed to be one, and the residual I attributed to degenerate rotation
        # was partly my own double-assignment.  The spoke's direction_correspondence()
        # is greedy on descending |overlap| and returns a true permutation; this uses
        # the same rule.
        Mabs = np.abs(M)
        perm = np.full(r, -1, np.int64)
        taken = np.zeros(r, bool)
        order = np.dstack(np.unravel_index(np.argsort(Mabs, axis=None)[::-1],
                                           Mabs.shape))[0]
        for i, j in order:
            if perm[i] < 0 and not taken[j]:
                perm[i], taken[j] = int(j), True
                if (perm >= 0).all():
                    break
        # perm[i] = which POST direction pre-direction i became.  The book slot i must
        # therefore drive post-direction perm[i].
        inv = np.empty(r, np.int64)
        inv[perm] = np.arange(r)                      # post j <- pre inv[j]
        best = inv
        sgn = np.sign(M[best, np.arange(r)])
        sgn[sgn == 0] = 1.0
        n_moved = int((perm != np.arange(r)).sum())
        n_dup_argmax = int(r - len(set(Mabs.argmax(axis=0).tolist())))

        before = c @ A0.T
        naive = c @ A1.T
        perm = (c[:, best] * sgn[None, :]) @ A1.T
        carry = (c @ M) @ A1.T                        # the correct post-deletion answer

        def l2(a, b):
            return float(np.sqrt(((a - b) ** 2).sum(axis=1)).mean())

        scale = float(np.sqrt((carry ** 2).sum(axis=1)).mean())
        d_real, d_naive, d_perm = l2(before, carry), l2(naive, carry), l2(perm, carry)
        tot["before_carry"] += d_real ** 2
        tot["naive_carry"] += d_naive ** 2
        tot["perm_carry"] += d_perm ** 2
        tot["scale"] += scale ** 2
        rows.append({"well": k, "r": r, "reordered": n_moved,
                     "argmax_double_assignments": n_dup_argmax,
                     "real_deletion_l2": d_real, "naive_gauge_l2": d_naive,
                     "perm_l2": d_perm, "scale": scale})
        log(f"  {k:4d} {r:4d}   {n_moved:3d}/{r:<4d}   {d_real:11.4f}   {d_naive:15.4f}   "
            f"{d_perm:16.4f}   (scale {scale:.3f}, argmax dup {n_dup_argmax})")

    real = float(np.sqrt(tot["before_carry"]))
    naive = float(np.sqrt(tot["naive_carry"]))
    perm = float(np.sqrt(tot["perm_carry"]))
    scale = float(np.sqrt(tot["scale"]))
    log("\n== whole-vector totals ==")
    log(f"  signal scale (post-deletion answer norm)      {scale:10.4f}")
    log(f"  REAL deletion effect   ||before - carry||     {real:10.4f}   "
        f"({100 * real / scale:6.2f}% of scale)")
    log(f"  NAIVE read error       ||naive  - carry||     {naive:10.4f}   "
        f"({100 * naive / scale:6.2f}% of scale)")
    log(f"  after permutation      ||perm   - carry||     {perm:10.4f}   "
        f"({100 * perm / scale:6.2f}% of scale)")
    rec = (naive - perm) / naive if naive > 0 else 0.0
    log(f"\n  permutation removes {100 * rec:.2f}% of the naive read error")
    log(f"  naive error / real deletion effect = {naive / max(real, 1e-12):.2f}x")
    verdict = ("THE NAIVE COLLAPSE IS MOSTLY GAUGE" if naive > 2 * real else
               "THE NAIVE COLLAPSE IS MOSTLY THE DELETION")
    log(f"\n  VERDICT: {verdict}")

    outd = Path("C:/chlu/outputs/w8-single-design/dev")
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "gauge_separation.json").write_text(json.dumps({
        "NOT_A_SCORED_RUN": True,
        "what": "separates slot-mismatch (gauge) from the deletion in the no-refit read",
        "caveats": [
            "FEATURE space, not the judge -- bounds what the judge can see, does not "
            "predict a valid-fraction",
            "ORACLE span coordinates as coefficients: a DIAGNOSTIC, disqualified as an "
            "arm by charter B.3",
            "val-leg partition, seed 0, probe rows drawn OUTSIDE the fit pool",
        ],
        "leg": "val", "K": K, "A": A, "n_probe": int(probe.size),
        "n_red_rows": int(red.sum()),
        "per_well": rows,
        "totals": {"scale": scale, "real_deletion_l2": real, "naive_l2": naive,
                   "perm_l2": perm,
                   "permutation_removes_frac_of_naive_error": rec,
                   "naive_over_real": naive / max(real, 1e-12)},
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    log(f"\nwrote {outd / 'gauge_separation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
