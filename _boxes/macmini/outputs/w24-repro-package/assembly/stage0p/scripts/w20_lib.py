"""w20 hub/shapes-data-matched -- shared library.

ONE rival set, data-matched to the store, replaces the two rival sets the draft carries:
  * w12 spoke/deletion-baselines' comparators: 131,072 params, fitted on 16,384 rows
    (that spoke's declared BUILD/SMOKE scale), 20 epochs, no tuning -> the T7 headline;
  * spoke-A's wave-4 set: 118,784 params, fitted on the 32,256-row Montero pool -> the
    ladder's `@ref` columns.
The store's own fit pool is 131,072 rows.  Referee M1 / M2 trace to that gap.

What lives here
---------------
* the pools (the store's EXACT pool per condition / rung, and its nested prefixes);
* `derangement` and `perturb_modes_w20` -- `landscape_shuffled_across_wells` as a TRUE
  DERANGEMENT (no well keeps its own block), with the permutation returned so every row
  can print it.  The historical `w12_crossover_lib.perturb_modes('well_permute')` rejects
  only the identity; at K = 2 that is the identity half the time before rejection and
  it can keep individual wells at any K (registry N350).  The other three controls are
  delegated to the historical function UNCHANGED, so they are bit-identical to T7's;
* `train_curve` -- `w4_learners.train` (Adam + warmup-cosine, MSE to the cached feature,
  identical optimiser, schedule and batching) with the loss curve RECORDED per epoch,
  and an optional held-out MSE per epoch.  Nothing about the optimisation is changed;
  only what is written down.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from chlu.paths import partition_file, run_dir  # noqa: E402

OUT = run_dir("shapes3d")
PART_VAL = str(partition_file("val"))
PART_TEST = str(partition_file("test"))
KSTAR_LADDER = run_dir("shapes3d", "ladder_nclu")

D = 2048
RUNGS = ("2", "3", "4", "5", "6")
FAMILIES = ("mlp", "lstm", "transformer", "cont_mlp")
POOL_SIZES = {"131k": 131_072, "32k": 32_256, "16k": 16_384}
#: the selection split inside the pool (the store's own address book uses a 90/10 split
#: of the same pool for its validation, `w12b_scored_leg`), fixed per seed.
SEL_VAL_FRAC = 0.10
SEL_RNG = 20_260_923
TENX = 10


def log(m=""):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def assert_worktree():
    """The imported package must be the one this script belongs to."""
    import chlu
    wt = Path(chlu.__file__).resolve()
    if ROOT not in wt.parents:
        raise SystemExit("the package resolves to %s, outside %s." % (wt, ROOT))
    return str(wt)


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------

def rung_partition(rung: str, seed: int) -> Path:
    # ⛔ MEASURED DEFECT, fixed 2026-09-24 ~00:30Z: the rung's POOL is seed-independent but
    # its DEAD-CHANNEL MASK is not (w12b built each leg's partitions from that seed's
    # features).  An earlier version read seed 0's file for every seed; seed-1/2 ladder
    # stores built with it did not reproduce w12b's ranks and were VOIDED.
    return KSTAR_LADDER / ("%s_seed%d_K32" % (rung, int(seed))) / "partitions.npz"


def load_pool(where: str, seed: int = 0):
    """where = 'val' | 'test' | 'rung:<k>' -> (pool int64[131072], dead bool[D])."""
    if where == "val":
        f = PART_VAL
    elif where == "test":
        f = PART_TEST
    elif where.startswith("rung:"):
        f = str(rung_partition(where.split(":", 1)[1], seed))
    else:
        raise ValueError(where)
    z = np.load(f, allow_pickle=True)
    return np.asarray(z["pool"], np.int64), np.asarray(z["dead"]).ravel().astype(bool), f


def selection_split(n: int, seed: int):
    rng = np.random.default_rng(SEL_RNG + int(seed))
    p = rng.permutation(int(n))
    nv = int(round(SEL_VAL_FRAC * n))
    return np.sort(p[nv:]), np.sort(p[:nv])


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def derangement(K: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random DERANGEMENT of range(K): no index maps to itself.

    Rejection sampling of uniform permutations; acceptance probability -> 1/e, so the
    expected number of draws is < 3 at any K >= 2.  Asserted, never assumed.
    """
    if K < 2:
        raise ValueError("no derangement of %d element(s) exists" % K)
    while True:
        p = rng.permutation(K)
        if not np.any(p == np.arange(K)):
            return p


def historical_well_perm(K: int, seed: int) -> np.ndarray:
    """The permutation `w12_crossover_lib.perturb_modes('well_permute')` USED for T7.

    Reproduced from its own RNG stream (`default_rng(seed*7919 + 20_260_921)`, first
    draw, identity rejected) so the historical control's fixed points can be PRINTED.
    """
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_921)
    perm = rng.permutation(K)
    while K > 1 and np.all(perm == np.arange(K)):
        perm = rng.permutation(K)
    return perm


def perturb_modes_w20(modes, kind: str, seed: int):
    """(perturbed modes, info).  `well_permute` is a DERANGEMENT; the rest are historical."""
    from w12_crossover_lib import perturb_modes
    if kind != "well_permute":
        return perturb_modes(modes, kind, seed), {"kind": kind, "definition": "historical"}
    Gs = [np.asarray(G, np.float32) for G in modes]
    K = len(Gs)
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_923)
    perm = derangement(K, rng)
    fixed = int(np.sum(perm == np.arange(K)))
    assert fixed == 0, "derangement has a fixed point -- refusing"
    out, crop = [], []
    for k in range(K):
        src = Gs[int(perm[k])]
        dst_w = Gs[k].shape[1]
        B = np.zeros((src.shape[0], dst_w), np.float32)
        m = min(dst_w, src.shape[1])
        B[:, :m] = src[:, :m]
        if src.shape[1] != dst_w:
            crop.append({"dst": k, "src": int(perm[k]), "src_w": int(src.shape[1]),
                         "dst_w": int(dst_w)})
        out.append(B)
    hist = historical_well_perm(K, seed)
    return out, {"kind": kind, "definition": "DERANGEMENT (w20)",
                 "perm": [int(x) for x in perm], "fixed_points": fixed,
                 "is_derangement": True, "crop_or_pad": crop,
                 "historical_T7_perm": [int(x) for x in hist],
                 "historical_T7_fixed_points": int(np.sum(hist == np.arange(K)))}


# ---------------------------------------------------------------------------
# training with the curve written down
# ---------------------------------------------------------------------------

def train_curve(model, Xin, Y, key, epochs: int, lr: float, wd: float,
                batch: int = 256, Xv=None, Yv=None, val_every: int = 1, logf=None):
    """`w4_learners.train`, verbatim in optimiser/schedule/batching, with the curve kept.

    curve[e] = {"epoch", "train_mse" (mean of the epoch's minibatch losses),
                "val_mse" (held-out, if given, every `val_every` epochs and at the end)}.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from chlu.eval.rivals import w4_learners as L

    n = int(Xin.shape[0])
    steps_per_epoch = max(n // batch, 1)
    total = steps_per_epoch * int(epochs)
    warm = max(total // 20, 1)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr, warmup_steps=warm,
        decay_steps=max(total, warm + 1), end_value=lr * 0.01)
    opt = optax.adamw(sched, weight_decay=wd)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, st, xb, yb):
        def loss_fn(mm):
            pred = jax.vmap(mm)(xb)
            return jnp.mean((pred - yb) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        upd, st = opt.update(grads, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, upd), st, loss

    Ynp = np.asarray(Y, np.float32)
    Xnp = L._as_model_input(Xin)
    curve = []
    first = last = None
    t0 = time.time()
    for ep in range(int(epochs)):
        key, sk = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sk, n))
        acc = []
        for i in range(steps_per_epoch):
            idx = perm[i * batch:(i + 1) * batch]
            model, opt_state, loss = step(model, opt_state, jnp.asarray(Xnp[idx]),
                                          jnp.asarray(Ynp[idx]))
            acc.append(loss)
        acc = [float(a) for a in acc]
        if first is None:
            first = acc[0]
        last = acc[-1]
        row = {"epoch": ep + 1, "train_mse": float(np.mean(acc)),
               "seconds": round(time.time() - t0, 1)}
        if Xv is not None and ((ep + 1) % val_every == 0 or ep + 1 == int(epochs)):
            row["val_mse"] = L.val_mse(model, Xv, Yv)
        curve.append(row)
        if logf is not None and ((ep + 1) % max(1, int(epochs) // 10) == 0 or ep == 0):
            logf("      epoch %d/%d train %.5f%s  (%.0fs)" % (
                ep + 1, epochs, row["train_mse"],
                (" val %.5f" % row["val_mse"]) if "val_mse" in row else "",
                time.time() - t0))
    return model, {"steps": total, "epochs": int(epochs), "lr": float(lr),
                   "weight_decay": float(wd), "batch": int(batch),
                   "loss_first": first, "loss_final": last, "curve": curve,
                   "seconds": round(time.time() - t0, 1)}


def jdump(obj, path: Path):
    from chlu.eval.rivals.w12_deletion_baselines import _pyify
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_pyify(obj), indent=1, default=str), encoding="utf-8")
    tmp.replace(path)



