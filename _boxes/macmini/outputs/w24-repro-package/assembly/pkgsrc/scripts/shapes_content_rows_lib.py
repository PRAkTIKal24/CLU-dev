"""shapes content rows lib."""

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
from nclu.paths import partition_file, run_dir

OUT = run_dir("shapes3d", "content_rows")
MATCHED_OUT = run_dir("shapes3d")
PART_VAL = str(partition_file("val"))
PART_TEST = str(partition_file("test"))
KSTAR_LADDER = run_dir("shapes3d", "ladder_nclu")
D = 2048
RUNGS = ("2", "3", "4", "5", "6")
FAMILIES = ("mlp", "lstm", "transformer", "cont_mlp")
SEL_VAL_FRAC = 0.1
SEL_RNG = 20260923
TENX = 10


def log(m=""):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def assert_worktree():
    import nclu

    wt = Path(nclu.__file__).resolve()
    if ROOT not in wt.parents:
        raise SystemExit("nclu resolves to %s, OUTSIDE %s. Refusing to continue." % (wt, ROOT))
    return str(wt)


def rung_partition(rung: str, seed: int) -> Path:
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
    return (np.asarray(z["pool"], np.int64), np.asarray(z["dead"]).ravel().astype(bool), f)


def store_pool_n(rung: str, seed: int) -> dict:
    import re

    leg = KSTAR_LADDER / ("%s_seed%d_K32" % (rung, int(seed)))
    lt = (leg / "LEG.json").read_text(encoding="utf-8")
    pt = (leg / "partitions.json").read_text(encoding="utf-8")
    a = sorted(set((int(x) for x in re.findall('"n_fit_rows":\\s*(\\d+)', lt))))
    b = sorted(set((int(x) for x in re.findall('"n_store_train":\\s*(\\d+)', pt))))
    if len(a) != 1 or len(b) != 1 or a[0] != b[0]:
        raise SystemExit(
            "store pool size ambiguous for rung %s seed %d: LEG n_fit_rows=%s, partitions n_store_train=%s -- refusing"
            % (rung, seed, a, b)
        )
    return {
        "n": a[0],
        "LEG.json": str(leg / "LEG.json"),
        "partitions.json": str(leg / "partitions.json"),
    }


def selection_split(n: int, seed: int):
    rng = np.random.default_rng(SEL_RNG + int(seed))
    p = rng.permutation(int(n))
    nv = int(round(SEL_VAL_FRAC * n))
    return (np.sort(p[nv:]), np.sort(p[:nv]))


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
    rng = np.random.default_rng(int(seed) * 7919 + 20260921)
    perm = rng.permutation(K)
    while K > 1 and np.all(perm == np.arange(K)):
        perm = rng.permutation(K)
    return perm


def perturb_modes_deranged(modes, kind: str, seed: int):
    """(perturbed modes, info).  `well_permute` is a DERANGEMENT; the rest are historical."""
    from shapes_nclu_lib import perturb_modes

    if kind != "well_permute":
        return (perturb_modes(modes, kind, seed), {"kind": kind, "definition": "historical"})
    Gs = [np.asarray(G, np.float32) for G in modes]
    K = len(Gs)
    rng = np.random.default_rng(int(seed) * 7919 + 20260923)
    perm = derangement(K, rng)
    fixed = int(np.sum(perm == np.arange(K)))
    assert fixed == 0, "derangement has a fixed point -- refusing"
    out, crop = ([], [])
    for k in range(K):
        src = Gs[int(perm[k])]
        dst_w = Gs[k].shape[1]
        B = np.zeros((src.shape[0], dst_w), np.float32)
        m = min(dst_w, src.shape[1])
        B[:, :m] = src[:, :m]
        if src.shape[1] != dst_w:
            crop.append(
                {"dst": k, "src": int(perm[k]), "src_w": int(src.shape[1]), "dst_w": int(dst_w)}
            )
        out.append(B)
    hist = historical_well_perm(K, seed)
    return (
        out,
        {
            "kind": kind,
            "definition": "DERANGEMENT",
            "perm": [int(x) for x in perm],
            "fixed_points": fixed,
            "is_derangement": True,
            "crop_or_pad": crop,
            "historical_T7_perm": [int(x) for x in hist],
            "historical_T7_fixed_points": int(np.sum(hist == np.arange(K))),
        },
    )


def train_curve(
    model,
    Xin,
    Y,
    key,
    epochs: int,
    lr: float,
    wd: float,
    batch: int = 256,
    Xv=None,
    Yv=None,
    val_every: int = 1,
    logf=None,
):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from nclu.eval.baselines import learners as L

    n = int(Xin.shape[0])
    steps_per_epoch = max(n // batch, 1)
    total = steps_per_epoch * int(epochs)
    warm = max(total // 20, 1)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warm,
        decay_steps=max(total, warm + 1),
        end_value=lr * 0.01,
    )
    opt = optax.adamw(sched, weight_decay=wd)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, st, xb, yb):
        def loss_fn(mm):
            pred = jax.vmap(mm)(xb)
            return jnp.mean((pred - yb) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        upd, st = opt.update(grads, st, eqx.filter(m, eqx.is_array))
        return (eqx.apply_updates(m, upd), st, loss)

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
            idx = perm[i * batch : (i + 1) * batch]
            model, opt_state, loss = step(
                model, opt_state, jnp.asarray(Xnp[idx]), jnp.asarray(Ynp[idx])
            )
            acc.append(loss)
        acc = [float(a) for a in acc]
        if first is None:
            first = acc[0]
        last = acc[-1]
        row = {
            "epoch": ep + 1,
            "train_mse": float(np.mean(acc)),
            "seconds": round(time.time() - t0, 1),
        }
        if Xv is not None and ((ep + 1) % val_every == 0 or ep + 1 == int(epochs)):
            row["val_mse"] = L.val_mse(model, Xv, Yv)
        curve.append(row)
        if logf is not None and ((ep + 1) % max(1, int(epochs) // 10) == 0 or ep == 0):
            logf(
                "      epoch %d/%d train %.5f%s  (%.0fs)"
                % (
                    ep + 1,
                    epochs,
                    row["train_mse"],
                    " val %.5f" % row["val_mse"] if "val_mse" in row else "",
                    time.time() - t0,
                )
            )
    return (
        model,
        {
            "steps": total,
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(wd),
            "batch": int(batch),
            "loss_first": first,
            "loss_final": last,
            "curve": curve,
            "seconds": round(time.time() - t0, 1),
        },
    )


def jdump(obj, path: Path):
    from nclu.eval.baselines.refusal_comparator import _pyify

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_pyify(obj), indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
