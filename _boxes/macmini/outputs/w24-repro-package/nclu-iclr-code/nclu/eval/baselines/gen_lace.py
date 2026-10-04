"""gen lace."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.ode import odeint

LACE_UP_DIM = 128
LACE_DEPTH = 4
LACE_LEAKY_SLOPE = 0.2
LACE_N_EPOCHS = 50
LACE_BATCH = 128
LACE_LR = 0.0001
LACE_WARMUP_ITERS = 1000
LACE_SIGMA = 0.1
LACE_BETA_0, LACE_BETA_1, LACE_T = (0.1, 20.0, 1.0)
LACE_ODE_T_END = 0.001
LACE_ODE_TOL = 0.001
LACE_SGLD_LR, LACE_SGLD_STD, LACE_SGLD_STEPS = (0.01, 0.01, 100)
LACE_SGLD_STD_CORRECTED = float(np.sqrt(2.0 * LACE_SGLD_LR))
ENERGY_FORMS = ("categorical", "continuous", "both")
SHRINKAGES = ("none", "ledoit_wolf", 0.001)
WHITENING_D_LIMIT = 20000


def ledoit_wolf_shrinkage(X: np.ndarray) -> float:
    X = np.asarray(X, np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = Xc.T @ Xc / n
    mu = float(np.trace(S) / d)
    d2 = float(np.sum((S - mu * np.eye(d)) ** 2) / d)
    norms = np.einsum("ij,ij->i", Xc, Xc)
    b2 = float((np.sum(norms**2) / n - np.sum(S * S) * 1.0) / (n * d))
    b2 = max(min(b2, d2), 0.0)
    return float(b2 / d2) if d2 > 0 else 1.0


@dataclass
class Whitening:
    mu: np.ndarray
    W: np.ndarray
    W_inv: np.ndarray
    meta: dict

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        return ((X - self.mu[None, :]) @ self.W.T).astype(np.float32)

    def inverse(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, np.float32)
        return (U @ self.W_inv.T + self.mu[None, :]).astype(np.float32)


def fit_whitening(
    X: np.ndarray, shrinkage="ledoit_wolf", eps: float = 1e-08, rel_eig_floor: float = 1e-08
) -> Whitening:
    X = np.asarray(X, np.float64)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    n, d = X.shape
    if d > WHITENING_D_LIMIT:
        raise ValueError(
            f"D={d} exceeds {WHITENING_D_LIMIT}: full whitening is infeasible and a DECLARED PCA projection is required. ⛔ This function will not silently fall back to channel-only whitening."
        )
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    S = Xc.T @ Xc / max(n, 1)
    if shrinkage == "none":
        lam = 0.0
    elif shrinkage == "ledoit_wolf":
        lam = ledoit_wolf_shrinkage(X)
    else:
        lam = float(shrinkage)
    lam = float(min(max(lam, 0.0), 1.0))
    target = float(np.trace(S) / d)
    S_shrunk = (1.0 - lam) * S + lam * target * np.eye(d)
    w, V = np.linalg.eigh(S_shrunk)
    floor = max(float(eps), float(rel_eig_floor) * float(np.max(w)))
    n_floored = int(np.sum(w < floor))
    w = np.maximum(w, floor)
    W = V * (w ** (-0.5))[None, :] @ V.T
    W_inv = V * (w**0.5)[None, :] @ V.T
    return Whitening(
        mu=mu.astype(np.float32),
        W=W.astype(np.float32),
        W_inv=W_inv.astype(np.float32),
        meta={
            "D": int(d),
            "N_train_features": int(n),
            "shrinkage": shrinkage if isinstance(shrinkage, str) else float(shrinkage),
            "shrinkage_intensity": lam,
            "rel_eig_floor": float(rel_eig_floor),
            "eigenvalue_min": float(w.min()),
            "eigenvalue_max": float(w.max()),
            "n_eigenvalues_floored": n_floored,
            "lace_option": "1 — full D x D shrinkage whitening, native (⛔ no PCA, ⛔ no channel-only)",
        },
    )


def whitening_check(wh: Whitening, X: np.ndarray, n: int = 8192, seed: int = 0) -> dict:
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    if X.shape[0] > int(n):
        idx = np.random.default_rng(int(seed)).choice(X.shape[0], int(n), replace=False)
        X = X[np.sort(idx)]
    U = wh.forward(X)
    C = np.cov(np.asarray(U, np.float64), rowvar=False)
    d = C.shape[0]
    return {
        "n": int(U.shape[0]),
        "mean_abs_mean": float(np.mean(np.abs(U.mean(axis=0)))),
        "mean_abs_diag_minus_one": float(np.mean(np.abs(np.diag(C) - 1.0))),
        "mean_abs_offdiag": float((np.abs(C).sum() - np.abs(np.diag(C)).sum()) / (d * d - d)),
        "roundtrip_max_abs_err": float(np.max(np.abs(wh.inverse(U) - np.asarray(X, np.float32)))),
    }


def dense_embedder_dims(input_dim: int, up_dim: int = LACE_UP_DIM, depth: int = LACE_DEPTH):
    """``OFFICIAL IMPLEMENTATION:`` ``np.linspace(input_dim, up_dim, depth).astype(int)``.

    For ``(512, 128, 4)`` this is ``[512, 384, 256, 128]`` — exactly ``PAPER:``
    Appendix A.3's "four-layer MLP … Linear 384 / 256 / 128 / #logits".  A 1x1
    conv on a 1x1 map *is* a Linear, so the two descriptions are the same network.
    """
    return [int(x) for x in np.linspace(int(input_dim), int(up_dim), int(depth)).astype(int)]


class LaceClassifier(eqx.Module):
    layers: list
    cls_heads: list
    cont_heads: list
    cardinalities: tuple

    def __init__(
        self,
        input_dim: int,
        cardinalities: Sequence[int],
        key,
        up_dim: int = LACE_UP_DIM,
        depth: int = LACE_DEPTH,
    ):
        dims = dense_embedder_dims(input_dim, up_dim, depth)
        n_trunk = len(dims) - 1
        keys = jax.random.split(key, n_trunk + 2 * len(cardinalities))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(n_trunk)]
        off = n_trunk
        self.cls_heads = [
            eqx.nn.Linear(dims[-1], int(c), key=keys[off + j]) for j, c in enumerate(cardinalities)
        ]
        off += len(cardinalities)
        self.cont_heads = [
            eqx.nn.Linear(dims[-1], 1, key=keys[off + j]) for j in range(len(cardinalities))
        ]
        self.cardinalities = tuple((int(c) for c in cardinalities))

    @property
    def param_dtype(self):
        return self.layers[0].weight.dtype

    def trunk(self, u):
        x = jnp.asarray(u, self.param_dtype).reshape(-1)
        for lin in self.layers:
            x = jax.nn.leaky_relu(lin(x), LACE_LEAKY_SLOPE)
        return x

    def __call__(self, u):
        h = self.trunk(u)
        return (
            [head(h) for head in self.cls_heads],
            jnp.stack([head(h)[0] for head in self.cont_heads]),
        )


def train_classifier(
    U_train: np.ndarray,
    cls_labels: np.ndarray,
    val_labels: np.ndarray,
    cardinalities: Sequence[int],
    key,
    n_epochs: int = LACE_N_EPOCHS,
    batch: int = LACE_BATCH,
    lr: float = LACE_LR,
    warmup: int = LACE_WARMUP_ITERS,
    log: Optional[Callable] = None,
):
    U = np.asarray(U_train, np.float32)
    Y = np.asarray(cls_labels, np.int32)
    V = np.asarray(val_labels, np.float32)
    n, d = U.shape
    key, mk = jax.random.split(key)
    model = LaceClassifier(d, cardinalities, mk)
    dtype = model.param_dtype
    Uj, Yj, Vj = (jnp.asarray(U, dtype), jnp.asarray(Y, jnp.int32), jnp.asarray(V, dtype))
    steps_per_epoch = max(1, int(np.ceil(n / float(batch))))
    steps = int(n_epochs) * steps_per_epoch
    warm = min(int(warmup), max(steps - 1, 1))
    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, float(lr), warm), optax.constant_schedule(float(lr))], [warm]
    )
    opt = optax.adam(sched)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, ub, yb, vb):
        logits, cont = jax.vmap(m)(ub)
        ce = 0.0
        for j, lg in enumerate(logits):
            logp = jax.nn.log_softmax(lg, axis=-1)
            ce = ce - jnp.mean(jnp.take_along_axis(logp, yb[:, j][:, None], axis=-1))
        ce = ce / len(logits)
        l2 = 0.5 * jnp.mean((cont - vb) ** 2)
        return ce + l2

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(m, s, ub, yb, vb):
        loss, grads = grad_fn(m, ub, yb, vb)
        updates, s = opt.update(grads, s, m)
        return (eqx.apply_updates(m, updates), s, loss)

    hist = []
    bs = min(int(batch), n)
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (bs,), replace=False)
        model, opt_state, loss = step(model, opt_state, Uj[idx], Yj[idx], Vj[idx])
        hist.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    lace clf step {i + 1}/{steps}  loss={hist[-1]:.5f}")
    return (
        model,
        {
            "steps": int(steps),
            "steps_per_epoch": int(steps_per_epoch),
            "n_epochs": int(n_epochs),
            "batch": int(bs),
            "lr": float(lr),
            "warmup_iters": int(warm),
            "loss_first": hist[0] if hist else None,
            "loss_final": hist[-1] if hist else None,
        },
        key,
    )


def classifier_accuracy(model, U: np.ndarray, Y: np.ndarray, batch: int = 512) -> dict:
    """Per-factor top-1 of LACE's own classifier (its supervision, made visible)."""
    fwd = eqx.filter_jit(jax.vmap(model))
    dtype = model.param_dtype
    hits = None
    n = int(U.shape[0])
    for i in range(0, n, batch):
        logits, _ = fwd(jnp.asarray(U[i : i + batch], dtype))
        pr = np.stack([np.asarray(jnp.argmax(lg, axis=-1)) for lg in logits], axis=1)
        h = pr == np.asarray(Y[i : i + batch], np.int32)
        hits = h if hits is None else np.concatenate([hits, h])
    per = hits.mean(axis=0)
    return {"per_factor": [float(x) for x in per], "mean": float(per.mean()), "n": n}


def _term_energy(logits, cont, k, cls_t, val_t, form, sigma):
    """One term's conditional energy (lower = better satisfied)."""
    e = 0.0
    if form in ("categorical", "both"):
        e = e - jax.nn.log_softmax(logits[k])[cls_t]
    if form in ("continuous", "both"):
        e = e + 0.5 * (cont[k] - val_t) ** 2 / sigma**2
    return e


def make_energy_fn(model, form: str = "both", sigma: float = LACE_SIGMA) -> Callable:
    if form not in ENERGY_FORMS:
        raise ValueError(f"unknown energy form {form!r} (have {ENERGY_FORMS})")
    n_f = len(model.cardinalities)

    def energy(u, cls_t, val_t, ex_f, ex_c, ex_v):
        logits, cont = model(u)
        e = 0.0
        for k in range(n_f):
            e = e + _term_energy(logits, cont, k, cls_t[k], val_t[k], form, sigma)
            extra = _term_energy(logits, cont, k, ex_c, ex_v, form, sigma)
            e = e + jnp.where(ex_f == k, extra, 0.0)
        return e

    return energy


def _batched_energy(energy_fn):
    def E(U, cls_t, val_t, ex_f, ex_c, ex_v):
        return jax.vmap(energy_fn)(U, cls_t, val_t, ex_f, ex_c, ex_v)

    return E


def beta_of_t(t):
    """``OFFICIAL IMPLEMENTATION:`` ``beta_t = beta_0 + t*(beta_1 - beta_0)``."""
    return LACE_BETA_0 + t * (LACE_BETA_1 - LACE_BETA_0)


def sample_ode(
    model,
    req,
    key,
    form: str = "both",
    sigma: float = LACE_SIGMA,
    tol: float = LACE_ODE_TOL,
    t_end: float = LACE_ODE_T_END,
    batch: int = 256,
) -> np.ndarray:
    energy_fn = make_energy_fn(model, form, sigma)
    E = _batched_energy(energy_fn)
    dtype = model.param_dtype

    def field(U, tau, cls_t, val_t, ex_f, ex_c, ex_v):
        t = LACE_T - tau
        g = jax.grad(lambda u: jnp.sum(E(u, cls_t, val_t, ex_f, ex_c, ex_v)))(U)
        return -0.5 * beta_of_t(t) * g

    n, d = (req.n, int(model.layers[0].weight.shape[1]))
    times = jnp.asarray([0.0, LACE_T - t_end], dtype)
    out = np.empty((n, d), np.float32)
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        key, sk = jax.random.split(key)
        u0 = jax.random.normal(sk, (sl.stop - sl.start, d), dtype)
        sol = odeint(
            field,
            u0,
            times,
            jnp.asarray(req.cls_targets[sl], jnp.int32),
            jnp.asarray(req.val_targets[sl], dtype),
            jnp.asarray(req.extra_factor[sl], jnp.int32),
            jnp.asarray(req.extra_cls[sl], jnp.int32),
            jnp.asarray(req.extra_val[sl], dtype),
            rtol=float(tol),
            atol=float(tol),
        )
        out[sl] = np.asarray(sol[-1], np.float32)
    return out


def sample_ld(
    model,
    req,
    key,
    std: float = LACE_SGLD_STD,
    lr: float = LACE_SGLD_LR,
    steps: int = LACE_SGLD_STEPS,
    form: str = "both",
    sigma: float = LACE_SIGMA,
    batch: int = 512,
) -> np.ndarray:
    energy_fn = make_energy_fn(model, form, sigma)
    E = _batched_energy(energy_fn)
    dtype = model.param_dtype

    def total_neg(U, cls_t, val_t, ex_f, ex_c, ex_v):
        return -jnp.sum(E(U, cls_t, val_t, ex_f, ex_c, ex_v)) - 0.5 * jnp.sum(U * U)

    grad_fn = eqx.filter_jit(jax.grad(total_neg))
    n, d = (req.n, int(model.layers[0].weight.shape[1]))
    out = np.empty((n, d), np.float32)
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        key, sk = jax.random.split(key)
        U = jax.random.normal(sk, (sl.stop - sl.start, d), dtype)
        args = (
            jnp.asarray(req.cls_targets[sl], jnp.int32),
            jnp.asarray(req.val_targets[sl], dtype),
            jnp.asarray(req.extra_factor[sl], jnp.int32),
            jnp.asarray(req.extra_cls[sl], jnp.int32),
            jnp.asarray(req.extra_val[sl], dtype),
        )
        for _ in range(int(steps)):
            key, nk = jax.random.split(key)
            U = (
                U
                + float(lr) * grad_fn(U, *args)
                + float(std) * jax.random.normal(nk, U.shape, dtype)
            )
        out[sl] = np.asarray(U, np.float32)
    return out


SAMPLERS = {
    "ode": {
        "kind": "ode",
        "note": "OFFICIAL IMPLEMENTATION: VP probability-flow ODE, dopri5, atol=rtol=1e-3",
    },
    "ld_published": {
        "kind": "ld",
        "std": LACE_SGLD_STD,
        "note": "OFFICIAL IMPLEMENTATION: --sgld_std 1e-2 (T_eff = 0.005, ~200x cold)",
    },
    "ld_corrected": {
        "kind": "ld",
        "std": LACE_SGLD_STD_CORRECTED,
        "note": "DERIVED: std = sqrt(2*lr) — the Langevin-consistent noise for PAPER: Eq. (2)",
    },
}


def tuning_grid(
    sigmas: Sequence[float] = (0.03, LACE_SIGMA, 0.3, 1.0),
    forms: Sequence[str] = ENERGY_FORMS,
    shrinkages: Sequence = SHRINKAGES,
) -> list[dict]:
    """Every (sigma, energy form, shrinkage) trial, in a deterministic order."""
    out = []
    for sh in shrinkages:
        for form in forms:
            for s in sigmas:
                out.append({"sigma": float(s), "form": str(form), "shrinkage": sh})
    return out
