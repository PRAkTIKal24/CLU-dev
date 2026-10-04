"""⭐ The **LACE** competitor arm, re-implemented on our frozen features.

Nie, Vahdat & Anandkumar, *"Controllable and Compositional Generation with
Latent-Space Energy-Based Models"*, NeurIPS 2021 (arXiv:2110.10873).  Constants
pinned from the paper and from the official code at
``NVlabs/LACE@dcfd4a57bee40fcb17ac8304413f1e35a64323cd`` (5 commits total; HEAD
*is* the paper release) via ``gen-lace-spec``.

⚠ **Licence.** The official implementation is under an **NVIDIA non-commercial**
licence.  ⛔ **Nothing is vendored here.**  This module is a clean re-implementation
written from the pinned specification; the constants it reproduces are facts about
the published method, each tagged ``PAPER:`` / ``OFFICIAL IMPLEMENTATION:`` /
``DERIVED:`` at its definition.

⛔ **`DO_NOT_USE_AS_A_BAR` (R9.8).** LACE's published CIFAR-10 FID 6.63 / ACC
0.972, FFHQ-glasses FID 20.93 / ACC 0.998 and unseen-combination 0.94–0.98 are
StyleGAN at up to 1024² on *their* splits with *their* labels.  Only this
re-implementation, on our features and our splits, is comparable to anything we
report.

---

## The one hard dependency, and why whitening is mandatory (R9.1)

LACE's ``½‖z‖²`` prior term is **not a modelling choice**: it is the *exact*
log-density of StyleGAN's z-space, which is ``N(0, I)`` by construction.  That
exactness is what licenses the shipped ODE drift to **omit the prior score
entirely** — ``VPODE.forward`` calls ``ccf.get_cond_energy``, not ``ccf``.  On raw
post-ReLU conv features the prior score does *not* cancel and the shipped drift
would be the **wrong vector field**.

⛔ A bare ``½‖x‖²`` on raw features is therefore a **silent crippling and is
refused**.  ⛔ A learned flow prior is also refused: it breaks the invariance
identity and stops being LACE.  What we do instead is fit a **shrinkage Gaussian**
over the cached train features and run the whole method in whitened coordinates
``u = Σ^{-1/2}(x − μ)``, where ``p₀(u) = N(0, I)`` *exactly* and Appendix A.2's
VP-SDE invariance — hence the ODE drift — survives verbatim.

The bridge measured ``D = 2048 ≤ 20000`` and recorded ``lace_option = 1``
(whitens natively).  ⛔ No PCA projection and no channel-only fallback are used or
needed here; :func:`fit_whitening` raises rather than degrade silently.

## Concessions granted to LACE, declared not hidden (R9.3)
1. **Dense factor labels** over the train split.  Our store gets none.  This makes
   LACE the stronger baseline, which is the direction we want.
2. **A validation split** for tuning the OR/NOT constants (``cof`` and the ``0.1``
   in ``α``).  Freezing them at the published 20/0.1 is the easiest way to hobble
   OR/NOT — the authors' own comments are literally ``# these values work well``.
   ⚠ Every request in our A/B/C protocol is a pure **conjunction**, so OR and NOT
   are never exercised by a reported number; the operators are implemented
   faithfully and the constants are calibrated as a declared side diagnostic.

## Constant traps this module does **not** fall into (R9.6)
* ``n_epochs = 50`` — the shipped ``run_clf.sh``.  ⛔ Not the appendix's rendered
  "100", ⛔ not argparse's 200.
* ``batch = 128`` (FFHQ) / 512 (CIFAR-10).  ⛔ The argparse default 64 is *not*
  what was run.
* The staircase decay (``decay_epochs [160, 180]``) **never fires** at 50 epochs;
  the effective schedule is a 1000-iteration linear warmup then a constant lr.
* ``dis_temp`` is **not** a clean temperature (it divides the gathered target
  logit but not the ``logsumexp``) and is ⛔ not swept.
* OR is **non-commutative** (``cof`` multiplies the left operand only) and there
  is a hard **10-operand** limit (operands are parsed as single characters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.ode import odeint

# --- published constants, each tagged with its provenance ------------------
#: ``PAPER:`` Appendix A.3 / ``OFFICIAL IMPLEMENTATION:`` ``DenseEmbedder(up_dim=128, depth=4)``.
LACE_UP_DIM = 128
LACE_DEPTH = 4
#: ``OFFICIAL IMPLEMENTATION:`` ``nn.LeakyReLU(0.2)`` between DenseEmbedder blocks.
LACE_LEAKY_SLOPE = 0.2
#: ``OFFICIAL IMPLEMENTATION:`` ``run_clf.sh --n_epochs 50``.  ⛔ NOT the appendix's 100, NOT argparse's 200.
LACE_N_EPOCHS = 50
#: ``OFFICIAL IMPLEMENTATION:`` ``run_clf.sh`` FFHQ batch 128 (CIFAR-10 512). ⛔ NOT argparse's 64.
LACE_BATCH = 128
#: ``OFFICIAL IMPLEMENTATION:`` ``lr=1e-4`` with a 1000-iteration linear warmup; the
#: staircase decay milestones are unreachable at 50 epochs (``DERIVED:``).
LACE_LR = 1e-4
LACE_WARMUP_ITERS = 1000
#: ``OFFICIAL IMPLEMENTATION:`` continuous-attribute energy divides by ``sigma**2`` with sigma=0.1.
#: ⚠ ``PAPER:``/``OFFICIAL IMPLEMENTATION:`` inconsistency kept faithfully: the training
#: loss (``utils.get_loss``) is a plain ``0.5*L2`` with **no** sigma.
LACE_SIGMA = 0.1
#: ``OFFICIAL IMPLEMENTATION:`` VP-SDE ``beta_0=0.1``, ``beta_1=20``, ``T=1``.
LACE_BETA_0, LACE_BETA_1, LACE_T = 0.1, 20.0, 1.0
#: ``OFFICIAL IMPLEMENTATION:`` ``integration_times = linspace(T, 1e-3, 2)``, dopri5, atol=rtol=1e-3.
LACE_ODE_T_END = 1e-3
LACE_ODE_TOL = 1e-3
#: ``OFFICIAL IMPLEMENTATION:`` ``--sgld_lr 1e-2 --sgld_std 1e-2 --n_steps 100``.
LACE_SGLD_LR, LACE_SGLD_STD, LACE_SGLD_STEPS = 1e-2, 1e-2, 100
#: ``OFFICIAL IMPLEMENTATION:`` ``logical_comb`` OR coefficient (``# these values work well``).
LACE_OR_COF = 20.0
#: ``OFFICIAL IMPLEMENTATION:`` ``logical_comb`` NOT numerator (``# this value works well``).
LACE_NOT_NUM = 0.1
#: ``DERIVED:`` (gen-lace-spec F5) operands are parsed as single characters.
LACE_MAX_OPERANDS = 10
#: ``DERIVED:`` (gen-lace-spec F2) ``T_eff = std**2 / (2*lr) = 0.005``: the shipped
#: Langevin is ~200x colder than its own target, i.e. near-deterministic gradient
#: descent.  The Langevin-consistent noise is ``sqrt(2*lr)``.
LACE_SGLD_STD_CORRECTED = float(np.sqrt(2.0 * LACE_SGLD_LR))

ENERGY_FORMS = ("categorical", "continuous", "both")
#: ⭐ ``"none"`` first: MEASURED to give the most exact N(0,I) prior (see
#: :func:`fit_whitening`), which is the identity LACE's ODE drift depends on.
SHRINKAGES = ("none", "ledoit_wolf", 1e-3)
#: ⭐ The bridge's own limit: above this a full ``D x D`` whitening is infeasible
#: and LACE would need a **declared** PCA projection (R14.3), never a silent drop.
WHITENING_D_LIMIT = 20_000


# ---------------------------------------------------------------------------
# ⭐ Whitening — mandatory, and the reason is in the module docstring
# ---------------------------------------------------------------------------


def ledoit_wolf_shrinkage(X: np.ndarray) -> float:
    """Ledoit-Wolf optimal shrinkage intensity toward ``(tr(S)/D)·I``.

    Closed form (Ledoit & Wolf 2004, eq. for the single-factor target); ⛔ no
    sklearn dependency is introduced for one scalar.
    """
    X = np.asarray(X, np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / n
    mu = float(np.trace(S) / d)
    d2 = float(np.sum((S - mu * np.eye(d)) ** 2) / d)
    # b2 = mean over samples of ||x x^T - S||_F^2 / d, capped at d2 (Ledoit-Wolf)
    norms = np.einsum("ij,ij->i", Xc, Xc)
    b2 = float((np.sum(norms**2) / n - np.sum(S * S) * 1.0) / (n * d))
    b2 = max(min(b2, d2), 0.0)
    return float(b2 / d2) if d2 > 0 else 1.0


@dataclass
class Whitening:
    """``u = W (x − μ)`` with ``W = Σ_λ^{-1/2}`` and its exact inverse.

    ⭐ In these coordinates ``p₀(u) = N(0, I)`` *exactly*, which is precisely the
    property LACE's ODE drift assumes when it omits the prior score.
    """

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
        return ((U @ self.W_inv.T) + self.mu[None, :]).astype(np.float32)


def fit_whitening(
    X: np.ndarray, shrinkage="ledoit_wolf", eps: float = 1e-8, rel_eig_floor: float = 1e-8
) -> Whitening:
    """Fit the shrinkage Gaussian over cached **train** features.

    Args:
        X: ``(N, D)`` cached train features (⛔ never a held-out row).
        shrinkage: ``"none"`` (⭐ the default arm — no shrinkage, see below),
            ``"ledoit_wolf"``, or an explicit intensity in ``[0, 1]``.
        rel_eig_floor: eigenvalues below ``rel_eig_floor * max_eigenvalue`` are
            floored.  ⭐ **This is what makes ``shrinkage="none"`` well-posed and
            is the right regulariser here.**  MEASURED on the shipped Shapes3D
            cache (16,384 train rows, D=2048): shrinkage *is itself* what breaks
            LACE's prior identity — ``mean |diag(cov(u)) - 1|`` is **0.1520** at
            the Ledoit-Wolf intensity (3.49e-4), **0.2422** at 1e-3, and
            **0.0343** with no shrinkage and this floor, which trips exactly the
            **72 near-null eigendirections** produced by the cache's dead
            (constant post-ReLU) coordinates and nothing else.  Those directions
            carry no information and *cannot* be whitened; every other direction
            is then whitened exactly.

    ⛔ Raises if ``D > WHITENING_D_LIMIT``: at that width the arm would need a
    **declared** PCA projection (R14.3, Head ruling — our choices never weaken a
    competitor), and silently falling back to channel-only whitening is exactly
    the degradation that ruling forbids.
    """
    X = np.asarray(X, np.float64)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    n, d = X.shape
    if d > WHITENING_D_LIMIT:
        raise ValueError(
            f"D={d} exceeds {WHITENING_D_LIMIT}: full whitening is infeasible and a "
            "DECLARED PCA projection is required (R14.3). ⛔ This function will not "
            "silently fall back to channel-only whitening."
        )
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    S = (Xc.T @ Xc) / max(n, 1)
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
    W = (V * (w ** -0.5)[None, :]) @ V.T
    W_inv = (V * (w**0.5)[None, :]) @ V.T
    return Whitening(
        mu=mu.astype(np.float32),
        W=W.astype(np.float32),
        W_inv=W_inv.astype(np.float32),
        meta={
            "D": int(d),
            "N_train_features": int(n),
            "shrinkage": (shrinkage if isinstance(shrinkage, str) else float(shrinkage)),
            "shrinkage_intensity": lam,
            "rel_eig_floor": float(rel_eig_floor),
            "eigenvalue_min": float(w.min()),
            "eigenvalue_max": float(w.max()),
            "n_eigenvalues_floored": n_floored,
            "lace_option": "1 — full D x D shrinkage whitening, native (⛔ no PCA, ⛔ no channel-only)",
        },
    )


def whitening_check(wh: Whitening, X: np.ndarray, n: int = 8192, seed: int = 0) -> dict:
    """⭐ Verify the claim: whitened train features really have ``mean≈0``, ``cov≈I``.

    ⛔ **Subsamples at RANDOM, never a head slice.**  The experiment's fit rows are
    ``np.sort``ed row indices, and a Shapes3D row index is a C-order encoding of the
    factor tuple — so ``X[:n]`` is not a sample of the training distribution, it is
    the **lowest-`floor_hue` sub-population**.  Whitening by the *full* covariance
    does not make a biased sub-population isotropic, so checking on a head slice
    measures the wrong thing and reports a violation that is not there (measured:
    ``mean |diag-1|`` 0.26 on the head slice vs 0.034 on a random draw, same
    whitening).  ⚠ This is a property of the CHECK, not of the whitening.
    """
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


# ---------------------------------------------------------------------------
# The classifier: LACE's DenseEmbedder, on a vector instead of a 1x1 map
# ---------------------------------------------------------------------------


def dense_embedder_dims(input_dim: int, up_dim: int = LACE_UP_DIM, depth: int = LACE_DEPTH):
    """``OFFICIAL IMPLEMENTATION:`` ``np.linspace(input_dim, up_dim, depth).astype(int)``.

    For ``(512, 128, 4)`` this is ``[512, 384, 256, 128]`` — exactly ``PAPER:``
    Appendix A.3's "four-layer MLP … Linear 384 / 256 / 128 / #logits".  A 1x1
    conv on a 1x1 map *is* a Linear, so the two descriptions are the same network.
    """
    return [int(x) for x in np.linspace(int(input_dim), int(up_dim), int(depth)).astype(int)]


class LaceClassifier(eqx.Module):
    """LACE's attribute classifier over whitened features.

    One shared DenseEmbedder trunk, then **per-factor** heads: categorical logits
    (LACE's discrete-attribute head) *and* a scalar regression output (LACE's
    continuous-attribute head).  Both are kept because Shapes3D's factors are
    ordinal-valued and the two energies behave very differently on an
    ``out_of_range`` request — see :func:`conditional_energy`.
    """

    layers: list
    cls_heads: list
    cont_heads: list
    cardinalities: tuple

    def __init__(self, input_dim: int, cardinalities: Sequence[int], key,
                 up_dim: int = LACE_UP_DIM, depth: int = LACE_DEPTH):
        dims = dense_embedder_dims(input_dim, up_dim, depth)
        n_trunk = len(dims) - 1
        keys = jax.random.split(key, n_trunk + 2 * len(cardinalities))
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(n_trunk)
        ]
        off = n_trunk
        self.cls_heads = [
            eqx.nn.Linear(dims[-1], int(c), key=keys[off + j])
            for j, c in enumerate(cardinalities)
        ]
        off += len(cardinalities)
        self.cont_heads = [
            eqx.nn.Linear(dims[-1], 1, key=keys[off + j]) for j in range(len(cardinalities))
        ]
        self.cardinalities = tuple(int(c) for c in cardinalities)

    @property
    def param_dtype(self):
        return self.layers[0].weight.dtype

    def trunk(self, u):
        # ⚠ §7.23: six test modules enable x64 at import time and eqx then builds
        # float64 weights; a hard-coded float32 input would crash. Cast to the
        # module's OWN parameter dtype (the gen_bridge exemplar).
        x = jnp.asarray(u, self.param_dtype).reshape(-1)
        for lin in self.layers:
            x = jax.nn.leaky_relu(lin(x), LACE_LEAKY_SLOPE)
        return x

    def __call__(self, u):
        h = self.trunk(u)
        return [head(h) for head in self.cls_heads], jnp.stack(
            [head(h)[0] for head in self.cont_heads]
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
    """Fit LACE's classifier on **whitened** train features + dense factor labels.

    ``PAPER:``/``OFFICIAL IMPLEMENTATION:`` cross-entropy for discrete attributes and a
    plain ``0.5·L2`` for continuous ones — ⚠ **with no sigma in the loss**, which is
    the inconsistency with the energy's ``/sigma**2``.  Reproduced faithfully.
    """
    U = np.asarray(U_train, np.float32)
    Y = np.asarray(cls_labels, np.int32)
    V = np.asarray(val_labels, np.float32)
    n, d = U.shape
    key, mk = jax.random.split(key)
    model = LaceClassifier(d, cardinalities, mk)
    dtype = model.param_dtype
    Uj, Yj, Vj = jnp.asarray(U, dtype), jnp.asarray(Y, jnp.int32), jnp.asarray(V, dtype)

    steps_per_epoch = max(1, int(np.ceil(n / float(batch))))
    steps = int(n_epochs) * steps_per_epoch
    warm = min(int(warmup), max(steps - 1, 1))
    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, float(lr), warm), optax.constant_schedule(float(lr))],
        [warm],
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
        return eqx.apply_updates(m, updates), s, loss

    hist = []
    bs = min(int(batch), n)
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (bs,), replace=False)
        model, opt_state, loss = step(model, opt_state, Uj[idx], Yj[idx], Vj[idx])
        hist.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    lace clf step {i + 1}/{steps}  loss={hist[-1]:.5f}")
    return model, {"steps": int(steps), "steps_per_epoch": int(steps_per_epoch),
                   "n_epochs": int(n_epochs), "batch": int(bs), "lr": float(lr),
                   "warmup_iters": int(warm), "loss_first": hist[0] if hist else None,
                   "loss_final": hist[-1] if hist else None}, key


def classifier_accuracy(model, U: np.ndarray, Y: np.ndarray, batch: int = 512) -> dict:
    """Per-factor top-1 of LACE's own classifier (its supervision, made visible)."""
    fwd = eqx.filter_jit(jax.vmap(model))
    dtype = model.param_dtype
    hits = None
    n = int(U.shape[0])
    for i in range(0, n, batch):
        logits, _ = fwd(jnp.asarray(U[i : i + batch], dtype))
        pr = np.stack([np.asarray(jnp.argmax(lg, axis=-1)) for lg in logits], axis=1)
        h = (pr == np.asarray(Y[i : i + batch], np.int32))
        hits = h if hits is None else np.concatenate([hits, h])
    per = hits.mean(axis=0)
    return {"per_factor": [float(x) for x in per], "mean": float(per.mean()), "n": n}


# ---------------------------------------------------------------------------
# ⭐ The conditional energy, and the AND / OR / NOT algebra
# ---------------------------------------------------------------------------


def _term_energy(logits, cont, k, cls_t, val_t, form, sigma):
    """One term's conditional energy (lower = better satisfied)."""
    e = 0.0
    if form in ("categorical", "both"):
        # ``PAPER:`` Eq. (5): E(z, c) = -log p(c | z) for a discrete attribute.
        e = e - jax.nn.log_softmax(logits[k])[cls_t]
    if form in ("continuous", "both"):
        # ``OFFICIAL IMPLEMENTATION:`` scaled L2 with sigma**2 = 0.01.
        e = e + 0.5 * (cont[k] - val_t) ** 2 / (sigma**2)
    return e


def make_energy_fn(model, form: str = "both", sigma: float = LACE_SIGMA) -> Callable:
    """``E(u, cls_t, val_t, extra_factor, extra_cls, extra_val) -> scalar``.

    ⭐ **AND is a sum of conditional energies** (``PAPER:`` §3.3).  A
    ``duplicate_factor`` request supplies an ``extra`` term on a factor that
    already has one, so both energies enter the sum and **cannot both be
    satisfied** — the residual is what makes the request detectable.

    ⚠ An ``out_of_range`` value is expressible by the **continuous** head only:
    ``cls_t`` is the nearest class, because LACE's categorical vocabulary is
    **closed** and literally has no symbol for an off-grid value (R9.7).  That
    asymmetry is LACE's structure, declared, not our thumb on the scale.
    """
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


def logical_comb(neg_energies, ops: str, cof: float = LACE_OR_COF,
                 not_num: float = LACE_NOT_NUM):
    """``OFFICIAL IMPLEMENTATION:`` ``FFHQ/utils.py::logical_comb``, on **negative** energies.

    ``ops`` is a string of single-character operands separated by ``*`` (OR),
    ``+`` (AND) or prefixed by ``-`` (NOT), exactly as in the original.

    ⚠ Three properties the paper does not advertise, reproduced here verbatim
    (``DERIVED:``, gen-lace-spec F5):
      1. ⛔ **OR is not commutative** — ``cof`` multiplies the *left* operand only,
         so ``'0*1' != '1*0'``, and a chained OR re-applies ``cof`` to the running
         accumulator at each fold.
      2. ⛔ **At most 10 operands** — operands are single characters.
      3. ``α = min(not_num / |a|, 1)`` is **scale-coupled** to the original's
         energy magnitudes; on our features it is a tuned constant, not a law
         (R9.3 grants tuning it on a validation split).
    """
    toks = [t for t in ops.replace("*", " * ").replace("+", " + ").replace("-", " - ").split() if t]
    n_ops = sum(1 for t in toks if t.isnumeric())
    if n_ops > LACE_MAX_OPERANDS:
        raise ValueError(
            f"{n_ops} operands: LACE parses operands as single characters, so "
            f"{LACE_MAX_OPERANDS} is a hard structural limit"
        )
    acc, pending, negate = None, None, False
    for t in toks:
        if t in ("*", "+"):
            pending = t
            continue
        if t == "-":
            negate = True
            continue
        a = neg_energies[int(t)]
        if negate:
            alpha = jnp.clip(not_num / jnp.abs(a), 0.0, 1.0)
            a = -alpha * a
            negate = False
        if acc is None:
            acc = a
        elif pending == "+":
            acc = acc + a
        elif pending == "*":
            acc = jnp.log(jnp.exp(acc) * float(cof) + jnp.exp(a))
        else:
            raise ValueError(f"two operands with no operator between them in {ops!r}")
    return acc


# ---------------------------------------------------------------------------
# Samplers (⭐ both Langevin temperatures — R9.4)
# ---------------------------------------------------------------------------


def _batched_energy(energy_fn):
    def E(U, cls_t, val_t, ex_f, ex_c, ex_v):
        return jax.vmap(energy_fn)(U, cls_t, val_t, ex_f, ex_c, ex_v)

    return E


def energy_of(model, U, req, form: str = "both", sigma: float = LACE_SIGMA,
              batch: int = 512) -> np.ndarray:
    """Composed conditional energy of each request at its sample (the refusal score)."""
    E = eqx.filter_jit(_batched_energy(make_energy_fn(model, form, sigma)))
    dtype = model.param_dtype
    out = []
    for i in range(0, int(U.shape[0]), batch):
        sl = slice(i, i + batch)
        out.append(
            np.asarray(
                E(
                    jnp.asarray(U[sl], dtype),
                    jnp.asarray(req.cls_targets[sl], jnp.int32),
                    jnp.asarray(req.val_targets[sl], dtype),
                    jnp.asarray(req.extra_factor[sl], jnp.int32),
                    jnp.asarray(req.extra_cls[sl], jnp.int32),
                    jnp.asarray(req.extra_val[sl], dtype),
                ),
                np.float64,
            )
        )
    return np.concatenate(out) if out else np.zeros(0, np.float64)


def beta_of_t(t):
    """``OFFICIAL IMPLEMENTATION:`` ``beta_t = beta_0 + t*(beta_1 - beta_0)``."""
    return LACE_BETA_0 + t * (LACE_BETA_1 - LACE_BETA_0)


def sample_ode(model, req, key, form: str = "both", sigma: float = LACE_SIGMA,
               tol: float = LACE_ODE_TOL, t_end: float = LACE_ODE_T_END,
               batch: int = 256) -> np.ndarray:
    """⭐ LACE-ODE: the VP probability-flow ODE, ``dopri5``, in whitened coordinates.

    ``OFFICIAL IMPLEMENTATION:`` ``dz_dt = -0.5 * beta_t * grad(get_cond_energy)`` with
    ``integration_times = linspace(T, 1e-3, 2)``.  ``get_cond_energy`` returns the
    **negative** energy and the prior score is **absent** — which is valid here
    *only* because we are in whitened coordinates where ``p₀ = N(0, I)`` exactly.

    Implemented as ``tau = T - t`` (``jax.experimental.ode.odeint`` integrates
    forward), so the vector field is ``du/dtau = -0.5 * beta(T - tau) * grad E``:
    a descent on the composed energy, identical to integrating LACE's field
    backwards in time.
    ⚠ ``jax.experimental.ode.odeint`` does not expose an NFE counter, so we report
    the tolerances and the solver rather than a fabricated NFE.
    """
    energy_fn = make_energy_fn(model, form, sigma)
    E = _batched_energy(energy_fn)
    dtype = model.param_dtype

    def field(U, tau, cls_t, val_t, ex_f, ex_c, ex_v):
        t = LACE_T - tau
        g = jax.grad(lambda u: jnp.sum(E(u, cls_t, val_t, ex_f, ex_c, ex_v)))(U)
        return -0.5 * beta_of_t(t) * g

    n, d = req.n, int(model.layers[0].weight.shape[1])
    times = jnp.asarray([0.0, LACE_T - t_end], dtype)
    out = np.empty((n, d), np.float32)
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        key, sk = jax.random.split(key)
        u0 = jax.random.normal(sk, (sl.stop - sl.start, d), dtype)
        sol = odeint(
            field, u0, times,
            jnp.asarray(req.cls_targets[sl], jnp.int32),
            jnp.asarray(req.val_targets[sl], dtype),
            jnp.asarray(req.extra_factor[sl], jnp.int32),
            jnp.asarray(req.extra_cls[sl], jnp.int32),
            jnp.asarray(req.extra_val[sl], dtype),
            rtol=float(tol), atol=float(tol),
        )
        out[sl] = np.asarray(sol[-1], np.float32)
    return out


def sample_ld(model, req, key, std: float = LACE_SGLD_STD, lr: float = LACE_SGLD_LR,
              steps: int = LACE_SGLD_STEPS, form: str = "both",
              sigma: float = LACE_SIGMA, batch: int = 512) -> np.ndarray:
    """LACE-LD (``sample_q_sgld``): ``z += lr * grad(-E_total) + std * randn``.

    ⭐ ``E_total`` here **includes** the ``0.5‖u‖²`` prior — unlike the ODE arm,
    ``sample_q_sgld`` calls ``ccf.forward``, not ``get_cond_energy``.

    ⚠ ``DERIVED:`` at the published ``std = lr = 1e-2`` the effective temperature is
    ``std²/(2·lr) = 0.005``, ~200x colder than Eq. (2)'s own target, so as-published
    LACE-LD is **near-deterministic gradient descent** and will not cover modes.
    R9.4 requires both this and ``std = sqrt(2·lr)``; LACE's best of the two is
    what gets reported.
    """
    energy_fn = make_energy_fn(model, form, sigma)
    E = _batched_energy(energy_fn)
    dtype = model.param_dtype

    def total_neg(U, cls_t, val_t, ex_f, ex_c, ex_v):
        return -jnp.sum(E(U, cls_t, val_t, ex_f, ex_c, ex_v)) - 0.5 * jnp.sum(U * U)

    grad_fn = eqx.filter_jit(jax.grad(total_neg))
    n, d = req.n, int(model.layers[0].weight.shape[1])
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
            U = U + float(lr) * grad_fn(U, *args) + float(std) * jax.random.normal(
                nk, U.shape, dtype
            )
        out[sl] = np.asarray(U, np.float32)
    return out


#: The sampler arms R9.4 requires, with their declared temperatures.
SAMPLERS = {
    "ode": {"kind": "ode", "note": "OFFICIAL IMPLEMENTATION: VP probability-flow ODE, dopri5, atol=rtol=1e-3"},
    "ld_published": {"kind": "ld", "std": LACE_SGLD_STD,
                     "note": "OFFICIAL IMPLEMENTATION: --sgld_std 1e-2 (T_eff = 0.005, ~200x cold)"},
    "ld_corrected": {"kind": "ld", "std": LACE_SGLD_STD_CORRECTED,
                     "note": "DERIVED: std = sqrt(2*lr) — the Langevin-consistent noise for PAPER: Eq. (2)"},
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
