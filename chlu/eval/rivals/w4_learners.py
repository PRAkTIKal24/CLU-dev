"""The generic learners, at MATCHED TOTAL PARAMETERS.

Four generic request-to-feature regressors, each solving exactly the problem the
prototype table solves in closed form: map a request over the six factors to a
point in the frozen encoder's 2048-dimensional feature space, then hand that
point to THE frozen decoder and THE frozen judge.

* :class:`MLPRival` - request one-hots (sum of cardinalities) -> 2048
* :class:`TransformerRival` - 6 factor tokens, 2 layers -> 2048
* :class:`LSTMRival` - 6 factor tokens, recurrent -> 2048
* :class:`ContMLPRival` - the six factor values as NUMBERS -> 2048. This one
  exists for row 2 and accepts an off-grid hue: its input is the min-max
  normalised value, so a hue between two grid points, or outside the grid
  entirely, is simply a different number. See :func:`offgrid_demo`.

Matched parameters
------------------
Every learner's TOTAL parameter count - embeddings, biases, norms, output head,
everything - is matched to a single declared target within +/-10 percent, and the
realised count is printed and stored beside every number it produces.
:func:`match_width` searches the family's width for the closest total rather than
trusting a formula, so an architecture change cannot silently drift the count.

If a family cannot be brought inside the band at any width, :func:`match_width`
raises with the achievable count on both sides. A mismatched rival is never
quietly shipped.

Tuning
------
Each learner gets an 8-trial budget, in two stages. Learning rate and weight
decay are selected on a SHORT schedule (6 trials) against validation MSE on the
cached features - the same tune-short idiom the LACE arm uses. The epoch axis is
then selected at FULL length (2 trials), because an axis whose trials all run for
the same number of epochs would be selected by the RNG stream rather than by the
data, and handing a rival a schedule chosen by noise is weakening it through our
own code. The specialists are entitled to 36; the generic learners realise 8, and
that asymmetry is declared rather than hidden.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

#: The tuning grid. lr x weight decay is selected on a short schedule (6 trials);
#: the epoch axis is then selected at FULL length (2 more), because an axis whose
#: trials all run for the same number of epochs is selected by the RNG stream
#: rather than by the data. 8 trials in total, identical for all four families.
LR_GRID = (3e-4, 1e-3, 3e-3)
WD_GRID = (0.0, 1e-4)
EPOCH_GRID = (20, 40)
#: Epochs used for the SHORT selection schedule.
TUNE_EPOCHS = 8
BATCH = 256
#: How far a realised parameter count may sit from the target.
PARAM_TOLERANCE = 0.10


# ---------------------------------------------------------------------------
# Parameter counting and width matching
# ---------------------------------------------------------------------------


def count_params(model) -> int:
    """TOTAL trainable parameters: every array leaf, including biases and norms."""
    return int(sum(np.asarray(x).size
                   for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))))


def match_width(build: Callable[[int], object], target: int, lo: int = 1,
                hi: int = 512, tol: float = PARAM_TOLERANCE,
                log: Optional[Callable] = None) -> tuple:
    """Smallest-error width whose TOTAL parameter count is within ``tol`` of target.

    The count is measured by building the model, never by a formula, so it cannot
    drift away from the architecture.

    Raises:
        AssertionError: if no width in ``[lo, hi]`` lands inside the band. The
            message carries the closest achievable count on each side, because
            "this family cannot be matched" is a reportable result and shipping a
            mismatched rival instead is not.
    """
    # The total count is strictly increasing in width for every family here, so
    # bisect rather than scan: building a model costs 1-2 s and a linear sweep
    # would spend minutes per family to land on the same width.
    seen: dict[int, int] = {}

    def n_at(w: int) -> int:
        if w not in seen:
            seen[w] = count_params(build(w))
        return seen[w]

    a, b = int(lo), int(hi)
    while a < b:
        mid = (a + b) // 2
        if n_at(mid) < target:
            a = mid + 1
        else:
            b = mid
    # `a` is the smallest width at or above target; the best is `a` or `a - 1`.
    for w in (a - 1, a, a + 1):
        if int(lo) <= w <= int(hi):
            n_at(w)
    best_w = min(seen, key=lambda w: abs(seen[w] - target))
    best_n = seen[best_w]
    best_err = abs(best_n - target) / float(target)
    below = max(((w, n) for w, n in seen.items() if n <= target),
                key=lambda t: t[1], default=None)
    above = min(((w, n) for w, n in seen.items() if n >= target),
                key=lambda t: t[1], default=None)
    if best_err > tol:
        raise AssertionError(
            f"cannot match {target} parameters within +/-{tol:.0%} at any width in "
            f"[{lo}, {hi}]: closest is width {best_w} at {best_n} params "
            f"({best_err:+.2%}); nearest below={below}, nearest above={above}. "
            "Report the achievable count and stop; do not ship a mismatched rival.")
    if log is not None:
        log(f"    matched width={best_w}: {best_n} params vs target {target} "
            f"({(best_n - target) / target:+.2%})")
    return int(best_w), int(best_n), float((best_n - target) / target)


# ---------------------------------------------------------------------------
# Input encodings
# ---------------------------------------------------------------------------


def factor_offsets(cardinalities: Sequence[int]) -> np.ndarray:
    """Start index of each factor inside the concatenated one-hot / token vocab."""
    return np.concatenate([[0], np.cumsum(np.asarray(cardinalities, np.int64))[:-1]])


def token_vocab(cardinalities: Sequence[int]) -> int:
    """Grid values, plus ONE "not requested" token per factor.

    Rows 3 and 4 ask partially-specified requests - a corner constrains some
    factors and leaves the rest unconstrained - and ``arm_features``'s contract
    spells an unconstrained factor as ``nan``. A token family with no way to say
    "absent" would have to answer those rows out of distribution, or not at all.
    """
    return int(sum(int(c) for c in cardinalities)) + len(list(cardinalities))


def onehot_inputs(cls_targets: np.ndarray, cardinalities: Sequence[int],
                  present: Optional[np.ndarray] = None) -> np.ndarray:
    """``(N, F)`` class indices -> ``(N, sum C_k)`` concatenated one-hots.

    An absent factor is the all-zero block, which is the natural spelling of
    "no value asserted" for a one-hot and needs no extra width.
    """
    cards = [int(c) for c in cardinalities]
    off = factor_offsets(cards)
    n, f = cls_targets.shape
    out = np.zeros((n, int(sum(cards))), np.float32)
    idx = np.asarray(cls_targets, np.int64) + off[None, :]
    rows = np.repeat(np.arange(n), f)
    flat = idx.reshape(-1)
    if present is None:
        out[rows, flat] = 1.0
    else:
        m = np.asarray(present, bool).reshape(-1)
        out[rows[m], flat[m]] = 1.0
    return out


def token_inputs(cls_targets: np.ndarray, cardinalities: Sequence[int],
                 present: Optional[np.ndarray] = None) -> np.ndarray:
    """``(N, F)`` class indices -> ``(N, F)`` shared-vocabulary token ids.

    An absent factor gets its own reserved token, ``sum(C_k) + factor_index``.
    """
    cards = [int(c) for c in cardinalities]
    off = factor_offsets(cards)
    tok = (np.asarray(cls_targets, np.int64) + off[None, :]).astype(np.int32)
    if present is not None:
        base = int(sum(cards))
        absent = base + np.arange(len(cards), dtype=np.int32)[None, :]
        tok = np.where(np.asarray(present, bool), tok, absent).astype(np.int32)
    return tok


def continuous_inputs(val_targets: np.ndarray,
                      present: Optional[np.ndarray] = None) -> np.ndarray:
    """``(N, F)`` normalised values -> ``(N, 2F)``: the value, and a requested flag.

    The value channel carries the min-max normalised number, so an off-grid hue
    is simply a number between (or outside) the grid points. The flag channel is
    what lets an absent factor be absent rather than be zero, which is a real
    value on this scale.
    """
    V = np.asarray(val_targets, np.float32)
    P = (np.ones_like(V) if present is None
         else np.asarray(present, np.float32))
    return np.concatenate([np.where(P > 0, V, 0.0).astype(np.float32), P], axis=1)


# ---------------------------------------------------------------------------
# The four families
# ---------------------------------------------------------------------------


class MLPRival(eqx.Module):
    """Request one-hots -> hidden -> 2048 features."""

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def __init__(self, in_dim: int, width: int, out_dim: int, key):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(in_dim, width, key=k1)
        self.l2 = eqx.nn.Linear(width, out_dim, key=k2)
        self.in_dim, self.out_dim, self.kind = int(in_dim), int(out_dim), "mlp"

    def __call__(self, x):
        return self.l2(jax.nn.gelu(self.l1(x)))


class ContMLPRival(MLPRival):
    """The six factor values as NUMBERS -> hidden -> 2048 features.

    Its input is the min-max normalised factor value, not a class index, so an
    off-grid hue is simply a number that no training row carried. That is the
    whole reason this family is in the set: it is the only one that can enter
    row 2.
    """

    def __init__(self, in_dim: int, width: int, out_dim: int, key):
        super().__init__(in_dim, width, out_dim, key)
        object.__setattr__(self, "kind", "cont_mlp")


class _Block(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    n1: eqx.nn.LayerNorm
    n2: eqx.nn.LayerNorm
    f1: eqx.nn.Linear
    f2: eqx.nn.Linear

    def __init__(self, d: int, n_heads: int, ff: int, key):
        ka, k1, k2 = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(n_heads, d, key=ka)
        self.n1, self.n2 = eqx.nn.LayerNorm(d), eqx.nn.LayerNorm(d)
        self.f1, self.f2 = eqx.nn.Linear(d, ff, key=k1), eqx.nn.Linear(ff, d, key=k2)

    def __call__(self, h):
        a = jax.vmap(self.n1)(h)
        h = h + self.attn(a, a, a)
        b = jax.vmap(self.n2)(h)
        h = h + jax.vmap(lambda z: self.f2(jax.nn.gelu(self.f1(z))))(b)
        return h


class TransformerRival(eqx.Module):
    """6 factor tokens, 2 layers, mean-pooled -> 2048 features."""

    emb: eqx.nn.Embedding
    pos: jax.Array
    blocks: tuple
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    kind: str = eqx.field(static=True)

    def __init__(self, vocab: int, n_tokens: int, d: int, n_layers: int,
                 n_heads: int, out_dim: int, key):
        ke, kp, kh, *kb = jax.random.split(key, 3 + n_layers)
        self.emb = eqx.nn.Embedding(vocab, d, key=ke)
        self.pos = jax.random.normal(kp, (n_tokens, d)) * 0.02
        self.blocks = tuple(_Block(d, n_heads, 2 * d, k) for k in kb)
        self.norm = eqx.nn.LayerNorm(d)
        self.head = eqx.nn.Linear(d, out_dim, key=kh)
        self.kind = "transformer"

    def __call__(self, tok):
        h = jax.vmap(self.emb)(tok) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h.mean(axis=0)))


class LSTMRival(eqx.Module):
    """6 factor tokens read sequentially; the final hidden state -> 2048 features."""

    emb: eqx.nn.Embedding
    cell: eqx.nn.LSTMCell
    head: eqx.nn.Linear
    hidden: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def __init__(self, vocab: int, d: int, out_dim: int, key):
        ke, kc, kh = jax.random.split(key, 3)
        self.emb = eqx.nn.Embedding(vocab, d, key=ke)
        self.cell = eqx.nn.LSTMCell(d, d, key=kc)
        self.head = eqx.nn.Linear(d, out_dim, key=kh)
        self.hidden, self.kind = int(d), "lstm"

    def __call__(self, tok):
        e = jax.vmap(self.emb)(tok)
        h = c = jnp.zeros(self.hidden)
        for i in range(e.shape[0]):
            h, c = self.cell(e[i], (h, c))
        return self.head(h)


# ---------------------------------------------------------------------------
# Builders - one per family, each taking a single width knob
# ---------------------------------------------------------------------------


def builders(cardinalities: Sequence[int], out_dim: int, key) -> dict:
    """``{family: (build_fn, input_kind)}`` - each build_fn takes only a width."""
    cards = [int(c) for c in cardinalities]
    n_tok = len(cards)
    in_oh = int(sum(cards))
    vocab = token_vocab(cards)

    def _heads(d: int) -> int:
        for h in (4, 3, 2, 1):
            if d % h == 0:
                return h
        return 1

    return {
        "mlp": (lambda w: MLPRival(in_oh, w, out_dim, key), "onehot"),
        "cont_mlp": (lambda w: ContMLPRival(2 * n_tok, w, out_dim, key), "continuous"),
        "transformer": (
            lambda w: TransformerRival(vocab, n_tok, w, 2, _heads(w), out_dim, key),
            "tokens"),
        "lstm": (lambda w: LSTMRival(vocab, w, out_dim, key), "tokens"),
    }


def encode_inputs(kind: str, req, cardinalities: Sequence[int],
                  present: Optional[np.ndarray] = None) -> np.ndarray:
    """Pack a :class:`DenseRequests` into the input a family consumes.

    ``present`` is an ``(N, F)`` boolean: False means that factor was not
    requested. ``None`` means every factor was requested, which is what the cell
    harness always asks.
    """
    if kind == "onehot":
        return onehot_inputs(req.cls_targets, cardinalities, present)
    if kind == "tokens":
        return token_inputs(req.cls_targets, cardinalities, present)
    if kind == "continuous":
        # The normalised value, NOT the class index: an off-grid hue is a number
        # that lands between (or outside) the grid points and is passed through.
        return continuous_inputs(req.val_targets, present)
    raise ValueError(f"unknown input kind {kind!r}")


#: Probability that a training request is left fully specified. The rest are
#: partially specified, because rows 3 and 4 ask partial requests and a rival that
#: has never seen one answers them out of distribution. Declared training data
#: augmentation, not a change to the architecture or the parameter budget.
P_FULL_REQUEST = 0.5


def dropout_present(n: int, n_factors: int, rng: np.random.Generator) -> np.ndarray:
    """``(N, F)`` request masks: half fully specified, half a random subset."""
    present = np.ones((n, n_factors), bool)
    partial = rng.random(n) >= P_FULL_REQUEST
    k = rng.integers(1, n_factors, size=n)  # 1..F-1 factors retained
    for i in np.nonzero(partial)[0]:
        keep = rng.choice(n_factors, size=int(k[i]), replace=False)
        m = np.zeros(n_factors, bool)
        m[keep] = True
        present[i] = m
    return present


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, Xin, Y: np.ndarray, key, epochs: int, lr: float,
          wd: float, batch: int = BATCH, log: Optional[Callable] = None,
          rng: Optional[np.random.Generator] = None) -> tuple:
    """Adam + cosine decay on MSE to the cached feature. Returns (model, info).

    ``Xin`` may be an array, or a callable ``rng -> (N, in_dim)`` re-drawn once per
    epoch. The callable form is how request dropout is applied: a fresh mask every
    epoch covers far more of the partial-request space than one fixed draw, at the
    cost of a numpy encode per epoch.
    """
    resample = callable(Xin)
    rng = rng or np.random.default_rng(0)
    X0 = Xin(rng) if resample else Xin
    n = int(X0.shape[0])
    steps_per_epoch = max(n // batch, 1)
    total = steps_per_epoch * int(epochs)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr, warmup_steps=max(total // 20, 1),
        decay_steps=total, end_value=lr * 0.01)
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

    Yj = jnp.asarray(Y)
    Xj = jnp.asarray(X0)
    first = last = None
    for ep in range(int(epochs)):
        if resample and ep > 0:
            Xj = jnp.asarray(Xin(rng))
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, n)
        for i in range(steps_per_epoch):
            idx = perm[i * batch:(i + 1) * batch]
            model, opt_state, loss = step(model, opt_state, Xj[idx], Yj[idx])
            if first is None:
                first = float(loss)
            last = float(loss)
        if log is not None and (ep + 1) % 10 == 0:
            log(f"      epoch {ep + 1}/{epochs}: train mse {last:.5f}")
    return model, {"steps": total, "epochs": int(epochs), "lr": float(lr),
                   "weight_decay": float(wd), "batch": int(batch),
                   "loss_first": first, "loss_final": last,
                   "request_dropout": bool(resample),
                   "p_full_request": P_FULL_REQUEST if resample else 1.0}


def predict(model, Xin: np.ndarray, batch: int = 1024) -> np.ndarray:
    fwd = eqx.filter_jit(lambda m, x: jax.vmap(m)(x))
    out = [np.asarray(fwd(model, jnp.asarray(Xin[i:i + batch])), np.float32)
           for i in range(0, int(Xin.shape[0]), batch)]
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def val_mse(model, Xin: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean((predict(model, Xin) - np.asarray(Y, np.float32)) ** 2))


def fit_family(name: str, build: Callable, in_kind: str, width: int, ctx: dict,
               key, log: Callable = print) -> dict:
    """Tune on the short schedule, then retrain the selection at full length."""
    cards = ctx["cardinalities"]
    n_f = len(cards)
    n_fit = int(ctx["req_fit"].n)
    # Training inputs are re-drawn each epoch with a fresh request-dropout mask, so
    # a partially-specified request (rows 3 and 4) is in distribution rather than
    # something the rival is asked to answer having never seen one.
    def Xf(rng):
        return encode_inputs(in_kind, ctx["req_fit"], cards,
                             dropout_present(n_fit, n_f, rng))

    # Validation is on FULL requests: that is the statistic the matched comparison
    # is made on, and keeping it unaugmented is what makes the cost of the
    # augmentation visible rather than hidden inside the selection.
    Xv = encode_inputs(in_kind, ctx["req_val"], cards)
    Yf = np.asarray(ctx["X_fit"], np.float32).reshape((n_fit, -1))
    Yv = np.asarray(ctx["X_val"], np.float32).reshape((Xv.shape[0], -1))
    rng = np.random.default_rng(int(ctx["seed"]) * 104_729 + 20_260_919)

    t0 = time.time()
    # Stage 1: lr x weight decay on the SHORT schedule. The epoch axis is
    # deliberately NOT here: every trial would run for the same TUNE_EPOCHS, so
    # "epochs" would be selected by nothing but the RNG stream - which can hand
    # the rival 20 epochs when 40 is better. Selecting a rival's schedule by noise
    # is weakening a competitor through our own code.
    short = []
    for lr in LR_GRID:
        for wd in WD_GRID:
            key, tk = jax.random.split(key)
            m, _ = train(build(width), Xf, Yf, tk, TUNE_EPOCHS, lr, wd, rng=rng)
            v = val_mse(m, Xv, Yv)
            short.append({"lr": lr, "weight_decay": wd, "short_val_mse": v})
            log(f"    [{name}] short trial {len(short)}/{len(LR_GRID) * len(WD_GRID)} "
                f"lr={lr:g} wd={wd:g}: val mse {v:.5f}")
    sbest = short[int(np.argmin([t["short_val_mse"] for t in short]))]

    # Stage 2: the epoch axis, at FULL length, where it actually has an effect.
    full = []
    fitted = {}
    for ep in EPOCH_GRID:
        key, fk = jax.random.split(key)
        m, info = train(build(width), Xf, Yf, fk, int(ep), float(sbest["lr"]),
                        float(sbest["weight_decay"]), rng=rng)
        v = val_mse(m, Xv, Yv)
        full.append({"lr": sbest["lr"], "weight_decay": sbest["weight_decay"],
                     "epochs": int(ep), "full_val_mse": v})
        fitted[int(ep)] = (m, info)
        log(f"    [{name}] full trial epochs={ep}: val mse {v:.5f}")
    fbest = full[int(np.argmin([t["full_val_mse"] for t in full]))]
    trials = short + full
    best = dict(fbest)
    log(f"    [{name}] SELECTED lr={best['lr']:g} wd={best['weight_decay']:g} "
        f"epochs={best['epochs']} (validation MSE on cached features)")

    model, info = fitted[int(best["epochs"])]
    feats = {}
    for rs in ("B", "MS"):
        Xr = encode_inputs(in_kind, ctx[f"req_{rs}"], cards)
        feats[rs] = predict(model, Xr)
    # What the request-dropout augmentation bought, and what it cost: validation
    # MSE on fully specified requests (the matched comparison) beside validation
    # MSE on partially specified ones (rows 3 and 4).
    pmask = dropout_present(int(ctx["req_val"].n), n_f,
                            np.random.default_rng(int(ctx["seed"]) + 7))
    partial_mse = val_mse(model,
                          encode_inputs(in_kind, ctx["req_val"], cards, pmask), Yv)
    return {
        "family": name, "width": int(width), "input_kind": in_kind,
        "param_count": count_params(model),
        "tuning": {
            "budget": len(trials), "realised": len(trials), "entitled": 36,
            "criterion": ("validation MSE against the cached feature. Two stages: "
                          "lr x weight-decay on the short schedule, then the epoch "
                          "axis at FULL length, because an axis selected on a "
                          "schedule that does not vary it is selected by noise."),
            "short_trials": short, "full_trials": full,
            "trials": trials, "selected": best},
        "fit": info,
        "val_mse_full_schedule": val_mse(model, Xv, Yv),
        "val_mse_partial_requests": float(partial_mse),
        "partial_requests": {
            "supported": True,
            "p_full_request_in_training": P_FULL_REQUEST,
            "absent_encoding": {
                "onehot": "the factor's one-hot block is all zeros",
                "tokens": "a reserved token per factor, sum(C_k) + factor_index",
                "continuous": "a second channel per factor carrying a requested flag",
            }[in_kind],
            "note": ("rows 3 and 4 ask partially specified requests; this rival was "
                     "trained with request dropout so an absent factor is in "
                     "distribution rather than answered blind"),
        },
        "wall_clock_s": float(time.time() - t0),
        "_feats": feats, "_model": model, "_in_kind": in_kind,
    }


# ---------------------------------------------------------------------------
# Row 2: the off-grid hue
# ---------------------------------------------------------------------------


def offgrid_values(grid, factor: str, n_between: int = 3) -> dict:
    """Normalised values BETWEEN this factor's grid points, plus the grid itself.

    Row 2 asks for a value the grid does not contain. Midpoints between adjacent
    grid points are the honest version of that: they are inside the factor's
    realised range, so nothing is being extrapolated off the end of the world,
    but no training row carries them.
    """
    k = list(grid.names).index(factor)
    vals = np.asarray(grid.values[k], np.float64)
    lo, hi = float(vals[0]), float(vals[-1])
    span = max(hi - lo, 1e-12)
    ongrid = [(float(v) - lo) / span for v in vals]
    off = []
    for i in range(len(vals) - 1):
        for j in range(1, int(n_between) + 1):
            w = j / (int(n_between) + 1.0)
            off.append((float(vals[i]) * (1 - w) + float(vals[i + 1]) * w - lo) / span)
    return {"factor": factor, "factor_index": int(k),
            "grid_values": [float(v) for v in vals],
            "normalised_ongrid": ongrid, "normalised_offgrid": off}


def offgrid_demo(model, in_kind: str, ctx: dict, factor: str = "object_hue",
                 n_between: int = 3, log: Optional[Callable] = None) -> dict:
    """Evaluate a rival at hues the grid does not contain - row 2, demonstrated.

    Only families whose input is the VALUE can do this. A family that consumes a
    class index or a one-hot has a closed vocabulary and cannot express an
    off-grid hue at all, so it gets a dash and this footnote rather than a number.
    A dash is never a win for us.
    """
    if in_kind != "continuous":
        return {
            "NOT_APPLICABLE": True,
            "reason": (f"this family's input is {in_kind!r}, a closed vocabulary over "
                       "the realised grid values. It cannot express an off-grid hue, "
                       "so row 2 gets a dash and this footnote, never a number."),
        }
    spec = offgrid_values(ctx["grid"], factor, n_between)
    base = np.asarray(ctx["req_B"].val_targets, np.float32)
    k = spec["factor_index"]

    def _at(v: float) -> np.ndarray:
        vals = base.copy()
        vals[:, k] = np.float32(v)
        return predict(model, continuous_inputs(vals))

    out = []
    for v in spec["normalised_offgrid"]:
        F = _at(v)
        out.append({"normalised_value": float(v),
                    "feature_mean": float(F.mean()),
                    "feature_norm_mean": float(np.linalg.norm(F, axis=1).mean())})
    on = []
    for v in spec["normalised_ongrid"]:
        F = _at(v)
        on.append({"normalised_value": float(v),
                   "feature_norm_mean": float(np.linalg.norm(F, axis=1).mean())})
    norms_off = [r["feature_norm_mean"] for r in out]
    norms_on = [r["feature_norm_mean"] for r in on]
    rec = {
        "ACCEPTED": True,
        "factor": factor,
        "n_offgrid_values": len(out),
        "n_ongrid_values": len(on),
        "offgrid": out, "ongrid": on,
        "offgrid_norm_mean": float(np.mean(norms_off)),
        "ongrid_norm_mean": float(np.mean(norms_on)),
        "offgrid_within_ongrid_range": bool(
            min(norms_on) * 0.8 <= np.mean(norms_off) <= max(norms_on) * 1.2),
        "note": ("the rival accepts a hue no training row carried and returns a "
                 "feature of comparable norm; this establishes only that it CAN "
                 "enter row 2, not that it is right there - the row itself is a "
                 "hub's measurement, not this spoke's claim"),
    }
    if log is not None:
        log(f"    [row 2] off-grid {factor}: {len(out)} values accepted; mean feature "
            f"norm {rec['offgrid_norm_mean']:.3f} against {rec['ongrid_norm_mean']:.3f} "
            f"on-grid")
    return rec
