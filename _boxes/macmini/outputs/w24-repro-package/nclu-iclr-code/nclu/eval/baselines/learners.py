"""learners."""

from __future__ import annotations
from nclu.paths import run_dir
import time
from typing import Callable, Optional, Sequence
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

LR_GRID = (0.0003, 0.001, 0.003)
WD_GRID = (0.0, 0.0001)
EPOCH_GRID = (20, 40)
TUNE_EPOCHS = 8
BATCH = 256
PARAM_TOLERANCE = 0.1
FAMILY_BUDGETS = {"mlp_180k": 179968}


def count_params(model) -> int:
    """TOTAL trainable parameters: every array leaf, including biases and norms."""
    return int(
        sum(
            (np.asarray(x).size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
        )
    )


def match_width(
    build: Callable[[int], object],
    target: int,
    lo: int = 1,
    hi: int = 512,
    tol: float = PARAM_TOLERANCE,
    log: Optional[Callable] = None,
) -> tuple:
    seen: dict[int, int] = {}

    def n_at(w: int) -> int:
        if w not in seen:
            seen[w] = count_params(build(w))
        return seen[w]

    a, b = (int(lo), int(hi))
    while a < b:
        mid = (a + b) // 2
        if n_at(mid) < target:
            a = mid + 1
        else:
            b = mid
    for w in (a - 1, a, a + 1):
        if int(lo) <= w <= int(hi):
            n_at(w)
    best_w = min(seen, key=lambda w: abs(seen[w] - target))
    best_n = seen[best_w]
    best_err = abs(best_n - target) / float(target)
    below = max(((w, n) for w, n in seen.items() if n <= target), key=lambda t: t[1], default=None)
    above = min(((w, n) for w, n in seen.items() if n >= target), key=lambda t: t[1], default=None)
    if best_err > tol:
        raise AssertionError(
            f"cannot match {target} parameters within +/-{tol:.0%} at any width in [{lo}, {hi}]: closest is width {best_w} at {best_n} params ({best_err:+.2%}); nearest below={below}, nearest above={above}. Report the achievable count and stop; do not ship a mismatched rival."
        )
    if log is not None:
        log(
            f"    matched width={best_w}: {best_n} params vs target {target} ({(best_n - target) / target:+.2%})"
        )
    return (int(best_w), int(best_n), float((best_n - target) / target))


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
    return int(sum((int(c) for c in cardinalities))) + len(list(cardinalities))


def onehot_inputs(
    cls_targets: np.ndarray, cardinalities: Sequence[int], present: Optional[np.ndarray] = None
) -> np.ndarray:
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


def token_inputs(
    cls_targets: np.ndarray, cardinalities: Sequence[int], present: Optional[np.ndarray] = None
) -> np.ndarray:
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


def continuous_inputs(val_targets: np.ndarray, present: Optional[np.ndarray] = None) -> np.ndarray:
    """``(N, F)`` normalised values -> ``(N, 2F)``: the value, and a requested flag.

    The value channel carries the min-max normalised number, so an off-grid hue
    is simply a number between (or outside) the grid points. The flag channel is
    what lets an absent factor be absent rather than be zero, which is a real
    value on this scale.
    """
    V = np.asarray(val_targets, np.float32)
    P = np.ones_like(V) if present is None else np.asarray(present, np.float32)
    return np.concatenate([np.where(P > 0, V, 0.0).astype(np.float32), P], axis=1)


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
        self.in_dim, self.out_dim, self.kind = (int(in_dim), int(out_dim), "mlp")

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
        self.n1, self.n2 = (eqx.nn.LayerNorm(d), eqx.nn.LayerNorm(d))
        self.f1, self.f2 = (eqx.nn.Linear(d, ff, key=k1), eqx.nn.Linear(ff, d, key=k2))

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

    def __init__(
        self, vocab: int, n_tokens: int, d: int, n_layers: int, n_heads: int, out_dim: int, key
    ):
        ke, kp, kh, *kb = jax.random.split(key, 3 + n_layers)
        self.emb = eqx.nn.Embedding(vocab, d, key=ke)
        self.pos = jax.random.normal(kp, (n_tokens, d)) * 0.02
        self.blocks = tuple((_Block(d, n_heads, 2 * d, k) for k in kb))
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
        self.hidden, self.kind = (int(d), "lstm")

    def __call__(self, tok):
        e = jax.vmap(self.emb)(tok)
        h = c = jnp.zeros(self.hidden)
        for i in range(e.shape[0]):
            h, c = self.cell(e[i], (h, c))
        return self.head(h)


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
        "mlp_180k": (lambda w: MLPRival(in_oh, w, out_dim, key), "onehot"),
        "cont_mlp": (lambda w: ContMLPRival(2 * n_tok, w, out_dim, key), "continuous"),
        "transformer": (
            lambda w: TransformerRival(vocab, n_tok, w, 2, _heads(w), out_dim, key),
            "tokens",
        ),
        "lstm": (lambda w: LSTMRival(vocab, w, out_dim, key), "tokens"),
    }


def encode_inputs(
    kind: str, req, cardinalities: Sequence[int], present: Optional[np.ndarray] = None
) -> np.ndarray:
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
        return continuous_inputs(req.val_targets, present)
    raise ValueError(f"unknown input kind {kind!r}")


P_FULL_REQUEST = 0.5


def dropout_present(n: int, n_factors: int, rng: np.random.Generator) -> np.ndarray:
    """``(N, F)`` request masks: half fully specified, half a random subset."""
    present = np.ones((n, n_factors), bool)
    partial = rng.random(n) >= P_FULL_REQUEST
    k = rng.integers(1, n_factors, size=n)
    for i in np.nonzero(partial)[0]:
        keep = rng.choice(n_factors, size=int(k[i]), replace=False)
        m = np.zeros(n_factors, bool)
        m[keep] = True
        present[i] = m
    return present


def _as_model_input(X) -> np.ndarray:
    """float32 for continuous inputs, dtype PRESERVED for integer token inputs."""
    A = np.asarray(X)
    return A if np.issubdtype(A.dtype, np.integer) else A.astype(np.float32)


def train(
    model,
    Xin,
    Y: np.ndarray,
    key,
    epochs: int,
    lr: float,
    wd: float,
    batch: int = BATCH,
    log: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
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
    Xnp = _as_model_input(X0)
    first = last = None
    for ep in range(int(epochs)):
        if resample and ep > 0:
            Xnp = _as_model_input(Xin(rng))
        key, sk = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sk, n))
        for i in range(steps_per_epoch):
            idx = perm[i * batch : (i + 1) * batch]
            model, opt_state, loss = step(
                model, opt_state, jnp.asarray(Xnp[idx]), jnp.asarray(Ynp[idx])
            )
            if first is None:
                first = float(loss)
            last = float(loss)
        if log is not None and (ep + 1) % 10 == 0:
            log(f"      epoch {ep + 1}/{epochs}: train mse {last:.5f}")
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
            "request_dropout": bool(resample),
            "p_full_request": P_FULL_REQUEST if resample else 1.0,
        },
    )


def predict(model, Xin: np.ndarray, batch: int = 1024) -> np.ndarray:
    fwd = eqx.filter_jit(lambda m, x: jax.vmap(m)(x))
    out = [
        np.asarray(fwd(model, jnp.asarray(Xin[i : i + batch])), np.float32)
        for i in range(0, int(Xin.shape[0]), batch)
    ]
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def val_mse(model, Xin: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean((predict(model, Xin) - np.asarray(Y, np.float32)) ** 2))


def fit_family(
    name: str,
    build: Callable,
    in_kind: str,
    width: int,
    ctx: dict,
    key,
    dropout: bool = False,
    log: Callable = print,
) -> dict:
    cards = ctx["cardinalities"]
    n_f = len(cards)
    n_fit = int(ctx["req_fit"].n)
    if dropout:

        def Xf(rng):
            return encode_inputs(in_kind, ctx["req_fit"], cards, dropout_present(n_fit, n_f, rng))
    else:
        Xf = encode_inputs(in_kind, ctx["req_fit"], cards)
    Xv = encode_inputs(in_kind, ctx["req_val"], cards)
    Yf = np.asarray(ctx["X_fit"], np.float32).reshape((n_fit, -1))
    Yv = np.asarray(ctx["X_val"], np.float32).reshape((Xv.shape[0], -1))
    rng = np.random.default_rng(int(ctx["seed"]) * 104729 + 20260919)
    t0 = time.time()
    short = []
    for lr in LR_GRID:
        for wd in WD_GRID:
            key, tk = jax.random.split(key)
            m, _ = train(build(width), Xf, Yf, tk, TUNE_EPOCHS, lr, wd, rng=rng)
            v = val_mse(m, Xv, Yv)
            short.append({"lr": lr, "weight_decay": wd, "short_val_mse": v})
            log(
                f"    [{name}] short trial {len(short)}/{len(LR_GRID) * len(WD_GRID)} lr={lr:g} wd={wd:g}: val mse {v:.5f}"
            )
    sbest = short[int(np.argmin([t["short_val_mse"] for t in short]))]
    full = []
    fitted = {}
    for ep in EPOCH_GRID:
        key, fk = jax.random.split(key)
        m, info = train(
            build(width),
            Xf,
            Yf,
            fk,
            int(ep),
            float(sbest["lr"]),
            float(sbest["weight_decay"]),
            rng=rng,
        )
        v = val_mse(m, Xv, Yv)
        full.append(
            {
                "lr": sbest["lr"],
                "weight_decay": sbest["weight_decay"],
                "epochs": int(ep),
                "full_val_mse": v,
            }
        )
        fitted[int(ep)] = (m, info)
        log(f"    [{name}] full trial epochs={ep}: val mse {v:.5f}")
    fbest = full[int(np.argmin([t["full_val_mse"] for t in full]))]
    trials = short + full
    best = dict(fbest)
    log(
        f"    [{name}] SELECTED lr={best['lr']:g} wd={best['weight_decay']:g} epochs={best['epochs']} (validation MSE on cached features)"
    )
    model, info = fitted[int(best["epochs"])]
    feats = {}
    for rs in ("B", "MS"):
        req = ctx.get(f"req_{rs}")
        if req is None:
            continue
        feats[rs] = predict(model, encode_inputs(in_kind, req, cards))
    pmask = dropout_present(int(ctx["req_val"].n), n_f, np.random.default_rng(int(ctx["seed"]) + 7))
    partial_mse = val_mse(model, encode_inputs(in_kind, ctx["req_val"], cards, pmask), Yv)
    return {
        "family": name,
        "width": int(width),
        "input_kind": in_kind,
        "variant": "dropout" if dropout else "full_only",
        "trained_on": "half fully specified requests, half a random subset of 1-5 factors, re-drawn every epoch"
        if dropout
        else "fully specified six-factor requests only",
        "use_for": [
            "rows 1, 2, 5, 6 - the query sets, where an unrequested factor is genuinely nan"
        ]
        if dropout
        else [
            "rows 3 and 4 - the pair and n-way corners, whose runner builds FULL six-factor requests"
        ],
        "param_count": count_params(model),
        "tuning": {
            "budget": len(trials),
            "realised": len(trials),
            "entitled": 36,
            "criterion": "validation MSE against the cached feature. Two stages: lr x weight-decay on the short schedule, then the epoch axis at FULL length, because an axis selected on a schedule that does not vary it is selected by noise.",
            "short_trials": short,
            "full_trials": full,
            "trials": trials,
            "selected": best,
        },
        "fit": info,
        "val_mse_full_schedule": val_mse(model, Xv, Yv),
        "val_mse_partial_requests": float(partial_mse),
        "partial_requests": {
            "supported": bool(dropout),
            "p_full_request_in_training": P_FULL_REQUEST if dropout else 1.0,
            "caveat": None
            if dropout
            else "this variant never saw a partially specified request in training; val_mse_partial_requests below is what it does with one anyway, and is NOT a number to quote as this rival's partial-request performance - use the _dropout variant for that",
            "absent_encoding": {
                "onehot": "the factor's one-hot block is all zeros",
                "tokens": "a reserved token per factor, sum(C_k) + factor_index",
                "continuous": "a second channel per factor carrying a requested flag",
            }[in_kind],
            "note": "rows 3 and 4 ask partially specified requests; this rival was trained with request dropout so an absent factor is in distribution rather than answered blind",
        },
        "wall_clock_s": float(time.time() - t0),
        "_feats": feats,
        "_model": model,
        "_in_kind": in_kind,
    }


def offgrid_values(grid, factor: str, n_between: int = 3) -> dict:
    """Normalised values BETWEEN this factor's grid points, plus the grid itself.

    Row 2 asks for a value the grid does not contain. Midpoints between adjacent
    grid points are the honest version of that: they are inside the factor's
    realised range, so nothing is being extrapolated off the end of the world,
    but no training row carries them.
    """
    k = list(grid.names).index(factor)
    vals = np.asarray(grid.values[k], np.float64)
    lo, hi = (float(vals[0]), float(vals[-1]))
    span = max(hi - lo, 1e-12)
    ongrid = [(float(v) - lo) / span for v in vals]
    off = []
    for i in range(len(vals) - 1):
        for j in range(1, int(n_between) + 1):
            w = j / (int(n_between) + 1.0)
            off.append((float(vals[i]) * (1 - w) + float(vals[i + 1]) * w - lo) / span)
    return {
        "factor": factor,
        "factor_index": int(k),
        "grid_values": [float(v) for v in vals],
        "normalised_ongrid": ongrid,
        "normalised_offgrid": off,
    }


def offgrid_demo(
    model,
    in_kind: str,
    ctx: dict,
    factor: str = "object_hue",
    n_between: int = 3,
    log: Optional[Callable] = None,
) -> dict:
    """Evaluate a rival at hues the grid does not contain - row 2, demonstrated.

    Only families whose input is the VALUE can do this. A family that consumes a
    class index or a one-hot has a closed vocabulary and cannot express an
    off-grid hue at all, so it gets a dash and this footnote rather than a number.
    A dash is never a win for us.
    """
    if in_kind != "continuous":
        return {
            "NOT_APPLICABLE": True,
            "reason": f"this family's input is {in_kind!r}, a closed vocabulary over the realised grid values. It cannot express an off-grid hue, so row 2 gets a dash and this footnote, never a number.",
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
        out.append(
            {
                "normalised_value": float(v),
                "feature_mean": float(F.mean()),
                "feature_norm_mean": float(np.linalg.norm(F, axis=1).mean()),
            }
        )
    on = []
    for v in spec["normalised_ongrid"]:
        F = _at(v)
        on.append(
            {
                "normalised_value": float(v),
                "feature_norm_mean": float(np.linalg.norm(F, axis=1).mean()),
            }
        )
    norms_off = [r["feature_norm_mean"] for r in out]
    norms_on = [r["feature_norm_mean"] for r in on]
    rec = {
        "ACCEPTED": True,
        "factor": factor,
        "n_offgrid_values": len(out),
        "n_ongrid_values": len(on),
        "offgrid": out,
        "ongrid": on,
        "offgrid_norm_mean": float(np.mean(norms_off)),
        "ongrid_norm_mean": float(np.mean(norms_on)),
        "offgrid_within_ongrid_range": bool(
            min(norms_on) * 0.8 <= np.mean(norms_off) <= max(norms_on) * 1.2
        ),
        "note": "the rival accepts a hue no training row carried and returns a feature of comparable norm; this establishes only that it CAN enter row 2, not that it is right there - the row itself is a study's measurement, not this study's claim",
    }
    if log is not None:
        log(
            f"    [row 2] off-grid {factor}: {len(out)} values accepted; mean feature norm {rec['offgrid_norm_mean']:.3f} against {rec['ongrid_norm_mean']:.3f} on-grid"
        )
    return rec


class ContMLPProtocolMatched(eqx.Module):
    """Three Linear layers, GELU between, none on the output. Depth 2."""

    layers: list
    kind: str = eqx.field(static=True)

    def __init__(self, key, n_in: int, hidden: int, n_out: int, depth: int = 2):
        ks = jax.random.split(key, depth + 1)
        dims = [int(n_in)] + [int(hidden)] * int(depth) + [int(n_out)]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(int(depth) + 1)
        ]
        self.kind = "cont_mlp_notune"

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        return self.layers[-1](x)


def pick_hidden(target_params: int, n_in: int, n_out: int, depth: int = 2) -> int:
    """Their width rule, reimplemented from the formula rather than imported."""
    best, bestd = (8, None)
    for h in range(8, 2048):
        p = n_in * h + h + (depth - 1) * (h * h + h) + h * n_out + n_out
        d = abs(p - int(target_params))
        if bestd is None or d < bestd:
            best, bestd = (h, d)
    return int(best)


def normalise_values(values: np.ndarray, grid) -> np.ndarray:
    """Raw factor values -> [0, 1] by each factor's own realised grid range."""
    V = np.asarray(values, np.float64).copy()
    for k in range(V.shape[1]):
        g = np.asarray(grid.values[k], np.float64)
        V[:, k] = (V[:, k] - g[0]) / max(g[-1] - g[0], 1e-12)
    return V.astype(np.float32)


def fit_protocol_matched(
    grid,
    rows: np.ndarray,
    Y: np.ndarray,
    target_params: int,
    seed: int = 0,
    steps: int = 4000,
    batch: int = 512,
    lr: float = 0.001,
    log: Callable = print,
) -> tuple:
    X = normalise_values(grid.values_of(np.asarray(rows, np.int64)), grid)
    Yn = np.asarray(Y, np.float32)
    h = pick_hidden(target_params, X.shape[1], Yn.shape[1])
    model = ContMLPProtocolMatched(jax.random.PRNGKey(int(seed)), X.shape[1], h, Yn.shape[1])
    npar = count_params(model)
    log(
        f"    cont_mlp_notune: hidden={h} params={npar:,} target={target_params:,} ({100.0 * npar / target_params:.1f}% of matched)"
    )
    opt = optax.adam(float(lr))
    state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, st, xb, yb):
        def loss(mm):
            return jnp.mean((jax.vmap(mm)(xb) - yb) ** 2)

        lv, g = eqx.filter_value_and_grad(loss)(m)
        up, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return (eqx.apply_updates(m, up), st, lv)

    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    lv = None
    for i in range(int(steps)):
        idx = rng.integers(0, X.shape[0], int(batch))
        model, state, lv = step(model, state, jnp.asarray(X[idx]), jnp.asarray(Yn[idx]))
        if i % 1000 == 0 or i == int(steps) - 1:
            log(f"      step {i + 1}/{steps} mse={float(lv):.5f} ({time.time() - t0:.0f}s)")
    return (
        model,
        {
            "protocol": "matched to an earlier study's in-harness cont_mlp",
            "protocol_source": str(run_dir("shapes3d/unseen_valid/code/cont_mlp.py")),
            "independent_reimplementation": True,
            "hidden": int(h),
            "n_params": int(npar),
            "target_params": int(target_params),
            "pct_of_matched": float(100.0 * npar / target_params),
            "depth": 2,
            "activation": "gelu",
            "steps": int(steps),
            "batch": int(batch),
            "lr": float(lr),
            "weight_decay": 0.0,
            "schedule": None,
            "sampling": "with replacement, np.random.default_rng(seed).integers",
            "n_rows_fit": int(X.shape[0]),
            "selection_set": None,
            "final_minibatch_mse": float(lv) if lv is not None else None,
            "wall_clock_s": float(time.time() - t0),
            "why": "ships ALONGSIDE the tuned cont_mlp, never instead of it: it exists only to make the implementation-independence row differ in implementation alone. An untuned rival is a weakened rival.",
        },
    )
