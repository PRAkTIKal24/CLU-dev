"""Class-IL baselines for the w25 continual-learning entry (``exp_cl_entry``).

The mandatory comparison set of `continual-learning-recon` §1.3, reimplemented in
JAX/Equinox because the field's harnesses (Mammoth, Avalanche) are PyTorch and this
repo is not. Two disciplines follow from that (N78 — *a rescued baseline must be
tuned, or the win is fake*):

1. **Every baseline shares the backbone, the optimizer, the iteration budget and the
   evaluation code with every other baseline** — including the CLU entry's memory
   budget. Nothing is tuned per method beyond the one hyper-parameter each method is
   defined by (``ewc_lambda``, ``si_c``, ``lwf_alpha``), which is swept over a small
   grid on seed 0 when ``tune_grid`` is set, and the best value is then used for all
   seeds. The grid and the winner are reported.
2. **The known nulls are labelled as such.** EWC/SI/LwF collapse to ≈chance in
   Class-IL *by construction* (van de Ven & Tolias 2019); their collapse is NEVER
   presented as a CLU win — it is the published behaviour of the method class the
   entry belongs to, and it is reproduced here to show the harness is calibrated.

Methods: ``finetune`` (the null), ``ewc``, ``si``, ``lwf`` (rehearsal-free);
``er``, ``derpp``, ``icarl``, ``gdumb`` (replay/exemplar, at matched memory);
``joint`` (offline upper bound). The kNN-in-φ laundering control lives in
``exp_cl_entry`` because it shares the entry's ``φ``.

Class-IL protocol: task identity is **not** given at test time; logits are masked to
the classes **seen so far** and the arg-max is taken over that set (the standard
Class-IL read-out, and the one that makes EWC/SI ≈ 20 % on Split-MNIST).
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

#: methods that never store a raw exemplar (the class the entry competes in)
REHEARSAL_FREE = ("finetune", "ewc", "si", "lwf")
#: methods that keep raw exemplars (reported, never claimed to be beaten)
REPLAY = ("er", "derpp", "icarl", "gdumb")
ALL_METHODS = REHEARSAL_FREE + REPLAY + ("joint",)
#: methods with NO episodic memory ⇒ horizontal lines on a memory-budget frontier
BUDGET_INDEPENDENT = REHEARSAL_FREE + ("joint",)


# ---------------------------------------------------------------------------
# ⛔ THE BYTE ACCOUNTING (w26 `matched-bytes-frontier`, PREREG §1)
# ---------------------------------------------------------------------------


def floats_per_stored_item(cfg, dim: int, n_classes: int) -> dict:
    """Floats kept **per stored item** by each method — the frontier's x-axis.

    Pinned before any run and published in the report; a referee checks this line
    first. Unit = float32; every scalar an implementation keeps per item is counted
    at one float even when it is an ``int``/``bool`` — deliberately **conservative
    against CLU**, which keeps more per-item scalar bookkeeping than a raw buffer.

    * CLU store: ``AtomStorePotential.centers[i]`` (``phi_dim``) + ``payloads[i]``
      + ``amps[i]`` + ``active[i]``, plus the controller's ``ItemRecord``
      non-redundant scalars (``leak``, ``permanent``, ``born``, ``last_used``,
      ``item_id``, ``slot``) when ``count_controller_record_floats`` is set. The
      record's ``center``/``payload``/``base_amp`` are duplicates of store arrays
      and are NOT double-counted.
    * kNN-in-φ ring buffer: key + label + age (the balanced-LRU drop rule needs the
      age). ⇒ it gets **more** keys than the store gets wells at the same bytes,
      which is the point: the launder must not be under-resourced.
    * ER/GDumb/iCaRL: raw exemplar + label. DER++ additionally stores the logit
      vector it distils against (``n_classes`` floats) — counted.
    * EWC/SI/LwF/finetune/joint keep **no** episodic memory (their consolidation
      state is *fixed*, not per-item ⇒ see :func:`fixed_state_floats`).
    """
    rec = 6 if getattr(cfg, "count_controller_record_floats", True) else 0
    clu = int(cfg.phi_dim) + 3 + rec
    return {
        "clu_entry": clu,
        "knn_phi_same_keys": clu,
        "knn_phi_ringbuffer": int(cfg.phi_dim) + 2,
        "er": dim + 1,
        "derpp": dim + 1 + int(n_classes),
        "gdumb": dim + 1,
        "icarl": dim + 1,
        "finetune": 0,
        "ewc": 0,
        "si": 0,
        "lwf": 0,
        "joint": 0,
    }


def fixed_state_floats(cfg, dim: int, n_classes: int) -> dict:
    """Per-method **fixed** state (not per item) — reported, and used for the
    secondary "all-fixed-state-charged" frontier.

    The CLU entry runs zero gradient steps: its only fixed state is the frozen
    PCA-``phi_dim`` read-in (``mean`` + ``components``). Every gradient baseline
    carries the shared backbone, and EWC/SI/LwF carry a further 1–2 copies of it
    (Fisher/ω + reference θ, or the frozen previous model). Adam moments are
    transient optimizer state and are excluded for everyone.
    """
    if cfg.backbone == "mlp":
        w, d = int(cfg.mlp_width), int(cfg.mlp_depth)
        sizes = [dim] + [w] * d
        backbone = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(d))
        backbone += w * int(n_classes) + int(n_classes)
    else:  # small CNN — counted from the same shapes ConvNet builds
        ch = list(cfg.cnn_channels)
        c_in = [3] + ch[:-1]
        backbone = sum(ci * co * 9 + co for ci, co in zip(c_in, ch, strict=True))
        flat = ch[-1] * (32 // 8) * (32 // 8)
        backbone += flat * int(cfg.mlp_width) + int(cfg.mlp_width)
        backbone += int(cfg.mlp_width) * int(n_classes) + int(n_classes)
    phi = int(cfg.phi_dim) * dim + dim  # PCA components + mean
    out = {m: backbone for m in ALL_METHODS}
    out["ewc"] = backbone * 3  # + Fisher diag + reference θ
    out["si"] = backbone * 3  # + ω + reference θ
    out["lwf"] = backbone * 2  # + the frozen previous model
    out["clu_entry"] = phi
    out["knn_phi_same_keys"] = phi
    out["knn_phi_ringbuffer"] = phi
    return out


def items_for_budget(cfg, budget_floats: int, dim: int, n_classes: int) -> dict:
    """How many items each method may keep at a **matched byte budget**."""
    per = floats_per_stored_item(cfg, dim, n_classes)
    return {
        m: (int(budget_floats // f) if f > 0 else 0) for m, f in per.items()
    }


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------


class MLPNet(eqx.Module):
    """van de Ven's Split-MNIST backbone: 2 hidden ReLU layers, one shared head."""

    layers: list
    head: eqx.nn.Linear

    def __init__(self, dim, n_classes, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        sizes = [dim] + [width] * depth
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]) for i in range(depth)
        ]
        self.head = eqx.nn.Linear(width, n_classes, key=keys[-1])

    def features(self, x):
        h = x
        for lin in self.layers:
            h = jax.nn.relu(lin(h))
        return h

    def __call__(self, x):
        return self.head(self.features(x))


class ConvNet(eqx.Module):
    """Small from-scratch CNN for Split-CIFAR-10 (the weight class of the entry;
    ResNet-18 is out of budget on this machine and would not change the ordering)."""

    conv: list
    lin: eqx.nn.Linear
    head: eqx.nn.Linear
    shape: tuple = eqx.field(static=True)

    def __init__(self, shape, n_classes, width, key, channels=(32, 64, 64)):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        c = shape[0]
        self.shape = shape
        ch = list(channels)
        self.conv = [
            eqx.nn.Conv2d(c, ch[0], 3, padding=1, key=k1),
            eqx.nn.Conv2d(ch[0], ch[1], 3, padding=1, key=k2),
            eqx.nn.Conv2d(ch[1], ch[2], 3, padding=1, key=k3),
        ]
        flat = ch[2] * (shape[1] // 8) * (shape[2] // 8)
        self.lin = eqx.nn.Linear(flat, width, key=k4)
        self.head = eqx.nn.Linear(width, n_classes, key=k5)

    def features(self, x):
        h = x.reshape(self.shape)
        for conv in self.conv:
            h = jax.nn.relu(conv(h))
            h = eqx.nn.MaxPool2d(2, 2)(h)
        return jax.nn.relu(self.lin(h.reshape(-1)))

    def __call__(self, x):
        return self.head(self.features(x))


def make_net(cfg, dim, n_classes, key):
    if cfg.backbone == "mlp":
        return MLPNet(dim, n_classes, cfg.mlp_width, cfg.mlp_depth, key)
    if cfg.backbone == "cnn":
        return ConvNet((3, 32, 32), n_classes, cfg.mlp_width, key,
                       channels=tuple(cfg.cnn_channels))
    raise ValueError(f"unknown backbone {cfg.backbone!r}")


# ---------------------------------------------------------------------------
# Small helpers (params / penalties / batched forward)
# ---------------------------------------------------------------------------


def _params(model):
    return eqx.filter(model, eqx.is_inexact_array)


def _zeros_like_params(model):
    return jax.tree_util.tree_map(jnp.zeros_like, _params(model))


def _quad_penalty(model, ref, imp):
    """``Σ_i imp_i (θ_i − ref_i)²`` over the model's array leaves."""
    diffs = jax.tree_util.tree_map(
        lambda p, r, i: jnp.sum(i * (p - r) ** 2), _params(model), ref, imp
    )
    leaves = jax.tree_util.tree_leaves(diffs)
    return sum(leaves) if leaves else jnp.asarray(0.0)


def batched_logits(model, X, chunk=1024):
    f = eqx.filter_jit(jax.vmap(model))
    out = [np.asarray(f(jnp.asarray(X[i : i + chunk]))) for i in range(0, len(X), chunk)]
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


def batched_features(model, X, chunk=1024):
    f = eqx.filter_jit(jax.vmap(lambda x: model.features(x)))
    out = [np.asarray(f(jnp.asarray(X[i : i + chunk]))) for i in range(0, len(X), chunk)]
    return np.concatenate(out, axis=0)


def _ce(logits, y, mask):
    """Cross-entropy restricted to the ``mask``ed (seen) classes."""
    logits = jnp.where(mask[None, :], logits, -1e9)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def class_il_predict(model, X, seen_mask, chunk=1024):
    """Class-IL read-out: arg-max over the classes seen so far (no task id)."""
    lg = batched_logits(model, X, chunk)
    lg = np.where(np.asarray(seen_mask)[None, :], lg, -np.inf)
    return lg.argmax(axis=1)


# ---------------------------------------------------------------------------
# The generic task trainer (one closure per method family)
# ---------------------------------------------------------------------------


def _train_task(
    model, X, y, seen_mask, cfg, key, extra_loss=None, buffer=None, aux=None,
    derpp=None,
):
    """Train ``model`` on one task's data.

    ``extra_loss(model, aux, xb) -> scalar`` adds the method's regularizer — it is
    handed the **current minibatch** ``xb`` so a distillation term can be evaluated
    where the published method evaluates it (LwF; see ``run_baseline_stream``).
    ``buffer`` (X, y) adds a replayed batch per step (ER/iCaRL).
    ``derpp`` = ``{"X","y","Z","alpha","beta"}`` adds DER++'s two independently
    drawn buffer terms: ``alpha·‖f(x)−z‖²`` (logit distillation against the logits
    stored at insertion) + ``beta·CE(f(x), y)``.
    """
    opt = optax.adam(cfg.baseline_lr)
    opt_state = opt.init(_params(model))
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    mask = jnp.asarray(seen_mask)
    n = X.shape[0]
    bs = min(cfg.baseline_batch, n)
    has_buf = buffer is not None and len(buffer[0]) > 0
    if has_buf:
        Xb_all, yb_all = jnp.asarray(buffer[0]), jnp.asarray(buffer[1])
        nb = Xb_all.shape[0]
        bsb = min(cfg.baseline_batch, nb)
    has_der = derpp is not None and len(derpp["X"]) > 0
    if has_der:
        Xd = jnp.asarray(derpp["X"])
        yd = jnp.asarray(derpp["y"])
        Zd = jnp.asarray(derpp["Z"])
        nd = Xd.shape[0]
        bsd = min(cfg.baseline_batch, nd)
        alpha_d = float(derpp["alpha"])
        beta_d = float(derpp["beta"])

    @eqx.filter_value_and_grad
    def loss_fn(m, xb, yb, xr, yr, xa, za, xc, yc):
        lg = jax.vmap(m)(xb)
        loss = _ce(lg, yb, mask)
        if has_buf:
            loss = loss + _ce(jax.vmap(m)(xr), yr, mask)
        if has_der:
            loss = loss + alpha_d * jnp.mean((jax.vmap(m)(xa) - za) ** 2)
            loss = loss + beta_d * _ce(jax.vmap(m)(xc), yc, mask)
        if extra_loss is not None:
            loss = loss + extra_loss(m, aux, xb)
        return loss

    @eqx.filter_jit
    def step(m, opt_state, xb, yb, xr, yr, xa, za, xc, yc):
        loss, grads = loss_fn(m, xb, yb, xr, yr, xa, za, xc, yc)
        updates, opt_state = opt.update(grads, opt_state, _params(m))
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss, grads

    dummy = X[:1], y[:1]
    dummy_z = jnp.zeros((1, 1))
    grads_last = None
    for _ in range(cfg.baseline_iters):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        idx = jax.random.choice(k1, n, (bs,), replace=n < bs)
        if has_buf:
            jdx = jax.random.choice(k2, nb, (bsb,), replace=nb < bsb)
            xr, yr = Xb_all[jdx], yb_all[jdx]
        else:
            xr, yr = dummy
        if has_der:
            # DER++ draws TWO independent buffer batches (Buzzega et al. 2020 §3)
            ad = jax.random.choice(k3, nd, (bsd,), replace=nd < bsd)
            cd = jax.random.choice(k4, nd, (bsd,), replace=nd < bsd)
            xa, za, xc, yc = Xd[ad], Zd[ad], Xd[cd], yd[cd]
        else:
            xa, za, xc, yc = dummy[0], dummy_z, dummy[0], dummy[1]
        model, opt_state, loss, grads_last = step(
            model, opt_state, X[idx], y[idx], xr, yr, xa, za, xc, yc
        )
    return model, key, float(loss), grads_last


# ---------------------------------------------------------------------------
# Method-specific state updates
# ---------------------------------------------------------------------------


def _fisher_diag(model, X, y, mask, cfg, key, n_samples=256):
    """Diagonal empirical Fisher: mean squared gradient of the (masked) log-lik."""
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n = min(n_samples, X.shape[0])
    idx = jax.random.choice(key, X.shape[0], (n,), replace=False)

    @eqx.filter_grad
    def g_of(m, x, yy):
        lg = jnp.where(jnp.asarray(mask), m(x), -1e9)
        return optax.softmax_cross_entropy_with_integer_labels(lg[None, :], yy[None])[0]

    fisher = _zeros_like_params(model)
    for i in idx:
        g = _params(g_of(model, X[i], y[i]))
        fisher = jax.tree_util.tree_map(lambda f, gg: f + gg**2, fisher, g)
    return jax.tree_util.tree_map(lambda f: f / float(n), fisher)


def _reservoir_update(buffer, X, y, cap, rng, seen):
    """Reservoir sampling (ER, Chaudhry et al. 2019) — the standard online buffer."""
    Xb, yb = buffer
    Xb, yb = list(Xb), list(yb)
    for xi, yi in zip(X, y, strict=True):
        seen += 1
        if len(Xb) < cap:
            Xb.append(xi)
            yb.append(yi)
        else:
            j = rng.integers(0, seen)
            if j < cap:
                Xb[j] = xi
                yb[j] = yi
    return (np.asarray(Xb), np.asarray(yb)), seen


def _reservoir_update_der(buffer, X, y, Z, cap, rng, seen):
    """Reservoir sampling carrying DER++'s stored logits ``Z`` alongside ``(x, y)``.

    ⚠ Deviation from the online DER++ of Buzzega et al. 2020, stated rather than
    hidden: this harness trains a whole task and *then* updates the buffer (exactly
    as its ER does), so ``Z`` is the logit vector of the model **at the end of the
    task the item belongs to** rather than at the single online step that saw it.
    Both are "the response of the network that knew this item"; the end-of-task
    version is the stronger teacher, which is generous to the baseline.
    """
    Xb, yb, Zb = (list(buffer[0]), list(buffer[1]), list(buffer[2]))
    for xi, yi, zi in zip(X, y, Z, strict=True):
        seen += 1
        if len(Xb) < cap:
            Xb.append(xi)
            yb.append(yi)
            Zb.append(zi)
        else:
            j = rng.integers(0, seen)
            if j < cap:
                Xb[j], yb[j], Zb[j] = xi, yi, zi
    return (np.asarray(Xb), np.asarray(yb), np.asarray(Zb)), seen


def _balanced_buffer(Xs, ys, cap):
    """GDumb's greedy balancer: an equal share of the budget per class seen."""
    classes = np.unique(ys)
    per = max(1, cap // len(classes))
    keep = []
    for c in classes:
        idx = np.flatnonzero(ys == c)[:per]
        keep.append(idx)
    keep = np.concatenate(keep)[:cap]
    return Xs[keep], ys[keep]


def _class_quotas(classes, budget):
    """Split a **total item budget** as evenly as possible across classes.

    ⚠ w26 fix: the previous call site used ``m_per = max(1, budget // n_classes)``,
    so at a budget below one item per class iCaRL kept ``n_classes`` exemplars —
    i.e. it silently ran **over** its budget (2.5× at the smallest frontier point).
    On a matched-BYTES frontier that is an unfair advantage, so the budget is now
    allocated exactly: ``budget // C`` each, with the first ``budget % C`` classes
    getting one more.
    """
    c = list(classes)
    base, rem = divmod(int(budget), max(1, len(c)))
    return {cls: base + (1 if i < rem else 0) for i, cls in enumerate(c)}


def _herd_exemplars(model, X, y, m_per_class, cfg, budget=None):
    """iCaRL herding: greedily pick exemplars whose running feature mean tracks the
    class mean (Rebuffi et al. 2017 §3.4). ``budget`` caps the TOTAL kept items."""
    feats = batched_features(model, X, cfg.eval_chunk)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    classes = np.unique(y)
    quota = _class_quotas(classes, budget) if budget is not None else None
    keep = []
    for c in classes:
        m_c = m_per_class if quota is None else quota[c]
        idx = np.flatnonzero(y == c)
        mu = feats[idx].mean(axis=0)
        chosen, running = [], np.zeros_like(mu)
        for k in range(min(m_c, len(idx))):
            cand = np.array([i for i in idx if i not in chosen])
            score = np.linalg.norm(mu - (running + feats[cand]) / (k + 1), axis=1)
            chosen.append(int(cand[int(score.argmin())]))
            running = running + feats[chosen[-1]]
        keep.extend(chosen)
    return np.asarray(keep, dtype=int)


def _nme_predict(model, X, ex_X, ex_y, cfg):
    """iCaRL's nearest-mean-of-exemplars classifier (no logits involved)."""
    fe = batched_features(model, ex_X, cfg.eval_chunk)
    fe = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-9)
    classes = np.unique(ex_y)
    means = np.stack([fe[ex_y == c].mean(axis=0) for c in classes])
    means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    fq = batched_features(model, X, cfg.eval_chunk)
    fq = fq / (np.linalg.norm(fq, axis=1, keepdims=True) + 1e-9)
    d = ((fq[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    return classes[d.argmin(axis=1)]


# ---------------------------------------------------------------------------
# The stream runner — returns the (T, T) accuracy matrix
# ---------------------------------------------------------------------------


def run_baseline_stream(method: str, stream, cfg, seed: int, hyper: Optional[dict] = None):
    """Run one baseline over the whole stream; return the Class-IL accuracy matrix.

    ``A[t, i]`` = accuracy on task ``i``'s **test** data after training through task
    ``t``, with the read-out masked to the classes seen up to ``t``.
    """
    hyper = hyper or {}
    key = jax.random.PRNGKey(seed + 4242)
    key, mk = jax.random.split(key)
    rng = np.random.default_rng(seed + 77)
    n_classes = cfg.n_tasks * cfg.classes_per_task
    dim = int(stream["train_X"][0].shape[1])
    model = make_net(cfg, dim, n_classes, mk)

    T = cfg.n_tasks
    A = np.zeros((T, T))
    seen_mask = np.zeros(n_classes, dtype=bool)
    buffer = (np.zeros((0, dim), np.float32), np.zeros((0,), int))
    der_buffer = (
        np.zeros((0, dim), np.float32),
        np.zeros((0,), int),
        np.zeros((0, n_classes), np.float32),
    )
    seen_count = 0
    ex_idx_all = (np.zeros((0, dim), np.float32), np.zeros((0,), int))
    ref = imp = None
    si_omega = si_ref = None
    prev_model = None
    diag = {"hyper": dict(hyper), "final_loss": []}

    for t in range(T):
        Xt, yt = stream["train_X"][t], stream["train_y"][t]
        seen_mask[stream["task_classes"][t]] = True

        if method == "joint":
            Xt = np.concatenate(stream["train_X"][: t + 1])
            yt = np.concatenate(stream["train_y"][: t + 1])

        extra, aux, buf, der = None, None, None, None
        if method == "ewc" and ref is not None:
            lam = hyper.get("ewc_lambda", cfg.ewc_lambda)
            extra = lambda m, a, xb: 0.5 * a["lam"] * _quad_penalty(m, a["ref"], a["imp"])  # noqa: E731
            aux = {"ref": ref, "imp": imp, "lam": lam}
        elif method == "si" and si_ref is not None:
            c = hyper.get("si_c", cfg.si_c)
            extra = lambda m, a, xb: a["c"] * _quad_penalty(m, a["ref"], a["imp"])  # noqa: E731
            aux = {"ref": si_ref, "imp": si_omega, "c": c}
        elif method == "lwf" and prev_model is not None:
            alpha = hyper.get("lwf_alpha", cfg.lwf_alpha)
            old_mask = jnp.asarray(stream["seen_mask_upto"][t - 1])

            # ⚠ w26 retune (task item 5): the distillation target is evaluated on
            # the CURRENT minibatch, as in Li & Hoiem 2016 / van de Ven & Tolias
            # 2019. w25 drew ONE fixed sub-batch per task and reused it for all
            # `baseline_iters` steps, which starves the distillation term and cost
            # ~4 pp against the published Split-MNIST Class-IL value.
            def extra(m, a, xb):
                lg_old = jax.vmap(a["prev"])(xb)
                lg_new = jax.vmap(m)(xb)
                om = a["old_mask"]
                p = jax.nn.softmax(jnp.where(om, lg_old, -1e9) / cfg.lwf_temp)
                q = jax.nn.log_softmax(jnp.where(om, lg_new, -1e9) / cfg.lwf_temp)
                return -a["alpha"] * jnp.mean(jnp.sum(p * q, axis=-1))

            aux = {"prev": prev_model, "alpha": alpha, "old_mask": old_mask}
        elif method in ("er", "icarl") and len(buffer[0]) > 0:
            buf = buffer
        elif method == "derpp" and len(der_buffer[0]) > 0:
            der = {
                "X": der_buffer[0], "y": der_buffer[1], "Z": der_buffer[2],
                "alpha": hyper.get("derpp_alpha", cfg.derpp_alpha),
                "beta": hyper.get("derpp_beta", cfg.derpp_beta),
            }

        if method == "si":
            # path-integral importance needs per-step (Δθ, g); approximate it with
            # the standard end-of-task estimate over the task's parameter movement
            theta_start = _params(model)

        if method == "gdumb":
            # GDumb never trains on the stream: it only fills the buffer and then
            # retrains from scratch at evaluation time.
            buffer_all = (
                np.concatenate([buffer[0], Xt]) if len(buffer[0]) else Xt,
                np.concatenate([buffer[1], yt]) if len(buffer[1]) else yt,
            )
            buffer = _balanced_buffer(buffer_all[0], buffer_all[1], cfg.memory_items)
            key, tk, mk2 = jax.random.split(key, 3)
            model = make_net(cfg, dim, n_classes, mk2)
            model, key, floss, _ = _train_task(
                model, buffer[0], buffer[1], seen_mask, cfg, tk
            )
        else:
            key, tk = jax.random.split(key)
            model, key, floss, _ = _train_task(
                model, Xt, yt, seen_mask, cfg, tk, extra_loss=extra, buffer=buf,
                aux=aux, derpp=der,
            )
        diag["final_loss"].append(floss)

        # ---- post-task consolidation -------------------------------------
        if method == "ewc":
            key, fk = jax.random.split(key)
            f_new = _fisher_diag(model, Xt, yt, seen_mask, cfg, fk, cfg.fisher_samples)
            imp = f_new if imp is None else jax.tree_util.tree_map(
                lambda a, b: a + b, imp, f_new
            )
            ref = _params(model)
        elif method == "si":
            theta_end = _params(model)
            delta = jax.tree_util.tree_map(lambda a, b: a - b, theta_end, theta_start)
            # importance ∝ |Δθ| / (Δθ² + ξ): the SI form with the path integral
            # approximated by the net task displacement (documented simplification)
            omega_new = jax.tree_util.tree_map(
                lambda d: jnp.abs(d) / (d**2 + cfg.si_xi), delta
            )
            si_omega = omega_new if si_omega is None else jax.tree_util.tree_map(
                lambda a, b: a + b, si_omega, omega_new
            )
            si_ref = theta_end
        elif method == "lwf":
            prev_model = model
        elif method == "er":
            buffer, seen_count = _reservoir_update(
                buffer, Xt, yt, cfg.memory_items, rng, seen_count
            )
        elif method == "derpp":
            Zt = batched_logits(model, Xt, cfg.eval_chunk)
            der_buffer, seen_count = _reservoir_update_der(
                der_buffer, Xt, yt, Zt, cfg.memory_items, rng, seen_count
            )
        elif method == "icarl":
            Xseen = np.concatenate([np.asarray(buffer[0]), Xt]) if len(buffer[0]) else Xt
            yseen = np.concatenate([np.asarray(buffer[1]), yt]) if len(buffer[1]) else yt
            n_seen_classes = int(seen_mask.sum())
            m_per = max(1, cfg.memory_items // max(1, n_seen_classes))
            keep = _herd_exemplars(
                model, Xseen, yseen, m_per, cfg, budget=cfg.memory_items
            )
            buffer = (Xseen[keep], yseen[keep])
            ex_idx_all = buffer

        # ---- evaluate on every task seen so far --------------------------
        for i in range(t + 1):
            Xe, ye = stream["test_X"][i], stream["test_y"][i]
            if method == "icarl" and len(ex_idx_all[0]) > 0:
                pred = _nme_predict(model, Xe, ex_idx_all[0], ex_idx_all[1], cfg)
            else:
                pred = class_il_predict(model, Xe, seen_mask, cfg.eval_chunk)
            A[t, i] = float(np.mean(pred == ye))

    n_mem = int(len(der_buffer[0]) if method == "derpp" else len(buffer[0]))
    per_item = floats_per_stored_item(cfg, dim, n_classes).get(method, 0)
    diag["memory_items"] = n_mem
    diag["memory_floats"] = int(n_mem * per_item)
    diag["floats_per_item"] = int(per_item)
    diag["fixed_state_floats"] = int(
        fixed_state_floats(cfg, dim, n_classes).get(method, 0)
    )
    return A, diag


# ---------------------------------------------------------------------------
# CL metrics — GEM formulas (Lopez-Paz & Ranzato 2017), pinned by the recon §1.4
# ---------------------------------------------------------------------------


def cl_metrics(A):
    """``ACC``, ``BWT``, ``forgetting`` and ``LA`` from a (T, T) matrix.

    ``LA`` (learning accuracy, ``mean_i A[i,i]``) is the **anti-degeneracy readout**
    added in w26: forgetting is trivially ≈0 for a method that never learned, so a
    low-forgetting claim is only meaningful next to the LA it was achieved at.
    """
    A = np.asarray(A, dtype=float)
    T = A.shape[0]
    acc = float(np.mean(A[T - 1, :]))
    la = float(np.mean([A[i, i] for i in range(T)]))
    bwt = (
        float(np.mean([A[T - 1, i] - A[i, i] for i in range(T - 1)]))
        if T > 1
        else 0.0
    )
    forget = (
        float(np.mean([max(A[t, i] for t in range(i, T)) - A[T - 1, i]
                       for i in range(T - 1)]))
        if T > 1
        else 0.0
    )
    return {"ACC": acc, "BWT": bwt, "forgetting": forget, "LA": la,
            "final_per_task": A[T - 1, :].tolist()}
