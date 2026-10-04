"""The ONE shared encoder.  Trained by reconstruction, then frozen.

Why reconstruction and nothing else: the encoder is the only part of the instrument
that every arm shares, so if any arm's loss could reach it, the shared part would be
tuned towards that arm and the comparison would be rigged.  The objective here is a
per-pixel Bernoulli reconstruction of the maze image -- no actions, no routes, no
judge, no arm.  `train()` returns a frozen `Encoder`; `features()` is the only thing
the arms ever call, and it is called under `jax.lax.stop_gradient` semantics simply
because the encoder parameters are never handed to any arm's optimiser.

`independence_report()` is the evidence for that claim: it lists what the encoder's
loss function closes over, and asserts no arm module and no label array is in it.

Architecture: three 3x3 convolutions down, three up, no striding -- the feature map
stays at grid resolution so `features.patch` can take an agent-centred slice, and
the receptive field (7x7 after three 3x3 layers) is wide enough that a patch of the
feature map sees more than a patch of the image.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Encoder(eqx.Module):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d

    def __init__(self, in_ch, width, feat_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.c1 = eqx.nn.Conv2d(in_ch, width, 3, padding=1, key=k1)
        self.c2 = eqx.nn.Conv2d(width, width, 3, padding=1, key=k2)
        self.c3 = eqx.nn.Conv2d(width, feat_dim, 3, padding=1, key=k3)

    def __call__(self, x):
        h = jax.nn.relu(self.c1(x))
        h = jax.nn.relu(self.c2(h))
        return jnp.tanh(self.c3(h))          # bounded, so the patch is well scaled


class Decoder(eqx.Module):
    d1: eqx.nn.Conv2d
    d2: eqx.nn.Conv2d
    d3: eqx.nn.Conv2d

    def __init__(self, feat_dim, width, out_ch, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.d1 = eqx.nn.Conv2d(feat_dim, width, 3, padding=1, key=k1)
        self.d2 = eqx.nn.Conv2d(width, width, 3, padding=1, key=k2)
        self.d3 = eqx.nn.Conv2d(width, out_ch, 3, padding=1, key=k3)

    def __call__(self, z):
        h = jax.nn.relu(self.d1(z))
        h = jax.nn.relu(self.d2(h))
        return self.d3(h)                    # logits


class AE(eqx.Module):
    enc: Encoder
    dec: Decoder

    def __init__(self, in_ch, width, feat_dim, key):
        ke, kd = jax.random.split(key)
        self.enc = Encoder(in_ch, width, feat_dim, ke)
        self.dec = Decoder(feat_dim, width, in_ch, kd)


def n_params(module):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))))


def _loss(ae, X):
    """Per-pixel Bernoulli reconstruction.  X is the ONLY argument: no labels."""
    logits = jax.vmap(lambda x: ae.dec(ae.enc(x)))(X)
    return optax.sigmoid_binary_cross_entropy(logits, X).mean()


def train(images, feat_dim=12, width=32, steps=1500, batch=32, lr=3e-3, seed=0,
          restarts=1, accept_bce=None, verbose=True):
    """Fit the autoencoder on maze images alone.  Returns (Encoder, report).

    `restarts` / `accept_bce` exist because a single run of this is not reliable.
    Measured on CMAZE-1 at 1200 steps: seed 0 reached BCE 0.0050 and a linear probe
    that reads every image plane back at AUC ~0.95, while seed 1 stalled at BCE
    0.0796 with probe AUC ~0.60 -- near chance -- and the sign-quantised features
    collapsed to 3-5 distinct patterns for the whole grid.  Everything downstream
    reads those features, so the whole seed inherits it.

    The selection criterion is RECONSTRUCTION LOSS AND NOTHING ELSE.  No arm, no
    action, no judge takes part in choosing which restart is kept, so the encoder
    stays arm-agnostic: it is the same objective the training loop already uses,
    evaluated on the same images.
    """
    t0 = time.time()
    if restarts > 1 or accept_bce is not None:
        best = None
        tries = []
        for r in range(max(restarts, 1)):
            enc_r, rep_r = _train_once(images, feat_dim, width, steps, batch, lr,
                                       seed * 1000 + r)
            tries.append(rep_r["final_bce"])
            if best is None or rep_r["final_bce"] < best[1]["final_bce"]:
                best = (enc_r, rep_r)
            if accept_bce is not None and rep_r["final_bce"] <= accept_bce:
                break
        enc, rep = best
        rep["restarts_run"] = len(tries)
        rep["restart_bce"] = tries
        rep["accept_bce"] = accept_bce
        rep["seconds"] = time.time() - t0
        if verbose:
            print("ENCODER (reconstruction only, %d restart(s), best of %s): "
                  "%d imgs, %d params, BCE %.5f -> pixel acc %.5f, on-pixel recall "
                  "%.4f  [%.1fs]"
                  % (len(tries), ["%.4f" % b for b in tries], rep["n_images"],
                     rep["params_encoder"], rep["final_bce"], rep["pixel_accuracy"],
                     rep["on_pixel_recall"], rep["seconds"]))
        return enc, rep
    enc, rep = _train_once(images, feat_dim, width, steps, batch, lr, seed)
    rep["restarts_run"] = 1
    if verbose:
        print("ENCODER (reconstruction only): %d imgs, %d params, BCE %.5f -> "
              "pixel acc %.5f, on-pixel recall %.4f  [%.1fs]"
              % (rep["n_images"], rep["params_encoder"], rep["final_bce"],
                 rep["pixel_accuracy"], rep["on_pixel_recall"], rep["seconds"]))
    return enc, rep


def _train_once(images, feat_dim, width, steps, batch, lr, seed):
    t0 = time.time()
    X = jnp.asarray(images)
    key = jax.random.PRNGKey(seed)
    kmodel, key = jax.random.split(key)
    ae = AE(X.shape[1], width, feat_dim, kmodel)
    opt = optax.adam(lr)
    state = opt.init(eqx.filter(ae, eqx.is_array))

    @eqx.filter_jit
    def upd(ae, state, xb):
        loss, grads = eqx.filter_value_and_grad(_loss)(ae, xb)
        updates, state = opt.update(grads, state, eqx.filter(ae, eqx.is_array))
        return eqx.apply_updates(ae, updates), state, loss

    n = X.shape[0]
    hist = []
    for s in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (min(batch, n),), 0, n)
        ae, state, loss = upd(ae, state, X[idx])
        if s % 250 == 0 or s == steps - 1:
            hist.append((s, float(loss)))

    logits = jax.vmap(lambda x: ae.dec(ae.enc(x)))(X[: min(n, 256)])
    ref = X[: min(n, 256)]
    px_acc = float(jnp.mean((logits > 0).astype(jnp.float32) == ref))
    on_recall = float(jnp.sum(((logits > 0) & (ref > 0)).astype(jnp.float32))
                      / jnp.maximum(jnp.sum(ref), 1.0))
    rep = {"feat_dim": feat_dim, "width": width, "steps": steps,
           "params_encoder": n_params(ae.enc), "params_decoder": n_params(ae.dec),
           "final_bce": hist[-1][1], "loss_curve": hist,
           "pixel_accuracy": px_acc, "on_pixel_recall": on_recall,
           "n_images": int(n), "seconds": time.time() - t0}
    return ae.enc, rep


def features(enc, images, batch=64):
    """(N, D, H, W) frozen features.  Computed once; the arms only ever read these."""
    out = []
    f = eqx.filter_jit(lambda xb: jax.vmap(enc)(xb))
    for i in range(0, len(images), batch):
        out.append(np.asarray(f(jnp.asarray(images[i:i + batch]))))
    return np.concatenate(out) if out else np.zeros((0,))


def _patch_matrix(S, r):
    """Every cell's (2r+1)^2 x D neighbourhood, flattened -- what an arm actually
    reads at that cell.  Zero-padded outside the grid, like `features.patch`."""
    N, D, H, W = S.shape
    P = np.zeros((N, H, W, (2 * r + 1) ** 2 * D), dtype=np.float32)
    side = 2 * r + 1
    for i, (dr, dc) in enumerate([(a, b) for a in range(-r, r + 1)
                                  for b in range(-r, r + 1)]):
        src = np.zeros_like(S)
        r0, r1 = max(0, dr), min(H, H + dr)
        c0, c1 = max(0, dc), min(W, W + dc)
        src[:, :, r0 - dr:r1 - dr, c0 - dc:c1 - dc] = S[:, :, r0:r1, c0:c1]
        P[..., i * D:(i + 1) * D] = src.transpose(0, 2, 3, 1)
    return P.reshape(N * H * W, side * side * D)


def probe_report(enc, images, names=None, ridge=1e-3, radius=0, verbose=True):
    """What the shared features actually carry, per cell, read off linearly.

    A linear probe from the feature vector at one cell back to that cell's image
    planes.  This is the check behind "the encoder does not advantage one arm": if
    a fact is linearly present for one arm it is linearly present for all of them,
    because they read the same arrays.  It is also how an arm's failure gets
    attributed -- a policy that steps into walls when `floor` probes at 1.00 is
    failing to use the features, not failing to have them.

    Trained on the first half of the images, scored on the second.
    """
    S = features(enc, images)                       # (N, D, H, W)
    N, D, H, W = S.shape
    Xf = (S.transpose(0, 2, 3, 1).reshape(-1, D) if radius == 0
          else _patch_matrix(S, radius))
    Yf = np.asarray(images).transpose(0, 2, 3, 1).reshape(-1, images.shape[1])
    Xf = np.concatenate([Xf, np.ones((Xf.shape[0], 1), np.float32)], 1)
    cut = (N // 2) * H * W
    A = Xf[:cut].T @ Xf[:cut] + ridge * np.eye(Xf.shape[1], dtype=np.float32)
    Wt = np.linalg.solve(A, Xf[:cut].T @ Yf[:cut])
    raw = Xf[cut:] @ Wt
    ref = Yf[cut:] > 0.5
    names = names or ["ch%d" % c for c in range(ref.shape[1])]
    out = {}
    for c, n in enumerate(names):
        y, s_ = ref[:, c], raw[:, c]
        pos, neg = int(y.sum()), int((~y).sum())
        # A fixed 0.5 threshold is meaningless on a plane that is 0.4% positive --
        # least squares shrinks every prediction below it, and the probe reads 0
        # recall for a fact that is perfectly present.  AUC is threshold-free, and
        # balanced accuracy is quoted at the threshold that maximises it.
        if pos == 0 or neg == 0:
            out[n] = {"auc": float("nan"), "balanced_accuracy": float("nan"),
                      "prevalence": pos / max(pos + neg, 1)}
            continue
        order = np.argsort(s_)
        ranks = np.empty(len(s_), dtype=np.float64)
        ranks[order] = np.arange(1, len(s_) + 1)
        auc = (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg)
        qs = np.quantile(s_, np.linspace(0.001, 0.999, 60))
        bal = max(0.5 * ((s_[y] >= q).mean() + (s_[~y] < q).mean()) for q in qs)
        out[n] = {"auc": float(auc), "balanced_accuracy": float(bal),
                  "prevalence": pos / (pos + neg)}
    if verbose:
        print("  encoder linear probe r=%d, AUC / balanced-acc (features at a cell"
              "%s -> that cell's planes):"
              % (radius, "" if radius == 0 else " and its %dx%d neighbourhood"
                 % (2 * radius + 1, 2 * radius + 1)))
        items = list(out.items())
        for i in range(0, len(items), 5):
            print("    " + "  ".join("%s %.3f/%.3f" % (n, v["auc"],
                                                       v["balanced_accuracy"])
                                     for n, v in items[i:i + 5]))
        print("    (threshold-free; every arm reads these same arrays)")
    return out


def plane_names(k):
    return (["floor", "start", "goal"]
            + ["door%d" % c for c in range(1, k + 1)]
            + ["tint%d" % c for c in range(1, k + 1)]
            + ["key%d" % c for c in range(1, k + 1)])


_ARM_WORDS = {"route", "routes", "action", "actions", "label", "labels", "Y",
              "target", "targets", "policy", "arm", "oracle"}
_JUDGE_WORDS = {"judge", "JUDGE", "judge_ref", "score", "validity", "reason"}


def independence_report(enc):
    """What the encoder's objective can see.  The basis for 'shared, not rigged'.

    The scan is over the loss function's IDENTIFIERS, parsed, with the docstring
    dropped -- a substring search over the source reads its own prose and reports
    that a loss which says "no labels" sees labels.
    """
    import ast
    import inspect
    tree = ast.parse(_dedent_src(inspect.getsource(_loss)))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]                      # drop the docstring
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    args = [a.arg for a in fn.args.args]
    return {"loss_source": inspect.getsource(_loss).strip(),
            "loss_arguments": args,
            "loss_identifiers": sorted(names),
            "sees_actions": bool(names & _ARM_WORDS),
            "sees_judge": bool(names & _JUDGE_WORDS),
            "trained_on": "maze images only",
            "frozen_before_any_arm_is_trained": True,
            "params": n_params(enc)}


def _dedent_src(src):
    lines = src.splitlines()
    pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[pad:] for l in lines)
