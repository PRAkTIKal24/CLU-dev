"""gen bridge."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from nclu.data.gen_splits import file_sha256

PRETEXTS = ("recon", "cls")
IMAGE_SHAPES = {"shapes3d": (3, 64, 64), "dsprites": (1, 64, 64)}
IMAGE_KEYS = {"shapes3d": "images", "dsprites": "imgs"}
UINT8_SCALES = {"shapes3d": 255.0, "dsprites": 1.0}


def dataset_module(dataset: str):
    """Return the loader module for ``dataset`` (Shapes3D only in this package)."""
    if dataset == "shapes3d":
        from nclu.data import shapes3d as mod
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected shapes3d)")
    return mod


def image_shape(dataset: str) -> tuple[int, int, int]:
    try:
        return IMAGE_SHAPES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {dataset!r}") from exc


def stream_images(
    dataset: str, rows: np.ndarray, root: str | Path | None = None, block: int = 8192, handle=None
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        c, h, w = image_shape(dataset)
        return np.empty((0, c, h, w), dtype=np.float32)
    mod = dataset_module(dataset)
    key = IMAGE_KEYS[dataset]
    order = np.argsort(rows, kind="stable")
    srows = rows[order]
    own = handle is None
    f = mod.open_images(root) if own else handle
    try:
        data = f[key]
        n = int(data.shape[0])
        rest = tuple((int(x) for x in data.shape[1:]))
        c, h, w = (1, rest[0], rest[1]) if len(rest) == 2 else (rest[2], rest[0], rest[1])
        out = np.empty((rows.size, c, h, w), dtype=np.float32)
        if int(srows[-1]) >= n:
            raise IndexError(f"row {int(srows[-1])} is outside {dataset} ({n} rows)")
        start = 0
        for lo in range(0, n, block):
            if start >= srows.size:
                break
            hi = min(lo + block, n)
            end = start + int(np.searchsorted(srows[start:], hi, side="left"))
            if end == start:
                continue
            chunk = np.asarray(data[lo:hi])
            sel = srows[start:end] - lo
            imgs = chunk[sel]
            if imgs.ndim == 3:
                imgs = imgs[:, None, :, :]
            else:
                imgs = np.transpose(imgs, (0, 3, 1, 2))
            scale = float(UINT8_SCALES.get(dataset, 255.0)) if imgs.dtype == np.uint8 else 1.0
            out[order[start:end]] = imgs.astype(np.float32) / scale
            start = end
        if start != srows.size:
            raise RuntimeError(f"{dataset}: streamed {start} of {srows.size} requested rows")
    finally:
        if own:
            f.close()
    return out


def param_hash(model) -> str:
    """sha256 over an ``eqx.Module``'s array leaves (shape + dtype + bytes).

    Deterministic across processes: leaves are taken in ``jax.tree`` order,
    which is the module's own field order, and every leaf contributes its shape
    and dtype as well as its bytes, so two differently-shaped models with the
    same byte count cannot collide.
    """
    params, _ = eqx.partition(model, eqx.is_array)
    hasher = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        arr = np.asarray(leaf)
        hasher.update(str(arr.shape).encode())
        hasher.update(str(arr.dtype).encode())
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def param_count(model) -> int:
    """Number of float parameters carried by ``model``."""
    params, _ = eqx.partition(model, eqx.is_array)
    return int(sum((np.asarray(x).size for x in jax.tree_util.tree_leaves(params))))


class GenEncoder(eqx.Module):
    """Small conv encoder: ``[conv(k, stride) -> GroupNorm -> relu] x len(channels)``.

    The **penultimate feature map** is the output of the last block (post-ReLU),
    i.e. what LACE would have to whiten.  GroupNorm (not BatchNorm) because it is
    stateless and identical at fit and read time — which is what "frozen" has to
    mean.
    """

    convs: list
    norms: list
    in_shape: tuple
    feat_shape: tuple

    def __init__(self, in_shape, channels, kernel, stride, groups, key):
        keys = jax.random.split(key, len(channels))
        convs, norms = ([], [])
        c_in = int(in_shape[0])
        pad = max((int(kernel) - int(stride)) // 2, 0)
        for c_out, k in zip(channels, keys, strict=True):
            convs.append(eqx.nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad, key=k))
            norms.append(eqx.nn.GroupNorm(min(groups, c_out), c_out) if groups > 0 else None)
            c_in = c_out
        self.convs = convs
        self.norms = norms
        self.in_shape = tuple((int(x) for x in in_shape))
        self.feat_shape = tuple(
            (
                int(x)
                for x in jax.eval_shape(
                    self.__call__, jax.ShapeDtypeStruct(self.in_shape, self.param_dtype)
                ).shape
            )
        )

    @property
    def param_dtype(self):
        return self.convs[0].weight.dtype

    def __call__(self, x):
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.relu(x)
        return x

    @property
    def feature_dim(self) -> int:
        """``D = C·H·W`` of the penultimate map — the LACE design constraint."""
        c, h, w = self.feat_shape
        return int(c * h * w)


def _n_upsamples(feat_shape, out_shape) -> int:
    """Number of x2 steps from the feature map to the image (raises if inexact)."""
    n_up, side = (0, int(feat_shape[1]))
    while side < int(out_shape[1]):
        side *= 2
        n_up += 1
    if side != int(out_shape[1]):
        raise ValueError(f"feature side {feat_shape[1]} does not reach {out_shape[1]} by x2 steps")
    return n_up


class GenDecoder(eqx.Module):
    """Conv renderer: penultimate feature map -> pixels in [0, 1].

    ``[bilinear x2 -> conv3 -> GroupNorm -> relu] x n`` then a final conv +
    sigmoid, with ``n`` fixed by the encoder's spatial reduction so the output
    lands exactly on the image size.
    """

    convs: list
    norms: list
    feat_shape: tuple
    out_shape: tuple

    def __init__(self, feat_shape, out_shape, channels, groups, key, blocks_per_stage=1):
        n_up = _n_upsamples(feat_shape, out_shape)
        b = max(1, int(blocks_per_stage))
        chans = list(channels[:n_up])
        while len(chans) < n_up:
            chans.append(chans[-1] if chans else feat_shape[0])
        chans = [c for c in chans for _ in range(b)] + [out_shape[0]]
        keys = jax.random.split(key, len(chans))
        convs, norms, c_in = ([], [], int(feat_shape[0]))
        for i, (c_out, k) in enumerate(zip(chans, keys, strict=True)):
            convs.append(eqx.nn.Conv2d(c_in, c_out, 3, padding=1, key=k))
            last = i == len(chans) - 1
            norms.append(
                eqx.nn.GroupNorm(min(groups, c_out), c_out) if groups > 0 and (not last) else None
            )
            c_in = c_out
        self.convs = convs
        self.norms = norms
        self.feat_shape = tuple(feat_shape)
        self.out_shape = tuple(out_shape)

    @property
    def blocks_per_stage(self) -> int:
        return (len(self.convs) - 1) // _n_upsamples(self.feat_shape, self.out_shape)

    def __call__(self, z):
        x = jnp.asarray(z, self.convs[0].weight.dtype)
        b = self.blocks_per_stage
        n_conv_up = len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            if i < n_conv_up and i % b == 0:
                x = jax.image.resize(
                    x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear"
                )
            x = conv(x)
            if norm is not None:
                x = norm(x)
            if i < len(self.convs) - 1:
                x = jax.nn.relu(x)
        return jax.nn.sigmoid(x)


class FactorHeads(eqx.Module):
    linears: list

    def __init__(self, in_dim, cardinalities, key):
        keys = jax.random.split(key, len(cardinalities))
        self.linears = [
            eqx.nn.Linear(in_dim, int(c), key=k) for c, k in zip(cardinalities, keys, strict=True)
        ]

    def __call__(self, z):
        flat = jnp.asarray(z, self.linears[0].weight.dtype).reshape(-1)
        return [lin(flat) for lin in self.linears]


class FactorProbe(eqx.Module):
    convs: list
    norms: list
    heads: FactorHeads

    def __init__(self, in_shape, channels, groups, cardinalities, key):
        keys = jax.random.split(key, len(channels) + 1)
        convs, norms = ([], [])
        c_in, h, w = in_shape
        for c_out, k in zip(channels, keys[:-1], strict=True):
            convs.append(eqx.nn.Conv2d(c_in, c_out, 4, stride=2, padding=1, key=k))
            norms.append(eqx.nn.GroupNorm(min(groups, c_out), c_out) if groups > 0 else None)
            c_in = c_out
            h, w = (h // 2, w // 2)
        self.convs = convs
        self.norms = norms
        self.heads = FactorHeads(int(c_in * h * w), cardinalities, keys[-1])

    def __call__(self, x):
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.relu(x)
        return self.heads(x)


def model_dtype(model):
    """Parameter dtype of ``model`` — under ``jax_enable_x64`` eqx builds float64
    weights and a float32 input would crash the conv (phi_encoders precedent)."""
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.result_type(leaves[0]) if leaves else jnp.float32


def _sgd_loop(
    model, loss_fn, data, steps, batch, lr, key, log: Optional[Callable] = None, schedule=None
):
    opt = optax.adam(lr if schedule is None else schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    n = int(data[0].shape[0])
    batch = min(int(batch), n)
    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(m, s, *arrays):
        loss, grads = grad_fn(m, *arrays)
        updates, s = opt.update(grads, s, m)
        return (eqx.apply_updates(m, updates), s, loss)

    history = []
    for i in range(int(steps)):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        model, opt_state, loss = step(model, opt_state, *[a[idx] for a in data])
        loss = float(loss)
        history.append(loss)
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    step {i + 1}/{steps}  loss={loss:.5f}")
    return (model, history, key)


def _xent(logits_list, labels):
    """Mean over factors of the per-factor cross-entropy (labels: (B, n_factors))."""
    total = 0.0
    for j, logits in enumerate(logits_list):
        logp = jax.nn.log_softmax(logits, axis=-1)
        total = total - jnp.mean(jnp.take_along_axis(logp, labels[:, j][:, None], axis=-1))
    return total / len(logits_list)


def train_encoder(
    images: np.ndarray,
    pretext: str,
    cfg,
    key,
    labels: Optional[np.ndarray] = None,
    cardinalities: Optional[Sequence[int]] = None,
    log: Optional[Callable] = None,
):
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}, got {pretext!r}")
    in_shape = tuple(images.shape[1:])
    key, ek, ak = jax.random.split(key, 3)
    encoder = GenEncoder(
        in_shape,
        tuple(cfg.enc_channels),
        int(cfg.enc_kernel),
        int(cfg.enc_stride),
        int(cfg.enc_groups),
        ek,
    )
    dtype = model_dtype(encoder)
    X = jnp.asarray(images, dtype)
    if pretext == "recon":
        aux = GenDecoder(
            encoder.feat_shape, in_shape, tuple(cfg.dec_channels), int(cfg.dec_groups), ak
        )

        def loss_fn(m, xb):
            enc, dec = m
            recon = jax.vmap(lambda x: dec(enc(x)))(xb)
            return jnp.mean((recon - xb) ** 2)

        data = (X,)
    else:
        if labels is None or cardinalities is None:
            raise ValueError("pretext='cls' needs labels and cardinalities")
        aux = FactorHeads(encoder.feature_dim, cardinalities, ak)
        Y = jnp.asarray(labels, jnp.int32)

        def loss_fn(m, xb, yb):
            enc, heads = m
            logits = jax.vmap(lambda x: heads(enc(x)))(xb)
            return _xent(logits, yb)

        data = (X, Y)
    model, history, key = _sgd_loop(
        (encoder, aux), loss_fn, data, cfg.enc_steps, cfg.enc_batch, cfg.enc_lr, key, log
    )
    return (model[0], model[1], history)


@dataclass(frozen=True)
class DecoderRecipe:
    """Resolved decoder hyperparameters (a config, not a fitted artifact)."""

    name: str
    channels: tuple
    groups: int
    blocks_per_stage: int
    steps: int
    batch: int
    lr: float
    loss: str
    edge_weight: float
    edge_blur: int
    grad_weight: float
    lr_schedule: str
    warmup_frac: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "channels": [int(c) for c in self.channels],
            "groups": int(self.groups),
            "blocks_per_stage": int(self.blocks_per_stage),
            "steps": int(self.steps),
            "batch": int(self.batch),
            "lr": float(self.lr),
            "loss": self.loss,
            "edge_weight": float(self.edge_weight),
            "edge_blur": int(self.edge_blur),
            "grad_weight": float(self.grad_weight),
            "lr_schedule": self.lr_schedule,
            "warmup_frac": float(self.warmup_frac),
        }


DECODER_RECIPES: dict = {
    "baseline": {},
    "v1_capacity_schedule": {
        "dec_width_mult": 2.0,
        "dec_blocks_per_stage": 2,
        "dec_steps": 1800,
        "dec_lr_schedule": "cosine",
        "dec_warmup_frac": 0.05,
    },
    "v2_object_weighted": {
        "dec_loss": "object_weighted",
        "dec_edge_weight": 8.0,
        "dec_edge_blur": 5,
        "dec_grad_weight": 1.0,
    },
}


def resolve_decoder_cfg(cfg, recipe: Optional[str] = None) -> DecoderRecipe:
    """Fold ``DECODER_RECIPES[name]`` onto ``cfg``'s ``dec_*`` knobs.

    The knobs remain the source of truth; a recipe is a named set of overrides,
    so ``dec_recipe="baseline"`` (the default) changes nothing.
    """
    name = str(recipe if recipe is not None else getattr(cfg, "dec_recipe", "baseline"))
    if name not in DECODER_RECIPES:
        raise ValueError(f"unknown decoder recipe {name!r} (have {sorted(DECODER_RECIPES)})")
    over = DECODER_RECIPES[name]

    def g(field, default=None):
        return over.get(field, getattr(cfg, field, default))

    width = float(g("dec_width_mult", 1.0))
    chans = tuple((max(1, int(round(float(c) * width))) for c in g("dec_channels")))
    return DecoderRecipe(
        name=name,
        channels=chans,
        groups=int(g("dec_groups")),
        blocks_per_stage=int(g("dec_blocks_per_stage", 1)),
        steps=int(g("dec_steps")),
        batch=int(g("dec_batch")),
        lr=float(g("dec_lr")),
        loss=str(g("dec_loss", "mse")),
        edge_weight=float(g("dec_edge_weight", 8.0)),
        edge_blur=int(g("dec_edge_blur", 5)),
        grad_weight=float(g("dec_grad_weight", 1.0)),
        lr_schedule=str(g("dec_lr_schedule", "constant")),
        warmup_frac=float(g("dec_warmup_frac", 0.05)),
    )


def make_decoder(feat_shape, out_shape, recipe: DecoderRecipe, key):
    """Build (do not fit) the recipe's decoder for an arbitrary feature shape."""
    return GenDecoder(
        feat_shape,
        out_shape,
        recipe.channels,
        recipe.groups,
        key,
        blocks_per_stage=recipe.blocks_per_stage,
    )


def _fdiff(x, axis: int):
    """Forward difference along ``axis``, zero-padded so the shape is preserved."""
    d = jnp.diff(x, axis=axis)
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, 1)
    return jnp.pad(d, pad)


def edge_weight_map(images, alpha: float, blur: int):
    y = jnp.mean(images, axis=1)
    g = jnp.abs(_fdiff(y, -2)) + jnp.abs(_fdiff(y, -1))
    k = max(1, int(blur))
    if k > 1:
        g = jax.lax.reduce_window(g, 0.0, jax.lax.add, (1, k, k), (1, 1, 1), "SAME") / float(k * k)
    g = g / (jnp.mean(g, axis=(-2, -1), keepdims=True) + 1e-08)
    return 1.0 + float(alpha) * g


def decoder_loss_fn(recipe: DecoderRecipe) -> Callable:
    if recipe.loss == "mse":

        def loss_fn(dec, zb, xb):
            return jnp.mean((jax.vmap(dec)(zb) - xb) ** 2)

        return loss_fn
    if recipe.loss == "object_weighted":
        alpha, blur, lam = (recipe.edge_weight, recipe.edge_blur, recipe.grad_weight)

        def loss_fn(dec, zb, xb):
            recon = jax.vmap(dec)(zb)
            w = jax.lax.stop_gradient(edge_weight_map(xb, alpha, blur))
            se = jnp.sum((recon - xb) ** 2, axis=1)
            num = jnp.sum(w * se, axis=(-2, -1))
            den = float(xb.shape[1]) * jnp.sum(w, axis=(-2, -1))
            l_w = jnp.mean(num / den)
            l_g = jnp.mean((_fdiff(recon, -2) - _fdiff(xb, -2)) ** 2) + jnp.mean(
                (_fdiff(recon, -1) - _fdiff(xb, -1)) ** 2
            )
            return l_w + lam * l_g

        return loss_fn
    raise ValueError(f"unknown dec_loss {recipe.loss!r} (have 'mse', 'object_weighted')")


def decoder_schedule(recipe: DecoderRecipe):
    if recipe.lr_schedule == "constant":
        return None
    if recipe.lr_schedule == "cosine":
        warm = int(max(0, round(recipe.warmup_frac * recipe.steps)))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=recipe.lr,
            warmup_steps=warm,
            decay_steps=max(1, recipe.steps),
            end_value=0.0,
        )
    raise ValueError(f"unknown dec_lr_schedule {recipe.lr_schedule!r} (have 'constant', 'cosine')")


def train_decoder_from_features(
    feat_shape,
    features: np.ndarray,
    images: np.ndarray,
    cfg,
    key,
    log: Optional[Callable] = None,
    recipe: Optional[str] = None,
):
    rec = resolve_decoder_cfg(cfg, recipe)
    in_shape = tuple(images.shape[1:])
    key, dk = jax.random.split(key)
    decoder = make_decoder(feat_shape, in_shape, rec, dk)
    dtype = model_dtype(decoder)
    Z = jnp.asarray(features, dtype)
    X = jnp.asarray(images, dtype)
    decoder, history, key = _sgd_loop(
        decoder,
        decoder_loss_fn(rec),
        (Z, X),
        rec.steps,
        rec.batch,
        rec.lr,
        key,
        log,
        schedule=decoder_schedule(rec),
    )
    return (decoder, history, rec)


ENCODE_PRECISION = "highest"
_PRECISION_ANNOUNCED: set[tuple[str, str, str]] = set()


def encode_precision_record(precision: str = ENCODE_PRECISION) -> dict:
    resolved = jax.lax.Precision(precision)
    try:
        devs = jax.devices()
        backend, kind, n_dev = (devs[0].platform, devs[0].device_kind, len(devs))
    except RuntimeError as exc:
        backend, kind, n_dev = (f"unavailable:{exc.__class__.__name__}", "unknown", 0)
    return {
        "requested": str(precision),
        "resolved": str(resolved),
        "backend": backend,
        "device_kind": kind,
        "n_devices": int(n_dev),
        "jax_version": jax.__version__,
    }


def _announce_encode_precision(log: Callable) -> dict:
    """Log the precision record once per process, per distinct device/precision."""
    rec = encode_precision_record()
    key = (rec["resolved"], rec["backend"], rec["device_kind"])
    if key not in _PRECISION_ANNOUNCED:
        _PRECISION_ANNOUNCED.add(key)
        log(
            f"  ⭐ encode precision: requested={rec['requested']} resolved={rec['resolved']} backend={rec['backend']} device={rec['device_kind']} n_devices={rec['n_devices']} (jax {rec['jax_version']})"
        )
    return rec


def encode_images(
    encoder, images: np.ndarray, batch: int = 512, log: Optional[Callable] = None
) -> np.ndarray:
    dtype = model_dtype(encoder)

    def _fwd(x):
        with jax.default_matmul_precision(ENCODE_PRECISION):
            return jax.vmap(encoder)(x)

    fwd = eqx.filter_jit(_fwd)
    _announce_encode_precision(print if log is None else log)
    out = []
    n = int(images.shape[0])
    for i in range(0, n, batch):
        chunk = jnp.asarray(images[i : i + batch], dtype)
        out.append(np.asarray(fwd(chunk), dtype=np.float32))
    if not out:
        return np.zeros((0,) + encoder.feat_shape, np.float32)
    return np.concatenate(out)


def load_feature_cache(out_dir: str | Path, verify: bool = True):
    out_dir = Path(out_dir)
    if not (out_dir / "features.npy").exists():
        from nclu.paths import features_root
        from nclu.eval.baselines.common import load_gen2_cache

        return load_gen2_cache(features_root() / out_dir.parent.name / out_dir.name, verify=verify)
    with open(out_dir / "feature_cache.json") as fh:
        manifest = json.load(fh)
    feat_path = out_dir / manifest["features_file"]
    if verify:
        got = file_sha256(feat_path)
        if got != manifest["features_sha256"]:
            raise AssertionError(
                f"feature cache checksum mismatch at {feat_path}: {got} != {manifest['features_sha256']}"
            )
    rows = np.load(out_dir / manifest["rows_file"])
    features = np.load(feat_path, mmap_mode="r")
    return (rows, features, manifest)


def features_for_rows(cache_rows: np.ndarray, features, rows: np.ndarray) -> np.ndarray:
    """Gather cached features for ``rows`` (raises if any row is not cached)."""
    rows = np.asarray(rows, dtype=np.int64)
    pos = np.searchsorted(cache_rows, rows)
    if np.any(pos >= cache_rows.size) or np.any(
        cache_rows[np.minimum(pos, cache_rows.size - 1)] != rows
    ):
        missing = rows[
            (pos >= cache_rows.size) | (cache_rows[np.minimum(pos, cache_rows.size - 1)] != rows)
        ]
        raise KeyError(
            f"{missing.size} requested row(s) are not in the feature cache (first: {int(missing[0])}) — the cache scope is too narrow"
        )
    return np.asarray(features[pos], dtype=np.float32)


CODE_GRADS = ("straight_through", "stop")


def _store_flat_views(store, zf):
    dt = store.modes[0].dtype
    Z = jnp.asarray(zf, dt).reshape((zf.shape[0], -1))
    return [Z[:, jnp.asarray(w, jnp.int32)] for w in store.well_channels]


def store_codes_traced(store, zf):
    out = []
    for z, m in zip(_store_flat_views(store, zf), store.modes, strict=True):
        d2 = jnp.sum(m * m, axis=-1)[None, :] - 2.0 * (z @ m.T)
        out.append(jnp.argmin(d2, axis=-1))
    return jnp.stack(out, axis=1)


def gather_from_codes_traced(store, codes):
    idx = jnp.asarray(codes, jnp.int32)
    n = int(idx.shape[0])
    out = jnp.zeros((n, store.dim), store.modes[0].dtype)
    for j, (w, m) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        out = out.at[:, jnp.asarray(w, jnp.int32)].set(jnp.take(m, idx[:, j], axis=0))
    return out


def quantise_through_store(
    store, zf, code_grad: str = "straight_through", bypass_alpha: float = 0.0
):
    if code_grad not in CODE_GRADS:
        raise ValueError(f"code_grad must be one of {CODE_GRADS}, got {code_grad!r}")
    Z = jnp.asarray(zf, store.modes[0].dtype).reshape((zf.shape[0], -1))
    codes = store_codes_traced(store, Z)
    gather = gather_from_codes_traced(store, codes)
    dec_in = gather
    if code_grad == "straight_through":
        dec_in = dec_in + (Z - jax.lax.stop_gradient(Z))
    if float(bypass_alpha) != 0.0:
        dec_in = dec_in + float(bypass_alpha) * Z
    return (dec_in, gather, codes)


def realised_channel_bits(store, codes) -> dict:
    C = np.asarray(codes, np.int64).reshape((np.asarray(codes).shape[0], -1))
    n_distinct = int(np.unique(C, axis=0).shape[0]) if C.size else 0
    k, m = (int(store.n_wells), int(store.n_modes))
    return {
        "design_bits": float(k * np.log2(m)),
        "design_formula": "K*log2(M)",
        "n_wells": k,
        "n_modes": m,
        "max_code_configs": int(m**k) if k * np.log2(m) < 62 else None,
        "n_requests": int(C.shape[0]) if C.size else 0,
        "n_distinct_code_configs": n_distinct,
        "realised_bits_used": float(np.log2(n_distinct)) if n_distinct > 0 else 0.0,
        "note": "design_bits is the DECLARED constant and is a property of the store, not of the run; realised_bits_used is how much of it this request set actually exercised.",
    }


def assert_no_feature_bypass(
    store,
    zf,
    code_grad: str = "straight_through",
    bypass_alpha: float = 0.0,
    jitter: float = 0.001,
    seed: int = 0,
) -> dict:
    from nclu.core.gen_store import address_codes, read_from_codes

    X = np.asarray(zf, np.float32).reshape((np.asarray(zf).shape[0], -1))
    if X.shape[0] == 0:
        raise ValueError("assert_no_feature_bypass needs at least one feature row")
    dec_in, gather, codes = quantise_through_store(store, X, code_grad, bypass_alpha)
    dec_in = np.asarray(dec_in)
    codes_np = np.asarray(codes, np.int64)
    failures = []
    shipped_codes = np.asarray(address_codes(store, X), np.int64)
    code_delta = int(np.max(np.abs(codes_np - shipped_codes))) if codes_np.size else 0
    if code_delta != 0:
        failures.append(
            f"(1) store_codes_traced disagrees with the shipped address_codes (max |delta| = {code_delta}): the trainer is not addressing the store the way the evaluation does"
        )
    shipped_read = np.asarray(read_from_codes(store, shipped_codes)).reshape(X.shape[0], -1)
    read_delta = float(np.max(np.abs(np.asarray(gather) - shipped_read)))
    if read_delta != 0.0:
        failures.append(
            f"(1) the traced gather is not read_from_codes (max |delta| = {read_delta})"
        )
    X_alt = np.asarray(shipped_read, np.float32)
    alt_in, _, alt_codes = quantise_through_store(store, X_alt, code_grad, bypass_alpha)
    alt_codes = np.asarray(alt_codes, np.int64)
    same = np.all(alt_codes == codes_np, axis=1)
    features_differ = float(np.max(np.abs(X_alt - X))) if X.size else 0.0
    if not bool(same.any()):
        raise AssertionError(
            "the collision test could not be constructed: re-addressing the store with its own modes moved every code — the store has duplicate modes"
        )
    collision_delta = float(np.max(np.abs(np.asarray(alt_in)[same] - dec_in[same])))
    if collision_delta != 0.0:
        failures.append(
            f"(2) COLLISION: two DIFFERENT feature tensors that address the SAME K codes produced DIFFERENT decoder inputs (max |delta| = {collision_delta}). ⛔ The decoder input is therefore NOT a function of the codes alone: an encoder -> decoder bypass exists and the K*log2(M)-bit channel claim is void."
        )
    rng = np.random.default_rng(int(seed))
    X_j = (X + float(jitter) * rng.standard_normal(X.shape)).astype(np.float32)
    jit_in, _, jit_codes = quantise_through_store(store, X_j, code_grad, bypass_alpha)
    jit_codes = np.asarray(jit_codes, np.int64)
    stable = np.all(jit_codes == codes_np, axis=1)
    jitter_delta = (
        float(np.max(np.abs(np.asarray(jit_in)[stable] - dec_in[stable])))
        if bool(stable.any())
        else 0.0
    )
    if jitter_delta != 0.0:
        failures.append(
            f"(3) JITTER: perturbing the features by {jitter} moved the decoder input by {jitter_delta} on rows whose codes did not change"
        )
    surrogate_delta = float(np.max(np.abs(dec_in - np.asarray(gather))))
    if surrogate_delta != 0.0:
        failures.append(
            f"(4) VALUE: the decoder's input is NOT the pure codebook gather (max |delta| = {surrogate_delta}) — a feature tensor is reaching the decoder."
        )
    if failures:
        raise AssertionError(
            "⛔ NO-BYPASS CHECK FAILED on the joint (unfrozen) path — "
            + str(len(failures))
            + " leg(s):\n  - "
            + "\n  - ".join(failures)
        )
    out = {
        "code_grad": str(code_grad),
        "bypass_alpha": float(bypass_alpha),
        "codes_match_shipped_address_codes": True,
        "gather_matches_read_from_codes": True,
        "max_abs_delta_decoder_input_vs_pure_gather": surrogate_delta,
        "collision_test_n_rows": int(same.sum()),
        "collision_test_feature_max_abs_difference": features_differ,
        "max_abs_delta_same_codes_different_features": collision_delta,
        "jitter": float(jitter),
        "jitter_test_n_rows": int(stable.sum()),
        "max_abs_delta_under_jitter": jitter_delta,
        "n_checked": int(X.shape[0]),
    }
    out.update(realised_channel_bits(store, codes_np))
    return out


def joint_loss_terms(model, xb, code_grad: str, bypass_alpha: float = 0.0):
    encoder, store, decoder = model
    z = jax.vmap(encoder)(xb)
    zf = z.reshape((z.shape[0], -1))
    dec_in, gather, _ = quantise_through_store(store, zf, code_grad, bypass_alpha)
    recon = jax.vmap(decoder)(dec_in.reshape((-1,) + tuple(store.feat_shape)))
    l_rec = jnp.mean((recon - jnp.asarray(xb, recon.dtype)) ** 2)
    l_code = jnp.mean((gather - jax.lax.stop_gradient(zf)) ** 2)
    l_commit = jnp.mean((jax.lax.stop_gradient(gather) - zf) ** 2)
    return (l_rec, l_code, l_commit)
