"""hub/store-content — earlier-layer activations from the FROZEN encoder.

⛔ Writes nothing into the worktree and modifies no frozen asset.  The encoder is
loaded from the pinned bridge directory, its parameter hash is asserted against the
bridge manifest, and the forward pass is re-implemented here **only** to expose the
per-block intermediates that ``GenEncoder.__call__`` discards.  ⭐ The final block's
output of :func:`block_activations` is asserted bit-identical to ``encoder(x)`` on
every call path that matters, so "the same encoder" is a measurement, not a claim.
"""
from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

from chlu.core.gen_bridge import ENCODE_PRECISION


def block_activations(encoder, x):
    """Return the post-ReLU output of EVERY conv block, innermost last.

    ``x`` is ``(C, H, W)`` (single image; vmap for a batch).  The loop is the body of
    ``GenEncoder.__call__`` verbatim — same casts, same order, same ops.
    """
    x = jnp.asarray(x, encoder.convs[0].weight.dtype)
    outs = []
    for conv, norm in zip(encoder.convs, encoder.norms):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        x = jax.nn.relu(x)
        outs.append(x)
    return outs


def _pool_to(a, hw: int):
    """Average-pool ``(N, C, H, W)`` down to ``(N, C, hw, hw)`` by exact block means.

    ⛔ Requires ``H % hw == 0`` — a non-integral factor would silently resample and
    the stored content would no longer be the layer's own spatial average.
    """
    n, c, h, w = a.shape
    if h % hw or w % hw:
        raise ValueError(f"cannot block-average {h}x{w} down to {hw}x{hw}")
    fh, fw = h // hw, w // hw
    return a.reshape(n, c, hw, fh, hw, fw).mean(axis=(3, 5))


#: ⭐ The content menu.  Every entry lands on the SAME 4x4 spatial grid as the
#: penultimate map, so the concatenation is along the channel axis and the store's
#: channel partition extends to it without reinterpretation.
#: ``(block index, pooled channels, dims)``.  block 3 == the penultimate map.
CONTENT_BLOCKS = {
    "b1": (0, 32, 512),    # (32,32,32) -> (32,4,4)
    "b2": (1, 64, 1024),   # (64,16,16) -> (64,4,4)
    "b3": (2, 128, 2048),  # (128,8,8)  -> (128,4,4)
    "p":  (3, 128, 2048),  # (128,4,4)  -> (128,4,4)   the penultimate map itself
}


def encode_blocks(encoder, images: np.ndarray, which, batch: int = 256):
    """``(N,C,H,W) images -> {name: (N, c, 4, 4) float32}`` for ``which`` block names.

    ⭐ The wrapper mirrors :func:`chlu.core.gen_bridge.encode_images` **exactly** —
    ``eqx.filter_jit`` around a function that enters ``default_matmul_precision`` at
    TRACE time and then ``jax.vmap``s — because that is what decides the accumulation
    order, and hence whether the last block comes out bit-identical to the frozen
    path.  Measured: it does.  ⛔ A plain ``jax.jit(vmap(...))`` here differed by 1
    float32 ULP and would have put my earlier-layer content on a different
    computation from the cache's penultimate content.
    """
    import equinox as eqx

    which = list(which)
    need = max(CONTENT_BLOCKS[w][0] for w in which) + 1
    dtype = encoder.convs[0].weight.dtype

    def _fwd(x):
        with jax.default_matmul_precision(ENCODE_PRECISION):
            return jax.vmap(lambda z: block_activations(encoder, z)[:need])(x)

    fwd = eqx.filter_jit(_fwd)
    acc = {w: [] for w in which}
    n = int(images.shape[0])
    for i in range(0, n, int(batch)):
        xb = jnp.asarray(images[i : i + int(batch)], dtype)
        outs = fwd(xb)
        for w in which:
            a = np.asarray(outs[CONTENT_BLOCKS[w][0]], np.float32)
            acc[w].append(_pool_to(a, 4))
        del outs
    return {w: (np.concatenate(acc[w]) if acc[w]
                else np.zeros((0, CONTENT_BLOCKS[w][1], 4, 4), np.float32))
            for w in which}


def encode_blocks_raw(encoder, images: np.ndarray, batch: int = 256):
    """Unpooled last-block output, for the bit-identity check against the cache."""
    import equinox as eqx

    dtype = encoder.convs[0].weight.dtype

    def _fwd(x):
        with jax.default_matmul_precision(ENCODE_PRECISION):
            return jax.vmap(lambda z: block_activations(encoder, z)[-1])(x)

    fwd = eqx.filter_jit(_fwd)
    out = []
    for i in range(0, int(images.shape[0]), int(batch)):
        out.append(np.asarray(fwd(jnp.asarray(images[i:i+int(batch)], dtype)), np.float32))
    return np.concatenate(out) if out else np.zeros((0,) + tuple(encoder.feat_shape), np.float32)
