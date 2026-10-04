"""gen classifier."""

from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from nclu.core.gen_bridge import FactorHeads


class ExternalFactorCNN(eqx.Module):
    convs: list
    norms: list
    proj: eqx.nn.Linear
    heads: FactorHeads

    def __init__(self, in_shape, channels, kernel, stride, groups, hidden, cardinalities, key):
        keys = jax.random.split(key, len(channels) + 2)
        convs, norms = ([], [])
        c_in, h, w = (int(x) for x in in_shape)
        pad = max((int(kernel) - int(stride)) // 2, 0)
        for c_out, k in zip(channels, keys[: len(channels)], strict=True):
            convs.append(
                eqx.nn.Conv2d(c_in, int(c_out), int(kernel), stride=int(stride), padding=pad, key=k)
            )
            norms.append(
                eqx.nn.GroupNorm(min(int(groups), int(c_out)), int(c_out))
                if int(groups) > 0
                else None
            )
            c_in = int(c_out)
            h = (h + 2 * pad - int(kernel)) // int(stride) + 1
            w = (w + 2 * pad - int(kernel)) // int(stride) + 1
        self.convs = convs
        self.norms = norms
        self.proj = eqx.nn.Linear(int(c_in * h * w), int(hidden), key=keys[-2])
        self.heads = FactorHeads(int(hidden), cardinalities, keys[-1])

    def __call__(self, x):
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.silu(x)
        return self.heads(jax.nn.silu(self.proj(x.reshape(-1))))
