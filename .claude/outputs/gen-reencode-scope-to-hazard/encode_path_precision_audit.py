"""⭐ ITEM 5 — one capped pass over the ENCODE path looking for any op still on a
default precision.

`gen-encode-precision` forced HIGHEST on two conv equations.  This does not re-derive
that; it takes the jaxpr of the **shipped** callable `encode_images` hands to
`eqx.filter_jit` and enumerates EVERY equation, flagging every one that carries a
`precision` parameter (the only parameter `jax.default_matmul_precision` can reach)
and every one that reduces (which it provably cannot reach).

Run:  PYTHONPATH=<tree> <main-venv>/bin/python encode_path_precision_audit.py
"""
import json
from collections import Counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.core import gen_bridge as GB

cfg = get_default_config().gen_bridge
enc = GB.GenEncoder(
    in_shape=(3, 64, 64),
    channels=list(cfg.enc_channels),
    kernel=int(cfg.enc_kernel),
    stride=int(cfg.enc_stride),
    groups=int(cfg.enc_groups),
    key=jax.random.PRNGKey(0),
)

# ⭐ capture the callable the SHIPPED code hands to filter_jit — not a reconstruction
captured = {}
real_filter_jit = eqx.filter_jit


def _spy(fn, *a, **k):
    captured["fn"] = fn
    return real_filter_jit(fn, *a, **k)


GB.eqx.filter_jit = _spy
try:
    GB.encode_images(enc, np.zeros((2, 3, 64, 64), np.float32), batch=2, log=lambda _m: None)
finally:
    GB.eqx.filter_jit = real_filter_jit

jaxpr = jax.make_jaxpr(captured["fn"])(jnp.zeros((2, 3, 64, 64), jnp.float32))


def walk(jp, depth=0):
    for eq in jp.eqns:
        yield depth, eq
        for v in eq.params.values():
            inner = getattr(v, "jaxpr", None)
            if inner is not None:
                yield from walk(inner, depth + 1)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    ij = getattr(x, "jaxpr", None)
                    if ij is not None:
                        yield from walk(ij, depth + 1)


eqns = list(walk(jaxpr.jaxpr))
counts = Counter(str(eq.primitive) for _, eq in eqns)

PRECISION_OPS, REDUCTIONS, OTHER = [], [], []
for _, eq in eqns:
    name = str(eq.primitive)
    if "precision" in eq.params:
        PRECISION_OPS.append((name, str(eq.params["precision"]),
                              str(eq.params.get("preferred_element_type"))))
    elif name.startswith("reduce_") or name in {"reduce_sum", "reduce_max", "argmax"}:
        REDUCTIONS.append(name)
    else:
        OTHER.append(name)

report = {
    "encoder": {
        "channels": list(cfg.enc_channels), "kernel": int(cfg.enc_kernel),
        "stride": int(cfg.enc_stride), "groups": int(cfg.enc_groups),
        "feat_shape": [int(x) for x in enc.feat_shape],
    },
    "ENCODE_PRECISION": GB.ENCODE_PRECISION,
    "n_equations": len(eqns),
    "primitive_histogram": dict(sorted(counts.items())),
    "ops_carrying_a_precision_param": [
        {"primitive": p, "precision": pr, "preferred_element_type": pet}
        for p, pr, pet in PRECISION_OPS
    ],
    "ops_on_a_DEFAULT_precision": [
        {"primitive": p, "precision": pr} for p, pr, _ in PRECISION_OPS
        if "HIGHEST" not in pr
    ],
    "reductions_precision_cannot_reach": sorted(set(REDUCTIONS)),
    "everything_else": sorted(set(OTHER)),
}
print(json.dumps(report, indent=2))
with open("encode_path_precision_audit.json", "w") as fh:
    json.dump(report, fh, indent=2)
