"""Isolate the ARITHMETIC cost of Precision.HIGHEST from the per-call jit cost.

`encode_images` builds a fresh jit wrapper on every call (true on `main` too), so
its wall time is dominated by trace+compile.  Here the executable is compiled
once per arm and then hammered, so the delta is the precision's own cost.
"""
import json, time
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from chlu.core.gen_bridge import GenEncoder

enc = GenEncoder((3, 64, 64), [32, 64, 128, 128], 4, 2, 8, jax.random.PRNGKey(0))
x = jnp.asarray(np.random.default_rng(1234).random((64, 3, 64, 64), dtype=np.float32))

def arm(prec):
    def f(z):
        if prec is None:
            return jax.vmap(enc)(z)
        with jax.default_matmul_precision(prec):
            return jax.vmap(enc)(z)
    g = eqx.filter_jit(f)
    y = g(x); y.block_until_ready()
    ts = []
    for _ in range(40):
        t = time.perf_counter(); g(x).block_until_ready(); ts.append(time.perf_counter() - t)
    return np.asarray(y), sorted(ts)

y_def, t_def = arm(None)
y_hi, t_hi = arm("highest")
print(json.dumps({
    "backend": jax.devices()[0].platform,
    "device_kind": jax.devices()[0].device_kind,
    "default_min_s": t_def[0], "default_med_s": float(np.median(t_def)),
    "highest_min_s": t_hi[0], "highest_med_s": float(np.median(t_hi)),
    "ratio_min": t_hi[0] / t_def[0],
    "ratio_med": float(np.median(t_hi) / np.median(t_def)),
    "bitwise_equal": bool(np.asarray(y_def).tobytes() == np.asarray(y_hi).tobytes()),
    "max_abs_delta": float(np.max(np.abs(np.asarray(y_def, np.float64) - np.asarray(y_hi, np.float64)))),
}, indent=2, sort_keys=True))
