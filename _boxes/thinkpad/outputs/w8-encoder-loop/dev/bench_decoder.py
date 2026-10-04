"""DEV BENCH: what does one v1_capacity_schedule decoder step actually cost on this box?

The winner recipe hard-codes dec_steps=1800 and OVERRIDES cfg.dec_steps (gen_bridge.py:620,
resolve_decoder_cfg's `g()` reads the recipe first).  Its own comment prices 4800 steps at
~25 h of CPU for three seeds.  This hub refits a decoder per ARM, so the per-step cost
decides the whole design.  Measured, not extrapolated.
"""
import os, time
import numpy as np, jax, equinox as eqx

t0 = time.time()
def L(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

from chlu.config import get_default_config
from chlu.core.gen_bridge import (
    make_decoder, resolve_decoder_cfg, decoder_loss_fn, decoder_schedule,
    param_count, stream_images,
)

cfg = get_default_config().gen_bridge
feat_shape = (128, 4, 4)
img_shape = (3, 64, 64)

for name in ("v1_capacity_schedule", "baseline"):
    rec = resolve_decoder_cfg(cfg, name)
    key = jax.random.PRNGKey(0)
    dec = make_decoder(feat_shape, img_shape, rec, key)
    L(f"{name}: channels={rec.channels} blocks={rec.blocks_per_stage} "
      f"steps={rec.steps} batch={rec.batch} params={param_count(dec)}")

    import jax.numpy as jnp
    import optax
    rng = np.random.default_rng(0)
    Z = jnp.asarray(rng.standard_normal((rec.batch,) + feat_shape, dtype=np.float32))
    X = jnp.asarray(rng.random((rec.batch,) + img_shape, dtype=np.float32))
    lf = decoder_loss_fn(rec)
    opt = optax.adam(rec.lr)
    params, static = eqx.partition(dec, eqx.is_inexact_array)
    ostate = opt.init(params)
    gf = eqx.filter_value_and_grad(lambda p, s, z, x: lf(eqx.combine(p, s), z, x))

    @eqx.filter_jit
    def step(p, s, o, z, x):
        loss, g = gf(p, s, z, x)
        u, o2 = opt.update(g, o, p)
        return eqx.apply_updates(p, u), o2, loss

    params, ostate, loss = step(params, static, ostate, Z, X)  # compile
    jax.block_until_ready(loss)
    t1 = time.time()
    N = 5
    for _ in range(N):
        params, ostate, loss = step(params, static, ostate, Z, X)
    jax.block_until_ready(loss)
    sps = (time.time() - t1) / N
    L(f"{name}: {sps:.3f} s/step  ->  {rec.steps} steps = {rec.steps*sps/60:.1f} min "
      f"| 300 steps = {300*sps/60:.1f} min | 600 steps = {600*sps/60:.1f} min")
