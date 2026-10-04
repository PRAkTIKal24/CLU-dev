"""SMOKE 1: can a gradient reach the encoder through store -> decoder -> judge?
Nothing here is a number; it is a plumbing proof."""
import os, time, numpy as np, jax, jax.numpy as jnp, equinox as eqx
t0=time.time()
def L(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

from chlu.utils.checkpoints import load_model
from chlu.core.gen_bridge import param_hash, param_count, model_dtype, quantise_through_store
from pathlib import Path
import json

BR = Path(os.environ["CHLU_GEN_BRIDGE_ROOT"])
DL = Path(os.environ["CHLU_GEN_DECLIFT_DIR"])
CL = BR.parent.parent / "gen-external-classifier"
seed = 0
enc = load_model(BR / "shapes3d_recon" / f"seed_{seed}" / "encoder.pkl")
dec = load_model(DL / f"decoder_v1_s{seed}.pkl")
cls = load_model(CL / f"seed_{seed}" / "external_classifier.pkl")
L(f"encoder {param_hash(enc)[:12]} params={param_count(enc)} feat_shape={enc.feat_shape} D={enc.feature_dim}")
L(f"decoder {param_hash(dec)[:12]} params={param_count(dec)} out_shape={dec.out_shape}")
L(f"classifier {param_hash(cls)[:12]} params={param_count(cls)}")

# fake images
rng = np.random.default_rng(0)
x = jnp.asarray(rng.random((4,3,64,64), dtype=np.float32))
z = jax.vmap(enc)(x)
L(f"z {z.shape} {z.dtype}")
img = jax.vmap(dec)(z)
L(f"decoded {img.shape}")
lg = jax.vmap(cls)(img)
L(f"logits: {len(lg)} heads, shapes {[tuple(l.shape) for l in lg]}")

# gradient through the whole chain into the encoder
def loss(e):
    zz = jax.vmap(e)(x)
    im = jax.vmap(dec)(zz)
    ll = jax.vmap(cls)(im)
    return sum(jnp.mean(jax.nn.log_softmax(l)[:,0]) for l in ll)
g = eqx.filter_grad(loss)(enc)
gn = sum(float(jnp.sum(jnp.abs(a))) for a in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array)))
L(f"|grad| into encoder through decoder+judge = {gn:.6g}  -> {'DIFFERENTIABLE' if gn>0 else 'NO GRADIENT'}")
