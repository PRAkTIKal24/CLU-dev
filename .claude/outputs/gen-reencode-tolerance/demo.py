"""Print the measured statistics behind each acceptance item (real numbers for the report)."""
import json
import jax, numpy as np, equinox as eqx
from chlu.config import get_default_config
from chlu.core.gen_bridge import encode_images as REAL
from chlu.experiments import exp_gen_store as X

import jax.numpy as jnp
class Tiny(eqx.Module):
    W: jnp.ndarray
    feat_shape: tuple = eqx.field(static=True)
    def __call__(self, x):
        return jnp.reshape(jnp.reshape(x, (-1,)) @ self.W, self.feat_shape)

cfg = get_default_config()
OUT = {}

def run(tag, cache_rows, feats, cman, fresh=None, enc=None, imgs=None):
    lines = []
    X.stream_images = lambda *a, **k: (imgs if imgs is not None else np.zeros((cache_rows.size,1,1,cman["D"]), np.float32))
    if enc is None:
        X.encode_images = lambda *a, **k: np.asarray(fresh)
        e = object()
    else:
        X.encode_images = REAL
        e = enc
    try:
        s = X._assert_cache_reencodes("shapes3d", cfg, e, cache_rows, feats, cman, n=64, log=lines.append)
        OUT[tag] = {"verdict": "PASS", "stats": s, "log": lines}
    except AssertionError as exc:
        OUT[tag] = {"verdict": "RAISED", "message": str(exc), "log": lines}
    print(f"--- {tag}: {OUT[tag]['verdict']}")
    for l in lines: print("   ", l.strip())
    if OUT[tag]["verdict"] == "RAISED": print("    RAISE:", OUT[tag]["message"][:400])

rng = np.random.default_rng(0)
feats = rng.uniform(4.0, 8.0, size=(8, 64)).astype(np.float32).astype(np.float16).astype(np.float32)
cr = np.arange(8, dtype=np.int64)
cman = {"dtype": "float16", "D": 64, "feat_shape": (1,1,64), "n_rows": 8}

# (2) the cluster case
f = feats.copy(); f[3,7] = np.float32(f[3,7] + np.float32(0.00390625))
run("A2_cluster_one_float16_ulp", cr, feats, cman, fresh=f)

# magnitude scaling
f1 = np.full((4,64), 1.0, np.float32); c1 = np.arange(4, dtype=np.int64)
g = f1.copy(); g[1,5] = np.float32(1.0 + 0.00390625)
run("A2b_same_abs_delta_near_1.0", c1, f1, dict(cman, n_rows=4), fresh=g)

# (3) half the tensor
f = feats.copy(); fl = f.reshape(-1); fl[::2] = (fl[::2] + np.float32(0.00390625)).astype(np.float32)
run("A3_half_the_tensor_1ulp", cr, feats, cman, fresh=f)

# (1) POSITIVE CONTROL
d, n = 32, 8
k = jax.random.PRNGKey(20260915); kw, ki = jax.random.split(k)
W = jax.random.normal(kw, (d, d)) * 0.5 + 6.0
enc = Tiny(W=W, feat_shape=(1,1,d))
imgs = np.asarray(jax.random.uniform(ki, (n,1,1,d)), np.float32)
fc = REAL(enc, imgs, 512).astype(np.float16).astype(np.float32).reshape((n,d))
cm2 = {"dtype": "float16", "D": d, "feat_shape": (1,1,d), "n_rows": n}
run("A1_true_encoder", np.arange(n, dtype=np.int64), fc, cm2, enc=enc, imgs=imgs)
for eps in (1.01, 1.0001, 1.000001):
    run(f"A1_POSITIVE_CONTROL_perturbed_x{eps}", np.arange(n, dtype=np.int64), fc, cm2,
        enc=eqx.tree_at(lambda m: m.W, enc, enc.W * eps), imgs=imgs)

json.dump(OUT, open("/Users/user/Desktop/CHLU/.claude/outputs/gen-reencode-tolerance/demo.json","w"), indent=1)
