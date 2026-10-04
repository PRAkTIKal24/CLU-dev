"""SPOKE I minimal reproducer. Read-only. Prints the three failing fingerprints
plus a ladder of ever-smaller probes, so the Mac can run the SAME file and diff."""
import sys, platform, hashlib, os
import jax, jaxlib, numpy as np, jax.numpy as jnp
import equinox as eqx

def h(*arrs):
    hs = hashlib.sha256()
    for a in arrs:
        a = np.asarray(a)
        hs.update(str(a.shape).encode()); hs.update(str(a.dtype).encode())
        hs.update(np.ascontiguousarray(a).tobytes())
    return hs.hexdigest()

print("=" * 78)
print("python  :", sys.version.split()[0], "|", platform.platform(), platform.machine())
print("jax     :", jax.__version__, "jaxlib", jaxlib.__version__,
      "numpy", np.__version__, "equinox", eqx.__version__)
print("backend :", jax.default_backend(), jax.devices(),
      "| x64:", jax.config.jax_enable_x64,
      "| matmul_prec:", jax.config.jax_default_matmul_precision)
print("threads : OMP=%s MKL=%s" % (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS")))
print("=" * 78)

# ---- LADDER 0: raw jax PRNG, nothing else -------------------------------
key = jax.random.PRNGKey(20260914)
print("L0a key bits          :", h(jax.random.key_data(key) if hasattr(jax.random,'key_data') else key))
ek, dk, sk = jax.random.split(key, 3)
print("L0b split(key,3)      :", h(ek, dk, sk))
print("L0c uniform(ek,(4,4)) :", h(jax.random.uniform(ek, (4, 4))))
print("L0d normal(ek,(4,4))  :", h(jax.random.normal(ek, (4, 4))))

# ---- LADDER 1: a bare equinox Conv2d, same args the encoder's 1st block uses
k2 = jax.random.split(ek, 2)
c0 = eqx.nn.Conv2d(3, 8, 4, stride=2, padding=1, key=k2[0])
print("L1a Conv2d(3,8,...)   :", h(c0.weight, c0.bias))
print("L1a weight[0,0,0,:]   :", np.asarray(c0.weight)[0, 0, 0, :].tolist())

# ---- LADDER 2: the real GenEncoder + the three failing hashes -----------
sys.path.insert(0, os.environ["CHLU_ROOT"])
from chlu.config import get_default_config
from chlu.core.gen_bridge import (GenEncoder, make_decoder, param_hash, resolve_decoder_cfg)
from chlu.core.gen_store import (address_codes, channel_correlation, correlation_grouping,
                                 init_store, store_bytes, well_channels_from_labels)
import chlu
print("chlu    :", chlu.__file__)

IN_SHAPE = (3, 16, 16)
enc = GenEncoder(IN_SHAPE, (8, 16), 4, 2, 4, ek)
enc_hash = param_hash(enc)
print("L2a enc_hash          :", enc_hash)
print("    expected (Mac)    : 7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693")
print("    MATCH             :", enc_hash == "7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693")
print("    feat_shape        :", [int(x) for x in enc.feat_shape], "expected [16, 4, 4]")
print("    n leaves          :", len(jax.tree_util.tree_leaves(eqx.partition(enc, eqx.is_array)[0])))
for i, lf in enumerate(jax.tree_util.tree_leaves(eqx.partition(enc, eqx.is_array)[0])):
    a = np.asarray(lf)
    print(f"    leaf[{i}] {str(a.shape):18s} {a.dtype} sha={h(a)[:16]} first3={a.ravel()[:3].tolist()}")

dbase = make_decoder(enc.feat_shape, IN_SHAPE,
                     resolve_decoder_cfg(get_default_config().gen_bridge), dk)
dbh = param_hash(dbase)
print("L2b dec_hash_baseline :", dbh)
print("    expected (Mac)    : 457977e44d7b2a6ac30742fa5632aad8d31cfab67c71c61e24265fb50b38fea3")
print("    MATCH             :", dbh == "457977e44d7b2a6ac30742fa5632aad8d31cfab67c71c61e24265fb50b38fea3")

dv1 = make_decoder(enc.feat_shape, IN_SHAPE,
                   resolve_decoder_cfg(get_default_config().gen_bridge, "v1_capacity_schedule"), dk)
print("L2c dec_hash_v1       :", param_hash(dv1))
print("    expected (Mac)    : 40236b47ba4f53c913b5ac247af50abad16fd742d04c54fdacc3f588a34251df")
print("    MATCH             :", param_hash(dv1) == "40236b47ba4f53c913b5ac247af50abad16fd742d04c54fdacc3f588a34251df")

# ---- LADDER 3: computed (not just initialised) quantities ---------------
rng = np.random.default_rng(0)
imgs = rng.random((32,) + IN_SHAPE).astype(np.float32)
print("L3a imgs sha          :", h(imgs))          # pure numpy PCG64 -> must match
Z = np.asarray(jax.vmap(enc)(jnp.asarray(imgs)))   # first real FORWARD PASS
print("L3b Z sha (fwd pass)  :", h(Z))
print("    Z[0,0,0,:4]       :", Z[0, 0, 0, :4].tolist())
Zf = Z.reshape((Z.shape[0], -1))
labels = correlation_grouping(channel_correlation(Zf), 3, method="auto")
print("L3c labels            :", np.asarray(labels).tolist())
wc = well_channels_from_labels(labels, 3)
store = init_store(Zf, wc, 4, enc.feat_shape, sk, seed=0)
sh = param_hash(store)
print("L3d store_hash        :", sh)
print("    expected (Mac)    : a2a916f4a054a1d8574cbd87646d356b078d6906cd67cbb848353ac4da18dea1")
print("    MATCH             :", sh == "a2a916f4a054a1d8574cbd87646d356b078d6906cd67cbb848353ac4da18dea1")
print("    well_sizes        :", [len(w) for w in store.well_channels], "expected [8, 23, 225]")
print("    codes_sum         :", int(address_codes(store, Zf).sum()), "expected 84")
print("    bits_per_read     :", store.bits_per_read, "expected 6.0")
print("    store_bytes       :", store_bytes(store)["store_bytes"], "expected 4096")
print("=" * 78)
