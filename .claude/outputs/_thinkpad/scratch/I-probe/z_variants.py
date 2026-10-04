"""Is store_hash a KNIFE-EDGE on last-bit conv arithmetic? Reproduce the
cross-machine situation LOCALLY by computing the SAME Z several numerically
equivalent ways and seeing whether store_hash moves."""
import os, sys, hashlib
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.environ["CHLU_ROOT"])
from chlu.config import get_default_config
from chlu.core.gen_bridge import GenEncoder, param_hash
from chlu.core.gen_store import (address_codes, channel_correlation, correlation_grouping,
                                 init_store, well_channels_from_labels)
def h(a):
    a=np.asarray(a); return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
IN=(3,16,16); SEED=20260914
ek,dk,sk = jax.random.split(jax.random.PRNGKey(SEED),3)
enc = GenEncoder(IN,(8,16),4,2,4,ek)
print("enc_hash:", param_hash(enc), "(matches Mac:", param_hash(enc)=="7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693", ")")
imgs = np.random.default_rng(0).random((32,)+IN).astype(np.float32)

variants = {
 "vmap (what the test does)": lambda: np.asarray(jax.vmap(enc)(jnp.asarray(imgs))),
 "python loop, one image"   : lambda: np.stack([np.asarray(enc(jnp.asarray(im))) for im in imgs]),
 "jit(vmap)"                : lambda: np.asarray(jax.jit(jax.vmap(enc))(jnp.asarray(imgs))),
 "vmap on 2 half-batches"   : lambda: np.concatenate([np.asarray(jax.vmap(enc)(jnp.asarray(imgs[:16]))),
                                                      np.asarray(jax.vmap(enc)(jnp.asarray(imgs[16:])))]),
}
ref=None; res={}
for name,f in variants.items():
    Z=f(); res[name]=Z
    if ref is None: ref=Z
    d=np.abs(Z.astype(np.float64)-ref.astype(np.float64))
    ulp=np.abs((Z.view(np.int32).astype(np.int64)-ref.view(np.int32).astype(np.int64)))
    Zf=Z.reshape((Z.shape[0],-1))
    lab=correlation_grouping(channel_correlation(Zf),3,method="auto")
    wc=well_channels_from_labels(lab,3)
    st=init_store(Zf,wc,4,enc.feat_shape,sk,seed=0)
    print(f"\n-- {name}")
    print(f"   Z sha            : {h(Z)}")
    print(f"   bits differ vs 1st: {int((Z!=ref).sum())}/{Z.size}  max|d|={d.max():.3e}  max ULP={int(ulp.max())}")
    print(f"   labels identical : {np.array_equal(np.asarray(lab),np.asarray(correlation_grouping(channel_correlation(ref.reshape((32,-1))),3,method='auto')))}")
    print(f"   well_sizes       : {[len(w) for w in st.well_channels]}   codes_sum: {int(address_codes(st,Zf).sum())}")
    print(f"   store_hash       : {param_hash(st)}")
    print(f"   == Mac a2a916f4  : {param_hash(st)=='a2a916f4a054a1d8574cbd87646d356b078d6906cd67cbb848353ac4da18dea1'}")
