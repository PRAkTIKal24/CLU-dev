import numpy as np, jax, hashlib
from chlu.core.clu_system import CluSystemConfig, build_system
cfg = CluSystemConfig(addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=8,
    min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=0, d_safe_override=0.05,
    read_steps=800, address_steps=40, n_query_per_item=2)
s = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
for i,k in enumerate([[0.4,0.1],[-0.5,0.3],[0.1,-0.6]]):
    s.write_stream([{"item_id": i, "address": np.array(k), "payload": 0.2*(i-1)}])
q0 = np.zeros((3, s.store.dim), dtype=np.float32); q0[:,0] = [0.1,-0.3,0.6]
qs = np.asarray(s.read(q0).state.q_star)
def h(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
print("amp     ", h(s.store.V.learned.amp))
print("centers ", h(s.store.V.learned.centers))
print("logw    ", h(s.store.V.learned.log_width))
print("q_star  ", h(qs))
print("n_bytes ", s.n_bytes())
print("losses  ", ["%.12e"%x for x in s._losses])
print("probe   ", {k: float(v) for k,v in s.self_probe().items() if k in ("acq","strict","decode")})
