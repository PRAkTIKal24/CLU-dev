"""M1: is a co-scaled COMPACT atom dead on the shipped scattered init?
Same shape as ERRATA-C2W8 §3's timing probe: 3 designed-site writes at addr_dim=8."""
import numpy as np, jax, time
from chlu.core.clu_system import CluSystemConfig, build_system

MED_NN = 0.1407  # seed-0 measured task-1 key spacing (pass 1); probe only
def run(kernel, width, local, cutoff=2.5):
    cfg = CluSystemConfig(addr_dim=8, payload_dim=1, capacity=16, budget=16, seed=0,
                          leak=0.02, stage_lifetimes=True, d_safe_override=0.88*MED_NN,
                          write_steps=300, read_steps=800, address_steps=400,
                          n_query_per_item=2, atom_width=width, atom_kernel=kernel,
                          atom_kernel_cutoff=cutoff, atom_site_local_init=local,
                          atom_site_local_radius=width)
    s = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 8)); A /= np.linalg.norm(A, axis=1, keepdims=True); A *= 0.8
    t0 = time.time()
    for i in range(3):
        s.write_stream([{"item_id": i, "address": A[i], "payload": 0.1*(i-1)}],
                       key=jax.random.PRNGKey(100+i))
    d, w = s.well_fits()
    rows = np.asarray(s.store.group_rows(0), dtype=bool)
    c = np.asarray(s.store.atoms.centers)[rows]
    z = np.zeros(9); z[:8] = A[0]; z[8] = -0.1
    dist = np.linalg.norm(c - z[None, :], axis=1)
    R = cutoff*width
    print(f"{kernel:20s} s={width:.4f} local={local} n_atoms={cfg.n_atoms} "
          f"depth={np.round(d,4).tolist()} nearest_own={dist.min():.4f} R={R:.4f} "
          f"in_support={(dist<=R).sum()}/{len(dist)} probe_strict={s.self_probe()['strict']:.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

run("gaussian", 0.3, False)              # the pass-1 store
run("wendland", 0.5*MED_NN, False)       # M1: compact + scattered init  -> predicted DEAD
run("wendland", 0.5*MED_NN, True)        # compact + site-local init
run("wendland", 1.0*MED_NN, True)
