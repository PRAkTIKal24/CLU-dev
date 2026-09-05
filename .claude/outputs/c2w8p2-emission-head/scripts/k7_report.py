import numpy as np, jax
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import capture_radii, flatten_unused_groups, plant_item

def sysm(cap=4, **o):
    cfg = CluSystemConfig(addr_dim=2, payload_dim=1, capacity=cap, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=0, d_safe_override=0.05,
        read_steps=800, address_steps=40, n_query_per_item=2, **o)
    return build_system(cfg, key=jax.random.PRNGKey(0), loud=False)

def m(s, site): return float(capture_radii(s, np.atleast_2d(site), n_dirs=16, steps=8, seed=0)[0])

s = sysm(); plant_item(s,0,np.array([0.5,0.0]),payload=0.1,depth=1.2,width=0.25); flatten_unused_groups(s)
a = m(s, np.array([0.5,0.0,0.1]))
s = sysm(); plant_item(s,0,np.array([0.5,0.0]),payload=0.1,depth=0.0,width=0.25); flatten_unused_groups(s)
b = m(s, np.array([0.5,0.0,0.1]))
out={}
for R in (0.30,0.60):
    s = sysm(); plant_item(s,0,np.array([+R,0.0]),payload=0.0,depth=1.2,width=0.25)
    plant_item(s,1,np.array([-R,0.0]),payload=0.0,depth=1.2,width=0.25); flatten_unused_groups(s)
    out[R]=m(s,np.array([R,0.0,0.0]))
print("K7-a isolated well  :", repr(a), " predicted 0.99609375")
print("K7-b flat site      :", repr(b), " predicted 0.0")
print("K7-c R=0.30         :", repr(out[0.30]), " predicted 0.31000")
print("K7-c R=0.60         :", repr(out[0.60]), " predicted 0.62001")
print("K7-c ratio          :", out[0.60]/out[0.30], " predicted 2.000")
