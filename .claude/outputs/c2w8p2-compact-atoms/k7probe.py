import numpy as np, jax, time
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import plant_item, flatten_unused_groups, capture_radii
from chlu.core.soft_certificate import capture_radius

def tiny(capacity=2, d_safe=0.05, seed=0, **over):
    cfg = CluSystemConfig(addr_dim=2, payload_dim=1, capacity=capacity, atoms_per_item=8,
                          min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
                          d_safe_override=float(d_safe), address_steps=40,
                          n_query_per_item=2, **over)
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)

# --- K7-1/2 synthetic
z = np.array([0.2, -0.1, 0.05])
R_true = 0.37
def relax_in(pts):
    p = np.asarray(pts, dtype=float)
    d = np.linalg.norm(p - z[None, :], axis=1, keepdims=True)
    return np.where(d <= R_true, z[None, :], z[None, :] + 10.0 * (p - z[None, :]))
print("K7-1", capture_radius(relax_in, z, n_dirs=32, r_hi=1.0, steps=12, tol=0.01, seed=0))
def relax_out(pts):
    p = np.asarray(pts, dtype=float)
    return z[None, :] + 5.0 * (p - z[None, :])
print("K7-2 floor", capture_radius(relax_out, z, n_dirs=32, r_hi=1.0, steps=12, tol=0.01, seed=0)["capture_radius"], "predicted", 0.01/5.0)
def relax_far(pts):
    return np.repeat((z + np.array([9.0, 0.0, 0.0]))[None, :], len(np.atleast_2d(pts)), axis=0)
print("K7-2b", capture_radius(relax_far, z, n_dirs=32, r_hi=1.0, steps=12, tol=0.01, seed=0)["capture_radius"])

# --- K7-3 planted two-well
t0=time.time()
s = tiny(capacity=2, d_safe=0.01, seed=0, read_steps=400)
sites = np.array([[0.30, 0.0], [-0.30, 0.0]])
for i in range(2):
    plant_item(s, i, sites[i], payload=0.0, depth=1.0, width=0.15, leak=0.0)
flatten_unused_groups(s)
z1 = np.array([0.30, 0.0, 0.0])
r = capture_radius(s._relax_points, z1, n_dirs=64, r_hi=1.0, steps=10, tol=0.15, seed=0)
print("K7-3", r, "settle", s._relax_points(z1[None, :].astype(np.float32)), time.time()-t0)

# --- K7-4 planted flat off-origin
s2 = tiny(capacity=2, d_safe=0.01, seed=0, read_steps=400)
plant_item(s2, 0, np.array([0.6, 0.0]), payload=0.0, depth=1e-9, width=0.15, leak=0.0)
flatten_unused_groups(s2)
zf = np.array([0.6, 0.0, 0.0])
print("K7-4", capture_radius(s2._relax_points, zf, n_dirs=32, r_hi=1.0, steps=10, tol=0.15, seed=0)["capture_radius"],
      s2._relax_points(zf[None, :].astype(np.float32)))

# --- K7-5 flat AT origin (declared false positive)
s3 = tiny(capacity=2, d_safe=0.01, seed=0, read_steps=400)
plant_item(s3, 0, np.array([0.0, 0.0]), payload=0.0, depth=1e-9, width=0.15, leak=0.0)
flatten_unused_groups(s3)
z0 = np.zeros(3)
print("K7-5", capture_radius(s3._relax_points, z0, n_dirs=32, r_hi=1.0, steps=10, tol=0.15, seed=0)["capture_radius"])
