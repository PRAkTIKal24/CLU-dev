"""Run every G-ADDR designed control + the Ruling-3 counterfactual, dump JSON."""
import json
import sys

import jax
import numpy as np

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import (
    displaced_write_counterfactual,
    flatten_unused_groups,
    gate_addr,
    plant_item,
)

ADDR_DIM = 8
SPACING_REF = 0.10


def planted(n=6, depth=1.0, width=0.30, radius=0.7, seed=0):
    cfg = CluSystemConfig(
        addr_dim=ADDR_DIM, payload_dim=1, capacity=int(n), atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=0.05, read_steps=120, address_steps=60,
        n_query_per_item=4, atom_width=float(width))
    s = build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)
    c = np.zeros((n, ADDR_DIM))
    for i in range(n):
        c[i, i % ADDR_DIM] = radius * (1.0 if i < ADDR_DIM else -1.0)
    for i in range(n):
        plant_item(s, i, c[i], payload=0.15 * (i - n / 2.0), depth=depth,
                   width=width, leak=0.0)
    flatten_unused_groups(s)
    return s


def addr(s, **kw):
    kw.setdefault("spacing", SPACING_REF)
    kw.setdefault("n_query_per_item", 4)
    kw.setdefault("n_dirs", 8)
    kw.setdefault("bisect_steps", 6)
    kw.setdefault("seed", 0)
    return gate_addr(s, **kw)


out = {}
print("C+ positive ...", flush=True)
sp = planted()
out["C_plus_positive"] = addr(sp)
print("N2 permutation ...", flush=True)
out["N2_permutation"] = addr(sp, permute=True)
print("N1' narrow wells ...", flush=True)
out["N1prime_narrow_wells"] = addr(planted(width=0.03))
print("scale-only ...", flush=True)
for a in (0.8, 1.25):
    out[f"S_scale_{a}"] = addr(planted(width=0.30 * a, radius=0.7 * a),
                               spacing=SPACING_REF * a)
print("R3 counterfactual ...", flush=True)
r3 = {}
for local_init in (True, False):
    for dmult in (1.0, 2.0):
        cfg = CluSystemConfig(
            addr_dim=ADDR_DIM, payload_dim=1, capacity=2, atoms_per_item=16,
            min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=0,
            d_safe_override=0.05, read_steps=200, address_steps=100,
            n_query_per_item=2, write_steps=200, atom_width=0.20,
            atom_site_local_init=bool(local_init), atom_site_local_radius=0.20)
        s = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
        a = np.zeros(ADDR_DIM); a[0] = 0.4
        dl = np.zeros(ADDR_DIM); dl[1] = 0.30 * dmult
        r3[f"local_init={local_init},delta={0.30 * dmult:.2f}"] = \
            displaced_write_counterfactual(s, 0, a, 0.2, delta=dl, seed=0)
# arm A's kernel too
for kern in ("wendland",):
    cfg = CluSystemConfig(
        addr_dim=ADDR_DIM, payload_dim=1, capacity=2, atoms_per_item=16,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=0,
        d_safe_override=0.05, read_steps=200, address_steps=100,
        n_query_per_item=2, write_steps=200, atom_width=0.20,
        atom_kernel=kern, atom_kernel_cutoff=2.5,
        atom_site_local_init=True, atom_site_local_radius=0.20)
    s = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
    a = np.zeros(ADDR_DIM); a[0] = 0.4
    dl = np.zeros(ADDR_DIM); dl[1] = 0.30
    r3[f"armA_kernel={kern},delta=0.30"] = \
        displaced_write_counterfactual(s, 0, a, 0.2, delta=dl, seed=0)
out["R3_counterfactual"] = r3


def j(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [j(v) for v in o]
    return o


path = sys.argv[1]
with open(path, "w") as f:
    json.dump(j(out), f, indent=2)
print("wrote", path)
for k in ("C_plus_positive", "N2_permutation", "N1prime_narrow_wells",
          "S_scale_0.8", "S_scale_1.25"):
    g = out[k]
    print(f"{k:24s} A1={g['A1']['correct_basin_rate']:.4f} "
          f"(thr {g['A1']['threshold']:.4f}) A2={g['A2']['never_addressed_frac']:.4f} "
          f"A3a={g['A3']['A3a_cue_margin']:+.4f}±{2*g['A3']['A3a_se_paired']:.4f} "
          f"vor={g['A1']['voronoi_only_rate']:.4f} lau={g['A3']['A3a_launder_rate']:.4f} "
          f"any={g['A1']['any_basin_rate']:.4f} PASS={g['gate_addr_pass']}")
for k, v in r3.items():
    print(f"R3 {k:38s} follow={v['follow_fraction']:.4f} "
          f"moved={v['moved_off_key']:.4f} |delta|={v['delta_norm']:.3f} "
          f"resid={v['residual_to_displaced_target']:.4f} can_move={v['attractor_can_move']}")
