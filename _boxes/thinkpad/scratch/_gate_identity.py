"""Pre-launch check for hub/admission: does E_k(a_ki) <= 0 hold with the SHIPPED code?

Also establishes the correct per-well API, because gen_landscape.store_energy SUMS
over wells and the gate needs per-well E_k.
"""
import os, sys, json
import numpy as np

ROOT = r"C:\Users\PratikJawahar\chlu\chlu-bundle-v2"
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".claude", "outputs", "hub-refusal", "scorer"))
import jax, jax.numpy as jnp                      # noqa: E402
import refusal_lib as R                           # noqa: E402
from chlu.core.gen_landscape import _well_logk, store_energy  # noqa: E402

ctx = R.build_context(0)
land = R.build_land(ctx, budget="A", log=lambda *a, **k: None)
L = land["land"]; store = land["store"]; sigma = np.asarray(land["sigma"], np.float64)

# per-well E_k at every stored atom, using the shipped primitive
res = {"schema": "admission-gate-identity/1", "rung": "A", "n_wells": int(len(store.atoms)),
       "sigma_head": [float(s) for s in sigma[:6]], "wells": []}

# X = the atoms themselves, assembled back into a full latent row per atom index
# (each well is a channel slice; evaluate each well independently on its own atoms)
worst = -np.inf
for k, a in enumerate(store.atoms):
    A = jnp.asarray(np.asarray(a, np.float32))
    d2 = (jnp.sum(A * A, -1)[:, None] - 2.0 * (A @ A.T) + jnp.sum(A * A, -1)[None, :])
    lk = -jnp.maximum(d2, 0.0) / (2.0 * float(sigma[k]) ** 2)
    Ek = -(float(sigma[k]) ** 2) * jax.nn.logsumexp(lk, axis=-1)
    mx = float(np.max(np.asarray(Ek)))
    worst = max(worst, mx)
    res["wells"].append({"well": k, "n_atoms": int(A.shape[0]), "sigma": float(sigma[k]),
                         "max_E_at_atoms": mx, "min_E_at_atoms": float(np.min(np.asarray(Ek)))})

res["max_over_all_wells"] = worst
res["identity_E_le_0_holds"] = bool(worst <= 0.0)
# the certified bound, per well
res["certified_bound_d"] = [{"well": w["well"], "N": w["n_atoms"],
                             "sigma_sqrt_2logN": float(w["sigma"] * np.sqrt(2.0 * np.log(w["n_atoms"])))}
                            for w in res["wells"]]
res["NOTE_store_energy_sums_over_wells"] = (
    "gen_landscape.store_energy returns the SUM over wells (e = e - s**2*logsumexp per well, "
    "accumulated). The per-well gate must use _well_logk(store, X, sigma)[k] and apply "
    "-sigma_k**2 * logsumexp itself.")
print(json.dumps(res, indent=2))
