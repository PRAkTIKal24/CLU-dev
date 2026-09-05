"""K1 in-system: the outer LM loss's gradient into every declared leaf, +-P1."""
import json
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from chlu.data.enwik8 import contiguous_batches
from chlu.experiments.exp_anti_erosion import _prepare_erosion
from chlu.training.train_cluformer import build_arm, loss_fn, plan_pass, solve_arms

def n(x):
    return float(jnp.sqrt(jnp.sum(jnp.asarray(x) ** 2)))

out = {}
for cell in ("p1_off", "p1_on"):
    pcfg, (tr, va, te), ks, km, prov = _prepare_erosion(cell, 0, steps=20, eval_batches=1)
    specs, _ = solve_arms(pcfg, ks)
    m = build_arm("clu_store", pcfg, specs, key=km)
    x0, y0 = next(iter(contiguous_batches(va, batch=pcfg.batch, seq_len=pcfg.seq_len, n_batches=1)))
    tk, tg = jnp.asarray(x0, jnp.int32), jnp.asarray(y0, jnp.int32)
    pl, _ = plan_pass(m, tk, pcfg)
    loss, g = eqx.filter_value_and_grad(loss_fn)(m, tk, tg, pl)
    rows = {}
    for li, gb in enumerate(g.blocks):
        a = gb.cell.clu.potential_net.learned
        rows[f"layer{li}"] = {
            "atom_centers": n(a.centers), "atom_log_width": n(a.log_width),
            "atom_amp": n(a.amp),
            "exact_zero": bool(np.all(np.asarray(a.centers) == 0.0)
                               and np.all(np.asarray(a.log_width) == 0.0)
                               and np.all(np.asarray(a.amp) == 0.0)),
            "log_mass": n(gb.cell.clu.log_mass),
            "log_gamma_addr": n(gb.cell.log_gamma_addr),
            "log_gamma_read": n(gb.cell.log_gamma_read),
            "psi_res_gate": n(gb.cell.psi_res_gate),
            "psi": float(np.sqrt(sum(float(jnp.sum(jnp.asarray(l) ** 2)) for l in
                    jax.tree_util.tree_leaves(eqx.filter(gb.cell.psi, eqx.is_inexact_array))))),
            "phi": float(np.sqrt(sum(float(jnp.sum(jnp.asarray(l) ** 2)) for l in
                    jax.tree_util.tree_leaves(eqx.filter(gb.phi, eqx.is_inexact_array))))),
        }
    out[cell] = {"loss": float(loss), "layers": rows}
print(json.dumps(out, indent=1))
