"""K2 reference capture — run at UNMODIFIED code (main @ 104ca19).

Dumps bit-level fingerprints of the run-2-config toy store cell's write output
and of the outer loss's gradients, so the `erosion_partition=False` arm can be
proven bit-identical AFTER the P1/I1 build lands.
"""
import hashlib
import json
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.data.enwik8 import contiguous_batches
from chlu.experiments.exp_psi_residual import _prepare_residual
from chlu.training.train_cluformer import (
    build_arm, loss_fn, plan_pass, solve_arms,
)


def h(x) -> str:
    a = np.asarray(x, dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main(seed: int = 0) -> int:
    pcfg, (tr, va, te), k_solve, k_model, prov = _prepare_residual(
        "run1", seed, steps=2, eval_batches=1, source="q_star", gain=1.0,
        trainable=True)
    specs, ledger = solve_arms(pcfg, k_solve)
    model = build_arm("clu_store", pcfg, specs, key=k_model)
    x0, y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                          seq_len=pcfg.seq_len, n_batches=1)))
    tk = jnp.asarray(x0, dtype=jnp.int32)
    tg = jnp.asarray(y0, dtype=jnp.int32)
    plans, _ = plan_pass(model, tk, pcfg)

    blk = model.blocks[0]
    cell = blk.cell
    hh = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    hh = hh + model.pos[: hh.shape[1]][None]
    z = jax.vmap(blk.chunk_latents)(hh)
    pl0 = jax.tree_util.tree_map(lambda a: a[0], plans[0])
    st = cell.init_state()
    for c in range(int(z.shape[1])):
        pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl0)
        st = cell.write(st, z[0, c], pc)
    r = np.asarray(cell.read(st, z[0, 0]))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tk, tg, plans)
    g = grads.blocks[0].cell

    def n(x):
        return float(jnp.sqrt(jnp.sum(jnp.asarray(x) ** 2)))

    out = {
        "seed": seed,
        "commit": "104ca19",
        "cell": "run1 (h1b_m0.6 + psi residual q_star) = the CSF3 run-2 config",
        "n_chunks": int(z.shape[1]),
        "state_hash": {"centers": h(st.centers), "log_width": h(st.log_width),
                       "amp": h(st.amp), "codebook": h(st.codebook)},
        "state_stat": {"amp_sum": float(jnp.sum(st.amp)),
                       "centers_sum": float(jnp.sum(st.centers)),
                       "log_width_sum": float(jnp.sum(st.log_width))},
        "read_hash": h(r),
        "read": [float(v) for v in r],
        "loss": float(loss),
        "grad_norms": {
            "atom_centers": n(g.clu.potential_net.learned.centers),
            "atom_log_width": n(g.clu.potential_net.learned.log_width),
            "atom_amp": n(g.clu.potential_net.learned.amp),
            "log_mass": n(g.clu.log_mass),
            "log_gamma_addr": n(g.log_gamma_addr),
            "log_gamma_read": n(g.log_gamma_read),
            "psi_res_gate": n(g.psi_res_gate),
            "phi": float(jnp.sqrt(sum(jnp.sum(jnp.asarray(l) ** 2) for l in
                                      jax.tree_util.tree_leaves(
                                          eqx.filter(grads.blocks[0].phi,
                                                     eqx.is_inexact_array))))),
            "psi": float(jnp.sqrt(sum(jnp.sum(jnp.asarray(l) ** 2) for l in
                                      jax.tree_util.tree_leaves(
                                          eqx.filter(g.psi, eqx.is_inexact_array))))),
        },
        "grad_atom_hash": {
            "centers": h(g.clu.potential_net.learned.centers),
            "log_width": h(g.clu.potential_net.learned.log_width),
            "amp": h(g.clu.potential_net.learned.amp),
        },
        "cell_ledger": ledger.get("clu_store"),
        "jax": jax.__version__,
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
