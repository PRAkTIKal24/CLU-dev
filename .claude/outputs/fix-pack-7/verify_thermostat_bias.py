"""Item 3 verification (pre-registered in PREREG.md).

Reproduce the theorist's harmonic-well bias collapse (f5-corrigendum-2 §3,
fixes IV): B := Var(q)/(T/k) - 1 for the coded relativistic Langevin.

  - "fdt" (biased, relativistic):   B ~ -0.31 / -0.54 / -0.73 at Theta=0.5/2/8
  - "fdt_relativistic" (exact fix): B collapses to the O(eps^2) shadow floor
  - "fdt" Newtonian control:        B ~ +1e-4, flat in T

Run with the repo venv:
  PYTHONPATH=<worktree> <repo>/.venv/bin/python verify_thermostat_bias.py
"""

import jax
import jax.numpy as jnp

from chlu.core.integrators import langevin_step

K_SPRING = 1.0
M = jnp.ones(1)
M0 = 1.0
DT = 0.05
GAMMA = 0.1
N_WALKERS = 8000
N_STEPS = 12000
BURN = 4000


def rel_H(c):
    def H_fn(q, p):
        return c * jnp.sqrt(jnp.sum(p * p / M) + (M0 * c) ** 2) + 0.5 * K_SPRING * jnp.sum(q * q)
    return H_fn


def newt_H():
    def H_fn(q, p):
        return 0.5 * jnp.sum(p * p / M) + 0.5 * K_SPRING * jnp.sum(q * q)
    return H_fn


def measure(H_fn, noise_mode, m_eff, T, c, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), N_WALKERS)

    def one(wkey):
        q0 = jax.random.normal(jax.random.fold_in(wkey, 3), (1,)) * jnp.sqrt(T / K_SPRING)
        p0 = jax.random.normal(jax.random.fold_in(wkey, 7), (1,)) * jnp.sqrt(T)

        def step(carry, _):
            q, p, k = carry
            qn, pn, kn = langevin_step(
                H_fn, q, p, DT, GAMMA, T, k,
                noise_mode=noise_mode, m_eff=m_eff, rest_mass=M0, c=c,
            )
            return (qn, pn, kn), qn[0]

        (_, _, _), qs = jax.lax.scan(step, (q0, p0, wkey), None, length=N_STEPS)
        return qs[BURN:]

    qs = jax.vmap(one)(keys)
    var_q = float(jnp.var(qs))
    return var_q / (T / K_SPRING) - 1.0


if __name__ == "__main__":
    print(f"{'Theta(=T)':>10} {'fdt(biased)':>14} {'fdt_relativistic':>18} {'newt fdt ctrl':>14}")
    for T in (0.5, 2.0, 8.0):
        c = 1.0
        m_eff_rel = M0 * M  # relativistic effective inertia
        b_fdt = measure(rel_H(c), "fdt", m_eff_rel, T, c, seed=1)
        b_fix = measure(rel_H(c), "fdt_relativistic", m_eff_rel, T, c, seed=1)
        b_newt = measure(newt_H(), "fdt", M, T, c, seed=1)
        print(f"{T:>10.2f} {b_fdt:>14.5f} {b_fix:>18.5f} {b_newt:>14.5f}")
