"""B5 — the read-length law, measured directly on the payload trajectory.

The shipped read phase is UNDAMPED (`gamma_read = 0.0`), so the theorist's overdamped
`tau_y = eta / (kappa (g0 + A))` cannot be the operative law: with eta = 0 the payload
coordinate does not relax, it OSCILLATES about `abar` with `omega = sqrt(kappa * G)`, and
the tail average returns `abar` only once the averaging window covers a period.

This measures the period directly (zero crossings of `q2 - mean` over the read phase) for a
grid of (g0, A) and compares it against both candidate laws. One store, one query set, no
vmap — cheap and decisive.
"""
import json
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mia-decay-measurement"))

import mia_harness as MH  # noqa: E402

from chlu.core.memory_potentials import AtomStorePotential  # noqa: E402
from chlu.experiments.exp_learned_memory import model_for  # noqa: E402
from chlu.experiments.exp_sequential_write import two_phase  # noqa: E402

CFG, DIM, CAP = MH.CFG, MH.DIM, MH.CAP


def period_of(gate, g0, A, a_i=1.0, c=(0.0, 0.0)):
    """Measured oscillation period of q2 during the (undamped) read phase."""
    V = AtomStorePotential(dim=DIM, capacity=CAP, alpha=CFG.atom_alpha, s=CFG.atom_width,
                           kappa=CFG.payload_kappa, payload_gate=gate, payload_g0=g0)
    V = V.with_item(list(c), a_i, amp=A)
    site = np.zeros((1, DIM)); site[0, :2] = c
    Q0, P0, _ = MH.make_queries_at(jax.random.PRNGKey(0), jnp.asarray(site), 4, CFG,
                                   dim=DIM)
    _, traj = two_phase(model_for(V, DIM), Q0, P0, CFG, CFG.gamma_address, CFG.gamma_read)
    y = np.asarray(traj[:, :, 2])                  # (n_query, read_steps)
    t = np.arange(y.shape[1]) * CFG.dt
    per, amp = [], []
    for row in y:
        z = row - row.mean()
        s = np.sign(z)
        cross = np.where(np.diff(s) != 0)[0]
        if len(cross) >= 3:
            per.append(2.0 * float(np.mean(np.diff(t[cross]))))
        amp.append(float(z.max() - z.min()))
    return (float(np.mean(per)) if per else float("nan"), float(np.mean(amp)),
            float(np.abs(y[:, -int(CFG.tail_frac * y.shape[1]):].mean() - a_i)))


if __name__ == "__main__":
    rows = []
    for gate, g0 in ((False, 0.0), (True, 0.05), (True, 0.005)):
        for A in (1.0, 0.5, 0.2, 0.1, 0.06, 0.051):
            T, amp, err = period_of(gate, g0, A)
            G = (g0 + A) if gate else 1.0          # baseline stiffness is kappa, flat in A
            rows.append({"gate": gate, "g0": g0, "A": A, "G": G, "T_meas": T,
                         "T_sqrt_law": float(2 * np.pi / np.sqrt(CFG.payload_kappa * G)),
                         "osc_amp": amp, "tail_err": err,
                         "window": CFG.tail_frac * CFG.read_steps * CFG.dt})
            print(f"gate={gate} g0={g0} A={A:<6g} G={G:.4f}  T_meas={T:8.3f}  "
                  f"T_pred={rows[-1]['T_sqrt_law']:8.3f}  osc_amp={amp:.3f}  "
                  f"tail_err={err:.4f}", flush=True)
    with open(os.path.join(HERE, "osc_period.json"), "w") as fh:
        json.dump({"read": {"dt": CFG.dt, "read_steps": CFG.read_steps,
                            "tail_frac": CFG.tail_frac, "gamma_read": CFG.gamma_read,
                            "n_subsample": CFG.n_subsample}, "rows": rows}, fh, indent=2)
    r = [x for x in rows if x["gate"]]
    lo = np.log([x["G"] for x in r]); lt = np.log([x["T_meas"] for x in r])
    slope = float(np.polyfit(lo, lt, 1)[0])
    print(f"\nlog-log slope d ln T / d ln G = {slope:.4f}  "
          f"(sqrt law: -0.5, overdamped law: -1.0)")
