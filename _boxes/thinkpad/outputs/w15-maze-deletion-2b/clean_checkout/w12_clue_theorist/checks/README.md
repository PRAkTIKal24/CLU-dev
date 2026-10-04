# `checks/` — the numerical sanity checks behind `CLUE.md`

Pure CPU, 2 threads, a few seconds in total. No scored run, no GPU, no write outside this
directory. Each script prints its own verdicts and writes a JSON beside itself.

```
cd <your CHLU worktree>                  # so `chlu` resolves from YOUR checkout
OMP_NUM_THREADS=2 python checks/check_00_env.py
OMP_NUM_THREADS=2 python checks/check_01_zero_dynamics.py
OMP_NUM_THREADS=2 python checks/check_02_crr.py
OMP_NUM_THREADS=2 python checks/check_03_provenance.py
OMP_NUM_THREADS=2 python checks/check_04_byte_ledger.py
```

| script | what it establishes | reads |
|---|---|---|
| `check_00_env.py` | `chlu.__file__` resolves **inside your worktree**, not through the shared editable `.pth`. ⛔ If this fails, every number in the others is void. | `chlu` |
| `check_01_zero_dynamics.py` | the zero-dynamics limit, probes **A–F**: the addressed point is in the atom span; an in-well settle is a no-op at 0/1/10/1000 steps; the two falsifiers (address off the span; atoms as points) both move; `λ=0` returns the static read exactly; the coercivity certificate reproduces from first principles | `chlu.core.gen_read` (production) + a **synthetic** store |
| `check_02_crr.py` | the CRR, probes **G–N**: the production integrator is the shorts' damped-Verlet map; D17 is the exact log-derivative; the fixed-`h/h*` sign trap; where `+1.1385` sits on the recorded γ grid; the underdamped floor is curvature-independent and D18 is exact; what "factor 1.10" is; the `d14_confinement` tautology | `chlu.core.integrators` (production) + `w6-potentials/test/s*/results.json` (read-only) |
| `check_03_provenance.py` | re-derives **B1** and **B2** from the recorded per-corner estimand, and scans all 56 deposited files for the D17/D18 constants | `w6-potentials/**` (read-only) |
| `check_04_byte_ledger.py` | the settle ladder is byte-identical **within** each seed, the 524,288 B store pin is exact on all three, and the single read-side column printed in `SCORED-TABLE.md` is the seed-0 value | `w6-compose-physics/scored/**/arm.json` (read-only) |

**Environment overrides** (both default to the paths used when these were run):
`W12_WORKTREE` (check_00), `W12_EVIDENCE` (check_02, check_03), `W12_COMPOSE` (check_04).

⚠ **What `check_01` is and is not.** The wave-6 fitted stores (`arm.pkl`, `rows.npz`) are **not** in
`/c/chlu/evidence/outputs/` — only their JSON summaries and prose are. `check_01` therefore exercises
the **production read path** (`span_basis`, `render_span`, `TensorAddress`) on a small **synthetic**
store built to the same shape (two rank-deficient wells, one full-rank). It checks the **algebra and
the code path**; it does **not** re-run any wave-6 scored number. Every claim it supports is labelled
that way in `CLUE.md`.

Run environment when these numbers were produced: Windows 11, python 3.11.9, numpy 2.4.1, jax 0.9.0,
`OMP_NUM_THREADS=2`. All five scripts exited 0.
