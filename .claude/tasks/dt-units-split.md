# Task: dt-units-split — separate the data sampling interval from the Verlet integrator step, then re-measure (w20)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/dt-units-split.md` · **Branch:** `agent/experiment-engineer/dt-units-split`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/clu-latent-io-audit.md` (the w19 audit that found the units bug) · `chlu/eval/config.py` · `chlu/eval/clu_scorer.py`
- **Why:** the w19 audit found `dt=0.05` on cycle-indexed data inflates K by 400×, making `E_reg` 99.2% of the loss and 99.8% of the mass gradient, the rollout 98.3% ballistic, and `H ≈ K` at corr 0.999996. **All SDR / anti-collapse work is gated on this.** The audit called it a one-line fix. **It is not** — see Item 1.

## Item 1 — the actual defect (Hub-verified, confirm before fixing)
`cfg.dt` is read for **two physically distinct quantities**:
- **data sampling interval** — `clu_scorer.py:340, 405, 476`: `p = (q_{t+1} − q_t) / dt`. On cycle-indexed C-MAPSS the true Δt is **1 cycle**.
- **Verlet integrator step** — `clu_scorer.py:398, 491, 508`: `model(q, p, horizon, dt, gamma)`.

Both come from `chlu/eval/config.py:389` (`dt: float = 0.05`). ⚠ **Naively setting it to 1.0 fixes the momentum scale and simultaneously multiplies the integrator step by 20×.** Verlet stability needs `dt·ω ≲ 2`, so that risks trading a units bug for an integration blow-up — and we would re-run FD001 on a silently unstable integrator.

**Do:** introduce a separate `data_dt` (default **1.0**, cycle units) used *only* for finite-difference momentum construction; leave `dt` as the integrator step. Audit **every** `cfg.dt` read site in `chlu/eval/` and `chlu/training/` and classify each as data-interval or integrator-step before changing anything. Report the classification table. If any site is genuinely ambiguous, flag it rather than guessing.

## Item 2 — retune the integrator step
With momenta now O(20×) smaller, the previous `dt=0.05` may no longer be the right integrator step. Report a short stability/accuracy scan (energy drift over a fixed horizon vs `dt`) and recommend a default. State the `dt·ω` margin you are running at.

## Item 3 — re-measure what the bug was suppressing
With the split in place, re-run and report against the w19 numbers (which are the baseline to beat, all measured at the conflated `dt=0.05`):
1. **Loss composition** — is `E_reg` still 99.2% of the loss and 99.8% of the mass gradient?
2. **`H` vs `K` correlation** — still 0.999996? (i.e. does the potential now participate in its own Hamiltonian?)
3. **Ballistic fraction** — still 98.3% free-streaming?
4. **`R-1` (mass-spread) usability** — w19 found it bit-identical at λ=0/1/50 with relative gradient 1.6e-6 at `dt=0.05`, but usable (7%) at `dt=1.0`. Re-run the λ∈{0,1,50} scan and report whether R-1 is now a live lever. **This is the gate on SDR coming off hold.**
5. **Mass spectrum** — w19: `log_mass` moves +3.56 but as common mode (differential:common = 1:39), `M_max/M_min` ending at 1.153 *below* its 1.265 init. Does the differential:common ratio change once `E_reg` stops dominating? Report the ratio and the final `M_max/M_min`.
6. **FD001 arms** — re-run the baseline + `mass_lr_mult=10` + relax-lever arms. w19 baseline 0.6540, mass arm 0.7092, both plateauing ≈0.714 vs raw-stats 0.7486.

## Acceptance
The classification table (Item 1), a retuned integrator default with its stability margin (Item 2), and the six-row before/after comparison (Item 3) — each row stating whether the w19 finding **survives, changes, or reverses**. Tests green. If the re-run changes any w19 conclusion, say so explicitly and loudly; that is the most valuable output of this task.

⚠ **Do not tune toward a better FD001 number.** This is a correctness fix; the FD001 delta is a measurement, not a target. If the number gets worse, report it worse.
