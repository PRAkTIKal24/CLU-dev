# V5 — Retention (STODY short)

Experiment data and figure-generation scripts for the "Retention" workshop short,
accepted at the NeurIPS 2026 AI for Stochastic Dynamics workshop.

## Reproducing figures

```bash
cd projects/v5-retention/plots

# Main-text figures
python -I r2_fig1.py                    # fig1_damping_optimum.png
R2_FULL=1 python -I r2_fig1.py         # figA1_damping_optimum_full.png (appendix)
python -I r2_fig2_vault.py              # fig2_vault.png
python -I r2_fig2ti.py                  # fig2_two_instruments.png

# Appendix B (forgetting theory)
python -I r2_figB_dlaw.py               # figB_dlaw.png
python -I r2_figB_signflip.py           # figB_signflip.png
python -I r2_figB_massive_vs_flat.py    # figB_massive_vs_flat.png

# Appendix C (gate characterization)
python -I r2_figC_lambda_coset.py       # figC_lambda_coset.png
python -I r2_figC_register_capacity.py  # figC_register_capacity.png
python -I r2_figC_Tstar.py              # figC_Tstar.png
python -I r2_figC2.py                   # figC2_vault_emergent.png
```

Output goes to `plots/renders/` by default; override with `FRP_OUT=<dir>`.

Dependencies: numpy, matplotlib (no chlu package required).

## Data

- `data/v5-gate/` — gate-characterization experiments (Jacobian spectra, v-curves, half-lives, vault designed-arm runs).
- `data/v5-vcurve-validation/` — v-curve validation milestone (emergent and designed mechanism sweeps).
- `data/t-lever-forgetting/` — forgetting-theory experiments (d-law cells, half-life cells, Jacobian sweep).
