# v3-scaling-figure — results-analyst report
Task + acceptance criterion: generate V3's headline O(N)-vs-O(1) interference-scaling figure. **Status: done.**

**Figure:** `.claude/outputs/v3-interference-ntk/fig_scaling_curve.png` (script `.claude/scratch/v3-interference-ntk/fig_scaling_curve.py`, run with main venv `./.venv/bin/python`; no chlu/JAX import, ~1 s).

**What's plotted.** Per-unit *received* interference S_B = Σ_{A≠B} R_{B←A} vs lattice size N∈{4,8}, log-y, two curves in the v3 style (modular = green #2a7 circles; monolith = red #c33 squares). Data source: `.claude/outputs/v3-interference-ntk/interference_init.json` `R_matrix` fields — for each run S_B is the row-sum of off-diagonal entries (received basin displacement at unit B from all other units' CD updates); modular uses the `banded` runs (banded ≡ uniform to machine precision per the item-1b finding). **Error bars = std across the 3 seeds** {0,1,2} of the per-seed unit-mean S_B (seed error bars, as requested — not the units×seeds pooled std the report table quoted).

**Numbers (reproduce Item-3 table):** modular 6.79e-05 ±9.2e-06 (N=4) → 1.74e-04 ±3.8e-05 (N=8), ~flat O(1); monolith 6.35e-01 ±1.3e-01 (N=4) → 1.38e+00 ±2.3e-01 (N=8), ≈×2 per N-doubling, **crossing the labeled S=1 "received interference = unit's own signal" reference line between N=4 and 8** (monolith self-interferes catastrophically as it widens; modular firewall stays ~10⁻⁴). This is the V3-short headline Fig 1.

Provenance inherited verbatim from `v3-interference-ntk.md` flag-provenance table (commit `9a13455`; lattice unit_dims=[2]×N, potential=mlp hidden=32, kinetic=newtonian_learned, spring/chain coupling κ=0.05, CD probe η=0.05 r=0.5, seeds {0,1,2}). No code touched; repo read-only.

## Proposed handover updates (for the Hub)
- V3-short Fig 1 asset ready: `.claude/outputs/v3-interference-ntk/fig_scaling_curve.png` (per-unit S_B vs N, modular O(1) ~1e-4 flat vs monolith O(N) crossing S=1 at N=8, 3-seed error bars). Embeds into v3-revision-2; numbers match §8/Item-3 already-logged table (no new claims).
