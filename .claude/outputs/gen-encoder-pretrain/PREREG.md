# PREREG — `gen-encoder-pretrain`
**Filed 2026-09-14, BEFORE the first probe was fitted or any accuracy was measured.**
Base `main @ 994ca9b`, branch `agent/experiment-engineer/gen-encoder-pretrain`, worktree `../CHLU-gen-pretrain`.

## DIAL DECLARATION (echoed from the task)
- **Dial:** the encoder's pretext. ⛔ The store is NOT touched; the encoder stays FROZEN for phase 1. This produces a
  *candidate* cache, it does not change the pipeline.
- **Laundering control:** the current `recon` encoder (`gen_bridge.train_encoder(pretext="recon")`), same probe, same
  rows, same seeds — a full row on every table. Plus a **compute control** (`recon_control_2x`, the same encoder at 2x
  steps) so a compute win cannot be mistaken for a representation win.
- **Falsifies:** `scale` **and** `orientation` probe ≥ ~0.90 on SEEN data (spread grid) for some variant, or a swap
  within the family, BEFORE the store is touched. ⛔ If no variant clears it: report and STOP.
- **Does NOT falsify:** losing on hue factors (already ~saturated for every arm); a corner-only pass (see P7);
  the VAE comparison arm underperforming the AE headline (expected — see P5).

## 1. The bar
**PASS iff, on the SPREAD grid, on SEEN held-back rows, for at least one variant, `scale` ≥ 0.90 AND `orientation`
≥ 0.90 (3-seed mean, with the per-seed minimum also stated).** The corner pool (216 matched-seen rows) is reported
**separately and never substituted** for the grid number.

## 2. The instrument (fixed here, before measuring)
- **Headline probe = LINEAR.** `FactorHeads(D, cardinalities)` — one `eqx.nn.Linear` per factor on the flattened,
  z-scored penultimate feature map. This is the *same head family* as the bridge's `cls` diagnostic and the external
  classifier's output stage, and it is scored by **the same functions** (`chlu.eval.gen_probe.probe_accuracy`)
  as every other number in the phase.
- **Capacity check = MLP probe** (one hidden layer, SiLU), reported **beside** the linear column, never instead of it.
  ⭐ Pre-registered reading: linear FAIL + MLP PASS ⇒ *the factor is present but not linearly readable*; that is a
  **different finding** from a pass and is reported as such, ⛔ not as clearing the bar.
- Probe fit rows, eval rows, steps, lr, batch, standardisation: **identical for every arm and every seed.**
- `n_values_realised` beside every number; chance = `1 / n_values_realised` and the majority-class rate, both reported.

## 3. Pools (SEEN data only; the held-out cell is never probed here)
- **SPREAD GRID:** rows drawn from `split.train_rows` outside Montero's photometric corner. Realises all 8 `scale`
  and all 15 `orientation` values ⇒ chance 0.125 / 0.0667 — the same denominators `gen-navigation` reported.
- **CORNER:** the 216 `matched_seen_rows` (seen combinations inside the corner). Realises **3** `scale` and **3**
  `orientation` values ⇒ chance 0.333 / 0.333. ⚠ A number here is not comparable to a grid number.

## 4. The arms (reduced scale, declared; ⛔ no augmentation-contrastive arm — it trains invariance to exactly
   `scale` and `orientation`)
| arm | family | what differs from the control |
|---|---|---|
| `recon_control` | control | nothing — the shipped `pretext="recon"` code path, called directly |
| `recon_control_2x` | compute control | 2x `enc_steps`, otherwise identical |
| `ae_deep` | headline (AE) | wider + 2-blocks-per-stage throwaway decoder (reconstruction *pressure*) |
| `mae` | headline (MAE) | random 8x8 patch masking of the input, loss on masked pixels |
| `ae_hires` | headline (AE) | last encoder conv at stride 1 -> features 128x8x8 (D=8192, still LACE-feasible) |
| `vae` | declared comparison arm | 1x1 mu/logvar heads, reparameterised, `mse + beta * KL`; features = `mu` |
⛔ Every arm gets the **same** `enc_steps` (except the declared compute control), the same rows, the same lr, the
same batch, the same probe. Parameter count and wall-clock are reported per arm.

## 5. Predicted values — committed here, measured after
Derivation: reconstruction MSE on 64x64 Shapes3D must reproduce hue exactly (global, low-frequency, cheap) but pays
only a small pixel penalty for a blob of slightly wrong size, so `scale` is the factor a uniform-MSE objective is
most willing to blur. `orientation` (camera azimuth) moves the wall/floor boundary across the whole frame, so it is
cheaper to keep than `scale` but is a 15-way ordered read. This is the same asymmetry `gen-navigation` measured
downstream (`scale` 0.229 / `orientation` 0.236 on the read; both far below 1).

| # | prediction | central | interval |
|---|---|---|---|
| **P1** | `recon_control`, spread, LINEAR, hues (floor/wall/object) | 0.97 | ≥ 0.90 each |
| **P2** | `recon_control`, spread, LINEAR, **`scale`** | **0.55** | 0.35 – 0.75 |
| **P3** | `recon_control`, spread, LINEAR, **`orientation`** | **0.60** | 0.40 – 0.85 |
| **P4** | best AE/MAE variant beats `recon_control` on `scale` by | **+0.12** | +0.00 – +0.30 |
| **P5** | `vae` scores **below** the best AE on `scale` (KL trades fidelity) | yes | — |
| **P6** | `recon_control_2x` gains over `recon_control` on `scale` | **+0.05** | +0.00 – +0.15 |
| **P7** | corner `scale` > grid `scale` for every arm (3 classes vs 8) | yes | — |
| **P8** | **HEADLINE VERDICT: no variant clears 0.90 on BOTH `scale` and `orientation` on the spread grid at this reduced scale ⇒ FAIL ⇒ report and stop.** | FAIL | — |

## 6. The two worlds this also adjudicates (R40.A.2's pre-registered reading, restated)
- If `recon_control` **already** clears ~0.90 on both at the feature level, then the encoder is **not** the defect and
  the phase's `scale`/`orientation` loss is **downstream** (World A: the store/quantiser drops it). ⭐ That outcome
  makes this spoke's answer *"no new encoder is needed"*, which is a result, not a null.
- If `recon_control` lacks it and a variant recovers it, World B, and the candidate cache is worth promoting —
  ⛔ subject to the full re-measurement list in the report.

## 7. Stop rules
- ⛔ On FAIL: report and stop. Do not touch `gen_store.py`, `gen_compose.py`, `gen_dream.py`, `chlu/eval/`, or any
  shipped artifact path. Nothing overwrites, re-fits or invalidates the frozen cache.
- ⛔ `chlu/core/gen_bridge.py` stays byte-identical to `main` — asserted by `git diff` **and** by a test that the
  control arm's encoder `param_hash` equals `gen_bridge.train_encoder`'s for the same key and config.
