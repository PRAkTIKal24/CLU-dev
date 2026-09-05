# regime-remap-2000ep — results-analyst report

Task + acceptance criterion: re-map the CLU-gate-vs-Hopfield regime grid at **2000 training epochs** (vs the 500-ep v1-hopfield-stress baseline) to resolve **CM-8** (frozen PROVISIONAL). Deliverables: corrected regime map + honest CM-8 rewrite (Item 1); epoch-scaling frontier / capacity-wall probe at kv∈{32,64,96,128} (Item 2); Hopfield compute-parity fairness check (Item 3); negatives written per C-9 (Item 4).

Status: **partial (headline decided; grid still filling)** — see §Coverage. Repo **read-only** (no tracked files touched); all artifacts under gitignored `.claude/`. Laptop CPU (warm JAX 0.9.0 session), f32 training.

---

## 0. Setup — apparatus, commands, provenance

**Apparatus (reused verbatim, no core edits).** Each grid cell trains a per-episode CLU memory EBM via `train_generative` (genuine PCD; persists its own buffer) on MQAR key‖value patterns, fits per-episode calibration heads on a write-time self-test, then runs the deployment ladder (governed relaxation) + the modern-Hopfield baseline on the *same* cue. Driver calls `chlu.experiments.exp_v1_calibration._regime_cell` + `_regime_metrics` **verbatim** (the same functions v1-hopfield-stress and anchor-robustness Item-2 used), so the **only** variable vs the 500-ep baseline is `train_epochs`. Metrics per cell: `fidelity` (fraction of stored patterns correctly recalled when relaxed from themselves), `clu_gate_acc` (learned-τ operating point, `p_exit`=0.5), `clu_full_acc` (full-budget = 3000 Verlet steps), `hop_acc`, `savings` = CLU-full-cost / CLU-gate-cost.

- Driver: `.claude/scratch/regime-remap-2000ep/driver.py` (one JSON per (axis,N,kv,stress,seed,epochs,vocab); idempotent/resumable).
- Batch runner: `batch.py <jobs_file>` (one warm JAX process; resumable).
- Command (per cell): `PYTHONPATH=/Users/user/Desktop/CHLU .venv/bin/python driver.py <axis> <N> <kv> <stress> <seed> <episodes> <epochs> [vocab]`.

**Flag-provenance (per PROTOCOL §5) — `train_epochs` is THE flag, tabled everywhere below.**

| flag | value | flag | value |
|---|---|---|---|
| commit | `63fea62` (main, w6-integrated) | JAX | 0.9.0 (main venv) |
| experiment | `experiment_v1_gate` regime map | dtype | f32 |
| **train_epochs** | **{500, 1000, 2000, 4000}** (the arm) | kinetic_energy_mode | **relativistic** |
| potential_type | **mlp** (coercive, F5 Prop-10) | rest_mass / c | 1.0 / 1.0 |
| embed_dim | 16 (CLU dim = 32) | hidden_dim | 128 |
| vocab_size | 256 (**512** for kv128 only) | regime_n_clusters | 8 |
| train_lr | 1e-3 | train_batch_size | 32 |
| train_k_steps | 50 | train_buffer_capacity | 256 |
| train_friction (sleep γ) | 0.3 | train_temperature | 0.3 |
| train_input_noise_sigma | 0.05 | reinit_prob | 0.25 |
| **langevin_noise** | **legacy** (FDT-violating, §7.9; default) | persistent buffer | PCD (internal) |
| clamp_key / clamp_outputs | True / True | energy_weight | 1.0 |
| relax_steps | 300 | governor_sensitivity | 0.95 |
| calib_features | r_margin | calib_p_exit | 0.5 |
| calib_n_stages × stage_steps | 3 × 900 (cost ladder 300/1200/2100/3000) | hopfield_beta | 20.0 |
| gap_distribution | uniform | seeds | {42,43,44,45,46} (Item1) / {42,43,44} (Item2) |
| episodes/cell | 5 (corr=0 column) / 3 (stress, frontier) | comparable_margin | 0.03 |

Note: `sleep_temperature=0.3` is *active* here (sleep_friction=0.3>0, so the w4 "temperature is a no-op at γ=0" caveat does not apply). `langevin_noise=legacy` is the FDT-violating default that every prior regime run used — kept for baseline continuity.

---

## 3. Item 3 — Hopfield compute-parity fairness check (COMPLETE)

Modern-Hopfield retrieval is *iterable* (`z_{t+1}=softmax(β z_t Xᵀ)X`, key-half clamped to the cue each step). I swept iteration count × β to answer the reviewer question *"did you give Hopfield the same budget?"*. corr=0, 3 seeds × 3 episodes; script `hopfield_cost.py`, data `hopfield_cost.json`.

**Accuracy vs n_iter (mean over 3 seeds):**

| cell | β | 1 iter | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| N128/kv32 | 5 | 0.976 | 0.976 | 0.979 | 0.979 | 0.979 |
| N128/kv32 | 20 | 0.976 | 0.976 | 0.976 | 0.976 | 0.976 |
| N256/kv64 | 5 | 0.967 | 0.969 | 0.969 | 0.969 | 0.969 |
| N256/kv64 | 20 | 0.969 | 0.969 | 0.969 | 0.969 | 0.969 |
| N384/kv96 | 5 | 0.941 | 0.950 | 0.949 | 0.949 | 0.949 |
| N384/kv96 | 20 | 0.947 | 0.947 | 0.947 | 0.947 | 0.947 |

At β≥5 Hopfield reaches its ceiling in **a single matvec**; 10× more iterations change accuracy by ≤0.003. Only at weak β=2 does the 1→2 iter step help (kv96 0.817→0.926), converging by iter 2. **Hopfield's cost floor is 1 matvec (O(kv·d)); extra budget buys it nothing.**

**Consequence for the savings claim (the P8 no-strawman, in reverse):** the "9–10× savings" reported at 2000 ep is **CLU-gate cost vs CLU-full-budget cost** (3000 Verlet steps), i.e. an *intra-CLU compute-rationing* number — it is **NOT** savings relative to Hopfield. Hopfield attains ≥0.947 accuracy at ~1 matvec, still far cheaper per query than even the gated CLU (which runs ≥300 Verlet steps, each a potential-MLP forward+grad). The honest parity statement: *at matched accuracy, Hopfield is the cheaper retriever; the CLU's compute-rationing advantage is measured against a full-budget CLU, and "matching Hopfield" means the gate's accuracy reaches Hopfield's, not that it does so at lower cost than Hopfield.*

---

<!-- Items 1, 2, 4 filled from the running grid below -->

# w8 completion — Items 1/2/4 (results-analyst, appended 2026-07-08)

Status: **DONE — full grid complete.** All 198 jobs finished (0 failures, 11834 s wall).
`runs/*.json` = 198 cells: corr=0 capacity axis (paired, n=8), full epoch-scaling frontier
(16 cells × 3 seeds), **and** both stress axes — correlation ρ∈{0.5,0.9} and eval_noise
σ∈{0.3,0.6}, 3 caps × 5 seeds × {500,2000} ep. (An earlier revision of this section marked the
stress columns pending; they are now complete and are reported below — they **materially qualify**
the headline: the accuracy reversal is specific to clean/correlated-key retrieval at kv≤64 and
**does not survive cue noise**.)

Repo read-only. All artifacts under gitignored `.claude/`. JAX was **warm** (cached from the
Item-3 session): first cell 108 s, no 20-min cold start. Laptop CPU, f32.

**Flag-provenance — `train_epochs` is THE arm (same table as §0; only the swept fields repeated):**

| flag | value |
|---|---|
| commit | `63fea62` (main, w6-integrated) · JAX 0.9.0 (main venv, `/Users/user/Desktop/CHLU/.venv`) |
| driver | `.claude/scratch/regime-remap-2000ep/driver.py` → `V1._regime_cell`+`_regime_metrics` VERBATIM |
| batch | `batch.py jobs_w8.txt` (one warm process, idempotent, log `batch_w8.log`) |
| **train_epochs** | **{500, 1000, 2000, 4000}** (the swept arm) |
| kinetic_energy_mode | relativistic · potential_type mlp · rest_mass/c 1.0/1.0 |
| embed_dim 16 (CLU dim 32) · hidden_dim 128 · regime_n_clusters 8 | vocab 256 (**512** for kv128) |
| calib_p_exit 0.5 (learned-τ gate) · cost ladder 300/1200/2100/3000 Verlet · hopfield_beta 20.0 |
| **seeds** | Item1 corr=0: {42,43,44,45,46} ne5 **+** {42,43,44} ne3 = **n=8 pooled**; frontier: {42,43,44} ne3 (n=3) |
| episodes/cell | 5 (ne5) or 3 (ne3), pooled within (cell,seed) |
| langevin_noise | legacy (FDT-violating default; baseline continuity) |

`savings` = full-budget CLU cost / gated CLU cost = **intra-CLU compute-rationing** (see the P8-reverse
caveat, restated in §Item-4). It is **NOT** savings vs Hopfield.

---

## Item 1 — corr=0 capacity axis, paired 500 vs 2000 ep (COMPLETE, n=8 pooled seeds)

Data: `runs/correlation_N*_kv*_s0.0_*_ep{500,2000}_ne{3,5}_*.json`. Mean±std over 8 pooled seeds.
Figure: `.claude/outputs/regime-remap-2000ep/fig_regime_map.png`.

| cell (N,kv) | ep | n | CLU fidelity | gate acc | full-budget acc | Hopfield acc | Δ(gate−hop) | intra-CLU savings |
|---|---|---|---|---|---|---|---|---|
| N128/kv32 | 500  | 8 | 0.76±0.09 | 0.31±0.04 | 0.30±0.02 | 0.98±0.01 | **−0.67** | 1.2× |
| N128/kv32 | 2000 | 8 | **1.00±0.00** | **1.00±0.00** | 1.00±0.00 | 0.98±0.01 | **+0.02** | 9.9× |
| N256/kv64 | 500  | 8 | 0.43±0.05 | 0.06±0.02 | 0.06±0.02 | 0.97±0.01 | **−0.91** | 1.0× |
| N256/kv64 | 2000 | 8 | **1.00±0.00** | **0.99±0.00** | 0.99±0.00 | 0.97±0.01 | **+0.02** | 9.5× |
| N384/kv96 | 500  | 8 | 0.40±0.03 | 0.02±0.01 | 0.02±0.01 | 0.95±0.01 | **−0.93** | 1.0× |
| N384/kv96 | 2000 | 8 | 0.97±0.01 | 0.91±0.02 | 0.91±0.02 | 0.95±0.01 | **−0.04** | 6.2× |

**Reading.** At the 500-ep baseline every cell is a Hopfield blowout (gate 0.02–0.31 vs Hopfield
0.95–0.98) — the "Hopfield-dominant" v1-hopfield-stress map was **under-trained**, confirmed. At
2000 ep the CLU **storage fidelity** jumps to ≈1.0 (kv32/kv64 exactly; kv96 0.97) and gated accuracy
tracks it. **kv32 and kv64 close AND reverse the accuracy gap** (Δ +0.02 each, gate 0.99–1.00 ≥
Hopfield). **kv96 does NOT close at 2000 ep** (gate 0.91 vs Hopfield 0.95, Δ −0.04) — the residual
gap is a fidelity deficit (0.97<1.00, i.e. ~3% of stored patterns still mis-relax). So at a fixed
2000-ep budget: **2 of 3 capacity points close; kv96 remains under-trained** (see Item 2 — it closes
at 4000 ep).

---

## Item 2 — epoch-scaling frontier {500,1000,2000,4000} × kv{32,64,96,128} (COMPLETE, n=3, ne3)

Data: `runs/correlation_N*_kv*_s0.0_seed4{2,3,4}_ep{500,1000,2000,4000}_ne3_*.json`.
Figures: `fig_frontier.png`, **`fig_frontier_clean.png`** (gate acc + fidelity vs epochs, Hopfield band shaded).
Hopfield band across these cells: **0.947–0.976** (epoch-independent, 1 matvec — Item 3).

| kv (N,vocab) | metric | 500 ep | 1000 ep | 2000 ep | 4000 ep | best ep (Δ vs Hop) |
|---|---|---|---|---|---|---|
| kv32 (N128,v256) | fidelity | 0.74±0.10 | 1.00±0.01 | 1.00±0.00 | 0.99±0.01 | |
| kv32 | **gate acc** | 0.30±0.04 | 0.98±0.02 | **1.00±0.01** | 0.89±0.05 | **2000 (+0.02)** |
| kv64 (N256,v256) | fidelity | 0.41±0.02 | 0.82±0.04 | 1.00±0.00 | 1.00±0.00 | |
| kv64 | **gate acc** | 0.06±0.02 | 0.42±0.02 | **0.99±0.01** | 0.97±0.01 | **2000 (+0.02)** |
| kv96 (N384,v256) | fidelity | 0.39±0.02 | 0.24±0.06 | 0.97±0.01 | 1.00±0.00 | |
| kv96 | **gate acc** | 0.02±0.00 | 0.11±0.01 | 0.92±0.02 | **0.975±0.00** | **4000 (+0.03)** |
| kv128 (N384,v512) | fidelity | 0.40±0.02 | 0.09±0.01 | 0.71±0.06 | 1.00±0.00 | |
| kv128 | **gate acc** | 0.01±0.00 | 0.04±0.00 | 0.60±0.06 | **0.95±0.01** | **4000 (+0.004, tie)** |

**Does a capacity wall reappear beyond kv64? — At a FIXED budget, yes; as a hard capacity limit, no.**
1. **Fixed 2000-ep budget: a wall is visible** — gate acc falls 0.99 (kv64) → 0.92 (kv96) → 0.60 (kv128).
   Storage fidelity at 2000 ep falls 1.00 → 0.97 → 0.71. So at the budget the paper's regime map used,
   capacity beyond kv64 is under-served.
2. **The wall is an EPOCH-BUDGET wall, not a capacity wall.** At 4000 ep **all four** kv reach
   fidelity ≈ 1.00 and gate acc ≥ (or ≈) the Hopfield band: kv96 → 0.975 (+0.03), kv128 → 0.95 (tie,
   Δ +0.004). **Required epochs scale with kv:** kv32 saturates by ~1000, kv64 by ~2000, kv96 by ~4000,
   kv128 by ~4000. The compute–fidelity frontier is a **diagonal ridge**, not a ceiling.
3. **Over-training is real and cheap to hit (negative, Item 4).** kv32 gate acc **drops 1.00→0.89 from
   2000→4000 ep** — below its own Hopfield (0.976). Small memories over-train: past their saturation
   epoch the EBM degrades. So there is **no single epoch at which all four cells simultaneously beat
   Hopfield** — each capacity has its own optimum.
4. **Non-monotone fidelity dip at intermediate epochs (negative).** kv96 fidelity goes 0.39(500)→
   **0.24(1000)**→0.97(2000), and kv128 0.40(500)→**0.09(1000)**→0.71(2000). Fidelity gets *worse*
   before it gets better — a transient PCD/buffer instability at intermediate epochs for large kv.
   Do not read the 1000-ep point as monotone progress.

---

## Item 1 (cont.) — STRESS columns, paired 500 vs 2000 ep (COMPLETE, n=5 seeds ne5/ne3)

`savings`=intra-CLU. Δ = gate − Hopfield; **CLOSE** = Δ ≥ −0.01. Data: `runs/{correlation,eval_noise}_*`.

**Correlation-ρ axis** (2000 ep rows; 500-ep rows all Δ ≪ 0, omitted for brevity — see tables.md):

| cell | ρ | fid | gate acc | Hop acc | Δ | savings | closes? |
|---|---|---|---|---|---|---|---|
| N128/kv32 | 0.5 | 1.00 | 1.00±0.00 | 0.98 | +0.02 | 10.0× | ✅ |
| N256/kv64 | 0.5 | 1.00 | 0.99±0.01 | 0.97 | +0.02 | 9.3× | ✅ |
| N384/kv96 | 0.5 | 0.97 | 0.90±0.01 | 0.95 | −0.05 | 5.5× | ❌ |
| N128/kv32 | 0.9 | 0.97 | 0.87±0.06 | **0.72** | **+0.16** | 5.6× | ✅ |
| N256/kv64 | 0.9 | 0.89 | 0.67±0.07 | **0.59** | +0.08 | 2.6× | ✅ |
| N384/kv96 | 0.9 | 0.68 | 0.36±0.05 | 0.52 | −0.16 | 1.3× | ❌ |

*Reading:* ρ=0.5 is identical to ρ=0 (kv32/kv64 reverse, kv96 fails). At **ρ=0.9 Hopfield itself
collapses** (0.72/0.59/0.52 — softmax attention degrades on strongly-correlated keys), so the CLU
reversal at kv32/kv64 **widens** (Δ +0.16/+0.08) — but this is Hopfield falling, not the CLU rising
(CLU gate also drops to 0.87/0.67). kv96 still fails (CLU fidelity itself collapses to 0.68).

**Eval-noise σ axis** (2000 ep rows) — **THE clean negative:**

| cell | σ | fid | gate acc | Hop acc | Δ | savings | closes? |
|---|---|---|---|---|---|---|---|
| N128/kv32 | 0.3 | **1.00** | 0.90±0.03 | 0.94 | −0.05 | 4.6× | ❌ |
| N256/kv64 | 0.3 | **1.00** | 0.81±0.03 | 0.89 | −0.08 | 3.4× | ❌ |
| N384/kv96 | 0.3 | 0.97 | 0.62±0.04 | 0.85 | −0.23 | 2.2× | ❌ |
| N128/kv32 | 0.6 | **1.00** | 0.36±0.06 | 0.71 | −0.35 | 1.3× | ❌ |
| N256/kv64 | 0.6 | **1.00** | 0.29±0.04 | 0.57 | −0.28 | 1.3× | ❌ |
| N384/kv96 | 0.6 | 0.97 | 0.19±0.01 | 0.52 | −0.33 | 1.3× | ❌ |

*Reading:* **under noisy cues NO cell closes, at any capacity, even kv32.** CLU storage fidelity is
still ≈1.0 (the patterns are stored) — but the **gated relaxation is far less robust to cue noise than
Hopfield's single matvec** (gate 0.36 vs Hopfield 0.71 at σ=0.6/kv32). The 2000-ep accuracy reversal
is a **clean-cue phenomenon**; it inverts once the cue is corrupted. This is the sharpest negative in
the study and it is the axis most relevant to real retrieval.

**Tally at 2000 ep (15 non-frontier cells):** correlation axis **6/9 close** (all kv≤64: ρ∈{0,0.5,0.9});
eval-noise axis **0/6 close**. Overall **6/15**. Adding the frontier: kv96 also closes on the *clean*
axis at 4000 ep, kv128 ties at 4000 ep.

---

## Item 4 — negatives (per C-9), and the P8-in-reverse cost caveat restated

**Cells / regimes that do NOT close (with the wall location):**
- **ENTIRE eval-noise axis (0/6)** — under cue noise σ∈{0.3,0.6}, no cell closes at any capacity, even
  kv32 (Δ −0.05 at σ=0.3, −0.35 at σ=0.6). CLU fidelity ≈1.0 but the gated relaxation is markedly less
  robust to cue corruption than Hopfield's single matvec. **This is the dominant negative** and the axis
  most relevant to real retrieval.
- **kv96 at 2000 ep, all correlations** — gate 0.90–0.91 < Hopfield 0.95 (Δ −0.04…−0.05 at ρ≤0.5; −0.16 at ρ=0.9 where CLU fidelity collapses to 0.68). Closes only at 4000 ep (clean axis).
- **kv128 at ≤2000 ep** — gate 0.60 at 2000 ep, ≪ Hopfield 0.95. Reaches only a **tie** (Δ +0.004) at 4000 ep, not a clear reversal.
- **kv32 at 4000 ep** — **over-trained**: gate 0.89 < its Hopfield 0.976 (Δ −0.09). Over-shooting the saturation epoch re-opens the gap.
- **Intermediate-epoch fidelity collapse** at kv96/kv128 (1000 ep dip to 0.24 / 0.09) — training is non-monotone for large kv.
- **Wall locations:** (a) capacity/epoch wall — the clean-cue reversal holds for **kv ≤ 64 at 2000 ep**;
  kv ∈ {96,128} needs 4000 ep; no *hard* capacity wall within kv∈[32,128]. (b) **Noise wall — a hard one:**
  any cue noise removes the reversal at *every* capacity (Hopfield wins). (c) ρ=0.9 hurts both retrievers
  but Hopfield more, so the reversal *widens* at kv≤64 (an artifact of Hopfield's fragility, not CLU strength).

**Cost caveat restated verbatim from Item 3 (the P8-in-reverse discipline — attach to EVERY accuracy-improvement claim above):**
> *At matched accuracy, Hopfield is the cheaper retriever.* Hopfield attains ≥0.947 accuracy in **~1 matvec** (O(kv·d)); extra iteration budget buys it ≤0.003 (Item 3, complete). The "6–10× savings" figures in the tables above are **CLU-gate cost vs CLU-full-budget cost (3000 Verlet steps)** — an **intra-CLU compute-rationing** number, NOT savings relative to Hopfield. "kv32/kv64 close and reverse the gap" and "kv96 closes at 4000 ep" mean the CLU gate's **accuracy reaches/exceeds Hopfield's accuracy**, at a per-query cost (≥300 Verlet steps, each a potential-MLP forward+grad) that remains **far above** Hopfield's single matvec. The CLU's asset here is escalatable accuracy under a rationing gate, not cheaper retrieval.

---

## CM-8 accuracy-story replacement (one paragraph for the Hub to splice into `claims_matrix.md`)

> **CM-8 accuracy (SETTLED, full grid, w8):** the "Hopfield-dominant 26/26" regime map was **an
> under-training artifact** (500 ep: CLU gate 0.02–0.31 vs Hopfield 0.95–0.98). Trained to convergence
> (2000 ep) CLU storage fidelity → ≈1.0, and the accuracy reversal appears **but is regime-specific,
> not general.** On **clean or correlated-key** cues, gated-CLU accuracy **reverses** Hopfield for
> **kv ≤ 64** (gate 0.99–1.00 vs 0.97–0.98, Δ +0.02, n=8; holds at ρ=0.5, and *widens* to Δ +0.08…+0.16
> at ρ=0.9 — but only because Hopfield collapses on correlated keys, 0.72/0.59, not because CLU rises).
> **kv96 needs 4000 ep** to reverse (0.975 vs 0.947, clean axis); **kv128 only ties** at 4000 ep. The
> capacity wall beyond kv64 is an **epoch-budget wall, not a hard capacity limit** (required epochs scale
> with kv), and small cells **over-train** (kv32 gate falls 1.00→0.89 from 2000→4000 ep). **The reversal
> does NOT survive cue noise:** at eval-noise σ∈{0.3,0.6} **no cell closes at any capacity** (gate 0.36
> vs Hopfield 0.71 at σ=0.6/kv32) — the CLU relaxation gate is markedly less noise-robust than Hopfield's
> single matvec, despite CLU fidelity ≈1.0. **Tally at 2000 ep: 6/15 cells close** (all correlation-axis,
> kv≤64); eval-noise axis 0/6. **Cost is unchanged and controlling:** Hopfield remains the cheaper AND
> more noise-robust retriever (≥0.947 in ~1 matvec); the CLU "6–10× savings" is intra-CLU (gate vs
> full-budget), not vs Hopfield. Net: *the CLU gate reaches/exceeds Hopfield accuracy only for
> clean/correlated cues at kv≤64 (kv-scaled epochs extend this on the clean axis); Hopfield keeps both
> the cost and the cue-noise-robustness advantage. The CLU asset is escalatable accuracy under a
> rationing gate on clean retrieval, not a general accuracy or cost win.*

---

## Limitations / confounds
- **Stress columns now COMPLETE** (correlation ρ∈{0.5,0.9}, eval_noise σ∈{0.3,0.6}, n=5). The reversal is
  established as **clean/correlated-cue, kv≤64 only**; it inverts under cue noise (0/6). The ρ=0.9 "win"
  is a Hopfield-fragility artifact (both retrievers degrade; Hopfield more), not a CLU gain — state it as such.
- **Frontier n=3** (ne3, seeds 42–44); Item-1 corr=0 is n=8, stress n=5. Over-train and dip effects are
  seen in all 3 seeds but std is non-trivial (kv32@4000 gate 0.89±0.05; ρ=0.9 gate ±0.06–0.07).
- **CPU / small-D / laptop / f32 / MQAR vocab-256(512).** Same scope caveats as all V1 regime work.
- **`langevin_noise=legacy`** (FDT-violating default) retained for baseline continuity — an FDT-correct
  sampler could shift the epoch-to-close curve.
- **Hopfield band 0.947–0.976 is not perfect** — the reversals at kv96/kv128 (Δ +0.03 / +0.004) are
  small relative to seed spread; "parity" is the honest word for kv128, "reversal" only for kv96.

## Recommended next experiments
1. **Diagnose the noise-robustness gap** (the dominant negative): why does the CLU relaxation gate lose
   0.35 accuracy to Hopfield under σ=0.6 cue noise when storage fidelity is 1.0? Likely the governed
   relaxation over-commits to the noisy cue; test a noise-aware τ / longer relax budget / denoising init.
   This is the axis that most threatens the V1 accuracy narrative — settle it before drafting.
2. Add **kv160/kv192 at {4000,8000} ep** to find where the epoch-budget genuinely diverges (or a real
   capacity wall finally appears at fixed CLU dim 32).
3. **Early-stopping on write-time fidelity** — since small cells over-train, a fidelity-plateau stop
   would both save compute and avoid the kv32@4000 regression; quantify the compute saved.
4. Sweep **CLU dim** (embed_dim) at fixed kv to separate "needs more epochs" from "needs more capacity".

## Proposed handover updates (for the Hub)
- **§1.6 / CM-8 accuracy:** splice the CM-8 replacement paragraph above. Concrete numbers to fold in:
  500-ep baseline gate 0.02–0.31 vs Hopfield 0.95–0.98 (under-training confirmed); 2000-ep clean-axis
  gate kv32 1.00 / kv64 0.99 / kv96 0.91 vs Hopfield 0.98/0.97/0.95 (2/3 close+reverse); 4000-ep kv96
  0.975 (+0.03) and kv128 0.95 (tie) — **epoch-budget wall, not capacity wall**; kv32 over-trains
  1.00→0.89. **Two hard qualifiers that MUST travel with the reversal:** (1) **eval-noise 0/6 — the
  reversal dies under cue noise** (gate 0.36 vs Hopfield 0.71 @σ=0.6/kv32) despite fidelity 1.0;
  (2) **ρ=0.9 "win" is a Hopfield-collapse artifact** (Hopfield 0.72/0.59, CLU also drops). Tally:
  **6/15 cells close at 2000 ep, all correlation-axis kv≤64.**
- **Cost framing UNCHANGED** (Item 3, final): Hopfield ≥0.947 in ~1 matvec; "6–10× savings" is intra-CLU.
  **New parity point: Hopfield is also the more cue-noise-robust retriever.**
- **§5 provenance:** driver + `jobs_w8.txt` + `runs/*.json` (**198 cells, complete, 0 failures, 11834 s**)
  under `.claude/scratch/regime-remap-2000ep/`; figures under `.claude/outputs/regime-remap-2000ep/`.
- **No code bugs hit.** `_regime_cell`/`_regime_metrics` behaved; `savings` field = full/gate cost as documented.
- **Flag for experiment-engineer (not a bug, a design note):** the non-monotone fidelity dip at 1000 ep
  for kv≥96 suggests PCD buffer instability at intermediate epochs — worth a look if a smoother
  training curve is wanted, but it self-heals by 2000 ep so not blocking.
