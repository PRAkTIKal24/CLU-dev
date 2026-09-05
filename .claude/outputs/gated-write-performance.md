# gated-write-performance — experiment-engineer report

**Task + acceptance criterion:** run the `write_mode ∈ {linear, gated}` fix properly across all
three families (adding T=128, parity T=64, MQAR kv=4 T=128; 3 seeds, symmetric monotone LR-rescue,
matched 40k block params, γ=0 for MQAR) to ask *is gated CLU now competitive*, then search for a
physics **edge** vs a matched, equally-gated GRU/SSM (3a extrapolation, 3b capacity, 3c robustness),
with honest cost. Acceptance: the corrected three-family table + the levelling caveat stated in
those words + the edge search as performance curves vs matched gated baselines + cost; ≥3 seeds;
fairness category of every knob stated; tests green.

**Status: done.** All items run at the published-numbers budget (1200/400), 3 seeds, symmetric rescue
on every variant, matched 40k params. Item 1 (18 cells) + 3b capacity (20 cells) + cost from one run
(`gated_write_core.json`); 3a extrapolation + 3c robustness from a second (`gated_write_edges.json`).
Full suite **570 passed**. Total harness wall-clock ~2h20m on a contended machine.

**One-line answer: the gate gets CLU *off the floor and to GRU-level* but beats no baseline on any
family (levelling, not beating); of the three edge-candidates, only the capacity axis survives — gated
CLU at γ=0 beats a matched, rescued, gated GRU/SSM at kv≥4 — while extrapolation and robustness are
both clean falsifications with CLU the *worse* primitive.**

> ⚠ **LEVELLING, NOT BEATING (Item 2, required verbatim).** The gate imports a capability every
> baseline already has (GRU gates, selective-SSM input-dependent Δ, softmax QK). Adding it to CLU is
> **levelling, not beating.** Item 1 answers only *"is CLU now competitive"*, never *"does CLU win"*.
> A gated CLU that merely ties the GRU is **not** a result the program needs. The gate is
> **category-(b)-adjacent**: it is a CLU-internal knob (lives in `CLUBlock`, touches no shared slot),
> but the *capability* it adds is one the baselines have — argued explicitly, not assumed.

> **⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary):**
> 1. **The `primitive-harness §1b` "19× capacity gap" is inflated by an un-rescued GRU.** With the GRU
>    rescued as hard as the CLU, its kv=16 accuracy recovers 0.008→**0.085**, so the honest gated-CLU
>    margin is **2.1× (0.180 vs 0.085), not 19×.** Every site quoting the 19×/17×-chance figure needs
>    the rescued-baseline number (§3).
> 2. **The founding-CHLU "symplectic ⇒ stable long-horizon extrapolation" intuition is FALSIFIED in
>    the trained, driven primitive slot** — the block is the *worst* of three extrapolators, blowing
>    up past the trained horizon (§2). Any "CLU extrapolates" claim must be scoped to the
>    autonomous/designed setting (Exp A), not the trained sequence primitive.
> 3. **Prop-2 barrier confinement gives no measured input-noise robustness in the trained block** (§4),
>    reinforcing `primitive-harness` reconciliation #1 (designed-landscape property, does not transfer).

---

## Flag-provenance table

| item | value |
|---|---|
| commit | `d0b3f4a` (code; branch `agent/experiment-engineer/gated-write-performance`, base local `main` @ `8519df6`). Result JSONs in `.claude/outputs/` (not committed, protocol §2). |
| seeds | 3 per cell: `42, 1042, 2042` (`cfg_seed(i)=1000i+42`) — identical to primitive-harness/gamma-read |
| budget | **published-numbers budget**: `train_steps=1200`, `tune_steps=400`, LR grid `{3e-4,1e-3,3e-3}`, batch 32, eval batch 256, grad-clip 1.0, block-param budget **40 000** — so the re-run baselines double as a reproduction check of primitive-harness |
| shared slot | `d_model=64`, `n_layers=2`; embedding + learned pos-embed + residual/LayerNorm + head byte-identical across primitives |
| CLU physics fixed | `kinetic_mode=newtonian_learned`, `potential_type=mlp` (hidden 32, incl. 0.05‖q‖²), `dt=0.1`, `clu_steps=1`, `read_mode=endpoint` |
| CLU knobs varied | `write_mode ∈ {linear, gated}`; `gamma`: **0.0 for MQAR** (gamma-read §1: +0.040, monotone), **0.05 (shipped) for adding/parity** — not tuned per family |
| rescue | **symmetric monotone LR-rescue applied to EVERY variant** (baselines and both CLU rows) — re-run the two non-selected LRs at full length, adopt only if it wins on the 3-seed mean |
| langevin/temperature/sleep | **N/A** — deterministic Verlet, no Langevin, no thermostat, no sleep |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** — main venv reused per §4 (worktree, no `uv sync`) |
| machine | shared; load noted in the cost table |

**Fairness category of every knob touched (task's absolute rule):**
- `write_mode` (linear→gated): **category (b)-adjacent** — a CLU-internal knob that imports a
  capability the baselines already have. Stated as levelling, above.
- `gamma` (0 for MQAR): **category (a)** — a knob no other primitive has, lives in `CLUBlock`.
- The shared slot, data, optimiser, LR grid, step budget, seeds, and the rescue pass are **identical
  for every primitive**; no baseline was disadvantaged or under-tuned.

---

## 1. Item 1 — the corrected three-family table

Six variants per family, matched **40 000 block params**, symmetric monotone LR-rescue on **every**
variant, 3 seeds. **The baselines and the linear CLU reproduce the published numbers** (built-in
validity check): GRU adding 0.0007 / MQAR 0.4863 / parity 1.000; attention MQAR 0.9945 / adding
0.0001; **clu_linear MQAR 0.3864, parity 0.5380, adding 0.1825 all match primitive-harness /
gamma-read to ≤1 in the 4th dp.**

**adding T=128** (MSE↓):

| variant | MSE | best_lr | rescued | γ |
|---|---|---|---|---|
| mlp (no-mixing control) | 0.1825 ± 0.0009 | 3e-4 | ✓ | — |
| gru | **0.0007 ± 0.0003** | 3e-3 | ✓ | — |
| ssm | 0.0008 ± 0.0010 | 3e-3 | — | — |
| attention | **0.0001 ± 0.0000** | 3e-4 | ✓ | — |
| clu_linear (shipped) | 0.1825 ± 0.0034 | 3e-3 | ✓ | 0.05 |
| **clu_gated** | **0.0023 ± 0.0009** | 1e-3 | ✓ | 0.05 |

**parity T=64** (accuracy↑):

| variant | accuracy | best_lr | rescued | γ |
|---|---|---|---|---|
| mlp | 0.5109 ± 0.0041 | 1e-3 | ✓ | — |
| **gru** | **1.0000 ± 0.0000** | 3e-4 | — | — |
| ssm | 0.6379 ± 0.0134 | 3e-3 | — | — |
| attention | 0.5474 ± 0.0033 | 3e-3 | — | — |
| clu_linear (shipped) | 0.5380 ± 0.0058 | 1e-3 | ✓ | 0.05 |
| **clu_gated** | **0.5774 ± 0.0071** | 3e-3 | — | 0.05 |

**MQAR T=128 kv=4** (accuracy↑, γ=0 for CLU):

| variant | accuracy | best_lr | rescued | γ |
|---|---|---|---|---|
| mlp | 0.0124 ± 0.0044 | 1e-3 | ✓ | — |
| gru | 0.4863 ± 0.0191 | 3e-3 | — | — |
| ssm | 0.0120 ± 0.0039 | 3e-3 | — | — |
| **attention** | **0.9945 ± 0.0020** | 3e-3 | — | — |
| clu_linear | 0.3864 ± 0.0032 | 3e-3 | — | 0.0 |
| **clu_gated** | **0.4993 ± 0.0036** | 3e-3 | — | 0.0 |

**Reading it — the gate levels, it does not beat (Item 2 confirmed).**
- **adding: off the floor by 79×** (linear 0.1825 → gated 0.0023) — the gate supplies the value×marker
  conjunction, exactly the `gamma-read-sweep §5` diagnosis. But gated CLU is **3.3× worse than the
  GRU** and **23× worse than attention**. Competitive; not winning.
- **parity: the gate does NOT solve it** (0.5380 → 0.5774, barely off chance) — as pre-registered, a
  multiplicative *write* supplies a conjunction, not the running XOR *state* parity needs. GRU=1.000.
- **MQAR: the gate ties the GRU** (0.3864 → 0.4993 vs GRU 0.4863, +0.013, ~0.7σ) but stays far below
  attention (0.9945). A tie with the GRU is, per the task, **not a result the program needs.**

**Item-1 verdict: gated CLU wins 0 of 3 families on absolute performance.** It becomes a
*competitive mid-tier recurrence* — off the floor on adding, GRU-level on MQAR — but beats no
baseline on any family. **This is levelling, not beating**, exactly as the fairness statement (Item 2)
requires it be called.

## 2. Item 3a — long-horizon extrapolation (train at T=64, test at 2T, 4T)  ⛔ EDGE FALSIFIED

**adding** (MSE↓), matched gated GRU/SSM, 3 seeds:

| variant | T=64 (in-dist) | T=128 (2×) | T=256 (4×) | drop 64→256 |
|---|---|---|---|---|
| **clu_gated** (γ=0) | **0.0020 ± 0.0011** | **2.0143 ± 1.8836** | **2.9571 ± 1.0544** | **+2.955** |
| gru | 0.0002 ± 0.0001 | 0.0239 ± 0.0097 | 0.1030 ± 0.0270 | +0.103 |
| ssm | 0.0007 ± 0.0006 | 0.0141 ± 0.0115 | 0.0174 ± 0.0089 | +0.017 |

**parity** (acc↑):

| variant | T=64 | T=128 | T=256 | drop 64→256 |
|---|---|---|---|---|
| clu_gated (γ=0) | 0.5590 ± 0.0116 | 0.5282 ± 0.0054 | 0.5153 ± 0.0010 | −0.044 |
| **gru** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | **+0.000** |
| ssm | 0.6328 ± 0.0110 | 0.5658 ± 0.0067 | 0.5323 ± 0.0043 | −0.101 |

**The founding-CHLU extrapolation edge is FALSIFIED, and hard.** On adding, gated CLU **solves the
task in-distribution** (T=64 MSE 0.0020 — competitive!) but its driven-Verlet rollout **blows up
on unseen lengths**: MSE 2.01 at 2× and 2.96 at 4× (both *worse than predicting the target mean*,
MSE≈0.18 — the state diverges past the trained horizon). The GRU degrades to 0.024/0.103 and the
SSM barely moves (0.014/0.017). On parity the **GRU extrapolates perfectly to 4× (1.000)** while
gated CLU sits at chance at every length. **CLU is the *worst* extrapolator of the three on both
families — the opposite of my pre-registered "candidate edge".** Figure: `extrapolation.png`.

*(Mechanism, offered not proven: the block trains the write/read maps and the learned `V_θ` only on
statistics of length-T rollouts; a longer driven rollout accumulates more momentum impulses and
integration steps than anything seen in training, and with no energy budget on the input current the
symplectic map has no reason to stay bounded off-distribution. This is a concrete, testable failure
of the "symplectic ⇒ stable extrapolation" intuition in the *trained, driven* slot.)*

## 3. Item 3b — capacity under item load (MQAR kv-sweep, gate + γ=0)  ⭐ THE ONE SURVIVING EDGE

MQAR T=128, accuracy↑, matched 40k params, **symmetric rescue applied to every variant** (so the
GRU is tuned as hard as the CLU — this is stricter than `primitive-harness §1b`, whose capacity axis
did **not** rescue the baselines):

| variant | kv=2 | kv=4 | kv=8 | kv=16 |
|---|---|---|---|---|
| gru | **0.9967 ± 0.0033** | 0.4863 ± 0.0191 | 0.1772 ± 0.0091 | 0.0850 ± 0.0175 |
| ssm | 0.0124 ± 0.0072 | 0.0120 ± 0.0039 | 0.0088 ± 0.0008 | 0.0212 ± 0.0180 |
| **attention** | 0.8874 ± 0.1524 | **0.9945 ± 0.0020** | **0.9382 ± 0.0820** | **0.8561 ± 0.1701** |
| clu_linear (γ=0) | 0.5723 ± 0.0120 | 0.3864 ± 0.0032 | 0.2386 ± 0.0024 | 0.1475 ± 0.0025 |
| **clu_gated (γ=0)** | 0.7018 ± 0.0101 | **0.4993 ± 0.0036** | **0.3034 ± 0.0061** | **0.1795 ± 0.0008** |

**clu_gated − GRU:** kv2 **−0.295**, kv4 **+0.013**, kv8 **+0.126**, kv16 **+0.095**.
**clu_linear − GRU:** kv2 −0.425, kv4 −0.100, kv8 **+0.061**, kv16 **+0.063**.

**⭐ The capacity edge survives matched gating AND matched tuning — this is the one axis where CLU's
physics beats the GRU/SSM.**
- The GRU is a *better* memory when there are few items (kv=2: 0.997 vs 0.702) and **collapses** as
  items accumulate (0.997 → 0.486 → 0.177 → 0.085). Gated CLU degrades far more gracefully
  (0.702 → 0.499 → 0.303 → 0.180) and **crosses above the GRU at kv=4 and beats it by 1.7× at kv=8,
  2.1× at kv=16.** The SSM is at the floor throughout.
- **The gate moves the crossover *earlier*** (kv≈8 for linear → kv≈4 for gated) and lifts the whole
  curve (+0.11–0.13 at every kv), but the graceful-capacity property **pre-exists the gate**:
  `clu_linear` *also* beats the GRU at kv≥8. So the edge is a **CLU architectural property (how a
  fixed state is used), not a gate artifact** — the gate levels absolute performance; the physics
  supplies the flat capacity curve, and it survives gating.
- **Two honest caveats that keep this from being oversold.** (a) **Attention beats everyone at every
  kv** (0.86–0.99) — an O(T) KV cache has no capacity ceiling at these scales, so this is a
  GRU/SSM-beating edge, not a state-of-the-art one. (b) **The margin is smaller than
  `primitive-harness §1b` reported**, because that measurement left the GRU *un-rescued* (kv=16 GRU
  0.008 = chance); with the GRU rescued as hard as the CLU it recovers to 0.085, so the honest kv=16
  gap is CLU 0.180 vs GRU 0.085 (2.1×), not the 19× previously quoted. **This is a downstream
  reconciliation item.** Figure: `capacity_crossover.png`.

## 4. Item 3c — robustness to input noise at inference  ⛔ EDGE FALSIFIED

Adding T=128, Gaussian noise added to the **value channel only** (markers intact) at inference; MSE
(↓), 3 seeds, matched gated GRU/SSM:

| variant | σ=0.0 | σ=0.05 | σ=0.1 | σ=0.2 | σ=0.4 | slope 0→0.4 |
|---|---|---|---|---|---|---|
| **clu_gated** (γ=0.05) | 0.0151 | 0.0189 | 0.0308 | 0.0862 | 0.4213 | **+0.406** |
| gru | 0.0008 | 0.0061 | 0.0213 | 0.0760 | 0.2432 | +0.242 |
| ssm | 0.0008 | 0.0058 | 0.0212 | 0.0813 | 0.2832 | +0.282 |

**The robustness edge is FALSIFIED.** Gated CLU is worse at **every** noise level *and* degrades on a
**steeper** slope (+0.406 vs GRU +0.242). Barrier confinement (Prop 2) produces **no measured
input-noise robustness** in the trained, gated block — consistent with `primitive-harness`
reconciliation #1 (Prop-2 isolation is a designed-landscape property that does not transfer to a
learned block). Note gated CLU's clean-input MSE here (0.0151) is ~19× the GRU's (0.0008): it is off
the no-mixing floor but not competitive on absolute error.

## 5. Item 4 — cost (gated CLU vs baselines)

Round-robin interleaved (5 rounds × 25 steps, median), MQAR kv=4 T=128, matched 40k params. Ratios
are contention-robust (`primitive-harness §4`); FLOPs are exact/load-independent.

| primitive | ms/step | ×GRU wall | fwd FLOPs | ×GRU FLOPs |
|---|---|---|---|---|
| gru | 27.1 | 1.00× | 6.06 M | 1.00× |
| ssm | 19.2 | 0.71× | 14.6 M | 2.41× |
| attention | 25.0 | 0.92× | 26.2 M | 4.33× |
| clu_linear | 41.3 | 1.52× | 12.6 M | 2.09× |
| **clu_gated** | **40.2** | **1.48×** | **13.1 M** | **2.16×** |

**The honest multiple: gated CLU costs 1.48× the GRU wall-clock and 2.16× its FLOPs** (the extra
`w_gate` adds ~4% FLOPs over linear CLU, and is if anything slightly *faster* in wall-clock here —
within noise). Consistent with `primitive-harness`'s 1.41× GRU / 0.48× attention. Per the Head, cost
is stated, not competed on; the one surviving edge (capacity) does **not** cost 10× — it is ~1.5×.

## 6. Pre-registration scorecard (`PREREG.md`, written before any full-length run)

| # | pre-registered call | measured | verdict |
|---|---|---|---|
| I1-adding | gated CLU gets off the floor to 0.003–0.03, **LOSE** to GRU/SSM/attn | **0.0023**, off floor 79×, loses (3.3× worse than GRU) | ✅ (marginally below range, LOSE confirmed) |
| I1-parity | gate ≠ XOR; 0.53–0.75, **LOSE** to GRU | **0.5774** (barely off chance), GRU 1.000 | ✅ |
| I1-MQAR | gate helps binding 0.45–0.70; **LOSE to attn, TIE/near GRU** | **0.4993**, ties GRU 0.486, loses attn 0.995 | ✅ |
| I1-global | gated CLU **wins 0 of 3 on absolute**; levelling not beating | 0 of 3; ties GRU on MQAR only | ✅ |
| 3a-extrap | **CANDIDATE EDGE, tie-to-win** on relative drop | ⛔ **FALSIFIED** — CLU is the *worst* extrapolator (MSE 2.01 at 2× vs GRU 0.024); parity GRU perfect | ❌ **falsified, hard** |
| 3b-capacity | ⭐ **STRONGEST EDGE, WIN vs GRU at kv≥8**, survives gate, crossover moves toward CLU | **CONFIRMED** — clu_gated beats GRU at kv≥4 (+0.013/+0.126/+0.095), crossover kv8→kv4 | ✅ ⭐ |
| 3c-robustness | weak edge / tie | ⛔ **FALSIFIED** — CLU worse at every σ *and* steeper slope | ❌ |
| headline | the capacity edge is the one I expect to survive | it is the **only** edge that survived | ✅ |

**Scorecard: 5 clean confirmations (incl. the whole Item-1 levelling call and the headline), 2 clean
falsifications of edge-candidates (extrapolation, robustness), 0 reversed-after-seeing-data.** The one
edge I pre-registered as strongest (capacity) is the one that survived; the two I hedged as
weak/candidate both failed. A pre-registered edge that survives is evidence; two that fail are
findings (both unflattering to CLU, which is the honest direction).

## 7. How I verified

- **Full suite: `570 passed` in 500 s** (`pytest tests/ -q -p no:randomly --no-cov`), incl. **9 new
  tests** in `tests/test_gated_write.py`. `ruff check` on all touched files → clean.
- **New tests pin what could fake this comparison:** cfg-override restore (and restore-on-exception,
  so a CLU knob never leaks into a baseline); the two CLU rows differ only in the write current
  (gated ≠ linear map); the extrapolation harness really evaluates a block at **2× the trained
  length** without a shape crash and returns finite numbers; and the robustness noise leaves the
  **marker channel untouched** (else it changes the task rather than stressing it).
- **Built-in reproduction check (unplanned, and it passed):** the re-run baselines and linear CLU
  reproduce prior published numbers to ≤1 in the 4th dp — GRU adding 0.0007 / MQAR 0.486 / parity
  1.000; attention MQAR 0.9945; **clu_linear MQAR 0.3864, parity 0.5380, adding 0.1825** all match
  `primitive-harness`/`gamma-read-sweep`. The comparison sits on the same measuring stick as the
  prior waves.
- **Zero divergences / NaNs** in any Item-1 or 3b cell; the driven-Verlet block is numerically stable
  *within* the trained horizon (its **only** instability is the off-distribution extrapolation blow-up
  in 3a, which is the finding).
- Every number is generated from the result JSONs by `.claude/scratch/gated-write-performance/
  make_tables.py` / `make_figs.py`; nothing transcribed by hand.
- **Artifacts** in `.claude/outputs/gated-write-performance/`: `PREREG.md`, `gated_write_core.json`
  (Item 1 + 3b + cost), `gated_write_edges.json` (3a + 3c), `capacity_crossover.png`,
  `extrapolation.png`, `pytest.log`, `edges.log`.

## Git footprint

- **Branch** `agent/experiment-engineer/gated-write-performance`, off local `main` @ `8519df6`.
  Rebase onto `main` = no-op (base unmoved); `origin/main` untouched (§7.21).
- **Worktree** `../CHLU-gatedwrite` (main checkout was on another agent's branch
  `hopfield-capacity-benchmark`, clean, at main's head — per §3.2 I never edited it; created a
  worktree off `main`). Main venv reused via `PYTHONPATH` (JAX stayed 0.9.0, no `uv sync`). Branch
  ref verified from the main repo. Remove with `git worktree remove ../CHLU-gatedwrite` after review.
  **No collision.**
- **Commit (1 code + results not committed):** `d0b3f4a` *add w22 gated-write performance test
  (exp_gated_write)*. Result JSONs live under `.claude/outputs/` (protocol §2), **not** committed to
  the tracked `results/` dir.
- **Files touched (3 + 1 new test, all surgical):** `chlu/config.py` (+7 fields in
  `ExperimentPrimitiveHarnessConfig`, all new defaults, shipped harness unchanged),
  `chlu/experiments/exp_gated_write.py` (new module, reuses primitive-harness helpers),
  `chlu/cli/experiment_cmd.py` (+1 subparser +1 cmd fn), `tests/test_gated_write.py` (+9 tests).
  **No shared physics/training/plotting/data code, and no shipped harness code, touched.** The gate
  itself (`CLUBlock.write_mode`) was already on `main` from `gamma-read-sweep`.
- **Commands:** `pytest tests/ -q -p no:randomly --no-cov` → 570 passed; `ruff check` → clean;
  `python -m chlu.experiments.exp_gated_write [--items ...] [--families ...] [--quick]`;
  CLI `chlu exp-gated-write`.
- **Unresolved conflicts:** none.

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** *The gated-write fix was run properly (3 families, 3 seeds,
   symmetric rescue, matched 40k params).* **Gated CLU wins 0 of 3 on absolute performance** — off
   the adding floor by 79× (0.1825→0.0023) but 3.3× worse than the GRU; ties the GRU on MQAR
   (0.4993 vs 0.4863 at γ=0); still near chance on parity (gate supplies a conjunction, not the XOR
   *state* parity needs). **Levelling, not beating** — the gate imports a capability every baseline
   already has. Cost 1.48× GRU wall-clock.
2. ⭐ **§6 / §8 — the ONE surviving edge (decision-relevant):** *on the capacity axis (MQAR item-load
   sweep), gated CLU at γ=0 degrades far more gracefully than a matched, **rescued**, gated GRU and
   crosses above it at kv≈4, beating it 1.7× at kv=8 and 2.1× at kv=16.* Survives matched gating AND
   matched tuning; the graceful-capacity property pre-exists the gate (linear CLU also crosses at
   kv≈8), so it is a CLU architectural property, not a gate artifact. **Attention still beats all**,
   so it is a GRU/SSM-beating edge, not SOTA. This is the program's strongest CLU-favourable
   performance signal and it is now measured under the strictest fairness (rescued baselines, gated,
   matched params, 3 seeds).
3. ⚠ **DOWNSTREAM RECONCILIATION (needs an owner):** `primitive-harness §1b`'s capacity numbers left
   the **GRU un-rescued** (kv=16 GRU = 0.008 = chance → the "19× / 17× chance" claim). With the GRU
   rescued as hard as the CLU it recovers to **0.085**, so the honest kv=16 margin is CLU 0.180 vs
   GRU 0.085 = **2.1×, not 19×.** Any site quoting the 19× capacity gap must be corrected to the
   rescued-baseline margin.
4. **§7 / §8 — two edge-candidates CLOSED as falsified.** (a) *Long-horizon extrapolation is NOT a
   CLU edge* — the trained driven-Verlet block is the **worst** extrapolator of the three (adding MSE
   blows up to 2.01 at 2×, 2.96 at 4×, worse than predicting the mean), refuting the founding-CHLU
   "symplectic ⇒ stable extrapolation" intuition **in the trained, driven slot**. (b) *Input-noise
   robustness is NOT a CLU edge* — gated CLU is worse at every noise level and degrades on a steeper
   slope; Prop-2 barrier confinement does not transfer to the trained block (consistent with
   `primitive-harness` reconciliation #1).
5. **New CLI/config surface:** `chlu exp-gated-write [--items item1 3a 3b 3c cost] [--families ...]
   [--quick] [--out …]`; `ExperimentPrimitiveHarnessConfig` gains `gw_train_steps`, `gw_tune_steps`,
   `gw_mqar_gamma`, `gw_extrap_train_T`, `gw_extrap_mults`, `gw_extrap_families`, `gw_noise_grid` —
   all default-preserving.
6. **Cost/runtime note for future scoping:** the full symmetric-rescue three-family + 20-cell
   capacity run took **~2h20m** on this (contended) machine at the 1200/400 budget. The rescue pass
   roughly triples per-cell cost; the capacity axis (5 variants × 4 kv) is the expensive part.
   Consider running the capacity edge *without* the baseline rescue when only the crossover shape is
   needed (it is directly comparable to the un-rescued `primitive-harness §1b`).
