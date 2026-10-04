# primitive-harness — experiment-engineer report

**Task + acceptance criterion:** build a fair, reusable, honest harness that drops CLU into a standard architecture slot against MLP / GRU / SSM / attention at matched parameter budget, on ≥2 task families reported separately, with stated compute cost and the recall-vs-distractors figure.
**Status: done** — all 3 task families, both MQAR axes (distractors + capacity), symmetric LR-rescue pass, and an interleaved cost benchmark. 45 cells × 3 seeds.

> ⚠ **HEADLINE: the task file's structural prediction is FALSIFIED, and it was falsified in the direction I pre-registered.** The task states *"attention degrades from interference and CLU does not, because of barrier confinement (Prop 2)."* Measured: **attention is flat (0.996 → 0.996 across an 8× distractor sweep, drop 0.001) and CLU is the one that degrades (0.409 → 0.272, drop 0.137).** Attention is the *strongest* primitive on MQAR by a wide margin, not the interference-limited one.
>
> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).**
> 1. **Any site claiming Prop-2 hard read isolation as a CLU advantage over softmax attention on a recall benchmark must be re-scoped.** Prop 2 governs a *hand-designed* `RingRegisterPotential` with no training (`clu-retrieval-demo`); it does not transfer to a trained CLU block, and the trained block loses to attention 0.27 vs 0.996 at T=256.
> 2. **"Retrieval as a fixed point / durability 1.000 out to 1200 steps" must not be quoted as evidence of recall capability in a learned setting.** Those numbers are from a designed landscape holding pre-written items; here the CLU must *learn* to write, address and read, and it reaches 0.41 at best.
> 3. **CLU wins zero of three task families.** It is mid-tier on recall (beats SSM and the no-mixing control, loses to GRU and attention) and is **at the no-mixing control floor** on long-range integration (0.182 vs control 0.183) and **at chance** on state tracking (0.538 vs chance 0.5). Any "general primitive" framing must carry this.
>
> ⭐ **THE ONE GENUINELY CLU-FAVOURABLE RESULT — and it is on the capacity axis, where the programme's memory claims live.** As the number of *stored items* grows at fixed sequence length, **CLU degrades far more gracefully than the GRU and crosses above it**: GRU 0.997 → 0.486 → 0.177 → **0.008 (chance)** for kv = 2 → 4 → 8 → 16, while CLU goes 0.525 → 0.346 → 0.237 → **0.154**. At kv=16 the GRU has collapsed to chance and **CLU still retains 19× the GRU's accuracy and 17× chance.** This is a *capacity* claim, it is the axis the barrier-confinement story should predict, and it is the strongest available evidence for the primitive framing. ⚠ It does **not** rescue the headline: attention holds 0.856 ± 0.170 at kv=16 and beats CLU at every point.

---

## Flag-provenance table

| item | value |
|---|---|
| commit | `1627369` (branch `agent/experiment-engineer/primitive-harness`, base `main` @ `089cc6e`) |
| seeds | 3 per cell: `42, 1042, 2042` (`cfg_seed(i) = 1000*i + 42`) |
| slot | embedding + learned positional embedding → **2 ×** [block + residual + LayerNorm] → linear head; `d_model=64` |
| budget | **block** params matched to **40 000** (match error ≤ 2.4%, see §3); total params ≈ 81.4–82.4 k |
| optimizer | Adam + global-norm clip 1.0; **LR grid `{3e-4, 1e-3, 3e-3}` identical for every primitive** |
| schedule | LR selection at **400** steps, final runs at **1200** steps, batch 32, eval batch 256 |
| CLU block | `kinetic_mode=newtonian_learned`, `potential_type=mlp` (PotentialMLP, hidden 32, incl. 0.05‖q‖² confinement), **dt=0.1, γ=0.05**, 1 Verlet step/token, `d_clu=83` |
| langevin_noise | **N/A** — deterministic Verlet only; no Langevin, no temperature, no sleep phase |
| MQAR | vocab 256 (chance ≈ 0.008), kv=4 for the distractor sweep, T=128 for the capacity sweep, uniform gaps |
| rescue pass | applied to the **distractor sweep, adding and parity**; **NOT** to the capacity sweep (stage C) — those cells carry the equal-budget short-selection protocol only |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** — main venv reused per protocol §4 (no worktree sync) |
| machine | shared; load average **277 → 2.5** across the session (other agents' jobs). See §4 on why this does not contaminate the cost table. |

**Not tuned, and deliberately so:** the CLU's physics knobs (`dt`, `γ`, `clu_steps`, `kinetic_mode`) were fixed *before* any result was seen and never adjusted. No post-hoc CLU-only knob was added. Every primitive got the same three LRs, the same steps, the same seeds.

---

## 1. Item 1 — the drop-in slot, and what CLU conceded

**The interface** (`chlu/core/blocks.py`): `block(x: (T, d_model)) -> (T, d_model)`, causal in `T`. All five primitives implement exactly this and are swapped by name; embedding, positional embedding, residual/LayerNorm, head, optimizer, schedule and data are byte-identical across primitives (pinned by `test_shared_scaffolding_is_identical_across_primitives`).

**CLU fits the slot without special-casing the harness.** It conceded exactly three things, all pre-registered:

| # | concession | why it is forced |
|---|---|---|
| 1 | **Driven, not autonomous** — input enters as a momentum impulse `p += W_in x_t` | the slot must absorb an exogenous token stream; the autonomous Hamiltonian has no input channel. **H is no longer conserved.** |
| 2 | **Dissipative, not symplectic** — γ=0.05 > 0 | a readable state must be a fixed point (`clu-retrieval-demo` §6: 1.000 at γ=0.02 vs 0.813 at γ=0). Pinned by `test_clu_block_gamma_is_dissipative_by_default`. |
| 3 | **Carry width 2·d_clu** — state is (q,p) | parameter matching solves for `d_clu` separately. |

**No fourth concession was needed** — as pre-registered. In particular the driven Verlet rollout was **numerically stable at every sequence length and LR tried: zero divergences, zero NaNs, across all 30 cells × 3 seeds + rescue probes.** No clipping, no bounded input kick, no per-family tuning. This is a genuine positive finding: *the CLU is a well-behaved drop-in recurrent block.* Gradients reach both `potential_net` and `log_mass` (`test_gradients_flow_through_the_clu_block`).

⚠ **Fairness note that cuts against CLU-favouring readings:** all primitives receive the learned positional embedding, which recurrent primitives do not need. This *helps* the recurrent baselines (GRU/SSM/CLU) and is the conservative choice.

## 2. Item 3 — per-family results (NEVER averaged)

### Family 1 ⭐ MQAR associative recall — accuracy vs distractors (kv=4, chance ≈ 0.008)

| primitive | T=32 (24 distr.) | T=64 (56) | T=128 (120) | T=256 (248) | **drop 32→256** |
|---|---|---|---|---|---|
| MLP (control, no mixing) | 0.012 ± 0.004 | 0.012 ± 0.004 | 0.012 ± 0.004 | 0.012 ± 0.004 | +0.000 |
| GRU | 0.666 ± 0.031 | 0.577 ± 0.034 | 0.486 ± 0.019 | 0.441 ± 0.036 | **+0.225** |
| SSM (S4D/Mamba-lite) | 0.072 ± 0.041 | 0.014 ± 0.004 | 0.012 ± 0.004 | 0.011 ± 0.003 | +0.061 |
| **Attention** | **0.996 ± 0.002** | **0.998 ± 0.001** | **0.994 ± 0.002** | **0.996 ± 0.002** | **+0.001** |
| **CLU** | 0.409 ± 0.005 | 0.374 ± 0.006 | 0.346 ± 0.008 | 0.272 ± 0.005 | **+0.137** |

**Ranking: attention ≫ GRU > CLU > SSM > MLP.** Figure: `harness_fig1_recall.png` (panel a).

- **The structural prediction fails.** Attention does not degrade from interference at these scales — it is *flat to three decimals* while the sequence grows 8×. The mechanism is not mysterious: exact attention has an O(T) KV cache, so distractors cost it nothing.
- **CLU degrades, as the information-theoretic argument requires.** Its carry is a fixed `2·83` floats; distractors dilute a fixed-capacity state. Barrier confinement can protect an already-stored item; it cannot create capacity.
- **CLU is nonetheless clearly *doing the task*** — 0.27–0.41 against a 0.012 no-mixing floor and a 0.008 chance rate is real associative binding, learned end-to-end, with no hand-designed landscape.

### Family 1b ⭐ MQAR — accuracy vs number of STORED ITEMS (T=128 fixed, the capacity axis)

| primitive | kv=2 | kv=4 | kv=8 | kv=16 | **drop 2→16** |
|---|---|---|---|---|---|
| MLP (control) | 0.011 ± 0.007 | 0.012 ± 0.004 | 0.009 ± 0.001 | 0.009 ± 0.001 | +0.002 |
| GRU | **0.997 ± 0.003** | 0.486 ± 0.019 | 0.177 ± 0.009 | **0.008 ± 0.001** | **+0.989 (total collapse)** |
| SSM (Mamba-lite) | 0.012 ± 0.007 | 0.012 ± 0.004 | 0.009 ± 0.001 | 0.010 ± 0.000 | +0.002 (at floor throughout) |
| **Attention** | 0.887 ± 0.152 | **0.994 ± 0.002** | 0.938 ± 0.082 | **0.856 ± 0.170** | +0.031 |
| **CLU** | 0.525 ± 0.024 | 0.346 ± 0.008 | **0.237 ± 0.002** | **0.154 ± 0.002** | **+0.371** |

**CLU − GRU: −0.471 (kv=2) → −0.140 (kv=4) → +0.060 (kv=8) → +0.147 (kv=16).** The crossover sits between kv=4 and kv=8. Figure: `harness_fig1_recall.png` panel (b).

Reading this honestly: the GRU is a *better* memory when there are few things to remember and a *catastrophically worse* one as items accumulate; the CLU trades peak accuracy for a much flatter capacity curve. Both are fixed-state recurrences at identical parameter count, so this is a difference in **how state is used**, not how much there is — which is the first result in this harness that the barrier/landscape story could plausibly explain. It should be handed to the theorist alongside `address-space-dimension-scaling`.
⚠ Attention's error bars at kv=2 and kv=16 (±0.15, ±0.17) are large — one seed underperforms in each. Its *mean* still exceeds CLU's best cell everywhere.

### Family 2 — adding problem, T=128 (long-range integration; MSE, lower better)

| primitive | MSE | best lr |
|---|---|---|
| MLP (control) | 0.183 ± 0.003 | 3e-4 |
| GRU | 0.001 ± 0.000 | 3e-3 |
| SSM (S4D/Mamba-lite) | 0.001 ± 0.000 | 3e-3 |
| Attention | **0.000 ± 0.000** | 3e-4 |
| **CLU** | **0.182 ± 0.003** | 3e-3 |

⚠ **CLU fails this family outright — it sits exactly on the no-mixing control floor (0.182 vs 0.183),** i.e. it learned the target's mean and nothing else, while every other mixing primitive solved the task to ≤0.001 (a 180× gap). **This falsifies my own pre-registration (P2: "CLU ≤0.05, within 3× of the SSM"), and it is the most surprising result of the wave**: a symplectic oscillator is *structurally* a linear SSM with imaginary eigenvalues, which is the object S4 approximates, so this is the family CLU was expected to win. It did not learn to integrate at all. This is now the sharpest open question about the primitive (§7).

### Family 3 — parity, T=64 (state tracking; accuracy, chance 0.5)

| primitive | accuracy | best lr |
|---|---|---|
| MLP (control) | 0.511 ± 0.004 | 1e-3 |
| **GRU** | **1.000 ± 0.000** | 3e-4 |
| SSM (S4D/Mamba-lite) | 0.638 ± 0.013 | 3e-3 |
| Attention | 0.547 ± 0.003 | 3e-3 |
| **CLU** | 0.538 ± 0.006 | 1e-3 |

**CLU is at chance (0.538), below even attention.** GRU's perfect score confirms the task is solvable in the slot at this budget, so this is a CLU failure, not a task artifact. Predicted ≥0.80 — **falsified.**

**Per-family verdict: CLU wins 0 of 3 families.** It is a credible mid-tier recall primitive and a non-performer on the other two.

## 3. Item 2 — matched budgets, and the honest compute price

Matching is on **block** parameters (embedding/pos/head are identical for all primitives, so including them would dilute the match); totals are reported alongside.

| primitive | width knob | block params | total params | match err | fwd FLOPs (T=128) | ms/step | ×GRU wall | ×GRU FLOPs |
|---|---|---|---|---|---|---|---|---|
| MLP (control) | 155 | 40 118 | 81 590 | 0.3% | 14.96 M | 15.9 | 0.51× | 2.46× |
| GRU | 49 | 40 014 | 81 486 | 0.0% | 6.06 M | 31.4 | 1.00× | 1.00× |
| SSM | 101 | 40 052 | 81 524 | 0.1% | 14.71 M | 22.7 | 0.72× | 2.41× |
| Attention | 82 | 40 960 | 82 432 | 2.4% | 26.27 M | 31.8 | 1.01× | 4.33× |
| **CLU** | **83** (`d_clu`) | **39 886** | 81 358 | 0.3% | **12.68 M** | **44.2** | **1.41×** | **2.09×** |

**The honest multiple: CLU costs 1.41× the GRU and 1.39× attention in wall-clock, while using 0.48× attention's FLOPs.** Per Head's ruling this is stated, not competed on — and it is *far cheaper than I pre-registered* (P4 predicted 3–15× GRU and ≥5× attention; **falsified in CLU's favour**). The shape of the prediction did hold: CLU's cost is **wall-clock-dominated, not FLOPs-dominated** — it is the cheapest mixing primitive by FLOPs after the GRU, and the most expensive by wall-clock, because a `∇V` evaluation per token in a sequential scan parallelises poorly.

**Measurement integrity.** The machine was shared with other agents (load average 277 at one point, which roughly doubled step times). Timing primitives sequentially would have rewarded whichever ran during a quiet period. The table above comes from `benchmark_cost`: a **round-robin interleaved** benchmark (5 rounds × 25 steps each, cycling primitives), run on a quiet machine (load 2.5), median over rounds, IQR ≤ 2.0 ms. FLOPs come from XLA's own cost analysis (exact, load-independent) — never a hand-rolled count that could flatter one primitive.

## 4. Item 4 — the baselines are real, and how much tuning each got

**Tuning budget is equal by construction:** identical LR grid `{3e-4, 1e-3, 3e-3}`, identical 400-step selection runs, identical 1200-step final runs, identical seeds. **3 tuning runs + 3 final runs per primitive per cell — no primitive received more.**

Then, because equal-budget selection can still be *uninformative* (on hard recall cells nothing has learned by step 400, so all three LRs read at chance and the "winner" is noise), I ran a **symmetric LR-rescue pass**: re-run the two non-selected LRs at **full length** for **every** primitive in **every** cell, adopt a better one if found. This is monotone and symmetric — it can only *raise* a score and applies equally to CLU — so it cannot be a route to a CLU win. **+2 full-length runs per primitive per cell.**

**Total tuning spend, identical per primitive: 5 full-length-equivalent training runs + 3 short runs per cell.**

Rescue outcome (this is the baseline-integrity evidence):

| cell | rescued? | note |
|---|---|---|
| MQAR, all T, **SSM** | **no** | probed at all three LRs at full length; **none beat its selected result.** Its weakness is real, not a tuning artifact. |
| MQAR, all T, **CLU** | **no** | likewise — CLU's recall numbers are robust to LR choice. |
| MQAR, all T, GRU / attention | no | already at their best LR (3e-3). |
| MQAR, all T, MLP (control) | yes | 0.008 → 0.012. Still the floor. |
| adding, GRU / attention / **CLU** | yes | GRU 8e-4→7e-4, attn 2e-4→1e-4, **CLU 0.1831→0.1825 (still the control floor)** |
| parity, MLP / **CLU** | yes | MLP 0.507→0.511, **CLU 0.523→0.538 (still chance)** |

**Crucially, the rescue helped CLU on both families it failed and did not change either verdict** — CLU's adding failure and parity failure survive the full LR grid at full length. That is the strongest available evidence that these are properties of the primitive, not of my tuning.

⚠ **Honest limitation on one baseline — do NOT read this as "CLU beats Mamba".** My `SSMBlock` is a **simplified S4D/Mamba-lite**: real diagonal recurrence + input-dependent Δ + output gate, but **no short causal convolution and no proper input-dependent B/C projections**, which are exactly the ingredients Zoology identifies as load-bearing for recall. Its MQAR score (0.011–0.072) should be treated as a **lower bound on a real Mamba**, not a measurement of one. Two things argue it is not simply broken: it solves the adding problem to 0.001 (tied with GRU) and reaches 0.638 on parity, and it survived the full-length rescue. **A genuine Mamba baseline is the top follow-up before any SSM comparison is published.**

## 5. Pre-registration scorecard (`PREREG.md`, written before any harness run)

| # | prediction | measured | verdict |
|---|---|---|---|
| P1 | task's claim (attention degrades, CLU flat) will FAIL | attn drop 0.001, CLU drop 0.137 | ✅ **the wave's headline** |
| P1 | attention ≥0.90 @T=32, ≥0.85 @T=256, drop ≤0.10 | 0.996 / 0.996 / 0.001 | ✅ |
| P1 | CLU 0.35–0.85 @T=32; 0.15–0.55 @T=256 | 0.409 / 0.272 | ✅ |
| P1 | CLU drop ≥0.15 across the sweep | **0.137** | ❌ marginally — CLU decays a little more gently than predicted |
| P1 | GRU ≥0.70 @T=32 | 0.666 | ❌ marginally |
| P1 | SSM ≥0.70 @T=32, 0.35–0.75 @T=256 | **0.072 / 0.011** | ❌ **badly** — see the Mamba-lite caveat |
| P1 | ranking attn > {ssm,gru} > clu > mlp | attn > gru > **clu > ssm** > mlp | ❌ CLU beat the SSM |
| P1b | attention ~flat, ≥0.80 at kv=16 | 0.856 ± 0.170 | ✅ |
| P1b | **CLU degrades monotonically, ≥0.25 drop kv=2→16** | **0.525 → 0.154, drop 0.371** | ✅ **capacity argument confirmed** |
| P2 | CLU ≤0.05 MSE, within 3× of SSM | **0.182 = control floor, 180× worse** | ❌ **falsified — the wave's biggest surprise** |
| P2 | ssm ≤0.02, attn ≤0.05, gru ≤0.10, mlp ≈0.17 | 0.001 / 0.000 / 0.001 / 0.183 | ✅ (GRU better than predicted) |
| P3 | GRU ≥0.95 | 1.000 | ✅ |
| P3 | attention ≤0.75 | 0.547 | ✅ |
| P3 | SSM 0.60–0.95 | 0.638 | ✅ |
| P3 | **CLU ≥0.80** | **0.538 (chance)** | ❌ **falsified** |
| P4 | CLU wall-clock 3–15× GRU, ≥5× attention | **1.41× / 1.39×** | ❌ falsified — CLU is far cheaper than predicted |
| P4 | cost is wall-clock-dominated, not FLOPs-dominated | 0.48× attn FLOPs, 1.39× attn wall | ✅ |
| P5 | CLU drop-in with exactly 3 concessions, no 4th | exactly 3; zero divergences | ✅ |

**Six falsifications, three of them about CLU and all three unflattering** (adding, parity, and a ranking where the SSM was supposed to beat it). One falsification (P4) favours CLU. The pre-registration did its job: the adding-problem prediction was mine, was theoretically motivated, and was wrong by 180×.

## 6. How I verified

- **Full suite: `428 passed` in 490 s** (`pytest tests/ -q -p no:randomly`), including the **36 new tests** in `tests/test_primitive_harness.py` (baseline on `main` was 392). `ruff check chlu/ tests/` → **All checks passed.**
- New tests pin the fairness properties a reviewer would attack: **causality of every primitive** (perturbing input *t* must not move any output before *t*, *and* must move the output at *t* — a block that ignores its input would otherwise pass), shape-interchangeability in both I/O modes, parameter-match tolerance, identical shared scaffolding, the γ>0 concession, gradient flow into `potential_net`/`log_mass`, task-generator correctness, and **two regression tests for the rescue-pass monotonicity bug below**.
- Every number above is generated from the result JSONs by `.claude/scratch/primitive-harness/make_tables.py`, not transcribed by hand.
- Artifacts: `harness_fig1_recall.png`, `harness_fig2_families_cost.png`, `PREREG.md` in `.claude/outputs/primitive-harness/`; raw JSONs under `CHLU-primitive/stage{A,B,C}/results/` (A and B also as `*_rescued.json`) and `results/primitive_cost_benchmark.json`. Drivers in `.claude/scratch/primitive-harness/` (`run_stages.py`, `run_rescue.py`, `make_tables.py`, `parse_log.py`).

### Bugs I found by running it (each would have produced a wrong number)
1. **Winner's curse in my own rescue pass.** The probe that triggers a rescue is a *single* seed but the reported number is a 3-seed mean, so a lucky probe could replace a good result with a worse average — observed live on `adding_T128/mlp` (0.1825 → 0.1832 on a lower-is-better metric). A pass whose entire purpose is protecting baselines must never degrade one. Fixed (adopt only if it still wins on the full n-seed mean), with regression tests in both directions. **The affected runs were produced before the fix; the correction is applied at report time from the stored `pre_rescue_metric_mean`, so every number above is the corrected one.**
2. **`eqx.filter_jit` does not expose `cost_analysis`** (JAX 0.9), so FLOPs silently read `NaN`. Fixed by closing over the model and lowering a plain `jax.jit`. Had I not checked, the compute table would have been empty in exactly the place the task demands a number.
3. **Uninformative LR selection** (the reason the rescue pass exists at all) — SSM picked its LR from three runs that were all at chance.
4. **Un-jitted batch generation cost more per step than the training step**, and the MQAR generator dominated. Moved to a cached `jax.jit`; it sits outside the timed region so it never entered the cost table.

## 7. Open questions / follow-ups / risks

1. ⭐ **Why does the CLU fail the adding problem?** This is the sharpest open question. A γ=0 symplectic oscillator is a near-perfect integrator, yet the block scores exactly the no-mixing floor. Leading hypothesis: **γ=0.05 per token is far too dissipative for T=128** — the state's memory half-life is ~`2ln2/γ` ≈ 28 tokens, so information from the first marker is gone by the readout. That predicts a **γ sweep should show a sharp transition**, and it is cheap. ⚠ **This is exactly the tension `clu-retrieval-demo` §6 identified** (readability needs γ>0; propagation needs γ→0) now showing up in a *trained* setting. **I did not run this** — it would be a post-hoc CLU-only knob, forbidden by my own pre-registration. It should be a separate, pre-registered task.
2. ⭐ **The CLU/GRU capacity crossover is the result most worth following up.** It is the only axis on which CLU beats a standard primitive, it is *predicted* by the barrier-confinement story, and it was measured at matched parameters with an equal tuning budget. Two cheap strengtheners before it is quoted anywhere: (a) extend to kv=32/64 to see whether CLU's curve keeps its shape or also collapses; (b) **the stage-C cells did not get the rescue pass** (run out of session time) — they carry the equal-budget short-selection protocol only. Rescue them before publication: `--rescue-from stageC/results/exp_primitive_harness.json`.
3. **The SSM baseline is Mamba-lite, not Mamba** (§4). A real Mamba (short conv + input-dependent B/C) is the top follow-up before any SSM claim is published.
4. **Budget is small** (40 k block params, 1200 steps, T ≤ 256, 2 layers, vocab 256). Conclusions are about *this* regime. Attention's flatness would likely hold at larger T; CLU's decay might steepen. GRU at 2000 steps scored 0.892 at T=32 vs 0.666 at 1200 — **absolute numbers are budget-sensitive, the ranking was stable across both budgets I ran.**
5. **`clu_steps=1`** (one Verlet step per token). More integration steps per token would raise both cost and expressivity; untested.
6. **Do not quote the "8-item ceiling"** (2-D-ring artifact). This harness runs `d_clu=83`, an address space far above the ring regime, so that ceiling does not bound these results. Coordinate the kv-sweep result with `address-space-dimension-scaling`.

## Git footprint
- **Branch** `agent/experiment-engineer/primitive-harness`, off local `main` @ `089cc6e`. Rebase onto `main` = no-op; `origin/main` untouched (§7.21).
- **Worktree** `../CHLU-primitive` — the main checkout had another agent's branch (`dt-units-split`) checked out, so per protocol §3.2 I never touched it. **No collision.** Branch ref verified from the main repo (w4 lesson). Worktree left in place for review; remove with `git worktree remove ../CHLU-primitive`.
- **Commits (7):** `669880b` blocks+tasks · `7bf64d0` harness · `28d79b0` rescue+compile cache · `3e330a0` figures · `7de1eea` checkpointing · `750f112` jit batching · `1627369` winner's-curse fix + cost benchmark.
- **Files: all new except two surgical additions** — `chlu/core/blocks.py`, `chlu/data/seq_tasks.py`, `chlu/experiments/exp_primitive_harness.py`, `chlu/experiments/plot_primitive_harness.py`, `tests/test_primitive_harness.py` (new); `chlu/config.py` (+1 dataclass at the 3 required sites), `chlu/cli/experiment_cmd.py` (+parser +`cmd_exp_primitive_harness`). **`utils/plotting.py` and all shared physics/training code untouched.**
- **Commands:** `pytest tests/ -q -p no:randomly` → 426 passed; `ruff check chlu/ tests/` → clean; `python -m chlu.experiments.exp_primitive_harness --quick [--rescue]`, `--benchmark`, `--rescue-from`.

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** *The CLU has now been benchmarked as a general sequence primitive against MLP/GRU/SSM/attention at matched parameters.* **It wins 0 of 3 families**: mid-tier on associative recall (0.41→0.27 vs attention's flat 0.996), at the no-mixing control floor on long-range integration, at chance on state tracking. Cost 1.41× GRU wall-clock at 0.48× attention's FLOPs. **The one favourable result: on the capacity axis CLU degrades far more gracefully than the GRU and crosses above it at kv≈8 (kv=16: CLU 0.154 vs GRU 0.008 = chance)** — a difference in *how* a fixed state is used, at identical parameter count.
2. **§7 — new known issue (scientific, high priority):** *the CLU block does not learn long-range integration in the drop-in slot.* Candidate cause: per-token friction γ=0.05 gives a ~28-token memory half-life. Needs a pre-registered γ sweep.
3. **Claims-matrix / paper-text warning:** **Prop-2 read isolation and the "durability 1.000 / retrieval as a fixed point" numbers are properties of a hand-designed landscape and must not be cited as evidence of learned recall performance.** The trained block loses to attention by 0.73 absolute at T=256. This is reconciliation item 1 above and needs an owner.
4. **New CLI/config surface:** `chlu exp-primitive-harness [--quick] [--families ...] [--steps N] [--rescue] [--rescue-from PATH] [--benchmark]`; `ExperimentPrimitiveHarnessConfig` (registered at all three config sites per the documented three-site rule).
5. **Reusable methodology worth adopting programme-wide:** (a) the **symmetric, monotone LR-rescue pass** as the standard answer to "was the baseline tuned as hard?"; (b) the **round-robin interleaved cost benchmark**, because this machine is shared and sequential timing is not reproducible — **load average reached 277 during this wave and roughly doubled step times**; (c) **per-family reporting with a no-mixing control**, which is what made "CLU is at the control floor" legible rather than "CLU got 0.18".
6. **Env note for §4:** `eqx.filter_jit` has no `cost_analysis` in JAX 0.9 — use a plain `jax.jit` closure for FLOP counts.
