# gamma-read-sweep — experiment-engineer report

**Task + acceptance criterion:** is CLU's 0-of-3 sequence failure an artifact of the γ=0.05/token dissipation setting inherited from a retracted w19 measurement? Sweep γ (7 points × 3 families × 3 seeds), cross it with a new trajectory read mode, sweep `clu_steps` with cost, and produce a corrected table **iff** a better configuration exists.

**Status: done.** All four items run: **48 pre-registered sweep cells** (Item 1: 21, Item 2: 24, Item 3: 3) × 3 seeds × 3-LR selection, plus 3 clearly-labelled exploratory cells, 2 interleaved cost benchmarks and 3 linear-information probes. Full suite green (526 passed).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).**
> 1. **"0 of 3 stands, and it hardens."** **16 of 19 pre-registered predictions confirmed cleanly, 2 marginal, 1 not triggered; none falsified.** γ is not the limiter on any family. `primitive-harness` §7.1's leading hypothesis — *"γ=0.05 per token is far too dissipative for T=128"* — is **refuted** and should be retired from the open-questions list, not left live.
> 2. **`primitive-harness` concession #2's justification must be corrected wherever it is quoted.** "γ>0 is required for a readable state (1.000 at γ=0.02 vs 0.813 at γ=0)" rests on the retracted w19 single-phase measurement. Measured here: γ=0 is fine — **zero divergences and zero NaNs in every γ=0 cell (9 full-length runs in Item 1 alone, more in Item 2)**, and on MQAR γ=0 is the *best* setting of the seven. The docstrings and the test that carried this claim are corrected on my branch; any paper/claims-matrix text is not.
> 3. ⭐ **A new, mechanistically-verified cause of the adding/parity failure, with a one-line fix — and it is NOT a CLU advantage.** The CLU block's write current `p += W_in x_t` is **unconditionally linear in the token**, so the state can carry `Σ_t v_t` and `Σ_t m_t` but never their product. Measured: at the readout the state decodes `Σ_t v_t` at **R² = 1.000** and both marker positions at **R² ≈ 0.99** at *every* γ including 0.1, while the task target — a value×marker *conjunction* — sits at **R² ≈ 0.02–0.09**. Adding a multiplicative input gate (the ingredient the GRU, the selective SSM and attention all already have) takes the adding MSE from **0.1816 (= the no-mixing control floor) to 0.0028**, a **65× improvement at matched parameters**. ⚠ **This is CLU catching up to a standard ingredient, not beating anything** — it is an exploratory arm outside the pre-registered grid and needs its own pre-registered, full-baseline follow-up before any claim is made.

---

## Flag-provenance table

| item | value |
|---|---|
| commit | `af02a22` (block/config/tests) + `834b9f3` (sweep driver/CLI) (branch `agent/experiment-engineer/gamma-read-sweep`, base local `main` @ `31c3e15`) |
| seeds | 3 per cell: `42, 1042, 2042` (`cfg_seed(i)=1000i+42`) — identical to `primitive-harness` |
| shared slot | **byte-identical to the shipped harness**: `d_model=64`, `n_layers=2`, block-param budget **40 000**, LR grid `{3e-4,1e-3,3e-3}`, **tune 400 / train 1200** steps, batch 32, eval batch 256, grad-clip 1.0 |
| CLU physics held fixed | `kinetic_mode=newtonian_learned`, `potential_type=mlp` (PotentialMLP, hidden 32, incl. 0.05‖q‖²), **dt=0.1** |
| CLU knobs SWEPT | **γ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}**; `read_mode ∈ {endpoint, trajectory}`; `clu_steps ∈ {1,2,4}` |
| CLU knob swept EXPLORATORY (labelled) | `write_mode ∈ {linear, gated}` — **outside the pre-registered grid** |
| `d_clu` | 83 (endpoint, all γ, all `clu_steps`) · 53 (trajectory, K=2) · 31 (trajectory, K=4) · 65 (gated write). All matched to 40 k block params, **max param error 0.50 %** |
| families | adding T=128 · parity T=64 · MQAR T=128 kv=4 vocab 256 — the exact three shipped cells |
| langevin_noise / temperature / sleep | **N/A** — deterministic Verlet, no Langevin, no thermostat, no sleep phase |
| baselines | **NOT re-run and NOT re-tuned.** All baseline numbers quoted are `primitive-harness`'s post-rescue values. Nothing in the shared slot changed, so they remain directly comparable |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** — main venv reused per protocol §4 (worktree, no `uv sync`) |
| machine | shared; load average **7 → 56 → 9** across the session (a concurrent `exp_potential_class` job). Metrics are load-independent; the one wall-clock table is round-robin interleaved and its load is stated in place |

**Fairness category of every knob I touched (task's absolute rule):** γ, `read_mode`, `clu_steps`, `write_mode` are **all category (a) — knobs no other primitive has.** They live entirely inside `CLUBlock`. The shared slot, the data, the optimiser, the LR grid, the step budget and the seeds were not touched, so **no baseline was disadvantaged and none needed re-running.** No category-(b) change was made.

**Reproducibility check (unplanned, and it passed).** Three cells were re-run in *separate processes* hours apart as by-products of Items 2 and 3. Every one reproduced its Item-1 value to 4 dp: adding γ=0 `0.1825±0.0032` (twice), γ=0.05 `0.1831±0.0060` (twice), γ=0.02 `0.1816±0.0043` (twice), parity γ=0 `0.5334±0.0053` (twice). And **my γ=0.05 MQAR cell reproduced the shipped harness's post-rescue CLU number exactly: 0.3464 vs 0.3464.**

---

## 1 ⭐ Item 1 — the γ sweep. The decisive measurement, and it is a null.

| family (metric) | γ=0 | γ=0.001 | γ=0.005 | γ=0.01 | γ=0.02 | γ=0.05 | γ=0.1 | shipped γ=0.05 | spread | best γ |
|---|---|---|---|---|---|---|---|---|---|---|
| **adding_T128** (mse) | 0.1825 ± 0.0032 | 0.1822 ± 0.0032 | 0.1821 ± 0.0031 | 0.1817 ± 0.0045 | 0.1816 ± 0.0043 | 0.1831 ± 0.0060 | 0.1839 ± 0.0075 | 0.1825 | **0.0023** | **0.02** |
| ↳ seed-σ within cell | 0.0032 | 0.0032 | 0.0031 | 0.0045 | 0.0043 | 0.0060 | 0.0075 | | (mean 0.0045) | |
| ↳ best lr | 0.003 | 0.001 | 0.001 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | | | |
| ↳ diverged seeds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | | |
| **parity_T64** (accuracy) | 0.5334 ± 0.0053 | 0.5323 ± 0.0021 | 0.5228 ± 0.0042 | 0.5349 ± 0.0032 | 0.5207 ± 0.0047 | 0.5233 ± 0.0051 | 0.5368 ± 0.0054 | 0.5380 | **0.0161** | **0.1** |
| ↳ seed-σ within cell | 0.0053 | 0.0021 | 0.0042 | 0.0032 | 0.0047 | 0.0051 | 0.0054 | | (mean 0.0043) | |
| ↳ best lr | 0.001 | 0.003 | 0.0003 | 0.001 | 0.0003 | 0.0003 | 0.003 | | | |
| ↳ diverged seeds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | | |
| **mqar_T128_kv4** (accuracy) | 0.3864 ± 0.0032 | 0.3857 ± 0.0086 | 0.3773 ± 0.0081 | 0.3770 ± 0.0060 | 0.3669 ± 0.0127 | 0.3464 ± 0.0080 | 0.3337 ± 0.0024 | 0.3464 | **0.0527** | **0** |
| ↳ seed-σ within cell | 0.0032 | 0.0086 | 0.0081 | 0.0060 | 0.0127 | 0.0080 | 0.0024 | | (mean 0.0070) | |
| ↳ best lr | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | | | |
| ↳ diverged seeds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | | |

**Read it family by family.**

- **adding T=128 — completely flat.** Spread over the whole 7-point grid is **0.0023, which is 0.50× the mean within-cell seed σ (0.0045)**. Every cell sits on the no-mixing control floor (0.1825). γ=0 is not better than γ=0.1; the best cell (0.1816 at γ=0.02) is 0.0009 from the worst-case shipped value and 260× worse than attention. **There is no transition, sharp or otherwise, anywhere near the derived γ\* = 2ln2/T = 0.01083 — or anywhere else.**
- **parity T=64 — flat at chance.** 0.5207–0.5368 across the grid; spread 0.0161 ≈ 3.8σ but **non-monotone** (the *best* cell is γ=0.1, the *most* dissipative). No cell beats the shipped 0.5380. The finite-memory model `acc ≈ (h+(T−h)/2)/T` predicts **0.71 at γ=0.05 and ~1.00 at γ≤0.01**; measured is 0.52–0.54 everywhere, i.e. the dissipation hypothesis over-predicts by 0.2–0.5 absolute at every point.
- **MQAR T=128 kv=4 — a real, clean, monotone, and SMALL effect.** 0.3864 → 0.3857 → 0.3773 → 0.3770 → 0.3669 → 0.3464 → 0.3337 as γ goes 0 → 0.1: **perfectly monotone**, spread 0.0527 = 7.5× the seed σ. **γ=0 buys +0.040 over the shipped γ=0.05 (+11.6 % relative).** This is the only place γ mattered — and it changes nothing: CLU is still far below the GRU (0.486) and attention (0.9945), and the shipped ranking is unaffected.

**Verdict: the hypothesis under test is refuted.** CLU's 0-of-3 is not a dissipation artifact. This is the task's second listed outcome — *"no movement at all … the primitive genuinely cannot integrate, which is the more serious finding"* — for adding and parity, with a monotone-but-shallow gain confined to MQAR.

Figure: `gamma-read-sweep/gamma_sweep.png`.

## 2 ⭐ Why — an information audit that localises the failure precisely

Item 1 says "changing γ does not change the score". The prior question is *what is in the state at the readout*. I ridge-regressed (held-out R², 50/50 split, 4096 sequences) six quantities from the CLU state at position T−1, across the same γ grid.

**Untrained block, `d_clu`=83, T=128, endpoint read, `clu_steps`=1** (R² from the concatenated 2-layer (q,p) state):

| decoded quantity | nature | γ=0 | γ=0.01 | γ=0.05 | γ=0.1 |
|---|---|---|---|---|---|
| `Σ_t v_t` (all 128 values) | **linear** functional of x | **1.000** | **1.000** | **1.000** | **1.000** |
| position of the far marker | **linear** in the marker channel | 0.995 | 0.995 | 0.995 | 0.992 |
| position of the near marker | **linear** | 0.995 | 0.995 | 0.994 | 0.993 |
| **the task target `v_far+v_near`** | **BILINEAR** (value × marker) | **0.077** | 0.074 | 0.052 | 0.094 |
| `v` at the far marker | bilinear | 0.133 | 0.134 | 0.086 | 0.150 |
| `v` at the near marker | bilinear | 0.006 | 0.003 | 0.013 | 0.013 |

**Two things follow, and they close the case.**

1. **The history is not lost — at any γ.** A linear functional spanning all 128 tokens is recoverable at **R² = 1.000 even at γ=0.1**, where the amplitude surviving from token 0 is 0.001. Attenuation in a noiseless linear system is not information loss; the decoder simply re-weights. The "memory half-life ≈ 28 tokens" framing describes an *amplitude*, not an information horizon, and the two are not the same thing. (Individual-token deconvolution is ill-conditioned so its R² is low — 0.02–0.4 — but it too shows **no γ-dependent horizon**: the very first token, at lag 127, decodes at R² **0.84 at γ=0 and 0.66 at γ=0.1**.)
2. **The missing object is the *conjunction*.** The state knows every value and knows where the markers are, but cannot pair them. That is exactly what `p += W_in x_t` cannot do: it is unconditionally linear in the token, so the register accumulates `Σ_t a·v_t` and `Σ_t b·m_t` *separately* and never their product. The GRU (gates), the selective SSM (input-dependent Δ) and attention (softmax QK) all have an explicit multiplicative input path, and all three solve adding to ≤0.001.

**Confirmation (the exploratory arm, §5): a multiplicative gate restores the conjunction, and γ has nothing to do with it.** With `write_mode="gated"` — same probe, same untrained setting, matched 40 k params — R² for the task target jumps **0.077 → 0.960 at γ=0** and **0.052 → 0.988 at γ=0.05**. It is *highest at the most dissipative setting*. After 1200 training steps the trained gated block's state decodes the target at **R² = 0.999** and each marker's value at **0.985**, against **0.021 / 0.009** for the trained linear block; and the learned gate is measurably marker-selective (mean gate output **0.109** on a non-marker token vs **0.407** on a marker token at the same value — a 3.7× multiplicative selectivity, learned).

Figure: `gamma-read-sweep/memory_probe.png`. Raw: `memory_probe.json`, `memory_probe_gated.json`, `memory_probe_trained.json`.

## 3 Item 2 — trajectory read vs endpoint read

Implemented as the Prop-11 **endpoint-fiber** read: `y_t = W_out[q_1;p_1;…;q_K;p_K]` over the `K=clu_steps` intra-token sub-steps, versus the shipped `y_t = W_out[q_K;p_K]`.

> ⭐ **First, a structural finding that changes how the shipped harness should be read.** At `clu_steps=1` the fiber has **one element**, so the two read modes are **the same map, bit-exactly** — same `w_out` shape, same key, same output. The shipped harness ran `clu_steps=1`. **Its read-mode axis is therefore degenerate by construction, not merely measured flat**, and no (γ × read-mode) table at `clu_steps=1` can contain information. Proven, pinned by a bit-exact unit test, *and* measured end-to-end below (spread exactly 0.0000 across two independent trained cells per γ). The 2-D table is only meaningful at `clu_steps>1`, which is why it was run at `clu_steps=2`.

### adding_T128, clu_steps=1  ⚠ read modes are the SAME MAP at clu_steps=1 (fiber of one)
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.1825 ± 0.0032 | 0.1831 ± 0.0060 |
| trajectory | 83 | 0.1825 ± 0.0032 | 0.1831 ± 0.0060 |
| **traj − end** | | +0.0000 | +0.0000 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0000

### adding_T128, clu_steps=2
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.1827 ± 0.0036 | 0.1820 ± 0.0036 |
| trajectory | 53 | 0.1832 ± 0.0038 | 0.1819 ± 0.0039 |
| **traj − end** | | +0.0004 | -0.0001 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0006

### parity_T64, clu_steps=1  ⚠ read modes are the SAME MAP at clu_steps=1 (fiber of one)
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.5334 ± 0.0053 | 0.5233 ± 0.0051 |
| trajectory | 83 | 0.5334 ± 0.0053 | 0.5233 ± 0.0051 |
| **traj − end** | | +0.0000 | +0.0000 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0000

### parity_T64, clu_steps=2
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.5337 ± 0.0056 | 0.5345 ± 0.0048 |
| trajectory | 53 | 0.5253 ± 0.0086 | 0.5321 ± 0.0051 |
| **traj − end** | | -0.0084 | -0.0023 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = -0.0061

### mqar_T128_kv4, clu_steps=1  ⚠ read modes are the SAME MAP at clu_steps=1 (fiber of one)
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.3864 ± 0.0032 | 0.3464 ± 0.0080 |
| trajectory | 83 | 0.3864 ± 0.0032 | 0.3464 ± 0.0080 |
| **traj − end** | | +0.0000 | +0.0000 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0000

### mqar_T128_kv4, clu_steps=2
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.2952 ± 0.0069 | 0.3298 ± 0.0092 |
| trajectory | 53 | 0.2881 ± 0.0089 | 0.3034 ± 0.0092 |
| **traj − end** | | -0.0072 | -0.0264 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0192

**γ grid reduced to the two ends {0, 0.05} for this item** (stated per the no-hidden-scoping rule): the registered prediction R2 is the *interaction* `(traj−end)|_{γ=0} − (traj−end)|_{γ=0.05}`, which needs exactly the two endpoints; Item 1 already supplies the full 7-point γ row at (endpoint, K=1). This kept a 24-cell table affordable on a contended machine.

**Result: the trajectory read never helps, and on two of three families it hurts.**

| family (K=2) | traj − end @ γ=0 | traj − end @ γ=0.05 | interaction | seed σ | verdict |
|---|---|---|---|---|---|
| adding (MSE ↓) | **+0.0004** (worse) | **−0.0001** (better) | **+0.0006** | 0.0037 | null; both ≈10× below σ |
| parity (acc ↑) | **−0.0084** (worse) | **−0.0023** (worse) | −0.0061 | 0.0060 | fiber is worse at both γ |
| MQAR (acc ↑) | **−0.0072** (worse) | **−0.0264** (worse) | **+0.0192** | 0.0085 | fiber is worse at both γ |

- **The fiber read loses on the parameter trade it has to win.** At matched 40 k block params it costs state width (`d_clu` 83 → 53 at K=2, → 31 at K=4) and buys nothing on any family. That trade is the honest interpretation of the whole item: *the intra-token rollout carries no information the endpoint does not, at least not enough to pay for the register width it consumes.*
- **The task's registered prediction — "the trajectory read is the one that benefits from γ→0" — is not supported, with one marginal caveat I will not overstate.** On adding the interaction is `+0.0006`, 33× below my 0.02 threshold. On parity it is `−0.0061`, the *wrong sign*. On **MQAR** it is `+0.0192` — the *right* sign and only just under my registered 0.02 threshold — so I score R2 as **marginal-pass rather than clean-pass**. But note what that number is made of: the fiber read is worse than the endpoint read at *both* γ (−0.0072 and −0.0264); the "interaction" is only that it is *less bad* at γ=0. **A read mode that loses everywhere cannot be rescued by the shape of how it loses.** If a follow-up wants to press this, it should run the full γ grid on MQAR at K≥2 with more seeds — this is a 2-point, 3-seed estimate with σ=0.0085.
- ⚠ **A side finding worth flagging: `clu_steps=2` badly hurts MQAR.** At γ=0, endpoint accuracy falls **0.3864 (K=1) → 0.2952 (K=2)**, a drop of 0.091 = 13σ, at identical parameters and identical γ. Adding and parity are unaffected (0.1825→0.1827, 0.5334→0.5337). More intra-token integration *destroys* recall specifically. This was not a pre-registered question and I have no mechanism for it beyond the obvious (more rotation per token scrambles the association), but it is a clean, large, reproducible effect and it reinforces Item 3's verdict.

## 4 Item 3 — `clu_steps`, with the cost

Run at the best γ from Item 1 on adding (γ=0.02; note that "best" is within noise — the point is not that 0.02 is special).

| `clu_steps` | `d_clu` | memory half-life (tok) | best lr | **MSE (adding, γ=0.02)** | **ms/step** | **× wall-clock** | fwd FLOPs | × FLOPs |
|---|---|---|---|---|---|---|---|---|
| **1** (shipped) | 83 | 68.6 | 3e-4 | **0.1816 ± 0.0043** | 36.4 | **1.00×** | 8.47 M | 1.00× |
| **2** | 83 | 34.3 | 3e-3 | **0.1824 ± 0.0031** | 65.5 | **1.80×** | 8.52 M | 1.01× |
| **4** | 83 | 17.2 | 3e-4 | **0.1817 ± 0.0034** | 133.0 | **3.66×** | 8.62 M | 1.02× |
| 1, trajectory read | 83 | 68.6 | — | (= endpoint, identity) | 36.5 | 1.00× | 8.47 M | 1.00× |
| 2, trajectory read | 53 | 34.3 | — | see §3 | 60.0 | 1.65× | 9.01 M | 1.06× |
| 4, trajectory read | 31 | 17.2 | — | not run | 100.0 | 2.75× | 9.49 M | 1.12× |

**⚠ A confound I registered in advance and that the code makes real:** friction is applied `p ← (1−γ)p` once per Verlet **sub-step**, so `clu_steps=K` damps K times per *token* and the memory half-life shortens by exactly K (68.6 → 34.3 → 17.2 tokens here). More integration therefore *buys more dissipation* at fixed γ. It made no difference — which is itself further evidence that dissipation is not the binding constraint. (Anyone sweeping `clu_steps` again should sweep γ *per token*, not per sub-step, or the two knobs are confounded.)

**The honest multiple: `clu_steps=4` costs 3.66× the wall-clock of `clu_steps=1` and buys 0.0001 MSE (0.1817 vs 0.1816) — i.e. nothing, at nearly 4× the price.** Cost is essentially **linear in `clu_steps`**, and, exactly as in w20, **wall-clock-dominated rather than FLOPs-dominated**: XLA's FLOP count moves only 1.00× → 1.02×, because each ∇V is cheap but they serialise inside the token scan. The fiber read is *cheaper* per step than the endpoint read at the same K (2.75× vs 3.66× at K=4) purely because the matched-parameter search shrank `d_clu` to 31 to pay for the wider `w_out`.

*Cost measured round-robin interleaved (5 rounds × 25 steps, median over rounds), on a quiet machine — load average 8.3, **IQR ≤ 2.7 ms**. A second run under load 16 gave 1.00× / 1.82× / 3.84×, so the ratios are contention-robust.*

## 5 ⚠ EXPLORATORY ARM (outside the pre-registered grid, labelled as such)

**What.** One knob, `write_mode`: `"linear"` = the shipped `p += W_in x_t`; `"gated"` = `p += (W_in x_t) ⊙ σ(W_gate x_t)`. Adding problem T=128 only, γ ∈ {0, 0.02, 0.05}, 3 seeds, **the same shared slot and the same 40 k block-param budget** (the search pays for the gate by shrinking `d_clu` 83 → 65).

| γ | linear write (pre-registered arm) | **gated write (exploratory)** | ratio |
|---|---|---|---|
| 0 | 0.1825 ± 0.0032 | **0.0085 ± 0.0060** | 21× |
| 0.02 | 0.1816 ± 0.0043 | **0.0028 ± 0.0007** | **65×** |
| 0.05 | 0.1831 ± 0.0060 | 0.0154 ± 0.0082 | 12× |

Reference levels: no-mixing control **0.1825**; GRU **0.0007**; SSM **0.0008**; attention **0.0001**.

**Three things to say about this, in order of importance.**

1. **It is not a CLU win and must not be reported as one.** Every baseline in the harness *already has* input-conditioned multiplication. Adding it to the CLU is levelling the architecture, not beating anyone: the gated CLU at 0.0028 is still **4× worse than the GRU and 28× worse than attention** on this family. What it establishes is *the cause of the failure*, which is the thing the task asked for.
2. **It converts the negative result into a positive, testable design statement.** "The CLU cannot integrate" is imprecise. The measured statement is: *a CLU whose write current is linear in the token cannot represent input-conditioned selection, and that — not dissipation — is why it sits at the no-mixing floor on the adding problem.* The probe (§2) and the ablation agree, at init and after training.
3. **γ becomes a real knob only once the primitive can do the task.** In the linear arm the γ spread was 0.0023 (0.5σ, noise). In the gated arm it is **0.0126 across the same three points, with the optimum at γ=0.02** — i.e. near the derived γ\* = 0.0108 — and γ=0.05 clearly worse (5.5×). **The γ\* prediction was untestable in the shipped configuration** because the model was not solving the task at any γ. ⚠ Three γ points, seed σ up to 0.0082 (γ=0 seeds 0.0026/0.0168/0.0060); treat the γ-optimum as suggestive, not measured.

**What this arm does NOT license.** No full-family run, no MQAR/parity, no LR-rescue pass, no baseline re-run, no pre-registration. It is a 3-cell diagnostic. **Follow-up needed: a properly pre-registered "gated write current" task** running all three families with the symmetric rescue pass, stating that the gate is a category-(b)-adjacent change (it imports a capability the baselines already have, so it is levelling — but that argument must be made explicitly, not assumed).

## 6 Item 4 — the corrected three-family table

**The pre-registered condition for producing one is "iff a better configuration is found". Within the pre-registered grid, it was not.** Two of three families move by less than the seed noise; MQAR moves by +0.040 at γ=0, which is real but changes no ranking and no verdict. I therefore did **not** re-run the three families with the LR-rescue pass at a "winning" configuration, because there is no winning configuration to justify the compute — and the honest table is the one below.

| family (metric) | shipped CLU (γ=0.05) | **best over the whole γ grid** | Δ | best baseline | verdict |
|---|---|---|---|---|---|
| adding T=128 (MSE ↓) | 0.1825 | **0.1816** (γ=0.02) | −0.0009 (0.2σ) | attention 0.0001 | **still exactly the no-mixing control floor (0.1825).** 0 of 3 stands |
| parity T=64 (acc ↑) | 0.5380 | **0.5368** (γ=0.1) | −0.0012 (**worse**) | GRU 1.0000 | **still chance.** 0 of 3 stands |
| MQAR T=128 kv=4 (acc ↑) | 0.3464 | **0.3864** (γ=0) | **+0.0400** (5.7σ) | attention 0.9945 | improved, still 4th of 5. Ranking unchanged |

**Recommended default change (one line, and it is free):** for MQAR-like recall the CLU block should run **γ=0**, which is +0.040 accuracy at identical cost and identical parameters. I have **not** changed `clu_gamma`'s default (0.05) on my branch — the default is what makes the w20 numbers reproduce, and changing it is the Hub's call. See *Proposed handover updates*.

## 7 Pre-registration scorecard (`PREREG.md`, written before any sweep cell ran)

I nominated **H2** (the primitive genuinely cannot integrate; γ is not the limiter) over **H1** (dissipation artifact) in advance, with three stated reasons.

| # | prediction | measured | verdict |
|---|---|---|---|
| A1 ⭐ | adding: **no sharp transition**; min over the γ grid ≥ 0.15 | **0.1816** | ✅ |
| A2 ⭐ | adding: **no cell reaches the near-marker level 0.0833** | min 0.1816 | ✅ |
| A3 | adding: spread over the grid ≤ 0.035 | **0.0023** (0.5σ) | ✅ |
| A4 | adding: MSE(γ=0) ∈ [0.15, 0.20] | **0.1825** | ✅ |
| A5 | (H1 branch: transition at γ∈[0.005,0.02], MSE ≤0.05 below it) | **not triggered** | — |
| A6 | ≤1 of 3 seeds diverges at γ ≤ 0.001 | **0 of 18** final-run seeds at γ ≤ 0.001; **0 divergences and 0 NaNs anywhere in all 48 cells** (144 final runs + LR selection) | ✅ |
| P1 ⭐ | parity: accuracy ≤ 0.60 at every γ | max **0.5368** | ✅ |
| P2 | parity: best-γ − shipped ≤ +0.06 | **−0.0012** (no improvement at all) | ✅ |
| P3 | parity: the finite-memory model over-predicts at every γ | predicts 0.71–1.00, measured 0.52–0.54 | ✅ |
| M1 | MQAR: best-γ ∈ [0.34, 0.46] | **0.3864** | ✅ |
| M2 ⭐ | MQAR: best-γ < 0.486 (the GRU) — no ranking change | 0.3864 | ✅ |
| M3 | MQAR: monotone improvement as γ falls, between +0.02 and +0.11 | **+0.0400, perfectly monotone in all 7 points** | ✅ |
| R0 ⭐ | read modes are the SAME MAP at `clu_steps=1`; measured spread exactly 0.0000 | **0.0000** (bit-exact in unit test *and* end-to-end in 3 trained cell pairs) | ✅ |
| R1 | traj − end on adding at K>1: \|Δ\| ≤ 0.02 | **+0.0004 / −0.0001** | ✅ |
| R2 ⭐ | **no** γ × read-mode interaction (< 0.02); the task's "trajectory benefits from γ→0" does **not** hold | adding **+0.0006** (opposite sign) · parity **−0.0061** (wrong sign) · **MQAR +0.0192** (right sign, just under the 0.02 threshold) | ◐ **marginal** on MQAR — but the fiber read is *worse* at both γ there, so the interaction is only "less bad at γ=0" |
| R3 | trajectory read shrinks `d_clu` at matched params | 83 → 53 (K=2) → 31 (K=4) | ✅ |
| S1 | `clu_steps` wall-clock ≈ 1.0 / 1.7 / 3.1 ×, each within ±0.5 | **1.00 / 1.80 / 3.66** (quiet machine) | ◐ K=2 ✅ (0.10 off), **K=4 ❌** (0.56 outside): cost is *linear* in `clu_steps`, not sub-linear |
| S2 ⭐ | `clu_steps` 1→4 improves adding MSE by ≤ 0.02 | **0.0001** (0.1816 → 0.1817) at **3.66× the cost** | ✅ |
| F1 ⭐ | **no better configuration found; 0 of 3 stands** | confirmed; MQAR +0.040 changes no ranking | ✅ |

**16 clean passes, 2 marginal (R2 on MQAR only; S1 at K=4), 1 not triggered, 0 falsified.**

Two honest caveats on that record, because a 16/19 scorecard is exactly the kind of thing that should be read sceptically:
- **S1's miss is a cost multiple, not a scientific claim**, and it misses in the direction *unflattering* to CLU — integration cost is fully linear in `clu_steps`, not sub-linear as I hoped.
- **R2's marginal on MQAR is the only place a pre-registered null was close to failing**, and I have flagged it rather than rounding it into the ✅ column. It should be settled by a follow-up with the full γ grid at K≥2 and more seeds, not by me re-reading a 2-point estimate.

**No prediction was adjusted after seeing data**, no cell was dropped, and nothing was run outside the pre-registered grid except the explicitly-labelled §5 arm. The one strong prediction I would have most liked to be wrong about — A1/A2, the dissipation transition — was the first thing measured and it was flat.

## 8 How I verified

- **Full suite: `526 passed` in 445 s** (`pytest tests/ -q -p no:randomly --no-cov`), including **11 new tests** in `tests/test_primitive_harness.py`. `ruff check chlu/ tests/` → **All checks passed.**
- **New tests pin exactly the things that could have faked this result:** the `clu_steps=1` read-mode identity (bit-exact — a difference there would be a bug, not a finding); fiber-read causality (perturbing token *t* must move output *t* and nothing before it); `det J = 1` at γ=0 (so the γ=0 end of the grid really is the conservative map); the `2ln2/γ` half-life formula matching what the integrator does; `_sweep_cell` restoring config so cells cannot leak; **and that the gated write is bit-identical to shipped when off.**
- **γ is provably live** (a null result's first failure mode is a disconnected knob): the final-token momentum norm falls monotonically 39.66 → 38.33 → 33.77 → 29.44 → 23.57 → 15.22 → 10.02 across the γ grid at fixed seed and input.
- **The default block is bit-identical to `main`** after my edits — `make_block("clu", 64, 83, key=PRNGKey(0))` output max-abs-diff **0.0**, `w_in` and `log_mass` array-equal. (The gate key is folded in from `k2` rather than taken from a 4-way split precisely so this holds; a 4-way split would have silently re-randomised every published w20 CLU cell.)
- Every number in this report is generated from the result JSONs by `.claude/scratch/gamma-read-sweep/make_tables.py`; nothing is transcribed by hand.
- **Artifacts** in `.claude/outputs/gamma-read-sweep/`: `PREREG.md`, `item1_gamma.json`, `item2_read.json`, `item3_steps.json`, `exploratory_gated_write.json`, `clu_steps_cost_benchmark.json`, `memory_probe{,_gated,_trained}.json`, `gamma_sweep.png`, `memory_probe.png`. Drivers in `.claude/scratch/gamma-read-sweep/`.

### What I found by running it

1. **The read-mode axis of the shipped harness is degenerate, not flat.** I nearly reported a "measured 0.0000 spread" as a finding before realising it is an identity at `clu_steps=1`. It is now a proof plus a bit-exact test, and the 2-D table was moved to `clu_steps=2` where the axis exists.
2. **My first memory-probe run silently omitted the gate.** `states_at_readout` re-implements `CLUBlock.__call__` to extract internal states, and I forgot the `⊙ σ(W_gate x)` line — so the trained gated model probed at R²=0.019 for the target while actually scoring MSE 0.0026, an obvious contradiction that flagged the bug. With the gate applied it reads **0.999**. Had I not cross-checked the probe against the model's own loss, I would have reported "even the gated model doesn't encode the target", which is the exact opposite of the truth.
3. **Quick-mode param matching hits the width floor** (`width_search_lo=4`) at `target_block_params=4000` with a K=2 fiber read, giving 30 % param error. Harmless (smoke only, and the real budget matches to ≤0.50 %), but worth knowing before anyone quotes a `--quick` number.

## 9 Open questions / follow-ups / risks

1. ⭐ **The gated write current needs a pre-registered follow-up** (§5): all three families, symmetric LR-rescue, explicit fairness argument, and the honest framing that it imports a capability every baseline already has. Until then it is a 3-cell diagnostic, not a result.
2. **MQAR at γ=0 is a free +0.040** and the only γ-sensitive family. Worth combining with the capacity axis (`primitive-harness` §7.2, the kv-sweep where CLU crosses above the GRU) — **that kv sweep was run at γ=0.05, so its CLU numbers are ~0.04 pessimistic and its crossover point may move in CLU's favour.** Cheap to re-check; I did not, because it is outside my three families.
3. **Budget-bounded.** 40 k block params, 1200 steps, 2 layers. A null at this budget is a null at this budget; the adding null is nevertheless robust (7 γ × 3 seeds × 3 LRs all at the control floor, plus an information-theoretic explanation that does not depend on budget).
4. **The information probe is at initialisation** (plus one trained pair). Training reshapes `V_θ`; it cannot make the per-token write multiplicative, which is the claim — but a full trained-probe sweep across γ would harden it.
5. **`clu_steps` and γ are not orthogonal** in the current code (K damps K times per token). If anyone sweeps `clu_steps` again, sweep `γ_per_token` rather than `γ_per_substep`, or the two knobs are confounded.
6. **Unexplained side finding worth a look: `clu_steps=2` costs MQAR 0.091 accuracy at γ=0** (0.3864 → 0.2952, 13σ, identical params and γ), while adding and parity are unmoved. More intra-token integration selectively destroys *recall*. Not pre-registered, no mechanism offered, but large and clean.
7. **R2's marginal on MQAR** (§3) is the one pre-registered null that came close to failing; it wants the full γ grid at K≥2 with ≥5 seeds to settle.

## Git footprint

- **Branch** `agent/experiment-engineer/gamma-read-sweep`, off local `main` @ `31c3e15`. Rebase onto `main` = no-op (base unmoved); `origin/main` untouched (§7.21).
- **Worktree** `../CHLU-gammaread` (main venv reused via `PYTHONPATH`, no `uv sync` — JAX stayed at 0.9.0 per §4). Branch ref verified from the main repo. Remove with `git worktree remove ../CHLU-gammaread` after review. **No collision:** the main checkout was clean and on `main` throughout; I never edited it.
- **Commits (2, verified from the MAIN repo per the w4 lesson):** `af02a22` *CLUBlock: make gamma a swept knob; add fiber read + gated write* · `834b9f3` *add the w21 gamma/read-mode/clu_steps sweep driver + CLI*
- **Files touched (4 + 1 test file, all surgical):** `chlu/core/blocks.py` (CLUBlock: `read_mode`, `write_mode`, corrected the docstring carrying the retracted w19 justification), `chlu/config.py` (+4 fields in `ExperimentPrimitiveHarnessConfig`, all defaulting to shipped behaviour), `chlu/experiments/exp_primitive_harness.py` (+`run_gamma_read_sweep`, `_sweep_cell`, `_write_sweep`, `memory_half_life_tokens`, `benchmark_clu_cells`, CLI flags), `chlu/cli/experiment_cmd.py` (+2 parser flags, +1 branch), `tests/test_primitive_harness.py` (+11 tests, 1 docstring/assertion correction). **No shared physics, training, plotting or data code touched.**
- **Commands:** `pytest tests/ -q -p no:randomly --no-cov` → 526 passed; `ruff check chlu/ tests/` → clean; `python -m chlu.experiments.exp_primitive_harness --gamma-sweep [--sweep-items …] [--quick]`, `--clu-steps-benchmark`.
- **Unresolved conflicts:** none.

## Proposed handover updates (for the Hub)

1. **§7 — CLOSE the known issue opened by `primitive-harness` §7.1 / handover-update 2.** *"The CLU block does not learn long-range integration in the drop-in slot; candidate cause: γ=0.05 gives a ~28-token half-life; needs a pre-registered γ sweep."* **Run and refuted.** Replace with: *the failure is not dissipation — the adding MSE is flat to 0.5σ across γ ∈ [0, 0.1] — it is that `CLUBlock`'s write current `p += W_in x_t` is unconditionally linear in the token and so cannot represent an input-conditioned (value × marker) conjunction. Measured: the readout state decodes Σ_t v_t at R²=1.000 and the marker positions at R²≈0.99 at every γ, the task target at R²≈0.02–0.09. A multiplicative input gate at matched parameters takes the MSE 0.1816 → 0.0028 (65×).*
2. **§6 ground truth — amend the w20 entry.** "CLU wins 0 of 3 families" **stands and is now stronger**: it survives a 7-point γ sweep, a read-mode cross, and a `clu_steps` sweep, all at 3 seeds. Add: *MQAR is γ-sensitive (γ=0 is +0.040 over the shipped γ=0.05, perfectly monotone); adding and parity are not (spread 0.5σ and non-monotone).*
3. **§3 / config defaults — a decision for the Hub, not taken by me.** `ExperimentPrimitiveHarnessConfig` gains `clu_read_mode="endpoint"`, `clu_write_mode="linear"`, `clu_gamma_sweep`, `clu_read_mode_sweep`, `clu_steps_sweep`. **All defaults preserve shipped behaviour and w20 reproduces bit-for-bit.** `clu_gamma` is left at 0.05 for that reason, **but the measured best value for MQAR is 0.0** — if the Hub prefers correctness over reproduction, that is the one default worth changing, and `test_clu_block_gamma_is_dissipative_by_default` (which now pins reproducibility, not physics) would need its assertion updated with it.
4. **Claims-matrix / text warning — the retracted w19 justification has a third site.** Beyond the two already logged, `primitive-harness`'s **concession #2** ("γ>0 is REQUIRED for a readable state, 1.000 at γ=0.02 vs 0.813 at γ=0") is inherited from it. Corrected in code on my branch; **any paper or claims-matrix text repeating it needs the same fix.** The correct statement is `address-space-dimension-scaling` §4's: *retrieval of the stored **value** requires dissipation; addressing does not* — and in a *trained* driven block, neither does the sequence task: γ=0 ran 9 full-length runs in Item 1 with zero divergences, and nothing diverged anywhere in 48 cells.
5. **⚠ A number elsewhere is now known to be pessimistic.** `primitive-harness` §1b's **capacity axis** (the one genuinely CLU-favourable result, where CLU crosses above the GRU at kv≈8) was run at γ=0.05. MQAR at γ=0 is +0.040. **Those CLU numbers are a lower bound, and the crossover may sit at a lower kv than reported.** Worth a cheap re-check before that result is published — it is the programme's strongest CLU evidence and it is currently understated.
6. **Methodology worth adopting:** the **linear information probe** (ridge-decode a task's required quantity from the state, contrasting a *linear* functional of the input against the *bilinear* one the task needs). It localised a failure that three separate sweeps could only describe, in about 90 seconds of compute, and it is architecture-agnostic.
