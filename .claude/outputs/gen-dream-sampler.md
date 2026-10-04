# gen-dream-sampler — experiment-engineer report

**Task + acceptance criterion:** implement conditional dreaming by clamped-well Langevin on the store's
landscape, decode through the ONE shared frozen decoder, and emit an audit trail that lets a referee
convict the sampler — **acceptance is the CONTRAST (full rollout vs the zero-step control), not the accuracy.**

**Status: done.** ⛔ **The pre-registered falsifier FIRED. The dial's answer is NO: clamping the wells a
request addresses does NOT demonstrably steer what comes out.** Headline contrast
`full − zero_step = +0.0095 ± 0.0082` (2/3 seeds positive) against a pre-registered threshold of `+0.03`
and an inert line of `+0.01`. ⭐ It is the **informative** version of that failure, not the vacuous one —
see the three-line summary below.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 5 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⛔ **`store_plan.json`'s `labels` vector indexes ENCODER CHANNELS, and every bridge seed trained its
>    own encoder — so "every later ladder rung loads this file VERBATIM" is well-defined on ONE seed only.**
>    I fell into this trap myself and caught it post-run (§5b). The plan records
>    `encoder_param_hash = 98d3260…` = **seed 0's** encoder; seeds 1 and 2 are `fc05e2c5…` / `154d4986…`.
>    **Measured cost: the store ceiling reads 0.5579 / 0.5628 on seeds 1–2 under the plan's labels vs
>    0.6208 / 0.6243 under their own, and the 3-seed spread collapses 13× (±0.0279 → ±0.0021).**
>    ⛔ **This is a ladder-contract defect, not just mine: the plan's wording must say that only
>    `K`/`n_modes`/`D`/`store_bytes` are portable and the grouping must be re-derived per seed.**
>    **Owner needed before R1…R5.** (§5b)
> 2. ⭐ **`gen-baselines`'s byte-match target of `524,288 B` is incomplete for this arm.** The landscape
>    carries an inter-well PMI coupling field of **163,840 B** in addition to the store ⇒ total **688,128 B**.
>    The read channel is **unchanged at 30 bits** (R20's design constant untouched). (§3, §10)
> 3. ⭐ **`gen-cell-run` cannot read a full-request contrast.** On the cell-native `full` protocol the
>    addressing head clamps **4.42 of 5 wells** (confidence 0.756), the relaxed set is degenerate (0.58
>    wells) and `Δ = −0.0005 ± 0.0036` **for a reason that has nothing to do with the sampler**. Registered
>    in advance as CH-3. The cell needs a ruling: is B/C scored on the clamp alone, or on a partial-request
>    protocol? (§4)
> 4. ⚠ **`store_removed` is a correct but NON-discriminating ablation on the A cell.** It collapses to one
>    code configuration and a constant image (0.1213 ± 0.0139, P4 held) because it ablates the **addressing**,
>    not the **dreaming**. The ablation that isolates dreaming is `no_coupling`, and it carries the finding. (§6)
> 5. ⚠ **`UF_HIDDEN` on harness-written JSON under `.claude/`** (handover §7.12 family) killed the run's own
>    aggregate stage with `PermissionError` **reading its own output**. `chflags -R nohidden` is the fix.
>    Will bite every gen spoke. (§11)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)

- **Dial:** *isolation* — whether clamping the wells a request addresses actually steers what comes out.
- **Laundering control:** ⭐ **the zero-step rollout** — decode the clamp directly, no Langevin, same clamp,
  same seed. It ran on **every** number below as a full column, never a footnote, plus a second,
  **stronger** trivial substitute (`prior_mode`: relaxed wells at their most-likely mode, still no Langevin).
- **Falsifies:** the zero-step control matches the full rollout ⇒ the sampler is inert and we are decoding
  the clamp. ⛔ **THIS FIRED.** Also filed as falsifying: flat energy traces, or a chain that never leaves
  its initialisation — ⭐ **neither fired**, which is exactly what makes the result informative.
- **Does NOT falsify:** blurry or low-fidelity samples in absolute terms. Every accuracy below is reported
  against **both** ceilings (R23.5) and never absolutely.

### ⭐ The three lines the Hub needs

1. **The chain genuinely samples** — 95.3 % of rollouts change a code, 499 of 512 distinct code
   configurations (vs 278 for the clamp alone), acceptance 0.952, energy moves by 1.0× its own scale, the
   softmax is not saturated (`max π` = 0.020). ⛔ "Inert" here does **not** mean "broken".
2. **It moves exactly one thing, and it moves it tightly: `shape`, the ONLY compositional factor (R16),
   `Δ = +0.0215 ± 0.0033`, 3/3 seeds, 6.5× its own spread.** Every other factor is inside its spread, which
   is why the six-factor aggregate washes out to `+0.0095 ± 0.0082`.
3. ⛔ **But the separability control gets essentially all of it.** `no_coupling` (λ = 0, where the clamp
   **provably** cannot reach a relaxed well) scores `Δshape = +0.0200 ± 0.0181`, so the
   **steering-attributable part is `+0.0015 ± 0.0204` — zero.** ⇒ the `shape` gain is the chain **settling
   onto a mode**, not the clamp **steering** anything. **That is the dial's answer, and it is a NO.**

## FLAG-PROVENANCE TABLE (mandatory, protocol §5)

| | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-dream-sampler` — `5bd25e5`, `4d5a35c`, `9fb80e5`, `48f0a6d`, `973e740` |
| base | `main @ 9a4457d` |
| dataset / pretext / **condition** | `shapes3d` / `recon` / **`recomb2element`** (Montero). ⛔ No `cls`, no dSprites, **no B, no C, no kill-rule verdict** |
| seeds | **{0, 1, 2}**; spread = population std over seeds (ddof=0) |
| ⭐ **decoder** | **`v1_capacity_schedule`** (gen-decoder-lift's winner), from `winner/decoder_v1_s{0,1,2}.pkl`, **param hash AND recipe name asserted** per seed: `01ef91f75eeaae7b…` / `001011bc603a7cb9…` / `2d7eea75add77044…`; the encoder each was fitted against is asserted against the bridge manifest too. ⛔ **NOT** the `dec_recipe="baseline"` default |
| probe / encoder | bridge's frozen `probe.pkl` / `encoder.pkl`, both hash-asserted; cache re-encode check **bit-identical (max abs Δ = 0.0)** every seed |
| feature cache | seed 0 `scope="full"`; seeds 1–2 `scope="used"` (9,288 rows — the R19 non-uniformity), coverage extended through the frozen encoder (32,199 rows/seed) |
| store | `K=5`, `n_modes=64`, `D=2048`, `store_bytes = 524,288 B`, `bits_per_read = 30.0` — **all pinned from `store_plan.json`**; `n_store_train=32768`, `train_steps=1200` (`gen_store` defaults) |
| ⭐ **grouping** | **`grouping_source="per_seed"`** — re-derived by the plan's own procedure on each seed's own train activations at the plan's `K`, and **verified bit-identical to the plan on seed 0** (whose encoder wrote it). Well sizes `[303,305,573,501,366]` / `[397,300,307,300,744]` / `[283,283,390,616,476]`. ⛔ See reconciliation item 1 |
| ⭐ coupling | inter-well **PMI field** from 32,768 train codes, `coupling_strength = 1.0`, smoothing 1.0 — **+163,840 B, EXTRA to `store_bytes`** |
| addressing head | naive-Bayes / PoE over per-(factor,value) code histograms, smoothing 1.0, `clamp_threshold = 0.5` |
| sampler | **`mala` — the Metropolis correction IS applied** (primary); `ula` (same proposal, correction removed) and `uld` (repo `langevin_step`, **`noise_mode="fdt"`**, `m_eff = 1`) as companions. ⛔ `legacy` noise never used (§7.22) |
| steps / step size | `steps = 200`; step size **tuned per protocol to an ACCEPTANCE target** (0.7 ± 0.1) on **train** requests: **0.1** (`full`) / **0.025** (`partial`), identical on all 3 seeds |
| ⭐ `tau` (calibrated) | `tau_calibration="mode_distance"`: `1.0 → 0.25` × the store's own median inter-mode distance ⇒ absolute **11.13→2.78 / 11.22→2.81 / 11.56→2.89** |
| ⭐ `T` (calibrated) | `temp_calibration="energy_scale"`: `0.05 → 0.005` × median abs E(x₀) ⇒ `full` **1.518 / 1.496 / 1.697** → ÷10; `partial` **0.432 / 0.521 / 0.197** → ÷10 |
| requests | `n_dream = 512` rows of the bridge's own `eval_a` (**A split only**); `init_mode = "centroid"`; `request_drop_factors = 2` (`partial`) |
| JAX / env | JAX **0.9.0**, **main venv** (protocol §4 — ⛔ no `uv sync` in the worktree), CPU only |
| wall clock | **1,545 s** for 3 seeds (seed 0 reused its store checkpoint; seeds 1–2 refit, 1,200 steps each) |
| reproducibility | ⭐ seed 0 reproduced **bit-identically** across three independent invocations (`0.3507 / 0.3310`, 4 dp); rollout keys from `zlib.crc32`, ⛔ never Python's per-process-randomised `hash()` |

---

## 1. Preconditions, checked mechanically first

| Check | Result |
|---|---|
| `git cat-file -e main:chlu/core/gen_store.py` | ✅ exit 0 |
| store artefact on disk | ✅ `gen-store-grouping/runs/shapes3d_recon/store_plan.json` (+ `store_bytes.json`, 3 seed manifests) |
| **delete-the-store delta non-trivial** | ✅ **`+0.4752`** vs the programme's near-zero line of 0.01 — **475× clear**. There is something worth sampling. Precondition **MET** |
| ⛔ the decoder trap | ✅ caught at the top of every seed: the run refuses to proceed unless the decoder's **hash and recipe name** are gen-decoder-lift's winner |
| ⛔ no-bypass on the dream path | ✅ `max abs Δ` vs an independent NumPy gather **= 0.0**, **30.0 bits/read**, all 3 seeds |
| worktree | fresh `../CHLU-gen-dream`; ⛔ none of the parked worktrees touched |

Read: `AGENT_PROTOCOL.md`, `gen-clu-phase1-rulings.md` (R1–R24), `gen-store-grouping.md`,
`exp_c_dreaming.py`, `integrators.py`, and — per R13 — **`handover_context.md` §7 Known Issues only**.

---

## 2. ⛔⭐ The finding the module is built around: a per-well store energy is SEPARABLE

Before any code: if the landscape is `E(x) = Σ_k E_k(x_k)`, the conditional law of the **relaxed** wells
given the **clamped** wells does not depend on the clamp **at all**. Conditional dreaming is then *provably*
inert — no sampler can steer an unaddressed well from a clamped one, and the zero-step control would match
the full rollout **by construction**. So the landscape carries two terms (both declared in `PREREG.md` §1
before any run):

1. the **within-well** mixture free energy over the store's own modes (`−τ² logsumexp_m(−d²/2τ²)`), and
2. ⭐ an **inter-well compatibility field** `J[k,l][m,m']` = the **pointwise mutual information of the
   empirical code co-occurrence on the TRAIN split** — a Potts/MRF over the `K` code variables, **learned
   from data, label-free**, never a function of the features at read time.

**Declared, not silent** (item 2): `K(K−1)/2 · M² · 4 = 163,840 B` on top of `store_bytes = 524,288 B`.
⛔ **It does not widen the read channel**: the decode is still
`x → address_codes → read_from_codes → decoder`, the encoder→decoder channel is the same **30 bits**
(`K·log₂M = 5·6`), `read_from_codes` never sees the features, and `assert_no_bypass` passed on the dream
path's own store on every seed.

⭐ **And the separability argument is MEASURED, not argued**: `coupling_strength = 0` (`no_coupling`) runs as
a full arm on every number (§4), and
`tests/test_gen_dream.py::test_separability_POSITIVE_CONTROL_lambda_zero_makes_the_clamp_irrelevant`
reproduces the failure deliberately — at λ=0 a *different* clamp leaves the relaxed coordinates
**bit-identical**; switching the coupling on breaks that.

---

## 3. ⭐ THE SAMPLER, DECLARED (the Head's sampler ruling)

- **Family:** overdamped Langevin on `x ∈ R^D`, clamped coordinates frozen by **masking the drift and the
  noise to the free block** — so the clamp is exact (clamp *drift* = 0.0 by construction) and the two
  Gaussian proposal densities differ only on the free block, which is what makes the MH ratio exact for the
  **conditional** target `p(x_free | x_clamped)`.
- **Step size** `0.1` (`full`) / `0.025` (`partial`), **tuned per protocol on TRAIN requests to an
  ACCEPTANCE target**, ⛔ never to an accuracy. Identical on all three seeds.
- **Steps** 200. **Noise convention** `x' = x − ε∇E + sqrt(2εT)·ξ`, T in energy units; the `uld` companion
  uses the repo's `langevin_step` at **`noise_mode="fdt"`** (§7.22 — ⛔ `legacy` never used).
- **Annealing:** `τ` and `T` both geometric, in **calibrated** units (§5a).
- ⭐ **Metropolis: YES — applied, in the primary sampler.**

### ⭐ …and "unnecessary at this scale" is MEASURED, not assumed

Du et al. (ICML 2023, arXiv:2302.11552) conclude that for composed energies **the sampler, not the model, is
the failure point**, and the critique bites hardest on **uncorrected** samplers over *products of diffused*
models. Rather than assert either way I ran `ula`: **the identical proposal with the accept/reject step
removed — one bit of difference, same keys, same seeds.**

| sampler | headline aggregate | vs MALA | measured acceptance |
|---|---|---|---|
| **`mala`** (corrected, primary) | **0.3485 ± 0.0141** | — | **0.9518 ± 0.0006** |
| `ula` (uncorrected, same proposal) | 0.3473 ± 0.0083 | **0.0012** | (same proposal) |
| `uld` (repo underdamped, `fdt`) | 0.3459 ± 0.0084 | 0.0026 | 0.948 |

⭐ **P6 HELD: 0.0012 ≪ 0.02.** The correction changes essentially nothing **because the measured acceptance
is 0.95** — at that acceptance the corrected and uncorrected chains are nearly the same chain, and the
**trace says so** rather than a paragraph saying so. Two structural reasons it does not bite here, offered
as argument and labelled as such: (a) nothing in this composition is a *diffused* model — the coupling is a
PMI field fitted on undiffused code co-occurrence, so the invariance Du et al. exploit is not being broken;
(b) the continuous relaxation is only a **search device over a 30-bit discrete code space** — what reaches
the decoder is `K` integers, so discretisation bias has one quantisation step in which to matter.
⛔ **I keep the correction anyway** (one extra gradient per step) and log acceptance every step in **every**
sampler, including the unadjusted ones, so the size of what `ula`/`uld` skip is on disk for a referee who
does not trust the paragraph above.

---

## 4. ⛔⭐ THE RESULT — full rollout vs the zero-step control, 3 seeds

⭐ **Headline protocol = `partial`**, chosen by the rule pre-registered in `PREREG.md` §4 ("the protocol
whose mean relaxed-well count is ≥ 1; ties go to `full`") — ⛔ not by outcome. On the cell-native `full`
protocol the addressing head clamps **4.42 ± 0.03 of 5** wells, leaving 0.58 relaxed: the degenerate case
CH-3 registered in advance.

| protocol · arm · sampler | full rollout | **zero-step** | **Δ (full − zero)** | seeds + | relaxed |
|---|---|---|---|---|---|
| ⭐ **`partial` · clu · mala** | **0.3485 ± 0.0141** | **0.3391 ± 0.0117** | ⛔ **+0.0095 ± 0.0082** | 2/3 | 3.45 |
| `partial` · clu · ula | 0.3473 ± 0.0083 | 0.3391 ± 0.0117 | +0.0083 ± 0.0042 | **3/3** | 3.45 |
| `partial` · clu · uld | 0.3459 ± 0.0084 | 0.3391 ± 0.0117 | +0.0068 ± 0.0052 | **3/3** | 3.45 |
| ⭐ `partial` · **no_coupling** · mala | 0.3429 ± 0.0076 | 0.3391 ± 0.0117 | **+0.0039 ± 0.0046** | 2/3 | 3.45 |
| `partial` · **store_removed** · mala | 0.1213 ± 0.0139 | 0.1213 ± 0.0139 | +0.0000 ± 0.0000 | 0/3 | 3.45 |
| `full` · clu · mala (⚠ degenerate relax set) | 0.5605 ± 0.0077 | 0.5611 ± 0.0041 | −0.0005 ± 0.0036 | 2/3 | 0.58 |
| `full` · clu · ula | 0.5642 ± 0.0035 | 0.5611 ± 0.0041 | +0.0031 ± 0.0030 | 2/3 | 0.58 |
| `full` · clu · uld | 0.5605 ± 0.0060 | 0.5611 ± 0.0041 | −0.0005 ± 0.0041 | 1/3 | 0.58 |
| `full` · no_coupling · mala | 0.5595 ± 0.0019 | 0.5611 ± 0.0041 | −0.0016 ± 0.0043 | 1/3 | 0.58 |
| `full` · store_removed · mala | 0.1196 ± 0.0151 | 0.1196 ± 0.0151 | +0.0000 ± 0.0000 | 0/3 | 0.58 |

**Per-seed headline:** Δ = **+0.0197 / −0.0003 / +0.0090**.

⛔ **P1 FALSIFIED.** `+0.0095` is below the pre-registered inert line of `+0.01`, far below the `+0.03`
threshold, is only **1.16× its own seed spread**, and the sign flips on one seed. **On this cell the full
rollout and the zero-step control are the same number: we are decoding the clamp.**

### ⭐ …but the aggregate is the wrong place to look, and R16 said so in advance

Per-factor, headline arm (`partial` · clu · mala):

| factor | full rollout | **Δ (full − zero-step)** | per-seed | Δ / spread |
|---|---|---|---|---|
| ⭐ **`shape`** (the ONLY compositional factor, R16) | 0.3538 ± 0.0226 | ⭐ **+0.0215 ± 0.0033** | +0.0261 / +0.0182 / +0.0203 | ⭐ **6.5×**, **3/3** |
| `floor_hue` | 0.5529 | +0.0197 ± 0.0124 | +0.0171 / +0.0060 / +0.0360 | 1.6×, 3/3 |
| `scale` | 0.1651 | +0.0107 ± 0.0029 | +0.0087 / +0.0086 / +0.0147 | 3.7×, 3/3 |
| `object_hue` | 0.3241 | +0.0084 ± 0.0285 | +0.0486 / −0.0086 / −0.0147 | 0.3×, 1/3 |
| `wall_hue` | 0.5509 | +0.0069 ± 0.0160 | +0.0119 / −0.0147 / +0.0235 | 0.4×, 2/3 |
| `orientation` | 0.1445 | −0.0104 ± 0.0129 | +0.0058 / −0.0114 / −0.0258 | −0.8×, 1/3 |

⭐ **The sampler moves `shape` — the one factor carrying compositional information — at 6.5× its own seed
spread, 3/3 seeds, clearing P2's `+0.02` threshold.** The six-factor aggregate is diluted by three factors
that are pure noise here, which is precisely R16's argument for reading the compositional term separately.
**P2 HELD.**

### ⛔⭐ …and the separability control takes it away again. THIS is the dial's answer.

| decomposition on the headline protocol | aggregate | **`shape`** |
|---|---|---|
| `clu` Δ (coupling on) | +0.0095 ± 0.0082 | **+0.0215 ± 0.0033** |
| `no_coupling` Δ (λ = 0 — the clamp **provably** cannot reach a relaxed well) | +0.0039 ± 0.0046 | **+0.0200 ± 0.0181** |
| ⭐ **steering-attributable = clu − no_coupling** | **+0.0056 ± 0.0078** (0.7× spread, 2/3) | ⛔ **+0.0015 ± 0.0204** (0.07× spread, 2/3) |

⛔ **The steering-attributable part is zero on both.** Whatever the rollout buys on `shape`, a landscape in
which the clamp is *mathematically incapable* of influencing a relaxed well buys the same. ⇒ **the gain is
the chain settling onto a mode from its centroid initialisation, not the clamp steering anything.**
**The isolation dial reads NO, and the control that says so is the one registered in advance to say it.**

### ⭐ The second, stronger trivial substitute — and the one place the sampler does win

`prior_mode` (relaxed wells placed at their **most-likely** mode; still **zero** Langevin steps) is a harder
control than `zero_step`, because it is the best no-inference guess available:

| | `prior_mode − full rollout` | reading |
|---|---|---|
| aggregate | **−0.0228 ± 0.0188** (full wins, 2/3) | the rollout beats the best zero-inference guess overall |
| `wall_hue` | −0.1044 ± 0.0503 | where that win comes from |
| **`shape`** | **+0.0089 ± 0.0456** | ⛔ **a tie** — on the compositional factor the trivial substitute is not beaten |

⭐ **The honest scorecard:** the rollout beats the *best* zero-inference guess on the aggregate, ties it on
the only compositionally meaningful factor, and is indistinguishable from its own separability control
everywhere. ⛔ None of that supports a steering claim.

### ⭐ The chain is NOT broken — which is what makes the above a finding, not a bug report

The second falsifier the task named ("flat energy, or a chain that never leaves its init") **did not fire**:

| audit column (headline arm) | measured |
|---|---|
| fraction of rollouts changing ≥1 code | **0.9531 ± 0.0166** |
| distinct code configurations / 512 | **499.3 ± 5.4** (zero-step: **278.3 ± 11.3**) |
| energy change at the FIXED reference `τ` | **−1.015 × its own scale** (⇒ non-flat by 200×) |
| mean acceptance probability / accepted fraction | **0.9518 ± 0.0006** / **0.9517 ± 0.0007** |
| mean top-1 occupancy `π` | **0.0204 ± 0.0002** (⇒ softmax **not** saturated) |
| mean clamp pull (force the landscape puts on the clamp) | **0.0044 ± 0.0001** (`full` protocol: 0.0229) |

⇒ **The chain mixes, accepts, and explores 1.8× as many configurations as the clamp alone specifies — and it
does not convert into steering.**

⚠ **The energy sign.** `energy_relative_drop` is **negative**: the chain **thermalises upward** from a
low-energy initialisation (the mode centroid) to the typical energy at `T`. That is what a *sampler* does;
an optimiser would go down. P7's non-flatness threshold holds by 200× while its directional wording is
falsified — **both reported.** ⛔ Endpoints are read off `energy_ref` (the state at a **fixed** `τ`), never
off the annealed `energy` trace — the trap `gen-store-grouping` filed for annealed training losses, pinned
here by `test_summary_reads_the_FIXED_tau_energy_never_the_annealed_one`.

---

## 5. ⛔ Two defects I found in MY OWN harness, and what each cost

### 5a. The frozen chain — caught by a pre-registered amendment (`PREREG.md` A1, 2026-09-12 00:34)

The **first** 3-seed run (preserved at `runs_absolute_tau/` + `run_absolute_tau.log`) measured
`moved = 0.05` — the chain was frozen. Diagnosed from the acceptance/code-change columns alone (⛔ no
accuracy number used), amended **before any post-fix number existed**:

| # | defect | evidence | fix |
|---|---|---|---|
| 1 | ⛔ **`τ` in ABSOLUTE units**, inherited from the store's *training* schedule. On a 2048-dim cache `d²` ~ hundreds, so `exp(−d²/2τ²)` is **saturated one-hot at every τ in the schedule** ⇒ the coupling's gradient vanishes and the basins are isolated | `moved = 0.05`; `max π ≈ 1` | `tau_calibration="mode_distance"`: τ becomes a **fraction of the store's own median inter-mode distance**. Realised **11.1 → 2.78**, not 1.0 → 0.2 |
| 2 | ⛔ step size tuned **once on `full`, reused for `partial`** — the free block differs ~6× in dimension and MALA is dimension-sensitive | acceptance **0.592** (`full`) vs **0.052** (`partial`) at the same ε | tuned **per protocol** |
| 3 | ⛔ `energy_relative_drop` divided by a per-sample `abs(E₀)` that can be ~0 | **`ΔE = −149011.5`** on one arm | ratio of means over `max(abs E₀, abs E_T)` |

⭐ **A1-P10, filed at the same moment — "the fix raises `moved` above 0.5, and if it does NOT the mixing
failure is structural and Du et al. bites here, in those words" — HELD: 0.05 → 0.953 ± 0.017.** So the
freeze was a tuning artefact, the fix is a declared sampler-tuning change justified on sampler diagnostics,
and **`mean_max_pi` is now a logged audit column on every rollout** so this can never again be an inference
from behaviour.

### 5b. ⭐⛔ The grouping — a LADDER-CONTRACT defect, not only mine (item 1)

After the 3-seed run completed I noticed my store ceiling read **0.6195 / 0.5579 / 0.5628** — a spread 14×
wider than R0's own `0.5997 ± 0.0019`. Cause:

> **`store_plan.json`'s `labels` vector indexes ENCODER CHANNELS.** `exp_gen_store` writes the plan at
> `i == 0` from **seed 0's** correlation grouping and records `encoder_param_hash = 98d3260…`. Every bridge
> seed trained its **own** encoder (`fc05e2c5…`, `154d4986…`), so channel `c` of seed 1 has no relation to
> channel `c` of seed 0. ⛔ **Taking the plan's labels verbatim on another seed is a size-matched RANDOM
> partition wearing the clustered grouping's name** — and the plan's own note, *"every later
> factorisation-ladder rung loads this file VERBATIM"*, invites exactly that.

`gen-store-grouping` itself did **not** make this mistake (it re-derived the grouping per seed); **my
harness did.** Fixed in `973e740`: the grouping is re-derived per seed by the plan's own procedure at the
plan's `K`, and ⭐ **the re-derivation is verified, not asserted — on the seed whose encoder wrote the plan
the labels must come back bit-identical, or the run raises.** It did: *"grouping re-derivation reproduces
the plan BIT-IDENTICALLY on its own seed"*, and seed 0's numbers are unchanged to 4 dp.

**Measured cost of the defect** (defective run preserved at `runs_plan_grouping/`):

| | seed 0 (the plan's own encoder) | seed 1 | seed 2 | 3-seed spread |
|---|---|---|---|---|
| store ceiling under the **plan's** labels | 0.6195 | **0.5579** | **0.5628** | **± 0.0279** |
| store ceiling under **per-seed** labels | 0.6195 | **0.6208** | **0.6243** | ⭐ **± 0.0021** |
| realised well sizes | `[303,305,573,501,366]` | `[397,300,307,300,744]` | `[283,283,390,616,476]` | — |

⭐ **The spread collapses 13×**, and the corrected ceiling (0.6215 ± 0.0021) lands where R22 predicts
(R0's 0.5997 + V1's ≈ +0.011). ⛔ **The headline verdict did not change in kind** — Δ went
`+0.0049 ± 0.0108` → `+0.0095 ± 0.0082`, still below the inert line — **but P2 flipped from NOT HELD
(−0.0118 ± 0.0308) to HELD (+0.0215 ± 0.0033), so the defect was masking the one real signal in the run.**
⛔ **Any ladder rung that consumes `store_plan.json` across seeds is exposed to this.**

---

## 6. ⭐ Two ceilings on every number (R23.5)

Protocol-matched (scored under the same request mask), 512 A-split requests, 3 seeds:

| | aggregate | `shape` |
|---|---|---|
| **DECODER ceiling** (real cached features → the frozen decoder → probe) | **0.9910 ± 0.0029** | 0.9733 ± 0.0133 |
| ⭐ **STORE ceiling** (real features → store read → same decoder → probe) | **0.6215 ± 0.0021** | 0.6107 ± 0.0252 |
| `full`-protocol dream (`clu`/`mala`) | 0.5605 ± 0.0077 | 0.5397 ± 0.0166 |
| `partial`-protocol dream (`clu`/`mala`) | 0.3485 ± 0.0141 | 0.3538 ± 0.0226 |
| `store_removed` (constant image) | 0.1213 ± 0.0139 | 0.2407 ± 0.0248 |

| fraction of… | `full` protocol | `partial` protocol |
|---|---|---|
| the **decoder** ceiling | **0.5656 ± 0.0063** | 0.3518 ± 0.0132 |
| ⭐ the **STORE** ceiling | ⭐ **0.9019 ± 0.0128** | 0.5612 ± 0.0168 |

⭐ **P8 HELD on both clauses** (predicted 0.45 of the decoder ceiling, band [0.20, 0.70]; measured **0.3518**
headline / **0.5656** on `full`), **and the pre-registered "cannot beat the store ceiling" held.**

⭐ **The single most important number for the Hub: on cell-native full requests the dream reaches 90.2 % of
the store ceiling.** ⛔ Quoting the decoder ceiling alone would make this read as a 57 % result, when the
object being sampled is a **30-bit** channel whose own ceiling is 0.62. **The binding constraint on this
cell is store capacity — not the sampler and not the decoder**, which is exactly the axis R23.5 defers.

⚠ **`store_removed` is correct but non-discriminating here** (item 4): with the store's content deleted the
read collapses to **one** code configuration and a constant image at 0.1213 (P4's ~0.125 ± 0.03 **HELD**),
with its own Δ **exactly 0.0000** — nothing left to sample. It ablates the **addressing**, not the
**dreaming**; `no_coupling` is the ablation that isolates dreaming, and it carries the finding (§4).

---

## 7. Pre-registration scorecard

`PREREG.md` mtime **2026-09-11 23:15 CEST**, before `gen_dream.py` existed; amendment **A1** at
**2026-09-12 00:34**, before any post-fix number. Machine-readable: `scorecard.json`
(defective-grouping run preserved at `scorecard_plan_grouping.json`).

| # | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| **P1** | full − zero-step ≥ +0.03, ≥2/3 seeds (point +0.08) | +0.03; **<+0.01 ⇒ inert** | **+0.0095 ± 0.0082**, 2/3, 1.16× spread | ⛔ **FALSIFIED — SAMPLER IS INERT** |
| **P2** | Δ`shape` ≥ +0.02 (point +0.06) | +0.02 | ⭐ **+0.0215 ± 0.0033**, **3/3**, **6.5× spread** | ⭐ **HELD** |
| **P3** | `no_coupling` abs Δ < 0.02 **and** < `clu`'s | 0.02 | **+0.0039 ± 0.0046** (clu +0.0095) | ✅ **HELD** |
| **P4** | `store_removed` ≈ 0.125 ± 0.03, own Δ ≈ 0 | — | **0.1213 ± 0.0139**, Δ **+0.0000** | ✅ **HELD** |
| **P5** | MALA acceptance in [0.4, 0.95] (point 0.70) | — | **0.9518 ± 0.0006** | ⚠ **NOT HELD** — 0.0018 above the band (§11.3) |
| **P6** | abs(MALA − ULA) < 0.02 ⇒ the correction is unnecessary at this scale | 0.02 | **0.0012** | ⭐ **HELD — and therefore EARNED** |
| **P7** | non-flat energy **and** ≥50 % of rollouts change a code | 0.005 / 0.5 | **−1.015** (200×), **0.953** | ✅ **HELD** on both; ⛔ the **sign** is a rise, reported |
| **P8** | 0.20–0.70 of the decoder ceiling; cannot beat the store ceiling (point 0.45) | — | **0.3518** (`full`: 0.5656); **0.561 / 0.902** of the store ceiling | ✅ **HELD**, both clauses |
| **P9** | clamp drift exactly 0; clamp **pull** > 0 | — | drift **0.0** by construction; pull **0.0044** / **0.0229** | ✅ **HELD** |
| **A1-P10** | the τ calibration raises `moved` above 0.5 | 0.5 | **0.953 ± 0.017** (from 0.05) | ⭐ **HELD** |

**Counter-hypotheses.** ⛔ **CH-1 (the dream is decoding its clamp) is CONFIRMED** on this cell.
⭐ **CH-2 (Du et al.'s mixing failure) is REJECTED** — the chain mixes at 0.95 and accepts at 0.95.
⛔ **CH-3 (degenerate clamp) is CONFIRMED for the `full` protocol** (0.58 relaxed wells), which is why the
rule filed in advance moved the headline to `partial`.

---

## 8. Audit traces on disk — every reported rollout

**66 `.npz` files**:
`runs/shapes3d_recon/seed_{0,1,2}/traces/<protocol>__<arm>__<sampler>__<variant>_s<seed>.npz`
(22/seed = 2 protocols × {clu × 3 samplers, clu zero_step, clu prior_mode, store_removed × 3 variants,
no_coupling × 3 variants}). Each carries, **per step and per request**:

`energy` (annealed-τ, ⛔ never differenced) · **`energy_ref`** (same state at a **fixed** τ — the only one
that may be differenced) · `accept_prob` · `accepted` · `clamp_pull` · `tau` · `temperature` · `code_init` ·
`code_final`. Shapes: `energy (201, 512)`, `accept_prob (200, 512)`.

- **Headline trace:** `runs/shapes3d_recon/seed_0/traces/partial__clu__mala__full_s0.npz`
- **Its laundering control:** `…/partial__clu__mala__zero_step_s0.npz` (0 steps; empty acceptance array — the
  control has no sampler, by definition)
- **The separability control:** `…/partial__no_coupling__mala__full_s0.npz`
- **Pre-fix (frozen-chain) traces:** `runs_absolute_tau/…`; **defective-grouping traces:** `runs_plan_grouping/…`
- **Figures:** `runs/shapes3d_recon/seed_*/dream.png` (energy traces + full-vs-zero-step bars against both ceilings)

---

## 9. Reproducing the exact numbers

```
PYTHONPATH=<worktree> CHLU_GEN_DATA_ROOT=<...>/.claude/data/gen \
CHLU_CLAUDE_ROOT=<...>/.claude \
  /Users/user/Desktop/CHLU/.venv/bin/python -m chlu.experiments.exp_gen_dream \
  --dataset shapes3d --pretext recon --stage dream --out <OUT>
# then, for the scorecard (recomputes the audit statistics from the on-disk traces):
python .claude/scratch/gen-dream/score_prereg.py
```
Equivalent CLI: `chlu exp-gen-dream --dataset shapes3d --pretext recon` (subparser verified to dispatch to
`cmd_exp_gen_dream`, including `--condition recomb2element_wide`).

---

## 10. Byte accounting — ⛔ `gen-baselines` needs this (item 2)

| structure | bytes | note |
|---|---|---|
| the store (modes) | **524,288** | unchanged; `n_modes · D · 4`, independent of `K` |
| ⭐ the inter-well PMI coupling | **163,840** | `K(K−1)/2 · M² · 4`; **EXTRA** — a byte-matched baseline must be told |
| **total for the dreaming arm** | **688,128** | |
| the addressing head | not counted here | per-(factor,value) code histograms — same supervision class as the kNN arm's per-factor prototypes (R8) |
| ⭐ **read channel** | **30.0 bits** | ⛔ **unchanged** — R20's declared design constant is untouched |

---

## 11. Environment findings the next gen spoke needs

1. ⛔ **`UF_HIDDEN` on harness-written JSON** (handover §7.12 family). The manifests were written with the
   macOS hidden flag set and the run's own `aggregate` stage died with
   `PermissionError: Operation not permitted` **reading its own output**. All seeds had already been
   written, so nothing was lost; `chflags -R nohidden` fixes it (now done in the pipeline script).
2. ⚠ **A concurrent `pytest` killed a 3-seed run.** The first overnight run died silently ~10 h in, during
   seed 1's store training, while a full `pytest` ran on the same machine. ⛔ **Do not run a JAX test suite
   beside a JAX harness here.** Relaunched alone, it completed in 26 minutes.
3. ⚠ **A MALA acceptance target is not reachable on a coarse ε grid.** Acceptance stepped
   `0.05 → 0.41 → 0.82` at successive halvings, so the search oscillated; it now returns the
   **closest-to-target** ε from its history rather than the last one. Residual consequence: the realised
   acceptance is **0.9518**, 0.0018 above the pre-registered [0.4, 0.95] band ⇒ **P5 reported as NOT HELD**
   rather than rounded into the band.
4. **R19's cache non-uniformity confirmed and handled**: seed 0 `scope="full"`, seeds 1–2 `scope="used"`
   ⇒ 32,199 rows re-encoded through the frozen encoder per seed, bit-identity check (`max abs Δ = 0.0`)
   passing every time.
5. **§7.23 (x64 at module import) did not bite** — every entry point casts its input to the store's own
   parameter dtype (`gen_bridge` exemplar), pinned by the module's own float64 test.

---

## 12. ⛔ CSF3 — NOT RUN (2FA, unreachable from an agent session)

Everything above ran **locally on CPU**. ⛔ No job was submitted and I report no unrun job as a pass.
⚠ **The launcher has no `dream` stage and no `CONDITION` axis** (the latter is R19's open scoping item), so
the stage is overridden through `EXTRA_ARGS` — argparse takes the **last** `--stage`, **verified**.

```
GEN_GIT_COMMIT=$(git rev-parse HEAD) \
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=store,\
ARMS=clu store_removed no_coupling,SEEDS=0 1 2,\
PAYLOAD_MODULE=chlu.experiments.exp_gen_dream,\
EXTRA_ARGS=--stage dream --condition recomb2element,\
GEN_GIT_COMMIT=$GEN_GIT_COMMIT" \
  -a 0-8%3 -t 06:00:00 scripts/csf3/job_gpu_gen.sh
```

- **Realised output paths:**
  `$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/store/<arm>_s<seed>/artifacts/shapes3d_recon/seed_<seed>/{dream_manifest.json,dream.png,traces/*.npz}`,
  plus `run_manifest.json` and `logs/stage.log` under each `<arm>_s<seed>/`.
- ⚠ `ARMS` **must** be set: the launcher's default (`clu store_removed lace knn`) contains arms this module's
  `--arm` rejects, and every task would die on argparse.
- ⚠ The array writes one `(arm, seed)` per task, so `--stage aggregate` must be run **once afterwards** over
  the collected run dirs (same shape as `exp_gen_store`'s).
- **Every cluster acceptance item: NOT RUN.** Local CPU only, 1,545 s for 3 seeds.

---

## 13. Suite

| | count |
|---|---|
| **before** (`main @ 9a4457d`, throwaway detached worktree `../CHLU-gd-base`, main venv) | **2112 passed, 0 failed** (50:39) |
| **after** (`agent/experiment-engineer/gen-dream-sampler`) | **2146 passed, 0 failed** (43:26) |
| delta | ⭐ **+34 = exactly the 34 collected tests in `tests/test_gen_dream.py`** (31 functions, one parametrised over the 3 samplers) |

⭐ **All 34 pass in the FULL suite, not only in isolation** — the §7.23 x64-at-module-import hazard that broke
9 of `gen-feature-bridge`'s tests did not recur, because every entry point casts its input to the store's own
parameter dtype (the `gen_bridge` exemplar). ⛔ No `uv sync` was run in any worktree (main venv, JAX 0.9.0,
protocol §4), so the w6 resolver hazard does not apply. Run **sequentially, never concurrently** (§11.2).
Logs: `suite_before.log`, `suite_after.log`, driver `pipeline.log`.

---

## Git footprint

- **Branch:** `agent/experiment-engineer/gen-dream-sampler` (worktree `/Users/user/Desktop/CHLU-gen-dream`),
  base `main @ 9a4457d`.
- **Commits** (`main..HEAD`):
  - `5bd25e5` add gen_dream: the clamped-well Langevin landscape + audit trail
  - `4d5a35c` wire GenDreamConfig through the three config sites
  - `9fb80e5` add exp-gen-dream: the zero-step control on every arm
  - `48f0a6d` tests: the separability control, the decoder guard, the Gibbs check
  - `973e740` re-derive the grouping per seed: the plan's labels index ENCODER channels
- **Files touched:** `chlu/core/gen_dream.py` (new), `chlu/experiments/exp_gen_dream.py` (new),
  `tests/test_gen_dream.py` (new), `chlu/config.py` (**append-only**: `GenDreamConfig` + the three house
  wiring sites), `chlu/cli/experiment_cmd.py` (**append-only**: one subparser + one `cmd_` function).
  ⛔ Nothing under `chlu/eval/`, `scripts/csf3/`, `gen_store.py` or `gen_bridge.py` was modified.
  ⚠ **Declared deviation from the task's "append at the end of `config.py`":** a dataclass appended *below*
  `CHLUConfig` cannot be referenced by its `default_factory` (NameError at class creation). I used the anchor
  `GenBridgeConfig`/`GenStoreConfig` actually use — immediately before `ProjectConfig`, i.e. the end of the
  *dataclass block* — which is also what "wire it at the three house sites exactly as `GenDataConfig` and
  `GenBridgeConfig` do" requires. **The conflict rule is unaffected: keep BOTH blocks.**
- **Rebased onto local `main`** (⛔ not `origin/main`, §7.21 / protocol §3.5): **no-op — `main` is still at
  `9a4457d`**, unmoved, so the rebase reported "up to date".
- ⭐ **Branch ref verified FROM THE MAIN REPO** (protocol §3.2), after the rebase:
  `git log --oneline main..agent/experiment-engineer/gen-dream-sampler` shows all **5** commits, and
  `git diff --stat` shows **3,006 insertions / 0 deletions across 5 files — not one existing line modified.**
  The worktree `../CHLU-gen-dream` is **left in place for Hub review**; the throwaway `../CHLU-gd-base`
  (detached at `9a4457d`, used only for the before-suite) was removed after that verification.
- **Unresolved conflicts:** none.
- **Artifacts** (all under `.claude/`): `outputs/gen-dream-sampler/{PREREG.md, scorecard.json, run_main.log,
  run_main_died.log (the pytest-contention kill, §11.2), suite_before.log, suite_after.log, pipeline.log}`, `runs/shapes3d_recon/seed_*/` (manifests, 66 traces,
  figures); **preserved defective runs** `runs_absolute_tau/` (frozen chain) and `runs_plan_grouping/`
  (+ `run_plan_grouping.log`, `scorecard_plan_grouping.json`);
  `scratch/gen-dream/{score_prereg.py, run_all.sh, smoke_core.py}`.

## Open questions / follow-ups / risks

1. ⛔ **The five reconciliation items at the top need a Hub owner** — item 1 (the ladder contract) **before
   R1…R5**, items 2–4 before `gen-baselines` / `gen-cell-run`.
2. ⭐ **What this spoke thinks the Hub should conclude.** The dream reaches **90.2 % of the store ceiling** on
   cell-native requests: **the addressing head already extracts nearly everything a 30-bit channel can
   carry, so there is nothing left for a sampler to find.** The inertness is a statement about **store
   capacity**, not about Langevin. ⛔ **Do not respond to it by tuning the sampler.** The levers that could
   make dreaming non-trivial are (a) more channel — which R23.5 explicitly defers to a later pre-registered
   axis; (b) requests that *under*-determine the wells, which is what `partial` is, and where the only tight
   signal (`shape`, +0.0215 at 6.5× spread) lives; (c) **the B/C cells, where the clamp is out of
   distribution and the clamp-decoding shortcut may not be available.** ⭐ **(c) is the interesting one and
   it is not mine to run.**
3. ⚠ **The coupling field is my design decision — pre-registered, but not Head-ruled.** If the Hub wants the
   dream to run on the store's landscape *as shipped* (separable), §2's argument says conditional dreaming
   is inert **by construction** and the capability needs a different mechanism. **That is a decision, not a
   re-run.** ⚠ And note the measured result: on this cell the coupling's *steering* contribution is
   `+0.0056 ± 0.0078` (aggregate) / `+0.0015 ± 0.0204` (`shape`) — **not distinguishable from zero**, so the
   coupling currently pays 163,840 B for nothing measurable.
4. ⚠ **`shape` at +0.0215 ± 0.0033 (6.5× spread, 3/3) is the one tight number in the run, and I cannot
   attribute it to steering.** It survives the zero-step control but not the separability control. If the
   Hub wants it attributed, the experiment is: vary `coupling_strength` over a grid and check monotonicity —
   cheap (~8 min/point, the store is checkpointed), and **not pre-registered**, so it must be filed first.
5. ⚠ **`n_dream = 512` of 1024 available `eval_a` rows**, and the `partial` protocol drops 2 of 6 factors at
   random (seeded, fixed per request). A per-dropped-factor breakdown of the contrast was not run and would
   be cheap.

## Proposed handover updates (for the Hub)

- **§3 config:** new `GenDreamConfig` block (dataclass + `load_config` + `save_config`, append-only) and the
  new `chlu exp-gen-dream` subcommand. Notable defaults: `sampler="mala"` (**the Metropolis correction IS
  applied**), `samplers=["mala","ula","uld"]`, `variants=[...,"zero_step",...]` (⛔ the laundering control is
  never optional), `arms=["clu","store_removed","no_coupling"]`, `decoder_source="winner"` +
  `expected_decoder_recipe="v1_capacity_schedule"` (⛔ **not** the `dec_recipe="baseline"` default),
  `step_size=None` (tuned to an **acceptance** target), `tau_calibration="mode_distance"`,
  `temp_calibration="energy_scale"`, `coupling_strength=1.0`, **`grouping_source="per_seed"`**.
- **§7 NEW known issue — ⭐ `store_plan.json`'s `labels` index ENCODER CHANNELS and are portable across
  RUNGS, not across SEEDS.** Each bridge seed trained its own encoder, so the plan's vector is a
  size-matched *random* partition on any seed but the one that wrote it (measured: store ceiling
  0.5579/0.5628 → 0.6208/0.6243, 3-seed spread ±0.0279 → ±0.0021). Only `K`/`n_modes`/`D`/`store_bytes` are
  portable; re-derive the grouping per seed and assert bit-identity on the plan's own seed.
- **§7 NEW known issue — a per-well store energy is SEPARABLE, so conditional dreaming on it is provably
  inert.** Any "clamp some wells, relax the rest" design needs an explicit inter-well term, and
  `coupling_strength = 0` must be **run as a control**, not argued about.
- **§7 NEW known issue — an annealing parameter in ABSOLUTE units silently freezes a sampler.** `τ` taken
  from the store's *training* schedule saturates the soft assignment one-hot on a 2048-dim cache
  (`moved = 0.05`). Calibrate `τ` to the store's own inter-mode distance and **log `mean_max_pi`** so
  saturation is a number, not an inference. (Sibling of `gen-store-grouping`'s "annealed losses are not curves".)
- **§7 NEW known issue — `UF_HIDDEN` on harness-written JSON under `.claude/`** makes a later read fail with
  `PermissionError`; `chflags -R nohidden` is the fix. Extends §7.12 from the editable-install `.pth` to
  ordinary output files.
- **§7 NEW discipline note — never derive a PRNG key from Python's `hash()` of a string** (randomised per
  process via `PYTHONHASHSEED`; a "seeded" rollout becomes irreproducible across runs). `zlib.crc32` here.
- **§7 NEW discipline note — a MALA acceptance target is not reachable on a coarse ε grid**; take the
  closest-to-target ε from the search history and report the **realised** acceptance, not the targeted one.
- **§7 NEW environment note — do not run a JAX test suite beside a JAX harness on this machine**; it killed
  a 3-seed run 10 hours in, silently.
- **Rulings ledger:** the phase's **conditional-dreaming arm exists and runs**; its **zero-step control
  fires** (`+0.0095 ± 0.0082`, P1 falsified); the only tight signal is on the compositional term
  (`shape`, +0.0215 ± 0.0033, 6.5× spread) and is **not attributable to steering** (the separability control
  matches it); and the binding constraint it identifies is the **30-bit store capacity** (R23.5's deferred
  axis), not the sampler. `gen-baselines`'s byte-match target for this arm is **688,128 B**.
