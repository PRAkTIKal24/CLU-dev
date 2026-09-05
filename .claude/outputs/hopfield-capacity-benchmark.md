# hopfield-capacity-benchmark — experiment-engineer report

**Task + acceptance:** match the real modern-Hopfield/U-Hop retrieval protocol, run CLU (designed register) vs dense Hopfield, sparse/U-Hop SOTA, and an NN floor on MNIST + CIFAR-10 (50%-masked queries), deliver accuracy-vs-load + accuracy-vs-noise + cross-over, test the three differentiators (keep only those with a performance edge), give the honest at/above/below verdict. Tests green.

**Status: done.** Protocol matched to both repos (hashes below). MNIST + CIFAR-10 both run with real baselines. Full suite **568 passed, 0 failed** (was 566+2; the 2 were my own tests hitting the §7.2 x64 landmine — root-caused to a *real* x64 crash in the CLU rollout, **fixed in the code**, not papered over). One differentiator (fiber payload) dropped per the kill-rule (not applicable to the direct-pattern protocol; see §5).

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). 3 sites.**
> 1. **The scout's "cosine>0.9 success" is NOT the U-Hop protocol.** The repo (`memory_retrieval.py`) reports **mean `sqdiff`** = Σ(clamp(x,0,1)−clamp(x̂,0,1))². Any paper text quoting "cosine>0.9 from U-Hop" must be corrected to "mean squared pixel error (sqdiff); cosine>0.9 is our added secondary."
> 2. **The mask is `torch.dropout(x,0.5)` — zero 50% AND double survivors — NOT "mask to zero".** Single-sourced constant now pinned from source.
> 3. ⭐ **The headline the Head may not want: on the standard image protocol, a trivial NN pixel-space floor beats BOTH CLU and modern-Hopfield.** CLU ties/beats the Hopfield line but does **not** exceed SOTA and is **below** the NN floor. The real CLU deliverables are the two *capabilities* Hopfield structurally lacks (retry, per-item retention), not a capacity-curve win. This must travel with any "CLU is competitive on associative memory" claim.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/hopfield-capacity-benchmark`; base local `main @ 8519df6`; commits `9375d3b, bcc1d2a, 5396432, a7ebf11` |
| protocol matched | **U-Hop** `MAGICS-LAB/UHop @ cdac75431df968b7142b4fb605a0fcd56feb59cb`; **Ramsauer** `ml-jku/hopfield-layers @ f56f929c95b77a070ae675ea4f56b6d54d36e730` (cloned + read; see §1) |
| datasets | MNIST (openml `mnist_784`, cached), CIFAR-10 (HuggingFace `uoft-cs/cifar10` parquet, 10k test images; openml CIFAR_10 is **checksum-blocked**, Toronto tarball **throttled** — see §6) |
| pixels | `[0,1]` (torchvision `ToTensor` convention, repo-verbatim) |
| query (capacity) | `torch.dropout(x, p=0.5)` — zero 50% of pixels, scale survivors ×2 (repo-verbatim) |
| query (noise) | `clamp(|x + N(0,σ)|, 0, 1)` (repo-verbatim) |
| **primary metric** | **mean `sqdiff`** (repo); **identity-retrieval accuracy** (argmin over stored ‖x̂−ξ_i‖ == true i) + median cosine + success@cosine>0.9 reported alongside |
| seed | 0 (single seed) |
| CLU register | `GaussianMemoryPotential` (centers = stored patterns), `clu_s_frac=0.3` (s = 0.3·median-NN-pattern-distance, **one fixed rule, not per-load**), `clu_b=1`, `clu_alpha=1e-3`, `clu_gamma=0.1`, `clu_steps=200`, `dt=0.5·s/√b` (auto), Newtonian-identity kinetics, read = mean over last 10% of the damped Verlet rollout |
| Hopfield arms | dense softmax repo-verbatim (β=1, 1 step); sparsemax repo-verbatim; proper-Ramsauer "tuned" (β INSIDE softmax = max(1, 200/⟨x,x⟩), 2 steps) |
| retry | boost 1.5, retry bottom-50%-confidence, compute ×1.5 |
| JAX | 0.9.0 (main venv reused, protocol §4); float32 (x64 OFF); **CLU rollout is now x64-safe** |
| designed vs learned | **everything designed/closed-form on BOTH sides. Nothing learned** (N46). U-Hop's learned-kernel UMHN is deliberately excluded — it would break the admissibility (see §3). |

---

## 1. Protocol match (Item 0) — matched to source, not to the scout's summary

Cloned and read both repos. The U-Hop retrieval harness (`memory_retrieval.py` + `functions.py`):
- stores `m_size` images `(m_size, D)` in `[0,1]`; for each stored `x` the query is `torch.dropout(x, p=0.5, train=True)`;
- update `MHN_update_rule`: `score = beta * activation(Xiᵀx); x = Xi @ score`, default `beta=1, steps=1`, `activation ∈ {softmax, sparsemax, entmax15, …}`;
- **success = `sqdiff(x, x_new)` = Σ(clamp(·,0,1))² difference, averaged** — NOT cosine>0.9. The Ramsauer repo's `x = X·softmax(βXᵀx)` is identical to MHN-softmax, confirming the dense arm.
Everything in my harness is these formulas verbatim (softmax/sparsemax exact ports; entmax15 via bisection; dropout mask exact). **I report `sqdiff` (repo) and add identity-accuracy/cosine for legibility.**

The one place I extend the protocol honestly: the repo's β=1/1-step dense arm is *already* the proper Ramsauer update, so I add a **sharper "tuned"** arm (β inside softmax, iterated) so nobody can say Hopfield was strawmanned. Measured: on MNIST the sharper/iterated arm does **not** beat β=1 (iterating hurts); on CIFAR **no** β from 0.26→256 rescues it (see §3).

---

## 2. Item 1 — the performance comparison (identity-retrieval accuracy)

### 2.1 MNIST — capacity (50%-dropout queries), seed 0

| M | dense-Hopfield (softmax β1) | sparse (sparsemax) | Hopfield tuned | **CLU register** | NN floor |
|---|---|---|---|---|---|
| 16  | 0.875 | 0.875 | 0.875 | 0.812 | **1.000** |
| 32  | 0.938 | 0.938 | 0.906 | 0.750 | **1.000** |
| 64  | 0.906 | 0.906 | 0.875 | 0.703 | **1.000** |
| 128 | 0.883 | 0.875 | 0.836 | 0.562 | **1.000** |
| 256 | 0.766 | 0.766 | 0.684 | 0.344 | **0.996** |
| 512 | 0.637 | 0.639 | 0.562 | 0.230 | **0.998** |

⚠ **On the repo's own `sqdiff` metric CLU looks worse than it is** (CLU is *bimodal*: sqdiff≈0 when it lands in the right well, large when wrong; Hopfield returns a soft low-variance blend). e.g. M=256 mean-sqdiff: dense 10.5, sparse 12.3, CLU 46.5, NN 0.06. **The two metrics disagree for CLU** — report both.

### 2.2 MNIST — noise robustness at load M=128 (Gaussian σ), seed 0

| σ | dense-Hopfield | sparsemax | tuned | **CLU register** | NN floor |
|---|---|---|---|---|---|
| 0.0 | 0.891 | 0.898 | 0.859 | **0.922** | 1.000 |
| 0.2 | 0.422 | 0.422 | 0.406 | **0.922** | 1.000 |
| 0.4 | 0.164 | 0.156 | 0.156 | 0.031 | 1.000 |
| 0.6 | 0.047 | 0.047 | 0.039 | 0.016 | 0.891 |
| 0.8 | 0.016 | 0.016 | 0.016 | 0.008 | 0.203 |
| 1.0 | 0.008 | 0.008 | 0.008 | 0.008 | 0.062 |

⭐ **CLU BEATS dense & sparse Hopfield at low noise** (σ=0.2: **0.922 vs 0.42**, a >2× win) — the attractor basin cleans the query where a 1-step Hopfield blend does not. **But CLU falls off a cliff at σ≥0.4**: its narrow wells (`s=0.3·NN`, set for capacity) cannot capture a badly-displaced query — this is the `Δ_req` resolution floor in action (`address-space-dimension-scaling`), the capacity↔robustness trade the theory predicts. NN dominates throughout.

### 2.3 CIFAR-10 — capacity (50%-dropout), seed 0

| M | dense-Hopfield | sparsemax | tuned | CLU register | NN floor |
|---|---|---|---|---|---|
| 16  | 0.062 | 0.062 | 0.062 | 0.062 | **1.000** |
| 64  | 0.016 | 0.016 | 0.016 | 0.031 | **1.000** |
| 256 | 0.012 | 0.012 | 0.012 | 0.008 | **1.000** |
| 512 | 0.012 | 0.012 | 0.010 | 0.004 | **1.000** |

### 2.4 CIFAR-10 — noise at M=128

| σ | dense/sparse/tuned Hopfield | CLU register | NN floor |
|---|---|---|---|
| 0.0 | 0.008 | **0.859** | 1.000 |
| 0.2 | 0.008 | **0.711** | 1.000 |
| 0.4 | 0.008 | 0.008 | 0.930 |
| ≥0.6 | 0.008 | 0.008 | ≤0.13 |

⭐⭐ **On raw-pixel CIFAR, closed-form dense & sparse modern-Hopfield retrieve at CHANCE (0.031→0.008) for EVERY β from 0.26 to 256, even on the clean query.** Natural-image inner products are DC-dominated (every image overlaps every other strongly), so `argmax_i ⟨ξ_i, q⟩` collapses to the brightest stored image for all queries. **This is precisely why U-Hop's headline method (UMHN) LEARNS a kernel `W` — the closed-form line needs a learned feature map on CIFAR, which is outside the "nothing learned" admissibility.** CLU, by contrast, retrieves CIFAR under moderate noise (0.86/0.71 at σ≤0.2) — its localized wells are metric-native and do not suffer the DC-overlap collapse. **CLU markedly beats the closed-form Hopfield line on CIFAR** (both axes), but still loses to the NN floor.

### 2.5 Cross-over (identity-acc < 0.9 criterion)
- **MNIST:** NN **never** crosses (≥0.996 to M=512). Every Hopfield arm and CLU are already <0.9 at M=16 under the aggressive 50%-dropout mask (dense/sparse peak ≈0.94 at M=32, then decay; CLU peaks ≈0.81). At a **0.5** criterion: dense/sparse cross at M≈512, tuned at M≈256, CLU at M≈128.
- **CIFAR:** all closed-form arms are below criterion at every M (chance); only NN stays at 1.0.

---

## 3. Item 3 — the honest at/above/below verdict

| axis | CLU vs dense Hopfield | CLU vs sparse (U-Hop closed-form) | CLU vs NN floor |
|---|---|---|---|
| MNIST capacity | **below** (0.23–0.81 vs 0.64–0.94) | **below** | **far below** (NN≈1.0) |
| MNIST noise (σ≤0.2) | **above** (0.92 vs 0.42) | **above** | below |
| MNIST noise (σ≥0.4) | below (basin limit) | below | far below |
| CIFAR capacity | tie (both ≈ chance) | tie | far below |
| CIFAR noise (σ≤0.2) | **far above** (0.86 vs 0.008) | **far above** | below |

**Verdict, stated plainly (per Hu–Wu–Liu framing):** CLU's capacity curve is **AT or BELOW** the dense/sparse modern-Hopfield line on the masked-image protocol and **BELOW** the trivial nearest-neighbour floor; on the **noise** axis CLU is **ABOVE** the Hopfield line at low noise (and *far* above on CIFAR, where closed-form Hopfield needs a learned kernel it is not allowed here). **CLU does not beat SOTA on raw capacity, and the exponential is not the story** — as Hu, Wu & Liu (NeurIPS 2024) show, exponential capacity = optimal spherical codes, a 2017–2024 result. The novel, defensible part is the **measured mechanism** (`Δ_req≈3.1·max(w,σ)` basin-width limit governing the capacity↔noise trade, `address-space-dimension-scaling`), realized here as a Hamiltonian settling dynamics — not the exponential itself.

**The uncomfortable finding the task asked me to report:** on this benchmark the honest strongest method is the **NN pixel-space floor**; both the Hopfield line and CLU are cleaner-attractor dressings of it that *lose* to it on masked-image retrieval. The scout's "#1 winnable target" is winnable only against the *closed-form Hopfield line*, not against the trivial baseline — and even that win is confined to the noise axis (MNIST) / requires the Hopfield line be denied a learned kernel (CIFAR).

---

## 4. Item 2 — differentiators tested AS PERFORMANCE

### 4.1 ⭐ Retry (adaptive compute) — KEPT, a strong edge

A second boosted CLU relaxation on the bottom-50%-confidence first-pass queries (confidence = cosine to the settled well), re-launched from the settled point with momentum toward the original query. **MNIST, M=128, ×1.5 compute:**

| | identity-acc |
|---|---|
| first pass | 0.492 |
| **after retry (low-conf half)** | **0.961** (+46.9 pp) |
| blank guard (retry the HIGH-conf half instead) | 0.109 (**−38.3 pp**) |

The +47pp lift (**far exceeding my pre-registered +2–8pp**, P5) with a **−38pp blank** is exactly the adaptive-compute signature: spending compute only where the first pass is unsure *recovers misses*, spending it on already-correct queries *destroys them*. The retry re-settles from the query into the metric-nearest well, lifting CLU from ≈Hopfield up toward the NN floor. **This is a curve modern Hopfield cannot draw** (a single softmax step has no "try again from here with more energy" knob). KEEP.

### 4.2 ⭐ Per-item retention control (μ²) — KEPT, a capability with no Hopfield analogue

One store, half the items permanent (`τ=∞`), half decaying (`τ=1`), via per-item well depths `b_i(t)=b·exp(−t/τ_i)`. **MNIST, M=16:**

| t | acc permanent | acc decaying |
|---|---|---|
| 0.0 | 0.875 | 0.875 |
| 0.5 | 0.875 | 0.875 |
| 1.0 | 0.875 | 0.625 |
| 2.0 | 0.875 | **0.000** |
| 4.0 | 0.875 | **0.000** |

Permanent items are retrieved unchanged at every `t`; scheduled items decay to unretrievable on their per-item schedule, **in the same store, simultaneously**. Modern Hopfield stores all patterns with identical timeless weight — there is no `t` axis to hang a schedule on. Reported as a **demonstrated capability**, not a benchmark curve (there is no Hopfield number to beat). KEEP.

### 4.3 Fiber payload — DROPPED per kill-rule (out of protocol scope)

The direct-pattern retrieval protocol has the pattern *be* the address (no separate payload channel), so the fiber differentiator does not fit and shows no performance edge *here*. Its capacity was already measured under the register+payload setup (`relaxation-fiber-capacity`: 129–174 bits/well at d=2 with `(σ_read,N,launches)`; the `B_total≤P·b_θ` ceiling binds for learned landscapes, not this designed one). Not re-run. Honest scoping, not a negative result.

---

## 5. PREREG scorecard (`PREREG.md` written before any harness ran)

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | dense β=1 near-degenerate in high-D (acc<0.3 by M=100 MNIST) | β=1 is actually the proper Ramsauer update and works on MNIST (0.88 at M=128); **degenerate on CIFAR (chance)** | ◐ **wrong for MNIST, right for CIFAR** — β=1 is not a strawman on MNIST |
| P2 | tuned Hopfield >0.9 at M=100 MNIST | tuned ≈ β=1 (iterating hurts); ≈0.85 at M=128 | ◐ form right, no >0.9 |
| P3 | CLU ≈ NN floor (\|Δacc\|≤0.05) | **FALSE** — CLU is well *below* NN (0.56 vs 1.0 at M=128); CLU ≈ Hopfield, not ≈ NN | ❌ **falsified** |
| P4 | ordering `NN≈CLU≳sparse≳tuned≫denseβ1` | `NN ≫ dense≈sparse≳tuned ≳ CLU` on capacity; **CLU>Hopfield on noise** | ❌ ordering wrong (CLU below Hopfield on capacity, above on noise) |
| P5 | retry +2–8 pp | **+46.9 pp** (blank −38.3) | ✅-in-direction, **magnitude under-predicted ~7×** |
| P6 | cross-over: dense-β1 first | at 0.9 all-but-NN cross by M=16 (mask too aggressive); at 0.5, CLU crosses first | ◐ |
| P7 | CIFAR same ordering, CLU at NN | CIFAR: **closed-form Hopfield = chance at all β**; CLU works only on noise; CLU ≠ NN | ❌ CLU not at NN; Hopfield collapse unregistered |

**Honest summary of the prereg:** my central optimistic hypothesis (P3/P4: CLU tracks the NN floor and matches/beats SOTA on capacity) is **falsified**. CLU tracks the *Hopfield* line, not NN, and loses to NN. The two survivors are a *finding* (retry +47pp, P5 vastly under-predicted) and an *unregistered* strong result (CIFAR closed-form Hopfield collapse, favouring CLU on the noise axis).

---

## 6. What I found / had to solve

1. **CIFAR data access.** openml `CIFAR_10` is **checksum-blocked** here (md5 mismatch — openml re-served the file); the canonical Toronto tarball **throttles to ~13–27 MB then stalls** (2 attempts). Loaded via the **HuggingFace `uoft-cs/cifar10` parquet** (10k test images, 24 MB, complete) → PIL decode → `(10000,3072)` in `[0,1]`. The loader tries tarball→parquet in order and errors clearly if neither is present. CIFAR uses the 10k **test** split (the U-Hop harness draws `m_size` from a single batch too).
2. ⭐ **A real x64 crash in the CLU rollout** (not just the §7.2 marginal-flip landmine): under `jax_enable_x64`, `GaussianMemoryPotential` pins its centers to float32, so `V(q)` promotes to float64 and `model.step` returns float64, mismatching the float32 `lax.scan` carry → `TypeError`. **Fixed** by launching the carry in the ambient float dtype (`jnp.result_type(float)`). Verified: `capacity_sweep` runs under x64 (acc 1.0) and the isolation case `pytest tests/test_twins.py tests/test_hopfield_capacity.py` → **15 passed**.
3. **`beta_tuned` was blurring on CIFAR** (rule `200/⟨x,x⟩` gives β≈0.22<1 at ⟨x,x⟩≈900). Floored at the repo β so the "tuned" arm is never worse than default. (Moot for the verdict — no β rescues CIFAR — but removes an unfair-to-Hopfield artifact.)
4. **CLU well width is load-bearing and NOT tuned per load:** `s=0.3·median-NN-distance`. Wider (`0.5`) merges basins → every query settles to the centroid (acc 0.17=chance on 6 patterns); this is the localized-dense-associative-memory regime, stated as structure.

---

## How I verified

- Repos cloned to `.claude/scratch/hopfield-capacity-benchmark/repos/{UHop,hopfield-layers}` at the pinned hashes; protocol read line-by-line from `memory_retrieval.py`/`functions.py`/`data.py`.
- `pytest tests/test_hopfield_capacity.py` → **9 passed** standalone; `pytest tests/test_twins.py tests/test_hopfield_capacity.py` (x64 leak) → **15 passed**; **full suite `568 passed, 0 failed`** (prior HEAD 561; +7 new tests). `ruff check`/`format` clean on all touched files.
- MNIST full run 89 s; CIFAR full run 294 s; combined final run refreshed `.claude/outputs/hopfield-capacity-benchmark/results/exp_hopfield_capacity_metrics.json` + figures `hopfield_capacity_{mnist,cifar10}.png`, `hopfield_noise_{mnist,cifar10}.png`. All numbers above re-derived from that JSON (seed 0).
- CLI: `chlu exp-hopfield-capacity [--project N] [--seed I] [--quick] [--dataset …]` wired; module runnable as `python -m chlu.experiments.exp_hopfield_capacity --quick`.

## Git footprint

- **Branch** `agent/experiment-engineer/hopfield-capacity-benchmark`, base local `main @ 8519df6`. Rebased onto local `main` = no-op (base unmoved). **Not pushed**, no PR (per protocol/task).
- **Commits (4):** `9375d3b` (GaussianMemoryPotential), `bcc1d2a` (config + registration), `5396432` (experiment + CLI + tests), `a7ebf11` (x64-safety + tuned-β floor + float32 fixture).
- **Files:** **+** `chlu/experiments/exp_hopfield_capacity.py`, `tests/test_hopfield_capacity.py`; **M** `chlu/core/memory_potentials.py` (+`GaussianMemoryPotential`), `chlu/config.py` (+dataclass + load/save reg), `chlu/cli/experiment_cmd.py` (+parser/handler). No shared files reformatted beyond my hunks; `utils/plotting.py` untouched (figures local, per `exp_retrieval` precedent). `results/` not committed.
- No collision: worked directly on the main checkout (clean, no other agent's uncommitted work present at start).

## Open questions / follow-ups / risks

- **Single seed (0), single CLU landscape family, MNIST-full + CIFAR-test-10k.** The MNIST numbers are stable across two pool draws (§2 vs an earlier n=1000 probe); CIFAR uses the 10k test split only.
- **The NN floor dominates the whole protocol** — this is a property of masked-image retrieval, not of CLU. If the program wants a *capacity win*, this protocol does not deliver one; the honest artifacts are (a) CLU > closed-form Hopfield on noise, (b) the CIFAR Hopfield-needs-a-learned-kernel finding, (c) retry (+47pp) and retention (capability) — none of which is a capacity-curve headline.
- **CLU noise cliff at σ≥0.4** is the fixed narrow-well width; a wider `s` would raise noise tolerance but lower capacity (the `Δ_req` trade). I did not sweep `s` per-axis (would be per-axis tuning); a single `s`-vs-(capacity,noise) Pareto is the clean follow-up.
- **entmax15** activation is implemented (bisection) and available but not in the default arm list (softmax+sparsemax); enable via `activations`.
- **U-Hop's UMHN (learned kernel)** is deliberately excluded (it learns) — but it is the *actual* CIFAR SOTA, so a fair "CLU is competitive" claim on CIFAR must note that the closed-form Hopfield line it beats is not the U-Hop headline.

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new benchmark result.** On the matched modern-Hopfield/U-Hop retrieval protocol (MNIST+CIFAR-10, dropout-0.5 mask, mean-sqdiff metric): a **designed CLU register (Gaussian dense associative memory + damped Verlet) TIES/loses to closed-form dense & sparse Hopfield on masked-capacity, BEATS them on noise robustness (MNIST σ≤0.2: 0.92 vs 0.42; CIFAR σ≤0.2: 0.86 vs 0.008), and loses to a trivial NN pixel-space floor on both axes.** CLU does **not** beat SOTA; the exponential is not novel (Hu–Wu–Liu). Do NOT claim a capacity win on this benchmark.
2. **⭐ §6 — the two real CLU deliverables here are CAPABILITIES, not the curve:** (a) **retry** = +46.9 pp at ×1.5 compute with a −38 pp blank guard (an adaptive-compute curve Hopfield cannot draw); (b) **per-item retention** = permanent + scheduled-decay items in one store (no Hopfield time-axis analogue). Frame the benchmark around these.
3. **⭐ Prior-art/positioning finding:** **closed-form modern-Hopfield retrieves raw-pixel CIFAR at chance for all β** — U-Hop's own headline (UMHN) *learns* a kernel to fix exactly this. So "nothing learned on both sides" is **MNIST-admissible but CIFAR-fragile**: on CIFAR the SOTA line requires learning. Cite when positioning the "designed CLU is admissible" claim.
4. **§7 correction to the scout/task premise:** the U-Hop success metric is **mean `sqdiff`**, not cosine>0.9; the mask is `torch.dropout(0.5)` (zero + ×2), not mask-to-zero. Pinned from `MAGICS-LAB/UHop @ cdac754`.
5. **§7 — resolved a real x64 crash** in a new code path (CLU `lax.scan` carry dtype under `jax_enable_x64`); the §7.2 landmine now also has a *code*-level trap (float32-pinned sub-modules inside an x64-promoted rollout), not just marginal test flips. New code launches carries in `jnp.result_type(float)`.
6. **New CLI/config surface:** `chlu exp-hopfield-capacity`, `ExperimentHopfieldCapacityConfig`; new core class `GaussianMemoryPotential`. `clu_s_frac=0.3` is measurement-derived (basin-merge at ≥0.5), not arbitrary.
