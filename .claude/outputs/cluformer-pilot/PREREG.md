# PREREG — `cluformer-pilot` (tier iii: the full C2W1 CLU as a streaming block's memory)

**Written 2026-08-01, BEFORE any run** (protocol §5 pre-registration rule; task §4 "⛔ BEFORE ANY RUN").
Nothing in this file is measured. Every number is either (a) pure arithmetic on declared geometry, or
(b) a signed prediction with its derivation. Author: `experiment-engineer`. Base `main @ 21a6dc4`.

**Head rulings echoed (§0, binding):** (1) **the C2W1 FULL store** — no full-CLU feature turned off;
(2) **compute on CSF3** for every 26–47 M run.

**Dial:** tier iii. **Control:** the SYSTEM-LEVEL SWAP (matched-state GRU / TTT-class cell in the same
block). **Not** the tier-i settle-deleted launder.

---

## 0. The configuration this PREREG is written against

Two scales. Everything except the memory **cell** is bit-identical across arms at a given scale.

| | **TOY** (local, laptop) | **PILOT** (CSF3, the 26–47 M leg) |
|---|---|---|
| stream | enwik8, byte-level | enwik8, byte-level (WT-103 loader built, not the venue) |
| `d_model` / `n_layers` | 64 / 2 | **512 / 12** |
| total params | ≈ 0.25 M | **≈ 28.3 M** ✔ in [26, 47] M |
| chunk granularity `C` | 32 tokens | **64 tokens** (§0.3 mitigation 1) |
| seq len | 256 | 1024 |
| store `addr_dim d` / `payload_dim m` | 2 / 1 | **8 / 4** |
| store `capacity K` / `atoms_per_item` | 8 / 128 | **32 / 256** |
| `n_atoms` (w23 floor `max(A·K, 384, 512·√2^d)`) | 1024 | **8192** |
| ψ (learned trajectory DeepSets) width | 48 | **128** |
| per-read Verlet budget | 32 + 32 | **64 + 64 = 128** (default); D5 sweeps {16,32,64,128,256,1200} |
| write inner steps per chunk | 4 | **4** |
| seeds | 3 (0,1,2) | 3 (0,1,2) |

**Levers ON (ruling 0.1 — none turned off):** learned `V_θ` store (not arrays) · derived addressing ·
admission policy · per-item lifetimes (`leak`) · masked/local write · **permitted basin interaction under
the soft certificate SC-1…SC-7, `B` declared** · learned φ in · **learned trajectory ψ out** · two-phase
relaxation · **γ as a trainable selector** (the 14× channel, §A13 design rule) · trajectory *and* settled
point available to ψ · confidence-gated retry · controller verb set {admit, place, evict, decay, route,
retry, stop} + {anneal, expand} · all 13 monitors live.

**Levers staged, not deleted, with the band declared:**
- `write_steps` **300 → 4 per chunk**. This is the §0.3 chunk-granularity re-budget, not a deletion: the
  same masked local `write_loss` objective, fewer inner steps, differentiably unrolled (the Titans/TTT
  convention, which the swap control also gets). ⚠ **Monitor #13 (maturity, floor 40 steps) is therefore
  PREDICTED TO TRIP on every chunk, and that is a reported artifact, not a bug.**
- Per-read Verlet steps **1200 → 128**. Same §0.3 mitigation; D5 measures the curve.

**Declared modelling choice, made BEFORE measuring (§4 requires it stated up front):** ψ's width is chosen
so that a **two-sided-matched** TTT-class swap *exists* (see §5). A narrower ψ makes the CLU cell's
`params ≈ state` and then **no** cell with input/output projections can match both. Choosing ψ to make the
control constructible strengthens the control; it is declared here so it cannot be presented later as a
convenience.

**Causality convention (declared, so no leak can be laundered as a win).** At chunk boundary `c`:
**read first, then write.** Query `= φ(pool(chunk c−1))` is run against a store holding chunks `0…c−2`;
the retrieved vector is added to *every* token of chunk `c`; only then is chunk `c−1` written. No token
ever sees its own chunk's pooled summary. Identical convention in every arm.

---

## 1. ⭐ THE DIRECTIONAL FALSIFIER (D4) — the binding one

> **DF1 (primary; sign + threshold + tolerance + seeds).**
> At the PILOT scale (28.3 M, enwik8, 3 seeds, seed-mean held-out bits-per-character):
> **tier iii is ALIVE only if `bpc(CLU) ≤ bpc(GRU_matched-params) − 0.02` AND
> `bpc(CLU) ≤ bpc(TTT_matched-both) − 0.02`, with the CLU seed-mean below both baselines'
> seed-means by more than the sum of the two ±1 s.e. bars.**
> If either inequality fails, **tier iii is NOT alive at 26–47 M** and I say so.

> **DF2 (monotone trend — the scale leg).**
> The margin `Δ(scale) = bpc(CLU) − bpc(GRU_matched-params)` must be **non-increasing** from TOY
> (0.25 M) to PILOT (28.3 M) by at least 0.01 bpc. **A flat or worsening Δ removes "needs > 500 M" from
> the menu**: an effect that is not already moving toward the CLU across a 113× scale span cannot be
> claimed to appear at 18× further scale.

> **DF3 (dynamic evaluation — pre-committed, non-renegotiable, D3).**
> Krause et al. (ICML 2018) dynamic evaluation is applied **to every arm** in the same table.
> **If `Δ_dyneval = bpc_dyn(CLU) − bpc_dyn(GRU_matched-params)` is ≥ `Δ_static − 0.02` (i.e. the CLU's
> advantage does not survive), the primary is dead.** Stated in the task, restated here, and not
> renegotiable at reporting time.

**Derivation of the thresholds.** 0.02 bpc is ≈ 1.5 % of a plausible pilot-scale byte-LM bpc (~1.3–1.6),
and is above the seed-to-seed spread we should expect: at 28 M with 3 seeds and a fixed data order the
usual byte-LM seed s.e. is ~0.005–0.01 bpc, so 0.02 is ≈ 2–4 s.e. The cell is **5.7 % of the model's
parameters** (arithmetic, §5) — a swap of 5.7 % of the parameters that moves bpc by less than 0.02 is not
distinguishable from optimisation noise, so a smaller threshold would be a falsifier I could not fail.

**⛔ A falsifier I *can* fail.** My own signed prediction (§3) is that **DF1 FAILS**. I am pre-registering
a bar I expect to miss, and the miss is the finding.

## 2. THE SCALE FALSIFIER (D4 §4.2) — when "needs > 500 M" is honest, and when it is not

- **"Needs > 500 M" is available ONLY if DF2 passes** — i.e. Δ improves for the CLU by ≥ 0.01 bpc from
  TOY to PILOT — **and** the CLU arm's own loss curve is not saturated (train bpc still falling at the
  compute budget's end on ≥ 2 seeds). Then it is a **feasibility finding, flagged to the Head, never a
  null.**
- **"Needs > 500 M" is NOT available and must not be written** if Δ is flat (|ΔΔ| < 0.01) or worsening.
  In that case the honest conclusion is: **the full CLU as a block memory does not beat its own swap at
  26–47 M and shows no trend that predicts it would at 500 M.**
- **Third possibility, pre-registered:** the run does not reach the pilot scale at all (see §7). Then S4
  is a **declared NOT-RUN**, never a null, and DF1/DF2/DF3 are **NOT SCORED**.

## 3. PREDICTED SWAP MARGINS (signed, per arm, derived — not guessed)

Held-out bpc, enwik8, PILOT scale, seed-mean. **Sign convention: positive = CLU is WORSE.**

| comparison | **predicted margin (bpc)** | range | derivation |
|---|---|---|---|
| **CLU − GRU_matched-params** | **+0.12** | [+0.02, +0.45] | (i) the CLU cell delivers `dim = 12` floats per chunk from `K = 32` items; the GRU delivers `h = 203` floats of unconstrained state — **17× the read width** at matched parameters. (ii) `trainability-spike` Stage 0: the trajectory channel carried **0 qualifying cells of 588 × 3 seeds** on a *frozen* store; here it is trained, which is the one thing that could move it, so I do not predict a total wipe-out. (iii) `orgdiv-prereg` Theorem O1 (**2 distinct settled points from 4000 queries**) says the settled-point image is tiny; the trajectory is the only channel with width, and it costs 17.1×. |
| **CLU − TTT_matched-both** | **+0.08** | [−0.05, +0.35] | The TTT cell has the same `params ≈ state` shape as the store and the same chunk convention, so it is the *harder* control on bytes but the *easier* one on optimisation-friendliness (one closed-form linear inner step vs 4 unrolled physics steps + 128 Verlet steps). I predict a smaller gap than the GRU's because TTT's read is also a low-rank projection, but still positive. |
| **CLU − memory-deleted (+0 B) control** | **−0.06** | [−0.20, +0.02] | The memory must at least beat *no memory*. If this is ≥ 0 the block's memory is inert and **S1 has failed in-system**. |
| **CLU − blank-store control** | **−0.05** | [−0.18, +0.02] | Same bar against a store that is read but never written (collapse mode #4). |
| **GRU − memory-deleted** | **−0.10** | [−0.30, −0.02] | The swap is expected to use its state. |

**⛔ Pre-committed (FB3 spirit, task §5): if the matched-state GRU wins, I say the matched-state GRU wins.
That is a result of the pilot and I will not re-frame it.**

## 4. PREDICTED EFFECT OF DYNAMIC EVALUATION (D3), per arm

Dynamic evaluation (Krause et al. 2018; the published criterion-3 weakness at **1.08 bpc**) adapts *all*
weights on the test stream with SGD. Predicted absolute improvement:

| arm | predicted Δbpc from dyn-eval | derivation |
|---|---|---|
| GRU_matched-params | **−0.12** | dyn-eval's gain is largest where the recurrent state is smallest, because SGD on the weights substitutes for the missing state. `h = 203` floats is small. |
| TTT_matched-both | **−0.07** | TTT already performs a test-time weight update; dyn-eval partially duplicates it. |
| **CLU** | **−0.05** | The CLU *also* already adapts at test time (the store is written on the test stream). **This is the pre-registered mechanism by which the CLU's advantage, if any, is expected to shrink under dyn-eval: dyn-eval is a substitute for exactly the thing the CLU sells.** |
| memory-deleted | **−0.15** | Nothing else adapts, so dyn-eval carries all of it. |

⇒ **Predicted `Δ_dyneval` (CLU − GRU) = +0.12 − (−0.05) + (−0.12) = +0.19 bpc**, i.e. **worse than
static.** By DF3's own rule this would make **the primary dead**. I am pre-registering that outcome as my
expectation, so that if the CLU *does* survive dyn-eval it is evidence rather than a rescued number.

## 5. PREDICTED BYTE LEDGER (two-sided, learned-initial-state rule applied to `V_θ` too)

**Rule (PREREG-Bprime §4):** *an initialisation is PARAMETERS; only the per-sequence deviation is STATE.
Both declared.* Applied to the store's `V_θ` init exactly as to the GRU's learned `h₀` and TTT's `W₀`.

### 5.1 PILOT (`d = 8, m = 4, dim = 12, K = 32, n_atoms = 8192, ψ width 128`)

| arm | **cell params** | **cell state (floats)** | **cell state (B)** | matched? |
|---|---|---|---|---|
| **CLU (full C2W1 store)** | `V_θ` init 114 688 + ψ 20 228 = **134 916** | `V_θ` deviation 114 688 + retained codebook `K·d` 256 = **114 944** | **459 776** | — (reference) |
| **GRU, matched-params** | `h = 203` → **134 398** (**−0.38 %**) | **203** | **812** | params ✔ · **state ✘ (CLU = 566.2× the GRU)** |
| **GRU, matched-state-bytes** | `h = 114 944` → **39 642 461 452** | 114 944 | 459 776 | state ✔ · **params ✘ — 1 525× an entire 26 M model** |
| **TTT-class, matched-both** (`W ∈ R^{657×175}`) | **134 944** (**+0.02 %**) | **114 975** (**+0.03 %**) | **459 900** | **✔ both, to 0.03 %** |
| memory-deleted (+0 B) | 0 | 0 | 0 | trivial substitute |

**Shared and identical on every arm (ledgered separately, not part of the cell):** embedding `256×512`,
learned positional `1024×512`, per-layer LayerNorm ×2, the intra-chunk causal depthwise conv (k = 4), the
token-wise MLP (4×), **φ: `512 → 12` (6 156 params)**, assimilation `12 → 512`, head `512×256`.
**φ is bit-identical across arms and its bytes are ledgered on every arm** (task §Dial declaration).

### 5.2 ⛔ PRE-REGISTERED PREDICTION: the falsifier "the swap is not a swap" **FIRES for the GRU**

This is arithmetic, not a measurement, so I register it as a *prediction that the measurement will
reproduce*: **matched params and matched state-bytes cannot both be hit for a GRU cell**, because a GRU's
`params/state = Θ(h)` while the CLU store's `params/state = 1.017` (its parameters *are* its state).
At `h` matching the store's state the cell alone is **1 525× a 26 M model**.
**Consequence I commit to now (task §5, "report the same day"):**
1. I report it as an **FB2-shaped finding at tier iii**, in the first screen, on the day it is confirmed.
2. I do **not** proceed on a one-sided match alone. The comparison is **bracketed**: the
   **TTT-class cell carries the two-sided match**, and the GRU is reported with *both* its ledger columns
   published and its state deficit (566×) stated in the table, never hidden.
3. If the TTT arm also fails to match both within 2 %, **tier iii has no control this wave** and I stop.

### 5.3 The item count the store gets at matched state-bytes (D2's real consequence)

Corrected byte law **`[A(D+2)+d]/(d+m)`** with `A = 256, D = 12, d = 8, m = 4`:
**`(256·14 + 8)/12 = 299.33×` per item.**
At the store's own 114 944 floats of state, a plain table row costs `d+m = 12` floats ⇒ the table gets
**9 578 rows** where **the store gets 32 items — 299× fewer.**
*(TOY: law `214.00×`; 1 712 rows vs 8 items.)*
⚠ **Floor check:** the law's floor is **2.40× at `n_spec = 1`, A = 1** — we are 125× above the floor
because `A = 256`. Predicted and declared, not discovered later.

### 5.4 TOY ledger (the scale I will actually run locally)

| arm | cell params | cell state (floats / B) |
|---|---|---|
| CLU | 5 120 + 2 737 = **7 857** | 5 120 + 16 = **5 136 / 20 544 B** |
| GRU matched-params (`h = 48`) | **7 731** (−1.60 %) | **48 / 192 B** — CLU = **107.0×** |
| GRU matched-state (`h = 5 136`) | **79 222 803** = 3× a 26 M model | 5 136 / 20 544 B |
| TTT matched-both (`W ∈ R^{442×12}`) | **8 029** (+2.19 %) | **5 304 / 21 216 B** (+3.27 %) |

⚠ At TOY the TTT match is only within **2.2 % / 3.3 %** (integer rounding on a small budget), against the
2 % bar of §5.2(3). **Pre-registered: the 2 % bar is scored at PILOT, where it is 0.03 %; TOY's 2–3 % is
declared as a rounding artifact of the small budget, not a failed match.**

## 6. CHUNK GRANULARITY AND THE PER-READ VERLET BUDGET (§0.3, both arms)

- **Chunk `C = 64` tokens (PILOT) / 32 (TOY).** ⭐ **The swap control's cell gets the identical chunk
  convention** — one state update and one read per chunk, never per token — so the comparison is matched
  (task §0.3.1 is explicit that this is required).
- **Per-read Verlet budget: 64 (phase 1, `γ_address`) + 64 (phase 2, `γ_read`) = 128 steps.**
  Predicted consequence, registered: at `γ_address = 0.05`, `ρ = √(1−γ) = 0.97468`, so `ρ^64 = 0.195`
  — **the read does NOT settle inside the budget.** I predict **monitor #1 (overdamping / `ρ_conv`) does
  not trip** (`ρ_conv` will be *large*, not small — #1 trips on over-convergence) and that
  **`residual = ‖∇V(q*)‖` stays O(1e-1), reported per arm.** The read is an *anytime* read by
  construction; that is D5's axis, and the price is declared, not hidden.
- **Sequence-scale arithmetic (the §2.2 constraint, faced):** PILOT = 1024 tokens / 64 = **16 chunk
  boundaries per sequence per layer × 12 layers = 192 reads/sequence**, each 128 Verlet steps ⇒
  **24 576 Verlet steps per sequence** (vs 1024 × 1200 × 12 = 14.7 M for a naïve per-token settle — a
  **600× reduction**, which is precisely what chunk granularity buys).
- **Trajectory ψ price:** 17.1× the point read (`trainability-spike` §3). Registered and carried on every
  trajectory number in the report.

## 7. WALL-CLOCK AND ALLOCATION BUDGET, AND WHAT I CUT FIRST

**⛔ Declared BEFORE the first job, per §0.2.**

| | budget | falsifier |
|---|---|---|
| **Local (laptop, this session)** | **≤ 8 h wall-clock total**, ≤ 1 worktree, no 26–47 M anything | if S1+S2 are not standing at 6 h, stop and report a partial |
| **CSF3** | **≤ 108 A100-hours** = 3 arms × 3 seeds × ≤ 12 h, `-p gpuA -G 1`, ≤ 4 concurrent (free tier) | if a single arm×seed will not finish inside `-t 12:00:00`, **stop and report** — do not resubmit at larger `-t` |

**⚠ Registered before running anything: I have no CSF3 route from this machine.**
`ssh csf3` fails at DNS (`Could not resolve hostname csf3.itservices.manchester.ac.uk`) — the runbook
records that off-campus access needs GlobalProtect VPN, which this agent cannot establish. **Therefore
the honest prior on S4 is that it is submitted by the Head, not by me**, and I pre-register that
**S4 will be reported as a declared NOT-RUN-BY-AGENT with a ready-to-submit job script, never as a null,
and DF1/DF2/DF3 will be NOT SCORED** unless the Head runs it and the artifacts come back.

**Cut order if the budget is exceeded (declared in advance):**
1. **D5** (the anytime shape curve) — secondary by the task's own wording.
2. **The TTT arm at PILOT** (keep it at TOY, where the two-sided match is demonstrated).
3. **`n_layers` 12 → 8 with `d_model` 512 → 640** (keeps the model at ~26 M, the bottom of the band).
4. **Never:** the seed count (3 is the floor), the swap control, the dyn-eval column, or a monitor.

## 8. SECONDARY PRE-REGISTERED PREDICTIONS (scored in the report)

| # | prediction | why |
|---|---|---|
| P1 | **`‖∂L/∂φ‖` through the trajectory read is > 1e3× the settled-point arm's** (which will be **0.0 exactly** under the implicit settle) | `trainability-spike` measured 2.42e6 on a probe; in-system through 12 layers I predict the ratio survives but shrinks — point estimate **1e4**, range [1e3, 1e7] |
| P2 | **Monitor #13 (maturity) TRIPS on every chunk** (4 write steps < the 40 floor) | arithmetic; declared in §0 |
| P3 | **Monitor #9 (lifetimes) TRIPS** | uncleanable-by-any-verb, `full-clu-harness` PREREG §0, inherited |
| P4 | **Monitor #8 (certificates) TRIPS** once SC-1…SC-7 are on with permitted basin interaction | `full-clu-harness` S4: basin interaction and the merge certificate are mutually exclusive by construction |
| P5 | **Monitor #10 (dead axis) does NOT trip on `traj_stride`** in this build | R-3: the shipped ψ never read the buffer; **my ψ is a trajectory ψ, so the buffer is consumed** |
| P6 | **Monitor #4 (blank) does NOT trip** | `trainability-spike` §5.2 refuted the leak for a *pooled* DeepSets ψ (0.148 vs 0.125 chance); my ψ is DeepSets, and `AttentionPsi` stays quarantined |
| P7 | **Monitor #12 (starvation) TRIPS** — a few chunks capture most writes | allocation collapse is a gradient attractor (T3 corollary); I initialise φ's address head away from the corner and report `‖∂L/∂φ_addr‖` + address-utilisation entropy at init as the liveness anchor |
| P8 | **γ (trainable selector) moves further than M** in relative terms over training | friction is the 14× stronger channel (§A13) — predicted `|Δγ|/γ₀ ≥ 3 × |ΔM|/M₀` |
| P9 | **`d/s` at the pilot store stays ≥ 3** using **`bprime-c6`'s ruler (`s = 0.40`, achieved separation)**, not T5.5's refuted `1.9` | the ruler is named in every `d/s` statement per the task's errata |
| P10 | **The store's `sep/σ_q` degrades below 5.15 once SC is on** (monitor #8's N2 bar) | same mechanism as P4 |

## 9. WHAT WOULD MAKE ME SAY "S1 FAILED" (the first-order finding, task §5)

- A collapse mode fires **and its designed restoring verb cannot clear it** inside the stream. I name the
  mode, its monitor, and the verb that failed. That is intervention §5's thesis under test and a loud
  failure is the deliverable.
- Or: the block's memory output is measurably inert — `CLU − memory-deleted ≥ 0` (§3 row 3).

## 10. WHAT WOULD MAKE ME SAY "S2 FAILED"

Gradients do not flow end-to-end through the trajectory read at usable wall-clock — concretely
**`‖∂L/∂φ‖ < 1e-8` (numerically dead) or > 300 s per training step at TOY scale**. That is **T3 biting
in-system**, a first-order finding, not a setback, and it is reported with the measured norm.

---

*Filed before any code was run against any data. Predictions above are scored verbatim in
`.claude/outputs/cluformer-pilot.md` §PREREG scorecard.*
