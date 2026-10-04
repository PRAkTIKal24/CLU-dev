# cluformer-pilot — experiment-engineer report

Task + acceptance criterion: build the **full C2W1 CLU store as a streaming block's memory** on a real
stream, with the **system-level swap** as its control; acceptance is `full-clu-harness`'s, inherited
verbatim — **"the system runs the stream without tripping a silent collapse mode. Does not collapse, not
wins."**
Status: **partial — S1 ◐ / S2 ✅ / S3 ✅ / S4 ⛔ NOT RUN (declared, never a null).**

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5, first-10-lines rule)
> **R1 — ⛔ "The swap is not a swap" FIRES for the GRU, and it is arithmetic, not a measurement.** A GRU
> cell has `params = Θ(h²)` against `state = Θ(h)`; the CLU store's parameters **are** its state
> (`params/state = 1.675` measured at toy, → 1.02 at pilot geometry). Matched params ⇒ the CLU carries
> **100.9× the GRU's state bytes**; matched state-bytes ⇒ the GRU cell alone is **79 469 668 params**
> (3.06× an entire 26 M model at toy; **1 525× at pilot geometry**). ⭐ **A TTT-class cell CAN match both
> (+0.046 % params, −0.097 % state)** and is therefore the two-sided control. *(Owner: Hub — this is
> FB2's shape at tier iii and §5 says report it the same day. Both columns are published in
> `swap_ledger`; the pilot proceeds bracketed by BOTH one-sided GRU columns plus the two-sided TTT arm,
> never on a single one-sided match.)*
> **R2 — the store is measurably INERT inside the block, and the cause is NOT the write step budget.**
> Live-store and blank-store held-out NLL are equal to **0.00e+00 / −5.96e-08 / 0.00e+00** (3 seeds) —
> float32 round-off. Self-probe acquisition sits **exactly at chance** on all 3 seeds, and a
> write-budget sweep (4 → 16 → 64 inner steps, 16×) moves acquisition **not at all** while well depth
> saturates at 0.045 against the shipped store's fitted `D = 0.46–0.80`. *(Owner: Hub → whoever holds the
> tier-ii/tier-iii write objective; §7 below has the mechanism and the three candidate fixes.)*
> **R3 — `full-clu-harness`'s "the shipped read is within 2.3× of monitor #1's band edge" does not
> survive chunk granularity.** At the affordable 48-step read `ρ_conv = 0.834–0.836`; at the shipped
> 1200-step read the same block gives **5.06e-07 / 2.60e-07**. **The read simply does not settle at
> chunk granularity**, monitor #1 trips on *under*-convergence on 3/3 seeds, and every γ-band statement
> must now be scoped by *read budget* as well as by harness. *(Owner: curator/theorist.)*
> **R4 — monitor #11's `saddle_reach_threshold` divides by zero on a group whose wells were never dug**
> (`D ~ 1e-86` ⇒ depth-weighted width underflows ⇒ `2·α·s² == 0.0`). It crashed seed 2 of the shipped
> run. Guarded in my caller, **not** in `monitors.py` (read-only to me). *(Owner: engineer — the guard
> belongs in `saddle_reach_threshold` itself.)*

---

## ⭐ FIRST SCREEN

**Head ruling 0.1 echoed — WHICH CLU: the C2W1 FULL STORE, no full-CLU feature turned off.** Honoured
and asserted in code (`tests/test_cluformer_pilot.py::test_no_full_clu_feature_is_turned_off`): learned
`V_θ` store (never arrays), derived addressing, admission, per-item lifetimes, masked/local C3 write,
permitted basin interaction with the **soft certificate ON**, learned φ in, **learned trajectory ψ out**,
two-phase relaxation, **γ and M as trainable selectors**, trajectory *and* settled point available to ψ,
confidence-gated retry, the controller's verb set, all 13 monitors + M14 live. ⛔ **The block-form
`CLUBlock` is NOT the memory** — it is the w20/w21 driven-Hamiltonian recurrence with no store, it is
ruled out, and the memory slot here holds `CluStoreCell`, which composes
`LearnedVStore` + `write_loss` + `atom_write_mask_fn` + `CluControllerV0` + `default_registry` through
their public API. **Zero edits to any store file.**

**Head ruling 0.2 echoed — WHERE THE COMPUTE RUNS: CSF3.** ⛔ **I have no CSF3 route.**
`ssh csf3` fails at DNS (`Could not resolve hostname csf3.itservices.manchester.ac.uk`) — off-campus
access needs GlobalProtect VPN, which this agent cannot establish. **This was pre-registered before any
run** (PREREG §7). Consequence: **every 26–47 M number is NOT RUN**, a ready-to-submit job script is
delivered (`scripts/csf3/job_gpu_cluformer.sh`, `bash -n` clean, UNTESTED-ON-CLUSTER), and **DF1/DF2/DF3
are NOT SCORED at pilot scale.**

**§3 stage reached: S3.** S1 ◐ (the block runs a real stream and **5 of 14 monitors trip, none silently**
— but the memory is inert, see R2) · S2 ✅ (**the single most valuable result: gradients flow end-to-end
`token → φ → store → trajectory ψ → loss`**) · S3 ✅ (the swap is defined, ledgered and trained on
identical data order and seeds) · **S4 ⛔ NOT RUN.**

### ⭐ The directional falsifier, verbatim as pre-registered, and its verdict

> **DF1 (primary).** At the PILOT scale (28.3 M, enwik8, 3 seeds, seed-mean held-out bpc): *tier iii is
> ALIVE only if `bpc(CLU) ≤ bpc(GRU_matched-params) − 0.02` AND
> `bpc(CLU) ≤ bpc(TTT_matched-both) − 0.02`, with the CLU seed-mean below both baselines' seed-means by
> more than the sum of the two ±1 s.e. bars.*

**VERDICT — at PILOT scale: ⛔ NOT SCORED (S4 NOT RUN).**
**At TOY scale (0.16 M, 3 seeds, reported as an instrument reading and explicitly NOT a pilot number):
DF1 FAILS, and it fails in the direction I pre-registered.** Paired per-seed margins
(`CLU − opponent`, **positive = CLU worse**):

| comparison | per-seed | mean ± s.e. | verdict |
|---|---|---|---|
| CLU − GRU (matched params) | +0.0085, +0.0018, +0.0213 | **+0.0106 ± 0.0057** | CLU worse on **3/3** |
| CLU − TTT (matched both) | +0.0366, +0.0159, −0.0055 | **+0.0157 ± 0.0122** | CLU worse on 2/3 |
| **CLU − memory-DELETED (+0 B)** | +0.0181, +0.0019, +0.0012 | **+0.0070 ± 0.0055** | ⛔ **the memory is a net COST on 3/3** |
| CLU − echo (+0 B trivial reader) | −0.0140, −0.0148, −0.0027 | −0.0105 ± 0.0039 | CLU beats the trivial reader |

⛔ **Pre-committed and honoured (task §5, FB3 spirit): the matched-state GRU wins, and I say so.**

**DF2 (the monotone scale leg): ⛔ NOT SCORED.** It needs the TOY *and* the PILOT margin; only the TOY
leg exists. ⚠ **Therefore "needs > 500 M" is NOT available and is not written anywhere in this report**
(PREREG §2 pre-committed exactly this).

**DF3 (dynamic evaluation): ⛔ MOOT, not passed.** Dyn-eval (Krause et al., ICML 2018), strictly causal,
LR swept per arm, **in the same table**: every arm improves by a uniform **−0.0018 to −0.0020 bpc**, so
the CLU has no advantage for dyn-eval to erase. DF3's pre-committed consequence ("if the dividend
vanishes with dynamic evaluation in the table, the primary is dead") **cannot fire because the primary
was already dead at DF1**. Reported as MOOT, never as a pass.

---

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch | `agent/experiment-engineer/cluformer-pilot`, worktree `../CHLU-pilot`, base local `main @ 21a6dc4` |
| commits (7) | `477a61b` · `aebfc70` · `fae6c29` · `f173db6` · `a4e1848` · `7bc166a` · `7f1b9f1` — see §12 |
| env | **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), no worktree `uv sync` (w6 hazard avoided). **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, numpy 2.4.1, **CPU** |
| ⚠ **JAX on CSF3** | **UNKNOWN — no run happened there.** The PREREG required reporting the resolved JAX version on *both* machines; only the local one exists. `scripts/csf3/setup_env_job.sh` pins `uv sync --frozen --extra cuda` off the same lock, so parity is *expected*, not *verified*. **Flagged.** |
| artifacts | `.claude/outputs/cluformer-pilot/` — `PREREG.md`, `pilot_toy_seed{0,1,2}_S3.json`, `pilot_toy_aggregate.json`, `pilot_toy_panels.png`, `write_budget_sweep.json`, `shipped_budget_control.json`, `run_toy_s3.log`, `run_toy_seed2.log` |
| command | `PYTHONPATH=. python -u -m chlu.experiments.exp_cluformer_pilot --stage s3 --scale toy --seeds 0 1 2 --steps 200 --d5` |
| **seeds** | **0, 1, 2 — three seeds on every reported number.** Nothing single-seed is offered as a result. |
| **scale** | ⛔ **TOY: `d_model 64`, `n_layers 2`, `seq_len 512`, `batch 4`, 159 830 params.** ⛔ **NOT 26–47 M and never reported as such.** The 28.3 M PILOT config is declared in `exp_cluformer_pilot.PILOT` and in the job script; it was not run. |
| data | **enwik8**, byte-level, canonical **90/5/5 positional** split, first **4 000 000 B** staged (prefix, deterministic); vocab 256 (196 byte values observed); sha256 of the archive `547994d9…2534bc` |
| store | learned `V_θ` = `DesignFreedomPotential(rung="free_mlp", family="atoms")`, `addr_dim 2`, `payload_dim 1`, `dim 3`, **`n_atoms 1024`** (= `max(128·8, 384, 512·√2²)`), `atoms_per_item 128`, `capacity 8`, **`budget 6`** (real capacity pressure), `atom_width 0.3`, `atom_init_scale 1.0`, `atom_depth_init 1e-4`, `confine (α) 0.05`, masked write ON |
| levers | **ALL stage flags TRUE** (`lifetimes, admission, capacity_pressure, deletion, basin_interaction, retry, trajectory_read`); **`soft_certificate = True`**, `ζ = 0.6`, `sep_expected = 0.6492` ⇒ **`d_safe = 0.3895`** (SC-1; the derived in-band radius would be 0.9864); `leak 0.02`, `amp_floor 0.05`, `retry_tau 0.5`, `retry_max_rounds 1` |
| read | `dt 0.05`, **`γ_address 0.05`, `γ_read 0.02` — both TRAINABLE**, **24 + 24 (+24 gated retry) Verlet steps**, `traj_stride 8`, `kinetic_mode newtonian_learned` with `M` trainable (init `I`), read_mode **trajectory**, ψ = `DeepSetsPsi(hidden 32, depth 2)` over the strided buffer |
| write | masked/local, **4 inner steps**, **sign-SGD at lr 0.05** (see §7.1 — load-bearing), `n_perturb 8`, `σ_addr 0.25`, `σ_pay 0.6`, `margin 0.15`, `barrier 0.2`, `barrier_pairs "nn"`, crowd targets = the live codebook |
| chunk | **C = 32 tokens** ⇒ 16 chunk boundaries/sequence/layer; **identical convention in every arm** |
| φ | shared, **bit-identical across arms** (asserted), `d_model → 3` MLP, **calibrated gain 3.839 / 6.356 / 5.650** (seeds 0/1/2) — the declared anti-collapse initialisation, §6.3 |
| optimiser | AdamW, warmup-cosine, peak `lr 1e-3`, `warmup 20`, `grad_clip 1.0`, `wd 0.0`, **200 steps** (409 600 tokens), **identical schedule, data order and seeds in every arm** |
| dyn-eval | SGD, LR grid `{1e-4, 1e-3, 1e-2}`, best per arm, 4 contiguous test batches, **strictly causal** (score before update) |
| langevin / temperature | **N/A** — deterministic, `T = 0`, `p₀ = 0`, no Langevin step anywhere. §7.22's `langevin_noise` discipline does not apply. |
| wake–sleep / Lyapunov | **N/A** — neither `train.py` nor `train_generative.py` is used. This is a third path (`train_cluformer.py`), byte-LM cross-entropy. |
| wall clock | **3 seeds = 1.35 h of measured run** (1741 + 1725 + 1411 s) + probes; **total agent session ≈ 13 h against a declared 8 h local budget — OVERRUN, see §10** |
| CSF3 allocation | **0 A100-hours spent** of ≤108 budgeted. Not reachable. |

---

## 1. What I built

1. **`chlu/data/enwik8.py`** (new) — byte-level, canonical **positional** 90/5/5, download-once staging
   through the concurrency-safe `download_file`, a **contiguous order-preserving** evaluation iterator
   (the only legal one for an arm with a persistent memory) and a **seeded** training iterator (identical
   data order across arms is a requirement of the swap, not a nicety).
   **`chlu/data/wikitext.py`** (new) — same surface, byte + word level, vocabulary from **train only**.
   Declared **BUILT, NOT THE VENUE**: byte-level enwik8 removes the tokenizer as a confound.
2. **`chlu/core/blocks.py`** (additive; `CLUBlock`'s 3-way key split untouched and now regression-tested)
   — `CluStoreCell` (the full store as a cell), `StreamBlock`/`StreamModel` (the shared shell),
   `MatchedGRUCell`, `MatchedTTTCell`, `NullMemoryCell`, `EchoMemoryCell`, `WritePlan`,
   `solve_matched_gru`, `solve_matched_ttt`, `swap_ledger`, `store_byte_law`.
3. **`chlu/training/train_cluformer.py`** (new, mine) — the **decision-replay** two-pass forward, the
   blocking shell-identity assertion, dyn-eval, the S2 gradient probe, the 13-monitor pass with real
   context, the φ-gain calibration, the anytime curve.
4. **`chlu/experiments/exp_cluformer_pilot.py`** (new) + **`scripts/csf3/job_gpu_cluformer.sh`** (new).
5. **Tests: +52** (`test_blocks.py` 22, `test_cluformer_pilot.py` 16, `test_data_enwik8.py` 8, +6 within).

⚠ **No CLI hook.** `chlu/cli/experiment_cmd.py` is read-only to this task (`bprime-c6` owns it), so the
runner is invoked as a module. Flagged for the Hub — the general repo convention wants a hook.

### 1.1 ⭐ The one design decision the Hub should review first: **decision replay**

The C2W1 controller's verbs are **discrete** and branch on `numpy` values, so it cannot run inside a
traced/differentiated forward (`np.asarray(tracer)` raises). Deleting it is forbidden (ruling 0.1). So
the forward is split: **pass 1 (concrete, detached)** runs the *real* `CluControllerV0` — admission,
placement, eviction, decay, every guard — and emits a `WritePlan` per layer; **pass 2 (differentiable)**
replays it. Nothing about the controller is weakened. What is given up is `∂(decision)/∂θ`, which is
**zero anyway** (the verbs are discrete) and which T3's corollary identifies as a gradient *attractor*
rather than a channel worth having. Cost: the plan pass is **77–84 % of the CLU arm's wall clock**
(§8.2) — a real, reported price.

### 1.2 The causality convention (declared before measuring)

At chunk boundary `c`: **read first, then write.** Query `= φ(pool(chunk c−1))` runs against a store
holding chunks `0…c−2`; the retrieved vector is added to *every* token of chunk `c`; only then is chunk
`c−1` written. No token ever sees its own chunk's pooled summary. Reading a store that was just handed
the answer is an echo and is the cheapest way to launder a win. Tested
(`test_the_block_is_causal_in_the_chunk_sense`).

---

## 2. ⭐⭐ THE SWAP TABLE — 3 seeds, dyn-eval column **in the table**, byte ledger attached

⛔ **TOY scale (159 830 params). This is an instrument reading, not a tier-iii result, and no line of it
may be quoted as a 26–47 M number.**

| arm | **bpc static** (mean ± s.e., 3 seeds) | **bpc dyn-eval** (mean ± s.e.) | cell params | **cell state B** | CLU−arm (paired) | total params |
|---|---|---|---|---|---|---|
| **CLU (full C2W1 store)** | **4.6191 ± 0.0143** | **4.6173 ± 0.0143** | 8 616 | **20 576** | — | 159 830 |
| GRU, **matched params** | **4.6086 ± 0.0183** | 4.6067 ± 0.0183 | 8 682 (**+0.77 %**) | 204 (**CLU = 100.9×**) | **+0.0106 ± 0.0057** | 159 962 |
| TTT-class, **matched BOTH** | **4.6034 ± 0.0207** | 4.6014 ± 0.0205 | 8 620 (**+0.046 %**) | 20 556 (**−0.097 %**) | **+0.0157 ± 0.0122** | 159 838 |
| memory-deleted (+0 B) | 4.6121 ± 0.0184 | 4.6103 ± 0.0184 | 0 | 0 | **+0.0070 ± 0.0055** | 142 598 |
| echo (+0 B trivial reader) | 4.6296 ± 0.0161 | 4.6278 ± 0.0161 | 0 | 0 | −0.0105 ± 0.0039 | 142 598 |
| ⛔ GRU, **matched state-bytes** | **NOT CONSTRUCTED** | — | **79 469 668** | 20 576 | — | *3.06× a 26 M model* |
| **blank-store control** (CLU) | **4.6191** — *bit-identical to the live CLU* | — | 8 616 | 20 576 | **0.00** | 159 830 |

**Shared shell, bit-identical on every arm and asserted at runtime** (`assert_shared_shell_identical`):
**142 598 params / 570 392 B** — embedding `256×64`, learned positional `512×64`, 3 LayerNorms/layer, the
intra-chunk causal depthwise conv (k = 4), **φ (`64→3`, identical, ledgered on every arm)**, the
assimilation projection `3→64`, the 4× token-wise MLP, the head. **Only `block.cell` differs.**

- **Dyn-eval is uniform**: −0.0018 (CLU), −0.0019 (GRU), −0.0020 (TTT), −0.0018 (none/echo). Best LR
  `1e-2` for every arm. It re-orders nothing.
- ⭐ **The blank-store control is the decisive row.** Live-store and blank-store held-out NLL differ by
  **0.00e+00 / −5.96e-08 / 0.00e+00** across seeds — i.e. **not at all, at float32 resolution.**

### 2.1 The two-sided byte ledger (learned-initial-state rule applied to `V_θ` too)

*Rule (PREREG-Bprime §4): an initialisation is **PARAMETERS**; only the per-sequence deviation is
**STATE**. Both declared. Applied to `V_θ` exactly as to a GRU's `h₀` and a TTT cell's `W₀`.*

| arm | PARAMS (init) | STATE (per-sequence deviation) |
|---|---|---|
| CLU | `V_θ` init 5 120 f + ψ 3 424 f + selectors 72 f = **8 616** | `V_θ` deviation 5 120 f + retained codebook `K·dim` 24 f = **5 144 f = 20 576 B** |
| GRU (matched params) | cell 8 631 f + learned `h₀` 51 f = **8 682** | `h` **51 f = 204 B** |
| TTT (matched both) | `W₀` 5 136 f + 4 thin projections + η = **8 620** | `W` **5 139 f = 20 556 B** |

**`params/state` = 1.675 (CLU), 170.2 (GRU), 1.677 (TTT).** That ratio *is* R1: a cell whose parameters
are its state can be matched on both axes; a cell whose parameters are quadratic in its state cannot.

### 2.2 ⭐ The item count the store gets at matched state-bytes (D2's real consequence)

Corrected law **`[A(D+2)+d]/(d+m)`** with `A = 128, D = 3, d = 2, m = 1`: **`(128·5 + 2)/3 = 214.00×`
per item.** At the store's own 5 144 floats of state a plain table row costs `d+m = 3` floats, so the
table gets **1 714 rows where the store gets 8 items — 214× fewer.**
⚠ Floor check: the law's floor is **2.40× at `n_spec = 1`, `A = 1`**; we sit **89× above the floor**
because `A = 128`. (Pre-registered pilot-geometry value: **299.33×**, 9 578 rows vs 32 items.)

---

## 3. ⭐ S2 — THE RESULT THE TASK CALLS THE MOST VALUABLE THING IT CAN PRODUCE

`‖∂L/∂φ‖` end-to-end through the block, **same model, same parameters, same plan**, only the read mode
switched at call time (a runtime override, so the two arms are literally the same object):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **trajectory read `‖∂L/∂φ‖`** | **1.022e-02** | **3.713e-03** | **4.815e-02** |
| settled-point read `‖∂L/∂φ‖` (**shipped sign-SGD write**) | **0.0 exactly** | **0.0 exactly** | **0.0 exactly** |
| **ratio** | **∞** | **∞** | **∞** |
| settled-point `‖∂L/∂φ‖` (**plain-SGD write**, diagnostic) | 1.551e-06 | 2.463e-06 | 2.446e-11 |
| **ratio (plain-SGD write)** | **6 558.5** | **1 503.1** | **1.97e9** |
| `‖∂L/∂γ‖` — trajectory / settled point | 4.72e-05 / **0.0** | 6.42e-05 / **0.0** | 1.96e-04 / **0.0** |
| `‖∂L/∂M‖` — trajectory / settled point | 2.45e-04 / **0.0** | 3.77e-05 / **0.0** | 5.89e-04 / **0.0** |

✅ **S2 MET.** Gradients flow `token → φ → store → trajectory ψ → loss` at usable wall-clock
(§8.2). Three things this establishes that the frozen-store probe could not:

1. ⭐ **T3 holds in-system, and the settled-point arm's γ- and M-gradients are `0.0` BITWISE on 3/3
   seeds.** `γ` and `M` really are trainable selectors *only* through the trajectory channel — measured
   inside a 2-layer language model, not on a probe.
2. ⭐ **The exact zero has TWO causes and I separated them.** `jnp.sign` has zero derivative, so the
   shipped **sign-SGD** inner write severs `∂(store state)/∂φ` *by construction*. Re-running the probe
   with a plain-SGD write leaves that channel open and isolates the theorem's own contribution:
   **1.5e3 – 2.0e9, geometric mean ≈ 2.3e5.** ⚠ **Anyone quoting "∞" must quote which write.**
3. ⚠ **The in-system ratio is NOT `trainability-spike`'s 2.42e6 and should not be conflated with it.**
   That number was a frozen store with no write path; here φ also writes, and on a differentiable write
   the ratio drops by up to 3 orders (1.5e3 on seed 1).

⚠ **A cost comparison this is NOT.** Wall-clock traj/point ≈ **0.96–2.3×**, *not* the 17.1× of
`trainability-spike`, because my settled-point arm still runs both rollouts and *then* solves the
implicit system — it is a gradient control, not an optimised point read. **The 17.1× price stands; this
number does not refute it and must not be quoted against it.**

### 3.1 The anti-attractor anchor (T3 corollary, §D1 design rule)

⛔ **There are no policy logits to differentiate**: the C2W1 controller's policy is **rule-based**, not
logit-parameterised. The learned object that steers allocation is **φ's address head**, and that is what
is reported: `‖∂L/∂φ_addr‖ = 1.84e-02 / 1.46e-02 / (seed 2)` at init, with normalised slot entropy
**0.618 / 0.727** (seed 0) and **0.819 / 0.853** (seed 1) per layer — **not at the degenerate corner.**
The φ-gain calibration (§6.3) is the declared initialisation that keeps it there.

---

## 4. ⭐ ALL 13 MONITORS + M14 — trip states as artifacts (the inherited acceptance criterion)

Final state, **identical on all 3 seeds**. `applicable = False` is reported as **inapplicable**, never as
a pass.

| # | monitor | state | value | why |
|---|---|---|---|---|
| 1 | overdamping | ⛔ **TRIPPED** 3/3 | `ρ_conv = 0.836 / 0.834 / 0.834` | band `ρ_conv ≤ 1e-6`; it trips on **UNDER**-convergence — **the 48-step read does not settle** (`δ = 2.95`, `corr(q₀,q*) = 0.99999982`) |
| 2 | settle→arg-min | ⚠ **INAPPLICABLE** | — | needs the arg-min assignment over ≥2 live keys on the same queries; the block's queries are not per-item probes |
| 3 | vacuous gate | ✅ clear | fire-rate **0.25** ∈ (0,1) | 16/64 offers refused (seed 0); 22/64 and 4/64 on seeds 1/2 |
| 4 | blank | ✅ clear | 0.167 vs chance 0.167, bar 0.623 | ⚠ **clear because the LIVE store is also at chance** — no leak, and no signal either |
| 5 | addressing | ⛔ **TRIPPED** 3/3 | `acq = 0.167 / 0.167 / 0.200` = **exactly chance**; `acq_strict = 0.0` | the store does not retrieve its own items |
| 6 | objective divergence | ⚠ **INAPPLICABLE** | — | needs ≥3–4 consolidation windows; 200 steps at `monitor_every 100` gives 3 observations through one persistent registry, still short |
| 7 | mass gauge | ⚠ **INAPPLICABLE** (n/a by design) | — | a `pytest` gauge, not a runtime trip |
| 8 | certificates | ⛔ **TRIPPED** 3/3 (**N2 + N4**) | `sep/σ_q = 3.349 / 3.262 / 3.312` (bar 5.15); `λ_min = +0.100` ✓; `N4 = 0.0062` | ⭐ **exactly the `full-clu-harness` S4 finding**: permitted basin interaction and the merge certificate are mutually exclusive by construction. The soft certificate is ON, so this is the *declared* price, not a surprise |
| 9 | lifetimes | ✅ clear | `Δ_ret = 7.8e-86` | ⚠ clear **because the wells have no depth to differ in** — see R2 |
| 10 | dead axis | ⚠ **INAPPLICABLE** | — | tier (b) semantic knob sweep not run in-block; tier (a) declared unimplemented (`knob_tier_a_implemented: false`), as `full-clu-harness` did |
| 11 | reach | ✅ clear | worst margin `+inf`, 0/6 unreachable | `a_U = ∞` — no saddle exists at these depths, so every excursion is "reachable"; ⚠ vacuous at `D ≈ 0` |
| 12 | starvation | ⛔ **TRIPPED** 3/3 | fairness **0.000**, oldest-retention-drop **1.000** | slot allocation is unequal and the oldest item has lost all its depth |
| 13 | maturity | ✅ clear (**demotes, never trips**) | `write_steps = 4`, floor 40 ⇒ **`promotable = 0`** | ⭐ **every reading in this report is formally NON-PROMOTABLE under N94.** Stated, not hidden |
| M14 | guard liveness | ⛔ **TRIPPED** 3/3 | **12 of 14 guards never fired** | only `admit.merge` (16/22/4) and `decay.permanent` (64) fire. Never fired: `admit.{priority,reach,budget}`, `place.{lambda_min,injective}`, `evict.{persistence,class_i,set_function}`, `route.signal`, `retry.budget`, `anneal.return`, `expand.monotone` |

**Acceptance verdict — honest form.** ⭐ **Nothing collapses *silently*: 10 of 14 monitors are
applicable, 5 trip, and every trip is named with its value.** But the criterion's *spirit* — a system
that runs the stream in a productive band — is **not** met: #5 says the store retrieves at chance and #12
says allocation is unfair, and neither could be cleared by its designed restoring verb inside the budget
this venue affords (§7). ⛔ **I therefore report S1 as ◐ PARTIAL, not ✅.**

---

## 5. ⛔ THE FIRST-ORDER FINDING: the store is INERT inside the block, and the write budget is not why

Three independent measurements, all 3 seeds:

1. **Live == blank.** Held-out NLL with the real write stream vs a store never written:
   `Δ = 0.00e+00 / −5.96e-08 / 0.00e+00`. The stored content changes the model's output by **at most
   float32 round-off**.
2. **Self-probe = chance.** Re-reading each live item at its own recorded address and decoding to the
   nearest stored payload: `acq = 0.167 / 0.167 / 0.200` against chances of `0.167 / 0.167 / 0.200`.
   Strict acquisition **0.000**. The blank store scores **identically**.
3. **The anytime curve is flat** (§9): 6 → 96 Verlet steps per read moves bpc by ≤ 7.5e-4.

### 5.1 The mechanism, isolated (`write_budget_sweep.json`)

Sweeping the one lever chunk granularity re-budgeted, **at fixed everything else**:

| inner write steps | 4 | 16 | **64** |
|---|---|---|---|
| self-probe acquisition | 0.250 | 0.250 | 0.250 |
| chance | 0.250 | 0.250 | 0.250 |
| median fitted well depth `D` | 0.0331 | 0.0448 | **0.0442** |
| wall (one 4-chunk pass) | 14 s | 28 s | 109 s |

⭐ **16× the write budget buys 1.33× the depth and ZERO acquisition, and depth saturates by 16 steps.**
Against the shipped store's fitted `D = 0.46–0.80` (`full-clu-harness` §3.4) these wells are **10–18×
too shallow**, and more steps do not fix it. ⛔ **So "the write is starved of steps" is REFUTED as the
explanation.**

### 5.2 The read budget, isolated (`shipped_budget_control.json`)

Same block, same φ/ψ/controller; only the budgets move to the shipped C2W1 values:

| configuration | write steps | Verlet/read | **`ρ_conv`** | residual | depth | acq / chance |
|---|---|---|---|---|---|---|
| affordable (chunk-granular) | 4 | 48 | **0.2079** | 6.35e-02 | 0.064 | 0.50 / 0.50 |
| read at shipped budget | 4 | **1200** | **5.06e-07** | 1.48e-07 | 0.064 | 0.50 / 0.50 |
| write at shipped budget | **300** | 48 | 0.7984 | 4.11e-01 | 0.122 | 0.50 / 0.50 |
| **both at shipped C2W1 budget** | **300** | **1200** | **2.60e-07** | 1.25e-07 | 0.122 | 0.50 / 0.50 |

⭐ **The read-budget leg reproduces `full-clu-harness` exactly**: at 1200 steps `ρ_conv = 2.6–5.1e-07`
against their `4.3e-07`. **The two-phase relaxation is fine — it simply is not run to convergence at
chunk granularity** (→ R3). ⚠ **The acquisition leg of this control is UNDER-POWERED and I will not
claim it**: `batch = 1, seq_len = 128` admits only 2 items, so chance is 0.50 and 0.50 is
uninformative. **Declared NOT-CONCLUSIVE, not a null.**

### 5.3 What I believe is happening, labelled as a hypothesis

Three facts constrain it: depth saturates at ~0.045 regardless of steps; `sep/σ_q = 3.3` (below the 5.15
bar, because the soft certificate deliberately shrank `d_safe` from 0.986 to 0.389 so basins may
interact); and 128 atoms per group are initialised **scattered at scale 1.0** (`atom_local_radius = 0.0`,
the historical default) in a `dim = 3` ball. **Hypothesis (untested): a few unrolled steps cannot gather
128 scattered atoms into a well at the target, so the group's depth *at the site* stays ~0 no matter how
many steps are spent — the binding constraint is atom *placement at init*, not optimisation.** The
shipped store escapes this with 300 Adam steps on a **static, non-streaming** objective.
**Three candidate fixes, none built** (§8.2 of the intervention doc forbids me moving toward a
degenerate configuration to obtain a clean number, and all three are design changes, not knob turns):
`atom_local_radius > 0` (the N98 localized init, a shipped lever at its designed band) · a
payload-carrying `ψ` residual · a write objective with a *trajectory* term (`trainability-spike`'s own
recommended next experiment, still unbuilt).

---

## 6. What ran correctly, and the three defects I found by running it

### 6.1 ⛔ Eviction bookkeeping (found, fixed, results re-run)

`_controller_plan_for_lane` sliced `ctrl.log` by a **count** of past evictions and looked the victim up
by item id — but the allocator has already deleted the evicted record. Consequences: **133 evictions
reported from 64 offers**, and **not a single `reset` ever marked**, so evicted atom groups were never
re-drawn. Now detected as a set difference of live slots across the offer (plus the evict-then-reuse
case). Post-fix: 24 / 19 / 37 evictions from 64 offers. ⚠ **The whole 3-seed table was re-run after the
fix; the bpc figures were unchanged to 4 d.p., which is itself evidence for §5's inertness.**

### 6.2 ⛔ The inner write was arithmetically inert (found, fixed, and it is load-bearing)

With plain SGD at `atom_depth_init = 1e-4` the write objective's gradients are ~1e-4, the atoms moved
~1e-6/step, and **the read output moved by 7.5e-9 — below float32 resolution.** The memory was inert *by
arithmetic* and would have been reported as a null. The shipped store digs its wells with **300 Adam
steps at 3e-3**, and Adam moves a parameter by ~`lr` per step *regardless of gradient magnitude*.
**Sign-SGD reproduces that behaviour statelessly**; a real Adam would add two moment tensors per sequence
(2× the store's state bytes) and break the D2 ledger. Post-fix the read moves 3.5e-4 over 8 writes — a
**10⁴** improvement. ⚠ **This is why §3's settled-point number is exactly 0** and why §3 point 2 exists.

### 6.3 The declared anti-collapse initialisation of φ

A `tanh` MLP on LayerNormed pooled chunk summaries emits addresses with RMS ≈ 0.125 at `d_model = 64` —
far inside the merge radius — so **at gain 1.0 the store starts life refusing 60/64 offers: the
degenerate corner, reached at step 0.** The rule applied is principled and scale-free, declared in
PREREG, and identical in every arm because it depends only on the shared shell: **set the gain so the RMS
address norm equals the store's `ball_radius`.** Measured gains **3.839 / 6.356 / 5.650**; refusals fall
to 16–22/64. ⚠ Even so, the *pairwise* separation of chunk latents at init is only ~0.04 against
`d_safe = 0.389` — **enwik8 chunk summaries are not naturally `d_safe`-separated in the store's address
ball**, and that is a reportable property of the venue.

### 6.4 A crash in a read-only file's helper (R4)

`saddle_reach_threshold(D, s, α, ‖c‖)` computes `β = D/(2αs²)`; a group whose wells were never dug has
`D ~ 1e-86` and a depth-weighted width that underflows, so `2αs² == 0.0` → `ZeroDivisionError`. It
**killed seed 2 of the shipped run**. Guarded in my caller (`D > 1e-12 and s > 1e-6`; an item with no
well is *absent*, not *unreachable*). ⛔ The guard belongs in `monitors.py`, which is read-only to me.

---

## 7. PREREG SCORECARD (`.claude/outputs/cluformer-pilot/PREREG.md`, written before any run)

| # | prediction | outcome |
|---|---|---|
| **DF1** | CLU beats both matched swaps by ≥0.02 bpc at 28.3 M | ⛔ **NOT SCORED at pilot (S4 not run).** At toy: **FAILS**, in the predicted direction |
| **§3** | `CLU − GRU = +0.12` bpc, range [+0.02, +0.45] | ◐ **sign correct, magnitude far over-predicted**: **+0.0106 ± 0.0057** at toy. *(A 0.16 M model trained 200 steps is not the object the prediction was derived for; scored as directionally confirmed, quantitatively NOT scored.)* |
| §3 | `CLU − TTT = +0.08`, range [−0.05, +0.35] | ◐ same: **+0.0157 ± 0.0122** (in range) |
| **§3** | `CLU − memory-deleted = −0.06` (the CLU must beat *no memory*) | ⛔ **REFUTED, and this is the report's most important refutation: +0.0070 ± 0.0055 — the memory is a net COST on 3/3 seeds.** By my own PREREG §9 this means **S1 has failed in-system** |
| §3 | `CLU − blank-store = −0.05` | ⛔ **REFUTED — 0.0000, bit-identical** |
| **§4** | dyn-eval improves arms by 0.05–0.15 bpc; `Δ_dyneval(CLU−GRU) = +0.19` ⇒ primary dead | ⛔ **REFUTED on magnitude** (uniform −0.0019 for every arm) and **MOOT on conclusion** (no advantage existed to erase). ⚠ 4 test batches on an under-trained model is a weak dyn-eval; that weakness is *conservative* — it cannot have hidden a CLU advantage |
| **§5** | GRU cannot match both axes; state-matched GRU ≈ 1 525× a 26 M model | ✅ **CONFIRMED by construction: 100.9× state deficit at matched params; 79 469 668 params at matched state (3.06× a 26 M model at toy geometry)** |
| §5 | TTT matches both within 2 % at pilot | ✅ **+0.046 % params / −0.097 % state at toy** — better than the pre-registered toy tolerance (2.2 %/3.3 %) |
| §5.3 | byte law `214.00×` at toy / `299.33×` at pilot; 1 714 rows vs 8 items | ✅ **exact** |
| §6 | the 48-step read does **not** settle (`ρ^64 = 0.195`) | ✅ **CONFIRMED: `ρ_conv = 0.834–0.836`** |
| §6 | **monitor #1 does NOT trip** ("#1 trips on over-convergence") | ⛔ **REFUTED, and my reasoning was wrong**: #1's predicate is `ρ_conv > 1e-6`, i.e. it trips on **under**-convergence. It trips on 3/3 seeds |
| P1 | `‖∂L/∂φ‖` ratio ≥ 1e3, point estimate 1e4 | ✅ **∞ (sign-SGD write) / 1.5e3–2.0e9 (plain-SGD write)** — inside the registered range |
| P2 | monitor #13 (maturity) **TRIPS** every chunk | ◐ **wrong mechanism, right consequence**: #13 does not *trip*, it **demotes** (`promotable = 0`). Every reading here is formally non-promotable under N94 |
| P3 | monitor #9 (lifetimes) trips | ⛔ **REFUTED — clear**, but vacuously: `Δ_ret = 7.8e-86` because the wells have no depth to differ in |
| **P4** | monitor #8 (certificates) trips once SC is on | ✅ **CONFIRMED, 3/3, on N2 + N4** (`sep/σ_q = 3.26–3.35` vs the 5.15 bar) |
| P5 | monitor #10 does **not** trip on `traj_stride` (my ψ consumes the buffer) | **NOT SCORED — #10 is INAPPLICABLE** (no in-block knob sweep run) |
| P6 | monitor #4 (blank) does not trip | ✅ **CONFIRMED** — but vacuously (live is also at chance) |
| P7 | monitor #12 (starvation) trips | ✅ **CONFIRMED, 3/3** (fairness 0.000) |
| **P8** | γ moves ≥3× further than M in relative terms (the 14× channel) | ⛔ **REFUTED: `|Δγ_read|/γ₀ = 0.0155 / 0.0051 / 0.0095` vs `|ΔM|/M₀ = 0.0093 / 0.0098 / 0.0061`** — ratio **1.67 / 0.52 / 1.56**, i.e. comparable, and M wins on seed 1. Gradient norms agree (`‖∂L/∂M‖ > ‖∂L/∂γ‖` on 2/3 seeds) |
| P9 | `d/s ≥ 3` using `bprime-c6`'s ruler (`s = 0.40`, achieved separation) | ◐ **borderline**: achieved `sep = 0.4893–0.5024`; against `bprime-c6`'s **`s = 0.40`** that is **`d/s = 1.22–1.26`**, well **below** 3 and far below their measured 4.3. ⚠ **Ruler named as required**; the pilot's store is NOT in the designed-gate regime |
| P10 | `sep/σ_q` falls below 5.15 with SC on | ✅ **CONFIRMED: 3.26–3.35** |
| §7 | S4 will be a declared NOT-RUN-BY-AGENT; DF1/2/3 NOT SCORED | ✅ **CONFIRMED (the CSF3 DNS failure was pre-registered)** |

**Score: 9 confirmed · 4 partial · 7 refuted · 2 not scored.** Four of the seven refutations
(memory-deleted, blank, dyn-eval magnitude, monitor #1's direction) are the substance of §5 — *a
pre-registered prediction that fails is a finding.*

---

## 8. Compute: the §2.2 constraint, faced

### 8.1 Chunk granularity and the per-read Verlet budget (both arms)

- **Chunk `C = 32` (toy) / 64 (pilot)** — one read + one write per chunk, **identical in every arm**.
- **Per-read budget 24 + 24 = 48 Verlet steps, +24 for the always-computed gated retry** (a
  data-dependent step count is not a static shape, so the retry round is computed and *then* gated; the
  compute is charged, the benefit accrues only where the gate fires).
- **Sequence-scale arithmetic:** 512/32 = 16 boundaries × 2 layers × 48 steps = **1 536 Verlet
  steps/sequence** vs 512 × 1200 × 2 = **1 228 800** for a naïve per-token settle — an **800×**
  reduction, which is exactly what chunk granularity buys. (PILOT: 24 576 vs 14.7 M, **600×**.)
- ⚠ **The 17.1× trajectory-vs-point price (`trainability-spike` §3) travels with every trajectory number
  here and is not re-measured** (§3's wall-clock ratio is a gradient control, not a cost comparison).

### 8.2 Wall clock, and where it actually goes

| arm | total (s, seed 0/1/2) | train (s) | **plan-pass fraction** |
|---|---|---|---|
| CLU | 1332 / 1299 / 1018 | 987 / 858 / 651 | **0.772 / 0.827 / 0.839** |
| GRU (matched params) | 83 / 93 / 94 | 68 | 0.919 |
| TTT (matched both) | 80 / 82 / 82 | 67 | 0.888 |

⭐ **The CLU arm is 14–16× the GRU's wall clock end-to-end, but 77–84 % of its step is the Python
controller, not physics.** Netting the plan pass out: CLU ≈ 225 s vs GRU ≈ 6 s of actual compute per 200
steps — **~37× on physics alone.** ⚠ The plan pass runs for *every* arm (the swap controls get the same
concrete pass so the decision and data order are identical), so it is fairness-neutral but it dominates,
and at the PILOT scale (96 lanes × 16 chunks) it is ~2 s/step of pure Python before a single Verlet step.
**This is a real scalability finding and it is the first thing to optimise if tier iii continues.**

---

## 9. D5 — the anytime shape curve (secondary; ⚠ **SHAPE claim only**, §A3)

bpc vs Verlet-steps-per-read, evaluated on the trained CLU arm (the model was trained at 48):

| Verlet/read | 6 | 12 | 24 | **48 (trained)** | 96 |
|---|---|---|---|---|---|
| seed 0 | 4.599653 | 4.599630 | 4.599570 | 4.599633 | 4.600315 |
| seed 1 | 4.647052 | 4.647044 | 4.647015 | 4.646973 | **4.646892** |
| seed 2 | 4.610947 | 4.610921 | 4.610815 | 4.610764 | 4.611014 |

⛔ **There is no shape to claim. Total variation across a 16× compute range is ≤ 7.5e-4 bpc**, an order
below the seed s.e. (0.014). Consistent with §5: a memory that carries nothing cannot be read better by
reading it longer. ⚠ The anytime figure is **occupied** (DEQs, EBTs, Titans-Revisited); no uniqueness is
claimed here and none could be.

---

## 10. Budgets, declared vs spent

| | budget (PREREG §7) | spent | verdict |
|---|---|---|---|
| local wall clock | **≤ 8 h** | **≈ 13 h** (of which 1.35 h is the reported 3-seed run) | ⛔ **OVERRUN, declared.** Cause: two correctness defects found late (§6.1, §6.2) each forced a full re-run, plus one crash (§6.4). I judged that reporting an inert memory caused by my own optimiser (§6.2) would have been a false first-order finding, and re-running was the honest call |
| local worktrees | ≤ 1 | 1 (`../CHLU-pilot`) | ✅ |
| 26–47 M locally | forbidden | none | ✅ |
| **CSF3** | ≤ 108 A100-hours | **0** | ⛔ **unreachable — see the first screen** |
| cut order if exceeded | D5 → TTT at pilot → layers | **nothing was cut**; the overrun was absorbed in wall clock | — |

---

## 11. ⛔ DECLARED NOT-RUNs (never reported as nulls)

- **S4 — the 26–47 M CSF3 run: NOT RUN.** No cluster route from this machine (DNS/VPN). `--scale pilot`
  and `scripts/csf3/job_gpu_cluformer.sh` are ready; the Head can submit. **DF1, DF2, DF3 are NOT SCORED
  at pilot scale.** ⚠ Every §2 number is TOY scale.
- **"Needs > 500 M": NOT AVAILABLE and not written.** PREREG §2 made it conditional on DF2 passing, and
  DF2 needs the pilot leg.
- **WikiText-103: loader built, NOT RUN.** enwik8 is the declared venue.
- **Monitors #2, #6, #10 (tier a and b) and #7: INAPPLICABLE, not passing.**
- **The §5.2 acquisition control: UNDER-POWERED (2 live items, chance 0.50), declared NOT-CONCLUSIVE.**
- **Multi-layer / deeper stacks, larger `capacity`, `atom_local_radius > 0`, a trajectory write term:
  NOT RUN.** All are §5.3 hypotheses.
- **The `overload`/gym families: not used as a claim venue** (§A14.8), and not used at all here.

---

## 12. Git footprint

- **Branch** `agent/experiment-engineer/cluformer-pilot`, **worktree `../CHLU-pilot`**, base local
  `main @ 21a6dc4` (**did not move under me; `git rebase main` is a no-op**). **Not pushed, not merged.**
- **Commits (7):**
  `477a61b` enwik8 + WikiText-103 loaders ·
  `aebfc70` the tier-iii block + matched swap cells ·
  `fae6c29` trainer, runner, CSF3 job script, tests ·
  `f173db6` module-scoped float32 fixture (x64 ordering hazard) ·
  `a4e1848` eviction-bookkeeping fix + real monitor context + φ-gradient decomposition
  (⚠ this commit also swept in the plotting panels and the D5 runtime Verlet-budget override — mine,
  but not named in its subject) ·
  `7bc166a` the zero-depth reach-threshold guard (R4) ·
  `7f1b9f1` plot panel (d): `inapplicable` rendered as a THIRD state, not as "clear".
- **Files touched — all within my declared ownership, zero read-only violations** (checked file by file):
  `chlu/core/blocks.py` (additive) · `chlu/data/enwik8.py` (new) · `chlu/data/wikitext.py` (new) ·
  `chlu/training/train_cluformer.py` (new) · `chlu/experiments/exp_cluformer_pilot.py` (new) ·
  `scripts/csf3/job_gpu_cluformer.sh` (new) · `tests/test_blocks.py`, `tests/test_cluformer_pilot.py`,
  `tests/test_data_enwik8.py` (new).
- **NOT touched:** `chlu/config.py` · `chlu/core/{clu_system,admission,placement,memory_potentials,
  integrators,controller,clu_controller,monitors,soft_certificate,psi_readout,implicit_grad}.py` ·
  `chlu/eval/**` · `chlu/cli/experiment_cmd.py` · `chlu/experiments/memory_gym.py`.
  ⭐ **`chlu/core/psi_readout.py` was in my ownership list and I did not need to edit it** — `PsiSpec`
  + `DeepSetsPsi` were sufficient, and **`AttentionPsi` stays quarantined and unused**.
- **Config lives in my own modules** (`StreamMemoryConfig` in `blocks.py`, `PilotConfig` in
  `train_cluformer.py`), per the standing `chlu/config.py` read-only rule.
- **Tests:** `pytest -q` full suite — see §13. `ruff check chlu/ tests/ scripts/` clean at every commit.
- **Anonymisation grep** (`pratik|@postgrad`) over every file I added: **CLEAN**.
- ✅ **Branch ref verified FROM THE MAIN REPO** (protocol §3.2, the w4 lesson):
  `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/cluformer-pilot`
  → **all commits present** (verified at `7bc166a`; `7f1b9f1` added after and confirmed on the branch).
  ⚠ **Worktree deliberately NOT removed** — left in place for Hub review; remove with
  `git worktree remove ../CHLU-pilot` afterwards.
- ⚠ **One near-miss to log:** a `git add -A` intended for the worktree ran in the **main repo** after a
  `cd` into `.claude/outputs` moved the session cwd. **No damage** — `.claude/**` is gitignored, git
  reported `nothing to commit, working tree clean`, and `main` is untouched. Recorded because §3.2's
  "never `git add` in the shared checkout" precedent cost another agent ~90 foreign lines.

## 13. Test suite

- **Before my fixture fix:** `10 failed, 1172 passed` — **all 10 mine**, and all passed in isolation.
  Cause: the handover §7.2 x64 hazard *with a twist* — several modules enable `jax_enable_x64` at
  **module import**, and a *function*-scoped float32 fixture is set up **after** module-scoped ones, so
  my store cell was constructed in float64 and exercised in float32. Reproduced the exact ordering
  (`pytest tests/test_lattice.py tests/test_blocks.py tests/test_cluformer_pilot.py`): **10 failed
  before, 60 passed after** with an autouse **module-scoped** fixture.
- ✅ **Final full suite on the branch: `1182 passed, 24 warnings in 1025.40s (17:05)` — zero failures**
  (main venv, JAX 0.9.0, worktree). The pre-branch suite was **1 136**, I added **46**, and
  `1 136 + 46 = 1 182` exactly: **nothing pre-existing broke.**
- **New tests: 46** (`test_blocks.py` 22 · `test_cluformer_pilot.py` 16 · `test_data_enwik8.py` 8).
  They encode the *claims*: ruling 0.1's lever set, the bit-identical shell (and that
  the check can fail), the GRU two-sided impossibility, the TTT two-sided match, C3-write locality, a
  refused offer leaving `V_θ` bit-identical, eviction **re-drawing** (never zeroing), the payload-zero
  launch, chunk causality, the blank plan never writing, `∂q*/∂q₀ = 0`, and γ/M being trajectory-only
  selectors.

---

## 14. Open questions / follow-ups / risks

1. ⭐ **The decisive next experiment is NOT more scale.** §5.1 shows the store is inert at 16× the write
   budget and §5.2 shows the read is fine at the shipped budget. Running 28.3 M on CSF3 *before* §5.3's
   placement hypothesis is tested would spend ~108 A100-hours to reproduce a mechanism already isolated
   at 0.16 M for 1.35 CPU-hours. **My recommendation to the Hub: gate S4 on a cheap `atom_local_radius`
   / trajectory-write-term probe first.** That is a Hub routing decision, not mine.
2. **Toy scale is a real limitation, stated plainly.** 200 steps × 2 048 tokens = 409 600 tokens; every
   arm sits at 4.60–4.63 bpc, i.e. barely past unigram statistics. The arms are separated by 0.01–0.02
   bpc against a seed s.e. of 0.014–0.021. **Only the *paired* per-seed margins (§DF1 table) are
   defensible, and only because their signs replicate 3/3.**
3. **The plan pass dominates the clock** (77–84 %). Before any pilot run, the controller loop needs
   vectorising or the GPU will idle behind Python.
4. **Monitor #6 never became applicable.** Three observations through one persistent registry is short of
   its window; a longer run or a smaller `monitor_every` fixes it cheaply.
5. **`params/state = 1.675` at toy vs a predicted 1.02 at pilot geometry.** ψ is a larger fraction of the
   cell at toy. The TTT two-sided match gets *easier* at pilot, so R1's conclusion strengthens with
   scale; the GRU's impossibility strengthens too (100.9× → 566×).
6. **Risk:** the whole build rests on decision replay (§1.1). If a reviewer rejects "the controller's
   decisions are detached", the fallback is a differentiable relaxation of admission — which would be a
   *different* (and weaker) claim about the C2W1 controller. Worth a Hub ruling before the pilot run.

---

## Proposed handover updates (for the Hub)

**§2/§3 (architecture + config).**
- New modules: `chlu/data/{enwik8,wikitext}.py`, `chlu/training/train_cluformer.py`,
  `chlu/experiments/exp_cluformer_pilot.py`, `scripts/csf3/job_gpu_cluformer.sh`; `chlu/core/blocks.py`
  gains `StreamMemoryConfig`, `WritePlan`, `CluStoreCell`, `MatchedGRUCell`, `MatchedTTTCell`,
  `NullMemoryCell`, `EchoMemoryCell`, `StreamBlock`, `StreamModel`, `solve_matched_{gru,ttt}`,
  `swap_ledger`, `store_byte_law`.
- ⚠ **No CLI hook** (`experiment_cmd.py` read-only this wave) — run by module invocation. A hook is owed.
- **A THIRD training path exists.** `train.py` = dynamics Wake–Sleep (A/B); `train_generative.py` = EBM
  PCD (C); **`train_cluformer.py` = byte-LM cross-entropy with the store as a layer.** Never conflate.
- Config for the pilot lives in `PilotConfig`/`StreamMemoryConfig`, **not** `chlu/config.py`.

**§7 (known issues / live) — add:**
- ⚠ **`saddle_reach_threshold` divides by zero when `D → 0`** (`2αs² == 0.0`). Crashed a run. Guard owed
  in `chlu/core/monitors.py`. *(R4)*
- ⚠ **The x64 ordering hazard bites MODULE-SCOPED fixtures too.** A function-scoped float32 fixture runs
  *after* module-scoped ones; any new test module that builds JAX objects in a module-scoped fixture must
  make its float32 fixture `scope="module", autouse=True`. Cost me 10 red tests.
- ⚠ **A learned store written by plain SGD from `atom_depth_init = 1e-4` is arithmetically inert**
  (read moves 7.5e-9, below float32). Any streaming/inner-loop write must use an Adam-like *step-size*
  rule (sign-SGD is the stateless one). *(§6.2)*
- ⚠ **`ρ_conv` bands are budget-scoped, not just harness-scoped**: 2.6e-07 at 1200 steps, **0.834 at
  48**. *(R3)*

**§8/§10 (record) — add:**
- ⭐ **Tier iii, first datum: the full C2W1 store runs inside a streaming block on enwik8, trains
  end-to-end through the trajectory read (`‖∂L/∂φ‖` ratio ∞ / 1.5e3–2.0e9), and its stored content
  changes the model's output by 0.00e+00 bpc.** The memory is a **net cost** vs a memory-deleted control
  (+0.0070 ± 0.0055 bpc, 3/3 seeds) and loses to a matched-params GRU (+0.0106 ± 0.0057, 3/3).
- ⭐ **T3 confirmed in-system and strengthened:** the settled-point read's γ- and M-gradients are `0.0`
  **bitwise** on 3/3 seeds inside a language model.
- ⛔ **FB2's shape at tier iii (R1):** matched params and matched state-bytes cannot both be hit for a
  GRU cell (79.5 M params at matched state, toy geometry; 1 525× a 26 M model at pilot geometry). **A
  TTT-class cell can (0.05 % / 0.10 %) and is the two-sided control.** Both columns published.
- ⛔ **`bprime-c6`'s ruler applied (P9): the pilot store sits at `d/s = 1.22–1.26`, NOT in the
  designed-gate regime** (`bprime-c6` measured 4.3 on the shipped rig).
- ⚠ **Monitor #13 demotes rather than trips**: at 4 inner write steps every reading in this report is
  **formally non-promotable under N94**. Any downstream quotation must carry that.
