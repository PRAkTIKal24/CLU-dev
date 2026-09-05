# c2w7-read-cardinality — experiment-engineer report

**Task + acceptance criterion (one line):** build the §A21 C2W7 cardinality iteration —
multiplicity-as-counting-code, overlap-as-importance weighting, the batch-level anti-collapse
regularizer (built OFF), the launch-collapse monitor, live A1-strong launders and the reader-class
re-registration — adjudicate the gate mechanically and run the ORGANIZER SWAP iff it fires.
**Status: done** (all six deliverables built and measured; **the gate does not fire, so NO swap was
run**, per the task file; the negatives are reported as negatives).

> ## ⛔ THE ONE-LINE VERDICT
> **The read's EXPRESSIVITY is repaired and the expressivity is entirely LAUNCH-SIDE.**
> R1 goes `0.0023 → 0.0656 ± 0.0239` (bar 0.02) — ⭐ **28×, and it CLEARS for the first time in the
> programme** — the read now commits to a cardinality (`F_hat = 4.54 ± 0.12`, gated set exactly `F`
> on 39.4 % of queries where iteration 1 had 5.79 ± 0.90 members and never 4). ⛔ **And guard 1
> fires harder than it did in iteration 1: `read − live launder = −0.0070 ± 0.0016`, now
> significantly NEGATIVE on 5/5 seeds — the settle makes the answer WORSE than the head's own
> launches, and the mechanism is measured: the settle scatters a committed allocation of 4.83
> distinct wells into 5.67, because the store's attractors are not at its designed anchors.**
> ⛔ **The gate fails 2 of 4 (`G1`, and `S_eff = 16.77 ± 0.85` marginally above its band ⇒ labelled
> COLLAPSED by the mechanical rule) ⇒ NO ORGANIZER SWAP was run.**

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. ⭐⭐ **Every arm this programme has scored at ≈ 0 may have been scored through a reader that
   destroys its own signal.** The class is fitted by **least squares** while the metric is a
   **thresholded** exact-set accuracy. Measured here, 5 seeds, unseen, on the *same* latent:
   `count_table` (72 params, lstsq) **0.0000 ± 0.0000** vs `count_identity` (**0** params)
   **0.0539 ± 0.0207**; a 2-parameter gain+bias reader is shrunk just as hard (`a ≈ 0.5`, 0.000).
   ⛔ **This re-scopes `tierii-read-fix` §13.3 ("is the cap what zeroes every arm?"): it is not the
   capacity cap, it is the fitting criterion.** Filed as `AMENDMENT-C2W7`. Someone must decide
   whether the *shipped* reader class needs a zero-parameter member everywhere.
2. **`k` is NOT a capacity dial once the head commits to a cardinality.** Guard 4 measured
   `k = 12 → 24` at 2× read flops: best-reader score **0.0430 → 0.0430** and `coverage_raw`
   **0.2344 → 0.2344** — identical to 4 decimal places (iteration 1 measured +0.2383 coverage).
   The ledger check still raises on a mismatch, but **"doubling `k` raises the score" is refuted at
   a multiplicity head** and any inherited wording must be scoped.
3. **Guard 2's statistic needed a second detach and the reason is a finding.** At a multiplicity
   head the counting-code channel has a *second* differentiable path — `F_hat` multiplies `m` — that
   survives an `argmax` assignment: with it live the hard/soft gradient ratio is **0.9988**, i.e.
   **the head still trains under `argmax`, through cardinality**. Guard 2's registered question
   (does the *assignment* backprop?) needs `detach_f`; both numbers are reported below.
4. **The `S_eff` band's upper edge is being read as a collapse and it is the opposite failure.**
   `S_eff = K·F/#wells-ever-occupied`, so `S_eff > 16` means **fewer than all 32 wells were
   visited** (measured 30.6 of 32; per seed 30/32/28/31/32). The mechanical rule labels the run
   COLLAPSED and I report it as such — but the physical reading is *slight under-usage*, not the
   34–51 concentration `orgdiv-cat-test` §5.2 measured. The band needs a two-sided label.
5. **`K0`'s bar and any `F`-commitment are in DIRECT TENSION by construction** (§1). K0 asks for
   `≥ F` distinct wells *reachable*; a commitment to exactly `F_hat` fails it whenever `F_hat < F`.
   Buying `K0 ≥ 0.90` cost **0.044 of launch-code exact-set** (0.143 → 0.099, SEEN). ⛔ **K0 can no
   longer be quoted as "the read is healthy"** — a *collapsed* head (F_hat saturated at `f_max`)
   scores `K0 = 1.000` while reading 0.0000 (measured, §7).
6. **The out-of-class ceiling at this cell is `0.691` (`d = 8`, ONE draw, `σ_q = 0.15`)** —
   recomputed live, quoted only with its `(d, draws)` noise model, never as a bar (reconciliation 1
   of iteration 1, honoured).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial:** TIER ii — the organization dividend, **read-cardinality iteration (iteration 2)**.
- **Control:** the ORGANIZER SWAP, **gated** — ⛔ the gate did not fire, so the swap was **NOT
  RUN**. The live controls that did run are the four §A20.3(c) guards, above all **G1 recomputed
  LIVE on this cell's OWN learned launches** (amendment A1, strong form).
- **Falsifies:** the re-registered falsifiers of `PREREG.md` §3 (K0, R1, R2, G1–G4, M15, `S_eff`),
  each with sign/threshold/seeds filed **before** the harness ran.
- **Does NOT falsify:** the nulls' failure (not my claim); anything tier i/iii; ψ's marginal value
  (A2 — this vehicle contains no ψ at all, verified by grep, §2).

---

# 1. ⭐ K0 — THE PRE-CONDITION, ADJUDICATED FIRST (store-free, before the build)

**Bar 0.90.** Designed init, 5 seeds, 1 280 unseen queries (`k0` stage), and **re-checked on the
TRAINED head** inside `arms` (a trained head can collapse K0 — that is what monitor #15 watches).

| head | `F_hat` | **K0 = P(≥F distinct)** | precision | launch top-`F` exact-set |
|---|---|---|---|---|
| iteration 1's designed head (`k = 12`, no commitment) | — | 1.0000 | 0.274 | — |
| ⭐ **the multiplicity head, designed init (REGISTERED)** | **4.70** | ⭐ **0.9195 → PASS** | **0.549** | **0.1898** |
| the same, **after training** (5 seeds) | 4.42 | **0.9070 ± 0.0125 → PASS** | 0.460 | 0.103 |
| ⛔ the same at `head_lr = 3e-2` (iteration 1's lr; a collapsed head) | **8.01** | **1.0000** | **0.124** (= chance 4/32) | **0.000** |

⭐ **The last row is the reason K0 must never be quoted alone** (reconciliation 5): a head whose
cardinality estimator has saturated at `f_max` passes K0 *perfectly* and reads **0.0000** on every
reader, with the launch-collapse monitor tripping at `S_marg = 6.9 < 16`. K0 is a **reachability**
statement, nothing more.

⛔ **The registered cell pays for K0.** On SEEN (3 seeds, store-free): `card_b = 0.0` → K0 0.789 /
exact 0.143 · `0.3` → 0.880 / 0.135 · **`0.7` (registered) → 0.953 / 0.099** · `1.1` → 0.979 /
0.063. The K0 bar costs **0.044 of exact-set**, and the SEEN→unseen gap (`card_b = 0.5` gave 0.912
SEEN but **0.871** unseen) is why the registered value carries margin.

**The decoder study that fixed the head** (SEEN, 3 seeds, store-free — deviation D12):
top-`F` of raw overlaps **0.023** · OMP/greedy **0.008** · ⭐ **non-negative ISTA (200 unrolled
steps) 0.19** · exhaustive `C(32,4)` **0.72** (⛔ out-of-class). Greedy pursuit is *provably* outside
its own guarantee here (coherence `1/√d = 0.35` vs the required `1/(2F−1) = 0.14`).

---

# 2. Flag provenance (mandatory — every quantitative result in this report)

Commits `5a952f0` · `92aea1b` · `3c19878` · `baf361f` on branch **`c2w7-read-cardinality`**, base
local `main @ 104ca19`, worktree `../CHLU-c2w7`, **main venv reused** (protocol §4, w6 lesson):
**jax 0.9.0, equinox 0.13.4, float32**. **Seeds 0–4** on every claim number.

| flag | value | note |
|---|---|---|
| `n_wells / f_subset / n_items / n_unseen` | 32 / 4 / 128 / **256** | D15 (compute); 1 280 scored queries/arm, grain 7.8e-4 |
| `atoms_per_well a` / `payload_dim m` / `addr_dim d` | 32 / 8 / **8** | measured constraints (K1/K2b) + D6 |
| `payload_radius` / `atom_payload_init_radius` | 0.5 / 0.5 | D10 (basin reach, §7 known issue) |
| `s_measured` → measured `d/s` | 0.2879 → **2.700 / 2.784 / 2.786 / 2.886 / 2.747** | ✅ 5/5 inside the soft-certificate band [2.5, 2.9] |
| `depth_ratio` | **3.0** | claim cell keeps the registered heterogeneity; `1.0` is a diagnostic axis only |
| **`k_particles`** | **12** | ⭐ ledgered on every arm (guard 4); `k ∈ {16, 24}` declared NOT-RUN |
| head: `ista_steps / ista_lam / ista_eta` | **200 / 0.05 / 0.30** | D12; `lax.scan`, learned λ/η |
| head: `card_a / card_b / commit_eps / rank_sigma` | **0.85 / 0.7** / 0.005 / 0.35 | designed init, tuned on **SEEN only** (§1) |
| head: `conf_b / conf_w` | **0.10 / 0.15** | D14 (a hard gate deletes a well from the count) |
| head: `rho / p0_gain / learned_p0` | 0.25 / 1.0 / True | learned `p₀` = **reach lever only** (§A14.1) |
| `occ_tau / payload_ref / weight_mode / count_agg` | 0.25 / 0.5 / **both** / **sum** | all four weight modes + `noisy_or` reported (one read pass) |
| **`lambda_anticollapse`** | ⛔ **0.0 (OFF)** | doctrine §3.3; the ON arm (λ = 1.0) is a labelled second state, §6 |
| query noise | `σ_q = 0.15`, **ONE draw per query** | D7 — strictly less launch information than the refuted protocol |
| read budget | **400 + 800** Verlet steps, `dt = 0.05` | every γ statement budget-scoped |
| `gamma_address / gamma_read` | 0.05 / 0.02 | one claim cell (the γ axis is a declared NOT-RUN) |
| `kinetic_mode` | `newtonian_learned`, per-particle `mass_override` | shipped Prop-6 per-address mass |
| head training | **60 steps @ lr 3e-3**, batch 16, settle **60 + 120** | ⚠ read-budget-scoped; lr tuned on SEEN (3e-2 destroys the head, §7) |
| organizer (physics) | **60 Adam steps @ 3e-3, through the settle**, settle 60 + 120 | loss 1.78–2.92 → 1.53–2.35 |
| `tol` | `0.25 × RMS‖y − ȳ‖` = **0.2338** | **chance = 0.0000** (constant predictor, `m = 8`) |
| ψ | ⛔ **none in this vehicle** (grep: 0 hits for `psi` in `factored_store` / `multiwell_read` / `multiplicity_read` / `null_arms`) | A2 honoured; no hot-swap possible |
| bytes | store **73 728 B** · head **568 B** (142 params) · φ **1 152 B** · **total 75 448 B**, identical on every arm | `read_flops/query` **2.507e8** |

---

# 3. ⚖ THE GATE — ADJUDICATED MECHANICALLY (5 seeds; ⛔ reported, not interpreted)

| check | statistic | measured | bar | clears? |
|---|---|---|---|---|
| **R1** | exact-set occupancy of the asserted set `{j : m_j ≥ 0.5}` | ⭐ **0.0656 ± 0.0239** | `mean − 2 SE > 0.02` ⇒ 0.0417 | ✅ **YES** |
| **G1** | `read − live launder` (worst reader) | ⛔ **−0.0070 ± 0.0016** | `mean − 2 SE > 0` | ⛔ **NO** |
| **S_eff** | `K·F / #wells ever occupied` | **16.774 ± 0.853** | in [8, 16] | ⛔ **NO ⇒ COLLAPSED** |
| **M15** | launch-collapse trips on the claim cell | **0 of 5** (`S_marg = 23.16 ± 2.66`) | no trip | ✅ **YES** |

⛔ **`SWAP_RUNS = False`. The ORGANIZER SWAP WAS NOT RUN** (the task file's instruction, honoured
literally; `--force-swap` exists and was deliberately not used, so no labelled-diagnostic swap
exists either). The wave's product is the cardinality mechanism + the monitor/regularizer
measurements.

---

# 4. ⭐⭐ THE ARMS (5 seeds, `Q_unseen`, exact-set accuracy, `chance = 0.0000`)

| reader | params | **physics** | **live launder `L_a`** | **designed (untrained) head** | **G1 = physics − `L_a`** |
|---|---|---|---|---|---|
| ⭐ `count_identity` (NEW, **0** params) | 0 | **0.0539 ± 0.0207** | **0.0609 ± 0.0218** | ⭐ **0.0695 ± 0.0268** | ⛔ **−0.0070 ± 0.0016** |
| `count_table` (NEW, lstsq) | 72 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `well_table` (hard) | 72 | 0.0008 ± 0.0016 | 0.0000 ± 0.0000 | 0.0023 ± 0.0031 | +0.0008 ± 0.0016 |
| `soft_well_table` (D8 twin) | 72 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0016 ± 0.0019 | 0.0000 |
| `sum_linear` | 136 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mlp` | 108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `knn` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| — | | **best 0.0539** | **best 0.0609** | **best 0.0695** | **`G1_min` −0.0070 ± 0.0016** |

- ⛔ **G1 FIRES, and unlike iteration 1 it is now significant**: `−0.0078 / −0.0078 / −0.0039 /
  −0.0078 / −0.0078` — **5/5 seeds negative**, `mean − 2 SE = −0.0086 < 0`. The launder is *this
  cell's own trained head* with the landscape deleted and the payload table retained, scored through
  the same reader class at the same `k`.
- ⭐ **The designed (untrained) head beats the trained one** (0.0695 vs 0.0539; R1 0.0883 vs 0.0656;
  gated precision 0.571 vs 0.477; launch top-`F` 0.184 vs 0.103) — **w20's "free learning erases
  design", measured at the claim budget** on the same organized store. The head training *was*
  selected on SEEN (where it improved 0.039 → 0.063); it does not transfer.
- **In-sample liveness (the L1-style anchor):** on SEEN the memorising reader `knn` scores
  **1.0000** on 5/5 seeds while scoring 0.0000 on `Q_unseen` — rule 4 doing its job, and the
  cleanest evidence that the split is sound while the vehicle is not.

## 4.1 ⭐ THE MECHANISM BEHIND G1 — the settle SCATTERS the commitment

| statistic (unseen, 5 seeds) | **launch (launder)** | **settled (physics)** | direction |
|---|---|---|---|
| distinct wells occupied, raw | **4.834 ± 0.125** | **5.668 ± 0.396** | ⛔ the settle **adds 0.83 wells** to a `F_hat = 4.54` commitment |
| occupancy precision, raw | **0.4578 ± 0.0260** | **0.4107 ± 0.0191** | ⛔ −0.047 |
| occupancy precision, gated | 0.4672 ± 0.0252 | 0.4773 ± 0.0162 | +0.010 (n.s.) |
| coverage (`A ⊆` gated set) | 0.1180 ± 0.0369 | 0.1133 ± 0.0388 | −0.005 (n.s.) |
| exact-set occupancy (**R1**) | 0.0641 ± 0.0260 | **0.0656 ± 0.0239** | +0.002 (n.s.) |
| `S_marg` (marginal usage perplexity) | 23.39 ± 2.45 | 23.16 ± 2.66 | — |

⭐ **The reading:** the head allocates `k = 12` particles onto ~4.8 distinct wells; the settle
delivers them to **5.67** distinct wells — i.e. particles that were committed *to the same well*
end in *different* basins. Iteration 1 measured why: **the store's actual attractors sit 1.09 away
(1.4 `sep`) from its designed anchors**, so a launch aimed at anchor `j` can fall into a
neighbouring basin. A counting code is exactly the representation that this scattering corrupts:
it converts a clean `3/3/3/3` allocation into `3/2/3/2/2`, and `m_j` stops landing on 1.
⛔ **R2 (registered as "the settle destroys addressability", fires iff settled < launched) does NOT
fire — but the failure at a multiplicity head is the opposite sign, and R2 as registered cannot see
it.**

## 4.2 The cardinality mechanism itself — it works

| statistic (physics, unseen) | iteration 1 | **this iteration** |
|---|---|---|
| the read's asserted-set size | **5.79 ± 0.90** (never 4) | **4.194 ± 0.224** |
| fraction of queries asserting **exactly `F`** | *(no mechanism)* | **0.394 ± 0.031** |
| `F_hat` (the head's own commitment) | *(none)* | **4.542 ± 0.122** |
| exact-set occupancy (**R1**) | 0.0023 ± 0.0047 | ⭐ **0.0656 ± 0.0239** (**28×**) |
| `P(≥ F distinct)`, settled | 1.0000 | 0.9234 ± 0.0172 |
| best reader, unseen | 0.0008 ± 0.0016 | ⭐ **0.0539 ± 0.0207** (**67×**) |
| out-of-class ceiling at the cell | 0.695 (`d=8`, 1 draw) | **0.691** (recomputed live) |

## 4.3 The weight-mode / dedupe-verb ablation (⭐ free — all five ride on ONE read pass)

`count_identity`, unseen, 5 seeds. The registered claim cell is **`both`**.

| aggregation | physics | launder |
|---|---|---|
| **`both`** (descent × overlap — REGISTERED) | **0.0539 ± 0.0207** | 0.0609 ± 0.0218 |
| `descent` only | 0.0555 ± 0.0222 | 0.0609 ± 0.0218 |
| `overlap` only | 0.0578 ± 0.0215 | 0.0609 ± 0.0218 |
| `none` (a plain count) | **0.0578 ± 0.0219** | 0.0609 ± 0.0218 |
| `noisy_or` (the refuted set-union verb, same weights) | 0.0578 ± 0.0235 | 0.0602 ± 0.0227 |

- ⛔ **Overlap-as-importance weighting COSTS 0.0039 at this cell** (`none` 0.0578 → `both` 0.0539),
  and the reason is structural: the metric is **all-or-nothing**, so any per-particle re-weighting
  that is not exactly uniform pushes `m_j` off 1 *even when the set is right*. Deliverable 2 is
  built and measured; it is **not** free, and the direction is registered here rather than assumed.
- ⭐ **The dedupe verb is LIVE but the two verbs agree at this head, and there is a proof-shaped
  reason.** `counting_code` returns 3.0 where `noisy_or` returns 1.0 (pytest-asserted) — so the
  verbs are *not* bit-identical as they were in iteration 1 §6. But after the importance
  normalisation `m_j = F_hat · cnt_j / Σ cnt`, a **uniform** allocation (`k/F_hat` particles per
  named well) gives `m_j = 1` under **both** verbs. They can only differ when multiplicities are
  *unequal*, which is exactly what a clean commitment prevents. Measured gap: **0.0039**.
  ⚠ **P13 (my prediction that the verbs would differ by > 0.05) FAILS, and the failure is the
  finding**: a working `F`-commitment makes the dedupe verb *irrelevant*, not live.

---

# 5. ⭐ THE FOUR §A20.3(c) GUARDS + THE NEW MONITOR — each on a designed negative

⛔ A guard that cannot fire is N74's vacuous gate. Every row has a demonstrated negative, all
`pytest`-asserted in `tests/test_multiplicity_read.py` (iteration 1's precedent).

| guard | statistic | **measured** | designed negative | fires? |
|---|---|---|---|---|
| **G1** live launch-only launder (A1 strong) | `read − launder`, worst reader | ⛔ **−0.0070 ± 0.0016** (5/5 seeds) | a 0-step read equals the launder: `max|Δz| = 0.0` **bit-identical**, `max|Δm| = 7.7e-5` (float32 round-off through the descent-gate constant) | ⛔ **FIRES** |
| **G2** soft-assignment training signal | `‖∇_head‖` hard/soft, assignment isolated | soft **0.9586**, hard **0.0 EXACTLY**, ratio **0.0** | `argmax` one-hot, magnitude **and** cardinality channels detached | ✅ fires on the negative |
| **G2′** ⭐ *the same probe with `F_hat` LIVE* | ratio hard/soft | **0.99877** (39.398 / 39.447) | — | ⛔ **the head still trains under `argmax`, through cardinality** (reconciliation 3) |
| **G3** staged store-then-launch | `‖∇_store‖` blank vs written | historical-init blank **6.70e-8** vs written **15.58** ⇒ ratio **4.3e-9**; designed-init blank **26.04** (alive) | two blanks (designed-init and historical-scatter) | ✅ fires (8 orders) |
| **G4** `k` on the byte ledger | `k = 12` vs `k = 24` at 2× flops | ⛔ best **0.0430 → 0.0430**, coverage **0.2344 → 0.2344** (identical), distinct 5.72 → 5.99; `assert_k_matched` **raises** on a mismatched ledger | a mismatched ledger | ⚠ **the ledger check fires; the capacity CLAIM is refuted** (reconciliation 2) |
| **M15** launch collapse (**NEW**, `monitors.py` #15) | `S_marg = exp(H(p̄))`, band `≥ 0.5 N_a = 16` | live read **27.86 → quiet**; claim cell **23.16 ± 2.66 → quiet** | input-independent allocation ⇒ **4.000 → TRIPS**; one-well head ⇒ **1.000 → TRIPS**; ⭐ and it **tripped on a real training run** (`head_lr = 3e-2`: `S_marg = 6.9`) | ✅ fires, incl. once un-forced |

⭐ **G3 reproduces iteration 1's finding and adds one:** the store gradient is dead at the
*historical* init (6.7e-8) and alive at the *designed* init (26.04) — but the **head's** gradient no
longer collapses on a blank store (0.64× of written, vs iteration 1's 1/1300). At a multiplicity
head the head trains off the **address** channel, which exists without a write. **The staging
ordering is still correct for the store, and is no longer forced by the head's gradient.**

---

# 6. (3) + (4) THE ANTI-COLLAPSE REGULARIZER — ⛔ BOTH STATES REPORTED (doctrine §3.3)

5 seeds, head retrained under each state, everything else identical.

| state | `λ` | `S_marg` | `max_j p̄_j` | per-query perplexity *(diagnostic only)* | best reader | **R1** | monitor trips |
|---|---|---|---|---|---|---|---|
| ⭐ **OFF (REGISTERED)** | **0.0** | **21.22** | 0.1067 | 3.94 | **0.0516** | 0.0547 | 0/5 |
| ON | 1.0 | 20.32 | 0.1111 | 4.13 | 0.0500 | 0.0539 | 0/5 |

- ⭐ **The activation rule behaved exactly as doctrine §3.3 specifies, and it was exercised for
  real.** The monitor is quiet on the claim cell ⇒ the regularizer stays **OFF**; turning it on
  moves nothing (`ΔS_marg = −0.90`, `Δbest = −0.0016`, both within noise). ⛔ **It is not vacuous:**
  the same monitor **tripped** on the `head_lr = 3e-2` training run (`S_marg = 6.9 < 16`, §7), which
  is the regime the regularizer exists for.
- ⭐ **The MARGINAL/per-query distinction is not decorative:** on the claim cell the *per-query*
  perplexity is **3.94** (≈ `F`: every query concentrates its `k` particles onto ~4 wells — the
  design working) while the *marginal* is **21.22**. A regularizer that penalised concentration
  would be attacking the mechanism. Pytest asserts the penalty is `< 0.15` on a
  per-query-concentrated / marginally-flat batch and `> 1.5` on a marginally collapsed one.

---

# 7. THE LEVERS + THE `depth_ratio` DIAGNOSTIC AXIS (5 seeds, designed head)

| lever | `w\|A` | `w\|¬A` | distinct | prec (raw) | prec (gated) | cov (gated) | **R1** | best reader |
|---|---|---|---|---|---|---|---|---|
| **`p₀` ON** (registered) | **0.815** | 0.503 | 5.73 | 0.472 | 0.589 | **0.100** | 0.0578 | **0.0477** |
| **`p₀` OFF** | 0.611 | 0.220 | 5.54 | 0.386 | 0.599 | **0.012** | 0.0109 | **0.0063** |
| `depth_ratio = 1` ⛔ **DIAGNOSTIC, never a claim cell** | 0.855 | 0.602 | 5.70 | 0.493 | 0.577 | 0.132 | **0.0727** | **0.0617** |

- ⭐ **Learned `p₀` is LOAD-BEARING and its effect has moved from reach to the score: 7.6×**
  (0.0063 → 0.0477), coverage **8.3×** (0.012 → 0.100). Iteration 1 measured it buying coverage ×12
  and *no* selectivity; at a multiplicity head a particle that cannot reach its committed well
  corrupts the count directly, so reach now **is** score. ⛔ It remains a **reach lever only**
  (§A14.1): gated precision is unchanged/slightly worse (0.589 vs 0.599).
- ⭐ **`depth_ratio = 1` beats the claim cell by +0.0140 (+29 %) on the best reader and +0.0149 on
  R1** — iteration 1's follow-up 2 confirmed in the same direction at a *different* read protocol.
  ⛔ Per the Hub's pre-registration this is a **diagnostic axis and never a claim cell**; it is
  reported so the standing conflict (the registered ≥3× heterogeneity exists so F5 is falsifiable,
  and it costs the read) stays visible with a number attached.
- ⛔ **A collapsed head is what the `head_lr` sweep produced and it is worth its own line.** At
  iteration 1's `head_lr = 3e-2`: `F_hat` saturates at `f_max = 8`, `rho` **flips sign**
  (0.25 → −0.176), occupancy precision falls to **0.124** (chance = 4/32 = 0.125), launch top-`F`
  exact-set → **0.000**, every reader → **0.0000**, and **M15 trips at `S_marg = 6.9`**. The
  registered `3e-3` was selected on SEEN (0.039 designed → 0.063 trained at the training budget).

---

# 8. PREREG SCORECARD (`.claude/outputs/c2w7-read-cardinality/PREREG.md`, filed before the harness)

| # | registered | measured | verdict |
|---|---|---|---|
| **K0** | ≥ 0.90 store-free, designed init | **0.9195** (trained head **0.9070**) | ✅ **PASS, adjudicated first** |
| **R1** | fires ≤ 0.001, **clears > 0.02** | **0.0656 ± 0.0239** ⇒ `mean − 2 SE = 0.0417` | ✅✅ **CLEARS** (iteration 1: 0.0023) |
| **R2** | fires iff settled < launched − 2 SE | settled **5.67** vs launched **4.83** | ✅ does not fire — ⚠ **but see §4.1: the failure has the opposite sign and R2 cannot see it** |
| **G1** | fires iff `≤ 0` | **−0.0070 ± 0.0016** | ⛔ **FIRES** (significant, 5/5 seeds) |
| **G2** | ratio `< 1e-3` | **0.0 exactly** (assignment isolated) | ✅✅ (⚠ 0.99877 with `F_hat` live) |
| **G3** | blank `< 1e-6` while written `> 1e-2` | **6.70e-8** vs **15.58** | ✅ |
| **G4** | doubling `k` raises the score | **0.0430 → 0.0430** | ⛔ **the capacity claim is REFUTED at a multiplicity head** |
| **M15** | band `S_marg ≥ 16` | **23.16 ± 2.66**, 0/5 trips; negatives 4.00 / 1.00 trip | ✅ live and non-vacuous |
| **S_eff** | in [8, 16] or COLLAPSED | **16.774 ± 0.853** (30.6 of 32 wells) | ⛔ **COLLAPSED by the rule** (⚠ upper edge — reconciliation 4) |
| **P1** physics best reader | 0.10, [0.02, 0.25] | **0.0539** | ✅ in band |
| **P2** live launder | 0.12, [0.03, 0.28] | **0.0609** | ✅ in band |
| **P3** `G1_min` | −0.02, [−0.12, +0.03] | **−0.0070** | ✅ in band |
| **P4** R1 | 0.12, [0.02, 0.30] | **0.0656** | ✅ in band |
| **P5** `F_hat` | 4.5, [4.0, 5.0] | **4.542** | ✅✅ |
| **P6** gated set has exactly `F` | 0.55, [0.30, 0.80] | **0.394** | ✅ in band |
| **P7** settled distinct wells | 4.4, [3.5, 5.5] | **5.668** | ⛔ **above band** (the scattering of §4.1) |
| **P8** `S_eff` | 16.0, [8, 16] | **16.774** | ⛔ just outside |
| **P9** `S_marg` | 28, [20, 32]; no trip | **23.16**, no trip | ✅ |
| **P10** `‖∇_hard‖/‖∇_soft‖` | 0 exactly, [0, 1e-8] | **0.0** | ✅✅ |
| **P11** 0-step read vs launder | bit-identical | `max\|Δz\| = 0.0` | ✅ |
| **P12** `count_identity − count_table` | +0.10, [+0.02, +0.25] | **+0.0539** | ✅ in band |
| **P13** dedupe verb differs by > 0.05 | > 0.05 | **0.0039** | ⛔ **FAILS — and the failure is a finding** (§4.3) |
| **P14** `none − both` | +0.02, [−0.02, +0.10] | **+0.0039** | ✅ in band |
| **P15** K0 on the trained head | 0.90, [0.80, 1.00] | **0.9070** | ✅✅ |
| **P16** anti-collapse ON − OFF | `ΔS_marg` +1.0, [−1, +6]; \|Δscore\| < 0.03 | **−0.90**, `Δscore = −0.0016` | ✅ (ΔS_marg at the band edge) |
| **P17** the organizer swap | **NOT RUN** (gate predicted to fail on G1) | **NOT RUN** (gate failed on G1 **and** `S_eff`) | ✅✅ |

**Score: 18 ✅ · 5 ⛔.** ⭐ **The pre-registered priors were right in both directions:**
`P(R1 clears) = 0.70` → it cleared; `P(G1 clears) = 0.15` → it did not; `P(the gate fires) = 0.12` →
it did not. ⛔ **And the prediction that failed (P13) is the one that taught something**: a working
cardinality commitment does not make the dedupe verb live, it makes it *irrelevant*.

---

# 9. How I verified (commands + observed output)

```
git worktree add ../CHLU-c2w7 -b c2w7-read-cardinality main
# main venv reused (protocol §4, w6 lesson); cwd = the worktree; jax 0.9.0 / eqx 0.13.4
PYTHONPATH=$PWD .venv/bin/python .claude/scratch/c2w7-read-cardinality/k0_design.py sweep   # ⛔ SEEN only
PYTHONPATH=$PWD .venv/bin/python -m ruff check chlu/ tests/            # All checks passed!
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-card --quick --out-dir …/quick          # 6 stages, ~2 min
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-card --stages k0 arms --seeds 0 \
    --out-dir …/pilot1                                                 # ⚠ the head-collapse pilot (§7)
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-card --seeds 0 1 2 3 4 \
    --stages k0 arms guards --out-dir …/main                           # 1 587 s (26.4 min)
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-card --seeds 0 1 2 3 4 \
    --stages guards regularizer levers --out-dir …/extra               # 2 246 s (37.4 min)
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_multiplicity_read.py -q   # 16 passed in 52 s
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q
#   ⭐ 1379 passed, 0 failed (31 warnings, 2 006.58 s) — baseline 1363 on main @ 104ca19 + my 16
```
Artifacts under `.claude/outputs/c2w7-read-cardinality/`: `PREREG.md` · `main/` (every seed × reader
× arm) · `extra/` (guards with the isolated G2, regularizer, levers) · `pilot1/` (**the labelled
head-collapse diagnostic**) · `quick/` · `k0_design_sweep.json` · `run_main.log`, `run_extra.log`,
`pytest_full.log`. Scratch: `.claude/scratch/c2w7-read-cardinality/k0_design.py`.

**⚠ FAILURES AND MISSTEPS, REPORTED.**
1. ⛔ **The first design (greedy successive-suppression pursuit with a learned residual-norm
   stopping rule) was built, measured and DISCARDED** before the prereg was filed: exact-set 0.008
   on SEEN, *below* the 0.023 of plain top-`F` ranking. The cause is a theorem, not a bug (coherence
   0.35 ≫ the `1/(2F−1) = 0.14` OMP needs). Replaced by unrolled non-negative ISTA (0.19) —
   registered as **D12**, with the whole comparison table in the prereg.
2. ⛔ **The first pilot at `card_b = 0.5` failed K0 on unseen (0.871 < 0.90)** although it passed on
   SEEN (0.912). Re-tuned on **SEEN only** to `card_b = 0.7` (SEEN 0.953 → unseen **0.9195**). The
   failing pilot is retained (`pilot1/`) and is not quoted as a claim cell.
3. ⛔ **The same pilot's trained head COLLAPSED at iteration 1's `head_lr = 3e-2`** (§7) — caught by
   the new monitor, then fixed by a SEEN-only lr sweep. No unseen number was ever used to choose it.
4. ⚠ `guards` ran twice: once inside `main/` (before the `detach_f` isolation existed, reporting the
   uninformative G2 ratio 0.9988) and once in `extra/` (with it, ratio **0.0**). **§5 quotes
   `extra/`**; both files are retained and the difference is explained in reconciliation 3.

---

# 10. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)
1. ⛔ **THE ORGANIZER SWAP** — the gate did not fire. `stage_swap` is built, tested and wired
   (`--force-swap` exists); it was **deliberately not forced**, so there is no labelled-diagnostic
   swap either, and **no `OD_min` / F1 / F2 / F5 number exists this wave.**
2. `k = 16` / `k = 24` as scored arms (`k = 24` appears only as guard 4's capacity probe).
3. A ψ A/B — this vehicle contains **no ψ**; ψ's marginal value belongs to CSF3 run-1/run-2.
4. N2 / N3′ / N4 / N5 as swap arms; and the swap's **robustness arm** (N1′ with the head re-fitted).
5. The γ axis — one claim cell at `γ_address = 0.05`, `γ_read = 0.02`.
6. `d ∈ {4, 16, 24}` as scored cells (the ceiling is recomputed at `d = 8` only).
7. The consolidate/trash stage (iteration 1's (d)) — C2W8 owns it.
8. The `pilot1/` single-seed run at `card_b = 0.5` / `head_lr = 3e-2` — a **labelled diagnostic**
   (the K0-miss and the head collapse), never a claim cell.

# 11. Git footprint
- **Branch:** `c2w7-read-cardinality` (off local `main @ 104ca19`), worktree `../CHLU-c2w7`.
  ⛔ Not pushed, no PR, no merge. `origin` untouched, `clu-dev` untouched.
- **Commits** (verified from the MAIN repo, protocol §3.2 — the wave-4 lesson):
  - `5a952f0` `[experiment-engineer] multiplicity-as-counting-code: the read's F-commitment (charter §A21, C2W7)`
  - `92aea1b` `[experiment-engineer] monitors: the launch-collapse row (#15), additive`
  - `3c19878` `[experiment-engineer] exp_tierii_cardinality + the chlu exp-tierii-card CLI hook`
  - `baf361f` `[experiment-engineer] tests: the F-commitment, the counting code, the monitor's negatives, the four guards`
- **Files touched (the declared ownership list, exactly):** `chlu/core/multiplicity_read.py`
  (**new**) · `chlu/experiments/exp_tierii_cardinality.py` (**new**) ·
  `tests/test_multiplicity_read.py` (**new**) · `chlu/core/monitors.py` (**+1 class, +1 SEVERITY
  row, +1 `default_registry` line, +1 `__all__` entry — additive only, no existing line altered**) ·
  `chlu/cli/experiment_cmd.py` (**+1 subcommand block, +1 handler, no existing line altered**).
  ⛔ **Not touched:** `chlu/training/train_cluformer.py`, `chlu/core/blocks.py`, `scripts/csf3/`
  (C2W6/CSF3 territory) · `chlu/core/psi_readout.py` · `chlu/core/factored_store.py` (no additive
  hunk turned out to be needed — the new module composes with its public API) ·
  `chlu/core/multiwell_read.py` (imported, **not modified**) · `chlu/core/null_arms.py` (public API
  only) · `chlu/config.py`.
- **Rebase:** onto local `main` (⚠ **not** `origin/main`, §7.21) — base unmoved, no-op.
- **Worktree:** verified from the MAIN repo (`git -C /Users/user/Desktop/CHLU log --oneline
  main..c2w7-read-cardinality` = 4 commits) **before** `git worktree remove ../CHLU-c2w7`; **wt2 is
  free**, the branch remains for review.
- **Concurrent work:** `../CHLU-c2w6` (`c2w6-anti-erosion`) was live throughout. **Zero file
  overlap** — C2W6 owns `train_cluformer.py` + `blocks.py`, neither of which I touch;
  `monitors.py` was ceded to C2W7 on the §10 record and my hunk is additive. The shared main
  checkout was never edited.

# 12. Open questions / follow-ups / risks
1. ⭐⭐ **The vehicle's verdict is now mechanistic, not statistical.** The store cannot beat its own
   launches because (i) every well is written on *every* query, so the landscape carries no
   query-specific information, and (ii) the settle actively *scatters* a committed allocation
   (4.83 → 5.67 distinct wells) because the store's attractors are ~1.4 `sep` from its anchors.
   **A tier-ii dividend on this family needs a store whose geometry is query-conditional, or a
   family where the landscape is not written identically for every query.** This is the question I
   would put to the Advisor before a third read iteration is funded.
2. ⭐ **The reader-fitting pathology (reconciliation 1) is the highest-value loose thread.** It is
   cheap to re-check on banked artifacts: re-score `orgdiv-null-arms` and `tierii-read-fix` latents
   through a zero-parameter identity reader. If their zeros survive, the C2W5/C2W6 conclusions are
   untouched and the finding is scoped to this wave; if they do not, several published zeros move.
3. **The `depth_ratio` conflict now has a number on both iterations** (+29 % on the best reader
   here, +0.035 gated coverage in iteration 1). Someone has to adjudicate whether F5's falsifiability
   or the read's selectivity moves.
4. **`S_eff`'s band is one-sided in the code** (`8 ≤ S_eff ≤ 16` with `S_eff = K·F/W`): the upper
   edge is "all wells used", so exceeding it is *under-usage*, not concentration. The label
   COLLAPSED is doing two jobs.
5. **Risk to how this gets quoted.** "R1 clears / the read commits to `F` / the cardinality
   mechanism works" are all true and all *expressivity* statements. ⛔ **They must never be quoted
   without G1 in the same paragraph** (the standing never-quote of iteration 1, inherited and now
   sharper: G1 is significant).

---

## Proposed handover updates (for the Hub)

- **§3 CLI/config — NEW:** `chlu exp-tierii-card [--stages k0 arms guards regularizer levers swap]
  [--seeds …] [--organize-steps N] [--k-particles K] [--lam-on F] [--force-swap] [--quick]
  [--out-dir D]`; new module `chlu/core/multiplicity_read.py` with `MultiplicityConfig` (lives next
  to its code — the `CatTestConfig` / `MultiWellReadConfig` precedent, **not** in `chlu/config.py`).
- **§2 architecture — NEW:** `chlu/core/monitors.py` now carries **row #15 `launch_collapse`**
  (severity I, marginal-usage perplexity, band `≥ 0.5 N_a`), in `default_registry`. ⚠ Anything
  asserting "13 monitors + M14" is now "…+ #15".
- **§7 Known Issues — NEW (open, HIGH, program-wide):** *a least-squares-fitted reader can score an
  informative latent at exactly 0 under a thresholded metric.* Measured: set exactly right on 18 %
  of queries, identity residual 0.006 vs `tol` 0.234, but lstsq shrinks `diag(W)` to 0.40 and the
  good queries land at 0.537 > `tol` ⇒ **0.000 fitted vs 0.054 identity, 5 seeds**. Remedy: keep a
  **zero-parameter** member in every reader class (`count_identity` is the exemplar).
- **§7 Known Issues — NEW (open):** *`K0` is a REACHABILITY statement and is passed trivially by a
  collapsed head* — a head with `F_hat` saturated at `f_max` scores `K0 = 1.000` while every reader
  reads 0.0000. Never quote K0 without R1 **and** the launch-collapse monitor.
- **§7 Known Issues — NEW (mechanism, standing):** *the settle scatters a committed allocation* —
  4.83 launched distinct wells → 5.67 settled, occupancy precision 0.458 → 0.411, because the
  store's attractors sit ~1.4 `sep` from its designed anchors (iteration 1's §7 measurement, now
  with a consequence). Any counting/multiplicity read pays for this directly.
- **§7 Known Issues — RE-SCOPE:** *"doubling `k` raises the score"* (iteration 1's guard 4) is
  **refuted at a multiplicity head**: `k = 12 → 24` at 2× flops moves best-reader and coverage by
  **0.0000**. `k` remains ledgered; it is a resolution dial, not a capacity dial, once the head
  commits.
- **Registry/doctrine candidates:** (i) ⭐ *a guard's designed negative belongs in the test suite* —
  all five here (incl. the new monitor's two) are `pytest`-asserted; (ii) ⭐ *the monitored-first/
  regularized-second order was exercised for real* — M15 tripped on an actual training run
  (`head_lr = 3e-2`) before any regularizer was switched on, and stayed quiet on the claim cell
  where the regularizer then measured inert; (iii) *a launch-information ceiling must be quoted with
  its `(d, draws)` noise model* — recomputed live here at **0.691**.

---

## ⛔⛔ DATED CURATOR ERRATUM BANNER (2026-08-06, `doc-curator-c2w7-fold`, [C2W7] — body above UNTOUCHED, C-3 precedent)

**Authority:** charter **ADDENDUM 9 §A27.2** (Head-ratified 2026-08-06) + **ADDENDUM 8 §A24/§A26.4**.
**Sources:** `.claude/outputs/reader-fitting-audit.md` §1/§3/§4/§6/§9 · the 2026-08-06 `[C2W7]` §10
entry. ⛔ **No measured number in this report is retracted. Three quotation forms move.**

**1. ⛔⛔ RECONCILIATION 1's HEADLINE SENTENCE IS RE-SCOPED — NOT WITHDRAWN — AND ITS ORIGINAL FORM IS
A NEVER-QUOTE.** ~~*"Every arm this programme has scored at ≈ 0 may have been scored through a reader
that destroys its own signal."*~~ is **MEASURED FALSE for C2W5.** ⭐ **The binding replacement,
verbatim (§A27.2):**

> *A fitted reader destroys signal only in proportion to the fraction of queries whose asserted set is
> **already exactly right** — which is why it cost C2W7 (~18 %) and cost C2W5 nothing (2/2560 ·
> 3/1280 · 0/2560).*

**Evidence:** `reader-fitting-audit` re-scored three banked cells × 5 seeds through zero-parameter
identity readers and returned **SURVIVES** — **`null*` = 0.00117 UNMOVED**, **`OD_min` −0.00078
identical**, **N1 1.0000/0.0000 unmoved**, **three reproduction gates bit-for-bit** (including
non-zero, seed-asymmetric cells), **largest identity-minus-fitted gain anywhere +0.0023** (21× below
the 0.05 bar). ⭐⭐ **This report's MECHANISM is CONFIRMED, not refuted** — the lstsq shrinkage
reproduces at every cell (`diag(W)` **0.13–0.45**) and the `tol` crossing **fires on two of them**,
with the identity reader keeping **every** correct query and the fitted reader **none** — **it simply
had nothing to destroy there.** ✅ **Add.5 §A20 stands as written; NO erratum is owed; the wave's
caveat is LIFTED.** ⛔⛔ **And the refuted hope, recorded explicitly: adding the zero-parameter reader
makes iteration 1's `G1` WORSE, −0.0016 → −0.0023 — the fitting artifact was not hiding a physics
dividend.** ⚠ **The honest qualifier travels with SURVIVES: the tie is a tie at ≈ 0, so it rules out a
reader artifact and does not make any arm alive.** ⭐ **This report's own remedy is now standing
doctrine, in two parts (§A26.3/§A28.3): `K6` — report the exactly-right fraction BEFORE any reader is
fitted — and every reader class carries a zero-parameter member, ADDED, never substituted, never
reported alone** (it can be **strictly worse**: +0.0109 on a real cell, > 0.99 on a designed control).
(**N237**.)

**2. ⛔ THE `S_eff` LABEL IS RE-REGISTERED (§A26.4) AND THIS REPORT'S §3 LINE IS RE-LABELLED.**
Reconciliation 4 was right and the ruling goes further: `S_eff = K·F/W = 512/W` with `W ≤ N_a = 32` ⇒
**`S_eff ≥ 16` always**, so the `[8, 16]` band's lower half is **unreachable by construction** and the
band is **RETIRED** in favour of direct wells-visited `W/N_a` with **two-sided labels**. ⭐ **This
report's *"`S_eff` = 16.774 ± 0.853 ⇒ COLLAPSED by the rule"* is re-labelled *"30.6 of 32 wells
visited — slight UNDER-usage"***, ⛔ **the OPPOSITE failure to C2W5's 34–51 concentration, which the
same word described** (and "COLLAPSED" is now reserved for concentration). ⚠ **No verdict moves: the
gate fails on `G1` alone as substance (§A26.1).** (**N239**.)

**3. ⛔ THE UN-FORCED M15 TRIP IS `S_marg ≈ 7.9`, NOT 6.9.** §5/§7 quote **6.9**; the Hub verified
from raw at the close that the **`pilot1` arms stage reads 7.9173**. ⭐ **Quote ≈7.9 and stage-label
it** — both are ≪ the 16 band, so **the trip is real either way** and no conclusion changes.
(§A24 footnote i; **N240**.)

**⚠ Two standing scope statements now attach to every quotation of this report (§A28.4, Head-ratified):**
⛔⛔ **no full-CLU verdict may be stated or implied — this vehicle has no ψ at all, no consolidation or
trash, no traversal, no persistent store, and runs in the factored-store gym, not the streaming
block**; ⚠ **and the missing components are NOT the deepest defect and must never be offered as the
explanation** (the landscape is written identically for every query, so the settle cannot beat its own
launches by construction). ⛔ **The read track is PAUSED (§A26.6): there is no third read iteration and
none is scoped.**
