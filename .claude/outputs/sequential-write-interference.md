# sequential-write-interference — experiment-engineer report

**Task + acceptance criterion:** does an MVC-0 admission gate stop new writes destroying stored items — gated vs ungated on the exact w20 failing setup, the K=1..16 sequential-write retention curve, the cross-primitive comparison with compute-to-criterion and retrieval-cost scaling, the parametric/contextual scope statement, ≥5 seeds, tests green.
**Status: done.** All four items run. **The headline is the loud negative the task asked me to look for, and it is sharper than "the gate does not transfer": the gate is *arithmetically incapable of firing* on the w20 geometry, and where it can fire it preserves items only by refusing to store new ones. What actually rescues a learned landscape is a different mechanism entirely (a structured write operator), and I measured it.**

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Four items.**
> 1. **The Hub's adjudication table must be re-scoped.** "ungated 8.39 / gated 8.0e-5" and "ungated 0.35 / gated 1.000 selectivity" are **designed-atom-landscape numbers on a crowded random-placement task**. They are reproduced here (§1.4: my independent atom write gives **1.008e-4** at exactly `d = d_safe`, within 26 % of the theorist's 8.0e-5) — but they **do not transfer to a learned landscape**, and the spacing gate contributes **nothing** to them on the w20 geometry. The claim "locality is a certificate the controller CHECKS" survives *only* for local write operators; for a global-support write the controller can check and **refuse**, nothing more.
> 2. **"An admission gate suppresses interference by ~5 orders" must not be stated about CLU generally.** Measured suppression of w20's corruption: spacing gate **1.00×** (identical to all reported digits), C3 gate **∞ but at zero items stored**, anchored/structured write **103×**. Nobody measured 5 orders on a learned landscape.
> 3. **The Head's "wasted compute reorganising to conserve key info" hypothesis is REFUTED as stated** (§3.3). Compute-to-criterion is **flat in K for every primitive** (mlp 10.8→14.6, gru 15.8→13.4, attention 13.0→9.4, clu 5.6→12.0 steps over K=1→64). The cost of conserving prior information does not show up as more steps — it shows up as **impossibility**: the joint criterion is censored at **100 %** for mlp/gru/clu by K=32.
> 4. **`primitive-harness`'s "CLU degrades more gracefully than the GRU" does not generalise to parametric sequential writes.** Here CLU is the **worst** of the four at K=64 (mean retention 0.16 vs gru 0.57, mlp 0.43, attention 0.34). The graceful-capacity result is a *contextual* MQAR property; this is a *parametric* store. Both must carry the label.

---

## Flag-provenance table

| item | value |
|---|---|
| commit | `2be2a9f` (branch `agent/experiment-engineer/sequential-write-interference`, base local `main` @ `31c3e15`) |
| seeds | **0, 1, 2, 3, 4** (5 seeds) on every arm of every item; arms are **paired** — identical proposal sequence per seed, so arms differ only in the controller |
| landscape (item 1) | `DesignFreedomPotential`, rungs `designed` (0 learned params) / `sites_learned_payload` (4 481) / `free_mlp` (4 481); K=4 ring, f=1.0, λ=1.0, barrier 0.2, payload_kappa 1.0, bump_width 0.05, payload_seed 0 — **identical geometry to w19/w20** |
| landscape (item 2, designed) | **`AtomDictionaryPotential`** (new): α=0.02, s=0.35, A=1.0, κ=1.0 — the theorist's S3 values |
| landscape (item 2, learned) | `free_mlp` rung (**not** the w20 rung: `sites_learned_payload` carries a designed K-well ring at radius f that would fight every off-ring site) |
| write objective | `training/train_memory.py`, Adam(w) lr 3e-3, wd 1e-4, n_perturb 32, σ_addr 0.25, σ_pay 0.6, margin 0.15, barrier 0.2. **600 steps** for write A / each sequential write; **300 steps** for write B |
| admission gate | `d_safe = 4.4·s`: **1.10** on learned landscapes (s = σ_addr = 0.25), **1.54** on the atom store (s = 0.35). δ_budget **0.10**, C3 re-checked every **25** write steps, 400 relocation candidates, proposals uniform in a **disk of radius 2.0** |
| retrieval | TWO-PHASE, γ_address **0.05** × 400 steps → γ_read **0.0** × 800 steps, dt 0.05, tail 25 %, 8 subsample points (w20 §5: fidelity depends on γ_address only) |
| queries | 32/item (item 1), 16/item (item 2); jitter σ = f·0.15, σ_p 0.05; **q2(0)=p2(0)=0 always** (anti-decoration guard) |
| metric | **value recovery only** (basin AND \|read − stored\| < tol), never classification. tol = min(0.1, **0.35 × codebook spacing**) = 0.100 at K=4, **0.0467** at K=16 |
| blank controls | on **every** reported cell of items 1 and 2: identical architecture, identical writes, **all payloads zero**, scored against the **real** codebook. All 24 item-1 blanks and all 5 item-2 blanks read **0.000** |
| cross-primitive slot | `primitive-harness` verbatim: `chlu/core/blocks.py`, d_model 64, 2 layers, block params matched to **40 000** (mlp 40 118 / gru 40 014 / attention 40 960 / clu 39 886; max match error 2.4 %), Adam + global-norm clip 1.0, LR grid **{3e-4, 1e-3, 3e-3}** identical for every primitive, symmetric monotone rescue at K=16 **and** at K=64 |
| kv task | vocab 128 (chance 0.0078), key length 4 tokens, values sampled without replacement, criterion = argmax correct at the last position, ≤200 Adam steps per item, checked every step |
| langevin_noise | **N/A** — deterministic Verlet throughout; no Langevin, no temperature, no sleep phase |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** — main venv reused per protocol §4 (no worktree `uv sync`) |
| machine | shared; load average 17–21 during items 1–3. **FLOPs come from XLA cost analysis (exact, load-independent)**; the only wall-clock numbers reported are item 4's, taken in a single interleaved pass, and they are secondary to the FLOP counts |

**Reproducibility check:** item 1 was run twice, in two independent processes on two different days' worth of machine load, once before and once after an output-field addition. **Maximum absolute difference across all 120 cells × 5 reported quantities: 0.0.**

---

## 0. Headline

1. **The spacing gate cannot fire on the w20 geometry, and does not.** Site spacing on the K=4 ring is `2f·sin(π/4) = 1.4142`; `d_safe = 4.4·σ_addr = 1.10`. Every w20 write is already admissible ⇒ gated and ungated are **identical to all reported digits** (corruption `0.682 ± 0.30` both; strict-A-after-B `0.000` both). Pre-registered as P1; confirmed.
2. **The 5 orders are a property of the WRITE OPERATOR, not of the gate.** An atom write's induced fixed-point drift at a stored minimum is `1.008e-4` at `d = d_safe` and `4.57e-1` at `d = 0.5 s` — the `exp(−d²/2s²)` law, measured over 8 distances. My independent implementation lands within **26 %** of the theorist's `8.0e-5`. A learned MLP write has no such factor: measured drift `1.69–3.73`.
3. **On a learned landscape the gate preserves items only by refusing to store new ones.** C3-gated writes were truncated at **0 of 300 steps on 100 % of cells**: corruption exactly 0.000 and strict-A held at 1.000, with **strict-B = 0.000** — the new item is simply not there. In the sequential curve, `learned_gated` admits **10.0 ± 1.4** sites and actually writes **1.0**.
4. **What does rescue a learned landscape is a structured write operator, and it is not a gate.** Anchoring the stored items in the write loss (rehearsal from the codebook, C3 option (b)) gives corruption `6.6e-3 ± 3e-3` with **strict-A 1.000 and strict-B 0.975** — a **103×** suppression at *no capacity cost*. This is the only arm that both preserves and stores.
5. **The sequential curve separates the arms at the first subsequent write.** CLU learned+ungated: item-1 retention `1.00 → 0.00` after **one** write. CLU designed+gated: `1.00` at every K, mean retention `1.00` at every K, at the price of admitting only **6.0 ± 0.9 of 16** proposals. Designed+**ungated** (same local write operator, only placement differs) falls to **0.16**. *The gate's entire contribution is visible in that one contrast: 1.00 vs 0.16.*
6. **Cross-primitive: catastrophic forgetting under sequential parametric writes is universal, and CLU is the worst of the four.** At K=64 mean retention: gru 0.57, mlp 0.43, attention 0.34, **clu 0.16**; item-1 retention 0.00 for mlp/gru/clu by K=32, attention 0.20 at K=64. **No primitive has anything resembling the designed+gated arm's flat 1.000.** The axis that matters is the controller, not the architecture — but that means our own *learned* CLU is on the wrong side of it with everyone else.
7. **Compute-to-criterion is flat in K for every primitive — the Head's "wasted compute" hypothesis is refuted as stated**, and replaced by something stronger: the *joint* criterion (store item K without dropping 1..K−1) becomes **unreachable**, censoring 100 % by K=32 for mlp/gru/clu.

---

## 1. Item 1 — the gate on the EXACT w20 failing setup (5 seeds)

`decisions` = how the controller disposed of B's proposal. `steps` = write steps actually committed of a 300-step budget. Value-recovery metric throughout; **every cell's blank control reads 0.000**.

### 1.1 Ring proposal — the w20 geometry verbatim

| rung (learned params) | arm | decisions | steps | **corruption** | strict A before | **strict A after** | **strict B** |
|---|---|---|---|---|---|---|---|
| designed (0) | *all four* | admit 5/5 | n/a | **0.000 ± 0** | 1.000 | 1.000 | 1.000 |
| sites_learned_payload (4481) | **ungated** | admit 5/5 | 300 | **0.682 ± 0.30** | 1.000 | **0.000** | 1.000 |
| sites_learned_payload | **gated_spacing** | admit 5/5 | 300 | **0.682 ± 0.30** | 1.000 | **0.000** | 1.000 |
| sites_learned_payload | **gated_c3** | admit 5/5 | **0 ± 0** | **0.000 ± 0** | 1.000 | **1.000** | **0.000** |
| sites_learned_payload | **anchored** | admit 5/5 | 300 | **0.0066 ± 0.003** | 1.000 | **1.000** | **0.975 ± 0.05** |
| free_mlp (4481) | ungated | admit 5/5 | 300 | 0.686 ± 0.19 | 0.931 ± 0.13 | 0.215 ± 0.25 | 0.938 ± 0.12 |
| free_mlp | gated_spacing | admit 5/5 | 300 | 0.686 ± 0.19 | 0.931 ± 0.13 | 0.215 ± 0.25 | 0.938 ± 0.12 |
| free_mlp | gated_c3 | admit 5/5 | 0 ± 0 | 0.000 ± 0 | 0.931 ± 0.13 | 0.931 ± 0.13 | 0.000 |
| free_mlp | anchored | admit 5/5 | 300 | 0.0489 ± 0.054 | 0.931 ± 0.13 | 0.942 ± 0.08 | 0.950 ± 0.06 |

- **w20 replicates.** `sites_learned_payload` ungated: strict-A `1.000 → 0.000`, exactly w20 §3's headline; corruption `0.682 ± 0.30` across 5 seeds vs w20's single-seed `0.495` — same band, and the 5-seed mean is *worse*.
- **`gated_spacing` ≡ `ungated`, digit for digit.** Zero refusals, zero relocations, in 40/40 cells. **P1 confirmed.**
- **`gated_c3` refuses everything.** Not "truncates to <25 % of budget" as pre-registered, but **0 of 300 steps on 100 % of cells**: the *first* 25-step chunk already exceeds `δ_budget = 0.10`. The gate's output is a refusal, and the price is `strict_B = 0.000`. **P4 confirmed, more extremely than predicted.**
- ⚠ **The `designed` rung of item 1 is degenerate and its 0.000 must not be quoted as a gate result.** It has **zero trainable parameters** (recorded in the JSON), so "writing B" changes nothing at all — w20's designed 0.000 has the same cause. The non-degenerate designed *write* is item 2's atom dictionary.

### 1.2 Disk proposal — making the gate non-vacuous

Proposing B uniformly in a disk of radius 2.0 puts it inside `d_safe` most of the time, so the gate fires: **admit 1/5, relocate 4/5 — an 80 % intervention rate** (P3 predicted ≥35 %; confirmed).

| rung | arm | corruption | strict A after | strict B |
|---|---|---|---|---|
| sites_learned_payload | ungated | 2.70 ± 1.3 | 0.000 | 1.000 |
| sites_learned_payload | **gated_spacing (relocated!)** | **2.56 ± 1.3** | **0.000** | 0.988 ± 0.03 |
| sites_learned_payload | gated_c3 | 0.000 | 1.000 | 0.000 |
| sites_learned_payload | anchored | 0.0145 ± 0.027 | 0.992 ± 0.02 | 0.400 ± 0.49 |
| free_mlp | ungated | 0.906 ± 0.36 | 0.006 ± 0.01 | 0.688 ± 0.41 |
| free_mlp | **gated_spacing (relocated!)** | **0.740 ± 0.32** | 0.200 ± 0.27 | 0.969 ± 0.05 |
| free_mlp | anchored | 0.0227 ± 0.036 | 0.946 ± 0.10 | 0.762 ± 0.39 |

⭐ **This is the cleanest possible statement of the negative.** Here the spacing gate *does* fire, on 80 % of writes, and it relocates B to a certified-safe distance — and the stored item is destroyed anyway: corruption `2.56` vs `2.70`, strict-A `0.000` either way. **Spacing certification is worth a 5 % corruption reduction on a learned landscape.** The certificate is sound; the write operator it certifies is not local, so the certificate is about the wrong thing.

### 1.3 The C3 first-order law

`||H⁻¹∇δV(q*)||` vs the measured relaxation-endpoint shift, median ratio per cell:

| cell | median pred/meas | within 2× |
|---|---|---|
| free_mlp, ring, **ungated** | 0.269 | 40 % |
| free_mlp, ring, **anchored** (small perturbation) | 0.682 | **80 %** |
| sites_learned_payload, ring, ungated | 0.811 | 80 % |
| sites_learned_payload, ring, anchored | 0.657 | 80 % |
| free_mlp, disk, gated_spacing | 0.274 | 20 % |

**The law is a small-perturbation law and behaves like one:** it holds (80 % within 2×) exactly where the perturbation is small — the anchored arm, drift ~1e-2 — and under-predicts by ~4× where the write is destructive (drift ~2). **P6's learned half is confirmed in the destructive cells and refuted in the small ones**, which is the physically correct behaviour and means the controller's C3 estimate is *conservative in the right direction*: it under-reports damage it is about to do, so a budget check on it is safe only because the budget is crossed long before the law breaks.

### 1.4 ⭐ Where the 5 orders actually come from (direct measurement, 8 distances)

Predicted fixed-point drift at a stored atom's minimum when a second atom is written at distance `d` (α=0.02, s=0.35, A=1):

| d/s | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | **4.4 (= d_safe)** | 5.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|
| drift | 4.57e-1 | 3.52e-1 | 1.13e-1 | 1.28e-2 | 4.97e-4 | **1.008e-4** | 6.76e-6 | 3.18e-8 |

- **`1.008e-4` at exactly `d_safe` — an independent reproduction of the theorist's `8.0e-5` to within 26 %**, from a different implementation, in a different codebase, with the same α, s, A. The `8.0e-5` is real.
- The suppression from the merger band to `d_safe` is **4537×** (3.66 orders), not 5; the theorist's 5 orders (8.39 → 8.0e-5) includes fully merged wells at much smaller `d`.
- **All of it is `exp(−d²/2s²)`.** Nothing in this table depends on the gate; the gate's only job is to keep `d` on the right-hand side of it.

---

## 2. Item 2 — the sequential-write curve (K = 1..16, one item at a time, 5 seeds)

Retention = value recovery of item 1 after *n* subsequent write **attempts**; mean retention averages over all items believed stored. `n_admitted` = placement decisions; `n_written` = writes actually committed (the C3 check can admit a site and then refuse the write).

| arm | admitted / 16 | **written** | item-1 retention @ K = 1, 2, 4, 8, 16 | mean retention @ K=16 |
|---|---|---|---|---|
| **CLU designed + gated** | 6.0 ± 0.9 | 6.0 | **1.00 1.00 1.00 1.00 1.00** | **1.00** |
| CLU designed + ungated | 16.0 | 16.0 | 1.00 0.80 0.59 0.39 **0.16** | 0.11 |
| **CLU learned + gated** | 10.0 ± 1.4 | **1.0** | **1.00 1.00 1.00 1.00 1.00** | 0.10 |
| **CLU learned + ungated** | 16.0 | 16.0 | 1.00 **0.00** 0.00 0.00 0.00 | 0.07 |
| CLU learned + anchored | 16.0 | 16.0 | 1.00 0.74 0.34 0.24 0.17 | **0.48** |

Blank controls: **0.000 on all five arms** (written with zero payloads, scored against the real codebook). Guard `blank_is_informative` = True (min |codeword| 0.0667 > tol 0.0467).

**Readings, in order of how much they matter:**

1. ⭐ **The gate's value is the designed+gated vs designed+ungated contrast: 1.00 vs 0.16 at K=16, same write operator, only placement differs.** This is the one place in the whole wave where an admission gate demonstrably saves stored items, and it is *only* available because the write is local. Cost: 10 of 16 proposals refused.
2. **CLU learned + ungated dies on the first subsequent write** — `1.00 → 0.00` at K=2. **P7 confirmed exactly.** The crossover K where designed+gated overtakes learned+ungated is **K = 2** (**P11 confirmed**), with a gap of 0.70 by K=4 (predicted ≥0.6).
3. **CLU learned + gated's flat 1.000 is an artefact of refusal, and the `n_written` column is what exposes it.** It admits 10 sites and writes 1. Its item-1 curve is perfect because after the first write *nothing else is ever written*; its mean retention falls to 0.10 = 1/10 for the same reason. **P10 confirmed: the gate's benefit on a learned landscape is entirely refusal.** Any report of "learned+gated retains item 1 at 1.000" without `n_written` is misleading, and I would have published exactly that had I not recorded it.
4. **Anchoring holds the *mean* (0.48, 4.4× the ungated arm) but not item 1 (0.17).** Rehearsal from the codebook converts catastrophic forgetting into a capacity-limited fit: the MLP cannot hold 16 items, and which item it drops is not controlled. So the mechanism that rescued a single write (§1, 103×) **does not scale to 16**.
5. **Admission count 6.0 ± 0.9 of 16 — P9 FALSIFIED** (predicted 8–14). The prediction's arithmetic used the theorist's *box* `U[−2,2]²` (area 16, hex bound 7.8); the harness proposes in a **disk of radius 2** (area 12.57, hex bound **6.1**). Measured 6.0 ± 0.9 matches the disk bound to 2 %. The falsification is mine, not the physics'; the packing bound is confirmed once the right area is used.

**Figure:** `sequential_write_fig1_retention.png` (panel a: item-1 retention per arm; panel b: mean retention with admission counts; panel c: the cross-primitive curve).

---

## 3. Item 3 — the cross-primitive comparison (parametric writes, matched params, 5 seeds)

**The task's protocol, followed verbatim** (`primitive-harness` §4): identical LR grid `{3e-4, 1e-3, 3e-3}`, identical step budgets, identical seeds `0–4`, block params matched to 40 000 (max match error 2.4 %), plus the **symmetric monotone rescue** — every non-selected LR re-run at full length for *every* primitive, adopted only if the **full 5-seed mean** improves.

⚠ **Honest note on the tuning, stated before the numbers.** The LR *selection* was **uninformative**: at the selection budget (K=8) all three LRs score `1.000` for every primitive, so the "winner" (3e-4 for all four) is arbitrary — the exact failure mode `primitive-harness` §6 flagged. What makes the table defensible is the rescue, which **did** run all three LRs at full length at **both** K=16 and K=64 for all four primitives: **none improved on any primitive.**

### 3.1 At the matched K=16 (the CLU arm's item count) — nobody forgets

| primitive | block params | item-1 retention @K=16 | mean retention @K=16 | joint censoring @K≥8 |
|---|---|---|---|---|
| mlp | 40 118 | 0.40 ± 0.49 | 0.70 ± 0.22 | 0.34 |
| gru | 40 014 | 0.80 ± 0.40 | 0.85 ± 0.18 | 0.15 |
| **attention** | 40 960 | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.00** |
| clu | 39 886 | 0.60 ± 0.49 | 0.80 ± 0.27 | 0.14 |

**P12 is falsified, badly.** I pre-registered "every primitive: item-1 ≤0.30, mean ≤0.40 at K=16" on the strength of w20's CLU-landscape result. Measured: item-1 `0.40–1.00`, mean `0.70–1.00`. **Sixteen sequential parametric writes do not catastrophically forget in any standard primitive at this budget** — and attention does not forget at all. This cell cannot discriminate, which is why the extended sweep exists.

### 3.2 ⭐ Extended to K=64 (selected LR, same seeds, its own rescue) — where each primitive breaks

| K | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| **mlp** item-1 | 1.00 | 1.00 | 1.00 | 0.40 | 0.20 | **0.00** | 0.00 | 0.00 |
| **mlp** mean | 1.00 | 1.00 | 1.00 | 0.70 | 0.55 | 0.52 | 0.54 | **0.43 ± 0.07** |
| **gru** item-1 | 1.00 | 1.00 | 1.00 | 0.80 | 0.20 | **0.00** | 0.00 | 0.00 |
| **gru** mean | 1.00 | 1.00 | 1.00 | 0.85 | 0.60 | 0.51 | 0.59 | **0.57 ± 0.04** |
| **attention** item-1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.60 | **0.20** |
| **attention** mean | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | 0.82 | **0.34 ± 0.10** |
| **clu** item-1 | 1.00 | 1.00 | 1.00 | 0.60 | **0.00** | 0.00 | 0.00 | 0.00 |
| **clu** mean | 1.00 | 1.00 | 1.00 | 0.80 | 0.42 | 0.33 | 0.23 | **0.16 ± 0.02** |

- **Catastrophic forgetting under sequential parametric writes is universal** — every primitive's item-1 retention reaches 0 or near-0, and no primitive holds a flat curve. That much of the framing survives.
- ⚠ **But CLU is the WORST of the four**, on both axes: it loses item 1 first (by K=24, before everyone else) and has the lowest mean retention at K=64 (`0.16` vs gru `0.57`). **The gate-vs-no-gate contrast (§2: 1.00 vs 0.07 at K=16) is much larger than any architecture difference — and our own learned CLU sits at the bottom of the architecture ranking.** Any "CLU is a better memory primitive" framing must confront this table.
- **Attention is the last to break and then breaks hardest** (flat 1.00 out to K=24, then 0.34 by K=64, below gru and mlp). Its per-token embedding rows act as a partially *local* parameter store, which is the same mechanism the atom dictionary exploits deliberately.
- **P13** (primitive spread ≤0.35 ≪ the CLU controller gap ≥0.6): spread **0.30** at K=16 ✅ vs a controller gap of **0.93**; spread **0.419** at K=64, marginally over the registered bound but still less than half the controller gap. ◐ — the *claim* (the controller matters more than the architecture) holds at both K; the registered *number* does not at K=64.

### 3.3 ⭐ Compute-to-criterion — the Head's hypothesis, measured and refuted

Mean steps for item K to reach criterion (argmax correct at the last position), over K:

| primitive | K=2 | K=16 | K=64 | **growth K=2→64** |
|---|---|---|---|---|
| mlp | 9.4 | 11.8 | 14.6 | **1.55×** |
| gru | 12.4 | 13.2 | 13.4 | **1.08×** |
| attention | 12.2 | 11.4 | 9.4 | **0.77×** |
| clu | 6.6 | 7.2 | 12.0 | **1.82×** |

**P14 is falsified.** I registered "≥2× rise from K=2 to K=16 for every primitive"; measured `0.93–1.26×` at K=16 and `0.77–1.82×` even at K=64. **The Head's "wasted compute reorganising to conserve key information" does not appear as more steps.** It is not there at K=16, and at K=64 attention actually gets *faster*.

The second half of P14 (the rise is flat across primitives) holds within a factor `2.36` (max/min of the growth factors), just over the registered 2× bound. ◐.

**What replaces the hypothesis — the joint criterion, and it is stronger.** The measurement that *does* move is the fraction of runs in which item K can be written **without** dropping items 1..K−1 below threshold:

| joint-criterion censoring | K=8 | K=16 | K=24 | K=32 | K=64 |
|---|---|---|---|---|---|
| mlp | 0.00 | 0.80 | 1.00 | 1.00 | 1.00 |
| gru | 0.00 | 0.40 | 0.80 | 1.00 | 1.00 |
| attention | 0.00 | 0.00 | 0.00 | 0.20 | 1.00 |
| clu | 0.00 | 0.40 | 1.00 | 1.00 | 1.00 |

**The cost of conserving prior information is not extra compute — it is infeasibility.** Beyond K≈24 no number of steps within the budget lets mlp/gru/clu add an item without losing one. **P15 confirmed in the extended regime (mean censoring over all `(primitive, K≥8)` cells: 0.711) and falsified in the matched K≤16 sweep (0.156)** ⇒ ◐; the honest statement is that censoring is a steep function of K, not a constant, and it is what "catastrophic forgetting" actually looks like when you instrument the write loop rather than the final accuracy.

⚠ **Scope of §3, restated:** these are **parametric** writes for every primitive including the transformer. See §4.1.

---

## 4. Item 4 — retrieval cost scaling in K, and the scope statement

### 4.1 The scope statement (required — this is the referee's first question)

**CLU's writes in this entire report are PARAMETRIC**: into `V_θ` (items 1–2, at write time) or into the block's weights (item 3, by gradient descent). **Attention's headline memory is CONTEXTUAL**: a KV cache written at inference, never touching a weight. These are *different capabilities*, and the comparison that is defensible here is:

| arm | capability exercised | comparable to |
|---|---|---|
| CLU landscape (items 1–2) | parametric store + physical read | MLP/FFN parametric memory |
| CLU block, GRU, MLP (item 3) | parametric store by sequential gradient updates | each other |
| transformer (item 3) | **parametric** — its weights are what is written | the above; reported *for completeness* |
| transformer, in-context (item 4 only) | **contextual** | nothing in this report |

**No contextual-memory claim is made anywhere from this experiment.** The transformer's strong showing in §3 is a *parametric* result (its weights hold the items), and its `O(K)` read cost below is a *contextual* measurement that has no CLU counterpart because CLU as built has no context window.

### 4.2 Retrieval cost (FLOPs from XLA cost analysis — exact, load-independent)

| read | K=1 | K=2 | K=4 | K=8 | K=16 | scaling |
|---|---|---|---|---|---|---|
| **parametric** mlp | 3.998e5 | 3.998e5 | 3.998e5 | 3.998e5 | 3.998e5 | **flat (exactly)** |
| **parametric** gru | 1.892e5 | … | … | … | 1.892e5 | **flat** |
| **parametric** attention | 4.100e5 | … | … | … | 4.100e5 | **flat** |
| **parametric** clu | 3.893e5 | … | … | … | 3.893e5 | **flat** |
| **contextual** attention (T = 2K+2) | 4.10e5 | 6.23e5 | 1.067e6 | 2.019e6 | **4.207e6** | **grows** |
| **CLU landscape rollout** (1200 steps) | 0.538 ms | 0.535 | 0.550 | 0.528 | 0.539 | **flat, 4.2 % spread** |

- **Contextual attention log-log slope of FLOPs vs K: 0.841 over the full grid, 1.06 between K=8 and K=16.** P16 predicted [0.9, 2.1] — **marginally falsified on the full-range fit**, because at K=1 (T=4) the embedding/head/MLP floor dominates and drags the fit down; the asymptotic slope is the honest number and it is `1.06` (linear in K, i.e. linear in T at these lengths — the `T²` attention term has not yet taken over at T=34).
- **CLU's rollout is O(steps) and K-independent to 4.2 %**, as claimed — now measured, not asserted. ⚠ The comparison is not apples-to-apples in absolute terms: a 1200-step rollout at 0.54 ms is ~6× the wall-clock of a 4.2e6-FLOP attention read, so *K-independence is not the same as cheapness*. What the measurement licenses is the **scaling** statement only.

---

## 5. PREREG scorecard (`PREREG.md` written before the module existed)

| # | prediction | measured | verdict |
|---|---|---|---|
| P1 | spacing gate changes nothing on the w20 ring; admission rate 1.000 | identical to all digits; 40/40 admits | ✅ |
| P2 | `sites_learned_payload` stays destroyed under the spacing gate (strict-A ≤0.10, corruption 2e-2…8e-1) | 0.000; 0.682 ± 0.30 | ✅ |
| P3 | crowded proposals ⇒ ≥35 % refuse-or-relocate; designed gated drift <1e-3 vs ungated >1e-1 | **80 %** relocation; 1.008e-4 vs 4.57e-1 (§1.4) | ✅ |
| P4 | C3 gate truncates ≥80 % of learned writes to <25 % of budget; corruption <1e-2 but strict-B <0.5 | **100 % truncated to 0 %**; corruption 0.000, strict-B 0.000 | ✅ (more extreme than predicted) |
| P5 | anchoring rescues: corruption <5e-2 with strict-B ≥0.9 | **6.6e-3** with strict-B **0.975** | ✅ |
| P6 | first-order drift law within 2× on ≥60 % of designed writes; <60 % on learned | designed half **unmeasured** (item-1 designed rung is a no-op); learned **40 % destructive / 80 % small-perturbation** | ◐ **split — and the split is the physics** |
| P7 | learned+ungated item-1 <0.5 after exactly ONE write, ≤0.2 after | **1.00 → 0.00** at one write | ✅ |
| P8 | designed+gated item-1 ≥0.95 at every K, mean ≥0.90 | **1.00 / 1.00** | ✅ |
| P9 | 8–14 of 16 proposals admitted | **6.0 ± 0.9** | ❌ **falsified — my arithmetic used a box, the code uses a disk** (disk bound 6.1) |
| P10 | learned+gated does not rescue; its benefit is refusal | admits 10, **writes 1** | ✅ |
| P11 | crossover at K=2, gap ≥0.6 by K=4 | K=2; gap 0.70 | ✅ |
| P12 | every primitive forgets: item-1 ≤0.30, mean ≤0.40 at K=16 | item-1 **0.40–1.00**, mean **0.70–1.00** | ❌ **falsified badly** (holds only at K≈64, and not even then for mean) |
| P13 | primitive spread ≤0.35 ≪ the CLU gated/ungated gap ≥0.6 | spread **0.30** @K=16, **0.419** @K=64; gap **0.93** | ◐ claim ✅, registered number ❌ at K=64 |
| P14 | steps-to-criterion rises ≥2× from K=2 to K=16; flat across primitives | rise **0.93–1.26×** @K=16, **0.77–1.82×** @K=64; cross-primitive max/min **2.36** | ❌ **rise falsified — the Head's hypothesis** ; ◐ flatness |
| P15 | joint criterion censored on ≥50 % of (primitive, K≥8) cells | **0.711** extended (K≤64); **0.156** matched (K≤16) | ◐ ✅ extended, ❌ matched — censoring is steep in K |
| P16 | parametric flat for all; contextual attention slope 0.9–2.1; CLU rollout <15 % | flat / **0.841** full-range (1.06 asymptotic) / **4.2 %** | ◐ two ✅, slope marginally ❌ |

---

## 6. How I verified

- **Environment:** `PYTHONPATH=/Users/user/Desktop/CHLU-seqwrite /Users/user/Desktop/CHLU/.venv/bin/python …` — the **main venv reused** per protocol §4 (w6 lesson); **no worktree `uv sync`**, so JAX stays at **0.9.0** (equinox 0.13.4, optax 0.2.6).
- **Full suite: `533 passed` in 509.88 s** (`pytest tests/ -q -p no:randomly --no-cov`), including the **18 new tests** in `tests/test_sequential_write.py`. `ruff check chlu/ tests/` → **All checks passed.** (⚠ `ruff format --check` is *not* clean on this repo at baseline — 42 files on `main` — so I formatted only my three new files and touched no shared formatting.)
- **`python -m chlu.experiments.exp_sequential_write --quick` → exit 0**; `chlu exp-sequential-write --help` parses (via `PYTHONPATH` — `uv run chlu` is broken on this machine, §7.12).
- **Every number in this report is generated from the results JSON by** `.claude/scratch/sequential-write-interference/make_tables.py`, **not transcribed by hand.**
- **Artifacts** in `.claude/outputs/sequential-write-interference/`: `PREREG.md` (written before the module existed), `exp_sequential_write_metrics.json` (all four items merged), `sequential_write_fig1_retention.png`. Drivers/scratch in `.claude/scratch/sequential-write-interference/` (`make_tables.py`, `merge.py`, `run1.py`, `rerun12.py`, `run3.py`, `run4.py`, logs).
- **Reproducibility:** item 1 was executed twice in independent processes (before and after an output-field addition, under different machine load). **Maximum absolute difference over 120 cells × 5 quantities: 0.0.**
- **Timing integrity:** item 4 is the only wall-clock-sensitive measurement and was run in a single process; FLOPs come from **XLA's own cost analysis** (exact, load-independent), and the FLOP columns — not the wall-clock — carry the scaling claims. Machine load 17–21 throughout (shared with other agents).

### Bugs I found by running it (each would have produced a wrong number)

1. ⭐ **The blank controls were scored against the zeros they were written with.** An *empty* designed store therefore reported `blank mean_strict = 1.000` — a tautology, not a control, and I would have published "designed+gated retains 1.000" with a blank that also said 1.000. Fixed: the blank writes zeros and is scored against the **real** codebook; all 29 blank cells now read **0.000**. (w20's method finding, re-earned the hard way.)
2. **Item 1 had no blank control at all.** Added, one per `(rung, proposal, arm)`; all 24 read 0.000.
3. ⭐ **A codebook containing an exact 0 makes the blank control vacuous.** A blank store reads ~0, so any codeword within the value tolerance of 0 is "retained" by nothing at all — measured at K=5, where `designed_payloads` puts an exact 0 in the grid: **blank item-1 retention 1.000 on an empty landscape.** The two K values this report uses (4, 16) clear it by 0.333 and 0.067 against tolerances 0.100 and 0.047; `blank_is_informative` now records this and a test pins it.
4. **The value tolerance was wider than half the codebook spacing at K=16** (0.100 vs spacing 0.133), making "the stored value came back" ambiguous between neighbouring codewords. Capped at 0.35 × spacing (w20's *ratio* at K≤8), which is strictly stricter and so cannot manufacture a positive.
5. **`n_admitted` is not `n_written`.** The C3 gate admits a *site* and can then refuse the *write*. Reporting only `n_admitted` made `learned_gated`'s flat 1.000 look like a gate success; it is 10 admitted and **1 written**. Both counts are now recorded and the figure legend carries them.
6. **The sequential arm was originally run on the w20 rung** (`sites_learned_payload`), whose designed K-well ring at radius `f` fights every off-ring site — first-write retention was 0.56–0.62 instead of 1.000, which would have been read as "learning cannot even store one item". Switched to `free_mlp`, where the first write scores 1.000.
7. **`forward_flops` wants a batched input**; passing a single sequence silently returned `NaN`, i.e. the entire item-4 FLOP table would have been empty in exactly the place the task demands a number (the same trap `primitive-harness` §6.2 hit, in a new place).
8. **The extended sweep initially inherited the K=16 LR with no rescue**, on the table that is the deliverable. Given its own symmetric monotone rescue (all three LRs, full length, all four primitives, adopt only on the full 5-seed mean).

---

## Git footprint

- **Branch** `agent/experiment-engineer/sequential-write-interference`, off local `main` @ `31c3e15`. **Rebase onto `main` = no-op** ("up to date"); `origin/main` untouched (§7.21 — never invoked).
- **Worktree** `../CHLU-seqwrite` (protocol §3.2). The main checkout was clean and on `main`, but w21 is a parallel wave and `chlu/config.py` + `chlu/cli/experiment_cmd.py` are the classic collision files, so I never touched the shared checkout. **No collision observed.** Branch ref verified from the main repo (w4 lesson): `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/sequential-write-interference` lists all 8 commits. Worktree left in place for review; remove with `git worktree remove ../CHLU-seqwrite`.
- **Commits (8):** `505361e` admission gate + atom store · `ac71954` experiment + config + CLI · `08b3548` tests · `49fffdc` full-resolution compute-to-criterion + extended-K sweep · `4621801` **blank-control fixes** · `0fe954b` extended-K symmetric rescue · `941c9f5` `n_learned_params` per cell · `2be2a9f` figure fix.
- **Files — all new except three surgical additions.** New: `chlu/core/admission.py`, `chlu/experiments/exp_sequential_write.py`, `tests/test_sequential_write.py`. Modified: `chlu/core/memory_potentials.py` (**appended** `AtomDictionaryPotential`; existing classes untouched), `chlu/config.py` (+1 dataclass at the **three** required registration sites), `chlu/cli/experiment_cmd.py` (+parser +`cmd_exp_sequential_write`). **`utils/plotting.py`, `chlu/core/blocks.py`, `chlu/experiments/exp_primitive_harness.py`, `chlu/experiments/exp_learned_memory.py` and all shared physics/training code untouched** — the harness is *imported*, per the task's "reuse it, do not rebuild". `results/` deliberately not committed.
- **Commands run:** `pytest tests/ -q -p no:randomly --no-cov` → 533 passed; `ruff check chlu/ tests/` → clean; `python -m chlu.experiments.exp_sequential_write [--quick] [--items 1 2 3 4]`.

## Open questions / follow-ups / risks

1. ⭐ **The strongest result in this report rests on a landscape family only ~120 lines old.** `AtomDictionaryPotential` is new, and the `designed_gated` 1.000-at-every-K curve is its output. It is *sanity-checked* against the theorist (§1.4: `1.008e-4` vs their `8.0e-5` at `d = d_safe`, independent implementation) and against its own blank control (0.000), but a second pair of eyes on that class is the cheapest way to de-risk the wave's headline.
2. **The learned arms are untuned.** One write configuration, chosen a priori and inherited from w20. Their failures are honest baselines, not proven irreducible — in particular `learned_anchored` (rehearsal) plateaus at mean 0.48 at K=16 and might do better with more steps or a capacity-matched MLP. **The gate/no-gate conclusion does not depend on this** (it is a comparison at *fixed* write configuration), but the "learning cannot do it" reading does.
3. **`δ_budget = 0.10` is doing a lot of work in the C3 arm, and the arm is saturated.** The very first 25-step chunk exceeds it on 100 % of cells, so I measured "refuse everything", not a trade-off curve. **A budget sweep (`δ_budget` × `c3_chunk_steps`) is the obvious cheap follow-up** and would turn a binary into a Pareto front: how much of a learned write can you commit per unit of stored-item damage?
4. **The cross-primitive task is small** (vocab 128, 4-token keys, 40 k block params, ≤200 steps/item, ≤64 items). Conclusions are about this regime. In particular the *ranking* at K=64 (gru > mlp > attention > clu) is at a point where three of the four have already collapsed and might not be stable; the *shapes* (attention holds longest, clu breaks first) are the robust part.
5. **The key→value dataset gives every primitive a partially local parameter store for free** — the embedding table has one row per token, so a write touches only its key's 4 rows plus the shared blocks. This is a genuine confound for "how local is the write operator", and it flatters every primitive relative to a dense-input task. Worth re-running with continuous (dense) keys.
6. **`d_safe` on a learned landscape is a guess.** There is no atom width to point at, so I used the write objective's `σ_addr = 0.25`. If a learned write has an effective width much larger than `σ_addr`, the honest `d_safe` is larger and the gate would fire on the w20 ring after all. ⚠ **This is the one assumption that could overturn §1's headline**, and it is testable: measure the actual spatial extent of `δV` from a learned write and set `d_safe` from that. I did not.
7. **`learned_gated` never got to test relocation properly at K=16**, because the C3 check refused before placement mattered. The spacing gate's *placement* value on a learned landscape is therefore measured only in item 1's disk arm (5 % corruption reduction), on a single subsequent write.
8. **No CLU arm exercises contextual memory anywhere**, so nothing here bears on the attention-vs-CLU comparison that a reviewer will actually care about. §4.1 says so; it remains the largest gap between what we can measure and what the claim would need.

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry (this supersedes the Hub's w21 framing of the gate).** *The MVC-0 admission gate has now been run on the exact w20 failing setup.* **The spacing gate cannot fire on the w20 ring geometry** (spacing 1.4142 vs `d_safe = 4.4·σ_addr = 1.10`) and gated ≡ ungated to all digits (corruption `0.682 ± 0.30`, strict-A `1.000 → 0.000`). Where it *is* made to fire (crowded proposals, 80 % relocation rate) it reduces corruption by **5 %** and the item is still destroyed. The **C3 check** works — and its only available output is **refusal** (0 of 300 steps committed on 100 % of learned cells; item preserved, new item not stored). **The theorist's `8.0e-5` is confirmed** (independent reproduction: `1.008e-4` at `d = d_safe`) **but it is a property of the LOCAL ATOM WRITE, not of the gate**: the drift law is `exp(−d²/2s²)`, measured over 8 distances.
2. **§6 — the positive result.** *An admission gate does protect a **designed** store, decisively:* sequential K=1..16, same local write operator, only placement differs — **designed+gated item-1 retention `1.000` at every K and mean retention `1.000`, vs designed+ungated `0.16` / `0.11`** — at the cost of admitting `6.0 ± 0.9` of 16 proposals (which matches the disk packing bound 6.1 to 2 %). **Refusal is the mechanism, and it has a price in capacity that must always be quoted with the retention number.**
3. **§7 — new known issue (methodological, high priority, program-wide).** *A blank control must be scored against the **real** codebook, never against the zeros it was written with* — the tautological version reports **1.000 for an empty landscape**. *And a codebook containing a value within the read tolerance of 0 makes any blank control vacuous* (measured at K=5: blank item-1 retention 1.000 on nothing stored). Both are now guarded in code (`blank_is_informative`) and tested. This is the second wave running in which the blank-control protocol has had to be strengthened.
4. **§7 — new known issue (scientific).** *The value-recovery tolerance must be capped at a fraction of the codebook spacing.* At K=16 on `[-1,1]` the w20 absolute tolerance 0.1 is 0.75 × the spacing, so "the stored value came back" is ambiguous between neighbouring codewords. `payload_tol_frac = 0.35` (w20's ratio at K≤8) is now the default.
5. **Claims-matrix / paper-text warnings — three, all needing an owner.**
   (a) "An admission gate suppresses interference by ~5 orders" is **designed-landscape-only**; on a learned landscape the measured suppressions are spacing gate **1.00×**, C3 gate **∞ at zero items stored**, structured (anchored) write **103×**.
   (b) The Head's **"wasted compute reorganising to conserve key information" is refuted as stated** — compute-to-criterion is flat in K for all four primitives (growth `0.77–1.82×` over K=2→64). The correct replacement claim is *infeasibility*: joint-criterion censoring `0.00 → 1.00` between K=8 and K=32.
   (c) `primitive-harness`'s **"CLU degrades more gracefully than the GRU"** is a *contextual* MQAR result and **does not generalise**: under parametric sequential writes CLU is the **worst** of mlp/gru/attention/clu at K=64 (mean retention 0.16 vs 0.57/0.43/0.34).
6. **New CLI/config surface:** `chlu exp-sequential-write [--project N] [--seed I] [--quick] [--items 1 2 3 4]`; `ExperimentSequentialWriteConfig` (registered at all three config sites). Two defaults carry reasoning and should not be silently changed: `sequential_rung = "free_mlp"` and `payload_tol_frac = 0.35`. New module `chlu/core/admission.py` (the MVC-0 gate; non-learned, reusable) and `AtomDictionaryPotential` in `chlu/core/memory_potentials.py` (MVC-0 §4's designed store — **the first landscape in the repo that supports writing one more item**).
7. **For the theorist.** Two things worth a proposition. (i) *The spacing gate is a certificate on the argument of the write operator's kernel; it is informative iff the operator has a kernel that decays.* A necessary condition to state formally: the gate can bound stored-item damage only if `‖∂_θV(q*_i)·δθ‖` admits a factor decaying in `d(q_new, q*_i)`, which a global-support `V_θ` does not have. (ii) The C3 first-order law is **conservative in the wrong direction** where it matters: it *under*-predicts measured drift by ~4× on destructive writes (median ratio 0.269) while tracking within 2× on small ones (0.68–0.81, 80 % of cells). A controller using it as a safety budget is safe only because the budget is crossed long before the law breaks — that should be stated as a condition, not assumed.
8. **Requested next task (from §6/§7 above):** a **`δ_budget` × `c3_chunk_steps` sweep** turning the C3 arm's binary refusal into a Pareto front (stored-item damage vs fraction of the new write committed), and a measurement of the **effective spatial extent of a learned write** so `d_safe` on a learned landscape stops being a guess (risk 6).
