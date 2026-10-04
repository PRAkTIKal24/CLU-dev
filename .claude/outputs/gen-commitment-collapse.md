# gen-commitment-collapse — experiment-engineer report

**Task + acceptance criterion:** measure distinct codes on the 72 held-out (B) requests vs **stopping rule ·
multiplicity · launch temperature**, with the matched-seen (MS) companion on every number —
**PASS = ≥60/72 distinct on B AND the store→decoder arm within seed spread of `knn_proto` (0.8817 ± 0.0023).**
**Status: done.** ⛔ **VERDICT: FAIL, on all four pre-registered configs, 3/3 seeds. Pre-registered as a FAIL (A5)
and measured as one.** Per R29 that routes to new approaches, ⛔ not to abandonment and ⛔ not to re-tuning.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary)
> 1. ⛔⛔ **The task's (and R30's) stated MECHANISM is wrong and must not be written down.** *"Montero's cell is a
>    colour corner, so the three hue factors contribute the same large log-conditional to every one of the 72
>    requests"* — **the hues are NOT constant on B**: each takes **2 values** (the corner is `hue ≥ 0.8`, i.e. two
>    grid values per hue, 2·2·2·3·1·3 = 72). ⭐ The true mechanism is the **opposite**: the hues are the only thing
>    the code tracks. Measured `code_flip_rate` on B (3-seed mean): floor 0.98 · wall 1.00 · object 0.56 · **scale
>    0.20 · shape 0.00 · orientation 0.13**. **The committed code is ≈ the hue triple (8 combinations) and almost
>    nothing else.** (§4)
> 2. ⭐⭐ **R29 Q2's rider, ANSWERED** (it was the reason this spoke had to persist tuples): **|B ∩ A| = 0 of 9 / 0 of
>    8 / 0 of 10** and **|B ∩ MS| = 9 of 9 / 7 of 8 / 10 of 10**. ⇒ **every code the store gives to a held-out
>    request is a code it also gives to a SEEN request in the same corner, and NONE of them is a code it uses
>    anywhere else in the factor space.** ⭐ R30's "the corner collapses, not the hold-out" is now a *set identity*,
>    not an inference from two rates. (§6)
> 3. ⚠ **"9 of 72" is a two-sided statement and the second side is new:** the collapse is **not** a shortage of
>    channel. The same store, same requests, at **160.2 bits** (k = 8, 5.34× the design constant) renders **70.3 ±
>    1.2 distinct** B codes and **still** scores `scale` **0.000** and `orientation` **0.194**. ⭐ **Distinctness and
>    correctness are separable and we have now separated them.** (§5, §7)
> 4. ⚠ **The `argmax` commitment is provably temperature-blind** (softmax is monotone) — measured bit-identical
>    across τ ∈ [0.25, 8] on A, B and MS. ⛔ Any future task or draft that lists "launch temperature" as a knob on
>    the *shipped* read is listing a no-op. It acts only through `sample`/`margin`. (§5, P0)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *commitment* — how a request's posterior over modes becomes an integer code.
- **Laundering control:** ⭐ **the matched-seen (MS) corner, on every single number** (66 sweep configs × 3 seeds +
  4 scored configs × 3 seeds). A knob that lifts B while MS stays flat is a **B-specific artefact**.
  ⭐ **Result: no such knob exists here** — over the 66 configs, Pearson *r*(B distinct fraction, MS distinct
  fraction) = **0.9907**, max (B−MS) fraction gap = **+0.096**, and **0 configs** exceed +0.10. Every knob moves
  the **corner**, ⛔ never the hold-out alone.
- **Falsifies (R29 pivot item 1):** ≥60/72 distinct on B **and** ≥ 0.8794 in-corner `shape` balanced accuracy.
  ⛔ **Not reached. FAIL.**
- **Does NOT falsify:** losing on *level* to the unbounded-channel prototype arm (R23.5's standing reading), or a
  distinct count that rises together with MS — that is the corner moving, which is the honest direction.

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | `d5d406d` (branch tip; every number below produced at this tree — the last code edit, `factor_leverage`, landed **before** the 3-seed commit run and before any accuracy run) |
| branch / base | `agent/experiment-engineer/gen-commitment-collapse` off **`main @ 5fee3f4`** (the task's base; rebase onto `main` is a no-op, `main` did not move) |
| worktree / venv | `../CHLU-gen-collapse`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero) |
| seeds | **0, 1, 2** (3 seeds, spread on every number) |
| decoder | `decoder_source="winner"`, recipe **`v1_capacity_schedule`**, hashes `01ef91f75eea…` / `001011bc603a…` / `2d7eea75add7…` — **asserted per seed in BOTH sub-stages** (⛔ `dec_recipe` still defaults to `"baseline"`, the recipe that failed the gate) |
| store | ⭐ **`gen-read-compose`'s own per-seed checkpoints, loaded, never retrained** — hashes `3e3424a1ea40…` / `76827ad534ee…` / `8449e6057dac…`; `K=5`, `n_modes=64`, `D=2048`, **30 bits/read**, 524,288 B |
| grouping | `grouping_source="per_seed"` (R26.5); seed 0 re-derivation asserted **bit-identical** to `store_plan.json`; store well sizes re-checked against it on every seed |
| addressing | `address_smoothing = 1.0`; requests = `requests_from_labels(..., protocol="full", n_drop=0)` |
| rows | `store_train` 32,768 · A **512** · B **72** · MS **198 / 199 / 199** |
| sweep grid | τ ∈ {0.25, 0.5, 1, 2, 4, 8} × rules {argmax, sample, margin, topk} × δ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} × k ∈ {2, 3, 4, 8} = **66 configs**; `sample_draws=3`, `sample_seed=20260913`, `topk_weight="posterior"` |
| accuracy leg | **4 pre-registered configs only**, `eval_batch = 256`, positives = B renders, negatives = MS renders, statistic = **in-corner `shape` balanced accuracy** (R20/R23), AUROC as mandatory companion |
| ⛔ NOT used | `soft_read`, any feature tensor on the arm path, `DreamLandscape`, `dream_rollout`, `clamp_from_posterior`, `fit_code_coupling` |
| wall clock | commit stage **3.4 s / 84.9 s / 88.2 s** (seed 0's cache is `scope="full"`; seeds 1–2 re-encode 32,199 rows through the frozen encoder) · accuracy leg **~54 s per seed for all 4 configs**. CPU only |
| ⛔ CSF3 | unreachable (2FA) — local CPU only, as the task states |

---

## 1. ⭐ The plumbing check nobody asked for, and it is exact
The shipped baseline (`argmax|tau=1`) was re-measured end to end through this new harness, from
`gen-read-compose`'s store checkpoints:

| quantity | this spoke (seeds 0/1/2) | `gen-read-compose` (R28) | verdict |
|---|---|---|---|
| distinct codes A / B / MS | 494,493,498 · **9,8,10** · 35,24,24 | 494,493,498 · 9,8,10 · 35,24,24 | ⭐ **identical** |
| in-corner `shape` balacc | 0.6610 / 0.6809 / 0.6858 | 0.6610 / 0.6809 / 0.6858 | ⭐ **identical to 4 dp** |
| mean top-1 confidence, B | 0.8560 / 0.8482 / 0.7936 | 0.8560 / 0.8482 / 0.7936 | ⭐ **identical** |
| A aggregate | 0.6174 ± 0.0070 | 0.6174 ± 0.0070 | ⭐ **identical** |

⇒ **P4a/A1 HELD.** The commitment rule is the only thing that changes between R28's numbers and everything below.

## 2. ⭐ LEG 1 — the diagnosis: the posterior is NOT degenerate, so the sweep was allowed to run

| set | mean top-1 | mean top1−top2 margin | entropy bits/well (max 6) | degenerate? | argmax distinct |
|---|---|---|---|---|---|
| **A** | 0.7563 ± 0.0053 | 0.6032 ± 0.0080 | 1.0922 ± 0.0231 | **no** | 494/493/498 of 512 |
| **B** | 0.8326 ± 0.0278 | 0.7384 ± 0.0470 | 0.9117 ± 0.0797 | **no** | **9/8/10 of 72** |
| **MS** | 0.7655 ± 0.0263 | 0.6288 ± 0.0365 | 1.1277 ± 0.1162 | **no** | 35/24/24 of ~198 |

⭐ **P1 HELD on all three clauses** (top-1 ∈ [0.78, 0.88]; entropy ∈ [0.5, 1.5] bits; margin > 0.60). No well has
top-1 > 0.99 (**0.0 %** on B), so the posterior retains real mass on the runner-up: **a stopping rule *could*
move the code, and the sweep was not a foregone conclusion.** ⚠ The mass is there; the **ordering** is what does
not move (§4).

## 3. ⛔ LEG 1b — the task's mechanical hypothesis: CONFIRM/REFUTE, and it REFUTES

| prediction (PREREG P2) | measured | verdict |
|---|---|---|
| **P2a** the 3 hues supply ≥60 % of B's absolute per-well evidence | **66.4 %** (0.213 + 0.274 + 0.177) | ⭐ **HELD** |
| **P2b** the hue deltas have **zero** across-request variance on B | **0.412 / 0.464 / 0.263** — ⛔ **not constant.** The only zero-variance factor is `shape` (one class), share 0.124 | ⛔ **FALSIFIED** |
| **P2c** dropping the 3 hues from the B request raises distinct codes 9 → ≥40 | **9 / 9 / 9** — ⛔ *exactly* the baseline | ⛔ **FALSIFIED** |

⭐ **Why P2c is the interesting failure.** Removing 66 % of the evidence — the entire colour corner — changes the
number of distinct B codes by **zero** (A drops 494 → 275/281/292, so the counterfactual is live and does bite
where there is information to lose). **The corner's hue evidence is not what is drowning the other factors; there
is nothing to drown.**

## 4. ⭐⭐ LEG 1c — the actual mechanism, from `factor_leverage` (3-seed means)
Edit one factor of a request to another value the set realises; does the committed K-tuple move at all?

| set | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| **A** flip rate | 1.00 | 1.00 | 0.96 | 0.71 | 0.79 | **0.83** |
| **A** codes reachable alone | 5.3 | 10.0 | 6.7 | 2.3 | 1.7 | **5.7** |
| **B** flip rate | 0.98 | 1.00 | 0.56 | **0.20** | **0.00** | **0.13** |
| **B** codes reachable alone | 2.0 | 2.0 | 1.7 | **1.3** | 1.0 | **1.3** |
| **MS** flip rate | 0.99 | 1.00 | 0.63 | **0.28** | 0.84 | **0.17** |
| evidence share (B) | 0.213 | 0.274 | 0.177 | 0.075 | 0.124 | 0.137 |

⭐ **`orientation` carries 13.7 % of B's evidence and moves the code 13 % of the time; on A it moves it 83 % of the
time and reaches 5.7 codes by itself.** So the address book **did** learn orientation — ⛔ **it is outvoted inside
the corner, not missing.** Same story for `scale` (7.5 % share, flip 0.20 on B vs 0.71 on A).
⭐ This is exactly the distinction `factor_leverage` was built to make (and its unit test proves it can make it):
**"never learned" and "outvoted" have different fixes, and this is the second.**

⚠ **The sharpest single sentence, from the persisted tuples** (`determinants.json`, seed 0): on B the committed
code is a **deterministic function of (floor_hue, wall_hue, object_hue, scale)** and `orientation` is **ignored
entirely**; grouping B's requests by the hue triple alone gives **8 groups**, and 8 of the 9 codes are constant
within them. ⇒ **the 30-bit read transmits the hue corner and, at most, one extra bit.**

## 5. ⭐ LEG 2 — the sweep: 66 configs × 3 seeds, ⛔ every row with its MS companion
Full table: `tables.md` §4 · raw: `runs/shapes3d_recon/seed_*/sweep_s*.json`. The rows that matter:

| config | B / 72 | MS / ~198 | A / 512 | realised width | abstain (B) |
|---|---|---|---|---|---|
| `argmax` τ = 0.25 … 8 (**6 rows**) | **9.0 ± 0.8** (all six identical) | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `sample` τ = 0.25 | 10.3 ± 1.7 | 35.7 ± 4.5 | 497.3 ± 2.6 | 30.0 bits | 0.00 |
| `sample` τ = 1 | 38.7 ± 0.9 | 120.3 ± 3.3 | 505.7 ± 0.5 | 30.0 bits | 0.00 |
| ⭐ `sample` τ = 2 | **71.7 ± 0.5** | 198.0 ± 0.8 | 512.0 ± 0.0 | 30.0 bits | 0.00 |
| `margin` τ = 1, δ = 0.75 | 17.7 ± 4.8 | 29.7 ± 6.8 | 365.3 ± 9.9 | 30.1 bits ⛔ | 0.37 |
| ⭐ `margin` τ = 2, δ = 0.25 (**sweep max, ties δ=0.75**) | **17.7 ± 5.0** | 34.3 ± 10.1 | 393.0 ± 7.8 | 30.1 bits ⛔ | 0.34 |
| `margin` τ ≥ 4 (most δ) | **1.0 ± 0.0** (total collapse) | 1.0 | 1.0 | 30.1 bits ⛔ | 1.00 |
| `topk` k = 2 (τ-invariant) | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | ⛔ **54.9 bits** | 0.00 |
| `topk` k = 4 | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | ⛔ **96.4 bits** | 0.00 |
| ⭐ `topk` k = 8 | **70.3 ± 1.2** | 193.3 ± 3.1 | 512.0 ± 0.0 | ⛔ **160.2 bits** | 0.00 |

- ⭐ **P0 HELD, measured not assumed:** under `argmax` the codes are **bit-identical for every τ** on A, B and MS.
- ⭐ **P3 scorecard:** temperature-on-argmax **HELD** (9 at every τ) · `sample` τ = 1 predicted ≥55, **measured
  38.7 ⇒ FALSIFIED** (τ = 2 is where it saturates) · `margin` predicted ≤30, **measured max 23 ⇒ HELD** ·
  `topk` k = 2 predicted 15–40, **measured 32.7 ⇒ HELD**; k = 4 predicted 30–60, **measured 50.7 ⇒ HELD**.
- ⛔ **Realised width is reported for every config** and **every multiplicity row exceeds R20's declared 30-bit
  design constant**: k = 2 → 54.9 bits, k = 3 → 76.7, k = 4 → 96.4, k = 8 → **160.2 bits = 5.34×**. ⭐ Those rows
  are a **declared change to the claim**, not a free win. `margin` is 30.11 bits (M + 1 symbols per well: the
  abstain symbol is real information the decoder acts on) and is flagged for the same reason, honestly but
  trivially.

## 6. ⭐ LEG 4 — the persisted tuples, and R29 Q2's owed answer
Artefacts: `codes_{A,B,MS}_seed{0,1,2}.json` (per-request **request vector + integer code tuple**, 9 files).

| pair | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **\|B ∩ A\|** | **0** of 9 | **0** of 8 | **0** of 10 |
| **\|B ∩ MS\|** | **9** of 9 | **7** of 8 | **10** of 10 |
| **\|MS ∩ A\|** | **0** of 35 | **0** of 24 | **0** of 24 |

⭐ **P5 HELD in the MS direction and FALSIFIED in the A direction** (predicted ≥5 of 9 in A; measured **0**).
⇒ The corner's codes are a **disjoint region of the code space**: the store answers held-out requests with codes
it *does* use — but only ever for **seen requests in the same corner**. ⛔ "Are the 9 codes seen configurations?"
**Yes, and only the corner's.**

## 7. ⭐⭐ LEG 3 — the accuracy leg at the pre-registered shortlist (⛔ both numbers or neither)
`PREREG.md` filed **17:42**; its shortlist **addendum at 17:48:20**, after `sweep_s0.json` (**17:47:22**) and
**before** the first accuracy output `scores_s0.json` (**17:51:30**) — mtime-verifiable, in that order.
⚠ **Declared:** the shortlist was fixed on **seed 0's** sweep (seeds 1–2 had not finished). Checked post-hoc
against the 3-seed means, each entry is still its knob's maximum **up to ties**: `sample` τ = 2 (71.7) ties τ = 4
(72.0 ± 0.0) and τ = 8; `margin` τ = 2, δ = 0.25 (17.7 ± 5.0) ties τ = 1, δ = 0.75 (17.7 ± 4.8); `topk` k = 8 is
the outright maximum. ⛔ No entry would have been displaced by a materially better config.

| config | B distinct | MS distinct | **in-corner `shape` balacc** | AUROC | A aggregate | width | **PASS?** |
|---|---|---|---|---|---|---|---|
| `argmax\|tau=1` (**shipped**) | 9.0 ± 0.8 | 27.7 ± 5.2 | **0.6759 ± 0.0107** | 0.7429 ± 0.0080 | 0.6174 ± 0.0070 | 30.0 bits | ⛔ FAIL (distinct) |
| `margin\|tau=2\|δ=0.25` | 17.7 ± 5.0 | 34.3 ± 10.1 | **0.5872 ± 0.0445** | 0.6583 ± 0.0465 | 0.3735 ± 0.0151 | 30.1 ⛔ | ⛔ FAIL (both) |
| ⭐ `sample\|tau=2` | **71.7 ± 0.5** | 198.0 ± 0.8 | **0.6319 ± 0.0332** | 0.6817 ± 0.0437 | 0.4368 ± 0.0066 | 30.0 bits | ⛔ **FAIL (accuracy)** |
| ⭐ `topk\|k=8` | **70.3 ± 1.2** | 193.3 ± 3.1 | **0.7408 ± 0.0323** | 0.7910 ± 0.0107 | 0.6603 ± 0.0105 | 160.2 ⛔ | ⛔ **FAIL (accuracy)** |
| *target* `proto`/`knn_proto` | 72 / 72 | — | *0.8817 ± 0.0023* | *0.9908* | *0.8104* | unbounded | — |

### ⭐ The two results the task was shaped to catch, and both fired
1. ⭐⭐ **"Distinct count rises and accuracy falls" — CONFIRMED.** `sample` τ = 2 takes B from 9.0 → **71.7 of 72**
   (it clears the ≥60 floor) while balanced accuracy **falls** 0.6759 → **0.6319** (−0.0440) and A aggregate
   **falls 0.6174 → 0.4368** (−0.18). ⭐ **That is noise, priced.** A2 predicted < 0.60; measured 0.6319 ⇒ the
   *direction* held, the *level* was pre-registered too pessimistically — recorded as a **split**.
2. ⭐ **The best commitment here needs 5.34× the channel and still loses by 0.14.** `topk` k = 8 is the only config
   that beats the shipped baseline (**+0.0649**, 3/3 seeds) and it does so at **160.2 bits**. It lands at
   **0.7408 ± 0.0323** — statistically indistinguishable from `gen-read-compose`'s `clu_wide` (0.7484 ± 0.0307,
   the full 320-weight mixture), i.e. ⭐ **k = 8 of 64 modes already recovers the whole soft read**, and the full
   soft read was already measured to lose to `knn_proto`. **A4 HELD** (predicted [0.70, 0.80]).
3. ⛔ `margin`+abstain is the one knob that is **worse on both axes at once** (its B distinct is the sweep's
   *third*-best, 17.7 ± 5.0, and it costs 0.09 balanced accuracy) (0.5872, A aggregate 0.3735):
   abstaining paints the well with its mean mode, which is the deletion control by another name. **A3
   FALSIFIED** (predicted ±0.05 of baseline; measured −0.089).

### ⭐ The number that settles the "is 9-of-72 a capacity limit?" question (R29 Q2's framing)
Per-factor probe accuracy on B, 3-seed mean:

| config | floor_hue | wall_hue | object_hue | **scale** | **orientation** | shape |
|---|---|---|---|---|---|---|
| `argmax` (9 codes) | 0.991 | 1.000 | 0.671 | **0.019** | **0.194** | 0.833 |
| `topk` k = 8 (**70 codes, 160 bits**) | 0.995 | 1.000 | 0.954 | **0.000** | **0.194** | 0.940 |
| `sample` τ = 2 (**72 codes**) | 0.648 | 0.593 | 0.398 | 0.125 | 0.144 | 0.630 |
| *decoder ceiling* | *1.000* | *1.000* | *1.000* | *0.958* | *1.000* | *0.940* |

⭐⭐ **Going from 9 distinct codes to 70 buys `object_hue` (+0.28) and `shape` (+0.11) and buys `scale` and
`orientation` NOTHING (0.000 and 0.194, unchanged).** ⇒ **R29's reading is confirmed and sharpened: 30 bits was
never the binding constraint — but neither is the commitment rule.** The store's modes do not *contain* this
corner's scale and orientation, so no rule that points at them can emit them. ⛔ The fix is upstream of the
commitment: in what the wells are fitted to represent.

## 8. How I verified (commands + real output)
```
# targeted tests, main venv, worktree:
tests/test_gen_compose.py .............................35 passed in 6.81s   (22 pre-existing + 13 new)
tests/{test_gen_compose,test_config,test_gen_dream,test_gen_store}.py → 107 passed in 218.89s
# ⛔ NO full suite started (task item 8; 16 GB machine, concurrent spokes)
ruff check chlu/core/gen_compose.py chlu/experiments/exp_gen_compose.py chlu/config.py → All checks passed!
# the run, per seed (both sub-stages assert the decoder):
⭐ decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b… (hash + recipe asserted)
⭐ store loaded from …/gen-read-compose/…/store_s0.pkl (hash 3e3424a1ea40017e…, K=5, M=64, 30 bits/read)
⭐ request-only path asserted (no feature argument on ANY arm; soft_read not imported)
  [B] top1=0.8560 margin(top1−top2)=0.7753 entropy=0.857/6.0 bits per well ⇒ not degenerate
  [B] ⭐ factor leverage: floor_hue=1.00|2 wall_hue=1.00|2 object_hue=1.00|2 scale=0.08|1 shape=0.00|1 orientation=0.00|1
⭐ hue-dropped counterfactual: A=275/512 B=9/72 MS=22/198
⭐ |B ∩ A| = 0 of 9 B codes · |B ∩ MS| = 9 of 9
  [topk|tau=1|margin=0|k=8] B distinct=72/72 (MS 189/198) · in-corner shape balacc=0.7172 … width=160.2 bits ⛔ EXCEEDS 30 ⇒ FAIL
```
- ⛔ **The no-bypass guarantee was re-asserted on the new path**, not inherited: `assert_request_only` passes on
  every seed; the pure-pointer rules route through the **shipped** `read_from_codes` (a unit test asserts
  bit-equality with the weighted path's inputs), and no arm takes a feature tensor. The A/B/MS feature tensors are
  **not even gathered** by this stage.
- ⚠ **One test of mine failed first and I fixed the test, not the code:** `test_factor_leverage_…` initially
  asserted a flat factor reaches 0 codes alone; it reaches **1** (a factor that changes nothing still emits the
  same single code). Recorded because a test that has never failed has never been tested.

## 9. Reproducing
```
PYTHONPATH=<worktree> CHLU_GEN_DATA_ROOT=<repo>/.claude/data/gen CHLU_CLAUDE_ROOT=<repo>/.claude \
CHLU_GEN_BRIDGE_ROOT=<repo>/.claude/outputs/gen-feature-bridge/runs \
  /Users/user/Desktop/CHLU/.venv/bin/python -u -m chlu.experiments.exp_gen_compose \
  --stage commit --dataset shapes3d --pretext recon                       # 66-config sweep + tuples, 3 seeds
  … --stage score --config 'argmax|tau=1|margin=0|k=1' --config 'sample|tau=2|margin=0|k=1' \
    --config 'margin|tau=2|margin=0.25|k=1' --config 'topk|tau=1|margin=0|k=8'
python table.py && python plot.py && python determinants.py     # every table/figure above, from disk
```
**Artefacts** (all under `.claude/outputs/gen-commitment-collapse/`): `PREREG.md` (+ its dated shortlist addendum) ·
`codes_{A,B,MS}_seed{0,1,2}.json` (**the owed tuples**) · `runs/shapes3d_recon/seed_*/commit_manifest_s*.json`,
`sweep_s*.json`, `scores_s*.json`, `commit_cache_s*.npz` · `tables.md` + `table.py` · `determinants.json` +
`determinants.py` · `collapse.png` + `plot.py` · `logs/{commit_all,score_all,commit_s0}.log`.
⛔ **Nothing is NOT RUN.** All 3 seeds × both sub-stages × 66 sweep configs × 4 scored configs completed.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-commitment-collapse`, worktree `../CHLU-gen-collapse`, base
  **`main @ 5fee3f4`** (as the task named). Rebase onto `main`: **no-op, `main` has not moved.** ⭐ Verified from
  the MAIN repo: `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-commitment-collapse`
  shows all three commits. ⚠ **The worktree is left in place** (the protocol's removal step is deliberately not
  taken — the Hub may want to re-run the score stage; it costs one `git worktree remove`).
- **Commits (3):** `f91b6f5` core commitment machinery + `GenCommitConfig` · `46b947a` the `commit`/`score` stages ·
  `d5d406d` 13 new tests.
- **Files touched (4, all inside my declared ownership):** `chlu/core/gen_compose.py` (**append-only**, +387) ·
  `chlu/experiments/exp_gen_compose.py` (+523: one appended section plus **exactly two** in-place lines — the
  `--stage` choices list and `main`'s pre-dispatch — and the `if __name__` block moved to the file tail so the new
  section is defined before `main` dispatches to it; total deletions in this file: **2 lines**) ·
  `chlu/config.py` (**append-only at EOF**, +83, ⛔ **0 deletions**, `# --- BEGIN/END gen-commitment-collapse ---`) ·
  `tests/test_gen_compose.py` (**append-only**, +214).
- ⛔ **Untouched:** `gen_dream.py` (⭐ `fit_address_book`/`address_requests` were *diagnosed*, not edited — see the
  follow-ups), `gen_store.py`, `gen_bridge.py`, `gen_certificate.py`, `gen_request.py`, `chlu/eval/rivals/`,
  `chlu/data/`, `scripts/csf3/`, `chlu/cli/`.
- ⚠ **`chlu/config.py` merge note for the Hub:** my block is a single contiguous region at EOF and **adds no
  `CHLUConfig` field and no `load_config`/`save_config` site** (it is reached through `gen_commit_config()`
  precisely so the tail anchor stays liftable). Merge order a7 → collapse → classifier as R30 says; if the anchor
  conflicts, take the peer's whole file, re-apply my delimited block, assert **0 deletions**.
- **Conflicts:** none. **No push, no PR.**

## Open questions / follow-ups / risks
1. ⭐⭐ **The actionable finding, and it is upstream of everything this spoke owns.** Orientation and scale have
   real conditionals (A: flip 0.83/0.71, 5.7/2.3 codes reachable alone) that are **outvoted inside the corner**
   (B: 0.13/0.20). Three candidate fixes, none tried here, all outside my file ownership or my task: **(a)**
   per-factor evidence weights in `address_requests` (down-weight the 3 hues, which are 66 % of the evidence and
   6 % of what a reader wants), **(b)** ⭐ a *contrastive* commitment — commit on `s − mean(s)` over the request
   set, which is transductive and must be declared as such, **(c)** the real one: a store whose wells are fitted
   so that scale/orientation own modes at all. ⛔ **(c) is the only one that can move `scale` off 0.000, because
   §7 shows no commitment rule can.**
2. ⚠ **`sample` at τ ≥ 2 renders 512/512 distinct A codes and drops A aggregate to 0.44.** If anyone is tempted to
   quote sampling as a diversity mechanism, ⛔ it is measurably a corruption mechanism on this cell.
3. ⚠ **`topk` k = 8's 160.2 bits is my pointer-bits accounting** (`K·log2 C(64,8)`), plus **K·(k−1) = 35 real
   weights** because the launch is posterior-weighted. A uniform-weight k-particle launch would be the same
   pointer bits with no reals; ⛔ not run, not claimed.
4. ⚠ **One dataset, one condition, one pretext, one corner.** Nothing here speaks to the wide condition, to
   dSprites, or to any cell where more than 2 hue values are realised. ⭐ The B cell realising only **2 values per
   hue** is central to §4 and may not generalise.
5. ⚠ **MS is 198 rows on seed 0 and 199 on seeds 1–2** (the store-train draw differs) — inherited from
   `gen-read-compose`, not a defect, but the negatives are not byte-identical across seeds.
6. ⛔ **No kill-rule verdict, no C set, no refusal number** — out of scope.

## Proposed handover updates (for the Hub)
- **§7 NEW (mechanism, binding on every draft):** ⛔ **"The colour corner's hues are constant across the 72 B
  requests" is FALSE** — each hue takes 2 values; the 72 = 2·2·2·3·1·3. The correct statement is **"the committed
  code is ≈ a function of the hue triple, and `orientation` has ~zero leverage inside the corner."** R30's table
  and this task file both carry the wrong version.
- **§7 NEW (mechanism, standing):** **`argmax` commitment is temperature-invariant by construction** (softmax is
  monotone) — verified bit-identical over τ ∈ [0.25, 8]. ⛔ Do not list launch temperature as a knob on the
  shipped read.
- **§7 NEW (instrument):** `gen_compose.factor_leverage` separates **outvoted** from **absent** factors, and
  `commitment_channel` prices any rule against R20's 30-bit constant. Both have tests that fail on purpose.
- **§7 NEW (config):** `GenCommitConfig` lives at the tail of `chlu/config.py` and is reached via
  `gen_commit_config(config)`; it is deliberately **not** a `CHLUConfig` field, so the merge anchor is one
  contiguous block (0 house sites, 0 deletions).
- **Ledger (R29/R30 follow-through):** **the commitment PASS FAILED, 3/3 seeds, on all four configs** · the 9
  codes are **⊂ MS's codes and disjoint from A's** (R29 Q2 answered) · **no B-specific artefact exists** (r = 0.99
  across 66 configs) · **distinctness ≠ correctness**: 70 codes at 160 bits still score `scale` 0.000 · the best
  rule (`topk` k = 8, 0.7408) **ties `clu_wide`** and therefore inherits its loss to `knn_proto`.
