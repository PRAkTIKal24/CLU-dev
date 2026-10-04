# gen-read-compose — experiment-engineer report

**Task + acceptance criterion:** build the cell's composition path `request → read → code → frozen decoder → probe`
(R26 ruling 3) and report A and B on it, four arms, ≥3 seeds — **acceptance is that the read path beats
`store_removed` on `shape`**.
**Status: done.** ⭐ **The falsifier did NOT fire: `read − store_removed = +0.1759 ± 0.0107` in-corner `shape`
balanced accuracy, 3/3 seeds, 16.4× the largest arm spread.** ⛔ No kill-rule verdict is rendered here.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 7 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⭐ **Both ceilings reproduce R22's to four decimal places, independently**: decoder **0.9414 ± 0.0179**
>    (R22: 0.9414 ± 0.0179) and store **0.6866 ± 0.0681** (R22: 0.6866 ± 0.0681), from a store re-fitted here
>    from scratch. ⇒ **the store ceiling stops being a quoted number and becomes a measured one in this cell.**
>    ⚠ Stated precisely: I did not compare checkpoints, so this is a four-decimal numeric match, not a proof of
>    bit-identity — but the fit is deterministic given (seed, grouping, cache), so it is the strongest plumbing
>    evidence available and it was not designed as a check. (§4)
> 2. ⭐ **The 30-bit read is statistically indistinguishable from the store ceiling** (`read − store ceiling =
>    −0.0107 ± 0.0576`, **0.2× spread**, 1/3 seeds positive). ⇒ **addressing from the REQUEST costs nothing
>    measurable against addressing from the image's own features.** The binding constraint on this path is
>    **store capacity**, not addressing — which is the axis R23.5 deferred, and this cell now has a number for it. (§5)
> 3. ⭐ **`CLU-wide` does NOT tie `knn_proto`** (`−0.1400 ± 0.0307`, 4.6× spread, **0/3**). R27's pre-registered
>    reading *"if composed modes are just prototype means with machinery attached, it will tie"* is **rejected** —
>    and so is any hope that the like-for-like channel closes the gap. **Both directions are reported.** (§5)
> 4. ⚠⭐ **The 72 B requests collapse to 9.0 ± 0.8 distinct code configurations** (of 72; A: 495.0 ± 2.2 of 512).
>    The capacity statement in its sharpest form, and **invisible in every accuracy number**. `clu_wide` and
>    `proto` render 72/72 distinct vectors. (§6b)
> 5. ⚠ **`gen-dream-sampler`'s `full`-protocol zero-step number (0.5611) is NOT this arm's A-baseline.** My P4
>    pre-registered it and it is **falsified**: the read reads **0.6174 ± 0.0070**. Cause: the dream clamp addressed
>    only the **confident** wells (4.42/5) and initialised the rest at `init_mode`; the read sets **every** well from
>    the posterior argmax. ⭐ **Worth +0.056 aggregate** — a real, if small, finding about the retired path. (§7)
> 6. ⚠ **P10 falsified for a KNOWN reason, and it re-confirms R19's carry-forward:** B `shape` accuracy (0.8333)
>    **exceeds** A `shape` (0.5977) because B realises exactly one class. ⛔ Nobody may read raw B-`shape` as a
>    composition score; the in-corner balanced accuracy exists for this. (§7)
> 7. ⚠ **R26's "0.86 exact-set" is not reproduced as an exact-set here.** Request-vs-feature code agreement is
>    **0.871 per WELL** but **0.510 per full K-tuple** (and `0.871^5 ≈ 0.50`). ⛔ Whichever of the two R26 meant,
>    the quote needs a unit. (§6b)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *isolation* — does composing a request through the store's READ produce the requested combination?
- **Laundering control:** ⭐ **two, both full arms on every number** — `store_removed` (the identical read path,
  store content ablated by the `mean` deletion) and `proto` (a **no-store** additive per-factor prototype
  composition, R8's declared embedding).
- **Falsifies:** the read path does not beat `store_removed` on `shape` beyond seed spread. ⛔ **It did not fire.**
- **Does NOT falsify:** absolute numbers below the decoder ceiling, or losing to an unbounded-channel arithmetic
  baseline. Every row below is read against **both** ceilings (R23.5).

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | `6b8082a` (`chlu/**` identical at branch tip `82bf5b2`; the only later `chlu/**` edit was `ruff --fix` deleting an unused `typing.Sequence` import — no semantic change, and **seed 0 re-scored at tip from its own store checkpoint: every arm number and BOTH ceilings bit-identical**, §8) |
| branch / base | `agent/experiment-engineer/gen-read-compose` off **`main @ 1bcfcb9`** (⚠ the task named `3faef27`; local `main` had advanced by the `gen-baselines` merge and `3faef27` is its ancestor — §9) |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero) |
| seeds | **0, 1, 2** (3 seeds, spread on every number) |
| decoder | `decoder_source="winner"`, recipe **`v1_capacity_schedule`**, param hashes `01ef91f75eea…` / `001011bc603a…` / `2d7eea75add7…` — **hash AND recipe asserted per seed before any number** (⛔ `dec_recipe` still defaults to `"baseline"`, the recipe that failed the gate) |
| store | `K = 5`, `n_modes = 64`, `D = 2048`, `bits_per_read = 30`, `store_bytes = 524,288` |
| grouping | `grouping_source="per_seed"` (R26.5) — re-derived through `exp_gen_dream.resolve_grouping` **verbatim**; reproduces `store_plan.json` **bit-identically** on seed 0 (its own encoder), re-derived on seeds 1–2 |
| store training | `gen_store.train_steps = 1200`, `n_store_train = 32,768`, Adam, final loss 31.65 / 29.74 / 34.36 |
| addressing | `address_smoothing = 1.0`; requests = `requests_from_labels(..., protocol="full", n_drop=0)` (cell-native) |
| ablation | `store_removed_delete_mode = "mean"` (the fairest deletion: on-manifold, correctly scaled, zero information) |
| rows | `store_train` 32,768 · A **512** · B **72** · matched-seen **198 / 199 / 199** |
| eval | `eval_batch = 256`; probe + encoder hashes asserted against `bridge_manifest.json`; cache re-encode check bit-identical (64 rows/seed) |
| ⛔ NOT used | `DreamLandscape`, `dream_rollout`, `clamp_from_posterior`, MALA/ULD, `fit_code_coupling`, `soft_read` |
| wall clock | seed 0 **4,908 s** (⚠ a 12m46s XLA compile under load-average ~110 from concurrent spokes), seeds 1–2 **872 s / 899 s**; CPU only |
| ⛔ CSF3 | unreachable — local CPU only. Submit line in §10. |

---

## 1. What I built
- **`chlu/core/gen_compose.py`** (new) — the composition path and its guarantees:
  `request_posterior` (softmax of `address_requests`' log-scores) → `compose_codes` (per-well argmax = the
  k-particle launch read) → `read_compose` (= `read_from_codes`, ⛔ a pass-through, not a reimplementation, so the
  scored path *is* the one whose signature carries the guarantee) → the frozen decoder. Plus `clu_wide_vector`
  (R27's arm), `store_removed_read`, `assert_request_only`, `deployed_byte_ledger`, `channel_ledger`,
  `metric_invariance_check`, `code_agreement`.
- **`chlu/experiments/exp_gen_compose.py`** (new) — the four-arm cell on A, B and R16's matched-seen companion.
- **`tests/test_gen_compose.py`** (new, **22 tests, all green**) — five of them fail on purpose when the guarantee
  is broken (§8).
- **`chlu/config.py`** — `GenComposeConfig` appended at the END + the three house wiring sites, ⛔ nothing else touched.
- **`chlu/cli/experiment_cmd.py`** — one appended subparser `exp-gen-compose` + its command function.

## 2. ⛔ The trap, and what I did about it
`gen_store.soft_read(store, X, tau)` computes `sum_m pi_m · mode_m` — **exactly** the composed per-well mode
vector `CLU-wide` needs — **but takes the features**. Using it would be an encoder → decoder path and would void
the no-bypass guarantee. ⭐ `clu_wide_vector(store, posterior)` takes the **posterior**, and `request_posterior`
derives it from the **request**. `assert_request_only` checks four legs and **all four passed on every seed**:

| leg | result |
|---|---|
| signatures on the whole path | `address_requests(book, requests)` · `compose_codes(posterior)` · `read_from_codes(store, codes)` · `clu_wide_vector(store, posterior)` — ⭐ **not one feature argument** |
| posterior provenance | re-derived from the requests, **bit-identical** |
| ⛔ `soft_read` not imported | checked **by name** in the module's own globals — passes |
| numeric regression guard | read and `clu_wide` outputs **bit-identical** under feature replacement (72 decoy rows) |

`assert_no_bypass` (the shipped check) also passes on this store: pure codebook gather, `max |Δ| = 0.0`,
30 bits/read, codes demonstrably live (`output_changed_when_codes_changed = 3.856`).

## 3. ⭐ The byte ledger — 524,288 B, coupling EXCLUDED

| structure | bytes | note |
|---|---|---|
| ⭐ **the deployed store (modes)** | **524,288** | `n_modes · D · 4 = 64 · 2048 · 4`; **asserted in-run**, the run aborts on any other value |
| ⛔ the inter-well PMI coupling | **163,840**, **NOT deployed** | R26 ruling 3 dropped dreaming as the composition mechanism; `fit_code_coupling`/`coupling_bytes` are referenced only by that landscape, so the field is never fitted here |
| ⛔ `gen-baselines` amendment A5's target | ~~688,128~~ | **not inherited** — correct only while dreaming was on the critical path. Recorded in the manifest as `superseded_target_do_not_inherit` |
| the CLU addressing head | 74,240 | excluded from the store count by `store_bytes.json`'s own convention, **reported anyway** |
| the `proto` arm's head | 475,136 | excluded by the **same** convention (`FactorPrototypes.byte_ledger`) — ⚠ **6.4× larger than ours**, which matters the day anyone counts heads |
| ⭐ read channel | **30.0 bits** | R20's declared design constant, untouched |

**Realised channel width, so the like-for-like is auditable rather than asserted:**
`read` **30 bits** (discrete pointer) · `clu_wide` **320 real weights** (dof 315; mean posterior entropy
**4.29 / 4.27 / 5.12 bits**; output still inside the span of the store's `K·M` modes) · `proto` **2,048 floats,
unbounded**. ⛔ `proto`/`knn_proto` are **byte**-matched, never channel-matched.

## 4. ⭐ THE GATE STATISTIC — in-corner `shape` balanced accuracy (R20/R23)
Positives = the arm's renders of the 72 held-out requests; negatives = its renders of the matched-seen requests
(same colour corner, a seen shape). **Chance 0.5.** AUROC is the mandatory companion and ⛔ never the gate.

| row | **balanced accuracy** | AUROC | per seed | vs decoder ceiling | vs store ceiling |
|---|---|---|---|---|---|
| **decoder ceiling** (real features) | **0.9414 ± 0.0179** | 0.9886 ± 0.0119 | 0.9223 / 0.9365 / 0.9653 | — | — |
| **STORE ceiling** (⭐ measured here) | **0.6866 ± 0.0681** | 0.7414 ± 0.0295 | 0.5972 / 0.7001 / 0.7624 | 0.729 | — |
| ⭐ **`read`** (30-bit) | **0.6759 ± 0.0107** | 0.7429 ± 0.0080 | 0.6610 / 0.6809 / 0.6858 | **0.718 ± 0.007** | **0.993 ± 0.086** |
| ⭐ **`clu_wide`** | **0.7484 ± 0.0307** | 0.7931 ± 0.0095 | 0.7146 / 0.7889 / 0.7415 | **0.795 ± 0.034** | **1.099 ± 0.094** |
| **`store_removed`** (laundering) | **0.5000 ± 0.0000** | 0.5000 ± 0.0000 | 0.5000 / 0.5000 / 0.5000 | 0.531 | 0.736 |
| **`proto`** (laundering, no store) | **0.8817 ± 0.0023** | 0.9908 ± 0.0085 | 0.8788 / 0.8844 / 0.8819 | 0.937 | 1.297 |
| ⛔ *reference row:* `knn_proto` (R27) | *0.8884 ± 0.0088* | *0.9912* | — | — | — |
| ⛔ *reference row:* channel-matched `knn` (R27) | *0.5206 ± 0.0487* | *0.5441* | — | — | — |

**TPR / TNR** (⛔ TPR is *detection*, never the gate): `read` 0.833 / 0.519 · `clu_wide` 0.958 / 0.538 ·
`store_removed` **0.000 / 1.000** · `proto` 1.000 / 0.763.

⭐ **A second cross-spoke consistency check, unplanned:** my `proto` arm (R8's `FactorPrototypes`, fitted on my
own store-train rows, decoded through the same winner) lands at **0.8817 ± 0.0023** against `gen-baselines`'
`knn_proto` **0.8884 ± 0.0088** — inside one seed spread, from a different row draw. ⇒ the two spokes' prototype
arms are the same estimator, and the reference row is reproducible.

⭐ **The two ceilings reproduce R22 to four decimals from an independently re-fitted store** (R22: decoder
0.9414 ± 0.0179, store 0.6866 ± 0.0681). The fit is deterministic given (seed, grouping procedure, cache), so
this is the strongest plumbing check available and it passed without having been designed as one. ⚠ It is a
numeric match, ⛔ not a checkpoint comparison.

## 5. ⭐ The comparisons that carry the result

| contrast | value | seeds + | × spread | reading |
|---|---|---|---|---|
| ⭐⭐ **`read` − `store_removed`** | **+0.1759 ± 0.0107** | **3/3** | **16.4×** | ⭐ **the falsifier did not fire: composition IS the store's** |
| ⭐ `read` − **store ceiling** | **−0.0107 ± 0.0576** | 1/3 | 0.2× | ⭐ **request-addressing costs nothing measurable vs feature-addressing** |
| `clu_wide` − `read` | **+0.0725 ± 0.0252** | **3/3** | 2.9× | widening the channel to request-derived mixture weights buys +0.07 |
| `clu_wide` − store ceiling | +0.0618 ± 0.0596 | 2/3 | 1.0× | `clu_wide` passes the 30-bit ceiling, at 1.0× spread |
| ⛔ `proto` − `clu_wide` | **+0.1333 ± 0.0285** | **3/3** | 4.7× | ⛔ **`CLU-wide` does not tie the prototype arm — it loses** |
| ⛔ `clu_wide` − `knn_proto` (0.8884) | **−0.1400 ± 0.0307** | 0/3 | 4.6× | the same statement against R27's own reference row |
| ⛔ `proto` − `read` | **+0.2058 ± 0.0090** | 3/3 | 22.8× | the unbounded-channel arithmetic baseline wins by 0.21 |
| `read` − decoder ceiling | −0.2655 ± 0.0102 | 0/3 | 26.1× | the honest distance to the top |

⭐ **R27's pre-registered `CLU-wide` reading, executed:** *"if composed modes are just prototype means with
machinery attached, this arm will tie `knn_proto` and we will say so."* **It does not tie** (0/3 seeds, 4.6×
spread) — so composed modes are *not* prototype means with machinery attached. ⛔ **The mirror statement is
equally true and is not softened: the like-for-like channel does not close the gap either.** Our own arm at an
audited 320-weight channel sits **0.14 below** a 2,048-float additive mean.

## 6. A and B — ⛔ `shape` separate from the aggregate, always

| arm | A aggregate | A `shape` | B aggregate | B `shape` |
|---|---|---|---|---|
| **decoder ceiling** | 0.9910 ± 0.0029 | 0.9733 ± 0.0133 | 0.9830 ± 0.0061 | 0.9398 ± 0.0131 |
| **STORE ceiling** | 0.6215 ± 0.0021 | 0.6107 ± 0.0252 | 0.6312 ± 0.0109 | 0.8565 ± 0.0755 |
| ⭐ `read` | **0.6174 ± 0.0070** | 0.5977 ± 0.0375 | **0.6181 ± 0.0204** | 0.8333 ± 0.1227 |
| ⭐ `clu_wide` | **0.6610 ± 0.0103** | 0.6445 ± 0.0345 | **0.6844 ± 0.0129** | 0.9583 ± 0.0589 |
| `store_removed` | 0.1196 ± 0.0151 | 0.2467 ± 0.0136 | 0.0833 ± 0.0393 | **0.0000 ± 0.0000** |
| `proto` | 0.8104 ± 0.0163 | 0.6960 ± 0.0367 | 0.8102 ± 0.0254 | 1.0000 ± 0.0000 |

⚠ **B `shape` is a one-class DETECTION rate** (R19's carry-forward) — it is *higher* than A `shape` for every live
arm and ⛔ must not be read as a composition score. §4's balanced accuracy is the like-for-like number.

**Per-factor on B** (mean over seeds) — ⭐ where the store arms actually lose:

| arm | floor_hue | wall_hue | object_hue | scale | orientation | shape |
|---|---|---|---|---|---|---|
| decoder ceiling | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 0.940 |
| `read` | 0.991 | 1.000 | 0.671 | **0.019** | **0.194** | 0.833 |
| `clu_wide` | 0.995 | 1.000 | 0.958 | **0.000** | **0.194** | 0.958 |
| `proto` | 1.000 | 1.000 | 1.000 | **0.028** | 0.833 | 1.000 |
| `store_removed` | 0.167 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 |

⭐ **`scale` is lost by EVERY composition arm including the no-store one (0.00–0.03 against a 0.958 ceiling)** —
so it is a property of *composition on this corner*, not of the store. ⛔ `orientation` is different: the store
arms hold 0.194 where `proto` holds 0.833, and that gap **is** ours.

## 6b. ⭐ The capacity finding the accuracies hide
| quantity | A | B |
|---|---|---|
| distinct code configurations emitted | 495.0 ± 2.2 of 512 | ⭐ **9.0 ± 0.8 of 72** |
| distinct rendered vectors, `read` | — | **9.0 ± 0.8** of 72 |
| distinct rendered vectors, `clu_wide` / `proto` | — | **72.0 / 72.0** |
| distinct rendered vectors, `store_removed` | — | **1.0** (⭐ the ablation is provably dead) |
| request → feature code agreement (diagnostic, ⛔ no arm consumes it) | exact-set 0.5104 ± 0.0106, per-well **0.8710 ± 0.0021** | exact-set 0.5833 ± 0.3007, per-well 0.8954 ± 0.0806 |
| request posterior confidence on B | — | 0.8326 ± 0.0278 |

⭐ **The 30-bit read maps the 72 B requests onto 9 distinct codes.** It cannot express the held-out cell's
diversity at all, and that ceiling is invisible in every accuracy column.

⚠⭐ **A number the Hub should re-check before it is quoted again.** R26 ruling 3 calls the launch read "the
validated mechanism (**0.86 exact-set**; +0.18 `shape` read margin)". On this cell, request-addressing agrees with
feature-addressing on **0.871 ± 0.002 of WELLS** but on only **0.510 ± 0.011 of full K-tuples** (A; B 0.583 ±
0.301). ⛔ **I cannot reproduce 0.86 as an exact-set on this substrate** — 0.87 is what our *per-well* rate reads.
⭐ The two are not interchangeable (`0.871^5 ≈ 0.50`, which is exactly our exact-set), and the difference is the
whole of §6b's capacity finding. **Provenance of R26's 0.86 is the Hub's to confirm; it is not a `gen_compose`
number.**

## 7. ⭐ Pre-registration scorecard
`PREREG.md` filed **2026-09-13 00:47:31**, before a line of `gen_compose.py` existed (mtime-verifiable; the first
harness output is from 01:05).

| # | prediction | measured | verdict |
|---|---|---|---|
| **P1** ⭐ | `read` beats `store_removed` by ≥ +0.10 | **+0.1759 ± 0.0107**, 3/3, 16.4× spread | ⭐ **HELD** |
| P2 | `read` in-corner = 0.65 (0.55–0.72) | **0.6759 ± 0.0107** | **HELD** |
| P3 | `store_removed` = 0.5000 (0.49–0.51) | **0.5000 ± 0.0000** (TPR 0, TNR 1, one constant image) | **HELD** |
| **P4** | `read` A aggregate = 0.56 (0.52–0.60), from dream's zero-step 0.5611 | **0.6174 ± 0.0070** | ⛔ **FALSIFIED** — see below |
| P5 | `store_removed` A aggregate = 0.12 (0.09–0.15) | **0.1196 ± 0.0151** (dream measured 0.1196 ± 0.0151 — identical) | **HELD** |
| P6 | `proto` ≈ 0.8884 (0.85–0.93) and beats `read` | **0.8817 ± 0.0023**, +0.2058 over `read` | **HELD** (pre-registered as a loss for us) |
| P7 | `clu_wide` beats `read` by +0.02…+0.10, loses to `proto`, 0.68–0.85 | **+0.0725**, loses by 0.1333, **0.7484** | **HELD** on all three clauses |
| **P8** | invariance: balacc/AUROC bit-identical; **raw accuracy moves > 0.05** | Δbalacc = **0.0**, ΔAUROC = **0.0** (3/3) ⇒ ⭐ **first clause HELD**; raw accuracy moved only −0.0019 / −0.0049 / +0.0078 ⇒ ⛔ **second clause FALSIFIED** | **SPLIT** |
| P9 | byte ledger = 524,288 B exactly, coupling excluded | **524,288**, asserted in-run | **HELD** |
| **P10** | B `shape` > `store_removed`'s **and** A-vs-B `shape` gap ≥ 0.10 | 0.8333 vs 0.0000 ✓, but B `shape` **exceeds** A `shape` by +0.236 | ⛔ **FALSIFIED** (second clause) |

**Why P4 failed, and it is a finding rather than a miss.** `gen-dream-sampler`'s `full`-protocol zero-step control
clamps only the wells whose posterior clears `clamp_threshold` — **4.42 of 5** — and initialises the remainder at
`init_mode`. The read arm sets **every** well from the posterior argmax. ⭐ **Addressing the last 0.58 wells is
worth +0.056 aggregate.** ⛔ Consequence for the Hub: 0.5611 is *not* this path's A-baseline and must not be
quoted as one.

**Why P8's second clause failed.** The clause was a non-vacuity check: a re-partition that changes the class
balance should move a *prior-weighted* statistic. On the real ceiling the probe's TPR and TNR are both ≈0.94, so
raw accuracy is nearly balance-insensitive **on this data** and barely moves. ⛔ The check is therefore
non-vacuous in principle but under-powered here; `tests/test_gen_compose.py::test_metric_invariance_check_is_not_vacuous`
demonstrates it at TPR 0.95 / TNR 0.55 (raw accuracy moves 0.067, balanced accuracy moves 0.0). ⭐ **The
invariance itself — the thing R26's standing rule asks for — held exactly, three seeds out of three, before any
arm was scored.**

**Why P10 failed** — R19's carry-forward, reproduced: Montero's cell realises exactly one `shape` class, so B
`shape` is a detection rate against a majority of 1.000 while A `shape` is 4-class discrimination. ⛔ This is the
whole reason R20 moved the gate onto in-corner balanced accuracy, and the failure re-confirms it.

## 8. How I verified (commands + real output)
```
# metric invariance, BEFORE any arm was scored (in-run, each seed):
⚠ metric invariance (R26 standing rule): balacc Δ=0.00e+00 AUROC Δ=0.00e+00 under a re-partition ⇒ ✓ INVARIANT
# decoder guard, each seed:
⭐ decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b… (hash + recipe asserted)
# byte ledger, asserted (the run aborts otherwise):
⭐ byte ledger: deployed store 524288 B (⛔ coupling 163840 B EXCLUDED …) + addressing head 74240 B
# the aggregate, 3 seeds:
⭐ falsifier: read − store_removed = +0.1759 ± 0.0107 (3/3 seeds positive, 16.4× the largest arm spread)
```
- **New tests:** `tests/test_gen_compose.py` — **22 passed in 45.16 s.** Five fail on purpose when the guarantee
  breaks: `soft_read`'s output is shown *moving* with the features while `clu_wide_vector`'s is bit-identical;
  `assert_request_only` is seen to raise both when a feature argument appears on the path **and** when `soft_read`
  is imported by name; `store_removed` is shown to emit **one** vector for every request; the invariance check is
  shown non-vacuous; and the falsifier is shown to report a dead read as dead.
- ⚠ **Three of my own tests failed on first run and I fixed the tests, not the code** — the byte fixture chopped
  `D = 2048` into five *equal* wells and lost the remainder (523,520 B, not 524,288 — exactly the arithmetic a byte
  ledger exists to catch); the invariance fixture's class-balance shift was too small to clear its own threshold
  (0.0429 < 0.05); and `save_config` takes a `Path`. Recorded because a test that has never failed has never been
  tested.
- **Lint:** `ruff check` clean on all five touched files. ⚠ `ruff format --check` reports my files (and the shipped
  `exp_gen_dream.py` / `exp_gen_baselines.py`) as unformatted — **the repo is not `ruff format`-clean**, so I did
  **not** reformat: doing so would have produced a large diff in shared files for no behavioural reason.
- ⭐ **Seed-0 re-score at branch tip** (`runs_s0_recheck/`, store loaded from the seed's own checkpoint):
  **every arm number and both ceilings bit-identical** to the original run — so the post-run `ruff --fix` is
  provably a no-op on the results.
- **Suite (main venv, worktree, `PYTHONPATH=<worktree>`):**

  | | measured |
  |---|---|
  | **before** — `main @ 1bcfcb9`, throwaway detached worktree `../CHLU-gr-base` | **2,251 tests collected** (`--collect-only`) |
  | **after** — branch tip `82bf5b2` | **2,273 collected · ⭐ 2,260 passed, 13 skipped, 0 failed (1:24:24)** |
  | delta | **+22 = exactly the 22 new tests** in `tests/test_gen_compose.py`; ⭐ **all 22 pass in the full suite, not only in isolation** — §7.23 did not bite |
  | the 13 skips | the known `importorskip("pandas")` family — venv extras, ⛔ not a defect |

  ⚠ **Declared deviation:** the **before** leg is a `--collect-only` at the base commit, ⛔ not a full run
  (`gen-baselines` measured 2:25:49 for one, and the machine was at load-average ~110 from concurrent spokes for
  most of this session). The after leg is a full green run, the delta is exactly my new tests, and `main @ 1bcfcb9`
  is the Hub's own merged-green state. HEAD was `82bf5b2` before **and** after the run (last commit ~08:2x, suite
  08:46 → 10:11); the tracked tree is clean.

## 9. Reproducing the numbers
```
PYTHONPATH=<worktree> CHLU_GEN_DATA_ROOT=<repo>/.claude/data/gen CHLU_CLAUDE_ROOT=<repo>/.claude \
CHLU_GEN_BRIDGE_ROOT=<repo>/.claude/outputs/gen-feature-bridge/runs \
  /Users/user/Desktop/CHLU/.venv/bin/python -u -m chlu.experiments.exp_gen_compose \
  --dataset shapes3d --pretext recon --stage compose --out <OUT>
# then: --stage aggregate (pools all three seeds into compose_results.json)
python3 .claude/outputs/gen-read-compose/table.py    # regenerates every table above from the manifests
```
Equivalent CLI: `chlu exp-gen-compose --dataset shapes3d --pretext recon` (subparser verified to dispatch to
`cmd_exp_gen_compose`).
⚠ The `--quick` smoke run's outputs were **deleted, not archived**, so no quick number can ever be quoted from
this directory.
**Artefacts:** `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_{0,1,2}/compose_manifest.json` (every
number above, incl. the assertions) · `.../compose_results.json` (the 3-seed aggregate) · `seed_*/compose.png` ·
`tables.md` + `table.py` · `runs_s0_recheck/` (the tip re-score) · `run_main.log` / `run_s12.log` /
`run_s0_recheck.log` / `suite_after.log` / `pytest_new.log` · `PREREG.md`.

## 10. ⛔ CSF3 submit line (not run — cluster unreachable from here)
```
sbatch scripts/csf3/job_gpu_gen.sh --stage compose --dataset shapes3d --pretext recon \
       --condition recomb2element --seed 0    # (repeat --seed 1, 2)
```
⚠ `job_gpu_gen.sh` has **no `compose` stage** — it needs one appended next to `certificate` (R26.7's owner),
and `gen_data.data_root` must still **fail loudly** rather than default. ⛔ Not edited here: `scripts/csf3/` is
outside my file ownership.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-read-compose`, worktree `../CHLU-gen-read`, base **`main @ 1bcfcb9`**.
  ⚠ **Deviation, declared:** the task named `main @ 3faef27`; local `main` had advanced to `1bcfcb9` (the
  `gen-baselines` merge) and `3faef27` is an ancestor of it. I branched from the tip so the `knn_proto` reference
  rows R27 asks me to quote come from the merged state rather than from a report. Rebase onto `main` is a no-op.
- **Commits (4):**
  `41c68b8` core `gen_compose` + `GenComposeConfig` + the three house sites ·
  `6b8082a` `exp_gen_compose` + the CLI subparser ·
  `27ff8c8` `tests/test_gen_compose.py` ·
  `82bf5b2` fix the three self-inflicted test failures.
- **Files touched:** `chlu/core/gen_compose.py` (new) · `chlu/experiments/exp_gen_compose.py` (new) ·
  `tests/test_gen_compose.py` (new) · `chlu/config.py` (**append-only at the tail + the three house sites**) ·
  `chlu/cli/experiment_cmd.py` (**append-only**: one subparser block + one command function).
  ⛔ `gen_dream.py`, `gen_store.py`, `chlu/eval/rivals/`, `chlu/data/`, `scripts/csf3/` untouched.
- **Conflicts:** none. **No push, no PR** — branch left for review.

## Open questions / follow-ups / risks
1. ⭐ **The capacity axis is now the live one.** The read sits at the store ceiling (0.2× spread) and the store
   ceiling sits 0.26 below the decoder's. R23.5 deferred "capacity vs bypass risk" to a later pre-registered axis —
   this cell says the deferral is now what blocks the number.
2. ⚠ **`clu_wide` is a *reported* arm with no ablation of its own.** I ran `store_removed` against the `read` path
   (the pre-registered falsifier). A mean-deleted `clu_wide` would also be constant, so its falsifier is
   structurally identical — but it was not run and I am not claiming it was.
3. ⚠ **Seed 0's matched-seen set is 198 rows, seeds 1–2's is 199** (the store-train draw differs per seed). Same
   as `gen-baselines` — not a defect, but the negatives are not byte-identical across seeds.
4. ⚠ **One dataset, one condition, one pretext.** Nothing here speaks to the wide condition or to dSprites.
5. ⛔ **No C set, no refusal number, no partial-request completion** — out of scope by the task.
6. ⚠⚠ **MERGE HAZARD the Hub must own: three live branches append to the SAME `chlu/config.py` tail anchor.**
   Mine (+130), `agent/experiment-engineer/gen-metric-invariance` (+83) and `agent/experiment-engineer/gen-c-set-a7`
   (+14). ⛔ A hunk-by-hunk resolution at that anchor is how one spoke's dataclass silently disappears — take one
   side's whole file, re-apply the other's block, and **assert zero deletions**. My block is delimited by
   `# --- BEGIN/END gen-read-compose` at the file tail plus **exactly three** house sites (`CHLUConfig` field,
   `load_config`, `save_config`), each also delimited, so it lifts cleanly.
7. ⚠ **There are now TWO invariance instruments and they must not be conflated.** Mine
   (`gen_compose.metric_invariance_check`) checks the **cell's gate statistic** — balanced accuracy under a
   class-balance re-partition — and runs in-line before any arm is scored. `gen-metric-invariance` is a separate
   spoke checking the **ladder's x-axis** (DCI-C) under a different partition, per R26 ruling 2. Same standing
   rule, different metrics, different owners.

## Proposed handover updates (for the Hub)
- **§7 NEW (config, resolved-by-design):** `gen_compose.expect_deployed_store_bytes = 524288` and the run
  **aborts** on any other value. The 688,128 B target is recorded in every manifest as
  `superseded_target_do_not_inherit` so a future spoke cannot silently inherit `gen-baselines`' amendment A5.
- **§7 NEW (mechanism, standing):** `gen_store.soft_read` is an **encoder → decoder path**. Any arm that wants a
  composed mode vector must derive its weights from the request (`gen_compose.request_posterior`), and
  `gen_compose.assert_request_only` is the check. ⭐ It fires on both failure modes and has tests that prove it.
- **§7 NEW (instrument):** `gen_dream.resolve_grouping` asserts the per-seed re-derivation reproduces
  `store_plan.json` bit-identically **on the plan's own seed**, which cannot hold on a smoke-sized subsample —
  hence `gen_compose.quick_grouping_source = "plan"`. ⚠ `exp_gen_dream --quick` on seed 0 hits this assertion and
  will fail; not fixed here (⛔ `gen_dream.py` is outside my ownership).
- **Ledger (R26/R27 follow-through):** the store ceiling **0.6866** is now **measured**, not quoted, and matches
  R22 to 4 dp; `CLU-wide` **does not tie** `knn_proto` (−0.1400, 4.6× spread, 0/3); the read is **indistinguishable
  from the store ceiling**; `gen-dream-sampler`'s 0.5611 is **not** this path's A-baseline (+0.056).
