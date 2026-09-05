# c2w10-lifecycle-mechanics — experiment-engineer report

**Task + acceptance criterion (one line):** build the C2W10 THREE-STATE lifecycle (PROTECTED ⇄ ACTIVE
→ TRASH) with the kill-conditions committed **before** the verbs, and file `LIFECYCLE-MECHANICS-DONE.json`
+ `USAGE-TELEMETRY.json` with `lifecycle_mechanics_done` computed mechanically as the AND over L1–L7.
**Status: done.** All seven legs landed (`lifecycle_mechanics_done = true`); **L4 is UNEXERCISED on the
stream** and labelled as such; the **real-stream legs are a declared NOT-RUN** because
`BENCHMARK-GATE.json` does not exist.

⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary); detail in §8:**
**(R1)** the task file **and** `PREREG-C2W10.md` §4 L5-b **misattribute C2W6's numbers**: `events
27/40/70, rates 0.593/0.050/0.043` is the **`p1_off` GUARD-OFF BASELINE**, not `p1_on_i1_on`. The I1 arm
is **44/62/59 events, 6/0/0 pre-guard violations**. Add.7 §A22's row is correct as printed.
**(R2)** §A23.2's *"zero post-guard violations"* was **never in the artifact**
(`n_rewrite_violations_post_guard` is `null` in all three C2W6 files); I recomputed it and it **is 0/0/0**
— the claim now has evidence for the first time.
**(R3)** L3's registered wording ("never useful **since first appearance** over `k` boundaries") and its
registered designed negative (a) ("useful in stream 1 only ⇒ trashed at `k`") are **not jointly
satisfiable**. Both readings ship; the default is declared; the difference is pytest-asserted.
**(R4)** L2 cannot be scored on the trailing window without breaking "**within** `d_demote` chunks"
(a window of length `W` delays demotion to `W−1+d_demote`).
**(R5)** the registered L6 assertion "nets to analytic `exp(−leak·Δt)` **to 1e-9**" holds only in
**float64**; the shipped store's amplitudes are float32 and the floor is **≈5.3e-8**. Both are pinned.
**(R6)** the base test count at `9e0bb25` in a fresh worktree is **1564 collected / 1562 selected**, not
the task file's 1555.
**(R7)** the operating point that clears the I2 spoke's `n_live ≥ 64` **per seed** needs
`d_safe_frac = 0.60`, not the carried 0.88 (at 0.88 seed 1 reached only 52 live wells).

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** **lifetimes + admission**, as a full-system component build. ⛔ No paper number, no VALUE
  number, no tier-ii/tier-iii verdict, no full-CLU verdict.
- **Laundering control:** none required, **and that is the point** (§A33.1). No launder margin appears
  anywhere in this build; none was computed.
- **Falsifies:** any designed negative that cannot fail ⇒ that leg does not ship.
- **Does NOT falsify:** an empty trash population (K-C: UNEXERCISED, not broken, not working).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ `M` never appears. ⛔ N94 respected: every
  reported cell runs **write inner steps = 40**; the `--quick` smoke cell is flagged non-promotable with
  its reason string.

---

## 1. Build order — kill-conditions FIRST (acceptance item 3)

| # | commit | what | observed |
|---|---|---|---|
| 1 | **`162fdba`** | **designed negatives, RED against a stub** (`tests/test_store_lifecycle.py` + an unimplemented `chlu/core/store_lifecycle.py`) | **32 failed, 1 passed** — intentional |
| 2 | **`255aa1d`** | the implementation the negatives gate | **34 passed** |
| 3 | **`11b2dd2`** | substrate: synthetic regime-switcher, frozen loader + sha256 gate, decimation | **21 passed** |
| 4 | **`b5a3ceb`** | the rig, config group, CLI command, cross-stream telemetry | **12 passed** |
| 5 | **`9459263`** | L5's designed negative on the **live store path** | **2 passed** |
| 6 | **`6e0c325`** | the §7.23 ordering fix the full suite caught (§7.2) | **52 passed** paired |

The single test green at step 1 (`test_l3_depth_never_enters_the_usefulness_criterion`) inspects the
criterion's *signature* for a depth argument, which the stub also lacked — green for the right reason at
both commits.

## 2. Flag provenance (every number below)

| item | value |
|---|---|
| commit | **`6e0c325`**; base **`main @ 9e0bb25`**; branch `agent/experiment-engineer/c2w10-lifecycle-mechanics`, worktree `../CHLU-c2w10` |
| env | **main venv reused** (protocol §4), **no worktree `uv sync`** ⇒ **JAX 0.9.0** |
| seeds | **0, 1, 2** (cells + the drift-free control); tests seeded per case |
| rig | `CluSystem`, learned `V_theta` (`DesignFreedomPotential`, rung `free_mlp`, family `atoms`) |
| carried rig facts | `atom_site_local_init=True` (placing write) · `atom_kernel=wendland`, cutoff 2.5 · `atom_width = 1.5 × measured distinct-item spacing` (co-scaled) · **`addr_dim = 12`** (d = 16 = declared NOT-RUN) |
| address block | **cheap UNFITTED random projection**, **0 fit steps**, standardisation + unit-ball scale from **stream 0 only** (§A31.4 inversion). φ params on every byte ledger (484 B) |
| store | `capacity 72`, **`budget = well_budget = 64`**, `leak 0.02`, `stage_lifetimes=True`, `n_atoms = 32 768`, `dim = 13` |
| admission | `d_safe = d_safe_frac × distinct-item spacing`; **reported run `d_safe_frac = 0.60`** (companion run at the carried 0.88 in `run1/`) |
| lifecycle | `h_hi 2` · `h_lo 1` · `window 2` · `d_dwell 3` · `d_demote 2` · `k_streams 3` · `trash_criterion last_k_streams` · `censoring_guard True` · `f_max 0.25` · `refresh_monotonic False` (ships OFF) |
| read/write budget | chunk `C = 8`, `offers_per_chunk 3`, **`write_steps = 40` (N94 floor ⇒ promotable)**, `read_steps 200`, `address_steps 100`, **`read_batch 128`** |
| stream | synthetic regime-switcher, `schedule (0,1,2,0,1,2)` = **6 streams / 5 change points / 3 revisits**, `n_anchors 96`, `jitter 0.02`, `n_per_stream 64` ⇒ 384 instances, 48 chunks |
| decimation | **`m = 1`**, selected by the registered rule (§5.4) |
| `gamma_phi` | **ON** in every lifecycle cell (L3's first experimental use); **OFF** in the L7 identity check |
| wall | **1252 / 1172 / 1227 s per seed**, 4027 s total incl. the control |
| venue | ⛔ **synthetic = MECHANICS instrument, NEVER a claim venue (§A14.8)** |
| declared NOT-RUNs | the real INSECTS stream (no `BENCHMARK-GATE.json`) · d = 16 · merge verbs / K9 · prune-by-depth · the anytime curve · any VALUE cell / tier-ii / full-CLU verdict |

## 3. The seven legs (`LIFECYCLE-MECHANICS-DONE.json`)

`lifecycle_mechanics_done = **true**`, computed in code as the AND over `legs[L1..L7].landed`, each of
which is a **pytest exit status**, not a judgement.

| leg | landed | designed negative(s) — all green | shown able to FAIL by | on the stream |
|---|---|---|---|---|
| **L1 PROMOTION** | ✅ | a single burst reaching `h_hi` does **not** promote (dwell reaches 2 < 3); a well below `h_lo` never promotes | `d_dwell := window` ⇒ the burst **does** promote; `h_hi := 0` ⇒ a never-read well promotes | **60 promotions** (27/13/20) |
| **L2 DEMOTION** ⭐ | ✅ | the planted early-popular-then-abandoned well **demotes within `d_demote`** and is **not** trashed by the demotion; the demoted well's depth follows `exp(−leak·Δt)` again | `demote := False` ⇒ it stays PROTECTED forever | **53 demotions** (24/12/17) |
| **L3 TRASH** | ✅ | (a) useful-in-stream-1-only ⇒ trashed at `k`; (b) useful in **every** stream ⇒ **never** trashed; (c) censoring guard: a well admitted in the last stream is **never** trashed | (b) the discriminating input flipped (hits zeroed ⇒ trashed); (c) `censoring_guard := False` ⇒ the young well **is** trashed; (a) `trash_criterion := since_first_seen` ⇒ does **not** fire | **117 routings** (37/42/38), `trash_bytes` 2220/2520/2280 B |
| **L4 PROTECTED FRACTION** | ✅ | forcing every item's usage high **trips `protected_saturation` and REFUSES**, never silently protects all | `f_max := 1.0` ⇒ everything protects and nothing trips | ⚠ **UNEXERCISED** — see §3.1 |
| **L5 I1 REFRESH** | ✅ | with the guard OFF a planted destructive rewrite **reduces the depth** — asserted both on replayed depths **and on the live store path** | the same events with `refresh_monotonic := True` end at `min(d_before, d_after·gain²)` | not driven in the cell (by design: the cell never rewrites a live well) |
| **L5-b CROSS-IMPL** | ✅ | E1/E2/E3 against C2W6's **own recorded events** — §4 | guard OFF on the same events reproduces **16/2/3** and **6/0/0** post-guard violations | — |
| **L6 NETTING** | ✅ | netted ≡ raw **bitwise** at `leak = 0`; netted > raw strictly at `leak > 0`; a well with no writes nets to the analytic law | the same function, both branches; and the float32-floor bound (R5) | **every** curve emitted raw **and** netted |
| **L7 OFF** | ✅ | OFF is **bit-identical** (leaves + read output) **and parameter-count-identical** (`n_params = 492 493`), `trash_attached = false`, `trash_bytes = 0` | the trash region is attached only when the verb is ON | γ_φ OFF regressions still green (3 passed) |

### 3.1 K-C applied: **L4 is UNEXERCISED at the measured operating point**
`protected_cap = floor(0.25 × 64) = 16`; the measured protected population peaked at **3, 1, 3** wells,
so `n_promote_refused = 0` on every seed and `protected_saturation` **never tripped on the stream**.
Per the registered K-C rule this is reported as **UNEXERCISED — not working and not broken**. The verb
itself is green on the planted population (the negative *and* its can-fail twin), so the leg lands; what
is absent is a *stream* population that reaches the bound.
**Why the bound is far away is measurable, not mysterious (§3.2):** promotions are undone almost as fast
as they are made.

### 3.2 A mechanics finding the Hub should see: **promotion/demotion churn is ≈ 1:1**
Across the three seeds: **60 promotions vs 53 demotions**, and the log shows the pattern item-by-item —
a well promotes at chunk *c* and demotes at *c+1* or *c+2*, repeatedly (e.g. seed 0 item 22: promoted at
chunks 10, 15, 36; demoted at 11, 18, 38). This is the asymmetric hysteresis working exactly as
registered — promotion is sticky (`d_dwell = 3 > window = 2`), demotion is prompt (`d_demote = 2`) — but
at this read budget usage is **bursty on a 2-chunk timescale**, so the protected set never accumulates.
⇒ **If the wave wants stable protection, `d_demote` must exceed the usage's burst gap**; that is a
parameter decision for the Hub, not a defect, and it is why L4 is unexercised.

## 4. L5-b — the cross-implementation validation, and the reconciliation I owned first

**The reconciliation, resolved from the RAW artifact before the test was pinned** (`PREREG.md` §P0,
filed before the harness ran). Read directly out of
`.claude/outputs/c2w6-anti-erosion/erosion_*_records.json`:

| C2W6 cell | flags | events/seed | pre-guard violations | rate |
|---|---|---|---|---|
| `p1_off` | P1 OFF, I1 OFF | **27 / 40 / 70** | 16 / 2 / 3 | **0.593 / 0.050 / 0.043** |
| `p1_on` | P1 ON, I1 OFF | 37 / 78 / 53 | 1 / 0 / 0 | 0.027 / 0 / 0 |
| `p1_on_i1_on` | P1 ON, **I1 ON** | **44 / 62 / 59** | **6 / 0 / 0** | 0.136 / 0 / 0 |

⇒ **`[0.593, 0.050, 0.043]` is the guard-OFF baseline (`p1_off`), not the I1 arm** (R1). Add.7 §A22's
`OFF → ON under P1` row is the `p1_off → p1_on` contrast and is **correct as printed**. §A23.2's "0
post-guard violations" is the `p1_on_i1_on` cell, whose raw `n_rewrite_violations = 6/0/0` counts
**pre-guard** events (the ones the guard repaired) — `exp_anti_erosion.post_guard_violations`' own
docstring says so.

**R2:** `n_rewrite_violations_post_guard` is **`null` in all three artifacts** (the function post-dates
them). Recomputing it with the **shipped** function (imported read-only) over the stored per-event
depths: **`p1_on_i1_on` → 0 / 0 / 0** and **`p1_off` → 16 / 2 / 3**. **P0-a CONFIRMED** (prior 0.85).

**The equivalence, on C2W6's own 302 rewrite events** (registered tolerances E1/E2/E3):

| test | tolerance | measured |
|---|---|---|
| **E1** violation flag reproduces `blocks.py`'s | 0 mismatches | **0 mismatches / 302 events**, and my rate equals the banked rate to 1e-12 on all 9 cells |
| **E2** refresh factor, amplitude units | max rel. dev ≤ 1e-6 | **5.36e-8** on the guard-ON cell (`p1_on_i1_on`) |
| **E3** post-guard violations under my guard | 0 | **0 / 0 / 0** on every cell, including replaying `p1_off`'s 16/2/3 |

⚠ **E2 is stated on the guard-ON cell only, by construction**: the guard-OFF cells recorded
`refresh_factor = 1.0` for every event (the guard was off), so comparing my *guard-ON* factor to their
*guard-OFF* record gives deviations of 1.65 / 0.21 / 0.85 — an **arms mismatch, not a divergence**. The
test is pinned to `p1_on_i1_on`.
⚠ The two implementations differ only in **units**: `blocks.py` multiplies `amp` by `f`, the store level
multiplies **depth** by `f²` via `LearnedVStore.scale_group_amplitude`. Same arithmetic, same cap.
⚠ The L5-b tests **skip** where the gitignored `.claude/` artifact tree is absent (they read C2W6's raw
JSON). They ran here; the numbers above are the evidence.

**The "up to budget" cap is real and binds:** on the live-store negative, `d_before/d_after = 27 >
gain² = 16`, so the guard restored 16× and no further. That is §A23.2's registered semantics and the
same cap `blocks.py` applies; the test asserts `depth_guarded == min(d_before, d_after·gain²)` and
records which branch it is in, rather than asserting only the easy one.

## 5. The run (MECHANICS; ⛔ no claim, no verdict)

### 5.1 Per seed (`run2/`, `d_safe_frac = 0.60`)

| seed | offered/admitted/refused/evicted | **n_live_max** | coverage | promote / demote / trash | never-read | hits median / max |
|---|---|---|---|---|---|---|
| 0 | 144 / 77 / 67 / 14 | **64** | 0.464 | 27 / 24 / 37 | 20 / 63 | 19 / 281 |
| 1 | 144 / 69 / 75 / 6 | **64** | 0.452 | 13 / 12 / 42 | 44 / 63 | 0 / 774 |
| 2 | 144 / 75 / 69 / 12 | **64** | 0.514 | 20 / 17 / 38 | 24 / 63 | 15 / 663 |

**`n_live ≥ 64` on 3/3 seeds ⇒ the I2 spoke's I2-a power precondition is met** (P6-a **CONFIRMED**,
prior 0.70 — but only after the operating-point change in R7; at the carried `d_safe_frac = 0.88`
(`run1/`) it was **64 / 52 / 64**, i.e. 2/3 seeds).

**Q4 (Hub's prior 0.45) — CONFIRMED, with a caveat I am flagging rather than burying:** L3's target
population is **not** empty (117 routings), so **K-C does not fire for L3**. The caveat: the trash
criterion is driven by `read_hits`, and **48–55 % of reads land in no basin at all** (`n_unassigned`
3225 / 3297 / 2926 of 6016), because `covered` is a **launch-point** test. A well that is never covered
is never credited, so part of the trashed population is "never *reachable* by this read", not "never
useful". `read_coverage` therefore travels with every usage number in the deliverable, and it is the
ceiling on the proxy's resolution.

### 5.2 Byte ledger (per cell, every component named)
`clu_store 1 969 920 B` (= `n_atoms 32 768 × (dim 13 + 2) × 4`) + `codebook 3 024 B` + `φ params 484 B`
+ **`trash_bytes` 2 220 / 2 520 / 2 280 B** (= `K × (dim+2) × 4`, `K` = 37/42/38 holes) ⇒
**total 1 975 648 / 1 975 948 / 1 975 708 B ≈ 1.884 MiB.** `gamma_phi_enabled = true` in every cell.

### 5.3 Monitors
Tripped: **`vacuous_gate`** (seeds 0, 2) and **#12 `starvation`** (all three) — the same pair C2W8 saw
on this substrate. **`protected_saturation`** (the new row) is registered, reported by name on every
cell, and **did not trip** (§3.1). The row's severity class is **II**, added as a one-line hunk to
`monitors.SEVERITY`; the monitor object itself lives in `store_lifecycle.py` and attaches through the
registry's public `register()`.

### 5.4 Pricing and decimation (evidence for the Hub to file into `PREREG-C2W10.md` §9)
Probe at the declared operating point: write **4.17 s** (steady, first write 7.45 s pays the compile),
read **11.83 s** per chunk at `read_batch = 128`, 48 chunks × 144 writes ⇒ **projected 1168 s/seed**
against the **7200 s** target. Ladder: every `m ∈ {1,2,5,10}` meets it ⇒ **the registered rule selects
`m = 1`** (smallest `m` meeting the target). Measured wall: 1252 / 1172 / 1227 s/seed. `n94_ok = true`.
Structure at `m = 1`: 6 runs, **5 change points**, regime sequence `[0,1,2,0,1,2]`, **3 revisits**,
64 instances per segment — and the structure assertion is shown to **fail** at an `m` that empties a
segment.

### 5.5 The drift-free control
`promote 22 / demote 21 / trash 37` against seed 0's `27 / 24 / 37` — i.e. **the lifecycle verbs behave
the same with and without regime switching**. That is the *correct* MECHANICS reading, and it is worth
stating plainly: **the verbs are driven by usage, and usage is a property of the address stream, not of
the label map.** A lifecycle that fired differently here would be responding to drift it cannot see.
⛔ This is not a VALUE control and no retention/adaptation quantity was computed.

### 5.6 The real stream — **declared NOT-RUN**
`.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json` **does not exist** (the directory is empty),
so no frozen file and no sha256 exist yet. The real-stream legs are a **NOT-RUN with that reason, never
a null**. ⛔ I did not download or freeze anything. The machinery ships and is pytest-pinned on the
synthetic: the loader **recomputes the digest and raises on a mismatch**, a missing gate file is handled
as a NOT-RUN rather than a crash, decimation refuses an `m` off the registered ladder, and structure
preservation is asserted with counts.

## 6. The deliverables (exact paths, acceptance item 1)
- **`.claude/outputs/c2w10-lifecycle/LIFECYCLE-MECHANICS-DONE.json`** — L1…L7 booleans, each with its
  designed negatives, the can-fail mutation that makes each one fail, the pytest command + summary,
  `exercised_on_stream` + the K-C verdict where it is false, `lifecycle_mechanics_done = true` computed
  as the AND, the byte ledgers, monitor trips, pricing, OFF identity and the NOT-RUN list.
- **`.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json`** — per item `hits_by_stream` +
  `first_seen_stream`, per item **raw AND netted** depth curves with the cumulative decay factor each
  was netted by, `n_live` at every measurement point per seed, `n_seeds = 3`, **`n_live_max = 64`**,
  `n_live_max_per_seed = {0:64, 1:64, 2:64}`, `n_seeds_meeting_64 = 3`, `read_coverage` per seed, and
  the three caveats (depth ≠ importance; synthetic ≠ claim venue; coverage bounds the proxy).
- Companion: **`run1/`** (the same cells at the carried `d_safe_frac = 0.88`), **`PREREG.md`**,
  **`make_deliverables.py`** (the generator — nothing in it decides anything).

## 7. How I verified (commands + observed output)

* Env: main venv reused, `PYTHONPATH=/Users/user/Desktop/CHLU-c2w10
  /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the worktree. **No worktree `uv sync`** ⇒
  **JAX 0.9.0** (no w6-style drift).
* `pytest tests/test_store_lifecycle.py` — **32 failed, 1 passed** at `162fdba` (the kill-conditions),
  **34 passed** at `255aa1d`.
* `pytest tests/test_stream_sources.py` — **21 passed**. `pytest tests/test_persistent_store.py` —
  **14 passed** (12 at `b5a3ceb`, +2 live-L5 in `9459263`).
* `pytest tests/test_well_lifecycle.py tests/test_store_lifecycle.py` — **52 passed** (the x64-ordering
  pairing that caught §7.2).
* `pytest tests/test_well_lifecycle.py -k gamma_phi` — **3 passed** (the γ_φ OFF regressions still green).
* `pytest tests/test_config.py` — **7 passed** (the mutate-every-group round-trip, with the new group).
* `python -m chlu.experiments.exp_persistent_store --quick` — real end-to-end smoke, every leg driven.
* `python -m chlu.experiments.exp_persistent_store --seeds 0,1,2` (+ the `d_safe_frac = 0.60` variant) —
  the two runs in §5, 4027 s for the reported one.
* `ruff check` on **every** file touched — **All checks passed** (one B007 found and fixed in-flight).
* CLI: the new `exp-persistent-store` parser builds and dispatches to `cmd_exp_persistent_store`.

### 7.1 Full suite (acceptance item 6) — **the checkout is named**
Counts are comparable only within one checkout, so I measured the base in a **fresh worktree of
`9e0bb25`** rather than quoting the task file:

```
base @ 9e0bb25 (fresh worktree /tmp/c2w10base):   1564 collected   (1562 selected, minus the
                                                   network-hitting tests/test_download_concurrency.py pair)
branch @ 6e0c325 (worktree ../CHLU-c2w10):        1633 collected  (1631 selected)
1564 + 69 (mine: 34 store_lifecycle + 21 stream_sources + 14 persistent_store) = 1633 ✓
```
⚠ **R6: the task file's "1555 selected" does not reproduce; I measure 1562.** I did not chase the
difference — it is outside my scope — but the arithmetic above is internally consistent and the base
figure is reproducible with one command.

**Run 1** (`suite1.log`, at `b5a3ceb`, before the last two commits): `1 failed, 1628 passed, 2
deselected in 2939 s`. **The failure was mine** — §7.2.
**Run 2** (`suite2.log`, at the final commit `6e0c325`):

```
$ pytest -q --no-cov --deselect tests/test_download_concurrency.py     # on 6e0c325
1631 passed, 2 deselected, 29 warnings in 2600.47s (0:43:20)
```
**GREEN: 1631 passed / 0 failed.** Count arithmetic, in one line:
**base 1564 collected (fresh worktree @ `9e0bb25`) + 69 mine = 1633 collected; − 2 network-hitting
deselected = 1631 selected, all passed.**
Mine = **69** = 34 (`test_store_lifecycle.py`) + 21 (`test_stream_sources.py`) + 14
(`test_persistent_store.py`; 12 at `b5a3ceb`, +2 in `9459263`).

### 7.2 The full suite caught a bug of mine (the §7.23 ordering hazard, again)
`test_l6_..._float32_floor` asserts the shipped store does **not** reach the registered 1e-9 netting
bound, because its amplitudes are float32. The dtype was **inherited**: run alone it is float32 and the
assertion is correct; after any module that enables `jax_enable_x64` the same store is float64, the
bound *holds*, and a correct negative assertion goes red. Green alone, red in the suite — the exact
class C2W8 §8.1 recorded twice. Fixed in `6e0c325` by pinning the flag **function-scoped** (a
module-scoped x64 fixture is itself the N211 hazard) and restoring it in a `finally`.

## 8. Reconciliation detail + scored pre-registrations

**PREREG scorecard** (`.claude/outputs/c2w10-lifecycle/PREREG.md`, filed **before** the harness ran):

| # | prediction | outcome |
|---|---|---|
| **P0-a** | recomputed post-guard violations = 0/0/0 (`p1_on_i1_on`) and 16/2/3 (`p1_off`) | ✅ **CONFIRMED**, exactly |
| **P1 E1/E2/E3** | 0 flag mismatches; factor dev ≤ 1e-6; 0 post-guard violations | ✅ **CONFIRMED** (0 / 5.36e-8 / 0) |
| **P2-a** | the burst negative passes on the first build at `window 2, d_dwell 3` | ✅ **CONFIRMED** |
| **P3-a** | forcing usage high leaves `n_protected == floor(0.25·budget)` with ≥ 1 refusal + trip | ✅ **CONFIRMED** in the planted test; ⚠ **UNEXERCISED on the stream** (§3.1) |
| **P4-a** | the analytic netting law holds to ≤ 1e-9 | ⚠ **PARTIALLY REFUTED**: holds in float64, floor ≈ **5.3e-8** in the shipped float32 store (R5). The residual risk I named in the prereg is the one that fired |
| **P5** | the smallest ladder `m` meeting the target | ✅ `m = 1`, with the evidence table |
| **P6-a** | `n_live_max = 64` reachable within the wall-clock target | ✅ **CONFIRMED** (3/3 seeds) — **but only after R7's operating-point change** |
| **P7** | any leg whose negative cannot fail ships `false` | no leg hit this |

**R3 in full — the two readings of L3.** The registered wording is "never useful **since first
appearance** over `k` stream boundaries"; the registered designed negative (a) is "useful in stream 1
only ⇒ trashed at `k`". An item useful in stream 1 is *not* never-useful-since-first-appearance, so (a)
can only fire under a **trailing-window** reading. Both ship:
`last_k_streams` (**default**) = zero hits in each of the last `k` streams — satisfies (a), (b) and (c),
and the never-useful-at-all well is the special case whose whole history is zero;
`since_first_seen` = the literal wording, under which (a) does **not** fire.
`test_l3_the_two_readings_of_the_criterion_differ_and_both_ship` asserts the difference so it cannot be
silently resolved later. **The Hub should ratify which reading §A20.6 intends.**

**R4 in full.** Scoring demotion on the trailing window makes "PROTECTED → ACTIVE **within `d_demote`**
chunks" false as written: a window of length `W` carries the last pre-abandonment hits forward for
`W−1` further chunks, delaying demotion to `W−1+d_demote` (measured: 3 chunks at `W=2, d_demote=2`).
Demotion is therefore scored on **the chunk's own hits** while promotion keeps the window. Promotion
sticky, demotion prompt — and it is the only version in which the rich-get-richer negative bites.

**The derivation behind L1's parameters (PREREG §P2), because it is a guard-validity result:** a single
burst satisfies the trailing-window test for **exactly `window` consecutive chunks**, so the hysteresis
binds — and the burst negative can fail at all — **iff `d_dwell > window`**. With `d_dwell ≤ window` the
designed negative is *arithmetically incapable of failing*: the vacuous-guard defect class. This is now
a `ValueError` at construction, not a comment.

## 9. Git footprint

* **Worktree** `../CHLU-c2w10`, **branch `agent/experiment-engineer/c2w10-lifecycle-mechanics`** off
  **`main @ 9e0bb25`** (named base, per the task file).
* Commits (verify from the main repo with `git log --oneline 9e0bb25..agent/experiment-engineer/c2w10-lifecycle-mechanics`):
  * `162fdba` — the designed negatives, RED against a stub (kill-conditions first)
  * `255aa1d` — `chlu/core/store_lifecycle.py` + `controller.set_permanence` + one `monitors.SEVERITY` row
  * `11b2dd2` — `chlu/experiments/stream_sources.py` + `tests/test_stream_sources.py`
  * `b5a3ceb` — `chlu/experiments/exp_persistent_store.py`, config group, CLI command, telemetry
  * `9459263` — L5's designed negative on the live store path
  * `6e0c325` — the §7.23 ordering fix
* **Files created:** `chlu/core/store_lifecycle.py`, `chlu/experiments/stream_sources.py`,
  `chlu/experiments/exp_persistent_store.py`, `tests/test_store_lifecycle.py`,
  `tests/test_stream_sources.py`, `tests/test_persistent_store.py`.
* **Files edited (all owned, minimal hunks):** `chlu/core/controller.py` (**one new method**,
  `set_permanence`; LRU/staleness semantics untouched, `last_used` not written) ·
  `chlu/experiments/usage_telemetry.py` (additive: `hits_by_stream`, `first_seen_stream`,
  `set_stream`, `cross_stream_summary`; `current_stream` defaults to 0 so every pre-C2W10 call site is
  bit-identical) · `chlu/config.py` (**additive only** — one new group + its three wiring sites, no
  existing default touched) · `chlu/cli/experiment_cmd.py` (**additive only** — one command) ·
  `chlu/core/monitors.py` (**one line** in `SEVERITY`, inside a `BEGIN/END c2w10-lifecycle` fence; the
  registry supports additive rows via `register()`, so the monitor object itself lives in my module).
* **Not touched:** every file on the DO-NOT-TOUCH list — `well_lifecycle.py` (imported read-only for
  `designed_decay_factors`), `clu_system.py` (**called**, never edited), `soft_certificate.py`,
  `blocks.py`, `train_cluformer.py`, `exp_cluformer_pilot.py`, `scripts/csf3/`, `factored_store.py`,
  `multiplicity_read.py`, `multiwell_read.py`, `psi_readout.py`, `null_arms.py`, `exp_cat_test.py`,
  `exp_tierii_*`, `exp_null_arms.py`, `admission.py`, `placement.py`, `exp_anti_erosion.py`
  (**imported** for `post_guard_violations`).
* **No conflicts.** The shared main checkout (on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`)
  was **never edited**. Rebase onto the named base `main` is a **no-op** (`main` has not moved from
  `9e0bb25` in this worktree's view). ⛔ Nothing pushed, no PR.
* ⚠ I created a **temporary worktree `/tmp/c2w10base` at `9e0bb25`** purely to measure the baseline test
  count; it is removed at the end of this task (`git worktree remove /tmp/c2w10base`). No branch, no
  commits.
* **Worktree-priority check (Head ruling 4):** at spawn time `git worktree list` showed **only** the
  shared checkout, and **no C2W11 branch or worktree existed** — the engineer slot was free, so I
  proceeded rather than waiting. Recorded so the Hub can audit the decision. ⚠ A C2W11 worktree
  (`../CHLU-c2w11a`, branch `c2w11-substrate-and-kills`) **appeared mid-flight**; there is **no
  collision** (separate worktree, disjoint file ownership, and I never touched the C2W11 files on the
  DO-NOT-TOUCH list), but the Hub should know the two ran concurrently.

## 10. Open questions / risks

1. **Which reading of §A20.6 is intended (R3)?** Both ship and the default is declared, but only the
   Advisor/Hub can ratify. This changes *which* wells are trashed, not whether the verb works.
2. **`d_demote` vs the usage burst gap (§3.2).** At the measured operating point the protected set never
   accumulates (churn ≈ 1:1), which is why L4 is unexercised. If the wave wants protection to *stick*,
   the parameter needs re-registering — that is a Hub decision, not a spoke one.
3. **Launch-point coverage bounds everything downstream.** 45–51 % of reads are credited; the rest are
   credited to nobody by design. Any I2 correlation computed on `read_hits` inherits that attenuation,
   and the I2 spoke should quote `read_coverage` beside its ρ.
4. **The trashed population is partly "unreachable", not "useless"** (§5.1). The verb is exercised, but
   its target set is not a pure never-useful set at this coverage.
5. **The real stream is missing.** Every real-stream leg, the `m` selection *for the benchmark*, and the
   sha256 reproduction are NOT-RUN until `BENCHMARK-GATE.json` lands. The code path is built and tested.
6. **L5-b's tests skip without the gitignored artifact tree.** The evidence is in §4 and in the
   deliverable; if the Hub wants it permanently green in CI, the 302 events would have to be vendored
   into `tests/`, which I did not do on my own authority.

## Proposed handover updates (for the Hub)

1. **§7 Known Issues — new entry (R1/R2):** *C2W6's `[0.593, 0.050, 0.043]` is the `p1_off` GUARD-OFF
   baseline (27/40/70 events), **not** `p1_on_i1_on` (44/62/59 events, 6/0/0 pre-guard). §A23.2's "zero
   post-guard violations" was never recorded in the artifact and has now been **recomputed as 0/0/0**.*
   Both the C2W10 task file and `PREREG-C2W10.md` §4 carry the wrong attribution and should be amended.
2. **§7 — new entry (R5):** *the L6 netting law reaches the registered 1e-9 only in float64; the shipped
   store's amplitudes are float32 and the achievable floor is ≈5.3e-8.* Any future prereg quoting 1e-9
   should name the dtype.
3. **§7 — new entry (the coverage/geometry trap):** *sizing `d_safe` from the median nearest-neighbour
   distance of stream **instances** measures the within-item jitter on any revisiting stream, collapses
   `min_sep`, and drives the read's launch-point coverage to ~13 % — i.e. a usage proxy that is ~0 for a
   purely geometric reason.* `exp_persistent_store.distinct_key_spacing` is the fix and is pytest-pinned.
4. **§3 CLI/config:** new command **`chlu exp-persistent-store`**; new config group
   **`experiment_persistent_store`** (additive; every lifecycle verb ships **OFF**, so `main` behaviour
   is preserved).
5. **§2 architecture:** new module **`chlu/core/store_lifecycle.py`** (the three-state lifecycle; the
   store-level `refresh_monotonic` guard lives here, **not** in the frozen `blocks.py`) and
   **`chlu/experiments/stream_sources.py`** (synthetic regime-switcher + frozen-stream loader +
   decimation). New monitor row **`protected_saturation`** (severity II).
6. **Test-count baseline (R6):** the handover/task-file figure **1555** does not reproduce; a fresh
   worktree at `9e0bb25` collects **1564** (1562 selected). Branch tip `6e0c325` collects **1631**.
7. **Operating point (R7):** the I2 spoke's `n_live ≥ 64` **per seed** needs `d_safe_frac = 0.60`; at the
   carried 0.88 seed 1 reached only 52 live wells. I left the config **default at 0.88** (the carried
   C2W8 rig fact) and ran the reported cells with an explicit override rather than silently changing a
   carried default — the Hub should decide whether to move it.
