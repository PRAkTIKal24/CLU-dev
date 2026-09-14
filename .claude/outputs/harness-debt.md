# harness-debt — experiment-engineer report (C2W4)

Task + acceptance criterion: **fix `memory_gym.byte_ratio_law`'s denominator (P0), land monitor #6's
missing `+eps_acq` half (P0), and publish ONE re-score diff for each — no silent re-scores.**
Status: **done** (D1 · D2 · D3 · D4 landed and verified; **D5 declared NOT-RUN — it requires
`clu_system.py`, which I do not own; see §7**).

## ⭐ RECONCILIATION LIST — FIRST 10 LINES (protocol §5 corollary; `doc-curator-c2w3-sync` consumes §8)
1. ⭐⭐ **Monitor #6's "27 post-repair" is CONFIRMED, not provisional — the PROVISIONAL qualifier LIFTS.**
   `58 pre-repair → 27 loss-half-only → **27** both halves`, **0 cells changed, 0 trips added, 0 removed.**
2. ⛔ **`doctrine-repairs.md` §2.3's "2 recovered false negatives" must NEVER be quoted without the band
   it needs.** It is real, it is exactly the two cells named, and it requires the **resolution** floor
   `1/(n_probed·W) ≈ 4.2e-2` — **6 orders above** the round-off band both legs actually ship. At the
   shipped band it is **0 recoveries**. Quoting "2 added" against the shipped monitor is wrong.
3. ⛔ **The 4 `manifold` C2W1 cells re-score 43.33× → 52.00× (+8.667, +20 %) and their `floor_note`
   2.00× → 2.40×.** 24/28 cells **bitwise unchanged**; measured min ratio **2.2824× unchanged**.
4. ⚠ **Open Hub/Head decision (not mine to take):** whether `eps_acq_rel` should be raised from the
   round-off band to the resolution band. It is a **one-knob, 27 → 29** decision, fully priced in §4.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial / pillar:** **none — instrument repair.** No dividend measured, no claim made.
- **Laundering control:** n/a. The control that matters is the **before/after diff** — every number I
  changed is a number someone already published, and both diffs are printed in full below.
- **Falsifies:** the corrected law failing to reproduce the measured ledger (it does not: **28/28, 0
  ulp**) · any `TRIP → no-trip` flip from a band that only widens the trip condition (there are **none**).
- **Does NOT falsify:** the byte-floor theorem (an accounting identity; only the *shipped formula* and
  one *published sentence* were wrong, both conservatively) · monitor #6's C2W2 dead-band (kept verbatim)
  · the theorist's two predicted recoveries not firing at the shipped band (a **finding**, §4.3).

---

# 1. ⭐ TABLE (a) — the byte-ledger diff, all 28 C2W1 cells

Re-scored **offline** from the recorded C2W1 artifact
(`.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json`); the gym was **not re-run**.
Script `.claude/scratch/harness-debt/d1_byte_law.py`, raw
`.claude/outputs/harness-debt/d1_byte_law_diff.json`.

| cell | n_spec | A | **measured** ratio | published (shipped law) | **corrected law** | Δ | floor_note |
|---|---|---|---|---|---|---|---|
| `overload/ref3@s0` | 0 | 18/17 | 2.2824 | 2.2824 | 2.2824 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/ref8@s0` | 0 | 54/17 | 5.2471 | 5.2471 | 5.2471 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/ref16@s0` | 0 | 108/17 | 9.6941 | 9.6941 | 9.6941 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/load1x@s{0,1,2}` | 0 | 32 | 45.6000 | 45.6000 | 45.6000 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/load1x_ref8@s0` | 0 | 8 | 12.0000 | 12.0000 | 12.0000 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/load1x_ref3@s0` | 0 | 3 | 5.0000 | 5.0000 | 5.0000 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/load1x_shipped@s{0,1,2}` | 0 | 341 | 478.2000 | 478.2000 | 478.2000 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/base@s{0,1,2}` | 0 | 198/17 | 17.1059 | 17.1059 | 17.1059 | **0 (bitwise)** | 2.20× → 2.20× |
| `overload/reach_free@s0` | 0 | 198/17 | 17.1059 | 17.1059 | 17.1059 | **0 (bitwise)** | 2.20× → 2.20× |
| `aggregate/base@s{0,1,2}` | 0 | 192/5 | 54.5600 | 54.5600 | 54.5600 | **0 (bitwise)** | 2.20× → 2.20× |
| `aggregate/tight@s{0,1,2}` | 0 | 192/5 | 54.5600 | 54.5600 | 54.5600 | **0 (bitwise)** | 2.20× → 2.20× |
| `recency/base@s{0,1,2}` | 0 | 192/5 | 54.5600 | 54.5600 | 54.5600 | **0 (bitwise)** | 2.20× → 2.20× |
| ⛔ **`manifold/base@s0`** | **1** | 32 | **52.0000** | **43.3333** | **52.0000** | **+8.6667** | ⛔ **2.00× → 2.40×** |
| ⛔ **`manifold/base@s1`** | **1** | 32 | **52.0000** | **43.3333** | **52.0000** | **+8.6667** | ⛔ **2.00× → 2.40×** |
| ⛔ **`manifold/base@s2`** | **1** | 32 | **52.0000** | **43.3333** | **52.0000** | **+8.6667** | ⛔ **2.00× → 2.40×** |
| ⛔ **`manifold/ridge@s0`** | **1** | 32 | **52.0000** | **43.3333** | **52.0000** | **+8.6667** | ⛔ **2.00× → 2.40×** |

**Totals (exact integer / rational arithmetic, `fractions.Fraction`, 0 ulp):**

| check | result |
|---|---|
| byte decomposition `V_θ = 4N_at(D+2)` · `code = 4Kd` · `launder = 4K(d+m)` · `keys = 4Kd` · `payloads = 4Km` | ✅ **28/28 exact (integers)** |
| **corrected** law `[A(D+2)+d]/(d+m)` == measured `full/launder` | ✅ **28/28 exact (rationals)** |
| **shipped** law `A(D+2)/D + d/D` == measured | ⛔ **24/28** |
| corrected == shipped **BITWISE** on the 24 `n_spectator = 0` cells | ✅ **24/24** (the fix is written in two-term form on purpose so this holds) |
| published `closed_form_ratio` reproduced bitwise by my re-implementation of the old code | ✅ **28/28** (the diff is not an artefact of my re-implementation) |
| cells changed | **4**, all `n_spectator = 1`, all by **exactly +8.6666666667** |
| floors printed across the 28 cells | before `{2.00, 2.20}` → after `{2.20, 2.40}` |
| **measured min ratio (the never-quote-adjacent `17.11×`/`2.28×` anchors)** | **2.2824×, unchanged** — it comes from `ByteAccount`, which I did not touch |

⭐ **Direction: entirely conservative.** The store costs **more** relative to the table than was
published; **no published claim was inflated**, and the theorem's *reuse licence* (`PREREG-Bprime.md`
§7) stands. `bprime-rivals` does **not** re-measure it.

## 1.1 What landed (D1)
`chlu/experiments/memory_gym.py::byte_ratio_law` now returns
`A·(D+2)/(d+m) + d/(d+m)`, `D = d + m + n_spectator`. The docstring states the corrected law, the
**24/28-vs-28/28 history**, the invisibility of the bug at `n_spectator = 0`, and the conservative
direction. ⭐ **The `floor_note` needed NO code edit:** it lives in `chlu/experiments/exp_memory_gym.py:551`
(not in `memory_gym.py` as theorist C1's location line said) and is *derived* by calling
`byte_ratio_law(1.0, …)`, so it corrects itself. **I therefore touched no file outside my ownership.**

## 1.2 What landed (D2 — the missing regression test)
`tests/test_memory_gym.py`:
- `test_byte_ratio_law_matches_the_measured_ledger` is **parametrised over `n_spectator ∈ {0, 1}`**
  (it previously passed `0` literally) and now also asserts the ledger **structurally, as integers**:
  `full == 4[N_at(D+2) + K·d]` and `launder == 4K(d+m)`.
- `test_cell_reports_all_three_harness_native_controls_and_a_byte_ledger` gains **`manifold`** — the
  only family with a spectator dim, and the reason the bug survived three waves of green tests.
- **New** `test_byte_ratio_law_is_correct_on_a_spectator_dim` pins **52.00×**, the **2.40×** floor, the
  pre-erratum **43.33×** it replaces, and the **`n_spec = 0` bit-identity gate** (`float.hex()` equality
  against the pre-erratum expression, for `A ∈ {3, 8, 32, 341, 198/17, 192/5}` — the C2W1 budgets).
- ⛔ **Theorist C3 (`chlu/eval/dividend.py::byte_account`) NOT touched — it is `bprime-rivals`' file
  this wave.** I assert the identity on the gym side only, reading `ByteAccount`'s public fields. **No
  routing request needed:** the gym-side assertion did not require the `dividend.py` change.

---

# 2. ⭐ TABLE (b) — the monitor-#6 trip-state diff at three predicate settings

**Method = the C2W2 D4 pattern, verbatim: the store was NEVER re-run.** The 112 recorded
monitor-#6 readings from `phi-particle-head`'s 28-cell re-score
(`.claude/outputs/phi-particle-head/exp_phi_particle_gym-rescore_seed0.json`, each carrying
`slope_write_loss`, `slope_acq` and the loss band `eps` in force) were re-scored offline through the
new free function. Script `.claude/scratch/harness-debt/d4_monitor6_rescore.py`, raw
`.claude/outputs/harness-debt/d4_monitor6_diff.json`.

⚠ **One honest methodological note.** The recorded readings pre-date this wave and do **not** carry
`scale_acq = max|acq|` over the window. `acq` is the self-probe **acquisition rate, a proportion in
[0,1]**, so `scale_acq ≤ 1` and therefore `eps_acq = eps_acq_rel·scale_acq ≤ **1e-9 exactly**` at the
shipped default, for every reading. I re-scored at that **upper bound** — the most generous band the
shipped default can produce. A reading that does not flip at the bound cannot flip at its true band, so
the null below is a null full stop, not an artefact of the missing field.

| cell | pre-repair (`eps=0, eps_acq=0`) | **loss half only** (C2W3 shipped ⇒ the published 27) | ⭐ **both halves (mine)** | changed? | *[sensitivity, NOT shipped]* `eps_acq = 1/24` |
|---|---|---|---|---|---|
| `overload/ref3@s0` | 0 | 0 | **0** | — | 0 |
| `overload/ref8@s0` | 1 | 1 | **1** | — | 1 |
| `overload/ref16@s0` | 2 | 2 | **2** | — | 2 |
| `overload/load1x@s0` | 2 | 1 | **1** | — | 1 |
| `overload/load1x@s1` | 2 | 1 | **1** | — | 1 |
| `overload/load1x@s2` | 1 | 0 | **0** | — | 0 |
| `overload/load1x_ref8@s0` | 3 | 0 | **0** | — | 0 |
| `overload/load1x_ref3@s0` | 3 | 0 | **0** | — | 0 |
| `overload/load1x_shipped@s0` | 4 | 1 | **1** | — | 1 |
| `overload/load1x_shipped@s1` | 2 | 1 | **1** | — | 1 |
| `overload/load1x_shipped@s2` | 3 | 0 | **0** | — | 0 |
| ⚠ `overload/base@s0` | 2 | 2 | **2** | **— (no change)** | **3** ← theorist's recovery, band-gated |
| `overload/base@s1` | 0 | 0 | **0** | — | 0 |
| `overload/base@s2` | 3 | 3 | **3** | — | 3 |
| `aggregate/base@s0` | 1 | 0 | **0** | — | 0 |
| `aggregate/base@s1` | 1 | 1 | **1** | — | 1 |
| `aggregate/base@s2` | 1 | 0 | **0** | — | 0 |
| `recency/base@s0` | 1 | 0 | **0** | — | 0 |
| `recency/base@s1` | 2 | 1 | **1** | — | 1 |
| `recency/base@s2` | 1 | 0 | **0** | — | 0 |
| `manifold/base@s0` | 4 | 1 | **1** | — | 1 |
| `manifold/base@s1` | 2 | 1 | **1** | — | 1 |
| `manifold/base@s2` | 3 | 0 | **0** | — | 0 |
| `aggregate/tight@s0` | 4 | 3 | **3** | — | 3 |
| `aggregate/tight@s1` | 4 | 3 | **3** | — | 3 |
| `aggregate/tight@s2` | 2 | 2 | **2** | — | 2 |
| `manifold/ridge@s0` | 2 | 1 | **1** | — | 1 |
| ⚠ `overload/reach_free@s0` | 2 | 2 | **2** | **— (no change)** | **3** ← theorist's recovery, band-gated |
| **TOTAL (112 readings)** | **58** | **27** | ⭐ **27** | **0 cells changed** | *29* |

**Gates:**

| gate | result |
|---|---|
| the free function reproduces the **recorded pre-repair** flag (`eps = eps_acq = 0`) | ✅ **112/112** |
| the free function reproduces the **recorded post-repair** flag (loss half only, `eps_acq = 0`) | ✅ **112/112** — ⭐ **this is the blocking `eps_acq_rel = 0` bit-identity gate, and it is green** |
| `TRIP → no-trip` flips at the shipped band (**must be 0** — falsifier) | ✅ **0** |
| `TRIP → no-trip` flips at *any* band up to `eps_acq = 1.0` (must be 0 by monotonicity) | ✅ **0** |
| **every other monitor bit-identical** | ✅ **by construction** — the diff re-scores *recorded* readings; nothing else was recomputed. The underlying run's own diff already reports `monitors_changed_other_than_6: []` across all 28 cells |

## 2.1 ⭐⭐ THE SENTENCE THAT LIFTS THE PROVISIONAL QUALIFIER (stated once, with its diff beside it)

> **Monitor #6's post-repair trip count on the 28-cell C2W1 gym is 27 — CONFIRMED, no longer
> PROVISIONAL. The C2W4 `+eps_acq` half changes it by ZERO: `58 pre-repair → 27 loss-half-only → 27
> both halves`; 0 cells changed, 0 trips added, 0 trips removed, every other monitor bit-identical.**

The `"58 trips"` never-quote rule is unaffected: it is still *"58 **pre-repair**"*, and the artefact
count is still **31 of 58**, not 29.

---

# 3. What landed (D3)

`chlu/core/monitors.py`:
- `objective_divergence_predicate(slope_loss, slope_acq, eps=0.0, **eps_acq=0.0**)` — still a **free
  function** (recorded readings stay re-scorable offline), fourth argument defaults to `0.0` so **every
  existing call site is unchanged**. Predicate: `slope_loss < -eps and slope_acq <= +eps_acq`.
- `ObjectiveDivergenceMonitor.__init__` gains `eps_acq_rel = 1e-9`, `eps_acq_floor = 1e-30` — **built
  the same relative way as the loss band** (`eps_acq = eps_acq_rel · max|acq|` over the window), with
  `eps_acq_rel = 0.0` restoring the C2W2–C2W3 predicate exactly.
- Readings now carry `eps_acq_dead_band`, `acq_scale`, `eps_acq_rel` **and `tripped_loss_half_only`**
  beside the existing `tripped_pre_repair` — so a *future* wave can re-score this wave's diff the same
  way this wave re-scored C2W2's, from the artifact alone.
- Docstrings (class + module monitor table) state both halves, the 31/58 count, and — explicitly — that
  the shipped band is a **round-off** floor and **not** `doctrine-repairs` §2.3's **resolution** floor.

Four new tests in `tests/test_monitors.py`: the recovered false negative (with the loss-half-only
predicate asserted `False` on the same reading) · a genuine acquisition rise **not** swallowed ·
⭐ the blocking `eps_acq_rel = 0` gate over **200 random 4-point windows spanning 20 decades**, asserting
`off.tripped == on.detail["tripped_loss_half_only"]` **and** monotonicity (`not (off and not on)`) ·
the predicate's monotonicity in `eps_acq` with the `+7.84e-4` cell pinned at both bands.

---

# 4. ⛔ THE FINDING: the two predicted recoveries, and the band that gates them

**They do not materialise at the shipped band, and I did not tune the band to make them.**

| | |
|---|---|
| readings whose **loss** leg passes (`slope_loss < -eps`) | **29** of 112 |
| …of which `slope_acq > 0` — i.e. **every** reading that could possibly flip | ⭐ **exactly 2** |
| which ones | **`overload/base@s0`** (`slope_loss = -4.0293e-2`, `slope_acq = +7.8431e-4`, `eps = 3.360e-10`) and **`overload/reach_free@s0`** (`slope_loss = -3.9900e-2`, same `slope_acq`) |
| ⭐ | **exactly the two cells `doctrine-repairs.md` §2.3 named, and no others** — the theorist's *cell-level* prediction is confirmed |
| the shipped band's reach | `eps_acq = eps_acq_rel·max|acq| ≤ **1e-9**` ⇒ **6.9 orders below** `7.84e-4` ⇒ **0 flips** |
| the band the prediction needs | `doctrine-repairs` §2.3's **resolution** floor `max(8u·Y_a, 1/(n_probed·W)) ≈ 1/24 = 4.17e-2` ⇒ **2 flips** |

**Decision-invariance sweep** (flip count vs `eps_acq`, all 112 readings — the analogue of C2W2's
"12.8 orders" argument):

| `eps_acq` | 0 | 1e-18 | 1e-12 | **1e-9 (shipped UB)** | 1e-6 | 1e-4 | 3.7e-4 | 1e-3 | 1e-2 | **4.17e-2 (resolution)** | 1e-1 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trips | 27 | 27 | 27 | ⭐ **27** | 27 | 27 | 27 | 29 | 29 | *29* | 29 | 29 |
| added | +0 | +0 | +0 | ⭐ **+0** | +0 | +0 | +0 | +2 | +2 | *+2* | +2 | +2 |

The single transition is at `slope_acq = 7.8431e-4`. **The decision is invariant over ~9 orders below it
and over the whole remaining range above it (up to `eps_acq = 1`, where the band saturates the
proportion) — so the count is 27 or 29 and nothing else can happen.** The choice between them is a
**doctrine** choice, not a tuning choice.

## 4.1 ⚠ The open decision I am NOT taking (Hub/Head)
The two bands answer different questions:
- **round-off band (shipped, both legs):** *"is this slope distinguishable from zero in float64?"* — the
  C2W2 half answers this, and my half answers it symmetrically. This is what my task file specified.
- **resolution band (`doctrine-repairs` §2.3, acq leg only):** *"is this slope larger than the
  quantity's own quantum `1/n_probed`?"* — a **stronger and different** claim; `acq` is a proportion over
  `n_probed` self-probes, and a slope of `7.8e-4` extrapolates to `2.4e-3` of a probe over the window,
  i.e. **below one probe**. That is a defensible position, and it is what the theorist derived.
Note the asymmetry if it were adopted: §2.3 also specifies a resolution floor on the **loss** leg
(`1e-3·Y_l/W`), which is **not** shipped either. **Adopting one leg's resolution floor and not the
other's would re-create exactly the half-repair this task exists to close.** ⇒ I shipped the symmetric
round-off pair; raising both to resolution floors is a scoped follow-up with a priced consequence
(27 → 29 on the acq leg; the loss leg is unpriced and would need its own re-score).

---

# 5. PREREG scorecard
`.claude/outputs/harness-debt/PREREG.md`, written and committed **before** any re-score was computed.

| # | registered | measured | verdict |
|---|---|---|---|
| A1 | corrected law exact **28/28** (rationals, 0 ulp) | **28/28** | ✅ |
| A2 | shipped law **24/28**, failures = the 4 `manifold` cells | **24/28**, exactly those 4 | ✅ |
| A3 | `43.33 → 52.00`, shift **+8.6667** identical on the `manifold/base` cells (`ridge` free) | **+8.6666666667 on all four**, `ridge` included | ✅ (ridge shares `A = 32`) |
| A4 | floor `2.00× → 2.40×` on those 4; `2.20×` elsewhere unchanged | exactly that | ✅ |
| A5 | ⭐ 24 `n_spec=0` cells **bitwise** identical (registered as `==`, `1e-16`-equal would be a *reported deviation*) | **24/24 bitwise** (`float.hex()` equality) | ✅ |
| A6 | measured min ratio **2.28×** unchanged | **2.2824×** unchanged | ✅ |
| B1 | ⭐ blocking: `eps_acq_rel = 0` reproduces the shipped predicate bit-for-bit on all 112 readings | **112/112** (and pre-repair **112/112**) | ✅ |
| B2 | ⭐⭐ **0 cells flip at the shipped default; count stays 27** | **0 flips, 27** | ✅ |
| B3 | ⛔ the theorist's 2 recoveries do **NOT** materialise at this band (`7.84e-4` ≫ `1e-9`) | **they do not** | ✅ (registered as a *prediction*, so this is a confirmation, not an excuse) |
| B4 | sensitivity at the resolution band: **the 2 named cells DO flip**; unknown whether others do | **exactly those 2, and provably no others** (only 2 flip candidates exist in the whole record) | ✅, sharper than registered |
| B5 | **no `TRIP → no-trip` anywhere, at any `eps_acq`** | **0** | ✅ (falsifier did not fire) |
| B6 | no other monitor moves | **none** (by construction; underlying artifact reports `monitors_changed_other_than_6: []`) | ✅ |
| C | D5 registered as *conditional and likely NOT-RUN* | **NOT-RUN**, reason §7 | — |

**Nothing in the PREREG failed.** The two predictions that could most easily have been retro-fitted
(B2/B3) were the ones I registered *against* the theorist's published expectation, and the measurement
went the way I registered.

---

# 6. Flag provenance (mandatory)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/harness-debt` off local `main @ d4f56c8` (worktree `../CHLU-debt`, slot 3/3) |
| commits | `8e122bf` (D1+D2), `cf244a4` (D3+D4 code/tests), `f7e28a8` (docstring erratum 29→31) |
| environment | **reused the MAIN venv** (`/Users/user/Desktop/CHLU/.venv/bin/python`, `PYTHONPATH=/Users/user/Desktop/CHLU-debt`, cwd in the worktree) — **no `uv sync` in the worktree**, so the w6 JAX-drift hazard does not apply. **Resolved JAX = 0.9.0** (same interpreter as `main`). |
| my own runs | **no store was written and no gym cell was run** by me except through `pytest`. Both diffs are **offline re-scores of recorded artifacts.** |
| byte-ledger source | `.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json` — C2W1 gym, `base_seed = 0`, **`quick = False`**, 28 cells = 4 families × arms × seeds {0,1,2} per the C2W1 plan; `sd_convention = sample sd (ddof=1)`; `d = addr_dim = 4`, `m = payload_dim = 1`, `n_spectator = 1` on `manifold` only (`FAMILY_DEFAULTS["manifold"]`), `stage_admission = True` all cells, `d_safe_override = 0.58` (`overload`) / `0.32` (`aggregate/tight`), `stage_lifetimes = True, leak = 0.06` (`recency`), `ridge_write = True` (`manifold/ridge`) |
| monitor-#6 source | `.claude/outputs/phi-particle-head/exp_phi_particle_gym-rescore_seed0.json` — `part = gym-rescore`, `seed = 0`, **`quick = 0`**, 28 cells / **112 applicable readings**, `wall = 597.8 s`; monitor #6 `window = 3` (n = 4-point windows), `eps_rel = 1e-9`, `eps_floor = 1e-30` |
| my non-default flags | `eps_acq_rel = 1e-9` (**new default**), `eps_acq_floor = 1e-30` (**new default**); re-scores evaluated at `eps_acq ∈ {0, 1e-18 … 1.0}` as tabulated, with `1e-9` as the shipped **upper bound** (see §2's note) and `1/24` labelled **sensitivity, not shipped** |
| seeds | none drawn by me except `numpy.random.default_rng(0)` inside the new `eps_acq_rel = 0` gate test (200 windows) |
| exactness | §1 uses `fractions.Fraction` (integer/rational, 0 ulp) and `float.hex()` for bit-identity; §2 uses the shipped free function on recorded float64 fields |

---

# 7. ⛔ DECLARED NOT-RUN (never reported as a null)
**D5 — monitor #2's domain guard (theorist C8, P1): NOT RUN.** Reason, with evidence:
`SettleArgminMonitor` does not compute `r_i` at all. It consumes `ctx.get("reads", "covered")`, and
`covered` — together with the `sep/2` inradius proxy — is computed at
**`chlu/core/clu_system.py:1309`**:
```python
r_i = 0.5 * _min_separation(centers)
cov = np.min(np.linalg.norm(q0n[:, None, :] - centers[None, :, :], axis=-1), axis=1) <= r_i
```
`λ_min` is **not in that scope at all** (it is produced later, in `certificates()`), so the domain guard
(`r_i := 0` and INAPPLICABLE wherever `λ_min,i ≤ 0`), the corrected inradius and the capture-radius leg
**all require editing `clu_system.py`, which §7 of my task file makes read-only to me**. ⛔ **STOP and
report, per the ownership rule.** Two further reasons not to force it: `bprime-c6` is re-locating
`B = 0.33` with the corrected inradius **this wave** (their number must land first), and the silent
`max(λ, 1e-9)` clamp at `clu_system.py:906` would have to be adjudicated at the same time.
**Routing request to the Hub:** D5 needs an owner who holds `clu_system.py`, sequenced after
`bprime-c6` reports.

**Also NOT RUN (declared):** any re-run of the gym store (forbidden by the task — both diffs are
offline) · theorist **C3** (`dividend.py::byte_account`'s structural identity) — `bprime-rivals`' file;
**no routing request needed**, the gym-side assertion did not require it · any change to `B = 0.33` ·
any edit to `PREREG-Bprime.md` (Hub ruling 6: the correction is an erratum, not a revision).

---

# 8. ⭐ DOCUMENTS THAT NOW NEED UPDATING (`doc-curator-c2w3-sync` consumes this list directly)

| # | file · section | change |
|---|---|---|
| 1 | **`claims_matrix.md` §0** (never-quote list) | ⭐ **Retire the PROVISIONAL qualifier on monitor #6's "27 post-repair"** — it is **CONFIRMED** (§2.1). Keep *"58 trips"* never-quote **without "pre-repair"**, and keep the artefact count at **31 of 58**. **Add:** *"2 recovered false negatives" is never-quote without the band* (`doctrine-repairs` §2.3's resolution floor `≈4.2e-2`); at the **shipped** band it is **0**. |
| 2 | **`claims_matrix.md` §0 + wherever the byte law appears** | *"verified to 1e-9 in all 28 cells"* → **24/28 under the pre-erratum formula; 28/28 exact under the corrected law `[A(D+2)+d]/(d+m)`** (this is reconciliation 2, already the curator's; my measurement confirms it cell-by-cell, table (a)) |
| 3 | **`ERRATA-Bprime.md`** (new, per Hub ruling 6 — beside the un-edited `PREREG-Bprime.md` §7) | file the byte-law erratum with **table (a)** as its evidence: 24 cells bitwise unchanged, 4 `manifold` cells `43.33× → 52.00×` (+8.667, +20 %), floor `2.00× → 2.40×`, **conservative direction**, reuse licence **stands**, `bprime-rivals` does **not** re-measure |
| 4 | **`memory-gym-v0.md` §2 / §3.1 / its PREREG-B1** | the byte-law sentence (as #2) **and** the `manifold` cells' published ratio/floor (`43.33/2.00` → `52.00/2.40`). ⚠ Its **R2 "29 of 58"** is still live in that file — it is **31** |
| 5 | **`track2-admissibility.md` §2** | the byte-law sentence (as #2) |
| 6 | **charter `§A2.3`** (Advisor's annotation — curator **verifies, does not edit**) | confirm the annotation now matches the *measured* table (a), incl. the **2.40×** floor |
| 7 | **`negative_results.md`** | the N-entry carrying monitor #6's dead-band result: **"×58 pre-repair → ×27 after the C2W2 loss half → ×27 after the C2W4 acq half; 0 new trips at the shipped band; all other monitors bit-identical"**; the artefact count **31**. New candidate entry: *a half-landed monitor repair moved the trip count in one direction only for two waves, and the missing half turned out to move it by zero — the asymmetry was the defect, not the count* |
| 8 | **`negative_results.md` / `future_work.md`** | new candidate: *the byte-law denominator bug was invisible to every test because no test exercised a spectator dim* — the **test-coverage** lesson, not just the formula |
| 9 | **`doctrine-repairs.md` §2.3** (registry-side annotation; the file itself is a spoke output) | annotate that its **`eps_acq` is a RESOLUTION floor**, that the shipped `eps_acq_rel = 1e-9` is a **round-off** floor, and that its **"2 added"** is correct **only** at the resolution band — otherwise the sentence reads as a prediction that failed, which it is not |
| 10 | **`handover_context.md` §7 / §10** | reconciliations **1** and **4** are **CLOSED** (this report); reconciliation **2** is the curator's and is unblocked by table (a); **OQ-3** (`memory_gym.py` has no owner) is **CLOSED** |
| 11 | **`handover_context.md` §3 (config defaults)** | new `ObjectiveDivergenceMonitor` kwargs `eps_acq_rel = 1e-9`, `eps_acq_floor = 1e-30`; new reading fields `eps_acq_dead_band`, `acq_scale`, `eps_acq_rel`, `tripped_loss_half_only`; `objective_divergence_predicate` gains a 4th arg `eps_acq = 0.0` (back-compatible) |

⛔ **Sites that must NOT be touched:** `PREREG-Bprime.md` (Hub ruling 6) · the `17.11×` / `2.2824×`
**measured** anchors (unchanged) · `sep/2` and `λ_min > 0` never-quote entries (untouched by me).

---

# 9. Verification — commands and actual output

```
# D1/D2 offline byte-ledger diff (exact rational arithmetic, 28 recorded cells)
PYTHONPATH=/Users/user/Desktop/CHLU-debt .venv/bin/python .claude/scratch/harness-debt/d1_byte_law.py
  byte decomposition exact (integers)    : 28/28
  CORRECTED law == measured (rationals)  : 28/28
  SHIPPED   law == measured (rationals)  : 24/28
  before==after BITWISE                  : 24/28
  changed cells : 4 -> ['manifold/base@s0','manifold/base@s1','manifold/base@s2','manifold/ridge@s0']
  deltas on changed cells : [8.6666666667]
  published ratios on changed cells : [43.3333] -> [52.0]
  floors before : [2.0, 2.2]   floors after : [2.2, 2.4]
  measured min ratio (unchanged) : 2.2824
  published closed_form == recomputed before (bitwise): 28/28

# D4 offline monitor-#6 re-score (112 recorded readings, store NOT re-run)
PYTHONPATH=/Users/user/Desktop/CHLU-debt .venv/bin/python .claude/scratch/harness-debt/d4_monitor6_rescore.py
  free fn reproduces recorded PRE  (eps=eps_acq=0) : 112/112
  free fn reproduces recorded POST (loss half only): 112/112
  TRIPS  pre-repair           : 58
  TRIPS  loss half only       : 27
  TRIPS  BOTH halves SHIPPED  : 27
  TRIPS  [sensitivity] 1/24   : 29
  cells changed by the SHIPPED acq band : 0 []
  readings TRIP -> no-trip (shipped band, MUST be 0)  : 0 []
  readings whose LOSS leg passes : 29 ; of which slope_acq > 0 : 2
  smallest/largest positive slope_acq among them : 7.843137e-04 / 7.843137e-04

# lint
PYTHONPATH=... .venv/bin/python -m ruff check chlu/ tests/
  All checks passed!
```

## 9.1 ⭐ Full suite — **1068 passed, 0 failed, 31 warnings, 1236.72 s (20:36)**

```
PYTHONPATH=/Users/user/Desktop/CHLU-debt /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q -p no:cacheprovider
  1068 passed, 31 warnings in 1236.72s (0:20:36)
```

**Every delta accounted for: 1061 (C2W3 close) + 7 = 1068. Zero regressions, zero deletions.**

| where | test | Δ |
|---|---|---|
| `tests/test_memory_gym.py` | `test_byte_ratio_law_matches_the_measured_ledger` — was 1 case, now parametrised over `n_spectator ∈ {0, 1}` | **+1** |
| `tests/test_memory_gym.py` | `test_byte_ratio_law_is_correct_on_a_spectator_dim` — **new** (52.00×, 2.40× floor, bit-identity gate) | **+1** |
| `tests/test_memory_gym.py` | `test_cell_reports_all_three_harness_native_controls_and_a_byte_ledger` — was `{aggregate, recency}`, now `+ manifold` | **+1** |
| `tests/test_monitors.py` | `test_monitor6_acq_dead_band_recovers_a_false_negative` — **new** | **+1** |
| `tests/test_monitors.py` | `test_monitor6_acq_dead_band_does_not_swallow_a_real_acquisition_rise` — **new** | **+1** |
| `tests/test_monitors.py` | `test_monitor6_eps_acq_zero_reproduces_the_loss_half_only_predicate_exactly` — **new** (200 random windows, 20 decades) | **+1** |
| `tests/test_monitors.py` | `test_monitor6_predicate_is_monotone_in_eps_acq` — **new** | **+1** |
| | **total** | **+7** |

⚠ Two pre-existing tests were **modified but not replaced** and still pass unchanged in intent:
`test_monitor6_dead_band_does_not_trip_on_a_numerically_zero_slope` and
`test_monitor6_still_trips_on_a_genuine_divergence` — **docstring-only** edits correcting the live
"29 of 58" erratum to **31** (commit `f7e28a8`).

---

# 10. Git footprint
- **Branch:** `agent/experiment-engineer/harness-debt`, base local `main @ d4f56c8`, worktree
  `/Users/user/Desktop/CHLU-debt` (slot 3/3). ⛔ **Not pushed.** `origin` untouched.
- **Commits (3, verified from the MAIN repo — `git -C /Users/user/Desktop/CHLU log --oneline
  main..agent/experiment-engineer/harness-debt`):**
  `8e122bf` D1+D2 byte law · `cf244a4` D3+D4 monitor #6 · `f7e28a8` docstring erratum 29→31.
- **Rebase:** `git rebase main` → *"Current branch … is up to date"* (base never moved; `origin/main` is
  stale by design and was **not** used as a rebase target, per protocol §3.5).
- **Diff vs `main`:** 4 files, **+278 / −50**
  (`chlu/core/monitors.py` +113/−… · `chlu/experiments/memory_gym.py` · `tests/test_memory_gym.py` ·
  `tests/test_monitors.py`).
- **Files touched (all four inside my declared ownership, nothing else):**
  `chlu/experiments/memory_gym.py` · `chlu/core/monitors.py` · `tests/test_memory_gym.py` ·
  `tests/test_monitors.py`.
- **Not touched, deliberately:** `chlu/experiments/exp_memory_gym.py` (the `floor_note` self-corrects) ·
  `chlu/eval/dividend.py` (`bprime-rivals`') · `chlu/core/clu_system.py` (`D5`'s blocker) ·
  `chlu/config.py`.
- **Collisions:** none. Main checkout clean and at `d4f56c8` throughout; `../CHLU-c6` and
  `../CHLU-rivals` (concurrent spokes) never entered.
- ⭐ **Worktree `../CHLU-debt` REMOVED — slot 3/3 is FREE for `cluformer-pilot`.** Removal was done
  **after** verifying from the main repo that the branch ref carries all three commits, and re-verified
  after removal (`git log --oneline main..agent/experiment-engineer/harness-debt` still lists
  `f7e28a8 · cf244a4 · 8e122bf`). Worktree was clean (`status --untracked-files=all` empty) at removal.
- ⛔ **Not pushed anywhere.** `origin` untouched; `clu-dev` untouched. Branch left for Hub review.

---

# 11. Open questions / follow-ups / risks
1. ⚠ **The `eps_acq` band decision (§4.1)** — round-off (shipped, 27) vs resolution (29). Needs a Hub/
   Head ruling; if resolution is chosen, the **loss** leg's resolution floor must be priced in the same
   task or the half-repair returns.
2. **D5 needs an owner who holds `clu_system.py`,** sequenced after `bprime-c6`'s `B` re-location (§7).
3. ⚠ **The C2W1 artifact on disk still prints the old `closed_form_ratio` and `floor_note`** for the 4
   `manifold` cells. I did **not** rewrite it (re-writing a published artifact is exactly the silent
   re-score this task forbids). Any consumer must apply table (a). If the Hub wants the artifact
   regenerated, that is a re-run and a separate decision.
4. **`acq` being a proportion in [0,1] is load-bearing** for the shipped band's `≤1e-9` bound. If a
   future self-probe reports an unbounded acquisition quantity, the bound and this re-score's
   upper-bound argument both need revisiting (the *code* is fine — it uses the real `max|acq|`).
5. The `manifold` family is **struck by FB4 as protocol-invalid**, so the 4 corrected cells are not
   quotable as dividends regardless. The correction matters because the **law** and the **floor** are
   quoted, not because those cells are.

---

## Proposed handover updates (for the Hub)
- **§7 (known issues):** ⛔ **remove** *"monitor #6's repair is half-landed"* — **both halves are landed**
  (`cf244a4`); the post-repair count is **27, CONFIRMED**. ⛔ **remove** *"`memory_gym.byte_ratio_law`
  divides by the store dim"* — fixed (`8e122bf`), 28/28 exact. ⭐ **add:** *"`doctrine-repairs` §2.3's
  2 recovered false negatives require the RESOLUTION band `≈4.2e-2`; the shipped round-off band
  (`≤1e-9`) yields 0. Never quote '2 added' against the shipped monitor."*
- **§3 (config/CLI):** `ObjectiveDivergenceMonitor(eps_acq_rel = 1e-9, eps_acq_floor = 1e-30)` (new,
  default-on, symmetric with `eps_rel`/`eps_floor`); `eps_acq_rel = 0.0` restores the C2W2–C2W3
  predicate exactly. `objective_divergence_predicate` gains `eps_acq = 0.0` as a 4th positional arg
  (back-compatible). New reading fields `eps_acq_dead_band`, `acq_scale`, `eps_acq_rel`,
  `tripped_loss_half_only`. ⚠ These live in `chlu/core/monitors.py`, **not** `chlu/config.py` (standing
  C2 rule).
- **§10 reconciliations:** **1 CLOSED** · **4 CLOSED** · **2 UNBLOCKED** (curator has the measured table)
  · **OQ-3 CLOSED** (`memory_gym.py` now has a fix and a regression test; ownership for future waves
  still to be assigned).
- **New standing lesson (worth a line in §7 or the doctrine):** *a formula bug that is invisible at the
  default geometry will survive every green test unless a test exercises the non-default geometry.* The
  byte law was wrong for three waves because no test set `n_spectator > 0`, and the fix is one
  `@pytest.mark.parametrize`.
