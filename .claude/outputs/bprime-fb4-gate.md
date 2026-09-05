# bprime-fb4-gate — experiment-engineer report (D0 verdict · D1 · D2 · D3)

Task + acceptance criterion: fire or clear **FB4** by the pre-registered D0.1 rule (D0), then land
`store_write_mask_factory` (D1) and SC-1…SC-7 + monitor #3's replacement leg (D2/D3), each **default-off
and bit-identical**, with the C2W1 trip-state diff as the acceptance.
Status: **done — D0, D1, D2, D3 all delivered.** *(D0 was reported to the Hub as an interim first, as
tasked; this file supersedes it and keeps the verdict in the first 10 lines.)*

## ⭐⭐ THE FB4 VERDICT (first 10 lines, as tasked)

> ### ◐ **FB4 PARTIAL = {`overload`, `recency`, `manifold`}. FB4 does NOT fire.**
> ### ⇒ **`bprime-rivals` is released against ONE surviving family: `aggregate`.**

| family (arm) | metric | `M` | `blank` | **`sub`** (which) | **`attn`** | **`S(f)`** | SE(paired) | legs `S≥0.95` / `sub≥attn−2SE` | **verdict** |
|---|---|---|---|---|---|---|---|---|---|
| `overload` (`load1x_shipped`, 478.2×) | `decode` | 1.0 | 0.1667 | **1.0000** (`settle_deleted`) | **1.0000** | **1.0000** | 0.0000 | ✅/✅ | ⛔ **SATURATED** |
| `aggregate` (`base`, 54.56×) | `neg_mae` | 0.0 | −0.4221 | **−0.2081** (`knn2_idw`/`knn2_mean`) | −0.2493 | **0.5068** | 0.0171 | ❌/✅ | ✅ **SURVIVES** |
| `recency` (`base`, 54.56×, pair-restricted) | `acc` | 1.0 | 0.5463 | **1.0000** (`order_aware_pair_+0B`) | 0.4755 | **1.0000** | 0.0221 | ✅/✅ | ⛔ **SATURATED** |
| `manifold` (`base`, 52.0×) | `r2` | 1.0 | −0.0001 | **1.0000** (`echo_+0B`) | 0.0000 | **1.0000** | 0.0000 | ✅/✅ | ⛔ **SATURATED** (predicted) |

**Declared secondary** (the same rule with the arg-min launder excluded from the +0 B reader set, PREREG
§1.1(c)): **PARTIAL = {`recency`, `manifold`}**, survivors **{`aggregate`, `overload`}**, `S(overload)`
falling to **0.6500**. ⭐ **`overload`'s verdict is entirely a function of that one definitional choice,
pre-registered before the run. The Hub must pick which one `bprime-rivals` inherits.**

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R1 — the recency null must be re-stated a *second* time.** The post-fix `−0.0028 ± 0.0619` is
> reproduced here exactly, but the family is now **struck as protocol-invalid**: a +0 B reader of the
> table's row order answers the restricted question at **1.0000 by construction**. Any site reporting a
> recency dividend (pre- or post-fix) inherits this. *(Owner: Hub + doc-curator.)*
> **R2 — `overload@load1x_shipped`'s table launder sits at the metric's exact maximum (1.0000, 3/3).**
> Valid as a **byte-frontier** anchor, not as reader discrimination. *(Owner: Hub / B′ draft.)*
> **R3 — the CLU reads BELOW its own blank store on `recency` even post-fix** (0.4769 vs 0.5463).
> *(Owner: whoever owns the family after the strike — possibly nobody, since it is struck.)*
> **R4 — ⛔ monitor #6's C2W2 repair landed ONLY HALF the theorist's spec.** The one-sided dead band
> (`slope_loss < −eps`) is in; **the `+eps_acq` half is not**, so the two predicted *recovered false
> negatives* did not materialise. Measured at `overload/base@s0`: `slope_loss = −4.03e−2` (genuine),
> `slope_acq = +7.84e−4`, shipped predicate `slope_acq <= 0` ⇒ **still suppressed**, monitor #6 `clear`
> where `doctrine-repairs.md` §2.3 predicted a trip. `monitors.py` is mine this wave and the fix is two
> lines, but landing it would re-score another agent's published post-repair count, so **I did not touch
> it.** *(Owner: Hub — decide, then whoever owns `monitors.py` next wave.)*

**⭐ DIAL DECLARATION (echoed).** **Dial:** *none — instrument validation + harness infrastructure.* No
performance claim, no leaderboard, no dividend. **Laundering control:** this task *is* the control's
audit — every family carries its `+0 B` substitute · settle-deleted/matched-bytes launder · blank-store
control · **and the new full-attention arm**, on identical φ with φ-bytes ledgered on every arm
(asserted in code). **Falsifies:** all four families saturate ⇒ wave pauses (**did not happen**).
**Does NOT falsify:** an individual family saturating; the attention arm winning on a metric-native
family (metric-native-ceiling theorem); the soft certificate showing no dividend (§A2.1 predicts it).

⚠ **The FB4 `attention` arm is a TABLE READER.** It consumes the launder's own `(key, payload)` table
plus a scalar temperature and **never sees a trajectory**. It is **not** `AttentionPsi` and inherits
none of that quarantine. Do not conflate them in either direction.

---

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/bprime-fb4-gate` — **`7eb4d03`** (D0) · **`bc2c49c`** (D1) · **`08f0a37`** (D2/D3), off local `main @ 6ff4c1d` (**unmoved** ⇒ rebase is a no-op; verified from the main repo) |
| worktree | `../CHLU-fb4` (B′ slot 1 of 2); the main checkout was never edited |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0** (no `uv sync` in the worktree) |
| commands | `PYTHONPATH=../CHLU-fb4 .venv/bin/python -u -m chlu.experiments.exp_fb4_gate --save-dir …` · the same with `--quick` · `run_experiment_memory_gym(families=…, arms=['load1x_shipped','base'])` (the C2W1 anchor re-run) · `pytest -q` |
| artifacts | `.claude/outputs/bprime-fb4-gate/{PREREG.md, exp_fb4_gate_metrics.json, exp_fb4_gate.png, run_full.log, postD3/, run_postD3.log, run_gym_anchor.log, sc_demo.json, pytest_full.log}` · `.claude/scratch/bprime-fb4-gate/results/exp_memory_gym_metrics.json` (the re-run anchor) |
| **seeds** | **{0, 1, 2} on every FB4 cell** (12 cells, 0 degenerate, 0 errors, **2.2 min**); the C2W1 anchor re-run is 15 cells × its own seeds |
| statistics | mean ± **sample** sd (`ddof=1`), `SE = sd/√3`; the attention leg uses the **paired** SE of `sub_s − attn_s`; unpaired SEs also emitted |
| **anchors** | `overload` @ `load1x_shipped` (**478.2×**, `atoms_per_item=341` ⇒ 2046 atoms, `d_safe_override 0.58`, `n_offer=capacity=budget=6`, `collision_offer=False`, `consolidate_every=4`, `n_query_per_item=4`) · `aggregate`/`recency`/`manifold` @ `base` (192 atoms, `capacity=budget=6`, `consolidate_every=2`) |
| ⛔ **recency flags** | **`restrict_index_to_pair=True`** (C2W2 D4) + `stage_lifetimes=True`, **`leak=0.06`**; both coverages emitted (§3.3) |
| manifold flags | `n_spectator=1` (⇒ `dim=6`), `deletion=False`, `revisit=False`, `collision_offer=False` |
| store / write / read | the shipped gym path, **imported unchanged**: `DesignFreedomPotential(free_mlp, atoms)`, masked/local write 300 steps Adam(3e-3, wd 1e-4), `σ_addr 0.25`, `σ_pay 0.6`, `margin 0.15`, `barrier 0.2`; read `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, 400+800 steps, `kinetic_mode newtonian_learned` |
| query law | **`σ_q = 0.15` ISOTROPIC**, `payload_tol 0.1`, `sep/σ_q` **9.06** (overload, manifold) / **7.59** (aggregate, recency) |
| langevin / temperature / λ_lyap | **N/A** — deterministic read, `T = 0`, `p₀ = 0`, no Langevin step; the write is `train_memory_landscape`'s static objective, **not** `train.py`/`train_generative.py` |
| attention arm | `softmax(q·kᵀ/(τ√d))`, value-weighted; `τ` grid-fitted (`logspace(−2,2,41)`) on an independent draw of the same query law (`rng = default_rng(seed+20260731)`); ledger = **table bytes + 4 B** |
| **byte ledger** | `full/launder` = **57384/120 B (478.2×)** · **5456/100 B (54.56×)** ×2 · **6240/120 B (52.0×)**. ⛔ **No cell here is a byte-matched dividend** (min ratio measured anywhere: **17.11×**) |
| φ ledger | `phi_family = identity_launch`, **`phi_bytes = 0`**, `phi_id` = content hash of `(q0, keys)`, **identical on all five arms of every cell, asserted in code** (raises `PhiMismatchError`) |
| D1/D2/D3 flags | `store_write_mask_factory = None`, `store_write_mask_kwargs = {}`, `soft_certificate = False` — **all default-off**; D3's monitor-#3 leg is **default-ON** because it is a *repair*, exactly like C2W2's #6 dead band |
| SC constants | `ζ = 0.6` · **`B = 0.33`** (domain **`s/sep ∈ [0.15, 0.30]`**; ⛔ `sep/2` is never-quote as a certified inradius) · `κ = 3`, `η = 0.10`, `ρ_C3 ∈ [1/3, 3]`, `δ_num = 1e−6`, `λ_floor = 0`, capture 32 dirs |
| **NOT-RUN (declared, never a null)** | `overload` at the base atom budget (reconciliation 6: 0/18 admissible incl. the Gaussian control) · the annealed +0 B read variant on the FB4 arms · any rival family (`bprime-rivals`'s) · a *trained* attention reader · a soft-certificate **sweep** over `B` (one demonstration cell only, §5.3) · re-locating `B` with R1's corrected inradius (OQ-A; still owed) · monitor #6's missing `+eps_acq` half (R4 — deliberately not landed) |

---

# PART A — D0: THE FB4 GATE

## A1. What I did
1. **`chlu/eval/fb4_gate.py` (new)** — the attention arm (`attention_logits/weights/read`,
   `fit_attention_temperature`) and the pre-registered rule (`saturation`, `family_saturated`,
   `fb4_verdict`) with its constants **frozen in code and pinned by a test**.
2. **`chlu/experiments/exp_fb4_gate.py` (new)** — the 12-cell runner. It **imports** the gym and
   `eval/dividend.py` and **edits neither** (both outside my ownership), so the write/read/query/score
   path is the shipped one byte-for-byte. Two new pure table readers: the attention arm and
   `order_aware_pair_launder`.
3. **`tests/test_fb4_gate.py` (new)** — 18 tests.
4. **PREREG filed before any measured run**, with the rule verbatim, the four predicted `S(f)`, the
   predicted attention ordering, and the D1/D2 bit-identity predictions.

## A2. How I verified — and the load-bearing fidelity check

| check | command | observed |
|---|---|---|
| smoke | `… exp_fb4_gate --quick --seeds 0` | 4 cells, exit 0 |
| the run | §0 | **12/12 cells, 0 degenerate, 0 errors, 2.2 min** |
| ⭐ **bit-fidelity to the on-disk C2W1 artefact** | per-seed diff vs `memory-gym-v0/exp_memory_gym_metrics.json` | **exact on every shared arm** |
| tests | `pytest tests/test_fb4_gate.py -q` | **18 passed in 9.44 s** |
| φ invariant | in-code assert per cell | 12/12; a deliberate mismatch raises `PhiMismatchError` (tested) |

My runner rebuilds the cell (I may not edit `exp_memory_gym.py`), so fidelity is **measured**:

| cell | arm | C2W1 on disk (s0/s1/s2) | this run |
|---|---|---|---|
| `overload/load1x_shipped` | `full` · `launder` · `blank` · `knn2_idw` | 1.000000/0.958333/0.958333 · 1.0 · 0.166667 · 0.9167/0.5833/0.625 | **identical** |
| `aggregate/base` | `full` · `launder` · `blank` · `knn2_*` | −0.682608/−0.384693/−0.511032 · −0.496261/−0.413103/−0.432255 · −0.438906/−0.404201/−0.423079 | **identical** |
| `manifold/base` | `full` · `blank` · `echo` | 0.000110/−0.019171/−0.521617 · −8.5e−05 · 1.0 | **identical** |
| `recency/base` | **pre-fix** `acc` (emitted as a coverage) | 0.430556/0.200000/0.275000 | **identical** |

And the **C2W2 D4 numbers are independently reproduced**: seed-0 out-of-pair rate **0.1944** (the quoted
19.4 %), trajectory `0.4306 → 0.5556`, launder `0.4861 → 0.5139`, post-fix 3-seed dividend
**−0.0028 ± 0.0619**. Two independent branches, digit-for-digit.

## A3. Findings

### A3.1 The rule, applied
* `S(overload) = (1.0000 − 0.1667)/(1 − 0.1667) = 1.0000`; `attn = 1.0000` on 3/3 (fitted `τ = 0.01`,
  the arg-max limit). ⭐ **The CLU itself reads 0.9722 ± 0.0139 — below three different readers of a
  table costing 1/478th of its bytes.**
* `S(aggregate) = (−0.2081 + 0.4221)/(0 + 0.4221) = 0.5068` — fails the ceiling leg by a wide margin.
  Its target is a convex combination **dropped at construction** if it lands within `payload_tol` of a
  stored payload, so no table reader can be exact. This is the family that survives.
* `S(recency) = (1.0000 − 0.5463)/(1 − 0.5463) = 1.0000`. ⛔ **The pair-restricted `order_aware`
  substitute is exactly 1.0000 on 3/3 seeds**: once the D4 fix makes every arm answer the 2-way
  question, a reader of the table's own row order answers it perfectly at +0 B. **The fix that made the
  family scorable is the same fix that makes it protocol-invalid.**
* `S(manifold) = 1.0000` (`echo` = 1.0000 at +0 B). **Predicted; not news.**

### A3.2 ⭐ The attention arm, per family

| family | fitted `τ` (3 seeds) | `attn` | vs `sub` | reading |
|---|---|---|---|---|
| `overload` | 0.01/0.01/0.01 (grid floor ⇒ arg-max limit) | **1.0000** | **=** | attention rediscovers the arg-min launder and is likewise at the metric's exact maximum |
| `aggregate` | 0.126/0.158/0.100 (**interior optimum**) | **−0.2493 ± 0.0138** | **worse by 0.041** | ⛔ **a pre-registered prediction of mine FAILS**: I predicted `attn ≳ sub`. A *global* softmax over all K rows with one temperature cannot be as local as a hard 2-NN mean. Attention is **not** uniformly the strongest table reader |
| `recency` | degenerate (**τ not identifiable** — a positive scalar cannot reorder two logits) | **0.4755 ± 0.0221** | far worse | the `(key, payload)` table has **no time column** |
| `manifold` | degenerate | **0.0000 exactly** | far worse | the table's spectator column is written **zero**; any convex combination is 0 |

⭐ **The substantive content of FB4:** "everything is at ceiling" is **false** — but on three of four
families *something costing ≤ 4 B* is at the exact maximum, and it is never the CLU.

### A3.3 The recency switch, auditable (both coverages, as tasked)

| seed | out-of-pair rate | `acc` unrestricted **K-way** (⛔ the defect, never a null) | `acc` restricted 2-way (traj) | (point) | `launder` |
|---|---|---|---|---|---|
| 0 | **0.1944** | 0.4306 | **0.5556** | 0.5556 | 0.5139 |
| 1 | 0.3250 | 0.2000 | 0.3375 | 0.3375 | 0.4625 |
| 2 | 0.3750 | 0.2750 | 0.5375 | 0.5375 | 0.4625 |
| **mean** | 0.298 | **0.3019** (defect) | **0.4769** | 0.4769 | 0.4796 |

⚠ Post-fix the **trajectory and point read-outs are identical on every seed** (Δ = 0.0000, 3/3):
C1W1's dead-trajectory-axis result survives the fix.

### A3.4 The byte ledger (two-sided, every arm)

| family | `full_bytes` | `launder_bytes` | `ratio` | attention arm | `phi_bytes` |
|---|---|---|---|---|---|
| `overload@load1x_shipped` | 57 384 | 120 | **478.2×** | 124 B (table **+4 B**) | 0 |
| `aggregate@base` | 5 456 | 100 | **54.56×** | 104 B | 0 |
| `recency@base` | 5 456 | 100 | **54.56×** | 104 B | 0 |
| `manifold@base` | 6 240 | 120 | **52.0×** | 124 B | 0 |

⛔ **No C2W3 cell is a byte-matched dividend; the minimum ratio measured anywhere is 17.11×.** The
falsifier *"the attention arm cannot be given a commensurate ledger"* **did not fire**: one float32
temperature, 4 B, nothing else.

### A3.5 PREREG scorecard

| prediction | outcome |
|---|---|
| `S(overload) = 1.000` [0.93, 1.00], saturated | ✅ **1.0000** |
| `S(aggregate) = 0.507` [0.40, 0.70], not saturated | ✅✅ **0.5068** — to three decimals |
| `S(recency) = 1.000` via a pair-restricted `order_aware`, 1.0000 by construction | ✅✅ **1.0000, 3/3 exactly** |
| `S(manifold) = 1.000` | ✅ **1.0000** |
| **verdict `PARTIAL = {overload, recency, manifold}`, survivor `{aggregate}`** | ✅ **exactly as pre-registered** |
| secondary `PARTIAL = {recency, manifold}`, `S_excl(overload) = 0.650` | ✅ confirmed |
| `attn(overload)` ∈ [0.95, 1.00] | ✅ 1.0000 |
| `attn(aggregate)` ∈ [−0.25, −0.15] **and `attn ≳ sub`** | ◐ **value in range (−0.2493); the ORDERING is REFUTED** |
| `attn(recency)` ≈ 0.50 | ✅ 0.4755 |
| `attn(manifold)` exactly 0.0000 | ✅✅ exactly 0.0 |
| `blank(recency)` ≈ 0.50 | ◐ **0.5463** — above 2-way chance **and above the CLU (0.4769)** |
| D1/D2 flag-off bit-identity (**exact**, not statistical) | ✅✅ **exact, end-to-end** (§B3, §C4) |
| D3 spearman/sign-flip claim (theorist's, not re-measured here) | **NOT RE-RUN** — carried, not confirmed |

**Score: 10 confirmed (3 sharper than predicted) · 2 partial · 0 wrong-direction.** The one refuted
ordering (`attention ≳ kNN`) would have made the instrument look *stronger*; it failed.

### A3.6 ⭐ My objection to the rule (filed alongside the verdict, per the task)
I do not object to the constants or the arithmetic. I object to one thing:

> **The rule cannot distinguish "the family is substitutable" from "the family's *anchor* is
> substitutable".** `overload` saturates only at `load1x_shipped` — the cell chosen because the base
> budget is unusable — and at that anchor the table launder is at the metric's exact maximum precisely
> *because* the anchor was picked where the store finally works. (At 12.0× the same family's launder is
> still 1.000 while the CLU is 0.333, so the family is substitutable everywhere measured — but the rule
> would return the same verdict even if it were not.) **Recommendation, not a change I made:** if
> `overload` survives under the secondary definition, it survives as a **byte-frontier** instrument, not
> a reader-discrimination one, and B′ should say which it is using it for.

### A3.7 Consequences for `bprime-rivals` + design rules for a replacement family
* FB4 does not fire ⇒ the protocol is not invalid ⇒ **the wave does not pause.**
* Primary: **one** surviving family. Six rivals against one surviving synthetic family is a thin
  cross-family audit; the Hub chooses (a) `aggregate` only, (b) + `overload`-as-frontier (secondary), or
  (c) a **new** family built to these rules:
  1. the answer is not recoverable from the table's **row order** (kills `recency`);
  2. the answer is not the query itself or a function of it alone (kills `manifold`/`echo`);
  3. the store's operating point is not one where the arg-min table is at the metric's exact maximum
     (kills `overload@load1x_shipped`);
  4. ⭐ `aggregate` satisfies all three for one generalisable reason — **its target is constructed to be
     absent from the table.** *"The answer is provably not in the table"* is the only property that has
     survived a +0 B audit in two waves (C2W1 0-for-4, C2W3 1-of-4).

---

# PART B — D1: `store_write_mask_factory` (rider A)

## B1. The defect, stated precisely (it is not the one the task file describes, and the difference matters)
`store_potential_factory` lets a new family supply its own potential. The task says an unmasked **leaf**
breaks C3 locality. Measured: `train_memory_landscape` trains **only `V.learned`**
(`trainable_filter`), so a leaf *outside* `.learned` is never written at all — it cannot break anything.
**The real hazard is a leaf *inside* `.learned` that is not one of `centers`/`log_width`/`amp`**, which
is exactly what `atom_write_mask_fn` masks and nothing else. Such a leaf **is** trained and **is not**
masked ⇒ writing item `j` moves a parameter every other item's read depends on. My first toy family
reproduced the task's description and the test failed *because the hazard was not real in that form*;
the shipped test now instantiates the real one.

## B2. What landed
* `CluSystemConfig.store_write_mask_factory: Optional[str] = None` + `store_write_mask_kwargs`, resolved
  by **`resolve_store_write_mask_factory`** — the same `pkg.module:attr` mechanism, **mirrored error
  messages** (`ValueError` on a bad path, `TypeError` on a non-callable).
* Invoked `factory(cfg=, store=, slot=, default_mask_fn=, **kwargs) -> (updates -> updates) | None`, so a
  family **composes with** the shipped row mask instead of replacing it.
* Module docstring gains seam **(c)** beside (a) and (b).

## B3. Tests (5 new, in `tests/test_clu_system.py`, 32 passed)

| test | asserts |
|---|---|
| `…an_unmasked_shared_leaf_BREAKS_C3_locality` | the toy family's `shared_tilt` **is not** bit-identical after writing another item (the atoms still are) — the failure, asserted |
| `…the_familys_own_mask_RESTORES_C3_locality` | ✅ the **passing partner**: same family, same write, its own mask ⇒ `shared_tilt` **bit-identical** |
| `…freeze_false_reproduces_the_break_through_the_SAME_code_path` | a controlled comparison: the hook is not what fixes it, the mask is |
| `…defaults_to_None_and_is_inert` | ⛔ **blocking**: absent from `as_flag_table()`, and a pass-through factory gives a **bit-identical store** |
| `…rejects_a_bad_import_path` | the two error types |

---

# PART C — D2/D3: SC-1…SC-7 + monitor #3's replacement leg (rider B)

## C1. What landed (`chlu/core/soft_certificate.py`, new; wired in `clu_system.py`/`monitors.py`)

| item | landed as | note |
|---|---|---|
| **SC-1** | `d_safe = ζ·sep_expected` (`ζ = 0.6`), `sep_expected` = min pairwise separation of the **designed site set** (independent of `s_max`, `σ_q` — the whole point). `R_cert = 2s_max + κ′σ_q` **still computed, still reported, no longer the gate** | ⚠ an explicit `d_safe_override` **still wins** and is reported as the legacy path — the override **was** the soft certificate, undeclared, and is now retired by declaration rather than silently overruled |
| **SC-2** | `cert_margin = sep_after − R_cert` and `deficit_rel` logged on **every admitted write**; `R_cert` is computed even in shipped runs | reporting only |
| **SC-3** | `deficit_rel ≤ B` (**`B = 0.33`**) **and** `mean ≤ B/2`; both legs reported | ⛔ **a TRIP of monitor #3, never a refusal** — asserted by a test that checks the write still lands |
| **SC-4** | (i) `λ_min,i > λ_floor` per site; (ii) the C3 calibration leg (= D3); (iii) monitor #2's corrected-radius clause | (i)/(ii) landed; (iii) is R1's corrected inradius and is **NOT RUN here** (it is the theorist's, and it is the prerequisite for re-locating `B`) |
| **SC-5** | the give-up statement travels **in the artifact** (`SC5_STATEMENT`) | pinned by a test |
| **SC-6** | `λ_min > 0` **AND a measured capture radius** (32-direction bisection, min over directions) `≥ σ_q` | ⛔ **INAPPLICABLE, never "passed", when the basin was not measured**; a test encodes the counter-example *basin 0.000 at `λ_min = +0.910`* |
| **SC-7** | **falsifier, not code** — recorded verbatim: *if a shared/factored store's wells cannot hold `λ_min > λ_floor` at any admissible `B`, basin interaction and non-degeneracy are genuinely disjoint and that is a Head escalation.* On the theorist's grid they are **not** disjoint: at `B = 0.33`, `λ_min = +3.19` with `ρ_ex = 0.294` | in the report **and** in `soft_certificate.py` |
| **D3** | monitor #3 leg (ii) `corr(gate_margin, post_write_drift)` **retired to a reported diagnostic that may not trip** (annotated with its `d/s ≳ 4` domain; the gym runs at ~1.9–2) → replaced by `ρ_C3 = median(Δ/B) ∈ [1/3, 3]` and `P[Δ > 3B] ≤ 0.10`, **INAPPLICABLE below 3 qualifying pairs** | **fix S3 lands with it**: ⛔ no `max(λ, 1e−9)` clamp — a non-minimum pair is **disqualified**, not rescued into a perfect certificate. Zero extra cost (`_c3_check` already computed every term; `_lambda_min_at` refactored to return the per-site vector it already built) |

## C2. ⭐ THE TRIP-STATE DIFF — 15 C2W1 anchor cells, re-run on the branch, diffed against the **on-disk** artefact
`.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json` (never a freshly generated baseline).

**Every `full` / `launder` / `dividend` number is bit-identical on all 15 cells.** Exactly **two**
monitors change state anywhere; **all 11 other monitors + M14 are bit-identical**; **zero trips were
added**.

| cell | #3 old → new | #6 old → new (shipped read) | #6 (annealed) | mapped repair |
|---|---|---|---|---|
| `aggregate/base@s0` | **TRIP → clear** | clear → clear | TRIP → clear | **D3** (+ C2W2 #6, annealed) |
| `aggregate/base@s1` | **TRIP → clear** | TRIP → TRIP | clear | **D3** |
| `aggregate/base@s2` | **TRIP → clear** | clear | TRIP → clear | **D3** (+ C2W2 #6) |
| `manifold/base@s0` | TRIP → TRIP (leg i, `f = 0.000`) | **TRIP → clear** | TRIP → clear | **C2W2 #6 dead band** |
| `manifold/base@s1` | TRIP → TRIP (`f = 0.000`) | clear | clear | — |
| `manifold/base@s2` | TRIP → TRIP (`f = 0.000`) | **TRIP → clear** | TRIP → clear | **C2W2 #6** |
| `overload/base@s0` | **TRIP → clear** | clear | TRIP → TRIP | **D3** |
| `overload/base@s1` | **TRIP → clear** | clear | clear | **D3** |
| `overload/base@s2` | **TRIP → clear** | TRIP → TRIP | TRIP | **D3** |
| `overload/load1x_shipped@s0` | TRIP → TRIP (`f = 0.000`) | **TRIP → clear** | TRIP → clear | **C2W2 #6** |
| `overload/load1x_shipped@s1` | TRIP → TRIP (`f = 0.000`) | clear | clear | — |
| `overload/load1x_shipped@s2` | TRIP → TRIP (`f = 0.000`) | **TRIP → clear** | TRIP → clear | **C2W2 #6** |
| `recency/base@s0` | **TRIP → clear** | clear | TRIP → clear | **D3** (+ C2W2 #6) |
| `recency/base@s1` | **TRIP → clear** | TRIP → TRIP | TRIP → clear | **D3** (+ C2W2 #6) |
| `recency/base@s2` | **TRIP → clear** | clear | TRIP → clear | **D3** (+ C2W2 #6) |

**#3: 9 cells TRIP → clear, all attributable to D3 and to nothing else.** In every one of the nine the
fire-rate leg was **in band** (`f = 0.200` / `0.053`) and utilisation was **unchanged to the digit**
(e.g. 0.145 → 0.145), so the old trip was the retired correlation leg alone (`validity` = −0.868, 0.092,
−0.514, 0.181, −0.117, 0.255, −0.853, 0.092, −0.516 — every one below the 0.30 bar). Post-repair the C3
leg is **INAPPLICABLE** (0–2 qualifying pairs: the sites are not minima, which is **#8-N3's** trip, not
#3's — no double-counting) or **applicable and passing** (`ρ_C3` = 0.806 / 0.972 / 0.931 / 0.543, all
inside `[1/3, 3]`, `P[Δ>3B] = 0.000`). ⭐ **This reproduces the theorist's predicted effect exactly:
"18/28 cells become INAPPLICABLE on leg (ii), and the shipped-anchor cells pass with `ρ_C3` computable
rather than a correlation" — the three `load1x_shipped` cells give `ρ_C3` = 0.9502 / 0.8635 / 0.9497.**

**#3 still trips on 6 cells** — every one at `f = 0.000` (leg i, unchanged, and the exact pathology SC-1
exists to break).
**#6: all changes are the already-merged C2W2 dead band** (`phi-particle-head`, in `6ff4c1d`). The four
shipped-read removals — `manifold/base@s0`, `manifold/base@s2`, `overload/load1x_shipped@s0`,
`overload/load1x_shipped@s2` — are **exactly four of the theorist's six named removals** (the other two
cells are not in this 15-cell set). ⇒ **no unexplained trip-state change anywhere. Acceptance met.**
⛔ **But see R4:** the *added* half of that same repair did **not** materialise — §C3.

## C3. ⛔ R4, with its evidence (monitor #6's repair is half-landed)
`doctrine-repairs.md` §2.3 specifies `tripped = slope_loss < −eps_loss and slope_acq <= +eps_acq`, and
calls the `+eps_acq` *"the half nobody asked for"*, predicting **2 recovered false negatives**
(`overload/base@s0`, `overload/reach_free@s0`). The shipped implementation has the one-sided dead band
only (`eps_dead_band`, `eps_rel`), and at `overload/base@s0` I measure

    slope_write_loss = -4.0293e-2   (genuine, 8 orders above eps_dead_band = 3.36e-10)
    slope_acq        = +7.8431e-4   => the shipped `slope_acq <= 0` suppresses the trip
    monitor #6 state = clear        (the repair predicted TRIP)

Two lines in `monitors.py` (mine this wave) would land it, but doing so would **re-score another agent's
published post-repair count mid-wave**, so I did not. **Hub decision.**

## C4. ⛔ The blocking regression: default-off is bit-identical — verified THREE ways
1. **Unit** (`tests/test_soft_certificate.py`): explicit-off vs default-off give a bit-identical written
   store leaf-by-leaf, `d_safe` is the shipped derived radius, and the monitors see **no**
   soft-certificate block at all.
2. **End-to-end, FB4**: all **12** FB4 cells re-run after D1+D2+D3 reproduce the pre-D1 run
   **digit-for-digit** on all five arms (`postD3/exp_fb4_gate_metrics.json` vs `exp_fb4_gate_metrics.json`).
3. **End-to-end, C2W1 anchor**: all **15** re-run gym cells reproduce the on-disk C2W1 `full`/`launder`/
   `dividend` **digit-for-digit** (§C2).

## C5. The soft certificate, demonstrated once (and its price, unsoftened)
One cell, `aggregate/tight@s0` — the arm C2W1 could only run by setting `d_safe_override = 0.32`
"deliberately out of band":

| arm | `d_safe` source | `full` | `launder` | dividend | #3 | SC-3 |
|---|---|---|---|---|---|---|
| shipped | `d_safe_override = 0.32` | −0.559014 | −0.579749 | +0.020735 | **clear** | n/a |
| **soft certificate** | **SC-1** `0.6·sep_expected`, override removed | **−0.559014** | **−0.579749** | **+0.020735** | ⛔ **TRIP** | `max deficit_rel` **1.988**, `mean` **0.981**, `B = 0.33` |

⭐ **Two results in one row.** (i) SC-1 **reproduces the override's admissions bit-for-bit** — the hack
really was the soft certificate, undeclared, and it is now declared. (ii) That arm was running at
**6.0× the declared violation budget and nothing said so**; now monitor #3 trips and the write still
lands, which is precisely SC-3's contract.
⚠ **Caveat that must travel with it:** this cell sits at `s/sep ≈ 1.1`, far **outside** the
`s/sep ∈ [0.15, 0.30]` domain in which `B = 0.33` was located, so the *number* 1.988 is a measurement
and 6.0× is **not** a calibrated statement about how far out of band it is.
⚠ **The price, carried unsoftened (the theorist's, not re-measured here):** decoupling buys `ρ_ex` up to
**6.3×** at a `λ_min` cost of **2.2–6.0×**, and **the dividend in that region stays ≈0**
(+0.0043 … −0.0067). ⛔ **The relaxation is a PRECONDITION, not a result** — §A2.1 predicted exactly
this and the theorist said so. ⚠ **`B = 0.33`'s outer edge was located by a broken proxy**; R1's
corrected inradius is a *prerequisite* for setting `B`, `sep/2` is **never-quote** as a certified
inradius, and re-locating `B` is **NOT RUN** (OQ-A, still owed).

---

## 4. Test suite

* `tests/test_fb4_gate.py` — **18 new**, incl. a test pinning the two pre-registered constants so a later
  edit cannot silently move the gate.
* `tests/test_clu_system.py` — **5 new** (rider A), **32 passed** in the file.
* `tests/test_soft_certificate.py` — **19 new**, incl. the blocking default-off bit-identity test, the
  SC-3 "trips-but-does-not-refuse" test, the S3 no-clamp test, and the SC-6 `λ_min = +0.910` / basin
  0.000 counter-example.
* `ruff check chlu/ tests/` — **All checks passed.** (⚠ `ruff format` is **not** clean repo-wide —
  `exp_memory_gym.py` and `dividend.py` also "would reformat" — so I formatted nothing.)
* ✅ **Full suite on the branch: `1026 passed, 0 failed` in 15 m 58 s** (`pytest -q`, worktree, main
  venv, JAX 0.9.0) — exactly `main`'s **984 + my 42**, **no regressions**. Log:
  `.claude/outputs/bprime-fb4-gate/pytest_full.log`.

## 5. Git footprint

* Branch **`agent/experiment-engineer/bprime-fb4-gate`**, worktree **`../CHLU-fb4`**, base local
  **`main @ 6ff4c1d`** — **verified from the MAIN repo** that the branch ref carries all three commits
  (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/bprime-fb4-gate`), per
  protocol §3.2's wave-4 lesson. `main` is unmoved ⇒ rebase is a no-op. Main checkout clean throughout
  (⚠ `../CHLU-route3` was live the whole time — I never touched the shared checkout). Worktree removed
  **after** that verification; `git worktree list` now shows only `main` + `route3`, and
  `git log -1 agent/experiment-engineer/bprime-fb4-gate` = `08f0a37`.
* Commits: **`7eb4d03`** (D0) · **`bc2c49c`** (D1) · **`08f0a37`** (D2/D3).
* Files touched (7): **new** `chlu/eval/fb4_gate.py`, `chlu/experiments/exp_fb4_gate.py`,
  `chlu/core/soft_certificate.py`, `tests/test_fb4_gate.py`, `tests/test_soft_certificate.py`;
  **edited** `chlu/core/clu_system.py` (config fields ×2 pairs, seam (c) in `_write_item`,
  `resolve_store_write_mask_factory`, SC-1 in `make_controller`, `_c3_check` per-pair records,
  `_lambda_min_per_point` refactor, `_cert_margin_now`, `soft_certificate_state`, `_relax_points`,
  `observe` extras), `chlu/core/monitors.py` (**`VacuousGateMonitor` only** — leg ii′, leg iv, the band
  string), `tests/test_clu_system.py` (append-only + two import lines).
* **Read-only compliance: zero violations.** `chlu/config.py`, `chlu/eval/dividend.py`,
  `chlu/experiments/{memory_gym,exp_memory_gym}.py`, `chlu/core/psi_readout.py`,
  `chlu/eval/rivals/**` — **imported, never edited**. `chlu/core/admission.py` (mine) needed **no**
  change: SC-1 changes the radius handed to the gate, not the gate.
* ⚠ **No CLI hook added.** `chlu/cli/experiment_cmd.py` is in neither my owned nor my read-only list and
  another C2W3 engineer may be adding a hook to it; per §5's stop-rule I did not touch it. The
  experiment runs as `python -m chlu.experiments.exp_fb4_gate`. **Owed: a 2-line additive parser block.**
* Not pushed, not merged. Branch left for Hub review.

## 6. Open questions / follow-ups / risks

1. ⭐ **Hub must choose the primary vs secondary `sub` definition** — it determines whether B′ has one
   surviving family or two. Both were pre-registered; neither was tuned.
2. **One surviving synthetic family is a thin instrument.** §A3.7's four design rules are the cheapest
   route to a fifth family; building one is **NOT RUN** and is not in my scope.
3. **R4** (monitor #6's half-landed repair) needs an owner and a ruling — the fix is two lines in a file
   I hold, but it re-scores a published count.
4. **`B = 0.33` is still located by a broken ruler** (theorist OQ-A). SC-3 is landed and *monitored*, but
   the budget's edge should be re-located with R1's corrected inradius before any C2W3 result leans on
   the number. The domain `s/sep ∈ [0.15, 0.30]` travels with `B` in code (`BUDGET_DOMAIN`) and in
   every report line.
5. **SC-6's capture radius costs a rollout per consolidation** (32 directions × 12 bisection steps,
   batched). It is default-on **only when the soft certificate is on**, and reports **INAPPLICABLE**
   rather than passing when `capture_dirs = 0`.
6. `blank(recency) = 0.5463 > chance` and `> full` (R3).
7. **Not measured by me:** D3's spearman `+0.914 vs +0.412` and `0/12 vs 1/12` sign flips are the
   theorist's numbers, carried and **not re-run**; what I measured is the *trip-state consequence* on 15
   real cells (§C2).

---

## Proposed handover updates (for the Hub)

**§10 / record.**
- ⭐⭐ **FB4 is adjudicated: PARTIAL = {`overload`, `recency`, `manifold`}; FB4 does NOT fire; the
  surviving family is `aggregate`** (secondary definition: `{aggregate, overload}`).
  `S = 1.0000 / 0.5068 / 1.0000 / 1.0000`, pre-registered before the run and hit exactly, including
  `S(aggregate)` to three decimals.
- ⭐ **The full-attention arm exists and is cheap** — a table reader at **+4 B**: at the metric's exact
  maximum on `overload`, **beaten by a 2-NN mean on `aggregate`**, structurally incapable on
  `recency`/`manifold`. ⚠ **It is not `AttentionPsi`** and must never be quoted as one.
- ⭐ **The C2W2 D4 recency fix is independently reproduced digit-for-digit** (out-of-pair 0.1944,
  0.4306→0.5556, launder 0.4861→0.5139, post-fix dividend **−0.0028 ± 0.0619**) — and the fix
  **strikes the family**, because the same 2-way restriction makes a +0 B row-order reader exact.
- ⭐ **New design rule, twice-tested:** *"the answer is provably not in the table"* is the only family
  property that has survived a +0 B substitute audit (C2W1 0-for-4, C2W3 1-of-4).
- ⭐ **D3 landed and its effect is measured on 15 real cells:** monitor #3's correlation leg is retired
  to a non-tripping diagnostic; **9 of 15 cells stop tripping #3**, every one of them with the fire-rate
  leg in band and utilisation unchanged, and the three shipped-anchor cells now report a computable
  `ρ_C3` (0.9502 / 0.8635 / 0.9497) instead of a sign-unstable correlation.
- ⭐ **SC-1 reproduces `d_safe_override` bit-for-bit** on `aggregate/tight@s0` — the override *was* the
  soft certificate, undeclared — **and that arm is measured at `max deficit_rel = 1.988` against
  `B = 0.33`**, which monitor #3 now trips on while the write still lands.

**§7 (known issues / live).**
- ⛔ **The `recency` gym family is protocol-invalid.** Neither `0.3019 ± 0.0679` (scoring-domain defect)
  **nor** `−0.0028 ± 0.0619` (post-fix) may be quoted as a *dividend null*.
- ⛔ **`overload@load1x_shipped`'s table launder is at the metric's exact maximum (1.0000, 3/3).** Quote
  it as a **byte-frontier** anchor only.
- ⛔ **NEW (R4): monitor #6's C2W2 repair landed only the false-positive half.** The `+eps_acq` half of
  `doctrine-repairs.md` §2.3 is absent, so the 2 predicted recovered false negatives did not materialise
  (evidence: `overload/base@s0`, `slope_loss = −4.03e−2`, `slope_acq = +7.84e−4`, state `clear`).
  ⇒ **"6 removed / 2 added" is not what shipped: it is 6 removed / 0 added.**
- ⚠ The FB4 `attention` arm is a **table reader**, not `AttentionPsi`.
- ⚠ `B = 0.33`'s domain `s/sep ∈ [0.15, 0.30]` must travel with it; `sep/2` remains never-quote.

**§2/§3 (architecture + config).**
- **New:** `chlu/eval/fb4_gate.py` · `chlu/experiments/exp_fb4_gate.py` · `chlu/core/soft_certificate.py`
  · `tests/test_{fb4_gate,soft_certificate}.py`. Run FB4 with
  `python -m chlu.experiments.exp_fb4_gate [--quick] [--families …] [--seeds …] [--save-dir …]`.
  **No CLI hook** (ownership stop-rule) — one owed 2-line addition to `chlu/cli/experiment_cmd.py`.
- **New `CluSystemConfig` fields, all default-off/inert** (they live in `clu_system.py`, **not** in
  `chlu/config.py`, per the standing C2 rule): `store_write_mask_factory=None`,
  `store_write_mask_kwargs={}`, `soft_certificate=False`, `soft_certificate_kwargs={}`.
  `SoftCertificateConfig` (ζ, B, κ, η, ρ-band, δ_num, λ_floor, capture dirs) lives in
  `chlu/core/soft_certificate.py`.
- **`chlu/core/clu_system.py` seam (c)** joins (a) write-objective and (b) store-potential:
  a new store family can now register **its own write mask** from config alone —
  **this unblocks `route3-stage2`'s slotted store.**
