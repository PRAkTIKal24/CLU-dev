# PREREG-erasure-1 — the adversary, and the three claims it scores

**Hub:** `hub/erasure` · **branch/worktree:** `hub/erasure` @ `afd37cf` ·
`C:/chlu/worktrees/CHLU-hub-erasure`
**Written:** 2026-09-18, **before any claim-1, claim-2 or claim-3 cell was run.**
**⛔⛔ SCORED RUNS SPENT BY THIS PREREG: ZERO.** The `deletion` target keeps all 3 banked.
Nothing here touches the held-out sets A/B/C. Every number is a mechanism measurement on the
pinned `value_holdout` condition and its own controls, and ⛔ none is quoted as a scored number.

---

## 0. What was already measured before this file was written, and why that is not cheating

Charter §A.1 says iterate freely on MECHANISM; pre-registration binds SCORED runs. The target
file §2 says the adversary must be **built first** and its budget and move-set **declared before
it runs**. Building it required calibration, and the calibration **killed the obvious adversary**.
Those measurements are in `LOG.md` under 2026-09-18 22:48 and 23:05 and are summarised in §3
below. ⛔ No claim-1/2/3 cell had been run when this file was written. The three smoke cells run
at `n_store_train = 4096`, `adv_steps = 3` are **plumbing only** and are reported nowhere.

## 1. The question, and what would falsify it

**Claim 1 (headline).** Deleted content is **unrecoverable** from the refitted CLU store and
**recoverable** from a byte-matched **gated** table, under the **same** adversary.
**Falsified if** the adversary recovers from `clu_deleted` as readily as from `basis_head_gated`
— then the erasure claim is dead and the deletion target with it, and **that is the result**.

**Claim 2.** Rebuild churn, with per-request predictions **persisted**, the identical-live-set
rebuild as the null, ≥ 3 store seeds.
**Falsified if** churn does not survive persisting per-request predictions.

**Claim 3.** A conjunction deletion is surgical in the CLU and not expressible by the gate.
**Falsified if** deleting a conjunction breaks rows 3–4 for the survivors.

## 2. The frozen instrument (charter §A.4) — read-only, and asserted at run open

| component | pin | assertion |
|---|---|---|
| encoder feature cache | `shapes3d_recon/seed_0/features.npy`, sha `d6728f01…`, **scope `full`, 480,000 rows** | `assert_cache_is_read_only` — `shortfall = 0` |
| decoder | `decoder_v1_s0.pkl`, hash `01ef91f7…` | asserted against `classifier_gate.json`'s `decoder.param_hash_per_seed["0"]` |
| external classifiers ×3 | `seed_{0,1,2}/external_classifier.pkl` | `classifier_gate.json` must read `gate: PASS` |
| store grouping | `store_plan.json` — ward, K = 5, well sizes `[303,305,573,501,366]` | labels must partition D = 2048 |
| splits | `value_holdout/factor_index.npz` — train 284,472 / test 195,528 | `assert_value_absent` per pool |

⭐ **The cache is `full`, so every run here only READS it.** ⇒ no run can extend it, the
re-encode gate cannot bind, and **there is no GPU case in this hub** (w3 §8). Everything below
is local CPU, 4 threads, as the machine ruling allows.

## 3. ⛔⛔ THE ADVERSARY — and the two moves that were KILLED by their own controls

### 3.1 What the adversary holds
The arm's **deployed weights** (all of them, as bytes), the **request interface**, the frozen
decoder, and the factor grid. ⛔ It does **not** hold the training rows, the feature cache, or
any held-out label.

### 3.2 The scope ruling, forced by measurement
"Recoverable from the store" ⛔ cannot mean "the frozen decoder can be driven to draw it."
Measured: 64 **free signed** coefficients over a **random Gaussian** span matched only in
per-coordinate mean/std reach `object_hue = 5` at **0.672 unanimous over three judges** — better
than the same sweep over real atoms from a pool that never held the value (0.594) and better than
coordinate-shuffled atoms (0.562). An L1 cap does not rescue it: at the tightest cap tried,
`L1 ≤ 1.0`, sensitivity 0.625 against specificity **0.484**.
⇒ **the unconstrained and L1-capped signed sweeps are INADMISSIBLE.** They are retained in the
code and reported as failed moves, with those numbers, and ⛔ never as evidence.

The zero-move baseline localises the fault to the **sign**, not to the target: with no
optimisation at all, 128 random **convex** combinations of atoms from a pool that never held the
value read the value at **0.000 / 0.000** (hue / cube), while 128 random **signed** L1 = 1
combinations of the *same atoms* read **0.125 / 0.195**.

### 3.3 The declared move-set (budget in brackets) — ⛔ fixed before the run
| move | what it does | budget |
|---|---|---|
| **M1 direct request** | name the deleted value on-grid in an ordinary 6-term request | 512 requests |
| **M2 weight read** | decode **every deployed parameter** as a feature vector and judge it: `clu` → the N stored train rows; `knn_proto` → the 58 prototypes; `basis_head` → the 62 dictionary atoms **and** the 58 head rows pushed through the dictionary. ⛔ No gate can stop it: it never asks | all parameters |
| **M3 indirect** | (a) the deleted factor's term **omitted**, so the arm must fill it from what it still holds; (b) off-grid terms bracketing the deleted value | 512 + 512 |
| **M4a convex span sweep** | `c = softmax(θ)` over the arm's **own deployed span**, gradient ascent on the **attacked judge**'s `log p(target)` | 96 candidates × 100 steps, lr 0.2 |
| **M4b signed span sweep** | the same with free/L1-capped signed coefficients | ⛔ **INADMISSIBLE** — reported, never quoted |
| **M5 read-reachable** | sample the deployed read's own reachable set (for `clu`, per-well atom-index tuples); no gradient | 512 tuples |

**Per-cell wall-clock cap: 900 s.** Every cell writes its own JSON the moment it completes and
is skipped on resume, so a kill costs one cell.

### 3.4 Judges — one is attacked, two are held out
Judge **0** is the only one that sees a gradient. **`unanimous_hit` (all three judges) is the
headline**; the attacked judge's own rate is reported beside it and is inflated by construction.
Instrument ceiling on REAL carrying rows: **1.000** (hue), **0.930** (cube) — so a zero on a
deleted store cannot be blamed on the judges.

### 3.5 ⭐ The calibration gate, applied BEFORE any cell is read
Two of the nine conditions exist only to calibrate, and they run in **every** cell block:
- **sensitivity** = `clu_full`, the store fitted on a pool that **HOLDS** the value;
- **specificity** = `random_span`, a dimension-matched random span, plus `clu_shuffled`, the
  charter §C.2 landscape-removed control.

**Rule, declared now: a cell is read only if `sensitivity − specificity ≥ 0.10`.** Otherwise the
verdict is `ADVERSARY NOT CALIBRATED` and ⛔ no erasure claim is read from it.

## 4. Claim 1 — conditions, all at matched bytes

| condition | arm | fit pool | gate | role |
|---|---|---|---|---|
| `clu_full` | clu | `L_full` | — | **sensitivity ceiling** |
| `clu_deleted` | clu | `L_deleted` | — | **the treatment** |
| `clu_placebo` | clu | `L_placebo` | — | equal rows deleted, none carrying |
| `clu_shuffled` | clu | `L_deleted`, coords shuffled | — | landscape-removed (§C.2) |
| `basis_head_gated` | basis_head | `L_full` | 8-byte bitmap | ⭐⭐ **THE LIKE-FOR-LIKE** |
| `basis_head_refit` | basis_head | `L_deleted` | — | the baseline doing what we do |
| `knn_proto_gated` | knn_proto | `L_full` | 8-byte bitmap | second control |
| `knn_proto_refit` | knn_proto | `L_deleted` | — | second control, refitted |
| `random_span` | — | random, matched marginals | — | **specificity control** |

**Byte ledger, asserted at run open** (`exp_gen_cell.arm_byte_ledger`'s convention):
`clu` **524,288 B exactly** (`64 × 2048 × 4`) · `knn_proto` **475,136 B** (`58 × 2048 × 4`) ·
`basis_head` **522,288 B** (`m = 62`, the largest the budget allows: `m(D·4 + 58·4)`).
The **8-byte gate is ADDED** to the gated arms' total — a declared handicap on our side.

**Pools.** `L_full` = 32,768 rows drawn from the universe of rows in no held-out *combination*
cell; `L_deleted` = `L_full` minus **every** row carrying the target; `L_placebo` = `L_full`
minus an **equal number** of non-carrying rows. ⇒ `|L_deleted| = |L_placebo|`.

**Targets** (⛔ neither is `scale` or `orientation`, which the code refuses to construct):
`object_hue = 0.5` (circular axis, no physical default, read back at 1.000) and
`shape = cube` (categorical, no retained near-neighbour; ⛔ not the sphere).

**Expected direction.** `clu_full` high · `clu_deleted` ≈ `random_span` ≈ 0 ·
`basis_head_gated` and `knn_proto_gated` high **on M2**, because the content is still in their
parameters · `clu_shuffled` ≈ 0.
**Margin: 0.10 unanimous-hit**, declared now.

## 5. Claim 2 — rebuild churn

Live set 32,768; delete **8,234** rows (wave 2's size, reproduced exactly); **4,096** fixed
requests, the same set for every arm and store. Per-request predictions from **all three judges**
are **persisted** to `claim2__<arm>__seed<s>__predictions.npz` (`requests`, `before`, `null`,
`deleted`). Churn = fraction of requests whose read-back factor tuple changed.
**Null = the identical-live-set rebuild, which must give exactly 0.0000** — if it does not, the
pipeline is nondeterministic and no churn number is readable. Arms: `clu`, `basis_head`,
`knn_proto`. **Seeds 0, 1, 2.**
Prior under test (wave 2, aggregates only): clu 18.8 % ± 0.8 % vs table 1.40 % ± 0.05 %, 13.4×.
⛔ Quoted as the prior, never as this hub's number.

## 6. Claim 3 — conjunction deletion

**Declared cell: `shape = 3` (oblong/capsule) — NON-CYLINDER**, and neither the cube (this hub's
value target) nor the sphere (the shape a degraded render degrades towards).

Three pre-registered conjunctions, each with a **size-matched placebo drawn from inside the
cell**: `object_hue=2 ∧ shape=3` · `floor_hue=7 ∧ shape=3` · `wall_hue=4 ∧ shape=3`.
**The pre-deletion level is reported beside every post-deletion number** (target §4: wave 2's
conjunction was held weakly at 0.3672 and the effect is unreadable without it).
Retention criterion: **unanimous over three judges on BOTH factors at once**; the two
single-factor rates are reported beside it.

**The gate contrast, computed exactly:** after a conjunction deletion both values still occur
separately, so the 8-byte per-factor bitmap changes **zero bits** and keeps serving the deleted
conjunction. The only gate that can express it is the full joint table (480,000 combinations,
59 KB) — which then refuses **every** combination absent from the live set, i.e. every row-3 and
row-4 request.

## 7. The scoreboard (w3 §2)

All seven rows, `basis_head` first and `knn_proto` second on every row. Rows 5, 6 and 7 carry the
**8-byte rule's score beside them** and the §1 negative in the row's own note. ⛔ No row-5 or
row-6 number is quoted as evidence for the store.

## 8. Floors (charter §C), binding

1. Every arm beats the matched-bytes lookup beyond seed spread, or the store is **decorative and
   is reported as such, in those words**.
2. Every arm carries a landscape-removed control that must collapse — here `clu_shuffled`.
3. Any arm whose settled state sits on its nearest atom is a **lookup table and is reported as
   one**. ⚠ `x0` is zero settling steps and returns a per-well stored atom **by construction**;
   that is declared here, not discovered by a referee.
