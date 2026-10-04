# REPORT — `hub/deletion-2` · exact structural deletion on the generative store
**Session `clu-dev-1e [eb2e09]`, Opus. START_UTC 2026-09-17T11:03:31Z. Machine FORGIS_14
(Windows 11 AMD64, 4 threads, CPU jax 0.9.0, numpy 2.4.1, Python 3.11.16).**
Branch `hub/deletion-2` @ `945a3e9` (base/iter2). Seed 0. Dataset shapes3d, pretext `recon`.
Detail and every raw number: `FINDINGS-clu-dev-1e.md`. Pre-registration: `PREREG-01-design.md` (mine,
11:05Z) and `PREREG-deletion-cell.md` + amendments A2–A4 (the co-agent's, 11:20Z onward).
⚠ Two agents worked this hub concurrently; see `INCIDENT-two-agents-clu-dev-1e.md`.

---
## VERDICT IN ONE LINE
**NOT SOLVED as a win. The pre-registered headline is a BANKED TIE** (churn-corrected collateral
Δ = **+0.0030**, 95 % CI [−0.0071, +0.0112], m = 0.020, 0 of 6 deletions outside the margin).
⭐ **Two results are real, non-by-construction and were not asked for**: a compositional-deletion win
(§4) and a mechanism-level cost that the ported method does not price (§3). **One charter condition
fails outright** (§5). ⛔ **The scored run was NOT spent** (§7) — 0 of 3 remain unused.

---
## 1. The blocking gate — PASSED, and stronger than required
`train_store` is **deterministic at the shipped cell scale** (K=5, M=64, D=2048, n_live=32,768,
1200 steps, batch 128). **Three independent OS processes from three different launch paths: 22/22
arrays bitwise identical**, `store_digest d0eacf3b…`. ⭐ They ran at 554.9 / 617.9 / 728.3 s — a
**1.31× wall-clock spread under different machine contention** — and still produced identical bytes,
which rules out the audit's top-ranked risk (BLAS thread count flipping a greedy `ward` merge).
A fourth confirmation through the whole eval path: `eval_a` predictions agree at exactly **1.000000**.

⚠ **Scope (measured, not assumed):** against the Mac manifest the **channel partition is
ISA-invariant** — dead/live channels 71/1977, the rejected `average` attempt `[1973,1,1,1,1]`, the
accepted `ward` attempt, the final `well_sizes` all identical, and `loss_first` bit-identical — but
`loss_final` differs by **+1.27 %**. ⇒ ⭐ **Byte-exact deletion provenance is a WITHIN-MACHINE
property.** ⛔ Not a parity claim.

## 2. Conditions 3 and 4 — provenance and rivals: MET, and by construction
`delete(v) := rebuild(live ∖ rows(v))`, so `hash(delete) == hash(build(pool ∖ rows(v)))` is a
tautology about the implementation once §1 holds. **Verified bit-exactly anyway**: after deleting
`object_hue=5`, `cond_2[5]` is exactly `log(1/64) = −4.1588830947875977` on all 320 entries (max
deviation `0.000e+00`), exactly 1 of 10 values uniform, no other factor touched.
⭐ **Method-section fact, per the task file's own instruction.** Rivals took the identical predicate
(`knn_proto` refitted in-process on the same live vector, `live_sha256` re-asserted).
`lace` / `ebm_compose`: **NOT RUN** — no exact delete operator, and retrain-without-`v` did not fit.

## 3. ⭐⭐ THE HEADLINE FINDING — canonical rebuild is not answer-preserving
| rung | comparison | n_live | ARI | CLU churn | CLU kappa | knn churn |
|---|---|---|---|---|---|---|
| 1 | identical live set | 32768 | **1.0000** | **0.0000** | **1.0000** | **0.0000** |
| 2 | **same n_live, different arbitrary rows** (4 replicates, 6 pairs) | 24534 | 0.681 ± 0.082 | **0.1883 ± 0.0084** | **0.597 ± 0.030** | **0.0140 ± 0.0005** |
| 3 | semantic deletion vs undeleted | 32768/24534 | 0.410 | 0.2122 | 0.5434 | 0.0644 |

> **Change nothing about a canonical store except which arbitrary 8,234 of 32,768 rows you drop,
> rebuild by the ported mechanism, and 18.8 % ± 0.8 % of its answers change — while a byte-matched
> prototype table under the identical operation changes 1.40 % ± 0.05 %. A 13.4x ratio,
> over 6 pairwise comparisons among 4 independent same-size rebuilds.**

Rung 1 proves this is not build noise. Rung 3 adds almost nothing over rung 2 ⇒ **essentially all the
churn is recanonicalisation; the semantic content of the deletion is second-order.** Holds on all six
deletions and all ten placebos (CLU churn 0.179–0.207, kappa 0.55–0.64; table 0.002–0.022).

### ⭐⭐⭐ And there is no small-deletion regime
Across **ten** semantically-empty rebuilds spanning `n_deleted` **852 → 8,234** (2.6 % → 25.1 % of the
pool, a **9.66×** range):

Across **7 distinct sizes** with within-size replicates at three of them:

**16 semantically-empty rebuilds, 7 distinct sizes, replicates at 4 of them:**

| | r(n_deleted) | max/min over the 9.7× span |
|---|---|---|
| **CLU** | **+0.016** | **1.110×** |
| **`knn_proto`** | **+0.989** | **3.589×** |

⭐ Restricting to the **four banded sizes** (n = 852, 2142, 4095, 8234): CLU churn is
**0.1929 ± 0.0046, 0.1910 ± 0.0036, 0.1920 ± 0.0033, 0.1931 ± 0.0035** — a **9.7× change in rows
removed producing a 1.011× change in answer churn.**

**The table's disturbance scales with what you remove; the CLU's does not.** ⇒ ⭐⭐ **the behavioural
cost of a deletion does not shrink with its size — deleting 852 rows (2.6 % of the pool) costs
essentially what deleting 8,234 (25.1 %) costs.** For a deployed system that is the practically
decisive fact: the price of honouring one erasure request is the price of a full recanonicalisation,
every time.

⚠ **Stated precisely, because a looser version would be an over-claim.** The CLU's residual variation
*is* detectable — between-size sd 0.00694 against a pooled within-size replicate sd of 0.00239, a
2.90× ratio — so this is **"does not scale with n"**, ⛔ **not "is constant"**. The variation is
non-monotone and driven by one single-replicate point (n = 3224, churn 0.2074); excluding it the other
six sizes span 1.034×. ⚠ Smallest measured is 852 rows; ⛔ I do not extrapolate to a single row.

⭐ *Retention You Can Predict, Scope and Delete* (§2.3 / App. F / G.8) prices the mechanism honestly in
**time** (O(n) per delete) and guarantees **byte-identity with a store that never held the item** —
this hub confirms both exactly. It makes **no claim about behavioural stability**, and this is the
measurement showing why that gap matters: a practitioner deleting one record would reasonably expect
the system's other answers to hold, and a fifth of them do not. ⛔ The table is immune only because it
has no layout to recanonicalise — the same property that makes it unable to forget a conjunction (§4).

## 4. ⭐⭐ THE ONE GENUINE WIN — the CLU forgets a conjunction; the table cannot
Delete `(object_hue = 5 ∧ shape = 2)`, 852 rows. **Both parts survive individually and neither
addressing term is emptied** (verified: `cond_2[5]` and `cond_4[2]` both still non-uniform), so the
request stays fully addressable. On the 1,024 requests naming both:

| | joint render rate | placebo control |
|---|---|---|
| CLU | 0.3672 → **0.0107** | 0.3037 |
| `knn_proto` | 0.9092 → **0.5488** | 0.9053 |

⭐ The table **re-composes the deleted conjunction from parts it still holds** — its `oh=5` prototype is
still fitted on hue-5 rows of other shapes, its `shape=2` prototype on shape-2 rows of other hues.
The CLU's store was **retrained with no `(oh5, shape2)` row present, so no mode encodes the
combination and the composed code has nothing to point at.** Normalised for the CLU's lower starting
point the gap widens: it retains **2.9 %** of its pre-deletion capability, the table **60.4 %**.
Collateral cost of this deletion: Δ = −0.0058, a tie. ⇒ **forgotten at no differential cost.**
⚠ One conjunction, one seed, and the CLU held it weakly to begin with (0.3672 vs 0.9092).

## 5. ⛔ CONDITION 1 IS NOT UNIFORMLY MET — and the failure is the useful part
- `object_hue`: met — **0.000000** of 4,096 dev renders show hue 5 afterwards. ⛔ But `knn_proto`
  achieves it **strictly better** at both ends (1.0000 → 0.0000 vs 0.5107 → 0.0049). By construction.
- ⛔ **`orientation`: FAILED BY BOTH ARMS.** The median-index rule selects `orientation = 7` = exactly
  **0.0°, the null pose**. `cond_5[7]` is bit-exactly uniform — and the CLU still renders orientation 7
  on **25.0 %** of the requests naming it and **12.6 %** of all dev renders (§6a bar: 0.117). The rival
  fails worse (0.5977).
  ⭐⭐ **Because a blurry reconstruction looks unrotated, the value is what the model degrades
  *towards*.** Deleting every row that carries it, and proving the addressing term bit-exactly gone,
  does not stop the decoder producing it.
  ⇒ **The mechanism guarantees byte-identity with a store that never held `v`; it does NOT guarantee
  no request can elicit `v`.** Any generative method with a default output cannot delete that default
  by removing rows, however canonical the rebuild.
- ⛔ **`scale`: ALSO FAILED BY BOTH ARMS**, and more starkly — `knn_proto`'s accuracy on the deleted
  value goes **UP** (0.8096 → 0.8252) and it still renders `scale = 4` on **54.5 % of all dev
  requests**; the CLU on 23.6 %. `scale` index 4 = **1.0357, the default object size**.
- Conjunction: **met by the CLU, failed by the rival** (§4).

⭐⭐ **The failures are predictable in advance and separate perfectly 4–2.** `scale` and
`orientation` are exactly the two factors whose median index is a **physical null** (default size,
null pose). The four without one — three hues and shape — **all PASS in both arms**
(CLU post 0.0020–0.0605; §6b down to 0.00000–0.07471). ⇒ **which deletions will fail condition 1 is
identifiable from the factor's geometry before any deletion is run.** ⚠ Perfect separation on n = 6,
one seed, and the link was drawn post-hoc — a hypothesis, ⛔ not a law.

⇒ Store-level deletion removes a value's **addressing** exactly and reliably; whether the value becomes
**ungenerable** depends on whether the decoder can produce it unprompted — a property of the value,
not of the deletion.

## 6. Condition 2 — preserved: the pre-registered headline, a TIE
Churn-corrected Δ = `[C_clu(del) − C_clu(plc)] − [C_knn(del) − C_knn(plc)]`, per PREREG §7 + A3.12.
floor_hue **+0.01162**, wall_hue **+0.01514**, object_hue **−0.01957**, scale −0.00195,
shape **+0.01074**, orientation +0.00216.
**Mean +0.003023, sd 0.012774, bootstrap 95 % CI [−0.00714, +0.01121].**
⇒ **R1: TIE.** 0 of 6 below −m, 0 of 6 above +m. ⭐ Banked.
⚠ The **uncorrected** Δ is −0.0033; the correction the PREREG registered as fair moves the number
*toward* the CLU overall, but *against* it on `object_hue`. Both are reported.

## 7. ⛔ THE SCORED RUN — NOT SPENT. 0 of 3 used on this target, on any machine.
PREREG §9 makes B permissible once DEV clears (it did); it is not mandatory. Reasons, quantitative:
1. ⭐ **The outcome is a foregone tie.** Pooled B half-width ≈ **0.01664** against m = 0.020, with DEV's
   effect at +0.0030 ± 0.0058. A *loss* needs a true effect ≤ −0.037, a *win* ≥ +0.037 — both ~6 DEV
   standard errors away. A run whose verdict is determined in advance buys no information.
2. ⭐ **B cannot test §4**, the one genuine win: B is entirely `shape = cylinder`; the conjunction is
   `shape = 2`.
3. ⚠ **Validity, not power:** pre-deletion on B the CLU is at exactly **1.0000 on all three hues**
   (collateral can only register as damage) and at 0.0417 `scale` / 0.0000 `orientation`, both *below*
   the fully-ablated store's 0.3333, against a best-constant floor of **0.5278**.

⚠ I first argued B lacked power and **withdrew that**: I expected the six deletions to be redundant on
B's shared 72 rows; measured pairwise correlation is **r = −0.0045**, i.e. effectively independent, so
pooling does reach adequate power. The decision rests on 1–3, not on power.
⭐ **Recommendation:** preserve all three for the 3-seed final once full-scope caches exist.

## 8. Cost line
**Rebuild cost per delete: 501–728 s `unit_wall_s`**, ⚠ plus **95–283 s of JAX teardown** excluded from
that figure — true process wall is what a delete actually costs (`det_B` 1011 s). Contention noise
measured on three identical-work replicates: **≥ 1.42×**. `O(n)` per delete, as the ported mechanism
states openly. CLU store 524,288 B; `knn_proto` 475,136 B (**the rival is the smaller model, 90.6 %**),
and its whole D-float vector reaches the decoder — an unbounded channel against the CLU's 30 bits.

## 9. ⛔ What I got wrong — all registered in advance, all refuted
1. **ARI direction.** Predicted `ARI(plc_f2v5)` 0.6–0.9 and above the deletion's 0.5656. Measured
   **0.4924**, below.
2. **Movers enrichment.** Predicted channels changing wells would be enriched for the deleted factor's
   selectivity. Measured ratio **0.966, p = 0.63**; `shape` (2.264) and `orientation` (2.468) enriched
   instead, both untouched.
3. **The diversity account**, on the falsification test I chose in advance: predicted
   `ARI(del_f3v4) > 0.5656`; measured **0.4418**. Across six, r(pre-accuracy, ARI) = **+0.5232**,
   predicted negative, and driven entirely by one outlier. ⛔ Dead.
4. ⛔⛔ **Over-generalising from `del_f2v5`** — the worst. I analysed one deletion in depth, concluded
   the churn correction "penalises the arm" and that deletion churn is "asymmetrically downward", and
   gave the second a clustered significance test that made an n = 1 pattern look established. Across
   six deletions `f2v5` is the **only** negative and it is the outlier; three of five move net upward.
   Both conclusions retracted.

## 10. Unexplained, reported as such
`ARI(det_C, del_f0v5) = 1.0000`. Deleting all 3,269 `floor_hue = 5` rows left the channel partition
**bit-identical** (0 of 2048 channels moved) — while its own size-matched placebo moved 982 and every
other deletion moved 864–1246. Verified not an artefact: different live set, different `store_digest`,
trained modes differing by up to 3.75 per well. ⛔ **I have no explanation and have not invented one.**

## 11. Elapsed and status
**START_UTC 2026-09-17T11:03:31Z · cell complete and deliverables written at 17:35Z ·
ELAPSED 6 h 32 m of the 12 h true wall clock** (deadline 23:03:31Z, ~5 h 28 m unused).
**28 of 28 planned units complete**: 3 determinism builds, 6 deletions + 6 size-matched placebos,
1 conjunction + its placebo, 10 further placebo replicates across 7 sizes, and the 2-unit
path-independence pair. **16 semantically-empty rebuilds** underpin §3.

⛔ **I held zero compute from 11:20Z onward.** The cell was built by the co-agent's surviving detached
driver; **every number in this report I re-derived from raw artefacts** (`build.json`, `modes.npz`,
`preds.npz`, `rival_preds.npz`) with my own code, not read out of `analyse.py`.
⛔ **No sub-agent of mine is running** (two were spawned, both completed: AUDIT-01 and INHERITED-FACTS).

**Return pack:** `hub-deletion.return.tar.gz`, 311 files, 30 MB,
`sha256 b4cecd3cb9a6b3921951a42a0b8ecb4defcd9c59835aba20261506015b2d12c2`.
**Code bundle:** `hub-deletion-2.bundle`, whole branch `hub/deletion-2`, 3.2 MB, verified
*"records a complete history"* (12 commits, 2,493 insertions).

### ⭐ SOLVED TRUTHFULLY: **NOT as a win — YES as an honest measurement.**
- The pre-registered object (**collateral**) is a **banked TIE**: Δ = +0.0030, CI [−0.0071, +0.0112],
  0 of 6 deletions outside ±m. Per R1 a tie is banked and reported.
- **Three findings beyond it are real and non-by-construction**: canonical rebuild churns ~19 % of
  answers *independently of how much is deleted* (§3); the CLU forgets a conjunction the byte-matched
  table re-composes (§4); and store-level deletion cannot remove a value that is the decoder's default
  output, which is predictable from the factor's geometry before the deletion is run (§5).
- **One charter condition fails outright** and is reported as a failure (§5).
- ⛔ **The scored run was not spent** (§7). 0 of 3 remain unused.
- ⛔ Four of my own predictions were refuted and are listed (§9); one unexplained result is reported
  as unexplained (§10); one over-claim of mine was corrected by later replicates (§3's precise wording).
