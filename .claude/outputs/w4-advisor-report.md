# WAVE 4 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-20. One spoke + four hubs, FORGIS_14, `base/iter4 = 9dfed5e`
(pin + 9 additive files, 0 deletions, all seven frozen harness files byte-identical).**
**Wall clock 2026-09-19 16:52Z → 22:06Z. All five finished. Zero GPU. Record:
`.claude/outputs/_thinkpad/w4/` (592 files).**
⚠ Every number is quoted from an agent's own report. ⛔ **The Hub has re-derived no statistic.**

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| `spoke-a` | instrument delivered + **two findings larger than the wave** | — |
| ⛔ `hub/read` | **LOST to a parameter-matched MLP — and so does the best possible read** | 1/3 |
| ⛔ `hub/unseen-valid` | **LOST to a size-matched continuous MLP**, two independent implementations | 1/3 |
| ⛔ `hub/placement` | **bar failed on all three seeds; the dial is exhausted** | 1/3 (2,3 returned) |
| ⭐ `hub/field-store` | **delivered; the central property holds EXACTLY** | 0/0 |

⭐⭐ **Wave 4's headline is a structural negative, and it is the most decision-relevant result the
programme has produced: at a matched parameter budget, no read of the wave-3 store can beat a plain
one-hot MLP — not the incumbent's read, not this hub's, not a perfect one.**

---

## 1. ⛔⛔ `hub/read` — the ceiling result

`mlp_180k`, a one-hot MLP with **no store, no atoms, no landscape and no address book**, beats the
addressed read by **−11.320 novel tuples/corner on row 4** and **−8.328 on row 3**, all three
generation-2 seeds, every bootstrap interval excluding zero.

⭐⭐ **And then the result that matters more.** `oracle_span` — the requested row's true feature
projected onto span(G_k), i.e. **the best answer any read of these atoms can produce**, a declared
non-arm — is *itself* below the matched learner on all three seeds: **49.361 vs 56.327, mean −6.966.**
⇒ **The remaining read headroom is +4.354 (arm → oracle). Capturing every bit of it still leaves the
arm 6.97 short.** ⛔ The read dial cannot reach the rival set on this store.

**What it did beat, and this is not nothing:** `table` **+34.238** · `cont_mlp` **+33.299** ·
`pvm_table` **+29.054** (⭐ so the store *is* earning its bytes against the per-value-mean specialist
that motivated the hub) · `tensor_addr_sum` **+32.408** (⭐ the tensor **product** is the mechanism).
⚠ Against `mlp`, `lstm` and `transformer` the sign **flips across seeds** ⇒ ⛔ no claim, either way.

⭐ **Scope, from `hub/field-store`'s correction and the Hub endorses it:** this ceiling is a property
of the wave-3 store's **64 fitted modes**, not of 64 atoms per well in general. **A store that
*chooses* its atoms spans a different subspace at the same 524,288 B.** ⛔ Whether some other
524,288 B of content clears the learner is **open**, and `hub/read` names it as the first thing wave 5
should measure.

## 2. ⛔ `hub/unseen-valid` — row 2 belongs to a generic learner

The hub built the second judge, **measured a 0.3218 span on it, declared a margin inside that span** —
and was then falsified by the rival its own task file named.

| falsifier | built by | params | ARM − rival | seeds with CI95 < 0 |
|---|---|---|---|---|
| `cont_mlp_ARMMATCHED` | the hub | 573,440 (100.1 % of the arm) | **−0.0192 ± 0.0028** | **3/3** |
| `cont_mlp_notune_572k` | **`spoke-a`, independently** | 573,440 | **−0.0170 ± 0.0024** | **3/3** |

Matched in population, parameters and cache. ⛔ Its dial declaration said *"falsified if the
continuous-input MLP matches the arm."* **It does not match it — it beats it.**
⭐ **The kernel-settling mechanism is a null for a structural reason:** the pinned read **already is
the chord midpoint**, and no two-atom bracket exists to fuse.
⚠ **The second judge has a broken-clock defect a constant render exploits** — reported, not
footnoted; the row needs **both** judges. ⭐ The instrument is the contribution that outlives the negative.

## 3. ⛔ `hub/placement` — the dial is exhausted, and it returned budget

Row 4 vs the label-MI partition, three generation-2 seeds: **+1.388 / −12.755 / +2.163**, mean
**−3.068**, seed spread **14.918**. Bar not met; ⛔ §A.3 forbids moving it.
⭐ **But bars 2 and 3 pass hugely:** vs `ward_balanced` (equal wells, **no label**) **+23.653**
(p = 4.4e-12); vs its own size-matched floor **+31.408**.
⇒ ⭐ **Where channels go matters enormously, and *using labels* to decide it is most of that — but
re-deriving how labels become a partition is worth ~1 tuple/corner with the wrong sign on another
seed.** ⛔ Runs 2 and 3 **returned unspent**: the corrected probe exists *because* bar 1 failed, so
selecting on it and scoring would be test-informed selection.

## 4. ⭐ `hub/field-store` — delivered, and the central property is exact

**Acceptance 2 holds EXACTLY.** Changed-byte positions, direct buffer comparison:

| operation | bytes changed | **outside the permitted region** |
|---|---|---|
| evict `object_hue = 0.0` (9 items) | 26,275 of 2,054,866 | **0** |
| ⭐⭐ **the hole — red ∧ cylinder** | **1** | **0** |
| decay ×0.5 on 8 items | 8 | **0** |
| per-item lifetime := 0.25 on 3 items | 3 | **0** |
| ⚠ **negative control** — nudge one address-book row by 1e-3 | 5,762 | — |

`assert_no_refit` passes on every operation (14 named structures byte-identical). ⭐ **The negative
control is there because an assertion that has never fired is not an assertion.**
⭐⭐ **The hole is the escape from wave 3's trap:** it refuses one conjunction while **96 unseen
conjunctions stay answered at 0.0000 refusal** and every other render stays bit-identical.
**Acceptance 1:** row 3 inside the declared ±1.0 band; rows 1 and 4 **ahead** of the incumbent
(r4 **31.375 ± 3.927** vs 26.472 ± 3.783). ⛔ It claims nothing. **Wave 5 scores it.**

## 5. ⭐⭐ `spoke-a` — two findings larger than its own job

**(a) Generation 1 fails its own encoder's re-encode check, on all three seeds.**

| seed | generation 1 | generation 2 |
|---|---|---|
| 0 | ⛔ **FAIL** max **51 ULP**, 1280/131072 (0.98 %) | ✅ PASS 1.00 ULP, 1 element |
| 1 | ⛔ **FAIL** max **81 ULP**, 640/131072 (0.49 %) | ⚠ FAIL 3.00 ULP, 3 elements |
| 2 | ⛔ **FAIL** max **59 ULP**, 1231/131072 (0.94 %) | ✅ PASS 1.00 ULP, 2 elements |

The gate's bound is **one** ULP; generation 1 sits at 50–80× it. ⭐ The signature (≈53 ULP on ≈1 % of
elements) is exactly the cross-device hazard generation 2 was commissioned to remove — **measured, not
argued, and it fell out of running the harness's own gate.**
⛔⛔ **Consequence: a re-score on generation 1 was never possible.** The gate enforces only when the
cache is *extended*, and a full-scope cache never triggers it — **so the check existed, was correct,
and was structurally unable to see the thing it was built to catch.**
⚠ It does **not** invalidate any published generation-1 number; those numbers were computed on those
features. It means generation-1 and generation-2 numbers are measurements on **different feature
sets**, which can now be demonstrated rather than asserted.

**(b) The re-score is reassuring in the way that counts.** ⭐ **The three deterministic arms —
`clu`, `knn_proto`, `store_removed` — moved by EXACTLY 0.0000 on every seed**, not "within
tolerance". Every arm that moved is a **sampler** arm. One trip of the seed-spread rule:
`lace/ld_corrected` at 1.17× spread, one seed, on a companion leg rather than LACE's published row.
⚠ Held with its limits: 2 of 3 seeds (seed 1's gen-2 re-score is a declared NOT-RUN), and `dynread`
is a declared absence, not a zero.

⭐ **The byte column now tells the real story:** **LACE is 95.1× the store budget** (12,464,319 params —
it had been printing a bare dash), PoE-EBM 27.1×, and **the prototype table at 0.91× leads the whole
rival set at 0.9165**. ⛔ The two arms that beat nothing are the two that are 27× and 95× over budget.

⭐ **Its own worst defect, caught and reported:** a first version of the generation comparison reported
*"no published arm moves by more than its seed spread"* computed from **zero** generation-1 data.
**An empty comparison is not a passing comparison, and a verdict function that cannot tell the
difference will always fail in the reassuring direction.** Fifteen minutes on disk; no hub read it.

---

## 6. What the Advisor is asked to rule on

1. ⛔⛔ **Does the oracle-ceiling result retire the read dial?** No read of the wave-3 store reaches a
   parameter-matched MLP, proven via a declared non-arm. ⭐ `hub/field-store`'s scoping — that this is
   the ceiling of *fitted* modes and a store that *chooses* its atoms is untested — is the only
   opening, and wave 5's `hub/compose-field` is already pointed at it.
2. ⭐⭐ **What survives of wave 3's composition win?** It was **+11.612 over `basis_head`** — an
   ABLATION under wave 4's rule. Against the full rival set the same object **loses to a matched MLP**.
   ⛔ The Hub's reading: **wave 3's win is not a result in wave 4's terms** and must not be quoted as
   one. **This is the Advisor's call and it bears directly on the paper's primary claim.**
3. **Is the label-MI partition now the settled answer on placement?** Bars 2–3 say placement matters
   enormously and labels are most of it; bar 1 says re-deriving the mapping is worth nothing.
4. ⚠ **A conflict between the rival set and the estimand**, raised independently by `hub/placement`:
   the rival rows are a **single 72-request cell**, while the wave rule requires ≥ 49 paired corners
   and forbids a single cell deciding a target. ⭐ **Wave 5 needs this closed before it scores.**
5. ⚠ **Parameter-matching and byte-matching cannot both hold.** The pinned store alone is 131,072
   parameters — 10.3 % above the prototype table — so no CLU arm can be parameter-matched to this
   rival set at the 524,288 B pin. `spoke-a` added `mlp_180k` as a second-budget rival; **the
   programme should declare which matching is primary.**
6. **Generation hygiene:** generation-1 numbers can never be re-scored on this box. Every wave-4+
   number is generation 2. ⛔ Nothing from waves 2–3 may share a table with them.

## 7. Limits
Seed 0/1/2 on generation 2 throughout (⭐ the three-seed rule is now met for wave-4 numbers) ·
the Hub re-derived nothing · `hub/unseen-valid` produced **0 commits** — its work lives in outputs, so
it can be read but not re-run · ⚠ `spoke-a` finished with **12 files uncommitted and no bundle**; the
Hub committed them at the boundary (`b14870b`, 5,064 insertions) and bundled the branch.
