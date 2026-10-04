# PREREG — C2W8 "Consolidation + trash": the well lifecycle

**Filed 2026-08-06 by the C2W8 research-lead Hub, BEFORE any harness cell of this wave runs.**
Base: `main @ d70898b` (= `clu-dev/main`, tree clean, zero worktrees, suite 1410/0).
Binding scope: charter **§A21 C2W8 row**; design inputs **§A20.3(d)** + **§A20.6-P2** + **Addendum 9 §A28.3**.
Bound by: intervention doc §5 (anti-collapse), §6 (benchmark criteria), §8 (prohibitions);
Add.6 standing rules; `AGENT_PROTOCOL.md` §7.

This document registers the wave's **numeric predictions**, its **kill-conditions**, and the
**instrument definitions** they are computed on. Kill-conditions K1–K5 are built and asserted
**before** the mechanisms they can kill (standing doctrine, §A12: *build the kill-condition before
the thing it can kill*). This file is never edited after the first cell runs; corrections go in a
dated `ERRATA-C2W8.md` block beside it.

---

## 0. The one-sentence question

**Does the designed well lifecycle — over-dig freely → merge-to-budget on mechanical criteria →
prune-below-budget as a controller decision → pruned wells routed to `γ_φ(q)` — earn a slot, when
measured as full CLU ± consolidation on long streams under capacity pressure?**

Not "does consolidation help in isolation" (§8.1 forbids it). The rig is the **full CLU system with
exactly one capability toggled**, per Add.6's binding design rule.

---

## 1. Three mechanics this wave is designed AGAINST (Add.9 §A28.3, all measured, all corrected on
   the record — none of them is an open question this wave re-opens)

1. **Erosion drives depth TO ZERO ⇒ an eroded well ceases to exist as an attractor.** There is no
   "deep well holding a useless zero latent" to go hunting for. ⇒ **Design consequence:** the prune
   verb's target population is *live attractors that are never read*, not eroded wells. Any well
   below the measured attractor floor is already gone and is a bookkeeping problem, not a lifecycle
   problem. The census (§4, K1) measures the two populations separately and reports both.
2. **Depth ≠ usefulness.** I2 = `NO_USAGE_STRUCTURE` (§A23.5), and the depth-as-feature-importance
   quotation caveat is **ACTIVE and may not be lifted by this wave**. ⇒ **Design consequence:** the
   prune criterion **may not be depth**. It must be usage. K3 is the pytest-asserted proof that the
   shipped criterion is not a depth policy wearing a usage costume.
3. **The optimizer's erosion is CHURN, not curation** — a 112× transient trough at step 150 that
   recovers (§A27.1). ⇒ **Design consequence:** optimizer erosion cannot be trusted as a deletion
   policy, which is exactly why P1 removes the outer loss's authority to delete. **The designed
   channels this wave builds are the replacement for that authority**, and they are answerable to
   the ledger in a way the optimizer never was.

⛔ **None of the three is re-measured here.** They are inputs. A C2W8 artifact that re-argues any of
them is out of scope.

---

## 2. Two INHERITED BUILD REQUIREMENTS — mandatory before any flattening/erosion-rate claim

**B1 — net the designed decay out of every erosion curve (§A27.1, A1).** The designed decay law is
exact (predicted per-tick drop `0.039211` vs measured median `0.039211` over 717 readings) but its
**exponent drifts because `last_write_chunk` moves 0→12 within a slot**, so a raw curve overstates
recovery by up to **34 %** (netting moved C2W6's E1 seed 0 from 9.78× → 6.47×). ⇒ **Every depth /
erosion curve in this wave is reported RAW and NETTED, side by side, per slot, keyed by
`last_write_chunk`.** A claim quoted off a raw curve alone is non-compliant. Direction is known and
conservative (netting makes an arm look *more* eroded, never less) — that does not excuse omitting
it, because this wave's P2 leg is *depth restoration*, i.e. exactly the quantity netting corrects.

**B2 — usage telemetry at I2 grade.** "Truly useless" must be **computable**, or the prune verb is a
guess wearing the word "decision". Registered instrument (§3.2), with the three prerequisites the
C2W6 adjudication established as binding:
- an **item-id key**, never a slot key (slot ≠ well: same-slot site drift is 0.32–0.67× the
  between-slot spread at place radius 0.30);
- a **single registered primary proxy** (§3.2), decided here and not selected on results;
- the **LOO probe reported only with its ICC(1,1)**, and **never as a decision input** — C2W6
  measured ICC negative on 3/3 seeds ⇒ attenuation ceiling 0.000 ⇒ `ρ(LOO)` is *undefined, not a
  null*, and this wave does not repeat that error.

⚠ **B2 is scoped as a DECISION INPUT, not as an I2 re-measurement.** The I2 correlation test
(does usefulness predict erosion?) stays **deferred to C2W10** per §A23.5, and its registered rule
is known to be self-capping and must be re-registered as a test-against-0 before wells are bought.
**This wave uses the telemetry to answer "which wells were never read", which needs no correlation
and no effect size.** ⛔ No C2W8 artifact reports an I2 verdict.

---

## 3. Instrument definitions (fixed here; a cell that needs a different definition files an
   ERRATA block BEFORE it runs)

### 3.1 Well states
For live item `i` with center `c_i`, amplitude `a_i`, width `s_i`:
- `depth_i` — well depth at its own site, geometric-scale reporting (the C2W6 estimator ruling).
- `is_attractor(i)` — `λ_min` at the relaxed site `> 0` **and** `depth_i ≥ θ_att`, where `θ_att` is
  the **measured** capture floor on this rig (SC-6's 32-direction bisection at one site per
  consolidation), not a guessed constant. **Measured before use and reported.**
- ⚠ `is_attractor` is measured, never assumed, precisely because of mechanic 1 (§1).

### 3.2 Usage telemetry `U` (the B2 instrument)
Keyed by **item id**. Accumulated over the whole stream, written by the existing `Controller.touch`
path extended to record reads:
- **REGISTERED PRIMARY PROXY: `read_hits(i)`** = the number of stream reads whose settled point is
  assigned to well `i`'s basin (`_assign` against the live codebook). Chosen because it is
  (a) computable online at O(1) per read, (b) independent of depth (mechanic 2), and (c) the
  quantity the Head's `γ_φ` criterion actually names ("never useful since first appearance").
- **SECONDARY, REPORTED-NOT-DECIDING:** `loss_contribution(i)` by leave-one-out. **Reported only
  beside its ICC(1,1); if ICC ≤ 0 it is labelled UNDEFINED and no number is quoted from it.**
- ⛔ Depth is **not** a usage proxy and does not enter `U`.

### 3.3 The two census populations (the K1 instrument)
At the end of an over-dug stream, at the registered operating point:
- **Prunable population `P`** = fraction of live wells with `is_attractor(i) = True`
  **AND** `read_hits(i) = 0` **AND** not protected (`leak = 0` permanent cohort excluded).
- **Mergeable population `M`** = fraction of live-well PAIRS `(i, j)` admissible for merge under
  the mechanical criteria: payload distance below the registered threshold **AND** center
  separation below the SC-1/SC-2 certificate radius (i.e. the pair is already the near-duplicate
  that over-digging is *supposed* to produce and that the certificate wanted separated).
- Both are reported per seed with their designed negatives (§5, K1).

### 3.4 Over-dig factor
`overdig = n_items_admitted / well_budget`. **Registered at ≥ 2.0** for every census and claim cell —
"over-dig freely" is a measured condition, not an adjective. Cells below 2.0 are diagnostics.

### 3.5 Byte ledger
Every arm, launder included: store bytes + φ bytes + **the trash field's own parameters**
(`K` holes × (`d` center coords + radius + strength)). ⚠ **γ_φ holes are bytes.** A trash region that
is not on the ledger is a hidden capacity increase — the C2W3 `allocate` collapse mode (§A9.6
"ledger drift") in a new costume.

---

## 4. Stage structure (one spoke, two stages — the C2W3 `route3-stage1-plus-2x2` shape, chosen
   deliberately so the kill precedes the build inside one task file)

- **Stage 1 (the rig + the instrument + the census).** Port the CL stream onto the **full CLU
  system** (`CluSystem`, learned `V_θ`) with the CL harness supplying the stream, the baseline
  table and the launder; build the `U` telemetry (§3.2) and the B1 netting; run the census (§3.3).
  Report the **mechanical stage-2 unlock verdict** (K1).
- **Stage 2 (the lifecycle), unlocks ONLY on K1.** Merge-to-budget · prune-below-budget as a
  controller decision · trash routing to `γ_φ(q)` (first use) · depth restoration (A20.6-P2) ·
  the ± consolidation ablation on the long stream.

⛔ **The store is the full CLU's learned `V_θ`, not the CL harness's designed per-item Gaussian
array.** The w25 designed-store numbers (+0.510 / −0.153 / −0.036) enter as a **labelled reference
row**, never as an arm — moving toward per-item arrays to obtain a clean number is intervention
§8.2, and this wave does not do it.

---

## 5. KILL-CONDITIONS (built and asserted before the mechanisms they can kill)

**K1 — THE CENSUS PRECONDITION (the K6 pattern applied to the lifecycle). Stage-1 deliverable;
gates stage 2 mechanically.**
> Registered reading, computed before any verb exists:
> - **UNLOCK** iff `P ≥ 0.05` **or** `M ≥ 0.05`, mean over seeds, on the ≥ 2× over-dug rig.
> - **KILL** iff `P < 0.05` **and** `M < 0.05` on every seed.
>
> **If KILL fires:** prune-below-budget and merge-to-budget have nothing to act on at this
> operating point. Stage 2 is **not built**, the wave re-prices to instrument-only, and **the
> census IS the finding** — reported as a vacuity result with its mechanism, exactly the C2W7
> lesson (*the artifact was real and had nothing to destroy*) caught **before** a build instead of
> after one. That outcome is a wave product, not a failure, and it is published as such.

⚠ K1's own validity needs designed negatives: a **hand-constructed store with 4 known-unread
attractors** must read `P ≥ 4/n_live`, and a **hand-constructed store with 3 known near-duplicate
pairs** must read `M ≥ 3/n_pairs`. Both **pytest-asserted**. A census instrument that cannot see a
planted population cannot license a kill.

**K2 — TRASH-REGION FIRST-USE SAFETY (built before the trash verb).**
`γ_φ(q)` ships **OFF**, and OFF is **bit-identical AND parameter-count-identical** to the pre-build
path (the P1 / psires precedent). Two designed negatives, both pytest-asserted:
- (a) a hole placed **at** a written well's site measurably destroys that well's retrievability
  (the field does something);
- (b) a hole placed **far from every well** leaves every read **bit-identical** (the field does not
  leak).
⛔ Reddening either test un-ships the flag (the standing C2W6 rule for K1/K2 fingerprints).

**K3 — PRUNE IS A DECISION, NOT DEPTH (built before the prune verb). This is mechanic 2 (§1)
instantiated as a test.**
Two planted wells, pytest-asserted:
- a **deep but never-read** well **IS** pruned;
- a **shallow but frequently-read** well is **NOT** pruned.
⛔ If the shipped criterion cannot separate these two cases, it is a depth policy and **it does not
ship** — regardless of how well it performs on the benchmark.

**K4 — MERGE DOES NOT SILENTLY SPEND BYTE-EXACT DELETION (§A9.9, standing).**
Deletion is measured as a **CURVE**: exactness preserved on the unmerged (private) fraction,
**measured degradation** on the merged fraction. Designed negative, pytest-asserted: deleting an
item that was merged shows a measurable departure from `AUC = 0.5000`, **or** the merge is recorded
in the ledger as deletion-destroying for that item. ⛔ Byte-exact deletion (AUC 0.5000 ± 0.0000) is
the program's banked capability and **is never spent silently**.

**K5 — EVERY PERFORMANCE CELL CARRIES ITS TIER-APPROPRIATE CONTROL.**
kNN-in-φ at matched memory (N89 / CM-22(i), the banked mandatory launder) on **every** cell, plus
the ± consolidation ablation (the Add.6 "full CLU with X vs full CLU without X" form), plus the
§3.5 byte ledger on all arms including the launder. ⛔ A consolidation number without its launder in
the same row is non-compliant.

---

## 6. NUMERIC PREDICTIONS (registered priors — the Hub's, before any cell)

| # | quantity | prediction | prior |
|---|---|---|---|
| **N1** | prunable population `P` at `overdig ≥ 2` | **0.15 – 0.45** | P(K1 kills on `P`) = 0.20 |
| **N2** | mergeable pair population `M` at `overdig ≥ 2` | **0.10 – 0.35** | P(K1 kills on `M`) = 0.25 |
| **N3** | **K1 KILL fires** (both below 0.05, every seed) | — | **0.12** |
| **N4** | consolidation ACC dividend (ON − OFF, matched budget, Split-MNIST extension) | **+0.00 to +0.04** | P(> 0 beyond 2 SE, multi-seed) = **0.45** |
| **N5** | ⭐ **launder margin** (CLU − kNN-in-φ at matched memory) — banked at **−0.036** | **−0.03 to +0.01** | P(**crosses 0** beyond 2 SE) = **0.20** |
| **N6** | depth restoration (P2): median well depth after consolidation, **decay-netted**, vs at-write | **≥ 0.5×** on ≥ 2/3 seeds | 0.60 |
| **N7** | Split-CIFAR strong-φ: CLU ACC lift over the banked PCA-φ null | **≥ +0.10** | 0.55 |
| **N8** | Split-CIFAR strong-φ: CLU beats its **own** kNN-in-φ launder | — | **0.15** |
| **N9** | trash routing changes ACC at all (‖Δ‖ > 2 SE) beyond what prune alone gives | — | 0.35 |

⭐ **N5 is the number that matters and I am predicting it does NOT cross.** See §7.

---

## 7. ⛔ A CORRECTION THE WAVE IS SCOPED ON (Hub, flagged to the Head + Advisor)

The §A21 C2W8 row says of the Split-MNIST extension: *"we hold a laundered win"*. **What is
actually banked is the opposite sign on the launder leg.** `claims_matrix.md` CM-23(q) fixes three
sentences that always travel together:

> **+0.510** over the rehearsal-free class · **−0.153** vs iCaRL · **−0.036 LAUNDERED**

plus the **CIFAR-10 NULL as a scope clause**, and the Head's Addendum-2 (2026-07-28) ruling that
this **does not count as an external benchmark won** — supplementary only.

⇒ **We hold a rehearsal-free-class win that is 0.036 BELOW its own kNN-in-φ launder.** The wave is
therefore scoped so that **the launder margin (N5), not the +0.510, is the quantity consolidation
must move.** Beating the rehearsal-free class again would restate a banked supplementary result;
crossing the launder would be the first time the store beats its own trivial substitute on an
external benchmark. ⛔ **No C2W8 artifact may quote "+0.510" without "−0.036 laundered" in the same
paragraph** (CM-23(q), inherited verbatim).

---

## 8. THE SPLIT-CIFAR RETRY'S MANDATORY PROVENANCE

The Split-CIFAR strong-φ arm is **registered as a re-price of the CIFAR null**, whose diagnosed
cause was the **feature space** (w25: "CIFAR-10 = NULL — the feature space, not the discipline"),
which the strong-φ policy §A4.3 addresses directly (`phi_encoders.py`: `randconv` / `convae` /
`simclr`). ⛔ **It is NEVER quoted without that provenance**, in any artifact, draft or table. The
registered form:

> *"Split-CIFAR was a null at frozen-PCA φ; re-priced at strong φ (arm, bytes ledgered), it reads X."*

⛔ A re-price is not a new benchmark entry, and a favourable X does not retire the null — it scopes
it to the feature space, which is what was diagnosed.

---

## 9. Declared NOT-RUNs (declared in advance; **never reported as nulls** — standing rule)

- **The I2 correlation test** (does usefulness predict erosion?) — deferred to C2W10 per §A23.5,
  with its rule re-registration as a test-against-0 as C2W10's entry condition. C2W8 builds and
  uses the telemetry; it reports **no I2 verdict**.
- **The cross-stream / persistent-store criterion** ("wells never useful over k streams → trash",
  the A20.6 Head addition) — that is **C2W10's**, and it needs this wave's trash plumbing plus
  C2W10's cross-stream telemetry. C2W8 builds the plumbing and stops.
- **Wormholes / traversal / learned p₀ as a reach lever** — C2W9, gated on its own
  traversal-failure signature. Not this wave.
- **Any tier-ii verdict / organizer swap** — unreached for a third wave (§A26.1); the next tier-ii
  iteration is **write-side** (§A26.6) and is the Head's decision, not this wave's.
- **Any full-CLU verdict** — ⛔ forbidden outright (§A28.4). C2W8 has no ψ work, no traversal, no
  persistent store, and its result is **vehicle-scoped and component-scoped**.
- **CSF3** — untouched. Runs 1/2 are the Head's; run 3's config decision consumes C2W6 + C2W7, not
  this wave.

---

## 10. Epoch / promotion discipline

**N94** applies to every cell: any reading below the undemoted write-step floor is labelled
**non-promotable** and travels with that label (monitor #13). The census may run at demoted budget
and say so; **no claim cell may.** Multi-seed (≥ 3, ≥ 5 where a margin is the claim) before any
number leaves the wave.

---

*Filed by the C2W8 Hub, 2026-08-06, before any harness cell. Corrections go in a dated
`ERRATA-C2W8.md`; this file is not edited.*
