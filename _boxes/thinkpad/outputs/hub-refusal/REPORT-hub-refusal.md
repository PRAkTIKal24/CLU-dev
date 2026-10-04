# `hub/refusal` — REPORT

**Agent:** `autonomous-hub` · **Branch:** `hub/refusal` @ `8e36bc9` (tree = pin `4125fce7…`)
**Machine:** FORGIS_14, Windows AMD64, CPU, 4 threads · **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-refusal`
**Start:** `2026-09-16T19:10:36Z` · **Deadline:** `2026-09-17T07:10:36Z` (12 h true wall clock)
**Running log:** `LOG.md` (dated bullets, written as I went) · **Pre-registration:** `PREREG-hub-refusal-1.md` + AMENDMENTS 1–9

---

## STATUS: ⛔ **NOT SOLVED — and the reason is measured, not guessed**

**Scored runs spent: 0 of the target's 3.** ⛔ No row of `A`, `B`, `C`, `A7` or `A7D` has
been scored by this hub. Two hard gates stand in front of the scored run, one of them
mine to respect and one of them a property of this machine:

1. ⛔ **The target-wide scored-run count is unknown to me** (COMMON §1c) and I was told not
   to spend the first one until a human states it. I have not.
2. ⛔⛔ **This box can produce SEED 0 ONLY.** Seeds 1 and 2 would require ~23,480 rows
   re-encoded through the frozen encoder; COMMON §4 says *"If you find yourself about to
   re-encode the feature cache, STOP and report instead."* I stopped. **The target
   requires 3 seeds, so the refusal win is NOT ACHIEVABLE ON THIS MACHINE at any budget.**

⭐ **What this hub did instead, at zero held-out cost:** it built and validated the scorer,
pre-registered the criterion, and then used the free half of the charter (§A.1: *"Iterate
freely on MECHANISM"*) to answer the mechanism question in analogue. **The answer is a
clear negative with a structural explanation**, and it is the most useful thing here.

---

## 1. THE ONE-PARAGRAPH ANSWER

A single landscape-derived criterion — `S_max = max_t [E_t(x_settled) − E_store(x_settled)]`,
the worst-satisfied request term at the settled state — **does not refuse both kinds**, and
the two reasons are different and both structural.
⭐ **`duplicate_factor` IS visible to the landscape, but only at the LAUNCH**: the criterion
separates it from its A7 control at **0.7511** (rung A) / **0.8124** (rung B) with zero
settling, and **1000 steps of descent destroy that**, down to **0.5825** / **0.6606** —
⭐ with **75 % of the loss inside the first TEN steps**.
⛔ **`out_of_range` is invisible at every rung, every arm and every step budget
(0.48–0.56)** — because the address book is a distribution over *stored atoms*, and **a
simplex has no outside**: however far off-grid a value goes, the expert it induces is
still satisfiable by some real atom.
⛔⛔ **And the store is NOT load-bearing at either budget**: with every atom's content
destroyed, the criterion scores **0.7809** at the launch against the full store's `0.7511`,
and `0.5888` settled against `0.5825`. **Task falsifier 4 fires — the landscape is
decorative for this criterion**, and the small signal that exists is a property of the
**address book's index structure**, not of the stored atoms.
⛔ These are measured on a **fit-derived synthetic analogue at seed 0**, ⛔ not on the
benchmark. They are stated as predictions the C set can refute.

---

## 2. WHAT WAS BUILT, AND THAT IT IS THE SHIPPED READ

`scorer/refusal_lib.py` **calls** `exp_gen_dynamical_read.build_landscape`,
`gen_landscape.settle_chain`, `gen_landscape.request_energy` and `convergence_report`.
⛔ It re-implements none of them, and ⛔ it edits no manifest-pinned file — everything lives
under `$CHLU_HUB_OUTPUTS/hub-refusal/`.

**The one new piece of machinery** is `term_experts`: one expert per requested **term**
instead of per **factor**, because the shipped `request_experts` takes a dense `(n, F)`
array of grid class indices and **cannot express either invalid kind** (one factor twice;
a value with no class index). Off-grid values are resolved by the **shipped R8 rule**,
`gen_request._extrapolating_index`, imported not re-derived, with the extrapolated
conditional renormalised over the well's atoms.

**Validation, measured (seed 0, rung A, request set `MS` — seen rows, zero held-out cost):**

| quantity | Mac artefact / log | this box | verdict |
|---|---|---|---|
| `term_experts` vs shipped `request_experts` on valid | — | `max_abs_delta = 0.0`, **bit-identical** | ⭐ it *is* the shipped arm on the accept half |
| λ from the coercivity certificate | `0.0485403` | `0.04854` (λ\* `0.05393`) | ⭐ same |
| byte ledger rung A | 524,288 B store | identical, **PREREG §3 match `True`** | ⭐ same |
| `E first → final` on `MS` | `-602.588 → -690.003` | `-602.5878520927 → -690.0028908469` | ⭐ 6 s.f. |
| `frac_still_descending` on `MS` | `0.6919191919191919` | `0.6919191919191919` | ⭐ **exact** |
| `x_settled` | — | max\|Δ\| `2.4e-3` on rms `0.710`, mean\|Δ\| `8.2e-6` | ⭐ ~1e-5 rel. |
| grouping re-derivation | — | reproduces the pinned plan **bit-identically** | ⭐ |
| `request_off` `frac_still_descending` | §E.4: "**1.00** on both request-off rows" | `1.0` | ⭐ exact |

⛔ Per COMMON §4 the parity check is a **diagnostic**; ⛔ no Mac-produced quantity enters any
number reported here.

---

## 3. THE PRE-REGISTERED CRITERION (unchanged, and it stays unchanged)

> **`S_max = max_t r_t(x_settled)`**, `r_t = E_t − E_store` at the reference bandwidth;
> refuse iff `S_max > quantile(S_max on the repaired stratified **val** split, q = 0.95)`.

- **`q` is not mine.** `0.95` is the harness's shipped `refusal_valid_quantile` — the same
  constant, to the digit, that every rival's published threshold uses. ⛔ Not swept.
- **`val` is seen-valid and not held out**, so the threshold spends **zero** held-out budget.
- **`max_t`, not `Σ_t`, and the reason was declared before any invalid row was seen:**
  `Σ_t` is a **term counter** (7-term `duplicate_factor` vs 6-term valid), which the
  registry itself says scores AUROC 1.0000 on that half. `S_sum` was pre-registered as a
  companion **explicitly rejected in advance**, so its number could never be adopted later.

---

## 4. THE RESULTS LEDGER

| # | run | scored? | seeds | status | where |
|---|---|---|---|---|---|
| 0 | rival re-score (a **JOIN** on `gen-val-split-fix/refusal_scores.json`) | ⛔ no — no new computation on any held-out row | 3 | done | `RIVAL_ROWS.json` |
| 1 | free units `val` + `MS` × {full, request_off, zero_step, store_removed} | ⛔ no (`val`/`MS` are not held out) | **0 only** | done; seeds 1–2 **aborted by the re-encode gate** | `runs/LEDGER.jsonl` |
| 2 | plumbing smoke (fit rows; shape/finiteness/branch coverage only) | ⛔ no | 0 | PASS | `scorer/plumbing_smoke.py` |
| 3 | synthetic mechanism probe, rung A — **corner bases, superseded by AMD 5** | ⛔ no | 0 | done, **confounded** | `SYNTHETIC_PROBE_seed0_A_CORNER-BASES-AMD5-SUPERSEDED.json` |
| 4 | synthetic mechanism probe, rung A — random bases (AMD 5) | ⛔ no | 0 | done | `SYNTHETIC_PROBE_seed0_A_AMD5.json` |
| 5 | synthetic mechanism probe, rung B (AMD 6) | ⛔ no | 0 | done | `SYNTHETIC_PROBE_seed0_B_AMD6.json` |
| 6 | laundering curve, step ladder (AMD 7) | ⛔ no | 0 | done | `LAUNDERING_CURVE_seed0_A.json` |
| 7 | §C.4 floor check (lookup-table test) | ⛔ no | 0 | **PASS** | `FLOOR_CHECKS.json` |
| 8 | simplex-blindness measurement, 6 severities (AMD 8) | ⛔ no | 0 | done | `SIMPLEX_BLINDNESS_seed0_A.json` |
| 9 | §C.2 floor check **at the launch** (AMD 9) | ⛔ no | 0 | ⛔ **FAILS — store not load-bearing** | `SYNTHETIC_PROBE_seed0_A_AMD9.json` |
| 10 | independent replication from the clean worktree (§A.5) | ⛔ no | 0 | ⭐ **BIT-IDENTICAL** | `REPLICATION.json` |
| — | **THE SCORED RUN** | — | — | ⛔ **NOT RUN — gated on the Head's count, and single-seed here in any case** | — |

⭐ **Every number above is Windows-produced, from valid or fit rows only.**
⛔ `gen-arms-a7-wiring/refusal_scores.json` was never opened; `scorer/score.py` asserts the
**directory** (not the basename) before reading.

---

## 5. THE BAR, RE-DERIVED (the rivals' AUROCs **and their pairs**)

3-seed mean, rivals at **their own** threshold (set `A`, `q = 0.95`). `RIVAL_ROWS.json`.

| rival | pooled C-vs-B | A7ctrl **oor** | A7ctrl **dup** | accept **B** | accept **A7** | reject oor | reject dup |
|---|---|---|---|---|---|---|---|
| hist/knn · knn_proto | 0.8457 | **0.9712** | 0.3103 | 1.0000 | **0.4120** | **0.8426** | 0.3519 |
| hist/knn_compsel | 0.8522 | **0.9738** | 0.3261 | 1.0000 | 0.3981 | 0.8241 | 0.3241 |
| hist/lace · ld_published | 0.8097 | 0.6267 | 0.7846 | 0.9444 | **0.4028** | 0.0833 | **1.0000** |
| hist/lace · ode | 0.7220 | 0.4443 | **0.9906** | 0.9583 | 0.9537 | **0.0278** | **1.0000** |
| depl/knn · knn_proto | 0.8357 | **0.9583** | 0.3090 | 1.0000 | 0.3843 | 0.8148 | 0.3889 |

⭐ **Two facts this table establishes that the task file's three numbers do not show:**
1. ⛔ **No existing arm refuses both kinds.** Best "both kinds" row is kNN at
   **(0.8426, 0.3519)**; LACE's best dup row refuses **2.8 %** of `out_of_range`.
2. ⭐⭐ **Every rival except `lace/ode` falsely refuses ~60 % of the VALID A7 control**
   (kNN accepts `B` at 1.0000 and `A7` at **0.4120**), because R8's embedding adds a
   repeated term's deviation twice. ⭐ **Any future refusal claim must print `accept A7`
   beside `accept B`**, or it is reporting half a number.

---

## 6. THE MECHANISM RESULT — ⛔ fit-derived synthetic, seed 0, **NOT the benchmark**

### 6.1 `duplicate_factor` is visible — **at the launch**, and the settling destroys it

| rung | arm | AUROC(dup vs a7) | AUROC(oor vs valid) | AUROC(**a7 vs valid**) |
|---|---|---|---|---|
| A (64 atoms) | launch (0 steps) | ⭐ **0.7511** | 0.5527 | **0.4818** |
| A | settled (1000) | 0.5825 | 0.5381 | **0.4904** |
| B (512 atoms) | launch (0 steps) | ⭐ **0.8124** | 0.5522 | **0.5093** |
| B | settled (2000) | 0.6606 | 0.4832 | **0.4819** |

⭐ **The laundering costs −0.169 (rung A) and −0.152 (rung B) AUROC — it is not a rung
artefact.** And the step ladder says *where*:

| steps | 0 | 10 | 25 | 50 | 100 | 250 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|
| AUROC(dup vs a7) | **0.7511** | **0.6250** | 0.5918 | 0.5848 | 0.5825 | 0.5819 | 0.5823 | 0.5825 |
| mean `S_max`(dup) | **8.029** | **1.730** | 1.847 | 2.069 | 2.249 | 2.448 | 2.568 | 2.653 |

⭐⭐ **74.8 % of the loss is gone in the first TEN of a thousand steps**, and after ~50 the
curve is flat. ⚠ Meanwhile `still_descending = 0.67` — **the chain is still descending in
energy while being completely flat in discriminative content.** Energy descent and refusal
information are decoupled.

**Why, mechanically, and §C.4 says it in one line.** The §C.4 floor check passes: the
settled state sits **0.70 of the atom spread** from its nearest atom (rms `14.98` vs
`21.26`), while the launch sits **exactly on** an atom (`0.0000`). ⭐ The launch is a real,
satisfiable configuration, so a conflicted request's worst term is glaring there. Ten MALA
steps later the state is out in the space *between* atoms, where **no term is badly
violated because none is exactly satisfied either**. ⭐⭐ **§C.4 passing and the refusal
criterion failing are the same fact**: the freedom that makes the read "not a lookup
table" is what dissolves the evidence of unsatisfiability.

### 6.2 ⛔⛔ `out_of_range` is STRUCTURALLY invisible — a prediction, with its argument

`0.4814–0.5558` in **every** arm, at **both** rungs, at **every** step budget, at three
severities (`w ≈ 2, 3, 4`).

The address book is `log P(atom | f = v)` — for each value, **a point on the simplex over
the well's stored atoms**. R8's extrapolation applied to it yields
`(1−w)·log_cond[i] + w·log_cond[j]`, renormalised — **still a distribution over those same
atoms.** ⭐ **The simplex has no outside.** However far off-grid the value goes, the expert
is satisfiable by *some* stored atom, and `r_t = −σ² log E_p(x)[e^{W_t}]` goes strongly
negative the moment the state sits there — which is exactly where the launch's argmax puts
it and where the chain keeps it.

⭐ **Contrast, and it explains the rival exactly:** R8 gives kNN the *same* rule in
**feature space** (`proto = P[i] + w(P[j] − P[i])`, `|w| > 1`), which moves the target
**off the feature manifold** so its nearest-neighbour distance grows without bound — the
module's own words, "the embedding moves further off-manifold the more invalid the request
is". kNN scores **0.9712**. ⛔ The landscape cannot do that.

> ⛔⛔ **PREDICTION, stated before the C set is scored so that the C set can refute it:
> no criterion derived solely from the settled energy and the atom address book will clear
> the `out_of_range` bar.** To see `out_of_range` a criterion must touch the *value
> geometry* — and a criterion that does is no longer "a function of the read's own settled
> energy and its convergence", which the task file requires in bold.
> ⭐ **That is a genuine tension between the target's two clauses**, and it is the sharpest
> thing this hub found.

### 6.3 ⭐ The one prediction that landed — and it is worth keeping

**P3, term-count invariance.** `AUROC(a7 vs valid)` = **0.4818 / 0.4904 / 0.5093 / 0.4819**
across rungs and arms, and **flat at 0.490–0.509 across the entire step ladder**. ⭐ `S_max`
is **provably not a term counter** — the axis on which kNN (0.3103) and `ld_published`
(0.7846 with 60 % false refusals on A7) both break. ⛔ It is not, on its own, a win.

### 6.4 ⛔⛔ THE STORE IS NOT LOAD-BEARING — measured at **both** the launch and the settled state

My task file's falsifier 4: *"⭐ **The store is load-bearing** — `store_removed` destroys
the criterion. ⛔ If an energy computed without the store refuses just as well, the
landscape is decorative."* §C.2's control, with every channel column of every atom
independently permuted (**byte count unchanged, marginals unchanged, no atom a train row**):

| arm | AUROC(dup vs a7) | mean `S_max` dup |
|---|---|---|
| `zero_step` — launch, **full store** | 0.7511 | 8.0294 |
| ⛔ **`store_removed_zero_step`** — launch, **atoms destroyed** | ⛔ **0.7809** | 3.7720 |
| `full` — settled, full store | 0.5825 | 2.6534 |
| ⛔ `store_removed` — settled, atoms destroyed | ⛔ **0.5888** | 0.6618 |

⛔⛔ **Destroying the store does not degrade the criterion at either step budget — it
slightly improves it. FALSIFIER 4 FIRES. THE LANDSCAPE IS DECORATIVE FOR THIS CRITERION.**

⭐ **Where the signal actually lives.** At the launch, `x0` is the argmax atom of the
request posterior `log_prior + Σ_t W_t`; a request with two incompatible terms has a
posterior that no single atom satisfies, so whichever atom is picked leaves one term badly
violated. That is a property of `log P(atom | f = v)`'s **index structure** — ⛔ not of what
the atoms *contain*, which is exactly why scrambling the contents leaves it intact.
⭐ **It is a naive-Bayes consistency check wearing a landscape's clothes.**

⭐ A built-in consistency check that passed: `request_off_zero_step` is **numerically
identical** to `zero_step` (`0.7511`, every column equal) — correct by construction, since
at zero steps β never enters (the launch uses the request *posterior*, never the request
*energy*). A difference there would have convicted my arm composition.

### 6.5 PREREG falsifiers, scored against the analogue

| falsifier | verdict (analogue) |
|---|---|
| **F3** — A7 refused like `duplicate_factor` (a term counter) | ⭐ **DOES NOT FIRE** — `0.48–0.51` at every rung, arm and step budget |
| **F4** — the launch refuses as well as the settled state | ⛔ **FIRES, and worse: the launch refuses BETTER** |
| **F5 / task falsifier 4** — `store_removed` does not collapse | ⛔⛔ **FIRES AT BOTH BUDGETS**: `0.7809` ≥ `0.7511` (launch), `0.5888` ≥ `0.5825` (settled) |
| **F6** — above chance on one kind, at chance on the other | ⛔ **FIRES** — `out_of_range` at chance everywhere, with the sign backwards |
| **F1 / F2** — the 3-seed pair against the rivals | ⛔ **NOT MEASURED** — gated, and single-seed here |

### 6.6 ⭐ INDEPENDENT REPLICATION (charter §A.5) — **BIT-IDENTICAL**

Charter §A.5: *"a separate spoke re-runs the hub's winning configuration from its PREREG
file on a clean checkout of the hub's branch. Unreplicated = not a result."*
`scorer/replicate.py` was run with `CHLU_ROOT` pointing at the **worktree**
(`/c/Users/PratikJawahar/chlu/CHLU-hub-refusal`) and the frozen data roots at the bundle —
so the **code** is the clean checkout's and the **data** is the pin's. It asserts
import-under-root first (`chlu` imported from `CHLU-hub-refusal/chlu/__init__.py`), then
re-runs the `MS` unit:

| field | bit-identical | max\|Δ\| | n |
|---|---|---|---|
| `S_max`, `S_sum`, `S_mean`, `e_total`, `e_store`, `launch_S_max`, `rel_tail_drop` | ⭐ **True** (all 7) | `0.000e+00` | 198 each |
| `x_settled` | ⭐ **True** | `0.000e+00` | **405,504** |

⭐ `experts_reproduction.bit_identical = True`; `frac_still_descending = 0.6919191919191919`
— the same value as the Mac's log. `REPLICATION.json`.

---

### 6.7 ⭐⭐ THE SIGN IS BACKWARDS ON `out_of_range` — the structural claim, MEASURED

AMENDMENT 8 turned §6.2's argument into a measurement: the **perturbed term's own
residual**, on-grid and at six extrapolation severities (128 fit-derived bases, rung A).

| severity (grid steps out) | `w` | `max_a P(atom\|term)` | entropy (nats, max `log 64` = 4.159) | **`r_t` of the perturbed term, settled** | **frac. SATISFIED (`r_t < 0`)** |
|---|---|---|---|---|---|
| **0 — on-grid control** | 0 | 0.0292 | 4.051 | **−16.211** | 0.688 |
| 1 | 2 | 0.0313 | 4.015 | −20.837 | 0.695 |
| 2 | 3 | 0.0366 | 3.973 | −25.753 | 0.727 |
| 3 | 4 | 0.0444 | 3.927 | −31.535 | 0.766 |
| 6 | 7 | 0.0743 | 3.755 | −51.274 | 0.852 |
| **12** | **13** | **0.1473** | **3.368** | **−87.097** | **0.891** |

⛔⛔ **The further off the grid a request goes, the MORE SATISFIED the landscape reports its
term to be — monotonically, over a 13× extrapolation.** `r_t` runs **−16.2 → −87.1**; the
fraction of off-grid terms the landscape calls *satisfied* rises **0.688 → 0.891**.
⛔ **The criterion does not merely fail to see `out_of_range`; it has the SIGN BACKWARDS on
it, and increasingly so with severity.**

⭐ **Mechanism, measured:** as `|w|` grows the renormalised extrapolation gets **sharper on
the same fixed set of stored atoms** — `max_a P` rises `0.0292 → 0.1473` (5.0×), entropy
falls `4.051 → 3.368`. A sharper distribution over **real atoms** is a term that is *easier*
to satisfy. ⭐ The residual ~0.54 AUROC is **collateral damage to the other five terms**, not
detection of the off-grid one: `d(settled → the expert's argmax atom)` stays flat at
`27.3–27.8` while `d(settled → nearest atom)` moves only `14.07 → 15.72`.

⚠ The renormalisation is mine (PREREG §2), but the direction is not: it comes from
**sharpening**, inherent to log-linear extrapolation. Dropping the renormalisation makes
`W` large and positive, so `E_p[e^W]` grows and `r_t` goes **more** negative still — same
direction, more extreme. ⚠ **That is an argument, not a measurement**, and it is labelled
as one.

> ⭐⭐ **DESIGN WARNING, with a falsifiable edge:** *any* confidence score derived from the
> atom address book is expected to be **ANTI-correlated with out-of-range severity**,
> because the address book is a simplex over stored atoms and R8's extrapolation makes an
> off-grid value a **sharper point on it**. ⛔ Cheap for the C set to refute.

---

## 7. ⛔ WHAT I DID NOT DO, AND WHY IT MATTERS MORE THAN WHAT I DID

⛔ **I did not change the criterion after seeing the probe.** PREREG AMENDMENT 3, written
*before* the probe ran, binds it: *"If this hub subsequently changes the criterion or the
threshold rule after seeing this probe, the zero-supervision claim is VOID."*
⭐ **The obvious better criterion is the LAUNCH-time statistic — 0.7511 / 0.8124 instead of
0.5825 / 0.6606 — and I can see it only because I looked at invalid examples.** Promoting
it would be exactly the failure my task file names: *"a threshold tuned on the very data it
is meant to refuse will look excellent and mean nothing."*
⭐ It is carried forward as a **recommendation for a PREREG-2 whose first line states that
its statistic was chosen after seeing invalid examples, and whose zero-supervision claim is
therefore explicitly VOID.** ⛔ Never as a silent edit to PREREG-1.

⛔ **I did not re-encode the feature cache** to unlock seeds 1–2, and ⛔ I did not ask for
permission to — COMMON §4 already answered that question.
⛔ **I did not weaken a check.** The preflight re-encode gate was called through its own
`will_extend=False` scoping API with a written reason, and `would_have_failed: true` is
recorded in every provenance file; the enforced gate at the point of extension was left at
full strength and **it is what stopped seeds 1–2**.
⛔ **I did not quote `gen-arms-a7-wiring/refusal_scores.json`.** `score.py` asserts the
directory before reading.
⛔ **I did not spend a scored run.**
⭐ **I withdrew my own recommendation** (§8.3) the moment the launch-time §C.2 control
refuted it, rather than leaving a plausible-looking suggestion in a report for someone else
to spend a slot on.

---

## 8. RECOMMENDATIONS (in priority order)

1. ⭐⭐ **Run the scored run on the Mac, not here.** `RUN-ON-THE-MAC.md` is four commands
   against an already-filed PREREG; ~30–45 min CPU for 3 seeds × 4 arms × 7 sets.
   ⛔ Spending a single-seed slot on this box burns a slot that cannot produce the headline.
2. ⭐ **Expect PREREG-1 to return a partial/negative on the benchmark**, and treat that as
   the point: §6.2's prediction is falsifiable and cheap to falsify.
3. ⛔⛔ **WITHDRAWN — do NOT spend a slot on the launch-time `S_max`.** An earlier draft
   of this report recommended it (+0.17 AUROC on `duplicate_factor` in analogue). ⛔ **The
   §C.2 floor check applied at the launch (AMENDMENT 9, §6.4) refutes it:
   `store_removed_zero_step` scores `0.7809` against the full store's `0.7511`.** The
   launch-time signal is **not landscape-derived** — it is the address book's index
   structure — so a refusal claim built on it would not be a CHLU claim at all, and judging
   by `ld_published`'s `0.7846` on `duplicate_factor` it is ⛔ **not unclaimed either**.
   ⭐ I am withdrawing my own recommendation before anyone acts on it; the §C floor would
   have rejected the number on arrival.
4. ⭐ **Report `accept A7` beside `accept B` on every refusal row, programme-wide.** Three
   of the five rival rows falsely refuse ~60 % of a valid control and the published AUROC
   table does not show it.
5. ⚠ **Fix the bundle, not the science:** seeds 1 and 2 need `cache_scope="full"` (or a
   cache covering `store_train`) for any second-machine work at all. This is the second
   hub-blocking consequence of the `"used"` scope; `gather_features`' docstring already
   flagged the first.
6. ⚠ **§E.4's open question is answered in analogue and the answer is NO.** "A landscape
   can be a bad generator and a good detector" — for this criterion the dynamics hurt
   *both*, in the same direction and by a comparable margin (`0.6852 → 0.2114` generation;
   `0.7511 → 0.5825` refusal).

---

## 9. COST LINE

| item | cost |
|---|---|
| Compute | **local CPU only**, 4 threads (`OMP`/`MKL`/`OPENBLAS`), ~1.6 h of process time |
| GPU | ⛔ **none** |
| GCloud / CSF3 | ⛔ **none** — nothing submitted, nothing spent |
| Money | ⛔ **£0.00** |
| Held-out rows scored | ⛔ **0** |
| Scored runs spent | ⛔ **0 of 3** |
| Manifest-pinned files edited | ⛔ **0** |
| Files deleted | ⛔ **0** (§0.1) |
| Write footprint | `$CHLU_HUB_OUTPUTS/hub-refusal/**` only |
| Git footprint | `hub/refusal` @ `8e36bc9`, **no commits** — all artefacts are under gitignored `.claude/`, carried by `--return-pack` |

---

## 10. ⛔ THE TWO QUESTIONS FOR THE HEAD

1. **How many of the three scored runs remain for the REFUSAL target, across all machines?**
2. **Given that this box can produce seed 0 only — spend a single-seed slot here, or hand
   the scored run to the Mac?** ⭐ **My recommendation: the Mac, and zero slots here.**

**Elapsed at filing: ~1.1 h of 12 h true wall clock. Task NOT SOLVED.** ⭐ It is not solved
because the measurement is gated and because this machine cannot produce the seeds the
target requires — ⛔ not because the clock ran out, and ⛔ not because a threshold was moved
until something passed.
