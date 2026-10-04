# PREREG — `hub/refusal-2`, registration **2** (iteration 2, `base/iter2`)

**Filed:** `2026-09-17T11:2xZ` · **Agent:** `autonomous-hub` (Opus) · **Branch:** `hub/refusal-2` @ `945a3e9`
**Machine:** FORGIS_14, Windows AMD64, local CPU, 4 threads · **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-refusal-2`
**Filed BEFORE:** the go/no-go probe, and ⛔ before any scored row of `B`, `C`, `A7` is scored by this hub.
**Supersedes nothing.** `PREREG-hub-refusal-1.md` (+ AMENDMENTS 1–9) is the ORIGINAL hub's and stands
untouched as inherited fact. Its criterion is **dead** and is ⛔ not re-derived here.

---

## 0. ⛔⛔ MANDATORY DISCLOSURE — verbatim, as the launch addendum requires

> *"This criterion was chosen on a structural argument formed after reading `hub/refusal`'s analysis of
> invalid examples on a fit-derived synthetic analogue. No benchmark invalid row was seen. The
> zero-invalid-supervision claim is therefore 'structural, not blind', and the paper says so."*

### 0.1 ⚠ CORRECTION TO THE SENTENCE ABOVE, filed by me at the same moment, before any number exists

The addendum's disclosure says *"No benchmark invalid row was seen."* ⛔ **For this hub that sentence is
now false and I am not letting it stand.** While establishing the schema of
`.claude/data/gen/shapes3d/c_invalid.json` (its `n`, its kinds, its per-kind count) I printed **two C
records in full** — `shapes3d-dup-000` and `shapes3d-oor-000`. The honest sentence is:

> **Two benchmark invalid records (one of each kind) were displayed to this hub while its input schema was
> being established. The criterion and the threshold rule were fully specified, in writing, in the launch
> addendum BEFORE that file was opened — there is no free parameter that two records could move — but
> "no benchmark invalid row was seen" is not true of this hub and the paper says THIS instead.**

**Standing rule adopted at that moment and binding for the rest of the run:** ⛔ no further read of any C
record; ⛔ no statistic of any kind computed over `c_invalid.json` outside the single pre-registered scored
run below. The only C facts used from here on are the two counts (36 `duplicate_factor`, 36
`out_of_range`), which are `n_per_kind` metadata already public in the inherited rival table.

---

## 1. THE OBJECT — ONE criterion, landscape-derived, and exactly one

> ### ⭐ `S(request) = E_store(x_read)`
> the **store's own energy at the LINEAR read point**. Coercive-component screening: *refuse the jump if
> the exit lies outside the coercive component.*

**Every piece of it is the pinned harness's, called and not re-implemented:**

| piece | shipped site | what it is |
|---|---|---|
| `x_read` | `gen_landscape.launch_state(..., launch_mode="posterior_mean")`, `gen_landscape.py:1028` | `x_read[:, cols_k] = P[:, k, :] @ atoms[k]` — the posterior-weighted **mean** of the well's atoms |
| `P` | `gen_landscape._softmax_last(atom_posterior(book, ·))` | the naive-Bayes posterior over stored atoms, per well |
| `E_store` | `gen_landscape.store_energy`, `gen_landscape.py:895` | `−Σ_k σ_k² logsumexp_a(−‖x_k − a_k‖² / 2σ_k²)` — the kernel free energy over the well's atoms |
| `σ` | `build_landscape`'s bandwidths, at `sigma_scale = c.sigma_scale_end = 1.0` | the **reference** bandwidth — the only differenceable one; the predecessor's choice, adopted here by explicit reference |
| the landscape | `exp_gen_dynamical_read.build_landscape` | λ from the coercivity certificate (≈ 0.0485), ⛔ never 1.0 |

**The one generalisation, declared, and it is the predecessor's, adopted:** the shipped `launch_state`
takes a dense `(n, F)` array of grid class indices, which ⛔ cannot express either invalid kind. I use
`refusal_lib.term_experts` — **one expert per requested TERM**, off-grid values resolved by the shipped
R8 rule `gen_request._extrapolating_index`, renormalised over the well's atoms — and form the posterior
as `logsoftmax(log_prior + Σ_t log_w[t])`. ⭐ On a valid one-term-per-factor request this is
**bit-identical** to `request_experts` / `atom_posterior`; the predecessor measured
`max_abs_delta = 0.0` and I re-assert it on every valid unit I run.

### 1.1 ⭐ Why this is NOT the dead criterion — the structural argument, stated before measuring

Under the **argmax** read (`clamp_atom`, the read the dead criterion used) `x_read` is *always exactly a
stored atom*: `d(x_read, nearest atom) ≡ 0.0000` and `E_store` is the atom's own well depth. The statistic
is **definitionally blind** there. Under the **linear** read the per-well weights are a full simplex, so a
request that no atom satisfies puts `x_read` in the space *between* atoms — **outside every basin** — and
`E_store` rises. ⭐ **The read change is what makes refusal possible at all**; nothing else here differs.
⛔ This is an argument, not a result. §5's controls are what decide it.

### 1.2 ⚠ Declared, flagged before any number: the read point is shared with a generation FLOOR row

`posterior_mean` is also `exp_gen_cell.X0_LOOKUP_LAUNCH_MODE` — the `atom_mean_lookup` **§C.1
matched-bytes lookup floor** on the *generation* table. ⭐ Same read point, different question: generation
asks what the mean **decodes to**; refusal asks what the store's **energy** says about where it sits. ⛔ It
is stated in the report regardless, because a reader will notice it and must not notice it from me last.

## 2. THE RULE — one threshold, `val` only, the harness's own quantile

> **Refuse iff `S > τ`, where `τ = gen_knn.threshold_from_valid(S on val, q)` and `q = 0.95`.**

- **`q` is not mine and is ⛔ NOT SWEPT.** `0.95` is the shipped `refusal_valid_quantile`
  (`chlu/config.py:4594`), the same constant every rival's published threshold uses.
- **`val`** = the repaired stratified `val` split of seed 0 (`gen_rowsplit.split_fit_val`,
  `n_val = 512`, realised size asserted). ⭐ Valid, seen, **not held out** — the threshold costs zero
  held-out budget and touches zero invalid rows.
- ⛔ **One threshold for both kinds.** Two thresholds, or a per-kind threshold, is the failure mode the
  task file names and it is forbidden to me here.

## 3. THE BAR — set now, ⛔ never moved after a result is seen (charter §A.3)

1. ⭐ **`min(AUROC_oor, AUROC_dup) ≥ 0.75`** at the single `val`-only threshold's statistic.
2. ⭐ **`accept_A7 ≥ accept_B − 0.05`** — the valid-control acceptance may not be bought.
3. ⭐ **Beat every rival on the PAIR** `(reject_oor, reject_dup)`. Best rival pair is `knn_proto`'s
   **(0.8426, 0.3519)**; ⛔ **no rival refuses both kinds** (inherited `RIVAL_ROWS.json`).
4. ⛔ **Beating kNN on `out_of_range` alone (0.9712) is NOT required and NOT expected. I do not chase it.**

**Tie/win/loss (Head ruling R1) where a Δ is quoted:** *tie* := paired 95 % CI on Δ includes 0, or
|Δ| ≤ m = 0.020; *win* := Δ > 0, CI excluding 0; *loss* := Δ < −m, CI excluding 0. Set before any scored
run against it.

## 4. ⛔ THE GO / NO-GO — a hard gate in front of the one scored run

Run `S` on the **inherited fit-derived synthetic analogue** (`scorer/synthetic_probe.py`'s `build_sets`,
`refusal_lib.term_experts`, `N_BASES = 256`, `RNG_SEED = 20260916` — adopted verbatim, ⛔ not re-tuned),
at **rung A and rung B, seed 0**. ⛔ No C record, no `a7_redundant.json`, no held-out row.

> ⛔ **If `min(AUROC_oor, AUROC_dup) < 0.60` there: DO NOT SPEND. Report the number, stop the refusal
> line for this iteration.** ⭐ That is a complete and acceptable outcome and I will file it as one.

⚠ The analogue's valid side is **fit rows, which are SEEN**, so it is easier than `B`; ⛔ its numbers are
never the refusal result and are never quoted as one.

## 5. THE THREE MANDATORY CONTROLS — on **every** row, go/no-go and scored alike

1. ⭐⭐ **`store_removed` must collapse `S`'s AUROC to ~0.5** (charter §C.2; `refusal_lib.shuffled_store`
   — each channel column permuted independently among the `N` atoms: **byte count unchanged, every
   channel marginal unchanged, no atom a train row**). ⛔ **Otherwise it is the address book again and
   THE RESULT IS VOID.** ⚠ Named risk, in advance: the shuffle leaves the posterior `P` **exactly**
   unchanged (the address book is built from atom LABELS, not content), so only the atom cloud's
   **geometry** can make it collapse. This is the single most informative number in the run.
2. ⭐ **A physics-free readout-margin control.** *Certified Access* §4.1 put the relaxation residual at
   ≤ +0.024 AUROC over a plain margin and a reviewer will ask. Defined here, before measuring, as
   **`M(request) = −Σ_k log max_a P(a | request, well k)`** — the address book's own confidence margin,
   ⛔ with no atom content, no kernel, no bandwidth and no energy in it. Reported beside `S` on every row.
   ⭐ **If `S` does not beat `M`, the physics is not doing the work and the report says exactly that.**
3. ⭐ **`accept_A7` printed beside `accept_B` on EVERY row.** Three of five rivals falsely refuse ~60 %
   of the valid A7 control and the published table hides it.

## 6. ⛔ COMPANIONS REJECTED IN ADVANCE — computed, logged, **never adopted**

| companion | why it is declared dead now |
|---|---|
| ⛔ **`d(x_read, nearest atom) / atom spread`** (the §C.4 statistic) | The addendum rejects it in advance and so do I. Declaring it now is what stops it becoming a second bite at the same data. |
| ⛔ `S` evaluated after settling (any step budget) | The criterion is the energy **at the read point**. A settled variant is a second object. |
| ⛔ `E_total`, `S_max`, `S_sum`, `S_mean` at the linear read | `S_max` is the **dead** criterion's family. |
| ⛔ any per-kind, swept, or re-fitted threshold | §2. |

⭐ **Binding, and this is the whole point of filing it:** if this hub changes the criterion, the read, the
threshold rule, the quantile or the bar **after** seeing the go/no-go or the scored row, the
zero-invalid-supervision claim is **VOID** and the report must say so in its first paragraph.

## 7. THE ONE SCORED RUN — budget, and what it is

- ⛔ **AT MOST ONE scored run this iteration, SEED 0 ONLY** (COMMON §1c). Reported as
  **"scored 1/3 — seed 0 only, 3-seed pending full caches"**. ⛔ Seeds 1 and 2 are not attempted:
  their caches are `scope="used"` and covering the cell needs ~23,480 new rows encoded, which is a
  re-encode and is forbidden.
- **Object:** `S` on sets `val` (threshold only), `B` (accept, held-out valid), `A7` (accept, valid
  control), `C` (refuse, both kinds), each under arms `linear` and `linear_store_removed`, with `M`
  computed alongside. Rung **A** (the matched 524,288 B rung) is the headline; rung **B** reported beside.
- **Gate:** it runs only if §4 clears **and** §5's control 1 collapses on the analogue.
- ⛔ **A shortfall > 0 anywhere means STOP and report the shortfall. It never means encode.**

## 8. FALSIFIERS — what kills this claim, written down before it can be argued away

| # | fires when | consequence |
|---|---|---|
| **F1** | `store_removed` does not collapse `S` to ~0.5 | ⛔⛔ **RESULT VOID.** The landscape is decorative; it is the address book again. Report as such, do not spend. |
| **F2** | `min(AUROC_oor, AUROC_dup) < 0.60` on the analogue | ⛔ **DO NOT SPEND.** Stop the line for this iteration. |
| **F3** | `accept_A7 < accept_B − 0.05` | ⛔ The refusal is bought by refusing a valid control — not a win. |
| **F4** | `S` does not beat the physics-free margin `M` by more than +0.024 AUROC | ⚠ Not a void, but the claim becomes *"a margin, computed with physics"* and the report leads with that. |
| **F5** | `S` clears one kind and is at chance on the other | ⛔ Not the claim. The claim is **ONE criterion, BOTH kinds**. |
| **F6** | the pair `(reject_oor, reject_dup)` fails to beat `knn_proto`'s `(0.8426, 0.3519)` | ⛔ Not a refusal win. Report the shortfall. |

## 9. WHAT WOULD MAKE THIS REGISTRATION WORTHLESS

⛔ Moving `q`, the read, the statistic or the bar after seeing a number · ⛔ a per-kind threshold ·
⛔ adopting a §6 companion · ⛔ quoting `gen-arms-a7-wiring/refusal_scores.json` (the superseded
broken-split file — my scorer asserts the **directory**, not the basename) · ⛔ quoting the analogue as
the benchmark · ⛔ re-encoding any row to reach seeds 1–2.

---

## AMENDMENT 1 — 2026-09-17T11:4xZ, ⛔ BEFORE the go/no-go was executed (no number exists yet)

The launch addendum names the gate as `min(AUROC_oor, AUROC_dup) < 0.6`. It does not say which
oor comparison. ⭐ **Pinning it now, so it cannot be picked afterwards:**

- **`AUROC_dup`** := `AUROC(S on syn_dup vs S on syn_a7)` — the **A7-CONTROLLED** one. ⛔ Not
  `dup vs syn_valid`, which a term counter would pass for free.
- **`AUROC_oor`** := `AUROC(S on syn_valid vs S on POOLED syn_oor_1..3)` — **pooled over all three
  severities**, ⛔ not the best severity. Per-severity values are reported beside it, never in place of it.
- **`accept`** on the analogue is `1 − refused_fraction` at the single `val` threshold, reported for
  `syn_valid` and `syn_a7` on every row.

Two structural self-checks are computed on every arm and are failures of MY code, not results, if they
move: `AUROC(val vs syn_valid) ≈ 0.5` (two valid sets) and `mean d(argmax read, nearest atom) = 0`
(the dead read is on an atom by construction).
