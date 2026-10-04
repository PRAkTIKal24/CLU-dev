# DIRECTIVE — refusal ruling + the re-posed ADMISSION hub, to the coordinator, from the Head and the Advisor (joint), 2026-09-17 (rev. 3: derived gate corrected, §1b property 1)

**Scope of this file.** It supersedes **§7.3 only** of the 2026-09-17 11:34 iteration-2 directive, which is
archived byte-identical at `_hub_scratch/advisor/DIRECTIVE-2026-09-17-1134-iter2-launch.md`
(sha256 `d2150885…`). **§1–§6 and §7.1, §7.2, §7.4 of that directive remain in force verbatim.**
Three hubs (`generation-table-2`, `composition-2`, `deletion-2`) are mid-run under it. ⛔ Nothing in this
file touches them. **Rev. 2 adds one new execution hub, `hub/admission` (§7.5), on the four threads the
refusal hub released.** Everything in rev. 1 about closing the *grammar-refusal* target stands.

**Reading guide.** §1 is two rulings: what is retired (unchanged from rev. 1) and what is re-posed (new).
§7.5 is the new hub's scope and is the part the coordinator turns into `LAUNCH-hub-admission.md`.

---

## 1. THE RULING

### 1a. Retired — grammar refusal under §C.2 (unchanged from rev. 1)

**The refusal target as chartered (`tasks/hub-refusal.md`: ONE landscape-derived criterion refusing
`out_of_range` and `duplicate_factor`, threshold on `val`, store load-bearing under the §C.2 atom shuffle) is
RETIRED. No PREREG-3 on it, no slot spent on it, on any object.** Reason, so you can tell when it stops
applying: a symbolic request enters the CLU only through the address book; the §C.2 control keeps the book
and shuffles atoms, so the posterior is unchanged (`refusal_lib.shuffled_store` docstring; PREREG-2 §5.1);
every read point is a function of book-derived quantities and fixed atoms; validity of a *symbolic* request
is a property of its grammar, which the book fully determines. Atoms can transform interface information but
cannot add validity information. Any `f(read)` is some `g(P, log_w)`, bounded above by the best interface
margin, and the §C.2 control is either not collapsed (measured: argmax `store_removed` 0.7809 ≥ 0.7511;
linear 32/32 and 64/64 shuffles win, `p = 1.0`, both rungs) or vacuous (book refit: everything collapses).
Falsifier 4 is unsatisfiable by construction for symbolic refusal. Two hubs measured it from opposite ends.
`duplicate_factor` in particular is a grammar error, caught at the interface, and is never again posed as a
landscape claim.

### 1b. Re-posed — ADMISSION: a certified, localised, live-set-exact gate inside the read (new)

**The Head's decision (this session): refusal is re-posed as admission on the paper's object, and a new
hub iterates on it.** The claim is not "the CLU detects invalid queries better than kNN". By §1a's ceiling
and by *Certified Access*'s own finding in its applications section (*"the residual is a valid gate feature
but adds at most +0.024 AUROC over a standard readout margin"*; negative findings: *"the raw energy signal
adds negligible predictive value over a standard readout margin"*), that claim is not available on any
query interface. The claim is four properties a distance gate bolted onto a rival does not have:

1. **Derived, not fitted.** The gate per well is the **zero level set of the kernel free energy**:
   `refuse well k iff E_k(x_k) > 0`, `E_k(x) = −σ_k² · log Σ_i exp(−‖x − a_ki‖² / 2σ_k²)`. Zero is not a
   choice: every stored atom has `E_k(a_ki) ≤ 0` because its own term contributes 1 to the sum, so the
   sub-level set at 0 is exactly "within reach of a stored atom" in the well's own units. The **certified
   sufficient condition** is the bound `d(x_k, nearest atom) ≥ σ_k · √(2 log N)`, beyond which `E_k > 0`
   whatever the atom configuration; it is reported as the certified fraction of refusals, ⛔ never as a
   second criterion. Inputs: `σ_k` from `atom_bandwidths` at the shipped rule, and `N`. ⛔ No quantile on
   `val`, no invalid example, no free parameter. Zero-supervision *including zero threshold fitting*.
   ⚠ **Correction (rev. 3, from the coordinator's trace):** rev. 2 derived this from "the pinned coercivity
   certificate `λ ≈ 0.0485`". That `λ` is `coupling_strength_used`, a dimensionless inter-well coupling
   budget (`gen_landscape.py:616`, `C ⪰ 0`), not a distance; and the kernel energy is coercive everywhere
   (`E_k ≥ min_i ‖x_k − a_ki‖²/2 − σ_k² log N`), so there is no coercive edge to be outside of. The
   *Certified Access* screen does not map onto this landscape. Struck.
2. **Localised.** The gate is per well; the report names the requested **term** the store cannot honour
   (via the per-term residual already in `refusal_lib.per_term_residuals`), because wells are
   correlation-partitioned channel groups, ⛔ not factors.
3. **Live-set exact.** Remove a value's rows and rebuild canonically: queries naming it are refused; write
   them back: accepted. The boundary is a byte-exact function of the live set. (Couples to `hub/deletion-2`;
   the exactness leg is phase 3, next iteration, once deletion-2's rebuild is inherited fact.)
4. **One object, two outputs.** The same launch, certificate and read yield a sample or a refusal. That is
   §C.3's *"same object delivers refusal"* leg, honestly stated.

**The query interface this defines (the Head's taxonomy, made mechanical).** A query is one instruction
**per factor**, from three sources: **symbol** (a value, grounded by the factored affine head of
`base/iter2` — ⛔ not the naive-Bayes book, which cannot leave the simplex), **exemplar** (a reference input
through the frozen encoder, contributing only the channel groups that carry that factor, per a *measured*
factor-to-well map), or **free** (launched from the well prior; dreaming). Sources mix across factors; that
mixing *is* composition. Before decoding, every addressed well is certified against its derived radius; any
uncertified well refuses the query and the per-term residual names the term. Seen generation, unseen-value
interpolation, compositional generation, out-of-range refusal, and out-of-vocabulary refusal (a value never
written, or deleted) all fall out of this one procedure.

**Controls that decide it — and the one that does not.** The §C.2 atom shuffle is ⛔ **not** the control
for admission (it tests decoding, and §1a shows it cannot certify an interface-facing claim). The controls
are: (a) a whole-latent kNN distance gate at the shipped `refusal_valid_quantile = 0.95` on the same
encoder; (b) the **fair** rival, a per-well-factored kNN gate at the same quantile — if it ties on the pair,
the claim is properties 1–4, not detection, and the report says so; (c) a single-well (`K = 1`) CLU on the
same atoms, which must lose localisation; (d) the live-set control, hold out → refused, write back →
accepted; (e) the physics-free head margin `M`, reported beside every number; (f) `accept_A7` beside
`accept_B`.

**Head amendment 2 to charter §B.3 (decided this session, to be countersigned upstream, §6.1):** a feature
tensor may enter launch **only in the declared exemplar query mode**, as a separately labelled arm.
`assert_no_bypass` must still pass on every symbol-mode read, and every generation number the paper reports
remains symbol-mode. Exemplar mode is a query interface, never a generation shortcut.

**The ruling stops applying** if §5.7–§5.10 fire: the derived radius does not calibrate, the factored kNN
ties on the pair *and* localisation does not beat the single-well control, or the encoder does not separate
held-out-value images. Each is a measurable stop, and the hub's first four hours are designed to hit them
early if they are going to.

---

## 2. WHAT THE COORDINATOR DOES NEXT — in order

1. **Do not touch the three running hubs.** No new addenda, no thread reallocation, no messages.
2. **Record the closure** of `hub/refusal` in your ledger: *RETIRED 2026-09-17, scored 0/3, both iterations,
   reason §1a; target re-posed as `hub/admission`, §1b.* ⛔ Do not delete or reuse `CHLU-hub-refusal*` or
   `.claude/outputs/hub-refusal*/`.
3. **Run the closure check** (≈ 2 s each, your own check or a 1-thread spoke; ⛔ no scored row): with the
   inherited `scorer/refusal_lib.py`, rung A and B, seed 0: (a) `max_abs(P_full − P_shuffled) == 0.0` over
   the 16 shuffles; (b) `matrix_rank(atoms[k]) == n_atoms` per well, real and shuffled. Write
   `_hub_scratch/reports/refusal-closure-check.md`. If (a) is not exactly zero, stop and escalate (§5.1).
4. **Compile `_hub_scratch/reports/refusal-closure.md`**: both hubs' headline numbers with controls; the §1a
   argument; `RIVAL_ROWS.json` with `accept_A7` beside `accept_B`; the two-record disclosure verbatim from
   `PREREG-hub-refusal-2.md` §0.1; the closure-check numbers. It becomes inherited fact for `hub/admission`.
5. **Adopt now, programme-wide:** every refusal/admission row, arm or rival, prints `accept_A7` beside
   `accept_B` at the same threshold. Into `LAUNCH-COMMON.md` and §6.
6. **Stage `hub/admission`** (COORDINATOR-LOOP step 2), from §7.5:
   a. `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-admission -b hub/admission base/iter2`.
      ⛔ Not from `8e36bc9`, ⛔ not from a peer's branch, ⛔ not from `hub/refusal-2`.
   b. Write `LAUNCH-hub-admission.md` from §7.5. It must carry, verbatim, the gate definition from §1b
      property 1 (the zero level set, the certified bound, the two inputs) and the sentence *"every stored
      atom has `E_k ≤ 0`; the PREREG verifies this identity numerically on the live store before any gate
      number"*. ⛔ The hub does not launch with any other radius — a radius the hub invents is a fitted
      radius. The *Certified Access* margin-comparison protocol (applications section) is quoted for control
      (e); ⚠ PDF rendering is unavailable on this box (`pdftoppm` missing) — quote from your own trace.
   c. Threads: **4** (the released ones). Sub-agents ≤ 2, `model: "opus"`. New `START_UTC.txt`.
   d. Pre-flight per `PREFLIGHT-12H.md` for a fourth concurrent process (page file already raised).
7. **The human launches** `hub/admission` on `opus[1m]`. You monitor only, watching §5.7–§5.10 in the first
   hours.
8. **The bandwidth defect and the two-record disclosure** are handled exactly as in rev. 1: upstream knob
   (§6.5), procedural fix (§6.3). Not work items.
9. **At 12 h**, collect all four reports; write the decision brief; **the Head and Advisor reconvene on
   `hub/admission`'s report** (the Head's instruction: reconvene when it is ready).

---

## 3. WHAT IS STILL FORBIDDEN — explicitly

- ⛔ **Any grammar-refusal PREREG** — a criterion whose claim is "refuses `out_of_range`/`duplicate_factor`
  better than rivals" under the §C.2 atom shuffle, on any object. §1a. This includes launch-time `S_max`,
  `E_store(x_read)`, the `τ`/`ρ` families, and the §C.4 distance *as a detection claim*.
  ✅ The admission hub's per-well certificate is the §C.4 quantity used *as a derived gate with the live-set
  control under §1b's claims*; that is permitted **only** inside `hub/admission` under its PREREG.
- ⛔ **Moving the gate off zero**, scaling `σ_k`, or replacing the level set with a quantile on `val`, on B,
  on any invalid or held-out row, or by sweeping until accept/refuse look good. The gate is `E_k > 0`,
  written in the PREREG before any number, and never moved. A hub that finds it does not calibrate
  **reports that** (§5.7); it does not fit a threshold. A fitted alternative may be reported beside it,
  labelled, never adopted.
- ⛔ **Using the §C.2 atom shuffle as evidence for or against an admission claim.**
- ⛔ **Exemplar mode on any generation number**, or any read where `assert_no_bypass` is expected to pass.
  Exemplar mode is its own labelled arm.
- ⛔ **Synthesising out-of-vocabulary images** (darkened "black" cubes, "pyramids"). Out-of-vocabulary is
  realised by **holding out a real value** (§7.5), so exemplar refusal is measured on real images.
- ⛔ Re-encoding any row. Held-out-value stores are rebuilt from the cached seed-0 features (full scope);
  a shortfall > 0 means stop and report.
- ⛔ Reading a concurrent peer's outputs (`hub-composition-2`, `hub-generation-table-2`, `hub-deletion-2`).
  ✅ `hub-refusal`, `hub-refusal-2`, and the iteration-1 hubs are inherited fact.
- ⛔ Editing any manifest-pinned file; changing any bandwidth in any pinned pipeline; deleting anything.
- ⛔ Quoting analogue numbers as benchmark numbers; quoting `gen-arms-a7-wiring/refusal_scores.json`.
- ⛔ Reporting any refusal/admission row without `accept_A7` beside `accept_B` and without the fair
  factored-kNN control beside it.
- ⛔ Writing "the CLU refuses what transformers decode as garbage" without the kNN-gated baseline in the same
  figure. That sentence is a figure, not a claim.

---

## 4. SCORED-RUN BUDGET

| target | spent | remaining | this iteration |
|---|---|---|---|
| **admission** (the refusal target, re-posed; inherits its 3) | 0/3 | 3 | **≤ 1, seed 0**, only after PREREG + DEV clears (§7.5) |
| generation-table | 0/3 | 3 | ≤ 1, seed 0 (archived §4, unchanged) |
| composition | 0/3 | 3 | 0 if same object as 7.1, else ≤ 1 (unchanged) |
| deletion | 0/3 | 3 | ≤ 1, seed 0 (unchanged) |

The admission scored run is the **B-accept / held-out-value-refuse pair** on held-out B rows at the derived
radius, with all six controls. Everything else in §7.5 is DEV/val and costs nothing.

---

## 5. WHAT WOULD FALSIFY THIS RULING — stop and come back on any of these

1. **Closure check (a) not exactly zero** — the §C.2 control changes the interface; §1a's premise is wrong.
2. **A counter-example to §1a:** a pre-registered symbolic criterion with full-store
   `min(AUROC_oor, AUROC_dup) ≥ 0.75` and book-kept `store_removed ≤ 0.55` on both kinds, both rungs.
3. (rev. 1 §5.3, now enacted by amendment 2) — exemplar mode is chartered; no longer a falsifier.
4. **Both §C.3 legs fail** — deletion-2 misses byte-exact provenance **and** composition-2 reports dreaming
   DEAD **and** admission's §5.7–§5.9 fire. Then the tie-is-win narrative has no leg. Head decision.
5. **Deletion-2's `train_store` is not deterministic on this box** (archived §5.5) — then the live-set
   exactness leg of admission is unreachable here too; phase 3 moves to the Mac.
6. **Shortfall > 0 on a held-out-value store rebuild** — the split is not what we think; stop, report rows.
7. **The derived radius does not calibrate:** at the derived radius, `accept_B < 0.50` while the
   whole-latent kNN gate at `q = 0.95` gives `accept_B ≥ 0.90` on the same rows. The "no tuning" property is
   dead; the hub reports the derived number and the fitted number side by side and **does not** fit a radius.
8. **No design advantage:** the per-well-factored kNN gate ties the pair under R1 **and** term localisation
   accuracy does not beat the single-well control beyond seed spread. Then admission is kNN with extra
   steps; the line stops and the report says so.
9. **The encoder does not separate:** exemplar launch points from held-out-value images fall inside the
   radius at a rate within 0.05 of valid images' rate. Exemplar refusal is dead; symbol and free modes
   continue; the paper never claims "launched far away by the encoder".
10. **The affine head's out-of-range extrapolation does not cross the level set** — measured in the first
    hour (§7.5 step 2). If `E_k(x_launch)` and `d/σ_k` are flat in severity, symbolic out-of-range refusal
    is dead on this head too, and the hub reports it before building anything else. The companion check,
    same hour: where valid `val` launches sit — if `E_k(x_launch) > 0` for most valid requests, the gate
    refuses everything and §5.7 has fired before B is touched.

---

## 6. WHAT GOES UPSTREAM — to the Mac, not on any hub's path

1. **Charter:** §B.3 amendment 2 (exemplar query mode, §1b) for the Head's countersign; §C.3's
   *"refusal with accept-unseen-valid"* leg re-worded to *"admission: certified, localised, live-set-exact,
   with accept-unseen-valid"*; the §1a argument as a footnote so grammar refusal is not re-attempted.
2. **`tasks/hub-refusal.md`:** RETIRED, pointer to `refusal-closure.md`; new `tasks/hub-admission.md` from
   §7.5 once this iteration's report is in; the C-set schema (`n = 72`; 36/36) written into both so no hub
   opens `c_invalid.json` to learn it.
3. **Reporting rule** into charter §A/§C and the paper's rival table: `accept_A7` beside `accept_B`; the
   fair factored-kNN gate beside every admission row.
4. **Tooling:** install `poppler` on FORGIS_14 so PDFs render for hubs and the coordinator.
5. **Bandwidth knob, no default change** (rev. 1 §6.4, unchanged): `address_bandwidth_frac = 1.0` in
   `GenDynamicalReadConfig`, docstring warning at `gen_landscape.py:733` citing
   `QUANTISATION_BOTTLENECK_seed0_{A,B}.json`.
6. **Full-scope caches for seeds 1–2** — in flight, unchanged.
7. **Paper text:** the refusal negative paragraph from `refusal-closure.md`; the admission section drafted
   from `hub/admission`'s report after we reconvene.

---

## 7. PER-HUB SCOPES

**7.1 `hub/generation-table-2`, 7.2 `hub/composition-2`, 7.4 `hub/deletion-2`:** unchanged; see the
archived directive. Mid-run; ⛔ not to be contacted.

**7.3 `hub/refusal` — CLOSED** (rev. 1). No relaunch; worktrees and outputs are inherited fact, read-only.
Closure spoke as in rev. 1 §2.3–§2.4.

### 7.5 `hub/admission` — NEW. Slug `hub-admission`, branch `hub/admission`, worktree `CHLU-hub-admission`, 4 threads, 12 h, seed 0

**Object it starts from.** The reference object on `base/iter2`, unchanged: K = 32 correlation-partitioned
wells, M = 26 modes per well, the factored affine head (`gen_factored_read.fit_factored_head`), rank-16
pair head, 469,504 B. ⭐ The same object as 7.1/7.2 — §C.3 coherence is the point. The landscape energy
and per-term residual machinery is called from `gen_landscape` / the inherited `scorer/refusal_lib.py`,
⛔ never re-implemented. Settle dynamics are optional and off by default (Head: nice-to-have); the gate is
evaluated **at the launch point**.

**Inherits.** `.claude/outputs/hub-refusal/**` and `hub-refusal-2/**` (both reports, `RIVAL_ROWS.json`,
`refusal_lib.py`, `term_experts`, `per_term_residuals`, the addressability/quantisation artefacts),
`_hub_scratch/reports/refusal-closure.md`, and the iteration-1 generation-table/composition reports.
⛔ Not the three concurrent hubs.

**What it builds, in order — each step logged as a dated bullet before the next starts.**
1. **Ground truth about the wells (≤ 1 h).** Confirm `feat_shape` and what the 2048 latent dimensions are
   (feature maps × spatial, or pooled); confirm `well_channels` for K = 32. Measure the **factor-to-well
   map**: per well, per factor, mutual information over shuffled-label floors (the refusal hub's method).
   Artefact `FACTOR_WELL_MAP_seed0.json`. ⭐ This is what makes exemplar mode and term localisation
   definable; without it the hub proceeds in symbol and free modes only.
2. **Does the affine head leave the radius? (≤ 1 h; §5.10).** For symbol-mode `out_of_range` at severities
   `w ∈ {2, 3, 4, 7, 13}` via the R8 rule on the one-hot design, measure `d(x_launch, nearest atom)/σ_k`
   per well and the per-term residual. It must **rise with severity**. If flat, report and stop the
   symbolic out-of-range leg.
3. **The derived gate — PREREG-1 before any accept/refuse number.** Register, verbatim from §1b
   property 1: the gate `E_k(x_k) > 0` per well at the launch point, with `E_k` the shipped
   `gen_landscape.store_energy` per well (called, not re-implemented) at `σ_k` from `atom_bandwidths` at
   the shipped rule; the certified bound `d ≥ σ_k √(2 log N)` reported as the certified fraction; the
   identity check `max_i E_k(a_ki) ≤ 0` on the live store, computed and logged as a self-check (a violation
   is a bug in the hub's code, not a result). State the term-localisation rule (the requested term with the
   largest residual among uncertified wells). ⛔ No number computed under the gate before the PREREG is
   filed. ⛔ No `λ`, no coercive component, no radius other than the bound. **Mandatory disclosure, verbatim:** *"This target was posed after two refusal hubs
   analysed invalid examples on a fit-derived synthetic analogue, and after two benchmark invalid records
   were displayed to a predecessor. The admission radius is derived from the store's geometry and the pinned
   coercivity certificate with no free parameter, so the zero-supervision claim is 'derived, not blind', and
   the paper says so."*
4. **Query sets (all from seed 0's cached features; ⛔ no re-encode).**
   - **seen:** `val` tuples, symbol mode.
   - **compositional (B):** the held-out valid compositions, symbol mode — ⛔ held out; touched only in the
     scored run.
   - **unseen value:** symbol mode with one continuous factor (scale, orientation) launched at an
     interpolated point between adjacent grid values, on `val` corners.
   - **out-of-range:** symbol mode, R8 severities as in step 2, on `val` corners, one factor at a time.
   - **out-of-vocabulary, symbol:** a request naming a value the store never wrote (see next item).
   - **out-of-vocabulary, exemplar (the "black cube" set, on real images):** rebuild the store from
     `store_train ∖ rows(v)` for one value `v` of each of the six factors (six stores), from cached features.
     Queries: real images of `v` through the frozen encoder, contributing only the channels the
     factor-to-well map assigns to that factor; the other factors by symbol. Also **write-back:** the full
     store, same images — must be accepted. This is the live-set control (d).
   - **A7:** the redundant-term valid control, printed beside B on every row.
   - **duplicate_factor:** interface grammar check only; reported as such, ⛔ no landscape number.
5. **DEV numbers (val corners, non-held-out) with all six controls,** then, if the bar is met on DEV,
   **PREREG-2 for the one scored run**: `accept_B` and `accept_A7` at the derived radius on held-out B;
   refuse rates on the out-of-range and out-of-vocabulary sets; the two kNN gates at `q = 0.95` on the same
   rows; `M`; the single-well control; term-localisation accuracy.
6. **The certified-access figure (DEV, unscored):** walk a launch point outward from a `val` atom along a
   ray; plot external-classifier accuracy of the decoded output vs `d/σ_k` for the CLU read and for a plain
   decoder of the same latent, with the derived radius and the kNN quantile drawn. Report where the
   certificate flips relative to where decode quality collapses. One figure; no claim beyond what it shows.

**Bar (set now; PREREG carries it; never moved).**
- **Pair at the derived radius:** win-or-tie under R1 vs the **factored** kNN gate on
  `(refuse_out_of_support, accept_B)`, where out-of-support pools out-of-range and out-of-vocabulary;
  `accept_A7 ≥ accept_B − 0.05`.
- **Localisation:** term-localisation accuracy on single-factor perturbations `≥ 0.50` (chance 1/6), and
  beats the single-well control beyond seed spread.
- **Live-set:** refuse rate on held-out-value exemplar queries ≥ 0.90 **and** accept rate on the same images
  after write-back ≥ 0.90. (Byte-exact provenance is phase 3, once deletion-2's canonical rebuild is
  inherited.)
- **No tuning:** the radius reported is the derived one. A fitted alternative may be *reported beside it*,
  labelled, never adopted.

**Mandatory controls, on every row:** (a) whole-latent kNN gate, `q = 0.95`; (b) per-well-factored kNN gate,
`q = 0.95`; (c) single-well CLU on the same atoms; (d) live-set hold-out/write-back; (e) physics-free head
margin `M`; (f) `accept_A7` beside `accept_B`. ⛔ Not the §C.2 shuffle.

**PREREG obligation.** PREREG-1 (radius derivation, localisation rule, query-set definitions, disclosure)
before any gate number; PREREG-2 (the scored run: rows, bar, controls, expected direction) before the scored
run. Amendments appended and dated. Bars never moved after a number.

**Scored budget:** ≤ 1, seed 0, reported as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.

**Report:** the pair with its controls; the localisation table; the live-set table; the figure; the
factor-to-well map; §5.7–§5.10 answered explicitly with numbers; a cost line; elapsed true wall clock;
solved-truthfully or not-solved. ⭐ "The derived radius does not calibrate and here is by how much" is a
complete and valuable answer.

---

*Everything not named here is governed by the archived 11:34 directive, `LAUNCH-COMMON.md`, and the charter.
Stopping at 12 h with "not solved, here is exactly how far I got" remains a success. Manufacturing a result
remains the one unrecoverable failure — and, for admission specifically, so does a radius that was fitted
and reported as derived.*
