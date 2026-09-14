# c2w8p2-instruments-and-debt — experiment-engineer report (wt3, riders 2 + 3a)

> ## ⛔ DATED CURATOR AMENDMENT — 2026-08-10 (`doc-curator-c2w8-pass3-close-fold`, **[C2W8-CLOSE]**) — **THE INVERSION QUALIFIER (filed for completeness) + THE `M` QUALIFIER CONFIRMED TRAVELLING**
> **⛔ The body of this report is NOT edited and nothing in it is deleted** (C-3 dated-banner precedent).
>
> ⚠⚠ **SWEEP CORRECTION, REPORTED NOT RESOLVED.** The Hub's worklist lists this file as *"strong φ ≥ 3 uses, inversion qualifier absent (4 hits)"*. **Re-swept per-file with a live positive control** (`DIAL DECLARATION` = 1): this file contains **ZERO** uses of *"strong φ"/"strong phi"* as a **concept**; its **4** hits are all the *filename* **`tests/test_cifar_strong_phi.py`** (lines 10, 181, 215, 259). ⇒ **the banner is filed for completeness, not because this report banks the claim.** ⛔ **The curator does not adjudicate the Hub's count; the discrepancy is on the record.**
>
> **The qualifier itself (charter Add.11 §A31.4, `c2w8p3-capture-strong-phi` §3.1):** on the census store at `d = 12`, the task-strong `simclr` encoder is the **address-WORST** arm beyond 2 SE (`simclr − randconv` **A1 = −0.1406 ± 0.0508, 0/3**; `simclr − pca` **−0.1276 ± 0.0589, 0/3**) while **unfitted `randconv` buys the geometry free (0 fit steps)**. ⛔⛔ **"strong φ" may never again be used as one undifferentiated notion.**
>
> ✅ **AND THE `M` VACUITY QUALIFIER IS CONFIRMED TRAVELLING — this file is its primary carrier and it is intact:** reconciliation item **3** and §7's closing item both state that pass-1 `M` = **0.2333 / 0.2417 / 0.2417** is **`vacuous_gate`-tripped** and *"must never be quoted as a merge population without the vacuity qualifier"*, and that **any census quoting `M` must state WHICH criterion produced it** (`mergeable_pairs` = pass 1's rule, **UNCHANGED**; **K9 is registered with two designed refusal negatives and the merge VERB stays UNBUILT** — `test_no_merge_verb_exists` asserts its absence). ⛔ **Unchanged at pass 3: no lifecycle verb was built, and the verbs are now also MOOT at this substrate (§A32.1).**
> **Registry:** `negative_results.md` **N277** (inversion) and **N250 / N263** (the `M` lineage); ledger ⟲ **C2W8 PASS-3 + CLOSE** addendum. **Sources:** `c2w8p3-capture-strong-phi` §3.1, charter **Add.11 §A31.4 / Add.11 §A32.1**, the **`[C2W8-CLOSE]`** §10 entry.

Task + acceptance criterion: **re-register the merge criterion (K9) so it can provably REFUSE on both
legs, and close the `cl_baselines.ConvNet` x64 coverage gap** — criterion registered in
`ERRATA-C2W8-PASS2.md` with two designed negatives pytest-asserted, no merge verb built; dtype fixed
with the CNN path covered under x64 and the x64-off path bit-identical.
Status: **done** (item 1 = declared NOT-RUN by Head ruling; items 2 + 3a delivered).

## ⭐ RECONCILIATION LIST (protocol §5 — first 10 lines; each needs an owner at review)
1. **`tests/test_cifar_strong_phi.py:66-72`** carries a 6-line comment saying the `"cnn"` backbone
   cannot be exercised because of the x64 dtype bug. **That comment is now STALE** (bug fixed here).
   The test's `backbone = "mlp"` choice can stay (cost), but the *stated reason* must change. ⛔ I did
   not edit it — it is the pass-1 CIFAR spoke's file. **Owner needed.**
2. **§7 (Known Issues)** gains one RESOLVED entry + one standing rule (dtype-strict primitives must
   promote to the **parameter** dtype, never assume the ambient one). Draft below.
3. **`M` is now criterion-ambiguous.** `well_lifecycle.mergeable_pairs` (pass 1's rule) is UNCHANGED
   and still what `census.json`'s `M` reports. Pass-1 `M` = **0.2333 / 0.2417 / 0.2417** must never be
   quoted as a merge population without the vacuity qualifier. Any future census that quotes `M` must
   state **which** criterion produced it. **Owner: the Hub / whoever re-runs the census.**
4. ⚠ **SECTION-NUMBER COLLISION in the shared `ERRATA-C2W8-PASS2.md`:** my K9 block (filed
   **2026-08-06**, line 70) and wt1 `c2w8p2-compact-atoms`'s block (filed **2026-08-07**, line 155)
   are **both numbered `## §2`**. Append-only discipline was honoured on both sides — nothing was
   overwritten — but the file now has two §2s. ⛔ I did not renumber wt1's. **Suggested resolution:
   renumber the LATER block (wt1's) to §3**, because the earlier one is cited by hash-stable
   references inside *tracked code* (`soft_certificate.py` docstring, `tests/test_merge_criterion.py`
   docstring, commit `c13f953`'s message). **Owner: the Hub at integration.**
5. **Item 1 (φ-dim co-scaling) is a DECLARED NOT-RUN, never a null** — one-paragraph discharge in §1
   below, with its scratch-probe provenance attached.

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** **none — instrument, registration and test-coverage debt.** ⛔ No claim cell, no
  performance number, no verdict.
- **Laundering control:** N/A (no performance claim; this spoke runs no cell).
- **Falsifies:** item 2's criterion cannot be made to refuse on either leg. **It was not falsified** —
  both legs refuse designed pairs (§2.3).
- ⛔ Item 1 runs no cell (Head ruling) and cannot falsify anything this wave.
- ⛔ N94 unchanged; no promotable reading produced.

---

## 1. ITEM 1 — the φ-dim / co-scaling finding: **DECLARED NOT-RUN, reported not re-derived**

**Banked finding, restated in the Head-ruled form and nothing more:** *the `d ≥ 16` inertness was
measured at a **fully co-scaled** atom budget — `ERRATA-C2W8.md` §3's `n_atoms` column matches
`round(512·√2^d)` exactly at 4/8/12/16 (2 048 / 8 192 / 32 768 / **131 072**) — so the store was **NOT
capacity-starved**; the binding constraint is **REACH**.* Mechanism (ERRATA §3's second table): the
nearest of *all* atoms recedes as `√dim` (0.294 at d=4 → **1.483** at d=16) while `atom_width` stays
0.3, so the write gradient's `exp(−r²/2s²)` underflows **6.18e-01 → 4.98e-06**. ⚠ **Provenance that
travels with the conclusion wherever it is quoted:** this arithmetic came from a **scratch probe of 3
designed-site writes** (`.claude/scratch/c2w8/timing16.py`), **not** from a censused CL-stream cell.
Per the Head ruling (`ERRATA-C2W8-PASS2.md` §1 Q1) the confirmation cell was **not bought**: this is a
⛔ **DECLARED NOT-RUN and must never be reported as a null**. Consequence carried as registered:
*whether "full CLU on a CL stream" has a non-empty operating window — and therefore whether **C2W10's
persistent store has a home** — is now answered by the **reach** route (learned p₀ / wormholes, C2W9
territory), not by buying atoms.* ⛔ I ran no `addr_dim = 16` cell, built nothing, and re-derived
nothing for this item.

---

## 2. ITEM 2 — the merge criterion, RE-REGISTERED (K9)

### 2.1 Registration (filed BEFORE the code; canonical site = `ERRATA-C2W8-PASS2.md` §2)
Appended as a dated block (append-only; §1 untouched), plus a copy at
`.claude/outputs/c2w8p2-instruments-and-debt/PREREG.md`. **Both predate commit `c13f953`**, which is
the first line of the predicate.

**What is retired, in numbers** (re-derived by me from the frozen `census.json`, matching the Hub):

| seed | key spacing `median_nn_task1` | pass-1 `R_cert` | ratio | pass-1 admitted | `payload_dist` values |
|---|---|---|---|---|---|
| 0 | 0.140714 | 1.540231 | **10.95×** | 28 / 120 | **{0.0}** |
| 1 | 0.137540 | 1.417728 | **10.31×** | 29 / 120 | **{0.0}** |
| 2 | 0.146832 | 1.648195 | **11.23×** | 29 / 120 | **{0.0}** |

**The re-registered criterion `M′`** (`chlu.core.soft_certificate.merge_admissible`), both legs able
to refuse **and both able to be INAPPLICABLE ⇒ refuse**:
- **geometry** — `center_sep ≤ rho_geom · key_spacing`, **`rho_geom = 1.0`**, `key_spacing` = the
  **measured** per-seed `median_nn_task1`. Commensurability = `rho_geom` = **1.00×** by construction
  (vs 10.31–11.23×). A missing/non-finite/non-positive spacing ⇒ **INAPPLICABLE ⇒ REFUSE** (an
  unmeasured ruler certifies nothing).
- **payload** — `payload_dist ≤ tau_payload · payload_scale`, **`tau_payload = 0.25`**, with
  `payload_scale` = median pairwise `‖a_i − a_j‖` over the **whole** live-well pair population
  (`payload_scale_from_pairs`), never the admitted subset (circular). ⭐ **Anti-vacuity clause:**
  `payload_scale ≤ 1e-9` ⇒ the channel carries no discriminative content ⇒ **INAPPLICABLE ⇒ REFUSE**.
  This is precisely pass 1's configuration (all payloads identical), and it now refuses.
- `merge_criterion_report` adds `refusal_rate` and `vacuous_gate_would_trip = refusal_rate ∈ {0,1}` —
  **monitor #3's own convention, applied in BOTH directions**.

⚠ **Assumption I made and am stating** (smallest reasonable reading of "an operating point where
`R_cert` is commensurate with the key spacing"): I did **not** re-tune `s_max`/`sigma_q` to shrink
`R_cert` itself — that would be changing the store's physics, which is **arm A's** job (compact
atoms), not a criterion re-registration, and it would silently move SC-1/SC-2 for everyone. Instead
the merge criterion **stops using `R_cert` as its geometric radius** and uses its own declared
`r_merge = rho_geom · key_spacing`. `R_cert` remains computed and reported exactly as before (SC-1's
whole content is that these are different objects). ⚠ Note for the Hub: if arm A lands, `s_max` falls
and `R_cert` will move *toward* the spacing on its own; the two radii are then close but still
distinct objects.

### 2.2 Pre-registered predictions vs measurement

| # | registered before the code | measured | verdict |
|---|---|---|---|
| **R1** | 28/29/29 pass-1-admitted pairs → **0 admitted**, refusal **1.000**, refused on **geometry** | 0 admitted on all 3 seeds; `refused_on == ["geometry"]`; minima `center_sep/spacing` = **2.0599 / 1.4942 / 1.1589** | ✅ held |
| **R2** | `vacuous_gate` still trips, at the **opposite** end (f = 1.000) | `vacuous_gate_would_trip is True` at `refusal_rate == 1.0`, 3/3 | ✅ held (registered **as expected, not a failure**) |
| **R3** | smallest admitting `rho_geom` = 2.0599 / 1.4942 / 1.1589 | reproduced to `abs=1e-3`; `rho·0.999` refuses and `rho·1.001` admits at each seed | ✅ held |

⚠ **Honest scope of R1:** this is a **re-scoring of banked pairs**, not a re-run census. `census.json`
stores only the **admitted** pairs, so the 92/91/91 pass-1-refused pairs are not available and **no
full-population `M′` is claimed**. What is proved is the load-bearing half: **every pair pass 1
admitted is refused by the re-registered criterion, on the geometry leg.**

### 2.3 ⭐ The designed proof of refusal (all pytest-asserted, `tests/test_merge_criterion.py`)

| # | pair (at `key_spacing = 0.140714`) | result | pass-1 rule on the same pair |
|---|---|---|---|
| **N-geom** | `center_sep = 0.30` (**2.13×** spacing), `payload_dist = 0.0` | **REFUSED on geometry** (`refused_on == ["geometry"]`, payload leg *passed* ⇒ the refusal is unambiguous) | **ADMITS** it |
| **N-pay** | `center_sep = 0.05`, `payload_dist = 0.9`, `payload_scale = 1.0` | **REFUSED on payload** (`payload_tol = 0.25`; geometry leg passed) | refuses |
| **N-degen** | `center_sep = 0.05`, `payload_dist = 0.0`, **`payload_scale = 0.0`** | **REFUSED — `payload_degenerate`** (the census's own configuration) | **ADMITS** it |
| **N-ruler** | any pair with `key_spacing ∈ {nan, 0, −1}` | **REFUSED — `geometry_inapplicable`** | — |
| **P-pos** | `center_sep = 0.05`, `payload_dist = 0.10` | **ADMITTED** ⇒ the criterion is not merely refusing everything | admits |

Plus a 4-pair population separating the legs (1 admitted / 1 geometry / 1 payload / 1 both ⇒
`refusal_rate = 0.75`, `vacuous_gate_would_trip is False`) — i.e. the criterion demonstrably occupies
the **non-degenerate** middle that pass 1's never did.

### 2.4 ⛔ Boundary held
**No merge verb was built.** `chlu/core/well_lifecycle.py` was **not modified** (read-only per the
task and the frozen-census rule); `mergeable_pairs` still implements pass 1's criterion verbatim, and
nothing in the harness calls the new predicate. A test (`test_no_merge_verb_exists`) asserts no
`merge_wells*` / `merge_pair` / `do_merge` symbol exists in either module, so the *next* spoke has to
delete an assertion to build one.

---

## 3. ITEM 3a — `cl_baselines.ConvNet` under `jax_enable_x64`

**Bug reproduced first** (worktree, pre-fix):
```
$ python -c "jax.config.update('jax_enable_x64', True); ConvNet((3,32,32),4,8,key,(4,4,4))(float32 image)"
weight dtype float64
ERROR: TypeError lax.conv_general_dilated requires arguments to have the same dtypes, got float32, float64.
```
**Cause:** conv weights are built at the **ambient** JAX dtype; `build_cl_stream` always supplies
**float32** images; `lax.conv_general_dilated` is dtype-**strict**. Several test modules set
`jax_enable_x64` **at module import**, so x64 is globally ON in a full-suite run ⇒ the entire
`backbone="cnn"` path was **untestable inside the suite** (§7.23 / N211 hazard class, 4th occurrence).

**Fix (1 line + comment, `ConvNet.features`):** promote the input to the **parameter** dtype —
`jnp.asarray(x, dtype=self.conv[0].weight.dtype).reshape(self.shape)`. At x64-off this is
float32→float32, i.e. a **no-op**. No parameter, shape or default changed; `MLPNet` untouched.

**Coverage now shipped (`tests/test_cl_baselines_x64.py`, 5 tests):**
1. CNN **forward under x64** on float32 images — weights `float64`, input `float32`, output finite and
   `float64`.
2. **`make_net(backbone="cnn")` + one `_train_task` step under x64** — forward + `filter_value_and_grad`
   + adam; loss finite, weights actually moved, grads finite. (The path, not just the kernel.)
3. ⭐ **x64-off bit-identity**: `features` and `__call__` compared with `np.array_equal` (exact, not
   `allclose`) against a **local reimplementation of the pre-fix code** (`_prefix_features`) over 4
   images — equal, dtype `float32` both sides.
4. x64-off **parameter count** hand-counted and unchanged.
5. promotion follows the weights (a float64 input is not downcast).

⚠ The x64 toggle is a **function-scoped fixture that restores the previous value** — N211's remedy;
it does not leak to other modules (verified by the full suite, §4).
⚠ **Verified the test is a real regression test:** under x64 the pre-fix code path
(`_prefix_features`) still raises the original `TypeError`, while the fixed path returns float64.
⛔ **No shipped result changes** — every real CIFAR run is x64-off, and the OFF path is bit-identical.

---

## 4. How I verified (commands + observed output)

All runs: **worktree `../CHLU-c2w8c`**, main venv reused (`PYTHONPATH=$PWD
/Users/user/Desktop/CHLU/.venv/bin/python`, per §4's w6 lesson — no worktree `uv sync`, so **no**
package-version drift).

| command | result |
|---|---|
| `pytest tests/test_merge_criterion.py -q` | **19 passed** (13.9 s) |
| `pytest tests/test_cl_baselines_x64.py -q --no-cov` | **5 passed** (11.0 s) |
| `pytest tests/test_soft_certificate.py tests/test_well_lifecycle.py tests/test_cl_entry.py tests/test_cifar_strong_phi.py -q --no-cov` (the neighbours of both edits) | **74 passed** (151 s) |
| `ruff check` on the 4 touched files | **All checks passed** |
| **full suite** `pytest tests -q --no-cov` | **1469 passed, 0 failed, 0 skipped** (3094.6 s = 51 min 35 s), 36 warnings (all pre-existing: vendored-TSB `DeprecationWarning`s + `exp_v1_hopfield_gate`'s empty-slice `RuntimeWarning`) |

**Count arithmetic:** **exactly additive, verified by collection at BOTH revisions** (not assumed):

```
pytest tests --collect-only -q   @ 80d7d4b (base, detached)  -> 1445 tests collected
pytest tests --collect-only -q   @ 42b781c (this branch)     -> 1469 tests collected
```
**1445 + 19 (`test_merge_criterion.py`) + 5 (`test_cl_baselines_x64.py`) = 1469**, and all 1469 pass
⇒ **zero regressions, zero pre-existing failures inherited.**
⚠ **Discrepancy worth one line for the Hub:** `PREREG-C2W8-PASS2.md` §0 records the base as
"**1443**/0 HEAD-stable" at `80d7d4b`; the base actually **collects 1445**. I measured the base
myself rather than trusting the banked figure, so the arithmetic above stands on measurement. (2
tests, no verdict rides on it — but the banked 1443 should be corrected so the next spoke's
arithmetic reconciles.)

---

## 5. Git footprint

- **Branch:** `agent/experiment-engineer/c2w8p2-instruments-and-debt` (off `main @ 80d7d4b`), worktree
  `../CHLU-c2w8c`. ⛔ Not pushed; left for review. Branch ref **verified visible from the main repo**
  (`git log main..agent/experiment-engineer/c2w8p2-instruments-and-debt` shows both commits).
- **Commits:**
  - `c13f953` `[experiment-engineer] K9: re-register the merge criterion so it can actually refuse`
    — `chlu/core/soft_certificate.py` (+157, additive only), `tests/test_merge_criterion.py` (new).
  - `42b781c` `[experiment-engineer] close the x64 coverage gap on cl_baselines.ConvNet`
    — `chlu/experiments/cl_baselines.py` (1 logical line + comment),
      `tests/test_cl_baselines_x64.py` (new).
- **Files touched: 4**, exactly the ones the task declared. ⛔ **Not touched:** `well_lifecycle.py`,
  `usage_telemetry.py`, `memory_potentials.py`, `emission_head.py`, `clu_system.py`, `config.py`,
  `friction_field.py`, `exp_cl_entry.py`, `phi_encoders.py`, C2W6/C2W7 files,
  `tests/test_cifar_strong_phi.py`.
- **Rebase onto local `main`:** no-op (base unmoved at `80d7d4b`); `origin/main` deliberately not used
  (§7.21). No conflicts. Working tree clean.
- `.claude/` writes: appended **§2** to `.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS2.md`
  (append-only, §1 untouched) + `.claude/outputs/c2w8p2-instruments-and-debt/PREREG.md` + this report.
- **Worktree removed after verification** (protocol §3.2, the wave-4 lost-commits precedent):
  `git log main..agent/experiment-engineer/c2w8p2-instruments-and-debt` **from the MAIN repo** showed
  both `c13f953` and `42b781c` **before** `git worktree remove ../CHLU-c2w8c`, and shows them after.
  Worktree count is back to 3 (`CHLU`, `CHLU-c2w8a`, `CHLU-c2w8b`) ⇒ the ≤3 cap the Head restores at
  the end of this pass is already satisfied from my side.
- ⚠ **Concurrency:** the shared main checkout had another agent's branch
  (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` @ `7fcef50`) checked out, so per §3.2 all
  work was done in a dedicated worktree; **the main checkout was never edited.** No file overlap with
  wt1/wt2 (`compact-atoms`, worktree `../CHLU-c2w8a`, still at `80d7d4b` when I branched).

## 6. Open questions / follow-ups / risks

1. **`rho_geom = 1.0` is a registered choice, not a measured optimum.** On the frozen census it
   refuses everything (correctly — there are no basins to merge), so the census cannot discriminate
   between `rho_geom ∈ (0, 1.16)`. R3 registers the per-seed thresholds so a later loosening is
   visible as a decision. **The criterion should be re-scored on the FIRST census that shows real
   capture** (arm A or B), and that re-scoring is where `rho_geom` earns or loses its value.
2. **The payload leg is untested against a non-degenerate real payload population**, because no
   shipped census has one (all `payload_dist ≡ 0`). Its designed negative proves it *can* refuse; its
   calibration (`tau_payload = 0.25`) is a registered prior, not a measurement.
3. **The full-population `M′` is unknown** — see §2.2's honesty note; it needs a re-run census, which
   this spoke is barred from producing.
4. If a future spoke wants merge to consult the criterion, it must **wire the predicate in
   deliberately** (nothing calls it today) — and `test_no_merge_verb_exists` will have to be edited,
   by design.

## Proposed handover updates (for the Hub)

- **§7 NEW (resolved):** *`cl_baselines.ConvNet` built conv weights at the ambient JAX dtype while
  `build_cl_stream` supplies float32 ⇒ the `backbone="cnn"` path raised
  `lax.conv_general_dilated requires arguments to have the same dtypes` under `jax_enable_x64` and was
  untestable in the full suite. **RESOLVED** at `42b781c` (promote to the parameter dtype; x64-off
  bit-identical, asserted) with regression coverage in `tests/test_cl_baselines_x64.py`. **No shipped
  result was affected** (all real runs x64-off).*
- **§7 NEW (standing rule, 4th occurrence of the hazard class):** *any **dtype-strict** JAX primitive
  (`lax.conv_general_dilated` and friends) must be fed at the **parameter** dtype — modules must never
  assume the ambient dtype matches their data. `jnp.asarray(x, dtype=<param>.dtype)` is a no-op at
  x64-off and is the shipped pattern. Test x64 with a **function-scoped** fixture that restores the
  previous value; module-scoped is N211's trap.*
- **§7 / doc drift:** `tests/test_cifar_strong_phi.py:66-72`'s comment documenting the above bug is
  now stale (reconciliation item 1 — needs an owner).
- **Criterion status:** K9 is **registered and proven able to refuse** (`ERRATA-C2W8-PASS2.md` §2),
  but the harness still runs pass 1's criterion. Any quotation of `M` must name its criterion; pass-1
  `M` = 0.2333/0.2417/0.2417 is **vacuous-gate-tripped** and carries that qualifier.
- **No config default changed** and **no new `chlu/config.py` field** — `MergeCriterionConfig` lives
  in `soft_certificate.py` per the C2 rule that a C2 config object lives in the C2-owned module.
