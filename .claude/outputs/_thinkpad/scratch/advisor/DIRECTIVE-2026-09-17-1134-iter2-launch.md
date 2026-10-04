# DIRECTIVE — to the Windows hub agent, from the Head and the Advisor (joint), 2026-09-17

**You are scoping and launching FOUR autonomous hubs on FORGIS_14, one per target, in parallel, 12 h
true wall clock each.** This file supersedes `LAUNCH-COMMON.md §1c` (the budget is now stated) and
`§4`'s "STOP and report" clause on the re-encode gate (replaced by §2.2 below). Everything else in
`LAUNCH-COMMON.md`, `PREFLIGHT-12H.md` and the charter still binds unless a line here says otherwise.

⭐ **The goal, in one line (Head):** ICLR-main-track results for the CLU in generation, composition,
refusal and deletion — as fast as possible, ~7 days for results including today, scientific and
honest. **What we strictly do not want is a glorified lookup table.** The Hamiltonian settle dynamics
are *nice to have*; the store, the memory modes, exact deletion and request-only addressing are the
core. A hub MAY modify the CLU for its task as long as that core is intact (charter §B).

---

## 1. THE RULING

**The re-encode gate stays at 1 ULP and is never loosened. It simply never binds on a scored run,
because no scored run on this box may extend the cache.** The gate exists to catch *writing new rows
through an encoder that does not reproduce the cache*. A run that encodes nothing new cannot trigger
that hazard, and the pin already ships the scoping for exactly that case
(`exp_gen_wells_route_b:382`, `will_extend=bool(cov["n_shortfall"] > 0)`). `exp_gen_cell.run_seed:1780`
never computes coverage and so cannot use it — that is an omission in the harness, not a decision
anyone made. **The Head authorises the one-line patch (§2.2) that gives `run_seed` the same
scoping.** Seed 0's cache is `scope="full"` (480,000 rows = the whole split) and provably cannot be
extended, so seed 0 scores here today. Seeds 1 and 2 (`scope="used"`, 9,288 rows) **cannot** run here
until the Mac ships full-scope caches; that is happening in parallel today and is not on your path.

*Why this is principled and not convenient:* the charter's declared property #3 already accepts ~53
ULP on ~1 % from the cluster's own L40S, post-precision-fix — so the 1-ULP bound has never once been
exercised by a non-trivial passing case (the Mac wrote the cache; its gate passes at delta = 0). The
Windows numbers (51/81/59 ULP on 0.49–0.98 %) are out-of-sample confirmation of a number the programme
banked before this box existed. **The one thing that would make this ruling wrong** is if the 51–81
ULP sits on normal-range activations rather than near-zero post-ReLU dust — §5.2 measures that in
two seconds and stops everything if so.

*The ruling stops applying* the moment any run on this box needs a row that is not in a Mac-written
cache. Then the gate binds at full strength, it will fail, and the correct action is STOP and report —
never a re-encode.

**Head rulings recorded today (2026-09-17), binding on all four hubs:**
- **R1 — Tie definition (charter §A / the ceiling rule):** *tie* := the paired 95 % CI on
  Δ(arm − rival) includes 0, **or** |Δ| ≤ m = 0.020. *Win* := Δ > 0 with CI excluding 0. *Loss* := Δ < −m
  with CI excluding 0. Set before any scored run against it (§A.3 satisfied). **A tie is banked and
  reported; the hub then keeps pushing for a win with its remaining time.**
- **R2 — Composition's bar is retired.** ≥ 60/72 distinct valid novel tuples is replaced by: *distinct
  valid novel tuples on B at matched bytes, win-or-tie vs `knn_proto` under R1*, plus the measured
  capacity law and the decoder ceiling reported beside it.
- **R3 — Cross-device provenance.** Every reported number carries the machine that produced it.
  Headline comparisons (arm vs rivals) are same-device by construction: the scored cell re-runs every
  arm, rivals included, on this box from Mac-encoded features. Mac-published rows are cited as prior
  rows with provenance, never mixed *inside* a paired comparison. "Too strict costs time we do not
  have" — do not chase bit-identity; do not block on it.
- **R4 — The PCA-affine dictionary is NOT adopted as a store.** Reported as a control only. Reason: a
  principal direction cannot be decoded or deleted; the deletion target needs atoms.
- **R5 — The trajectory-encoder idea (reading the encoder's intermediate layers as CLU trajectories)
  is deferred** to a later iteration. Measured reason: the pairwise request model reaches 0.9734 / 68
  of 72 on the *existing* latent, so latent richness is not the binding constraint on this benchmark.

---

## 2. WHAT THE HUB DOES NEXT — in order

### 2.1 Machine preconditions (before any worktree is created)
1. `PREFLIGHT-12H.md` steps 1–4 and 6, re-verified (sleep = 0, pending reboot = False, `PYTHONUTF8=1`
   proven in the launching shell, verifier green except the one overridden test).
2. **Raise the Windows page file.** `hub/generation-table` died at DEV-29 on
   `LLVM ERROR: Unable to allocate section memory / The paging file is too small` with three hubs.
   Four hubs will hit it sooner. Admin PowerShell, then reboot:
   ```powershell
   $cs = Get-CimInstance Win32_ComputerSystem; $cs | Set-CimInstance -Property @{AutomaticManagedPagefile=$false}
   $pf = Get-CimInstance Win32_PageFileSetting -Filter "Name='C:\\pagefile.sys'"
   if ($pf) { $pf | Set-CimInstance -Property @{InitialSize=98304; MaximumSize=131072} } else { New-CimInstance Win32_PageFileSetting -Property @{Name='C:\pagefile.sys'; InitialSize=98304; MaximumSize=131072} }
   ```
   (96–128 GB on a disk with 833 GB free.) Verify after reboot: `Get-CimInstance Win32_PageFileUsage`.
3. Core budget: **4 threads per hub** (`OMP/MKL/OPENBLAS_NUM_THREADS=4`), 4 hubs = 16. No hub exceeds
   its 4. Sub-agent cap per hub: 2, inside the 4 threads.

### 2.2 The authorised harness patch and the new base commit
4. In the bundle repo (`chlu-bundle-v2`, single commit `8e36bc9`), create **one new base commit**
   `base/iter2` containing, on top of the pin tree:
   a. The five new files from `hub/generation-table @ cb6d4d4` — `chlu/core/gen_factored_read.py`,
      `gen_partition.py`, `gen_address.py`, `gen_sensitivity.py`, `tests/test_gen_factored_read.py`
      (940 insertions, 0 modified files; verified byte-identical harness). This is the reference object.
   b. **The gate patch, exactly this and nothing more**, at `chlu/experiments/exp_gen_cell.py` just
      before the `_assert_cache_reencodes` call at line ~1780:
      ```python
      # [HEAD-AUTHORISED 2026-09-17] scope the preflight gate to the hazard — the route_b pattern.
      # Every row this run will gather is known here; if none is uncached, this run cannot extend
      # the cache and the (unchanged) bounds do not bind. gather_features re-checks at the point
      # of extension at full strength regardless.
      _need = np.unique(np.concatenate([pool] + [np.asarray(v) for v in row_sets.values()]))
      _n_shortfall = int((~np.isin(_need, crows)).sum())
      reenc = _assert_cache_reencodes(
          dataset, config, encoder, crows, feats_cache, cman, log=tlog,
          will_extend=bool(_n_shortfall > 0),
          reason=f"run_seed coverage: {_n_shortfall} of {_need.size} needed rows uncached",
      )
      tlog(f"cache coverage for this run: shortfall={_n_shortfall} (gate enforced={_n_shortfall > 0})")
      ```
      replacing the existing two-line call. ⛔ `ULP_BOUND`, `FRAC_BOUND` and `_assert_cache_reencodes`
      itself are **not touched**. If any other enforced site (`exp_gen_compose:408`,
      `exp_gen_baselines:1508`, `exp_gen_dream:519`, `exp_gen_atom_mean:802`,
      `exp_gen_dynamical_read:507`, …) is actually hit by a hub, it may receive the *same* pattern —
      logged as a dated bullet with the diff, never silently.
   c. Commit message: `[head-authorised] base/iter2: reference read + run_seed gate scoped to the hazard`.
   Record the full `git diff 8e36bc9 base/iter2 --stat` in `_hub_scratch/advisor/base-iter2.diffstat`.
5. **All four hubs branch from `base/iter2`.** `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-<slug> -b hub/<slug>-2 base/iter2`.
   ⛔ Not from `8e36bc9`, ⛔ not from a peer's branch, ⛔ not from the old `hub/*` branches.

### 2.3 Write four `LAUNCH-hub-<slug>.md` addenda from the scopes in §7, and launch
6. Rewrite `LAUNCH-COMMON.md` §1c to state the budget (§4 below) and §4 to state §2.2's rule in place of
   "STOP and report". Keep everything else. Add R1–R5 verbatim.
7. Each addendum names: slug, branch, worktree, 4 threads, the object it starts from, its bar (from §7),
   its mandatory controls, what it may read from its predecessor hub, and its PREREG obligation.
8. Launch all four. Record `START_UTC.txt` per hub. You (the scoping hub) then monitor only: you do not
   run experiments yourself.

### 2.4 At 12 h
9. Each hub files PREREG · ledger · LOG.md final section · cost line · `--return-pack` · whole-branch
   bundle, per COMMON §6. You collect the four reports into `_hub_scratch/reports/iter2-<slug>.md`
   (copies, not moves) and stop. The Head and Advisor read them and issue the next directive.

---

## 3. WHAT IS STILL FORBIDDEN — explicitly

- ⛔ **Re-encoding the feature cache, ever, for any seed, by any hub.** A shortfall > 0 means STOP and
  report the shortfall; it does not mean encode. (Seeds 1 and 2 are therefore off-limits for any scored
  cell this iteration.)
- ⛔ Editing any manifest-pinned file, any frozen artefact, `ULP_BOUND`, `FRAC_BOUND`, the external
  classifier, the splits, the rival rows, the byte-ledger or bypass assertions. The **only** harness edit
  is §2.2(b), applied in `base/iter2`, identical in every worktree.
- ⛔ Quoting `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` (superseded; check the directory).
- ⛔ Changing a bar after seeing a held-out result against it. R1/R2 were set before any scored run.
- ⛔ Mixing a Mac-produced and a Windows-produced quantity **inside one paired comparison**. Citing with
  provenance is allowed (R3).
- ⛔ A threshold, criterion or dictionary chosen after looking at invalid/held-out examples and then
  reported as if it were not. Disclosure is mandatory, not optional (see refusal scope).
- ⛔ Adopting an affine-PCA basis as the store (R4). ⛔ Starting the trajectory encoder (R5).
- ⛔ GPU runs, cloud spend, CSF3. ⛔ Deleting anything. ⛔ Reading a *concurrent* peer's outputs.
  ✅ Reading the three **predecessor** hubs' reports and outputs is allowed and expected — they are
  inherited facts now.
- ⛔ More than 4 threads per hub. ⛔ A second scored run per hub this iteration (§4).

---

## 4. SCORED-RUN BUDGET — stated

**Three per target, across every attempt and every machine. Zero spent so far on any target, on any
machine** (all three overnight hubs spent 0/3; the Mac has spent 0 on these four targets).

**This iteration: at most ONE scored run per hub, seed 0 only**, and only after (i) its PREREG names the
exact object, bar and controls, and (ii) its DEV number clears its bar on non-held-out corners with the
§C controls passing. A seed-0-only number is reported as *"scored 1/3 — seed 0 only, 3-seed pending
full caches"*. Budget after this iteration: 2 remaining per target, of which the 3-seed final in a later
iteration costs 1.

⭐ **Parity costs nothing extra.** The first scored cell includes the `clu` arm; its Windows seed-0
composition-B aggregate and distinct-tuple count against the Mac-published seed-0 `clu` row
(`gen-read-cell-arm/runs/cell/seed_0/…/cell_manifest.json`; the 3-seed row is 0.5949, 9.00 ± 0.82
tuples) **is** the parity measurement CONTEXT demanded. Bound: aggregate within the published 3-seed sd,
distinct tuples within ±1. Reported in `hub/generation-table`'s ledger as a labelled row, not a spend.

---

## 5. WHAT WOULD FALSIFY THIS RULING — stop and come back if you see any of these

1. **The ULP is on real activations.** `hub/generation-table` runs this at boot, before anything else
   (≈ 2 s): load seed 0's cache, take `crows[:64]`, encode exactly as `_assert_cache_reencodes` does,
   and for every differing element record `(|cached|, delta, ulp_delta)`. Write
   `$CHLU_HUB_OUTPUTS/hub-generation-table/reencode_magnitude.json` with: the `|cached|` of the
   argmax-ULP element; the count and max ULP of differing elements with `|cached| ≥ 2^-4`; the same
   for `< 2^-4`; `max_abs_delta`. **If any element with `|cached| ≥ 2^-4` shows `ulp_delta > 2`, all
   four hubs stop scored work and report** — the encoder arithmetic on this box genuinely differs and
   even a no-extension run is suspect. (Expected: the 51–81 ULP sits entirely on `|cached| < 2^-4`,
   where `np.spacing` has collapsed to the float16 subnormal gap of 6e-8 and ULP counting is
   meaningless — the pathology report I already found at `I-numerical-divergence.md:212`.)
2. **Parity fails** (§4 bound). Windows scored numbers do not enter the corpus; hubs revert to
   mechanism-only for the iteration and say so.
3. **A shortfall > 0 on seed 0.** Should be impossible (full scope). If it happens, the split or the
   cache is not what we think it is: stop, report the rows.
4. **§C.2 stops collapsing** for the linear read at the chosen K (known at K ≳ 128 — a partition gives K
   free scalars). A configuration where `store_removed` does not collapse is disqualified before it is
   scored, never after.
5. **Deletion byte-identity fails** because `train_store` is not deterministic given the live set on this
   machine. Report the two hashes and the batching; do not loosen "byte-identical" to "close".
6. A hub finds itself wanting to raise `ULP_BOUND`, widen `FRAC_BOUND`, or pass `will_extend=False`
   with a shortfall > 0. That is the exact loosening this ruling forbids; it is the signal to stop.

---

## 6. WHAT GOES UPSTREAM TO THE MAC — today, in parallel, not on the hubs' path

1. **Bridge caches for seeds 1 and 2 at `cache_scope="full"`** (seed 0's already is). This removes every
   extension on every machine permanently and unblocks the 3-seed finals next iteration.
2. **The §2.2(b) gate patch into `main`** (and the same pattern at the other enforced sites), so the next
   pin ships it and `base/iter2` needs no carry-forward.
3. **Reconcile charter property #3 with the gate**, using `reencode_magnitude.json`: either the ULP
   metric gets a magnitude floor (`ulp = np.spacing(max(|cached|, 2^-4))` or equivalent, *derived* from
   float16's normal range, not fitted to Windows) or property #3 is rewritten to say what the gate
   actually enforces. Humans decide; not a hub.
4. **Charter edits:** R1 (tie definition) into §A; R2 (composition bar) into the composition task file;
   the 0.5278 constant-output floor into §E.4; §A.2 rewritten to state the per-target budget and its
   count; `exp_gen_cell` persists per-render tuples for **every** arm (composition PREREG amendment 1).
5. **Merge `hub/generation-table`'s five files into `main`** so they are pinned, not carried.
6. **Re-pin v3** from the result; bundle to this box. Hubs in iteration 3 branch from v3.

---

## 7. THE FOUR HUB SCOPES — write each `LAUNCH-hub-<slug>.md` from its block

**Common to all four.** Start from the reference object on `base/iter2`: **K = 32 correlation-partitioned
wells · M = 26 k-means modes per well · request-addressed LINEAR read (`clu_wide_vector` over real
request-conditioned weights) · rank-16 pair head · 469,504 B = 0.988 × `knn_proto`'s 475,136 B**
(`hub-generation-table.md` §5). It is charter-legal: addressed not indexed, request-only
(`assert_request_only_factored` and `assert_no_bypass` pass), atoms in wells, matched bytes. A hub may
depart from it **for a measured reason logged as a dated bullet** — but note charter §C.3: the
generation *tie* becomes the win narrative only if **the same object** delivers deletion / refusal /
dreaming, so departures cost the paper coherence; depart only when the measurement demands it.
Every row every hub reports carries: the §C.1 **medoid** control (a dictionary of the real row nearest
each learned centre — the sharpest lookup-table test; report **K** with it, because the pass exists at
K = 32 and vanishes at K = 5), the §C.2 `store_removed` control, the §C.4 distance-to-nearest-atom over
atom spread, the constant-output floor 0.5278 where composition-B is the metric, and `n` corners/seeds.
Inherited nulls (charter §E.4) are not rediscovered: learned placement, codebook scale/orientation loss,
density-over-atoms descent 0.685 → 0.211, dreaming inert at every β, argmax read discards the span.

### 7.1 `hub/generation-table-2` — the headline
- **Target:** WIN or TIE (R1) vs `knn_proto` on composition-B aggregate at matched bytes, external
  classifier, shipped cell, seed 0 this iteration. On a tie, keep pushing for a win.
- **Starting numbers (DEV, 49 corners, predecessor):** arm 0.8135 / 49.9 tuples vs `knn_proto` 0.8307 /
  51.8 — Δ = −0.0172, CI [−0.0347, +0.0003], 16/49. Under R1 that is a **tie already** (CI ∋ 0 and
  |Δ| ≤ 0.020). Bank it, then push.
- **Levers the predecessor measured but did not exhaust:** partition K (5 → 64 lifted round-trip 6 → 47;
  §C.2 fails at K ≳ 128 — stay below), head rank, pair-term selection (`hue × scale`,
  `shape × orientation` carried the 0.9734 bound), medoid-vs-learned margin (+0.085 at K = 32).
  ⛔ Not the PCA basis (R4). ⛔ Not the argmax read.
- **Must do first:** §5.1 `reencode_magnitude.json`. Then the scored cell (seed 0, all arms) — this is
  the one scored run, and its `clu` row is the parity measurement (§4).
- **Reports:** the table with the 0.5278 floor drawn; the byte-identical linear-vs-argmax row on the
  shipped K = 5 store (+0.119, 6/6) as a standalone finding; per-render tuples for every arm persisted.

### 7.2 `hub/composition-2`
- **Target (R2):** distinct valid novel tuples on B at matched bytes, win-or-tie (R1) vs `knn_proto`;
  report the capacity law `rel-err = e^1.346 · bytes^−0.1651` (R² 0.9960) and the decoder ceiling
  67.67 ± 2.49 beside it. Identity to use: on B, `distinct_valid_novel = 72 × exact_match` (all 72
  targets distinct and absent from every train row) — so the headline column needs no new instrument.
- **What changed since the predecessor:** its "0 tuples, 147,000× bytes" is a law about the **argmax
  quantiser's self-aligned residual**, not about the store. The same store under the linear read renders
  35–50 tuples at the shipped bytes. Start there; the predecessor's VALID∧NOVEL instrument (24/24 frozen
  rows reproduced) is inherited and reused, not rebuilt.
- **Levers:** everything in 7.1 plus dreaming *on the linear read* at a declared finite T (the
  predecessor measured it inert on the argmax read; it may not be on a read that is not pinned to a
  vertex). If it is still inert, say DEAD in those words.
- **Scored run:** if this hub's object is the same as 7.1's, its readout comes from 7.1's scored cell
  (per-render tuples persisted) and this hub spends **0**. It spends its 1 only if its object differs,
  and says why.

### 7.3 `hub/refusal-2`
- **Target, re-posed on the predecessor's structural result:** the original criterion (settled energy
  of the argmax/MALA read) is measured structurally blind — the address book is a simplex over stored
  atoms and "a simplex has no outside"; `out_of_range` has the **sign backwards**, monotonically over
  13× extrapolation; the store is decorative for it (`store_removed` 0.7809 ≥ 0.7511). Do not re-derive.
- **The new criterion, pre-registered, ONE, landscape-derived, threshold on `val` only:**
  **coercive-component screening** — `S = E_store(x_read)`, the store's own energy at the linear read
  point (vocabulary and precedent: *Certified Access*, BIBO screen, "refuse the jump if the exit lies
  outside the coercive component"). Under the argmax read this carries zero information (the read is
  always on an atom, distance 0.0000); under the linear read the weights are unconstrained and an
  invalid request can put `x_read` outside every basin. Refuse iff `S > quantile_0.95(S on val)` —
  the harness's own shipped `refusal_valid_quantile`, not swept. **Companion, rejected in advance:**
  `d(x_read, nearest atom) / atom spread` (the §C.4 statistic) — computed and logged, never adopted.
- **Mandatory disclosure line in PREREG-1, verbatim:** *"This criterion was chosen on a structural
  argument formed after reading `hub/refusal`'s analysis of invalid examples on a fit-derived synthetic
  analogue. No benchmark invalid row was seen. The zero-invalid-supervision claim is therefore
  'structural, not blind', and the paper says so."*
- **Mandatory controls:** (a) `store_removed` must collapse `S`'s AUROC to ~0.5 — otherwise it is the
  address book again and the result is void; (b) a **physics-free readout-margin control**
  (*Certified Access* §4.1 measured the relaxation residual at ≤ +0.024 AUROC over a plain margin — a
  reviewer will ask); (c) **`accept A7` printed beside `accept B` on every row** (three of five rivals
  falsely refuse ~60 % of the valid A7 control; the published table hides it).
- **Bar:** `min(AUROC_oor, AUROC_dup) ≥ 0.75` at the single val-only threshold, `accept_A7 ≥
  accept_B − 0.05`, and beating every rival on the **pair** (best rival pair is kNN's (0.8426, 0.3519);
  no rival refuses both kinds). Beating kNN on `out_of_range` alone (0.97) is not required and is not
  expected.
- **Go/no-go before any scored spend:** run `S` once on the predecessor's synthetic analogue
  (`scorer/refusal_lib.py`, `term_experts`, inherited) at rung A and B, seed 0. If `min(oor, dup) < 0.6`
  there, do not spend; report the number and stop the refusal line for this iteration.
- **Scored run:** the C-set scoring uses seed 0's traces from the linear read; 1 spend max.

### 7.4 `hub/deletion-2` — nothing has run on this target yet; the cheapest genuine result on the board
- **Target (charter):** exact structural deletion of one factor value or one atom, on all four
  conditions: unrecoverable from every composition · every other factor preserved · byte-exact
  provenance · the rivals given the same deletion. ⛔ The trap the task file names: *store load-bearing
  by construction is a method-section fact, never a headline*; "the rival has no delete operator" is
  worth nothing.
- **The mechanism — port, do not reinvent:** *Retention You Can Predict, Scope and Delete* §2.3 /
  App. F / App. G.8. Deletion is exact when the store layout is a **canonical function of the live item
  set and the fixed geometry** (history-independent placement), so `delete(v)` := remove `v`'s rows from
  the live set and **recanonicalise (rebuild)**; the post-delete store is then **byte-identical** to a
  store that never held `v`. Cost: O(n) rebuild per delete — accepted openly in that paper; put it in
  the cost line. Its caveat carries over verbatim: *store-level guarantee only; the frozen encoder is a
  separate channel; no certified (ε, δ) unlearning is claimed.*
- **On the gen store, concretely:** (1) make `train_store` on the reference object a deterministic
  function of the live row set — fixed seed, **fixed batch shape** (report I: same batching ⇒ identical
  `store_hash`; changed batching ⇒ moves), no eviction (fixed K × M) — and **prove it**: two independent
  builds from the same live set must hash identical on this machine before anything else.
  (2) `delete_value(v)`: rebuild from `pool ∖ rows(v)`; assert `hash(delete(store, v)) ==
  hash(build(pool ∖ rows(v)))`. That equality **is** condition 3. (3) `delete_atom(j)` (zero the
  mode's column in dictionary and head) is **not** canonical — the other modes were fitted with `j`
  present — so it is reported as a *scoped* operation with its measured residue, never as exact.
- **The four conditions, measured on seed 0, val/DEV corners (no held-out spend until the final):**
  unrecoverable — external-classifier accuracy on factor `v`'s value drops to chance for every request
  naming it and no rendered tuple contains it; preserved — per-factor accuracy on requests **not**
  naming `v` within pre-deletion seed spread; provenance — the hash equality; rivals — `knn_proto` and
  atom-mean get row deletion (canonical by construction); `lace` / `ebm_compose` have no exact operator:
  retrain-without-`v` if it fits the budget, else **NOT RUN**, and either way a method-section line.
- **The non-by-construction headline — measure this:** **collateral**. For each deleted value `v`, the
  paired change in per-factor accuracy on compositions that do *not* contain `v`, CLU vs `knn_proto`,
  at matched bytes, over the 6 factors (one value each: floor hue, wall hue, object hue, scale, shape,
  orientation) — plus rebuild wall-clock per delete. That comparison is real whichever way it falls.
- **Existing pieces:** `chlu/core/gen_store.delete_store(mode=zero|mean|scramble)` is the *laundering
  control* (its own docstring); `gen-store-grouping/runs/shapes3d_recon/seed_*/store_manifest.json`
  already carries `delete_zero_* / delete_scramble_* / delete_mean_*` deltas — the pre-deletion control
  is partly in hand. Budget: a rebuild ≈ one `train_store` (546–1258 s per corner at 5 threads on the
  predecessor's ward wells) ⇒ plan ~20 rebuilds in 12 h, not 200.
- **Scored run:** 1 max, seed 0: the deletion cell on held-out B for the 6 deletions, PREREG'd with
  the collateral bar stated as a number before it runs.

---

*Elapsed-time policy, budget, resumability, logging and finishing rules: `LAUNCH-COMMON.md` §1b, §3, §5,
§6, unchanged. Stopping at 12 h with "not solved, here is exactly how far I got" is a success.
Manufacturing a result is the one unrecoverable failure.*
