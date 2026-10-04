# ITERATION 2 — COMPILED REPORT (coordinator, 2026-09-18)
**Four execution hubs, Windows/FORGIS_14, seed 0, 4 threads each. All finished; no process running.**
Sources: `hub-generation-table/LOG-iter2.md` · `hub-composition/iter2r/REPORT.md` ·
`hub-deletion.md` · `hub-admission/REPORT-hub-admission.md`.
⚠ **`hub-generation-table.md` and `hub-composition/REPORT.md` at the top level are ITERATION-1
documents** — do not read them as this iteration's. Iteration 2 lives in `LOG-iter2.md` and `iter2r/`.

---

## SCOREBOARD

| hub | verdict | scored | budget after |
|---|---|---|---|
| `generation-table-2` | ⚠ **TIED, not won** — banked | **1/3** | 2 left |
| `composition-2` | ⭐ **WIN on the pre-registered bar** — with three caveats that must travel | **1/3** | 2 left |
| `deletion-2` | ⚠ **BANKED TIE** + 2 unasked-for real results + 1 charter condition failed | **0/3** | 3 left |
| `admission` | ⛔ **NOT SOLVED** — dead on arrival, one measurable reason | **0/3** | 3 left |

**2 of 12 scored runs spent across the programme.** No hub moved a bar. No pinned file edited. Nothing deleted.

---

## 1. `hub/generation-table-2` — TIED
- **Every rival beaten outright; `knn_proto` tied.** Falsifier 1 part-answered; the matched-bytes leg not.
- ⭐⭐ **The read verb is the finding, and it got bigger on held-out data than on DEV: `+0.2014`,
  9 → 27 tuples**, byte-identical ledger. Confirmed at **43/43** corners as well as on `eval_b`.
- ⭐ **Parity measured and exact where it matters:** `clu` **0.5903 / 9 tuples** against the published
  Mac row **0.5949 / 9.00 ± 0.82**. ⇒ Windows numbers sit inside the published seed spread.
- Against `knn_proto` at 1.26× the bytes: **−0.0893 [−0.1025, −0.0760], 0/43.**
- ⭐ **Five of its own errors recorded as its own**, including a thread-budget breach and a
  pre-registered §C.1 margin that measured 0.0000. Its PREREG predicted *"a TIE, not a WIN"* — correct.

## 2. `hub/composition-2` — WIN, and the caveats are the important part
**On `eval_b`, seed 0, at 469,504 B = 0.9881 × `knn_proto`:**
`clu_linear` **42 / 72** vs `knn_proto` **12** · §C.1 matched-bytes lookup **0/0** · §C.2 landscape
removed **0/0** · constant-output floor **0** · decoder ceiling 67.
**Δ = +30.00 tuples, CI [+20.27, +39.73], McNemar exact p = 2.27 × 10⁻⁷ ⇒ WIN.**
⭐ **Independently replicated** by a §A.5 spoke barred from the hub's code: **25 vs 12, Δ = +13.00
[+3.88, +22.12]** — two builds, two code paths, two wins.

⛔⛔ **Three numbers that must travel with it, per the hub's own insistence:**
1. **Over 105 held-out-shaped corners in two location-disjoint families: Δ = +1.09, CI [−1.52, +3.69]
   — a TIE.** That is **the generalising estimate**, and it is *not* a tie for lack of power (CI ±2.6).
   Families agree to 0.24 tuples; three store seeds agree to 0.9.
2. ⛔ **The B level does not replicate — 42 (hub) vs 25 (spoke)** — because its PREREG failed to pin the
   store fit. ⭐ The hub refuses to quote "42" as reproducible.
3. The win is a single-cell result; the hub's own framing is *"the measurement of how little that
   single-cell win can bear."*

## 3. `hub/deletion-2` — banked tie, plus two results nobody asked for
- ⭐ **Blocking gate PASSED, stronger than required:** `train_store` deterministic at shipped cell scale —
  **three independent OS processes, three launch paths, 22/22 arrays bitwise identical**,
  `store_digest d0eacf3b…`, across a **1.31× wall-clock spread** (554.9 / 617.9 / 728.3 s).
- **Headline (pre-registered): BANKED TIE** — churn-corrected collateral Δ = **+0.0030**,
  CI [−0.0071, +0.0112], m = 0.020, **0 of 6 deletions outside the margin**.
- ⭐⭐ **§3 — canonical rebuild is NOT answer-preserving.** A mechanism-level cost the ported
  *Retention* method does not price. Not asked for; real; non-by-construction.
- ⭐⭐ **§4 — a genuine WIN: the CLU forgets a conjunction; the table cannot.** Compositional deletion.
- ⛔ **§5 — charter condition 1 (unrecoverable) is NOT uniformly met**, and the hub says the failure is
  the useful part.
- ⛔ **Scored run NOT spent — 0 of 3.**

## 4. `hub/admission` — NOT SOLVED, and the reason is exact
The gate is **correctly derived, correctly implemented, correctly certified**: identity
`max_i E_k(a_ki) ≤ 0` holds with 30 units to spare, the per-well reduction reproduces shipped
`store_energy` to **1.2e-07**, and **69–73 % of its refusals are certified** by the `σ_k√(2 ln N)` bound.

⛔ **It is dead on arrival for one measurable reason: the radius is in the wrong place.**
- Zero level set sits at **`d/σ_k ≈ 2.0 – 2.4`**.
- **The entire data manifold — every real encoder feature, held out or not — sits at `d/σ_k ≈ 0.79`.**
- **Why, and it is structural:** `σ_k` is `nn_median`, the median nearest-neighbour distance *between
  atoms*; the atoms **are train rows**, so they tile their own manifold at spacing ≈ `σ_k`
  **by construction**, and any real point is ≈ 1 `σ_k` from one of them **whatever the store contains**.
  The level set is pinned ~2.6× outside the data **by the bandwidth rule itself**.
- 8× the store closes the gap by only **~0.09** in `d/σ_k` while pushing the level set further out.
- ⇒ At the shipped `clamp_atom` launch the gate is **vacuous by algebra** (the launch point *is* an atom);
  at `posterior_mean` it is vacuous because a convex combination cannot leave the hull. Only a **linear
  extrapolation** (R8 affine head, severity `w ≥ 3`) ever reaches it.
- ⛔ Scored 0/3; `eval_b` never read.

⚠ **Coordinator note against itself:** the pre-launch `§0c` warning got the *quantity* right (the level
set is ~2.9 σ; the data is not) but the **direction wrong** — I predicted the gate would refuse
everything; it accepts everything. The failure is vacuity, not over-refusal.

---

## 5. ⛔⛔ PROCESS INCIDENT — two agents on one hub
`hub-deletion/INCIDENT-two-agents-clu-dev-1e.md`: **two autonomous agents executed `hub/deletion-2`
concurrently**, sharing one worktree and one output directory, writing the same tag names under `runs/`,
with `LOG.md` interleaving both RELAUNCH sections. Session `clu-dev-1e` filed the incident, noted that
*"a log two writers append to is not a reliable record of either"*, and — correctly per LAUNCH-COMMON
§-1 — logged it and carried on rather than escalating.
⚠ The final `hub-deletion.md` is authored by `clu-dev-1e`; the other agent's in-progress skeleton is
preserved at `hub-deletion/REPORT-skeleton-clu-dev-89-1127Z.md`. **Nothing was deleted.**
⭐ **This is a coordination failure at the coordinator/human level and needs a fix before iteration 3.**

## 6. Cross-cutting
- ⭐ **Parity is settled empirically**: generation-table-2's `clu` row lands inside the Mac's published
  seed spread. Windows numbers enter the corpus.
- ⭐ **The re-encode ruling held.** No hub re-encoded; no shortfall > 0 was hit; the gate stayed at 1 ULP.
- ⭐ **`accept_A7`-beside-`accept_B`** (§7b) was adopted mid-iteration; admission is the only hub it bound.
- ⛔ **Seeds 1–2 remain unreachable** on this box. Every number above is **seed 0 only**.
