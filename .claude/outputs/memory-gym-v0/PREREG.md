# PREREG — `memory-gym-v0` (Track 1, C2W1)

**Written before any gym cell was run.** Author: experiment-engineer. Branch base `main @ 4160cf7`.
Governing docs: charter §2.1/§2.2/§6.2 · intervention §6 (five criteria) / §8 (prohibitions) ·
`.claude/outputs/full-clu-harness.md` (the API as landed) · task file `.claude/tasks/memory-gym-v0.md`.

---

## 0. DIAL DECLARATION (echoed)

- **Dial (C2 form):** the **dynamics dividend** is the only KPI —
  `dividend ≡ (full CLU) − (its own settle-deleted / matched-bytes launder)`, same harness, same bytes,
  same φ. No accuracy number without its dividend; no dividend without its bytes.
- **Laundering control:** harness-native on **every** cell — (i) `settle_deleted_launder` (arg-min over
  the store's own keys, returning the true payload), (ii) `same_keys_null` (same keys, **permuted
  payloads**), (iii) blank/empty-store control. Plus **one family-specific strong classical control per
  family** (§3), because the frozen three are not the strongest classical method for a non-metric-native
  query and reporting only them would be a laundering by omission.
- **Falsifies (the gym, not the CLU):** same-keys launder ≥ CLU on **every** family **and** every family
  admits a classical provable ceiling ⇒ the gym is metric-native in disguise ⇒ redesign.
- **Does NOT falsify:** a dividend of ≈0 or negative at v0 (charter §6.2's own stated expectation);
  losing to a classical method on a metric-native protocol (metric-native-ceiling theorem).
- **A POSITIVE cell is suspicious, not a win** — all three controls + the family's strong control + a
  multi-seed re-run before it is written down, and it is reported *unexplained-pending-controls*.

## 1. Declared query law and geometry (the harness's narrow-band risk, faced)

`σ_q = 0.15` **isotropic**, identical to the harness's shipped value; `d = 4`, `m = 1`,
`ball_radius = 1.0`, `payload_tol = 0.1`, `kinetic_mode = newtonian_learned`, `dt = 0.05`,
`γ_address = 0.05`, `γ_read = 0.02`, `address_steps = 400`, `read_steps = 800`, `traj_stride = 8`,
`write_steps = 300` (N94-promotable), masked/local write, `confine = 0.05`.
`sep/σ_q` is **reported per cell**. The harness's clean band is `sep/σ_q ≥ 6.83` (decode 0.92) and its
collapsed band is 3.07 (decode 0.48). Two families deliberately place queries **between** wells
(midpoints), i.e. at ~`sep/2σ_q ≈ 3.4σ` from each centre — that is the point of the family, not an
accident, and the risk that it reproduces the collapsed picture is accepted and declared.

## 2. The byte ledger — and a pre-registered *structural* prediction about it

Ledger, per cell, both sides (float32 throughout):

| side | quantity |
|---|---|
| CLU | `V_θ = n_atoms · (dim + 2) · 4 B` (centres `dim`, `amp` 1, `log_width` 1) **+** codebook `K_live · d · 4 B` |
| CLU (reported, not counted by the frozen `byte_account`) | controller records ≈ `K_live · 4 · 4 B` (slot, payload, amp, leak) — eval-only bookkeeping |
| launder | `K · (d + m) · 4 B` |
| strong controls | k-NN-mean **+0 B** · insertion-order **+0 B** (a table's row order is free) · echo **+0 B** · timestamp column `+K·4 B` · shared metric `+d(d+1)/2 · 4 B = 40 B` |

⭐ **PREREG-B1 (the a-impossibility claim; derived, not guessed).** With one atom group per item (which
is what makes the write masked/C3-local) the byte ratio is

    ratio = full/launder = atoms_per_item · (dim + 2)/dim + d/dim
          = 1.4 · atoms_per_item + 0.8      (at d=4, m=1, dim=5)

**independent of `K`.** Therefore `ratio ≥ atoms_per_item`, and **matched bytes requires
`atoms_per_item < 1`, i.e. atoms shared between items — which the masked write forbids by
construction.** Consequences, pre-registered:
1. **Matched bytes is unreachable at v0**, not merely unachieved. Every gym cell will carry
   `matched=False` and none may be quoted as a dividend.
2. Because the byte-matched table always holds **more** items than the CLU does, a byte-matched table is
   never handicapped ⇒ **opening (a) cannot yield a positive dividend at this weight class.** The
   measurable quantity is the **frontier** (accuracy vs byte ratio), not the dividend.
3. **Predicted frontier** (the number I commit to): with `atoms_per_item ∈ {4, 8, 16, 32}` ⇒ ratio
   `{6.4×, 12.0×, 23.2×, 45.6×}`. **Predicted minimum ratio still reaching decode ≥ 0.80: 12.0×
   (`atoms_per_item = 8`)**, range [6.4×, 23.2×]. Derivation: the shipped band is 32 atoms/item and the
   w26 `atom_init_scale` lesson says the failure mode below the floor is *reach* (atoms cannot cover the
   payload excursion), which degrades gradually rather than cliff-edge; 8 atoms in `dim = 5` still spans
   the local tangent space, 4 does not.
   **Falsifier for the frontier prediction:** decode ≥ 0.80 at `atoms_per_item = 4`, or < 0.80 at 16.
4. This supersedes the task file's framing of route 2 ("load the store far beyond the launder's table"):
   at 20 B/entry the launder's table cannot be overloaded without ~270 items, which is out of reach at
   300 write-steps/item. **Declared as a route that arithmetic closes, with the arithmetic shown.**

## 3. The four families, one per charter §2.1 opening — with the metric-native argument each

Cross-cutting stream properties carried by **every** family (charter §2.2): capacity pressure ·
interference (crowd targets in the write loss) · **a deletion demand** · **a revisit** (the same address
re-offered later = regime re-identification) · ≥4 consolidation windows (so monitor #6 becomes
applicable for the first time).

### F1 `overload` — opening (a), beyond-capacity compression
- **Stream:** `n_offer = 3 × reference_capacity` items, all admitted (`capacity = n_offer`), into an atom
  budget sized for `reference_capacity` ⇒ **~3× overload in the quantity that matters (atoms per item),
  not in slots.** The reference arm is the same store at 1× load. Eviction is *not* the overload
  mechanism here: the harness's eviction **re-draws** the freed group (verbatim erasure), so a
  slot-overloaded CLU evicts exactly as verbatim as a table does and opening (a) would be untestable.
- **Metric:** `decode_all` over the full offered payload alphabet (chance `1/n_offer`) **and**
  `mae = mean|v_read − a_true|` (graded, sign-corrected for the dividend). ⭐ **The curve, not the
  endpoint:** accuracy vs atoms-per-item, and accuracy vs offer index.
- **Metric-native argument.** The best classical method inside the byte budget is a table; by PREREG-B1
  it is never budget-limited here, so **F1 fails criterion 4 as a dividend task and is retained only as
  the byte-frontier instrument.** Stated in advance rather than discovered at review.
- **Predicted dividend:** ⛔ **NEGATIVE.** `decode_all` point **−0.30**, range [−0.75, −0.05];
  `−mae` negative. Predicted `matched=False` at every cell.

### F2 `aggregate` — opening (b), non-metric-native queries
- **Stream:** 8 offers, budget 6 ⇒ real capacity pressure + deletion + revisit.
- **Query:** the midpoint of the two nearest live addresses (+ σ_q jitter). **Target = the mean of the
  two payloads** — an answer that is *not any stored item*. Payloads come from `designed_payloads`
  (a permuted even grid), and cells whose true pair-mean lies within `payload_tol` of any stored payload
  are excluded at construction so the arg-min launder cannot be accidentally right.
- **Metric:** `mae` to the pair-mean (primary, sign-corrected), `decode_pairmean` over the pair-mean
  alphabet, `beats_nearest_stored` rate.
- **Metric-native argument.** An arg-min lookup returns a *stored* payload; the target is a function of
  two, and by construction no stored payload is within `payload_tol` of it ⇒ the arg-min launder's error
  is bounded below by a positive constant. **Criterion 4 is satisfied against arg-min but NOT against
  aggregation-augmented classical baselines** — so the family carries its strong control:
  **`knn_mean_launder(k=2)`**, a 2-NN mean at **zero extra bytes**. That is the honest classical ceiling
  and it is expected to win.
- **Predicted dividend:** ≈0, point **−0.02** on `−mae` (normalised by the payload range), range
  [−0.25, +0.10]. **Mechanism for ≈0:** at `sep/σ_q = 6.8` the midpoint is ~3.4σ from each centre, so
  the settle falls into whichever well is nearer and returns *one stored payload* — i.e. it *is* the
  arg-min. A dividend requires the settle to stop somewhere between the wells, which needs overlapping
  basins. Hence the second arm `F2-tight` at `ball_radius = 0.45` (`sep/σ_q ≈ 3.1`): predicted **more**
  negative there (the harness measured CLU 0.479 vs its launder 0.854 in that regime).
  **vs `knn2_mean`: predicted strongly negative** (declared *does-not-falsify*: metric-native ceiling).

### F3 `recency` — opening (c), trajectory information
- **Stream:** lifetimes **ON** (`leak = 0.06`, per-write tick), 8 offers, budget 6, deletion + revisit.
  Older items are physically shallower (their own atom rows are scaled), so **recency is in the
  landscape and not in the address metric.**
- **Query:** the midpoint of two live addresses. **Target: which of the two was written more recently** —
  a binary answer that is *not* a stored value and is *not* determined by the metric.
- **Read-outs (the point-vs-trajectory ablation, handcrafted ψ, v0 limitation declared):**
  `traj` = soft time-occupancy of the phase-2 trajectory near each well → argmax;
  `point` = nearest centre to `q*`. The learned read-out is `trainability-spike`'s.
- **Metric-native argument.** Recency is not a function of the query–key metric, so no arg-min over
  `(keys, payloads)` can exceed chance ⇒ criterion 4 holds **against the frozen launder**. ⛔ **But it
  fails against a zero-byte substitute, and I pre-register that:** a table's **row order already
  encodes insertion order**, so `order_aware_launder` ("the later-inserted of the two nearest keys")
  answers this family **exactly, at +0 B.** It travels with every F3 number.
- **Predicted:** `recency_acc(traj)` **0.72**, range [0.50, 0.90]. **Dividend vs the frozen
  settle-deleted launder: POSITIVE, point +0.22, range [0.00, +0.40]** — mechanism: the deeper (more
  recent) well has the larger basin and the trajectory spends more time in it. **This positive cell is
  declared suspicious in advance**, and its resolution is also declared in advance:
  **`order_aware_launder = 1.000 at +0 B ⇒ F3's positive dividend is a laundering artefact of the
  frozen control's byte allocation, not a dynamics dividend.**
- **Point-vs-trajectory:** predicted `acc(traj) − acc(point)` ≈ **+0.02**, range [−0.05, +0.10] —
  i.e. **≈0**, consistent with the harness's monitor-#10 finding that the trajectory axis is
  semantically dead at this geometry. A flat result here is the pillar's honest v0 datum, not a null
  about the pillar.

### F4 `manifold` — opening (d), flat directions (scoped stub, blocker named in advance)
- **Store:** `n_spectator = 1` ⇒ `dim = 6`. The write objective constrains address+payload only, so the
  spectator axis is unconstrained by the *objective* — but **it is not flat in `V_θ`**: atoms are
  isotropic in `dim`, and `confine = 0.05` is coercive in every coordinate. **Named blocker, in advance:
  `train_memory_landscape` digs point wells, not valleys; a genuine manifold-valued memory needs a
  ridge-write (multi-row collinear targets for one item) and the controller has no verb for it.**
- **Measurement:** launch reads with the spectator coordinate swept over a grid; **`manifold_r2`** =
  R² of settled spectator vs launch spectator (does the store retain a *set* of settled states),
  `manifold_spread` = std(settled)/std(launch), plus the Hessian spectrum at each site
  (`λ_min`, softest-eigenvector spectator participation).
- **Arm `F4-ridge` (bonus, may be declared NOT RUN):** re-write one item with 5 collinear targets along
  the spectator axis inside its own atom mask, then re-measure. A measured blocker beats an asserted one.
- **Metric-native / laundering argument.** No lookup table can express a manifold, so the frozen
  launder scores `r2 ≈ 0` — **but the `echo_launder` (return the launch spectator coordinate) scores
  exactly 1.000 at +0 B.** Therefore any positive F4 dividend is **by-construction** and is reported as
  a *capability measurement*, never as a dividend (intervention §8.3).
- **Predicted:** `manifold_r2` **0.15**, range [0.00, 0.60] (coercivity + isotropic atoms should pull the
  spectator toward 0); `F4-ridge` r2 higher if the ridge write takes, and that is the interesting
  outcome either way. Dividend vs frozen launder: positive-by-construction, **not quotable**.

## 4. Pre-registered summary verdict (the sentence I commit to)

> **The gym's v0 dividend is ≈0 or negative on every family once zero-byte trivial substitutes are
> admitted.** F1 is negative and byte-unmatched by ≥6.4×; F2 is ≈0 because the settle *is* the arg-min at
> in-band separation; F3 is positive against the frozen launder and exactly beaten by a +0 B
> order-aware substitute; F4 is positive-by-construction and exactly beaten by a +0 B echo.
> **The gym falsifier does NOT fire** on its letter (CLU is predicted to exceed the same-keys null on F3
> and F4), but the *cheap-substitute audit* is the real finding and is reported in the first ten lines.

## 5. Monitors — first-fire predictions (an untested monitor is not green)

| monitor | prediction |
|---|---|
| **#3 vacuous gate** | ⭐ **FIRES for the first time** — under 3× overload the utilisation leg (`n_live/N_pack > 0.95`) or the fire-rate leg (`f = 0` on a farthest-point stream with no collision offer) trips. Confidence: high on at least one family. |
| **#4 blank** | does **not** fire (settled-point/occupancy read-outs see only the store's own basins; the harness's refutation #4 says the `q₀ = φ(x)` leak needs a ψ with address access). ⚠ For F3/F4 the read-out *does* use address geometry, so a first fire here is possible and would be a real result. |
| **#6 objective divergence** | ⭐ **becomes APPLICABLE for the first time** (≥4 consolidation windows). Predicted **trips on F1** (write loss falls while acquisition falls under 3× overload — that is literally mode #6) and clear elsewhere. |
| **#11 reach** | ⭐ **FIRES for the first time on F1** if the reach-stress item (`|a| = 1.6`) is admitted; if admission refuses it instead, `admit.reach` fires and #11 stays clear — **both outcomes are reportable and neither is a failure.** |
| #2 settle→arg-min | trips or is inapplicable on F2-wide (`D → 0`, Prop D2a), applicable with `D > 0` on F2-tight. |
| #5 addressing | trips on F1 (3× overload) and F2-tight. |
| #8 certificates | trips on F2-tight (N2 `sep/σ_q = 3.1 < 5.15`) and on F1 (crowding). |
| #9 lifetimes | trips on F3 by construction (that is the family's mechanism) — pre-declared uncleanable. |
| #12 starvation | trips on F1 (fairness `min D/max D` under 3× overload). |
| #10 dead axis | not swept in the gym (the harness owns the knob sweep); reported as inapplicable, never green. |

## 6. Compute order (declared; anything unreached is **NOT RUN**, never a null)

1. `--quick` smoke (1 seed, tiny store): plumbing only, not a result.
2. **F1 byte-parity frontier** — `atoms_per_item ∈ {4, 8, 16, 32}`, 1 seed. (Answers the Hub's hard problem first.)
3. **All four families, seeds {0, 1, 2}** — the acceptance criterion.
4. **F2-tight** (`ball_radius = 0.45`), seeds {0, 1, 2}.
5. **F4-ridge**, 1 seed.
6. *(optional, likely NOT RUN)* an anisotropic-query-law cell where `shared_metric_launder` can beat plain
   arg-min. With the declared isotropic σ_q the fitted metric is ≈ `I` and the shared-metric launder is
   expected to **tie** arg-min; it is still run (doctrine I-12: it has never been run anywhere).

**Statistics convention, declared once:** multi-seed cells report mean ± **sample** sd (`ddof = 1`) and
`SE = sd/√n`, `n = 3`. (The handover flags a cross-wave sd-convention split; this task uses sample sd.)

**Compute discipline:** 1 worktree (`../CHLU-gym`), main venv (JAX 0.9.0, no worktree `uv sync`).
No background sweeps without telling the Hub.
