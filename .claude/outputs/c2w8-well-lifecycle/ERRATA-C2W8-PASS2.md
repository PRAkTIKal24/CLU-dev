# ERRATA-C2W8-PASS2 — dated addenda to `PREREG-C2W8-PASS2.md`

Filed **before** the cells each block governs. `PREREG-C2W8-PASS2.md` is **not edited** (a revised
pre-registration stops being one). Spokes append their own dated blocks below; **append only, never
rewrite another block.**

---

## §1 — 2026-08-06, filed by the C2W8 Hub BEFORE any pass-2 cell: the four Head rulings that close the prereg's §7 carried questions

The prereg §7 carried four questions to the Head + Advisor. **All four are RULED, same session.**
None changes a registered prediction (P1–P6 stand as filed); they resolve scope and reporting.

### Q1 — ✅ RULED: **BANK the rider-1 finding and ROUTE to C2W9. The confirmation cell is NOT bought.**
The Hub established from banked evidence that the `(d, atom budget)` co-scaling **was honoured** —
`ERRATA-C2W8.md` §3's `n_atoms` column matches `round(512·√2^d)` exactly at every dimension
(4 → 2 048 · 8 → 8 192 · 12 → 32 768 · 16 → **131 072**), verified against `clu_system.py:314`.

⇒ **BANKED FINDING (quotable in this form):** *the `d ≥ 16` inertness was measured at a fully
co-scaled atom budget, so the store was **not capacity-starved**; the binding constraint is
**REACH**.* Mechanism, from `ERRATA-C2W8.md` §3's second table: the nearest of **all** atoms recedes
as `√dim` (0.294 at d=4 → **1.483** at d=16) while `atom_width` stays 0.3, so the write gradient's
`exp(−r²/2s²)` underflows **6.18e-01 → 4.98e-06**.

⇒ **ROUTED: the fix is learned p₀ / wormholes — C2W9 territory — not more atoms.** This is a
**registered scoping input to C2W9**, whose spawn trigger remains its own traversal-failure
signature (§A21); this ruling does **not** fire that trigger.
⚠ **Carried honestly:** the banked table came from a **scratch probe of 3 designed-site writes**
(`.claude/scratch/c2w8/timing16.py`), **not** a censused CL-stream cell. The Head has ruled the
confirmation cell **not worth its cost**, so ⛔ **this is a DECLARED NOT-RUN, never reported as a
null**, and the reach conclusion carries the probe's provenance wherever it is quoted.
⚠ **Consequence registered:** *whether "full CLU on a CL stream" has a non-empty operating window —
and therefore whether **C2W10's persistent store has a home** — is now answered by the reach route,
not by buying atoms.*

### Q2 — ✅ RULED: **the Hub's own aggregation is canonical; report BOTH to be safe.**
Canonical = the Hub's re-derivation from `census.json` (**median** over live wells): own
**0.518 / 0.282 / 0.123**, foreign **1.261 / 0.947 / 0.611**. The directive's near-identical figures
(own 0.52 / 0.30 / 0.17, foreign 1.29 / 0.96 / 0.62) come from a different aggregation.
⇒ **Every pass-2 arm reports own/foreign under BOTH aggregations (median AND mean), each labelled**,
so the two forms are reconciled once rather than circulating side by side. ⛔ **This changes nothing
about own/foreign's status: it remains a DIAGNOSTIC, never a target, never a gate leg** (prereg §3).
⚠ Also recorded: site drift's low end is **0.003** (seed 0), not 0.22.

### Q3 — ✅ RULED: **the pass-2 gate is BYTE-BLIND.**
Pass 2 is a **capture / instrument gate, not a performance claim**, so no gate leg reads bytes.
⛔ **The byte ledger is still reported on every arm including the launder** (K5 stands), and ⛔ **no
performance number is quoted at the pass-1 ratio** — `clu_total_bytes` **360 960** vs
`knn_launder_bytes` **288** = **1 253×**. Arm B changes this structurally (bytes move from the atom
store into head parameters) and its ledger remains its sharpest column.

### Q4 — ✅ RULED: **arm B may pass the capture gate in a configuration that is permanently
claim-barred, and both halves are reported.**
The private-well configuration is **laundered by construction** — the intervention doc's own
degenerate endpoint (§8 prohibition 2), reached faster. Registered expectation **P6 = 0.85** stands.
**K8 is confirmed as machine-checkable, not a promise:** `wells_per_item` and `vocabulary_shared`
declared and **pytest-asserted**, and a **`NO_TIER_II_CLAIM`** label that **travels with every number
from the arm**. The **shared-vocabulary interface is specified this wave; the factored store is NOT
built** and must not be foreclosed.

### Process ruling attached (Head, same session)
⭐ **The ≤3 engineer-worktree cap is LIFTED for this overnight pass only.** All three pass-2 engineer
spokes spawn in parallel alongside the concurrently-live `pilot-ttt-nan-and-d5-wiring` spoke (4
engineer worktrees). ⚠ **The cap returns to ≤3 at the end of this pass** — it is a one-pass
exception, not a standing change, and it is recorded here so it is not inherited by a later wave as
precedent.

---

## §2 — 2026-08-06, filed by `c2w8p2-instruments-and-debt` (wt3) BEFORE the criterion is measured on anything: **K9 — the merge criterion, RE-REGISTERED**

**Appended, not a rewrite of §1.** This block registers rider 2's replacement for pass 1's merge
criterion. It is filed **before** the predicate exists in code and **before** any refusal rate is
measured. ⛔ **No merge verb is built by this block or by the spoke that filed it** — the criterion is
a *predicate plus its refusal proof*, and merge stays unbuilt until the capture gate passes
(`PREREG-C2W8-PASS2.md` §6).

### 2.1 What is being retired, and exactly why (Hub-verified from the frozen `census.json`)

Pass 1's criterion (`well_lifecycle.mergeable_pairs`, which stays **read-only and unchanged**) admits
a pair iff `payload_dist <= payload_tol (= 0.1)` **and** `center_sep <= R_cert = 2 s_max + kappa' sigma_q`.

| seed | key spacing `median_nn_task1` | `R_cert` | `R_cert` / spacing | admitted | payload_dist values present |
|---|---|---|---|---|---|
| 0 | 0.140714 | 1.540231 | **10.95×** | 28 / 120 | **{0.0}** |
| 1 | 0.137540 | 1.417728 | **10.31×** | 29 / 120 | **{0.0}** |
| 2 | 0.146832 | 1.648195 | **11.23×** | 29 / 120 | **{0.0}** |

⇒ the geometric leg was **~10× the address resolution ⇒ it refused nothing**, and the payload leg
tested an **absolute** tolerance against a population whose payload distances are **identically 0 by
construction** (a class-incremental stream repeats classes). Monitor #3 `vacuous_gate` tripped
**3/3 at refusal rate 0.000**. `M` therefore measured *"the stream contains two items of the same
class"*. ⛔ **Shipping it would ship "collapse to one well per class" wearing a certificate costume.**

### 2.2 The re-registered criterion `M'` (`chlu.core.soft_certificate.merge_admissible`)

A live-well pair `(i, j)` is **admissible iff BOTH legs are APPLICABLE and both hold**:

* **geometry leg** — `center_sep <= r_merge`, with
  **`r_merge = rho_geom * key_spacing`, `rho_geom = 1.0` (REGISTERED)** and `key_spacing` the
  **measured** per-seed `geometry.median_nn_task1`, never a constant.
  *Content:* two centres may only be certified as one well when they are indistinguishable **at the
  address resolution the store actually has**. Commensurability is `r_merge / key_spacing = rho_geom`
  **by construction** (1.00× vs pass 1's 10.31–11.23×).
  **INAPPLICABLE ⇒ REFUSE** when `key_spacing` is missing/non-finite/`<= 0` (an unmeasured ruler
  never certifies).
* **payload leg** — `payload_dist <= tau_payload * payload_scale`, with
  **`tau_payload = 0.25` (REGISTERED)** and `payload_scale` the **measured spread of the payload
  channel over the whole live-well pair population** (median pairwise `‖a_i − a_j‖`), *not* an
  absolute constant and *not* measured on the admitted subset (that would be circular).
  ⭐ **The anti-vacuity clause:** if `payload_scale <= 1e-9` the payload channel carries **no
  discriminative content at all**, the leg is **INAPPLICABLE ⇒ REFUSE** — it must never pass by
  virtue of every payload being identical, which is exactly how pass 1 became vacuous.

Every evaluation returns its per-leg verdict and a `refused_on` list, so a refusal always names the
leg that refused it. A population-level report adds `refusal_rate` and
`vacuous_gate_would_trip = refusal_rate in {0.0, 1.0}` — **monitor #3's own f ∈ {0,1} convention,
applied to this criterion in BOTH directions** (a criterion that refuses everything is as
uninformative as one that refuses nothing, and is reported as such rather than celebrated).

### 2.3 Registered predictions (before the predicate exists; falsifiable)

1. **R1 — the frozen census is now entirely refused on geometry.** Every pair pass 1 admitted has
   `center_sep / key_spacing` **≥ 2.06 / 1.49 / 1.16** (seed 0/1/2 minima, from `census.json`), all
   **> rho_geom = 1.0** ⇒ **28 / 29 / 29 → 0 admitted**, refusal rate **1.000**, refused **on the
   geometry leg**. (⚠ This is a re-scoring of banked pairs, **not a re-run census**: the 92/91/91
   pairs pass 1 refused are not in the artifact, so no full-population `M'` is claimed.)
2. **R2 — `vacuous_gate` still trips on the frozen census, in the OPPOSITE direction** (f = 1.000,
   not 0.000). Registered *as expected*, not as a failure: pass 1 measured a store with essentially
   no basins (`capture_radius` 0.000 on 47/48), and **there is nothing there to merge**. The
   criterion's discriminating power is therefore proven by **designed positives and negatives**, not
   by this census.
3. **R3 — the smallest `rho_geom` that would admit ANY banked pair is 2.06 / 1.49 / 1.16** per seed.
   Registered so that any future loosening of `rho_geom` is visibly a *decision*, not a default.

### 2.4 The designed proof of refusal (pytest-asserted, `tests/test_merge_criterion.py`)

Both legs get a negative, and a positive proves the criterion is not merely refusing everything:

| # | pair | expected |
|---|---|---|
| **N-geom** | `center_sep = 0.30` (2.13× spacing), `payload_dist = 0.0` | **REFUSED on geometry** — ⭐ and pass 1's rule **ADMITS** the same pair (0.30 ≤ 1.54, 0.0 ≤ 0.1); the negative discriminates the two criteria |
| **N-pay** | `center_sep = 0.05`, `payload_dist = 0.9` at `payload_scale = 1.0` | **REFUSED on payload** |
| **N-degen** | `center_sep = 0.05`, `payload_dist = 0.0`, **`payload_scale = 0.0`** | **REFUSED — payload leg INAPPLICABLE** (the anti-vacuity clause; this is the frozen census's own configuration) |
| **P-pos** | `center_sep = 0.05`, `payload_dist = 0.10` at `payload_scale = 1.0` | **ADMITTED** |

⛔ **Declared NOT-RUN (never a null):** no fresh census, no merge verb, no `M'` on a re-run store, no
performance number of any kind. The criterion is *registered and proven able to refuse*; it is not
yet *used*.

*Filed by `c2w8p2-instruments-and-debt`, 2026-08-06, before the predicate was written.*

---

## §5 — 2026-08-07, filed by `c2w8p2-compact-atoms` (ARM A, wt1) BEFORE the cells it governs: the pilot grid is extended by one dial value, and why

> ⚠ **HUB RENUMBERING AT INTEGRATION (2026-08-07): this block was filed as `§2` and collided with
> wt3's earlier `§2` (the K9 merge criterion).** Append-only discipline was honoured on both sides —
> nothing was overwritten — so the Hub renumbered **this, the LATER block**, per wt3's own suggested
> resolution: wt3's `§2` is cited by hash-stable references inside **tracked code**
> (`soft_certificate.py` docstring, `tests/test_merge_criterion.py` docstring, commit `c13f953`).
> ⚠ Chronology, so the ordering is not misread: this block was filed **before** wt1's `§3` and `§4`,
> which cite each other and are therefore left untouched.


My own `PREREG.md` §5 declared the arm-A pilot grid as `atom_width_frac_spacing ∈ {0.5, 1.0}` on
**pilot seeds 7/8, disjoint from the census seeds 0/1/2**. Both pilot cells have now run (seed 7
only — **declared seed cut**, per the task's "cut seeds before cutting a cell"), and they diagnose a
mechanism the prereg did not anticipate. This block extends the grid to **`{0.5, 1.0, 1.5, 2.0}`**
and registers the prediction **before** the two new cells run. ⛔ No gate leg, no P1–P6 and no
acceptance criterion is changed; the census seeds are still run once, at the selected configuration.

**What the two pilot cells measured (seed 7, `median_nn_task1 = 0.1081`):**

| frac | atom `s` | support `R = 2.5 s` | `capture_radius > 0` | `capture >= σ_q` | own (med) | foreign (med) | `site_drift = 0` wells | `decode` |
|---|---|---|---|---|---|---|---|---|
| pass 1 (Gaussian, s = 0.3) | 0.300 | ∞ | **1/48 over 3 seeds** | — | 0.518/0.282/0.123 | 1.261/0.947/0.611 | 0 | 0.0625 = chance |
| 0.5 | 0.0541 | 0.135 | **16/16** | 0/16 | 0.193 | **1.3e-10** | 1/16 | 0.0625 |
| 1.0 | 0.1081 | 0.270 | **16/16** | **16/16** | 0.358 | **0.0020** | 6/16 | 0.0938 (+1.46 SE) |

**The diagnosis (this is the new mechanism).** `site_drift` is **bimodal**: 6 wells sit at *exactly*
0.0 with `λ_min ≈ 42–50`, and the other 10 sit at 0.45–1.18 with `λ_min = 0.1000 = 2α` — i.e. **the
bare confinement bowl, no well at all**. The split is not erosion: the census launches
`_relaxed_sites` from the address with the **payload channel at 0**, while the well sits at payload
`a_i = (label − 4.5)/9 ∈ [−0.5, +0.5]`. With a **compact** atom the force at the launch point is
**exactly zero** whenever `|a_i| > R`, so the read cannot feel its own well at all. At frac = 1.0,
`R = 0.270`, and exactly the items with `|a_i| ≲ R` are the six that hold their site. **G-CAP passes
(the well exists and captures a point placed at it) while G-DRIFT/G-DEC fail (the read's launch
manifold cannot reach it)** — the saddle-reach condition `|a_i| < a_U`, made absolute by compact
support.

**Registered prediction, before the cells run:**
* **frac = 1.5** (`R = 0.405 < max|a_i| = 0.5`): drift-0 wells **12–14 / 16**; `decode` **0.13–0.22**;
  foreign (med) **0.005–0.05**.
* **frac = 2.0** (`R = 0.540 > max|a_i| = 0.5`, i.e. **every** payload inside the support):
  drift-0 wells **≥ 14 / 16** ⇒ **G-DRIFT passes**; `decode` **≥ 0.15** ⇒ G-DEC passes;
  foreign (med) rises to **0.02–0.20** but stays **below** own. P(G-DRIFT passes at 2.0) = **0.7**.
* If frac = 2.0 also fails G-DRIFT, the arm's honest conclusion is that **reach alone does not close
  the payload-launch gap**, and that is the reported negative — not a further sweep.

⚠ Selection stays on **G-CAP/G-DEC/G-DRIFT on pilot seed 7 only**; ⛔ nothing is selected on
own/foreign (diagnostic, never a target).

## §3 — 2026-08-07, filed by `c2w8p2-compact-atoms` (ARM A, wt1) BEFORE the two ablation cells run: separating the arm's two levers

Arm A as run bundles **two** levers — the **compact kernel** (the sanctioned reach lever) and the
**site-local atom init at admission** (the companion lever, without which the compact kernel is
*exactly* dead: measured own depth 0.000, 0/512 own atoms in support, `ERRATA §2`/M1). The census
therefore cannot, by itself, say which lever did the work. Two ablation cells are registered here,
both on **pilot seed 7 only** (disjoint from the census seeds; **declared seed cut**), both at the
selected `atom_width_frac_spacing = 1.5`:

* **AB1 — Gaussian + site-local init** (`atom_kernel = "gaussian"`, `site_local_init = True`): is the
  *reach bound* doing the work, or only the *init*?
* **AB2 — compact kernel, no site-local init** (`wendland`, `site_local_init = False`): M1 re-checked
  at the selected width.

**Registered predictions, before the cells run:**
* **AB1.** The atoms are in the right place, so wells get dug and `capture_radius > 0` on a majority
  (**P = 0.8**); but at `s = 0.21` a *Gaussian* atom at a neighbouring site (spacing ≈ 0.108–0.147)
  still carries `exp(−0.14²/2·0.21²) ≈ 0.80` of its depth, so **foreign contribution returns**:
  predicted `foreign_median ≥ 0.5 × own_median` and **foreign > own on ≥ 8 of 16 wells** (arm A
  measured **0–3 of 16**). `decode` predicted **below arm A's 0.141–0.164**, band **0.06–0.14**, and
  P(AB1 clears G-DEC at 2 SE) = **0.35**.
* **AB2.** Nearest own atom ≈ 1.09 vs support `R = 2.5 s = 0.53` ⇒ **dead**: fitted depth **0.000**,
  `capture_radius` **0 on ≥ 15/16**, `decode` at chance. P = **0.9**.

⛔ These are ablations of the mechanism, not extra arms: the gate verdict stands on the three census
seeds already run at the selected configuration and does not move whatever AB1/AB2 say.

## §4 — 2026-08-07, filed by `c2w8p2-compact-atoms` (ARM A, wt1) BEFORE the cell it governs: AB1 promoted to census strength, because it FALSIFIED my §3 prediction

`ERRATA §3`'s ablations have run on pilot seed 7. **AB2 confirmed** (compact atoms without the
site-local init are exactly dead: depth 0.000, `capture_radius` 0/16, `λ_min = 0.1 = 2α` on all 16,
`decode` at chance — predicted P = 0.9). ⚠ **AB1 FALSIFIED my prediction, in the direction that
matters:** a **Gaussian** kernel at the same co-scaled width `s = 1.5 × spacing` **with** the
site-local init clears **all three gate legs** on seed 7 and is *better* than the compact arm on
G-DEC — `decode` **0.2344 (+8.03 SE)** vs the compact arm's **0.1797 (+5.48 SE)** on the same seed —
with a *lower* foreign contribution (median **0.0296** vs **0.0529**), where I had registered
`foreign ≥ 0.5 × own`, `foreign > own on ≥ 8/16` and `decode` **below** arm A's.

⇒ **The reach thesis survives; the attribution does not.** What closed the gate is the **co-scaled
width plus the site-local init** — both of which bound reach — and **not** the exactly-zero support.
Compactness is, at this operating point, **not necessary, and costs a little `decode`** (plausibly
because an exactly-zero force outside `R` also removes the pull on a read launched off the well).

Because this changes which lever the wave should carry forward, AB1 is promoted from a one-seed
ablation to the **same three census seeds (0/1/2) at the same selected width**, run once.
⛔ It is an **ablation of arm A's own mechanism, not a new arm**; the arm-A gate verdict already
recorded on seeds 0/1/2 does not move whatever AB1 says.

**Registered predictions, before the cell runs:** G-CAP majority-positive on **3/3** (P = 0.9);
`decode` **0.18–0.30** clearing 2 SE on **3/3** (P = 0.8); median `site_drift` **< 0.02** on 3/3
(P = 0.85); `foreign_median` **0.02–0.15** and foreign > own on **≤ 4 of 48**.
