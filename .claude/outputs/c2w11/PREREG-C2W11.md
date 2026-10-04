# PREREG — C2W11, THE COMPOSITIONAL WAVE

**Filed by the C2W11 research-lead Hub, 2026-08-10, BEFORE any spoke is spawned and before any cell
runs.** Binding design basis: charter **ADDENDUM 12 (§A33 north star + the MECHANICS/VALUE rule;
§A34 the ten-item ratified package)**, **ADDENDUM 11 (§A31–§A32)**, **§A13** (claim architecture v3),
**§A20.3** (the read-fix design inputs), and **`PREREG-TierII.md`** (the tier-ii metric, the cat-test
construction, the organizer swap, F1–F5 — still binding wherever this document does not amend it).

⛔ **Nothing in this document is a result.** Every number below is a *pre-registered prediction*, a
*threshold*, or a *quotation of a banked measurement with its provenance*.

---

## ⭐ AMENDMENT 1 (2026-08-10, same day, **BEFORE ANY CELL HAS RUN**) — the Advisor's five rulings

**Filed openly rather than silently: this document was written, sent for adjudication, and amended
before a single cell existed.** ⚠ The standing rule *"a revised pre-registration stops being one"*
governs preregs whose cells have run (the `PREREG-Bprime.md` precedent, deliberately not edited).
**Zero C2W11 cells have run.** Every change below is recorded here with its authority so the diff is
auditable; nothing is quietly overwritten.

| # | change | authority |
|---|---|---|
| 1 | **The substrate base moves to `main @ 2e1cdb2` or later** (§2.1) — C2W8 close is merged and its four repairs bind this wave | Advisor ruling 1, Hub-verified on disk |
| 2 | **V3 is re-formed as the SWAP-DIFFERENCED curve** (§5) — the shape claim demotes to MECHANICS | Advisor ruling 2 |
| 3 | **K8, the `K < N_a` structural cell**, added at the headline configuration only (§4) | Advisor ruling 3 |
| 4 | **M8, the curvature-spectrum check**, attached to loss term (c) (§6) | Advisor ruling 4(iii) |
| 5 | **Q1 and Q3 revised; Q11 added** (the false-positive prior); **the V1 interpretation asymmetry registered** (§8) | Advisor ruling 5 |

### ⛔⛔ 2.1 — THE SUBSTRATE BASE, AND THE TRAP INSIDE IT
**C2W8 close is merged: `main @ 2e1cdb2`, `gate_hardening_done = true`, all twelve items true**
(Hub-verified by reading the artifact's *content*, not its existence — this program has been bitten by
a gate firing on a file that merely exists). ⛔ **Every C2W11 engineer spoke bases on `2e1cdb2` or
later, NEVER on pass 3's `9e0bb25`.** Four repairs bind:
- **(a) A3 is DIAGNOSTIC by construction**, removed from the pass condition in census *code*. §A33.1 is
  now enforced by the instrument, not only by doctrine.
- **(b) `d_safe` AND the G-ADDR cue are sized on the STORE population.** ⭐⭐ **AND `70b11ae` declares
  the arm-facing consequence, which this Hub reads as a live trap for our own spine:** every arm
  store-config factory recovers the spacing as **`d_safe / d_safe_frac`**, so **an arm's CO-SCALED
  ATOM WIDTH now co-scales to the store population's spacing too.** The store population's spacing is
  **~3× the sizing set's** (0.445 vs 0.141, MNIST) ⇒ **`atom_width_frac_spacing = 1.5` NO LONGER MEANS
  WHAT IT MEANT WHEN IT WAS SELECTED.** ⛔ **The banked value is therefore NOT inherited. Spoke A
  RE-SELECTS the width against the store population and DECLARES its selection**, and the census's
  refuse-at-unselected-width guard (repair (d)) enforces that it cannot drift. ⚠ `d_safe_population =
  "sizing"` reproduces banked cells bit-exactly and is used **only** for reproduction, never for a
  claim cell.
- **(c) `covered` / `n_never_read` are SPLIT.** ⛔ **Use `settle_covered` and the settle-side telemetry
  for every addressability statement.** `launch_covered` is store-invariant by construction and is
  what produced the Advisor's own retracted erratum (§A31.1); it is retained only because monitor
  `settle_argmin` needs the launch-side U for Prop D1.
- **(d) The census REFUSES a non-selected width**, loudly.

---

# 0. THE WAVE'S QUESTION (§A33.2, verbatim scope)

> Does physics-trained organization produce a latent space that **GENERALIZES** to unseen feature
> combinations, reports **CALIBRATED** confidence including on novel features, and is **NAVIGABLE**
> at reasonable read compute — versus **matched-capacity non-physics organizers** (gradient placement
> / VQ / Titans-style write; same φ, same bytes, same capacity, same launch protocol, same reader
> class; **F3-grade tuning on the nulls**)?

**The only VALUE control is the ORGANIZER SWAP.** The settle-deleted / matched-bytes launder is an
**inherited tier-i diagnostic** and is reported as such, never as evidence. **Table-like inference
reads are explicitly PERMITTED on both arms** (`PREREG-TierII.md` §0); post-training reduction to a
table is a computational win (§A14.1).

---

# 1. ⭐⭐ THE MECHANICS/VALUE PARTITION (§A33.1 — governs every leg in this wave)

| | MECHANICS | VALUE |
|---|---|---|
| question | does the mechanism work? | is it worth anything? |
| level | component | **wave level only** |
| control | designed negatives (pytest-asserted) | **the organizer swap** |
| launder margins | ⛔ **DIAGNOSTIC ONLY — never a pass condition** | ⛔ same |

⛔ **NO launder margin is a pass condition anywhere in this wave.** Pass 3's A3 leg was the last of
its kind (§A33.1). The launch-only launder, the settle-deleted launder and the kNN-in-φ launder are
all reported, all labelled DIAGNOSTIC, and none of them can fail a gate.
⛔ **Per-feature G-ADDR is MECHANICS-ONLY and permanently barred from VALUE duty** (§A34.8).
⛔ **Every MECHANICS leg ships with a designed negative that is pytest-asserted.** A leg that cannot
fail on the degenerate configuration does not ship (the defect class caught three times in C2W8:
pass-1's vacuous `M`, pass-2's blind gate, pass-3's D2a-rewarding drift flag).

---

# 2. THE SUBSTRATE (carried, all pytest-pinned — §A34.10)

| element | value / source | status |
|---|---|---|
| **the placing write** | atoms are **placed**, not dragged by a 300-step gradient write (§A29.4(ii): a placing write gives foreign>own on **0/48** vs the gradient write's **45/48**) | carried, mandatory |
| **co-scaled widths** | `atom_width_frac_spacing = 1.5`, co-scaled to each seed's **measured** spacing (⚠ the shipped default **0.5** does NOT clear the pass-2 gate — the census must **refuse** at a non-selected width, C2W8-close item vi.5) | carried, mandatory |
| **`atom_site_local_init`** | compliant (R3 `attractor_can_move_off_the_key = true`, follow-fraction ≈1.008 at δ=0.30) | carried |
| **(d, atom budget) = ONE joint dial** | `n_atoms = round(512·√2^d)`; **d ≤ 12** (d=12 ⇔ 32 768 atoms). ⛔ **d = 16 is a declared NOT-RUN** — store measured INERT at a fully honoured 131 072-atom budget; the constraint is **REACH, not capacity** | binding ceiling |
| **address block ≠ task features** | the A31.4 inversion: task-strong φ is the address-**worst** arm beyond 2 SE (simclr−randconv A1 = −0.1406 ± 0.0508, 0/3); unfitted `randconv` buys the geometry for free. **The address block is its own head; cheap conv-class address geometry is a legitimate default** | binding design rule |
| **soft-certificate band** | `d/s ∈ [2.5, 2.9]` on **measured** `s` (estimator must subtract `α‖q‖²` — 1.44× inflation otherwise); ⛔ never ≥ 4.0 (provably zero dividend, D = 0.0000), never ≤ 2.01 (merger). `B ≥ 0.542` (`bprime-c6`'s re-located edge) | binding operating point |
| **P1 partition** | `erosion_partition=True` available; ⛔ **any change that reds K1's exact-zero probe or K2's fingerprint test UN-SHIPS it** | carried |
| **instrument suite** | census · G-ADDR · K9 · U telemetry · trash region · the promotion/trash pair | carried, **read-only this wave** |
| **2α coercivity floor** | `α = 0.05`, `2α = 0.1000`; `τ_max = Γ/2α`. `λ_min > 0` **does not certify a nonempty basin** (measured: wells with `λ_min > 0` and capture radius exactly 0.000) — **capture is the discriminating leg, λ_min is not** | carried caveat |

---

# 3. THE FAMILY AND THE READ (§A34.1 — feature-factored launches)

**The store:** `N_a` **SHARED feature wells** (the vocabulary). Each item is an `F`-subset of wells.
Each well carries `a` atoms and a payload `v_j ∈ R^m` **existing only in the store**. Sharing factor
`S = K·F/N_a`. ⛔ **Wells are never named semantically** — §2.6's claim form is copied verbatim into
every task file and every artifact.

**The launch (the wave's structural change, and the reason C2W5's cap may not bind):**
> **ONE PARTICLE PER SEMANTIC FEATURE CHANNEL of φ.** `k` is **structured by the encoder's
> decomposition, not free.** ⛔ **Binding is the READ + ψ's job** — the latent space may be
> disjoint/independent per feature; a set-level **DeepSets** ψ pools the landed particles into the
> downstream answer, with a **likelihood weighted by captured-vs-scattered particles**.
> ⛔ **No binding structure is built this wave.** Co-activation / wormhole edges remain the C2W9
> pointer.

⚠ **Why this is not C2W5 again, stated so it can be checked rather than believed.** C2W5 launched
`P = 4` particles from **one set-code** at fixed designed offsets; they occupied **2.20 of the 4
required** distinct wells, `≥ F` distinct wells were reachable on **5.0 %** of queries, and exact-set
occupancy was **0.0000 / 2 560**. That is a **launch-geometry cap that existed before any store was
written**. Feature-factored launches make the particles structurally distinct by construction rather
than by offset noise. **K0 (§4) measures exactly this, first, in seconds, with no store.**

⭐ **ψ ruling (Hub, mechanical):** **DeepSets pooled ψ only.** `AttentionPsi` is **QUARANTINED for
trajectory input** (C2W2 reconciliation 1, live in `chlu/core/psi_readout.py::AttentionPsiLeakError`);
the pooled DeepSets ψ is explicitly **not** quarantined. ⛔ **Any attention-ψ number is a declared
NOT-RUN this wave.**

---

# 4. ⛔⛔ THE KILL-CONDITIONS — BUILT AND RUN **BEFORE** THE THING THEY CAN KILL

**Standing doctrine (C2W3, adopted):** *build the kill-condition before the thing it can kill.* All of
K0–K7 run in **spoke A**, before the physics organizer and the null arms are funded at all. Every one
is **MECHANICS**. Spoke A's whole purpose is to be able to kill the wave cheaply.

| id | check | bar | if it fails | label |
|---|---|---|---|---|
| ⭐ **K0** | **launch expressivity, computed from launch geometry with NO store** (`orgdiv-null-arms` §12.2's proposed K0, now registered): fraction of unseen queries for which the feature-factored launch set can reach `≥ F` **distinct** feature wells; and mean distinct wells reachable | **≥ 0.80** distinct-`F` fraction; mean distinct ≥ `F − 0.5` | ⛔ **THE WAVE STOPS AT SPOKE A.** The read protocol is capped before any organizer exists — C2W5's death, reproduced. Report as a **structural cap**, not a physics null | MECHANICS |
| **K1** | write admissibility per `a` under the **placing write** at co-scaled width: endpoint write loss ≤ **0.05** · `λ_min > 0` at ≥ 90 % of wells · **SC-6 capture radius ≥ σ_q at ≥ 90 % of wells** | all three | the cell **ABSTAINS** (route-3 precedent); coverage reported first-class. If no affordable `a` passes ⇒ **F4 fires** | MECHANICS |
| **K2** | rule-4 assertions, **both halves**: set half `\|A ∩ B\| ≤ F−2` ∀ stored `B`; payload half `min_B ‖y(A) − y(B)‖ ≥ tol` | **100 %** of held-out queries | the split is **rejected and rebuilt** (⚠ the payload half is unsatisfiable at `m = 1` — measured 0.5 %; **`m ≥ 8` is required**, C2W5 deviation D1) | MECHANICS |
| **K3** | nearest-item table + the strongest **+0 B** substitute on the raw item table | ≤ **0.60** of metric range | ⛔ the family is **protocol-invalid** (FB4 killed 3 of 4 families this way) | MECHANICS |
| ⭐⭐ **K4** | the four leak controls — **run against the FULL trained read path including ψ at full capacity and the novelty head, with the store blanked**: (1) blank store · (2) query-only reader · (3) permuted payloads · (4) address-leak probe | all ≤ **chance + 0.05** | ⛔ **FAMILY VOID** | MECHANICS |
| **K5** | the **per-item table launder** (`K`-row table keyed by nearest stored item) through the **same** reader class | the read must beat it by **> 0.10** on ≥ 1 reader | ⛔ the read is table-expressible. ⚠ **Read C2W5's lesson before quoting a K5 failure:** it failed there **vacuously** — read 0.0000 *and* table 0.0000. **A K5 failure with every arm at ≈ 0 is a "not expressible at all" finding, not a "table-expressible" finding, and must be labelled so** | MECHANICS |
| ⭐ **K6** | ⭐⭐ **THE FIFTH-SESSION SLIP, NOW OWNED (spoke A).** §A28.3: the fraction of queries whose **asserted set is already exactly right** — computable in one line **BEFORE any reader is fitted**. It is the precondition that scopes the reader-fitting pathology | **reported**, and it gates interpretation: the pathology destroys signal only in proportion to this fraction (C2W5's cells: 2/2560 · 3/1280 · 0/2560; C2W7's: ~18 %) | not a kill — a **mandatory reported precondition**; a fitted-reader `0` is quotable only beside the **zero-parameter** member's score on the same latent (§A26.3) | MECHANICS |
| ⭐ **K7-CAP** | ⛔ **THE SP-1 LINEAR-CODE ESCAPE.** `y(x) = Σ_{j∈A} v_j` is a **linear code in `1_A` with `N_a·m` parameters**: an OLS fit on the true indicator scores **1.0000** exact-set with `‖v̂−v‖∞ = 4.25e-15` **on a blank store**. Assert: every reader in the class has `< N_a·m` params (measured C2W5: 104 / 72 / 0 / 92 vs bound 256) **and** the arm's own ψ + novelty head do not leak — the latter is **K4 leg 2 run at full ψ capacity**, which is why K4 is re-specified above | reader params `< N_a·m` **asserted**; K4 ≤ chance + 0.05 with full ψ | ⛔ wave-invalidating; the ψ budget is set by the **measured** leak, not chosen | MECHANICS |

### ⭐ K8 — THE STRUCTURAL CELL (`K < N_a`), added by Amendment 1 · **MECHANICS** · headline configuration ONLY

> ⭐ **A measured guard tells you the leak is small AT THIS OPERATING POINT. A structural
> impossibility tells you it CANNOT HAPPEN.**

**ψ-does-the-work is the single most likely FALSE-POSITIVE mode of this wave, and a false positive
here IS the tier-ii headline.** K7-CAP + K4-at-full-ψ are the *measured* guard and they stay primary
(measured-guard-not-fiat is correct doctrine: capping ψ by fiat would be choosing a design point to
make a measurement legible — intervention **Error 2**). **K8 adds the structural kill beside it.**

- **Construction:** one confirmatory cell at the **headline configuration**, re-drawn with
  **`K < N_a`** (e.g. `N_a = 32, F = 4, K = 24`). At `K < N_a` the `1_A ↦ y` design matrix is
  **rank-deficient**, so the SP-1 linear-code probe **provably cannot recover `v`** — verified at
  C2W5's `K = 12 < N_a = 16` fixture, where the probe reproduces `y` **without** recovering the
  payloads.
- **Registered reading:** the physics arm's V1 verdict at K8 must **agree in sign** with the headline
  cell. ⛔ **A V1 clear at `K > N_a` that does NOT survive K8 is reported as a ψ-capacity artifact,
  not as a tier-ii result.**
- **Scope, deliberately bounded:** ⛔ **ONE cell, headline configuration, NOT across the grid.** This
  bounds the statistical-power cost (fewer written items) while putting the structural kill exactly
  where a false positive would do damage.
- **Precedent this instantiates:** the program has been saved by structural kills three times —
  §A9.5's per-slot table launder killed a stage-2 build *before it existed*; the reader-audit's
  reproduction gates; K1's bitwise zero.

⭐ **Reader class (frozen before the first arm runs, `PREREG-TierII.md` §8.3 + §A26.3):** nearest-well
table · kNN · linear · small MLP · **plus a mandatory ZERO-PARAMETER member**. Identical
architectures, identical fitting budget, fitted on the **SEEN split only**, on **both** arms, params
ledgered on both. ⚠ Selection never touches `Q_unseen`; the seen-validation split must inherit the
family's **own** rule-4 held-out rule (`orgdiv-null-arms` §6 — otherwise selection runs on an easier
problem than the one being scored).

---

# 5. ⭐⭐ THE THREE VALUE LEGS (§A33.2 — wave level, organizer-swapped, ALL of them)

`OD(R) ≡ score(R ∘ z_phys) − score(R ∘ z_null*)` on `Q_unseen`; **`OD_min ≡ min_R OD(R)`**.
`null* = max over ALL non-physics arms AND their entire registered tuning grid` — **computed, not
estimated** (C2W5 computed it over 584 configs × 5 seeds; the standard is now permanent).
⛔ **F3-GRADE TUNING ON THE NULLS:** ≥ 5 lr × 3 capacity × 3 seeds per arm, held-out-from-seen
selection; **the physics arm gets the same budget, no more.** A hobbled null is the same referee
attack in mirror image (and C2W5's physics arm ran at **ONE** configuration — that is why its numbers
carry "physics, untuned and collapsed" and may not carry a headline).

### V1 — GENERALIZATION (leg i)
**Metric:** held-out exact-set accuracy on **rule-4-valid unseen combinations**, 5 seeds, `ddof = 1`,
`SE = sd/√5`. **Clears iff** `OD_min − 2 SE > +0.05` **and** `OD(R) > 0` for **≥ 3 of the 4** readers.
**Fires iff** `mean + 2 SE < +0.05`. **`|OD_min| ≤ 0.05` ⇒ TIE**, reported as a finding.
*(Inherited verbatim from `PREREG-TierII.md` §3.1 F1 — unamended.)*

### V2 — CALIBRATION, the graded-novelty read (leg ii) — ⭐ the Head-flagged novelty piece
**Construction (§A34.4):** per-particle capture / depth / residual diagnostics composed per feature +
**overlap-as-confidence** (§A20.3(c)); trained via **feature-dropout-as-pseudo-novelty**.
**Eval:** unseen queries with **0 / 1 / 2 novel feature channels** (a novel channel is a feature whose
well was never written).
**Two statistics, both registered:**
- **V2a — per-feature novelty AUROC** (known vs novel feature channel).
  **Floor (MECHANICS-adjacent, must hold before the swap is read):** `AUROC_phys > 0.60`. Below
  **0.55** the channel does not exist and V2 is a null **regardless of the swap**.
  **Swap:** clears iff `AUROC_phys − AUROC_null* > +0.05` beyond 2 SE, 5 seeds.
- **V2b — set-level answer ECE** on unseen combinations, physics vs `null*`, lower is better.
**The null arms MUST emit a matched confidence** — VQ: distance-to-codebook; N1: read-objective
residual; N5: the surprise gate. ⛔ An arm without a confidence channel is **not** scored as
"uncalibrated"; it is a declared NOT-RUN for V2.
⛔ **Designed negative (pytest-asserted):** a store with **permuted payloads** must give
`AUROC ≈ 0.5`. A novelty channel that cannot report chance on a scrambled store does not ship.

### V3 — NAVIGABILITY (leg iii) — ⭐ **RE-FORMED BY AMENDMENT 1: the SWAP-DIFFERENCED curve**

⚠ **The Hub's first form collapsed two different comparisons. They are now separated, and only the
second is the VALUE leg:**

| comparison | who has a curve | verdict |
|---|---|---|
| **(a)** CLU vs the static nulls **as readers** | only CLU | ⛔ **§8.3-barred as a primary claim** — the competition is absent by construction. Not the leg |
| ⭐ **(b)** physics-organized store vs **matched non-physics-organized store**, **BOTH read with the IDENTICAL k-particle anytime read** | **both** | ✅ **THIS IS THE VALUE LEG.** Not §8.3-barred: the competition is **present by construction**, which is exactly what the swap is for |

> ### ⭐ V3-PRIMARY (VALUE) — the swap-differenced curve
> **At matched read compute, does the physics-organized store NAVIGATE better than the matched
> non-physics organizer?** Score `OD_V3(b) ≡ s(physics, budget b) − s(null*_V3, budget b)` at every
> registered budget point, ≥ 6 points, 5 seeds. Clears at V1's bar sustained over the upper half of
> the budget axis. **The organizer swap hands the null arm the same reader class — it changes who
> decides where the wells go, not how they are read** (`PREREG-TierII.md` §1: identical φ, bytes,
> capacity, launch protocol **and reader class**; §A13: table-like inference reads permitted **on both
> arms**).

> ### V3-MECHANICS (demoted from primary by Amendment 1, and it still carries real information)
> The physics curve is **monotone and non-flat** — a navigation dial exists. ⭐ **N199 measured a FLAT
> curve when the store carried nothing:** *a memory that carries nothing cannot be read better by
> reading it longer.* **Non-flat is therefore positive evidence that something readable is in there**,
> and **flat is a MECHANICS FAILURE**, independently checkable and independently diagnosable.

> ### V3-REPORTED (the "and also" position — mandatory in every table, NEVER primary)
> **Compute-to-parity against EXTERNAL baselines.** ⚠ Banked reference, and **label it as what it is —
> a READ-COMPUTE RATIO**, not a wall-clock or training cost: the C2W5 physics read cost **3 360× N1's
> matched-capacity static read (6.88e7 vs 20 480 mult-adds) at a TIE** (`orgdiv-null-arms` §7).
> ⛔ **Absolute compute-matched parity as the primary is REJECTED** — it would make a 3 360× ratio the
> headline of a wave whose question is **organization, not efficiency**.

### ⛔ V3's null side — WHICH ARMS ADMIT A LANDSCAPE (registered by the Hub, because a spoke must not discover this mid-run)
V3(b) requires the null arm's organization to be **instantiable as a landscape** and read by the
identical k-particle anytime read. That is true of some arms and false of others:
- ✅ **N1** — *identical store parameterisation by construction*; it is the **headline** V3 null and
  the cleanest possible swap.
- ✅ **N2 (VQ)** and **N3 (static-geometric)** — codebook / fitted `(c, σ, b)` + payloads instantiate
  as well centres.
- ⛔ **N4 (kNN over raw rows)** and **N5 (Titans fast weights)** — **no landscape exists**; they are
  **declared NOT-RUN for V3** and their static scores are reported as **flat reference lines**.
  ⛔ They are never scored as "un-navigable" — that would be scoring an arm for not being the object
  under test.
⇒ **`null*_V3 = max over {N1, N2, N3}`**, computed over their registered grids.
⚠⚠ **And the mirror-image referee attack is registered in advance:** instantiating N2/N3's codebook as
a landscape requires choosing atom amplitudes, widths and counts. ⛔ **Those must be matched to the
physics arm's atom budget and F3-tuned like every other null hyperparameter.** *"We gave the null a
badly-instantiated landscape"* is the same attack as *"you hobbled the competition"*, and it is
closed by measurement, not by assertion.

---

# 6. THE MECHANICS LEGS (component, pass/fail, designed negatives mandatory)

| id | leg | bar | designed negative (pytest-asserted) |
|---|---|---|---|
| **M1** | **K0** (above) | ≥ 0.80 | a launch set collapsed to one channel must score ≈ chance |
| **M2** | **K1** write admissibility at the placing write | §4 | a store written at a non-selected width must **refuse to run** |
| **M3** | ⭐ **per-feature G-ADDR** (§A34.8, **MECHANICS-ONLY, barred from VALUE**): feature `f`'s particle resolves to feature `f`'s well **AND** lands inside that well's **measured** SC-6 capture radius | `≥ max(4·chance, chance + 2 SE)`; **report `margin_in_SE` beside every boolean** (pass 3: `randconv` failed by ONE read on 2/3 seeds) | (a) planted permutation (same store, wrong declared targets) must score 0 while `any_basin` stays 1.0000; (b) **narrow-wells** rig must fail. ⛔ **`any_basin` is reported and is NOT the leg** |
| **M4** | **sharing / refresh** (§A34.9(b)): a re-encountered feature **deepens the existing well**; K9 gates merges | monotone non-decreasing depth on rewrite, ≥ 90 % of events | a store that spawns a **private well per item** must FAIL the leg |
| **M5** | **anti-collapse**: direct **wells-visited `W/N_a`**, TWO-SIDED labels (§A26.4 — the `[8,16]` `S_eff` band is **RETIRED**; "COLLAPSED" is reserved for **concentration**, under-usage is labelled as under-usage) + the **marginal** well-usage monitor (M15; per-query concentration is confidence and is **never** penalized) | declared per run; a run outside the band is reported **COLLAPSED, not null** | M15 must trip on its two banked designed negatives (4.00 / 1.00) |
| ⭐ **M6** | ⛔ **DIAGNOSTIC (not a pass condition, §A33.1) — THE FOURTH-CONVERGENT-DATUM TEST.** Occupancy precision of the **raw launch geometry** vs **after the settle**. Banked C2W5: **0.4061 → 0.2967, dividend −0.1094**; distinct wells **2.20 → 1.70** | reported with sign and 2 SE | — (it is a diagnostic and cannot fail a gate) |
| **M7** | ⭐ **the curvature-shape term** (§A34.9(c)): does a within-well **soft direction survive superposition**? ⚠ §A4.2 **REFUTED** the tilt instantiation on a learned store (tilt monotonically *reduces* `λ_min`, +0.099 → −8.28; a written site's vacuum residual 0.140–0.343 vs a 0.167 random baseline). **This is measured, never assumed** | reported two-sided; the shipped confinement floors the soft mode at `2α`, so `τ_max = Γ/2α` travels with any lifetime statement | a store with the term's coefficient at **0** must be **bit-identical** to the shipped objective |
| ⭐ **M8** | ⛔⛔ **THE CURVATURE-SPECTRUM CHECK — term (c)'s CONSUMER** (added by Amendment 1). **At END of training**: the `λ_min` distribution and the **participation ratio** at written sites, physics arm vs the coefficient-zero arm | reported two-sided with the `2α = 0.1000` floor marked on the axis; a soft direction is a `λ` **at or near the floor that is not the floor itself** (⚠ banked trap: undug wells report `λ_min ≈ 0.0993` **because** `2α` is what `λ_min` reports when nothing was written) | a store trained with the coefficient at **0** must show **no excess** soft directions over the shipped objective |

> ⭐⭐ **Why M8 is mandatory and is not optional polish (Advisor ruling 4(iii) — and the reasoning is
> sharper than the Hub's original).** Term (c) **CREATES** the soft directions a future kinetics wave
> would exploit, and **w20's law says undefended designed structure gets ERASED**. Landing (c) now
> means C2W12 inherits **surviving** soft directions instead of fighting erasure. ⛔ **But (c) has no
> other consumer this wave** — so without M8 an undefended (c) could be **silently tuned to inertness
> and nobody would know**. M8 is the §A4.2 tilt lesson applied **prospectively**: the last time this
> program shipped a designed degeneracy, it did not survive superposition and that was measured only
> afterwards.

---

# 7. ⭐ THE C2W9 EMISSION TRIGGER (mandatory, mechanical, a FILE — never prose)

The reach monitor is **split** (§A21 C2W9 row, ruled):
- **COVERAGE failure** — a needed feature well lies **outside the union of the `k` launch diamonds**.
  *(a launch-head problem)* → **spoke A owns this statistic** (it is computable from launch geometry
  alone, alongside K0).
- **TRAVERSAL failure** — **in-flight evidence points outside the current particle's diamond**.
  *(the wormhole trigger)* → **spoke B owns this statistic** (it needs trajectories).

> ⛔ **If either fires, the owning spoke WRITES:
> `.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md`**
> carrying the **measured** signature (which mode, the fraction of queries, the per-feature
> breakdown, the reach radii, and the seeds). **Spoke A writes it if coverage fires; spoke B appends
> its own dated section if traversal fires.** It is **C2W9's spawn trigger and must be a file.**
> ⛔ If neither fires, the file is **NOT** created and both spokes state that explicitly in their
> reports — an absent trigger is a measurement, not an omission.

⚠ Banked scoping input, and it does **not** itself fire the trigger: the `d ≥ 16` inertness was
measured at a **fully honoured** atom budget ⇒ **the store was not capacity-starved; the binding
constraint is REACH** ⇒ the fix is learned `p₀` / wormholes (C2W9), not more atoms.

---

# 8. ⭐ THE HUB'S PRE-REGISTERED PREDICTIONS (numbers committed, per the w14 rule)

| # | quantity | **prediction** | reasoning |
|---|---|---|---|
| ⭐ **Q1** | **K0** distinct-`F` fraction under feature-factored launches | ⚠ **MOVED BY AMENDMENT 1: 0.65 → 0.78** that it clears 0.80 · point **0.88**, band [0.55, 0.98] | **Re-derived, as instructed, against the repaired sizing.** My 0.65 was set against **pass-3-era cue sizing**, which used the ~200-key sizing set. Sized on the **STORE population** the measured `σ_q/spacing` is **0.19–0.37**, not the ~1.07 the wrong-population number implied — queries sit ~3× closer to their own key than to a neighbour, so **the cue is materially easier than my prior assumed**. ⭐ **And K0 is a LAUNCH-geometry statistic measured with NO store**, so it is *not* contaminated by the open §A31.2 settle mystery. ⇒ **the geometric half of the risk is discharged; the entire residual risk is FEATURE-CHANNEL COLLINEARITY** (two channels of φ encoding to the same launch region), which is unmeasured and is precisely what K0 exists to measure. I move it deliberately now rather than explain a beaten prior at review |
| **Q2** | **K1** passes at `a ≤ 12` under the placing write (C2W5 needed `a = 32` with the gradient write) | **0.70** | the placing write removes the atom-dragging that manufactured foreign-atom domination (45/48 → 0/48) |
| ⭐ **Q3** | **M6**: the settle's occupancy dividend turns **non-negative** | ⚠ **MOVED BY AMENDMENT 1: 0.35 → 0.50** · point **+0.00**, band [−0.08, +0.08] | **Advisor challenge accepted, with the mechanism decomposed so the prediction stays interpretable either way.** C2W7's G1 was significantly negative with a **measured** mechanism — unstructured launches **plus broken capture (1 basin in 48)** — and **both are now fixed and measured** (46/48 basins; correct-basin 0.50). ⭐ The capture repair bears **directly** on M6: C2W5's settle scattered particles because attractors sat ~1.4 `sep` from the designed anchors; **with real basins at the sites the settle should move particles toward their wells rather than merging them** (2.20 → 1.70 was the merging signature). ⚠ **My one retained reservation, registered so it is not raised post-hoc:** feature-factored launches do **NOT** dissolve §A25.2's by-construction launder ceiling — **the launder receives the SAME launches**. What they dissolve is the **occupancy cap** and the **capture failure**. So I move to 0.50 on the capture mechanism, not on query-conditionality |
| **Q4** | **V1** `OD_min` clears +0.05 beyond 2 SE | **0.20** — **HELD** (inherits `PREREG-TierII.md` §3.6 **unchanged**: the substrate repairs are *addressability* repairs, not *composition* repairs) · `P(TIE) ≈ 0.55` · `P(physics loses by > 0.05) ≈ 0.25` | ⭐⭐ **THE INTERPRETATION ASYMMETRY, REGISTERED NOW SO IT CANNOT BE A POST-HOC UPGRADE AT REVIEW:** if V1 **clears** on a substrate whose repairs were **ADDRESSABILITY-ONLY**, that is **STRONGER** evidence than clearing after composition-targeted repairs — it would mean **composition was latent and mechanically blocked all along**. ⛔ The converse is registered with equal force: a clear after composition-targeted tuning would have been weaker, and no such tuning is funded this wave |
| **Q5** | **V2a** floor: `AUROC_phys > 0.60` | **0.55** · point **0.64**, band [0.50, 0.82] | capture radius, `λ_min` and drift are **measured, live** per-well diagnostics; the composition into a per-feature novelty score is untested |
| **Q6** | **V2a** clears the swap by > 0.05 beyond 2 SE | **0.30** | VQ's distance-to-codebook is a strong confidence baseline and costs nothing |
| **Q7** | **V3-primary(a)**: the anytime curve is monotone and non-flat | **0.60** | C2W5 measured the store **"POOR, not INERT"** (0.000 at `tol`, **0.45–0.58 at 4×tol**) — there is something for a budget dial to resolve |
| **Q8** | **V3-reported**: compute-to-parity ratio **< 100×** | **0.15** | banked 3 360× at tie |
| **Q9** | the **C2W9 trigger** fires (coverage **or** traversal) | **0.55** | reach is the named binding constraint at `d ≥ 16`; at `d = 12` it is open |
| **Q10** | **M7/M8**: a within-well soft direction survives superposition **and is visible in the end-of-training curvature spectrum** | **0.25** | §A4.2's refutation was on a *tilt* instantiation; a curvature-shape **defender** is a different object, but the `2α` floor is the ceiling either way |
| ⭐⭐ **Q11** | ⛔ **THE FALSE-POSITIVE PRIOR — `P(ψ does the work \| V1 clears)`** (added by Amendment 1) | **0.15** with **BOTH** K4-at-full-ψ **and** the K8 structural cell · **0.40** on the **measured guard alone** | **The Advisor is right that I had ten priors and not one of them was the false-positive prior — which this program's epistemics normally demand.** ⭐ The 0.15-vs-0.40 gap **is independently the argument for buying K8** (ruling 3): the structural cell more than halves the probability that a tier-ii headline is a ψ-capacity artifact. ⭐ Registering it now is what makes the K4 result **interpretable rather than merely reassuring** — a passing K4 moves this number, and by how much is fixed in advance |

### ⭐⭐ RUN ORDER, SHARPENED (Advisor, endorsed and binding)
> **K0 and M6 are the CHEAPEST KILL SIGNAL IN THE WAVE. Run them FIRST and report them BEFORE
> anything else.** K0 needs no store and costs seconds; M6 needs only a written store and the launch
> geometry.
> ⭐ **And the null outcome is a RESULT, not a wasted wave:** if the structural caps did **not** move
> despite **three measured substrate changes** (the placing write · co-scaled widths ·
> feature-factored launches), that is the **FIFTH convergent datum on write-side organization — this
> time with the substrate repairs CONTROLLED FOR**, which is strictly more than the previous four
> could say. It is reported as such and it is a paper section, not a discard.

⛔ **Every one of these is scored in the wave's closing report, right or wrong** (the C2W5 scorecard's
shape — *systematic over-prediction of the store's capability at every point where a number was put on
it* — is itself a banked finding, and this Hub is not exempt from it).

---

# 9. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)

1. **Two-scale hierarchical placement** (§A34.2) — priced, not built (§ Hub report ablation pricing).
2. **The three-state lifecycle** (§A34.3) — priced, not built; routed to C2W10, which owns I2's
   re-registration (the registered I2 rule is **self-capping**: `P(confirm | true ρ = +0.5)` = 0.424
   at 6×3 wells and only 0.468 at 100×3 — it must be re-registered as a **test-against-0** first).
3. **Kinetics-as-selector** (§A34.5) — the **head** is not built. ⭐ Only the **loss term (c)** rides
   (M7), so that the flat floors kinetics would need are defended rather than assumed.
4. **`d = 16`** — measured inert at an honoured budget; a declared NOT-RUN, not a null.
5. **Attention-ψ** — quarantined for trajectory input (C2W2 reconciliation 1); DeepSets only.
6. **Any inference-read claim** — the per-item slotted read is table-expressible and that axis is
   CLOSED (§A14.1). OQ-2 (learned `p₀` steering) remains the only pre-registered revival trigger,
   wave-boundary only.
7. **Wormholes / learned `p₀` as reach fixes** — C2W9 territory; this wave only **emits the trigger**.
8. **The trajectory write term** (`lambda_traj > 0`) — C2W2 measured it a **monotone cost**; not
   deployed. ⚠ It is registered as available *tooling* (§A14.1) and its non-deployment is a choice,
   stated.
9. ⭐ **N4 (kNN) and N5 (Titans) for VALUE leg V3** — **no landscape exists to read with the identical
   k-particle anytime read.** Declared NOT-RUN for V3; their static scores are reported as **flat
   reference lines**. ⛔ They are never scored as "un-navigable" — that would be scoring an arm for not
   being the object under test.
10. **`d_safe_population = "sizing"`** — used **only** to reproduce banked cells bit-exactly, and
    **never** for a claim cell. Any reproduction run is labelled as a reproduction.

---

# 10. ⭐ LEG-LABEL INDEX (§A33.1 compliance, checkable at a glance)

| leg | label | control |
|---|---|---|
| **K0–K8** (all kill-conditions) | **MECHANICS** | designed negatives, pytest-asserted |
| **M1, M2, M3, M4, M5, M7, M8** | **MECHANICS** | designed negatives, pytest-asserted |
| **M6** (occupancy dividend) · every launder margin · every byte ratio · V3-REPORTED compute-to-parity | ⛔ **DIAGNOSTIC** | **none — cannot fail any gate** |
| **V3-MECHANICS** (monotone + non-flat curve) | **MECHANICS** | N199's flat-curve reference |
| **V1** (generalization) · **V2a/V2b** (calibration) · **V3-PRIMARY** (swap-differenced curve) | ⭐ **VALUE** | ⭐ **the ORGANIZER SWAP, and nothing else** |
| **V2a's `> 0.60` floor** | **MECHANICS** (a precondition on V2's VALUE reading — below 0.55 the channel does not exist and V2 is a null **regardless of the swap**) | designed negative: permuted payloads ⇒ AUROC ≈ 0.5 |

⛔ **Per-feature G-ADDR (M3) is MECHANICS-ONLY and permanently barred from VALUE duty** (§A34.8).
⛔ **No launder margin is a pass condition anywhere in this wave.**

---

# 11. STANDING DISCIPLINE ON EVERY SPOKE (copied into every task file)

multi-seed before any number (**5 seeds** on claim cells, 3 on instrument cells, stated per cell) ·
**byte ledgers on every arm including launders and φ/projection params** · identical frozen φ,
byte-compared, on every arm · declared NOT-RUNs **never** nulls · **wells never named semantically**
(§2.6's sentence verbatim) · **depth is NOT feature importance** (§A23.5 ACTIVE) · **N94** epoch
discipline on every reading · every γ statement is **read-budget-scoped** · quote the **curve**, not
the endpoint · reconciliation list in the **first 10 lines** · **"CLU-former" stays a placeholder** ·
⛔ **never push `origin`; `clu-dev` only, and the Hub integrates.**

⛔ **No paper number. No full-CLU verdict.** The verdict surface is the **three VALUE legs**, and it is
**Advisor-adjudicated**.

*Filed by the C2W11 Hub, 2026-08-10, before any spoke was spawned.*
