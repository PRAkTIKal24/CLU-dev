# WAVE 6 — THE PHYSICS WAVE. REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-20. One spoke + three hubs + one retry, FORGIS_14,
`base/iter6 = df025685` (pin + 44 additive files, 0 deletions, frozen harness byte-identical).**
Launch 06:25Z; all five finished by 09:03Z. Zero GPU. Record: `.claude/outputs/_thinkpad/w6/`.
⚠ Every number is quoted from an agent's own report. ⛔ **The Hub has re-derived no statistic.**

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| ⭐⭐ `spoke/shorts-inventory` | **the law wave 6 was built on does not exist** | — |
| ⭐⭐ `hub/potentials` | **the shorts' retention law PREDICTS the store, parameter-free** | 1/3 |
| ⛔ `hub/compose-physics` | **the dynamics are worse than decoration for composition** | 1/2 (3/3 banked) |
| ⛔ `hub/retention` | **a lifetime is strictly worse than instant eviction** | 1 (3/3 returned) |
| ⭐ `w6.1/compose-physics` (R-B) | **the Hub's own hypothesis was false by construction** | 0 |

**Budget after wave 6:** composition **2/3** (3/3 banked) · deletion **2/3** (3/3 banked) ·
potentials **1/3** · refusal spent · placement exhausted.

⭐⭐ **The wave's shape: the physics theory is VALIDATED as a description of the dynamics, and
REFUTED as a contribution to composition. Both, cleanly, on the same object.**

---

## 1. ⭐⭐ `hub/potentials` — the shorts' law is right, and it beats a sighted rival

> **The Curvature-Retention Relationship predicts the field store's settle half-life,
> parameter-free, to a factor 1.10 at the median held-out corner — and beats a one-parameter rival
> FITTED WITH SIGHT OF THE HELD-OUT MEASUREMENTS on 49 of 49 corners, on every one of three seeds.**

Nothing fitted: `n_half = ln2 / (−ln|λ_slow|)`, `ε = 0.05` inherited verbatim from the shorts.
⭐ **Validated against the papers' own printed numbers before any data was touched** — V2 §4.1 floor
**27.03** vs **27.02681**; V5 `λ_ret` **0.998999499** vs **0.9989994995**; `n_half` **692.5** vs
**692.4538**. The repo's `velocity_verlet_step` **is** the shorts' map, asserted to **1.885e-08**, and
the leg refuses to run otherwise.
⭐ **Instrument floor measured before the bar was fixed** (median 0.0349); B1's 0.10 bar is ~3× it.
**B1 0.0433 / 0.0435 / 0.0431 · B2 49/49 ×3 · B3, B5, B6 PASS ×3 · B4 MISS ×3.** Seed spread **0.0004**.

⭐ **The sharpest number (D17):** derived overdamped branch slope **+1.1385**; measured **+1.1368 /
+1.1367 / +1.1368** — mean absolute error 0.0038–0.0042 against the textbook `+1` at 0.1430–0.1436,
**34–38× closer on all three seeds.** The shorts derive the slope must *exceed* +1 at finite γ. It does.
⚠ **D18's margin is thin on every seed and is reported as thin, not as a win.**

## 2. ⭐⭐ The inventory spoke — a declared absence, established twice
⛔ **"Mexican-hat width → interpolation floor" DOES NOT EXIST in the corpus.** "interpolation" occurs
**zero** times in the three papers; "Mexican hat" **once**, in V2 App. M.4, as a float64 round-off
testbed with no width and no separation parameter; "centre-surround", "DoG", "RBF" and "lateral
inhibition" occur **zero** times. Established **independently twice** — by the spoke and by
`hub/potentials`' own exhaustive search.
⭐⭐ **And the nearest genuinely derived object inverts it:** the floor of the V2/V5 retention curve is
**curvature-INdependent**. Substituting one for the other would have reversed the papers' finding.
⭐ `hub/potentials` **refused both near-substitutes** (an undefined table cell; a measured constant) as
forbidden, and reports the absence as its first result. ⚠ *"The exact-decay law"* was likewise
ambiguous — three distinct exact statements, not interchangeable.

## 3. ⛔ `hub/compose-physics` — the dynamics do not earn their place
> **Worse than decoration: at every strength at which they actually move the state they COST
> composition, and the only setting at which they tie the incumbent is the setting at which they do
> nothing.**

Row 4, mean paired Δ vs `pca_tensor`, beside the displacement the dynamics actually produce:

| leg | displacement | seed 0 / 1 / 2 |
|---|---|---|
| `pca_settle_zerostep` | 0.0000 | **0.000 / 0.000 / 0.000** — bit-identical to the incumbent |
| `pca_settle_f05_conv` | 0.009 | −0.531 / −0.510 / −0.388 (inside margin, negative every seed) |
| `pca_settle_onestep` | 0.046 | −2.735 / −3.796 / −2.959 |
| **`pca_settle`** | **0.39** | **−39.959 / −46.918 / −39.061** |

Row 3: `pca_settle` −15.800 / −25.533 / −20.350. **Every pre-registered bar negative, as declared in
advance.** ⚠ And it loses **while paying 81,312 B more** read-side. ⛔ **More bytes, fewer tuples.**

## 4. ⛔ `hub/retention` — a lifetime is strictly worse than instant eviction
> **It pays the same composition cost, it pays most of it EARLY, and it delivers the refusal LATE.**

**Bar 1 — its own falsifier — PASSES on all three seeds:** at the last horizon before the items die,
decay's composition cost differs from instant eviction's by **+0.0562 / +0.0337 / +0.0112** against a
declared margin of 1.0. ⛔ **Its DIAL DECLARATION is falsified in its own words, and it pre-registered
that it expected exactly that.** **Bar 2 DOMINATED on all three seeds** — the composition cost is paid
strictly before any refusal arrives (`t_pay(50 %)` = 7.992 / 5.994 / 7.992).
⭐ Run 3/3 **returned unspent**; it took one scored run because the horizon grid, clock and refusal
horizon are all derived from the papers' law in closed form.

## 5. ⭐ `w6.1/compose-physics` — the retry killed the Hub's own hypothesis
The Hub scoped this retry around the hypothesis that settling *leaves the principal subspace*.
⛔⛔ **False, and false by construction.** The settled state's off-span fraction is at most **2e−7** —
the same float32 rounding the incumbent carries **standing still** — and the orthogonal fraction
predicts the loss **at chance (AUC 0.457–0.559)**. The settle is parameterised in span coordinates and
emits through the same renderer, so `span(G_k)` is an **invariant subspace** of it; even the
channel-space kernel descent is confined, because its force is built from the atoms and **the atoms
are the span**.
⭐⭐ **The real mechanism, and it is a resolution law:** what breaks the read is motion *inside* the
subspace. The read's answers form a **dense lattice whose neighbouring cells are ~5.5 % of the state
norm apart**, the judge's decision cells are that lattice's cells, and **the dynamics step across them.**
⭐ 0 scored runs; composition stays 2/3 with 3/3 banked.

---

## 6. What the Advisor is asked to rule on

1. ⭐⭐ **The physics splits cleanly in two, and the paper must say which half it is claiming.**
   **Validated:** the shorts' retention law predicts this store's dynamics parameter-free, beating a
   sighted fitted rival 49/49 on three seeds. **Refuted:** those same dynamics cost composition at
   every strength that moves the state. ⛔ Both are measured on the same object.
2. ⭐ **May the CRR result carry a section?** It is the programme's cleanest theory-meets-measurement
   result. ⚠ It predicts **settle half-life**, not a taxonomy row — so it supports no scoreboard claim.
3. ⛔ **The dynamics are now out of the composition story.** With wave 5's result, **composition is
   carried by unsupervised PCA content plus a static tensor-product read** — no settling, no labels in
   the content. **Is that the paper's object?**
4. ⚠ **Retention is closed:** a lifetime buys nothing eviction does not, and delivers refusal later.
5. ⛔⛔ **The absence finding should reach the writing.** A law the programme has been planning
   experiments against for at least two waves **is not in its own papers**, and the nearest real
   result inverts it. ⭐ Two independent searches agree.
6. ⭐ **The compose/forget gap is unchanged and is now the only thing left.** Wave 7 is premise-dead
   (two gyms failed adoption gates; a composition-blind policy scores 0.9367–1.0000). **There is no
   scoped wave 8.**

## 7. Limits
All numbers 3 seeds on generation 2 · the Hub re-derived nothing · D18 thin on every seed, declared ·
B4 missed on all three seeds · the CRR result is about dynamics, not about any scoreboard row ·
⚠ a stray `C:\c\chlu\...` tree exists from a path mistake and two hubs correctly left it alone.
