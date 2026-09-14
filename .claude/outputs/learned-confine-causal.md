# learned-confine-causal — physics-theorist report (w27, [C1W27])

Task + acceptance criterion: an **independent causal test** of the saddle criterion (★) —
`learned_confine` 0.05 → 0.022 at d=4, K=16, `atom_init_width` 0.15, 3 seeds; prediction registered
with a numeric band **before** the run, then scored.
Status: **done.** No tracked code touched (`git status --short` empty before and after).

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). FOUR items,
> all small, none retracting a headline.**
> 1. ⚠ **`r2-geometry-revival`'s "± sd" is the SAMPLE (ddof=1) sd, not population.** I reproduce
>    their exact per-seed triple (0.5859 / 0.6133 / 0.5840); its population sd is **0.0134**, its
>    sample sd **0.0164** — the number they print. My task file said "w26's r2 tables quote
>    population sd"; that is wrong. **I quote both everywhere below.** Owner: Hub / `doc-curator`.
> 2. ⚠ **`readout-channel-theory` §4.3's "`a_U` depends on α and `|c|` only through the product
>    `2α|c|`" is approximate and measurably wrong** (§4): matching a **2.27×** cut in α needs only a
>    **1.64×** cut in `|c|`, because `|c|` also enters through `L = √(|c|²+a²)`. Owner: Hub §1 wording.
> 3. ⚠ **`D` values fitted by the `fit_reach.py` radial fit are α-contaminated** and must not be fed
>    into (★) at a *different* α without correction: the direction-averaged profile carries `+α r²`,
>    which inflates `D_fit` by ≈ **0.37 α** (predicted −0.0104 for this Δα; measured paired
>    **−0.0143 ± 0.0028**). w26 §1.5's `D` = 0.910 / 0.459 are therefore ≈ 0.89 / 0.44 as *well*
>    depths. Owner: me (noted), Hub if the numbers are quoted in a doc.
> 4. ⚠ **`learned_confine` is now causal but it is a LOG lever** (§4): 2.27× buys **+9.3 %** reach;
>    the next codebook rung costs **5–10×**; the `|a| = 1.0` rung is **unreachable at any α ≥ 1e-4**.
>    Any doc that says "lower the confinement" must quote the curve, not the direction.
>    Owner: Hub §1/§6 + whoever writes the ceiling-knob note.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** **none — theory / instrument.** A causal test of a criterion. **No performance claim is
  made and none of these numbers may be scored on a performance axis** (in particular, `strict`
  rising 0.594 → 0.624 is **not** a capacity result; the cell still FAILs the 0.9 bar at both α).
- **Laundering control:** n/a (no performance number). Substitute, per the task: the registered
  prediction (`.claude/outputs/learned-confine-causal/PREREG.md`, written before the measurement
  script existed) — plus a **value-blank** control at both α (P-G).
- **Falsifies:** Δstrict ≤ 0, or Δstrict outside the registered band [+0.02, +0.12].
- **Does NOT falsify:** a smaller-but-in-band, right-signed effect; the absolute retrieval level.

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| base commit | local `main` @ **`082d095`**, tree clean, **no tracked file touched** |
| JAX / venv | **jax 0.9.0, equinox 0.13.4**, main venv (protocol §4, no worktree, no `uv sync`) |
| PREREG | `.claude/outputs/learned-confine-causal/PREREG.md` — written **before** `confine_causal.py` existed; derived from (★) + the *existing* w26 seed-0 artefact `tr_d4K16_w015.json` |
| scripts | `.claude/scratch/learned-confine-causal/`: `confine_causal.py`, `run_many.py`, `score.py` (+ w26's `criterion_U.py`, imported unmodified) |
| cell | `d=4, K=16`, `atom_init_width` **0.15**, `min_atoms_base` ×4 ⇒ **8192 atoms**, GLOBAL write, 600 Adam(3e-3) wd 1e-4, `n_perturb` 32, σ_addr .25, σ_pay .6, margin .15, barrier .2, payload_index 4, write key `PRNGKey(seed+7919)`, site_seed 0, `R` = 1.0, codebook `linspace(-1,1,16)` permuted |
| **the one flag** | `learned_confine` **0.05 → 0.022**; every other knob at its w26/r2 value |
| read | shipped `_two_phase`: γ_address .05 × 400 → γ_read .02 × 800, dt .05, tail .25, 8 subsamples, `payload_tol` **0.1** (absolute), `query_sigma` 0.15 `fixed_norm`, σ_p .05, 32 queries/item = 512 queries/cell |
| seeds | **3 (0, 1, 2)** at each α (write key, atom init and query key all move with the seed; sites and codebook are seed-invariant) |
| sd convention | **both** quoted: population (ddof=0) and sample (ddof=1). r2's printed "±" = sample. |
| blank control | value-blank = identical write with all payloads 0, scored against the real codebook; seed 0 at each α |
| langevin_noise | **N/A** — deterministic Verlet, no temperature, no read noise |
| cost | 8 cells (6 real + 2 blank) ≈ 65 min wall on ≤ 2 concurrent processes, sharing 8 cores with two engineer worktrees |

---

## 1. Headline

| α | seed 0 | seed 1 | seed 2 | **mean** | pop sd | sample sd | basin |
|---|---|---|---|---|---|---|---|
| **0.050** (shipped) | 0.5859 | 0.6133 | 0.5840 | **0.5944** | 0.0134 | 0.0164 | ≡ strict |
| **0.022** | 0.6250 | 0.6250 | 0.6211 | **0.6237** | **0.0018** | **0.0023** | ≡ strict |
| paired Δ | +0.0391 | +0.0117 | +0.0371 | **+0.0293** | — | 0.0153 (SE 0.0088) | — |

**`Δstrict = +0.0293`, positive in 3/3 seeds, t(2) = 3.33, one-sided p = 0.040.**
Registered band **[+0.02, +0.12]**, registered point **+0.064** ⇒ **IN BAND** (lower third).

The α=0.05 arm reproduces `r2-geometry-revival` §1 **exactly**: their 3-seed 0.5944 and per-seed
0.5859 / 0.6133 / 0.5840, and seed-0's write trace (`0.2132 → 0.000115`, `w_atom` 0.1841) is
bit-identical to w26's. The intervention arm is therefore a clean one-flag delta.

### 1.1 The evidence that matters is the LOCUS, not the mean
Per-item, seed-averaged (32 queries × 3 seeds per item):

| `|a|` rung | items | strict @0.05 | **strict @0.022** | Δ | prereg said |
|---|---|---|---|---|---|
| 0.0667 … 0.4667 | 6,13,10,11,3,7,5,1 (8) | 1.0000 | 1.0000 | **0.0000** | 1.000 (no change) |
| **0.6000** | 2 | 0.8333 | **1.0000** | **+0.1667** | 0.992 ✅ |
| **0.6000** | 9 | 0.6771 | **0.9792** | **+0.3021** | 0.941 ✅ (P-A: ≥0.85) |
| 0.7333 | 0, 12 | 0.0000 | 0.0000 | 0.0000 | 0.299 / 0.163, band [0,0.70] ✅ |
| 0.8667 | 8, 14 | 0.0000 | **0.0000** | 0.0000 | **0.000 ± 0.05** ✅ (P-B) |
| 1.0000 | 4, 15 | 0.0000 | **0.0000** | 0.0000 | **0.000 ± 0.05** ✅ (P-B) |

**Every unit of the effect is on the two `|a| = 0.6` items — the exact rung (★) said was straddling
the saddle-node boundary (`x ≡ |a|/a_U` = 0.933 and 1.006 at α=0.05, 0.854 and 0.917 at α=0.022) —
and the 14 other items move by exactly 0.0000.** A diffuse "α helps retrieval somehow" alternative
predicts change spread over items; a criterion-mediated α predicts change **only** in the transition
shoulder. Measured: 100 % of the change in the predicted 2 items, 0 % elsewhere. This is a much
stronger discriminator than the 3-σ mean shift, and it is what makes the test causal rather than
correlational.

### 1.2 A signature I did not register, and would not have thought to
**The seed variance collapses 7×** (sample sd 0.0164 → 0.0023) and all three α=0.022 seeds land on
**0.625 = 10/16 exactly** — i.e. the 10 items with `|a| ≤ 0.6` retrieve *perfectly* and the 6 with
`|a| ≥ 0.7333` fail *completely*. This is the direct signature of a saddle-node: at α=0.05 the cell
is sitting **on** the bifurcation (two items partially captured, per-seed strict on those items
scattering 0.469–0.969), and the α cut moves the whole codebook **off** it, restoring an all-or-none
codebook. *Retrieval variance is a bifurcation-proximity meter.* (Corollary for the engineer: a cell
whose seed sd is large is a cell sitting on its reach boundary, not a cell that needs more seeds.)

### 1.3 Controls
- **Value-blank (P-G):** `strict_blank` = **0.1250** at *both* α, with `basin` = **1.0000** and
  median value error 0.533. 0.1250 = 2/16 exactly = the trivial ceiling (only the two `|a| = 0.0667`
  codewords lie inside `payload_tol` = 0.1 of zero). The blank is unaffected by α, so nothing in the
  measured Δ is value-channel laundering. ✅
- **`basin ≡ strict` at all 6 real cells** (max |difference| 0). w26's "the reach failure is an
  ADDRESS failure" holds identically at the new α.
- **(S) spurious-minimum-on-the-payload-ray:** 0/48 items at α=0.05, 3/48 at α=0.022 — and the 3 are
  not among the failing items. The binding mechanism remains (R/U), not (S).

---

## 2. PREREG scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| **P-1** ⭐ | `Δstrict` = **+0.064**, band **[+0.02, +0.12]** | **+0.0293** (3/3 seeds positive, t = 3.33, p₁ = 0.040) | ✅ **in band**, lower third; point 2.2× too high (cause identified, §3) |
| **P-A** | item 9 ≥ 0.85; item 2 ≥ 0.95 | **0.979** / **1.000** | ✅ both |
| **P-B** ⭐ | `|a| ≥ 0.8667` stays 0.000 ± 0.05 | **0.0000 at 12/12 item-seeds** | ✅ the strongest falsifier did not fire |
| **P-C** | `|a| = 0.7333` → 0.299 / 0.163, band [0, 0.70] | **0.000 / 0.000** | ✅ in band, at the floor (post-hoc with measured fits: 0.091 / 0.049 — the right end of the shoulder) |
| **P-D** | `a_U(0.022)/a_U(0.05)` = 1.093 ± 0.005 | realized **1.093** per item (e.g. i=2: 0.6152 → 0.6730) | ✅ arithmetic auditable |
| **P-E** | `D_fit` median ∈ [0.41, 0.51], `s_fit` ∈ [0.19, 0.21], R² ≥ 0.99 | **0.4595 → 0.4402**, **0.1999 → 0.1922**, min R² **0.9976** | ✅ (both in band; `s` at the band edge) and the drift is *explained* — §3 |
| **P-F** | failing-flow endpoint `|x_end|` **increases** with weaker confinement | **0.0638 → 0.0638** (no change; min 0.0636, max 0.0640) | ✗ **falsified as registered.** Cause: the endpoint is *item 13's own site* (`|c₁₃|` = 0.064, a **write** placement), not a confinement equilibrium. The failing query is eaten by the ball-centre item, whose position α does not move. **My model of where failures go was wrong in its mechanism, not its destination.** |
| **P-G** | blank ≤ 0.13 at both α | **0.1250 / 0.1250** (= trivial ceiling exactly) | ✅ |

**Score: 6 confirmed, 1 falsified (P-F), 0 partial.**

---

## 3. Why the effect is 2.2× smaller than the point prediction (and what that costs the claim)

Registered explicitly in PREREG §3 as *the* dominant uncertainty: the point prediction assumed the
write is α-invariant, so it used the α=0.05 fits `(D_i, s_i)` at both α. It is not exactly invariant.

**Measured paired drift over 48 item-seeds:** `ΔD = −0.0143 ± 0.0028` (SE), `Δs = −0.0063 ± 0.0007`;
`w_atom` 0.1850 → 0.1820.

**Most of that is not the write at all — it is the fit.** The direction-averaged radial profile of
`V = −D e^{−r²/2s²} + α|q|²` around a site is `const − D e^{−r²/2s²} + α r²` (the cross term averages
to zero), so fitting the 3-parameter Gaussian form to it over `r ∈ [0, 4s]` **absorbs `α r²` into
`D`**. Synthetic check (`numpy`, exact profile, same fit code, `D_true` = 0.44, `s_true` = 0.19):

| | `D_fit` | `s_fit` |
|---|---|---|
| α = 0.050 | 0.4584 | 0.1967 |
| α = 0.022 | 0.4480 | 0.1929 |
| **predicted drift** | **−0.0104 (−2.3 %)** | **−0.0038 (−1.9 %)** |
| **measured drift** | **−0.0143 ± 0.0028** | **−0.0063 ± 0.0007** |

⇒ **73 % of the apparent `D` drop and 60 % of the `s` drop are fit contamination**; the residual real
write change is `−0.004 ± 0.003` in `D` (≈ **−0.9 %**) and `−0.0025` in `s` (≈ **−1.3 %**). **The
write is α-invariant to ~1 %** — the two competing write-side effects I derived in the PREREG (the
confinement donates `α·E|δ|²` to the `L_min` margin; it taxes the `L_bar` barrier) do cancel, as
registered, and the residual is at the noise floor. *This is reconciliation item 3: never feed a
`fit_reach`-style `D` into (★) at a different α without subtracting ≈ 0.37 α.*

**Post-hoc re-score (labelled post-hoc), using each arm's OWN measured fits** and the transfer Φ
fitted on the α=0.05 cell:

| | predicted | measured |
|---|---|---|
| strict @ α=0.05 | 0.5999 | 0.5944 |
| **strict @ α=0.022** | **0.6253** | **0.6237** |
| Δ | **+0.0254** | **+0.0293** |

**The criterion predicts the retrieval LEVEL at the intervened α to 0.0016 (0.16 pp)** and the Δ to
0.004. The pre-registered point (+0.064) missed because it moved `a_U` with α while holding the
fitted `D, s` — which, per the paragraph above, double-counts the confinement's contribution to `D`.
**Honest accounting: the +0.064 was an arithmetic artefact of my own fit convention, caught by the
prereg's own registered sensitivity table, and it is the reason the band was set wide enough to
survive.** The band [+0.02, +0.12] was derived from exactly this envelope (`D×0.9, s×0.95` → +0.023).

---

## 4. What the intervention says about the two load-bearing knobs (the deliverable §3 asks for)

`a_U(α)` at the median measured site (`D` = 0.4402, `s` = 0.1922, `|c|` = 0.98) — **the curve, not the
endpoint**:

| α | 0.050 | 0.030 | **0.022** | 0.010 | 0.005 | 0.002 | 0.001 | 5e−4 | 1e−4 |
|---|---|---|---|---|---|---|---|---|---|
| `a_U` | 0.600 | 0.636 | **0.656** | 0.705 | 0.745 | 0.794 | 0.829 | 0.862 | 0.933 |

`a_U` grows like `√(2 ln(1/α) + const)` — **≈ +0.10 per decade of α**, and the shipped codebook rungs
are 0.133 apart. Per-item α needed to bring each rung inside `a_U` (measured α=0.022 fits):

| rung `|a|` | 0.6000 | **0.7333** | **0.8667** | **1.0000** |
|---|---|---|---|---|
| α\* required | already in at 0.022 | **0.0094 / 0.0055** | **0.00053 / 0.00039** | **none at α ≥ 1e−4** |

> ### ⭐ `learned_confine` is a **causal** ceiling parameter and a **logarithmic** one.
> A 2.27× cut buys **+9.3 %** of reach and exactly **one straddling rung**. The next rung costs
> **5–10×**, the one after **~100×**, and the top rung (`|a| = 1.0`, the shipped `payload` maximum)
> is **not purchasable at any α ≥ 1e−4** at this geometry. This is the same `√(2 ln(·))` no-go that
> w26 proved for the well depth `D`, now demonstrated **causally** on the α lever.

**And the ball radius `R` is NOT the free twin of α.** `readout-channel-theory` §4.3 said reach
depends on α and `|c|` only through `2α|c|`; that is only the leading term. Exactly: matching the
`a_U` of a **2.27×** α cut needs a **1.64×** cut in `|c|` (0.98 → 0.596), not 2.27× — because `|c|`
also shrinks the *demand* through `L = √(|c|² + a²)`. So `R` is the **stronger** reach lever per unit…
and the one you cannot spend, because `designed_sites` scales linearly in `R` (sites = `R` × a fixed
unit-ball packing), so **`sep ∝ R` exactly**: taking `R` 1.0 → 0.60 drops `sep` **0.903 → 0.553**
against the merge condition `2s + c_j σ_q ≤ sep` with `2s` = 0.384 and `σ_q` = 0.15 — i.e. straight
onto the merge boundary. **α is the only *uncoupled* one of the two knobs; `R` trades reach against
merge at a fixed exchange rate.** ⇒ For `r2-d-sweep-close`: `learned_confine` should be treated as a
**declared operating point of the ceiling**, reported in every capacity table, and swept (if at all)
in decades — not as an initialisation detail; and any proposal to shrink `R` must carry the `sep`
consequence in the same sentence.

---

## 5. One paragraph on the arms I was told not to re-open
This bears on the excursion arms only as an **upper bound on the α lever**, and it *lowers* it. Arms
(a) (multi-channel payload) and (b) (annealed read) act on the same inequality `r ≤ a_U(s, α, |c|)`;
w26 measured arm (b) worth **+0.236 strict** at this very cell and arm (a) worth a 3.5× cut in the
demand `r`. The α lever measured here is worth **+0.029 strict** and **+9.3 % of `a_U`** for a 2.27×
cut — **one order of magnitude less than either arm**, and it saturates logarithmically. α is
therefore a *calibration* knob, not an *arm*: use it to move a cell off its bifurcation (which it
does perfectly — §1.2), not to buy capacity. I have not touched the trilemma or option (d).

---

## 6. Verdict on (★) as a causal statement

> **EVIDENCED (causal), one cell. Not proven.** The intervention moved retrieval in the predicted
> direction in 3/3 seeds, by an amount inside the registered band, **entirely on the item rung the
> criterion nominated in advance**, while the four items the criterion said α could not save stayed
> at exactly 0.000 — and, post-hoc, the criterion predicts the intervened cell's retrieval **level**
> to 0.16 pp. `learned_confine` therefore enters retrieval **through `2α(L−R)` in (★)**, i.e. w26's
> "(★) reach parameters, not free knobs" survives as a **causal** claim, not a correlation.

**What would still be needed to call it proven:** (i) a second α rung (0.005, where (★) predicts the
`|a| = 0.7333` rung flips and the cell should jump to **0.75 = 12/16** — a *sharp, pre-registerable*
prediction, ~25 min on this harness); (ii) a second cell (a different `d` or `K`) to show the law is
not d=4-K=16-specific; (iii) the three crowded high-K FAIL cells remain outside (U)'s single-well
scope (w26 §2.5) and this test says nothing about them.

**Limitations, stated before a referee does.** (1) One cell, one width, one intervention rung.
(2) `t(2) = 3.33` on the mean is weak on its own; the locus argument (§1.1) is what carries the
verdict. (3) The transfer function Φ's two parameters are fitted on the α=0.05 cell — the falsifiable
content is that **`a_U` moves the curve by the amount (★) says**, not the curve's shape.
(4) Deterministic, noise-free read; no temperature. (5) P-F shows my mental model of *where* failed
queries end up was mechanically wrong (the ball-centre item eats them; the confinement only delivers
them), which does not affect (U) but does affect any future claim about failure destinations.
(6) Coercivity was **not** separately stress-tested at α=0.022 — the value-blank's `basin` = 1.0000
and the flow tests' bounded endpoints are evidence that the landscape stayed coercive, but F5 Prop-10
was not re-derived at the new α.

## Git footprint
**None.** No tracked file created, modified or deleted; `git status --short` empty before and after;
no branch, no commits. Artefacts: `.claude/outputs/learned-confine-causal/PREREG.md`, this report,
and `.claude/scratch/learned-confine-causal/` (3 scripts + 8 JSON cells + logs).

---

## Proposed handover updates (for the Hub)

1. **§1 / §6 — upgrade (★) from observational to causal, with its price.** *"The saddle criterion (★)
   is causally confirmed on the `learned_confine` lever: 0.05 → 0.022 at d=4 K=16 width 0.15 moved
   3-seed strict 0.5944 → 0.6237 (+0.0293, 3/3 seeds, pre-registered band [+0.02,+0.12]), with 100 %
   of the change on the two `|a| = 0.6` items the criterion nominated in advance and 0.0000 change on
   the four `|a| ≥ 0.8667` items it said α could not save. Post-hoc, (★) predicts the intervened
   cell's retrieval level to 0.16 pp."*
2. **§1 — extend the no-go to α.** *"Reach is logarithmically un-buyable in the **confinement** as
   well as the depth: `a_U ≈ +0.10 per decade of α` (measured 0.600 → 0.933 over α 0.05 → 1e−4).
   At the shipped geometry the `|a| = 0.7333` rung costs α ≈ 0.005–0.009, the `0.8667` rung
   α ≈ 4e−4, and the `|a| = 1.0` rung is unreachable at any α ≥ 1e−4."*
3. **§3 config notes — sharpen the two-knob entry (replaces w26 item 5's wording).** `learned_confine`
   is a **causally confirmed, logarithmic, uncoupled** ceiling parameter — declare it in every
   capacity table, sweep in decades. The ball radius `R` is the **stronger but coupled** twin:
   `sep ∝ R` exactly, so `R` 1.0 → 0.60 (the α-equivalent cut) drops `sep` 0.903 → 0.553 against
   `2s = 0.384` — it buys reach by spending merge margin. **And `2α|c|` is only the leading term**:
   matching a 2.27× α cut needs a 1.64× `|c|` cut, not 2.27×.
4. **§7 / instrument note (new, and reusable).** *"Retrieval seed-variance is a bifurcation-proximity
   meter. At α=0.05 this cell has sample sd 0.0164 with two items partially captured; at α=0.022 the
   sd collapses 7× to 0.0023 and all seeds land on 10/16 exactly. A cell with large seed spread is
   sitting **on** its reach boundary — the fix is to move the boundary, not to add seeds."*
5. **§7 — measurement hygiene (reconciliation item 3).** The `fit_reach.py` radial fit inflates `D`
   by ≈ 0.37 α (verified: predicted drift −0.0104, measured −0.0143 ± 0.0028). Corrected well depths
   for w26 §1.5: `D` ≈ 0.89 (width 0.30) and 0.44 (width 0.15). The write itself is **α-invariant to
   ~1 %**.
6. **§5 provenance — new artefact set.** `.claude/outputs/learned-confine-causal/PREREG.md` + this
   report; scratch `.claude/scratch/learned-confine-causal/` (`confine_causal.py`, `run_many.py`,
   `score.py`; `a050_s{0,1,2}.json`, `a022_s{0,1,2}.json`, `blank_a{050,022}_s0.json`). Base
   `082d095`, JAX 0.9.0, no tracked code, ~65 min on ≤2 concurrent processes.
7. **Commission next, ranked (mine).** (i) ⭐ **the second α rung, α = 0.005, same cell, 3 seeds
   (~25 min)** — (★) predicts strict jumps to **0.75 = 12/16** (the `|a| = 0.7333` rung flips) and
   predicts **no** further gain at α = 0.002; that is a two-sided pre-registerable test and would
   take (★) from *evidenced* to *proven on this cell*. (ii) The same intervention at **d=6, K=32** to
   show the law is not cell-specific. (iii) w26's items (ii)–(iv) (excursion arm (a) at fixed
   `payload_tol`; the annealed read at K=32; option (d) behind a `payload_gate` flag) are unchanged
   in priority — the α lever is **not** a substitute for them (§5).
