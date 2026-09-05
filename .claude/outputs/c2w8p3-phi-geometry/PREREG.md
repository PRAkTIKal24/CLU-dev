# PREREG — `c2w8p3-phi-geometry` (C2W8 pass 3, wt2)

**Filed 2026-08-09 by the experiment-engineer spoke, BEFORE the first Split-CIFAR-10
measuring cell.** Written after (a) a *synthetic* smoke of the harness on toy images
(`.claude/scratch/c2w8p3-phi-geometry/smoke.py` — no CIFAR, no encoder fitting, no
geometry number) and (b) a **timing** price re-used from
`c2w8-cifar-strong-phi/results/encoder_price.json`. ⛔ **No geometry quantity of any
kind has been measured on CIFAR-10 at the time of filing.**

Inherits: `PREREG-C2W8-PASS3.md` §3/§4/§5 + Q3, and `ERRATA-C2W8-PASS3.md` §1's four
Head rulings (R1 substrate · R2 map in scope + launder uses it · R3 the `d` rule · R4
NO-GO re-labels). ⛔ This file is **not edited** after filing.

Code under measurement: branch `c2w8p3-phi-geometry` @ `0d206b1`
(`chlu/experiments/exp_phi_geometry.py`, `phi_encoders.PhiProjection`).

---

## 0. DIAL DECLARATION (echoed)

- **Dial:** none — encoder/geometry **instrumentation**. No claim cell, no performance
  number, no verdict.
- **Laundering control:** N/A (nothing is scored) — **but the byte ledger is mandatory**:
  the map's parameters land on the ledger of every arm **including the launder**, and
  the launder demonstrably reads the **projected** φ (asserted in code, not intended).
- **Falsifies:** nothing this pass claims. A NO-GO **re-labels** the spine (R4).
- **N94:** every reading here is **non-promotable**.

---

## 1. The registered reading (verbatim, from `PREREG-C2W8-PASS3` §5)

> **GO** iff strong φ improves **σ_q / spacing** over the PCA reference **at the same
> `d`**, beyond noise, ≥ 3 seeds. **NO-GO** otherwise.

**Operationalised before measuring** (`geometry_verdict`, computed, never argued):
"improves" = **LOWERS** σ_q/spacing; paired per seed (the stream is bit-identical across
arms at a seed — fingerprinted); **GO iff paired mean improvement > 2·SE AND ≥ 3 seeds
positive**, at the primary key population **n = 16** and at the `d` the geometry
favours, where `d_favoured := argmin_d (strong arm's mean σ_q/spacing)`.

⚠ `d_favoured` is computed from **geometry alone** so that an inert store cannot leak
into the GO/NO-GO reason (R1). `d_recommended_operational` is reported **separately**.

---

## 2. The pricing that licenses the run (done BEFORE the prereg, timing only)

| item | number | source |
|---|---|---|
| `simclr` fit, `enc_steps = 8000`, `phi_dim = 256`, CIFAR fit pool ≈ 4966 | **0.226 s/step ⇒ 30.1 min/seed** | banked `encoder_price.json` (same config) |
| 3 seeds of `simclr` | **≈ 90 min** | above ×3 |
| `randconv` (unfitted control) | 2.9 s/seed | banked |
| `pca` reference SVDs (2 per `d`, cached per `(arm, phi_dim)`) | ≈ 15 s each | banked |
| **rider (a)** depth cell, `d = 16`, `n_atoms = 131 072`, 6 writes | **≈ 175 s** (banked: 87.1 s for 3 writes at `d=16`; the write cost is store-side and φ-independent) | `ERRATA-C2W8.md` §3 |
| `d = 12` / `d = 8` depth cells | ≈ 58 s / ≈ 24 s | same |

⇒ **rider (a) FITS at `d = 16` and will be RUN, not refused.** Total budget ≈ 1 h 45 m.
⛔ If the `d = 16` cell overruns 3× this price it is reported as a **declared NOT-RUN
with the arithmetic**, never as a truncated run.

**The joint dial, priced before any cell** (`min_atoms = round(512·√2^d)`):
`d = 8 → 8 192` · `d = 12 → 32 768` · `d = 16 → 131 072` · `d = 256 → 1.7e41`
(**forbidden**, not expensive) ⇒ the band is exactly `{8, 12, 16}`.

---

## 3. NUMERIC PREDICTIONS — committed before measuring

Substrate: Split-CIFAR-10 (R1), `task1_only`, seeds 0/1/2, `n_keys` primary **16**
(the pass-1/2 store item population), also reported at 64 / 200. Scale stated on every
row: `scale = 1/r95(‖φ_proj(fit_pool)‖)`, so the address ball has unit 95th-percentile
radius and every leg below is dimensionless.

| # | quantity | prediction (point, 80% band) | prior |
|---|---|---|---|
| **P1** | ⭐ **the §5 GO reading** — strong φ lowers σ_q/spacing vs PCA-φ at the same `d`, 3/3 seeds | **GO fires** | **0.70** |
| **P1b** | ratio σ_q/spacing (strong ÷ PCA reference) at `d = 8`, `n = 16` | **0.55** (0.30 – 0.95) | — |
| **P2** | absolute σ_q/spacing of the STRONG arm at the favoured `d`, `n = 16` | **> 0.30** — strong φ does **not** make the address space comfortable | **0.75** |
| **P2b** | σ_q/spacing bands, `d = 8`, `n = 16` | strong **0.3 – 1.2**; PCA reference **0.6 – 2.5** | — |
| **P3** | σ_q/spacing falls monotonically with `d` at fixed `n` ⇒ **`d_favoured` = 16** | **16** | **0.80** |
| **P3b** | strong arm σ_q/spacing(16) ÷ σ_q/spacing(8) | **0.65** (0.40 – 0.90) | — |
| **P4** | ⭐ **rider (a)** — the store is **INERT at `d = 16`** on CIFAR φ (median fitted depth < 1e-6) | **inert** | **0.80** |
| **P4b** | `d = 12` median fitted depth | in **(1e-6, 0.1)** — marginal | 0.50 |
| **P4c** | `d = 8` median fitted depth | in **[0.05, 0.6]** — digs | 0.80 |
| **P5** | ⭐ **rider (b)** — refusal rate under the **rig's own** `d_safe` pricing | **0.000** at every `d`, every arm | **0.85** |
| **P5b** | `d_safe_rig` ÷ population spacing (`n = 16`) | **0.35** (0.15 – 0.60) | — |
| **P5c** | refusal rate under the **re-priced** `d_safe` (= 0.88 × population median-NN) | **> 0**, band **0.05 – 0.35** | 0.75 |
| **P6** | the shipped **`truncate`** map vs the fitted `pca` map, strong arm, `d = 8` | **≈ equal**, ratio **0.85 – 1.30** | 0.60 |
| **P7** | map neutrality: `pca256→d` reproduces `pca@d` | σ_q/spacing within **1 %** | 0.90 |
| **P8** | the unfitted `gaussian` (JL) map vs the fitted `pca` map, strong arm | ratio **0.70 – 1.40**, **no directional call** | P(gaussian better) 0.45 |

### Derivations (so a hit is evidence and a miss is a finding)

**P1/P1b.** The normalisation fixes the *volume* (`r95 = 1`), so strong φ can only help
by **spreading items more uniformly inside the ball** (`PREREG-C2W8-PASS3` §4). PCA of
CIFAR **pixels** is dominated by DC/colour and low spatial frequency: the key cloud is a
thin anisotropic pancake, so its participation ratio should be well below `d`. The
`simclr` trunk is L2-normalised on a 256-sphere under a contrastive objective that
*explicitly* spreads augmentation-invariant content. At fixed `n`, median-NN spacing
scales roughly with the geometric mean of retained per-axis scales relative to `r95`;
a participation ratio 2–4× higher should buy **1.3–2.5× spacing**, i.e. σ_q/spacing
lower by that factor ⇒ ratio ≈ 1/1.8 ≈ 0.55.
⚠ This is *above* the Hub's registered Q3 prior of 0.55 because the mechanism is
structural rather than incidental. **If P1 misses, Q3's 0.55 was the better call and the
mechanism above is wrong** — which is a finding about φ, not about the store.

**P2.** Pass 1/2 measured MNIST PCA `d = 8` spacing 0.138–0.147 against σ_q = 0.15, i.e.
σ_q/spacing ≈ **1.02–1.09**. Even a 2× improvement lands ≈ 0.5. A store needs
`d_safe = 2 s_max + 2.576 σ_q` to fit *inside* the spacing for reliable addressing, so
σ_q/spacing must fall well below ~0.3 before the address space is comfortable. I predict
strong φ **does not get there** ⇒ **a GO and a still-unaddressable substrate are
compatible, and I register that now** so it cannot later be read as a contradiction.

**P3.** `(1/n)^{1/d}` rises with `d`, so at fixed `n` the ball has more room. The
geometry should therefore favour the *largest* feasible `d` — and the binding constraint
on `d` is the **atom law and the store's inertness**, not the geometry. ⇒ I predict the
report ends with **`d_favoured_by_geometry = 16` and `d_recommended_operational = 8`**,
i.e. a *conflict*, and the whole value of separating the two fields is that this conflict
is visible instead of silently resolved.

**P4.** The diagnosed cause of the banked `d ≥ 16` inertness is **store-side, not
φ-side**: atom centers are drawn `N(0, 1)` in `dim = d + 1`, so the nearest of `n_atoms`
atoms to a unit-norm site recedes (0.738 → 1.252 → 1.483 at `d` = 8/12/16) while
`atom_width` stays 0.3, and the write gradient's `exp(−r²/2s²)` factor falls
4.86e-2 → 1.65e-4 → **4.98e-6**. Since the projected CIFAR φ is *also* normalised to
`r95 = 1`, **the same arithmetic applies verbatim** ⇒ the banked MNIST inertness should
reproduce on CIFAR φ, with the co-scaling honoured (`atom_budget_honoured` is asserted
against `CluSystemConfig.n_atoms` in the cell). ⇒ **the binding constraint is REACH, not
capacity** — the Hub's reading from banked evidence, here confirmed or refuted on a
censused cell rather than a scratch probe.

**P5.** `d_safe = 0.88 × median-NN(≈200 task-1 keys)` but the gate then adjudicates a
≈16-item population. Spacing falls with `n` (roughly like `n^{-1/d}`), so at `d = 8`,
`(200/16)^{1/8} ≈ 1.37` ⇒ the sizing spacing is ~1.4× smaller than the population
spacing ⇒ `d_safe_rig/population_spacing ≈ 0.88/1.37 ≈ 0.64`… but the sizing set is also
drawn from **one task** (tighter still), so I widen the band down to 0.15 and take
**0.35** as the point estimate. At that ratio essentially nothing is within `d_safe` of
anything ⇒ **refusal rate 0.000, and monitor #3 is QUIET BECAUSE IT CANNOT FIRE.**

**P6.** ⚠ **The honest complication, registered rather than discovered later.** The
`simclr` read-in already ends in a **PCA head** (`enc_head = "pca"`), so φ's 256
coordinates are *already ordered by explained variance*. Truncating to the first `d` is
therefore close to keeping the top-`d` principal directions — nearly what the fitted map
recovers. ⇒ I predict `truncate ≈ pca` **on this encoder**, and if that holds, the
defect the wave is fixing is a defect of **principle and of the ledger** (a truncation
is not a declared, priced, fitted map and does not generalise to a φ without a PCA head)
rather than a large numerical loss **at this particular encoder**. ⛔ That distinction
travels with any quotation of P6.

---

## 4. What would make me report a NO-GO or an abort

- **NO-GO** (P1 misses): reported as such, with the ratio, and the spine is **re-labelled
  not blocked** (R4). ⛔ I will not re-fit, re-tune or re-pick `n`/`d` to convert it.
- **Inert store at every `d`** (including 8): reported in the **first 10 lines**, and the
  geometry is still reported — it is φ-side and does not depend on the store digging.
- **Rider (a) overrun** (> 3× the price above): declared **NOT-RUN with the arithmetic**.

---

## 5. Declared NOT-RUNs (never reported as nulls)

`convae` and every encoder arm beyond `simclr` + `randconv` · any ACC / performance
number · G-ADDR (wt1's) · the spine's capture gate (wt3's) · `generic_frozen` φ · any
merge/prune verb · MNIST (the pass-1/2 substrate — **excluded by R1**, not merely unrun).

*Filed before the first CIFAR-10 cell. Corrections, if any, go in a dated ERRATA block in
the report, never in this file.*
