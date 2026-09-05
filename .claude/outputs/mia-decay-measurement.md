# mia-decay-measurement — results-analyst report
Task + acceptance criterion: put retention and a per-example (U-LiRA-style) distinguishability curve on ONE `leak·t` axis, report where they cross, measure what survives `evict`, and run the TTL laundering line — pre-registered, ≥3 seeds, all numbers re-derived from a saved metrics JSON.
Status: **done.**

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). THREE items.**
> 1. ⛔ **`controller-mvp` §3(b)'s "leaky item retention = 1.00 through 8 ticks" is a BEST CASE, not the store's retention.** Its `retention_of` scores `evaluate_items` on a **one-element site array**, so `basin_of`'s argmin is vacuous and only the value criterion is tested; and it happens to probe `pay[1] = +0.1429`, the codebook value most robust to decay. Re-measured with the full live-site basin test and all 8 codebook values: retention at that same amplitude (A = 0.0608) is **0.886 ± 0.19**, and **0.500 ± 0.11 for the item whose payload is +1.0**. The N91/§6 line "a permanent item retains 1.000 while leaky wells decay and self-evict" is still true; the implied "decayed items answer perfectly right up to the floor" is not. **Owner needed: `doc-curator` (§6 + N91 wording) and `experiment-engineer` (D2 below).**
> 2. ⛔ **`AtomStorePotential.evict` is not erasure at the data-structure level** — `centers[slot]` and `payloads[slot]` keep the written values verbatim (measured max error **5.6e−8** and **0.0** over 3 072 evictions). Anything in the program that says "the item is gone / the slot is freed" must say "the item is removed from `V`; the row is not cleared." **Owner: `experiment-engineer` (one-line fix, no scientific result moves).**
> 3. ⚠ **The lifetimes dial is NOT payload-independent.** Retention near the floor correlates **r = −0.85** with `a_i²` and **r = +0.01** with `|c_i|`. Two items written with the *same* `leak` have different effective half-lives. **Owner: `experiment-engineer`/theorist — see D3.**

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** lifetimes.
- **Laundering control:** (i) **CLU-with-a-TTL-flag** — identical store/controller/adversary, item held at `amp ≡ 1` until the same expiry tick then `evict`ed; (ii) a **TTL vector-store** (nearest-neighbour dict, row deleted at expiry).
- **Falsifies:** distinguishability does not decay with amplitude.
- **Does NOT falsify:** distinguishability persisting *longer* than retention.
- **Verdict on the falsifier: it FIRES against an exact adversary and does NOT fire against a resolution-limited one.** Both are reported below; neither was chosen after the fact (PREREG §1 P3/P8).

---

## Flag-provenance table

| item | value |
|---|---|
| base commit | local `main` @ **`63c668d`** — **working tree clean, no tracked file modified** (analysis-only) |
| harness | `.claude/outputs/mia-decay-measurement/mia_harness.py` (+ `analyze.py`, `fig1.py`, `decomp.py`, `payload_dep.py`); run `PYTHONPATH=. .venv/bin/python mia_harness.py` |
| PREREG | `.claude/outputs/mia-decay-measurement/PREREG.md`, **written before the harness existed** |
| metrics | `mia_metrics.json` (1.07 MB), `tables.json`, `fig1_data.json`, `payload_dependence.json`, `retention_decomposition.json`; log `full_run.log` |
| seeds | **0, 1, 2** (target draw + world draw + query draw) |
| n | **8 targets × 3 seeds = 24 per-example values**; **128 paired IN/OUT worlds** per target; **16 queries** per world; 18 amplitude levels; 11 radii; 4 probe-noise levels |
| designed store | `AtomStorePotential(dim=3, capacity=8, α=0.02, s=0.35, s_pay=s, κ=1.0)` — the theorist S3 / `controller-mvp` values. **No learning anywhere.** |
| controller | shipped `Controller`, `d_safe = 4.4·s = 1.540`, `n_relocation_candidates = 400`, `budget = capacity = 8`, 1 target + 7 background |
| geometry | proposal disk `R = radius_for_capacity(8, 1.54) = 2.2869`; packing bound **8.00** |
| decay | amplitude set directly to `A`; `τ = leak·t = −ln A` is exact for the shipped law `amps *= exp(−leak)` (verified: P9). `amp_floor = 0.05` ⇒ **`τ_evict = ln 20 = 2.996`** |
| retrieval (shipped, unmodified) | two-phase, `dt 0.05`, `γ_address 0.05 × 400` → `γ_read 0.0 × 800`, tail 0.25, 8 subsamples, `q₂(0) = p₂(0) = 0` |
| queries | shipped `make_queries_at`: `σ_θ = 0.15`, `σ_p = 0.05`, `f = 1.0`, `n_query = 16` (Panel A) · deterministic ring of radius `r` (Panel B) |
| payload codebook | `designed_payloads(8, seed=0)`, spacing **2/7 = 0.2857** ⇒ **`payload_tol = min(0.1, 0.35×0.2857) = 0.1000`**. ⚠ **PREREG mis-stated the spacing as 1/7 and the tolerance as 0.05** — arithmetic slip in the prereg, corrected here; it loosens no criterion relative to the shipped default (`payload_tol = 0.1` is the shipped cap) |
| JAX / venv | **0.9.0**, main venv (protocol §4, no worktree sync) |
| runtime | **1 009 s** (single process, CPU) |
| N94 | no training anywhere; every fit/statistic is on a *designed* store — no epoch count applies |

---

## 0. Headline (six numbers)

1. **The store stops answering before it stops leaking — measured.** At the last amplitude before self-eviction (`A = 0.051`, `τ = 2.976`): **retention = 0.832**, query-MIA **AUC = 0.983** (**TPR = 0.858 @ FPR 1 %**), white-box MIA **AUC = 1.000**. The curves never cross; retention is below distinguishability at every `τ > 0` and the gap widens monotonically.
2. **Against an EXACT adversary, amplitude decay buys nothing.** White-box AUC is **1.000 at all 18 amplitude levels** down to the floor. The depth gap is **`0.3935·A` to four decimal places at every level** (pre-registered coefficient `1 − e^{−1/2} = 0.3935`) — so what decays is the **effect size** (`d′` 1 560 → 79.6, exactly linear in `A`), not the adversary's success. **The declaration's falsifier fires on the AUC metric, as pre-registered (P3).**
3. **Against a resolution-limited adversary, decay is genuinely graded — and the pre-registered law survives.** `A₇₅ = 2.57·σ_obs` measured **0.0780 / 0.2627 / 0.7537** at `σ_obs = 0.03 / 0.1 / 0.3` vs predicted **0.0771 / 0.2570 / 0.7710** (+1.2 %, +2.2 %, −2.2 %). The whole `AUC(A)` curve lies on the registered `Φ(0.3935A/1.5σ)` (fig 2).
4. **The laundering control decides the "physical amplitude vs bookkeeping flag" question, and the answer is conditional.** CLU-with-a-TTL-flag is a step. Against the exact adversary CLU's decay curve is within **0.017 AUC** of that step (0.983 vs 1.000) — *no measurable differentiator*. Against `σ_obs = 0.1` the two separate by **0.44 AUC** (0.559 vs 1.000) at `A = 0.06`. **The differentiator exists only relative to a stated adversary resolution.**
5. **Post-`evict` is clean only if placement is order-independent.** Paired-placement: **AUC = 0.5000 ± 0.0000** on all four statistics (`V` is bit-identical). Natural (history) OUT: the allocator trace gives **AUC = 0.99985 ± 0.0007, TPR = 1.000 @ FPR 1 %** — *after* the item is gone. And the raw arrays still hold the address and value verbatim (max err 5.6e−8 / 0.0).
6. **The graded, physical quantity is the BASIN RADIUS, not the AUC.** `R₅₀` (read radius at which retention halves) moves **1.146 → 0.752** as `A: 1 → 0.06`, matching the pre-registered saddle prediction (1.15/1.05/0.90/0.80/0.72 vs measured 1.146/1.083/0.979/0.874/0.752, all within ±0.20). The TTL vector-store's is a **constant hard step at `R_lookup = 0.77`**. *This is the one place the physical story has a measurement the boolean substitute cannot reproduce.*

---

## 1. Item 1 ⭐ — the two curves on one `leak·t` axis

**Figure:** `.claude/outputs/mia-decay-measurement/fig1_retention_vs_mia.png` (data: `fig1_data.json`).
AUCs are **direction-calibrated** (`max(AUC, 1−AUC)` per example, then averaged) — the LiRA adversary calibrates the sign of its statistic on shadow worlds; raw AUCs are also in `mia_metrics.json`.

| `A` | `τ = leak·t` | **retention** | MIA AUC TM-1 (query, paired) | TPR@FPR 5 % | **TPR@FPR 1 %** | MIA AUC TM-2a (white-box, paired) | `d′` (TM-2a) | MIA AUC TM-2a (history OUT) |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.000 | 1.0000 | 1.0000 | 1.000 | 1.000 | **1.0000** | 1 560 | 1.000 |
| 0.50 | 0.693 | 1.0000 | 1.0000 | 1.000 | 1.000 | **1.0000** | 780 | 0.933 |
| 0.30 | 1.204 | 1.0000 | 1.0000 | 1.000 | 1.000 | **1.0000** | 468 | 0.887 |
| 0.20 | 1.609 | 0.9990 | 1.0000 | 1.000 | 1.000 | **1.0000** | 312 | 0.855 |
| 0.15 | 1.897 | 0.9928 | 0.9999 | 1.000 | 0.998 | **1.0000** | 234 | 0.836 |
| 0.12 | 2.120 | 0.9830 | 0.9995 | 0.997 | 0.992 | **1.0000** | 187 | 0.824 |
| 0.10 | 2.303 | 0.9714 | 0.9986 | 0.993 | 0.979 | **1.0000** | 156 | 0.814 |
| 0.08 | 2.526 | 0.9397 | 0.9967 | 0.981 | 0.952 | **1.0000** | 125 | 0.807 |
| 0.07 | 2.659 | 0.9159 | 0.9941 | 0.969 | 0.932 | **1.0000** | 109 | 0.802 |
| 0.06 | 2.813 | 0.8858 | 0.9895 | 0.950 | 0.899 | **1.0000** | 93.6 | 0.796 |
| 0.055 | 2.900 | 0.8605 | 0.9859 | 0.932 | 0.879 | **1.0000** | 85.8 | 0.794 |
| **0.051** | **2.976** | **0.8321** | **0.9830** | **0.921** | **0.858** | **1.0000** | 79.6 | **0.791** |
| — (evicted) | 3.0+ | **0.0000** | **0.5000** | 0.000 | 0.000 | **0.5000** | 0 | **0.760** |

*(full 18-level table in `fig1_data.json`; per-example spread at the floor: retention std **0.274**, MIA-AUC std **0.034**.)*

### Where do they cross?
**They do not cross.** `MIA ≥ retention` at every `τ`, with the gap opening from `0.000` (`τ ≤ 1.2`) to **`+0.151`** at the floor and to **`+0.500 / +1.000`** immediately after eviction (paired / allocator-trace). Direction of the asymmetry: **retention decays first**. Registered prediction P6 ("MIA outlives retention — the store stops answering before it stops leaking") **confirmed**, and per the DIAL DECLARATION this direction is explicitly non-falsifying — it is the finding.

The same ordering holds on the *radius* axis (§3): at `A = 0.06`, retention halves at `r = 0.752` while the query-MIA is still at AUC 0.75 at `r = 1.382`; for `A ≥ 0.1` the MIA never falls to 0.75 anywhere in `r ≤ 1.4` while retention halves by `r ≤ 0.98`.

### Two threat models, two different stories (both pre-registered)
- **TM-2a, white-box, paired OUT (the exact adversary):** AUC **1.000 flat**. Mechanism: the background contributes `exp(−4.4²/2) = 6.3e−5` at `c_i` while the target contributes `0.3935·A ≥ 0.0197` at the floor. `d′` decays linearly (1 560 → 79.6, ratios exactly `1/A`) but AUC saturates. **Amplitude decay does not reduce an exact adversary's success at any point in the item's life.**
- **TM-2a, white-box, HISTORY OUT (the natural adversary):** the background differs between worlds, which supplies a real noise floor — and *here* the AUC **is graded**: 1.000 → 0.791, **saturating at the post-evict allocator-trace level 0.760**. This is the honest "graded" curve for a white-box attacker who does not get a matched-background oracle.
- **TM-1, query-only:** 1.000 → 0.983 (paired) / 0.965 (history). Mildly graded, always above retention.

---

## 2. Item 2 — what remains after `evict`

**Retention post-evict = 0.0000** (all 24 examples, all 128 worlds).

| statistic | OUT = paired-placement | OUT = history (natural) |
|---|---|---|
| `s1` value-return (query) | **0.5000 ± 0.0000** | 0.6015 ± 0.0915 (TPR@1 % 0.154) |
| `s2` address-capture (query) | **0.5000 ± 0.0000** | 0.7828 ± 0.1435 (TPR@1 % 0.377) |
| `s4` white-box address-channel depth | **0.5000 ± 0.0000** | **0.7599 ± 0.0644 — TPR@1 % = 1.000** |
| `s5` white-box full `V` | **0.5000 ± 0.0000** | 0.7106 ± 0.0796 — TPR@1 % = 1.000 |
| **`z_hole` = dist(`c_i`, nearest live site)** | 0.5000 ± 0.0000 | **0.99985 ± 0.00070 — TPR@1 % = 1.000** |
| `n_live` (admitted background count) | — | 0.8114 ± 0.0349 (raw AUC 0.189, i.e. IN admits *fewer*) |
| **raw array dump (TM-4)** | `centers[slot]` max err **5.62e−8** (float32), `payloads[slot]` max err **0.0**, `amps` 0.0, `active` 0.0 | same |

**Reading.** Three distinct residual channels, in descending severity:
1. **The allocator trace is a near-perfect membership oracle.** Under the shipped history-dependent placement (`refuse-and-relocate` + budget), writing item `i` guarantees a `d_safe = 1.54` exclusion disk around `c_i` that survives `i`'s eviction. Distance-to-nearest-live-site separates IN from OUT at **AUC 0.99985, TPR 1.000 @ FPR 1 %** — *after* the item is physically gone from `V`. This is exactly the N99 / recon-item-3 history-dependence gap, now with a number: **it is not a technicality, it is the dominant leak.** It is also, by construction, the **baseline that `order-independent-placement` must drive to 0.5**.
2. **The raw store is not erased.** `evict` clears `active` and `amps` only. An adversary with state-read access recovers `(c_i, a_i)` exactly. Invisible to `V` (the mask multiplies both terms), so *no measured `V`-based or query-based result changes* — but "the item is gone" is false at the data-structure level.
3. **The amplitude channel itself closes exactly.** With placement held fixed, every statistic is **0.5000 ± 0.0000** post-evict — the two stores are bit-identical. This is the clean part of the result and the sanity check that the harness is not manufacturing signal (P7a confirmed exactly).

**The partially-decayed worst case the recon predicted (`A ≈ 0.05–0.06`) was sampled densely** (6 of 18 levels in `[0.051, 0.10]`). It is not "neither present nor absent" to an adversary — it is **fully present** (white-box AUC 1.000, query AUC 0.983–0.999) while being only **83–97 % answerable**. The ambiguity is one-sided: it degrades the *user*, not the *attacker*.

---

## 3. Item 3 — the TTL laundering line

**Figures:** `fig2_resolution.png` (TM-3 + effect size), `fig3_radius.png` (radius sweep + TTL vector-store).

### (a) CLU-with-a-TTL-flag (same store, `amp ≡ 1` until expiry) — the tight control
| adversary | CLU decay at `A = 0.06` | TTL-flag control | separation |
|---|---|---|---|
| TM-2a exact white-box | **1.000** | 1.000 | **0.000 — no differentiator** |
| TM-1 query (paired) | 0.9895 | 1.0000 | 0.011 |
| TM-3, `σ_obs = 0.01` | 0.946 | 1.000 | 0.054 |
| TM-3, `σ_obs = 0.03` | 0.709 | 1.000 | 0.291 |
| **TM-3, `σ_obs = 0.1`** | **0.559** | **0.996** | **0.437** |
| TM-3, `σ_obs = 0.3` | 0.533 | 0.818 | 0.285 |

⛔ **Against an exact adversary the laundering control FIRES**: a boolean TTL flag and a physical `exp(−leak·t)` amplitude are *equally* detectable at every point in the item's life. ✅ **Against a resolution-limited adversary it does NOT fire**, and the separation follows the pre-registered `Φ(0.3935A/1.5σ)` law to ~2 %. **The honest paper sentence is therefore adversary-relative**, e.g.: *"amplitude decay reduces an attacker's advantage in proportion to `A` relative to its measurement resolution `σ`; against an exact-arithmetic attacker it does not, and neither does any deterministic bookkeeping alternative."*

### (b) TTL vector-store (nearest-neighbour dict, row-delete at expiry)
| launch radius `r` | 0.0 | 0.106 | 0.212 | 0.4 | 0.6 | 0.7 | **0.8** | 0.9 | 1.0 | 1.2 | 1.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TTL dict** retention (row live, **any age**) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| CLU `A = 1.00` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.977 | 0.821 | 0.663 | 0.440 | 0.324 |
| CLU `A = 0.20` | 1.00 | 1.00 | 1.00 | 0.997 | 0.974 | 0.869 | 0.727 | 0.595 | 0.475 | 0.323 | 0.273 |
| CLU `A = 0.06` | 1.00 | 0.947 | 0.855 | 0.819 | 0.690 | 0.570 | 0.434 | 0.343 | 0.289 | 0.236 | 0.198 |

`R₅₀` (retention = 0.5): **CLU 1.146 / 1.083 / 0.979 / 0.874 / 0.752** at `A = 1 / 0.5 / 0.2 / 0.1 / 0.06` — a **1.52×** contraction, matching the pre-registered saddle calculation (P5, all points inside ±0.20). **TTL dict: 0.75–0.77 constant, a hard step, independent of age.**

✅ **This is the differentiator that survives without an adversary-resolution caveat**: decay continuously shrinks the *addressing tolerance* of the item — the store answers approximate queries with progressively less slack — which a row-with-a-timestamp cannot express. It is a **retrieval-geometry** claim, not a privacy claim.

---

## 4. PREREG scorecard (`PREREG.md`, written before the harness existed)

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | retention ≥ 0.95 for all `A ≥ 0.051`; step at the floor | 1.000 to `A = 0.3`, then a graded tail to **0.832** at `A = 0.051` | ❌ **falsified as stated.** Retention is *not* a step; the tail is real (and payload-dependent, §5) |
| P2 | TM-1 AUC ≥ 0.99 ∀`A ≥ 0.05`; 0.50 post-evict paired | **0.983** at the floor; **0.5000** post-evict paired | ◐ direction right, the ≥0.99 bound falsified at the last two levels |
| P3 | TM-2a AUC = 1.000 ∀`A ≥ 0.05`; gap `= 0.3935·A`; `d′ ∝ A` | **1.0000 at all 18 levels**; gap/`A` = **0.3935 at all 18 levels**; `d′` 1 560 → 79.6 (exactly `∝ A`) | ✅ **confirmed to 4 dp**, including the pre-registered "the falsifier will fire on AUC" |
| P4 | `|s1(A=1) − s1(A=0.051)| < 0.01` (payload channel amplitude-independent) | **−0.0001 → −0.1331 (Δ = 0.133)** | ❌ **falsified.** `S(q)` *is* amp-independent, but the read *position* is not: a shallow well lets the query drift, so the value read degrades |
| P5 | `R₅₀ = 1.15/1.05/0.90/0.80/0.72` (±0.20); ratio 1.60 ± 0.25; TTL constant | **1.146/1.083/0.979/0.874/0.752**; ratio **1.524**; TTL **0.75 constant** | ✅ **confirmed**, all five inside the band |
| P6 | MIA outlives retention; radius gap ≥ +0.10 | no crossing; gap `+0.151` at the floor; radius gap `R₇₅^MIA − R₅₀^ret = +0.63` at `A = 0.06`, unbounded for `A ≥ 0.1` | ✅ **confirmed, more strongly than registered** |
| P7a | post-evict paired AUC = 0.500 ± 0.05 | **0.5000 ± 0.0000** on all four statistics | ✅ exact |
| P7b | history `AUC(z_hole) = 0.85 ± 0.10`; `AUC(n_live) = 0.60 ± 0.10` | **0.99985 ± 0.0007** and **0.811 ± 0.035** | ❌ **both outside the band — I under-predicted the allocator trace badly.** It is a near-perfect oracle, not a weak signal |
| P7c | `evict` leaves `centers`/`payloads` verbatim | max err **5.62e−8** / **0.0** over 3 072 evictions | ✅ confirmed |
| P8 | `AUC = Φ(0.3935A/1.5σ)`; `A₇₅ = 2.57σ` = 0.026/0.077/0.257/0.771 | 0.0780 / 0.2627 / 0.7537 at σ = 0.03/0.1/0.3 (**+1.2 %, +2.2 %, −2.2 %**); σ = 0.01 predicts `A₇₅ = 0.026`, **below the sampled grid — untested** | ✅ **confirmed** at 3 of 4 σ; curve matches point-by-point (fig 2) |
| P9 | `amps[t] = exp(−0.35t)`; evict at `t = 9` | `[1.0, 0.7047, 0.4966, 0.3499, 0.2466, 0.1738, 0.1225, 0.0863, 0.0608, 0]`; **evict at tick 9** | ✅ exact (reproduces `controller-mvp`'s 0.705) |

**Score: 5 confirmed, 1 partial, 4 falsified.** The two most decision-relevant falsifications (P4, P7b) both point the same way: **the store leaks through channels the amplitude dial does not control.**

---

## 5. Not pre-registered (exploratory, flagged as such): the lifetime dial is payload-dependent

`payload_dependence.json`. Per-example retention at the floor correlates **r = −0.846 with `a_i²`** and **r = +0.015 with `|c_i|`** (`n = 24`).

| target payload `a_i` | ±0.1429 | ±0.4286 | ±0.7143 | **−1.0** | **+1.0** |
|---|---|---|---|---|---|
| retention at `A = 0.06` | 0.999 / 1.000 | 0.997 / 0.997 | 0.969 / 0.962 | **0.663 ± 0.093** | **0.500 ± 0.110** |
| retention at `A = 0.051` | 0.998 / 1.000 | 0.993 / 0.992 | 0.949 / 0.929 | **0.544 ± 0.069** | **0.252 ± 0.210** |
| `τ₅₀` (retention halves) reached before `τ_evict = 2.996`? | no | no | no | no (0.544 at the floor) | **yes, `τ₅₀ = 2.814`** |

**Mechanism (from the shipped `V`).** The payload term is `0.5κ(q₂ − S(q_addr))²` and the anti-decoration guard forces `q₂(0) = 0`, so at launch the item's own site carries a payload **hill** of height `0.5κa_i²` (up to **0.5** for `a_i = ±1`) competing with an address **well** of depth `A` (0.05 at the floor — 10× smaller). Directly visible in the full-`V` probe: `s5` crosses zero between `A = 0.4` and `A = 0.3` (0.024 → −0.015) and is **−0.113 at the floor**, i.e. the decayed site becomes a net *maximum* of `V` at `q₂ = 0` for large `|a_i|`. Query MIA is unaffected (0.983 regardless of `a_i`) — **only the user loses, not the adversary.**

⇒ **Two items given the same `leak` do not have the same effective lifetime.** Reported as a defect of the dial, not of the decay law (the amplitude law itself is exact — P9).

---

## 6. How I verified / reconciliation with `controller-mvp` §3(b)

- `mia_harness.py` uses the shipped objects verbatim: `Controller`, `AtomStorePotential.with_item/evict/with_amps`, `admit_site`, `two_phase`, `make_queries_at`, `effective_payload_tol`, `model_for`. The only addition is a `jax.vmap` over the store arrays so 128 worlds read in one call (`eqx.filter_jit(jax.vmap(...))`); the per-trajectory computation is the shipped `CHLU.__call__` path.
- **Smoke first:** `--quick` (16 worlds, 2 targets, 3 levels) in 11 s, then the full run in 1 009 s, exit 0, no NaN/divergence (`finite` checks in the read path; no non-finite value appears in any statistic).
- **Independent re-derivation:** every number in this report comes from `mia_metrics.json` via `analyze.py` / `fig1.py` / `payload_dep.py`; nothing is transcribed from stdout.
- **P9 cross-check against `controller-mvp`:** `Controller.tick()` at `leak = 0.35` gives `0.7047` at `t = 1` — the same `0.705` they report — and self-eviction at tick 9.
- **The apparent conflict with `controller-mvp` §3(b) ("leaky item-1 retention 1.00 through 8 ticks") is fully explained, not a contradiction** (`retention_decomposition.json`): their `retention_of` passes a **1-element** `sites` array to `evaluate_items`, so `basin_of`'s `argmin` is always 0 and the basin criterion is vacuous; and their `leaky_id = 1` has payload `+0.1429`, the codebook's most decay-robust value. On the same amplitude with the full live-site basin test, that item scores **0.999** (agreeing with them) while the 8-value average is **0.886** and the `a = +1` item is **0.500**.
- **Sanity controls that had to come out exactly right, and did:** post-evict paired AUC = 0.5000 ± 0.0000 on 4 statistics × 24 examples; `s4` gap `= 0.3935·A` at all 18 levels (the closed-form `1 − e^{−1/2}`); the decay law reproducing an independently reported constant.

## Git footprint
**None.** No tracked file created, modified or deleted; `git status --short` empty before and after. No branch, no commits. All artefacts under `.claude/outputs/mia-decay-measurement/` and `.claude/scratch/mia-decay-measurement/` (both gitignored).

---

## 7. Limitations / confounds (stated before a referee does)

1. **Scope is the STORE only.** No `φ`, no learned `V`, no payload-encoder. `φ` and the payload channel are separate leak surfaces (recon §Item-3); nothing here supports a system-level erasure statement. **No `(ε,δ)` claim is made; the statement is a measured curve.** Words *certified / unlearning / privacy guarantee* are not used (CM-22 m/n/o).
2. **Small store.** capacity 8, `dim = 3`, 7 background items, one designed store family, Newtonian kinetic, `p₀ ≈ 0`. The allocator-trace magnitude in particular is expected to depend on the load factor (7/8 of the packing bound here — near-full, the most favourable case for a "hole" statistic). **It should be re-measured at low occupancy before being quoted as a general number.**
3. **The paired-placement OUT world is an idealisation** — it hands the adversary a background matched to the IN world. It is the right instrument for isolating the amplitude channel (and gives the exact 0.5 post-evict control), but the *natural* attack surface is the history condition, which is uniformly reported alongside.
4. **`TPR @ FPR = 1 %` is at the resolution limit of 128 OUT worlds** (the 1 % quantile sits between the top one and two samples). Treat the 1 % column as indicative; the 5 % column and the AUC are well resolved.
5. **`σ_obs` is a modelling choice, not a property of CLU.** TM-3 is "an adversary with finite measurement precision on the potential." I report the law and its constants; I do not claim any particular `σ_obs` is the right one, and at `σ_obs = 0.01` the predicted `A₇₅ = 0.026` lies below the floor so the law is untested there.
6. **Amplitudes were set directly rather than by ticking.** Justified because the shipped law is exactly `amps *= exp(−leak)` and P9 verifies it; but it means every `τ` here is a *pure* amplitude effect with no accumulated tick history (no interleaved writes, no background decay).
7. **Direction-calibrated AUC** (`max(AUC, 1−AUC)`) is used for the headline curves. Raw AUCs are in the JSON; two statistics (`s5`, `n_live`) have example-dependent sign and would be under-reported by the raw convention. LiRA TPRs are sign-agnostic by construction and tell the same story.

---

## 8. Defects flagged for `experiment-engineer`

| id | file / symbol | defect | suggested fix | does it move any published number? |
|---|---|---|---|---|
| **D1** | `chlu/core/memory_potentials.py`, `AtomStorePotential.evict` (L1017–1027) | clears `active` and `amps` only; `centers[slot]` and `payloads[slot]` retain the written address and value verbatim (measured max err 5.6e−8 / 0.0) | also zero `centers[slot]` and `payloads[slot]` | **No** — `V` multiplies both terms by `active`, so every physics result is unchanged. It changes what "the slot is freed" means |
| **D2** | `chlu/experiments/exp_controller_mvp.py`, `item_decay_demo.retention_of` | passes a **1-element** `sites` array to `evaluate_items` ⇒ `basin_of` argmin is always 0 ⇒ the basin criterion is vacuous; and `leaky_id = 1` is the codebook's most decay-robust payload | score against the store's full live-site set (as `_score` already does) and average over several `leaky_id`s | **Yes, for `controller-mvp` §3(b)/N91's decay-demo row**: retention at `A = 0.0608` is 0.886 (8-value mean), not 1.00. The permanent-item row (1.000 ∀t) is unaffected |
| **D3** | design-level: `AtomStorePotential.__call__` payload term + the `q₂(0) = 0` guard | `S(q)` does not scale with `amps`, so a decayed site is a payload **hill** of height `0.5κa_i²` against a well of depth `A`; effective lifetime correlates `r = −0.85` with `a_i²` | either scale the payload bump by the amplitude (`payloads * amps`) or launch reads at `q₂ = S(q_addr)` instead of 0 | **Yes, conceptually** — it is what makes the lifetimes dial payload-dependent. Needs a theorist ruling before changing the potential |

---

## 9. Recommended next experiments (ranked)

1. **`order-independent-placement`'s target is now numeric: drive post-evict `AUC(z_hole)` from 0.99985 to 0.5.** This harness is the acceptance test — re-run §2's history column against the new placement rule; nothing else needs to change. **Until it is 0.5, no deletion-flavoured sentence about this store is defensible.**
2. **Occupancy sweep on the allocator trace.** Repeat §2 at load factors 2/8, 4/8, 6/8, 8/8. My number is at 7/8 (near-full), the most favourable case for the hole statistic; the leak may be much weaker at low occupancy, which would scope the claim usefully.
3. **Fix D3 and re-measure §1 and §5.** If the payload bump is scaled by the amplitude, retention should become payload-independent *and* the retention curve should get steeper near the floor — testing whether the lifetimes dial can be made clean. Cheap (this harness, one config change).
4. **Sequential/interleaved decay.** Everything here is a single decaying item in a static background. Measure the same curves while other items are written and decaying concurrently — the "freed wells restore admission headroom" coupling (novelty-surface item 4 of the recon) is untested.
5. **Do NOT invest in a `σ_obs`-based privacy story without a threat-model owner.** The TM-3 law is clean and pre-registered, but the constant `σ_obs` is ours to choose, which a security referee will say makes it unfalsifiable. Its defensible use is *mechanistic* ("the leak signal is `0.3935·A`"), not *protective*.

---

## Proposed handover updates (for the Hub)

1. **§1.6 / R1 — a new, quotable result (the lifetimes dial's adversarial line).** *"On the designed store with MVC-0 decay (8 targets × 3 seeds × 128 paired worlds, per-example U-LiRA), retention falls to **0.832** at the last amplitude before self-eviction while per-example distinguishability is still **AUC 0.983 / TPR 0.858 @ FPR 1 %** (query) and **1.000** (white-box). The curves never cross: **the store stops answering before it stops leaking.** After `evict`, with placement held fixed the amplitude channel closes exactly (**AUC 0.5000 ± 0.0000**), but under the shipped history-dependent placement the allocator trace alone gives **AUC 0.99985, TPR 1.000 @ FPR 1 %**."*
2. **§1.6 — the laundering verdict, both halves, must travel together.** Against an **exact** adversary, CLU's `exp(−leak·t)` decay is **indistinguishable in AUC from a boolean TTL flag** (1.000 vs 1.000 white-box; 0.983 vs 1.000 query) — the "physical amplitude ≠ bookkeeping" differentiator **does not exist at the AUC level**. It appears only (a) relative to a stated adversary resolution — `A₇₅ = 2.57 σ_obs`, pre-registered and confirmed to 2 % — and (b) in the **retrieval geometry**: `R₅₀` contracts **1.146 → 0.752** (1.52×) as `A: 1 → 0.06` while a TTL dict's lookup radius is a constant step at 0.77. **(b) is the differentiator to lead with; it needs no adversary-model caveat.**
3. **§8 / negative registry — candidate new N (recommend registering).** *"Amplitude decay does not reduce an exact adversary's per-example distinguishability at any point in an item's life: white-box AUC = 1.000 at all 18 amplitude levels down to the floor; what decays is the effect size (`gap = 0.3935·A` to 4 dp, `d′` 1 560 → 79.6), not the attack's success. The graded curve exists only against a resolution-limited adversary."* Pre-registered as the expected outcome (PREREG P3), so this is a confirmed prediction, not a surprise.
4. **§7-CURRENT — two shipped-code defects (details in §8 above).** (D1) `AtomStorePotential.evict` leaves `centers`/`payloads` verbatim — "the slot is freed" ≠ "the row is cleared"; no physics number moves. (D2) `item_decay_demo.retention_of` scores the basin against a **single site**, making that criterion vacuous, and probes the codebook's most decay-robust payload — **`controller-mvp` §3(b)'s "leaky retention 1.00 through 8 ticks" is a best case; the 8-value mean at the same amplitude is 0.886, and 0.500 for `a = +1`.**
5. **§7 / theory queue — the lifetimes dial is not payload-independent.** Retention at the floor correlates `r = −0.846` with `a_i²` (`r = +0.015` with `|c_i|`), because the `q₂(0)=0` guard makes each site a payload hill of height `0.5κa_i²` against a well of depth `A`. Two items with the same `leak` have different effective half-lives; only the *user* is affected (query MIA is flat in `a_i`). Needs a theorist ruling on whether the payload bump should scale with the amplitude.
6. **N99 upgrade — the history-dependence gap now has a magnitude.** It is not a technicality: it is **the dominant residual leak**, and it is *stronger* than every amplitude-channel effect measured here. `order-independent-placement` now has a numeric acceptance test (drive post-evict `AUC(z_hole)` 0.99985 → 0.5) and this harness is it.
7. **§5 provenance — new artefact set.** `.claude/outputs/mia-decay-measurement/` : `PREREG.md`, `mia_metrics.json` (all 24 per-example values for every metric/level), `tables.json`, `fig1_data.json`, `payload_dependence.json`, `retention_decomposition.json`, `fig1_retention_vs_mia.png`, `fig2_resolution.png`, `fig3_radius.png`, and the five scripts. Base `63c668d`, JAX 0.9.0, seeds 0/1/2, 1 009 s, **no tracked code touched**.
8. **Wording that is now backed by measurement (safe to use):** *"retention and distinguishability are different curves and they separate: at one lifetime the store answers 83 % of queries while an adversary still separates membership at AUC 0.98."* · *"decay contracts the addressing tolerance of an item (`R₅₀` 1.15 → 0.75), which a timestamped row cannot express."* **Wording that is NOT backed and must not be used:** any claim that decay reduces distinguishability *per se*, and any claim that eviction removes the item — under the shipped placement it does not.
