# PREREG — the C3 Track-A rival ladder

**Filed BEFORE any ladder arm trains** (protocol §5; task `c3-rival-ladder-prereg`; charter §5 invariant
"prereg with numeric falsifiers BEFORE each job ladder"). Agent: `experiment-engineer`.
Branch `agent/experiment-engineer/c3-rival-ladder-prereg` off `agent/experiment-engineer/c3-csf3-harness @ f98f939`.
Date **2026-08-13**. ⛔ **Zero ladder arms trained under this document.**

> ⛔⛔ **THIS PREREG REPORTS A CONFLICT AND HANDS IT BACK (task §2's declared legitimate outcome).**
> There is **no config-only C3 geometry that fits the ruled ≈2 MB ceiling without descending below a
> design-ruled atom floor** — and the write-efficacy curve I measured for exactly that descent, over
> **4 seeds**, **cannot settle whether it is safe** (§2.4b: the pre-registered falsifier fires on 1 seed,
> reverses on another, and the median is monotone at 5.4×). Three options are costed in §2.3; the
> recommended one
> (**G-B: the full atom floor, in 3 of 12 layers**) is byte-, compute- and envelope-**identical** to the
> descent but breaks no design rule — and it is a `chlu/core/blocks.py` change, i.e. **not mine and not a
> config value**. ⛔ **Nothing on the ladder should train until the Advisor picks an option.**
> Everything downstream of the geometry (ceiling digit, arms, shrink knobs, job plan, predictions) is
> ready and is parameterised by the choice.

---

## ⭐⭐ AMENDMENT 1 — 2026-08-13, spoke `c3-gb-landing` (experiment-engineer)

⛔ **The document above is the ACCEPTED text and is not rewritten.** This amendment records what changed
after the Advisor and Head **RATIFIED G-B** on 2026-08-13, and adds the three things the ratification
made binding. Every addition is a new, marked section; nothing already accepted was edited except the
four status markers listed in (0).

| # | what changed | where |
|---|---|---|
| **0** | ⭐ **G-B is RATIFIED and BUILT.** §2's hand-back is CLOSED: `store_layers` landed on `StreamModel`/`PilotConfig` (branch `agent/experiment-engineer/c3-gb-landing`), the arithmetic is pinned by 27 test cases, and the ratified **1,380,864 B / 0.658×** reproduces off **built** cells. **K0 is DISCHARGED** (⛔ K0b never fires — it is G-A's gate and G-A was not chosen). | §0 K-table, §2.2, §2.3, §9.1 markers |
| **1** | ⭐ **THE LAYER PLACEMENT IS A DESIGN DECISION AND IT IS ARGUED** — which 3 of 12, why those, the alternatives, the precedent, and what placement is pre-registered to buy or cost. ⛔ **It is NOT a byte-fitting default:** the ceiling chose *how many*, it did not choose *which*. | **NEW §2.5** |
| **2** | ⭐ **PHASE 1 IS DECLARED, AND PHASE 1 IS NOT THE CLAIM.** This ladder = CLU + the TTT swap + the dyn-eval column + the slices. ⛔ **The six pinned rivals are NOT trained here** (phase 2, funded separately), so **charter §2's tier-iii primary claim WAITS for phase 2 and no phase-1 table may be quoted as it.** | **NEW §4.3** |
| **3** | ⭐⭐ **THE STORE-LIVENESS DIAGNOSTIC IS PRE-REGISTERED ON PHASE 1's FIRST RUNGS** (Head ruling) — the C2 flat-curve disjunction (*carries nothing* **vs** *cannot be addressed*) separated **at real scale, early**, with numeric falsifiers and a **kill condition (K6)**. ⛔ Not deferred to the end of the ladder. | **NEW §7.1**, **K6** in §0 |
| 4 | ⭐ **G-B's launch line now exists** (§6.3 said it did not, because the `blocks.py` work item was unbuilt). | **NEW §6.4** |
| 5 | ⚠ **A launch-package defect found while verifying this**: `ttt_normalized_write` is a `PilotConfig` field, so `RUN3-LAUNCH.md` §4's `MEM="… ttt_normalized_write=true"` is **silently dropped** *and* makes run 3 a **second-differing-key** refusal. Reported to the Hub; it changes nothing in this prereg except that **P4b's premise is intact** (the flag is not on run 3). | `c3-gb-landing.md` §RECONCILIATION |

---

## 0. ⛔ KILL CONDITIONS — FIRST, before anything that could motivate ignoring them

| # | condition (measured) | action |
|---|---|---|
| **K0** ⭐⭐ ✅ **DISCHARGED (AM.1, 2026-08-13)** | **THE GEOMETRY IS NOT RULED.** §2 reports a conflict with three options (G-A / G-B / G-C) and hands it back. | ⛔ **No ladder arm trains until the Advisor picks one.** ⭐ Recommendation: **G-B** … → ⭐ **RATIFIED G-B (Head + Advisor 2026-08-13) and BUILT** (`store_layers`, spoke `c3-gb-landing`). The gate is open. |
| **K0b** ⭐ ⛔ **NEVER FIRES (AM.1)** | **ONLY IF G-A IS CHOSEN — the gate job** (§6.2): one `(clu_store, seed 0)` leg at G-A, 4,000 steps, vs the banked pilot-geometry leg at the same 4,000 steps. If `bpc_static(G-A) − bpc_static(pilot) > 0.03` | ⛔ **STOP the ladder.** … → **G-A was not chosen; this gate is inert and its 11 h are not spent.** |
| **K6** ⭐⭐ **NEW (AM.1)** | **THE STORE IS INERT AT SCALE.** On the first rungs' diagnostic (**§7.1**): oracle-addressed payload recovery `frac_recovered ≤ 0.05` **and** `depth_ratio < 0.10`, on **≥2 of 3 seeds**. | ⛔ **STOP before submitting the remaining jobs.** The store carries nothing at this geometry, so the other 12 jobs buy shell comparisons only. Re-scope to the **write** (φ's launch head / `erosion_partition`), where C2 Add.16 localized the blocker. ⛔ **Do NOT answer it by adding rivals** (K1's rule applies unchanged). |
| **K1** ⭐⭐ | After 3 seeds, the **memory dividend** `bpc(blank_store) − bpc(clu_store) < +0.02` on ≥2 of 3 seeds (i.e. **P1's null survives**) | ⛔ **STOP.** There is no tier-iii claim to make at any byte budget. Do **not** extend the ladder to the six pinned rivals — building them costs weeks and would measure the shell, not the memory. Re-scope to φ's launch head (C2 Add.16's localized blocker). |
| **K2** | `ttt_matched` NaNs with `ttt_normalized_write=False` **and** with `True` | ⛔ **STOP.** No two-sided system swap exists ⇒ the primary claim has no control (charter §2) and GRU cannot substitute (it cannot be matched both ways). |
| **K3** | Any single `(arm, seed)` job exceeds **96 h** | ⛔ **STOP and re-cost.** Credits change scheduling, never controls (charter §4); the answer is fewer steps, never fewer seeds and never a dropped control. |
| **K4** | `chlu/eval/byte_ledger.build_byte_ledger` raises `StateByteBudgetError` or `UnledgeredArmError` | ⛔ Already mechanical — the run refuses to train. Never resolved by `enforce_state_byte_budget=false`. |
| **K5** | The retention slice's `assert_non_degenerate` tripwire fires | ⛔ **STOP.** The slice is measuring character frequency, not retention (harness §2.2). |

⭐ **K1 is the one that matters and it is deliberately pointed at us.** The pre-registered expectation
(**P1**) is that the dividend is **zero**. If that survives, this ladder's honest output is a negative
result, and the kill condition exists so that a null cannot be answered with "add more rivals".

---

## 1. What this document freezes

1. the **C3 CLU store geometry** (§2) — ⛔ **NOT frozen: a reported conflict with three costed options
   and a recommendation**, with the measurements that could and could not decide it;
2. the **ceiling digit** (§3), with the admissible-window arithmetic verified against the harness's
   pinned rival table rather than re-derived from memory;
3. every **arm's pinned config, provenance, shrink knob and expected occupancy** (§4–§5);
4. the **CLU/TTT-matched byte match ratio** (§5.1);
5. the **job plan** (§6) — 15 jobs inside 2×A100/4-day, costed off **measured** A100 walls;
6. **numeric point predictions with declared falsifiers** (§7).

---

## 2. ⛔⛔ THE GEOMETRY — a REPORTED CONFLICT, three costed options, and a recommendation

### 2.1 The fact that reframes the question (arithmetic, verified by test)

`CluSystemConfig.n_atoms` is **not** `atoms_per_item × capacity`:

```
n_atoms = capacity * ceil( max( atoms_per_item*K , min_atoms , round(min_atoms_base * min_atoms_c**addr_dim) ) / capacity )
```

At the **ruled `addr_dim = 8`** the w23 dimension-aware term is `512 · √2⁸ = 512·16 =` **8192**, and the
pilot's `K·A = 32·256` **ties** it exactly. Therefore, at the pilot geometry:

| knob moved | resulting `n_atoms` | CLU total state B | occupancy of 2 MiB |
|---|---|---|---|
| pilot `K=32, A=256` | 8192 | **5,523,456** | 2.634× |
| `A: 256 → 64` | **8192 (unchanged)** | **5,523,456 (unchanged)** | 2.634× |
| `A: 256 → 16` | **8192 (unchanged)** | **5,523,456 (unchanged)** | 2.634× |
| `K: 32 → 128, A=16` | **8192 (unchanged)** | 5,578,752 | 2.660× |

⇒ ⛔ **The harness's own remedy — "Shrink the store (capacity / atoms_per_item / addr_dim+payload_dim /
n_layers) until it fits" — is unachievable through `capacity` or `atoms_per_item`. They are byte-inert.**
(`tests/test_c3_geometry_freeze.py::test_atoms_per_item_is_byte_inert_at_the_shipped_w23_floor`, 6 cases,
plus 4 for `capacity`.)

The four levers that *can* move the byte count, and why three of them are closed:

| lever | effect | status |
|---|---|---|
| `addr_dim` | floor ∝ √2^d — the dominant term | ⛔ **CLOSED**: Hub ruling R2 + C3 Add.1 §4 (N312). Stays 8. |
| `n_layers` | state ∝ `n_layers` | ⛔ **CLOSED**: 12 layers at `d_model 512` is what puts the model at **28.56 M params**, inside the 26–47 M class the whole venue is defined at. Dropping to 4 layers would fit the ceiling and leave the weight class. |
| `payload_dim` (`dim`) | state ∝ `dim+2` | ⛔ **CLOSED at 12**: `dim=12` *is* the `d ≤ 12` reach ceiling design rule; and even `dim → 3` cannot fit 8192 atoms under 2 MiB (`8192·5·4·12 = 1.97 MB` needs `dim ≤ 3`, i.e. `payload_dim ≤ −5`). Arithmetically impossible. |
| **`min_atoms_base`** (the floor's HEIGHT) | `n_atoms` directly | ⭐ **THE ONLY OPEN LEVER** — and opening it is §2.3's STOP. |

### 2.2 ⛔ THE CONFLICT, IN ONE TABLE — and the three ways out

**Constraint set:** ceiling ≈2 MB (ruled) · `addr_dim = 8` (ruled) · 12 layers at `d_model 512` (the
26–47 M weight class) · `dim = 12` (the `d ≤ 12` reach ceiling) · the **w23 atom floor** `512·√2^d_addr = 8192`
(a design rule, empirically anchored). ⛔ **These five cannot all hold at once:** the floor alone forces
`8192·14·4·12 = 5,505,024 B`, i.e. **2.63× the ceiling**, and §2.1 shows no remaining config knob touches it.

| | atoms **per store layer** | store layers | **CLU total state B** | occ. of 2 MiB | breaks | `s/step` (2×A100, projected) | TTT `η·n/d` | needs |
|---|---|---|---|---|---|---|---|---|
| **pilot (as landed)** | 8192 | 12 | 5,523,456 | **2.634×** ⛔ | the ceiling | 14.51 | 3.004 ⛔ | — |
| **G-A** config-only descent | **2048** | 12 | **1,394,688** | **0.665×** ✅ | ⛔ **the w23 floor, by 4×** — and §2.4 measures against it | **4.18** | **0.751** ✅ | a config value |
| ⭐ **G-B** *(RECOMMENDED → ✅ **RATIFIED + BUILT**, AM.1)* | **8192** (floor intact) | **3 of 12** — ⭐ **which 3 is §2.5's DESIGN DECISION: `(2, 6, 10)`** | **1,380,864** *(reproduced off built cells, AM.1)* | **0.658×** ✅ | **nothing** | **4.18** | 3.004 ⛔ ⇒ `ttt_normalized_write=True` becomes mandatory | ~~a `chlu/core/blocks.py` task~~ ✅ **LANDED** (`store_layers`, `c3-gb-landing`) |
| **G-C** half descent | 4096 | 12 | 2,770,944 | 1.321× ⚠ | the w23 floor by 2× (the *bottom* of its own measured band) **and** the ceiling must move into `[2,770,944 , 3,145,728)` | 7.62 | 1.502 ⚠ | config + an Advisor re-rule of the digit |

⭐ **G-B and G-A are byte-, compute- and envelope-IDENTICAL** (1,380,864 vs 1,394,688 B; both **4.18 s/step**;
both **29.7 h** worst job at 20,000 steps; both 3.23× headroom) — because the store's bytes *and* its
compute are both **per store-bearing layer**. The difference is entirely in what they cost epistemically:
G-A buys the reduction by **starving each cell**, G-B by **using fewer cells**. Nothing in the architecture
requires a memory in every layer.

⚠ **G-B's own price, stated:** it keeps the pilot's *per-layer* cell, so `solve_matched_ttt` returns
`(2197, 52)` again and the `ttt_matched` arm's inner loop is **divergent again** (`η·n/d = 3.004`). ⇒ under
G-B the pending Hub ruling on `ttt_normalized_write` must be **decided in favour of `True`** before the
ladder runs, or the rival column NaNs exactly as it did at step 135/4000. G-A gets that cure for free (§5.2).

### 2.3 The CONDITIONAL freeze, and what each option trades

⛔ **I decline to freeze G-A outright**, because §2.4's measurement — the only measurement anyone has of the
descent — does not support it. The conditional freeze, in the Advisor's hands:

```
IF the Advisor accepts a blocks.py task  ⇒  ⭐ G-B:     [✅ RATIFIED + BUILT 2026-08-13 — AMENDMENT 1]
   addr_dim 8 · payload_dim 4 (dim 12) · capacity 32 · atoms_per_item 256 · min_atoms_base 512 (UNCHANGED)
   n_layers 12, CLU cell in 3 of them   ⇒ n_atoms 8192/store-layer, total 1,380,864 B (0.658x)
   store_layers = (2, 6, 10)            ⇒ the placement DECISION, argued in §2.5 (not a default)
   + `ttt_normalized_write = True` (mandatory, see above)

ELSE (config-only, ship this week)      ⇒  G-A, GATED BY K0:
   addr_dim 8 · payload_dim 4 (dim 12) · capacity 32 · atoms_per_item 64 · store: min_atoms_base 128
   ⇒ n_atoms 2048, total 1,394,688 B (0.665x); ttt_matched (2235,13), eta*n/d 0.751
```

**What G-A trades, stated as the task requires.** It runs the store at **¼ of the w23 dimension-aware atom
floor at `d_addr = 8`**, whose own provenance (`chlu/config.py` L1518–1540) says d=8 "reaches strict 1.000
by **4096–8192**" and that `min_atoms_base = 512` pins it "**with margin**".
- The argument **for** was a code fact: the landed C3 runs set **`atom_place_radius = 0.3`** (H1b,
  `blocks.py`), which re-draws the *written slot's own atoms* into a ball around the incoming chunk's
  address at write time, so the near-site atom count becomes `atoms_per_item` — **dimension-free** — and
  the geometric-thinning mechanism the `√2^d` floor compensates for should be structurally absent.
- ⛔ **§2.4 measured that argument and it did not survive on one of two seeds.**

**What G-B trades:** the assumption that the memory belongs in every layer — a modelling choice never
tested — plus the `ttt_normalized_write` ruling. It trades **no design rule and no measured quantity.**

**What G-C trades:** half the descent (to the *bottom* of the floor's own measured adequacy band) **and**
the ceiling digit, which would have to move to a round number in `[2,770,944 , 3,145,728)`. ⚠ That window
is real and **no rival grows inside it** (§3.3), but it is above the Head+Advisor's "≈2 MB" and is
**theirs to rule, not mine to take.** It is also the worst on compute (7.62 s/step, 54.2 h/job).

### 2.4 The geometry sweep, arm 1 — the pre-registered grid

Pre-registered in `PREREG-GEOMETRY-SWEEP.md` **before it ran**; 22 runs, 2 seeds, real enwik8, smoke shapes.
Full curve in `geometry-sweep.json`. Verdicts against the filed predictions:

| # | prediction | outcome |
|---|---|---|
| **G1** | `total_state_bytes(2048) = 1,394,688 B = 0.6650×` | ✅ **HELD** exactly, and reproduced by the built cell (test). |
| **G2** | clu s/step ratio 8192:2048 ∈ [1.5, 4.0] | ✅ **HELD: 2.51×** (seed-mean 1.460 vs 0.582 s/step) — sub-linear in atoms, monotone across all six points (0.511 · 0.486 · 0.582 · 0.676 · 0.921 · 1.460). |
| **G3** ⭐ | write efficacy at 2048 within 2× of 8192, on both seeds | ⛔ **FALSIFIED on the filed wording, NOT DECISIVE at 4 seeds — see §2.4b.** In the pre-registered arm the instrument was **dead** (`depth_ratio` `1e-28 … 1e-110`, **no ordering in `n_atoms`**: §7.27 destroys the store at every geometry). A **declared second arm** with the landed write levers revived it; the falsifier fires on **1 of 4** seeds (18.37×) and one seed runs the *other* way (0.83×), while the 4-seed **median** curve is monotone (5.4× between 2048 and 8192). |
| **G4** | smoke bpc cannot rank geometries | ✅ **HELD, far more strongly than predicted:** static bpc is **bit-identical to 6 d.p. across a 16× atom range and across all four iso-byte `capacity` splits** (seed-mean **6.4835** at `n_atoms` 512/1024/2048/3072/4096/8192 and at `K` 16/32/64/128) while the cell genuinely differs (state **30,208 → 460,288 B**, params **8,602 → 116,122**). Still true in the deployed-write arm (**6.4831–6.4838**). ⇒ **the store's read is numerically inert in the loss at smoke scale.** |
| **G5** | `dim 16` costs 1.286× the bytes and buys nothing | ✅ bytes **exactly** 1,794,048 / 1,394,688 = **1.2864×**; the "buys nothing" half inherits G3's dead instrument in the filed arm. |
| **B** | (no prediction filed) | ⚠ **Iso-byte spend is inert too**: at fixed `n_atoms = 2048`, `K ∈ {16,32,64,128}` gives the *same* bpc to 6 d.p. and 1.10–1.40× the s/step. `K = 32` is the cheapest of the four. |

### 2.4b ⭐⭐ THE DECLARED SECOND ARM (4 seeds) — the measurement that cannot settle it

⚠ **Declared deviation, stated before its result is used:** mid-flight I found that the landed C3 runs set
**`atom_place_radius = 0.3` / `write_margin = 0.6`**, which the pre-registered grid did **not**. Since that
is the config the ladder would actually run — and since H1b is the very mechanism my G-A argument leans on
— I ran a **second, additional** arm of axis A with those levers (`--deployed-write`, `write_inner_steps`
scaled 40→8 for wall-clock, declared). ⛔ The filed arm was **not** replaced; both are reported.

⭐ **The levers revive the instrument:** `depth_ratio_vs_untrained` goes from `1e-28 … 1e-110` (dead) to
**O(0.1–1)** and becomes **ordered in `n_atoms`**. ⭐ **After the first two seeds gave a 13× spread I ran
two MORE seeds rather than reporting n = 2** (`geometry-sweep-deployed-write-s23.json`), because a single
seed was carrying the whole effect. **All four are reported:**

| `n_atoms` | occ. of 2 MiB | `depth_ratio` seed 0 · 1 · 2 · 3 | **median** | `qstar_spread` median | s/step |
|---|---|---|---|---|---|
| 512 | 0.173× | 0.0816 · 0.0719 · 0.0330 · 0.8619 | 0.0767 | 0.3371 | 0.87 |
| 1024 | 0.337× | 0.0641 · 0.0735 · 0.0295 · 0.7784 | 0.0688 | 0.0756 | 0.94 |
| **2048** *(G-A)* | 0.665× | 0.1342 · 0.0644 · 0.0676 · 0.5087 | **0.1009** | 0.2683 | 0.94 |
| 3072 | 0.993× | 0.1915 · 0.0945 · 0.0991 · 0.6686 | 0.1453 | 0.4171 | 1.09 |
| 4096 *(G-C)* | 1.321× | 0.1961 · 0.0883 · 0.1316 · 0.6810 | 0.1639 | 0.3600 | 1.25 |
| **8192** *(the w23 floor; G-B's cell)* | 2.634× | **2.4648** · 0.1199 · 0.0560 · 0.9795 | **0.5497** | 0.4180 | 2.34 |

**G3's verdict, both readings, because the pre-registration was written for 2 seeds:**
- ⛔ **On the filed wording ("*either* seed worse than 2×") it FIRES** — seed 0 gives **18.37×**.
- ⚠ **Per seed it fires on 1 of 4:** `18.37× · 1.86× · 0.83× · 1.93×`. **Seed 2 runs the other way**
  (8192 is *worse* than 2048 for that seed).
- ⭐ **On the median across 4 seeds the curve IS monotone from 2048 upward** — 0.1009 · 0.1453 · 0.1639 ·
  0.5497 — i.e. **more atoms is directionally better for the write**, by **5.4×** between 2048 and 8192.
- ⚠ **But no geometry deepens wells on the median** (every median < 1): only seed 0 at 8192 (2.46) and
  seed 3 at 8192 (0.98, break-even) do. The §7.27 erosion is present everywhere; this measures *relative*
  erosion, not health.
- ⚠ **The PAIRED signal (§7.26: depth alone is not a health signal) is much weaker**: `qstar_spread`
  medians are 0.337 · 0.076 · 0.268 · 0.417 · 0.360 · **0.418** — 8192 is highest but 3072 is level with
  it, and 512 already beats 2048.

⇒ ⭐⭐ **HONEST VERDICT: the measurement is DIRECTIONALLY consistent with the w23 floor and NOT DECISIVE.**
The median says more atoms help the write; the per-seed spread is larger than the effect; the paired signal
does not corroborate; and it is smoke scale (`d_model 32`, 2 layers, 60 steps, `write_inner_steps 8` not
the landed 40). ⛔ **It is nowhere near strong enough to license a 4× descent below a design-ruled floor,
and it is nowhere near strong enough to forbid one either.**

⭐ **That is exactly why the recommendation is G-B: when the measurement cannot settle the question, prefer
the option that does not need it settled.** G-B costs the same bytes, the same compute and the same
envelope as G-A while descending below nothing. G-A would be a bet on an unresolved instrument.

⛔ **Verdict on G4 at 4 seeds: unchanged and stronger.** `clu_store` bpc at 2048 vs 8192, per seed:
`6.4202/6.4205 · 6.5462/6.5471 · 6.5287/6.5287 · 6.4471/6.4472` — a 4× atom change moves the loss by
`≤ 0.001 bpc` on every seed. **The store's read is inert in the loss.**

⛔ **Consequence, stated plainly: the geometry could NOT be frozen on measured behaviour.** What the sweep
*did* decide: the byte arithmetic (G1), the 2.51× compute saving (G2), that smoke bpc is useless for
ranking geometries (G4, now at 4 seeds), and that iso-byte `capacity` splits are inert (B). What it could
**not** decide: whether the w23 floor's height is real at `d_addr = 8` — the pre-registered falsifier fires
on 1 seed of 4 and reverses on another. **⛔ That is why the geometry is handed back rather than frozen.**

⚠ G4 has a second consequence the ladder must carry: **`bpc(none) − bpc(clu_store)` is not a memory
dividend.** At smoke scale the store is inert and the gap vs `none` is still +0.24…+0.35 bpc, i.e. purely
architectural. **The memory claim is `bpc(blank_store) − bpc(clu_store)` and nothing else** (P1).

### 2.5 ⭐⭐ **NEW (AMENDMENT 1)** — THE LAYER PLACEMENT IS A DESIGN DECISION, AND HERE IS THE ARGUMENT

⛔⛔ **The ceiling chose HOW MANY store layers. It did not choose WHICH, and it must not be allowed to.**
A reader who can derive the placement from the byte budget alone is reading a number that was fitted, not
designed. So: the count (3) comes from §2.2's arithmetic; the *placement* comes from the rule below, and
the rule is stated **before** any arm trains so that it can be wrong.

**THE DECISION: `store_layers = (2, 6, 10)`** (0-indexed, of 12) — the pilot's per-layer cell, unchanged,
in those three blocks; the other nine keep the **identical** shell (conv, φ, assimilation, MLP, norms,
residual) with a null cell in the slot. It is written on the launch line (§6.4) and emitted into every
artifact's flag table and byte ledger (`n_store_layers`, `store_layer_indices`).

**The rule that generates it**, stated generally so that a different count does not re-open the argument:

> *period ⌊12/n⌋, offset so that **at least two blocks sit below the first store layer** and **at least one
> block sits above the last**.*

At `n = 3` that is `(2, 6, 10)`. ⭐ If the ceiling ever admits `n = 4`, the rule gives `(2, 5, 8, 11)` →
and its own "≥1 above" clause forces `(1, 4, 7, 10)`; the placement is **derived, not re-fitted**.

**Why each clause:**

1. **≥2 blocks below the first store layer.** φ maps a *pooled chunk summary* to an address. At layer 0
   that summary is one depthwise conv and one LayerNorm away from the token embedding, i.e. an address
   space that is close to a bag-of-bytes; the store would be keyed on surface form. Two blocks of mixing
   first is the cheapest way to make the address a *composed* representation. ⚠ This is an argument, not a
   measurement — **P11 makes it falsifiable.**
2. **≥1 block above the last store layer.** The block's read is shifted by one chunk and enters the
   residual stream through `assim`; if the last store sits at layer 11 the retrieved content reaches only
   `norm_f` and the head, so the memory becomes a re-ranker of the final representation rather than
   something the network can *compute with*. ⛔ This is why the placement is `(2, 6, 10)` and **not**
   `(3, 7, 11)`, which has the same period.
3. **Spread, not clustered.** Adjacent blocks' chunk summaries are strongly correlated, so three *adjacent*
   stores would hold three near-copies of one memory — paying 3× the bytes for ~1× the content. Period 4
   maximises representational distance between the three at fixed count.

**Alternatives considered and rejected** (all cost the identical bytes and compute — this is a purely
epistemic choice):

| placement | rejected because |
|---|---|
| `(0, 1, 2)` contiguous-early | violates clauses 1 and 3: near-raw addresses **and** three correlated copies. ⛔ It is also the *default one would get for free* by writing `range(3)`, which is exactly the byte-fitting default this section exists to forbid. |
| `(9, 10, 11)` late | violates clause 2: nothing above to integrate the read; the memory can only re-rank. |
| `(3, 7, 11)` | same period, violates clause 2 at layer 11. |
| `(0, 4, 8)` | violates clause 1 (layer-0 address space); otherwise the closest competitor. |
| `(5, 6, 7)` middle-clustered | violates clause 3. |

**Precedent, named.** Periodic/hybrid placement of the global-context layer — rather than putting one in
every block — is the standard design of the recurrent-hybrid literature (Griffin, Jamba, Samba,
Zamba-style interleaving; "attention at ≈¼, ½, ¾ depth"). ⭐ **The novelty here is not the periodicity, it
is that the interleaved layer is an addressable store rather than attention**, and that the count is set by
a *pre-registered byte ceiling* rather than by taste. ⚠ Nothing in that literature says 3-of-12 is optimal,
and this prereg does not claim it is: it claims the placement is *argued and falsifiable*.

**⭐ What placement is PRE-REGISTERED to buy or cost** (predictions, falsifiers in §7/§7.1):

- **P10** *(conditional — see below)*: `bpc(2,6,10) − bpc(0,1,2)` at 4,000 steps is **0.000 ± 0.010 bpc**.
  Derivation: G4 (the store's read is numerically inert in the loss over a 16× atom range) and P1 (blanking
  the store costs ≤0.0013 bpc on 4/4 banked legs) together say the *contents* are not in the loss; if the
  contents are not, their *depth in the stack* cannot be either. ⛔ **FALSIFIER: |Δ| > 0.02 bpc.** ⭐ Its
  firing would be **good news twice over** — placement matters *and* the read is in the loss, which partly
  falsifies P1.
  ⚠ **Conditional by design, and this is a cost control, not a hedge:** the contrast job is run **only if
  §7.1's diagnostic says the store is in the loss at all** (L3 fires). Against an inert read the contrast
  measures one null against another and spends 11 h to do it.
- **P11**: in §7.1's oracle-addressed recovery, `frac_recovered` is **monotone non-decreasing in depth**
  across the three store layers (`layer 2 ≤ layer 6 ≤ layer 10`), because clause 1 asserts that composed
  addresses are better addresses. ⛔ **FALSIFIER: layer 2 strictly highest on ≥2 of 3 seeds** ⇒ clause 1 is
  wrong, the rule must be re-derived (and the natural repair is `(0, 4, 8)`).
- **P12** *(cost, not capability)*: the ladder's measured `s/step` at G-B is **4.18 ± 1.5** (§6.1) and the
  plan pass's share is **not** 3/12 of the pilot's — the host-side controller is skipped only on the nine
  null-cell layers, while φ, conv and the MLP still run in all twelve. ⛔ FALSIFIER: `s/step > 8` (⇒ **K3**).

---

## 3. ⭐ THE CEILING DIGIT

### 3.1 The ruled principle (Head+Advisor 2026-08-13), applied

> a **round** number that **both swap members fit under naturally**, with **no arm sitting exactly on it**,
> **all others shrink-to-match**, and **occupancy reported** per arm. Convention: **TOTAL state bytes, AS
> DEPLOYED — ⛔ no dtype normalisation.**

### 3.2 The admissible window — VERIFIED against the harness, not re-derived

`rival_reference_table()` re-run on this branch reproduces the task's pinned table **to the byte**
(artifact `rival-table-verified.json`):

| rival (natural, bf16, layer-summed) | bytes | occupancy of 2 MiB | shrink knob → value | after shrink |
|---|---|---|---|---|
| **ttt_linear** ⭐ *swap member* | 1,597,440 | **0.762×** | — (fits) | — |
| gated_deltanet2 | 3,145,728 | 1.500× | `n_heads` 4 → **6** | 2,097,152 |
| transformer_xl | 6,291,456 | 3.000× | `mem_len` 512 → **170** | 2,088,960 (0.996×) |
| mamba2 | 6,475,776 | 3.088× | `d_state` 128 → **39** | 2,075,616 (0.990×) |
| sliding_window | 12,582,912 | 6.000× | `window` 512 → **85** | 2,088,960 (0.996×) |
| ttt_mlp | 12,705,792 | 6.059× | `head_dim` 64 → **25** | 1,968,000 (0.938×) |

**Window arithmetic:**
`ceiling ≥ max(CLU_frozen, TTT_matched_frozen, ttt_linear) = max(1,394,688 ; 1,394,640 ; 1,597,440) = 1,597,440`
`ceiling < gated_deltanet2 = 3,145,728` (so GDN-2 and everything above it **shrinks**, never grows).
⇒ **admissible window = [1,597,440 , 3,145,728).**

### 3.3 ⭐ THE DIGIT: **2 MiB = 2,097,152 B — CONFIRMED, unchanged**

`chlu/eval/byte_ledger.MATCHED_STATE_BYTE_BUDGET` already holds this value. ⭐ **No edit is required**,
which closes reconciliation item 4 of `c3-csf3-harness.md` ("built against the recommendation; one-line
edit if it differs" — **it does not differ**). ⛔ I did not touch that file (it is `c3-run3-budget-exemption`'s).

It satisfies every clause of the ruling:
- **round** — a power of two, and the field's own unit;
- **both swap members fit naturally** — under **either** G-A or G-B: `clu_store` **0.665× / 0.658×**,
  `ttt_matched` **0.665× / 0.658×**, and the published-config `ttt_linear` **0.762×**; ⛔ none was shrunk
  to get there. (Under G-C the CLU arm is 1.321× and the digit would have to move — §2.2.);
- **no arm sits exactly on it** — no *natural* value equals 2,097,152. ⚠ **Disclosed:** GDN-2's *shrunk*
  value lands on it exactly, because `24·512²/6 · 2 B = 2,097,152` is an exact powers-of-two coincidence.
  That is the shrink solution, not a tuned ceiling; it is stated here so a reviewer meets it in our text.
- **all others shrink-to-match** — five of six rivals shrink; the solved knob values are in §3.2 and come
  from `shrink_to_budget()`, not by hand;
- **occupancy reported per arm** — §2.2 and §3.2.

⛔ **Not chosen and why:** widening to `[2,770,944 , 3,145,728)` would accommodate **G-C** (`n_atoms = 4096`)
**without making any rival grow** — the window is genuinely admissible under the task's own arithmetic —
but it moves off the Head+Advisor's "≈2 MB" and buys the CLU arm 1.32× more state than the swap member it
is matched against. ⛔ **It is the Advisor's to take, not mine**, and only as §2.2's G-C.

⭐ **The digit is INDEPENDENT of which geometry option is chosen, except G-C.** G-A and G-B both land at
0.66× of 2 MiB, so §3.3 stands whichever the Advisor picks between them; only G-C forces a re-rule.

---

## 4. Pinned arm configs, provenance, shrink knobs

### 4.1 The five arms the ladder TRAINS

| arm | config | provenance | state B (total) | occupancy |
|---|---|---|---|---|
| `clu_store` | §2.2/§2.3 geometry (**G-A or G-B**), all stage levers ON, `atom_place_radius 0.3`, `write_margin 0.6`, `write_inner_steps 40` | **this prereg** (geometry) + landed run-2 `flags` (write levers) | **G-A 1,394,688 / G-B 1,380,864** | **0.665× / 0.658×** |
| `ttt_matched` | **G-A:** `(k, n) = (2235, 13)`, `ttt_normalized_write = False`. **G-B:** `(k, n) = (2197, 52)` in the same 3 layers, ⛔ **`ttt_normalized_write = True` MANDATORY** (§2.2) | solved by `solve_matched_ttt` from the CLU cell's ledger — ⛔ **not a free choice** | **G-A 1,394,640 / G-B 1,370,928** | 0.665× / 0.654× |
| `gru_matched` | **G-A** `hidden = 158`, **G-B** `hidden = 229` (**params**-matched, per store layer) | `solve_matched_gru`; ⚠ the state-matched GRU is `hidden = state_floats` ⇒ **billions of params**, published as arithmetic only — the one-sidedness is the D2 finding, not an omission | 7,584 / 2,748 | 0.004× / 0.001× |
| `none` | null cell | the floor | 0 | 0 |
| `echo` | echo cell | the trivial-substitute **laundering control** | 0 | 0 |
| **`dyneval` column** ⭐ | `dyneval_lr` grid, **per arm**, its own column | Krause et al. 2019 (method); ⛔ their **0.94 bpc is at 277 M** and is never printed beside ours | — (parameter update, not state) | — |

### 4.2 The six rivals the ladder LEDGERS but does **NOT** build — ⛔ DECLARED NOT-RUN

⛔ **No implementation of Mamba-2, GDN-2, TTT-Linear, TTT-MLP, Transformer-XL or sliding-window attention
exists in this repository.** `RIVAL_SPECS` is a *ledger*, and its own docstring says so ("nothing here is
measured from a built model, because we do not build the rivals"). Their pinned configs, provenance
(paper table vs official implementation, incl. the `flash-linear-attention` 3× TRAP-2 row) and solved
shrink knobs are §3.2 and are carried into every artifact.

⚠ **This is a scope statement the Advisor should see explicitly:** charter §2's primary claim names "tuned
rivals (real Mamba/Mamba-2 · Gated DeltaNet-2 · sliding attention)". This ladder does **not** deliver them.
It delivers the **two-sided system swap** (CLU ↔ TTT-class cell) at matched params *and* matched state
bytes, plus the null/echo/GRU controls and the dyn-eval column. Building the six rivals is a separate,
multi-week engineering task, and **K1 exists so it is not started before the swap says there is anything
to defend.**

### 4.3 ⭐⭐ **NEW (AMENDMENT 1)** — THIS IS **PHASE 1**, AND ⛔ **PHASE 1 IS NOT THE CLAIM**

⛔⛔ **Read this before quoting any table produced under this document.**

**What phase 1 trains** (the 15 jobs of §6.2, at geometry G-B): `clu_store` · `ttt_matched` (the two-sided
system swap) · `gru_matched` · `none` · `echo` (the trivial-substitute laundering control), each with its
**dyn-eval substitute column** and the **within-document retention/revisit slices**, plus (on `clu_store`)
the **blank-store control**, the **D5 anytime curve** and — new in AM.1 — the **store-liveness diagnostic**
(§7.1).

**What phase 1 does NOT train:** ⛔ **the six pinned rivals — Mamba-2, Gated DeltaNet-2, TTT-Linear,
TTT-MLP, Transformer-XL, sliding-window attention. None of them is implemented in this repository**
(§4.2). `RIVAL_SPECS` is a **ledger**, and its own docstring says so. Their bytes, provenance and solved
shrink knobs are carried into every artifact so that the ceiling is auditable — **that is a byte table, not
a result.**

> ⛔⛔ **THEREFORE: charter §2's tier-iii PRIMARY CLAIM — "matched-bytes, matched-params, against TUNED
> RIVALS (real Mamba/Mamba-2 · Gated DeltaNet-2 · sliding attention)" — WAITS FOR PHASE 2. NO PHASE-1
> NUMBER, TABLE, FIGURE OR ABSTRACT SENTENCE MAY BE QUOTED AS IT.** Phase 2 is funded and scoped in
> parallel and is a separate, multi-week implementation task.

**What phase 1 *can* honestly support, stated positively so the boundary is usable:**

| ✅ phase 1 may say | ⛔ phase 1 may NOT say |
|---|---|
| "at matched params **and** matched state bytes, swapping the CLU store for a TTT-class cell in the identical shell changes bpc by X" (the **two-sided system swap**) | "the CLU beats Mamba-2 / GDN-2 / sliding attention at 2 MiB" — **none of them was run** |
| "the memory dividend `blank_store − clu_store` is X" (the P1 quantity) | "the CLU is state-of-the-art at this budget" |
| "the store is / is not addressable at scale" (§7.1) | any leaderboard positioning against the published enwik8 grid (⛔ 7.42: we are token-bound at ~1.8 epochs vs their ~50; the category error is already barred) |
| "the ladder's own controls (`none`, `echo`, `gru_matched`, dyn-eval) rank as follows" | "…therefore the primary claim holds" |

⭐ **Why this section exists at all.** A phase-1 table with five arms, a byte ledger and a rival reference
table *printed beside it* looks exactly like a matched-bytes rival comparison. Someone will quote it as
one — that is the predictable failure, and naming it in advance is cheaper than retracting it later (this
program has already spent two waves retracting one un-owned number). ⛔ **The rival reference table must
never be printed in the same table as our arms' bpc.** Bytes and bpc in one grid is the whole confusion.

⚠⚠ **ONE THING PHASE 2 MUST NOT INHERIT FROM G-B, flagged now because the rival arms are being built
concurrently.** `store_layers` applies to **every arm of the SWAP** — `clu_store`, `ttt_matched`,
`gru_matched`, `none`, `echo` — because the two-sided match is *per store-bearing layer* and the trivial
substitutes must occupy exactly the layers the store occupies. ⛔ **It must NOT be applied to a published
rival arm.** A Mamba-2 / GDN-2 / sliding-window layer is the block's **sequence mixer**, not a memory in a
slot; running one in 3 of 12 blocks with null cells between them is not that model. The matched-bytes
control for a rival is the one already built: **keep its 12 layers and shrink its own declared knob**
(`shrink_to_budget`, §3.2). ⛔ Byte-matching by *deleting nine of a rival's layers* would be hobbling, and
the anti-hobbling rule has already inverted one verdict in this program.

**K1 governs the transition.** If P1's null survives 3 seeds, ⛔ **phase 2 does not start**: building six
rivals to compare against a memory that is not in the loss would measure the shell. That is K1's existing
rule, restated here because §4.3 is where someone will look for permission to proceed.

---

## 5. The two-sided match, pre-registered

### 5.1 ⭐ CLU / TTT-matched byte match ratio

| geometry | CLU total | TTT total | **ratio** |
|---|---|---|---|
| pilot (`n_atoms 8192`) | 5,523,456 | 5,483,712 | 1.0072× |
| **G-A (`n_atoms 2048` ×12)** | **1,394,688** | **1,394,640** | **1.0000×** (1.0000344) |
| **G-B (`n_atoms 8192` ×3)** | **1,380,864** | **1,370,928** | **1.0072×** (the pilot's, per-layer-preserved) |

⇒ **the two-sided match survives under either option**, and under G-A it *improves* (1.0072 → 1.0000),
because `solve_matched_ttt` re-solves against the smaller cell. Params match to **1.00046×** at G-A.
⛔ Pre-registered: the ladder's ledger must print **1.0000× (G-A)** or **1.0072× (G-B)**; anything outside
`[0.99, 1.01]` means the solve moved and the swap is no longer two-sided.

### 5.2 ⭐ Under G-A only, the shrink also moves the TTT arm's inner-loop stability — **partially**

`MatchedTTTCell`'s update is non-expansive only while `η‖θ_K z‖² < 2`, and that product is `η·n/d`, a pure
function of the **solved** geometry (handover 7.30). Measured on this branch
(`ttt-stability-at-frozen-geometry.json`):

| geometry | (k, n) | analytic `η·n/d` | mean over 512 unit-RMS z | frac > 2 | worst direction | ‖W‖ over 16 writes |
|---|---|---|---|---|---|---|
| pilot — **and G-B, which keeps the per-layer cell** | (2197, 52) | **3.004** ⛔ | 2.993 | **76.2 %** | 6.182 | **×4.4e10** |
| **G-A** | (2235, 13) | **0.751** ✅ | 0.923 | **4.3 %** | **2.726** ⚠ | ×2.3e3 |
| (G-C, for reference) | (2220, 26) | 1.502 ⚠ | 1.627 | 26.0 % | 3.777 | ×3.3e6 |

⇒ **under G-A the divergence that NaN'd the rival column at step 135/4000 is reduced ~7 orders of
magnitude by geometry alone**, with `ttt_normalized_write` still at its shipped `False`. ⚠ **The cure is
partial and that is asserted in a test, not buried**: the worst-direction criterion is still `2.726 > 2`,
so a sufficiently coherent chunk stream can still amplify. **P4** is the prediction; the built-and-gated
`ttt_normalized_write=True` is the declared fallback if it fires.

⛔⛔ **Under G-B the divergence RETURNS UNCHANGED** — G-B shrinks the *number of cells*, not the cell, so
`solve_matched_ttt` sees the identical per-layer ledger and returns `(2197, 52)`, `η·n/d = 3.004`,
‖W‖ ×4.4e10 over 16 writes. ⇒ **choosing G-B means ruling `ttt_normalized_write = True` in the same
breath**; it is not optional there, and it is a claims-relevant flip of a published rival column
(handover 7.30). **That is G-B's real price and it belongs beside its zero design-rule cost.**

---

## 6. The job plan — 15 jobs, costed off MEASURED A100 walls

⭐ **Not the scout's assumed MFU — ours, measured.** Every phase second comes from the landed CSF3
artifact `csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json` (`host_rss` phase marks; 2×A100; JAX 0.9.0/gpu;
`steps=4000`, `eval_batches=dyneval_batches=40`, `write_inner_steps=40`, `plan_workers=8`).
⛔ Runs 1/2 are the pre-C3, over-budget geometry (C3 Add.1 §2) and are used **only as a timing basis** —
no bpc of theirs appears in any table of this ladder.

**Measured per-phase walls, 2×A100, pilot geometry:** `clu_store` train **58,025 s**, dyneval **59,673 s**,
static 319 s, blank_store 476 s, gradient probes 2,077 + 2,273 s; `gru_matched` train 4,265 s;
`ttt_matched` 2,643 s; `none` 2,968 s; `echo` 3,822 s.

### 6.1 ⭐ THE MEASURED MFU — and the reconciliation the scout's number needs

| arm | s/step (measured) | achieved FLOP/s (`C ≈ 6ND`, N = 28,556,792, D = 8,192 tok/step) | **MFU vs 2×A100 bf16 peak** |
|---|---|---|---|
| `clu_store` @ pilot geometry | **14.51** | 9.68 × 10¹⁰ | **0.0155 %** |
| `none` (null arm) | **0.742** | 1.89 × 10¹² | **0.303 %** |
| `clu_store` @ **frozen** geometry (projected) | **4.18** | 3.36 × 10¹¹ | **0.0538 %** |

⛔ **The scout's most pessimistic assumption (3 % MFU → 18 h/arm) is optimistic by ~200× for the CLU arm
and ~10× for a null arm of this shape.** Two structural reasons, both measured: the CLU block's cost is a
multi-step integrator over an atom dictionary (not counted by `6ND`, which is why MFU is the wrong
denominator for it), and the plan pass is **host-side Python** — `plan_pass_frac` is 0.166 on the CLU arm
and **0.88–0.95 on every other arm**, i.e. the cheap arms are not GPU-bound at all.

⭐ **The reconciliation: we are TOKEN-bound, not compute-bound, and the scout's costing assumed the wrong
budget.** The scout priced `D = 5×10⁹` bytes (≈55 epochs). At the measured throughput that is **700 h** for
one CLU arm. The pilot ran `4000 × 8192 = 32.8 M` tokens = **0.36 epochs**.

### 6.2 The plan (`scripts/c3_ladder_plan.py`, re-runnable)

**5 arms × 3 seeds = 15 jobs**, one `(arm, seed)` per array task, `#SBATCH -a 0-14%4`, via the harness's
`scripts/csf3/job_gpu_c3_seeds.sh` (resume-first; each arm its own `--out`; `.eqx` precondition checked in
the job script **and** in `run_pilot` via `resume_require_ckpt`).

| geometry | steps | tokens (epochs) | worst job | all fit 96 h? | makespan @%4 |
|---|---|---|---|---|---|
| pilot 8192 ×12 layers | 4,000 | 32.8 M (0.36) | 38.6 h | ✅ (2.49×) | 1.61 d |
| pilot 8192 ×12 layers | 20,000 | 164 M (1.82) | **103.1 h** | ⛔ **NO** | 4.29 d |
| ⭐ **G-B** 8192 × **3** layers | **20,000** | **164 M (1.82)** | **29.7 h** | ✅ **(3.23×)** | **1.70 d** |
| **G-A** 2048 ×12 layers | **20,000** | **164 M (1.82)** | **29.7 h** | ✅ **(3.23×)** | **1.70 d** |
| G-C 4096 ×12 layers | 20,000 | 164 M (1.82) | 54.2 h | ✅ (1.77×) | 2.46 d |
| G-A or G-B | 40,000 | 328 M (3.64) | 53.0 h | ✅ (1.81×) | 3.04 d |

⭐ **Either reduction is what makes a ≥1.8-epoch ladder possible at all**: at the landed pilot geometry
20,000 steps **does not fit one job**. **That is the trade, stated as the task requires: a 4× reduction in
store state — by starving cells (G-A) or by using fewer of them (G-B) — buys 5× more training tokens
inside the same envelope.** ⛔ G-B and G-A are indistinguishable on every number in this table; they differ
only in what they cost epistemically (§2.2).

**⭐ THE GATE JOB (K0b), ONLY IF G-A IS CHOSEN, run alone before the other 14:** `(clu_store, seed 0)` at
G-A, **4,000 steps** — deliberately the pilot's step count so it is directly comparable to the banked leg.
Projected **11.1 h**. Its only job is to answer "did the 4× atom shrink cost anything?" (**P7**).
⛔ Under G-B no such gate is needed for the *atoms* (the floor is intact), but `ttt_normalized_write=True`
must be ruled first (§2.2).

**Emitted per arm, per seed, non-negotiable:** the byte ledger (all 5 arms in every artifact, φ accounted),
the **retention/revisit slices** for the static column **and** the dyn-eval column with the three validity
controls, the dyn-eval substitute column, and (clu_store) the blank-store control + D5 anytime curve.

### 6.3 The literal launch — ⚠ ONE LITERAL COMMAND LINE PER SUBMISSION (zsh does not word-split)

```bash
# 0. stage serially, once (array tasks are HARD-BLOCKED from downloading)
sbatch --export=ALL,CORPUS=enwik8,STAGE_ONLY=1 -p serial -t 0:30:00 scripts/csf3/job_gpu_c3_seeds.sh

# 1. ⭐ THE K0b GATE JOB — ALONE, FIRST, AND ONLY UNDER G-A. (clu_store, seed 0), 4,000 steps (~11 h)
sbatch -a 0-0 -t 1-00:00:00 --export=ALL,CORPUS=enwik8,SCALE=pilot,STG=s4,N_SEEDS=1,SEED_BASE=0,D5=1,SLICES=1,ARM_LIST=clu_store,OUT_BASE=.claude/outputs/c3-ladder-gate,STORE="min_atoms_base=128 write_margin=0.6",MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true",SET="atoms_per_item=64 steps=4000 warmup=200 slice_batches=10 monitor_every=25 plan_workers=8 liveness_lanes=1" scripts/csf3/job_gpu_c3_seeds.sh

# 2. ONLY IF K0b PASSES: the G-A ladder, 15 tasks (5 arms x 3 seeds), <=4 concurrent, 20,000 steps
sbatch -a 0-14%4 -t 2-00:00:00 --export=ALL,CORPUS=enwik8,SCALE=pilot,STG=s4,N_SEEDS=3,SEED_BASE=0,D5=1,SLICES=1,OUT_BASE=.claude/outputs/c3-ladder,STORE="min_atoms_base=128 write_margin=0.6",MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true",SET="atoms_per_item=64 steps=20000 warmup=1000 slice_batches=10 monitor_every=25 plan_workers=8 liveness_lanes=1" scripts/csf3/job_gpu_c3_seeds.sh

# 3. verify BEFORE walking away
ls -l logs/clu-c3-*.out | wc -l          # expect 15
ls .claude/outputs/c3-ladder/*/ckpt_*_seed*.eqx | wc -l
```

⛔ **These two lines are the G-A launch. G-B has NO launch line yet** — it needs the `chlu/core/blocks.py`
change (a `store_layers` selection on `StreamModel`) that this task does not own, plus its own
`ttt_normalized_write=True`. **That is the work item the hand-back creates.**
→ ✅ **SUPERSEDED BY §6.4 (AMENDMENT 1): the work item is DONE and G-B's launch line now exists.**

### 6.4 ⭐ **NEW (AMENDMENT 1)** — THE G-B LAUNCH, and what is different about it

⚠ **ONE LITERAL COMMAND LINE PER SUBMISSION** (zsh does not word-split — §7.45).
⭐ Differences from §6.3's G-A lines, all of them consequences of the ratification:
`store_layers=2,6,10` **replaces** `atoms_per_item=64` + `min_atoms_base=128` (the floor is intact — that
is the point of G-B); and `ttt_normalized_write=true` is **SET**, because G-B keeps the pilot's per-layer
cell so `η·n/d = 3.004 ≥ 2` fires (§5.2, and PILOT-TTT-RULINGS ruling 1 already rules the flip).
⛔ **There is no K0b gate job under G-B** — the atom floor was never descended, so there is nothing to gate.

```bash
# 0. stage serially, once (array tasks are HARD-BLOCKED from downloading)
sbatch --export=ALL,CORPUS=enwik8,STAGE_ONLY=1 -p serial -t 0:30:00 scripts/csf3/job_gpu_c3_seeds.sh
```

```bash
# 1. ⭐ THE FIRST RUNGS: (clu_store, seed 0..2) ALONE, 20,000 steps, with the store-liveness
#    diagnostic (§7.1) read off their 4,000-step checkpoints BEFORE the other 12 jobs are submitted.
sbatch -a 0-2%3 -t 2-00:00:00 --export=ALL,CORPUS=enwik8,SCALE=pilot,STG=s4,N_SEEDS=3,SEED_BASE=0,D5=1,SLICES=1,ARM_LIST=clu_store,OUT_BASE=.claude/outputs/c3-ladder,STORE="write_margin=0.6",MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true",SET="store_layers=2,6,10 ttt_normalized_write=true steps=20000 warmup=1000 slice_batches=10 monitor_every=25 plan_workers=8 liveness_lanes=1" scripts/csf3/job_gpu_c3_seeds.sh
```

```bash
# 2. ⛔ ONLY IF §7.1's K6 DID NOT FIRE: the remaining 12 tasks (4 arms x 3 seeds), <=4 concurrent
sbatch -a 0-11%4 -t 2-00:00:00 --export=ALL,CORPUS=enwik8,SCALE=pilot,STG=s4,N_SEEDS=3,SEED_BASE=0,D5=1,SLICES=1,ARM_LIST="ttt_matched gru_matched none echo",OUT_BASE=.claude/outputs/c3-ladder,STORE="write_margin=0.6",MEM="atom_place_radius=0.3 write_inner_steps=40 remat_chunks=true",SET="store_layers=2,6,10 ttt_normalized_write=true steps=20000 warmup=1000 slice_batches=10 monitor_every=25 plan_workers=8 liveness_lanes=1" scripts/csf3/job_gpu_c3_seeds.sh
```

```bash
# 3. verify BEFORE walking away
ls -l logs/clu-c3-*.out | wc -l
grep -h '"n_store_layers"' .claude/outputs/c3-ladder/*/*.json | sort | uniq -c   # expect 3, never 12
```

⛔ **`store_layers` and `ttt_normalized_write` are non-default, so `as_flag_table()` emits them and the
ladder's resume fingerprint differs from every banked run-1/2/3 journal.** That is correct and intended —
it is a different geometry — but it means **no banked leg can be resumed into this ladder** and a fresh
`OUT_BASE` is mandatory (given above). ⚠ It is also why the ladder **cannot** ride run 3's
pre-registered-continuation exemption: it is not run 2 plus one flag, and the exemption refuses it **by
name** (asserted in `tests/test_c3_gb_geometry.py`). The ladder needs the **ceiling prereg** instead —
this document — which is the edit that flips `BUDGET_IS_INTERIM`.

⭐ **The ledger check that proves the geometry is the one you asked for**, on the first artifact:
`byte_ledger.arms.clu_store` must read `n_store_layers: 3`, `store_layer_indices: [2, 6, 10]`,
`total_state_bytes: 1380864`, `occupancy: 0.65845`, `within_budget: true`.
⛔ If it reads `n_store_layers: 12`, the `--set` did not land — **kill the job**, it is the wrong geometry.

⚠ **`atoms_per_item = 64` and `min_atoms_base = 128` are non-default**, so `as_flag_table()` emits them and
the ladder's resume fingerprint **differs from every banked run-1/2/3 journal**. That is correct and
intended — it is a different geometry — but it means ⛔ **no banked leg can be resumed into this ladder**,
and a fresh `OUT_BASE` is mandatory (given above). The `.eqx` precondition check
(`resume_require_ckpt=True` + the job script's pre-submit check) still guards every *re*-resume.

⚠ **Largest unmeasured item, flagged not buried: the slice phase.** It did not exist when the pilot legs
ran. It is priced as one `static`-shaped + one `dyneval`-shaped pass at `slice_batches`; since the measured
`dyneval` is **59,673 s**, running slices at the full `eval_batches = 40` would add ~16.6 h to the CLU job.
⇒ **`slice_batches = 10` is pinned here** (¼ of the eval budget, ≈4.2 h at the frozen geometry). If the
first job's slice phase exceeds 6 h, cut to 5 and record it.

---

## 7. ⭐ NUMERIC POINT PREDICTIONS, WITH FALSIFIERS

⚠ **Derivation basis.** P1–P3, P5 are derived from the four banked CSF3 legs (runs 1 & 2, seeds 0/1) —
⛔ **pre-C3, over-budget geometry, never quotable as budget-compliant ladder rows** (C3 Add.1 §2). They are
used here for exactly what a prereg needs: a prior committed to *before* the new measurement.

⚠ **Scope of the predictions.** P1–P3, P5–P6, P8–P9 are **geometry-independent** — they hold under G-A or
G-B alike, because §2.4b's own G4 says the store's contents are not in the loss and §6 says the two options
cost the same. **P4 and P7 are G-A-specific and are relabelled below.**

| # | prediction (20,000 steps, 3 seeds, enwik8 test split) | derivation | ⛔ FALSIFIER |
|---|---|---|---|
| **P1** ⭐⭐ | **The memory dividend is ZERO: `bpc(blank_store) − bpc(clu_store) = 0.000 ± 0.005` bpc.** | The four banked legs measured `+0.001158, +0.001276, −0.000263, −0.000139` (mean **+0.00051**) — blanking the store changes nothing. C2 Add.16 localized the blocker to φ's launch head, and the shrink does not touch φ. | **≥ +0.02 bpc on ≥2 of 3 seeds.** ⭐ *This falsifier firing is the GOOD outcome — it is the first evidence the store is in the loss at all.* |
| **P2** | The gap to the null is architectural: `bpc(none) − bpc(clu_store) ∈ [0.03, 0.08]` bpc | banked: 0.0512, 0.0458, 0.0406, 0.0479 | outside `[0.03, 0.08]` |
| **P3** | **`gru_matched` beats `clu_store` on static bpc on ≥2 of 3 seeds** | banked: 4 legs of 4, by 0.015–0.026 bpc | `clu_store` beats `gru_matched` on ≥2 of 3 ⇒ report it loudly; it would be the first such reversal |
| **P4** *(G-A only)* | `ttt_matched` completes 20,000 steps **without NaN on all 3 seeds** at `ttt_normalized_write=False` | §5.2: `η·n/d` 3.004 → **0.751**, ‖W‖ growth ×4.4e10 → ×2.3e3 | any NaN ⇒ flip `ttt_normalized_write=True` (built, gated) and re-run **that arm only**; record the flip as a claims-relevant config change |
| **P4b** *(G-B only)* | ⛔ **`ttt_matched` WILL NaN at `ttt_normalized_write=False`** — G-B reproduces the pilot's per-layer cell exactly, and the pilot NaN'd at step 135/4000 | §5.2: `(2197,52)`, `η·n/d = 3.004`, 76.2 % of chunks amplifying | it does **not** NaN in 20,000 steps ⇒ the divergence is not purely geometric and 7.30's mechanism needs revisiting |
| **P5** ⭐ | **The dyn-eval substitute column, re-measured at 28.56 M: `bpc_static − bpc_dyneval = 0.015 ± 0.008` bpc (0.8 ± 0.4 % relative)** | banked: 0.0144, 0.0153, 0.0159, 0.0153 (mean **0.0152**) | > 0.03 bpc (i.e. approaching the published 0.05) |
| **P5b** | ⇒ **the published dyn-eval purchase (0.99 → 0.94 bpc, −5.1 %, at 277 M) is ~3.3× larger than ours at 28.6 M** — pre-registering the category error the charter already forbids quoting across | arithmetic on P5 | our relative purchase ≥ 3 % |
| **P6** | Envelope: `clu_store` **4.18 ± 1.5 s/step** on 2×A100 at the frozen geometry; worst job ≤ 30 h; **all 15 jobs fit**; makespan < 2 days | §6.1 cost model (atom-linear store term; shell/store split measured from `none` vs `clu_store`) | s/step > 8, or any job > 96 h (⇒ **K3**) |
| **P7** ⭐ *(G-A only — THE GATE, and I now expect it to be CLOSE)* | **`bpc_static(G-A, 4k) − bpc_static(pilot, 4k) < 0.01` bpc** | G4 (the store's read is inert in the loss) + P1 (blanking the store costs nothing): if the *contents* are not in the loss, 4× fewer atoms cannot be. ⚠ **Countervailing evidence I am pre-registering AGAINST myself:** §2.4b measures the write **18.4× worse at 2048 on seed 0**, and 8192 is the only geometry where the writer deepens wells. If the bpc and the write instrument disagree, **the write instrument is the one that matters for a memory claim** | **> 0.03 bpc worse ⇒ K0b fires, the G-A ladder STOPS** and G-B/G-C go to the Advisor. ⚠ *If P7 holds while §2.4b's erosion stands, that is not a green light — it is more evidence for P1's null (the store is not in the loss either way).* |
| **P8** | The retention slice's per-bin gap `clu_store − blank_store` is **flat in distance** (no monotone widening) | P1: a zero scalar dividend cannot have a distance structure | monotone widening across ≥3 consecutive bins on ≥2 seeds ⇒ ⭐ the memory is doing something the scalar hides — **the most interesting possible outcome of this ladder** |
| **P9** | The ledger prints CLU/TTT = **1.0000× (G-A)** / **1.0072× (G-B)** and every arm `within_budget=true` at 2,097,152 B | §5.1 arithmetic | ratio outside `[0.99, 1.01]`, or any arm over budget (⇒ **K4**) |

⛔ **A prediction that survives is evidence; one that fails is a finding; an un-pre-registered agreement is
neither.** P1 and P7 are the two that decide whether this ladder produces a paper or a negative result, and
both are pre-registered **against** the outcome we would prefer.

---

## 7.1 ⭐⭐ **NEW (AMENDMENT 1)** — THE STORE-LIVENESS DIAGNOSTIC, ON PHASE 1's **FIRST** RUNGS

**Head ruling, 2026-08-13.** ⛔ **The C2 flat-curve disjunction must be separated at REAL SCALE, at the
FIRST rungs — not deferred to the end of the ladder.**

### 7.1.1 The disjunction, and why the ladder cannot proceed without resolving it

At smoke scale the store's read is **numerically inert in the loss**: bpc is **bit-identical to 6 d.p.
across a 16× atom range** while the cell's state moves **30,208 → 460,288 B** (§2.4's **G4**, now at 4
seeds), and `blank_store − clu_store` is `≤0.0013 bpc` on 4/4 banked pilot legs with the **sign flipping**
between runs (P1's basis). Two mutually exclusive worlds produce that same reading:

> **(i) the store CARRIES NOTHING** — the outer loop erodes the wells faster than the writer digs them
> (§7.27), so there is nothing to retrieve; **or**
> **(ii) the store CARRIES SOMETHING IT CANNOT ADDRESS** — the content is there, and φ's launch head cannot
> put a read particle where it would find it (C2 Add.16's localized blocker).

⭐ **This exact disjunction has been separated before, on this program's own instrument.** C2W11 measured
the *same store*, the *same physics*, the *same budget grid*: the **shipped read was flat at 0.0004** while
the **oracle-addressed read went 0.0223 → 0.8219 → 0.8711 and plateaued** (`c2w11-physics-organizer` R3,
§9), i.e. a **~2,000× separation produced by addressing alone**. ⇒ **flatness licenses "carries nothing OR
cannot be addressed", never "carries nothing".** ⛔ We reuse **the idea**, not that toy code.

⚠ **Why it must be early.** The two worlds imply *opposite* next moves — (i) says stop and fix the write,
(ii) says stop and fix the launch head — and **neither is "train 12 more arms"**. Learning which one we are
in *after* the makespan is spent buys nothing that could have changed a decision.

### 7.1.2 The instrument — three legs, all on the FIRST rungs' 4,000-step checkpoints

⭐ Run on the **`clu_store` seed-0/1/2 jobs' own 4,000-step checkpoints** (§6.4 step 1), reported **before
the remaining 12 jobs are submitted**. ⛔ Read-side only: **no arm is retrained**, no gradient is taken.

| leg | what it measures | how (reusing what exists) |
|---|---|---|
| **L0 — is anything written?** | `depth_ratio` = median fitted well depth of the live items vs the **untrained** reading of the same tokens; and `qstar_payload_spread`, the between-item range of the settled point's payload coordinate | ⭐ **`train_cluformer.store_health_probe` as shipped** (§7.27's in-flight watch), read at the three store layers instead of one. **Zero new physics.** |
| **L1 — is it ADDRESSABLE?** ⭐⭐ | **oracle-addressed payload recovery**: launch each live item's read at **its own recorded site** (`plan.sites[slot]`, + the same `σ_q` jitter the write used), settle with the **identical read budget and friction**, and compare the settled point's payload block against **that item's written payload**: `frac_recovered` = fraction with `‖pay(q*) − v_j‖ ≤ tol`, `tol` = ½ the minimum between-item payload separation | ⭐ A **small extension of an instrument that already runs**: `store_health_probe`'s `qstar_payload_spread` **already launches every item at its own recorded site** — it reports the spread and throws the identity away. L1 keeps the pairing. ⛔ **Negative control, mandatory:** the same measurement with the site→item pairing **shuffled** (must be ≈0). |
| **L2 — is it in the LOSS?** | `bpc(shipped read)`, `bpc(oracle-site read)`, `bpc(random-site read)`, `bpc(blank_store)` on the **same** eval batches | one extra `static`-shaped eval pass per variant (measured basis: 319 s each at pilot ⇒ **≈16 min per job**, against a 29.7 h job). ⛔ **`random-site` is the laundering control**: the oracle read is off-distribution for the trained `assim` head, and the random-site read is the *same* off-distribution perturbation **without** the right answer — so the addressing effect is `bpc(random) − bpc(oracle)`, never `bpc(shipped) − bpc(oracle)` alone. |

### 7.1.3 ⭐ PRE-REGISTERED NUMBERS, WITH FALSIFIERS AND A KILL CONDITION

| # | prediction (3 seeds, 4,000-step checkpoints, G-B, enwik8 test split) | derivation | ⛔ verdict it forces |
|---|---|---|---|
| **L-P1** ⭐ | **`frac_recovered ≥ 0.50`** under oracle addressing (median over the three store layers), while the shuffled control is `≤ 0.05` | C2W11 got **0.8621 ± 0.0036** exact-set on *unseen* queries under oracle addressing with per-particle payload error **0.0002** against `‖v_j‖ = 0.60`. Discounted to 0.50 because here the store is written by a **language-model outer loop** with §7.27 erosion live (4-seed median `depth_ratio` **0.5497** at this cell), not by C2W11's dedicated write | **holds ⇒ world (ii): ADDRESSING-BOUND.** The store carries content; the read cannot reach it |
| **L-P2** ⭐⭐ | **`Δbpc = bpc(random-site) − bpc(oracle-site) ≤ 0.005` bpc** | P1 + G4: if the contents are not in the loss, handing the read the right address cannot move the loss either | **holds ⇒ the read is not in the loss** even when correctly addressed ⇒ ⛔ the tier-iii memory claim has no mechanism at this geometry, whatever L-P1 says |
| **L-P3** | `depth_ratio ≥ 0.10` at **≥2 of 3** store layers | 4-seed median **0.5497** at `n_atoms = 8192` with the deployed write levers (`atom_place_radius 0.3`, `write_margin 0.6`), which the ladder runs | **fails ⇒ world (i): the wells are gone** |
| **L-P4** | `bpc(blank_store) − bpc(clu_store) = 0.000 ± 0.005` (**= P1, re-measured here 16,000 steps earlier**) | the four banked legs: `+0.001158, +0.001276, −0.000263, −0.000139` | an **early** read on the ladder's decisive question |

**⛔ THE THREE VERDICTS, DECLARED IN ADVANCE** (each is a *conjunction*, so no single soft reading can be
talked into the outcome someone prefers):

| verdict | fires when | what happens next |
|---|---|---|
| ⛔ **INERT — world (i)** | `frac_recovered ≤ 0.05` **AND** `depth_ratio < 0.10`, on **≥2 of 3 seeds** | ⛔⛔ **K6 FIRES: STOP. Do not submit the remaining 12 jobs.** The store carries nothing, so every other arm measures the shell. Re-scope to the **write** (φ's launch head, `erosion_partition`, the §7.27 erosion) — C2 Add.16 already localized the blocker there. ⛔ **Do NOT add rivals** (K1). ⭐ Cost avoided: ~1.7 days of makespan and 12 A100-jobs. |
| ⚠ **ADDRESSING-BOUND — world (ii)** | `frac_recovered ≥ 0.50` **AND** `Δbpc ≤ 0.005` | ⭐ The ladder **continues** (the swap and the controls are still worth measuring), but ⛔ **its headline changes before the results exist**: the honest output is *"the store is addressable in principle and its read does not reach the loss"* — a **mechanism finding**, not a memory dividend. ⛔ It does **not** justify starting phase 2 (§4.3), and P1's null is expected to survive. |
| ⭐ **LIVE — the read is in the loss** | `Δbpc ≥ 0.02` bpc with the random-site control within `0.005` of the shipped read | ⭐ **P1's falsifier is pointed at, early.** The ladder runs in full, **P10's placement contrast is released** (§2.5), and this becomes the first evidence the memory is in the loss at all. |
| ⚠ **UNRESOLVED** | any other combination (e.g. `0.05 < frac_recovered < 0.50`) | ⛔ **Report it as unresolved and say so in the first ten lines of the ladder report.** Do not round it to the nearest verdict. The ladder continues; the diagnostic is re-run on the 20,000-step checkpoints and both readings are published. |

⚠ **Declared limits of this instrument, stated before it runs.** (a) The oracle read is **off-distribution
for `assim`**, which is why L2's quantity is `random − oracle` and never `shipped − oracle`. (b)
`frac_recovered` is a **DIAGNOSTIC** and may never appear as a capability number in any table — it hands
the read the answer (the same bar C2W11 put on its own 0.8621). (c) `tol` is fixed **before** the
measurement, from the geometry (½ the minimum between-item payload separation), never tuned to the result.
(d) The three store layers are reported **separately**, never pooled — P11 is a statement about their
ordering.

---

## 8. Flag provenance for every number in this document

| | (A) geometry arithmetic | (B) TTT stability | (C) geometry sweep | (D) job plan |
|---|---|---|---|---|
| commit | `agent/experiment-engineer/c3-rival-ladder-prereg` (see report §Git) | same | same | same |
| basis | `PilotConfig`/`CluSystemConfig` arithmetic + built `CluStoreCell` | `MatchedTTTCell` at init, `PRNGKey(0)` | real enwik8, `data_bytes=600,000`, seeds **0,1** | landed `csf3_outs/run2` `host_rss` marks |
| hardware | CPU/float32/macOS, JAX 0.9.0 (main venv, never re-synced) | same | same | **2×A100, JAX 0.9.0/gpu** (not mine) |
| scale | `PILOT` (`d_model 512, n_layers 12, seq 1024, batch 8`) | n/a (cell only) | `d_model 32, n_layers 2, seq 256, batch 2, steps 60` | `PILOT`, `steps 4000` |
| non-default store | `min_atoms_base ∈ {512(pilot/G-B), 128(G-A), 256(G-C)}`, `min_atoms=1`, `addr_dim 8`, `payload_dim 4`, `capacity 32` | — | as left column, `min_atoms_base = n_atoms/16`; **deployed arm adds `write_margin 0.6`** | `write_margin 0.6` |
| non-default memory | pilot defaults | — | `chunk 32, address_steps 8, read_steps 8, traj_stride 4, psi_hidden 16, write_inner_steps 2, write_n_perturb 4`; **deployed arm: `atom_place_radius 0.3, write_inner_steps 8`** | `atom_place_radius 0.3, write_inner_steps 40, remat_chunks true, plan_workers 8` |
| sweep arms | — | — | **(i) pre-registered (placement OFF, 22 runs)** and **(ii) declared deployed-write (12 runs)** — ⛔ both reported, neither replaced | — |
| budget | `2,097,152 B`, `enforce=True` (⛔ `False` in the sweep **by declaration**, artifact records `enforced:false`) | — | same | same |
| `ttt_normalized_write` | **False** (shipped) | **False** | **False** | **False** |
| seeds | 0 | 0 (`PRNGKey(0)`) | 0, 1 | 0 (basis), 0/1/2 (planned) |

⛔ **No number in this document is a ladder result.** The only bpc values quoted are (i) smoke-scale, from a
script that says it is never a claim venue, and (ii) the four banked pre-C3 legs, quoted **as a prediction
basis** and labelled non-quotable at every appearance.

---

## 9. What the Advisor is being asked to accept

> ✅ **AMENDMENT 1 (2026-08-13): item 1 is CLOSED — the Advisor and Head RATIFIED G-B, and it is built.**
> What is *now* being asked to be accepted, in its place: **§2.5** (the layer placement, argued), **§4.3**
> (phase 1 is declared and is not the claim), **§7.1** (the store-liveness diagnostic + **K6**) and
> **§6.4** (G-B's launch line). Items 2–5 below stand unchanged.

1. ⛔⛔ **A GEOMETRY DECISION — this is the hand-back, and nothing trains until it is made** (§2.2):
   - ⭐ **G-B (recommended):** the full w23 atom floor (`n_atoms 8192`, no descent) with the CLU cell in
     **3 of 12 layers** ⇒ 1,380,864 B, **0.658×**. Identical bytes/compute/envelope to G-A. Breaks **no**
     design rule. **Costs: a `chlu/core/blocks.py` task (a `store_layers` selection on `StreamModel`) —
     not mine — plus ruling `ttt_normalized_write = True`, which G-B makes mandatory (§5.2).**
   - **G-A (config-only, ships this week, NOT recommended):** `min_atoms_base 512 → 128` ⇒ `n_atoms 2048`,
     1,394,688 B, **0.665×**. ⛔ A 4× descent below a design-ruled floor, and §2.4b's write-efficacy curve
     **fires its own pre-registered falsifier on seed 0 (18.4×)**. Would need the 11 h **K0b gate**.
   - **G-C (fallback):** `n_atoms 4096` ⇒ 2,770,944 B — requires **re-ruling the ceiling** into
     `[2,770,944 , 3,145,728)` (admissible; no rival grows) and is the worst on compute (7.62 s/step).
2. **The ceiling digit: 2 MiB, unchanged** (§3.3), with the window `[1,597,440 , 3,145,728)` verified.
3. **The scope statement** (§4.2): this ladder trains the **two-sided swap**, not the six pinned rivals,
   and **K1** stops it from growing into them on a null.
4. **The job plan** (§6.2) at 20,000 steps / 1.82 epochs, and the pinned `slice_batches = 10`.
5. **The predictions** (§7) — in particular that we expect **no memory dividend** and have written the
   kill condition that says so first.
