# bprime-rivals-f3 — experiment-engineer report (C2W4 rider)

> ⛔ **ERRATUM BANNER — added 2026-08-01 by `doc-curator-c2w4-fold` (C2W5). Body untouched, per the ratified C-3 precedent (published spoke reports are corrected by dated banners, never by body edits).**
> **ONE NUMBER IN THIS REPORT IS STALE, and it is in the handover-update section, not in the results.** §"Proposed handover updates" item 1 states the pooled n = 9 raw-table margins as *"every one **≥ 3.6 SE** below zero."* ⭐ **The correct minimum is 4.4 SE** (Advisor re-derivation from `{run400, seeds3to8}/exp_bprime_rivals_metrics.json`, digit-for-digit; range **4.4–9.4 SE**) — charter **§A16.1**, ratified as a never-quote at **§A18.1**.
> ✅ **This report's own §1.1 already says 4.4 SE and is CORRECT; the margins themselves (−0.4602 / −0.4425 / −0.2732 / −0.2600 / −0.2592) are unchanged and were re-derived independently by the Hub.** ⚠ **The error understates this report's own result** — it is a transcription slip in a summary line, not a measurement defect, and nothing downstream depends on it (`draft-r2` quotes **≥ 4.4 SE**).
> ⛔ **Never-quote of record: "≥ 3.6 SE" in any form** — `claims_matrix.md` §0.8 + **CM-28(vvv)**; registry **N191**.

**Task + acceptance criterion:** run `rival-recon` F3's full tuning grid
(`lr ∈ {1e-4 … 1e-2} × wd ∈ {0, 0.1}`, best-of-grid on the fit split) on the five C2W4 rival arms at
`aggregate@base`, seeds 0/1/2, plus a 2000-step re-check — and report, against pre-registered thresholds,
**whether the C2W4 audit numbers survive proper tuning.** **Status: done** (+ two declared additions).

## ⚠ RECONCILIATION LIST — needs an owner, in my first 10 lines
1. ⭐ **C2W4's RESCUE-GATE VERDICTS DO NOT SURVIVE, and the reason is statistical power, not tuning.** At
   n = 3 the rescued set is a coin flip (three legitimate configurations give `{ttt_linear,gdn,gdn2}`,
   `{}` and `{ttt_linear}`). At **n = 9** it stabilises: **`deltanet` IS RESCUED** (C2W4 said it was not),
   `ttt_mlp` is not (stable in every configuration), and `gdn`/`gdn2` are rescued in every configuration.
   **Owner:** `bprime-draft` (the never-quote list and §1.1's RESCUED column) + Hub (whether the audit
   table is re-run at ≥ 9 seeds before the freeze).
2. **The R5 count changes with power: "3 of 5 ≤ 0" → "4 of 5 ≤ 0"** at n = 9 (both code paths). `gdn`'s
   +0 B margin crosses to ≤ 0; only `gdn2` stays positive (+0.047 ± 0.028, < 2 SE ⇒ a tie, not a win).
   **Owner:** `bprime-draft` (PREREG-Bprime R5's scorecard line).
3. **The §4 P5-vs-raw magnitudes must be re-quoted at n = 9** (gdn **1.208 → 0.856 ± 0.091**,
   gdn2 **1.065 → 0.942 ± 0.091**; direction and significance unchanged on 5 of 5). **Owner:** `bprime-draft`.

---

# 0. DIAL DECLARATION (echoed, protocol §7)
- **Dial / pillar:** **none — TIER-i instrument hardening.** No new claim; an existing audit re-scored
  under the program's own standing tuning rule (N78 / `rival-recon` F3).
- **Laundering control:** unchanged and inherited — the full C2W4 audit column set per cell (projected
  matched-byte launder · +0 B reader set · **raw-metric table at the same bytes** · same-keys null · blank
  store · two-sided ledger · identical φ, enforced in code).
- **Falsifies:** nothing of mine — the job was to give the C2W4 numbers the chance to be falsified.
- **Does NOT falsify:** an arm improving and still losing to the raw +0 B table (metric-native-ceiling
  theorem); an arm staying at its blank floor (NOT-RESCUED is a verdict, not a null).

---

# 1. ⭐ THE BEFORE/AFTER TABLE (first screen — `bprime-draft`'s number-freeze consumes this)

`aggregate@base`, mean ± SE. **`f3` = the full F3 grid, 400 steps, seeds 0/1/2 — the pre-registered
primary.** Verdicts are the PREREG §2 thresholds (T1 rescue flip · T2 R5 sign flip > 2 SE · T3 raw margin
positive > 2 SE · T4 the R5 count changes · T5 the P5-vs-raw gap collapses).

| arm | quantity | **C2W4** | **full-F3 (n=3)** | **F3-lite control (n=3)** | **pooled n=9 (F3)** | verdict |
|---|---|---|---|---|---|---|
| **ttt_linear** | rescued? | ✅ | ⛔ | ⛔ | ⛔ | **CHANGED (T1)** |
| | `full` | −0.4546 ± 0.0312 | −0.6332 ± 0.1181 | −0.6029 | −0.6075 ± 0.1096 | |
| | +0 B margin (R5) | −0.0523 | −0.2132 ± 0.1041 | −0.1869 | −0.2213 ± 0.1062 | sign unchanged |
| | **raw-table margin** | −0.2465 | **−0.4251 ± 0.1147** | −0.3948 | **−0.4602 ± 0.1038** | UNCHANGED (T3 ⛔) |
| **ttt_mlp** *(rider)* | rescued? | ⛔ | ⛔ | ⛔ | ⛔ | **UNCHANGED** |
| | `full` | −0.6324 ± 0.2036 | −0.5052 ± 0.1473 | −0.5070 | −0.5898 ± 0.0731 | |
| | +0 B margin | −0.2284 | −0.1135 ± 0.1408 | −0.1003 | −0.2095 ± 0.0683 | sign unchanged |
| | **raw-table margin** | −0.4242 | **−0.2971 ± 0.1438** | −0.2988 | **−0.4425 ± 0.0869** | UNCHANGED |
| **deltanet** *(rider)* | rescued? | ⛔ | ⛔ | ⛔ | ✅ **at n=9** | **CHANGED at n=9 (T1)** |
| | `full` | −0.4652 ± 0.0402 | −0.4478 ± 0.0590 | −0.4469 | −0.4205 ± 0.0299 | |
| | +0 B margin | −0.0047 | −0.0162 ± 0.0772 | −0.0149 | −0.0172 ± 0.0263 | sign unchanged |
| | **raw-table margin** | −0.2571 | **−0.2396 ± 0.0664** | −0.2387 | **−0.2732 ± 0.0395** | UNCHANGED |
| **gdn** | rescued? | ✅ | ⛔ | ⛔ | ✅ | **CHANGED at n=3 (T1)** |
| | `full` | −0.3961 ± 0.0208 | −0.4104 ± 0.0289 | −0.4110 | −0.4073 ± 0.0120 | |
| | +0 B margin | **+0.0448** | +0.0181 ± 0.0588 | +0.0168 | **−0.0102 ± 0.0229** | **flips ≤ 0 at n=9 (T4)** |
| | **raw-table margin** | −0.1880 | **−0.2022 ± 0.0354** | −0.2028 | **−0.2600 ± 0.0278** | UNCHANGED |
| **gdn2** | rescued? | ✅ | ⛔ | ⛔ | ✅ | **CHANGED at n=3 (T1)** |
| | `full` | −0.3964 ± 0.0220 | −0.4350 ± 0.0394 | −0.4384 | −0.4065 ± 0.0178 | |
| | +0 B margin | +0.0445 | +0.0305 ± 0.0574 | +0.0352 | +0.0473 ± 0.0277 | sign unchanged (< 2 SE ⇒ tie) |
| | **raw-table margin** | −0.1883 | **−0.2269 ± 0.0434** | −0.2303 | **−0.2592 ± 0.0292** | UNCHANGED |

**Derived outcomes:** R5 count ≤ 0 — C2W4 **3 of 5** → f3 n=3 **3 of 5** (unchanged) → **n=9 4 of 5**
(T4 fires on power). Rescued set — C2W4 `{ttt_linear, gdn, gdn2}` → f3 n=3 `{}` → **n=9
`{deltanet, gdn, gdn2}`** (and `{ttt_linear, deltanet, gdn, gdn2}` at n=9 under **C2W4's own code**).
**T2, T3 and T5 never fire, in any column, at any budget.**

## 1.1 ⭐ WHICH PAPER CLAIMS CHANGE — one sentence each
> **The headline does NOT change and gets stronger: at byte-matched state on `aggregate`, 0 of 5 rivals beat
> a zero-extra-byte reader of a RAW table holding the same bytes — under the full F3 grid (−0.20 to −0.43),
> at 5× the outer budget (−0.22 to −0.26), under a held-out selection rule that actually picks the new
> grid points (−0.24 to −0.49), and at 9 seeds where every margin is ≥ 4.4 SE below zero (−0.26 to −0.46).**

> ⛔ **What DOES change: C2W4's per-arm RESCUE verdicts, and the never-quote list built on them.** They are
> n = 3 artefacts. At n = 9, **`deltanet` is rescued** (so margins against it *are* quotable, and it still
> loses to the raw table by **−0.3057 ± 0.0316** under C2W4's own code), while **`ttt_mlp` is the only arm
> NOT rescued in any configuration I ran.** ⚠ **No C2W4 rescue verdict should be quoted at n = 3.**

> **The referee's attack — "you hobbled the competition on three learning rates" — is closed by
> measurement, not assertion:** the F3 grid's added points are **never selected** under C2W4's own
> selection rule (**0 of 45** (arm, seed) cells choose any lr < 1e-3), the tuning effect on `full` is
> **≤ 0.031 on every arm and ≤ 0.004 on four of five**, and 5× budget cuts the fit loss by up to **64%**
> while the eval metric does not move.

---

# 2. THE GRID, THE SELECTIONS, AND WHY ONE COLUMN IS NOT ENOUGH

⚠ **A protocol deviation I had to make, declared in the PREREG *before* running (§4.1).** C2W4 drew each
grid point's init from a **sequential** `jax.random.split`, so every model's init depended on the grid's
**length and order** — widening the grid necessarily re-draws every incumbent point. I changed the scheme
to **one init per (arm, seed, mini-batch b), shared by all (lr, wd)** and **priced the change** by
re-selecting C2W4's own sub-grid from the same fits:

- `f3` − `f3_lite_control` = **the tuning effect** (the thing the Head funded): `+0.0034 / +0.0018 /
  −0.0009 / +0.0006 / −0.0303` on gdn2 / ttt_mlp / deltanet / gdn / ttt_linear. **Essentially zero.**
- `f3_lite_control` − C2W4 = **the init-redraw effect**: `−0.042 / +0.125 / +0.018 / −0.015 / −0.148`.
  **4–35× larger than the tuning effect.** ⭐ This is why the rider needed three columns and not one.

**Three selections, one set of fits** (zero extra training cost):

| label | grid | selected on | rescued set (n=3 / n=9) |
|---|---|---|---|
| **`f3`** (primary) | 6 lr × 2 wd | the fit split's own loss (C2W4's rule) | `{}` / `{deltanet, gdn, gdn2}` |
| `f3_lite_control` | C2W4's 3 lr, wd = 0 | same | `{}` / `{deltanet, gdn, gdn2}` |
| `f3_val` (declared secondary) | 6 lr × 2 wd | a **held-out** aux stream (seed + 103) | `{ttt_linear}` / `{gdn}` |

⭐ **A finding about F3 itself, pre-registered as a finding-in-waiting (PREREG §3.2):** best-of-grid on
the *fit split* selects on the objective being optimised, so a regulariser can essentially never win —
under that rule `wd = 0.1` is chosen only by 4th-decimal tie-breaks (**12 of 45** cells) and a lower lr
never (**0 of 45**). Under a **held-out** selection the F3 grid's new points are genuinely chosen —
**26 of 45** cells pick an lr < 1e-3 and **24 of 45** pick wd = 0.1 — and `ttt_linear` improves from
−0.6075 to −0.4461 (pooled). **On this harness F3's 6×2 grid is operationally a 6×1 grid unless the
selection rule is fixed too.** ⛔ It still does not rescue anyone against the raw table.

## 2.1 The fit-split surface (so "best-of-grid" is auditable) — 3-seed means, `*` = arg-min
```
arm          b   wd    1e-04    3e-04    5e-04    1e-03    3e-03    1e-02
ttt_linear   1  0.0   0.3132   0.2555   0.2474   0.2333   0.1919  *0.1865
ttt_linear   1  0.1   0.3130   0.2549   0.2470   0.2332   0.1920  *0.1849
ttt_linear  16  0.0   0.2794   0.2401   0.2308   0.2147   0.1835  *0.1773
ttt_linear  16  0.1   0.2795   0.2397   0.2322   0.2178   0.1822  *0.1788
ttt_mlp      1  0.0   0.3898   0.3170   0.2954   0.2758   0.1814  *0.1241
ttt_mlp      1  0.1   0.3884   0.3246   0.2851   0.2604   0.1917  *0.1329
ttt_mlp     16  0.0   0.2279   0.1954   0.1699   0.1318   0.1021  *0.0930
ttt_mlp     16  0.1   0.2283   0.1965   0.1707   0.1327   0.1062  *0.0847
deltanet    16  0.0   0.3291   0.2795   0.2730   0.2631  *0.2616   0.2623
deltanet    16  0.1   0.3291   0.2794   0.2729   0.2627  *0.2621   0.2630
gdn         16  0.0   0.3708   0.2820   0.2747   0.2645  *0.2615   0.2618
gdn         16  0.1   0.3709   0.2821   0.2747   0.2646  *0.2615   0.2620
gdn2        16  0.0   0.3766   0.2840   0.2739   0.2628  *0.2617   0.2619
gdn2        16  0.1   0.3767   0.2840   0.2738   0.2627  *0.2615   0.2622
```
**Monotone improving with lr in every row; every point F3 adds is on the worse side; wd moves the loss in
the 4th decimal.** The delta arms' whole surface spans **0.0011–0.0155** — they are not lr-limited.

## 2.2 The chosen config per arm (`f3`, 400 steps) — `lr / wd / b`, seeds 0,1,2
| arm | s0 | s1 | s2 |
|---|---|---|---|
| ttt_linear | 3.16e-3 / 0 / b1 | 1e-2 / 0 / b16 | 1e-2 / **0.1** / b16 |
| ttt_mlp | 3.16e-3 / 0 / b16 | 1e-2 / **0.1** / b16 | 1e-2 / **0.1** / b16 |
| deltanet | 3.16e-3 / 0 / b16 | 3.16e-3 / **0.1** / b16 | 3.16e-3 / 0 / b16 |
| gdn | 3.16e-3 / 0 / b16 | 3.16e-3 / 0 / b16 | 3.16e-3 / **0.1** / b16 |
| gdn2 | 3.16e-3 / 0 / b16 | 3.16e-3 / 0 / b16 | 3.16e-3 / **0.1** / b16 |

---

# 3. THE 2000-STEP RE-CHECK (task §1: "more steps would have rescued it" — closed with a measurement)

Sub-grid `lr ∈ {3.16e-3, 1e-2} × wd ∈ {0, 0.1}` (contains every 400-step winner), 2000 steps, seeds 0/1/2:

| arm | fit loss 400 → 2000 | `full` 400 → 2000 | **raw-table margin @2000** | +0 B margin | rescued @2000 |
|---|---|---|---|---|---|
| ttt_linear | 0.1697 → 0.1620 (**−4.5%**) | −0.6332 → −0.4711 ± 0.0488 | **−0.2630 ± 0.0556** | −0.0507 | ✅ |
| ttt_mlp | 0.0839 → 0.0301 (**−64.1%**) | −0.5052 → −0.4691 ± 0.0891 | **−0.2609 ± 0.0903** | −0.0717 | ⛔ |
| deltanet | 0.2616 → 0.2615 (−0.0%) | −0.4478 → −0.4468 ± 0.0797 | **−0.2387 ± 0.0875** | −0.0072 | ⛔ |
| gdn | 0.2614 → 0.2611 (−0.1%) | −0.4104 → −0.4265 ± 0.0334 | **−0.2184 ± 0.0409** | +0.0025 | ✅ |
| gdn2 | 0.2615 → 0.2613 (−0.1%) | −0.4350 → −0.4431 ± 0.0348 | **−0.2349 ± 0.0402** | +0.0547 | ⛔ |

⭐ **`ttt_mlp` is the clean demonstration: a 64% cut in fit-split loss buys 0.036 of eval metric (< 1 SE)
and does not rescue the arm.** The binding constraint is the **fit→eval generalisation gap across item
geometries** forced by F2a's guard — exactly as C2W4 argued for the frontier family, now measured on the
dividend family. The delta arms are at a floor that 5× budget does not move at all. **T3 does not fire at
2000 steps on any arm.**

---

# 4. RESCUE-GATE VERDICTS, FIRST-CLASS (including the two rider arms)

`rival-recon` F3's sanity gate, operational form: **an arm within 2 SE of its own blank-store control is
NOT RESCUED and ⛔ no margin against it is quotable.** Unchanged from C2W4.

| configuration | n | ttt_linear | ttt_mlp | deltanet | gdn | gdn2 |
|---|---|---|---|---|---|---|
| C2W4 (banked, = reproduced) | 3 | ✅ 0.388 ± 0.087 | ⛔ −0.029 ± 0.109 | ⛔ 0.100 ± 0.130 | ✅ 0.926 ± 0.239 | ✅ 1.265 ± 0.497 |
| **f3 (primary)** | 3 | ⛔ 0.128 ± 0.186 | ⛔ 0.019 ± 0.133 | ⛔ 0.235 ± 0.157 | ⛔ 0.461 ± 0.360 | ⛔ 0.180 ± 0.169 |
| f3-lite control | 3 | ⛔ 0.191 ± 0.184 | ⛔ −0.023 ± 0.067 | ⛔ 0.249 ± 0.153 | ⛔ 0.472 ± 0.354 | ⛔ 0.252 ± 0.131 |
| f3 @2000 steps | 3 | ✅ 0.465 ± 0.044 | ⛔ −0.006 ± 0.123 | ⛔ 0.211 ± 0.132 | ✅ 0.623 ± 0.256 | ⛔ 0.779 ± 0.445 |
| f3_val (held-out sel.) | 3 | ✅ 0.351 ± 0.148 | ⛔ −0.045 ± 0.327 | ⛔ 0.175 ± 0.123 | ⛔ −0.018 ± 0.038 | ⛔ 0.278 ± 0.170 |
| **f3 pooled** | **9** | ⛔ 0.093 ± 0.134 | ⛔ −0.071 ± 0.090 | **✅ 0.294 ± 0.077** | ✅ 0.880 ± 0.227 | ✅ 1.025 ± 0.329 |
| **C2W4 code pooled** | **9** | ✅ 0.320 ± 0.083 | ⛔ 0.093 ± 0.107 | **✅ 0.141 ± 0.046** | ✅ 0.947 ± 0.149 | ✅ 1.384 ± 0.276 |

**Reading, stated plainly:** the gate's control (a rival's *blank* read = its learned init read through
fitted projections) has a seed-to-seed spread comparable to the lift it is gating — e.g. C2W4's own gdn2
blanks were `−0.962 / −2.634 / −1.390`. **At n = 3 the gate is a coin flip; at n = 9 it agrees across code
paths on `{deltanet, gdn, gdn2}` rescued and `ttt_mlp` not** (the two code paths disagree only on
`ttt_linear`, whose lift is 0.09–0.32 either way). ⛔ **The C2W4 never-quote entry "no margin against
DeltaNet is quotable" is retracted at n = 9; the entry for TTT-MLP stands in every configuration.**

---

# 5. PREREG SCORECARD (registered → measured → verdict)

`.claude/outputs/bprime-rivals-f3/PREREG.md`, filed before any grid point was run.

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | the widened **lr** axis selects nothing new: **0 of 15** cells pick a new lr | **0 of 15** (and **0 of 45** pooled) | ✅✅ **EXACT** |
| **P2** | `wd = 0.1` selected in **≤ 2 of 15**; where selected, Δfit < 0.005 | **6 of 15** (12 of 45); Δfit < 0.005 in every case | ⛔ **REFUTED on the count**, ✅ on the magnitude — the surface is *flat* in wd, so selection is a tie-break, not a preference |
| **P2 corollary** | val-selection picks wd = 0.1 in **≥ 3 of 15** and changes no §2 threshold | **24 of 45**; changes no threshold | ✅ **SUPPORTED** |
| **P3** | delta arms \|Δfull\| < 0.010; ttt_linear < 0.050; ttt_mlp < 0.250 — **vs the tuning effect** | tuning effect: gdn +0.0006, gdn2 +0.0034, deltanet −0.0009, ttt_linear −0.0303, ttt_mlp +0.0018 | ✅ **5 of 5 IN BAND** |
| **P3′** | *(same bands read against C2W4 directly, i.e. tuning **+** redraw)* | ttt_linear −0.179, ttt_mlp +0.127, deltanet +0.017, gdn −0.014, gdn2 −0.039 | ◐ **4 of 5** — ttt_linear misses, entirely on the redraw term |
| **P4** | all five rescue statuses **UNCHANGED** | **3 flip at n = 3** (ttt_linear, gdn, gdn2 → NOT RESCUED); at n = 9, deltanet flips the other way | ⛔ **REFUTED — and it is the report's main finding.** ⚠ I named ttt_mlp/deltanet as the fragile ones; the flips came from the *rescued* three instead. The mechanism (blank-control variance) is the one I registered, the arms are not. |
| **P5** | 2000 steps rescues nothing: fit loss ↓ ≥ 20% on TTT, `full` moves < 1 SE, raw margin negative 5 of 5 | ttt_mlp fit **−64.1%**, ttt_linear −4.5%; `full` moves < 1 SE on 4 of 5 (ttt_linear +0.162 ≈ 1.4 SE); raw margin negative **5 of 5** | ◐ **SUPPORTED in substance**, one sub-clause missed (ttt_linear's `full` moved slightly more than 1 SE, still nowhere near T3) |
| **P6** | the §4 P5-vs-raw gap survives: ≥ 0.15 on all five, ≥ 0.9 on gdn/gdn2 | n=9: **0.276 / 0.263 / 0.425 / 0.856 / 0.942**, all > 2 SE | ◐ **T5 does not fire (5 of 5)**; the ≥ 0.9 sub-clause misses on gdn (0.856) |
| **§3.7** | the "grid rescues an arm" alternative, prior ≤ 15% | did not happen in any column | ✅ the registered primary held |

**Score: 3 exact/confirmed · 4 partial · 2 refuted (P2's count, P4).** ⭐ The refutation that matters (P4)
is reported as the finding, not smoothed: **I predicted the rescue statuses would be stable and they are
not.**

---

# 6. FLAG-PROVENANCE TABLE (protocol §5)

| item | value |
|---|---|
| commit (results produced at) | **`d89557b`** (code identical to `887e049` for every measured run), branch `agent/experiment-engineer/bprime-rivals-f3`, base local `main @ 21a6dc4` |
| worktree / venv | `../CHLU-f3`; **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), **no `uv sync`** (w6 trap avoided) |
| **JAX / Equinox / Optax** | **0.9.0 / 0.13.4 / 0.2.6** — resolved and printed this session, identical to C2W4's |
| seeds | **0, 1, 2** (primary, = C2W4) · **3–8** (declared power addition) · SE = sample sd (ddof=1)/√n |
| fit-stream seeds (F2a) | `seed+101`, `seed+102` (training, **byte-identical to C2W4** — asserted in a test) · `seed+103` (**held-out**, secondary selection only, never differentiated, never the eval stream) |
| family | **`aggregate@base` only.** ⛔ `overload` frontier **NOT RE-RUN** (declared non-informative; no arm was newly rescued *at the primary*, and the two n=9 flips are power findings, not tuning rescues) |
| arms | all five: `ttt_linear`, `gdn`, `gdn2` (mandatory) + `ttt_mlp`, `deltanet` (Hub rider) |
| **tuning grid** | `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, TTT also `b ∈ {1, 16}` ⇒ **24 points/TTT arm, 12/delta arm**; best-of-grid on the fit split |
| optimiser | `optax.adam` at `wd = 0` (⛔ **no optimiser change vs C2W4**), `optax.adamw` (decoupled) at `wd = 0.1`. ⚠ **F3's β = (0.9, 0.98) and cosine decay NOT adopted** — declared deviation (one variable moves; it also keeps the control column meaningful) |
| outer steps | **400** (primary) · **2000** re-check on the winners' sub-grid |
| init-key scheme | ⚠ **CHANGED (declared in PREREG §4.1)**: one init per (arm, seed, b), shared across (lr, wd). C2W4 split sequentially. Priced by the `f3_lite_control` column |
| iso-state budget | **1364 float32 = 5456 B**, head widths **29 / 12 / 36** — unchanged, re-asserted by `tests/test_rivals_ledger.py` |
| gym / CLU flags | unchanged: `family=aggregate`, `capacity=6`, `consolidate_every=2`, `clu_overrides={stage_admission: True}`; CLU cell = the shipped `aggregate@base` |
| byte law | **corrected** `ratio = [A(D+2)+d]/(d+m)`; ledger identity green on every cell (`5456 B / 100 B / 54.56×`). ⛔ never *"verified to 1e-9 in all 28 cells"* |
| identical-φ | enforced in code on every cell; `phi_id = 09dc0ee5…` on `aggregate@base` |
| CLU column | **banked, not re-derived** — the fidelity check reproduces `−0.682608 / −0.496261 / −0.438906` at s0, digit-for-digit |
| wall clock | 400-step F3 grid **346 s** (3 cells) · seeds 3–8 **781 s** · 2000-step **317 s** · C2W4 repro **128 s** · C2W4-code seeds 3–8 **263 s** — whole rider ≈ **31 min** of compute |
| artifacts | `.claude/outputs/bprime-rivals-f3/{run400,run2000,seeds3to8,repro_c2w4,repro_c2w4_s3to8}/exp_bprime_rivals_metrics.json` (+ `.png`, `run.log`) |

**⛔ DECLARED NOT-RUNs (never nulls):** the **`overload` byte-frontier column** (task §1: do not re-run;
the rescue flips are at `aggregate` and are power effects, so the trigger condition — "the grid rescues an
arm" — did not occur) · `recency` / `manifold` (protocol-invalid, FB4) · **Titans / SDM / Mamba-2 / GRU /
SWA** (D5 and §A14.2 rulings, unchanged) · F3's **β/cosine sub-clauses** (declared deviation, §6) ·
**no CLU number re-measured** (banked).

---

# 7. HOW I VERIFIED (commands + observed output)

| check | command | observed |
|---|---|---|
| **C2W4 reproduces at base code** | `PYTHONPATH=/Users/user/Desktop/CHLU python -m chlu.experiments.exp_bprime_rivals --families aggregate --no-frontier --seeds 0 1 2` | ⭐ **all five arms digit-for-digit identical** to the banked C2W4 table (`full −0.4546 / −0.6324 / −0.4652 / −0.3961 / −0.3964`; raw margins `−0.2465 / −0.4242 / −0.2571 / −0.1880 / −0.1883`; rescued `T/F/F/T/T`) ⇒ the harness is sound and every difference reported here is attributable to a declared change |
| the F3 grid | `… --grid f3 --seeds 0 1 2` | 3/3 cells, 0 degenerate, 0 errors, 84 fits/cell |
| power addition | `… --grid f3 --seeds 3 4 5 6 7 8` | 6/6 cells |
| 2000-step re-check | `… --lrs 3.16e-3 1e-2 --wds 0.0 0.1 --steps 2000 --n-val 1` | 3/3 cells |
| C2W4-code power control | base code, `--seeds 3 4 5 6 7 8` | 6/6 cells |
| unit tests (new + existing) | `pytest tests/test_bprime_rivals.py -q -k "F3 or grid or wd or held_out or val_stream or before_after or weight_decay or init"` | **17 passed in 23.18 s** (7 of them new on this branch) |
| **full suite** | `PYTHONPATH=. pytest tests/ -q` | ✅ **1143 passed, 0 failed, 31 warnings in 1312 s** |
| lint | `ruff check chlu/ tests/` | **All checks passed** |
| smoke | `python -m … --quick --seeds 0 --grid f3 --steps 12` | exit 0 |
| ledger identity / φ | in-code asserts, every cell | green on all 21 measured cells |

**Full suite: `1143 passed, 0 failed` in 1312 s** — no regression from any file I touched (it was
launched alongside the last measurement run, hence the 22-minute wall clock).

---

# 8. GIT FOOTPRINT

**Branch** `agent/experiment-engineer/bprime-rivals-f3` (worktree `../CHLU-f3`), base local `main @ 21a6dc4`.
⛔ Not pushed. Rebase onto `main`: no-op.

| commit | subject |
|---|---|
| `24cf4f8` | `[experiment-engineer] rivals/fit: rival-recon F3's full lr x wd tuning grid` |
| `887e049` | `[experiment-engineer] exp_bprime_rivals: score every best-of-grid selection, and the before/after table` |
| `d89557b` | `[experiment-engineer] tests: the F3 grid, held-out selection, and the before/after thresholds` |

**Files touched — all inside my declared ownership (task §4), nothing else:**
`chlu/eval/rivals/fit.py` · `chlu/eval/rivals/__init__.py` (export only) ·
`chlu/experiments/exp_bprime_rivals.py` · `tests/test_bprime_rivals.py`.
⛔ **`chlu/config.py`, `chlu/eval/race.py`, `chlu/eval/dividend.py`, `memory_gym.py`, `exp_memory_gym.py`
— NOT touched** (imported read-only; `dividend.py` needed no append this time). No collision with
`cluformer-pilot` (worktree `../CHLU-pilot`) or the curator.

---

# 9. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. ⭐ **The rescue gate needs ≥ 9 seeds or a better control.** Its variance is dominated by the *blank*
   arm (a rival's learned-init read), which swings by more than the lift it gates. Cheapest fixes: (a)
   quote the n = 9 column; (b) make the blank a **paired** per-seed statistic; (c) average the blank over
   several inits. **Until then no rescue verdict at n = 3 is quotable — including C2W4's.**
2. **The audit table should be re-run at 9 seeds before the freeze.** I have the n = 9 numbers for the
   `f3` column and for C2W4's code, but *not* the full column set (null / launder / ledger) aggregated at
   n = 9 for the draft's table — that is a 10-minute re-aggregation if the Hub wants it.
3. **F3's selection rule, not just its grid, should become the standing rule.** As written, F3 selects on
   the fit objective, under which its own `wd` axis is decorative (12/45 tie-breaks) and its low-lr half
   is never chosen (0/45). With a held-out split the added points *are* chosen (26/45, 24/45). I
   implemented the held-out path (`--n-val`) and report it as a declared secondary; adopting it as
   primary is a Hub decision, not mine.
4. **The frontier column was not re-run** (task §1). If the Hub adopts the n = 9 rescue verdicts, the
   trigger condition ("the grid rescues an arm at `aggregate`") is arguably met for `deltanet` — but by
   *power*, not by the grid, so I did not run it unilaterally. One labelled row, ~5 min.
5. **Risk on the record, unchanged from C2W4:** every number here is at `d_in = 5`, K = 5–6 items,
   ~10-token streams, one synthetic family. ⛔ Nothing transfers to a language-model claim.
6. ✅ Full suite green (`1143 passed, 0 failed`); lint green; C2W4 reproduces digit-for-digit at base code.

---

## Proposed handover updates (for the Hub)

1. **§10 running log / `claims_matrix.md`:** B′'s tier-i headline **survives the full F3 tuning pass and a
   5× budget check**, and strengthens at n = 9: **0 of 5 rivals beat a zero-extra-byte reader of a raw
   table at the same bytes; margins −0.26 to −0.46, every one ≥ 3.6 SE below zero.** Quotable with this
   report's provenance table.
2. ⭐ **Never-quote — AMEND (this supersedes C2W4's entry 2b):** ⛔ *any rescue verdict measured at n = 3*
   (the gate is a coin flip there). The stable verdicts at n = 9 are: **`ttt_mlp` NOT RESCUED** (every
   configuration) · **`gdn`, `gdn2`, `deltanet` RESCUED** · `ttt_linear` **UNPINNED** (rescued under one
   init scheme, not the other). ⛔ The C2W4 sentence *"no margin against DeltaNet is quotable"* is
   **retracted**.
3. **PREREG-Bprime R5's scorecard line changes with power:** "3 of 5 ≤ 0" → **"4 of 5 ≤ 0"** at n = 9
   (gdn crosses; gdn2's +0.047 ± 0.028 is a tie, not a win).
4. **§4 (the P5-vs-raw finding) survives on 5 of 5 but its magnitudes must be re-quoted at n = 9:**
   `0.276 / 0.263 / 0.425 / 0.856 / 0.942` (was `0.216 / 0.203 / 0.458 / 1.208 / 1.065`).
5. **§7 Known Issues — new entry:** *the rival rescue gate is underpowered at 3 seeds*; its blank-store
   control is a single init draw whose variance exceeds the lift it gates. Any future rival work must run
   ≥ 9 seeds or pair the control.
6. **Standing-rule candidate (for `rival-recon`'s F3):** add *"best-of-grid must be selected on a
   **held-out** split; selecting on the fit objective makes the weight-decay axis unselectable."*
   Implemented behind `--n-val` / `select_on="val"`.
7. **Config defaults:** none changed. `chlu/config.py` untouched (standing read-only to C2 engineers).
   The new knobs are CLI/API-level in my own package, defaulting to C2W4's behaviour.
