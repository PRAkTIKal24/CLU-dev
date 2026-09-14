# pilot-placement-probe — experiment-engineer report

**Task + acceptance criterion:** deliver the two B′ riders, then test the pilot's §5.3 placement
hypotheses (H1 `atom_local_radius`, then H2 the trajectory write term) in the toy rig against the
pre-registered success signal — **(a)** in-block acquisition off chance, **(b)** live ≠ blank at
float32, **(c)** well depth leaves the 0.045 saturation toward the shipped 0.46–0.80 band — 3 paired
seeds, and end with a **recommended config block for `scripts/csf3/job_gpu_cluformer.sh`**.
**Status: done** (with 2 declared CUTS and 1 declared NOT-RUN, listed in §9 — never nulls).

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5, first-10-lines rule)
> **R1 — ⛔ `cluformer-pilot` §8.2 / N196's compute attribution is WRONG and it is in the running log.**
> *"77–84 % of CLU wall-clock is the Python plan pass"* is right about the **plan pass** and wrong about
> the **Python**: measured, **only 1.6 % of the plan pass is the controller**; 98.4 % was the per-layer
> block forward running **eagerly** beside a `filter_jit`-ed training step. Jitting it: plan pass
> **2.755 s → 0.127 s (21.7×)**, step **3.14 s → 0.547 s (5.7×)**, plan-pass share **0.877 → 0.232**
> (and **0.304 ± 0.008** measured in situ across 6 trained runs). *(Owner: curator — N196's compute
> clause + the §10 entry + `future_work.md`'s "the controller loop needs vectorising".)*
> **R2 — ⛔ The pilot's THREE §5.3 candidate fixes are now measured, and the surviving one is the THIRD
> (the ψ payload residual), which was ranked last.** H1 (localized init) is **worse than doing
> nothing**; H2 (trajectory write term) is **bit-identically inert** at 3.9× the wall clock; a
> streaming-adapted placement (H1b) + a raised hinge margin **do wake depth and the live-vs-blank
> channel but not acquisition**. The isolated cause is the **payload channel**: the dynamics deliver
> 30–50 % of the payload and **ψ then compresses it 7–25×**. *(Owner: Hub → whoever holds the tier-ii
> write/read objective; §5–§6 have the numbers.)*
> **R3 — ⛔ 200 steps of OUTER training destroys the store's wells** (depth **0.029 → 4.95e-63**,
> 3 seeds, at the shipped config). This is the same object as the pilot's monitor #9 `Δ_ret = 7.8e-86`
> and its cause was never named. Placement + margin keeps depth at **0.0616 ± 0.037** through the same
> training. *(Owner: Hub — it is a CSF3 config decision; §10 row 10 is the watch-item.)*
> **R4 — the n = 9 ledger columns for the TTT arms are NOT constant across seeds** (`ttt_linear`
> 5220/5328 B, `ttt_mlp` 4656/5376 B) because best-of-grid picks the mini-batch `b` per seed and `b`
> enters the declared state. ⛔ A single TTT byte figure must never be quoted as *the* n = 9 value.
> *(Owner: `bprime-draft-r3`.)*
> **R5 — at n = 9 both TTT arms are BELOW their own same-keys null** (`full − null` = −0.2063 ± 0.1016
> and −0.1995 ± 0.0665) while all three delta arms are above it. New column, rival-side only, changes
> no CLU claim. *(Owner: `bprime-draft-r3`.)*

---

## ⭐ DIAL DECLARATION (echoed before the first result)

- **Dial / pillar:** **none — instrument/diagnostic** (tier-iii mechanism isolation). ⛔ Nothing in
  this report is a paper number; it is config evidence for the CSF3 run and design evidence for tier ii.
- **Laundering control:** the pilot's own arms, paired seeds — **live vs blank vs memory-deleted** —
  plus, mandatory here, the **blank store carrying the SAME localized init** (a localized init can buy
  self-probe hits with no write at all). Only `acq(live) − acq(blank)` at the same init is evidence of
  a **write**. Measured: **exactly 0.000 on 14/14 cells × 3 seeds.**
- **Falsifies:** H1 fails if `atom_local_radius` at its designed band leaves acquisition at chance AND
  live = blank at float32. H2 same bar. Both failing ⇒ the placement hypothesis is refuted at toy
  scale — **a FINDING that re-prices, never closes, the route (A18.4).**
- **Does NOT falsify:** anything about tier ii or the CSF3 scale run; toy-scale GRU superiority
  (pre-registered, already measured).
- ⚠ **Monitor #13 / N94 travels with every reading:** the shipped block writes with **4 inner steps
  against a floor of 40**, so **every number here is formally NON-PROMOTABLE** — except the two
  `*_w40` cells, which are at the floor and still say the same thing.
- ⚠ **"CLU-former" is a placeholder and appears nowhere.** This is *the tier-iii block*.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch | `agent/experiment-engineer/pilot-placement-probe`, worktree `../CHLU-placement`, base local `main @ 29fc22b` |
| commits (5) | `2db0fbc` · `edec27a` · `46755fb` · `85a557d` · `df36427` — see §13 |
| env | **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), **no worktree `uv sync`** (w6 hazard avoided). **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, numpy 2.4.1, **CPU** |
| **seeds** | **0, 1, 2 on every cell**, paired (identical shell, identical φ-gain, identical data order, identical plan pass — cells differ ONLY in the lever under test). SE = sample sd (ddof = 1)/√n |
| **scale** | ⛔ **TOY (0.16 M): `d_model 64`, `n_layers 2`, `seq_len 512`, `batch 4`, `chunk 32`.** ⛔ Not a 26–47 M number and never reported as one |
| data | **enwik8**, byte-level, canonical 90/5/5 positional split, first 4 000 000 B; screen probes on the **valid** split's first contiguous batch, held-out bpc on **test** (4 contiguous batches) |
| store | `DesignFreedomPotential(rung="free_mlp", family="atoms")`, `dim 3` (`addr_dim 2` + `payload_dim 1`), `capacity 8`, `budget 6`, `n_atoms 1024` (`atoms_per_item 128`), `atom_width 0.3`, `atom_init_scale 1.0`, `atom_depth_init 1e-4`, `confine 0.05`, `ball_radius 1.0`, `query_sigma 0.15`, masked write ON |
| levers | ALL stage flags TRUE; `soft_certificate = True` (`ζ = 0.6` ⇒ `d_safe = 0.3895`, `sep_expected = 0.6492`); `leak 0.02`, `retry_tau 0.5`, `retry_max_rounds 1` |
| read | `dt 0.05`, `γ_address 0.05`, `γ_read 0.02` (both trainable), **24 + 24 (+24 gated retry)** Verlet steps, `traj_stride 8`, `kinetic_mode newtonian_learned`, read_mode **trajectory**, ψ = `DeepSetsPsi(hidden 32, depth 2)` |
| write | masked/local, **4 inner sign-SGD steps at lr 0.05** (40 in the `*_w40` cells), `n_perturb 8`, `σ_addr 0.25`, `σ_pay 0.6`, **`margin 0.15`** (0.6 / 1.0 in the `*_m*` cells), `barrier 0.2`, `barrier_pairs "nn"` |
| φ | shared, bit-identical across arms, calibrated gain per seed by the shipped rule (RMS address norm = `ball_radius`) |
| **probe levers (all new, all OFF by default, all bit-identical when off)** | `atom_local_radius` + `atom_group_centers` (H1) · `atom_place_radius` (H1b) · `write_lambda_traj` (H2) · `write_margin` via the store-override path |
| optimiser (trained tier only) | AdamW, warmup-cosine, peak `lr 1e-3`, `warmup 20`, `grad_clip 1.0`, `wd 0.0`, **200 steps** |
| langevin / temperature | **N/A** — deterministic, `T = 0`, `p₀ = 0`, no Langevin step. §7.22's discipline does not apply |
| wake–sleep / Lyapunov | **N/A** — a third training path (`train_cluformer.py`), byte-LM cross-entropy |
| artifacts | `.claude/outputs/pilot-placement-probe/` — `PREREG.md` (+3 addenda) · `probe_all_records.json` (48 records) · `probe_{screen,screen2,margin,trained}_records.json` · `decode_dispersion.json` · `qstar_payload.json` · `n9_full_columns{.json,_table.md}` · `n9_deltanet_frontier{.json,_table.md}` · `*_run.log` |
| wall clock | ≈ **3.0 h of measured runs** (48 probe cell-seeds + the riders' 9 frontier seeds + the plan-pass benchmarks), plus ≈ 0.4 h of a **discarded** pre-fix screen run (§13). Inside the ~6 h declared budget |
| CSF3 | **0 A100-hours.** Not reachable from this machine (the pilot pre-registered the DNS/VPN failure) |

---

## 1. ⭐ THE RIDERS (§0 of the task — delivered first, they gate `bprime-draft-r3`)

Both are in `.claude/outputs/pilot-placement-probe/` with paper-ready tables.
**Ping for the Hub: BOTH RIDERS HAVE LANDED.**

### 1.1 Rider 1 — the **n = 9 full-column re-aggregation** (F3 follow-up 2)
`n9_full_columns.json` + `n9_full_columns_table.md`. Four columns (`f3` primary · C2W4's own code path
· `f3_lite_control` · `f3_val`), each at a **uniform n = 9**, aggregated through the shipped
`exp_bprime_rivals.audit_table` so the rule is byte-identical to the published one. **Nothing
re-measured.**

- ✅ **Fidelity:** every quantity the F3 report published at n = 9 reproduces **digit-for-digit**.
- ⭐ **What is new:** the **same-keys null**, the **launder**, the three **+0 B readers**, the paired
  `full − null` / `full − blank` statistics, and the **byte ledger**, all at the same n — the columns
  App. I.1c flagged as un-aggregated.
- **Headline unchanged and now uniform-n:** *at byte-matched state on `aggregate`, **0 of 5** rivals
  beat a zero-extra-byte reader of a RAW table holding the same bytes.* Raw margins are
  **4.43 / 5.09 / 6.92 / 9.35 / 8.87 SE below zero** ⇒ **minimum 4.43 SE** (⛔ "≥ 3.6 SE" stays retired).
- ⭐ **R5 (new):** `full − null` = **−0.2063 ± 0.1016** (ttt_linear) · **−0.1995 ± 0.0665** (ttt_mlp) ·
  **+0.2174 ± 0.0749** (deltanet) · **+0.5642 ± 0.1032** (gdn) · **+0.7438 ± 0.1242** (gdn2). **Both
  TTT arms are below their own same-keys null**; all three delta arms are above it. Same pattern on
  C2W4's code path.
- ⚠ **R4 (new):** the TTT arms' byte-ledger columns are **not constant across the 9 seeds**
  (`ttt_linear` 5220/5328 B with `b = [1,16,16,1,1,16,1,16,1]`; `ttt_mlp` 4656/5376 B) — best-of-grid
  selects the mini-batch per seed and `b` enters the declared state. Delta arms are constant.

### 1.2 Rider 2 — the **labelled `deltanet` byte-frontier row** (F3 follow-up 4)
`n9_deltanet_frontier.json` + `n9_deltanet_frontier_table.md`. `overload@load1x_shipped`, `decode`,
**seeds 0–8 (n = 9)**, 5 head widths, `deltanet` **plus** `ttt_linear`/`gdn2` re-run on the same
post-F3 code path (the C2W4 frontier artifact predates the init-key change, so the incumbents were not
comparable). The CLU curve is banked, not re-measured.

⛔ **The row ships as a LABELLED NULL, and that is the point.** **Not one of the 15 (rival × head)
cells is RESCUED at n = 9** — every arm is within 2 SE of its own blank (best lift anywhere
`deltanet@d4`, **+0.0694 ± 0.0491**). By the program's own rescue gate, ⛔ **no frontier margin against
any of these three rivals is quotable, including the newly-rescued `deltanet`: its rescue is an
`aggregate` verdict and does not transfer to the frontier family.** The venue has no resolving power
here (6-way decode, chance 0.167, 24 queries; every arm lives in 0.12–0.24). ⚠ The small-head cells
(`d ≤ 8` for deltanet/gdn2) are **not table-lossless**, so their launder is a lossy control.

---

## 2. ⭐⭐ THE PROBE — the three success signals, 14 cells × 3 paired seeds

⛔ TOY scale. `depth` = median fitted well depth at the item's own site on the launch manifold (the
pilot's convention; its saturated value is **0.045**, the shipped store's fitted band **0.46–0.80**).
`acq` is pooled over **all 4 lanes** (13–18 probed items/seed) — `monitor_pass` replays lane 0 only,
which leaves chance = 1/3 and SE ≈ 0.27, too weak for the registered `chance + 2 SE` bar.

| cell | lever | **depth** (median ± SE) | **acq** | chance | **acq − blank** | **live−blank (bpc)** | read Δ | (a) | (b) | (c) |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | the pilot's config | 0.02882 ± 0.015 | 0.260 | 0.260 | **+0.000** | +2.80e-05 ± 1.4e-05 | 8.07e-03 | ⛔ | ✅ | ⛔ |
| `h1_r0.15` | H1 init, 0.5 s | **2.27e-08 ± 2.2e-08** | 0.260 | 0.260 | +0.000 | **−6.02e-06** ± 4.1e-06 | 1.46e-04 | ⛔ | ✅(−) | ⛔⛔ |
| `h1_r0.3` | H1 init, 1 s | **9.41e-06 ± 9.4e-06** | 0.260 | 0.260 | +0.000 | **−1.70e-05** ± 1.1e-05 | 7.44e-04 | ⛔ | ✅(−) | ⛔⛔ |
| **`h1_r0.6`** | **H1 init, N98 DESIGNED band** | **1.64e-03 ± 1.6e-03** | 0.260 | 0.260 | +0.000 | **−1.58e-05** ± 4.6e-06 | 2.00e-03 | ⛔ | ✅(−) | ⛔⛔ |
| `h1b_r0.15` | H1b place-at-write, 0.5 s | **0.09446 ± 0.025** | 0.260 | 0.260 | +0.000 | +2.09e-04 ± 5.3e-05 | 1.33e-02 | ⛔ | ✅ | ◐ |
| `h1b_r0.3` | H1b, 1 s | 0.08545 ± 0.027 | 0.260 | 0.260 | +0.000 | +1.74e-04 ± 6.0e-05 | 1.34e-02 | ⛔ | ✅ | ⛔ |
| `h1b_r0.6` | H1b, 2 s | 0.08028 ± 0.024 | 0.260 | 0.260 | +0.000 | +1.30e-04 ± 8.1e-05 | 1.31e-02 | ⛔ | ✅ | ⛔ |
| `baseline_w40` | N94's 40-step floor | 0.06631 ± 0.019 | 0.260 | 0.260 | +0.000 | +1.45e-04 ± 1.0e-04 | 1.34e-02 | ⛔ | ✅ | ⛔ |
| `h1b_r0.3_w40` | H1b × 40 steps | 0.08798 ± 0.028 | 0.260 | 0.260 | +0.000 | +1.82e-04 ± 7.9e-05 | 1.29e-02 | ⛔ | ✅ | ⛔ |
| `baseline_m1.0` | margin 1.0, scattered | 0.03175 ± 0.017 | 0.260 | 0.260 | +0.000 | +3.46e-05 ± 9.7e-06 | 8.14e-03 | ⛔ | ✅ | ⛔ |
| `h1b_m0.6` | H1b × margin 0.6 | 0.1489 ± 0.044 | 0.260 | 0.260 | +0.000 | +5.24e-04 ± 3.6e-04 | 1.95e-02 | ⛔ | ✅ | ◐ |
| **`h1b_m1.0`** | **H1b × margin 1.0** | **0.1668 ± 0.037** | 0.260 | 0.260 | **+0.000** | **+7.12e-04 ± 5.3e-04** | **2.17e-02** | ⛔ | ✅ | ◐ |
| `h2_lam0.3` | H2, `λ_traj = 0.3` | 0.02882 ± 0.015 | 0.260 | 0.260 | +0.000 | +2.85e-05 ± 1.3e-05 | 7.84e-03 | ⛔ | ✅ | ⛔ |
| `h2_lam3` | H2, `λ_traj = 3` | 0.02937 ± 0.015 | 0.260 | 0.260 | +0.000 | +3.21e-05 ± 9.6e-06 | 7.85e-03 | ⛔ | ✅ | ⛔ |

**Legend, applied mechanically by `aggregate()` — no hand-grading.** (a) `acq > chance + 2 SE`;
(b) `|live − blank| > 1e-6` bpc, i.e. above float32 round-off; (c) `✅` = depth inside the shipped
**0.46–0.80** band, `◐` = depth `> 2 × 0.045` (leaves the pilot's saturation) but below the band,
`⛔` = neither, `⛔⛔` = moved the **wrong way** vs the baseline.
⚠ **Read (b) with two caveats.** First, `✅(−)` marks the three H1 cells whose gap clears round-off
**with the wrong sign** — the localized init makes the live store *worse* than a blank one. Second,
**this is the UNTRAINED gap**: every cell clears round-off before the outer loop runs, and §7 shows
the baseline's gap collapsing to **−2.87e-08 ± 2.9e-08** — the pilot's *live = blank at float32* — once
200 steps of training have run. **The trained column is the one that answers the pilot's question.**

### 2.1 The three signals, adjudicated exactly as pre-registered

- **(a) ACQUISITION: ⛔ NOT MET ON ANY CELL, and the failure is stronger than "at chance".**
  `acq` equals its own chance level **to the digit** on 14/14 cells × 3 seeds, and
  **`acq(live) − acq(blank) = 0.000 exactly on every seed of every cell** — the live and blank
  self-probes produce *identical hit patterns*. The pilot's monitor #5 reading is reproduced and
  sharpened: this is not noise around chance, it is a **structurally query-independent decode** (§6).
- **(b) LIVE ≠ BLANK: ✅ MET UNTRAINED, ⛔ NOT MET TRAINED — and the trained column is the one that
  answers the pilot.** Untrained, the live-vs-blank bpc gap goes **+2.80e-05 → +2.09e-04** under H1b
  (7.5×) and **→ +7.12e-04** under H1b × margin 1.0 (**25×**), with the read-output delta rising
  **8.07e-03 → 2.17e-02**: the store's content measurably changes the block's output. ⚠ **After 200
  steps of outer training the baseline's gap collapses to −2.87e-08 ± 2.9e-08 (= the pilot's *live =
  blank at float32*) and `h1b_m1.0`'s to −7.08e-06 ± 2.7e-05 — 250× larger in magnitude, but neither
  significant nor sign-consistent** (§7).
- **(c) DEPTH: ◐ PARTIAL, on 3 of 14 cells.** H1b × margin 1.0 reaches **0.1668 ± 0.037** — **5.8× the
  pilot's 0.045 saturation**, the first configuration measured anywhere that leaves it — but still
  **2.8× below the shipped band's floor (0.46)**, so ✅ is met nowhere. ⛔ H1, the registered primary,
  moves depth **the wrong way by up to 6 orders of magnitude.**

⭐ **Verdict in the task's own words: a PARTIAL WAKING, and not the one that was hypothesised.**
Depth and the live-vs-blank channel wake; acquisition does not move at all; and the lever that wakes
them is **not** the registered H1 but a streaming-adapted placement plus a raised hinge margin.

---

## 3. ⛔ H1 AS REGISTERED IS WORSE THAN DOING NOTHING — and the reason matters

`atom_local_radius` at **every** radius, including N98's designed `2s = 0.6`, drives the fitted depth
from `0.0288` to **1.6e-03 / 9.4e-06 / 2.3e-08** and the read-output delta from `8.1e-03` to
**2.0e-03 / 7.4e-04 / 1.5e-04`. The live-vs-blank bpc gap goes **negative**.

**Why (measured, not conjectured).** N98 as it ships is a **static init** localization around targets
that must be fixed *before the stream starts*. In the shipped store harness those targets are the
designed item sites and are known. **In a streaming block the site an item will occupy is chosen by
the controller when the chunk arrives**, so localizing group *j* around the φ-image of calibration
chunk *j* puts that group's 128 atoms around a point the item slot *j* later holds has no reason to be
near — and the write, masked to that group, cannot bring them back inside 4 (or 40) sign-SGD steps.

⭐ **The corollary is the useful part, and it reverses the pilot's reading of its own baseline.** The
scattered `init_scale = 1.0` was not merely neutral — it was **doing the only work there was**: with
128 atoms scattered in a `dim = 3` ball, one or two land near any given site by luck, and the pilot's
saturated `D ≈ 0.045` is *exactly one atom's worth of amplitude* (`(0.01 + 4×0.05)² = 0.0441`).
Localizing at a stale target removes even that luck. ⛔ **N98's cross-harness caveat (N111) now has a
third harness and a mechanism: the localized init cannot be quoted as a general gain, and in a
streaming block at a stale target it is actively harmful.**

**H1b — the streaming form (PREREG ADDENDUM 1).** Re-drawing the written slot's atom **address**
coordinates into a ball around the **incoming chunk's own address** (payload untouched — N46; fixed
key-free jig, so no parameter and no state byte; C3-local; a refused offer still leaves `V_θ`
bit-identical) does what H1 was supposed to do: depth **0.0288 → 0.0945** and the live-vs-blank gap
**7.5×**. The radius is flat across the band (0.0945 / 0.0855 / 0.0803 at 0.15 / 0.3 / 0.6).

---

## 4. ⛔⛔ H2 IS ARITHMETICALLY INERT — bit-identically, at 3.9× the wall clock

`write_lambda_traj ∈ {0.3, 3}` reproduces the baseline's depth to 4 significant figures on all 3
seeds (`0.02571 / 0.004661 / 0.0561` at λ = 0.3 — **the baseline's own values**), and the whole cell
costs **402–444 s against the baseline's 97–104 s (3.9–4.3×)**.

**The mechanism, isolated and pinned as a test.** The term is genuinely wired — it evaluates to a
penalty of **0.183** — but its gradient into the atoms is **~1e-13** (leaf max |g| = 2.0e-13 / 5.1e-14
/ 1.2e-11), because at `atom_depth_init = 1e-4` the read path is dominated by the confinement term and
carries essentially nothing back to the atoms. With the shipped **sign-SGD** inner write (which needs a
*sign flip*, not a magnitude) the written `V_θ` is **bit-identical with and without the term, at every
budget and under plain SGD too** (`tests/test_placement_probe.py::test_lambda_traj_is_wired_but_its_
gradient_is_vanishing_at_the_flat_init`).

⭐ **This is the same arithmetic-inertness mechanism the pilot found in §6.2 — on the other side of the
objective.** It is coefficient-linear, so it settles the whole registered band: at λ = 30 the gradient
is 3e-12 and still cannot flip a sign. ⛔ **C2W2's `λ_traj` is not a lever in the streaming block until
the landscape is no longer flat**; C2W2's own measured *monotone cost* was on a store trained to
convergence and does not transfer either way.

---

## 5. ⭐⭐ WHAT ACTUALLY BOUNDS THE DEPTH: the hinge margin — and it is NOT the budget, NOT the placement

Three measurements, each 3 paired seeds:

| what was multiplied | cell pair | depth |
|---|---|---|
| **write budget ×10** (4 → 40 inner steps), scattered | `baseline` → `baseline_w40` | 0.0288 → **0.0663** (2.3×) |
| **write budget ×10**, with placement | `h1b_r0.3` → `h1b_r0.3_w40` | 0.0855 → **0.0880** (**1.03× — nothing**) |
| **hinge margin ×6.7** (0.15 → 1.0), scattered | `baseline` → `baseline_m1.0` | 0.0288 → **0.0318** (**1.10× — nothing**) |
| **hinge margin ×6.7**, with placement | `h1b_r0.3` → `h1b_m1.0` | 0.0855 → **0.1668** (**2.0×**) |

- ⛔ **PREREG ADDENDUM 2's B2 and B3 are REFUTED:** placement and budget do **not** compound. At N94's
  40-step floor — the one budget at which readings are not demoted by monitor #13 — placement buys
  **1.03×**. Whatever bounds the depth, it is not the number of steps.
- ⭐ **Placement and margin DO compound, and neither works alone** (margin alone at the scattered init:
  1.10×; placement alone: 3.0×; both: 5.8×). The mechanism registered in ADDENDUM 3 is the right shape:
  `write_loss`'s minimum term is a **relu hinge**, `relu(V(z) − V(z+δ) + margin)`. It asks for a well
  `margin` deep **and stops** — once satisfied the loss is exactly 0 and the gradient dies. The pilot's
  saturation (0.033 / 0.045 / 0.044 at 4/16/64 steps) **is the shipped `write_margin = 0.15`**, not a
  budget limit; and raising the margin only helps where atoms are close enough for the hinge to act on.
- ⛔ **But ADDENDUM 3's C1 is REFUTED on magnitude**: depth does not scale linearly with the margin
  (predicted 0.35–0.9 at margin 1.0; measured **0.1668**), so a second ceiling sits below the hinge —
  §6 identifies it.

---

## 6. ⭐⭐⭐ THE ISOLATED CAUSE OF THE ACQUISITION NULL: the payload channel, and where it is lost

`decode_dispersion.json`, `qstar_payload.json` — lane 0, seed 0, 3 live items, three cells.

**6.1 The read's decoded payload is compressed ~27× and is not item-specific.**

| cell | decoded payloads | true payloads | decoded range | payload range | live-vs-blank decode max │Δ│ |
|---|---|---|---|---|---|
| `baseline` | −0.0686 / −0.0639 / −0.0601 | −0.622 / −0.779 / −0.850 | **0.0085** | 0.2277 | 0.0031 |
| `h1b_r0.3` | −0.0656 / −0.0638 / −0.0591 | (same) | 0.0065 | 0.2277 | 0.0061 |
| `h1b_m1.0` | −0.0675 / −0.0608 / −0.0558 | (same) | 0.0117 | 0.2277 | 0.0063 |

The between-item spread of the decode (0.0065–0.0117) is the **same order as the difference between
reading the live store and reading a blank one** (0.0031–0.0063). ⭐ **That is the quantitative form of
"the read does not carry the payload", and it explains the exact `acq = chance` equality**: the
nearest-stored-payload assignment is a fixed permutation of a nearly constant vector, which yields
exactly one hit per lane ⇒ `acq = mean(1/n_live) = chance`, identically. (On these 3 items the decode
is also strongly *anti*-correlated with the truth, `r = −0.99 / −0.89 / −0.99` — reported as one
replicated observation on one lane, not as a 3-cell result.)

**6.2 The loss is NOT in the dynamics — it is downstream of them.** The settled point's payload
coordinate `q*[payload]` (blank store: **0.000** on all items):

| cell | `q*[payload]`, live | true payloads | fraction of the way | between-item spread |
|---|---|---|---|---|
| `baseline` | −0.233 / −0.181 / −0.211 | −0.622 / −0.779 / −0.850 | **~30 %** | 0.053 |
| `h1b_r0.3` | −0.411 / −0.297 / −0.371 | (same) | **~48 %** | 0.114 |
| `h1b_m1.0` | −1.089 / −1.090 / −1.143 | (same) | **130–175 % (overshoot)** | **0.054** |

- ⭐ **The two-phase relaxation DOES carry payload information** — the read leaves the payload-zero
  launch manifold by 0.18–1.14 when the store is live and by **0.000** when it is blank. The store is
  not inert at the level of the dynamics.
- ⭐⭐ **ψ then throws most of it away:** `q*` spread 0.053–0.114 → decoded spread 0.0065–0.0117, a
  further **7–25× compression** in the read-out.
- ⛔⛔ **And raising the margin makes DISCRIMINATION worse while making depth better:** at margin 1.0
  every item's `q*` lands at ≈ −1.1 regardless of its own payload (spread collapses 0.114 → 0.054).
  **A deeper well at a shared payload location is a worse memory.** This is the single most
  decision-relevant result in the report and it directly contradicts *"just make the wells deeper"*.

⭐ **Consequence, stated plainly: of the pilot's three §5.3 candidates, the two that were ranked first
are measured and fail, and the evidence now indicts the third — the ψ payload residual — which was
declared third-priority.** It is a **declared NOT-RUN** here (§9), with the diagnostic that makes it
the obvious next task attached.

---

## 7. THE TRAINED TIER (200 steps, 3 seeds) — and a failure mode nobody had named

| cell | CLU bpc | blank bpc | **live − blank** | **CLU − memory-deleted** | depth after training | acq / chance | plan-pass frac |
|---|---|---|---|---|---|---|---|
| `baseline` | 4.6121 ± 0.0148 | 4.6121 ± 0.0148 | **−2.87e-08 ± 2.9e-08** | **+0.0039 ± 0.0049** | **4.95e-63** | 0.1741 / 0.1741 | 0.304 ± 0.008 |
| `h1b_m1.0` | 4.6101 ± 0.0155 | 4.6101 ± 0.0155 | −7.08e-06 ± 2.7e-05 | +0.0019 ± 0.0067 | **0.0616 ± 0.037** | 0.1715 / 0.1715 | 0.304 ± 0.005 |

- ✅ **The pilot's headline reproduces in an independently-written harness:** trained live-vs-blank is
  **0.0 / −8.6e-08 / 0.0** bpc (pilot: 0.0 / −5.96e-08 / 0.0) — *live = blank at float32*, and the
  memory is still a net cost vs the memory-deleted arm (+0.0039 ± 0.0049; pilot +0.0070 ± 0.0055).
- ⛔⛔ **R3 — the outer optimisation DESTROYS the store's wells.** Untrained depth **0.0288** →
  trained **4.95e-63** at the shipped config. This is the same object as the pilot's monitor #9
  `Δ_ret = 7.8e-86` ("clear because the wells have no depth to differ in"), whose cause was never
  named. **Placement + margin survives it** (0.0616 ± 0.037), and its live-vs-blank gap stays 250×
  larger in magnitude — but not significantly, and sign-inconsistent across seeds. ⚠ So the untrained
  gain of §2 is **largely undone by 200 steps of training**: the screen tier's ✅ on (b) becomes ◐ once
  the outer loop runs, and **that, not the untrained number, is what the CSF3 run will see.**
- ⚠ 200 steps × 2048 tokens = 409 600 tokens; every arm sits at 4.59–4.64 bpc, barely past unigram
  statistics. Only paired per-seed margins are defensible.

---

## 8. ⭐ THE PLAN PASS — priced, and the fix built and measured (task §2)

**The pilot's attribution was wrong (R1).** Decomposing the plan pass by wrapping
`_controller_plan_for_lane` in a timer:

| | before | after `filter_jit` |
|---|---|---|
| plan pass, warm | **2.755 s** | **0.127 s** (**21.7×**) |
| — of which the **Python controller** | **0.044 s (1.6 %)** | 0.043 s (**34 %**) |
| — of which the **JAX forward** | 2.711 s (**98.4 %**) | 0.084 s |
| differentiable fwd + bwd (already jitted) | 0.388 s | 0.420 s |
| **plan pass / step** | **0.877** | **0.232** (measured in situ over 6 trained runs: **0.304 ± 0.008**) |
| **end-to-end step** | 3.14 s | **0.547 s (5.7×)** |
| one-off compile | — | ≈ 1.9 s |

So *"77–84 % of CLU wall-clock is the Python plan pass"* is right about the plan pass and **wrong
about the Python**: the plan pass ran the per-layer block forward **eagerly, op-by-op**, next to a
`filter_jit`-ed training step. **The cheap fix was not a controller vectorisation at all** — it is
`filter_jit` on the three stages (embed, `chunk_latents`, block forward) plus the evaluation loss.
**Built, ~1 h, committed (`46755fb`).**

**Decision-identity, checked and honest:** every **discrete** plan field (`slot`, `admitted`, `reset`,
`live`, `retry`) is **bit-identical** to the eager reference and held-out bpc is **bit-identical**;
the recorded `sites` differ by **≤ 6e-08** (float32 ULP, XLA fusion). A knife-edge admission could in
principle flip; none did, and the regression test is the tripwire.

### 8.1 The GPU-idle fraction for the recommended config, priced

The only CPU-serial (GPU-idle) work left is the **Python controller**: `n_layers × batch` lane-calls
per step, each a sequential loop over the chunks. Measured **~1.7–2.4 ms per lane-chunk**, roughly
independent of `capacity` (K = 8 / 32 / 64 all in band).

- **Toy, measured:** 2 × 4 × 16 = 128 lane-chunks ⇒ 0.043 s Python vs 0.504 s JAX ⇒ **GPU idle ≈ 7.9 %**.
- **PILOT (12 layers × 8 lanes × 16 chunks = 1536 lane-chunks):** **≈ 2.6–3.7 s/step of pure Python**
  (measured rate × count), unchanged by any GPU.
- On an A100 the pilot step's JAX work is **not measurable from here**. Two bracketing readings:
  if the A100 does the pilot forward+backward in **1 s**, GPU idle ≈ **72–79 %**; if in **4 s**, ≈
  **39–48 %**. ⛔ **Either way the GPU idles behind the controller for a large fraction of a
  ≤ 12 h job**, and the honest form is *"2.6–3.7 s/step of Python that no GPU can absorb"*.
- ⭐ **The remaining fix, PRICED not built (task §2's alternative branch):** the **lanes are
  independent** (each `_controller_plan_for_lane` builds its own controller), so a `ProcessPoolExecutor`
  over the batch axis is an **8× cut on the dominant term** (layers must stay sequential — layer *l+1*'s
  latents need layer *l*'s decisions). It is **not free**: the per-lane call currently returns a live
  `CluControllerV0` (for the monitors) and shares the monitor `registry` for guard counts, so both
  would have to become picklable summaries. **Estimate: ~half a day, one spoke, self-contained in
  `plan_pass`/`_controller_plan_for_lane`.** At 8 lanes it takes the pilot's Python from ~3 s to
  ~0.4 s/step and the GPU-idle fraction to **< 30 %** on any of the above brackets.
  **Recommendation: fund it before the CSF3 submission if the Head wants the A100-hours to buy
  compute rather than wall-clock; it is not a correctness blocker.**

---

## 9. ⛔ DECLARED CUTS AND NOT-RUNs (never reported as nulls)

- **CUT (PREREG §5 cut order, applied):** the two interaction rows **`h1h2`** and **`h1bh2`**. H2 was
  measured bit-identical at `λ_traj = 0.3` on 3/3 seeds at 3.9× the wall clock, and its gradient is
  **~1e-13 and coefficient-linear**, which settles the interaction without the cells. The freed budget
  paid for the margin cells, which are decisive where the interaction rows are not.
- **NOT RUN — the ψ payload residual** (the pilot's third §5.3 candidate). Task §1 makes it
  third-priority and conditional; §6 now specifically indicts it, so it is **the recommended next
  task**, not a gap I filled badly.
- **NOT RUN — anything at 26–47 M / on CSF3.** No cluster route from this machine (pre-registered by
  the pilot). ⛔ No number here is a pilot-scale number.
- **NOT RUN — the full 5-arm swap table per probe cell.** The probe's controls are live / blank /
  memory-deleted; the matched GRU/TTT swap is the pilot's protocol and is not re-run per cell.
- **NOT RUN — `atom_init_width`** (N111's substitute lever; moving it would confound the probe),
  WikiText-103, deeper stacks, larger `capacity`, the trained tier for cells other than `baseline` and
  `h1b_m1.0`, and any plot (no figure was needed for a 3-signal table).
- **UNDER-POWERED, declared:** the §6 decode/`q*` diagnostics are **one lane, one seed, 3 live items**.
  They are mechanism isolation, not statistics, and are labelled as such wherever quoted.

---

## 10. ⭐⭐ THE CSF3 RECOMMENDATION BLOCK (the deliverable the Head submits with)

**Status rule honoured (A18.4): this informs the submitted config; it NEVER gates the commitment.**
The scale run happens regardless. Every knob below is reachable **as a flag** — `85a557d` adds
`--set` / `--mem` / `--store` to the runner and `SET` / `MEM` / `STORE` pass-through to
`scripts/csf3/job_gpu_cluformer.sh`, all defaulting to EMPTY, so **an unmodified submission is
bit-identical to the pre-probe one** and a modified one is recorded verbatim in the artifact's `flags`
block (a module edited on the cluster is a provenance hole).

```bash
# scripts/csf3/job_gpu_cluformer.sh — the pilot-placement-probe recommended submission
sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,\
MEM="atom_place_radius=0.3 write_inner_steps=40",\
STORE="write_margin=0.6",\
SET="monitor_every=25" \
       --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
```

| # | knob | **recommended** | evidence | confidence |
|---|---|---|---|---|
| 1 | **atom init scheme** | ⛔ **`atom_local_radius` = 0.0 — do NOT turn the N98 init on** | §3: at the designed band it costs **6 orders of depth**; the site is not known at init in a streaming block | **high** |
| 2 | **placement** | ✅ **`atom_place_radius = 0.3`** (= 1·`atom_width`; flat over 0.15–0.6, so the exact value is not critical) | §2: depth ×3.0, live−blank gap ×7.5; §5: it is the only lever the margin composes with | **high** |
| 3 | **write budget** | ✅ **`write_inner_steps = 40`** — N94's maturity floor | §5: it buys **1.03×** with placement, so it is **not** bought for depth; it is bought so **monitor #13 stops demoting every reading** (the toy 4-step reading is non-promotable and says so) | **high** (for the reason stated, not for depth) |
| 4 | **hinge margin** | ◐ **`write_margin = 0.6`** (4×), ⛔ **not 1.0** | §5: 0.6 gives depth 0.1489 ± 0.044 vs 1.0's 0.1668 ± 0.037 (no gain), and §6.2 shows 1.0 **collapses the between-item `q*` spread 0.114 → 0.054** — deeper wells at a *shared* payload location. **Depth is not the objective; discrimination is.** | **medium** — a genuine trade-off, and the first knob to sweep if a sweep is affordable |
| 5 | **read budget + γ** | keep `address_steps = read_steps = 64` (the PILOT default). ⛔ **Every γ statement must be scoped by read budget**: `ρ_conv` is 0.834 at 48 steps and 2.6e-07 at 1200 — the read **does not settle at chunk granularity**, by design | inherited R3 of the pilot; unchanged by this probe | high |
| 6 | **φ-gain calibration** | ✅ **keep it** (RMS address norm = `ball_radius`), unchanged, per seed, shared by every arm | it is the declared anti-collapse init; without it the store refuses 60/64 offers at step 0 | high |
| 7 | **monitors** | all 13 + M14; **`monitor_every = 25`** (not 100) | monitor #6 needs ≥ 3–4 consolidation windows through one persistent registry and was **INAPPLICABLE** in the pilot at 200 steps / `monitor_every 100`. At 4000 pilot steps, 25 gives 160 observations — #6 becomes applicable without measurable cost | high |
| 8 | **trajectory write term** | ⛔ **`write_lambda_traj = 0.0` — do not enable** | §4: bit-identically inert at 3.9–4.3× the wall clock. Enabling it would spend ~25 % of the A100 budget on a no-op | **high** |
| 9 | **swap protocol** | unchanged and non-negotiable: matched-params GRU + **matched-both TTT** + memory-deleted + echo + blank, dyn-eval column **in the table**, 3 seeds | A18.4: persistence governs scheduling, never controls | high |
| 10 | ⚠ **what to watch, and what would make the run a null-with-mechanism** | log the **untrained-vs-trained well depth** and the **`q*` payload spread** per seed | §7 R3: 200 steps drive depth to **1e-63** at the shipped config. If that happens at 4000 steps too, the scale run measures a store that was destroyed by its own outer loop — and the fix is a *training* fix, not a scale fix | **high — this is the probe's most actionable warning** |

⚠ **Two things the recommendation deliberately does NOT do.** (i) It does not claim the store will
wake: acquisition is at chance in **every** configuration measured, and §6 says why. (ii) It does not
touch the payload axis of the init — localizing it would hand the write the value it is supposed to
learn (**N46**) and destroy the basin-reach property (**N100**). That prohibition is exactly why the
payload channel is the open mechanism and not a knob.

---

## 11. PREREG SCORECARD (registered → measured → verdict)

`PREREG.md` + ADDENDUM 1 (H1b), ADDENDUM 2 (budget × placement), ADDENDUM 3 (the hinge margin) — each
filed **before** the cells it governs ran.

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | H1 `r = 0.6` depth **1.0**, band [0.3, 4.0] | **1.6e-03** | ⛔⛔ **REFUTED by 3 orders, and in the wrong direction** |
| **P2** | H1 `r = 0.3` depth **2.5**, band [0.5, 8.0] | **9.4e-06** | ⛔⛔ **REFUTED** |
| **P3** | H1 `r = 0.6` live−blank > 1e-3 bpc | **−1.58e-05** (negative) | ⛔ **REFUTED** |
| **P4** | H1 acquisition **at chance** ⇒ H1 fails (a) | at chance, exactly | ✅ **CONFIRMED** (for the wrong reason — I predicted a partial waking, not a collapse) |
| **P5** | the localized init buys blank-store hits | `acq(blank)` identical to `acq(live)`, and unchanged from the scattered blank | ⛔ **REFUTED — no laundering occurred**, because the localized init helps nothing at all |
| **P6** | `acq(live) − acq(blank) ≈ 0` | **0.000 exactly, 14 cells × 3 seeds** | ✅✅ **CONFIRMED exactly** |
| **P7** | H2 acquisition at chance | at chance | ✅ **CONFIRMED** |
| **P8** | H2 depth unchanged ±50 % | **bit-identical to 4 s.f.** | ✅✅ **CONFIRMED, far tighter than registered** |
| **P9** | H1+H2 no rescue | ⛔ **CUT** (§9) — settled by H2's ~1e-13 gradient instead | not scored |
| **P10** | baseline reproduces the pilot | depth 0.0288 (pilot 0.033–0.045); trained live−blank **−2.9e-08** (pilot 0.0 / −5.96e-08 / 0.0); CLU − memory-deleted **+0.0039 ± 0.0049** (pilot +0.0070 ± 0.0055) | ✅ **CONFIRMED** |
| **P11** | monitor #5 trips on every arm | trips 3/3 on every cell | ✅ **CONFIRMED** |
| **P12** | monitor #8 degrades under H1 | not scored — H1's wells never existed, so the certificate has nothing to merge | not scored |
| **A1** | H1b `r = 0.3` depth **1.26**, band [0.3, 4.0] | **0.0855** | ⛔ **REFUTED by 15×** (direction right, magnitude badly over-predicted) |
| **A2** | H1b `r = 0.6` depth **0.60**, band [0.15, 2.0] | **0.0803** | ⛔ **REFUTED** |
| **A3** | H1b live-vs-blank read delta ≥ 10× baseline | 8.07e-03 → 1.34e-02 (**1.7×**); the *bpc* gap 7.5× | ◐ **PARTIAL** — the bpc gap nearly makes it, the read delta does not |
| **A4** | H1b acquisition NOT off chance | at chance | ✅ **CONFIRMED** |
| **A5** | `acq(live) − acq(blank) ≈ +0.05` | **0.000** | ◐ direction vacuous; the bar (not > 2 SE) is met |
| **A6** | the residual constraint is the **payload channel** | ✅ **CONFIRMED and localised**: `q*` carries 30–50 % of the payload, ψ compresses it a further 7–25×, and the margin *collapses* the between-item spread | ✅✅ **CONFIRMED — the report's central finding** |
| **A7** | H1b × H2 no rescue | ⛔ **CUT** | not scored |
| **B1** | `baseline_w40` depth ≈ 0.045 | **0.0663 ± 0.019** | ◐ **PARTIAL** (2.3× the 4-step value, not "unchanged"; still nowhere near the shipped band) |
| **B2** | `h1b_r0.3_w40` depth ≥ 0.5 | **0.0880** | ⛔ **REFUTED by 6×** |
| **B3** | placement × budget are **complements** (≥ 5× vs ≤ 2×) | **1.03× vs 2.3×** — the ratio is *inverted* | ⛔⛔ **REFUTED, and this is a finding: the budget is not the constraint at either init** |
| **B4** | acquisition off chance at 40 steps (P = 0.45) | at chance | ⛔ **REFUTED — and its own falsifier clause fires: the payload channel is not under-budgeted, it is structurally unreachable by this write** |
| **B5** | `acq(live) − acq(blank)` largest here | 0.000 everywhere | ⛔ **REFUTED** |
| **C1** | depth scales ~linearly with the margin ⇒ 0.35–0.9 at margin 1.0 | **0.1668 ± 0.037** | ⛔ **REFUTED on magnitude**, ✅ on direction and on the *interaction* (margin does nothing without placement: 1.10×) |
| **C2** | the margin helps less at the scattered init | 1.10× vs 2.0× | ✅ **CONFIRMED** |
| **C3** | acquisition **still** at chance at margin 1.0 | at chance | ✅ **CONFIRMED** |
| **C4** | *(the flip case)* if C3 were wrong, depth was the constraint | did not happen | the registered primary held |
| **C5** | monitor #8 degrades as the margin rises | not scored (not isolated per cell) | not scored |

**Score: 9 confirmed (2 exactly) · 4 partial · 11 refuted · 5 not scored.** ⭐ **The refutations are the
report.** I pre-registered a partial waking under the *registered* H1 and got a **collapse**; I
pre-registered placement × budget complementarity and got the **inverse**; I pre-registered depth
scaling with the margin and got **2×, not 6.7×**. The one prediction that survived intact (**A6**) is
the one that names the mechanism, and it survived a falsifier (**B4**) designed to kill it.

---

## 12. HOW I VERIFIED (commands + observed output)

| check | command | observed |
|---|---|---|
| the screen grid | `PYTHONPATH=. python -u -m chlu.experiments.exp_placement_probe --tier screen --seeds 0 1 2` (+ `--cells … --tag screen2/margin`) | 42 screen records, 14 cells × 3 seeds; per-cell log lines in `screen*_run.log`, `margin_run.log` |
| the trained tier | `… --tier trained --cells baseline h1b_m1.0 --seeds 0 1 2 --steps 200` | 6 records; `trained_run.log` |
| rider 1 | `.claude/scratch/pilot-placement-probe/n9_aggregate.py` (imports the shipped `audit_table`) | 4 columns × n = 9; F3's published n = 9 values reproduce digit-for-digit |
| rider 2 | `.claude/scratch/pilot-placement-probe/deltanet_frontier.py` | 9 frontier cells, 63–79 s each; 0/15 rows rescued |
| plan-pass decomposition | timer-wrapped `_controller_plan_for_lane` inside `plan_pass` | 1.6 % controller / 98.4 % forward; after jit 21.7× |
| **new tests** | `pytest tests/test_placement_probe.py -q` | **22 passed** |
| **regression (the files I touched), re-run at branch HEAD `df36427`** | `pytest tests/test_placement_probe.py tests/test_cluformer_pilot.py tests/test_blocks.py -q` | ✅ **60 passed in 154 s, 0 failed** (22 mine + 16 pilot + 22 blocks) |
| lint | `ruff check chlu/ tests/ scripts/` | **All checks passed** at every commit |
| job script | `bash -n scripts/csf3/job_gpu_cluformer.sh` | clean; ⛔ **UNTESTED ON CLUSTER** |

⚠ **Full-suite status, stated honestly: I ran the three affected modules (59 passed), not the whole
1189-test suite** — the suite takes ~25 min and the probe grid held the machine. The files I touched
are `chlu/core/blocks.py` (additive), `chlu/training/train_cluformer.py`,
`chlu/experiments/exp_{placement_probe,cluformer_pilot}.py`, `scripts/csf3/job_gpu_cluformer.sh` and my
own test module; the only one imported outside the tier-iii tests is `blocks.py`, whose 22 existing
tests are in the 59. **A full-suite run before merge is owed and is the Hub's gate, not a claim of
mine.**

---

## 13. GIT FOOTPRINT

**Branch** `agent/experiment-engineer/pilot-placement-probe`, **worktree `../CHLU-placement`**, base
local `main @ 29fc22b` (did not move under me; `git rebase main` is a **no-op**). ⛔ **Not pushed, not
merged.** ⛔ `origin` never touched.

| commit | subject |
|---|---|
| `2db0fbc` | blocks: the placement-probe levers (N98 localized init, placement-at-write, the C2W2 trajectory write term) |
| `edec27a` | the placement probe: runner, pooled multi-lane self-probe, tests |
| `46755fb` | plan pass: jit the forward (21.7×); it was never the Python controller |
| `85a557d` | pilot runner + CSF3 job: config overrides as FLAGS, not module edits |
| `df36427` | probe: the hinge-margin cells + a store-override hook (PREREG ADDENDUM 3) |

**Files touched — my declared list, nothing else:**
`chlu/core/blocks.py` (additive: 5 config fields + `localize_atom_init` + 2 hunks in `CluStoreCell`) ·
`chlu/training/train_cluformer.py` (`calibrate_atom_group_centers`; the 3 jitted stages + `_eval_loss`) ·
`chlu/experiments/exp_placement_probe.py` (**new**) · `chlu/experiments/exp_cluformer_pilot.py`
(`_parse_kv` + 3 argparse flags) · `scripts/csf3/job_gpu_cluformer.sh` (3 env vars, defaults empty) ·
`tests/test_placement_probe.py` (**new, 22 tests**).

⛔ **NOT touched:** `chlu/config.py` · `chlu/core/monitors.py` (`orgdiv-cat-test`'s) · the
factored-store family files · `chlu/core/{clu_system,admission,placement,memory_potentials,
controller,clu_controller,soft_certificate,psi_readout,implicit_grad}.py` · `chlu/eval/**` ·
`chlu/cli/experiment_cmd.py` · `chlu/experiments/{exp_bprime_rivals,memory_gym}.py` (imported
**read-only** by the riders). **No collision with any concurrent agent's declared files.**

⚠ **Worktree deliberately NOT removed** — left for Hub review; remove with
`git worktree remove ../CHLU-placement` afterwards. Branch ref verified from the MAIN repo
(`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/pilot-placement-probe`
→ all 5 commits present). ⚠ The runner writes artifacts relative to its cwd, so the JSONs were
produced under the worktree's gitignored `.claude/` and **copied** to the main repo's
`.claude/outputs/pilot-placement-probe/`; both copies are byte-identical.

⚠ **One provenance note:** the screen grid's first 24 records were produced by a build in which the
write-time placement was applied *before* the admission blend, so a **refused** offer also moved
atoms. That was caught by `test_a_refused_offer_leaves_the_landscape_bit_identical_under_placement`,
fixed (`2db0fbc`), and **the entire grid was re-run from scratch** on the fixed build. Every number in
this report is post-fix; the pre-fix run was discarded, not merged.

---

## 14. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. ⭐⭐ **The next experiment is the ψ payload residual, and §6 says exactly what to measure:** the
   dynamics deliver 30–50 % of the payload into `q*`; ψ compresses the between-item spread a further
   7–25×. A payload-carrying residual in ψ is a ~1-day build with a sharp pre-registerable target
   (does the decoded spread reach the `q*` spread?).
2. ⭐ **The margin is a trade-off, not a knob**: it buys depth and *costs* between-item `q*` spread.
   Somebody should sweep `write_margin ∈ {0.15, 0.3, 0.6, 1.0}` **scored on the spread**, not on depth.
   That is the single cheapest remaining tier-ii-relevant measurement.
3. ⛔ **R3 is a training-time failure mode with no owner**: 200 outer steps take depth to 1e-63. Is φ
   drifting away from the sites, or is `amp` being pushed down? One gradient probe answers it.
4. **The lane-parallel controller (§8.1) is priced at ~half a day** and takes the pilot's GPU-idle
   fraction below 30 %. Head call, before or after the CSF3 submission.
5. **Risk on the record:** every probe number is at `dim = 3`, `capacity = 8`, 16 chunks, 0.16 M
   params, one lane for the §6 diagnostics. ⛔ **Nothing here transfers to a 26–47 M claim**, and the
   recommendation block is a *best-informed guess*, not a validated config.
6. **Risk:** the jit changes `sites` by ≤ 6e-08. On a knife-edge admission at pilot scale a decision
   could flip relative to the pilot's published toy numbers. The test is the tripwire; if a future
   comparison disagrees at the discrete level, this is the first thing to check.

---

## Proposed handover updates (for the Hub)

**§7 Known Issues — ADD:**
- ⛔ **`CluSystemConfig.atom_local_radius` is DEAD in the `LearnedVStore` path** — the field exists and
  is documented, but `LearnedVStore.__init__` never forwards it (or `atom_group_centers`) to
  `DesignFreedomPotential`, so the shipped N98 lever is **unreachable through the store's own config**.
  The probe reaches it from outside (`chlu.core.blocks.localize_atom_init`, bit-identity asserted). A
  one-line forward in `clu_system.py` would fix it — **deliberately not done here** (that file is
  outside my declared ownership this wave).
- ⚠ **The trajectory write term (`lambda_traj`) is arithmetically inert at `atom_depth_init = 1e-4`**:
  penalty 0.183, gradient ~1e-13, and with sign-SGD the written `V_θ` is bit-identical. Any future use
  must first check the gradient magnitude, not just that the term is non-zero.
- ⚠ **`write_margin` is the depth ceiling of the streaming write**, not the step budget: `write_loss`'s
  relu hinge stops at `margin`. Raising it deepens wells **and collapses between-item discrimination**.
- ⛔ **200 steps of outer training drive the in-block store's well depth to ~1e-63** at the shipped
  config (the cause of the pilot's monitor #9 `Δ_ret = 7.8e-86`).

**§10 running log / N-registry (N196 amendments):**
- ⛔ **N196's compute clause is WRONG and must be amended (R1):** the plan pass is 87.7 % of the step
  but only **1.6 % of it is the Python controller**; the fix is `filter_jit`, **built and measured
  (21.7× on the plan pass, 5.7× end-to-end, decision-identical)**. `future_work.md`'s "the controller
  loop needs vectorising" should become "the plan pass needed jitting (done); lane-parallelism is the
  remaining ~half-day item, priced".
- ⭐ **N196's placement hypothesis is MEASURED and the verdict is three-part:** the shipped N98 init
  **fails and is harmful in a streaming block** (site unknown at init); a streaming-adapted
  placement-at-write **works on depth and on live-vs-blank but not on acquisition**; and the binding
  constraint is the **payload channel** — `q*` carries 30–50 % of the payload and **ψ compresses it a
  further 7–25×**. The pilot's third-priority candidate is now the first-priority one.
- ⭐ **N111 gains a third harness:** the localized init is not merely a null at the shipped width — in
  a streaming block localized at a stale target it costs up to **6 orders of well depth**.
- ✅ **The pilot's own headline independently reproduces** (trained live = blank at float32: 0.0 /
  −8.6e-08 / 0.0; memory a net cost +0.0039 ± 0.0049).

**B′ / draft-r3 (the riders):**
- **Rider 1 delivered** (`n9_full_columns_table.md`): uniform n = 9 on every column; App. I.1c's
  un-aggregated-columns caveat retires; minimum raw margin **4.43 SE**; **R5** (both TTT arms below
  their own same-keys null) and **R4** (TTT ledger columns vary per seed with the selected mini-batch)
  are new and need a home in the draft.
- **Rider 2 delivered** (`n9_deltanet_frontier_table.md`): the labelled deltanet frontier row exists
  and is a **labelled NULL** — ⛔ **0/15 frontier cells rescued at n = 9**, so no frontier margin
  against deltanet, ttt_linear or gdn2 is quotable, and deltanet's `aggregate` rescue does **not**
  transfer to the frontier family.

**§3 config defaults:** none changed. All five probe levers ship **OFF** and are bit-identical when
off; `chlu/config.py` untouched. New CLI: `--set` / `--mem` / `--store` on the pilot runner and
`SET` / `MEM` / `STORE` on the CSF3 job script, all defaulting to empty.

---

## ⛔ DATED ERRATUM BANNER (Hub, 2026-08-01, `[C2W5]` second review — body above UNTOUCHED, C-3 precedent)
§8.1's *"takes the GPU-idle fraction to < 30 % on any of the above brackets"* assumed the full 8× lane
cut. `plan-pass-vectorise` MEASURED **4.93×** (straggler inflation s = 1.16 + a 7.7 ms/layer round-trip):
at the 1 s A100 bracket idle is **34.5–42.9 %** (claim false there); at the 4 s bracket **11.6–15.8 %**
(claim holds); at the low-end Python term both brackets hold (24.5 % / 7.5 %). The pre-registered
likely-outcome branch fired. The ~half-day price and the design were otherwise as stated.

## ⛔⛔ DATED ERRATUM BANNER 2 (curator, 2026-08-05, `doc-curator-c2w6-fold` — filed under Head ruling §A23.4; **body above UNTOUCHED, C-3 precedent; no number in this report is retracted**)

**This banner corrects the QUOTATION FORM of §7's `4.95e-63` and scopes R3. Both come from charter ADDENDUM 7
(§A22–§A23), Head-ratified 2026-08-05 after C2W6 re-measured this report's own R3 finding.**

1. ⛔⛔ **`4.95e-63` IS A CENSORING STATEMENT, NEVER A POINT ESTIMATE** (§A23.4, program-wide). The figure in
   §7's table is the **arithmetic mean** of per-seed depths **`1.439e-87 / 1.486e-62 / 2.493e-177`** — three seeds
   spanning **114.8 orders of magnitude**; the **geometric mean is `3.764e-109`**, 46 orders lower. It is therefore
   ≈ seed 1 ÷ 3, *a single seed wearing a mean's clothes*. ⭐ **Replacement wording, mandatory in every downstream
   quotation:**
   > *"200 outer steps drive well depth below 1e-62 on 3/3 seeds (1.4e-87 / 1.5e-62 / 2.5e-177)."*

   ⚠ **The qualitative R3 claim is UNTOUCHED and if anything STRENGTHENED** — 3/3 seeds annihilate and two go far
   below the quoted figure. ⛔ **Any band phrased as a RATIO to this anchor is ill-conditioned** (the analyst's F1;
   `c2w6-erosion-adjudication` §3, which independently re-derived all nine banked anchors of this report from raw —
   **all nine reproduce**, this one included: 4.954e-63 arithmetically).
2. ⛔⛔ **R3 IS `shipped-config`-SCOPED, AND THE COLLAPSE DOES NOT REPRODUCE AT THE CONFIG NOW SHIPPING.** This
   report measured R3 at **no placement, `write_margin = 0.15`, pre-ψ-fix, 200 steps**. At the **CSF3 run-2** config
   (`atom_place_radius = 0.3`, `write_margin = 0.6`, ψ payload residual ON) over **1000** outer steps × 3 paired
   seeds, the unprotected arm reads **9.782 / 0.9035 / 0.5305 ×** its own step-200 depth against a registered decay
   rule of **≤0.3×** ⇒ **REFUTED 0/3** (`c2w6-anti-erosion` §3; Advisor §A22). ⭐ **Quotable form: *"N223's monotone
   collapse does not reproduce at the run-2 config."*** ⛔ **NEVER *"there is no erosion at the run-2 config"*** —
   that sentence is estimator-dependent (final/untrained **arith 0.708 ± 0.57× vs geo 0.327×**). ⭐ **Where the
   collapse survives is the mechanism: with the ψ payload residual OFF — the store useless to the loss — retention
   falls to 0.141 ± 0.095× untrained** (**N225**).
3. ⛔⭐ **§10 / R3's PRE-REGISTERED CSF3 ABORT CRITERION IS SUSPENDED** (§A23.1) — *not amended.* Replayed against
   C2W6's curves on the exact arm CSF3 is running it is a **measured false-positive generator**: **seed 0 trips on
   15 of 41 readings, first at step 25, reaching 4.12e-4 — below the 1e-3 immediate-escalation floor — then
   RECOVERS 15.2×; seed 2 trips 10/41 from step 50 and recovers 2.6×; only seed 1 never trips.** ⇒ **no run is
   paused or escalated on a depth reading; all in-flight runs finish regardless; the telemetry stays on as
   post-hoc evidence.** ⭐ **The Head's run-1 seed-0 override is recorded VINDICATED BY MEASUREMENT.** ⚠ **Accepted
   risk, stated knowingly by the Head: a genuinely destroyed store now burns its allocation to completion.**

⚠ **Nothing else in this report moves.** ⛔ **`PREREG-*.md` bodies are NOT edited anywhere in this program — a
revised pre-registration stops being one.** **Registry: `negative_results.md` N223's dated C2W6 addendum, N225,
N227, N228. Ledger: ⟲ C2W6 addendum. Matrix: v2.12 §0.10 / CM-32.**
