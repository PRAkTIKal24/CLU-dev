# c3-rival-ladder-prereg — experiment-engineer report

**Task + acceptance criterion:** freeze the C3 CLU store geometry on measured behaviour, set the ceiling
digit inside the verified admissible window, and file `PREREG-C3-LADDER.md` with pinned arm configs,
shrink knobs, a 15+-job plan inside 2×A100/4-day with a measured MFU, numeric predictions with falsifiers
and kill-conditions-first — **with zero ladder arms trained.**

**Status: done — and the geometry is HANDED BACK as a reported conflict**, which task §2 names as a
legitimate outcome. Everything else is delivered and is parameterised by the one decision the Advisor owes.
⛔ **Zero ladder arms trained. No bpc produced that any table could quote.**
Branch green: **1,813 passed / 0 failed**; **3 new files, 0 modifications to existing tracked code.**

> ⚠⚠ **RECONCILIATION LIST — NEEDS AN OWNER (protocol §5 corollary, in the first 10 lines).**
> 1. ⛔⛔ **ADVISOR DECISION OWED: which C3 geometry (G-A / G-B / G-C).** There is **no config-only
>    geometry** that fits the ruled ≈2 MB ceiling without descending below the w23 atom floor, and the
>    write-efficacy curve I measured over **4 seeds cannot settle whether that descent is safe** (the
>    pre-registered falsifier fires on 1 seed and reverses on another; the median is 5.4× and monotone).
>    ⭐ Recommendation: **G-B** (full atom floor, CLU cell in **3 of 12 layers**) — identical
>    bytes/compute/envelope to the descent, **no design rule broken, and no measurement needed to justify
>    it**. **Blocks every ladder job.** (§F1, §F3)
> 2. ⛔ **NEW ENGINEERING TASK CREATED by G-B:** a `store_layers` selection on `StreamModel`
>    (`chlu/core/blocks.py`) — explicitly outside this task's ownership. Small, but somebody must own it.
> 3. ⛔ **G-B forces the pending `ttt_normalized_write` ruling to `True`** (it keeps the pilot's per-layer
>    cell, so the divergent `(2197,52)` inner loop returns). The two decisions must be taken together.
> 4. ⛔ **`byte_ledger.py`'s `StateByteBudgetError` remedy text is WRONG** — it tells the operator to shrink
>    `capacity`/`atoms_per_item`, which move **zero bytes**. ⛔ I did not edit that file (it is
>    `c3-run3-budget-exemption`'s). Replacement text in §Proposed handover updates.
> 5. ✅ **CLOSED: `MATCHED_STATE_BYTE_BUDGET = 2,097,152` needs no edit** — confirmed inside the verified
>    window `[1,597,440 , 3,145,728)`. Closes reconciliation item 4 of `c3-csf3-harness`.
> 6. ⚠ **The ladder does NOT train the six pinned rivals** (none is implemented). Charter §2's "tuned
>    rivals" clause is undelivered by this ladder — a scope statement the Advisor should see.
> 7. ✅ **Full suite GREEN: 1,813 passed / 0 failed** (41:10) at HEAD `07b126b` / `main` `0644c48`, with
>    **1,813 = 1,781 + 32** confirming the count is exactly my new file. ⚠ Two earlier attempts **died
>    silently** under contention with the concurrent spoke's own suite — recorded in §Suite because the
>    failure mode (no traceback, no summary) is worth the Hub knowing about on this machine.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial:** **none — pre-registration + geometry freeze.** ⛔ I measured no ladder result. Every bpc
  anywhere in my artifacts is either (a) smoke-scale, from a script whose first 12 lines say it is never a
  claim venue, or (b) one of the four **banked pre-C3 CSF3 legs**, used only as a *prediction basis* and
  labelled non-quotable at every appearance (C3 Add.1 §2).
- **Laundering control:** ⭐ the sweep *found one the ladder needs*. `bpc(none) − bpc(clu_store)` is **not**
  a memory dividend — at smoke scale the store's read is numerically inert in the loss (bpc bit-identical
  to 6 d.p. across a 16× atom range) yet the gap to `none` is still +0.24…+0.35 bpc. The memory claim is
  `bpc(blank_store) − bpc(clu_store)`, pre-registered as **P1**, with the `echo` trivial-substitute arm.
- **Falsifies the task:** a frozen geometry with no admissible ceiling; a prereg without numeric
  falsifiers; a plan that does not fit 2×A100/4-day. **The first one effectively fired and is reported as
  the §2 conflict**; the ceiling is admissible for the recommended option; 11 predictions carry falsifiers;
  the plan fits with 3.23× headroom.
- **Does NOT falsify:** the C3 geometry differing substantially from the pilot's. **It must — by 4×.**

**Pre-registration:** two filed, both **before** the runs they govern.
`PREREG-GEOMETRY-SWEEP.md` (input prereg, filed before the sweep): **G1 held · G2 held · G3 FALSIFIED on
its filed wording but NOT DECISIVE at 4 seeds · G4 held (harder than predicted) · G5 half-held.**
`PREREG-C3-LADDER.md` (the deliverable).
⭐ **G3 fired against my own argument and changed my recommendation** — then two extra seeds showed the
effect is not robust, which changed the *grounds* for the recommendation but not the recommendation. That
is the prereg rule working: *a prediction that survives is evidence, one that fails is a finding, an
un-pre-registered agreement is neither.*

---

## 0. Deliverables

| artifact | what |
|---|---|
| `.claude/outputs/c3-rival-ladder-prereg/PREREG-C3-LADDER.md` | ⭐ **the deliverable** — kill conditions first, the geometry conflict + 3 costed options, ceiling digit + window arithmetic, pinned arms/rivals + shrink knobs + occupancies, match ratios, 15-job plan, 11 numeric predictions with falsifiers, literal launch lines |
| `.../PREREG-GEOMETRY-SWEEP.md` | the input prereg, filed before the sweep ran |
| `.../geometry-sweep.json` | 22 runs (pre-registered grid), 2 seeds, real enwik8, smoke shapes |
| `.../geometry-sweep-deployed-write.json` + `...-s23.json` | **24 runs / 4 seeds** — the **declared second arm** with the landed write levers; ⭐ the measurement that reopened the question |
| `.../deploy-geometry-ledger.json` | 6 candidate geometries at PILOT shapes: CLU/TTT bytes, match ratio, solved `(k,n)`, GRU hidden |
| `.../ttt-stability-at-frozen-geometry.json` | `η‖θ_K z‖²` at 4 geometries |
| `.../rival-table-verified.json` | the pinned rival table + `shrink_to_budget()` solutions, re-run on this branch |
| `scripts/c3_geometry_sweep.py`, `scripts/c3_ladder_plan.py`, `tests/test_c3_geometry_freeze.py` | tracked code (§Git) |

---

## F1. ⛔⛔ FIRST-ORDER FINDING: the store cannot be shrunk the way everyone assumed

### F1.1 `atoms_per_item` and `capacity` are BYTE-INERT at `addr_dim = 8`

`CluSystemConfig.n_atoms` is not `atoms_per_item × capacity`; it is a `max` including the **w23
dimension-aware floor** `round(min_atoms_base · min_atoms_c^addr_dim)`. At the ruled `addr_dim = 8` that
term is `512 · √2⁸ =` **8192**, and the pilot's `32 × 256` **ties it exactly**:

| knob moved | `n_atoms` | CLU total state B |
|---|---|---|
| pilot `K=32, A=256` | 8192 | 5,523,456 |
| `A → 64` | **8192** | **5,523,456** |
| `A → 16` | **8192** | **5,523,456** |
| `K → 128, A=16` | **8192** | 5,578,752 |

⇒ the pilot's `atoms_per_item = 256` is **decorative**. Pinned by 10 test cases.

### F1.2 Therefore no config-only sub-2 MiB geometry exists

`addr_dim` closed (Hub R2 / N312). `n_layers = 12` is what puts the model at **28.56 M params**, inside the
26–47 M class the venue is defined at. `dim` cannot help: fitting 8192 atoms under 2 MiB needs `dim ≤ 3`,
i.e. `payload_dim ≤ −5`. **The only remaining lever is `min_atoms_base` — the floor's own height.**
⇒ **the conflict is structural, not a config oversight.**

### F1.3 The three options, costed (full table in the prereg §2.2)

| | atoms/store-layer | store layers | CLU total B | occ. | breaks | s/step | TTT `η·n/d` | needs |
|---|---|---|---|---|---|---|---|---|
| pilot (landed) | 8192 | 12 | 5,523,456 | 2.634× ⛔ | the ceiling | 14.51 | 3.004 ⛔ | — |
| **G-A** | 2048 | 12 | 1,394,688 | 0.665× ✅ | ⛔ **the w23 floor ×4** | **4.18** | **0.751** ✅ | a config value |
| ⭐ **G-B** *(recommended)* | 8192 | **3 of 12** | 1,380,864 | 0.658× ✅ | **nothing** | **4.18** | 3.004 ⛔ | a `blocks.py` task + `ttt_normalized_write=True` |
| **G-C** | 4096 | 12 | 2,770,944 | 1.321× ⚠ | floor ×2 **and** the ceiling digit | 7.62 | 1.502 ⚠ | config + a re-ruled digit |

⭐ **G-A and G-B are byte-, compute- and envelope-IDENTICAL** (both 4.18 s/step, both 29.7 h worst job,
both 3.23× headroom at 20,000 steps) — the store's bytes *and* its compute are both **per store-bearing
layer**. G-A buys the reduction by **starving each cell**; G-B by **using fewer cells**. Pinned by
`test_G_B_is_byte_compute_and_envelope_EQUIVALENT_to_the_descent`.

⚠ **G-B's price, pinned by its own test:** it keeps the pilot's per-layer cell, so `solve_matched_ttt`
returns `(2197, 52)` and the divergent TTT inner loop returns with it ⇒ **`ttt_normalized_write=True`
becomes mandatory**, which is a claims-relevant flip of a published rival column.

## F2. The ceiling digit: **2 MiB = 2,097,152 B — CONFIRMED, no edit needed**

`rival_reference_table()` re-run on this branch reproduces the task's pinned table **to the byte**
(`rival-table-verified.json`).
`ceiling ≥ max(CLU, TTT_matched, ttt_linear) = 1,597,440`; `ceiling < gated_deltanet2 = 3,145,728`
⇒ **window `[1,597,440 , 3,145,728)`; 2 MiB is inside it** under G-A **or** G-B (both ≈0.66× occupancy).
Occupancies: `clu_store` 0.665/0.658× · `ttt_matched` 0.665/0.654× · `ttt_linear` 0.762× (all natural);
GDN-2 `n_heads 4→6` · TXL `mem_len 512→170` · Mamba-2 `d_state 128→39` · sliding `window 512→85` ·
TTT-MLP `head_dim 64→25`. No **natural** value equals the ceiling. ⚠ Disclosed in the prereg: GDN-2's
*shrunk* value lands on 2,097,152 exactly (`24·512²/6·2 B`, a powers-of-two coincidence).
⭐ **This closes `c3-csf3-harness`'s reconciliation item 4:** the constant as shipped is correct.
⚠ Only **G-C** would force a re-rule (into `[2,770,944 , 3,145,728)` — admissible, no rival grows, but the
Advisor's call, not mine).

## F3. ⭐⭐ The geometry sweep — and the arm that changed my mind

**Arm 1 — the pre-registered grid** (22 runs, 2 seeds, real enwik8, smoke shapes, `steps=60`, 1,766 s):

| # | filed prediction | outcome |
|---|---|---|
| G1 | `total(2048) = 1,394,688 B = 0.6650×` | ✅ exact, reproduced by the **built** cell |
| G2 | s/step ratio 8192:2048 ∈ [1.5, 4.0] | ✅ **2.51×**, monotone over all six points |
| **G3** | write efficacy at 2048 within 2× of 8192 | ⛔ **instrument dead in this arm**: `depth_ratio` `1e-28 … 1e-110`, **no ordering in `n_atoms`** — §7.27 store destruction at every geometry. Revived by arm 2 below. |
| **G4** | smoke bpc cannot rank geometries | ✅ **held far harder**: bpc **bit-identical to 6 d.p.** across a 16× atom range *and* all four iso-byte `capacity` splits, while cell state differs 30,208 → 460,288 B ⇒ **the store's read is inert in the loss** |
| G5 | `dim 16` costs 1.286× bytes | ✅ exactly **1.2864×** |

**Arm 2 — the DECLARED second arm** (12 runs). ⚠ Mid-flight I found the landed C3 runs set
**`atom_place_radius = 0.3` / `write_margin = 0.6`**, which the filed grid did not — and H1b is the very
mechanism my "the floor's mechanism is void" argument leans on. So I ran axis A **again** with those levers
(`--deployed-write`; `write_inner_steps` 40→8 for wall-clock, declared). ⛔ The filed arm was **not**
replaced; both are reported.

⭐ **The levers revive the instrument** (`depth_ratio` goes from `1e-28…1e-110` to **O(0.1–1)** and becomes
ordered in `n_atoms`). ⭐ **After the first two seeds gave a 13× spread I ran two MORE rather than reporting
n = 2** — all four below:

| `n_atoms` | occ. | `depth_ratio` seed 0 · 1 · 2 · 3 | **median** | `qstar_spread` median | s/step |
|---|---|---|---|---|---|
| 512 | 0.173× | 0.0816 · 0.0719 · 0.0330 · 0.8619 | 0.0767 | 0.3371 | 0.87 |
| 1024 | 0.337× | 0.0641 · 0.0735 · 0.0295 · 0.7784 | 0.0688 | 0.0756 | 0.94 |
| **2048** *(G-A)* | 0.665× | 0.1342 · 0.0644 · 0.0676 · 0.5087 | **0.1009** | 0.2683 | 0.94 |
| 3072 | 0.993× | 0.1915 · 0.0945 · 0.0991 · 0.6686 | 0.1453 | 0.4171 | 1.09 |
| 4096 *(G-C)* | 1.321× | 0.1961 · 0.0883 · 0.1316 · 0.6810 | 0.1639 | 0.3600 | 1.25 |
| **8192** *(w23 floor; G-B's cell)* | 2.634× | **2.4648** · 0.1199 · 0.0560 · 0.9795 | **0.5497** | 0.4180 | 2.34 |

**G3's verdict, both readings (the prereg was written for 2 seeds):**
⛔ on the filed wording ("*either* seed worse than 2×") it **FIRES** (seed 0: **18.37×**);
⚠ per seed it fires on **1 of 4** (`18.37 · 1.86 · 0.83 · 1.93`) and **seed 2 runs the other way**;
⭐ the 4-seed **median** curve is **monotone from 2048 up** (0.1009 · 0.1453 · 0.1639 · 0.5497 — **5.4×**
between 2048 and 8192), i.e. more atoms is *directionally* better for the write;
⚠ but **no geometry deepens wells on the median** (all < 1 — §7.27 erosion everywhere), and the **paired**
signal §7.26 requires (`qstar_spread`) does **not** corroborate (3072 is level with 8192; 512 beats 2048).

⇒ ⭐⭐ **HONEST VERDICT: directionally consistent with the w23 floor, NOT DECISIVE.** Not strong enough to
license a 4× descent below a design-ruled floor, and not strong enough to forbid one. ⭐ **Which is
precisely why I recommend G-B: when the measurement cannot settle the question, prefer the option that
does not need it settled.** G-A would be a bet on an unresolved instrument for zero byte, compute or
envelope gain.

⛔ **G4 at 4 seeds, unchanged and stronger:** `clu_store` bpc at 2048 vs 8192 per seed —
`6.4202/6.4205 · 6.5462/6.5471 · 6.5287/6.5287 · 6.4471/6.4472`. A **4× atom change moves the loss by
`≤0.001 bpc` on every seed.** The store's read is inert in the loss.

## F4. The job plan — measured MFU, and the reconciliation the scout's number needed

Every phase second read off the landed **2×A100** artifact `csf3_outs/run2/…_PARTIAL.json` (`host_rss`
marks). ⛔ Runs 1/2 are the pre-C3 over-budget geometry, used **only as a timing basis**.

| arm | s/step (measured, 2×A100) | **MFU** (`C≈6ND`, N=28,556,792, 8,192 tok/step, peak 6.24e14) |
|---|---|---|
| `clu_store` @ pilot geometry | **14.51** | **0.0155 %** |
| `none` (null arm) | **0.742** | **0.303 %** |
| `clu_store` @ G-A **or** G-B (projected) | **4.18** | **0.0538 %** |

⛔ **The scout's most pessimistic 3 % MFU is optimistic by ~200× for the CLU arm.** Two measured reasons:
the CLU block's cost is a multi-step integrator over an atom dictionary (not counted by `6ND`, so MFU is
the wrong denominator for it), and the plan pass is **host-side Python** — `plan_pass_frac` 0.166 on the
CLU arm and **0.88–0.95 on every other arm**, i.e. the cheap arms are not GPU-bound at all.
⭐ **We are TOKEN-bound, not compute-bound.** The scout priced `D = 5×10⁹` bytes (≈55 epochs); at the
measured throughput that is **700 h** for one CLU arm. The pilot ran 0.36 epochs.

| geometry | steps | tokens (epochs) | worst job | all fit 96 h? | makespan @%4 |
|---|---|---|---|---|---|
| pilot 8192×12 | 4,000 | 32.8 M (0.36) | 38.6 h | ✅ 2.49× | 1.61 d |
| pilot 8192×12 | 20,000 | 164 M (1.82) | **103.1 h** | ⛔ **NO** | 4.29 d |
| **G-A or G-B** | **20,000** | **164 M (1.82)** | **29.7 h** | ✅ **3.23×** | **1.70 d** |
| G-C 4096×12 | 20,000 | 164 M (1.82) | 54.2 h | ✅ 1.77× | 2.46 d |
| G-A or G-B | 40,000 | 328 M (3.64) | 53.0 h | ✅ 1.81× | 3.04 d |

⭐ **Either reduction is what makes a ≥1.8-epoch ladder possible at all** — at the landed geometry, 20,000
steps does not fit one job. ⚠ **Largest unmeasured item, flagged not buried:** the slice phase (priced by
call shape; measured `dyneval` is 59,673 s, so slices at full `eval_batches` would add ~16.6 h) ⇒
**`slice_batches = 10` pinned.**

## F5. The two-sided match and the TTT stability

| geometry | CLU total | TTT total | ratio | `η·n/d` | worst dir. | ‖W‖/16 writes |
|---|---|---|---|---|---|---|
| pilot **and G-B** | 5,523,456 / 1,380,864 | 5,483,712 / 1,370,928 | 1.0072× | **3.004** ⛔ | 6.182 | ×4.4e10 |
| **G-A** | 1,394,688 | 1,394,640 | **1.0000×** | **0.751** ✅ | **2.726** ⚠ | ×2.3e3 |
| G-C | 2,770,944 | 2,770,560 | 1.0001× | 1.502 ⚠ | 3.777 | ×3.3e6 |

⇒ the two-sided match survives under every option, and **improves** under G-A. ⭐ Under G-A the divergence
that NaN'd the rival column falls ~7 orders of magnitude **by geometry alone**; ⚠ the cure is **partial**
(worst direction still 2.726 > 2) and that caveat is asserted by a test. ⛔ Under G-B it returns unchanged.

---

## How I verified (commands + observed output)

| check | result |
|---|---|
| `tests/test_c3_geometry_freeze.py` | ✅ **32 passed** in 24.2 s |
| `ruff check` on all three new files | ✅ All checks passed |
| geometry sweep, pre-registered grid | ✅ 22 runs / 1,766 s → `geometry-sweep.json` |
| geometry sweep, declared deployed-write arm | ✅ **24 runs / 4 seeds** → `geometry-sweep-deployed-write{,-s23}.json` (§F3) |
| rival table re-run vs the task's pinned table | ✅ reproduces to the byte |
| deploy-scale ledger, 6 candidate geometries | ✅ `deploy-geometry-ledger.json` |
| TTT stability, 4 geometries | ✅ `ttt-stability-at-frozen-geometry.json` |
| job plan, 4 geometry/step combinations | ✅ §F4 table |
| full suite (`pytest -q`, all files) | ✅ **1,813 passed / 0 failed** in 2,470 s (41:10) — **on the third attempt**; the first two died silently under contention. **§Suite** below. |

⚠ **Two real defects were caught by the new tests before they shipped**: a duplicate `write_inner_steps`
kwarg that would have killed the sweep's deployed-write arm with a `TypeError`, and a wrong null-arm MFU
constant. Both fixed in the committed code.
⚠ **The zsh no-word-splitting trap (§7.37) fired on me live** while sweeping the plan options with
`set -- $cfg`; every subsequent invocation is one literal command line, and so is every launch line in the
prereg.

### §Suite — the full test suite: ✅ **GREEN on the third attempt** (the first two died — reported)

```
PYTHONPATH=/Users/user/Desktop/CHLU-wt2 .venv/bin/python -m pytest -q -p no:cacheprovider --no-cov
1813 passed, 36 warnings in 2470.00s (0:41:10)
```

✅ **1,813 passed / 0 failed**, against **HEAD `07b126b`**, `main` **`0644c48`**, working tree clean,
verified on both sides of the run (the "a suite run needs a stable HEAD" rule).

**Arithmetic checked, not assumed:**

| run | collected |
|---|---|
| `--collect-only` on my branch | **1,813** |
| `--collect-only` in a clean detached worktree at `main @ 0644c48` | **1,781** |
| ⇒ | **1,813 = 1,781 + 32**, i.e. **exactly my new test file and nothing else** |

⚠ **Two earlier attempts DIED silently and I am recording it rather than only reporting the green.** Both
ended mid-progress-bar (at 3 % and ~19 %) with **no summary line and no Python traceback**. Throughout the
session a **second `pytest` was running against the same shared venv from `/Users/user/Desktop/CHLU-wt1`**
— the concurrent `c3-run3-budget-exemption` spoke's own full-suite run (`ps` showed its command line
carrying that `PYTHONPATH`, at 264 % CPU / ~20 % RSS; mine sat at ~25 % RSS). Two concurrent JAX suites on
one laptop is the most plausible cause, and it matches a host-memory death with no Python-level error.
⛔ **I never killed the other agent's process**; I killed only my own, twice, when the contention was
costing more than the information. The third attempt ran once the other suite had space and took
**41:10 — the same wall clock `c3-csf3-harness` reported**, which is the corroboration that contention,
not my code, was the cause.

⚠ **A liveness mis-read I made and corrected, because it is the exact failure the memory warns about:** I
briefly read `ps` elapsed times (`05:22`) as hours rather than `MM:SS` and concluded the suite had stalled.
Checking `date` on both sides corrected it. A wrong liveness read is how a healthy run gets killed.

**Intermediate verification, kept because it is what I would have reported had the third attempt failed
too:** a targeted run over every test file in the subsystem —
`test_c3_geometry_freeze.py`, `test_c3_csf3_harness.py`, `test_ttt_stability_and_d5_wiring.py`,
`test_pilot_checkpoint_resume.py`, `test_placement_probe.py` — gave **124 passed** in 667 s.

---

## Git footprint

**Branch `agent/experiment-engineer/c3-rival-ladder-prereg`**, in worktree `/Users/user/Desktop/CHLU-wt2`
(**wt2**, as the task directs; `wt1` is `c3-run3-budget-exemption`'s and was already live when I started).
⚠ **Base note:** the task's preferred base (`c3-run3-budget-exemption`) had **not landed** at spawn — it
existed only as a bare branch pointer at `f98f939` — so per the task's fallback I branched off
`agent/experiment-engineer/c3-csf3-harness @ f98f939` and treated `chlu/eval/byte_ledger.py` as
**READ-ONLY** (I read it; I did not modify one byte of it, and §F1.4's correction is handed to its owner
rather than applied). **Mid-task the Hub merged the harness into `main` (`0644c48`); I rebased onto local
`main` — clean, no conflicts.** ⚠ Per protocol §3.5 I did **not** rebase onto the stale `origin/main`.
Not pushed, no PR.

| commit | files | note |
|---|---|---|
| `a6ae6c8` | **new** `scripts/c3_geometry_sweep.py` (+330) | the smoke sweep + the declared deployed-write arm |
| `ccf7a23` | **new** `scripts/c3_ladder_plan.py` (+259) | the job plan, costed off measured A100 walls |
| `08b15cf` | **new** `tests/test_c3_geometry_freeze.py` (+287) | 30 cases |
| `07b126b` | all three (+71 net) | `n_store_layers` pricing + 2 tests, after the write curve reopened the question |

```
$ git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/c3-rival-ladder-prereg
07b126b  08b15cf  ccf7a23  a6ae6c8      (4 commits)
$ git -C /Users/user/Desktop/CHLU diff --stat main..agent/experiment-engineer/c3-rival-ladder-prereg
 scripts/c3_geometry_sweep.py     | 330 +++++
 scripts/c3_ladder_plan.py        | 291 +++++
 tests/test_c3_geometry_freeze.py | 326 +++++
 3 files changed, 947 insertions(+)
```

⭐ **Three NEW files, zero modifications to any existing tracked file.** ⛔ `chlu/eval/byte_ledger.py`,
`chlu/core/blocks.py`, `chlu/core/clu_system.py`, `chlu/config.py`, `chlu/training/train_cluformer.py`,
`chlu/experiments/exp_cluformer_pilot.py`, the CLI and `scripts/csf3/*` are **all untouched** — so **no
banked journal's resume fingerprint moves because of this branch**, and there is no overlap with the
concurrent `c3-run3-budget-exemption` spoke.

**Worktree-ref verification (protocol §3.2, the lost-8-commits precedent):** all 4 commits confirmed
visible on the shared ref **from the main repo** (output above).

⚠ **Post-hoc note, checked after the fact and worth recording:** `../CHLU-wt2` has since been **reassigned
by the Hub to the `c3-rival-mamba2` spoke** (and `wt1` to `c3-gb-landing`, `wt3` to `c3-rival-gdn2`).
⭐ **Re-verified after the handover: `agent/experiment-engineer/c3-rival-ladder-prereg` is still at
`07b126b` with all 4 commits and all 3 files present on the ref** (`git ls-tree` on the branch, from the
main repo). Nothing was lost — which is exactly the check §3.2 exists for, since the directory no longer
holds my branch. ⛔ **Review the branch, not the directory.**

Scratch: `.claude/scratch/c3-rival-ladder-prereg/` (probe scripts + run logs). Nothing left in the repo.

---

## Open questions / follow-ups / risks

1. ⛔⛔ **The geometry decision (§F1.3) is the whole gate.** Recommendation G-B, which creates a
   `chlu/core/blocks.py` work item and forces the `ttt_normalized_write` ruling.
2. ⛔ **K1 is pointed at us.** The pre-registered expectation is **no memory dividend**
   (P1: `blank − clu = 0.000 ± 0.005` bpc, from four banked legs measuring
   `+0.001158, +0.001276, −0.000263, −0.000139`, mean **+0.00051**, sign flipping between runs). If it
   survives 3 seeds the ladder's honest output is a negative result, and **K1 forbids answering it by
   adding rivals**.
3. ⚠ **The six pinned rivals are not implemented** — charter §2's "tuned rivals" clause is undelivered.
   Owner + budget needed *if* K1's falsifier fires.
4. ⚠ **§7.27 store destruction now blocks an instrument**, not just a watch item — it is why arm 1's G3
   was unmeasurable. ⭐ **But arm 2 shows the deployed write levers revive it**, which is a cheap,
   reusable fix for any future geometry question: measure with `atom_place_radius > 0`.
5. ⚠ **The decisive curve is still under-powered at n = 4 seeds and I ran the extra two rather than
   recommending them.** Result: the effect is real in the median (5.4×) but fires the pre-registered
   falsifier on only 1 of 4 seeds and *reverses* on another. ⇒ **a smoke-scale write-efficacy instrument
   with this variance cannot decide a geometry question.** If the Advisor wants the question settled
   empirically rather than by the G-B "breaks nothing" argument, it needs ≥8 seeds or a cluster-scale
   probe — both of which cost more than simply choosing G-B.
6. 🔍 **Not done, out of scope:** no rival implementations; no `blocks.py` change; no edit to
   `chlu/eval/byte_ledger.py`; no PG-19/FineWeb loaders; dyn-eval **pre-registered** (P5 = 0.015 ± 0.008
   bpc) but not run — running it is a ladder job.

---

## Proposed handover updates (for the Hub)

**§7 — new entries**

- **7.38 [NEW, FIRST-ORDER, BLOCKS THE C3 LADDER] `capacity` and `atoms_per_item` are BYTE-INERT at
  `addr_dim = 8`.** `n_atoms = max(A·K, min_atoms, round(min_atoms_base·√2^d_addr))`, and at `d_addr=8`
  the last term is **8192**, which the pilot's `32×256` ties exactly. Every knob the harness names as the
  way to shrink the store moves **zero bytes**. With `addr_dim` closed (R2/N312), `n_layers` closed by the
  26–47 M weight class and `dim` arithmetically unable to help, **`min_atoms_base` is the only config
  lever** — and lowering it descends below a design-ruled, empirically-anchored floor. **Advisor decision
  owed between three costed options; the recommended one is not a config change at all.**
- **7.39 [NEW, ⭐ the recommendation] The store need not live in every layer.** `n_atoms = 8192` (floor
  intact) in **3 of 12 layers** gives **1,380,864 B = 0.658× of 2 MiB** and **4.18 s/step** — byte-,
  compute- and envelope-**identical** to descending to `n_atoms = 2048` in 12 layers, while breaking **no**
  design rule. ⛔ Costs a `chlu/core/blocks.py` `store_layers` selection (unbuilt) **and** forces
  `ttt_normalized_write = True`, because it keeps the pilot's per-layer cell and hence the divergent
  `(2197,52)` TTT solve.
- **7.40 [NEW, corrective] `byte_ledger.StateByteBudgetError`'s remedy text is wrong.** Suggested
  replacement: *"Shrink the store — ⚠ at `addr_dim=8` the w23 atom floor (`min_atoms_base·√2^d`) dominates,
  so `capacity`/`atoms_per_item` are BYTE-INERT. The levers are: `min_atoms_base` (a claims-relevant
  descent below a design-ruled floor — prereg it), the number of store-bearing layers, or `dim`/`n_layers`
  (both move the reach ceiling or the weight class)."* Owner: `c3-run3-budget-exemption`.
- **7.41 [NEW, laundering] `bpc(none) − bpc(clu_store)` is NOT a memory dividend.** Smoke: the CLU arm's
  bpc is **bit-identical to 6 d.p. across a 16× atom range and four `capacity` splits** while cell state
  differs 15× — the store's read is numerically inert in the loss — yet the gap to `none` is still
  +0.24…+0.35 bpc. Corroborated at pilot scale: `blank_store` differs from `clu_store` by `≤0.0013 bpc` on
  4/4 banked legs **and the sign flips between run 1 and run 2**. **The memory claim is
  `blank_store − clu_store`; every C3 table must print it.**
- **7.42 [NEW, measured] The CLU arm runs at 0.0155 % MFU on 2×A100, not 3 %.** 14.51 s/step vs the null
  arm's 0.742 (`C≈6ND`, N=28.56 M). `plan_pass_frac` 0.166 on the CLU arm, **0.88–0.95 on every other arm**
  — the cheap arms are host-bound. ⇒ the scout's `D = 5×10⁹`-byte (≈55-epoch) costing is **700 h** for one
  CLU arm: **we are token-bound, not compute-bound**, and ⛔ **no C3 enwik8 number can be placed beside the
  published 1.00–1.06 bpc grid, which is obtained at ~50 epochs. Same category error as the dyn-eval 0.94
  and it needs the same never-quote row in `claims_matrix.md`.**
- **7.43 [NEW, measured, geometry, ⚠ NOT DECISIVE] The w23 atom floor is DIRECTIONALLY supported at
  `d_addr = 8` on real text, and no more than that.** With the landed write levers
  (`atom_place_radius 0.3`, `write_margin 0.6`), the 4-seed **median** trained/untrained well-depth ratio
  rises monotonically with atoms — **0.1009 (2048) → 0.1453 → 0.1639 → 0.5497 (8192)**, a **5.4×** span —
  but the pre-registered 2× falsifier fires on **1 seed of 4** and **reverses on another (0.83×)**, no
  geometry deepens wells on the median, and the paired `qstar_spread` signal does not corroborate.
  ⇒ **a smoke-scale write-efficacy probe with this variance cannot decide a geometry question**; a
  geometry choice must rest on what it breaks, not on this curve.
- **7.44 [escalation of 7.27, with a fix] Store destruction blocks the write-efficacy instrument — unless
  the deployed write levers are on.** At `atom_place_radius = 0` the depth ratio is `1e-28…1e-110` with no
  ordering in any geometry; at `0.3` it is O(0.1) and ordered. **Any future geometry probe must set the
  deployed levers or it measures nothing.**
- **7.45 [confirmed again, ops] The zsh no-word-splitting trap fired during this task** (`set -- $cfg`
  over a config list). One literal command line per invocation, including every launch line in the prereg.
- **7.46 [NEW, ops] Two concurrent JAX full-suite runs on this laptop kill one of them SILENTLY.** Two of
  my `pytest -q` runs died mid-progress-bar with **no summary line and no traceback** while the concurrent
  `wt1` spoke's suite was live (264 % CPU / ~20 % RSS each). The third, run with space, was green in
  **41:10** — the same wall clock `c3-csf3-harness` reported. ⇒ **the suite is not safely parallel across
  worktrees on this machine**; a spoke that needs a green suite should check `ps` for another
  `.venv/bin/python -m pytest` first, and a silent death must not be read as a failing test.

**§3 (CLI & config)** — ⛔ **no new `PilotConfig`/`StreamMemoryConfig` field was added**; nothing in this
branch moves any banked journal's fingerprint. ⚠ The *ladder's* own config would be non-default in
`atoms_per_item`/`min_atoms_base` (G-A) or would need a new `store_layers` field (G-B), so **no banked
run-1/2/3 leg can be resumed into the ladder** — intended; the prereg gives it a fresh `OUT_BASE`.

**§10 (C3W1 block):** the ladder prereg is filed and **gated on one Advisor decision**. The ceiling digit
is confirmed at 2 MiB with **no code change**. The measured MFU (7.42) and the token-bound finding change
what the C3 paper can claim against the published enwik8 grid and should reach `claims_matrix.md` this wave.
