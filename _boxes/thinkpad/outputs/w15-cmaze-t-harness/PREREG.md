# PREREG — w15/cmaze-t-harness (CMAZE-T, the transition instrument)

**Written 2026-09-22, BEFORE any number in `VAL-SMOKE.json` or `HARNESS-READY.json`.**
Scored runs budgeted and spent: **ZERO**. This spoke is instrument construction; the val
smoke is free under COMMON.md. No row of `/c/chlu/SCORED-LEDGER.md` is claimed, because
no scored run is spent.

---

## 0. What is being built

CMAZE-T: given `(instance, s_t = (cell, keystate), a_t)`, an arm emits the next state.
The engine judges the transition **exactly** — legality under RULES A+B, displacement,
and pocket — by integer comparison against `w9_harness.dynamics.step`, which is itself
already audited route-for-route against `w8_envscout.maze_colour.judge`.

Engine, solver and exact judge are **imported unmodified**. The one distributional
change the request layer makes is declared in `LOG.md` D7 (tinted door cells) and is a
change to the `tint` array of an `Instance`, of exactly the kind `w9_maze.task` already
makes; no engine code is edited.

## 1. The emission and the judgement (fixed here, not after the fact)

An arm emits `(legal, displacement, key_event, pocket_next)`. The harness applies
`(legal, displacement, key_event)` to the real `s_t` to build the **generated**
`s_{t+1} = (pos', ks')`. The transition is **correct** iff

1. the key event is coherent against `s_t` (no picking up a key that is not untaken or
   into a full pocket; no consuming a key that is not carried), **and**
2. `legal` equals the engine's legality, **and**
3. if legal: `pos'` and `ks'` equal the engine's `step()` output exactly, **and**
4. `pocket_next` equals the pocket of the true next state.

All comparisons are between small integers or tuples of them. **Tolerance used: none.**

A **rollout of length L** chains *generated* states: the state fed to step `t+1` is the
arm's own `ŝ_t`, never the engine's. A rollout is **valid** iff every step is correct
and the goal is entered within the instance's budget.

## 2. Rows, fixed here

`seen` · `unseen 2-way` (`pair`) · `unseen n-way` (`nway`) · `rollout validity` ·
`value hold-out` (never-seen colour; **scored, no refusal display** — refusal displays
are retired).

## 3. Hold-outs, fixed here, before a row is looked at

* **VALUE hold-out:** colour `k` (= 4). A transition is in the value row iff colour 4
  appears in ANY rule-bearing role — pocket, key-on-cell, closed-door colour, or tint.
  It is therefore a colour **never seen in a rule** in train.
* **2-way hold-out:** 8 `(word_i = a, word_j = b)` combinations drawn from the two
  rule-bearing pairs — pocket × door (RULE A) and pocket × tint (RULE B).
* **n-way hold-out:** 20 `(w_agent, w_door, w_tint, w_dir)` 4-tuples, drawn **evenly
  from five named families** of the word grid so the n-way row is not accidentally a row
  of cells no rule decides: `interaction` (closed door c, pocket c, tint c — LEGAL),
  `rule_b` (no door, pocket c, tint c — ILLEGAL), `rule_a` (door b, pocket c≠b —
  ILLEGAL), `open_door_tint` (OPEN door, pocket c, tint c — ILLEGAL), `free`
  (nothing bites — LEGAL).
* **Mazes:** train mazes and evaluation mazes come from **disjoint generator seed
  blocks**. Every evaluation maze is a random maze never seen at train, so a within-maze
  lookup cannot transfer and **cross-maze lookup is the floor**.

The hold-out recipe depends only on the WORD GRID and a seed. No row, no outcome and no
score enters it.

## 4. THE BARS — declared here

### 4.1 Acceptance bars (these gate `HARNESS-READY.json` `"status": "ready"`)

| # | bar | threshold |
|---|---|---|
| A1 | independent replay disagreements | **exactly 0**, over **≥ 50,000** transitions replayed, by **two** independent drivers |
| A2 | rows shown to a non-refuser arm for which no accepting emission exists | **exactly 0**, over **≥ 50,000** rows checked |
| A3 | train rows mentioning the held-out colour | **exactly 0** |
| A3b | held-out n-way / 2-way combinations present in train | **exactly 0** |
| A3c | the five words determine `(legal, disp, event, pocket)` | **0** word tuples with two outcomes |
| A4 | rule-B-bites fraction | **REPORTED COLUMN — no threshold, never an abort condition.** Reported on both the stratified bank and the unstratified natural distribution, each named. |
| A5 | VAL smoke | 3 seeds, **all** rows above, **every** rival, written to `VAL-SMOKE.json` + a table in `FINDINGS.md` |
| A6 | this file | exists before any number |

If A1, A2, A3, A3b, A3c and A5 all hold, `"status": "ready"`. A4 can take any value.

### 4.2 Config-A selection bar (val only, free)

Config-A (the CLU's `latent D`, `K`, `n_atoms`, `ridge`) is chosen by **val `exact` on
the `seen` row only**. ⛔ The held-out rows (`pair`, `nway`, `value`) and the rollout row
are **not** consulted in selection. Adoption bar:

* `seen exact ≥ 0.90`, **and**
* `landscape_removed` on `seen` `≤ 0.60`.

If no config clears the bar, the failure is DECLARED and the best config on `seen` is
carried forward with the bar stated as unmet. A config failing its val bar is declared
and the next config starts.

### 4.3 Charter-C bar for a STORE claim

All four controls — `landscape_removed`, `landscape_shuffled`,
`landscape_shuffled_across_wells`, `random_atoms` — must **collapse**, defined as
`control exact ≤ CLU exact − 0.15` on the `seen` row. If they do not all collapse, the
result is reported as an **interface** claim or as **nothing**, and which one is said
plainly.

### 4.4 The falsifier (the two-gym law), declared here

The instrument warning `composition_blind_scores_near_the_clu` fires iff

    max(BLIND-COLOUR, BLIND-SEQUENTIAL) >= CLU_exact - 0.05   on the `unseen n-way` row

⛔ This is a **reported finding, not a reason to abandon**. If it fires, the harness
ships with the warning written into `HARNESS-READY.json` under `"instrument_warnings"`.

Note that BLIND-COLOUR and BLIND-SEQUENTIAL here are **rule-knowing relaxed planners**,
not learned policies: each is optimal inside its own relaxed legality relation and is
then graded by the unrelaxed exact judge. They are therefore an **upper bound** on what
a composition-blind *policy* could achieve, which makes the falsifier conservative.

### 4.5 Deletion bars

* **Store-scoped exactness:** after deleting the rows, the store's per-well sufficient
  statistics must equal, to float32, those of a store **fitted without** those rows.
  Bar: `max |Δ| / scale ≤ 1e-5`. Reported as measured, not asserted.
* **Adaptation cost:** printed for both sides — CLU (seconds, passes over retained data,
  items kept) and rival **retrain-without** (seconds, gradient steps, examples seen).
  The rival **as-trained** row is printed beside them.
* **Deletion language:** store-scoped exactness; the codec and the state encoder are
  declared **OUTSIDE** the deletable object; post-deletion adaptation is light and
  data-free. ⛔ Never "no changes".

## 5. Matching discipline

* **Naive vs naive.** The MLP / LSTM / transformer rivals are matched on the **CLU's
  read-path float count** (store atoms + address book), by a width search that reports
  the count it **achieved** and the residual.
* The **memory ledger** (the float64 sufficient statistic that buys exact deletion) is
  printed separately and is **never** matched. ⛔ The inflated-rival row does not exist.
* The **lookup table is a printed floor, never a rival.**
* Every byte figure travels with its convention named; seed count is a column on every
  table.

## 6. Seeds

**3 seeds, all or none.** Seeds 0, 1, 2, each with its own generator seed block, its own
frozen codec, its own frozen state encoder and its own arms. ⛔ Never "9/9 seeds".

## 7. What would prove the idea wrong

1. A composition-blind planner scoring near the CLU on `unseen n-way` (§4.4) — the
   instrument would not be measuring composition.
2. The cross-maze lookup floor scoring near the CLU on `unseen n-way` — the row would be
   retrieval, not composition.
3. Charter-C controls failing to collapse (§4.3) — no store claim is available.
4. The five words failing to determine the transition (A3c) — hold-outs over word
   combinations would not be statements about the transition.
5. Rule B never rendering an action illegal (A4 = 0) — the first clause of the two-gym
   law would be unsatisfied. **Reported, not aborted on.**

---

# AMENDMENT 1 — 2026-09-23, still BEFORE any smoke number

Three changes, all prompted by a **12-maze / 8-val-maze DRY RUN** whose numbers are
recorded in `LOG.md` and in `logs/dry*.log` and which is **not** part of the VAL smoke.
No `val_seed*.json`, no `VAL-SMOKE.json` and no `HARNESS-READY.json` existed when this
amendment was written.

### A1.1 — the `landscape_shuffled` control was a no-op by construction
As first written, `landscape_shuffled` permuted the atom **rows** within each well. The
address book is fitted **after** the control is applied and `head_read` computes
`(Φ W_k) G_k`, so a row permutation of `G_k` is absorbed **exactly** by the matching
column permutation of `W_k`. Measured on the dry run: the row-permuted control scored
**0.8195** on `unseen n-way` against the CLU's **0.8195** — identical to every digit.
It is redefined as a permutation of each well's atom **channels** (columns), which moves
the deployed span and cannot be absorbed. Recorded rather than quietly fixed.

### A1.2 — the Config-A selection bar now requires ALL FOUR charter-C controls to collapse
§4.2's bar was `seen ≥ 0.90` and `landscape_removed ≤ 0.60`. On the dry run the config
that bar selected (`latent 64, K 8, n_atoms 6` → span fraction 6/8 = **0.75**) had
`random_atoms` at 0.6992 and `landscape_shuffled_across_wells` at 0.6090 against a CLU of
0.8195 on `unseen n-way` — i.e. **no store claim was available at the selected config**,
because a store at 0.75 span nearly fills its wells and random atoms span them too.

The bar becomes:

> `seen exact ≥ 0.90` **and** `max(all four charter-C controls) ≤ CLU − 0.15` on the
> `seen` row. Ranking: clears-the-bar first, then `seen exact`, then fewest read floats.

This is val-side model selection on val-only quantities; the held-out rows and the
rollout row are still **not** consulted. The config grid is widened to span the
saturation ratio explicitly. If no config clears the bar, that is DECLARED and the best
available config is carried forward with the bar stated as unmet.

### A1.3 — two measurement corrections
* **Rollout length.** The rollout row is the **full** optimal route (mean ≈ 45 steps), not
  a fixed-`L` prefix: an 8-step prefix of a 45-step route cannot reach the goal, so
  `validity` would be 0 for every arm by construction and would measure the truncation.
  An `L = 8` block is kept as a **declared diagnostic** whose readable number is
  `all_steps_exact_rate`.
* **The within-maze lookup column.** Its table is built from the val mazes'
  **natural-distribution** rows, a different sample of the same mazes, so the scored row
  is not in the table by construction. As first written it was built from the scored rows
  themselves and returned 1.0000 trivially. It remains **declared NOT transfer** and is
  printed only as the ceiling a lookup reaches when the maze repeats.

### A1.4 — rival training budget
Rival gradient steps raised from 2400 to **8000** (same budget for MLP, LSTM and
transformer, and for both deletion rivals). On the dry run at 200 steps the MLP and LSTM
were still emitting the majority class; a rival that has not converged is not a rival.
The loss curve is printed for every arm so the claim is checkable.

---

# AMENDMENT 2 — 2026-09-23, AFTER the first VAL smoke, with BOTH runs reported

⚠ Unlike Amendment 1 this one is written **after** numbers existed. Everything it
changes and everything it was prompted by is stated here, and the **pre-amendment run is
preserved in full** at `preamendment2/` (`val_seed{0,1,2}.json`,
`VAL-SMOKE-PREAMENDMENT2.json`, `HARNESS-READY-PREAMENDMENT2.json`, `seed*.log`). Both
sets of numbers are reported in `FINDINGS.md`. ⛔ Nothing is replaced silently.

### What the first smoke measured about its own instrument
The frozen codec's **own** encode→decode→judge ceiling, per row, 3 seeds:

| row | codec ceiling (mean ± sd) |
|---|---|
| seen | 1.0000 ± 0.0000 |
| unseen 2-way | 0.8722 ± 0.0750 |
| unseen n-way | 0.9495 ± 0.0409 |
| **value hold-out** | **0.4963 ± 0.0368** |

The train rows realise only **23–31** of the **45** distinct next-state codes the val
rows need, because a code carrying the held-out colour in its `key_event` or
`pocket_next` block never occurs in train. So on the value row **every arm that goes
through the codec was capped at ≈ 0.50, and the row was measuring the codec, not the
arm.** The CLU scored 0.4257 ± 0.0162 against that 0.4963 ceiling.

### The change
The frozen codec is fitted on the TRAIN codes **plus the enumerated emission alphabet**:
the full Cartesian product of the four emission blocks, 540 codes at k = 4
(`exp_w15_cmaze_t.enumerate_code_alphabet`).

**Why this leaks nothing.** The alphabet is a statement about the emission FORMAT. It
contains no row, no maze, no request word, no rule and no outcome — only that the
format's symbols exist, which the format already says. The **store** is still built from
the TRAIN codes only, the head is still fitted on the TRAIN rows only, and the value
hold-out colour still never appears in a train rule.

Nothing else changes: same banks, same hold-outs, same config grid, same bars, same
seeds, same rivals, same floors, same controls.

### Also recorded from the first smoke, and NOT changed
* **Config-A is selected per seed and came out different on each** (`64/8/3`, `64/4/3`,
  `128/8/3`), which is a large part of the seed-to-seed spread on the held-out rows
  (`unseen n-way` CLU 0.8019 / 0.3674 / 0.4441). ⭐ The recommendation to wave 16 is to
  **pin one Config-A across seeds**; it is left unpinned here because pinning it would
  require choosing from these seeds' numbers, and this spoke reports rather than tunes.
* **Charter-C per row.** On `seen` and `unseen n-way` all four controls collapse, so a
  STORE claim is available there. On `unseen 2-way` and `value hold-out` the CLU itself
  is low enough that the −0.15 margin is not met, so those rows are an **INTERFACE**
  claim or nothing. Stated per row in `FINDINGS.md`, never averaged away.

### AMENDMENT 3 — 2026-09-23, a measurement bug in the deletion row, fixed before the rerun

The first smoke's **store-scoped exactness** check compared the downdated store against
a reference built with `TransitionClu(Ztr[keep], ...)`, which **recomputes the ward
partition on the surviving rows**. The two objects therefore had different CHANNELS in
each well and the check reported `max_relative_delta = 1.26` while the per-well counts
matched exactly (11620 = 11620) — the counts matching while the statistics could not is
what gave the bug away.

The reference is now built at the **same well partition** (`well_channels=` override).
Re-measured on a 20-maze probe (`logs/probe_delete.log`):

    S_k   max relative delta  5.226e-16
    deployed atoms            0.000e+00  (exact)
    n after, downdated / fitted-without   2745 / 2745

The bar in §4.5 (`≤ 1e-5`) is unchanged. The pre-amendment deletion numbers are in
`preamendment2/`; the check FAILED there for the reason above and that is stated in
`FINDINGS.md` rather than dropped.
