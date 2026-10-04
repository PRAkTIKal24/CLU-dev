# HUB CHARTER — `hub/generation-table`

- **Agent:** `autonomous-hub` · **Branch:** `hub/generation-table` · **Worktree:** `../CHLU-hub-generation-table`
- **Base:** ⭐ **the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`** — ⛔ never a moving `main`, ⛔ never a peer's branch
- **Boot:** `.claude/autonomous-hub-charter.md` → `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7 → this file. ⛔ Nothing else.
- **Report:** `.claude/outputs/hub-generation-table.md` · ⭐ **running log: `.claude/outputs/hub-generation-table/LOG.md`, updated as you go, ⛔ never written at the end**
- **Limits:** ≤ 10 h wall clock · ⛔ **≤ 3 scored runs, all reported** · ⭐ CPU by default · ⛔ **GPU only for a headline**

---

## ⭐⭐ YOUR TARGET
**Show that the store is NOT DECORATIVE on the generation table — and then beat the known numbers and the rivals.**
⭐ **Metric:** composition-B aggregate, external classifier, the shipped cell, 3 seeds, **matched declared bytes**.
⭐ **Ceiling to win or tie:** `knn_proto` **0.8650 ± 0.0122**. `m = 0.020`; **WIN needs `Δ > m`**; `0 < Δ ≤ m` with
3/3 same sign is **TIE (favourable)**; ⛔ **a tie counts ONLY against the ceiling** — every like-for-like row must be
**beaten outright**.

### ⛔⛔ RULED (Head, 2026-09-16): YOU START FROM "IS THE STORE EARNING ITS BYTES", NOT FROM "BEAT 0.5949"
⛔ **At matched bytes the store is currently DECORATIVE and that is the measured starting position.** ⭐ Clearing
that is a **precondition, not an achievement** — and ⛔ **it is not sufficient: you must also beat the rivals and
the published numbers below.**

---

## ⭐ THE TABLE AS IT STANDS (frozen harness, repaired split, 3 seeds, external classifier)
| row | composition-B | distinct tuples / 72 | deployed bytes |
|---|---|---|---|
| `store_removed` (**the null**) | 0.1204 ± 0.0131 | 1.00 | 524,288 |
| `ebm_compose` | 0.1937 ± 0.0182 | 72.00 | 14,226,660 |
| ⛔ **`dynread` (the dynamical read — IT LOST)** | 0.2083 ± 0.0514 (A) / 0.2114 ± 0.1033 (B) | 13.3 / 16.0 | 1,048,576 / 8,388,608 |
| ⚠ **constant-output floor** (`x0/A_row_lookup`) | **0.3272 ± 0.0633** | ⛔ **2.67 ± 0.47** | — |
| `lace` | **0.3117 ± 0.0153** | 71.67 | not byte-matched |
| ⭐ **atom-mean lookup, matched bytes — THE §C FLOOR** | **0.3927 ± 0.0986** | — | 524,288 |
| `x0` addressed read, **matched bytes** | **0.5008 ± 0.0498** | — | 524,288 |
| ⭐ `clu` codebook read | **0.5949 ± 0.0082** | 9.00 ± 0.82 | 524,288 |
| `x0` addressed read, **8× bytes** | 0.6852 ± 0.0173 | — | ⛔ **4,194,304** |
| ⭐⭐ `knn_proto` — **THE CEILING** | **0.8650 ± 0.0122** | **56.00 ± 3.74** | 475,136 |

### ⛔⛔ THREE THINGS ABOUT THIS TABLE THAT WILL MISLEAD YOU IF YOU DO NOT READ THEM
1. ⛔⛔ **ROUGHLY THE BOTTOM THIRD IS FREE.** A **near-constant output — 2.67 distinct tuples of 72 — scores
   0.3272 and TIES `lace`.** composition-B aggregate is an unweighted mean over six per-factor accuracies and has
   a high free floor. ⇒ ⭐ **The honest dynamic range is ~0.33 → 0.865, not 0 → 0.865.** ⛔ A mid-0.4 is **barely
   above free**, ⛔ not halfway home. ⭐ **Report the distinct-tuple count beside EVERY aggregate you quote.**
2. ⛔ **`x0`'s 0.6852 is at EIGHT TIMES the bytes.** At matched bytes it scores **0.5008 and LOSES to `clu`**, and
   ⛔ **fails the §C floor against the atom-mean lookup** (2/3 seeds, intervals overlap). ⛔ **Never quote 0.6852
   without its byte multiple on the same line.**
3. ⚠ **`knn_proto` renders 56.00 ± 3.74 of 72, ⛔ NOT 72/72** — an earlier docstring said 72/72 and was never measured.
⚠ `lace` **0.3117** is the CELL harness row. ⛔ **0.3241 is the BASELINES instrument at 688,128 B and is not
comparable here.** Both are real; they are different instruments.

## ⭐ §E.4 — THE MECHANISM FACTS YOU INHERIT (measured, ⛔ do not re-derive)
- ⭐⭐ **The launch beats the settled state by a mile: 0.6852 → 0.2114 after 2000 steps.** The store, address book,
  request path and decoder all work. ⛔ **THE DYNAMICS ARE THE DEFECT** — descending the energy *loses* accuracy,
  so the landscape's minima are not at task-correct states. ⇒ ⛔ **"more steps" makes it worse.**
- ⭐ **Steering is real and attributable:** request-off (β = 0, same chain) collapses to **1 distinct tuple of 72**
  and to 0.0000 at rung B. The request energy carries 100 % of the distinctness — it is just small.
- ⭐ **Placement is NOT the lever:** learned placement nulled against a D-matched random floor (0.0019 vs a 0.10
  bar, `fraction_moved` 0.75 — a real null). ⭐ **The INHERITED ward/correlation grouping already beats that floor
  by 0.0975 — 50×.** The wells are in the right place.
- ⭐⭐ **The codebook is the largest identified loss in the programme: it costs 90.5 % of `scale` and 100 % of
  `orientation`**, while the partition costs 9.5 % and 0 %.

## ⛔ Falsifiers — answer all three in these words
1. **Beats `store_removed`, `ebm_compose`, `lace`, `clu` AND the matched-bytes atom-mean lookup outright**, beyond
   seed spread — and **WIN or TIE against `knn_proto`** by the rule above.
2. ⭐ **Clears the constant-output floor**: distinct factor tuples **well above 2.67/72**, reported beside the aggregate.
3. ⭐ **The store is load-bearing**: `store_removed` collapses your arm. ⛔ If deleting the store barely moves your
   number, you have built a decoration and ⛔ it does not count.

## ⭐ What would make this hub's answer worthless
- ⛔⛔ **Shrinking the store's role until something wins.** That is the degenerate solution charter §C exists to
  forbid. **Every arm must beat a matched-bytes lookup table.**
- ⛔ **A mid-0.4 reported as progress.** It is ~0.07 above a constant output.
- ⛔ **An aggregate quoted without its distinct-tuple count.**
- ⛔ **Spending the ten hours tuning the settling.** It is measured as harmful; the lever is elsewhere.
