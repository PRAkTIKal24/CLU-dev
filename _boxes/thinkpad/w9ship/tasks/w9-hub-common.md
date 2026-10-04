# WAVE 9 — THE MAZE. COMMON RULES.
Boot: `autonomous-hub-charter.md` → this file → your target file. ⛔ Nothing else.
**Base `base/iter8 = fa60b15`** (pin + 92 additive files, frozen harness byte-identical).

## 1. The binding rules (unchanged from wave 8)
1. ⭐ **SAME ENCODER FOR EVERY ARM.** One shared encoder (recon on maze images), every arm consumes
   its features, one shared action head, one **exact** judge. Unfrozen ⇒ each arm trains its own at
   **equal compute**.
2. ⛔⛔ **NO RUN-KILLING TRIPWIRES.** Fragment-lookup and composition-blind scores are **REPORTED
   COLUMNS, never abort conditions.** Every commissioned experiment runs to completion.
3. ⭐⭐ **TWO ROW CLASSES, label every row.** **RESULT** = vs learned baselines (MLP, LSTM,
   transformer) at matched parameters, table as a **printed floor**. **DISPLAY** = inductive bias;
   the **only** comparator is a transformer with a refuse token trained on impossible examples, and
   our line is **"no impossible example ever seen."** ⛔ A DISPLAY row is never quoted as a RESULT.
4. ⭐ Paper object = the **Naive-CLU**, composition-led. Physics is an extended section.
5. ⛔⛔ **THE GYM IS NEVER KILLED.** The maze runs regardless, on local CPU. A design weakness is a
   **correction to the spec**, never a reason to stop.

## 2. Estimand and budget
**≥ 49 paired corners; margin declared in the PREREG before the first held-out number. 3 seeds.**
**Budget: `maze` 3 runs.** ⛔ `single-design` 3/3 and `generation-table` run 3 are **BANKED and not
yours**. Ledger row before the run. All scored runs reported; the best never alone.

## 3. ⭐ THE ENVIRONMENT — CMAZE-1, adopted, and its numbers travel with every claim
`env-scout` built and gated it. **Adopted configuration: adversarial placement, NO acceptance
filter**, n=500, **exact judge, no tolerance anywhere**, `n_colours=4`, `braid=0.40` adversarial.

| planner | score |
|---|---|
| ORACLE (full composition) | **1.0000** |
| BLIND-B (rule A dropped) | 0.0000 |
| BLIND-COLOUR | 0.0840 |
| BLIND-A (rule B dropped) | 0.1280 |
| BLIND-FACTOR (A×B local only) | 0.2600 |
| ⚠ **BLIND-SEQUENTIAL** (key order from A) | **0.4980** |
| ⭐⭐ **FRAGMENT-LOOKUP r=1 / r=2** | **0.0000 / 0.0000** |

⭐ **Rule B bends the route in 81.6 % of episodes** (+14.8 steps; optimal 42.4 under A+B vs 27.6
under A alone) — the first environment to satisfy the two-gym law.
⚠⚠ **The 0.4980 blind-sequential column is a CAVEAT THAT TRAVELS WITH EVERY COMPOSITION CLAIM.**
Print it beside every RESULT row. ⚠ The uniform unfiltered generator yields 0.9100 — also printed.
⛔ MiniGrid/BabyAI were rejected **from the generator code**: DoorKey is **1.0000 composition-blind**.
⛔ A perfect maze has one route (braid=0.00 yields zero valid instances) — **the source must be braided.**

## 4. ⛔ KNOWN — do not rediscover
1. ⭐⭐ **THE RESULT (w5):** `pca_tensor` — per-well leading principal subspace, **unsupervised** —
   beats `mlp_180k` on rows 3 and 4, all three seeds. That object is the incumbent.
2. ⭐ **Exact deletion WORKS** (w8): downdate == refit to **1e-16…1e-14**, **9/9 across three seeds**,
   0 data passes, items never kept, corroborated by two independent implementations.
3. ⛔ **Compose and forget are incompatible AT MATCHED BYTES** (w8), and the mechanism is known: the
   covariance costs `Σ d_k²`, which **forces K ≥ 36**, and **above K ≈ 5 composition collapses**
   (0.0648 at K=40 vs 0.8918 at K=5, the latter at **7.48×** the pin). ⭐ It is a **budget** result.
4. ⛔ **The store does NOT force better features** (w8): BAR B fired. ⭐ But the decomposition is
   real — **unfreezing with the store on the gradient path HURTS** (5/6 seed-rows negative) and
   **the swap loss REPAIRS it** (4/6 positive, p ≤ 0.0003). ⭐ **Report decomposed, never as a net.**
5. ⛔ **The dynamics cost composition** (w6); **a lifetime is worse than instant eviction** (w6);
   **symbolic refusal ties an 8-byte rule** (w5); **row 2 belongs to a generic learner** (w4);
   **placement exhausted** (w4). ⛔ None of these is reopened.
6. ⚠ **Novelty:** the phrase *"multilinear/bilinear factored generation with a request-indexed tensor
   head"* is **Tenenbaum & Freeman 2000 / TensorFaces 2002** — ⛔ never claim it. ⭐ Exact deletion on
   this object is **unclaimed in the literature** and is the strongest claim the programme owns.

## 5. Process — the rules that have cost this programme real time
1. ⛔⛔ **NEVER a box-wide process kill.** Kill only your own PIDs, captured at launch.
2. ⛔⛔ **Do NOT end a turn while detached work runs** — nothing re-invokes you. ⭐ **Poll inside a
   turn.** ⚠ And **size work before launching**, rather than polling better — a wave-8 hub logged
   that fix against itself after repeating the mistake.
3. ⭐⭐ **Commit and bundle the branch WHOLE before you finish.** Verify outputs by **directory
   listing** (⛔ `git status` cannot see gitignored outputs), code by `git status` + `git bundle verify`.
One launcher per hub; `START_UTC.txt` first. `LOG.md` in dated bullets; final report is its last
section. ⛔ A declared NOT-RUN is never a null. Local CPU, 3 threads.
