# matched-bytes-frontier — experiment-engineer report

**Task + acceptance criterion:** measure the **forgetting-vs-BYTES** frontier on Split-MNIST
Class-IL — every method given the same byte budget at ≥5 points spanning 50×, ≥3 seeds, byte
accounting pinned first, DER++ added, the kNN-in-φ launder run at matched **bytes**, saturation
reported — and say whether CLU dominates any region against tuned ER / DER++ / GDumb / iCaRL.
**Status: done.** All 6 budget points × 3 seeds ran; the pre-registered falsifier **fired**.

> ⭐ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Five items.**
> 1. ⛔ **The primary claim FAILS as pre-registered.** At matched bytes the kNN-in-φ ring buffer
>    has **lower forgetting than the CLU store at all 6/6 budget points** (Δ = +0.032, +0.042,
>    +0.027, +0.039, +0.018, +0.005) and higher ACC at 6/6. **Fifth consecutive laundering.**
>    ⭐ Sharper: the *same-keys* launder — plain arg-min over the store's own wells, i.e. the
>    settle deleted — also beats the settle at **6/6** on forgetting *and* 6/6 on ACC. **On this
>    benchmark the physics is a small, consistent net negative.** File as SUPPLEMENTARY.
> 2. ⭐ **What IS real and new: at matched bytes CLU beats the entire replay field on BOTH axes.**
>    Lower forgetting than every raw-exemplar method at **5 of 6** budgets (all but the smallest),
>    and higher **ACC at 6 of 6** — including **0.903 ± 0.003 vs iCaRL 0.885 ± 0.002** at 1.2 MiB.
>    Under the strict pre-registered rule (sd-separated **and** LA-in-band) the dominance region is
>    **B = 314 000 floats (1.2 MiB) only**. ⚠ This is *partly by construction* (replay stores 785
>    floats/item by definition) — it is the supplementary claim, not the primary one.
> 3. ⛔ **LwF carried item CLOSED, but NOT as a fix.** ACC is flat at **0.196** across a **200×**
>    α sweep (0.5→100); the w25 fixed-distillation-batch bug was real but immaterial. What moves it
>    is the **cross-entropy scope**: current-task-only CE gives **0.556** at the real protocol —
>    past the published 0.239 in the *other* direction. Neither convention reproduces 23.9. **LwF
>    stays at 0.196 and must be flagged as a 4.3 pp-low reimplementation wherever quoted.**
> 4. ⚠ **Two baseline bugs fixed that change w25's published table:** iCaRL ran up to **2.5× over
>    its item budget** at small budgets, and `memory_floats` under-counted every method. Any w25
>    memory-per-item number should be re-derived from the new pinned accounting (**41**, not 32,
>    floats/item for the store ⇒ **19.1×**, not 24.5×, vs a raw exemplar).
> 5. ⭐ **The store saturates, as pre-registered — at exactly one point.** At B = 314 000 the store
>    holds **6 085 ± 66 of 7 658** allowed wells (6 027 / 6 177 / 6 051) (**fill 0.795**) because the *spacing gate*, not
>    the budget, becomes binding. `refused_full = 0` everywhere. The capacity law shows up inside a
>    benchmark. ⚠ But forgetting did **not** flatten there (0.076 → 0.049), falsifying my P3(b).

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** admission + isolation, measured on the **forgetting** axis.
- **Laundering control:** kNN-in-φ **at matched BYTES**, both forms, at **every** budget point —
  the ring buffer stores 34 floats/key against the store's 41 floats/well, so it was handed
  **1.21× more keys than the store had wells** at every point (184/153 … 9 235/6 085).
- **Falsifies:** CLU dominates no region; or the matched-bytes launder dominates everywhere.
- **Does NOT falsify:** losing ACC (it did not — it won ACC vs replay at 6/6); losing at large
  budgets where replay saturates; sitting below joint/offline (0.946).
- **Outcome against the declaration:** ⛔ **the falsifier fired.** The launder is never beaten.
  The replay-only region is real but is the supplementary claim.

---

## Flag-provenance table (governs every number in this report)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/matched-bytes-frontier`, base local `main @ ff85573` |
| commits (6) | `6556095` accounting+DER++ +LwF+iCaRL · `1380496` frontier harness · `0b441c6` tests · `3298eea` fill fraction · `822e9cf` `lwf_ce_scope` · `c049b4e` untrack smoke artifacts |
| worktree | `../CHLU-mbf` (protocol §3.2; a second engineer worktree `CHLU-cl-encoder` was live in parallel). **Main venv reused** via `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python` — no worktree `uv sync`, no JAX drift |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** (main venv) |
| harness | `chlu.experiments.exp_cl_entry.run_byte_frontier` (`--items frontier`), orchestrated per (seed, budget) by `.claude/scratch/matched-bytes-frontier/run_frontier.py` so a long run survives interruption. Every number re-derived from the shipped JSONs by `.claude/outputs/matched-bytes-frontier/render.py` (output verbatim in `RENDER.txt`) |
| dataset / scenario | **Split-MNIST Class-IL**, 5 tasks × 2 classes, from scratch, task identity **NOT** given at test; read-out masked to classes seen so far. 2 000 train / **1 000 test per task** (unchanged from w25) |
| seeds | **0, 1, 2** at every budget point (18 CLU runs, 72 replay runs, 15 budget-free runs) |
| budget grid | **6 280 · 19 625 · 39 250 · 78 500 · 157 000 · 314 000 floats** (24.5 KiB → 1 226 KiB, span **50.0×**); P5 = 200 raw exemplars = the w25 operating point |
| φ | `phi_arm=pca`, **`phi_dim=32`**, `task1_only` (PRIMARY; fit on task-1 classes only, from a pool disjoint from every stream item, frozen). `generic_frozen` **not run** (declared upper bound, not needed for a frontier) — a deliberate compute saving |
| store | `clu_s_frac=0.2`, `d_safe_mult=4.4`, `s_policy="refit"`, `clu_b=1.0`, `clu_alpha=1e-3`, `clu_gamma=0.1`, `clu_steps=150`, `dt=0.5·s/√b` (auto), `clu_tail_frac=0.1`, **`newtonian_identity`**, `rollout_chunk=256`, `allow_relocation=False`, class-balanced LRU eviction, **decay OFF** (`items=["frontier"]` never ticks) |
| baselines | shared MLP 784-400-400-10, Adam `lr=1e-3`, `baseline_iters=500`/task, `batch=128`, `fisher_samples=200`, `eval_chunk=1024`. **N78 tuning on seed 0 at P5 only, then frozen for every seed AND every budget:** `ewc_lambda ∈ {100,1e3,1e4} → 100` (edge), `si_c ∈ {0.1,1,10} → 0.1` (edge), `lwf_alpha ∈ {0.5,1,2,5,10} → 0.5` (edge, but the sweep is **flat to ±0.0004**), `derpp (α,β) ∈ {0.2,0.5,1.0}×{0.5,1.0} → (0.2, 0.5)`. Tuning cost 351 s |
| byte accounting | `count_controller_record_floats=True` (**conservative against CLU**), `bytes_per_float=4` |
| anti-degeneracy | `frontier_la_band=0.10`; `constant_predictor` control on the plot |
| PREREG | `.claude/outputs/matched-bytes-frontier/PREREG.md`, written **before** any harness ran (one same-day amendment, logged in-file, fixing 784→785 floats/item integers; smoke-test only had run) |
| wall clock | tuning 351 s + 18 (seed, budget) jobs, 3 090 s (seed 0) / ~3 400 s (seed 1, under load 35–41) / ~2 900 s (seed 2). Full run ≈ 2.7 h |

---

## §1 ⛔ THE BYTE ACCOUNTING (pinned in PREREG §1, before any run)

Unit = float32 = 4 B. Every per-item scalar is charged one float even when it is an `int`/`bool` —
**deliberately conservative against CLU**, which keeps more scalar bookkeeping per item than a raw
buffer does. `dim = 784`, `phi_dim = 32`, `n_classes = 10`. Held by
`tests/test_cl_entry.py::test_byte_accounting_is_pinned`.

| method | arrays actually kept per item | floats/item | B/item |
|---|---|---|---|
| **CLU store** (primary) | `centers[i]` 32 + `payloads[i]` + `amps[i]` + `active[i]` + `ItemRecord` non-redundant scalars (`leak`, `permanent`, `born`, `last_used`, `item_id`, `slot`) | **41** | 164 |
| CLU store (landscape only, secondary) | `centers`+`payloads`+`amps`+`active` | 35 | 140 |
| **kNN-in-φ ring buffer** (the launder) | key 32 + label + age | **34** | 136 |
| kNN-in-φ same keys (the launder) | shares the store's arrays | 41 | 164 |
| **ER / GDumb / iCaRL** | raw exemplar 784 + label | **785** | 3 140 |
| **DER++** | raw exemplar 784 + label + stored logits 10 | **795** | 3 180 |
| EWC / SI / LwF / finetune / joint | **no episodic memory** ⇒ budget-independent | 0 | 0 |

⇒ **19.1× more items for CLU than a raw-exemplar method at the same bytes** (w25's "24.5×" used
the landscape-only 32-float count and is superseded), and **the launder gets 1.21× more keys than
CLU gets wells.** ⛔ Nothing is smuggled: the CLU entry runs **zero gradient steps** — it has no
network, no optimizer state, no logits.

**Fixed (per-method, not per-item) state — reported, not in the primary x-axis:**

| method class | fixed state | floats |
|---|---|---|
| CLU entry + both launders | frozen PCA-32 read-in (`components` 32×784 + `mean` 784) | **25 872** |
| every gradient baseline | MLP 784-400-400-10 backbone | **478 410** |
| EWC / SI | + Fisher (or ω) + reference θ | 1 435 230 |
| LwF | + frozen previous model | 956 820 |

**The accounting rule was fixed in advance** (PREREG §1b): the primary x-axis is the
*episodic-memory* budget alone (the field's "buffer size"), which is the choice **unfavourable to
CLU** — charging φ to CLU while not charging the backbone to the baselines would be a double
standard. The secondary all-fixed-charged frontier is in `RENDER.txt`: there CLU's total at the
top point is **275 357 floats vs every replay method's 792 410**, i.e. the same conclusions with a
2.9× larger margin. **Neither axis was chosen after seeing results.**

---

## §2 ⭐ THE FRONTIER — forgetting vs BYTES (3 seeds, mean ± sd)

Figure: `cl_entry_byte_frontier_mnist.png` (forgetting · ACC · LA).

| method | 24.5 KiB | 76.7 KiB | 153 KiB | 307 KiB | **613 KiB** (w25 op. pt.) | 1 226 KiB |
|---|---|---|---|---|---|---|
| | *6 280 fl* | *19 625* | *39 250* | *78 500* | *157 000* | *314 000* |
| ⛔ **kNN-φ ring buffer** (launder) | **0.162** ± 0.018 | **0.106** ± 0.007 | **0.086** ± 0.005 | **0.069** ± 0.002 | **0.058** ± 0.002 | 0.045 ± 0.002 |
| ⛔ **kNN-φ same keys** (launder) | 0.179 ± 0.016 | 0.132 ± 0.010 | 0.101 ± 0.007 | 0.100 ± 0.005 | 0.065 ± 0.007 | **0.044** ± 0.002 |
| ⭐ **CLU entry** | 0.193 ± 0.016 | 0.149 ± 0.010 | 0.113 ± 0.006 | 0.108 ± 0.007 | 0.076 ± 0.008 | 0.049 ± 0.000 |
| **GDumb** | 0.128 ± 0.017 | 0.231 ± 0.059 | 0.152 ± 0.056 | 0.122 ± 0.033 | 0.093 ± 0.009 | 0.079 ± 0.008 |
| **iCaRL** | 0.208 ± 0.020 | 0.305 ± 0.043 | 0.212 ± 0.015 | 0.148 ± 0.011 | 0.106 ± 0.006 | 0.068 ± 0.003 |
| **DER++** (new) | 0.883 ± 0.032 | 0.693 ± 0.030 | 0.469 ± 0.017 | 0.311 ± 0.016 | 0.192 ± 0.006 | 0.119 ± 0.002 |
| **ER** | 0.879 ± 0.025 | 0.715 ± 0.013 | 0.514 ± 0.016 | 0.361 ± 0.011 | 0.264 ± 0.022 | 0.166 ± 0.003 |
| *EWC 0.993 · LwF 0.994 · finetune 0.994 · SI 0.991* (no memory ⇒ flat) | | | | | | |
| *joint (offline) 0.025 ± 0.004 · constant-predictor **0.000*** | | | | | | |

**ACC carried alongside (never the claim, but it settles the degeneracy question):**

| method | 24.5 KiB | 76.7 KiB | 153 KiB | 307 KiB | 613 KiB | 1 226 KiB |
|---|---|---|---|---|---|---|
| kNN-φ ring buffer | **0.731** | **0.818** | **0.856** | **0.885** | **0.905** | **0.921** |
| kNN-φ same keys | 0.698 | 0.782 | 0.827 | 0.846 | 0.888 | 0.908 |
| ⭐ **CLU entry** | **0.673** | **0.764** | **0.815** | **0.837** | **0.878** | **0.903** |
| iCaRL | 0.609 | 0.709 | 0.776 | 0.827 | 0.857 | 0.885 |
| DER++ | 0.284 | 0.434 | 0.614 | 0.738 | 0.830 | 0.884 |
| ER | 0.288 | 0.418 | 0.578 | 0.699 | 0.775 | 0.849 |
| GDumb | 0.448 | 0.541 | 0.664 | 0.743 | 0.795 | 0.829 |

(sd ≤ 0.027 everywhere; joint 0.946 ± 0.002, EWC/SI/LwF/finetune 0.195–0.196, constant 0.110.)

### What the frontier says, in four statements

1. ⭐ **CLU beats the whole replay field on ACC at 6/6 budgets** — by +0.064 over iCaRL at the
   smallest and **+0.018 at the largest (0.903 vs 0.885)**, and by 0.05–0.39 over ER/DER++/GDumb.
   *That is the reverse of w25*, where at matched **items** CLU sat 0.153 below iCaRL. Matching
   bytes rather than items flips the ACC ordering against replay entirely.
2. ⭐ **CLU has lower forgetting than every raw-exemplar method at 5 of 6 budgets** (all but the
   24.5 KiB point, where GDumb's near-degenerate 0.128 at LA 0.549 wins). Applying the **strict**
   pre-registered rule — strictly lowest, **separated by ≥1 sd**, **and** LA within 0.10 of the best
   LA — the region shrinks to **one point, 1 226 KiB** (Δ vs iCaRL −0.019, sd-separated, LA 0.942
   vs best 0.982). ⚠ **Knife-edge, disclosed:** 76.7 KiB is sd-separated (Δ −0.082 vs GDumb) and
   fails the LA clause by **0.007** (CLU LA 0.883 vs ER 0.991, band 0.890). 153/307/613 KiB pass
   the LA clause and are lower, but not sd-separated because GDumb's sd is 0.056/0.033/0.009.
   The strict reading is the one I registered, so it is the one I report.
3. ⛔ **The launder is never beaten — the declared falsifier.** The matched-bytes ring buffer has
   lower forgetting at **6/6** (Δ = +0.032, +0.042, +0.027, +0.039, +0.018, +0.005) and higher ACC
   at **6/6**. The gap *narrows* monotonically with budget (0.032 → 0.005) but never closes.
4. ⭐⛔ **The settle is a small net negative, and this is the cleanest measurement of it the
   program has.** The *same-keys* launder differs from the CLU line by exactly one thing: it skips
   the damped-Verlet settle and takes arg-min over the same wells. It is **better on forgetting at
   6/6** (mean Δ 0.011) and **better on ACC at 6/6** (mean Δ 0.013). No φ, no buffer, no admission
   policy differs. **On Split-MNIST Class-IL the physics costs ≈1 pp.**

---

## §3 ⭐ WHERE THE STORE SATURATES (a result, not an embarrassment)

| B (floats) | budget wells | live at end | fill | admitted fraction, tasks 1→5 | `refused_full` | saturated? |
|---|---|---|---|---|---|---|
| 6 280 | 153 | 153 | 1.000 | 0.63, 0.93, 0.70, 0.55, 0.58 | 0 | no |
| 19 625 | 478 | 478 | 1.000 | 0.58, 0.90, 0.67, 0.50, 0.60 | 0 | no |
| 39 250 | 957 | 957 | 1.000 | 0.57, 0.84, 0.63, 0.49, 0.61 | 0 | no |
| 78 500 | 1 914 | 1 914 | 1.000 | 0.57, 0.78, 0.60, 0.45, 0.60 | 0 | no |
| 157 000 | 3 829 | 3 829 | 1.000 | 0.57, 0.77, 0.60, 0.48, 0.61 | 0 | no |
| **314 000** | **7 658** | **6 085** ± 66 | **0.795** | 0.57, 0.77, 0.60, 0.48, 0.62 | **0** | ⭐ **YES** |

- ⭐ **The binding constraint changes identity between the last two points.** Below 314 000 floats
  the store is **budget-bound** (it fills its allowance exactly, evicting under the class-balanced
  LRU). At 314 000 it is **gate-bound**: 10 000 offers, ~6 085 admitted, and it simply never fills
  the 7 658 slots it was given. `refused_full = 0` everywhere, so no capacity *alarm* fired — the
  refusals are all spacing refusals, i.e. the address space ran out, not the allowance.
- **This is the packing law inside a benchmark.** The admitted fraction is stable at 0.57–0.62 per
  task from 39 250 floats upward — offering a 20 %-larger store does not buy 20 % more items.
- **Geometry** (mean over seeds): median-NN address spacing shrinks 4.409 → 3.107 as the store
  grows; `s` shrinks with it (0.870 → 0.622) because `s_policy="refit"`; σ_q 4.225 → 2.839; and the
  **corrected packing slack is essentially constant at 0.337–0.353** across a 50× budget range.
  The store runs ~3× past the packing bound at every budget — which is intrinsic to classification
  (a query is a different image, not a corrupted copy) and is why the gate, not the budget, ends up
  binding. *Never quote the retracted "1.08".*
- ⚠ **Registered prediction P3(b) FALSIFIED:** I registered that forgetting would flatten above
  157 000 floats (<0.02 change). It fell **0.076 → 0.049 (−0.027)**. Saturating the *item count* did
  not saturate the *forgetting* — the store at 79.5 % fill still improves, because what improves is
  class coverage per well, not well count alone.

---

## §4 DER++ (new baseline) and the LwF retune (carried item)

### DER++ — added, tuned, behaves exactly as the literature says
Reservoir buffer carrying the logits it distils against (**counted: +10 floats/item**), two
independently drawn buffer batches per step, α·‖f(x)−z‖² + β·CE. Grid `{0.2,0.5,1.0}×{0.5,1.0}`
swept on seed 0 at P5 → **(0.2, 0.5)**, ACC 0.835 (grid range 0.822–0.835), then frozen.
At P5: **ACC 0.830 ± 0.006, forgetting 0.192 ± 0.006** — inside my registered [0.78, 0.88] /
[0.10, 0.22] bands, and **above ER on both axes at 5 of 6 budgets** (its logit-distillation term is
an explicit anti-forgetting term ER lacks). No debugging clause needed. It is the strongest replay
method here on ACC at the top budget after iCaRL (0.884 vs 0.885) — its absence would indeed have
been noticed.
⚠ Deviation stated, not hidden: this harness trains a whole task then updates the buffer (as its ER
does), so `z` is the end-of-task logit rather than the single online step's. That is the **stronger**
teacher, i.e. generous to the baseline.

### LwF — retuned, and the carried item closes as **NOT A TUNING ISSUE**
| what I did | result |
|---|---|
| fixed the w25 bug: the distillation batch was drawn **once per task** and reused for all 500 steps; it is now the **current minibatch** (`extra_loss(m, aux, xb)`) | ACC 0.196 → **0.196**. The bug was real and **immaterial** |
| extended the α grid `{0.5,1,2}` → `{0.5,1,2,5,10}` (w25's winner sat at the edge) | ACC **0.1958 / 0.1958 / 0.1954 / 0.1956 / 0.1958** — flat to **±0.0004** |
| stress test α = 100 (200× the w25 value) | ACC **0.189**; the term is *active* (LA moves) but cannot stop the Class-IL collapse |
| diagnostic: mask the training CE to the **current task's** classes instead of all seen classes | ACC **0.556** at the real protocol (0.463 at a 10×-reduced one) |

**Reading.** The 4.3 pp gap against van de Ven & Tolias's published 23.85 is **not** a
hyper-parameter — a 200× α sweep does not move it. It is a **loss-decomposition convention**: the
"CE over all seen classes" convention gives 0.196 and the "CE over the current task" convention
gives 0.556, and the published value sits *between* them. Our convention is the one under which
**EWC 0.196 / SI 0.195 / finetune 0.196** reproduce their published 19.9–20.0 to ≤0.5 pp, so it is
the one that stays shipped; the alternative is now the config flag `lwf_ce_scope="current_task"`
so the measurement is reproducible without patching.
⛔ **LwF remains 4.3 pp below the published value and must be flagged wherever the table is
quoted.** It does not affect any conclusion here (LwF has no episodic memory; it is a flat line at
0.994 forgetting).

### Two fairness bugs found and fixed (they change w25's table)
1. **iCaRL over-ran its item budget.** `m_per = max(1, budget // n_classes)` kept one exemplar per
   class even when the budget was smaller than the class count ⇒ **10 items on a budget of 4**
   (2.5×) at the smallest frontier point. Budgets are now allocated exactly (`_class_quotas`).
   Regression-tested.
2. **`memory_floats` under-counted everyone** (`n × dim`, i.e. labels free, CLU's amps/active/record
   free). Now uses the pinned accounting.

---

## §5 PREREG scorecard (`PREREG.md`, written before any harness ran)

| # | registered | measured | verdict |
|---|---|---|---|
| P1a | CLU forgetting at P1–P3 ∈ [0.10, 0.22] | 0.193 / 0.149 / 0.113 | ✅ |
| P1b | ER at P1 ∈ [0.35, 0.70] | **0.879** | ❌ far above — 8 exemplars is worse than I allowed |
| P1c | iCaRL at P1 ∈ [0.15, 0.45] | 0.208 | ✅ |
| P1d | **CLU strictly below every replay method for B ≤ 39 250** (conf. 0.85) | ❌ at P1 (GDumb 0.128 < 0.193); ✅ at P2/P3 | ❌ **direction wrong**: CLU's replay advantage grows *with* budget, it is not a low-byte effect |
| **P2** | **the matched-bytes launder's forgetting ≤ CLU's at every point** (p = 0.70) | **6/6, Δ +0.005 … +0.042** | ✅ **the falsifier fired, as the more likely branch predicted** |
| P2′ | competing branch: CLU below the launder at P1–P2 by ≥0.01 (p = 0.30) | CLU is **above** at P1–P2 by 0.032 / 0.042 | ❌ (correctly given the lower prior) |
| P3a | `n_live` stops tracking the budget in P5–P6, at 5 000–8 000 wells, admitted 0.55–0.85 | saturates at **P6**, **6 085** wells, admitted 0.57–0.77 | ✅ |
| P3b | forgetting flattens above P5 (<0.02 change P5→P6) | **−0.027** | ❌ |
| P3c | `refused_full = 0` at every point | **0** at 18/18 runs | ✅ |
| P4 | DER++ at P5: ACC ∈ [0.78, 0.88], forgetting ∈ [0.10, 0.22] | **0.830 / 0.192** | ✅ |
| P5 | crossover with the best replay method at B ∈ [3×10⁴, 1×10⁵] | crossover is at **B ∈ [6.3×10³, 2.0×10⁴]**, below the window | ❌ |
| P6 | LwF retune reaches [0.21, 0.26] | **0.196**, flat over a 200× sweep | ❌ → reported as **unresolved-by-tuning**, with the cause localised (§4) |
| §2 | LA reported for every method; constant-predictor control computed | done; constant predictor ACC 0.110, forgetting **exactly 0.000** | ✅ |

**Score: 6 ✅ / 6 ❌.** The failures are the informative half — three of them (P1d, P5, P3b) all say
the same thing I had backwards: **the store's advantage over replay is a large-budget effect, not a
small-budget one**, because what CLU buys with bytes is class coverage in a 32-dim address space,
and coverage keeps paying long after 400 raw exemplars have stopped paying.

---

## §6 How I verified

- `PYTHONPATH=<worktree> .venv/bin/pytest tests/ -q -p no:randomly --no-cov` → **700 passed, 0
  failed** (603.83 s). Base `main @ ff85573` measured at **690 collected** ⇒ the branch adds **10** tests.
- `pytest tests/test_cl_entry.py` → **27 passed** (was 17 at w25). `pytest tests/test_config.py`
  → **7 passed**, including `test_every_group_round_trips_mutated` (the w23 four-site trap; all new
  knobs went into the existing `ExperimentClEntryConfig`, so no new group).
- `ruff check chlu/ tests/` → **All checks passed** after every change.
- Smoke: `python -m chlu.experiments.exp_cl_entry --quick --items frontier` → exit 0 (80.6 s,
  3 tiny budgets, all 4 replay methods + both launders + the store).
- Real runs: 18 (seed, budget) jobs + one tuning pass; per-job wall clock logged in
  `.claude/scratch/matched-bytes-frontier/seed{0,1,2}.log`. **Every number in this report is
  re-derived from the shipped JSONs** by `render.py`; its verbatim output is `RENDER.txt`.
- Budget compliance is machine-checked, not asserted:
  `test_byte_frontier_runs_and_holds_every_method_to_the_same_bytes` fails if any method's stored
  floats exceed the budget it was given, at any point.

---

## §7 Known limitations (stated, not hidden)

1. **One dataset.** Split-MNIST only, per the task's scope line. The extension point is
   Split-CIFAR-10 **iff `cl-encoder` clears its gate** — w25 showed PCA-32-on-CIFAR-pixels is not an
   address space (kNN over it tops out at 0.21), so re-running this frontier on CIFAR *before* a
   working φ would measure the φ, not the frontier.
2. **`generic_frozen` φ was not run** (compute). The frontier is the PRIMARY (leakage-free) arm
   only. w25 measured the strict-φ cost at +0.028 on MNIST, so the reference arm would shift every
   φ-based line — CLU **and both launders** — up together; the CLU-vs-launder ordering is
   differential and would not change.
3. **GDumb's sd is large** (0.059 / 0.056 / 0.033 at P2–P4) because it retrains from scratch each
   task. That is what blocks sd-separation at three budgets where CLU is nonetheless lower. More
   seeds would likely convert P4/P5 into sd-separated dominance-vs-replay points; 3 seeds is what
   the wave's compute allowed, and I report the strict reading rather than the hopeful one.
4. **The LA clause is a blunt instrument.** It excludes the 76.7 KiB point by 0.007. I registered
   the band in advance and applied it verbatim; a reader who prefers a 0.15 band gets two
   dominance-vs-replay points instead of one. Both readings are in `RENDER.txt`.
5. **The top budget is the only saturated point**, so "where the store saturates" is measured at
   one point, not resolved as a curve. A 2× and 4× extension above 314 000 floats would turn the
   saturation point into a saturation *law*; it costs ~20 and ~40 min/seed.
6. **DER++'s buffer logits are end-of-task, not online** (§4), and the tuning grid is 6 points on
   one seed at one budget. Both are stated in-code.
7. **`joint`, EWC, SI, LwF, finetune were run once per seed** and drawn as horizontal lines. That
   is correct (they hold no episodic memory) but it means their sd is a 3-seed sd, not a
   3-seeds-×-6-budgets sd.

---

## Git footprint

- **Branch** `agent/experiment-engineer/matched-bytes-frontier`, base local `main @ ff85573`.
  `git rebase main` → *up to date* (no-op; base did not move). ✅ **Verified from the MAIN repo**
  (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/matched-bytes-frontier`)
  that the shared ref carries **all 6 commits** — the w4 lost-commit mode is checked, not assumed.
- **6 commits:** `6556095` · `1380496` · `0b441c6` · `3298eea` · `822e9cf` · `c049b4e`.
- **Files (5, all in my declared ownership):**
  `chlu/experiments/cl_baselines.py` (+271/−…: byte accounting, DER++, LwF CE-scope + distillation
  batch, iCaRL quota fix, `LA`), `chlu/experiments/exp_cl_entry.py` (+479: `run_byte_frontier`,
  `frontier_table`, `frontier_verdict`, `constant_predictor_row`, `ring_budget`/`phi` args on
  `run_clu_entry`, two figures), `chlu/config.py` (+51, all on the **existing**
  `ExperimentClEntryConfig` — no new group, so no four-site registration risk),
  `chlu/cli/experiment_cmd.py` (+13: `--items frontier`, `--budgets`),
  `tests/test_cl_entry.py` (+207: 10 tests).
- ⛔ **Did NOT touch** `exp_phi_stream.py`, the φ config surface, or `chlu/core/*` (owned by
  `cl-encoder`, `placement-landing`, `r2-excursion-reach`).
- ⚠ **One self-inflicted slip, fixed:** a `git add -A` swept the untracked `results/` smoke
  artifacts into the index; removed in `c049b4e` (`git rm --cached`). I did **not** add `results/`
  to `.gitignore` — that is a shared file outside my scope; see handover item 5.
- **Worktree `../CHLU-mbf` left in place** (it holds `results/` run output, untracked); the Hub
  removes it at integration. Canonical JSONs, figures and `RENDER.txt` are in
  `.claude/outputs/matched-bytes-frontier/`.
- **No push, no PR.**

---

## Open questions / follow-ups / risks

1. ⭐ **The store's ACC now beats the whole replay field at matched bytes and loses only to a
   trivial kNN over the same features.** The program's genuine-win bar therefore points at one
   question and only one: *is there any read for which the settle beats arg-min over the same
   wells?* Six budget points × 3 seeds say no on this benchmark, by ≈1 pp on both axes. The w25
   R3-native retry cell is the only regime where it did not lose. **That contrast — settle loses on
   clean classification, ties on corrupted-query retrieval — is the sharpest surviving lead.**
2. **More seeds would probably buy two more dominance-vs-replay points** (P4/P5 are lower but not
   sd-separated, blocked entirely by GDumb's variance). Cheap: ~35 min/seed for the replay arms
   alone. Whether that is worth it depends on whether the Hub wants the *supplementary* claim
   stated as "one budget point" or "a region".
3. **Extend the grid upward** (628 000 / 1 256 000 floats) to turn the saturation point into a law.
   The prediction is that CLU's line flattens at ~6 100–6 500 wells while the ring buffer keeps
   improving to 18 000+ keys — i.e. the launder gap should *re-open* above P6. If it does not, the
   store's geometry is not the limiter and something else is.
4. **The LA clause needs a program-level convention.** I registered 0.10 and it excluded a point by
   0.007. Someone should decide once whether the anti-degeneracy band is 0.10 or 0.15, before the
   next frontier is drawn against a different band.
5. **LwF's 4.3 pp gap is now localised but not closed**, and the same "CE scope" question applies to
   *every* rehearsal-free baseline in the table. EWC/SI/finetune reproduce, so the convention is
   right for them — but nobody has checked whether the published LwF number was produced under a
   third convention again.

---

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** *The matched-BYTES forgetting frontier exists and has run
   (w26).* `chlu exp-cl-entry --items frontier [--budgets ...]`. **Split-MNIST Class-IL, 6 byte
   budgets spanning 50×, 3 seeds: at matched bytes the CLU store beats ER / DER++ / GDumb / iCaRL
   on ACC at 6/6 budgets (0.903 vs iCaRL 0.885 at 1.2 MiB) and on forgetting at 5/6 — but the
   matched-bytes kNN-in-φ launder beats it on both at 6/6, and so does the same-keys launder (=
   the settle deleted). Fifth consecutive laundering; the frontier claim is SUPPLEMENTARY.**
2. **Candidate N-entries (3).** (a) *"At matched BYTES the designed store dominates the replay
   field on Split-MNIST Class-IL (ACC 6/6, forgetting 5/6) — and is still laundered by kNN-in-φ at
   6/6"* — tier A, the contested-win attempt with its own falsifier. (b) ⭐ *"Deleting the settle
   improves the store: arg-min over the same wells beats the damped-Verlet read on both ACC and
   forgetting at 6/6 matched-byte budgets (mean +0.013 / −0.011)"* — tier A, the cleanest
   negative the program has on the read mechanism, because nothing but the settle differs.
   (c) *"The CL store's binding constraint switches from budget to spacing gate at ~314 000 floats:
   6 085 of 7 658 wells filled, admitted fraction stable at 0.57–0.62, refused_full = 0"* — tier A,
   the packing law inside a benchmark.
3. **Retractions / corrections to w25's table.** (i) "**24.5× fewer floats**" → **19.1×** under the
   pinned accounting (32 → 41 floats/item). (ii) iCaRL's w25 row was run **over budget** at small
   budgets (the bug is fixed here; at 200 items the w25 number is unaffected). (iii) w25's
   `memory_floats` under-counted every method. (iv) w25's forgetting comparison "CLU 0.169 vs ER
   0.264 at matched items" stands, but it is a matched-**items** statement and must not be quoted
   as a matched-bytes one.
4. **§7 — new config/CLI surface** (all on the existing `ExperimentClEntryConfig`, no new group):
   `frontier_budgets_floats/_methods/_fixed_methods/_seeds/_n_test_per_task/_max_clu_items/
   _tuning_budget/_la_band`, `derpp_alpha/_beta/_grid`s, `tune_derpp`, `lwf_ce_scope`,
   `count_controller_record_floats`, `bytes_per_float`; `lwf_alpha_grid` extended to
   `{0.5,1,2,5,10}`; `derpp` added to the default `baselines`; CLI `--items frontier`,
   `--budgets`. `cl_metrics` gains `LA`. **Two default-behaviour changes** (DER++ in the baseline
   list, LwF grid) — both intended, both reported.
5. **⚠ Repo hygiene (one line, not mine to land):** `results/` is untracked but **not** in
   `.gitignore`, so any agent's `git add -A` sweeps smoke-run artifacts into a commit (it caught me;
   `c049b4e`). Adding `results/` to `.gitignore` would close it for everyone.
6. **Test count:** the branch adds **10** tests; the full suite measured **700 passed / 0 failed**.
7. **The LwF finding belongs in `negative_results.md`:** *"LwF's Split-MNIST Class-IL score is set
   by the cross-entropy class-masking convention (0.196 for 'all seen', 0.556 for 'current task'),
   not by its distillation weight (flat over a 200× sweep). Our 4.3 pp deficit against the
   published 23.9 is a convention gap, not a tuning failure."*
