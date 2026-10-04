# PREREG — matched-bytes-frontier (w26)

**Written 2026-07-28, BEFORE any frontier harness ran.** Nothing below is informed by a
measurement taken this wave. The only inputs are (a) the w25 `cl-entry-build` numbers, which are
public in `.claude/outputs/cl-entry-build.md`, and (b) arithmetic on the byte accounting in §1.

Acceptance criterion being pre-registered: *does CLU dominate a region of the
**forgetting-vs-BYTES** frontier on Split-MNIST Class-IL against tuned ER / DER++ / GDumb / iCaRL
**and** the matched-BYTES kNN-in-φ launder?*

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** admission + isolation, measured on the **forgetting** axis (not ACC).
- **Laundering control:** kNN-in-φ **at matched BYTES** — the ring buffer gets its full
  byte-equivalent key count (more keys than the store has wells, see §1), run at **every** budget
  point, in both forms (same-keys and independent class-balanced ring buffer).
- **Falsifies:** CLU dominates **no** region of the frontier, or the matched-bytes launder is
  ≤ CLU's forgetting at every budget point.
- **Does NOT falsify:** losing ACC at every budget point; losing at large budgets where replay
  saturates and the store is geometry-bound; sitting below joint/offline.

---

## §1 ⛔ THE BYTE ACCOUNTING — pinned before any run

Unit = **float32 = 4 bytes**. Every scalar an implementation keeps per stored item is counted at
4 bytes even when it is an `int`/`bool` — deliberately **conservative against CLU** (CLU keeps more
scalar bookkeeping per item than a raw buffer does). Split-MNIST: `dim = 784`, `phi_dim = 32`,
`n_classes = 10`.

### 1a. Marginal (per stored item) — this is the frontier's x-axis

| method | arrays actually kept per item | floats/item | bytes/item |
|---|---|---|---|
| **CLU store (primary accounting)** | `centers[i]` (32) + `payloads[i]` (1) + `amps[i]` (1) + `active[i]` (1) + controller `ItemRecord` non-redundant scalars: `leak`, `permanent`, `born`, `last_used`, `item_id`, `slot` (6) | **41** | **164** |
| CLU store (landscape only, *secondary*) | `centers`+`payloads`+`amps`+`active` | 35 | 140 |
| kNN-in-φ **ring buffer** (the launder) | key (32) + label (1) + age (1, needed by the balanced-LRU drop rule) | **34** | **136** |
| kNN-in-φ **same keys** (the launder) | shares the CLU store's arrays | 41 | 164 |
| **ER** | `x` (784) + `y` (1) | **785** | 3 140 |
| **DER++** | `x` (784) + `y` (1) + stored logits `z` (10) | **795** | 3 180 |
| **GDumb** | `x` (784) + `y` (1) | **785** | 3 140 |
| **iCaRL** | exemplar `x` (784) + `y` (1); NME class means are recomputed from the exemplars at eval, never stored | **785** | 3 140 |
| EWC / SI / LwF / finetune / joint | **no episodic memory** ⇒ budget-independent (horizontal lines on the frontier) | 0 | 0 |

⇒ **at matched bytes CLU gets 785/41 = 19.1× the items of a raw-exemplar method**, and the
**launder gets 41/34 = 1.21× more keys than CLU has wells.** The launder is therefore run
*adversarially over-resourced*, which is the point (task §DIAL: "anything less is a rigged
control"). w25's quoted "24.5×" used the landscape-only 32-float count; **41** supersedes it here.

### 1b. Fixed (per-method, not per-item) — reported, NOT in the primary x-axis

| method class | fixed state | floats |
|---|---|---|
| CLU entry + both launders | frozen PCA-32 read-in φ: `mean` (784) + `components` (32×784) | **25 872** |
| every gradient baseline (finetune/EWC/SI/LwF/ER/DER++/GDumb/iCaRL/joint) | MLP 784-400-400-10 backbone: 313 600+400 + 160 000+400 + 4 000+10 | **478 410** |
| EWC | + Fisher diag + reference θ = 2× backbone | +956 820 |
| SI | + ω + reference θ = 2× backbone | +956 820 |
| LwF | + a frozen copy of the previous model = 1× backbone | +478 410 |
| (Adam moments, 2× backbone, are transient optimizer state and are excluded for everyone) | | |

**Registered accounting rule (decided now, not after seeing results):** the **primary** x-axis is
the *episodic-memory* budget only (§1a) — this is what the CL field means by "buffer size", and it
is the axis on which the frontier is drawn. §1b is published in the report and a **secondary
"all-fixed-state-charged" frontier** is drawn as a sensitivity, in which CLU/launder pay +25 872
and every baseline pays +478 410 (or more). Charging φ to CLU while not charging the backbone to
the baselines would be a double standard; charging neither is the field convention. **Both are
reported; neither is chosen after the fact.** The primary axis is the one that is *unfavourable to
CLU* between the two (CLU has no backbone at all: **zero** learned parameters).

⛔ No other state exists on the CLU side: the entry runs **zero gradient steps**, so there is no
model, no optimizer, and no logits to smuggle capacity through. `PhiStore.item_task` / `cohort`
are diagnostic dicts, not used by any read or write decision, and are excluded (and would add 2
floats/item if charged).

---

## §2 METRIC DEFINITION (pinned)

On the (T×T) Class-IL accuracy matrix `A[t,i]` (accuracy on task *i*'s held-out test set after
training through task *t*; read-out masked to classes seen up to *t*), T = 5:

- `ACC   = mean_i A[T-1, i]`
- `BWT   = mean_{i<T-1} ( A[T-1,i] − A[i,i] )`                        (GEM, Lopez-Paz 2017)
- `forgetting = mean_{i<T-1} ( max_{t≥i} A[t,i] − A[T-1,i] )`          (Chaudhry 2018)
- `LA    = mean_i A[i,i]`  — **added this wave**: learning accuracy, the anti-degeneracy readout.

⭐ **Degenerate-forgetting control (registered now).** Forgetting is trivially ≈0 for a method that
never learns. Therefore: (i) `LA` is reported for every method at every budget point, and (ii) a
**constant-predictor reference line** (predict the majority seen class) is computed, with its ACC
and its ≈0 forgetting, so a reader can see the floor that "low forgetting" must beat to mean
anything. **A CLU dominance region is only claimed where CLU's `LA` is within 0.10 of the best
method's `LA` at that budget.** This clause is registered *before* the measurement.

---

## §3 THE BUDGET GRID (pinned)

Six points, expressed as the byte budget `B` every method receives, and quoted in
raw-exemplar-equivalent items `N_raw = B / (785·4 bytes)`:

| point | B (floats) | B (KiB) | N_raw (ER/GDumb/iCaRL) | N_derpp | **N_clu** | N_ring (launder) | × the w25 CLU operating point (6 400 floats) |
|---|---|---|---|---|---|---|---|
| P1 | 6 280 | 24.5 | 8 | 7 | 153 | 184 | 0.98× |
| P2 | 19 625 | 76.7 | 25 | 24 | 478 | 577 | 3.07× |
| P3 | 39 250 | 153.3 | 50 | 49 | 957 | 1 154 | 6.13× |
| P4 | 78 500 | 306.6 | 100 | 98 | 1 914 | 2 308 | 12.27× |
| P5 | 157 000 | 613.3 | **200** ← the w25 operating point | 197 | 3 829 | 4 617 | 24.5× |
| P6 | 314 000 | 1 226.6 | 400 | 394 | 7 658 | 9 235 | 49.1× |

> *Amendment, same day, before any real-data run (smoke-test only had been run):*
> the draft grid used multiples of 784 (pixels) rather than 785 (pixels **+ label**),
> so P5 gave 199 raw exemplars instead of 200. The budgets above are the multiples of
> 785 that make P5 exactly the w25 operating point. Item counts move by ≤6; the span
> is unchanged at 50.0×. No prediction below depends on the difference.

Span **50.0×**, ≥5 points, as required. `N = floor(B / floats_per_item)`.
**Registered compute-contingency rule (stated in advance):** if P6 does not complete within the
wave's compute, it is dropped and *reported as dropped with the wall-clock evidence* — it is never
silently replaced by a cheaper point, and no dominance claim is made about a budget region that was
not measured (the NOT-RUN ≠ null rule). Seeds: **0, 1, 2** at every point.

Budget-independent methods (finetune/EWC/SI/LwF/joint) are run **once per seed** and drawn as
horizontal lines — running them six times would be six identical runs.

---

## §4 REGISTERED PREDICTIONS

Derived from w25's operating-point numbers (CLU forgetting 0.169 / BWT −0.169; ER 0.264; GDumb
0.093; iCaRL 0.105; φ-ring-buffer launder 0.152 — **all at 200 items each, i.e. NOT matched
bytes**) plus §1's 19.1× / 1.21× conversion factors.

**P1 — CLU dominates the replay methods at low bytes.** For `B ≤ 3.9×10⁴` floats (P1–P3) CLU's
forgetting is **strictly lower than every one of ER / DER++ / GDumb / iCaRL**, because at those
budgets they hold 8–50 raw exemplars (≤5 per class) while CLU holds 152–956 wells.
Registered numbers: CLU forgetting at P1–P3 ∈ **[0.10, 0.22]**; ER at P1 ∈ **[0.35, 0.70]**;
iCaRL at P1 ∈ **[0.15, 0.45]**. *Confidence: high (0.85).*
⚠ **Pre-registered caveat on P1's value:** this region is *partly* a by-construction win — replay
pays 785 floats/item by definition of what it stores. It is reported as a frontier region, but the
program's genuine-win bar is P2 below.

**P2 — the contested comparison is CLU vs the matched-bytes launder, and I predict CLU LOSES it.**
At matched bytes the ring buffer gets 1.21× more keys than CLU has wells, and at matched *items*
w25 already measured launder forgetting 0.152 < CLU 0.169. Registered:
**the matched-bytes launder's forgetting is ≤ CLU's at every one of P1–P6** — i.e. **the declared
falsifier fires** — with probability **0.70**. The competing hypothesis (registered so it can win):
**CLU's forgetting is below the launder's at the low-budget end (P1–P2) by ≥0.01**, because the
spacing gate spreads the store's wells over the address space while the ring buffer's balanced-LRU
keeps whatever arrived last, and at 152–478 items coverage, not recency, is what protects old
tasks. Probability **0.30**. ⭐ **If and only if this second branch fires, the wave has its
contested win, and its region is P1–P2.**

**P3 — where the store saturates (a result, not an embarrassment).** The store's live count is
capped by the spacing gate, not only by the budget. w25 measured admitted fractions 0.57–0.95 per
task at a 200-item budget over 10 000 offers. Registered: **`n_live` at the end of the stream stops
tracking the budget somewhere in P5–P6, saturating at 5 000–8 000 live wells out of 10 000 offers
(admitted fraction 0.55–0.85)**, and consequently **CLU's forgetting curve flattens above
`B ≈ 1.6×10⁵` floats (P5), changing by <0.02 between P5 and P6.** `refused_full` is predicted to
be **0** at every point (permanent cohort is off in the frontier config, so a victim always exists).

**P4 — DER++ (new baseline).** At P5 (197 items): ACC ∈ **[0.78, 0.88]**, forgetting ∈
**[0.10, 0.22]** — i.e. between ER (0.264) and iCaRL (0.105), because its logit-distillation term
is an explicit anti-forgetting term that ER lacks. If DER++ lands *below* ER on both ACC and
forgetting, the implementation is presumed wrong and is debugged before the table is published.

**P5 — the crossover.** CLU's forgetting curve is nearly flat in budget (capacity/geometry-bound,
P3) while iCaRL's improves with exemplars. Registered crossover with the best replay method at
**`B` between 3×10⁴ and 1×10⁵ floats (P3–P4)**; above it, iCaRL has the lower forgetting.

**P6 — LwF retune (carried item).** w25's LwF = 0.196 vs the published Split-MNIST Class-IL
23.9 ± 0.4 (van de Ven & Tolias 2019), 4.3 pp low, **with the chosen α at the grid edge (2.0)**.
Two registered causes, in order of suspicion: (i) the distillation batch is drawn **once per task**
and reused for all 500 steps (the published method distills on the *current* minibatch), (ii) the
α grid `{0.5,1,2}` was too narrow. Registered: after fixing (i) and extending the grid to
`{0.5,1,2,5,10}`, **LwF ACC ∈ [0.21, 0.26]**, i.e. reaching the published band. If it stays
≤ 0.20 the retune is reported as **unresolved**, with the sweep table, and the baseline is flagged
as a known 4-pp-low reimplementation wherever it is quoted.

---

## §5 REGISTERED OUTCOME READINGS (decided now, applied verbatim later)

| outcome | reading |
|---|---|
| **CLU's forgetting is strictly lowest (incl. both launders) at ≥1 budget point, with mean−sd separation, and its `LA` is within 0.10 of the best `LA` there** | ⭐ **CONTESTED WIN.** "CLU dominates the forgetting-vs-bytes frontier for B ∈ [·,·]." Report the region, the saturation point, ACC alongside, and the by-construction caveat for the replay-only part of the region. |
| **CLU's forgetting is lowest among ER/DER++/GDumb/iCaRL at some budgets but the matched-bytes launder is ≤ CLU everywhere** | ⛔ **THE FALSIFIER FIRED (5th consecutive laundering).** The frontier region is real but it is φ's and the buffer's. Filed as *supplementary* (a by-construction win over raw-exemplar storage), **not** as the primary claim. Say so in the first 10 lines. |
| **CLU is dominated at every point by some replay method** | ⛔ **CLAIM DEAD.** Report the frontier as a negative result with the crossover budget. |
| **CLU's forgetting is lowest only where its `LA` is >0.10 below the best** | ⛔ **DEGENERATE.** Not a win; it is the "a method that never learns never forgets" artefact, and it is reported as such. |
| **`n_live` saturates and forgetting flattens** | ⭐ Reported as the **capacity law inside a benchmark**, with `n_live`, admitted fraction and `refused_full` at every point — a result, not an embarrassment. |
| **LwF retune reaches [0.21,0.26]** | carried item CLOSED; w25's LwF row is superseded, and the `.md` sites quoting 19.6 are put on the reconciliation list. |

---

## §6 What would make me distrust my own result
1. A CLU dominance region that appears only at P1 (8 raw exemplars) — that is a statement about
   raw-pixel storage, not about the store. It is why P2's contested comparison is the real test.
2. A forgetting advantage with an `LA` deficit > 0.10 (§2 clause).
3. Any budget point where CLU's `n_live` ≪ budget while the launder's is full: then the methods are
   not actually at matched bytes and the point must be re-reported at *used* bytes as well.
