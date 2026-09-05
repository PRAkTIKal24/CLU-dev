# PREREG_CL_PHI — the `φ` protocol for the w25 continual-learning entry

**Status:** pre-registration. Written w24, **before** the CL entry is built and (for the
protocol clauses) before the cost-of-strictness harness was run. This document is what
protects the entry from the **leakage attack** — *"your feature extractor was trained on the
classes you claim not to have seen yet."* Anything the entry does with `φ` that is not
licensed below is out of protocol.

**Binding authority:** the Head's ruling (w24 task `phi-stream-discipline`), the w20 law
(`φ` is never trained through the store), N89 / CM-22(i) (kNN-in-φ is mandatory on every
claim), and the standing scope caveat (masked recall is appendix-only).

**Implementation:** `chlu/experiments/exp_phi_stream.py` (`PHI_REGIMES`,
`fit_pool_for_regime`, `build_stream_read_in`, `OnlineReadIn`) + config group
`experiment_phi_stream` (`phi_regimes`, `s_policy`, …).

---

## 1. The three regimes and their status

| regime | status | what `φ` may be fit on | when it is frozen | may it produce a headline number? |
|---|---|---|---|---|
| **`task1_only`** | ⭐ **PRIMARY** | data whose labels are in **task 1's classes only** (Split-MNIST: {0,1}), drawn from a pool **disjoint from every stored item** | at the end of task 1; **never updated again for the whole stream** | **YES — every headline number comes from here** |
| **`generic_frozen`** | **REFERENCE — declared upper bound** | a pool spanning **all** stream classes (the w23-style pool), or generic pretrained features | before the stream starts; never updated | **NO.** Reported only, always adjacent to the primary arm, always labelled *"declared upper bound — leaks future tasks"* |
| **`online`** | **NOT RUN in w25 unless separately approved** | data from tasks **already seen** by the stream, updated as tasks arrive | continuously | **NO.** Its own experiment; whether it enters the CL results is a **separate decision** by the Head |

**Rule of quotation.** Any table, figure, or sentence that quotes a `generic_frozen` number
without the primary `task1_only` number **next to it** and without the words *declared upper
bound* is a protocol violation. The reference arm exists to bound the loss, not to flatter it.

## 2. What is frozen, and when — the timeline

```
t = 0   draw the store region and the fit region from disjoint index sets
        |
        |-- generic_frozen φ is fit here (all classes) ......... REFERENCE ONLY
        |
t = 1   task 1 arrives ({0,1})
        |-- task1_only φ is fit here, on task-1 classes ONLY ... PRIMARY
        |-- ⛔ FREEZE. φ is never refit, fine-tuned, or re-keyed again.
        |-- task 1's items are written into the DESIGNED store
        |
t = 2..T  each later task arrives; items are written through the ALREADY-FROZEN φ.
          No gradient, no statistic, and no hyper-parameter of φ is updated.
```

**Store-side clauses (so "frozen φ" is not quietly undone by the store):**
- The store is **designed**, never learned. `φ` is never trained through it (w20 law).
- The well width `s` is set by the fixed rule `s = clu_s_frac · median-NN(φ keys)`. Under
  the default `s_policy="refit"` it is recomputed at each stream position **from the keys
  already in the store** — never from future data. This is legitimate (the store may inspect
  its own contents) but **must be stated**; `s_policy="task1_frozen"` is available if a
  referee objects, and the entry must report which was used.
- **No stored item is ever re-keyed.** If a future protocol change requires re-keying (the
  online regime does), that is a new experiment and a new pre-registration, and the question
  *"does re-keying count as replay?"* must be answered first.

## 3. What each arm may see — the exhaustive list

| object | `task1_only` (PRIMARY) | `generic_frozen` (REFERENCE) |
|---|---|---|
| φ fit pool: classes | task 1's classes only | all stream classes |
| φ fit pool: size | `n_fit_pool` (same for both — the regimes differ only in *which* classes) | `n_fit_pool` |
| φ fit pool vs stored items | **disjoint by construction** (separate index regions) | **disjoint by construction** |
| φ fit objective | unsupervised only — PCA reconstruction directions / AE reconstruction MSE. **No label, no retrieval loss, no store gradient.** | same |
| future-task data | **never** | yes (this is exactly why it is only a bound) |
| task identity at test time | **not given** (Class-IL) | not given |
| stored items | written through frozen φ | written through frozen φ |
| queries | identical in both regimes, generated once | identical |

## 4. The mandatory controls (every claim, no exceptions)

1. **kNN-in-φ in EVERY regime** (N89, CM-22(i)). If kNN-in-φ ties or beats CLU-in-φ, the
   report says so in the required words: *the win is φ's, not ours.*
2. **The primary/reference pair is always reported together**, so the cost of strictness is
   visible in the same table as the number it qualifies.
3. **The w25 CL baselines are unchanged by this document** and remain mandatory
   (`continual-learning-recon`): tuned ER + iCaRL + **GDumb at matched memory**, with EWC/SI
   as the known-null. This document governs `φ` only; it does not license skipping them.

## 5. ⭐ The exact sentence that will appear in the paper

> **The read-in `φ` is fit once, on unlabelled data from the first task's classes only, and
> is then frozen for the entire stream; it never sees data from any later task, is never
> trained through the memory store, and is never fit on a stored item. As a declared upper
> bound we also report a `generic_frozen` `φ` fit on a class-balanced pool spanning all
> stream classes — this arm deliberately leaks future tasks and is never quoted as a
> headline result.**

Methods-section companion sentence (for the store-side clauses):

> **The store is designed, not learned: stored addresses are `φ(x)` written as fixed
> Gaussian wells whose common width is set by a fixed rule from the nearest-neighbour
> spacing of the keys already stored, so no store hyper-parameter is ever a function of
> unseen data.**

## 6. The decision this pre-registration feeds (registered before measurement)

From `PREREG.md` P4, registered in advance: with `gap_end_of_stream` = the end-of-stream
accuracy gap `generic_frozen − task1_only`,

- **< 0.10** ⇒ `task1_only` is **viable as the primary arm**; build the w25 entry on it.
- **0.10 – 0.20** ⇒ viable **with a declared caveat** quantifying the cost of strictness.
- **> 0.20** ⇒ the strict `φ` **cannot represent later classes**; the entry must either build
  online `φ` first, or re-declare itself as a *pretrained-feature-extractor* entry (the
  L2P/DualPrompt precedent) rather than a from-scratch one — a weight-class change that the
  Head must approve.

## 7. ⭐ MEASURED OUTCOME (added 2026-07-24, after the harness ran — §1–6 are unchanged)

Split-MNIST, 5 tasks × 2 classes, 32 items/task ⇒ **M=160**, 3 seeds, `s_policy="refit"`.
`gap_end_of_stream` = `generic_frozen − task1_only`, identity accuracy, CLU-in-φ:

| φ arm | gap (identity) | gap (class) | verdict vs §6 rule |
|---|---|---|---|
| PCA-32 | **−0.023** | −0.004 | **< 0.10 ⇒ VIABLE** |
| AE-32 | **+0.006** | +0.006 | **< 0.10 ⇒ VIABLE** |

⇒ **`task1_only` is RATIFIED as the primary regime for the w25 entry.** The strict φ costs
essentially nothing at `phi_dim=32`; it is *better* on task-1 classes (gap −0.115: it spends
its whole budget on its own distribution, which is where the store is densest) and mildly
worse on the last task (+0.062 PCA / +0.125 AE) — a **tilt, not a growing deficit**.

**⚠ NEW BINDING CLAUSE — the cost is bought off by feature dimension, so `phi_dim` is now a
protocol parameter, not a free hyper-parameter.** A `phi_dim` sweep (3 seeds) shows the gap
is monotone in how tight the feature budget is:

| `phi_dim` | gap PCA | gap AE | kNN off ceiling? |
|---|---|---|---|
| 4 | **+0.117** | **+0.190** | yes (0.42–0.80) |
| 8 | +0.029 | +0.027 | yes (≈0.885) |
| 16 | +0.025 | +0.023 | nearly (0.95–1.00) |
| 32 | −0.023 | +0.006 | at ceiling (1.000) |

The representational deficit is **real at every `phi_dim`** — a task-1-only PCA basis
captures **10–18 pp less** of later-task pixel variance than the generic basis (and 4–16 pp
*more* of task-1 variance) — but at `phi_dim ≥ 16` it does not convert into retrieval loss,
because identity retrieval needs *distinctness*, not variance capture. **The w25 entry must
therefore report `phi_dim` alongside every strict-φ number and must not run the primary arm
below `phi_dim = 16`.** Below that the leakage-free arm pays a real double-digit penalty and
the entry would be quietly trading defensibility for accuracy.

## 8. Standing scope caveat (Head, binding — do not re-litigate)

Masked/static retrieval is a task where **equalling a simple baseline is our best case**,
because CLU *approximates* the nearest-neighbour method that wins it. Every retrieval number
produced under this protocol is **diagnostic of `φ`'s stream discipline**, not a competitive
claim. A tie with kNN is not a win, and masked recall is **permanently appendix-only**.
