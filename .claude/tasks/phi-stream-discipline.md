# Task: phi-stream-discipline — what may `φ` see in a continual stream? (resolve + pre-register) (w24)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/phi-stream-discipline.md` · **Branch:** `agent/experiment-engineer/phi-stream-discipline`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (**§3.2 worktrees mandatory**) · `.claude/outputs/phi-read-in.md` (the φ you are re-scoping; its store/φ separation and the laundering control) · `.claude/outputs/continual-learning-recon.md` (the CL protocols, the van de Ven taxonomy, the mandatory baseline table) · `.claude/negative_results.md` **N89** (tier A — the laundering negative)
- **This task de-risks w25's CL entry. It is a blocker: the CL entry cannot be built until this is settled and pre-registered.**

## Why
w23's `φ` was fit on a **disjoint pool that saw all ten classes**. In `phi-read-in` that was fair — the task was static recall. In **Class-IL it is data leakage**: `φ` must not be trained on data from tasks the model has not yet reached. A referee kills the entry in one line otherwise. The advisor flagged this; the Head has ruled the arms.

## ⭐ HEAD RULING (binding — build to this)
- **PRIMARY = task-1-only `φ`** (fit on task 1's classes, then frozen for the whole stream). This is the defensible arm and the one every headline number comes from.
- **REFERENCE = generic-frozen `φ`** (the w23-style pool, or generic pretrained features), carried as a **declared upper bound** — clearly labelled, **never quoted as the headline**.
- **ONLINE `φ` = later**, as its own experiment; whether it enters the CL results is a separate decision. Leave a clean extension point, do not build it now.

## Item 1 — implement the φ-stream regimes
Three regimes behind one flag: `task1_only` (primary) · `generic_frozen` (declared upper bound) · `online` (stub + interface only, not run). The store stays **designed** and `φ` is never trained through it (the w20 law, unchanged).

## Item 2 — ⭐ the deliverable: the cost-of-strictness curve
The number that decides whether the CL entry is viable: **how much does task-1-only `φ` lose to generic-frozen `φ`, and how does that gap grow as tasks accumulate?** Report identity-retrieval / downstream accuracy per task index over a Split-MNIST-shaped stream (5 tasks × 2 classes), both regimes, same store, same queries. A gap that widens steeply means `φ` from task 1 cannot represent later classes and the entry needs online `φ` — better to learn that now than in w25.

## Item 3 — the pre-registration for w25
Write the CL entry's `φ` protocol as a **pre-registration document** (`.claude/outputs/phi-stream-discipline/PREREG_CL_PHI.md`): which regime is primary, what is frozen when, what each arm may see, and the exact sentence that will appear in the paper describing φ's training data. This is what protects the entry from the leakage attack.

## Item 4 — carry the laundering control forward
**kNN-in-φ is mandatory on every claim** (N89, CM-22(i)). Report it in both regimes. If task-1-only `φ` makes kNN-in-φ *worse* while CLU-in-φ holds up, that is the first evidence of a store advantage — pre-register it as a watch-item, do not go hunting for it.

## Acceptance
The three-regime implementation; the cost-of-strictness curve with the laundering control in both regimes; the `PREREG_CL_PHI.md` protocol document. PREREG the main predictions before running. Tests green; `ruff` clean; config registered at **all three sites plus `save_config`**.

## ⚠ Standing scope caveat (Head, binding — do not re-litigate)
**Masked/static retrieval is a task where equalling a simple baseline is our best case, because CLU *approximates* the nearest-neighbour method that wins it.** Any retrieval number in this task is **diagnostic of φ's stream discipline**, not a competitive claim. Do not frame a tie with kNN as a win, and do not propose a lead claim on this axis (**Head ruling: masked recall is permanently appendix-only**).
