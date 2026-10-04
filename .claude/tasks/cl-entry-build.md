# Task: cl-entry-build — the CL entry: one build, three results (R4 + R3-native + R1-survivor) (w25)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/cl-entry-build.md` · **Branch:** `agent/experiment-engineer/cl-entry-build`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (**§3.2 worktree mandatory — 3 parallel engineer tasks this wave** · **§7 dial declaration, NEW and binding**) · `.claude/outputs/continual-learning-recon.md` (the target, protocols, baseline table) · `.claude/outputs/phi-stream-discipline/PREREG_CL_PHI.md` (**the φ protocol you build to, verbatim**) · `.claude/outputs/controller-mvp.md` (the machinery + the sizing rule) · `.claude/outputs/headroom-retry-benchmark.md` §5/§6 (the retry harness + why no oracle exists here) · `.claude/outputs/unlearning-recon.md` (naming rules) · `claims_matrix.md` **v2.3 + v2.3-HR**

## ⭐ DIAL DECLARATION (protocol §7)
- **Dials:** ALL FOUR meet an external benchmark here — admission/isolation (the anti-forgetting mechanism), lifetimes (scheduled retention on the live stream), compute-adaptive reads (retry in its native regime). This is the flagship: **the one build where the dials meet a benchmark that cannot be won by a lookup.**
- **Laundering control:** **kNN-in-φ at matched memory on every claim** (N89 discipline) — if kNN over the same φ features with a raw ring-buffer matches the CLU entry, the win is φ's/the buffer's, not the store's.
- **Falsifies:** losing the **replay-free class** (to EWC/SI/LwF-class methods, or to the kNN-in-φ launder) falsifies the entry. A retry curve that is flat, or whose lift is matched by kick/ensemble, falsifies the R3-native measurement.
- **Does NOT falsify:** sitting below replay/GDumb/iCaRL (**the Head's filing rule, CM-23(n): winning replay-free while below replay IS a publishable success**); any comparison to an oracle or to methods outside the rehearsal-free class.

## Why (the Head's w24 assessment, verbatim direction)
*"Build the CL entry. Now. It is the wave."* The CL entry is simultaneously **R4** (the replay-free sweep), **the native headroom regime for R3** (crowded-store retrieval of past items mid-stream — no mask-oracle exists there), and **the honest home for R1's surviving form** (scheduled per-item retention on a real stream). Three of the five results bottleneck on this build. Every blocker is cleared: target picked (recon), φ ratified (`task1_only`, `PREREG_CL_PHI.md`), controller built (MVC-0), sizing rule extracted.

## Item 1 — the entry
**Rehearsal-free Class-IL, Split-MNIST first, Split-CIFAR-10 as the hard rung.** van de Ven three-scenario taxonomy; task id NOT given; report **ACC + forgetting/BWT** (GEM formulas, pinned in the recon §1.4).
- **Architecture:** designed store (`AtomStorePotential`/Gaussian wells over `φ(x)`; payload = label/exemplar-free representation) + **`task1_only` φ** per `PREREG_CL_PHI.md` (**`phi_dim ≥ 16`**, quote it everywhere) + **MVC-0 controller** (admission + placement + eviction/decay).
- **⭐ The sizing rule baked in from day one** (controller-mvp handover item 4): size the address space so the packing bound ≥ the per-task item load, or accept abstention and report per-admitted WITH the admitted fraction. Betting per-offered on an undersized space reproduces the GRU loss (N91).
- ⚠ **Class-clustered stores are the store-geometry risk** (phi-stream §8: task-0 well-overlap 0.62–0.74 — dense same-class items collapse median-NN spacing). The controller's spacing gate is the native mitigation; report intervention rates per task.
- ⚠ **Compute the packing slack CORRECTLY** — the 1.08 figure was a unit artifact (vector-norm σ_q vs vector median-NN; corrected 0.227). Use the corrected computation from `headroom-retry-benchmark`.

## Item 2 — the mandatory baseline table (a submission is invalid without it)
Tuned **ER** · **iCaRL** · **GDumb at matched memory** (the pathology check) · **EWC/SI** (the known-null — never presented as a CLU win) · the **kNN-in-φ ring-buffer launder** (same memory budget). Cite-and-differentiate: **SQHN** (continuous landscape / per-item decay / retry / learned φ) and **PALL** (per-item vs per-task; no lifetimes). Harness: build inside/against **Mammoth** numbers where possible (recon §1.4); if reimplementing baselines, state the tuning discipline (N78).

## Item 3 — ⭐ the R3-native internal measurement (retry in its home regime)
Mid-stream and end-of-stream, run the **retry ladder on crowded-store retrieval of PAST-task items** — the regime where degradation comes from **store geometry** (class-clustered wells, packing pressure), not from a query mask. **State in the report why no oracle exists here:** there is no erasure mask to hand a baseline; the ambiguity is in the landscape. Controls per RUD-C: kick, ensemble, ungated, matched-compute feedforward (the honest floor here is kNN-in-φ — already your laundering control). Deliverable: the accuracy-vs-compute curve per task-age, with the mechanism controls. **This is the R3 dial's native-regime test named by the Head ruling on N95.**

## Item 4 — ⭐ the R1-survivor internal measurement (scheduled retention on the live stream)
Demonstrate **"designed scheduled retention"** in the running entry: a subset of items written permanent (`leak=0`), a subset with scheduled half-lives; show retention-per-item follows `exp(−leak·t)` on the stream while permanent items ride through all tasks at 1.0. ⛔ **Naming rules (CM-22 m/n/o): never "certified", never "unlearning", never "deletion by construction," never "exact deletion"** — the words are *scheduled per-item retention / scheduled forgetting*. This is a capability demonstration inside a benchmarked system, not a privacy claim.

## Item 5 — Split-CIFAR-10 (the de-risk the φ report demanded)
Everything above is MNIST-validated only; **CIFAR is where a strict φ should bite** (phi-stream §8). Add the labels branch to `load_labeled_images`, run the entry end-to-end on Split-CIFAR-10, and report the strict-φ cost there — whatever it is. If the task1_only φ collapses on CIFAR, that is a headline finding for the entry's scope, reported plainly.

## Acceptance
PREREG **before running** (predicted ACC/forgetting per method incl. the launder; retry-curve predictions; retention-law predictions; the CIFAR strict-φ cost band). The full baseline table both datasets; the R3-native curves with controls; the R1 retention demo; ≥3 seeds on every headline; laundering control on every claim; tests green; config at all sites + `save_config`; `ruff` clean. Echo the DIAL DECLARATION at the top of your report.

## ⚠ Standing traps
- The score sentence is fixed: **external benchmarks won = ZERO until this lands** — do not soften it, and do not claim "beats replay" under any outcome.
- `git -C <worktree>` explicitly, always (the cwd can silently revert).
- The store never sees φ's training data; φ never sees the stream beyond task 1 (`PREREG_CL_PHI.md` §"what each arm may see" is binding).
