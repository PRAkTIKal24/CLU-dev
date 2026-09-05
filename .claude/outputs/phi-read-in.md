# phi-read-in — experiment-engineer report

**Task + acceptance:** build the learned read-in `φ` around a DESIGNED key–value store
(phase-doctrine flagship); re-fight the w22 Hopfield/U-Hop protocol in **φ-space** with 4
lines (CLU-in-φ · kNN-in-φ · closed-form Hopfield-in-φ · raw-space CLU), two φ arms
(frozen PCA + separately-trained AE, never through the store), the **mandatory laundering
control**, and the Item-4 retry note — MNIST + CIFAR-10, pre-registered, tests green.

**Status: done.** All four items delivered on both datasets and both φ arms. New tests
green; full suite green after fixing one real config-serialization bug my change exposed.

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). 3 sites.**
> 1. **THE HEADLINE, and it is a decision-grade NEGATIVE about the store:** with a real
>    `φ`, **kNN-in-φ (the trivial feature baseline) beats or ties CLU-in-φ on EVERY axis,
>    EVERY dataset, BOTH arms.** The laundering control **FIRES on all 4 (dataset×arm)
>    cells** — CLU-in-φ *never* beats kNN-in-φ with the designed store. **In the task's
>    exact words: the win is φ's, not ours.** This is no longer excusable by a missing
>    embedding (task's final ⚠). CM-22(e)-class: do NOT claim the designed store beats a
>    trivial feature baseline.
> 2. **BUT the phase doctrine's PREMISE is validated:** a `φ` fixes the "CLU on raw data"
>    losses **for everyone**. On CIFAR, raw-space CLU is at **chance (0.06)** and φ lifts
>    CLU to **0.81–0.97**; closed-form Hopfield-in-φ recovers from raw chance (0.01) too
>    (P-φ confirmed). And **CLU-in-φ decisively beats closed-form Hopfield-in-φ** at load
>    and under noise (CIFAR M=256: 0.973 vs 0.008 AE-softmax). The store adds nothing over
>    kNN, but φ + the settling dynamics is a real, large lift over both raw-CLU and
>    closed-form Hopfield. Frame accordingly.
> 3. **Item 4 — the retry hook SURVIVES φ, strongly.** Distance-to-nearest-well at settle
>    separates correct/incorrect first-pass reads in φ-space at **AUROC 0.845–0.988**
>    across all 4 cells (pre-registered ≥0.65). Gates the `retry-compute-study` thread's
>    feature-space extension: GREEN.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/phi-read-in`; base **local `main @ 7ff0651`**; 3 commits `bd52530, 5549182, 62faecf` |
| worktree | built in `../CHLU-phi-read-in` (§3.2 — 4 parallel engineer tasks share `chlu/`); branch ref verified from main repo, then worktree removed. **Main checkout was NOT touched** (it currently holds another agent's `controller-mvp` branch + untracked `results/`; no collision) |
| datasets | MNIST (openml `mnist_784`, cached), CIFAR-10 (HuggingFace `uoft-cs/cifar10` parquet, 10k test — same source as w22) |
| pixels / mask / noise | `[0,1]`; capacity query `torch.dropout(x,p=0.5)` (zero + ×2); noise query `clamp(\|x+N(0,σ)\|,0,1)` — **repo-verbatim, reused from w22** |
| **primary metric** | **mean `sqdiff` in PIXEL space on the returned payload** (w22-comparable); identity-retrieval accuracy (payload index == true index) reported as the legible number |
| store | key = `φ(pattern)` written as a Gaussian well (`GaussianMemoryPotential`); payload = raw `x`; read-out ψ = damped-Verlet settle → payload of nearest well |
| φ-A (`pca`) | frozen PCA-k (SVD of centered fit pool), **fit on a DISJOINT pool** (`n_fit_pool=3000`), never sees the store — FAIR |
| φ-B (`ae`) | 1-hidden-layer AE (D→256→k tanh / k→256→D), Optax Adam, **reconstruction MSE only**, `epochs=400 lr=1e-3 batch=512`, fit on the DISJOINT pool — never sees store/wells/retrieval loss — FAIR. recon-MSE: MNIST **0.0170**, CIFAR **0.0234** |
| feature dim | `phi_dim (d) = 32` both arms; well width `s = 0.3·median-NN(φ)` (one fixed rule, not per-load) |
| CLU register | `clu_b=1, clu_alpha=1e-3, clu_gamma=0.1, clu_steps=200`, `dt=0.5·s/√b` (auto), newtonian-identity, read = mean of last 10% of the damped Verlet rollout — **reused w22 x64-safe `_settle_read`** |
| Hopfield-in-φ | dense softmax + sparsemax, β=1, 1 step (repo-verbatim), decoded to nearest stored φ → payload |
| store pool / loads | `n_data_pool=1500`; capacity `M∈{16,32,64,128,256}`; noise at fixed load **128**; σ∈{0,0.2,0.4,0.6,0.8,1.0} |
| seed | 0 (single seed) |
| JAX | main venv reused (`/Users/user/Desktop/CHLU/.venv`, JAX 0.9.0), float32 (x64 OFF), CPU |
| designed vs learned | **φ is learned/fit OFF the CLU side (w20's law); the store is DESIGNED.** kNN-in-φ / Hopfield-in-φ share the same φ and queries (fair controls) |

---

## 1. Item 2 — capacity in φ-space (identity-retrieval accuracy, 50%-masked queries)

### 1.1 MNIST — capacity, seed 0
| M | **CLU-in-φ** | **kNN-in-φ** | Hopfield-softmax-φ | Hopfield-sparse-φ | raw-space CLU (w22) |
|---|---|---|---|---|---|
| φ-A PCA | | | | | |
| 16  | 1.000 | **1.000** | 1.000 | 1.000 | 0.938 |
| 32  | 0.969 | **1.000** | 1.000 | 1.000 | 0.812 |
| 64  | 0.969 | **1.000** | 0.984 | 0.984 | 0.719 |
| 128 | 0.891 | **1.000** | 0.984 | 0.984 | 0.438 |
| 256 | 0.871 | **0.996** | 0.957 | 0.965 | 0.324 |
| φ-B AE | | | | | |
| 16  | 1.000 | **1.000** | 1.000 | 1.000 | 0.938 |
| 32  | 1.000 | **1.000** | 1.000 | 1.000 | 0.812 |
| 64  | 0.969 | **1.000** | 0.984 | 0.969 | 0.719 |
| 128 | 0.891 | **1.000** | 0.969 | 0.914 | 0.438 |
| 256 | 0.910 | **0.996** | 0.922 | 0.887 | 0.324 |

### 1.2 CIFAR-10 — capacity, seed 0  ⭐ (the decisive dataset)
| M | **CLU-in-φ** | **kNN-in-φ** | Hopfield-softmax-φ | Hopfield-sparse-φ | raw-space CLU (w22) |
|---|---|---|---|---|---|
| φ-A PCA | | | | | |
| 16  | 1.000 | **1.000** | 1.000 | 1.000 | 0.062 |
| 32  | 0.875 | **1.000** | 0.875 | 0.875 | 0.031 |
| 64  | 0.906 | **1.000** | 0.562 | 0.562 | 0.016 |
| 128 | 0.953 | **1.000** | 0.414 | 0.414 | 0.023 |
| 256 | 0.809 | **1.000** | 0.379 | 0.379 | 0.012 |
| φ-B AE | | | | | |
| 16  | 0.875 | **1.000** | 0.688 | 0.812 | 0.062 |
| 32  | 1.000 | **1.000** | 0.438 | 0.688 | 0.031 |
| 64  | 1.000 | **1.000** | 0.156 | 0.328 | 0.016 |
| 128 | 0.984 | **1.000** | 0.039 | 0.234 | 0.023 |
| 256 | 0.973 | **0.984** | 0.008 | 0.180 | 0.012 |

**Reads.**
- **kNN-in-φ wins/ties every cell.** It is 1.000 down to M=128 (0.996/0.984 at M=256) on
  both datasets/arms. The trivial feature baseline is the ceiling here.
- ⭐ **φ lifts CLU off the floor.** Raw-space CLU (w22 line) is 0.32 (MNIST) / **chance
  0.012** (CIFAR) at M=256; CLU-in-φ is **0.87–0.91 (MNIST) / 0.81–0.97 (CIFAR)**. On
  CIFAR that is a lift from chance to near-perfect — the phase-doctrine premise made real.
- ⭐ **CLU-in-φ ≫ closed-form Hopfield-in-φ at load on CIFAR** (M=256: PCA 0.809 vs 0.379;
  AE 0.973 vs 0.008). The localized-basin attractor does not suffer the inner-product
  concentration that flattens the softmax read. **On MNIST they are comparable** (Hopfield
  slightly ahead at high M: 0.957 vs 0.871 PCA) — so the CLU>Hopfield margin is a CIFAR /
  hard-feature-geometry effect, not universal (my prereg "CLU>Hopfield everywhere" was too
  broad — see scorecard P-hop).
- **P-φ confirmed:** a `φ` rescues closed-form Hopfield from the w22 raw-CIFAR chance
  collapse (0.012 → 0.38–1.0 at low M). But in AE space softmax Hopfield **re-collapses**
  at high M (0.008 at M=256) — AE features re-concentrate; sparsemax is more robust (0.180).

## 2. Item 2 — noise robustness in φ-space (fixed load 128)

### 2.1 MNIST
| σ | CLU-in-φ (PCA / AE) | kNN-in-φ (PCA / AE) | Hopfield-soft-φ (PCA / AE) | raw CLU |
|---|---|---|---|---|
| 0.0 | 0.891 / 0.898 | **1.000 / 1.000** | 0.984 / 0.961 | 0.898 |
| 0.2 | 0.898 / 0.914 | **1.000 / 1.000** | 0.914 / 0.922 | 0.906 |
| 0.4 | 0.852 / 0.773 | **0.938 / 0.898** | 0.250 / 0.531 | 0.008 |
| 0.6 | 0.180 / 0.203 | **0.352 / 0.320** | 0.062 / 0.102 | 0.008 |
| 0.8 | 0.039 / 0.031 | **0.109 / 0.102** | 0.055 / 0.031 | 0.008 |
| 1.0 | 0.023 / 0.039 | 0.062 / 0.047 | 0.023 / 0.023 | 0.008 |

### 2.2 CIFAR-10
| σ | CLU-in-φ (PCA / AE) | kNN-in-φ (PCA / AE) | Hopfield-soft-φ (PCA / AE) | raw CLU |
|---|---|---|---|---|
| 0.0 | 0.953 / 1.000 | **1.000 / 1.000** | 0.422 / 0.023 | 0.836 |
| 0.2 | 0.945 / 0.992 | **1.000 / 1.000** | 0.391 / 0.016 | 0.539 |
| 0.4 | 0.547 / 0.664 | **0.711 / 0.719** | 0.320 / 0.008 | 0.008 |
| 0.6 | 0.141 / 0.117 | 0.148 / 0.125 | 0.008 / 0.008 | 0.008 |
| ≥0.8| ≤0.062 | ≤0.070 | 0.008 | 0.008 |

**Reads.**
- **kNN-in-φ dominates the noise axis too** (laundering holds on noise). CLU-in-φ tracks
  just below it and **converges to it at high σ** (both fail together — the query has left
  the basin).
- ⭐ **CLU-in-φ ≫ closed-form Hopfield-in-φ** at moderate noise (MNIST σ=0.4: 0.852 vs
  0.250; CIFAR σ=0 AE: 1.000 vs 0.023). This mirrors the w22 "CLU beats one-step Hopfield
  on noise" result, now in φ-space — but it is a win over the *other clever method*, not
  over the trivial kNN.
- **raw-space CLU collapses at σ≥0.4** (the w22 narrow-well cliff); **φ-space CLU holds to
  σ≈0.4** (0.85 MNIST / 0.55–0.66 CIFAR). Another large φ-lift.

## 3. Item 3 — THE LAUNDERING CONTROL (pre-registered, mandatory)

Same `φ`, trivial store swap (kNN-in-φ). Verdict per cell = does CLU-in-φ *ever* beat
kNN-in-φ (identity-acc, tie band 0.03) on the capacity axis?

| dataset | arm | n loads | CLU wins | kNN wins | ties | max CLU margin | **verdict** |
|---|---|---|---|---|---|---|---|
| MNIST | PCA | 5 | **0** | 3 | 2 | 0.000 | **LAUNDERED** |
| MNIST | AE  | 5 | **0** | 3 | 2 | 0.000 | **LAUNDERED** |
| CIFAR | PCA | 5 | **0** | 4 | 1 | 0.000 | **LAUNDERED** |
| CIFAR | AE  | 5 | **0** | 1 | 4 | 0.000 | **LAUNDERED** |

**Stated in the task's required words:** *the win is φ's, not ours.* Across every dataset
and both φ arms, **CLU-in-φ never beats kNN-in-φ with the designed store** — it ties at
low load and loses at high load. **A CLU margin that exists ONLY with the designed store —
the result the program needs — does NOT appear on this benchmark.** Per the task's final
⚠, reported plainly: this is a decision-grade negative about the store, no longer
attributable to a missing embedding.

**What the store/dynamics DO add (the honest positive):** everything CLU-in-φ gains over
(a) raw-space CLU and (b) closed-form Hopfield-in-φ. Those are large (chance→0.97 on
CIFAR vs raw; +0.6–0.97 vs Hopfield at load) — but they are gains of *the settling
dynamics in feature space*, not gains over the trivial nearest-neighbour read of the same
features.

## 4. Item 4 — does the retry hook survive `φ`? (brief; full study = `retry-compute-study`)

Confidence = distance-to-nearest-well at settle. AUROC of (−distance) vs first-pass
correctness, φ-space, load 128, masked queries:

| dataset | arm | confidence AUROC | mean dist (correct / incorrect) |
|---|---|---|---|
| MNIST | PCA | **0.975** | 0.110 / 0.957 |
| MNIST | AE  | **0.974** | 0.056 / 0.396 |
| CIFAR | PCA | **0.988** | 0.257 / 1.734 |
| CIFAR | AE  | **0.845** | 0.013 / 0.040 |

**The retry trigger survives `φ` strongly** (all ≥0.845, pre-registered ≥0.65). Incorrect
reads settle 1.6–8.7× farther from the nearest well than correct reads. **GREEN for the
retry thread's feature-space extension.**

## 5. Packing-law occupancy (Item 1, matrix v2.1 §1) — `d`, well width, Δ_req

> ⛔ **CORRECTION (w24, `headroom-retry-benchmark` §3 + `lattice-capacity-theory`): the `spacing/Δ_req ≡ 1.08` figure below is a UNIT ARTIFACT and is RETIRED.** It was computed with a per-element `σ_q` against a vector median-NN, which pins the ratio at `1/(3.1·s_frac)` for *every* store regardless of the actual geometry. The corrected slack (w23 iid protocol, matched cell) is **≈0.227** — i.e. the store already ran **past** the packing bound, not marginally adequate at it. Do not quote 1.08 as a measured store property (the §7.19 "never quote 2.6" precedent). The qualitative reading below ("no slack; capacity decays faster than kNN at high load") is *strengthened*, not weakened, by the correction.

| dataset | arm | d | well width `w=s` | median NN spacing | Δ_req≈3.1·w | spacing/Δ_req |
|---|---|---|---|---|---|---|
| MNIST | PCA | 32 | 1.786 | 5.953 | 5.536 | **1.08** |
| MNIST | AE  | 32 | 0.856 | 2.853 | 2.654 | **1.08** |
| CIFAR | PCA | 32 | 3.050 | — | — | **1.08** |
| CIFAR | AE  | 32 | 0.410 | — | — | **1.08** |

Note: `spacing/Δ_req ≡ 1.08` by construction — the fixed rule `s=0.3·median-NN` puts the
store **exactly at the packing bound** (`1/(3.1·0.3)=1.075`). The occupancy is marginally
adequate (≥1), which is why CLU-in-φ settles to the right well most of the time; but there
is **no slack**, which is consistent with CLU-in-φ decaying at high load faster than kNN
(the basins begin to overlap as M grows and median-NN shrinks). A larger `s_frac` slack
would trade capacity for noise, exactly as w22's `Δ_req` analysis predicts.

## 6. PREREG scorecard (`PREREG.md` written before any harness ran)

| # | registered | measured | verdict |
|---|---|---|---|
| CLU-in-φ vs kNN capacity | TIE (laundering fires) | TIE at low M, **LOSE** at high M; laundering FIRES all 4 cells | ◐ laundering right; kNN is ≥, not =, so slightly optimistic |
| CLU-in-φ vs Hopfield capacity | WIN (both arms) | **CIFAR WIN (huge); MNIST ~tie/slight-LOSE at high M** | ◐ right on CIFAR, wrong on MNIST-high-M |
| CLU-in-φ vs raw-space CLU | WIN | **WIN, large** (chance→0.97 on CIFAR) | ✅ |
| noise vs kNN | TIE/LOSE | **LOSE** (kNN ≥ everywhere) | ✅ direction |
| noise vs Hopfield | TIE/WIN | **WIN**, large | ✅ |
| P-φ (Hopfield recovers on CIFAR) | above chance | YES (0.01→0.38–1.0); AE-softmax re-collapses at high M | ✅ (with a caveat) |
| P-lift (CLU-in-φ ≫ raw) | >0.75 at M=128, one arm | 0.89–0.98 at M=128 all arms | ✅ |
| P-arm (AE ≥ PCA at high load) | small AE edge ≤0.05 | MNIST +0.039, CIFAR +0.164 at M=256 | ✅ (CIFAR edge larger) |
| P-Item4 (confidence AUROC ≥0.65) | survives | **0.845–0.988** | ✅ |
| P-laundering (kNN within band, both arms) | CLU ties kNN | LAUNDERED all 4 (kNN ties-or-beats) | ✅ core claim |

**Honest summary:** the central optimistic hypothesis (a designed store beats the trivial
feature baseline) is **falsified** — laundered on all 4 cells. The survivors are a
validated *premise* (φ fixes the raw-data losses for everyone; P-lift, P-φ) and two
robust findings (**CLU-in-φ ≫ closed-form Hopfield-in-φ** at load/noise; **retry survives
φ** at AUROC 0.85–0.99).

## How I verified

- New tests `tests/test_phi_read_in.py` (9): φ store/fit disjointness, PCA & AE encoders,
  kNN/Hopfield-in-φ, `_auroc` (perfect + degenerate), laundering-control verdict (both
  branches), four-line capacity sweep, retry probe. **9 passed** (14.8s warm).
- **Full suite:** first run **592 passed / 1 failed** — the failure (`test_config.py::
  test_every_group_round_trips_mutated`) was a **real bug my change exposed**: `save_config`
  manually enumerates groups and I had not added `experiment_phi_read_in`, so a mutated
  config silently reverted to defaults on round-trip. **Fixed** (one line in `save_config`,
  folded into the config commit); `pytest tests/test_config.py` → **7 passed**; combined
  `test_phi_read_in + test_config` → **16 passed**. The only full-suite failure is resolved.
- **MNIST run** (both arms, full sweep) and **CIFAR-10 run** (both arms) completed clean;
  metrics saved to `.claude/outputs/phi-read-in/{mnist,cifar10}_metrics.json` + 8 figures.
  All numbers above re-derived from those JSONs (seed 0).
- `ruff check` clean on all four touched files. CLI wired: `chlu exp-phi-read-in
  [--project N] [--seed I] [--quick] [--dataset …] [--arms pca,ae]`; module runnable as
  `python -m chlu.experiments.exp_phi_read_in --quick`.

## Git footprint

- **Branch** `agent/experiment-engineer/phi-read-in`, base **local `main @ 7ff0651`**
  (verified merge-base). Built in worktree `../CHLU-phi-read-in`; branch ref confirmed from
  the main repo before the worktree was removed (§3.2). **Not pushed, no PR.**
- **Commits (3):** `bd52530` (ExperimentPhiReadInConfig + register + **save_config fix**),
  `5549182` (exp_phi_read_in.py), `62faecf` (CLI hook + tests). +971 lines, 4 files.
- **Files:** **+** `chlu/experiments/exp_phi_read_in.py`, `tests/test_phi_read_in.py`;
  **M** `chlu/config.py` (additive dataclass + load/save reg — no existing default changed),
  `chlu/cli/experiment_cmd.py` (parser + handler). No shared files reformatted beyond my
  hunks; `utils/plotting.py` untouched (figures local, w22 precedent); `results/` not
  committed.
- **No collision.** The shared main checkout currently holds another agent's
  `controller-mvp` branch (+ its untracked `results/`); I worked entirely in my own
  worktree and never edited the shared checkout. Rebased onto local `main` = no-op.

## Open questions / follow-ups / risks

- **Single seed (0), single φ family per arm, d=32.** kNN's dominance is large-margin; the
  CLU-vs-Hopfield gaps are large; the laundering verdict is unambiguous (max CLU margin
  0.000 everywhere). Digits/dim unswept.
- **The store's value is not visible on a static-recall benchmark.** kNN-in-φ is a perfect
  static content-addressable read of the same features; the store's *designed* capabilities
  (retry, per-item retention, deletion, sequential-write locality) are exactly the axes a
  masked-recall capacity/noise sweep cannot score. **If the program wants a designed-store
  win, it must be sought on a capability axis (continual-learning / retention / retry), not
  on this static φ-recall benchmark** — consistent with the w22 conclusion and the HG1
  (continual-learning) target the Head named for Phase 2.
- **`s_frac=0.3` sits exactly at the packing bound (slack ≡1.08).** A per-dataset `s_frac`
  Pareto (capacity↔noise) in φ-space is the clean follow-up but is per-axis tuning; I kept
  the single fixed rule for a fair curve.
- **AE-softmax Hopfield re-collapses at high M** in AE space — a learned manifold can
  re-concentrate inner products the way raw CIFAR did. Worth a note wherever "φ fixes
  Hopfield" is claimed: it fixes it for *localized* reads (CLU, sparsemax), not universally
  for softmax.

## Proposed handover updates (for the Hub)

1. **⭐ §6 / CM-22 / CM-23 — new decision-grade result (the doctrine flagship's verdict):**
   with a real learned read-in `φ` (frozen PCA **and** a separately-trained AE, both fit on
   a disjoint pool — never through the store), the DESIGNED key–value store **does not beat
   the trivial kNN-in-φ baseline on any axis, either dataset, either arm — the laundering
   control FIRES on all 4 cells (max CLU margin 0.000).** *"The win is φ's, not ours."*
   Forbid: "a designed CLU store beats a trivial feature baseline on associative recall."
   This closes the "it only loses because there's no embedding" escape hatch — the embedding
   now exists and the store still does not win the static benchmark.
2. **⭐ §6 — the phase-doctrine PREMISE is validated (the approved-wording positive):** `φ`
   fixes the "CLU on raw data" losses for everyone. Raw-space CLU is at **chance (0.06)** on
   CIFAR; **CLU-in-φ = 0.81–0.97**. Closed-form Hopfield-in-φ recovers from raw chance too
   (P-φ). And **CLU-in-φ ≫ closed-form Hopfield-in-φ** at load and noise (CIFAR M=256:
   0.973 vs 0.008). These are gains of the settling dynamics *in feature space*, not over
   kNN. Approved framing: *"a learned read-in makes the CLU register competitive with
   closed-form modern-Hopfield in feature space and far above its own raw-pixel line, but a
   trivial kNN in the same feature space is the ceiling on static masked recall."*
3. **§6 — the retry hook survives `φ`** (AUROC 0.845–0.988, confidence = dist-to-well).
   Unblocks the feature-space extension of `retry-compute-study`.
4. **§7 / config — bug found and fixed:** `save_config` manually enumerates groups; any new
   config group must be added there too or it silently reverts on round-trip (caught by
   `test_config::test_every_group_round_trips_mutated`). Fixed for `experiment_phi_read_in`.
   Flag for the other 3 concurrent engineer tasks adding config groups this wave.
5. **New CLI/config surface:** `chlu exp-phi-read-in`, `ExperimentPhiReadInConfig`, module
   `exp_phi_read_in`. Reuses w22's x64-safe rollout + scorer; no new core class.
6. **Direction implication (for the Head/Phase-2):** the designed store's advantage is a
   *capability*, not a static-recall number. This benchmark structurally cannot show it.
   The HG1 continual-learning target (retention/interference/retry) is where the designed
   store can beat something the feature map alone cannot — recommend Phase-2 scoping there,
   with kNN-in-φ carried as the mandatory laundering control on every claim.
