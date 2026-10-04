# cl-entry-build — experiment-engineer report

**Task + acceptance criterion:** build THE CL entry — rehearsal-free Class-IL (Split-MNIST →
Split-CIFAR-10) on a designed CLU store + task-1-only φ + MVC-0 controller, with the mandatory
baseline table, the R3-native retry measurement, and the R1 scheduled-retention demonstration;
PREREG first, ≥3 seeds, laundering control on every claim, tests green, config at all sites.
**Status: done** — all five items ran on both datasets at ≥3 seeds and are reported; six known
limitations are stated in §8, and four ⛔/⭐ results need an owner (list immediately below).

**Findings in one line:** *the entry is the best rehearsal-free method on Split-MNIST Class-IL
(0.707 vs 0.196, +0.51) while sitting below replay (−0.15 vs iCaRL) and being **laundered** by
kNN-in-φ at matched memory (−0.036); it is a **null on Split-CIFAR-10** (0.149, below LwF) because
PCA-32-of-pixels is not an address space; the store's real edge is **forgetting** (BWT −0.169 vs
−0.99 for the parametric rehearsal-free class, and better than ER at the same item budget with 24.5×
fewer floats); the retry dial **ties** the honest floor in its native regime while beating every
mechanism control; and the retention dial reproduces `exp(−leak·t)` to 7e-8 with permanent items at
1.000 through the whole stream.*

> **⭐ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Four items.**
> 1. ⭐ **The entry WINS the rehearsal-free class on Split-MNIST by +0.51 ACC (0.707 vs 0.196), 3 seeds,
>    and sits −0.15 below iCaRL** — the CM-23(n) filing case. ⚠ **Whether this counts as "an external
>    benchmark won" for the program's score sentence is a Hub/Head call, not mine**: the class it
>    beats is the **known null** (EWC/SI/LwF fail Class-IL ≈ chance *by construction*, and our
>    harness reproduces their published numbers to ≤0.5 pp), and the laundering control fires
>    (item 2). I have not softened the score sentence anywhere; I am handing over the measurement.
> 2. ⛔ **The kNN-in-φ laundering control FIRES on the entry (CLU 0.707 vs 0.742 launder, −0.036).**
>    The store does not beat a class-balanced ring buffer of the same φ keys. Fourth consecutive
>    wave of the same pattern. **The entry's ACC number is φ's and the buffer's, not the settle's.**
> 3. ⛔ **Split-CIFAR-10 is a NULL for the entry: CLU 0.149 ± 0.013, BELOW every rehearsal-free
>    baseline (LwF 0.162).** The strict-φ cost is ~0 (0.001) — the failure is the *feature space*,
>    not the stream discipline: kNN over the same φ only reaches 0.21. **The entry's scope is
>    MNIST-class data until a CIFAR-capable φ exists.**
> 4. ⭐ **R3-native (the regime where no oracle can be constructed) is the first cell in the
>    program's history where the anytime read is NOT below the honest floor: gated retry
>    0.885 ± 0.018 vs matched-compute feedforward-in-φ 0.877 ± 0.025 — a TIE (+0.8 ± 1.6 pp,
>    6 seeds, 3/6 positive), and +1.1 ± 1.4 pp over 1-shot kNN-in-φ.** (w24: −3.5 … −38 pp in 8/8
>    cells.) The *mechanism* margin is large and clean: **+4.6 ± 2.2 pp over random-kick, 6/6
>    seeds**, ensemble +4.4, ungated collapses to 0.000, auto-stop at 1.40 ± 0.20× compute.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dials:** all four. Admission/isolation = the anti-forgetting mechanism (Item 1/2);
  lifetimes = scheduled retention on the live stream (Item 4); compute-adaptive reads = retry in
  its native regime (Item 3).
- **Laundering control:** kNN-in-φ at matched memory on every claim — run in **two** forms (same
  keys ⇒ isolates the settle; independent class-balanced ring buffer ⇒ isolates the admission gate).
- **Falsifies:** losing the replay-free class, or being matched by the launder; a flat retry curve
  or a lift matched by kick/ensemble.
- **Does NOT falsify:** sitting below replay/GDumb/iCaRL (CM-23(n)); any oracle comparison.
- **Outcome against the declaration:** the *replay-free class* is won on MNIST and **lost on
  CIFAR-10**; the *launder* fires on both (the falsifier fires); the *retry mechanism* survives
  every control on both.

---

## Flag-provenance table (governs every number in this report)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/cl-entry-build`, base local `main @ 63c668d` |
| commits (9) | `b6aa1f5` store addr_dim · `1848691` labelled CIFAR + splits · `67b949b` baselines · `50e70b9` experiment · `61d4fc7` config+CLI · `d25ffce` tests · `e7f9e73` verdict/φ-floor/eviction fixes · `a95e213` τ-sweep level · `8267c89` regression tests for both fixes |
| worktree | `../CHLU-cl-entry` (protocol §3.2; 3 parallel engineer tasks this wave). **Main venv reused** via `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python` — no worktree `uv sync`, so no JAX drift |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** (main venv, protocol §4) |
| headline runs | MNIST: `results/exp_cl_entry_mnist_metrics.json` @ `e7f9e73`, seeds **0,1,2** · retry replication seeds **3,4,5** (`--items entry,retry --baselines none`) @ `e7f9e73` · CIFAR: `results/exp_cl_entry_cifar10_metrics.json` @ `e7f9e73`, seeds **0,1,2** |
| scenario | **Class-IL**, van de Ven & Tolias: 5 tasks × 2 classes, task identity **NOT** given at test time; read-out = arg-max over classes **seen so far** |
| φ | `phi_arm=pca`, **`phi_dim=32` (≥16, PREREG_CL_PHI §7 binding)**, `task1_only` = PRIMARY (fit on {0,1} images from a pool **disjoint** from every stream item, frozen at end of task 1, never refit), `generic_frozen` = declared upper bound (reported, never a headline) |
| store | designed `AtomStorePotential(addr_dim=32)` + MVC-0 `Controller(allow_relocation=False)`; read landscape = `GaussianMemoryPotential(centers, s, amps)`; `clu_s_frac=0.2`, `d_safe_mult=4.4` (⇒ `d_safe = 0.88 × median-NN`, the **sizing rule** made self-consistent), `s_policy="refit"`, `clu_b=1.0`, `clu_alpha=1e-3`, `clu_gamma=0.1`, `clu_steps=150`, `dt=0.5·s/√b` (auto), `clu_tail_frac=0.1`, **`newtonian_identity`**, `rollout_chunk=256` |
| memory | **200 items** for the store, ER, iCaRL, GDumb and both kNN-in-φ launders (matched **items**; float counts reported alongside: store 6 400 vs raw buffer 156 800 on MNIST / 614 400 on CIFAR) |
| baselines | shared MLP 784-400-400-10 (MNIST) / 3-layer CNN 16-32-32 + 128 (CIFAR), Adam `lr=1e-3`, `baseline_iters=500` (MNIST) / `150` (CIFAR) per task, `batch=128`, `fisher_samples=200`; **N78 tuning**: `ewc_lambda ∈ {100,1000,10000} → 100`, `si_c ∈ {0.1,1,10} → 0.1`, `lwf_alpha ∈ {0.5,1,2} → 2` (swept on seed 0, then fixed for all seeds) |
| stream sizing | MNIST 2 000 train / 1 000 test per task; CIFAR **reduced protocol** (`apply_cifar10`): 1 000 train / 500 test per task, 150 iters — stated, never quoted as literature-comparable Split-CIFAR-10 |
| retry | `retry_ladder=[0,1,2,4,8]`, `retry_boost=1.5`, `retry_step_frac=0.1`, `retry_tau=0.99` (+ sweep {0.99,0.999,0.9999,1.0}), corruption = **pixel-space** dropout `p ∈ {0.5, 0.8}` on stored items, embedded through the frozen φ |
| retention | `ticks_per_task=4` (20 ticks/stream), `permanent_per_task=10`, `leak_slow=0.0866` (t½=8 ticks), `leak_fast=0.3466` (t½=2), `amp_floor=0.05` |
| PREREG | `.claude/outputs/cl-entry-build/PREREG.md`, written **before** any harness ran |
| nothing learned on the stream | the CLU entry runs **zero gradient steps** on the stream; φ is fit once, unsupervised, off-stream (w20 law: φ is never trained through the store) |

---

## What I did

1. **Generalised the designed store to a φ-dimensional address block** (`AtomStorePotential.addr_dim`,
   per-well `amps` on `GaussianMemoryPotential`, an address-dimension-aware `Controller` with
   `allow_relocation=False`, a public `evict_item` verb and `live_amps`) — all additive and
   default-preserving, so w23/w24 objects are bit-identical.
2. **Built `exp_cl_entry.py`**: the Class-IL stream with the `PREREG_CL_PHI` fairness regions, the
   `PhiStore` (MVC-0 controller + class-balanced budget over φ addresses, payload = label), the
   Class-IL read-out (settle → landed well → its label), the two laundering lines, the R3-native
   retry harness with the RUD-C controls, and the scheduled-retention cohorts.
3. **Built `cl_baselines.py`**: finetune/EWC/SI/LwF/ER/iCaRL/GDumb/joint on a shared backbone,
   optimizer, iteration budget, memory budget and evaluation, with the N78 tuning pass.
4. **Added labelled CIFAR-10 + train/test splits** to `load_labeled_images`, wired
   `ExperimentClEntryConfig` at all three sites + `save_config`, added `chlu exp-cl-entry`, and
   wrote 15 tests.
5. **Ran it**: Split-MNIST 3 seeds (+3 replication seeds on the retry cell, +1 τ probe) and
   Split-CIFAR-10 3 seeds, after writing `PREREG.md`.

---

## Item 1 + 2 — the entry and the mandatory baseline table (Split-MNIST, 3 seeds)

| method | class | **ACC** | BWT | forgetting | mem items | mem floats |
|---|---|---|---|---|---|---|
| joint (offline) | upper bound | 0.946 ± 0.002 | −0.025 | 0.025 | — | — |
| **iCaRL** | replay | **0.859 ± 0.002** | −0.105 | 0.105 | 200 | 156 800 |
| **GDumb** (matched memory) | replay | **0.795 ± 0.012** | −0.093 | 0.093 | 200 | 156 800 |
| **ER** (tuned) | replay | **0.775 ± 0.018** | −0.264 | 0.264 | 200 | 156 800 |
| kNN-in-φ ring buffer, `generic_frozen` | ⚠ launder (upper bound φ) | 0.768 ± 0.008 | −0.126 | 0.126 | 200 | 6 400 |
| kNN-in-φ same keys, `generic_frozen` | ⚠ launder (upper bound φ) | 0.757 ± 0.007 | −0.131 | 0.131 | 200 | 6 400 |
| **kNN-in-φ ring buffer, `task1_only`** | ⛔ **the launder** | **0.742 ± 0.010** | −0.152 | 0.152 | 200 | 6 400 |
| CLU entry, `generic_frozen` φ | *declared upper bound* | 0.734 ± 0.005 | −0.151 | 0.151 | 200 | 6 400 |
| **kNN-in-φ same keys, `task1_only`** | ⛔ **the launder** | **0.728 ± 0.008** | −0.154 | 0.154 | 199 | 6 368 |
| ⭐ **CLU entry, `task1_only` φ (PRIMARY)** | rehearsal-free | **0.707 ± 0.012** | **−0.169** | 0.169 | 199 | 6 368 |
| EWC | rehearsal-free (known null) | 0.196 ± 0.000 | −0.993 | 0.993 | — | — |
| LwF | rehearsal-free (known null) | 0.196 ± 0.000 | −0.994 | 0.994 | — | — |
| finetune | rehearsal-free (the null) | 0.196 ± 0.000 | −0.994 | 0.994 | — | — |
| SI | rehearsal-free (known null) | 0.195 ± 0.001 | −0.991 | 0.991 | — | — |

**The three sentences the entry is allowed to say (computed, not asserted):**
1. ⭐ **`wins_rehearsal_free_class = True`, margin +0.510** over the best published rehearsal-free
   method (EWC 0.196). ⚠ **The margin is against the known null** — van de Ven & Tolias show
   regularisation methods collapse to ≈chance in Class-IL *by construction*; our harness
   reproduces 19.5–19.6 % against their published 19.9–20.0 %, which is the calibration check,
   **not** a CLU achievement. The correct sentence is *"the entry is the best rehearsal-free
   method on this benchmark, in a class whose incumbents are known to fail it."*
2. **`deficit_vs_replay = −0.153`** (vs iCaRL 0.859); below GDumb (−0.088) and ER (−0.068) too.
   **"Beats replay" is not claimed and is not true.** Per CM-23(n) this is a publishable success.
3. ⛔ **`laundered = True`, `clu_minus_launder = −0.036`.** kNN over the *same* φ keys scores
   0.728 and a class-balanced ring buffer of φ keys scores 0.742, both above the store's 0.707.
   **The win is φ's and the buffer's, not the store's** (N89/CM-22(i), required wording).

**Where the store *is* ahead, and it is not the ACC column.** Against the parametric rehearsal-free
methods the entry's **BWT is −0.169 vs −0.99**: it retains ~83 % of what it knew, they retain ~1 %.
Against **ER** — the strongest matched-memory replay method here — the entry's forgetting is
**0.169 vs 0.264**, i.e. the designed store forgets *less* than a replay buffer at the same item
budget while using **24.5× fewer floats**, and loses only on absolute accuracy. Forgetting/BWT, not
ACC, is the axis where the mechanism shows.

**Strict-φ cost (PREREG_CL_PHI §6 rule):** `generic_frozen − task1_only = +0.0275` ⇒ **< 0.10 ⇒ the
strict, leakage-free arm is viable as the primary arm**, confirming the w24 ratification at
`phi_dim=32` on a real Class-IL benchmark rather than on retrieval.

### Controller / sizing-rule accounting (per task, seed 0; seeds 1–2 identical to ±3 %)

| task | offered | admitted | refused (spacing gate) | admitted frac | s | d_safe | live |
|---|---|---|---|---|---|---|---|
| 0 | 2 000 | 1 244 | 756 | 0.622 | 0.487 | 2.141 | 200 |
| 1 | 2 000 | 1 906 | 94 | 0.953 | 0.626 | 2.755 | 200 |
| 2 | 2 000 | 1 398 | 602 | 0.699 | 0.785 | 3.453 | 200 |
| 3 | 2 000 | 1 142 | 858 | 0.571 | 0.808 | 3.557 | 199 |
| 4 | 2 000 | 1 161 | 839 | 0.581 | 0.858 | 3.774 | 199 |

- The store is **budget-bound, not gate-bound**: `n_live` saturates at the 200-item budget from
  task 0 on, so the *sizing rule* here is enforced by the **width** knob rather than by growing the
  address space (in φ-space the address space is where the data is; you cannot grow it). Setting
  `d_safe_mult · clu_s_frac = 0.88 ≤ 1` makes the spacing gate self-consistent with the width rule
  — the gate refuses 5–45 % of offers per task instead of refusing everything (the failure mode the
  controller-mvp handover warned about, N91).
- **Per-admitted is not reported as an accuracy number here** (deliberately): in Class-IL the test
  set is fixed and independent of what was admitted, so per-offered/per-admitted has no meaning for
  ACC. What *is* reported is the admitted fraction and the intervention rate, per task, as above.
- ⭐ **Corrected packing slack** (w24 A1 unit fix — σ_q is a displacement **norm**):
  **0.326–0.340** (median-NN 4.12–4.30, s 0.83–0.86, σ_q 4.08). The CL store runs **~3× past the
  packing bound**, which is *intrinsic* to classification (a query is a different image, not a
  corrupted copy of a stored one) and is exactly the geometric ambiguity Item 3 measures. Never
  quote the retracted "1.08".

### Cite-and-differentiate (required by Item 2; not re-derivable from our numbers, so stated as scope)

| prior work | what it owns | what this build does differently — **as measured here** |
|---|---|---|
| **SQHN** (Nature Comms 2024) | energy-based Hopfield-class store, online single-pass, replay-free via **discrete sparse-quantized one-hot codes + parameter isolation** | our store is a **continuous designed landscape** with (i) a payload channel separate from the address, (ii) **per-item decay schedules** (Item 4: exact `exp(−leak·t)`, permanent-vs-leaky cohorts in one store), (iii) a **relaxation + retry read** with a measured accuracy-vs-compute curve (Item 3), (iv) a **learned frozen φ** read-in. ⛔ We do **not** claim "energy store for replay-free CL" as new — SQHN owns it. |
| **PALL** (ICLR 2025) | **exact task unlearning** inside a CL architecture (task-specific sparse subnetworks + episodic rehearsal) — it occupies the CL ∩ forgetting cell | ours is **per-item, not per-task**, and it is a **lifetime**, not a deletion: a schedule set at write time, measured as an amplitude law with two causes of removal separated (schedule vs budget). ⛔ We make **no** deletion/unlearning claim (CM-22 m/n/o), and PALL rehearses — we do not. |
| **iCaRL** (CVPR 2017) | the fair, refereed episodic-memory slot: herding exemplars + NME | measured here at 0.859 (MNIST) / 0.419 (CIFAR) — **above us**. Our difference is the memory *object*: 200 × 32 floats of designed landscape vs 200 raw exemplars (24.5×/96× more floats), with better BWT than ER at the same item budget. |
| **GDumb** (ECCV 2020) | the pathology check — a dumb balanced buffer + retrain | run at matched memory (0.795 / 0.301). We are below it on MNIST by 0.088. Reported, not hidden. |

---

## Item 3 ⭐ — the R3-native measurement: retry where **no oracle exists**

**Why no oracle exists here (state this in any write-up).** The corruption is applied in **pixel
space** (dropout `p` on the stored image) and the store is addressed in **φ**. There is therefore no
"known-erased" coordinate subset *of the store's space* to hand a baseline, so the `masked-NN`
oracle that sat at 1.000 and beat CLU in all 8 w24 cells **cannot be constructed**. This is
precisely the "space the store is not metric-native to" that `headroom-retry-benchmark` named as
the only untested route. The honest floor is **kNN-in-φ** (1-shot) and its matched-compute version
**feedforward-in-φ** (k+1 augmented votes) — which is also the laundering control.

⚠ **A control had to be rebuilt.** `exp_retry_compute._feedforward_ladder` augments queries with
`clip(|q + noise|, 0, 1)` — correct for pixels in [0,1], **destructive** for a signed zero-mean φ
vector (the line collapsed to ~0.005 with k, i.e. it was not a floor at all). `_feedforward_ladder_phi`
uses additive noise scaled to the store's own median address spacing, no clipping. The pixel-TTA
line is kept in the JSON under an explicitly `INVALID_in_phi` name so the w24 comparison stays
auditable. **All floor numbers below are from the corrected control.**

### Headline cell: Split-MNIST, end-of-stream, crowded store (199–200 wells), corruption p=0.8

| seed | first pass | **gated best** | @compute | kick best | ensemble best | ungated @k=8 | kNN-in-φ (1-shot) | ff-in-φ (matched compute) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.814 | 0.864 | 1.80× | 0.814 | 0.814 | 0.000 | 0.859 | 0.864 |
| 1 | 0.850 | 0.865 | 1.20× | 0.850 | 0.850 | 0.000 | 0.855 | 0.860 |
| 2 | 0.845 | 0.885 | 1.40× | 0.845 | 0.845 | 0.000 | 0.880 | 0.890 |
| 3 | 0.830 | 0.905 | 1.40× | 0.830 | 0.830 | 0.000 | 0.895 | 0.895 |
| 4 | 0.884 | 0.910 | 1.40× | 0.884 | 0.889 | 0.000 | 0.915 | 0.915 |
| 5 | 0.810 | 0.880 | 1.20× | 0.810 | 0.815 | 0.000 | 0.840 | 0.840 |
| **mean ± sd (6 seeds)** | **0.839 ± 0.025** | **0.885 ± 0.018** | **1.40 ± 0.20×** | 0.839 ± 0.025 | 0.841 ± 0.026 | **0.000** | 0.874 ± 0.025 | 0.877 ± 0.025 |

- ⭐ **Mechanism (the dial claim): SURVIVES, cleanly.** gated − kick = **+4.6 ± 2.2 pp, 6/6 seeds**
  (the equal-energy random kick **never once** improves on the first pass); gated − ensemble =
  **+4.4 ± 2.2 pp**; **ungated-all collapses to 0.000** in 6/6 (the gate is load-bearing); the
  gated curve is monotone and **auto-stops at 1.2–1.8× compute** — it never spends the 9× the
  ladder offers.
- **Leaderboard vs the honest floor: a TIE, and that is the news.** gated − ff-in-φ =
  **+0.8 ± 1.6 pp (3/6 seeds positive)**; gated − 1-shot kNN-in-φ = **+1.1 ± 1.4 pp (5/6)**.
  Neither margin is separated from zero. ⚠ **PREREG discipline:** I registered that a CLU win here
  must be re-run before belief; seeds 3,4,5 are that re-run, and they turn an apparent
  +0.7 pp win (seeds 0–2) into a **tie**. Reported as a tie.
  **This is still the first regime in the program's history where the anytime read is not *below*
  the correct floor** (w24: −3.5 … −38 pp in 8/8 cells).
- **Confidence signal:** cosine-to-nearest-well AUROC **0.65–0.85** (informative); distance-to-well
  AUROC 0.51–0.60 (near-useless in φ). The gate ranks on cosine, which is the right choice here.
- **Mid-stream (store snapshot after task 3), same corruption, 3 seeds:** first 0.828 ± 0.012 →
  gated **0.858 ± 0.018 @ 1.30 ± 0.14×**; kick 0.828, ensemble 0.832, ungated 0.000; kNN-in-φ
  0.848, ff-in-φ 0.853 ⇒ gated − kNN **+0.010 ± 0.007**. **The picture does not change between
  mid-stream and end-of-stream**, i.e. the result is not an end-of-run artifact.

### Per task-age (the crowded-store question), MNIST p=0.8, 3 seeds

| stage | task-age 0 (oldest) | 1 | 2 | 3 | 4 (newest) |
|---|---|---|---|---|---|
| end-of-stream, first pass | 0.846 ± 0.073 | 0.858 | 0.858 | 0.825 | **0.796 ± 0.069** |
| end-of-stream, gated best | 0.906 @1.34× | 0.908 @1.53× | 0.883 @1.10× | 0.900 @1.40× | 0.820 @1.29× |
| lift | +0.060 | +0.050 | +0.025 | +0.075 | +0.025 |
| mid-stream (after task 3), first pass | 0.798 | 0.833 | 0.853 | — | — |
| mid-stream, gated best | 0.843 @1.64× | 0.879 @1.28× | 0.887 @1.14× | — | — |

⭐ **There is no forgetting-by-age in the store**: the spread across task ages is **6.2 pp** (≤10 pp
as pre-registered) and the trend is **opposite** to the registered direction — the *newest* task's
items are the hardest to recall (0.796), because they were admitted last into an already-crowded
store, while task 0's items (0.846) have had every later item pass the spacing gate against them.
Retry gives the largest lift exactly where the first pass is worst among the *old* items (+6.0 pp at
age 0 end-of-stream, +4.5 pp at age 0 mid-stream).

At **p=0.5 the cell is saturated** (first pass 0.99, floor 1.000) and every line is flat — reported
for completeness, no information.

### The τ-sweep (re-run at the hardest level after `a95e213`, seed 3)

In the three reported MNIST runs the τ-sweep executed on the *first* corruption level (p=0.5,
saturated) and therefore measured nothing; `a95e213` moves it to the hardest level and a dedicated
seed-3 probe (`--items entry,retry --baselines none`) supplies it:

| τ | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| 0.99 / 0.999 / 0.9999 / 1.0 (**all identical**) | 0.830 @1.0× | 0.870 @1.1× | 0.880 @1.2× | **0.905 @1.4×** | 0.900 @1.8× |

⭐ **τ is not the binding element of the gate in φ-space.** After a damped settle in a crowded
φ store essentially every read sits below cosine 0.99, so all four thresholds select the same
eligible pool; what produces the curve is the **ranking + the 10 %-per-round budget + the lock**.
The curve still auto-stops — accuracy peaks at k=4 (1.4×) and *falls back* at k=8 (1.8×) — but the
stop comes from exhausting the low-confidence tail, not from τ. ⚠ This is a **behavioural
difference from w24**, where τ=1.0 demonstrably over-retried and cost accuracy; the gate's
threshold clause should not be quoted as load-bearing in the φ-space regime.

---

## Item 4 ⭐ — scheduled per-item retention on the live stream (the R1 survivor)

⛔ **Naming (CM-22 m/n/o), enforced by a test:** this is **scheduled per-item retention / scheduled
forgetting**. It is **not** "certified", **not** unlearning, **not** deletion by construction, **not**
exact deletion, and **not** a privacy claim. It is a capability demonstration inside a benchmarked
system.

Three cohorts written into the **running entry** (same stream, same store, same evaluation), 20
decay ticks over the 5-task stream, `amp_floor = 0.05`:

| cohort | leak | half-life | measured amplitude vs `exp(−leak·t)` | retrieval retention (live items) | evicted **by schedule** | evicted by budget |
|---|---|---|---|---|---|---|
| **permanent** (`leak=0`) | 0.0 | ∞ | **exactly 1.000 at all 20 ticks** (max err **0.0**) | **1.00** throughout the stream | **0** | **0** |
| slow | 0.0866 | 8 ticks | 0.917 → 0.501, max err **6.8e-8** | 0.95–1.00 | **0** | 3 248 |
| fast | 0.3466 | 2 ticks | 0.707 → 0.477, max err **1.2e-8** | 0.90–1.00 | **71** (66 / 67 on seeds 1 / 2) | 3 244 |

- ⭐ **The scheduled law holds to numerical precision** (per-item `exp(−leak·t)`, max abs error
  ≤ 6.8e-8 over every snapshot), and **permanent items ride through all five tasks at retention
  1.000** while items around them decay and are removed.
- ⭐ **Only the fast cohort is ever removed by the schedule** (71/66/67 items across seeds), exactly
  as the arithmetic requires: `ln(1/0.05)/leak` = **8.6 ticks** (fast, inside the 20-tick stream)
  vs **34.6 ticks** (slow, outside it). Zero permanent items were removed by *either* cause.
- ⭐ **The two causes of forgetting are separated and never conflated** (new accounting in
  `e7f9e73`): `evicted_by_schedule` (the dial) vs `evicted_by_budget` (the capacity policy). In a
  200-item store fed 10 000 offers, **budget pressure removes 6 492 items and the schedule removes
  71 (≈91×)** — the honest framing is *"the schedule is a per-item control operating inside a
  capacity-bound store"*, not *"the store forgets on schedule"*.
- **The retrieval consequence is a threshold, not a graded fade** (pre-registered as T4): while a
  well is alive, cohort-mean retrieval retention stays ≈1.0 whatever its amplitude; the item stops
  being retrievable when the well crosses `amp_floor` and is removed. The *amplitude* decays
  smoothly; the *read* does not degrade until the well is gone.
- **The price of the dial:** end-of-stream ACC with decay ON = **0.701 / 0.679 / 0.708** (seeds
  0/1/2) vs **0.707 ± 0.012** with decay OFF ⇒ **the retention schedule costs ≈ 0.01 ACC** here
  (much less than the 0.05–0.25 I pre-registered), because budget eviction was already removing
  most items anyway.
- **Permanence costs budget forever:** permanent items are exempt from eviction, so `permanent_per_task`
  is a hard design knob (10/task ⇒ 48 of 200 slots by end of stream). With no cap, the store fills
  with permanent items and correctly starts refusing writes (`refused_full` capacity alarm — 0 in
  these runs because of the cap).

---

## Item 5 — Split-CIFAR-10: the de-risk the φ report demanded

⚠ **Reduced protocol** (`apply_cifar10`, stated in-code and here): small 3-layer CNN (16-32-32 +
128), 1 000 stream items and 150 optimizer steps per task, identical for every method. These are
**not** literature-comparable Split-CIFAR-10 numbers; they are an internally-matched comparison.

| method | class | **ACC** | BWT | forgetting | mem items | mem floats |
|---|---|---|---|---|---|---|
| joint (offline) | upper bound | 0.480 ± 0.005 | −0.113 | 0.155 | — | — |
| **iCaRL** | replay | **0.419 ± 0.003** | −0.199 | 0.202 | 200 | 614 400 |
| **ER** | replay | **0.369 ± 0.006** | −0.441 | 0.441 | 200 | 614 400 |
| **GDumb** | replay | **0.301 ± 0.023** | −0.217 | 0.227 | 200 | 614 400 |
| kNN-in-φ ring buffer, `generic_frozen` | ⚠ launder (UB φ) | 0.222 ± 0.014 | −0.231 | 0.231 | 200 | 6 400 |
| **kNN-in-φ ring buffer, `task1_only`** | ⛔ **the launder** | **0.219 ± 0.014** | −0.235 | 0.235 | 200 | 6 400 |
| kNN-in-φ same keys, `generic_frozen` | ⚠ launder (UB φ) | 0.210 ± 0.016 | −0.195 | 0.195 | 200 | 6 400 |
| **kNN-in-φ same keys, `task1_only`** | ⛔ **the launder** | **0.207 ± 0.014** | −0.198 | 0.198 | 200 | 6 400 |
| **LwF** | rehearsal-free (known null) | **0.162 ± 0.001** | −0.783 | 0.783 | — | — |
| EWC | rehearsal-free (known null) | 0.154 ± 0.004 | −0.696 | 0.696 | — | — |
| SI | rehearsal-free (known null) | 0.153 ± 0.005 | −0.739 | 0.739 | — | — |
| finetune | rehearsal-free (the null) | 0.152 ± 0.010 | −0.774 | 0.774 | — | — |
| CLU entry, `generic_frozen` φ | *declared upper bound* | 0.150 ± 0.021 | −0.150 | 0.155 | 200 | 6 400 |
| ⛔ **CLU entry, `task1_only` φ (PRIMARY)** | rehearsal-free | **0.149 ± 0.013** | **−0.157** | 0.160 | 200 | 6 400 |

⛔ **This is a NULL, and it is the entry's scope boundary.** `wins_rehearsal_free_class = False`
(−0.013 vs LwF); `deficit_vs_replay = −0.270`; `laundered = True` (−0.070). Per the pre-registered
F4 clause, this is reported as **the headline finding for the entry's scope**, not buried.

**The failure is the address space, not the stream discipline — three lines of evidence:**
1. **`strict_phi_cost = +0.0011`.** The leakage-free task-1-only φ costs *nothing* relative to the
   leaky all-classes φ (0.149 vs 0.150). Whatever is broken is broken in both. ⚠ This **falsifies my
   pre-registered F1 band (+0.02…+0.15)** and, more importantly, refutes the phi-stream report's
   expectation that *"CIFAR is where a strict φ should bite"* — **strictness does not bite; the
   feature space does.**
2. **kNN over the same φ keys only reaches 0.207–0.222.** A 200-prototype nearest-neighbour
   classifier on PCA-32-of-raw-CIFAR-pixels is a ~0.22 classifier. **No store built on those
   addresses can be competitive**, because the addresses do not separate the classes.
3. **The store's own retrieval is *perfect* on the same addresses** (Item 3 on CIFAR: stored-item
   recall 1.000 at p=0.5, 0.863 at p=0.8). The memory works; what it holds is uninformative about
   the label.

**The mechanism still shows, in the same place as on MNIST:** CLU BWT **−0.157** vs the parametric
rehearsal-free methods' **−0.70…−0.78** — a 4.5–5× smaller backward transfer loss, at 0.149 absolute.
The store does not forget; it never knew.

**Geometry (3 seeds):** median-NN address spacing 9.24–9.78, s 1.87–1.93, σ_q 8.85–9.14 ⇒ corrected
**packing slack 0.337–0.345** — essentially the same crowding as MNIST (0.33), so crowding is not
the differentiator either. Admitted fraction per task 0.46–0.79, store budget-bound at 200.

**Item 3 on CIFAR (for completeness):** the mechanism controls behave exactly as on MNIST — gated
0.863 → **0.945 @1.11×** at p=0.8 (+8.2 pp), kick **flat at 0.863** (6/6 cells), ensemble 0.867,
ungated collapses to 0.083 — but the **kNN-in-φ floor is 1.000** here, so the anytime read cannot
win (gated − kNN = **−0.055 ± 0.019** end-of-stream, −0.032 ± 0.009 mid-stream). Same structural
situation as w24: a saturated floor is unbeatable, which is why the MNIST p=0.8 cell (floor 0.874,
off ceiling) is the one that carries the R3-native result.

**Item 4 on CIFAR:** identical qualitative result — permanent amplitude exactly 1.000 (max err 0.0)
and retention 1.00 throughout; slow/fast follow `exp(−leak·t)` to ≤6.8e-8; **only the fast cohort is
schedule-evicted (82/75/76 items), zero slow, zero permanent**. Decay-ON ACC 0.126–0.146 vs 0.149.

---

## PREREG scorecard (`PREREG.md`, written before any harness ran)

| # | registered | measured | verdict |
|---|---|---|---|
| C1 | CLU beats every rehearsal-free baseline by > +0.40 (MNIST) | **+0.510** (0.707 vs EWC 0.196) | ✅ |
| C2 | CLU sits below replay, deficit −0.01…−0.10 | **−0.153** (vs iCaRL) | ◐ direction right, deficit **larger** than registered |
| C3 | the launder ties or beats CLU; CLU−kNN ∈ [−0.05,+0.01] | **−0.021** (same keys) / **−0.036** (ring buffer) | ✅ **laundering fires, as pre-registered** |
| C4 | CLU BWT ∈ [−0.20,−0.02]; EWC/SI ≤ −0.85 | CLU **−0.169**; EWC −0.993, SI −0.991, LwF −0.994 | ✅ |
| C5 | admitted-per-offered ≈0.02 (budget-bound); intervention > 0.90 by task 5 | admitted 0.57–0.97 **per offer at the gate**; store budget-bound at 200/10 000 offers = 0.02 net | ◐ ambiguous registration (gate-level vs net); both numbers reported |
| C6 | corrected packing slack < 1 (0.2–0.8) | **0.326–0.340** | ✅ |
| R1 | first-pass ∈ [0.55,0.90] (headroom present) | p=0.8: **0.839 ± 0.025** ✅; p=0.5: 0.99 (saturated) | ✅ at the level that matters |
| R2 | gated lift +2…+12 pp, monotone, auto-stop ≤2.0× | **+4.6 ± 2.2 pp @ 1.40 ± 0.20×** | ✅ |
| R3 | kick/ensemble flat-or-declining, gap ≥ +1 pp; ungated declines at k=8 | kick/ens **never above first pass** (6/6); ungated **0.000** | ✅ |
| R4 | kNN floor within ±5 pp and predicted ≥ CLU-gated in ≥half the cells | gated − ff-in-φ **+0.8 ± 1.6 pp (3/6)**; gated − kNN +1.1 ± 1.4 (5/6) | ◐ **a tie, not the registered loss** — and not a win either |
| R5 | task-age spread ≤10 pp, no monotone trend; if any, older worse | spread **6.2 pp**; trend present but **newest is worst** | ◐ ✅ on the bound, **direction falsified** |
| T1 | amplitude = `exp(−leak·t)` to <1e-6 rel | max abs err **6.8e-8** | ✅ |
| T2 | permanent retrieval retention = 1.000 at every tick | **1.000**, all 20 ticks, all seeds | ✅ |
| T3 | fast cohort self-evicts at tick 9 ± 1; slow never | **only fast evicts** (71/66/67); slow 0; arithmetic 8.6 vs 34.6 ticks | ✅ |
| T4 | retrieval = a step at eviction, not a smooth decay | confirmed (retention ≈1.0 while alive) | ✅ |
| T5 | retention arm costs 0.05–0.25 ACC | **≈0.01** (0.701/0.679/0.708 vs 0.707) | ❌ **over-predicted the cost** |
| F1 | CIFAR strict-φ cost +0.02…+0.15 | **≈+0.001** | ❌ (see §5 — the whole feature space fails, so strictness costs nothing) |
| F2 | CIFAR CLU ACC 0.22…0.38 | **0.149 ± 0.013** | ❌ **below the band** |
| F3 | CIFAR EWC/SI/finetune ≈0.19 ± 0.03; ER/iCaRL 0.30…0.55 | **0.152–0.162** (just below the band); ER 0.369, iCaRL 0.419, GDumb 0.301 ✅ | ◐ replay arm ✅, the null sits ~0.04 lower than registered (weak CNN ⇒ lower Class-IL null) |
| F4 | CIFAR: CLU still beats the rehearsal-free class by > +0.05, else headline finding | **CLU 0.149 < LwF 0.162 ⇒ the registered headline FINDING fires** | ❌ → reported as the finding |

**Score: 9 ✅ / 6 ◐ / 4 ❌.** The four failures are the informative ones: the CIFAR collapse (F1/F2/F4),
and the over-predicted retention cost (T5).

---

## How I verified

- `PYTHONPATH=<worktree> .venv/bin/pytest tests/test_cl_entry.py -q -p no:randomly --no-cov` →
  **15 passed** (462.91 s). Re-ran the three retention/naming/end-to-end tests after the
  eviction-accounting change → **3 passed** (734.44 s, machine under load); the two regression
  tests added in `8267c89` → **2 passed** (10.26 s). Test count is now **17**.
- `pytest tests/test_config.py` → **7 passed** (389.14 s) — this includes
  `test_every_group_round_trips_mutated`, the guard that a new config group is registered at all
  three sites **plus** `save_config` (the w23 trap).
- **Full suite:** `pytest tests/ -q -p no:randomly --no-cov` → **672 passed, 0 failed**
  (1 161.58 s, run concurrently with the CIFAR job). That is exactly the handover's 657 + 15 of the new
  tests (the last 2 were added after the suite run and pass individually; expect **674** at
  integration).
- `ruff check chlu/` → **All checks passed** (after each change).
- CLI: `chlu exp-cl-entry --help` parses (`--project/--seed/--quick/--dataset/--items/--baselines`).
- Smoke: `python -m chlu.experiments.exp_cl_entry --quick` → exit 0 (all 8 baselines + retry +
  retention on a tiny stream).
- Real runs: MNIST 3 seeds (≈75 min alone), retry replication seeds 3–5 (≈5 min each), CIFAR 3
  seeds (≈95 min). Every number above is re-derived from the shipped JSON by
  `.claude/outputs/cl-entry-build/render.py`.
- ⚠ **A first full MNIST run (superseded) exposed three real bugs** — the verdict counted our own
  `generic_frozen` arm as a "baseline", the feedforward control was pixel-space TTA applied to φ
  vectors, and eviction cause was not attributed. All three are fixed in `e7f9e73` and the reported
  runs are from the fixed code.

---

## §8 — Known limitations of this build (stated, not hidden)

1. **τ-sweep landed on the saturated corruption level** in the three reported MNIST runs; `a95e213`
   moves it to the hardest level and a dedicated seed-3 probe supplies the missing measurement
   (Item 3) — but that measurement is **one seed**, and it says τ is *not* binding in φ-space,
   which is a behavioural difference from w24 that deserves its own multi-seed check.
2. **Baselines are reimplementations, not Mammoth/Avalanche runs.** They are internally matched
   (same backbone/optimizer/iterations/memory/evaluation) and **three of the four known nulls
   reproduce the published Split-MNIST Class-IL values to ≤0.5 pp** — EWC 19.6 vs 20.0, SI 19.5 vs
   20.0, finetune 19.6 vs 19.9 — which is the calibration evidence that the Class-IL protocol is
   implemented correctly. iCaRL at 0.859 vs the published 0.946 reflects our 200-exemplar budget
   and reduced training, not a different method. ⚠ **LwF is the one baseline we do NOT reproduce**
   (19.6 vs the published 23.9, i.e. 4.3 pp low) — it should be re-tuned before this table is
   published; it does not affect any conclusion here (LwF is not the best rehearsal-free baseline
   on MNIST either way, and on CIFAR it is, where our CLU entry loses to it).
3. **SI uses an end-of-task displacement estimate** of the path integral, not the online
   accumulation; documented in-code. Its Class-IL collapse is the published behaviour either way.
4. **The CIFAR arm is a reduced protocol** (§5) and its φ is PCA-32 on raw pixels — the finding is
   about *that* φ, and a CNN/SSL φ is the obvious next test.
5. **The store's read decodes by nearest live address after settling** (the w23/w24 protocol).
   That final arg-min is classical indexing (N89 discipline) — which is precisely why both
   laundering lines are mandatory and reported.
6. **Retention snapshots measure live items only.** Items removed by budget pressure leave the
   cohort, so the cohort-mean retention curve cannot show the eviction step directly; the
   `evicted_by_schedule` counters (new) carry that information instead.

---

## Git footprint

- **Branch** `agent/experiment-engineer/cl-entry-build`, base local `main @ 63c668d`
  (⚠ `main` had moved 3 commits past the handover's `5ad04f6` — `1a8fa92`/`cee1f87`/`63c668d`,
  authored by Pratik Jawahar, which add three **cloud-sync duplicate files**
  `chlu/core/controller 2.py`, `chlu/experiments/exp_controller_mvp 2.py`,
  `tests/test_controller_mvp 2.py`. I did **not** touch them; see the handover note below).
- **9 commits** (all `[experiment-engineer]`-tagged, atomic):
  `b6aa1f5` · `1848691` · `67b949b` · `50e70b9` · `61d4fc7` · `d25ffce` · `e7f9e73` · `a95e213` · `8267c89`.
- **Files — 3 new, 5 surgical:** NEW `chlu/experiments/exp_cl_entry.py` (1 190 lines),
  `chlu/experiments/cl_baselines.py` (395), `tests/test_cl_entry.py` (17 tests). Modified
  `chlu/core/memory_potentials.py` (`addr_dim` on `AtomStorePotential`, optional `amps` on
  `GaussianMemoryPotential` — **both default-preserving**), `chlu/core/controller.py`
  (`addr_dim`/`allow_relocation`/`evict_item`/`live_amps`, all additive),
  `chlu/experiments/exp_phi_stream.py` (`load_labeled_images` split + CIFAR-10 labels — w24
  callers unaffected), `chlu/config.py` (+1 dataclass at all three sites),
  `chlu/cli/experiment_cmd.py` (+parser +`cmd_exp_cl_entry`).
- **Isolation:** dedicated worktree `/Users/user/Desktop/CHLU-cl-entry`, main venv reused
  (`PYTHONPATH` + `.venv/bin/python`, no worktree `uv sync`). ✅ **Verified from the MAIN repo**
  (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/cl-entry-build`)
  that the branch ref carries **all 9 commits** — the w4 lost-commit failure mode is checked, not
  assumed. `git rebase main` = *up to date* (no-op; base unmoved). A second engineer worktree
  (`CHLU-shard`, `lattice-sharded-store`) is live in parallel — no shared files were edited beyond
  the additive `config.py` / `experiment_cmd.py` registrations, which will conflict textually at
  the same insertion points and must be resolved **additively** (the w23/w24 precedent).
- **Worktree left in place** (it holds the run outputs under `results/`, untracked, per repo
  precedent); the Hub removes it at integration. Canonical copies of every JSON/figure quoted here
  are in `.claude/outputs/cl-entry-build/results/`.
- **No push, no PR.** Branch left for review.

---

## Open questions / follow-ups / risks

1. ⭐ **A CIFAR-capable φ is now the entry's single blocking dependency.** kNN over PCA-32 CIFAR
   features tops out at 0.21, so *no* store on those addresses can be competitive. The cheap next
   test is the existing `ae` arm at `phi_dim ∈ {32,64}`; the real answer is a small conv/SSL φ fit
   on task-1 classes only. Until then the entry's scope is MNIST-class data — say so.
2. **The launder fires on ACC but not on forgetting.** kNN-in-φ over a ring buffer beats the store
   on ACC (+0.036) — but the store's BWT/forgetting is better than ER's at the same item budget and
   24.5× fewer floats. If the program wants a *non-laundered* claim, forgetting-at-matched-bytes is
   the axis to formalise, not accuracy.
3. **The retry tie is one dataset, one corruption level, 6 seeds.** It is the best R3 result the
   program has (w24 lost by 3.5–38 pp) but it is a tie, and a second native regime (CIFAR at a
   working φ, or a cross-modal store) is needed before it is more than a lead.
4. **`permanent_per_task` is an unstudied knob** with an obvious capacity trade (permanent items
   never leave). A sweep would turn "the dial exists" into "the dial has a measured price curve".
5. **The mid-stream/end-of-stream retry pair shows no degradation with stream position** — the
   crowded-store hypothesis (older = harder) is *not* what the data says; it is the *newest*
   task that is hardest. That inverts an assumption in the R3 framing and should be checked before
   anyone writes "retrieval degrades as the store crowds with age".

---

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** *The CL entry exists and has run end-to-end (w25).*
   `chlu exp-cl-entry` (`exp_cl_entry.py` + `cl_baselines.py` + `ExperimentClEntryConfig`).
   **Split-MNIST Class-IL, 3 seeds, 200-item matched memory: CLU entry (task-1-only φ, phi_dim 32)
   = 0.707 ± 0.012 ACC / BWT −0.169; wins the rehearsal-free class by +0.510 over EWC (the known
   null); sits −0.153 below iCaRL; ⛔ the kNN-in-φ launder beats it by 0.036 ⇒ LAUNDERED.**
   **Split-CIFAR-10 (reduced protocol) = 0.149 ± 0.013 ⇒ BELOW the rehearsal-free class — a null,
   and the finding is that PCA-32-on-pixels is not an address space (kNN-in-φ only reaches 0.21).**
2. **Candidate N-entries (3).** (a) *"The designed store wins the rehearsal-free Class-IL class on
   Split-MNIST but is laundered by kNN-in-φ at matched memory"* — tier A, fourth consecutive
   laundering confirmation, and the first time the program has topped a *class* on an external
   benchmark's headline metric (with the two qualifiers of reconciliation item 1). (b) *"The CL entry
   collapses on Split-CIFAR-10 with a PCA φ; the deficit is the feature space, not the stream
   discipline"* — tier A, scope-defining. (c) *"In the R3-native regime (pixel-space corruption,
   φ-space store, no constructible oracle) gated retry TIES the matched-compute feedforward floor
   (+0.8 ± 1.6 pp, 6 seeds) while beating random-kick by +4.6 ± 2.2 pp"* — tier A, the first
   non-losing anytime-read result.
3. **Claims-matrix wording.** CM-23(n) is satisfied on MNIST **only**, and the win is over a class
   whose incumbents fail by construction — the two clauses must travel together. Nothing here
   licenses "beats replay" (deficit −0.153) or "beats a trivial feature baseline" (−0.036).
4. **§7 — new config/CLI surface:** `ExperimentClEntryConfig` (all three sites + `save_config`);
   `chlu exp-cl-entry`; `AtomStorePotential(addr_dim=…)`, `GaussianMemoryPotential(amps=…)`,
   `Controller(allow_relocation=…, evict_item, live_amps)`; `load_labeled_images(dataset, split)`
   now supports **cifar10 with labels** (`CHLU_CIFAR10_TARBALL` env override for worktrees, npz
   cache next to the tarball). All defaults preserve prior behaviour.
5. **⚠ Repo hygiene (not mine to fix):** local `main` is at `63c668d`, three commits past the
   handover's `5ad04f6`; those three commits **add** the cloud-sync duplicates
   `chlu/core/controller 2.py`, `chlu/experiments/exp_controller_mvp 2.py`,
   `tests/test_controller_mvp 2.py` (previously untracked, now tracked). `tests/test_controller_mvp 2.py`
   is collected by pytest as a second module and **double-counts 11 tests**, which will inflate the
   suite total the Hub compares against (612 → 657 → …). Someone with authority over `main` should
   remove them.
6. **Test count:** the branch adds **17** tests; the full suite was measured at **672 passed / 0
   failed** with 15 of them present, so integration should expect **674**.
7. **The w24 retry harness has a φ-space bug worth recording:** `_feedforward_ladder`'s
   `clip(|q+noise|,0,1)` TTA is only valid for pixel queries. Any future φ-space use of
   `exp_retry_compute` must use the corrected `_feedforward_ladder_phi` (in `exp_cl_entry.py`) or it
   will silently report a collapsing, meaningless "floor".
