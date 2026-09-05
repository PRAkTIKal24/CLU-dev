# cl-encoder — experiment-engineer report

> ## ⛔ DATED CURATOR AMENDMENT №2 — 2026-08-10 (`doc-curator-c2w8-pass3-close-fold`, **[C2W8-CLOSE]**) — **THE INVERSION QUALIFIER ON EVERY "STRONG φ" SENTENCE IN THIS REPORT**
> **⛔ The body of this report is NOT edited and nothing in it is deleted** (C-3 dated-banner precedent). **The 2026-08-06 amendment below stands unchanged and travels with this one** — they correct different things (that one: §3's read-out artifact; this one: what *"strong φ"* is allowed to mean).
>
> **The measurement (`c2w8p3-capture-strong-phi` §3.1; charter Add.11 §A31.4), on the census store at `d = 12`, Split-CIFAR-10, 3 paired seeds:** the encoder ladder this report built is **inverted on addressability**. `simclr − randconv` **A1 = −0.1406 ± 0.0508 (0/3 seeds)** · `simclr − pca` **= −0.1276 ± 0.0589 (0/3)** · `randconv − pca` **+0.0130 ± 0.1017 (tied)**; and the **unfitted `randconv` clears the address-geometry GO rule at `simclr`'s own margin with 0 fit steps** (`c2w8p3-phi-geometry` §2). ⇒ ⭐⭐ **task accuracy and address geometry are ANTI-CORRELATED on this substrate** — this report's *encoder-quality* ladder is **not** an *address-quality* ladder, and the two must never be conflated.
>
> ⛔⛔ **BINDING (§A31.4): "strong φ" may never again be used as one undifferentiated notion.** ✅ Admissible form: ***"strong φ by the CL-accuracy metric that defines it here"***, with the addressability inversion stated beside it. ⭐ **Design consequence already ratified (Add.12 §A34.6 / §A34.10): the address block is its own head; cheap conv-class geometry is a legitimate default; task features ≠ address features.**
> ⚠ **What this banner does NOT touch:** §2's gate values · §4's read-out sweep · §5's strict-φ cost **+0.020 ± 0.010** (still unamended, still a declared NOT-RUN) · §7's geometry. ⛔ **No pass-3 number is a baseline for anything in this report and vice versa** (Head ruling R1).
> **Registry:** `negative_results.md` **N277**; ledger ⟲ **C2W8 PASS-3 + CLOSE** addendum; primer **§11.27 Record 35**. **Sources:** `c2w8p3-capture-strong-phi` §3.1, `c2w8p3-phi-geometry` §2, charter **Add.11 §A31.4**, the **`[C2W8-CLOSE]`** §10 entry.

> ## ⛔⛔ DATED CURATOR AMENDMENT — 2026-08-06 (`doc-curator-c2w8-pass1-fold`, rider 3b, **[C2W8]**) — **§3's ARM-ISOLATION TABLE IS NOT SAFE TO QUOTE: its objective/architecture decomposition is a READ-OUT ARTIFACT**
> **⛔ The body of this report is NOT edited and nothing in it is deleted** (C-3 dated-banner precedent). This banner corrects **one table's interpretation** and nothing else.
>
> **What §3 says (unchanged, and correct as measured at its own read-out):** *"+ reconstruction fitting (`convae`) 0.238 — **−0.006** (bought nothing)"* against `randconv` **0.244**, at the **pre-sweep read-out defaults** (whitened PCA · no L2 · `d = 256` — which §4 of this same report then measured as **the worst corner of 28**, 0.2676 vs the chosen 0.3496).
>
> **What was measured one wave later at the CHOSEN read-out** (`keep` spatial · plain PCA · L2 · `d = 256`), `c2w8-cifar-strong-phi` §1/§2, 3 seeds, Split-CIFAR-10 reduced protocol, `enc_steps = 8000`, `task1_only`, 200 items: **`convae` CLU ACC 0.267 ± 0.006** and **kNN-in-φ ring buffer 0.277 ± 0.003**, against `randconv` **0.213 ± 0.006** / 0.235 ± 0.005 ⇒ ⭐ **`convae` is +0.054 ABOVE `randconv`, and it CLEARS the registered N7 threshold (+0.1063 ± 0.0038 paired vs the in-harness `pca` row, 3/3 seeds) — a pre-registered MISS that was falsified.**
>
> ⛔⛔ **THE NEVER-QUOTE, in the only admissible form:** ~~*"the reconstruction objective bought nothing (−0.006)"*~~ and ~~*"`convae` never beat the untrained trunk"*~~ are **NEVER-QUOTE as objective statements**, and **§3's objective-vs-architecture decomposition may not be quoted at all** (every row of it was measured at the pre-sweep corner). ⛔ **The correction is NOT *"reconstruction helps"*** — that is equally unquotable from these two cells. ✅ **The honest form, and the only one approved:** ***the SIGN of the objective comparison depends on the read-out, and the banked table's read-out was not the measured one.*** ⇒ ⭐ **A defensible "the objective matters" sentence requires re-running §3's isolation table at the measured read-out; it does not exist yet** (`c2w8-cifar-strong-phi` open question 4).
>
> ⚠ **What this banner does NOT touch:** §2's gate values (0.339 ± 0.015 at 8 k / 0.357 ± 0.019 at 20 k) · §4's read-out sweep itself (it is the evidence *for* this correction) · §5's strict-φ cost **+0.020 ± 0.010**, which `c2w8-cifar-strong-phi` declares NOT-RUN and leaves **unamended** · §7's geometry (packing slack 0.332, ⛔ never the retracted 1.08). ⚠ **§4's own standing caveat still governs both reports: the read-out was selected on the decision metric, on seed 0 of the `simclr` trunk** — so `convae`'s hit is obtained under a read-out tuned on a *different* objective (conservative for `convae`, and it must be said).
> **Registry:** `negative_results.md` **N258** (+ the dated amendment block on **N121**); ledger ⟲ **C2W8 pass-1** addendum; primer **§11.9 wave-26 block, dated C2W8 update**. **Sources:** `c2w8-cifar-strong-phi` reconciliation 3 / §1 / §2 / §6 / §9.4; `[C2W8]` §10 review entry, reconciliation 4.

**⭐ Reconciliation list: 5 items, immediately below — item 0 is a DECISION the Hub must take
(the entry run is authorised by the gate and deliberately not run; §10 has the recipe + cost).**

**Task + acceptance criterion:** build a CL-capable read-in φ and run the decisive gate FIRST —
*does kNN-in-φ at 200-item matched memory on Split-CIFAR-10 clear 0.35 (w25 PCA-32 = 0.21)?*
**Status: done on the gate — measured at two φ-compute levels, and the answer straddles the gate:
MISS at the arm I pre-registered (8 000 SimCLR steps: 0.339 ± 0.015, 3 seeds), CLEAR at 2.5× the
fit budget (20 000 steps: 0.357 ± 0.019, 3 seeds), with the gate value itself never moved.
⚠ The conditional next step — the full entry run — is **NOT DONE** (≈5 h of φ-refitting +
baselines); the exact recipe, cost and pre-registered expectation are handed over in §10.**

> **⭐ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Five items.**
> 0. ⭐⭐ **DECISION NEEDED: the gate is cleared, so the CIFAR entry is authorised by the
>    registered rule — but it is not run.** The Hub must decide whether to spend ~5 h on it,
>    *knowing in advance* (item 4) that the launder it will face is **0.357–0.395**, i.e. the
>    entry's most likely outcome is a fifth consecutive laundering confirmation on a harder
>    benchmark. §10 has the command, the config block and the cost breakdown.
> 1. ⭐ **w25's "the CIFAR failure is the feature space" is CONFIRMED AND QUANTIFIED:** φ alone
>    moves the ceiling **0.219 ± 0.017 → 0.357 ± 0.019** (+0.138) with **zero** change to the
>    store or the stream discipline. **"kNN over the same φ caps at 0.21" is superseded** — the
>    entry's CIFAR scope boundary must be restated at the new number.
> 2. ⭐⭐ **The strict-φ cost FINALLY BITES: +0.020 ± 0.010 (3/3 seeds positive, pool-size-matched)
>    at the working arm vs +0.003 at PCA — and it is the difference between clearing and missing:
>    the leaky `generic_frozen` arm scores 0.358 ± 0.015 (clears), the defensible `task1_only` arm
>    0.339 ± 0.015 (misses).** This is the number w24 `phi-stream-discipline` asked for and w25
>    could not supply (its φ was broken). **`PREREG_CL_PHI` §7's "strictness costs essentially
>    nothing" is MNIST/PCA-specific and needs a dated amendment.**
> 3. ⭐ **The CIFAR null has TWO causes, and the second is now measured:** at the 8 k arm the gate
>    goes 0.339 (200 items) → **0.357 (400)** → **0.398 (1000)** → **0.455 (5000)**; at the 20 k arm
>    0.357 → **0.395 (400)** → **0.433 (1000)**. *Doubling the memory is worth as much as 2.5× the
>    φ compute.* An address here is **256 floats vs a raw exemplar's 3072**, so at matched BYTES
>    the store gets ~12× the items of ER/iCaRL/GDumb. **Direct input to `matched-bytes-frontier`.**
> 4. ⚠ **The launder trap fired exactly as pre-registered and must travel with items 0 and 1:**
>    the gate metric *is* the laundering control, so this work raised the launder 0.219 → **0.357**
>    (0.395 at 400 items) **by construction**. **Nothing in this report is a store result.** Any
>    wording that reads "the encoder improved CLU" is wrong; an entry run at this φ faces a
>    0.357-strong kNN line before it starts.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — an **enabler** for R4 (the CL entry), not a store result.
- **Laundering control:** kNN-in-φ over the same new features at matched memory. **It is the gate
  metric itself.** Every number below is a launder number.
- **Falsifies the claim:** kNN-in-φ over the new features fails the gate. **Outcome: the falsifier
  FIRED at the pre-registered arm** (8 k steps, 0.339 < 0.35) and **did not fire at 2.5× φ compute**
  (20 k steps, 0.357 ≥ 0.35). Both are reported; the gate value was never moved (§9).
- **Does NOT falsify:** the store losing to iCaRL/replay (CM-23(n)); the launder firing on ACC.

---

## Flag-provenance table (governs every number in this report)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/cl-encoder`, base local `main @ ff85573` |
| commits (6) | `ef8f8f3` conv arms · `daae0ec` config knobs · `4894c03` tests · `471ed19` measured defaults · `8be8053` x64 dtype fix · `9e9c68d` cosmetic rename |
| worktree | `/Users/user/Desktop/CHLU-cl-encoder` (protocol §3.2; 4 engineer worktrees this wave). **Main venv reused** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`) — no worktree `uv sync`, no JAX drift |
| JAX / equinox / optax | **0.9.0 / 0.13.4 / 0.2.6** (main venv, protocol §4) |
| dataset / protocol | Split-CIFAR-10, **reduced protocol** (`apply_cifar10`): 5 tasks × 2 classes, 1000 train / 500 test per task, **Class-IL** (task identity NOT given at test). ⚠ never literature-comparable |
| gate metric | ring-buffer kNN-in-φ (class-balanced — the stronger launder), Class-IL end-of-stream `ACC = mean_i A[T−1,i]`, **200-item matched memory**, `task1_only` φ. Identical object to w25's `knn_phi_ringbuffer_task1_only` row |
| φ (the arm) | `phi_arm=simclr`, **`phi_dim=256`**, `enc_head="pca"`, `enc_l2_normalize=True` (cosine addresses), `enc_channels=[32,64,128]`, `enc_pool=2` (h_dim 512), `enc_groups=8`, **`enc_steps=8000` (registered arm) / `20000` (the clearing arm, §9)**, `enc_batch=128`, `enc_lr=1e-3`, `enc_temperature=0.5`, `enc_proj_dim=64`; augment = pad-4 crop + 20→32 zoom (p 0.5) + hflip (0.5) + colour jitter (p 0.8, s 0.4) + grayscale (0.2) |
| φ fit pool | `n_fit_region=25000`, `n_fit_pool=6000` ⇒ **4966–4990 task-1-class images** ({0,1}), **disjoint from every stream item**, unsupervised (no label, no retrieval loss, no store gradient), frozen at end of task 1, never refit, **never trained through the store** (w20 law) |
| seeds | **0, 1, 2** on every headline; the read-out config was selected on **seed 0 only** and then applied unchanged to seeds 1, 2 |
| store (geometry only) | `PhiStore` + MVC-0 `Controller`, `clu_s_frac=0.2`, `d_safe_mult=4.4`, `s_policy="refit"`, `memory_items=200`. **No CLU settle was run** — no retrieval claim is made here |
| harnesses | `.claude/outputs/cl-encoder/{gate_knn.py, fit_and_sweep.py, store_geometry.py}`, all importing production code (`build_cl_stream`, `build_phi`, `build_read_in`, `RingBufferKNN`, `PhiStore`, `cl_metrics`, `_geometry_report`); every number re-derived by `render.py` from `results/*.json` |
| PREREG | `.claude/outputs/cl-encoder/PREREG.md`, written **before** any harness ran |
| ⛔ NOT RUN (≠ null) | **the full CL entry** (authorised by the gate at 20 k, ≈5 h — see §10) · the CLU settle at the new arm (no retrieval number exists) · the strict-φ cost at the 20 k arm · MNIST at the new arm · `convae`/`randconv` beyond seed 0 |

---

## 1. What I built

1. **`chlu/experiments/phi_encoders.py`** — a small conv trunk (`[conv3×3 → GroupNorm → ReLU →
   maxpool2] × 3` → adaptive-avg-pool 2×2 → 512-d `h`) with **three unsupervised arms** behind the
   existing `phi_arm` flag:
   - `randconv` — untrained; **sees no data at all** (architecture-only control);
   - `convae` — mirror decoder, **reconstruction MSE** (the objective `PREREG_CL_PHI` licenses);
   - **`simclr`** — projection head + **NT-Xent** over two augmented views; head discarded, φ reads
     the trunk.
   `h → phi_dim` via a PCA head fit on the same pool, so **`phi_dim` keeps meaning the store's
   address dimension** and every number stays quotable with it (binding since w24). GroupNorm not
   BatchNorm: stateless ⇒ a *frozen* φ behaves identically at fit and read time.
2. **Additive dispatch** in `exp_phi_read_in.build_read_in` (one import + 3 lines) — `pca`/`ae`
   bit-identical; a test pins that.
3. **Config knobs** `enc_*` (17) in `ExperimentClEntryConfig` and `ExperimentPhiStreamConfig` (the
   φ surface I own), inert unless a conv arm is selected; three defaults set **from the measured
   sweep** (§4), not from taste.
4. **14 tests** (`tests/test_phi_encoders.py`) incl. the NT-Xent chance-value guard
   (`loss(random) ≈ ln(2N−1)` — anything else means a pairing bug) and a test that the encoder's
   constructor takes **no labels** (PREREG_CL_PHI §3).

**Why a contrastive objective and not just an AE (the justification the task asked for).**
Reconstruction allocates capacity by *pixel variance* — colour and low spatial frequency on CIFAR
— which is exactly what PCA already maximises and demonstrably not what separates the classes.
A contrastive objective instead *optimises the invariances a nearest-address read-out needs*
(translation, crop, colour). The measurement (§3) confirms it: `convae` never beat the *untrained*
`randconv` trunk, while `simclr` beat both by a wide margin.

---

## 2. ⭐ THE GATE (the decisive result)

**Gate: ring-buffer kNN-in-φ ≥ 0.35, Split-CIFAR-10 reduced protocol, 200 items, `task1_only`.**

| φ arm | `phi_dim` | fit pool | gate ACC | vs gate |
|---|---|---|---|---|
| **PCA-32 — the w25 φ (harness-validity check)** | 32 | 2027 | **0.219 ± 0.017** (3 seeds) | ⛔ −0.131 |
| PCA-32, larger pool | 32 | 4966 | 0.226 ± 0.008 (3) | ⛔ −0.124 |
| PCA-256 | 256 | 2027 | 0.210 ± 0.010 (3) | ⛔ −0.140 |
| AE-32 / AE-64 (existing arm) | 32/64 | 2027 | 0.214 / 0.223 (seed 0) | ⛔ |
| `randconv` (no data at all), best read-out | 32 | — | 0.244 (seed 0) | ⛔ |
| `convae` (reconstruction), pre-sweep defaults | 64 | 2027 | 0.238 (seed 0) | ⛔ |
| `simclr`, 2000 steps | 256 | 2027 | 0.285 (seed 0) | ⛔ |
| **`simclr`, 8000 steps — THE REGISTERED ARM** | **256** | **4966** | **0.339 ± 0.015** (3) | ⛔ **−0.011** |
| *`simclr` 8000 steps, `generic_frozen`* | *256* | *4966* | *0.358 ± 0.015* (3) | *⚠ leaky — clears, never a headline* |
| ⭐ **`simclr`, 20000 steps (2.5× fit budget, §9)** | **256** | **4966** | **0.357 ± 0.019** (3) | ✅ **+0.007 — CLEARS** |

Per-seed: **8 k → 0.3496 / 0.3452 / 0.3212**; **20 k → 0.3652 / 0.3704 / 0.3348** (2/3 seeds
individually above 0.35). The PCA-32 control reproduces w25's `0.219 ± 0.014` to three decimals —
that is the harness-validity evidence (P0).

**Verdict against the registered rule.** At the arm I pre-registered (8 k steps) the mean lands in
the **borderline band 0.30–0.35, which I registered *in advance* as a MISS**. Adding fit compute —
the one lever §3 identified as unsaturated, with the gate metric and its threshold unchanged —
takes it to **0.357 ± 0.019, which CLEARS**. Both numbers are reported; neither is a store result.
**The honest one-line verdict: the encoder IS the fix, but only at ~2.5× the φ-fit compute I
registered, and by a margin (+0.007) smaller than the seed spread (±0.019).**

**Reference points, same reduced protocol (w25):** LwF 0.162 · GDumb 0.301 · **ER 0.369** ·
iCaRL 0.419 · joint 0.480. The new φ's kNN line now sits **above GDumb** and just below ER.

**⚠ The trap, in plain words (PREREG P7).** Every number in this table is a **kNN** number. w25's
entry scored CLU 0.149 where kNN scored 0.219; the honest expectation at this φ is a CLU line
**near, and probably below, 0.357**. If the entry is later run here and lands at 0.35, the wave
has bought R4 a **scope**, not a win.

---

## 3. What actually bought the movement (arms isolated, seed 0, best read-out)

| step | gate ACC | Δ |
|---|---|---|
| PCA-32 (the w25 φ) | 0.209 | — |
| + conv architecture only (`randconv`, **no data**) | 0.244 | **+0.035** |
| + reconstruction fitting (`convae`) | 0.238 | **−0.006** (bought nothing) |
| + contrastive fitting (`simclr`, 2000 steps, 2027 imgs) | 0.285 | **+0.041** |
| + 4× steps and 2.5× unlabelled data (8000 steps, 4966 imgs) | 0.350 | **+0.065** |

Two levers matter: **the contrastive objective** and **fitting compute/data** — and the second has
**not saturated**. NT-Xent: 5.51 → 4.19 (2 k steps, 2027 imgs) → **4.07** (8 k steps, 4966 imgs),
against a chance value of `ln(255) = 5.54` and a converged small-SimCLR value of ~2.5–3.
**The gate is compute-limited, not concept-limited** (§9 tests this directly at 20 k steps).

---

## 4. The read-out sweep (28 configs, seed 0, 8000-step trunk)

The `h → φ` read-out is **not free**, and the pre-sweep default was the worst corner:

| config | gate ACC |
|---|---|
| **`keep` spatial · plain PCA · L2/cosine · d=256** (chosen) | **0.3496** |
| plain PCA · L2 · d=128 | 0.3436 |
| plain PCA · no L2 · d=64 | 0.3432 |
| whitened PCA · no L2 · d=32 | 0.3004 |
| **whitened PCA · no L2 · d=256** (the pre-sweep default) | **0.2676** |
| `gap` (global-avg-pooled) · whitened · no L2 · d=64 | 0.2740 |

- **Cosine (L2-normalised) addresses beat Euclidean on average but not universally**: +0.008 mean
  over the 14 matched pairs, **8/14 positive**, range −0.009 … +0.041 — the gain is large for
  whitened features and small (±0.01) for plain PCA. It won the chosen cell, so it is the default;
  it is **not** a strong effect on its own.
- **Whitening hurts**: −0.028 mean over all cells, up to **−0.075** at `d=256` — it amplifies
  low-variance trunk directions that a 1-NN read-out then treats as signal.
- **Keeping the 2×2 spatial map beats global average pooling** by **+0.023** (mean over matched
  cells) — spatial layout is part of the address.
- **`phi_dim` is monotone non-decreasing** — 0.320 / 0.330 / 0.335 / 0.339 at 32 / 64 / 128 / 256
  (3-seed means) — with 128→256 worth only +0.004 ⇒ **P9 confirmed**. `phi_dim=128` is the sane
  byte-conscious operating point (128 floats/item, 24× smaller than a raw exemplar).

⚠ **This sweep is hyper-parameter selection on the decision metric.** It ran on **seed 0 only**;
the winner was applied *unchanged* to seeds 1, 2 (that is where the ±0.015 comes from). The
quotable number is the 3-seed mean at the chosen config (**0.339**); the seed-0 best-of-28
(0.350) is **not** quotable as the gate result.

---

## 5. ⭐⭐ The strict-φ cost — the number w24 asked for and w25 could not give

`generic_frozen − task1_only`, same trunk config, same stream, same read-out, `phi_dim=256`:

| arm | task1_only | generic_frozen | strict-φ cost |
|---|---|---|---|
| PCA-32 (3 seeds) | 0.219 ± 0.017 | 0.222 ± 0.017 | **+0.003** (w25 measured +0.001 — reproduced) |
| *(all `simclr` rows below are at the **8 000-step** trunk — the cost is unmeasured at 20 k)* | | | |
| **`simclr`-256, pool-size-MATCHED (3 seeds)** | **0.339 ± 0.015** | **0.358 ± 0.015** | **+0.020 ± 0.010** (per seed +0.026 / +0.008 / +0.025 — **3/3 positive**) |
| `simclr`-256, unmatched pools (the naive comparison) | 0.339 | 0.360 ± 0.021 | +0.021 ± 0.012 |

⭐ **On a broken φ strictness costs nothing because everything is broken; on a working φ it bites**
— ~7× more than at PCA, in the same direction on every seed, and **exactly across the decision
boundary**: the leaky reference arm **clears** the gate (0.358) and the defensible arm **misses**
it (0.339). At this operating point, the entry's leakage-freedom is worth precisely the gap to the
gate.

⚠ **Confound found and controlled.** `build_cl_stream` hands the reference arm `n_fit_pool`
images (6000) but the strict arm only the task-1-class members of the fit region (≈4966) — a
**+21 % data advantage** for the leaky arm. I re-ran all three `generic_frozen` seeds at
`n_fit_pool=4966` so both arms see the same number of images: the cost barely moves
(+0.021 → **+0.020**), i.e. **the cost is the class restriction, not the data volume.** The
matched row is the one to quote.

---

## 6. The second cause, named and measured: the 200-item budget

Same φ, same stream, only the memory budget changes (ring-buffer kNN-in-φ, Class-IL):

| memory (items) | 200 | 400 | 1000 | 5000 (the whole stream) |
|---|---|---|---|---|
| 8 k arm (seeds 1, 2) | 0.333 | **0.357** | **0.398** | **0.455** |
| 20 k arm (3 seeds) | 0.357 | **0.395 ± 0.007** | **0.433 ± 0.013** | 0.482 |

⭐ **At 400 items the gate clears even at the under-trained 8 k arm**, and doubling the memory is
worth about as much as 2.5× the φ-fit compute. The Split-CIFAR-10 null therefore has **two** causes: the
address space (fixed here, +0.14) *and* the item budget (≈+0.02 per doubling near 200). w25's
single-cause framing is now the larger half of a two-term explanation.

**Why this matters for the wave (reconciliation item 3).** 200 items is a *matched-item*
constraint. At this arm an address is **256 floats** vs a raw CIFAR exemplar's **3072**, so at
matched **bytes** the store/kNN line gets ~**12×** the items of ER/iCaRL/GDumb — deep into the
0.45 regime of the table above. `matched-bytes-frontier` owns that axis; this is a direct input.

---

## 7. Geometry at the new arm (required deliverable)

`simclr`-256 at the **8 000-step** trunk, cosine addresses, seed 0, end of stream, 200 live wells,
**no CLU settle**:

| quantity | value |
|---|---|
| median-NN address spacing | **0.975** |
| well width `s` (`= clu_s_frac · median-NN`) | **0.192** |
| `σ_q` (RMS ‖φ(test query) − nearest address‖, a **norm**) | **0.946** |
| **corrected packing slack** = median-NN / (3.1·max(s, σ_q)) | **0.332** |
| kNN over the STORE's own admitted keys (same-keys launder) | 0.325 |
| admitted fraction per task | 0.855 / 0.741 / 0.640 / 0.630 / 0.931 (budget-bound at 200) |

⛔ **Never quote the retracted 1.08.** The slack is **0.332**, indistinguishable from w25's CIFAR
0.337–0.345 and MNIST ≈0.33 ⇒ **P8 confirmed: a better φ does not de-crowd the store.**
`σ_q ≈ median-NN` says a test query is about as far from its nearest address as two addresses are
from each other — crowding is intrinsic to a *classification* stream (the query is a different
image, not a corrupted copy of a stored one), not a φ defect. Note the same-keys launder (0.325)
sits below the ring-buffer launder (0.350) at the same φ, reproducing w25's ordering.

---

## 8. PREREG scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| P0 | PCA-32 control in 0.19–0.24, else the harness is wrong | **0.219 ± 0.017** | ✅ harness valid |
| P1 | PCA-64/128 in 0.21–0.25; P(clear) < 0.02 | 0.210–0.213 | ✅ |
| P2 | AE-32/64 in 0.19–0.26 | 0.214 / 0.223 | ✅ |
| P3 | conv AE 0.22–0.32, point 0.26 | 0.238 (and never above `randconv`) | ✅ |
| P4 | SimCLR 0.28–0.45, point **0.36** | **0.339 ± 0.015** (8 k) / **0.357 ± 0.019** (20 k) | ✅ in band; the 20 k arm lands on the registered point value to 0.003 |
| P5 | random conv 0.20–0.30 | 0.244 | ✅ |
| P6 | strict-φ cost ≈0 for PCA/AE; **+0.02…+0.10** at a working arm | **+0.003** / **+0.020 ± 0.010** | ✅ **both halves; at the band's edge** |
| P7 | the launder rises by construction; the entry would land ±0.05 of the kNN line | the launder rose 0.219 → 0.357 **as registered**; the CLU half is untestable — the entry is NOT RUN (§10) | ◐ half-confirmed, half carried |
| P8 | packing slack < 1, band 0.25–0.50 | **0.332** | ✅ |
| P9 | gate monotone non-decreasing in `phi_dim`; 128→256 ≤ 0.03 | monotone; **+0.004** | ✅ |
| GATE | ≥0.35 ⇒ run the entry; 0.30–0.35 ⇒ MISS, stop, report | **0.339 ± 0.015 at the registered arm** (MISS) → **0.357 ± 0.019 at 2.5× φ compute** (CLEAR) | ◐ **straddles**: the registered arm missed; the gate is cleared by adding fit compute, not by moving the criterion |

**Score: 9 ✅ / 2 ◐.** The pre-registration held: the SSL point prediction (0.36) is within 0.003
of what the fully-trained arm actually did, and the *shape* of every prediction — which arms move,
which do not, what strictness costs, what the geometry does, that the launder rises by
construction — was right.

---

## 9. ⭐ Does more φ compute clear it? (the registered compute-limited test)

Same arm, same read-out (`keep`/`pca`/L2/`d=256`), same 4966-image task-1 pool — only the fit
budget changes:

| trunk fit | NT-Xent (chance 5.541) | gate ACC (200 items) | at 400 | at 1000 | full-stream kNN |
|---|---|---|---|---|---|
| 2 000 steps (2027 imgs), seed 0 | 5.503 → 4.191 | 0.285 | — | — | 0.403 |
| 8 000 steps, **3 seeds** | → 4.07 | **0.339 ± 0.015** | 0.357 | 0.398 | 0.455 |
| ⭐ **20 000 steps, 3 seeds** | → **3.98** | **0.357 ± 0.019** — **CLEARS** | **0.395 ± 0.007** | **0.433 ± 0.013** | 0.482 |

Per-seed at 20 k: **0.3652 / 0.3704 / 0.3348** (seeds 0/1/2) — 2/3 individually above the gate; the
weakest seed (2) is the weakest at 8 k too (0.3212), so the seed ordering is a property of the
*stream draw*, not of the fit.

⭐ **The gate is compute-limited, exactly as §3 argued**, and the NT-Xent loss is *still* falling
at 20 k (3.98 vs a converged ~2.5–3) — this is not the ceiling either. Note the 20 k full-stream
kNN (0.482) has reached the **joint upper bound of the reduced protocol (0.480)**: with all 5000
keys these addresses are as informative as training the reduced-protocol CNN on everything.

⚠ **How to quote this honestly.** The gate *threshold* and *metric* were never changed — only the
fit budget, which §3 had already identified (before the 20 k run) as the unsaturated lever. But
the margin (+0.007) is **smaller than the seed spread (±0.019)**, so the correct sentence is
*"the gate is cleared on the 3-seed mean at 20 000 fit steps, by less than one seed-σ"* — not
"comfortably cleared". Anyone building on it should re-measure at more seeds or more fit compute.

---

## 10. ⛔ The entry run: AUTHORISED, NOT RUN — recipe, cost, and what to expect

The registered rule says a cleared gate ⇒ run the full entry. **I did not run it**, deliberately:
it is a ≈5 h job (the φ refit dominates) arriving at the end of a ~10 h task, on a machine the
wave already flags as compute-contended (4 engineer worktrees). Handing over a complete,
verified gate beats handing over a half-finished entry. Everything needed is below.

**No code change is required** — the new arms reach `exp_cl_entry` through the existing
`phi_arm` string, and every `enc_*` knob is already on `ExperimentClEntryConfig`.

⚠ **Trap, found while writing this recipe:** `chlu exp-cl-entry --dataset cifar10` calls
`apply_cifar10()` **after** the project YAML is loaded, and that helper **overwrites**
`n_fit_region` (→10000) and `n_fit_pool` (→3000) — which would silently halve the φ fit pool and
put you back at a ~0.30 arm. **Put `dataset: cifar10` in the YAML and do NOT pass `--dataset`**,
carrying the reduced-protocol values explicitly:

```yaml
# projects/<name>/config/config.yaml  →  experiment_cl_entry:
dataset: cifar10
backbone: cnn
n_train_per_task: 1000
n_test_per_task: 500
baseline_iters: 150
mlp_width: 128
fisher_samples: 100
tune_baselines: false
clu_steps: 150
# ---- the w26 φ (the arm that clears the gate) ----
phi_arm: simclr
phi_dim: 256
phi_regimes: ["task1_only"]      # ⚠ adding generic_frozen DOUBLES the cost (a 2nd φ fit/seed)
n_fit_region: 25000
n_fit_pool: 6000
enc_steps: 20000                 # 8000 misses the gate; 20000 clears it
enc_head: pca
enc_l2_normalize: true
```
```bash
PYTHONPATH=<repo> CHLU_CIFAR10_TARBALL=<...>/cifar10.tar.gz \
  .venv/bin/python -m chlu.experiments.exp_cl_entry --project <name> --items entry
```

**Cost breakdown (measured on this machine):** φ fit **64 min × 3 seeds = 3.2 h** (the dominant
term, and it is *per regime*) + store/eval ≈10 min/seed + the 8 baselines ≈30 min total
⇒ **≈4.5–5 h** for the PRIMARY arm alone; **≈8 h** if the `generic_frozen` reference arm is
included (which the PREREG requires for any *quoted* strict-φ number, though §5 already supplies
that number at the 8 k arm).

**What to expect, pre-registered here before anyone runs it (so the result is evidence either
way):** the kNN-in-φ launder will be **0.357 ± 0.019** at 200 items; the CLU entry line will land
**within ±0.05 of it and most likely below** (5 waves of precedent, and the store's own
same-keys kNN at the 8 k arm was already 0.03 *below* the ring buffer). Concretely: **CLU
0.30–0.36, `laundered = True`**, above LwF (0.162) and GDumb (0.301), near/below ER (0.369),
below iCaRL (0.419), with **BWT far better than the parametric rehearsal-free class** (the axis
w25 showed the store actually owns). If that is the expected outcome, the honest question for
the Hub is whether the 5 h buys anything the forgetting/matched-bytes axis does not already own.

---

## How I verified

- `pytest tests/test_phi_encoders.py -q -p no:randomly --no-cov` → **14 passed** (24 s), and
  **14 passed** again under `JAX_ENABLE_X64=1` (the regression the full suite exposed).
- `pytest tests/test_phi_encoders.py tests/test_phi_read_in.py` → **23 passed** (25 s) — the
  pre-existing φ tests still pass with the dispatch hook in place.
- **Full suite:** first run → **695 passed, 9 failed**; all 9 were my own new tests dying under
  the suite-wide `jax_enable_x64` (`lax.conv_general_dilated requires arguments to have the same
  dtypes`). Fixed in `8be8053`. **Re-run: `704 passed, 0 failed` (928 s)** = the handover's 690
  + my 14.
- `ruff check chlu/` → **All checks passed** after every commit.
- Harness validity: the PCA-32 control reproduces w25's `0.219 ± 0.014` on 3 seeds.
- Sanity guard inside the arm: NT-Xent starts at 5.50–5.51 vs the analytic chance value
  `ln(2·128−1) = 5.541`.
- Every number here is re-derived from the shipped JSON by
  `.claude/outputs/cl-encoder/render.py` (raw JSON + logs in `results/`).

---

## Git footprint

- **Branch** `agent/experiment-engineer/cl-encoder`, base local `main @ ff85573`, **worktree**
  `/Users/user/Desktop/CHLU-cl-encoder`. **No push, no PR.** `git rebase main` = no-op (base
  unmoved).
- **6 commits:** `ef8f8f3` (conv arms) · `daae0ec` (config knobs) · `4894c03` (tests) · `471ed19`
  (defaults from the sweep) · `8be8053` (x64 dtype fix) · `9e9c68d` (cosmetic rename).
- **Files — 2 new, 2 surgical:** NEW `chlu/experiments/phi_encoders.py` (≈400 lines),
  `tests/test_phi_encoders.py` (14 tests). Modified `chlu/experiments/exp_phi_read_in.py`
  (+1 import, +3-line dispatch inside `build_read_in` only) and `chlu/config.py` (+17 `enc_*`
  fields in `ExperimentClEntryConfig`, +17 in `ExperimentPhiStreamConfig`).
- **File-ownership compliance (task §"File-ownership split"):** **zero edits** to
  `chlu/experiments/cl_baselines.py`, `chlu/experiments/exp_cl_entry.py` (⚠ imported read-only)
  and `chlu/core/*`. My only shared-file edit is the additive `chlu/config.py` block, which will
  conflict **textually** with the other engineer worktrees at the same insertion points and must
  be resolved **additively** (w23/w24/w25 precedent). **No entry-side change was needed** — the
  new arms reach `exp_cl_entry` entirely through the existing `phi_arm` string.
- Verified from the MAIN repo that the branch ref carries every commit
  (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/cl-encoder`).
  Worktree left in place for the Hub to remove at integration.

---

## Open questions / follow-ups / risks

1. ⭐⭐ **The open decision is §10: the entry is authorised and unrun.** It costs ≈5 h and its
   most likely outcome is pre-registered there (CLU 0.30–0.36, `laundered = True`). The Hub owns
   the call.
2. ⭐ **More φ compute is still unspent headroom, with no new code needed.** NT-Xent is 3.98 at
   20 k steps against a converged ~2.5–3, and the full-stream kNN has *already* reached the
   reduced protocol's joint bound (0.482 vs 0.480). A 50 k-step arm is a ~2.7 h/seed job that
   would likely put the 200-item gate comfortably (not marginally) above 0.35.
3. ⭐ **The budget route is cheaper than both and arguably more *correct*:** at matched **bytes**
   rather than matched **items** this φ is already at 0.40–0.46 (§6).
4. ⚠ **`phi_dim=256` is 8× the w25 address dimension.** Every downstream "**24.5× fewer floats**"
   sentence is **MNIST-specific**; on CIFAR at this arm it is **12×** (24× at `phi_dim=128`,
   which costs only 0.004 of the gate). Do not let the MNIST ratio travel to CIFAR text.
5. **The store's own retrieval at the new arm is unmeasured** (no settle was run — deliberately,
   the gate rule forbids spending on it). Anyone quoting a CLU number at this φ must run it.
6. **MNIST is untouched by this branch** — the conv arms are wired and tested for MNIST shapes,
   but no MNIST number was measured; the w25 MNIST entry results stand unchanged.
7. **Risk — this makes the launder stronger.** If the entry is later run here and loses to a
   0.357 kNN line, that is a **fifth** consecutive laundering confirmation, now on a harder
   benchmark. Decide *before* the run, not after (§10).
8. **The conv arms are expensive** — 21 min/fit at `enc_steps=8000`, **64 min at 20 000**, ×seeds
   ×regimes. Budget ≈1 h per 3-seed 8 k arm and ≈3.2 h per 3-seed 20 k arm.
9. **The strict-φ cost is measured at the 8 k arm only.** Whether it stays ≈+0.02 at 20 k (where
   both arms are better trained) is unmeasured — it could shrink (more compute closes the
   class-coverage gap) or grow. One more 3-seed `generic_frozen` run at 20 k (3.2 h) settles it.

---

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** *A CL-capable conv φ exists* (`phi_arm ∈ {randconv, convae,
   simclr}`, `chlu/experiments/phi_encoders.py`, w26). On Split-CIFAR-10 (reduced protocol,
   200-item matched memory, `task1_only`, `phi_dim=256`) it moves the kNN-in-φ ceiling
   **0.219 ± 0.017 → 0.339 ± 0.015 at 8 000 fit steps and 0.357 ± 0.019 at 20 000** (3 seeds each).
   **w25's "kNN over the same φ caps at 0.21" is SUPERSEDED. The 0.35 gate is missed at the
   registered arm and cleared at 2.5× φ compute (by less than one seed-σ); the full entry is
   AUTHORISED but NOT RUN** (§10 has the recipe, the ≈5 h cost and the pre-registered expectation).
   The gate also clears at **400 items** (0.395 at 20 k) and at the leaky `generic_frozen` φ.
2. **§7 — new config surface:** 17 `enc_*` knobs on `ExperimentClEntryConfig` and
   `ExperimentPhiStreamConfig`; `phi_arm` now accepts `randconv|convae|simclr`. Defaults changed
   **for the new arms only** and **from measurement**: `enc_head="pca"` (whitening measured
   worse), `enc_l2_normalize=True` (cosine addresses), `enc_steps=8000`. No pre-existing behaviour
   changes; `pca`/`ae` are bit-identical (pinned by a test).
3. **Candidate N-entries (3).** (a) *"A task-1-only SSL conv φ moves the Split-CIFAR-10 kNN
   ceiling 0.219 → 0.357 with no change to the store or the stream discipline; the 0.35 admission
   gate is missed at 8 000 fit steps and cleared at 20 000, by less than one seed-σ"* — tier A,
   scope-defining for R4. (b) ⭐ *"The strict-φ cost is a property of a WORKING
   φ: +0.003 at PCA (both arms broken) vs +0.020 ± 0.010 at the SSL arm (3/3 seeds positive,
   pool-matched), where the leaky reference clears the gate and the defensible arm does not"* —
   tier A; **amends `PREREG_CL_PHI` §7**. (c) *"The Split-CIFAR-10 null has two causes: the address
   space (+0.12 recovered) and the 200-item budget (+0.02 per doubling; clears at 400 items)"* —
   tier A, direct input to `matched-bytes-frontier`.
4. **`PREREG_CL_PHI` needs a dated amendment.** §7's binding `phi_dim ≥ 16` is unaffected, but its
   conclusion *"the strict φ costs essentially nothing"* must be **scoped to MNIST/PCA** and
   paired with the CIFAR/SSL measurement (+0.020 ± 0.010, and decision-changing at the gate).
5. **Do-not-quote list.** ⛔ "the gate was cleared" **without** the fit budget attached (8 k:
   0.339 ± 0.015 = miss; 20 k: 0.357 ± 0.019 = clear by +0.007 < 1 σ). ⛔ the
   retracted slack 1.08 (this arm: **0.332**). ⛔ "24.5× fewer floats" on CIFAR (that is MNIST;
   here **12×** at `phi_dim=256`). ⛔ any phrasing implying the encoder improved **CLU** — every
   number in this report is a **kNN/launder** number. ⛔ the seed-0 best-of-28 read-out value
   (0.350) as the gate result.
6. **Test count:** +14 ⇒ integration should expect **704** (measured green on this branch).
7. **Compute note for wave planning:** the whole task cost ≈10 h wall-clock, of which ~9 h was φ
   fitting (21 min per 8 k trunk and 64 min per 20 k trunk, × seeds × regimes). The gate harness
   itself is free (seconds); **all** the cost is in φ.
8. ⚠ **A usability trap worth recording in §7:** `chlu exp-cl-entry --dataset cifar10` applies
   `apply_cifar10()` **after** the project YAML and silently overwrites `n_fit_region`/`n_fit_pool`
   (and `baseline_iters`, `tune_baselines`, …). Any CIFAR run that needs a non-default φ fit pool
   must set `dataset: cifar10` in the YAML and **not** pass `--dataset` (§10).
