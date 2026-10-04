# PREREG — `c2w8-cifar-strong-phi` (the Split-CIFAR strong-φ RE-PRICE)

**Filed 2026-08-06 by the C2W8 experiment-engineer, BEFORE any measuring cell of this task
ran.** Written after the *pricing* probe only (`price_encoder.py`, 50-step timing + parameter
counts — no accuracy is measured there) and before the first entry run.

Base: `main @ d70898b`, worktree `../CHLU-c2w8-phi`, branch `c2w8-cifar-strong-phi`.

⛔ **This file registers a re-price, not a benchmark entry.** The mandatory provenance
(PREREG-C2W8 §8) governs every sentence produced from it:

> *"Split-CIFAR was a null at frozen-PCA φ; re-priced at strong φ (arm named, bytes ledgered),
> it reads X."*

---

## 0. What is inherited and NOT re-derived (task §"Binding documents")

The numeric predictions are the Hub's, registered in `PREREG-C2W8.md` §6:

| # | quantity | registered prediction | registered prior |
|---|---|---|---|
| **N7** | Split-CIFAR strong-φ: CLU ACC lift over the banked PCA-φ null (0.149 ± 0.013) | **≥ +0.10** | 0.55 |
| **N8** | Split-CIFAR strong-φ: CLU beats its **own** kNN-in-φ launder | — | **0.15** |

I do not re-tune or move them. §2 below adds only the *point* values these imply on the arms I
am about to run, so that a miss is legible as a miss.

---

## 1. The run plan, priced BEFORE it runs (task §Budget)

Measured on this machine, `price_encoder.py` (50-step probe, per-step time nets out the ~3–4 s
JIT compile; the netted per-step figures reproduce `cl-encoder`'s 21 min @ 8 000 steps):

| arm | φ fit cost / seed | φ param floats (`phi_dim=256`) |
|---|---|---|
| `pca` (`phi_dim=32`, the banked reference) | ~2 s | **101 376** (`32·3072 + 3072`) |
| `randconv` | ~3 s (**sees no data**) | **225 536** |
| `convae` @ `enc_steps=8000` | **≈ 18–27 min** | **225 536** |
| `simclr` @ `enc_steps=8000` | **≈ 21–30 min** | **225 536** |
| *(`simclr` @ 20 000 — the `cl-encoder` gate-clearing arm)* | *≈ 64–75 min* | *225 536* |

**Declared allocation** (cut arms before seeds, task §Honesty):
- `simclr` @ `enc_steps=8000`, **5 seeds** — the headline-margin arm (N8);
- `randconv`, **5 seeds** — free, and the honest architecture-only control;
- `convae` @ `enc_steps=8000`, **3 seeds**;
- `pca`, `phi_dim=32`, **3 seeds** — the banked PCA-φ **reference row**, re-run *inside this
  harness at this commit* so N7 is a within-harness difference, not a cross-report subtraction;
- baselines (EWC/SI/LwF/finetune/ER/DER++/iCaRL/GDumb/joint) — **3 seeds, run once**; they are
  φ-independent, so re-running them per arm would buy nothing.

**⛔ DECLARED NOT-RUNs (never to be reported as nulls):**
1. **`simclr` @ `enc_steps=20000`** — the arm that clears `cl-encoder`'s 0.35 kNN gate by
   +0.007 (< 1 seed-σ). 5 seeds ≈ 5.3 h of φ fit alone, which does not fit beside the four arms
   above in this task's budget. **The headline arm here is therefore the *under-trained* one**,
   and every N7/N8 number carries `enc_steps=8000`.
2. **`generic_frozen`** — a second φ fit per seed (doubles the cost) and it is a declared upper
   bound that may never be a headline (w24 ruling). ⇒ **no strict-φ cost is measured here**;
   `cl-encoder` §5's `+0.020 ± 0.010` at this exact arm stands.
3. **MNIST at the strong φ** — out of scope; the w25 MNIST numbers stand unchanged.
4. **`online` φ** — the unrun stub, out of protocol.
5. **retry / retention / frontier items** — `--items entry` only.

---

## 2. Point predictions implied by the inherited priors (registered here, before the sweep)

Derived from `cl-encoder`'s **measured** kNN-in-φ ceilings on this exact protocol
(ring-buffer kNN, 200 items, `task1_only`) — those are launder numbers, not store numbers:

| arm | measured kNN ceiling | predicted CLU ACC | predicted N7 lift over 0.149 | N7 verdict |
|---|---|---|---|---|
| `pca`-32 (reference) | 0.219 ± 0.017 | 0.15 ± 0.02 (reproduces the banked null) | — | — |
| `randconv` | 0.244 (seed 0) | **0.19 – 0.25** | +0.04 … +0.10 | predicted **MISS** |
| `convae` @8k | 0.238 (seed 0) | **0.18 – 0.25** | +0.03 … +0.10 | predicted **MISS** |
| **`simclr` @8k** | **0.339 ± 0.015** | **0.28 – 0.34** | **+0.13 … +0.19** | predicted **HIT** |

⇒ **N7 registered as HIT, and it hinges entirely on the `simclr` arm.** P(N7 hit) = **0.80**
(above the Hub's 0.55, because the Hub's prior predates `cl-encoder`'s gate measurement, which
is a direct upper bound on what the store can score).

**N8 (the one that matters).** Registered as **MISS — the store does NOT beat its own launder.**
Predicted margin `CLU − best kNN-in-φ launder (same φ)` = **−0.08 … −0.01**, point **−0.04**.
P(N8 hit, i.e. margin > 0 beyond 2 SE) = **0.15** (the Hub's prior, unmoved). Evidence: MNIST
−0.036, CIFAR/PCA −0.070, and `cl-encoder`'s explicit warning that a stronger φ raises the
launder by construction.
⚠ Registered in advance: **N7 without N8 is the clean scope clause, not a disappointment** — it
says the feature space *was* the null's cause **and** the store still adds nothing over the
trivial substitute in the better space.

**Falsifier of the re-price (task §DIAL DECLARATION).** If `simclr` leaves CLU ACC statistically
unmoved from 0.149 (lift < +0.05, or overlapping ±2 SE), the null's diagnosed cause (feature
space) is **wrong**, the null re-prices to the **discipline**, and *that* becomes the headline.

**Byte-ledger prediction (§A4.3).** At `phi_dim=256` the frozen φ is **225 536 floats**, i.e.
**4.3× the store's own 200 × 265 = 53 000** memory floats — so on the total ledger the strong-φ
CLU arm is dominated by its encoder, and I predict CLU total ≈ **278 500 floats** vs ER/iCaRL
**≈ 695 900** (614 600 exemplar + 81 290 CNN) ⇒ CLU still **≈ 2.5× cheaper in bytes** while
scoring below them. Registered so the ledger cannot be read post-hoc as either an alibi or a
gotcha. ⚠ The launder carries the **same** 225 536 φ floats, so the ledger does **not** change
the N8 comparison — that is by design.

---

## 3. Instruments (fixed here)

- **ACC / BWT / forgetting** = GEM formulas via `cl_baselines.cl_metrics`, Class-IL, end of
  stream, on the shipped reduced CIFAR protocol (`apply_cifar10`: 1000 train / 500 test per
  task, `baseline_iters=150`, 3-layer CNN backbone) — ⚠ never literature-comparable.
- **The launder** = both shipped kNN-in-φ lines (same-keys and class-balanced ring buffer) at
  matched *items*, computed in the **same φ object** the arm uses. The reported N8 margin uses
  the **stronger** (max-ACC) of the two, as `entry_verdict` already does.
- **N7** = `mean_seeds(ACC[arm]) − mean_seeds(ACC[pca-32 reference, this harness])`, SE from the
  two seed-wise SDs (unpaired: different φ, different seed counts).
- **N8** = **paired per-seed** `ACC(CLU) − ACC(best launder)`, mean ± SE over seeds. Paired
  because both lines are computed on the identical stream, store and φ within a seed.
- **N94:** every cell is run at the shipped reduced protocol; the CLU store runs **zero gradient
  steps**, so the undemoted-write-step floor is not applicable to the store arm — the *baseline*
  arms run at the declared reduced `baseline_iters=150` and are labelled **non-promotable**
  accordingly, exactly as w25 labelled them.

---

*Filed before the first measuring cell. Corrections go in a dated ERRATA block beside this file;
this file is not edited after the first cell runs.*
