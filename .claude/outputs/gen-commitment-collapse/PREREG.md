# PREREG — gen-commitment-collapse
**Filed:** 2026-09-13, BEFORE any line of the commitment harness was written and before any number was measured.
**Agent:** experiment-engineer · **Branch:** `agent/experiment-engineer/gen-commitment-collapse` · **Base:** `main @ 5fee3f4`.

## The acceptance criterion, verbatim from the task (⛔ not re-worded, not softened)
> **PASS = ≥60/72 distinct codes on B AND the store→decoder arm within seed spread of `knn_proto`
> (in-corner shape balanced accuracy 0.8817 ± 0.0023).**
> ⛔ Fail is a legitimate, reportable outcome and routes to new approaches, never to abandonment.
> ⛔ Do not re-tune toward a pass.

**Laundering control on every number:** the matched-seen (MS) corner. A knob that lifts B's distinct count while
MS stays flat has found a **B-specific artefact**, not a fix.

## Reference rows this prereg is written against (from disk, `gen-read-compose` manifests, 3 seeds)
| quantity | value |
|---|---|
| distinct codes, argmax (shipped): A / MS / B | 494,493,498 of 512 · 35,24,24 of ~198 · **9,8,10 of 72** |
| mean top-1 posterior confidence: A / MS / B (seed 0) | 0.7546 / 0.7773 / **0.8560** |
| mean posterior entropy, summed over K=5 wells (B) | 4.285 / 4.269 / 5.122 bits ⇒ **≈0.86 bits per well** |
| shipped `read` (argmax) in-corner shape balacc | **0.6759 ± 0.0107** (0.6610 / 0.6809 / 0.6858) |
| `proto` = the `knn_proto`-equivalent target | **0.8817 ± 0.0023** (0.8788 / 0.8844 / 0.8819) |
| `clu_wide` (full posterior mixture, 320 weights) | 0.7484 ± 0.0307 |
| store ceiling / decoder ceiling | 0.6866 ± 0.0681 / 0.9414 ± 0.0179 |

## ⭐ P0 — an ANALYTIC prediction, derivable before any run (and therefore a plumbing check if it fails)
The shipped commitment is `argmax_m P(m)` and `P ∝ exp(s/τ)` is a strictly monotone reparametrisation of `s` for
every `τ > 0`. **⇒ launch temperature cannot change the shipped argmax code at all.** Predicted: distinct-code
counts under `rule=argmax` are **bit-identical for every τ in the sweep**, on A, B and MS. τ can only act through
`sample` (it changes the draw distribution) and `margin` (it changes the top1−top2 gap).
⛔ If this fails, my tempering is buggy, not a finding.

## P1 — the posterior is concentrated but NOT degenerate
Mean top-1 confidence on B ∈ [0.78, 0.88]; mean per-well entropy ∈ [0.5, 1.5] bits; mean top1−top2 margin on B
**> 0.60**. ⇒ the diagnosis leg says stopping rule / multiplicity **can** move the code in principle (unlike a
top-1 ≈ 1.0 posterior, where nothing downstream of `fit_address_book` could move).

## P2 — THE MECHANICAL HYPOTHESIS (the task's, restated so it can fail)
`address_requests` is a naive-Bayes sum of per-factor log-conditional deltas. Montero's cell is a colour corner
(all three hues ≥ 0.8), so the three hue factors contribute an **identical** delta to all 72 B requests.
Predicted:
- **P2a** the three hue factors supply **≥ 60 %** of the total absolute per-well evidence `|log P(m|f) − log P(m)|`
  summed over factors, on B;
- **P2b** the across-request **variance** of the hue deltas on B is **zero** (they are literally constant), while
  shape/scale/orientation deltas vary;
- **P2c** the counterfactual "drop the three hue factors from the B request" raises B's distinct-code count from
  **9** to **≥ 40 of 72**.
⛔ If P2c fails, the corner's hues are *not* what is drowning the argmax and the diagnosis is wrong.

## P3 — the cheap sweep, per knob (distinct codes on B of 72, argmax baseline = 9)
| knob | setting | predicted B distinct | predicted MS distinct (of ~198) | predicted realised width |
|---|---|---|---|---|
| temperature (argmax) | τ ∈ {0.25, 0.5, 1, 2, 4, 8} | **9 for every τ** (P0) | flat | 30.0 bits |
| stopping = sample | τ = 1 | **≥ 55** | **≥ 150** (rises with B ⇒ ⛔ not a B-specific artefact) | 30.0 bits |
| stopping = sample | τ = 0.25 | 15–45 | rises | 30.0 bits |
| stopping = margin+abstain | δ ∈ {0.1 … 0.9} | **≤ 30** at every δ (abstention merges more tuples than it splits) | rises slightly | ≈30.1 bits (M+1 symbols) |
| multiplicity k = 2 | top-2 per well | 15–40 | rises | **54.9 bits ⇒ EXCEEDS the 30-bit design constant** |
| multiplicity k = 4 | top-4 per well | 30–60 | rises | **≈96 bits ⇒ EXCEEDS** |

## P4 — the accuracy leg (in-corner shape balanced accuracy, B vs MS, 3 seeds)
- **P4a** `sample` at τ = 1 **raises distinct count and LOWERS accuracy** to **< 0.60** (below the argmax 0.6759).
  ⭐ This is the task's named failure mode — *distinct count is trivially maximised by noise* — and I predict it
  fires.
- **P4b** `topk` (k = 2) lands **between** argmax (0.6759) and `clu_wide` (0.7484), i.e. in **[0.66, 0.76]**, and
  **below `proto` 0.8817**.
- **P4c** margin+abstain lands within ±0.03 of the argmax baseline.
- **P4d** ⭐ **NO configuration reaches the PASS.** Specifically: no configuration achieves ≥60 distinct on B *and*
  ≥ 0.8794 (= 0.8817 − 1 seed spread) balanced accuracy. **I pre-register the verdict FAIL.**

## P5 — the persisted-code rider (R29 Q2 / R30's owed number)
Predicted per seed: **|B-codes ∩ A-codes| ≥ 5 of 9** and **|B-codes ∩ MS-codes| ≥ 6 of 9** — i.e. the nine answers
the store gives to the seventy-two held-out questions are **mostly codes it also emits for SEEN requests**, which
is what "the corner collapses, not the hold-out" predicts.

## Shortlist rule (⛔ committed before the sweep is run)
The accuracy leg runs on **at most 4 configs**: the shipped `argmax` baseline **plus** the best setting per knob
(stopping rule · multiplicity · temperature), chosen **on the distinct-code statistic alone**, ties broken toward
the smaller realised channel width. The shortlist is written into this file as a dated addendum **before** the
first accuracy run.

---

# ADDENDUM — THE SHORTLIST, filed 2026-09-13 before the first accuracy run
**Chosen on the distinct-code statistic ALONE** (seed 0's 66-config sweep,
`runs/shapes3d_recon/seed_0/sweep_s0.json`, written before this addendum and before any decoder was run on any
committed code), ties broken toward the smaller realised channel width. ⛔ No accuracy number existed for any of
these configs when this list was fixed.

⚠ **Why the temperature knob has no slot of its own:** P0 was confirmed on the sweep — under the shipped
`argmax` commitment τ is a **measured no-op** (B = 9 at τ = 0.25 … 8, A = 494, MS = 35, bit-identical). τ can only
act through `sample`/`margin`, so its slot is spent on the second stopping-rule flavour, and each shortlisted
config carries its own best τ.

| # | config | knob it represents | B distinct (seed 0) | MS distinct | A distinct | realised width |
|---|---|---|---|---|---|---|
| 1 | `argmax\|tau=1\|margin=0\|k=1` | **the shipped baseline** (mandatory) | 9 / 72 | 35 / 198 | 494 / 512 | 30.0 bits |
| 2 | `sample\|tau=2\|margin=0\|k=1` | stopping rule — **sample**, best τ | **72 / 72** | 197 / 198 | 512 / 512 | 30.0 bits |
| 3 | `margin\|tau=2\|margin=0.25\|k=1` | stopping rule — **margin + abstain**, best (τ, δ) | 23 / 72 | 41 / 198 | 404 / 512 | 30.1 bits |
| 4 | `topk\|tau=1\|margin=0\|k=8` | **multiplicity**, best k | **72 / 72** | 189 / 198 | 512 / 512 | ⛔ **160.2 bits = 5.34 × the 30-bit design constant** |

**Pre-registered predictions for the accuracy leg (in-corner `shape` balanced accuracy, B vs MS, 3 seeds):**
- **A1** config 1 (baseline) reproduces `gen-read-compose`'s `read` arm to within seed noise: **0.6759 ± 0.0107**.
  ⛔ If it does not, my path differs from the shipped one and nothing below is quotable.
- **A2** ⭐ config 2 (`sample`, τ = 2) **clears the ≥60 distinct floor and FAILS on accuracy, below 0.60** — the
  noise failure mode the task is shaped to catch.
- **A3** config 3 lands within ±0.05 of the baseline.
- **A4** config 4 (k = 8) lands **above** the baseline and **below** `proto` 0.8817 — it is a 160-bit channel
  approaching `clu_wide`'s full mixture (0.7484), so I predict **[0.70, 0.80]**.
- **A5** ⭐ **no config PASSES.** Verdict predicted: **FAIL**.
- **A6** every config that lifts B also lifts MS (already visible in the sweep) ⇒ ⛔ **no B-specific artefact is
  expected**; the knobs move the corner, not the hold-out.
