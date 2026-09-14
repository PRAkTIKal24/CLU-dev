# PREREG — gen-decoder-lift
**Written BEFORE any decoder is trained and before any diagnostic is run.** (mtime is the evidence; the
harness `.claude/scratch/gen-decoder-lift/*.py` did not exist when this file was written.)

Agent: experiment-engineer · base `main @ fcdca43` · branch `agent/experiment-engineer/gen-decoder-lift`.

## 0. DIAL DECLARATION (echoed)
- **Dial:** none — instrument repair plus one gating measurement. No store, no well, no arm, no CLU number.
- **Laundering control:** the **matched-seen** control (R16) — same hue corner (`floor_hue,wall_hue,object_hue ≥ 0.8`,
  `scale ≥ 1.1`, `orientation > 20`), **seen** shape (`shape ≠ cylinder`), minus every row used in fitting —
  reported as a companion row beside every number.
- **Falsifies:** `shape` ratio still `< 0.8` after both variants ⇒ option 2 (R16) activates. Pre-registered outcome.
- **Does NOT falsify:** no movement on `scale` (already 1.007 vs matched-seen; photometric, nothing to fix);
  aggregate not moving (four of six factors sit at ≈1.0 and carry no compositional information).

## 1. The measured quantity and the verdict rule (fixed here, not after the fact)
`shape_ratio(variant, normaliser) = mean_over_seeds{0,1,2}[ shape_B(seed) ] / mean_over_seeds[ shape_A(seed, normaliser) ]`
where `shape_X` = probe top-1 accuracy on the `shape` factor of **real cached features decoded through the variant's
frozen decoder**, and the two normalisers are
- **random-A**: `eval_a`, 1024 rows drawn from the train split and held back from every fitting stage;
- **matched-seen**: the R16 control set above (n ≈ 209 at seed 0; per-seed n reported).

**Verdict rule, pre-committed: `CLEARS 0.8` iff the 3-seed mean ratio ≥ 0.8 under BOTH normalisers.**
If the two disagree, that is reported as a split verdict and read as DOES NOT CLEAR.

**Baseline to beat** (recomputed by me from the three shipped `bridge_manifest.json`, `main @ fcdca43`):

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| `shape_A` (random-A) | 0.9043 | 0.8945 | 0.9238 | **0.9076** |
| `shape_B` | 0.5000 | 0.5694 | 0.7083 | **0.5926** |
| ratio | 0.5528 | 0.6366 | 0.7667 | **0.6530** |
| `shape_A` (matched-seen) | 0.8756 | *not yet measured* | *not yet measured* | — |

## 2. Why a decoder-only fix is predicted NOT to clear 0.8 (the Hub's standing expectation, adopted)
The `cls` diagnostic pretext reaches `shape` 0.6944 / 0.8988 = **0.773** — the *whole* pipeline under a strictly more
favourable pretext, and still below 0.8. A decoder-only change is a strictly smaller intervention than swapping the
pretext. ⇒ **predicted verdict for both variants: DOES NOT CLEAR.**

**Additional evidence I derived from `matched_control.json` before choosing the variants, which CHANGED one of them:**
`object_bg_contrast` is **0.2208 on B and 0.2227 on A_matched** (vs 0.2974 on A_random) — i.e. B and the matched-seen
control are photometrically *identical*, yet `shape` is 0.500 vs 0.8756. ⛔ **Therefore a purely contrast-normalising
loss cannot be the fix, and I dropped the contrast-normalisation term I had drafted.** The residual `shape` gap is a
genuine (shape × scale) composition failure: the decoder has rendered small cylinders and large non-cylinders, never a
large cylinder, and falls back to a scale-appropriate generic blob. What a pixel-MSE loss never pays for is the
*silhouette identity* of that blob — the object is a small pixel fraction and a wrong-but-right-sized blob costs almost
nothing (measured: decode MSE is 0.0027 on B vs 0.0031 on A — B is **not** harder to reconstruct).

## 3. THE TWO VARIANTS — declared here, before any run. Exactly two. Both will be reported.
Both are refit from scratch per seed against the **frozen** encoder and the **unmodified** feature cache.
The reference row is the **shipped baseline decoder** (`.pkl` loaded, not retrained), plus one bit-identity
regression retrain at seed 0 whose `param_hash` must equal `47c0f83196cd220d…`.

### V1 — `capacity_schedule` (the null control: "is it just too small / undertrained?")
Architecture and schedule change only; loss stays plain uniform pixel MSE.
`dec_width_mult = 2.0` (channels `[128,64,32] → [256,128,64]`), `dec_blocks_per_stage = 2` (one extra 3×3
conv+GroupNorm+ReLU per upsampling stage; **skip-free**, as now), `dec_steps 1200 → 4800`,
`dec_lr_schedule = "cosine"` with 5 % linear warmup from `dec_lr = 1e-3` to 0.
*Why:* it is the only variant that can distinguish "the objective is wrong" from "the renderer is under-resourced",
and it is the reading a reviewer will demand first. Nothing here is width-dependent: `dec_width_mult` multiplies the
existing channel list, and the number of upsampling stages is derived from `feat_shape` at construction.

### V2 — `object_weighted` (the objective fix; the task's clue)
Architecture, step count and LR schedule **identical to the shipped baseline** so the comparison isolates the loss.
Loss (all terms image-derived — ⛔ no factor labels enter the generative path, R1):
```
y   = mean_channels(x)                       # target luminance
g   = |∂_h y| + |∂_w y|                      # forward differences, edge-replicated
s   = boxblur_{k=5}(g);  s ← s / (mean(s)+1e-8)   # per-image, mean 1
w   = 1 + alpha * s                          # alpha = dec_edge_weight = 8.0
L_w = mean_i[ Σ_p w_p (x̂-x)²_p / (C · Σ_p w_p) ]
L_g = mean(|∂_h x̂ - ∂_h x|² + |∂_w x̂ - ∂_w x|²)
L   = L_w + lambda_g * L_g                   # lambda_g = dec_grad_weight = 1.0
```
`w` is a function of the target only (no gradient path). *Why:* both terms pay for the object's **silhouette**, which
uniform pixel MSE does not. Nothing width-dependent (the loss is in pixel space).

⛔ I will not try a third variant. If neither clears, option 2 activates.

## 4. Point predictions (committed; ranges are 90 % intervals, not escape hatches)

| # | quantity | point | range |
|---|---|---|---|
| P1 | **V1 `shape` ratio, random-A, 3-seed mean** | **0.66** | 0.58 – 0.74 |
| P2 | **V1 `shape` ratio, matched-seen, 3-seed mean** | **0.68** | 0.59 – 0.78 |
| P3 | **V2 `shape` ratio, random-A, 3-seed mean** | **0.72** | 0.60 – 0.84 |
| P4 | **V2 `shape` ratio, matched-seen, 3-seed mean** | **0.75** | 0.62 – 0.87 |
| P5 | **verdict, both variants** | **DOES NOT CLEAR 0.8** | — |
| P6 | V1 plain decode MSE on A (baseline 0.0031) | **≤ 0.0020** | 0.0010 – 0.0025 |
| P7 | V2 plain decode MSE on A **rises** (it trades pixels for identity) | **≥ 0.0035** | 0.0030 – 0.0070 |
| P8 | aggregate (6-factor mean) moves by | **< 0.03** either way | — |
| P9 | `scale` on B, matched-seen ratio, both variants | **stays ≈ 1.0** | 0.90 – 1.10 |

### Diagnostic D — is the shape information even IN the frozen features? (run before the variants)
A factor classifier fit on the **cached train-split features** (frozen encoder, cache untouched) and read on the
**cached B features**. Two heads: linear, and a 1-hidden-layer MLP (512 units). Labels are measurement only (R1).

| # | quantity | point | range |
|---|---|---|---|
| D1 | linear-on-features `shape` accuracy on **B** | **0.75** | 0.45 – 0.93 |
| D2 | MLP-on-features `shape` accuracy on **B** | **0.85** | 0.55 – 0.98 |
| D3 | same, on A (sanity) | **≥ 0.97** | 0.93 – 1.00 |

*Derivation:* a classifier only has to separate silhouettes, a renderer has to draw one; and locally a large cylinder
looks like a small cylinder, so a classifier should generalise across the (shape × scale) product far better than a
decoder must. **Pre-registered decision rule:**
- **D2 ≥ 0.85** ⇒ the information is present and the decoder is discarding it ⇒ there is real headroom for V2.
- **D2 ≤ 0.65** ⇒ the frozen features have already lost it ⇒ ⛔ **no decoder can fix this**, and the finding I report
  is "the fix requires an encoder change — Head decision, not mine" (task's stop condition). ⭐ **Both variants are
  run at 3 seeds regardless**, so this rule changes the *interpretation*, never the *experiment*.
- 0.65 < D2 < 0.85 ⇒ partial; report as such.

## 5. Invariants asserted (not assumed) in every run
1. `param_hash(encoder)` equals the shipped `bridge_manifest.json:encoder_param_hash`, per seed.
2. `feature_cache.json:features_sha256` re-verified on load, per seed, and the file's mtime unchanged.
3. R3.3 no-leak assertion re-run on each variant's decoder-training rows; the count is reported.
4. Matched-seen rows are disjoint from both the fitting rows and the held-out rows (assertion, per seed).
5. The **probe is not retrained** — the shipped `probe.pkl` is the instrument for every row in every table.

---

## 6. AMENDMENT — 2026-09-10, **before any V1 or V2 run** (compute sizing only)
Nothing measured on the acceptance quantities existed when this was written: at this timestamp the only completed runs
are (a) scoring the **shipped** decoder (which reproduces `gen-feature-bridge` §6 exactly and is the reference row, not
a variant) and (b) the in-flight baseline **bit-identity** retrain. Diagnostic D and both variants are unrun.

**Measured cost:** the baseline decoder (1200 steps, batch 128, 8192 train pairs) takes **≈19 min** on this CPU with the
test suite running concurrently (240 steps / 3.5–4.3 min, logged). V1 as pre-registered (`width 2.0`, `blocks 2`,
`4800 steps`) is **6.69× the baseline per-step MAC count** (hand-counted per conv: 592 M vs 88.4 M MAC/image) **× 4×
the steps = 27× baseline ≈ 8.5 h/seed ≈ 25 h for three seeds** — outside the Head's one-day bound for this whole task.

⇒ **V1's step count is cut from 4800 to 1800** (cosine + 5 % warmup, decaying to 0, unchanged). Architecture unchanged
(`width 2.0`, `blocks_per_stage 2`). New cost ≈ 10× baseline ≈ 3 h/seed contended. The "is it undertrained?" question is
then answered from the **reported training-loss curve** (V1's final loss vs the baseline's 0.00311) rather than assumed.
⛔ **No prediction changes:** P1/P2 already predicted no movement, and a smaller step budget cannot make the
pre-registered "no movement" easier to confirm in the direction that matters (it can only make V1 *weaker*, i.e. it
biases against the null control, which is the conservative direction for the conclusion "capacity is not the problem").
⚠ If V1 lands near its predicted value, this amendment is a caveat on the claim "capacity was ruled out", and it is
stated as such in the report.
