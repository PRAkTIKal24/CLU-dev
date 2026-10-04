# gen-variance-equalisation — experiment-engineer report

**Task + acceptance criterion:** does k-means drop `scale` because it is LOW-VARIANCE? — test the premise first
(Leg 1), then equalise per well and re-fit (Leg 2), report all six factors before/after with a TOTAL-RETAINED line
(Leg 3) against a pre-registered per-well capacity bound (Leg 4); acceptance = both pre-registered falsifiers
answered yes/no, `knn_proto` bit-identical, byte ledger in bytes.
**Status: done.**

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 5 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⭐⭐ **THE PREMISE IS TRUE AND THE REPAIR IS SELF-DEFEATING — the variance account survives but is no longer a
>    route to a fix.** `scale`/`orientation` *do* sit in low-variance directions (Leg 1, 2/3 seeds), and equalising
>    them **destroys the code**: TOTAL-RETAINED **3.678 ± 0.075 → 1.172 ± 0.076**, distinct codes on 4,096 rows
>    **3,638 → 626**. ⛔ **No draft may say "whitening / variance equalisation is the fix for the `scale` collapse."**
>    **Owner: Hub, before the R40.D design note is re-read.**
> 2. ⭐⭐ **THE MECHANISM, measured: the well features have EFFECTIVE DIMENSION ≈ 3–14, not 283–744**, and that is the
>    *only* reason a 64-word codebook compresses at all. Variance-weighted `M^(−2/PR)` predicts the unwhitened
>    relative read error at **0.2533 ± 0.0217** against a measured **0.2783 ± 0.0064**, and the whitened one at
>    **0.9787 ± 0.0006** against **0.9336 ± 0.0057** — parameter-free, both seeds-consistent.
>    ⇒ **k-means' reliance on high variance is not a bug to be removed; it is the compression.**
>    **Owner: Hub, into the design note's "why a bigger `M`, not a better metric" section.**
> 3. ⚠ **THE HUB's 6.87× BYTE FIGURE IS SEED-0-SPECIFIC.** Well sizes differ per seed, so per-well PCA whitening
>    costs **3,600,672 / 3,949,768 / 3,681,432 B = 6.87× / 7.53× / 7.02×**, mean **7.14×**. ⭐ But it is **fit-time
>    only (0 deployed bytes) on the request-side path** — *proved*, not claimed (max rel Δ = **0.000e+00**, 9/9
>    arm-seeds). **Owner: Hub, at the next byte-ledger review.**
> 4. ⭐ **THE SECONDARY ARM SPLIT THE FALSIFIERS AND THAT IS A FINDING, NOT A WIN.** Diagonal standardisation
>    (**16,384 B**, 3.1 % of budget) **doubles the distinct configuration count on the 72 held-out requests,
>    9.00 ± 0.82 → 15.00 ± 2.94** — the first thing in this phase to move that number *upward* — and raises
>    `orientation` retention 0.2425 → 0.2985, while **losing** `object_hue` (0.665 → 0.463) and `shape`
>    (0.555 → 0.423). ⛔ It is **not** the pre-registered pass (`scale` did not move) and ⛔ must not be promoted to
>    primary. **Owner: Hub — decide whether "distinct-of-72" is worth a follow-up arm at 3 % of budget.**
> 5. ⚠ **`gen-loss-attribution`'s ladder is REPRODUCED BIT-FOR-BIT by the `identity` control**, including its
>    distinct-code counts **3,598 / 3,594 / 3,723**. ⇒ the two spokes' numbers are interchangeable and any future
>    table may cite either. **Owner: Hub, at the next table review.**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *feature scaling before the store.* ⛔ The encoder, the decoder, the **partition** and `K` were all
  UNCHANGED — loaded, hash-asserted, and (for the partition) asserted **channel-for-channel**, not merely by size.
- **Laundering control:** ⭐ **`knn_proto` bit-identical.** ⭐ **It was: `max |Δ| = 0.0`, feature-array and
  prototype-embedding sha256 identical, 3/3 seeds.**
- **Falsifies the claim:** (a) the `scale`+`orientation` probe on the store code rises beyond seed spread; (b) the
  distinct configuration count rises. **Both were answered. Both are NO for the primary arm.**
- **Does NOT falsify:** a whitened arm that *trades* hue retention for `scale`/`orientation` retention at a flat
  total — the pre-registered CAPACITY reading. ⛔ **That is not what happened either**: the total collapsed.

## ⭐⭐ THE RESULT — one line, in the pre-registered words

⭐ **The premise HOLDS (2 of 3 seeds) and the intervention FAILS: falsifier (a) = NO, falsifier (b) = NO, the
pre-registered pass = NO.** Per-well PCA whitening lowers **every one of the six factors**, takes TOTAL-RETAINED from
**3.678 ± 0.075 to 1.172 ± 0.076**, and takes the distinct code count on 4,096 held-out rows from **3,638 ± 60 to
626 ± 126**. ⛔ **The variance hypothesis is confirmed as a DESCRIPTION and refuted as a REPAIR.**

⛔ **This is not the pre-registered flat-total/capacity reading.** `Δ TOTAL = −2.506` against a seed spread of
`0.0755` — the metric did not merely redistribute the modes, it **stopped the codebook working at all**.

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | **`7a88754`** (branch tip). ⚠ Per-seed artefacts were produced at **`cb879e6`**; `48fb2cb` and `7a88754` changed only `summarize`/the gate scope and the summary was **regenerated from the unchanged per-seed JSONs** (`skip_if_exists` HIT, 3/3). No number below was produced by a different tree than the one that computed its inputs. |
| branch / base | `agent/experiment-engineer/gen-variance-equalisation` off **`main @ 81ba81b`** |
| worktree / venv | `../CHLU-gen-varianceeq`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero). Grid 480,000 = 10·10·10·8·4·15 |
| seeds | **0, 1, 2**; `±` is **population sd, ddof = 0** |
| encoder · decoder · partition · `K` · `M` | ⛔ **ALL LOADED AND UNCHANGED.** Encoder hash asserted vs `bridge_manifest.json` **and** the feature-cache manifest; cache re-encode check **64 rows bit-identical**, 3/3. Decoder = `gen-decoder-lift`'s **`v1_capacity_schedule`** winner, `param_hash` **and recipe name** asserted (⛔ never `dec_recipe="baseline"`). Partition = `gen-commitment-collapse`'s `labels_vec`, asserted channel-for-channel against the shipped store. `K=5`, `M=64`, **30 bits/read** |
| ⭐ what DID change | **only the metric the per-well k-means sees.** Arms `identity` (control) · `pca_whiten` (primary) · `diag_std` (secondary), plus the `shipped` reference store |
| store re-fit | `init_store` (k-means++) + `train_store`, **`train_steps=1200`, `train_batch=128`, `train_lr=3e-3`, `commitment_beta=0.25`, `tau 1.0 → 0.1`**, `n_store_train=32768`, PRNG `PRNGKey(seed·10007 + 20260911)` — ⛔ **identical across arms**, exactly `gen-read-compose`'s recipe |
| rows | `store_train` **32,768** · `SPREAD_train` **12,288** · `SPREAD` (test) **4,096** · `CORNER_B` **72** · `CORNER_MS` **198/199/199**; the SPREAD draw is `exp_gen_attribution`'s RNG stream verbatim (`default_rng(seed·7919 + 20260914)`) |
| probe (⛔ identical at every point, every arm) | `WellProbe`, `probe_steps=3000`, `probe_batch=512`, `probe_lr=3e-3`, key `PRNGKey(20260914 + seed)`; the **encoder point is fitted once and shared by every arm** |
| Leg 1 | `spectrum_rows=16384`, a **seeded permutation** of `store_train` (⛔ never a prefix — see §7 L1); between-class variance **null-subtracted**, `premise_rank_ratio=2.0`, `premise_share_ratio=0.5` |
| transform | `whiten_eps=1e-6` (relative), **`scale_match_to_raw=True`** (matched-compute control, addendum 1), `round_trip_tol=1e-2` |
| decoded column | `decode_rows=1024`, `decode_batch=256` |
| ⛔ NOT used | `soft_read`, `DreamLandscape`, `dream_rollout`, the inter-well coupling, the external classifier, CSF3 |
| wall clock | **2,446 / 1,776 / 1,546 s** per seed (seeds 1–2 re-encode ~48k rows through the frozen encoder). CPU only, 16 GB |

**Pre-registration:** `.claude/outputs/gen-variance-equalisation/PREREG.md`, mtime **08:55:22**, ⛔ never edited.
Two addenda are **NEW FILES** (R34's seventh strike), both filed **before** the runs they govern:
`PREREG_ADDENDUM_1.md` **09:01:03** (the matched-compute scalar), `PREREG_ADDENDUM_2.md` **09:06** (the
null-subtraction), `PREREG_ADDENDUM_3.md` **11:21** (the spoke-level gate, filed **before** seed 2 was re-run).
Production artefacts `veq_s0.json` **10:47** → `veq_s1.json` **11:17** → `veq_s2.json` **11:47**.
⇒ **every number postdates its prereg on disk.**

---

## 1. ⛔ LEG 1 — THE PREMISE, TESTED BEFORE IT WAS ACTED ON

Each factor in **its own headline well** (`argmax_k variance_explained`), on 16,384 store-training rows, 3 seeds.
`VE` = share of the well's total variance the factor's class means account for. `rank_frac_median` = the normalised
principal-axis rank at which **half** the factor's between-class variance has accumulated — ⭐ **the literal
"does it live in low-variance directions?" number.** Both are **null-subtracted** (addendum 2).

| factor | n_val | **VE** | **`rank_frac_median`** | frac of signal in top 10 % of axes | var frac needed for half the signal |
|---|---|---|---|---|---|
| `floor_hue` | 10 | **0.7500 ± 0.0589** | **0.00342 ± 0.00060** | 0.999 | 0.477 |
| `wall_hue` | 10 | **0.6804 ± 0.0250** | **0.00336 ± 0.00013** | 0.997 | 0.525 |
| `object_hue` | 10 | 0.3094 ± 0.0092 | 0.00331 ± 0.00053 | 0.997 | 0.339 |
| **`scale`** | **8** | **0.0236 ± 0.0025** | **0.01358 ± 0.00458** | 0.984 | 0.628 |
| `shape` | 4 | 0.0503 ± 0.0110 | 0.00773 ± 0.00159 | 0.995 | 0.535 |
| **`orientation`** | **15** | **0.0995 ± 0.0406** | **0.00846 ± 0.00237** | 0.981 | 0.535 |

⭐ **The premise is real and the size of the effect is large.** The two hues account for **68–75 %** of their well's
variance; `scale` accounts for **2.4 %** — a **31×** gap. And `scale`'s signal sits **4.0×** deeper in the spectrum
than `floor_hue`'s, `orientation`'s **2.5×** deeper.

### ⛔ The verdict, and it is NOT unanimous

| seed | verdict | direction test | reference `rank_frac_median` (max) | `scale` | `orientation` |
|---|---|---|---|---|---|
| 0 | **HOLDS** | pass | 0.00328 | 0.01996 | 0.01047 |
| 1 | **HOLDS** | pass | 0.00333 | 0.00941 | 0.00977 |
| **2** | ⛔ **DEAD** | **fail** | 0.00420 | 0.01136 | **0.00513** |

⭐ **Spoke-level verdict under PREREG §3's own wording ("≥ 2 of 3 seeds"): HOLDS, 2/3.**
⛔ **Seed 2 fails on `orientation` alone** (`0.00513` against a `2 × 0.00420 = 0.00840` bar) — on that seed
`orientation` carries **0.1558** of its well's variance, 2.5× its cross-seed mean, i.e. the encoder gave it a
**high**-variance direction. ⚠ `orientation`'s `VE` has the largest seed spread of any factor
(**0.0995 ± 0.0406**, sd = 41 % of the mean). **The premise is a property of `scale` far more robustly than of
`orientation`.**

⚠ **The harness first applied the stop rule per seed** (stricter than the rule filed), stopped seed 2 at Leg 1, and
left it without arms. That was corrected to the filed ≥2-of-3 rule, declared in `PREREG_ADDENDUM_3.md` **before**
seed 2 was re-run, and the fix shipped as commit `48fb2cb`. **Seed 2's DEAD verdict is in every table.**

## 2. ⭐⭐ LEG 3 — THE SIX-FACTOR GRADED PROFILE, BEFORE AND AFTER, WITH THE TOTAL

SPREAD grid, 3 seeds. Retention = `(acc − chance)/(acc_encoder − chance)`; ⛔ the encoder point is **one probe**,
shared by every arm. ⛔ Degenerate factors are excluded from the total and the count travels with it.

| factor | n_val | chance | encoder | **`identity`** (control) | ⭐ **`pca_whiten`** (primary) | **`diag_std`** (secondary) |
|---|---|---|---|---|---|---|
| `floor_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | **0.997 ± 0.002** | 0.227 ± 0.025 | 0.990 ± 0.012 |
| `wall_hue` | 10 | 0.100 | 0.9997 ± 0.0001 | **0.995 ± 0.004** | 0.372 ± 0.035 | 0.991 ± 0.001 |
| `object_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | **0.665 ± 0.043** | 0.155 ± 0.031 | 0.463 ± 0.057 |
| **`scale`** | **8** | 0.125 | 0.9352 ± 0.0058 | **0.224 ± 0.004** | **0.086 ± 0.012** | 0.218 ± 0.013 |
| `shape` | 4 | 0.250 | 0.9960 ± 0.0008 | **0.555 ± 0.034** | 0.208 ± 0.022 | 0.423 ± 0.029 |
| **`orientation`** | **15** | 0.067 | 0.9915 ± 0.0018 | **0.243 ± 0.042** | **0.124 ± 0.020** | ⭐ **0.299 ± 0.040** |
| ⭐⭐ **TOTAL RETAINED** | — | — | — | ⭐ **3.678 ± 0.075** | ⛔ **1.172 ± 0.076** | **3.383 ± 0.035** |

⭐⭐ **TRADE vs REDUCTION, in words.**
- **`pca_whiten` is a REDUCTION, not a trade.** ⛔ **Every single factor falls** — including the two the intervention
  was designed to rescue. There is no redistribution to discuss.
- **`diag_std` is a genuine TRADE with a net loss.** `orientation` **+0.056** (the only factor to rise anywhere in
  this spoke) and `scale` **−0.006** (flat), paid for by `object_hue` **−0.202** and `shape` **−0.132**. Net
  **−0.295 ± 0.083**, i.e. **−8 %**. ⛔ Not flat (|Δ| > seed spread), so the pre-registered *"flat total ⇒ capacity is
  binding"* reading **does not fire literally** — but the *shape* of the diagonal arm's result (one factor up, two
  down, small net loss) is the trade-not-gain pattern that reading anticipates.

⭐ **The control is not a claim — it is a bit-level reproduction.** `identity` and `shipped` agree **to every printed
digit** on all 6 factors, on all 3 row sets, on the MI table, on the distinct counts (**3,598 / 3,594 / 3,723**, i.e.
`gen-loss-attribution`'s own figures) and on the decoded column. `gen-loss-attribution`'s ladder
(0.997 · 0.995 · 0.665 · 0.555 · 0.243 · 0.225) is **re-measured here, not transcribed**, and reproduces.

## 3. ⭐⭐ THE TWO PRE-REGISTERED FALSIFIERS, ANSWERED IN THOSE WORDS

**(a) Does the `scale` + `orientation` probe on the store code rise above the unwhitened baseline beyond seed
spread? → ⛔ NO.**

| factor | `identity` | `pca_whiten` | Δ | seed spread | rose? |
|---|---|---|---|---|---|
| `scale` | 0.2245 ± 0.0036 | 0.0861 ± 0.0119 | **−0.1383** | 0.0119 | ⛔ **NO** (it fell 11.6 spreads) |
| `orientation` | 0.2425 ± 0.0424 | 0.1238 ± 0.0202 | **−0.1187** | 0.0424 | ⛔ **NO** (fell 2.8 spreads) |

**(b) Does the distinct configuration count rise? → ⛔ NO.**

| count | `identity` | `pca_whiten` | Δ | seed spread |
|---|---|---|---|---|
| ⭐ **request-side distinct of 72 (CORNER_B)** — PRIMARY | **9.00 ± 0.82** | 5.67 ± 0.94 | **−3.33** | 0.94 |
| feature-side distinct codes / 4,096 SPREAD rows — secondary | 3,638.3 ± 59.9 | 626.0 ± 126.4 | **−3,012.3** | — |

⭐ **PRE-REGISTERED PASS (BOTH): ⛔ NO.** ⭐ **The two falsifiers agree — this is not the "one moved, the other did
not" split the task warned about.** Both moved, and both moved the **wrong way**.

### 3b. ⭐ The secondary arm SPLITS them — reported, ⛔ not promoted

| quantity | `identity` | `diag_std` | Δ | seed spread | rose? |
|---|---|---|---|---|---|
| retention `orientation` | 0.2425 ± 0.0424 | 0.2985 ± 0.0403 | **+0.0560** | 0.0424 | ⭐ **YES** |
| retention `scale` | 0.2245 ± 0.0036 | 0.2181 ± 0.0125 | −0.0064 | 0.0125 | **NO** |
| ⭐ **distinct of 72 (CORNER_B)** | **9.00 ± 0.82** | **15.00 ± 2.94** | **+6.00** | 2.94 | ⭐ **YES** |

⇒ **falsifier (a) = NO** (it requires **both** factors; `scale` did not move), **falsifier (b) = YES**,
**pre-registered pass = NO.** ⛔ **This is exactly the "probe rises but the distinct count does not (or vice versa)"
split the task names as *not* the pass**, and it is reported as the split it is. ⭐ **But the distinct-of-72 result is
substantive on its own terms: 9 → 15 of 72, at 16,384 B (3.1 % of the store budget).** Nothing else in this phase has
moved that number upward.

## 4. ⭐⭐ THE MECHANISM — why equalisation cannot work here (⚠ post-hoc, NOT pre-registered)

⛔ **Stated as an explanation formed after the measurement, with its evidence, not as a pre-registered prediction.**

A vector quantiser with `M` codewords on a source of **effective dimension** `PR = (Σλ)²/Σλ²` has, by high-rate
theory, a distortion ratio ≈ `M^(−2/PR)`. The shipped wells are **extremely anisotropic**:

| | `identity` | `pca_whiten` |
|---|---|---|
| per-well effective dimension `PR` | **3.12 … 13.63** (of `d_k` = 283…744) | **= `d_k` by construction** |
| variance-weighted `64^(−2/PR)` — **predicted** relative read error | **0.2533 ± 0.0217** | **0.9787 ± 0.0006** |
| **measured** relative read error (original feature space) | **0.2783 ± 0.0064** | **0.9336 ± 0.0057** |

⭐ **A parameter-free formula predicts both arms' read error to within 10 % and 5 %.** Whitening raises the effective
dimension from ~3–14 to 283–744 **by definition**, and 64 codewords in a 500-dimensional isotropic ball capture
essentially nothing.

⇒ ⭐⭐ **k-means' preference for high-variance directions is not a bug in the objective — it IS the compression.**
The anisotropy that makes `scale` cheap to discard is the same anisotropy that makes a 64-word codebook viable at
all. ⛔ **You cannot remove one without the other by rescaling the metric.**

## 5. ⭐⭐ LEG 4 — THE PER-WELL CAPACITY BOUND (pre-registered before measuring)

`K·log₂M` = **30.00 bits** vs `log₂(480,000)` = **18.87 bits** ⇒ ⭐ **the code is SUFFICIENT in aggregate**, and
⛔ **"30 bits" may not be written as the cause** (R42.2 re-affirmed; distinctness is 3,638 of 4,096).
**Per well it is not.** A well *carries* a factor iff `MI(code_k; f)/H(f) ≥ 0.25`; its **load** is the product of the
carried cardinalities, against `M = 64`.

| arm | wells over capacity (of 5) | per-well loads, seed 0 | what well 0 must resolve, seed 0 |
|---|---|---|---|
| `identity` / `shipped` | **4.67 ± 0.47** | **[1500, 150, 100, 400, 100]** | `floor_hue × wall_hue × orientation` = **1,500 vs 64** |
| `diag_std` | **4.33 ± 0.47** | [150, 150, 10, 100, 100] | `wall_hue × orientation` = 150 vs 64 |
| `pca_whiten` | **0.00 ± 0.00** | [1, 1, 1, 1, 1] | ⛔ **nothing** — no well clears the 0.25 MI bar at all |

⭐ **The shipped store is over its per-well capacity in 4.67 of 5 wells, by up to 23×.** ⛔ `pca_whiten`'s "0 wells
over capacity" is **not** a success: its wells carry *no* factor above threshold. The bound is read as
"how much does a well have to resolve", never as a score.

⛔ **The pre-registered flat-total reading does NOT fire.** `Δ TOTAL = −2.506` against spread `0.0755`. ⭐ **But the
capacity conclusion survives by a different and stronger route:** the metric arm shows that redistributing modes is
not even reachable — the intervention that was supposed to move the distribution destroyed the code. ⇒ **the repair
still goes to `M` (or to a different representation), never to the metric.**

## 6. ⭐ THE BYTE LEDGER, AND THE FIT-TIME-vs-DEPLOYED DECLARATION

⭐⭐ **BOTH WERE BUILT, AND BOTH ARE PRICED. The primary arm is FIT-TIME ONLY on the request-side path — and that is
PROVED, not asserted.** The codebook is fitted in the whitened metric and the `M` centres are then mapped back
through the inverse and stored **un-whitened** (`unwhiten_store`). The deployed cell read is
`request → address book → code → read_from_codes`, a plain gather that **never evaluates the transform**.
`assert_deployment_equivalence` requires `inverse(read(store_whitened, c)) == read(store_unwhitened, c)`:
**max relative Δ = 0.000e+00 on 9/9 arm-seeds.**
⛔ **Any path that maps a FEATURE VECTOR to a code at read time (`address_codes`) must evaluate the forward map and
pays the table below.**

| transform | formula | seed 0 | seed 1 | seed 2 | **vs the 524,288 B budget** | fit-time-only |
|---|---|---|---|---|---|---|
| `identity` (control) | — | **0** | **0** | **0** | 0× | 0 |
| `diag_std` (secondary) | `2·D·4` | **16,384** | **16,384** | **16,384** | **3.1 %** | 0 |
| ⭐ `pca_whiten` (primary) | `Σ_k d_k²·4 + D·4` | **3,600,672** | **3,949,768** | **3,681,432** | ⛔ **6.87× / 7.53× / 7.02×** (mean **7.14×**) | **0** |
| `pca_whiten` + explicit inverse | `+ D·4` | 3,608,864 | 3,957,960 | 3,689,624 | 6.88× / 7.55× / 7.04× | 0 |
| *(full-`D` whitening — ⛔ not the ruled arm)* | `D²·4` | *16,777,216* | *16,777,216* | *16,777,216* | *32×* | — |

⚠ **The Hub's 6.87× is the seed-0 number.** `Σ d_k²` is **898,120 / 985,394 / 918,310** — the well sizes differ per
seed, so the deployed price does too. ⭐ The seed-0 arithmetic (`898,120 × 4 + 2,048 × 4 = 3,600,672`) is re-derived
in `tests/test_gen_whiten.py`, ⛔ never hand-typed.
⛔ **The store itself is unchanged at 524,288 B in every arm** (`M·D·4`), as it must be — the transform is invertible
and preserves dimension.

**Round trip** `inverse(forward(x)) ≈ x`, worst observed over 2,048 held-out rows:
**`identity` 0.000e+00 · `pca_whiten` 2.384e-07 (rel 4.07e-08) · `diag_std` 2.384e-07 (rel 3.89e-08)** — float32
round-off, 3/3 seeds. (PREREG P12 predicted ≤ 1e-2: ⭐ held with five orders of magnitude to spare.)

## 7. ⭐ THE LAUNDERING CONTROL — `knn_proto` DID NOT MOVE

| check | result |
|---|---|
| sha256 of **every feature array**, before vs after the whitening pipeline | **identical, 3/3 seeds** |
| sha256 of the **`FactorPrototypes` embedding** of `SPREAD` and `CORNER_B` | **identical, 3/3 seeds** |
| `max |Δ|` on the embedding | **0.0, 3/3 seeds** |
| source: lines changed on the `knn_proto` path (`gen_request.py`, `gen_compose.py`, `exp_gen_cell.py`, `exp_gen_baselines.py`, `chlu/eval/**`) | **0** |

⭐ **The numerical check is the one that bites:** a source diff cannot see an **in-place mutation of the shared
feature array**, which is the realistic way a transform escapes its scope. It was taken before *and* after.
⚠ **Declared limit:** the *shipped* `knn_proto` balanced accuracy is a decoder+classifier cell result this spoke does
not re-run. What is proved is that its **inputs and its embedding are byte-identical** and that no file on its path
changed.

## 8. ⭐ WHY THE RULED ORDER MATTERS — measured, not argued

R43.1 rules that the grouping is fitted on **unwhitened** features **first**. The reason is measurable: per-well
whitening makes every within-well correlation the identity, destroying the structure the grouping was fitted on.

| seed | in-block \|corr\| fraction, unwhitened → whitened | **lift over a size-matched random partition**, unwhitened → whitened |
|---|---|---|
| 0 | 0.4030 → 0.1751 | **1.886 → 0.819** |
| 1 | 0.4208 → 0.1745 | **1.794 → 0.744** |
| 2 | 0.4035 → 0.1808 | **1.846 → 0.827** |

⭐ **A grouping fitted after whitening would have been WORSE THAN RANDOM** (lift < 1, 3/3 seeds) — and would have
moved the one component `gen-loss-attribution` exonerated. `assert_grouping_is_unwhitened` makes the order an
assertion in code, not a comment.

## 9. The decoded column — ⛔ through the WINNER decoder, on the INVERTED read

`gen-decoder-lift`'s `v1_capacity_schedule`, `param_hash` **and** recipe name asserted per seed; 1,024 SPREAD rows;
every read mapped **back** through the inverse first (⛔ a decoder fed whitened vectors measures the transform).

| arm | aggregate | `floor_hue` | `wall_hue` | `object_hue` | `scale` | `shape` | `orientation` |
|---|---|---|---|---|---|---|---|
| `identity` / `shipped` | **0.6225 ± 0.0124** | 0.998 | 0.989 | 0.686 | 0.208 | 0.599 | 0.255 |
| ⭐ `pca_whiten` | ⛔ **0.1952 ± 0.0318** | 0.215 | 0.201 | 0.208 | 0.130 | 0.315 | 0.102 |
| `diag_std` | 0.5642 ± 0.0018 | 0.993 | 0.978 | 0.474 | 0.202 | 0.476 | **0.263** |

⭐ **The pixels agree with the probe**: the whitened read does not render recognisable Shapes3D at all. ⛔ No arm is
scored on a whitened vector anywhere.

## 10. Pre-registered predictions vs measurement (verbatim ids)

| id | predicted | 80 % interval | measured | verdict |
|---|---|---|---|---|
| **L1** | `VE(floor_hue)` 0.28 | [0.10, 0.55] | **0.7500** | ⛔ FALSIFIED — above. Hue dominance is far stronger than I guessed |
| **L2** | `VE(wall_hue)` 0.26 | [0.10, 0.55] | **0.6804** | ⛔ FALSIFIED — above |
| **L3** | `VE(object_hue)` 0.10 | [0.02, 0.30] | **0.3094** | ⛔ FALSIFIED — marginally above |
| **L4** | `VE(shape)` 0.05 | [0.01, 0.15] | **0.0503** | ⭐ **HELD — to three decimals** |
| **L5** | ⭐ `VE(scale)` 0.02 | [0.003, 0.08] | **0.0236** | ⭐ **HELD** |
| **L6** | ⭐ `VE(orientation)` 0.02 | [0.003, 0.08] | **0.0995** | ⛔ FALSIFIED — above; `orientation` is *not* as low-variance as `scale` |
| **L7** | `rank_frac_median(floor_hue)` 0.006 | [0.001, 0.03] | **0.00342** | ⭐ HELD |
| **L8** | `rank_frac_median(wall_hue)` 0.006 | [0.001, 0.03] | **0.00336** | ⭐ HELD |
| **L9** | ⭐ `rank_frac_median(scale)` 0.05 | [0.01, 0.30] | **0.01358** | ⭐ HELD (low end) |
| **L10** | ⭐ `rank_frac_median(orientation)` 0.08 | [0.01, 0.40] | **0.00846** | ⛔ FALSIFIED — marginally **below** |
| **L11** | premise verdict | — | **HOLDS, 2/3 seeds** | ⭐ HELD, ⚠ not unanimous |
| **P1** | `identity` TOTAL 3.68 | [3.2, 4.1] | **3.678** | ⭐⭐ **HELD — to two decimals** |
| **P2** | ⭐ `pca` `scale` retention 0.38 | [0.20, 0.62] | **0.086** | ⛔ **FALSIFIED — far below** |
| **P3** | ⭐ `pca` `orientation` 0.40 | [0.20, 0.65] | **0.124** | ⛔ **FALSIFIED — far below** |
| **P4** | `pca` `floor_hue` 0.90 | [0.60, 0.999] | **0.227** | ⛔ FALSIFIED below |
| **P5** | `pca` `wall_hue` 0.90 | [0.60, 0.999] | **0.372** | ⛔ FALSIFIED below |
| **P6** | `pca` `object_hue` 0.55 | [0.30, 0.80] | **0.155** | ⛔ FALSIFIED below |
| **P7** | `pca` `shape` 0.50 | [0.25, 0.75] | **0.208** | ⛔ FALSIFIED below |
| **P8** | ⭐⭐ `pca` TOTAL 3.70 | [3.1, 4.4] | **1.172** | ⛔ **FALSIFIED — the central failure of my prediction** |
| **P9** | ⭐⭐ TOTAL change **+0.02, "FLAT"** | [−0.5, +0.7] | **−2.506** | ⛔ **FALSIFIED — 3.6× outside the interval** |
| **P10** | `diag_std` TOTAL change +0.01 | [−0.3, +0.4] | **−0.295** | ⭐ HELD (at the very edge) |
| **P11** | `pca` read MSE worse | ratio [1.0, 3.0] | **3.36×** | ⚠ **direction HELD, magnitude FALSIFIED** (just above) |
| **P12** | round-trip ≤ 1e-2 | [0, 1e-1] | **2.38e-07** | ⭐ HELD |
| **P13** | `identity` distinct-of-72 = 9 | [4, 20] | **9.00 ± 0.82** | ⭐⭐ **HELD — exactly** |
| **P14** | ⭐ `pca` distinct-of-72 = 16 | [6, 40] | **5.67** | ⛔ FALSIFIED — below |
| **P15** | `pca` distinct codes / 4,096 = 3,600 | [3,000, 4,000] | **626** | ⛔ **FALSIFIED — 4.8× below** |
| **P16** | wells over capacity, `identity` = 2.0 | [0, 4] | **4.67** | ⛔ FALSIFIED — above |
| **P17** | ⭐ same, `pca` = 3.0 | [1, 5] | **0.00** | ⛔ FALSIFIED — below (and for a reason that is not a win) |
| **A1** (add. 3) | seed-2 `pca` distinct < 1,500 and TOTAL > 1.0 below `identity` | — | **559**, Δ = **−2.55** | ⭐ HELD |
| **A2** (add. 3) | seed-2 `identity` distinct = **3,723**, row identical to `shipped` | — | **3,723**, identical | ⭐⭐ **HELD — exactly** |
| **A3** (add. 3) | falsifier answers do not change sign | — | they did not | ⭐ HELD |
| **AD1** (add. 1) | ⭐ unmatched `pca` TOTAL **worse** than matched | 0.2–1.5 lower | **0.4710 lower** (0.7812 vs 1.2522) | ⭐ **HELD** |

⭐ **14 held, 17 falsified, 1 mixed (P11: direction right, magnitude wrong) — and the falsified ones are the
result.** ⛔ **I predicted the intervention would work
(P2/P3/P8/P9: a trade at a flat total). It does the opposite, and it does so by a margin no reading of the numbers
can soften.** The two predictions that held *exactly* (P1 3.678, P13 = 9, A2 = 3,723) are the **control** ones — the
baseline is reproduced bit-for-bit, which is precisely what makes the primary arm's failure credible.

## 11. ⭐ THE CONFOUND CONTROL — the matched-compute scalar did NOT rescue the whitened arm

`PREREG_ADDENDUM_1.md` flagged that `train_store`'s `τ` is in **absolute distance units**, so an unmatched whitened
arm would see a different anneal. The matched-compute scalar (metric-neutral: k-means is equivariant under a uniform
scaling) was applied to every arm. ⭐ **It was then removed deliberately, seed 0, separate `--out`, to price the
confound:**

| retention, SPREAD, seed 0 | `identity` | ⭐ `pca_whiten` **matched** | `pca_whiten` **unmatched** |
|---|---|---|---|
| `floor_hue` | 0.9951 | 0.2620 | 0.1502 |
| `wall_hue` | 0.9957 | 0.3569 | 0.2364 |
| `object_hue` | 0.7213 | 0.1987 | 0.1233 |
| `scale` | 0.2221 | 0.0902 | 0.0547 |
| `shape` | 0.5338 | 0.2383 | 0.1396 |
| `orientation` | 0.1909 | 0.1061 | 0.0771 |
| **TOTAL RETAINED** | **3.6588** | **1.2522** | **0.7812** |
| distinct of 72 / distinct of 4,096 | 9 / 3,598 | 5 / 803 | 4 / 399 |
| relative read MSE | 0.2784 | 0.9293 | 0.9522 |

⭐ **The control worked and it did not save the arm.** Matching costs the unmatched arm **0.4710** of TOTAL-RETAINED
— inside addendum 1's pre-registered [0.2, 1.5] band (**HELD**) — but the **matched** arm still sits at **1.2522
against the control's 3.6588**. ⛔ **Of the unmatched arm's total deficit (2.8776 below the control), the τ-schedule/learning-rate confound
accounts for 0.4710 = 16 % and the METRIC accounts for the remaining 2.4066 = 84 %.** The per-well scalars were `α = 0.53…0.90`, i.e. a correction
of well under 2×, which is why it could never have closed a 3× gap.

## 12. ⚠ Limitations, stated rather than buried

- **L1 — a row-selection bug in Leg 1, found and fixed before any reported number.** `store_train` rows are **sorted**
  and the Shapes3D grid is a **lexicographic bijection**, so my first Leg-1 subsample (a prefix) restricted the
  outermost factor: `floor_hue` realised **5 of 10** values. Caught by `n_values_realised` being on the row (R31's
  fourth strike earning its keep) and fixed to a seeded permutation (`cb879e6`). ⛔ **No artefact from the prefix
  version was ever written** — the run that exposed it was killed at Leg 1 of seed 0 and `runs/` was verified empty
  at the time; every `veq_s*.json` in this report postdates the fix.
- **L2 — `orientation` is the weak leg of the premise.** Its `VE` spread is 41 % of its mean and seed 2 places it in
  a *high*-variance direction. The premise is much more robust for `scale`.
- **L3 — matched compute, declared.** All arms get the *same* 1,200 steps, key and batch. `pca_whiten`'s store-only
  loss is still falling at step 1,200 (354.8 → 242.5), so it is **not converged** — but its hard read **did** improve
  over the k-means++ init (0.287 → 0.207, 3/3 seeds), and the effective-dimension account (§4) predicts its final
  error to within 5 % *without any reference to training*, and §11's confound control puts only **16 %** of the
  unmatched whitened deficit on the schedule. ⛔ Under-training cannot explain a 0.98-vs-0.26 information-theoretic gap.
  ⚠ A converged whitened arm would still be a **different-compute** arm.
- **L4 — the transform is fitted on `store_train`, the same rows the codebook is fitted on.** That is the correct
  choice (it is part of the store's fit) but it means the transform sees 32,768 rows the probe's test set never
  touches; no leakage, but stated.
- **L5 — `decode_rows = 1024`, not the full 4,096.** The decoded column is a secondary read; the headline is the
  probe on the code.
- **L6 — shapes3d only.** ⛔ Nothing here is claimed for dSprites.
- **L7 — `diag_std`'s distinct-of-72 rise is 9 → 15 on a 72-row set with seed sd 2.94.** It clears the pre-registered
  bar but it is a small-n statistic; ⚠ it should be re-measured before anything is built on it.

## How I verified

```
# ⚠ plumbing smokes, numbers DISCARDED, separate --out so they cannot poison the artefacts
python -m chlu.experiments.exp_gen_variance_eq --quick --out .../scratch/gen-variance-equalisation/quick   # EXIT=0 (x3)
# production, 3 seeds
python -u -m chlu.experiments.exp_gen_variance_eq                                                          # EXIT=0
# sensitivity (addendum 1), seed 0, separate --out
python -u -m chlu.experiments.exp_gen_variance_eq --seeds 0 --arms identity pca_whiten \
       --no-scale-match --no-decode --out .../gen-variance-equalisation/sensitivity_no_scale_match   # EXIT=0
```
with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`,
`CHLU_CLAUDE_ROOT=/Users/user/Desktop/CHLU/.claude`,
`CHLU_GEN_BRIDGE_ROOT=/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs` (a worktree has no
`.claude/`) — all three needed, as `gen-loss-attribution` warned.

**Targeted tests, run from the worktree with the data root set:**
`tests/test_gen_whiten.py tests/test_config.py tests/test_gen_store.py tests/test_gen_compose.py
tests/test_gen_attribution.py`
⇒ ⭐ **122 passed · 0 skipped · 0 failed (3:39).** **Zero skips — no skip to name.**
(`tests/test_gen_whiten.py` alone: **26 passed · 0 skipped · 0 failed**.)
⛔ **No full suite started.**
`ruff check` clean on all four touched files. ⚠ `ruff format --check` would reformat all four — and, **measured, not
inherited**, it equally reformats `chlu/core/gen_store.py`, `chlu/core/gen_compose.py` and `tests/test_gen_store.py`
on unmodified `main`; the repo is not format-clean, so I matched house style rather than reformatting
shared-adjacent files.

**Four tests failed on first run and each failure was a real defect in the harness, not in the code under test:**
three were the **null-bias** in the between-class statistic (a planted trailing-axis signal was reported in the
*leading* axis — the estimator, not the features; fixed by the null subtraction, addendum 2), and one was a
`scale_match` default that made "whitening produces unit covariance" false (the matched map is that map times a
scalar; the test now pins the unmatched property). ⭐ **The premise gate was shown returning `DEAD` on a
deliberately high-variance dataset before it was trusted returning `HOLDS`.**

## Artefacts
`.claude/outputs/gen-variance-equalisation/` — `PREREG.md` (08:55, ⛔ unedited) · `PREREG_ADDENDUM_1.md` ·
`PREREG_ADDENDUM_2.md` · `PREREG_ADDENDUM_3.md` · `summary.json` · `tables.md` (12 sections: per-seed provenance,
the full Leg-1 spectrum incl. the raw/biased statistic, per-well VE matrix, effective dimension, the six-factor
profile, raw accuracies, MI, both falsifiers + the secondary arm, distinct counts, capacity, the byte ledger,
round-trip/deployment-equivalence, reconstruction MSE + the under-training check, the decoded column, both corners,
the laundering control, the order witness) · `variance_eq.png` · `production.log` · `sensitivity.log` ·
`runs/shapes3d_recon/veq_s{0,1,2}.json` · `sensitivity_no_scale_match/`.

## Git footprint
- **Branch** `agent/experiment-engineer/gen-variance-equalisation` off `main @ 81ba81b`; **worktree
  `../CHLU-gen-varianceeq`, left parked.**
- `a048650` `chlu/config.py` — **one contiguous append-only `BEGIN`/`END` block at EOF**, ⛔ 0 deletions of anyone
  else's lines (the single counted deletion is the missing trailing newline on the previous last line), no
  `CHLUConfig` field, no `load_config`/`save_config` site. Reached by `gen_variance_eq_config`.
- `4c7cc83` `chlu/core/gen_whiten.py` (new)
- `84f8057` `chlu/experiments/exp_gen_variance_eq.py` (new)
- `7f8df0f` `tests/test_gen_whiten.py` (new)
- `cb879e6` Leg-1 subsample fix (⛔ prefix → seeded permutation)
- `48fb2cb` premise gate adjudicated at the **spoke** level, as pre-registered; every arm adjudicated vs the control
- `7a88754` falsifier (b) must read the **72-request corner** count, not SPREAD's
- ⛔ **Nothing else touched.** `gen_store.py`, `gen_compose.py`, `gen_bridge.py`, `gen_dream.py`, `gen_request.py`,
  `chlu/eval/**`, `chlu/data/**`, `scripts/csf3/**` are **unmodified** — verified by `git diff --stat main..HEAD`.
  ⛔ **The existing store-fitting path is untouched; the unwhitened baseline is reproducible and was reproduced.**
  `git diff --stat main..HEAD` = **4 files, 2,478 insertions, 1 deletion** (that deletion is `config.py`'s
  missing trailing newline).
- ⛔ **No CLI hook added** — `chlu/cli/experiment_cmd.py` is outside the files I own and recent gen experiments run
  via `python -m` (`gen-loss-attribution` flagged the same). **If the Hub wants `chlu exp-gen-variance-eq`, that is a
  separate one-line task.**
- No conflicts. Rebase onto `main` is a no-op (`main` has not moved from `81ba81b`).
- ⛔ Not pushed, no PR, left for review.

## Open questions / follow-ups / risks
1. ⭐⭐ **The one arm worth a follow-up is the CHEAP one.** `diag_std` costs **16,384 B (3.1 %)**, doubles
   distinct-of-72 (**9 → 15**) and is the only thing in this phase to raise `orientation` retention
   (0.2425 → 0.2985). ⛔ It loses `object_hue` and `shape` and does **not** move `scale`. **Worth one scoped arm; ⛔
   not worth a claim.**
2. ⭐⭐ **The repair implied by §4 is `M`, and the cost is now computable.** To hold a fixed distortion at effective
   dimension `PR`, `M` scales as `exp(PR/2 · ln(1/D))`. ⚠ The design note's "much larger codebook with the byte
   ledger re-declared" fallback is the option this spoke's mechanism section supports; the metric option is closed.
3. ⚠ **`orientation` behaves differently from `scale` at every level** — higher `VE`, higher seed spread, the only
   factor that ever rises, and the one that killed seed 2's premise. ⛔ Treating them as one pair ("`scale` +
   `orientation`") has been the programme's shorthand and this spoke is evidence against it.
4. ⚠ **`pca_whiten` is not converged at 1,200 steps.** A converged-compute arm is a different (and more expensive)
   experiment; §4's effective-dimension law and §11's confound control both argue it cannot change the sign (the
   schedule accounts for 16 % of the unmatched deficit), but it has not been run.
5. ⚠ **The `identity == shipped` agreement is so exact that a reviewer will ask whether the control silently reuses
   the loaded store. It does not:** `run_arm` builds its store with `init_store` (k-means++ on the arm's own
   transformed features) and then `train_store`, from `PRNGKey(seed·10007 + 20260911)` — it never receives the loaded
   object. The agreement is exact because the identity transform is **literally** the identity (`α ≡ 1.0`, `μ ≡ 0`,
   `A ≡ I`, round-trip `0.000e+00`) and every other input is `gen-read-compose`'s. ⭐ That is reproducibility, which
   is the point of the control — but the check is worth one Hub minute.

## Proposed handover updates (for the Hub)
- **§7 NEW — the variance premise is TRUE and the repair is SELF-DEFEATING (this spoke).** `scale` carries
  **2.4 %** of its well's variance to `floor_hue`'s **75.0 %** and its signal sits **4×** deeper in the spectrum
  ⇒ the premise holds (2/3 seeds; seed 2 DEAD on `orientation`). But per-well PCA whitening takes TOTAL-RETAINED
  **3.678 → 1.172** and distinct codes **3,638 → 626**. ⛔ **Never write "whitening / variance equalisation fixes the
  `scale` collapse."**
- **§7 NEW — the effective-dimension law.** Well features have `PR ≈ 3–14` of `d_k = 283–744`; variance-weighted
  `M^(−2/PR)` predicts the read error at **0.253** (measured **0.278**) unwhitened and **0.979** (measured **0.934**)
  whitened. ⭐ **k-means' reliance on high variance IS the compression** — this is the quantitative reason the repair
  goes to `M`, not to the metric.
- **§7 amend — the byte figure.** Per-well PCA whitening is **6.87× only on seed 0**; the 3-seed range is
  **6.87–7.53×** (mean 7.14×) because the well sizes differ. ⭐ And it is **fit-time-only (0 deployed bytes) on the
  request-side path**, proved by `assert_deployment_equivalence` (max rel Δ = 0.0, 9/9).
- **§7 NEW — `train_store`'s `τ` is in ABSOLUTE distance units.** Any arm that changes the feature scale sees a
  different anneal *and* a different effective Adam step. ⭐ The fix is a **uniform per-well scalar**, which is
  **metric-neutral** (Euclidean k-means is equivariant under it) and costs 0 extra bytes. Measured here: it is worth
  **16 %** of a whitened arm's deficit. ⛔ Any future arm that rescales features must carry this control.
- **§7 NEW — standing trap: never take a PREFIX of a sorted row set on Shapes3D.** The grid is a lexicographic
  bijection, so a prefix restricts the outermost factor (measured: `floor_hue` 5 of 10 values). Every subsample must
  be a seeded permutation, and `n_values_realised` must be on the row — it is what caught this.
- **§3 config — new block `GenVarianceEqConfig`** (append-only at `config.py` EOF, reached via
  `gen_variance_eq_config(config)`; ⛔ deliberately **not** a `CHLUConfig` field, same rationale as
  `gen_attribution`/`gen_navigate`). Defaults: 3 seeds, `arms=[identity, pca_whiten, diag_std]`,
  `n_store_train` inherited from `gen_store` (32,768), `spectrum_rows=16384`, `whiten_eps=1e-6`,
  `scale_match_to_raw=True`, `premise_gate_scope="spoke"`, `stop_if_premise_dead=True`,
  `dec_recipe="v1_capacity_schedule"`.
- **Worktree ledger:** `../CHLU-gen-varianceeq` added, **parked** (not removed), 0 uncommitted files.
