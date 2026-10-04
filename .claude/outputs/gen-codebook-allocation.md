# gen-codebook-allocation — experiment-engineer report

**Task + acceptance criterion:** allocate per-well `M_k` **by measured effective dimension** instead of uniformly,
**at constant total bytes**, and report the six-factor retention ladder + read error against the uniform arm;
acceptance = PREREG before measuring · allocation from measured `PR` with bytes asserted · 6-factor ladder +
TOTAL-RETAINED, 3 seeds, `n_values_realised` on every row · distinct-of-72 both arms · the **fit** of the law ·
targeted tests green.
**Status: done.**

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⭐⭐ **D6 IS ANSWERED AND THE ANSWER IS NO — re-allocation is not a lever, and the reason is arithmetic, not
>    experimental.** The task's named rule (`equal_distortion`) **loses**: TOTAL-RETAINED **3.678 ± 0.075 →
>    3.172 ± 0.094**. And the allocation that *minimises* the read error at the same bytes (`mse_optimal`) **ties**:
>    Δ TOTAL **−0.012** against a seed spread of 0.075, measured read error **0.2795 ± 0.0059 vs 0.2783 ± 0.0064**.
>    ⭐ **Uniform `M = 64` is already at the byte-constrained optimum to within 1 %.** **Owner: Hub — §0.5's "a cheap
>    candidate nobody has tried" / D6 must be closed in the design note.**
> 2. ⭐⭐ **THE LAW'S *EXPONENT* IS WRONG, and this is a correction to design-note §0.5, not a detail.** Within each
>    well the measured exponent of `ε` in `M` is **−0.2631 ± 0.0555, essentially CONSTANT across `PR` = 3.12 … 13.63**,
>    against the law's `−2/PR` which spans **−0.642 … −0.147** — a 4.4× range the data does not show. `ε ≈ M^(−2/PR)`
>    survives as a **level** (pooled `r = 0.905`, median ratio 1.078) and fails as a **rate**. ⛔ Every §0.5 table that
>    extrapolates `M = ε^(−PR/2)` (the "6.5 million codewords" row) is built on the exponent that just failed.
>    ⚠ POST-HOC (`PREREG_ADDENDUM_1` §2), `M ∈ [4, 186]`, ⛔ not extrapolable. **Owner: Hub, before §0.5 is cited again.**
> 3. ⭐ **A SECOND, CHEAPER THING MOVED "DISTINCT OF 72" UPWARD — at NEGATIVE bytes.** `mse_optimal` takes the
>    request-side distinct count **9.00 ± 0.82 → 14.33 ± 3.30** (**3/3 seeds up**: 9→19, 8→12, 10→12) while **losing
>    nothing** on the ladder, at **−432 / −256 / −1,008 B**. `gen-variance-equalisation`'s `diag_std` got 9 → 15 but
>    **paid** `object_hue` and `shape`; this does not. ⛔ **NOT the pre-registered pass** (TOTAL did not rise) and
>    ⚠ a 72-row statistic on 3 seeds. **Owner: Hub — this is a second candidate for the D7 follow-up arm.**
> 4. ⚠ **A DEFECT IN MY OWN ADJUDICATION, declared not patched.** PREREG §4 enumerated PASS/TRADE/NULL with an
>    `else: LOSS` fallback, and the fallback labelled item 3 **"⛔ LOSS"** — factually false, nothing fell.
>    `PREREG_ADDENDUM_1` adds the **SPLIT** branch and narrows LOSS; ⛔ **no pre-registered answer changes**
>    (`equal_distortion` LOSS before and after; `mse_optimal` NOT-A-PASS before and after). **Owner: none needed —
>    recorded so a reader of the log is not surprised by two different words for one run.**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *codeword allocation across wells, at CONSTANT TOTAL BYTES.* ⛔ The encoder, the decoder, the
  **partition**, `K`, the store recipe, the row draw and the probe are all UNCHANGED — loaded, hash-asserted, and
  (for the partition) asserted **channel-for-channel**. The only thing any arm changes is `M_k`.
- **Laundering control:** ⭐ **uniform `M = 64` at the same total bytes — the shipped arm, re-fitted through the
  identical code path.** ⭐ **It reproduced the shipped store to every printed digit on all six factors, all three
  row sets, both distinct counts and the decoded column.** A second control rides along: a **random-code probe at
  each arm's own `M_k` widths**, because the one-hot probe width `Σ_k M_k` is arm-dependent.
- **Falsifies:** the scaling law's actionability. ⛔ **This is a CONTROL ARM for the dynamical read, ⛔ NOT a gate
  on it.**
- **Does NOT falsify:** failing to rescue the high-`PR` wells (§0.5 consequence 2 says in advance that it cannot);
  losing to `knn_proto` (metric-native-ceiling theorem).

## ⭐⭐ THE RESULT — one line, in the pre-registered words

⭐ **`equal_distortion` (the rule D6 names): pre-registered pass = ⛔ NO, reading = LOSS.**
⭐ **`mse_optimal` (the byte-constrained optimum): pre-registered pass = ⛔ NO, reading = SPLIT.**
⭐⭐ **Neither arm beats the uniform control on TOTAL-RETAINED, and the optimum cannot: uniform `M = 64` already sits
within 1 % of the best achievable variance-weighted read error at this byte budget.** ⭐ **The law is real and the
re-allocation it suggests is not a lever.**

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | per-seed artefacts produced at **`d0c090a`**; `summary.json` regenerated at **`ec24d8d`**. ⚠ Both were **rebased** onto the Hub's moved tip → **`61b816c`** and **`25a66a7`**; the rebase touched only `chlu/config.py`'s anchor. ⭐ **Verified bit-identical across the rebase**: `gen_alloc.py` `a09fa09cc0f5`, `gen_store.py` `d0d27d098b27`, `gen_whiten.py` `0a9a1fbd6903`, `exp_gen_codebook_alloc.py` `0c5206934f3f` (pre-fix) / `85d97c7f9dcf` (post-fix), config block `diff` empty. ⛔ **The only file differing between `d0c090a` and `ec24d8d` is `exp_gen_codebook_alloc.py`, and the difference is confined to `summarize`/`_verdict`** — no per-seed number was produced by a different tree than the one that computed its inputs. |
| branch / base | `agent/experiment-engineer/gen-codebook-allocation`, spawned off **`main @ 107fea9`**, rebased onto **`main @ 9c83e83`** (the Hub merged wave-9 `gen-atom-mean-control` + `gen-dynamical-read` mid-run) |
| worktree / venv | `../CHLU-gen-alloc`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero). Grid 480,000 = 10·10·10·8·4·15 |
| seeds | **0, 1, 2**; `±` is **population sd, ddof = 0** |
| encoder · decoder · partition · `K` | ⛔ **ALL LOADED AND UNCHANGED.** Encoder hash asserted vs `bridge_manifest.json` **and** the feature-cache manifest; cache re-encode check **64 rows bit-identical**, 3/3. Decoder = `gen-decoder-lift`'s **`v1_capacity_schedule`**, `param_hash` **and** recipe name asserted (⛔ never `dec_recipe="baseline"`). Partition = `gen-commitment-collapse`'s `labels_vec`, asserted channel-for-channel against the shipped store. `K = 5` |
| ⭐ what DID change | **only `M_k`.** Arms `uniform` (⭐ control, `M_k = 64`) · `equal_distortion` (⭐ primary, the D6 rule) · `mse_optimal` (secondary), plus the `shipped` reference store |
| ⭐ byte budget | `M·D·4` = **524,288 B**, `D = 2,048`, every seed. Enforced as **`bytes ≤ bytes(uniform)`** by `assert_bytes_not_exceeded`, which **raises** on `>` |
| the allocation input | per-well `PR = (Σλ)²/Σλ²` and `Σλ`, **LABEL-FREE**, on `spectrum_rows = 16384` rows drawn as a **seeded permutation** of `store_train` (⛔ never a prefix — veq §12 L1) |
| store re-fit | `init_store` (k-means++) + `train_store`, **`train_steps=1200`, `train_batch=128`, `train_lr=3e-3`, `commitment_beta=0.25`, `tau 1.0 → 0.1`**, `n_store_train=32768`, PRNG `PRNGKey(seed·10007 + 20260911)` — ⛔ **identical across arms**, `gen-read-compose`'s recipe verbatim |
| rows | `store_train` **32,768** · `SPREAD_train` **12,288** · `SPREAD` (test) **4,096** · `CORNER_B` **72** · `CORNER_MS` **198/199/199**; draw = `exp_gen_attribution`'s RNG stream verbatim (`default_rng(seed·7919 + 20260914)`) |
| probe (⛔ identical at every point, every arm) | `WellProbe`, `probe_steps=3000`, `probe_batch=512`, `probe_lr=3e-3`, key `PRNGKey(20260914 + seed)`; the **encoder point is fitted once and shared by every arm** |
| ⭐ per-arm random control | `random_codes_per_well` at **each arm's own `M_k`** (⛔ not one shared uniform control), `random_code_seed=20260914` |
| address book | `fit_address_book`, `address_smoothing = 1.0`; ⛔ the posterior is **masked to the allocation** before `compose_codes` (see §6) |
| decoded column | `decode_rows=1024`, `decode_batch=256` |
| ⛔ NOT used | `soft_read`, `DreamLandscape`, `dream_rollout`, the inter-well coupling, the external classifier, whitening, CSF3 |
| wall clock | **2,165 / 2,141 / 2,586 s** per seed. CPU only, 16 GB |

**Pre-registration:** `.claude/outputs/gen-codebook-allocation/PREREG.md`, mtime **15:18:59**, ⛔ never edited.
First production artefact `alloc_s0.json` **16:05**. `PREREG_ADDENDUM_1.md` is a **NEW FILE** (R34's seventh strike),
mtime **17:24**, filed **after** the run and **before** `summary.json` was regenerated **17:25**; it changes **no
pre-registered answer** (§4 of this report). ⇒ **every pre-registered number postdates its prereg on disk.**

---

## 1. ⭐⭐ THE ALLOCATIONS, AND THE BYTE GATE (acceptance 2)

`PR` is re-measured **label-free** in this harness and reproduces `gen-variance-equalisation`'s published values to
**max relative deviation 1.29 × 10⁻⁴** (3/3 seeds) — the allocation is derived from a **measurement**, not an
assumption, and the measurement is the one already on the record.

| seed | arm | `M_k` | store bytes | Δ vs uniform | bits/read | `Σ M_k` (probe width) |
|---|---|---|---|---|---|---|
| 0 | `uniform` | [64, 64, 64, 64, 64] | **524,288 B** | **+0** | 30.00 | 320 |
| 0 | ⭐ `equal_distortion` | **[4, 4, 160, 70, 5]** | 524,048 B | **−240** | 19.77 | 243 |
| 0 | `mse_optimal` | [51, 64, 72, 53, 77] | 523,856 B | **−432** | 29.84 | 317 |
| 1 | `uniform` | [64, 64, 64, 64, 64] | **524,288 B** | **+0** | 30.00 | 320 |
| 1 | ⭐ `equal_distortion` | **[11, 6, 4, 5, 164]** | 523,644 B | **−644** | ⛔ **17.72** | 190 |
| 1 | `mse_optimal` | [83, 62, 59, 78, 51] | 524,032 B | **−256** | 30.17 | 333 |
| 2 | `uniform` | [64, 64, 64, 64, 64] | **524,288 B** | **+0** | 30.00 | 320 |
| 2 | ⭐ `equal_distortion` | **[4, 4, 25, 186, 9]** | 523,496 B | **−792** | 19.35 | 228 |
| 2 | `mse_optimal` | [58, 54, 68, 46, 93] | 523,280 B | **−1,008** | 29.76 | 319 |

⭐ **The gate: `any_arm_over_budget = False`, max shortfall 0.192 %.** ⛔ **No arm is a bigger store.**

⚠ **The one place I departed from the task's wording, pre-registered in `PREREG.md` §1.1.** The task's falsifier #1
is *"total bytes are not exactly equal ⇒ it is a bigger store."* With integer `M_k` at unequal `d_k` (283 … 744),
`Σ M_k d_k = 131,072` **exactly** is a Diophantine problem with no guaranteed solution, so the enforceable — and
pre-registered — constraint is **`≤`, never `>`**, with the exact shortfall on every row. Every allocated arm is
**smaller** than the control by 240 … 1,008 B, so it is handicapped, never helped.

⛔ **The channel width is NOT preserved by the named rule.** `Σ_k log₂M_k` falls from **30.00** to **17.72 … 19.77
bits**, and on **seed 1** to **17.72 < log₂(480,000) = 18.87** — `aggregate_is_sufficient` flips to **False**, which
it has never been in this programme. ⭐ Pre-registered in `PREREG.md` §2.2 and confirmed. ⛔ (My further guess there —
that seed 1's distinct-of-72 would collapse *specifically* — is **FALSIFIED**: seed 1 went 8 → 9, while **seed 0**,
which stayed sufficient, collapsed 9 → 4.)

## 2. ⭐⭐ THE SIX-FACTOR LADDER + TOTAL-RETAINED (acceptance 3) — SPREAD, 3 seeds

⛔ Every row carries `n_values_realised`. Degenerate factors are excluded from the total and the count travels with
it (none were degenerate here: 6 of 6 counted, 3/3 seeds).

| factor | `n_val` | chance | encoder | **`shipped`** (ref) | ⭐ **`uniform`** (control) | ⭐ **`equal_distortion`** | **`mse_optimal`** |
|---|---|---|---|---|---|---|---|
| `floor_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | 0.9968 ± 0.0021 | **0.9968 ± 0.0021** | ⛔ **0.7563 ± 0.0215** | 0.9964 ± 0.0057 |
| `wall_hue` | 10 | 0.100 | 0.9997 ± 0.0001 | 0.9946 ± 0.0036 | **0.9946 ± 0.0036** | 0.9541 ± 0.0117 | 0.9936 ± 0.0012 |
| `object_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | 0.6649 ± 0.0432 | **0.6649 ± 0.0432** | 0.6670 ± 0.0310 | 0.6644 ± 0.0468 |
| **`scale`** | **8** | 0.125 | 0.9352 ± 0.0058 | 0.2245 ± 0.0036 | **0.2245 ± 0.0036** | ⛔ **0.1676 ± 0.0139** | 0.2243 ± 0.0018 |
| `shape` | 4 | 0.250 | 0.9960 ± 0.0008 | 0.5547 ± 0.0338 | **0.5547 ± 0.0338** | 0.5628 ± 0.0307 | 0.5425 ± 0.0293 |
| **`orientation`** | **15** | 0.067 | 0.9915 ± 0.0018 | 0.2425 ± 0.0424 | **0.2425 ± 0.0424** | ⛔ **0.0645 ± 0.0230** | 0.2451 ± 0.0416 |
| ⭐⭐ **TOTAL RETAINED** | — | — | — | **3.678 ± 0.075** | ⭐ **3.678 ± 0.075** | ⛔ **3.172 ± 0.094** | **3.666 ± 0.040** |
| random-code control, **each arm's own `M_k`** | — | — | — | −0.014 ± 0.018 | −0.014 ± 0.018 | −0.014 ± 0.008 | −0.000 ± 0.016 |
| one-hot probe width `Σ M_k`, per seed | — | — | — | [320,320,320] | [320,320,320] | **[243,190,228]** | [317,333,319] |

⭐ **`equal_distortion` is a REDUCTION, not the trade I pre-registered.** I predicted (`R4`) that `scale` would
**rise** to ≈ 0.32, because `scale`'s headline well is the highest-`PR` well in all three seeds and the rule gives
that well 70 / 164 / 186 codewords against 64. **It fell, to 0.1676.** `orientation` fell by **0.178**, `floor_hue`
by **0.240**. Only `shape` (+0.008) and `object_hue` (+0.002) moved up, both **inside** seed spread. ⛔ **Feeding a
well more codewords did not make it carry more of the factor whose signal lives there.**

⭐ **`mse_optimal` is a TIE, and that is the point.** Δ TOTAL = **−0.012** against a seed spread of **0.075**; every
one of the six factors is within spread. ⭐ **This is the pre-registered NULL reading on the ladder: uniform `M` is
already at the byte-constrained optimum.**

⚠ **The probe-capacity confound is controlled in the direction that matters.** The one-hot width is `Σ_k M_k`, which
is **arm-dependent**: `equal_distortion`'s probe is **narrower** (243/190/228 vs 320), so its loss cannot be
attributed to it being given less; and `mse_optimal`'s is **317/333/319**, i.e. ±4 % of the control — far too small
to manufacture a tie. The random-code control at each arm's own widths scores **≤ 0.0 TOTAL** everywhere, so no arm's
probe is reading its own width.

**Corner sets, reported separately and ⛔ never pooled into SPREAD** (TOTAL-RETAINED):
`CORNER_B` — uniform **3.047 ± 0.204**, `equal_distortion` 2.771 ± 0.171, `mse_optimal` **3.278 ± 0.196**.
`CORNER_MS` — uniform **3.812 ± 0.264**, `equal_distortion` 3.408 ± 0.341, `mse_optimal` 3.786 ± 0.099.

## 3. ⭐ DISTINCT CONFIGURATIONS ON THE 72 HELD-OUT REQUESTS (acceptance 4)

| arm | ⭐ **request-side distinct of 72** (PRIMARY) | per seed | feature-side distinct / 4,096 SPREAD |
|---|---|---|---|
| `shipped` (ref) | **9.00 ± 0.82** | [9, 8, 10] | 3,638.3 ± 59.9 |
| ⭐ `uniform` (control) | **9.00 ± 0.82** | [9, 8, 10] | 3,638.3 ± 59.9 |
| ⭐ `equal_distortion` | **8.00 ± 2.94** | [4, 9, 11] | 2,499.3 ± 263.0 |
| `mse_optimal` | ⭐ **14.33 ± 3.30** | **[19, 12, 12]** | 3,620.7 ± 31.4 |

⭐ **`mse_optimal` raises the number the pivot named, on 3/3 seeds, at NEGATIVE bytes and with no ladder cost.** It
is the second thing in this phase to move it upward; `gen-variance-equalisation`'s `diag_std` got **9 → 15** but paid
`object_hue` (0.665 → 0.463) and `shape` (0.555 → 0.423) for it, and cost **16,384 B**. This one costs **−432 /
−256 / −1,008 B** and loses nothing measurable.
⛔ **It is NOT the pre-registered pass** — the pass requires TOTAL-RETAINED to **rise**, and it did not.
⚠ **It is a 72-row statistic with a 3.30 seed sd, on 3 seeds, and it was not the arm the prereg pointed at.**
⛔ **Do not promote it. Replicate it.**

## 4. ⛔ THE VERDICT, and the defect in my own adjudication

| arm | pre-registered pass | reading | Δ TOTAL (spread) | Δ distinct-of-72 (spread) | bytes ≤ uniform |
|---|---|---|---|---|---|
| ⭐ `equal_distortion` | ⛔ **NO** | **LOSS** | **−0.506** (0.094) — 5.4 spreads | −1.00 (2.94) | ✔ [−240, −644, −792] |
| `mse_optimal` | ⛔ **NO** | ⭐ **SPLIT** | −0.012 (0.075) | **+5.33** (3.30) | ✔ [−432, −256, −1008] |

⚠ **A defect in the harness's own verdict logic, declared rather than quietly patched.** `PREREG.md` §4 enumerated
**PASS / TRADE / NULL** with an `else: LOSS` fallback. The run produced a fourth outcome — TOTAL **flat** while
distinct-of-72 **rose beyond spread** — and the fallback printed **"⛔ LOSS"**, which is factually false: nothing
fell. `PREREG_ADDENDUM_1` §1 adds the **SPLIT** branch and narrows LOSS to `Δ TOTAL < −spread`.
⭐ **No pre-registered answer moves:** `equal_distortion` was a LOSS before and after, `mse_optimal` was NOT-A-PASS
before and after. Only the *name* of one non-pass changed, from a false one to a true one. All five branches are now
pinned by `tests/test_gen_alloc.py::test_every_verdict_branch_is_reachable_and_correctly_named`, **including the one
that was mislabelled**.

## 5. ⭐⭐ THE LAW — THE FIT, NOT THE WIN (acceptance 5)

### 5a. Variance-weighted, per arm — the quantity `gen-variance-equalisation` measured
| arm | predicted `Σ v_k ε_k / Σ v_k` | **measured** relative read error | ratio |
|---|---|---|---|
| `shipped` (ref) | — | **0.2783 ± 0.0064** | — |
| ⭐ `uniform` (control) | 0.2533 ± 0.0217 | **0.2783 ± 0.0064** | 1.099 |
| ⭐ `equal_distortion` | 0.4446 ± 0.0145 | **0.4044 ± 0.0083** | 0.910 |
| `mse_optimal` | 0.2511 ± 0.0212 | **0.2795 ± 0.0059** | 1.113 |

⭐ **The control's 0.2783 ± 0.0064 reproduces veq's 0.2783 ± 0.0064 exactly** — the read error is re-measured here,
not transcribed.
⭐⭐ **The byte-constrained *optimum* does not beat uniform: +0.0012, i.e. 0.4 % WORSE, inside noise.** I
pre-registered (`L4`) that it would be *slightly better*; the **magnitude** prediction held (|Δ| < 0.01) and the
**sign** is falsified. ⇒ **the headroom from re-allocation is not merely small, it is not measurable.**
⛔ **`equal_distortion` is 45 % worse than the control on the read error** — as pre-registered. Equalising distortion
at a fixed budget levels the **easy** wells **up**.

### 5b. The pooled log-log fit (pre-registered, `L5`–`L8`)
`log ε_measured = slope · log ε_predicted + intercept`, **60 points** (arm × well × seed).

| quantity | predicted (80 % interval) | measured | verdict |
|---|---|---|---|
| slope | 0.85 [0.55, 1.20] | **0.5019** | ⛔ **FALSIFIED — below** |
| Pearson `r` (log) | 0.93 [0.75, 0.995] | **0.9048** | ⭐ HELD |
| median ratio measured/predicted | 1.10 [0.7, 1.8] | **1.0780** | ⭐ HELD |
| fraction within [0.5, 2.0]× | 0.80 [0.45, 1.00] | **0.767** | ⭐ HELD |

`R²(log) = 0.8187`; ratio range **[0.737, 2.728]**.
⚠ **Read the per-arm row before reading the pooled one.** `uniform`/`shipped` slope **0.4812** (`r` 0.8849),
`mse_optimal` **0.5060** (`r` 0.8916), but `equal_distortion` **0.0150** (`r` **0.0056**) — ⛔ **that arm has no
predictive signal at all, because its predicted `ε` is equalised by construction**, so there is nothing left to
predict. The pooled fit is carried by the two non-equalised arms.

### 5c. Per well, at the shipped `M = 64` — ⛔ where the law is systematically wrong
| `PR` band | measured / predicted `ε` |
|---|---|
| `PR` ≈ 3.1 … 4.0 (the **easy** wells) | **1.94 … 2.73** — ⛔ the law says they are far easier than they are |
| `PR` ≈ 5.6 … 8.5 | 0.96 … 1.35 |
| `PR` ≈ 10.5 … 13.6 (the **hard** wells) | **0.78 … 0.91** — the law says they are harder than they are |

⭐ **The law over-states the spread between easy and hard wells by roughly 2×**, and that spread is exactly what an
allocation rule trades on. ⛔ **This is why equal-distortion allocation is a bad trade even on its own terms: it
re-allocates against an exaggerated gradient.**

### 5d. ⚠ POST-HOC (`PREREG_ADDENDUM_1` §2) — **the exponent, measured directly**
The three allocations give every well a free `M` sweep. Fitting `log ε_measured` against **`log M`** within each
(seed, well), 15 wells:

| | value |
|---|---|
| **fitted exponent** | ⭐⭐ **−0.2631 ± 0.0555**, range [−0.373, −0.171] |
| the law's own `−2/PR` | **−0.6416 … −0.1467** (a **4.4×** range) |
| ratio fitted / law | median **0.642**, range [0.462, 1.418] |

⭐⭐ **The measured exponent is near-CONSTANT across `PR` = 3.12 … 13.63; the law says it should vary 4.4-fold. It
does not.** Empirically `ε ≈ 0.6 · M^(−0.26)` at this scale, essentially independently of `PR`.
⛔ **`ε ≈ M^(−2/PR)` survives as a LEVEL and fails as a RATE.**
⚠ **Limits, declared:** for several wells two of the three `M` values are nearly equal, so the slope is effectively
a two-point contrast; the measured range is `M ∈ [4, 186]` and ⛔ **it may not be extrapolated** (a plausible reading
is that at `M = 64` with `PR` ≈ 3–14 we are **not** in the high-rate regime the law assumes, and/or that a heavy
eigenvalue tail floors the distortion in a way `PR` alone does not capture — ⚠ **conjecture, untested**).

## 6. ⛔ THE CEILING — restated, because a partial fix is not a fix

`M = ε^(−PR/2)` per well at seed 0's `PR`, with the byte cost:

| target ε | modes needed per well | bytes | × the 524,288 B budget | feasible? |
|---|---|---|---|---|
| 0.278 (today's average) | 7 · 9 · **3,070** · 828 · 14 | 8,735,228 B | **16.7×** | ⛔ NO |
| 0.10 | 36 · 54 · **1,871,769** · 177,357 · 108 | 4.65 GB | ⛔ **8,861×** | ⛔ NO |
| 0.05 | 107 · 179 · **144,673,307** · 6,743,888 · 442 | 345 GB | ⛔ **658,239×** | ⛔ NO |

⭐ **No re-allocation at a fixed budget rescues a high-`PR` well, and this spoke does not claim otherwise.**
⚠ **But §5d says these rows are computed with the exponent that just failed.** Under the measured `ε ∝ M^(−0.26)`
the same arithmetic is far less catastrophic — and far less trustworthy, since it is an extrapolation four orders of
magnitude outside `M ∈ [4, 186]`. ⛔ **Neither table should be quoted as settled until the `M` sweep is run
properly** (see follow-ups).

## 7. ⭐ THE LAUNDERING CONTROLS

| check | result |
|---|---|
| ⭐ **`uniform` vs `shipped`, all six factors, all three row sets** | **identical to every printed digit**, 3/3 seeds |
| ⭐ **`uniform` vs `shipped`, distinct counts** | **9.00 ± 0.82** and **3,638.3 ± 59.9** — `gen-loss-attribution`'s and veq's own figures, **re-measured** |
| ⭐ **`uniform` vs `shipped`, decoded column** | **0.6225 ± 0.0124** both, per-factor identical |
| `knn_proto` feature + embedding sha256, before vs after | **BIT-IDENTICAL**, max \|Δ\| = **0.0**, 3/3 seeds |
| random-code control at **each arm's own** `M_k`, TOTAL-RETAINED | −0.014 / −0.014 / −0.014 / −0.000 |
| training improved the hard read over the k-means++ init | **True**, 3 arms × 3 seeds = 9/9 |
| mode usage | **every allocated mode is used**: 320/320, 243/243, 317/317 … 12/12 arm-seeds |

⭐ **The control is not a claim — it is a bit-level reproduction**, and it is what makes the two allocated arms'
results credible.

## 8. The decoded column — ⛔ through the WINNER decoder (`v1_capacity_schedule`)

| arm | aggregate | `floor_hue` | `wall_hue` | `object_hue` | `scale` | `shape` | `orientation` |
|---|---|---|---|---|---|---|---|
| `shipped` / ⭐ `uniform` | **0.6225 ± 0.0124** | 0.998 | 0.989 | 0.686 | 0.208 | 0.599 | 0.255 |
| ⭐ `equal_distortion` | ⛔ **0.5167 ± 0.0192** | **0.568** | 0.918 | 0.705 | 0.176 | 0.625 | **0.108** |
| `mse_optimal` | **0.6142 ± 0.0045** | 0.998 | 0.988 | 0.637 | **0.231** | 0.586 | 0.244 |

⭐ **The pixels agree with the probe.** `equal_distortion` visibly loses `floor_hue` and `orientation`;
`mse_optimal` is within noise of the control.

## 9. Per-well capacity (Leg-4-style, reported not adjudicated)

`n_wells_over_capacity` of 5 — `shipped`/`uniform` **[5, 5, 4]** · `equal_distortion` **[5, 4, 4]** ·
`mse_optimal` **[5, 5, 5]**. ⛔ Re-allocation does **not** reduce the number of wells over their per-well capacity;
it moves which wells are over it. `aggregate_is_sufficient` is **True** everywhere **except `equal_distortion` on
seed 1** (17.72 bits < 18.87).
⚠ **Declared limit:** `well_relevance` estimates MI with a plug-in estimator whose upward bias grows with the number
of occupied bins, and the arms now have **different** bin counts per well. The MI/capacity columns are therefore
**not** strictly comparable across arms; the retention ladder (§2), which is bias-free in that sense, is the
headline and the MI table is diagnostic only.

## 10. Pre-registered predictions vs measurement (verbatim ids)

| id | predicted | 80 % interval | measured | verdict |
|---|---|---|---|---|
| **A1** | harness `PR` vs veq's, max rel dev 0.08 | [0, 0.35] | **1.29e−4** | ⭐ HELD (bottom edge) |
| **A2** | `equal_distortion` worst shortfall 792 B | [0, 1,100] | **792 B** | ⭐⭐ **HELD — exactly** |
| **A3** | `mse_optimal` worst shortfall 1,008 B | [0, 1,500] | **1,008 B** | ⭐⭐ **HELD — exactly** |
| **A4** | any arm over budget = NO | — | **NO** | ⭐ HELD |
| **A5** | min `M_k` = 4 | [2, 12] | **4** | ⭐ HELD |
| **A6** | max `M_k` = 186 | [80, 400] | **186** | ⭐ HELD |
| **L1** | `uniform` measured ε 0.278 | [0.24, 0.32] | **0.2783** | ⭐⭐ **HELD — 3 dp** |
| **L2** | `equal_distortion` ε 0.455 | [0.36, 0.56] | **0.4044** | ⭐ HELD |
| **L3** | `mse_optimal` ε 0.276 | [0.24, 0.32] | **0.2795** | ⭐ HELD |
| **L4** | ⭐ `L3 − L1` **negative**, \|·\| < 0.01 | [−0.02, +0.01] | **+0.0012** | ⚠ **MIXED — interval HELD, SIGN FALSIFIED** |
| **L5** | ⭐ pooled log-log slope 0.85 | [0.55, 1.20] | **0.5019** | ⛔ **FALSIFIED — below** |
| **L6** | pooled `r` (log) 0.93 | [0.75, 0.995] | **0.9048** | ⭐ HELD |
| **L7** | median ratio 1.10 | [0.7, 1.8] | **1.0780** | ⭐ HELD |
| **L8** | ⭐ frac within [0.5, 2]× = 0.80 | [0.45, 1.00] | **0.767** | ⭐ HELD |
| **R1** | `floor_hue` 0.997 / **0.42** / 0.99 | — | 0.9968 / **0.7563** / 0.9964 | HELD / ⛔ **FALSIFIED above** / HELD |
| **R2** | `wall_hue` 0.995 / **0.45** / 0.99 | — | 0.9946 / **0.9541** / 0.9936 | HELD / ⛔ **FALSIFIED above** / HELD |
| **R3** | `object_hue` 0.665 / 0.52 / 0.67 | — | 0.6649 / 0.6670 / 0.6644 | ⭐ HELD ×3 |
| **R4** | ⭐ `scale` 0.224 / **0.32** / 0.23 | — | 0.2245 / ⛔ **0.1676** / 0.2243 | HELD / ⛔ **FALSIFIED below** / HELD |
| **R5** | `shape` 0.555 / 0.52 / 0.56 | — | 0.5547 / 0.5628 / 0.5425 | ⭐ HELD ×3 |
| **R6** | ⭐ `orientation` 0.243 / **0.28** / 0.25 | — | 0.2425 / ⛔ **0.0645** / 0.2451 | HELD / ⛔ **FALSIFIED below** / HELD |
| **R7** | ⭐⭐ TOTAL 3.68 / 2.51 / 3.69 | — | **3.678** / 3.172 / 3.666 | ⭐⭐ HELD ×3 |
| **R8** | ⭐ Δ TOTAL (`mse` − `uni`) +0.01 | [−0.20, +0.25] | **−0.012** | ⭐ HELD |
| **D1** | `uniform` distinct-of-72 = 9 | [6, 14] | **9.00** | ⭐⭐ **HELD — exactly** |
| **D2** | `equal_distortion` = 7 | [2, 16] | **8.00** | ⭐ HELD |
| **D3** | `mse_optimal` = 9 | [5, 15] | **14.33** | ⭐ HELD (upper end; the point estimate was low) |
| **D4** | `uniform` distinct / 4,096 = 3,638 | [3,300, 3,900] | **3,638.3** | ⭐⭐ **HELD — exactly** |
| **D5** | `equal_distortion` = 2,600 | [1,200, 3,700] | **2,499.3** | ⭐ HELD |
| **C1** | `shipped` reproduces `uniform` | — | **bit-identical** | ⭐⭐ HELD |
| **C2** | random control TOTAL < 0.35 every arm | [0, 0.6] | **≤ 0.000** | ⭐ HELD |
| **C3** | probe width: allocated arm **narrower** | — | 243/190/228 vs 320 | ⭐⭐ **HELD — exactly as predicted** |
| **C4** | `knn_proto` bit-identical | — | max \|Δ\| = 0.0, 3/3 | ⭐ HELD |
| **§2.2** | seed 1's distinct-of-72 collapses (bits < 18.87) | — | 8 → **9** (seed **0** collapsed instead) | ⛔ **FALSIFIED** |

⭐ **39 held, 6 falsified, 1 mixed.** ⛔ **The falsified ones are the result.** The six that failed are: the law's
**exponent** (`L5`), the **sign** of the optimum's advantage (`L4`), the four `equal_distortion` ladder cells where I
predicted a **trade** (`R1`, `R2`, `R4`, `R6`) — and the trade did not happen — plus the seed-1 clause. ⭐ The
predictions that held *exactly* (`L1` 0.2783, `R7` 3.678, `D1` = 9, `D4` = 3,638, `A2`/`A3` to the byte, `C3` to the
integer) are the **control and arithmetic** ones, which is what makes the failures credible.

## 11. ⚠ Limitations, stated rather than buried

- **L1 — three `M` values per well is a thin sweep.** §5d's exponent is the spoke's most consequential number and it
  rests on 3 points per well over `M ∈ [4, 186]`, with two of the three often close together. ⛔ **A proper 5–6-point
  `M` sweep per well, at a re-declared byte ledger, is the experiment that should settle §0.5.**
- **L2 — the allocated arms change the probe's input width.** Controlled (§2) but not eliminated: `Σ M_k` is
  243/190/228 and 317/333/319 against the control's 320. The direction is unfavourable to `equal_distortion` and
  ±4 % for `mse_optimal`, so neither result is manufactured by it; but a *strictly* width-matched comparison would
  need padding the one-hot, which would itself be a different probe.
- **L3 — `mse_optimal`'s distinct-of-72 rise is a 72-row statistic on 3 seeds** (sd 3.30), and it is **post-hoc**:
  the prereg pointed at TOTAL-RETAINED, not at it. ⭐ It is 3/3 in direction, which is why it is reported at all.
- **L4 — MI/capacity columns are not cross-arm comparable** (§9): different bin counts ⇒ different plug-in bias.
- **L5 — matched compute, declared.** All arms get the same 1,200 steps, key, batch and `τ` schedule. `τ` is in
  **absolute distance units** (veq's §7 trap), but **this spoke changes no feature scale** — only `M_k` — so the
  anneal every arm sees is identical by construction and no scale-matching control is needed or applied.
- **L6 — `shapes3d`/`recon` only, 3 seeds.** ⛔ Nothing here is claimed for dSprites.
- **L7 — the wells are re-fitted per seed**, so a well index is not the same object across seeds; every per-well
  table is per-seed and none is averaged across seeds.
- **L8 — the byte constraint is `≤`, not `==`** (§1). Pre-registered, and the shortfall (≤ 0.192 %) handicaps the
  allocated arms.

## How I verified

```
# ⚠ plumbing smoke, numbers DISCARDED, separate --out so they cannot poison the artefacts
python -m chlu.experiments.exp_gen_codebook_alloc --quick --out .../scratch/gen-codebook-allocation/quick  # EXIT=0
# production, 3 seeds
python -u -m chlu.experiments.exp_gen_codebook_alloc                                                        # EXIT=0
# summary regenerated after the verdict fix — skip_if_exists HIT 3/3, ⛔ no per-seed artefact recomputed
python -u -m chlu.experiments.exp_gen_codebook_alloc                                                        # EXIT=0
```
with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`,
`CHLU_CLAUDE_ROOT=/Users/user/Desktop/CHLU/.claude`,
`CHLU_GEN_BRIDGE_ROOT=/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs` (a worktree has no
`.claude/`) — all three needed.

**Targeted tests, from the worktree with the data root set, ⛔ no full suite.**
*Pre-rebase:* `test_gen_alloc · test_gen_store · test_gen_whiten · test_config · test_gen_compose ·
test_gen_attribution · test_gen_dream · test_gen_cell · test_gen_certificate · test_gen_navigate ·
test_gen_decoder_lift · test_gen_r0_wide` ⇒ ⭐ **352 passed · 0 skipped · 0 failed (3:32).**
*Post-rebase onto `main @ 9c83e83`* (wave 9 landed mid-run, so everything was re-run):
`test_gen_alloc · test_gen_store · test_gen_whiten · test_config · test_gen_compose · test_gen_attribution ·
test_gen_dream · test_gen_atom_mean · test_gen_dynamical_read` ⇒ ⭐ **282 passed · 0 skipped · 0 failed (2:47)**,
and `test_gen_cell · test_gen_certificate · test_gen_navigate · test_gen_decoder_lift · test_gen_r0_wide ·
test_gen_baselines` ⇒ ⭐ **195 passed · 0 skipped · 0 failed (0:43).**
⭐ **Zero skips anywhere — no skip to name.** `tests/test_gen_alloc.py` alone: **61 passed · 0 skipped · 0 failed.**
`ruff check` clean on all six touched files. ⚠ `ruff format --check` would reformat them — and, **measured**, it
equally reformats `chlu/core/gen_store.py` and `chlu/core/gen_compose.py` on unmodified `main`; the repo is not
format-clean, so I matched house style rather than reformatting shared files.

**One test failed on first run and it was a real defect in my test, not in the code:**
`test_more_codewords_lower_the_MEASURED_distortion` asserted `ε ≤ 1.05` at `M = 2`. ⭐ `init_store` is k-means++
**seeding only** (no Lloyd steps), so two sampled data points are legitimately a *worse* predictor than the mean
(measured **1.538**). The test now **asserts that** (`seen[0] > 1.0`) instead of hiding it, and keeps the law's
content: monotone decrease and `ε < 0.35` at `M = 128`.

## Artefacts
`.claude/outputs/gen-codebook-allocation/` — `PREREG.md` (15:18:59, ⛔ unedited) · `PREREG_ADDENDUM_1.md` (17:24) ·
`summary.json` · `tables.md` (11 sections: `PR`, the allocations + byte gate, the six-factor ladder, distinct-of-72,
the law in five parts incl. the post-hoc exponent, the ceiling, the verdict, the decoded column, the laundering
controls, mode usage, wall clock) · `codebook_alloc.png` · `production.log` ·
`runs/shapes3d_recon/alloc_s{0,1,2}.json`.
Scratch: `.claude/scratch/gen-codebook-allocation/` — `predict.py`/`predict.json` (the pre-registered arithmetic,
run **before** `PREREG.md` was filed), `tables.py`, `quick/` (⚠ smoke, numbers discarded).

## Git footprint
- **Branch** `agent/experiment-engineer/gen-codebook-allocation`, off **`main @ 107fea9`**, **rebased onto
  `main @ 9c83e83`**; **worktree `../CHLU-gen-alloc`, left parked.** ⛔ Not pushed, no PR.
- `61b816c` (was `d0c090a`) `chlu/core/gen_alloc.py` (new) · `chlu/core/gen_store.py` · `chlu/core/gen_whiten.py` ·
  `chlu/config.py` · `chlu/experiments/exp_gen_codebook_alloc.py` (new) · `tests/test_gen_alloc.py` (new)
- `25a66a7` (was `ec24d8d`) the SPLIT-vs-LOSS verdict fix + the post-hoc `M`-exponent instrument + 4 new tests
- `git diff --stat main..HEAD` = **6 files, 2,375 insertions, 15 deletions.**
- ⚠ **THE SHARED-FILE MERGE, done by the ruled rule.** `main` moved under me mid-run (the Hub merged
  `gen-atom-mean-control` and `gen-dynamical-read`), and the rebase conflicted **in `chlu/config.py` only**, at the
  EOF anchor — exactly the collision the task named. ⛔ **I did NOT merge hunk-by-hunk.** I took the **peer's whole
  file** and re-applied my block by its `BEGIN`/`END` markers, then **asserted programmatically that all 6,306 of
  the peer's lines survive as an unmodified prefix ⇒ 0 deletions**, and that my block is byte-identical across the
  rebase (`diff` empty). `chlu/config.py` is **+150 / −1**, the single deletion being the file's missing trailing
  newline.
- **Shared-file edits, named line-for-line:**
  `chlu/core/gen_store.py` — `n_modes_per_well` (new property); `n_modes` now **raises** on a non-uniform store;
  `bits_per_read` → `Σ_k log₂M_k` (**identical** for a uniform store); `store_bytes` emits `n_modes: None` +
  `n_modes_per_well` + `uniform_allocation` instead of a wrong scalar; `init_store` accepts `int | Sequence[int]`
  with the **uniform path bit-identical** (asserted).
  `chlu/core/gen_whiten.py` — `per_well_capacity` accepts a per-well sequence; **the scalar result is unchanged
  digit-for-digit** (asserted in `test_per_well_capacity_accepts_a_sequence_and_is_UNCHANGED_for_a_scalar`).
  ⛔ Nothing else touched: `gen_compose.py`, `gen_dream.py`, `gen_request.py`, `gen_bridge.py`, `chlu/eval/**`,
  `chlu/data/**`, `chlu/cli/**`, `scripts/csf3/**` are **unmodified** (`git diff --stat main..HEAD`).
- ⛔ **No CLI hook added** — ⚠ note that wave-9's `gen-dynamical-read` **did** add hooks to
  `chlu/cli/experiment_cmd.py`, so that file is now live territory; adding `chlu exp-gen-codebook-alloc` is a
  separate one-line task if the Hub wants it. This experiment runs via `python -m`.
- No unresolved conflicts. Branch ref verified **from the main repo** (`git -C /Users/user/Desktop/CHLU log
  main..agent/experiment-engineer/gen-codebook-allocation` shows both commits).

## Open questions / follow-ups / risks
1. ⭐⭐ **The experiment §0.5 actually needs is an `M` SWEEP, not an allocation.** §5d says the exponent is
   ≈ **−0.26 flat**, not `−2/PR`. A 5–6-point per-well sweep (`M` = 16, 64, 256, 1024, 4096) at a **re-declared byte
   ledger** would settle whether the law's rate is wrong or whether we are simply below the high-rate regime — and
   it is the only measurement that can price design-note candidate **C** honestly. ⚠ It is **not** constant-bytes,
   so it needs the Head's ledger decision first.
2. ⭐ **`mse_optimal` is a free 1.6× on distinct-of-72 and wants one replication arm**, ideally at 5+ seeds, beside
   veq's `diag_std`. ⛔ Neither should be promoted on 3 seeds of a 72-row statistic. ⚠ If both replicate, they are
   **composable** (one is a metric, one is an allocation) and that composition has never been measured.
3. ⚠ **Why did feeding `scale`'s well 2.9× the codewords LOWER `scale` retention?** `equal_distortion` gives the
   highest-`PR` well 70/164/186 modes and `scale` falls 0.2245 → 0.1676. The starved wells' collapse plausibly
   drags the joint code, but ⛔ that is a guess; the address book's per-well MI would answer it and this spoke did
   not isolate it.
4. ⚠ **`orientation` is again the odd one out** (veq's follow-up 3): it is the factor `equal_distortion` destroys
   most completely (−0.178, to 0.0645, i.e. **near zero**) while `scale` only drops 0.057. Treating them as one
   pair keeps failing.
5. ⚠ **The `PR` used for the allocation is measured on 16,384 of the 32,768 store-training rows.** It reproduces
   veq's to 1.3e−4, but the allocation and the codebook are fitted on overlapping rows; that is correct (both are
   part of the store's fit) but it is stated.

## Proposed handover updates (for the Hub)
- **§7 NEW — D6 IS CLOSED: re-allocation at constant bytes is NOT a lever.** The rule the design note names
  (`equal_distortion`) takes TOTAL-RETAINED **3.678 ± 0.075 → 3.172 ± 0.094** and the read error **0.278 → 0.404**;
  the byte-constrained **optimum** (`mse_optimal`) ties on both (Δ TOTAL **−0.012** vs spread 0.075; read error
  **+0.0012**). ⭐ **Uniform `M = 64` is already within ~1 % of the best achievable at 524,288 B.** ⛔ Never write
  "spending codewords where the distortion is fixes the `scale` collapse."
- **§7 NEW — ⛔ `ε ≈ M^(−2/PR)` HOLDS AS A LEVEL AND FAILS AS A RATE.** Measured within-well exponent
  **−0.2631 ± 0.0555**, near-constant over `PR` = 3.12 … 13.63, against a law exponent spanning
  **−0.642 … −0.147**. Pooled predicted-vs-measured `r = 0.905`, median ratio **1.078**, but slope **0.502**. ⛔ The
  design note's `M = ε^(−PR/2)` extrapolations (the "6.5 million codewords" and "6,149" rows) are computed with the
  exponent that failed and must be re-derived or explicitly caveated. ⚠ POST-HOC, `M ∈ [4, 186]`, not extrapolable.
- **§7 NEW — the law OVER-STATES the easy/hard spread by ~2×.** At `M = 64`, measured/predicted `ε` is
  **1.94 … 2.73** on the `PR` ≈ 3–4 wells and **0.78 … 0.91** on the `PR` ≈ 10–14 wells. ⭐ That gradient is exactly
  what an allocation rule trades on, and it is smaller than the law claims.
- **§7 NEW (D7 candidate 2) — `mse_optimal` moves "distinct of 72" 9.00 ± 0.82 → 14.33 ± 3.30, 3/3 seeds, at
  NEGATIVE bytes and with no ladder cost.** veq's `diag_std` moved it 9 → 15 but **paid** `object_hue` and `shape`
  and cost 16,384 B. ⛔ Not a pass; ⚠ a 72-row statistic on 3 seeds. **Worth one replication arm; ⛔ not a claim.**
- **§7 NEW — standing trap: `fit_address_book` is RECTANGULAR.** It builds `(K, max_k M_k)` with Laplace smoothing,
  so on a non-uniform store a short well gets non-zero mass on modes it does not own and `argmax` can emit an
  **out-of-range code** that `read_from_codes` would silently gather as a wrapped index. ⭐ Fix shipped:
  `gen_alloc.mask_posterior_to_allocation`, applied before `compose_codes`, plus a hard assertion on the composed
  codes. **Any future per-well-`M` work must use it.**
- **§7 NEW — `GenStore.n_modes` now RAISES on a non-uniform allocation.** A single `M` would be a wrong number, not
  an approximate one, and it silently reaches byte ledgers, one-hot widths and address books. Use
  `store.n_modes_per_well`. `bits_per_read` is now `Σ_k log₂M_k` (identical for uniform stores).
- **§3 config — new block `GenCodebookAllocConfig`** (append-only at `config.py` EOF, reached via
  `gen_codebook_alloc_config(config)`; ⛔ deliberately **not** a `CHLUConfig` field, same rationale as
  `gen_variance_eq`/`gen_attribution`). Defaults: 3 seeds, `arms=[uniform, equal_distortion, mse_optimal]`,
  `m_uniform=64`, `m_min=2`, `spectrum_rows=16384`, `per_arm_random_control=True`,
  `assert_bytes_within_budget=True`, `dec_recipe="v1_capacity_schedule"`.
- **Worktree ledger:** `../CHLU-gen-alloc` added, **parked** (not removed), 0 uncommitted files.
- ⚠ **Process note for the Hub:** `main` moved from `107fea9` to `9c83e83` **while this spoke was running**, and the
  rebase conflicted in `chlu/config.py` exactly as the task file predicted. The whole-file-take + block-re-apply
  rule worked and is now exercised twice; ⭐ the 0-deletion assertion should probably be a scripted check rather
  than a per-spoke habit.
